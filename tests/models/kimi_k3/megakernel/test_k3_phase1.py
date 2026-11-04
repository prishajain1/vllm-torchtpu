# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Strict verification tests for Kimi K3 megakernel Phase 1 isolation."""

from __future__ import annotations

import ast
import dataclasses
import importlib
import json
import operator
import pathlib
from typing import Any

import jax
import jax.numpy as jnp
import pytest
from jax._src.pallas.mosaic import lowering
from jax.experimental import topologies
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.models.kimi_k3.megakernel.kernels import (
    dspark_mla,
    dspark_mla_kernel,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kernels import (
    prefill_megakernel as _pmk_init,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kernels.prefill_moe import moe_down_valu

_Dynamic = Any
_CONST_10 = 10
_CONST_128 = 128
_CONST_16 = 16
_CONST_1792 = 1792
_CONST_2 = 2
_CONST_2048 = 2048
_CONST_256 = 256
_CONST_32 = 32
_CONST_4096 = 4096
_CONST_512 = 512
_CONST_64 = 64
_CONST_7 = 7
_CONST_7168 = 7168
_CONST_8 = 8
_CONST_896 = 896
_CONST_93 = 93


def test_k3_phase1_containment_and_pruned_dead_files() -> _Dynamic:
    """Verify clean import of megakernel, absence of 10 dead files, and no.

    legacy imports.
    """
    mk = importlib.import_module("vllm_torchtpu.models.kimi_k3.megakernel")
    base_dir = pathlib.Path(mk.__file__).resolve().parent

    pruned = [
        "kernels/prefill_moe/layer.py",
        "kernels/prefill_moe/shared_ring_kernel.py",
        "kernels/prefill_moe/routing_count_kernel.py",
        "kernels/prefill_moe/streaming_ring.py",
        "kernels/prefill_moe/routing_topk_kernel.py",
        "kernels/prefill_moe/packed_layout.py",
        "draft_layers.py",
        "draft.py",
        "draft_ring.py",
        "kernels/dspark.py",
    ]
    for dead_file in pruned:
        target = base_dir / dead_file
        assert not target.exists(), f"Dead file still exists: {target}"

    for live_file in [
        "kernels/prefill_moe/indexed_ragged_gmm.py",
        "kernels/prefill_moe/moe_down_valu.py",
    ]:
        target = base_dir / live_file
        assert target.exists(), f"Required live file missing: {target}"

    for py_file in base_dir.rglob("*.py"):
        if py_file.name in ("spec.py", "adapter.py"):
            continue
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "vllm_torchtpu.megakernel", (
                        f"Legacy import {alias.name} found in {py_file}"
                    )
                    assert not alias.name.startswith("vllm_torchtpu.megakernel."), (
                        f"Legacy import {alias.name} found in {py_file}"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module != "vllm_torchtpu.megakernel", (
                    f"Legacy import {node.module} found in {py_file}"
                )
                assert not node.module.startswith("vllm_torchtpu.megakernel."), (
                    f"Legacy import {node.module} found in {py_file}"
                )


def test_dspark_common_helpers_and_behavior() -> _Dynamic:
    """Verify dspark_common pack/unpack tracing, select_rows, rms_norm, and ring.

    helpers.
    """
    dspark_common = importlib.import_module(
        "vllm_torchtpu.models.kimi_k3.megakernel.kernels.dspark_common"
    )

    # pack/unpack tracing on CPU
    x = jnp.arange(16 * 128, dtype=jnp.float32).reshape(16, 128)
    jaxpr_pack = jax.make_jaxpr(dspark_common.pack)(x)
    assert jaxpr_pack.out_avals[0].shape == (8, 128)

    jaxpr_unpack = jax.make_jaxpr(lambda w: dspark_common.unpack(w, 16, 128))(
        jnp.zeros((8, 128), jnp.float32)
    )
    assert jaxpr_unpack.out_avals[0].shape == (16, 128)

    # select_rows on [16, 64]
    block = jnp.arange(16 * 64, dtype=jnp.float32).reshape(16, 64)
    selected = dspark_common.select_rows(block, offset=2, rows=4)
    expected = block[2:6]
    assert jnp.allclose(selected, expected)

    # rms_norm output
    x_bf16 = jnp.ones((2, 128), dtype=jnp.bfloat16)
    weight_bf16 = jnp.ones((128,), dtype=jnp.bfloat16) * 2.0
    normed = dspark_common.rms_norm(x_bf16, weight_bf16, eps=1e-5)
    assert normed.shape == (2, 128)
    assert jnp.allclose(normed, 2.0, atol=1e-2)

    # header bytes and ints roundtrip
    hdr_vals = jnp.array([[1, 255, 65535, 16777215]], dtype=jnp.int32)
    hdr_bytes = operator.attrgetter("_header_bytes")(dspark_common)(hdr_vals)
    hdr_ints = operator.attrgetter("_header_ints")(dspark_common)(
        hdr_bytes.reshape(1, 4, 4)
    )
    assert jnp.array_equal(hdr_vals, hdr_ints)

    # ring_rows and pack_ring/unpack_ring roundtrip
    class FakeCfg:
        """Manage fake cfg instance and state."""

        layers = 2
        window = 32
        head_dim = 64

    rows = dspark_common.ring_rows(FakeCfg())
    assert rows > 0

    ring = jnp.ones((1, 2, 32, 128), dtype=jnp.bfloat16)
    meta = jnp.array([[128, 0]], dtype=jnp.int32)
    packed_ring = dspark_common.pack_ring(ring, meta, blocks=1)
    unpacked_ring, unpacked_meta = dspark_common.unpack_ring(
        FakeCfg(), packed_ring[0], length=128
    )
    assert unpacked_ring.shape == (2, 32, 128)
    assert int(unpacked_meta[0]) == _CONST_128
    assert int(unpacked_meta[1]) == 0


def test_load_checkpoint_config_validation(tmp_path: _Dynamic) -> _Dynamic:
    """Verify config.from_checkpoint parses valid and rejects invalid."""
    load_mod = importlib.import_module(
        "vllm_torchtpu.models.kimi_k3.megakernel.kernels.load"
    )
    config = load_mod.Config

    base_config = {
        "hidden_size": 7168,
        "routed_expert_hidden_size": 3584,
        "moe_intermediate_size": 3072,
        "num_shared_experts": 2,
        "intermediate_size": 33792,
        "num_experts": 896,
        "num_experts_per_token": 16,
        "num_hidden_layers": 93,
        "num_attention_heads": 96,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "vocab_size": 163840,
        "attn_res_block_size": 12,
        "rms_norm_eps": 1e-5,
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
        "mla_use_nope": True,
        "mla_use_output_gate": True,
        "latent_moe_use_norm": True,
        "moe_renormalize": True,
        "moe_router_activation_func": "sigmoid",
        "hidden_act": "situ",
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "num_expert_group": 1,
        "routed_scaling_factor": 1.0,
        "linear_attn_config": {
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
            "full_attn_layers": [*list(range(4, 93, 4)), 93],
            "use_full_rank_gate": True,
        },
    }

    valid_dir = tmp_path / "valid_k3"
    valid_dir.mkdir()
    (valid_dir / "config.json").write_text(json.dumps(base_config))
    cfg = config.from_checkpoint(valid_dir)
    assert cfg.dim == _CONST_7168
    assert cfg.layers == _CONST_93
    assert cfg.experts == _CONST_896

    invalid_dir = tmp_path / "invalid_k3"
    invalid_dir.mkdir()
    bad_config = dict(base_config)
    bad_config["mla_use_nope"] = False
    (invalid_dir / "config.json").write_text(json.dumps(bad_config))
    with pytest.raises(ValueError, match="Unsupported K3 setting"):
        config.from_checkpoint(invalid_dir)


def test_prefill_moe_overflow_helper_and_bf16_reject() -> _Dynamic:
    """Verify KernelConfig rejects bfloat16 and wires live overflow helpers."""
    config_mod = importlib.import_module(
        "vllm_torchtpu.models.kimi_k3.megakernel.config"
    )
    with pytest.raises(
        ValueError, match="moe_reduce_dtype='bfloat16' is not supported"
    ):
        config_mod.KernelConfig(moe_reduce_dtype="bfloat16")

    cfg = config_mod.KernelConfig()
    assert cfg.moe_reduce_dtype == "float32"

    fused_mod = importlib.import_module(
        "vllm_torchtpu.models.kimi_k3.megakernel.kernels.prefill_moe.fused"
    )
    assert hasattr(fused_mod, "routed_experts_prepared")
    assert hasattr(fused_mod, "indexed_gate_up")
    assert hasattr(fused_mod, "down_valu")
    assert hasattr(fused_mod, "down_valu_compact")


def test_k3_mla_dspark_common_and_draft_class(tmp_path: _Dynamic) -> _Dynamic:
    """Verify shared dspark_common symbols and draft_mla.draft_class MLA.

    dispatch/validation.
    """
    dspark_common = importlib.import_module(
        "vllm_torchtpu.models.kimi_k3.megakernel.kernels.dspark_common"
    )
    for sym in (
        "ROWS",
        "OUT_COUNT",
        "OUT_BONUS",
        "OUT_DRAFTS",
        "pack",
        "unpack",
        "pack_ring",
        "unpack_ring",
        "ring_positions",
        "synthetic_count",
        "nucleus_accept",
        "bf16",
        "f32",
        "select_rows",
        "rms_norm",
    ):
        assert hasattr(dspark_common, sym), f"Missing symbol {sym} in dspark_common"

    draft_mla = importlib.import_module(
        "vllm_torchtpu.models.kimi_k3.megakernel.draft_mla"
    )
    draft_cls_fn = draft_mla.draft_class

    valid_dir = tmp_path / "valid_ckpt"
    valid_dir.mkdir()
    (valid_dir / "config.json").write_text(json.dumps({"model_type": "k3_dspark"}))
    cls = draft_cls_fn(valid_dir)
    assert cls == draft_mla.MlaDsparkDraft

    invalid_dir = tmp_path / "invalid_ckpt"
    invalid_dir.mkdir()
    (invalid_dir / "config.json").write_text(json.dumps({"model_type": "legacy_swa"}))
    with pytest.raises(ValueError, match="Unsupported checkpoint model_type"):
        draft_cls_fn(invalid_dir)


def test_pytree_dataclass_registration_and_fields() -> _Dynamic:
    """Verify test pytree dataclass registration and fields logic."""
    dm = importlib.import_module(
        "vllm_torchtpu.models.kimi_k3.megakernel.kernels.decode_megakernel"
    )
    k = importlib.import_module(
        "vllm_torchtpu.models.kimi_k3.megakernel.kernels.prefill_megakernel"
    )
    names = [f.name for f in dataclasses.fields(dm.MultiHeadLatentAttentionWorkspace)]
    assert len(names) == _CONST_10
    assert "maximum_vmem" in names
    assert "denominator_vmem" in names
    assert "accumulator_vmem" in names

    ws = dm.MultiHeadLatentAttentionWorkspace(*[jnp.zeros(1)] * 10)
    assert len(jax.tree.leaves(ws)) == _CONST_10

    w_ref = k.WeightsRef(weight=jnp.zeros(1), scale=jnp.zeros(1))
    assert len(jax.tree.leaves(w_ref)) == _CONST_2

    m_ref = k.MetadataRef(
        gm_id_to_group_id=jnp.zeros(1), gm_id_to_m_offset=jnp.zeros(1)
    )
    assert len(jax.tree.leaves(m_ref)) == _CONST_2


def test_k3_phase1_provenance_headers() -> _Dynamic:
    """Verify source provenance headers in all 11 adapted modules."""
    mk = importlib.import_module("vllm_torchtpu.models.kimi_k3.megakernel")
    base_dir = pathlib.Path(mk.__file__).resolve().parent

    modules_11 = [
        "common/pool_alias.py",
        "common/vmem_pool.py",
        "common/collectives32.py",
        "kernels/decode_megakernel.py",
        "kernels/dspark_mla.py",
        "kernels/dspark_mla_kernel.py",
        "kernels/mla_dcp.py",
        "kernels/prefill_megakernel.py",
        "kernels/prefill_moe/fused.py",
        "kernels/prefill_moe/indexed_ragged_gmm.py",
        "kernels/prefill_moe/moe_down_valu.py",
    ]
    gmm_provenance_modules = {
        "kernels/prefill_megakernel.py",
        "kernels/prefill_moe/indexed_ragged_gmm.py",
        "kernels/prefill_moe/moe_down_valu.py",
    }

    for rel_path in modules_11:
        target = base_dir / rel_path
        assert target.exists(), f"Module {rel_path} does not exist"
        content = target.read_text(encoding="utf-8")
        assert "Source provenance:" in content, (
            f"Missing 'Source provenance:' header in {rel_path}"
        )
        assert "inferact/k3-megakernel@30c9ba966b" in content, (
            f"Missing inferact provenance tag in {rel_path}"
        )
        if rel_path in gmm_provenance_modules:
            assert "vllm_torchtpu/kernels/megablox/gmm_v2.py@644426d851" in content, (
                f"Missing megablox gmm_v2 provenance tag in {rel_path}"
            )


def test_k3_pool_alias_chains_with_main() -> _Dynamic:
    """Verify AST isolation and runtime lower chaining with main pool_alias."""
    mk = importlib.import_module("vllm_torchtpu.models.kimi_k3.megakernel")
    base_dir = pathlib.Path(mk.__file__).resolve().parent
    target = base_dir / "common/pool_alias.py"
    content = target.read_text(encoding="utf-8")
    tree = ast.parse(content, filename=str(target))

    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            func_name = getattr(call.func, "id", None) or getattr(
                getattr(call.func, "value", None), "id", None
            )
            attr_name = getattr(call.func, "attr", None)
            assert func_name != "register_dataclass", (
                "Unexpected top-level register_dataclass call in pool_alias.py"
            )
            assert attr_name != "register_dataclass", (
                "Unexpected top-level register_dataclass call in pool_alias.py"
            )

    k3_pa = importlib.import_module(
        "vllm_torchtpu.models.kimi_k3.megakernel.common.pool_alias"
    )
    main_pa = importlib.import_module("vllm_torchtpu.megakernel.pool_alias")

    assert hasattr(k3_pa, "_registered_box"), (
        "Missing _registered_box in common/pool_alias.py"
    )
    assert 'vars(transform)["_validate_shape"]' not in content, (
        "vars(transform)['_validate_shape'] must be replaced with operator.attrgetter"
    )
    assert 'operator.attrgetter("_validate_shape")' in content, (
        "Missing operator.attrgetter('_validate_shape') in common/pool_alias.py"
    )

    orig_hook = vars(lowering)["_reshape_memref"]
    try:
        vars(k3_pa)["_patched_box"][0] = False
        vars(main_pa)["_installed"] = False
        vars(lowering)["_reshape_memref"] = orig_hook

        operator.attrgetter("_install")(main_pa)()
        operator.attrgetter("_ensure_patched")(k3_pa)()
        k3_orig = vars(k3_pa)["_original"]
        main_lower = vars(main_pa)["_lower"]
        assert k3_orig is main_lower, (
            f"k3_pa._original ({k3_orig}) is not main_pa._lower ({main_lower})"
        )
    finally:
        vars(lowering)["_reshape_memref"] = orig_hook


def _lower_dspark_mla_on_tpu7x(devices: _Dynamic) -> None:
    """Verify deviceless TPU v7x AOT lowering for fused_draft_local."""
    mesh_tp = Mesh(devices, ("tp",))
    c_draft = dspark_mla.MlaDraftConfig()
    r = _CONST_2
    rows = _CONST_8
    pages = _CONST_64
    block = _CONST_7
    dp = 1
    t = r * rows

    fused_draft = dspark_mla_kernel.fused_draft_local(
        c_draft, requests=r, pages=pages, block=block, rows=rows, attention_dp=dp
    )
    weights = {
        "ln1": jax.ShapeDtypeStruct((c_draft.layers, 1, c_draft.hidden), jnp.bfloat16),
        "ln2": jax.ShapeDtypeStruct((c_draft.layers, 1, c_draft.hidden), jnp.bfloat16),
        "q_norm": jax.ShapeDtypeStruct(
            (c_draft.layers, 1, c_draft.q_lora), jnp.bfloat16
        ),
        "kv_norm": jax.ShapeDtypeStruct((c_draft.layers, 1, 640), jnp.bfloat16),
        "context_norm": jax.ShapeDtypeStruct((1, c_draft.hidden), jnp.bfloat16),
        "final_norm": jax.ShapeDtypeStruct((1, c_draft.hidden), jnp.bfloat16),
        "mask_embed": jax.ShapeDtypeStruct((1, c_draft.hidden), jnp.bfloat16),
        "rot": jax.ShapeDtypeStruct((c_draft.rope, c_draft.rope), jnp.bfloat16),
        "qkv_tile": jax.ShapeDtypeStruct(
            (c_draft.layers, c_draft.hidden, _CONST_128), jnp.bfloat16
        ),
        "context_tile": jax.ShapeDtypeStruct(
            (c_draft.hidden, _CONST_128), jnp.bfloat16
        ),
        "q_b": jax.ShapeDtypeStruct(
            (c_draft.layers, c_draft.q_lora, _CONST_512),
            jnp.bfloat16,
        ),
        "w_uk": jax.ShapeDtypeStruct(
            (c_draft.layers, _CONST_2, c_draft.nope, c_draft.kv_lora), jnp.bfloat16
        ),
        "w_uv": jax.ShapeDtypeStruct(
            (c_draft.layers, _CONST_2, c_draft.kv_lora, c_draft.v_head),
            jnp.bfloat16,
        ),
        "o": jax.ShapeDtypeStruct(
            (c_draft.layers, _CONST_2 * c_draft.v_head, c_draft.hidden), jnp.bfloat16
        ),
        "gate": jax.ShapeDtypeStruct(
            (c_draft.layers, c_draft.hidden, _CONST_512), jnp.bfloat16
        ),
        "up": jax.ShapeDtypeStruct(
            (c_draft.layers, c_draft.hidden, _CONST_512), jnp.bfloat16
        ),
        "down": jax.ShapeDtypeStruct(
            (c_draft.layers, _CONST_512, c_draft.hidden), jnp.bfloat16
        ),
        "fc": jax.ShapeDtypeStruct((8960, _CONST_896), jnp.bfloat16),
        "w2t": jax.ShapeDtypeStruct(
            (_CONST_256, c_draft.vocab // _CONST_32), jnp.bfloat16
        ),
        "w1": jax.ShapeDtypeStruct((c_draft.vocab, _CONST_256), jnp.bfloat16),
        "rope": jax.ShapeDtypeStruct(
            (_CONST_512, _CONST_2 * c_draft.rope), jnp.float32
        ),
        "lm_head": jax.ShapeDtypeStruct(
            (c_draft.hidden, c_draft.vocab // _CONST_32), jnp.bfloat16
        ),
        "embed": jax.ShapeDtypeStruct(
            (c_draft.vocab // _CONST_32, c_draft.hidden), jnp.bfloat16
        ),
    }
    pools = tuple(
        jax.ShapeDtypeStruct((_CONST_16, _CONST_128, 4, 640), jnp.float8_e4m3fn)
        for _ in range(c_draft.layers)
    )
    aux = jax.ShapeDtypeStruct(
        (len(c_draft.aux_layers), t, c_draft.hidden), jnp.bfloat16
    )
    logits = jax.ShapeDtypeStruct((r, rows, c_draft.vocab // _CONST_32), jnp.float32)
    previous = jax.ShapeDtypeStruct((r, _CONST_128), jnp.int32)
    position = jax.ShapeDtypeStruct((r,), jnp.int32)
    forced_count = jax.ShapeDtypeStruct((r,), jnp.int32)
    forced_anchor = jax.ShapeDtypeStruct((r,), jnp.int32)
    tables = jax.ShapeDtypeStruct((r, pages), jnp.int32)

    part_a_sharded = jax.shard_map(
        fused_draft,
        mesh=mesh_tp,
        in_specs=P(),
        out_specs=P(),
        check_vma=False,
    )
    lowered_draft = jax.jit(part_a_sharded).lower(
        weights,
        pools,
        aux,
        logits,
        previous,
        position,
        forced_count,
        forced_anchor,
        tables,
    )
    assert "tpu_custom_call" in lowered_draft.as_text()


def _lower_moe_down_valu_on_tpu7x(devices: _Dynamic) -> None:
    """Verify deviceless TPU v7x AOT lowering for moe_down_valu.down_valu."""
    mesh_sp = Mesh(devices, ("k3_sp",))
    tokens_val = _CONST_4096
    columns_val = _CONST_1792
    tile_val = _CONST_128
    unroll_val = _CONST_8
    capacity_val = _CONST_512
    size_k = 1536
    size_group = 56
    num_quant_blocks = size_k // _CONST_32

    x_val = jax.ShapeDtypeStruct((capacity_val, size_k), jnp.bfloat16)
    rhs_val = jax.ShapeDtypeStruct((size_group, size_k // _CONST_2, 7168), jnp.uint8)
    scales_val = jax.ShapeDtypeStruct((size_group, num_quant_blocks, 7168), jnp.uint8)
    counts_val = jax.ShapeDtypeStruct((size_group,), jnp.int32)
    destinations_val = jax.ShapeDtypeStruct((capacity_val,), jnp.int32)
    probabilities_val = jax.ShapeDtypeStruct((capacity_val,), jnp.float32)
    tables_val = jax.ShapeDtypeStruct((3, 3, _CONST_32), jnp.int32)

    def _part_b_wrapper(operands: tuple[_Dynamic, ...]) -> _Dynamic:
        """Wrap down valu call for shard map."""
        return moe_down_valu.down_valu(
            *operands,
            tokens=tokens_val,
            columns=columns_val,
            tile=tile_val,
            unroll=unroll_val,
            layer=None,
            stream_metadata=False,
            metadata_cache_rows=0,
            metadata_row_capacity=capacity_val,
            ring_dtype=jnp.float32,
        )

    part_b_sharded = jax.shard_map(
        _part_b_wrapper,
        mesh=mesh_sp,
        in_specs=P(),
        out_specs=P(),
        check_vma=False,
    )
    lowered_valu = jax.jit(part_b_sharded).lower(
        (
            x_val,
            rhs_val,
            scales_val,
            counts_val,
            destinations_val,
            probabilities_val,
            tables_val,
        )
    )
    lowered_text = lowered_valu.as_text()
    assert "tpu_custom_call" in lowered_text


def test_k3_deviceless_tpu7x_aot_lowering() -> _Dynamic:
    """Verify deviceless TPU v7x AOT lowering for fused draft and down VALU."""
    assert _pmk_init is not None
    devices = topologies.get_topology_desc(
        topology_name="tpu7x_2x2x4", platform="tpu"
    ).devices

    _lower_dspark_mla_on_tpu7x(devices)
    _lower_moe_down_valu_on_tpu7x(devices)

    mk = importlib.import_module("vllm_torchtpu.models.kimi_k3.megakernel")
    base_dir = pathlib.Path(mk.__file__).resolve().parent
    valu_source = (base_dir / "kernels/prefill_moe/moe_down_valu.py").read_text(
        encoding="utf-8"
    )
    assert (
        "lambda: pltpu.make_async_copy(payload, o_ref, store_sem)" not in valu_source
    ), "Final store DMA in moe_down_valu.py must call .start() and .wait()"
    assert "for step_idx in range(state[0], stop):" not in valu_source, (
        "Traced state[0] must not be used in Python range() in moe_down_valu.py"
    )
