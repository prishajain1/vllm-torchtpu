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
"""vLLM runtime adapter (`Qwen38MoeProgram`, `layout_request`, `create`) for Qwen3.8."""

from __future__ import annotations

import contextlib
import dataclasses
import itertools
from collections.abc import Mapping
from typing import TYPE_CHECKING, TypedDict, Unpack

import jax
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import torch

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.megakernel_runtime.jax_bridge import (
    jax_to_torch_cpu,
    torch_cpu_to_jax,
)
from vllm_torchtpu.megakernel_runtime.plugin import (
    Bucket,
    Decline,
    EligibilityResult,
    ForwardView,
    LayoutDecline,
    RunnerShape,
    RunnerView,
    StepDecision,
    StepFn,
    WarmupCase,
)

from . import kernel, plan, weights
from .shapes import Qwen38Config, mini_parity_config

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from jax.sharding import Mesh

logger = init_logger(__name__)
_OP_COUNTER = itertools.count()

_PROD_NUM_LAYERS: int = 92
_PROD_HIDDEN_SIZE: int = 8192
_MINI_HIDDEN_SIZE: int = 128
_EXPECTED_TP_SIZE: int = 32
_EXPECTED_DCP_SIZE: int = 8
_MIN_BROADCAST_NDIM: int = 2

_ERR_POOL_UNKNOWN: str = "pool_layout_unknown"
_ERR_POOL_MISMATCH: str = "pool_layout_mismatch"
_ERR_UNSUPPORTED_ARCH: str = "unsupported_arch"
_ERR_UNSUPPORTED_GEOM: str = "unsupported_geometry"
_ERR_UNSUPPORTED_QUANT: str = "unsupported_quant"
_ERR_UNSUPPORTED_TP: str = "unsupported_tp_size"
_ERR_UNSUPPORTED_DCP: str = "unsupported_dcp_size"
_ERR_UNALIGNED_CACHE: str = "unaligned_cache_layout"
_ERR_BACKBONE_NOT_FOUND: str = "backbone_not_found"
_ERR_EMPTY_BUCKETS: str = "empty_buckets"


class Qwen38AdapterError(ValueError):
    """Raised when adapter options or configuration validation fails."""


def _build_pallas_step_op(
    op_prefix: str,
    body_fn: Callable[..., tuple[jax.Array, ...]],
    *,
    mutated_arg_indices: tuple[int, ...],
    out_rows: int,
    hidden_size: int,
) -> Callable[..., torch.Tensor]:
    """Wrap `body_fn` in a `torch.library.custom_op` preserving pool `data_ptr()`."""
    op_id = next(_OP_COUNTER)
    qual_name = f"megakernel_runtime::{op_prefix}_{op_id}"

    @torch.library.custom_op(qual_name, mutates_args=())
    def _op(operands: list[torch.Tensor]) -> list[torch.Tensor]:
        """Execute underlying Pallas body function on tensor inputs."""
        jax_args = [
            torch_cpu_to_jax(t.view(torch.uint8) if idx in mutated_arg_indices else t)
            for idx, t in enumerate(operands)
        ]
        jax_outs = body_fn(*jax_args)
        pool_outs = jax_outs[:-1]
        x_out = jax_outs[-1]
        torch_pools = [
            jax_to_torch_cpu(p, operands[idx].view(torch.uint8)).to(
                device=operands[idx].device
            )
            for p, idx in zip(pool_outs, mutated_arg_indices, strict=True)
        ]
        torch_x = jax_to_torch_cpu(x_out, operands[0]).to(device=operands[0].device)
        return [*torch_pools, torch_x]

    @_op.register_fake
    def _fake(operands: list[torch.Tensor]) -> list[torch.Tensor]:
        """Register fake kernel output shapes for tracing."""
        x_t = operands[0]
        fake_pools = [
            torch.empty_like(operands[idx].view(torch.uint8))
            for idx in mutated_arg_indices
        ]
        fake_x = x_t.new_empty((out_rows, hidden_size))
        return [*fake_pools, fake_x]

    def _step(*args: torch.Tensor) -> torch.Tensor:
        """Execute custom op and mutate pool buffers in place."""
        outs = _op(list(args))
        for p_out, idx in zip(outs[:-1], mutated_arg_indices, strict=True):
            args[idx].view(torch.uint8).copy_(p_out)
        return outs[-1]

    return _step


OPTION_KEYS: frozenset[str] = frozenset(
    {"buckets", "compile_hlo_check", "allow_mini_geometry", "interpret"}
)
DEFAULT_BUCKETS: tuple[int, ...] = (1, 2, 4, 8, 16)
SUPPORTED_ARCHS: frozenset[str] = frozenset(
    {
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen3_5MoeForCausalLM",
    }
)
SUPPORTED_MODEL_TYPES: frozenset[str] = frozenset(
    {
        "qwen3_5_moe",
        "qwen3_5_moe_text",
    }
)
MAMBA_SLOT_TOKENS: int = kernel.PAGES_PER_GDN_SLOT * 128


def layout_request(
    vllm_config: object, options: Mapping[str, object] | None = None
) -> dict[str, int]:
    """Request 128-token KV pages and 640-token (5-page) Mamba state slots."""
    del vllm_config, options
    return {"block_size": 128, "mamba_slot_tokens": MAMBA_SLOT_TOKENS}


def _parse_buckets(raw: object) -> tuple[int, ...]:
    """Parse comma-separated or sequence bucket sizes."""
    if raw is None or raw == "":
        return DEFAULT_BUCKETS
    if isinstance(raw, (list, tuple)):
        return tuple(sorted({int(x) for x in raw if int(x) > 0}))
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    return tuple(sorted({int(p) for p in parts if int(p) > 0}))


def _dcp_size(view: RunnerView) -> int:
    """Extract decode context parallel size from runner view."""
    r = getattr(view, "_runner", None)
    for cfg in (
        getattr(r, "parallel_config", None),
        getattr(getattr(view, "vllm_config", None), "parallel_config", None),
    ):
        val = getattr(cfg, "decode_context_parallel_size", None)
        if val is not None:
            return int(val)
    return 1


def _quant_method(hf_cfg: object) -> str:
    """Extract quantization method from HuggingFace config."""
    if hf_cfg is None:
        return "nvfp4"
    q_cfg = getattr(hf_cfg, "quantization_config", None)
    if isinstance(q_cfg, Mapping):
        return str(q_cfg.get("quant_method") or "nvfp4").lower()
    if q_cfg is not None:
        return str(getattr(q_cfg, "quant_method", "nvfp4") or "nvfp4").lower()
    return "nvfp4"


def _effective_slot_tokens(view: RunnerView) -> int:
    """Determine effective token count per Mamba state slot."""
    r = getattr(view, "_runner", None)
    vllm_cfg = getattr(view, "vllm_config", None) or getattr(r, "vllm_config", None)
    for cfg in (
        getattr(vllm_cfg, "cache_config", None),
        getattr(r, "cache_config", None),
    ):
        for attr in ("_megakernel_mamba_slot_tokens", "mamba_block_size"):
            slot_toks = getattr(cfg, attr, None)
            if slot_toks:
                return int(slot_toks)
    applied = getattr(vllm_cfg, "_megakernel_layout_applied", None)
    if isinstance(applied, Mapping) and applied.get("mamba_slot_tokens"):
        return int(applied["mamba_slot_tokens"])
    add_cfg = getattr(vllm_cfg, "additional_config", None)
    if isinstance(add_cfg, Mapping) and add_cfg.get("mamba_slot_tokens"):
        return int(add_cfg["mamba_slot_tokens"])
    bs = view.cache_block_size
    return int(bs) if bs else 0


class ProgramOptions(TypedDict, total=False):
    """Optional settings for Qwen38MoeProgram."""

    tp_rank: int
    decode_buckets: Sequence[int]
    compile_hlo_check: bool
    base_config: Qwen38Config | None
    interpret: bool | pltpu.InterpretParams | None


def _collect_attention_pools(
    backbone: torch.nn.Module, cfg: Qwen38Config
) -> tuple[list[torch.Tensor], list[str]]:
    """Collect and validate KV cache pools for attention layers."""
    pools: list[torch.Tensor] = []
    attn_names: list[str] = []
    for l_idx in range(cfg.L):
        if (l_idx + 1) % cfg.att_every != 0:
            continue
        layer = backbone.layers[l_idx]
        sa = getattr(layer, "self_attn", None)
        attn_mod = getattr(sa, "attn", sa)
        name = (
            getattr(attn_mod, "layer_name", None)
            or getattr(sa, "prefix", None)
            or f"model.layers.{l_idx}.self_attn.attn"
        )
        attn_names.append(str(name))
        kv = getattr(attn_mod, "kv_cache", None)
        if isinstance(kv, (list, tuple)):
            kv = kv[0] if kv else None
        if not isinstance(kv, torch.Tensor):
            err_msg = f"Missing attention kv_cache tensor on layer {l_idx}"
            raise LayoutDecline(_ERR_POOL_UNKNOWN, err_msg)
        valid_tails = {
            (2, cfg.K_R, cfg.D, cfg.page_size),
            (2, cfg.K_R, 4, cfg.D),
        }
        if kv.ndim not in (5, 6) or tuple(kv.shape[1:5]) not in valid_tails:
            err_msg = (
                f"Layer {l_idx} kv_cache shape {tuple(kv.shape)} != "
                f"(num_pages, 2, {cfg.K_R}, {cfg.D}, {cfg.page_size})"
            )
            raise LayoutDecline(_ERR_POOL_MISMATCH, err_msg)
        pools.append(kv)
    return pools, attn_names


def _collect_mamba_names(backbone: torch.nn.Module, cfg: Qwen38Config) -> list[str]:
    """Collect layer names for GDN linear attention layers."""
    mamba_names: list[str] = []
    for l_idx in range(cfg.L):
        if (l_idx + 1) % cfg.att_every == 0:
            continue
        layer = backbone.layers[l_idx]
        la = getattr(layer, "linear_attn", None)
        name = (
            getattr(la, "prefix", None)
            or getattr(la, "layer_name", None)
            or f"model.layers.{l_idx}.linear_attn"
        )
        mamba_names.append(str(name))
    return mamba_names


def _verify_gdn_pool_aliasing(
    backbone: torch.nn.Module, pools: list[torch.Tensor], cfg: Qwen38Config
) -> None:
    """Verify that GDN states share storage with corresponding attention regions."""
    for l_idx in range(cfg.L):
        if (l_idx + 1) % cfg.att_every == 0:
            continue
        la = getattr(backbone.layers[l_idx], "linear_attn", None)
        la_kv = getattr(la, "kv_cache", None)
        if isinstance(la_kv, (list, tuple)):
            la_kv = la_kv[0] if la_kv else None
        if isinstance(la_kv, torch.Tensor):
            reg_pool = pools[l_idx // cfg.att_every]
            if la_kv.data_ptr() != reg_pool.data_ptr():
                err_msg = (
                    f"GDN layer {l_idx} pool is not aliased with attention region "
                    f"{l_idx // cfg.att_every}"
                )
                raise LayoutDecline(_ERR_POOL_MISMATCH, err_msg)


def _prepare_input_tensors(
    ctx: object,
    batch_dims: tuple[int, int],
    cfg: Qwen38Config,
    dev: torch.device,
    backbone: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare padded activations and position vectors for step execution."""
    n_reqs, b_dim = batch_dims
    inputs_embeds = getattr(ctx, "inputs_embeds", None)
    input_ids = getattr(ctx, "input_ids", None)
    if inputs_embeds is not None:
        x_raw = inputs_embeds[:n_reqs]
    else:
        embed_fn = getattr(backbone, "embed_tokens", None)
        if callable(embed_fn) and input_ids is not None:
            x_raw = embed_fn(input_ids[:n_reqs])
        else:
            x_raw = torch.zeros((n_reqs, cfg.H), dtype=torch.bfloat16, device=dev)
            if input_ids is not None:
                ids_f = (
                    input_ids[:n_reqs]
                    .to(device=dev, dtype=torch.bfloat16)
                    .unsqueeze(-1)
                )
                x_raw = x_raw + ids_f * 0.01

    if x_raw.shape[0] < b_dim:
        pad_x = torch.zeros(
            (b_dim - x_raw.shape[0], cfg.H),
            dtype=torch.bfloat16,
            device=dev,
        )
        x_in = torch.cat([x_raw.to(torch.bfloat16), pad_x], dim=0)
    else:
        x_in = x_raw[:b_dim].to(torch.bfloat16).contiguous()

    positions = getattr(ctx, "positions", None)
    if positions is not None:
        pos_raw = positions[:n_reqs].to(device=dev, dtype=torch.int32).reshape(-1)
    else:
        pos_raw = torch.zeros((n_reqs,), dtype=torch.int32, device=dev)

    if pos_raw.shape[0] < b_dim:
        pad_p = torch.full(
            (b_dim - pos_raw.shape[0],), -1, dtype=torch.int32, device=dev
        )
        pos = torch.cat([pos_raw, pad_p], dim=0)
    else:
        pos = pos_raw[:b_dim].contiguous()
    return x_in, pos


def _prepare_page_table(
    attn_md: Mapping[str, object],
    attn_layer_names: Sequence[str],
    dims: tuple[int, int, int],
    dev: torch.device,
) -> torch.Tensor:
    """Extract and pad page table indices across attention layers."""
    n_reqs, b_dim, max_pages = dims
    pt_rows: list[torch.Tensor] = []
    for a_name in attn_layer_names:
        md_a = attn_md.get(a_name)
        bt = getattr(md_a, "block_tables", getattr(md_a, "block_table", None))
        if bt is None:
            bt_pad = torch.zeros((b_dim, max_pages), dtype=torch.int32, device=dev)
        else:
            bt_t = bt[:n_reqs, :max_pages].to(device=dev, dtype=torch.int32)
            if bt_t.shape[0] < b_dim or bt_t.shape[1] < max_pages:
                bt_pad = torch.zeros((b_dim, max_pages), dtype=torch.int32, device=dev)
                bt_pad[: bt_t.shape[0], : bt_t.shape[1]] = bt_t
            else:
                bt_pad = bt_t.contiguous()
        pt_rows.append(bt_pad)
    return torch.stack(pt_rows, dim=0)


def _prepare_mamba_slots(
    attn_md: Mapping[str, object],
    mamba_layer_names: Sequence[str],
    n_reqs: int,
    b_dim: int,
    dev: torch.device,
) -> torch.Tensor:
    """Extract and pad Mamba slot indices across GDN layers."""
    m_rows: list[torch.Tensor] = []
    for m_name in mamba_layer_names:
        md_m = attn_md.get(m_name)
        ms = getattr(md_m, "mamba_state_indices", None)
        if ms is None:
            ms_pad = torch.full((b_dim,), -1, dtype=torch.int32, device=dev)
        else:
            ms_t = ms[:n_reqs].to(device=dev, dtype=torch.int32).reshape(-1)
            if ms_t.shape[0] < b_dim:
                ms_pad = torch.full((b_dim,), -1, dtype=torch.int32, device=dev)
                ms_pad[: ms_t.shape[0]] = ms_t
            else:
                ms_pad = ms_t[:b_dim].contiguous()
        m_rows.append(ms_pad)
    return torch.stack(m_rows, dim=0)


class Qwen38MoeProgram:
    """`MonoProgram` implementation for Qwen3.8-2.4T TP32xDCP8 NVFP4."""

    display_name = "Qwen3.8-2.4T"

    def __init__(
        self,
        model: torch.nn.Module,
        **options: Unpack[ProgramOptions],
    ) -> None:
        """Initialize Qwen3.8-2.4T megakernel program instance."""
        self.model = model
        self.backbone = weights.find_qwen38_2_4t_backbone(model)
        self.tp_rank = int(options.get("tp_rank", 0))
        decode_buckets = options.get("decode_buckets", DEFAULT_BUCKETS)
        self.decode_buckets = tuple(int(b) for b in decode_buckets)
        self._compile_hlo_check = bool(options.get("compile_hlo_check", False))
        self.base_config = options.get("base_config") or Qwen38Config()
        self._interpret = options.get("interpret")

        self.mesh: Mesh | None = None
        self.w_dict: dict[str, torch.Tensor] = {}
        self.w_flat: list[torch.Tensor] = []
        self.attn_layer_names: list[str] = []
        self.mamba_layer_names: list[str] = []
        self.region_pools: list[torch.Tensor] = []
        self.rank_pools: list[list[torch.Tensor]] | None = None
        self._pool_snapshots: list[list[torch.Tensor]] | None = None

    def buckets(self, token_paddings: Sequence[int]) -> Sequence[Bucket]:
        """Compute decode buckets covering requested token padding thresholds."""
        pads = [int(p) for p in token_paddings if int(p) > 0]
        result: list[Bucket] = []
        for b in self.decode_buckets:
            out_rows = next((p for p in pads if p >= b), b)
            result.append(Bucket(T=1, B=b, out_rows=out_rows))
        return result

    def load(self, mesh: Mesh | None) -> None:
        """Bind device mesh and extract normalized weights for Pallas execution."""
        self.mesh = mesh
        norm_w = getattr(getattr(self.backbone, "norm", None), "weight", None)
        device = norm_w.device if isinstance(norm_w, torch.Tensor) else "cpu"
        self.w_dict = weights.extract_backbone_weights_torch(
            self.model, self.base_config, device=device
        )
        self.w_flat = [self.w_dict[k] for k in weights.WEIGHT_KEYS]

    def _collect_region_pools(self, backbone: torch.nn.Module) -> list[torch.Tensor]:
        """Collect and validate unified KV/GDN region cache pools."""
        cfg = self.base_config
        pools, attn_names = _collect_attention_pools(backbone, cfg)
        mamba_names = _collect_mamba_names(backbone, cfg)
        _verify_gdn_pool_aliasing(backbone, pools, cfg)

        if not self.attn_layer_names:
            self.attn_layer_names = attn_names
            self.mamba_layer_names = mamba_names
        return pools

    def bind(self, view: RunnerView) -> dict[str, int]:
        """Inspect runner view cache layout and allocate region pools."""
        slot_tokens = _effective_slot_tokens(view)
        if slot_tokens < MAMBA_SLOT_TOKENS:
            err_msg = (
                f"Qwen3.8 GDN state requires >= {MAMBA_SLOT_TOKENS} tokens "
                f"(5 pages) per slot, got cache_block_size={slot_tokens}"
            )
            raise LayoutDecline(_ERR_POOL_MISMATCH, err_msg)

        rank_backbones = getattr(self.model, "_rank_backbones", None)
        if rank_backbones is not None:
            self.rank_pools = [self._collect_region_pools(b) for b in rank_backbones]
            self.region_pools = self.rank_pools[0]
        else:
            self.region_pools = self._collect_region_pools(self.backbone)
            self.rank_pools = None

        num_pages = int(self.region_pools[0].shape[0])
        self.base_config = dataclasses.replace(
            self.base_config,
            num_pages=num_pages,
            max_pages_per_seq=int(view.max_pages_per_seq),
        )
        return {
            "num_pages": num_pages,
            "pages_per_gdn_slot": kernel.PAGES_PER_GDN_SLOT,
            "num_regions": len(self.region_pools),
        }

    def check_compiled_ir(self, debug_info: object, bucket: Bucket) -> None:
        """Verify compiled HLO contains no forbidden all-gather operations."""
        if not self._compile_hlo_check:
            return
        texts = getattr(debug_info, "stablehlo_forward_text", None) or ()
        for idx, text in enumerate(texts):
            plan.assert_no_all_gather(
                str(text), context=f"bucket=({bucket.T},{bucket.B}) graph={idx}"
            )

    @contextlib.contextmanager
    def warmup_session(self) -> Iterator[None]:
        """Context manager preserving cache pools during warmup compilation."""
        if self.rank_pools is not None:
            snaps = [
                [p.view(torch.uint8).clone() for p in r_pools]
                for r_pools in self.rank_pools
            ]
        else:
            snaps = [[p.view(torch.uint8).clone() for p in self.region_pools]]
        self._pool_snapshots = snaps
        try:
            yield
        finally:
            self._restore_pools()
            self._pool_snapshots = None

    def _restore_pools(self) -> None:
        """Restore initial cache pool tensor buffers from snapshot."""
        if self._pool_snapshots is None:
            return
        if self.rank_pools is not None:
            for r_pools, r_snaps in zip(
                self.rank_pools, self._pool_snapshots, strict=True
            ):
                for p, s in zip(r_pools, r_snaps, strict=True):
                    p.view(torch.uint8).copy_(s)
        else:
            for p, s in zip(self.region_pools, self._pool_snapshots[0], strict=True):
                p.view(torch.uint8).copy_(s)

    def warmup_inputs(self, bucket: Bucket, shape: RunnerShape) -> WarmupCase | None:
        """Construct synthetic dummy inputs for compile-time trace execution."""
        if self._interpret is not None or jax.devices()[0].platform == "cpu":
            return None
        cfg = self.base_config
        b_dim = int(bucket.B)
        dev = self.region_pools[0].device
        x_in = torch.zeros((b_dim, cfg.H), dtype=torch.bfloat16, device=dev)
        pos = torch.zeros((b_dim,), dtype=torch.int32, device=dev)
        page_table = torch.zeros(
            (cfg.NATT, b_dim, shape.blk_pad), dtype=torch.int32, device=dev
        )
        g_per_reg = max(1, cfg.att_every - 1)
        if cfg.num_pages >= (4 + g_per_reg * b_dim) * kernel.PAGES_PER_GDN_SLOT:
            local_g = (
                torch.arange(cfg.NGDN, dtype=torch.int32, device=dev) % g_per_reg
            )[:, None]
            b_idx = torch.arange(b_dim, dtype=torch.int32, device=dev)[None, :]
            mamba_slots = 4 + local_g * b_dim + b_idx
        else:
            mamba_slots = torch.full(
                (cfg.NGDN, b_dim), -1, dtype=torch.int32, device=dev
            )

        if self.rank_pools is not None:
            flat_pools = [p for r_pools in self.rank_pools for p in r_pools]
        else:
            flat_pools = list(self.region_pools)

        args = (x_in, pos, page_table, mamba_slots, *flat_pools, *self.w_flat)

        def _check(_out: object) -> None:
            """Restore pools after warmup step execution."""
            self._restore_pools()

        return WarmupCase(args=args, check=_check)

    def build(self, bucket: Bucket) -> StepFn:
        """Compile and return the executable PyTorch step function for `bucket`."""
        cfg = dataclasses.replace(self.base_config, B=int(bucket.B))
        out_rows = int(bucket.out_rows or bucket.B)
        multi_rank_host = self.rank_pools is not None
        num_pool_args = cfg.R * cfg.NATT if multi_rank_host else cfg.NATT

        if self._interpret is not None:
            interpret = self._interpret
        else:
            interpret = jax.devices()[0].platform == "cpu"

        def _body(*jax_args: jax.Array) -> tuple[jax.Array, ...]:
            """Execute sharded Pallas megakernel step on JAX arrays."""
            x_in = jax_args[0]
            pos = jax_args[1]
            page_table = jax_args[2]
            mamba_slots = jax_args[3]
            pools_in = list(jax_args[4 : 4 + num_pool_args])
            w_list = list(jax_args[4 + num_pool_args :])
            w_raw = dict(zip(weights.WEIGHT_KEYS, w_list, strict=True))

            if multi_rank_host:
                pools_by_reg = [
                    jnp.stack(
                        [pools_in[r * cfg.NATT + reg] for r in range(cfg.R)], axis=0
                    )
                    for reg in range(cfg.NATT)
                ]
                conv_init, rec_init = kernel.unpack_gdn_states_from_pools(
                    pools_by_reg, mamba_slots, cfg
                )
                pool_all_regs = jnp.stack(pools_by_reg, axis=1).view(jnp.float8_e4m3fn)
                prob = {
                    "x": jnp.broadcast_to(x_in[None, ...], (cfg.R, cfg.B, cfg.H)),
                    "pos": jnp.broadcast_to(pos[None, :], (cfg.R, cfg.B)),
                    "page_table": jnp.broadcast_to(
                        page_table[None, :, :, : cfg.max_pages_per_seq],
                        (cfg.R, cfg.NATT, cfg.B, cfg.max_pages_per_seq),
                    ),
                    "pool": pool_all_regs,
                    "conv": conv_init,
                    "rec": rec_init,
                    "weights": w_raw,
                }
                res = kernel.run_qwen38_dcp8_step(
                    prob,
                    cfg,
                    use_forced_routing=False,
                    return_kv_updates=True,
                    interpret=interpret,
                    mesh=self.mesh,
                )
                pool_updated_u8 = res["pool"].view(jnp.uint8)
                updated_by_reg = [pool_updated_u8[:, reg] for reg in range(cfg.NATT)]
                packed_by_reg = kernel.pack_gdn_states_into_pools(
                    updated_by_reg, res["conv"], res["rec"], mamba_slots, cfg
                )
                pools_out = [
                    packed_by_reg[reg][r]
                    for r in range(cfg.R)
                    for reg in range(cfg.NATT)
                ]
                x_res = res["x"][0]
            else:
                conv_1, rec_1 = kernel.unpack_gdn_states_from_pools(
                    pools_in, mamba_slots, cfg
                )
                pool_1_regs = jnp.stack(pools_in, axis=0).view(jnp.float8_e4m3fn)
                w_bcast = {
                    k: (
                        jnp.broadcast_to(v, (cfg.R, *v.shape[1:]))
                        if v.ndim >= _MIN_BROADCAST_NDIM and v.shape[0] == 1
                        else (
                            v
                            if v.shape[0] == cfg.R
                            else jnp.broadcast_to(v[None, ...], (cfg.R, *v.shape))
                        )
                    )
                    for k, v in w_raw.items()
                }
                prob = {
                    "x": jnp.broadcast_to(x_in[None, ...], (cfg.R, cfg.B, cfg.H)),
                    "pos": jnp.broadcast_to(pos[None, :], (cfg.R, cfg.B)),
                    "page_table": jnp.broadcast_to(
                        page_table[None, :, :, : cfg.max_pages_per_seq],
                        (cfg.R, cfg.NATT, cfg.B, cfg.max_pages_per_seq),
                    ),
                    "pool": jnp.broadcast_to(
                        pool_1_regs[None, ...], (cfg.R, *pool_1_regs.shape)
                    ),
                    "conv": jnp.broadcast_to(conv_1[None, ...], (cfg.R, *conv_1.shape)),
                    "rec": jnp.broadcast_to(rec_1[None, ...], (cfg.R, *rec_1.shape)),
                    "weights": w_bcast,
                }
                res = kernel.run_qwen38_dcp8_step(
                    prob,
                    cfg,
                    use_forced_routing=False,
                    return_kv_updates=True,
                    interpret=interpret,
                    mesh=self.mesh,
                )
                r_self = min(max(self.tp_rank, 0), cfg.R - 1)
                pool_self_u8 = res["pool"][r_self].view(jnp.uint8)
                updated_regs = [pool_self_u8[reg] for reg in range(cfg.NATT)]
                pools_out = kernel.pack_gdn_states_into_pools(
                    updated_regs,
                    res["conv"][r_self],
                    res["rec"][r_self],
                    mamba_slots,
                    cfg,
                )
                x_res = res["x"][r_self]

            if out_rows > cfg.B:
                pad = jnp.zeros((out_rows - cfg.B, cfg.H), dtype=x_res.dtype)
                x_res = jnp.concatenate([x_res, pad], axis=0)
            return (*pools_out, x_res)

        mutated_indices = tuple(range(4, 4 + num_pool_args))

        op_fn = _build_pallas_step_op(
            f"qwen38_2_4t_t1_b{cfg.B}_r{out_rows}",
            _body,
            mutated_arg_indices=mutated_indices,
            out_rows=out_rows,
            hidden_size=cfg.H,
        )

        def _step(
            x_in: torch.Tensor,
            pos: torch.Tensor,
            page_table: torch.Tensor,
            mamba_slots: torch.Tensor,
            *rest: torch.Tensor,
        ) -> torch.Tensor:
            """Run custom PyTorch step op for current bucket."""
            return op_fn(x_in, pos, page_table, mamba_slots, *rest)

        return _step

    def decide(self, _ctx: object, fwd: ForwardView) -> StepDecision | None:
        """Evaluate request batch eligibility and select target decode bucket."""
        attn_md = fwd.attn_metadata
        if not isinstance(attn_md, Mapping) or not self.attn_layer_names:
            return StepDecision(
                eligible=False, reason="stock_no_metadata", num_reqs=0, tokens_per_seq=0
            )

        a0 = attn_md.get(self.attn_layer_names[0])
        m0 = attn_md.get(self.mamba_layer_names[0]) if self.mamba_layer_names else None
        shape = fwd.batch_shape(a0, m0)
        if not getattr(shape, "eligible", False):
            return StepDecision(
                eligible=False, reason="stock_ineligible", num_reqs=0, tokens_per_seq=0
            )

        num_reqs = int(getattr(shape, "num_reqs", 0) or 0)
        toks_per_seq = int(getattr(shape, "tokens_per_seq", 0) or 0)
        if toks_per_seq != 1 or num_reqs <= 0:
            return StepDecision(
                eligible=False,
                reason="stock_prefill",
                num_reqs=num_reqs,
                tokens_per_seq=toks_per_seq,
            )

        bucket_b = next((b for b in self.decode_buckets if b >= num_reqs), None)
        if bucket_b is None or (1, bucket_b) not in fwd.steps:
            return StepDecision(
                eligible=False,
                reason=f"stock_t1_b{num_reqs}",
                num_reqs=num_reqs,
                tokens_per_seq=1,
            )
        return StepDecision(
            eligible=True,
            reason=f"mk_t1_b{bucket_b}",
            num_reqs=num_reqs,
            tokens_per_seq=1,
            bucket_t=bucket_b,
        )

    def forward(
        self, decision: StepDecision, ctx: object, fwd: ForwardView
    ) -> torch.Tensor:
        """Execute forward step using megakernel or dispatch to fallback."""
        cfg = self.base_config
        b_dim = int(decision.bucket_t or decision.num_reqs)
        n_reqs = int(decision.num_reqs)
        attn_md = fwd.attn_metadata or {}
        dev = self.region_pools[0].device

        x_in, pos = _prepare_input_tensors(
            ctx, (n_reqs, b_dim), cfg, dev, self.backbone
        )
        page_table = _prepare_page_table(
            attn_md,
            self.attn_layer_names,
            (n_reqs, b_dim, cfg.max_pages_per_seq),
            dev,
        )
        mamba_slots = _prepare_mamba_slots(
            attn_md, self.mamba_layer_names, n_reqs, b_dim, dev
        )

        if self.rank_pools is not None:
            flat_pools = [p for r_pools in self.rank_pools for p in r_pools]
        else:
            flat_pools = list(self.region_pools)

        step_fn = fwd.steps[(1, b_dim)]
        out = step_fn(x_in, pos, page_table, mamba_slots, *flat_pools, *self.w_flat)
        out_rows = int(fwd.num_tokens or n_reqs)
        return out[:out_rows]


def _validate_hf_architecture(
    text_cfg: object, hf_cfg: object
) -> EligibilityResult | None:
    """Validate architectures and model type."""
    archs = tuple(getattr(hf_cfg, "architectures", None) or ())
    mtype = str(
        getattr(text_cfg, "model_type", None)
        or getattr(hf_cfg, "model_type", None)
        or ""
    )
    if archs and not (set(archs) & SUPPORTED_ARCHS):
        return Decline(
            _ERR_UNSUPPORTED_ARCH,
            f"Qwen3.8-2.4T plugin does not support architectures {archs}",
        )
    if mtype and mtype not in SUPPORTED_MODEL_TYPES:
        return Decline(
            _ERR_UNSUPPORTED_ARCH,
            f"Qwen3.8-2.4T plugin does not support model_type={mtype!r}",
        )
    return None


def _validate_hf_geometry(
    text_cfg: object, options: Mapping[str, object]
) -> EligibilityResult | None:
    """Validate hidden layer dimensions and expert geometry."""
    num_layers = int(
        getattr(text_cfg, "num_hidden_layers", _PROD_NUM_LAYERS) or _PROD_NUM_LAYERS
    )
    hidden_size = int(
        getattr(text_cfg, "hidden_size", _PROD_HIDDEN_SIZE) or _PROD_HIDDEN_SIZE
    )
    allow_mini = bool(options.get("allow_mini_geometry", False))
    is_prod = num_layers == _PROD_NUM_LAYERS and hidden_size == _PROD_HIDDEN_SIZE
    is_mini = allow_mini and (num_layers in (4, 8)) and hidden_size == _MINI_HIDDEN_SIZE
    if not (is_prod or is_mini):
        return Decline(
            _ERR_UNSUPPORTED_GEOM,
            f"Qwen3.8-2.4T requires num_hidden_layers=92, hidden_size=8192 "
            f"(got {num_layers}, {hidden_size})",
        )

    ref_cfg = mini_parity_config() if is_mini else Qwen38Config()
    geom_checks = (
        ("num_attention_heads", ref_cfg.Q_H),
        ("num_key_value_heads", ref_cfg.KV_H),
        ("linear_num_key_heads", ref_cfg.KH),
        ("linear_num_value_heads", ref_cfg.VH),
        ("num_experts", ref_cfg.E),
        ("n_routed_experts", ref_cfg.E),
        ("num_experts_per_tok", ref_cfg.K),
    )
    for attr, expected_val in geom_checks:
        actual_val = getattr(text_cfg, attr, None)
        if actual_val is not None and int(actual_val) != expected_val:
            return Decline(
                _ERR_UNSUPPORTED_GEOM,
                f"Qwen3.8-2.4T requires {attr}={expected_val}, got {actual_val}",
            )
    return None


def _validate_hf_config(
    view: RunnerView, options: Mapping[str, object]
) -> EligibilityResult | None:
    """Validate HuggingFace model architectures, geometry, and quantization."""
    hf_cfg = view.hf_config
    if hf_cfg is None:
        return None
    text_cfg = getattr(hf_cfg, "text_config", hf_cfg)
    arch_err = _validate_hf_architecture(text_cfg, hf_cfg)
    if arch_err is not None:
        return arch_err
    geom_err = _validate_hf_geometry(text_cfg, options)
    if geom_err is not None:
        return geom_err

    q_method = _quant_method(hf_cfg)
    if q_method not in ("nvfp4", "fp4", "compressed-tensors", "modelopt"):
        return Decline(
            _ERR_UNSUPPORTED_QUANT,
            f"Qwen3.8-2.4T requires NVFP4 quantization, got {q_method!r}",
        )
    return None


def _validate_runner_view(view: RunnerView) -> EligibilityResult | None:
    """Validate parallelism topology, cache layout, and backbone availability."""
    tp_size = int(getattr(view, "tp_size", 0) or 0)
    if tp_size != _EXPECTED_TP_SIZE:
        return Decline(
            _ERR_UNSUPPORTED_TP,
            f"Qwen3.8-2.4T requires tp_size=32, got {tp_size}",
        )

    dcp_size = _dcp_size(view)
    if dcp_size != _EXPECTED_DCP_SIZE:
        return Decline(
            _ERR_UNSUPPORTED_DCP,
            f"Qwen3.8-2.4T requires decode_context_parallel_size=8, got {dcp_size}",
        )

    if not view.unified_kv_layout:
        return Decline(
            _ERR_UNALIGNED_CACHE,
            "Qwen3.8-2.4T requires unified KV/GDN layout (_unified_kv_layout=True)",
        )

    try:
        weights.find_qwen38_2_4t_backbone(view.model)
    except ValueError as exc:
        return Decline(_ERR_BACKBONE_NOT_FOUND, str(exc))
    return None


def create(
    view: RunnerView, options: Mapping[str, object]
) -> Qwen38MoeProgram | EligibilityResult:
    """Validate runner/checkpoint eligibility and construct `Qwen38MoeProgram`."""
    unknown = sorted(set(options) - OPTION_KEYS)
    if unknown:
        valid_keys = sorted(OPTION_KEYS)
        err_msg = f"unknown option {unknown[0]!r} for qwen38_2_4t (valid: {valid_keys})"
        raise Qwen38AdapterError(err_msg)

    hf_err = _validate_hf_config(view, options)
    if hf_err is not None:
        return hf_err

    runner_err = _validate_runner_view(view)
    if runner_err is not None:
        return runner_err

    buckets = _parse_buckets(options.get("buckets"))
    if not buckets:
        return Decline(_ERR_EMPTY_BUCKETS, "No valid decode buckets configured")

    hf_cfg = view.hf_config
    text_cfg = getattr(hf_cfg, "text_config", hf_cfg) if hf_cfg is not None else None
    num_layers_cfg = int(
        getattr(text_cfg, "num_hidden_layers", _PROD_NUM_LAYERS) or _PROD_NUM_LAYERS
    )
    base_cfg = (
        dataclasses.replace(mini_parity_config(batch_size=buckets[0]), L=num_layers_cfg)
        if num_layers_cfg in (4, 8)
        else Qwen38Config(B=buckets[0])
    )
    interpret_opt = options.get("interpret")
    interpret_val: bool | pltpu.InterpretParams | None = None
    if interpret_opt is True:
        interpret_val = pltpu.InterpretParams(
            detect_races=False, out_of_bounds_reads="raise"
        )
    elif isinstance(interpret_opt, (bool, pltpu.InterpretParams)):
        interpret_val = interpret_opt

    return Qwen38MoeProgram(
        view.model,
        tp_rank=view.tp_rank,
        decode_buckets=buckets,
        compile_hlo_check=bool(options.get("compile_hlo_check", False)),
        base_config=base_cfg,
        interpret=interpret_val,
    )
