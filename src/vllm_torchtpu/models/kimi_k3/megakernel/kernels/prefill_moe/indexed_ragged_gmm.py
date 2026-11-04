# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Indexed activation loads into the ragged GMM pipeline.

Source provenance: Adapted from inferact/k3-megakernel@30c9ba966b and
vllm_torchtpu/kernels/megablox/gmm_v2.py@644426d851.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.models.kimi_k3.megakernel.kernels import prefill_megakernel as k

_Dynamic = Any


class PackedRows:
    """Load only the static row bucket selected by the GMM."""

    def __init__(self, ref: _Dynamic) -> None:
        """Execute init logic."""
        self.ref = ref

    def reshape(self, rows: _Dynamic, width: _Dynamic) -> _Dynamic:
        """Execute reshape logic."""
        if not (rows == -1 and width == 2 * self.ref.shape[-1]):
            _err_msg = (
                "Assertion failed: rows == -1 and width == 2 * self.ref.shape[-1]"
            )
            raise AssertionError(_err_msg)
        return self

    def __getitem__(self, rows: _Dynamic) -> _Dynamic:
        """Execute getitem logic."""
        return self.ref[rows].reshape(-1, 2 * self.ref.shape[-1])


_PackedRows = PackedRows


def _indexed_gate_up_kernel(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Execute kernel logic for indexed gate up."""
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        counts,
        offset,
        layer,
        idx,
        x,
        weights,
        out,
        partial,
        acc,
        metadata,
        acts,
        sems,
    ) = (
        _p_next(_p_iter, _p_get("counts")),
        _p_next(_p_iter, _p_get("offset")),
        _p_next(_p_iter, _p_get("layer")),
        _p_next(_p_iter, _p_get("idx")),
        _p_next(_p_iter, _p_get("x")),
        _p_next(_p_iter, _p_get("weights")),
        _p_next(_p_iter, _p_get("out")),
        _p_next(_p_iter, _p_get("partial")),
        _p_next(_p_iter, _p_get("acc")),
        _p_next(_p_iter, _p_get("metadata")),
        _p_next(_p_iter, _p_get("acts")),
        _p_next(_p_iter, _p_get("sems")),
    )
    cfg, sub, tn = _p_kw["cfg"], _p_kw["sub"], _p_kw["tn"]
    weights = dataclasses.replace(
        weights,
        weight=weights.weight.at[layer[0]].bitcast(jnp.float4_e2m1fn),
        scale=weights.scale.at[layer[0]],
    )
    ng = k.fill_metadata(counts, offset, metadata, cfgs=cfg)
    (_, ws), ys = k.generate_block_specs(metadata, cfg)
    weights = k.FusedWeightsRef(
        gate=weights,
        up=jax.tree.map(lambda r: r.at[..., cfg.out_size_n :], weights),
    )
    ws = k.FusedWeightsRef(gate=ws, up=ws)

    def bounds(g: _Dynamic) -> _Dynamic:
        """Execute bounds logic."""
        start = metadata.gm_id_to_m_offset[g] // sub * sub
        end = pl.cdiv(metadata.gm_id_to_m_offset[g + 1], sub) * sub
        return (start, end - start)

    def start(g: _Dynamic, bank: _Dynamic) -> _Dynamic:
        """Execute start logic."""
        begin, nrows = bounds(g)

        @pl.loop(0, nrows)
        def row(i: _Dynamic) -> _Dynamic:
            """Execute row logic."""
            source = idx[begin + i]
            pltpu.make_async_copy(
                x.at[pl.ds(source, 1)], acts.at[bank, pl.ds(i, 1)], sems.at[bank]
            ).start()

    def inner(
        w: _Dynamic,
        y: _Dynamic,
        partial: _Dynamic,
        acc: _Dynamic,
        metadata: _Dynamic,
    ) -> _Dynamic:
        """Execute inner logic."""
        g = pl.program_id(1)
        bank = g % 2

        @pl.when(g == 0)
        def initial() -> _Dynamic:
            """Execute initial logic."""
            start(g, bank)

        _, nrows = bounds(g)
        active = acts.at[bank, pl.ds(0, nrows)]
        pltpu.make_async_copy(active, active, sems.at[bank]).wait()

        @pl.when(g + 1 < ng)
        def following() -> _Dynamic:
            """Execute following logic."""
            start(g + 1, 1 - bank)

        k.inner_kernel(
            _PackedRows(acts.at[bank]), w, y, partial, acc, metadata, cfgs=cfg
        )

    pltpu.emit_pipeline(
        inner,
        grid=(pl.cdiv(cfg.out_size_n, tn), ng, 1),
        in_specs=(ws,),
        out_specs=ys,
    )(
        weights,
        out.reshape(-1, sub, out.shape[-1]),
        scratches=(partial, acc, metadata),
    )


def indexed_gate_up(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Packed [T,2,K/2] source; bounded token indices and per-expert route counts.

    Runs the GMM weight/scale/output pipeline and inner arithmetic
    (``prefill_megakernel.inner_kernel``) on rows gathered by sorted token
    index. All index reads and row DMAs are bounded by aligned valid metadata
    rows, including partial expert tiles.
    """
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        latent,
        indices,
        rhs,
        sizes,
        scales,
        layer,
    ) = (
        _p_next(_p_iter, _p_get("latent")),
        _p_next(_p_iter, _p_get("indices")),
        _p_next(_p_iter, _p_get("rhs")),
        _p_next(_p_iter, _p_get("sizes")),
        _p_next(_p_iter, _p_get("scales")),
        _p_next(_p_iter, _p_get("layer")),
    )
    rows, width = (indices.shape[0], latent.shape[1] * latent.shape[2])
    limit = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)
    offset = jnp.zeros((1,), jnp.int32)
    cfg = k.make_gmm_configs(
        jax.ShapeDtypeStruct((rows, width), latent.dtype),
        jax.ShapeDtypeStruct(rhs.shape[1:], rhs.dtype),
        jax.ShapeDtypeStruct(scales.shape[1:], scales.dtype),
        sizes,
        offset,
        tile_info=k.calculate_tiling,
        vmem_limit_bytes=limit,
        out_dtype=None,
        acc_dtype=None,
        fuse_act="situ:4:25",
        rhs_quant_dtype=jnp.float4_e2m1fn,
    )
    tm, tn, tk = (cfg.tiles.tile_m, cfg.tiles.tile_n, cfg.tiles.tile_k)
    sub = cfg.dims.size_lhs_sublane
    if tk != width:
        _err_msg = f"Indexed gate/up requires full K: {cfg.tiles}"
        raise ValueError(_err_msg)
    max_gm = cfg.dims.size_group + pl.cdiv(rows, tm) - 1
    if rows % sub or latent.dtype != jnp.bfloat16 or indices.dtype != jnp.int32:
        _err_msg = "Expected sublane-aligned BF16 rows and int32 token indices"
        raise ValueError(_err_msg)
    smem_bytes = indices.size * 4 + (2 * max_gm + 1 + sizes.size + 2) * 4
    if smem_bytes > pltpu.get_tpu_info().smem_capacity_bytes:
        _err_msg = "The sorted index array does not fit in SMEM"
        raise ValueError(_err_msg)

    def kernel(*args: _Dynamic, **kwargs: _Dynamic) -> _Dynamic:
        """Execute kernel logic."""
        return _indexed_gate_up_kernel(*args, **kwargs, cfg=cfg, sub=sub, tm=tm, tn=tn)

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    scratch_shapes = (
        pltpu.VMEM((sub, tn), cfg.out_dtype),
        pltpu.VMEM((tm, 2 * tn), cfg.acc_dtype),
        k.MetadataRef(
            gm_id_to_group_id=pltpu.SMEM((max_gm,), jnp.int32),
            gm_id_to_m_offset=pltpu.SMEM((max_gm + 1,), jnp.int32),
        ),
        pltpu.VMEM((2, tm, 2, width // 2), latent.dtype),
        pltpu.SemaphoreType.DMA((2,)),
    )
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((rows, cfg.out_size_n), cfg.out_dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            in_specs=(hbm, k.WeightsRef(weight=hbm, scale=hbm)),
            out_specs=hbm,
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=limit, disable_bounds_checks=True
        ),
        cost_estimate=None,
        name=f"indexed_ragged_gu_{'prefetch'}" + "_bucket",
    )(
        sizes,
        offset,
        layer,
        indices,
        latent,
        k.WeightsRef(weight=rhs, scale=scales),
    )
