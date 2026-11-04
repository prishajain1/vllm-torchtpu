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

"""Kimi-K3 prefill megakernel: the 93 layers over a packed window at TP32.

Source provenance: Adapted from inferact/k3-megakernel@30c9ba966b and
vllm_torchtpu/kernels/megablox/gmm_v2.py@644426d851.

A window (512, 1024 or 2048..8192 tokens) packs up to ``segments``
sequences at 256-aligned offsets, described by an int32 ``segment_desc``
(offsets, lengths, KDA checkpoint positions). Activations are
sequence-parallel in 64-row stripes: rank r holds rows
``[2048 b + 64 r, 2048 b + 64 (r + 1))`` of every 2048-token block b, in
block order (``sp_token_index``), so each block is a whole ring all-gather
of its own; a 512- or 1024-token window is one block of 16- or 32-row
stripes (``window_stripe``). Weights are
family-major: each family holds every layer's copy, a runtime slot picks one.

The prefill program (``kimi_k3/prefill.py``) runs four ops per window:

* ``make_embedding_sp``, then ``make_prefill_stack_sp_indexed`` (depth 4):
  the embedding and layers 0..3 (a dense FFN at layer 0);
* ``make_regular_group8_sp_indexed``, eleven times: layers 4..91, as pairs
  of four-layer groups (three KDA layers, one MLA layer, each with its MoE);
* ``make_final_mla_sp_indexed`` and ``make_prefill_logits_sp``: layer 92
  and each sequence's first-token logits.

Per layer:

* KDA, ``make_norm_gather_kda_sp``: one Pallas call for the residual mixture
  and RMSNorm, the 32-rank all-gather of the window, the chunked gated delta
  rule (``kda_prefill_tiled_body``) and the output projection ring;
* MLA, ``make_norm_gather_sp`` then ``make_mla_attention_layer_sp_ring``:
  the projections, packed flash attention over the window, context-parallel
  attention over every rank's cached prefix pages (``CpPrefix``), the
  projection ring;
* MoE, ``make_indexed_moe_sp_ring``: sequence-parallel routing, then
  fused EP16/TP2 Pallas schedule for every window (512..8192) using FP32
  reduction (``moe_reduce_dtype="float32"``).

The sections run bottom-up: the TPU kernel library (SparseCore gathers,
TP32 rings, the grouped matmul), the K3 building blocks, the layers, then
the groups and the stack. Model math shared with the decode megakernel is
in ``helpers``. The SparseCore, grouped-matmul and counting-sort code is
adapted from ``vllm_torchtpu.kernels`` and ``vllm_torchtpu.layers`` (see
the header); the TP32 rings from ``vllm_torchtpu.kernels.kimi_k3.collectives``.
"""

from __future__ import annotations

import dataclasses
import functools
import operator
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax._src import api as _src_api
from jax._src import core as jax_core
from jax._src import lax as _src_lax
from jax._src import tree_util as _src_tree_util
from jax._src.pallas import core as pl_core
from jax._src.state.types import AbstractRef, TransformedRef
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.models.kimi_k3.megakernel.common import pool_alias
from vllm_torchtpu.models.kimi_k3.megakernel.config import active as _config

from .helpers import (
    e8m0_scale,
    expert_activation,
    kda_decay,
    kda_normalize_query_key,
)
from .prefill_moe import fused

_Dynamic = Any


def _dyn(val: _Dynamic) -> _Dynamic:
    """Coerce value to dynamic type for static analyzers."""
    return val


_CONST_12 = 12
_CONST_2 = 2
_CONST_256 = 256
_CONST_3 = 3
_CONST_31 = 31
_CONST_32 = 32
_CONST_384 = 384
_CONST_4 = 4
_CONST_56 = 56
_CONST_64 = 64
_CONST_7168 = 7168
_CONST_8 = 8
_CONST_8192 = 8192
_CONST_92 = 92
_CONST_93 = 93


# ============================================================================
# SparseCore kernels through core_map
# ============================================================================
#
# ``pl.kernel`` dispatches to ``mpmd_map``; ``core_map_kernel`` calls
# ``core_map`` directly on the given single SparseCore mesh.


def _empty_out_ref(out_type: _Dynamic) -> _Dynamic:
    # Mirrors JAX's private ``pl.kernel`` output-ref setup; recheck on JAX upgrades.
    """Execute empty out ref logic."""
    aval = operator.attrgetter("_convert_out_shape_to_aval")(pl_core)(out_type)
    memory_space = (
        None
        if isinstance(aval.memory_space, jax_core.MemorySpace)
        else aval.memory_space
    )
    value = _src_lax.empty(aval.shape, aval.dtype, out_sharding=aval.sharding)
    return jax_core.new_ref(value, memory_space=memory_space)


def core_map_kernel(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Near drop-in for ``pl.kernel`` that lowers via ``core_map``.

    Differs in that ``out_shape`` is renamed to ``out_type`` and
    ``scratch_shapes`` (list) is renamed to ``scratch_types`` (dict).
    """
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    body = _p_next(_p_iter, _p_get("body"))
    (
        out_type,
        mesh,
        scratch_types,
        compiler_params,
        interpret,
        cost_estimate,
        debug,
        name,
        metadata,
    ) = (
        _p_get("out_type"),
        _p_get("mesh"),
        _p_get("scratch_types", ()),
        _p_get("compiler_params"),
        _p_kw.get("interpret", False),
        _p_get("cost_estimate"),
        _p_kw.get("debug", False),
        _p_get("name"),
        _p_get("metadata"),
    )
    single_output = not isinstance(out_type, (tuple, list))
    out_types = (out_type,) if single_output else out_type

    @_src_api.jit
    def run(*operands: _Dynamic) -> _Dynamic:
        """Execute run logic."""
        arg_refs = _src_tree_util.tree_map(jax_core.new_ref, operands)
        out_refs = _src_tree_util.tree_map(_empty_out_ref, out_types)

        @pl_core.core_map(
            mesh,
            scratch_shapes=scratch_types,
            compiler_params=compiler_params,
            interpret=interpret,
            cost_estimate=cost_estimate,
            debug=debug,
            name=name,
            metadata=metadata,
        )
        def _(*scratch_refs: _Dynamic, **scratch_kwrefs: _Dynamic) -> _Dynamic:
            """Execute operation logic."""
            return body(*arg_refs, *out_refs, *scratch_refs, **scratch_kwrefs)

        outs = _src_tree_util.tree_map(lambda ref: ref[...], out_refs)
        return outs[0] if single_output else outs

    return run


# ============================================================================
# SparseCore ragged gather
# ============================================================================
#
# Rows gathered by index on the SparseCore (``ragged_gather_v2``): the routed
# MoE's move into expert-major order.


def calculate_col_size(hidden_size: int, packing: int) -> int:
    """Calculates the max column size bounded by VMEM limits and.

    hidden_size divisibility.
    """
    tpu_info = pltpu.get_tpu_info()
    sc_info = tpu_info.sparse_core
    if not (sc_info is not None):
        _err_msg = "SparseCore info is missing."
        raise AssertionError(_err_msg)
    lanes = sc_info.num_lanes

    match tpu_info.generation:
        case 6:
            target_bytes = (256 * 1024) * 0.8
        case 7:
            target_bytes = (512 * 1024) * 0.8
        case _:
            target_bytes = (128 * 1024) * 0.8

    # Calculate max safe column size based on VMEM budget and align to 128.
    num_buffers = 2
    bytes_per_col = (lanes + lanes // packing) * 4 * num_buffers
    max_safe_col = int((target_bytes // bytes_per_col) // 128) * 128

    # Search for the largest divisor of hidden_size bounded by max_safe_col.
    # The first divisor found is the maximum.
    start_col = (min(hidden_size, max_safe_col) // 128) * 128
    for c in range(start_col, 127, -128):
        if hidden_size % c == 0:
            return c
    return max_safe_col


class _MainKernelV2Runner:
    """Runner for main_kernel_v2 ragged gather execution."""

    def __getattr__(self, name: _Dynamic) -> _Dynamic:
        """Dynamic attribute access."""
        raise AttributeError(name)

    def __init__(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Initialize main kernel v2 runner."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            self.start_ref,
            self.end_ref,
            self.in_hbm_ref,
            self.indices_hbm_ref,
            self.out_hbm_ref,
            self.start_vmem_ref,
            self.end_vmem_ref,
            self.sem_ref,
        ) = (
            _p_next(_p_iter, _p_get("start_ref")),
            _p_next(_p_iter, _p_get("end_ref")),
            _p_next(_p_iter, _p_get("in_hbm_ref")),
            _p_next(_p_iter, _p_get("indices_hbm_ref")),
            _p_next(_p_iter, _p_get("out_hbm_ref")),
            _p_next(_p_iter, _p_get("start_vmem_ref")),
            _p_next(_p_iter, _p_get("end_vmem_ref")),
            _p_next(_p_iter, _p_get("sem_ref")),
        )
        (
            self.core_axis_name,
            self.subcore_axis_name,
            self.num_row_subchunks,
        ) = (
            _p_get("core_axis_name"),
            _p_get("subcore_axis_name"),
            _p_get("num_row_subchunks"),
        )
        tpu_info = pltpu.get_tpu_info()
        sc_info = tpu_info.sparse_core
        if not (sc_info is not None):
            _err_msg = "Assertion failed: sc_info is not None"
            raise AssertionError(_err_msg)
        self.num_simd_lanes = sc_info.num_lanes
        self.hidden_size = self.in_hbm_ref.shape[-1]
        self.dtype_bits = jax.dtypes.itemsize_bits(self.out_hbm_ref.dtype)
        self.packing = 32 // self.dtype_bits
        self.col_size = calculate_col_size(self.hidden_size, self.packing)

        if not isinstance(self.hidden_size, int):
            _err_msg = f"hidden_size must be int, got {type(self.hidden_size)}"
            raise TypeError(_err_msg)
        self.num_cores = jax.lax.axis_size(
            (self.core_axis_name, self.subcore_axis_name)
        )
        self.row_subchunk_size = self.num_simd_lanes
        self.row_chunk_size = self.row_subchunk_size * self.num_row_subchunks
        self.block_size = self.row_chunk_size * self.num_cores

        self.recv_sem = self.sem_ref.at[0]

        copy_start = pltpu.make_async_copy(
            self.start_ref, self.start_vmem_ref.at[:1], self.recv_sem
        )
        copy_end = pltpu.make_async_copy(
            self.end_ref, self.end_vmem_ref.at[:1], self.recv_sem
        )
        copy_start.start()
        copy_end.start()
        copy_start.wait()
        copy_end.wait()

        self.start = self.start_vmem_ref[...][0]
        self.end = self.end_vmem_ref[...][0]

        self.block_start = self.start // self.block_size
        self.block_end = pl.cdiv(self.end, self.block_size)
        num_blocks = self.block_end - self.block_start
        self.num_blocks = jnp.where(self.end <= self.start, 0, num_blocks)

        self.num_cols = pl.cdiv(self.hidden_size, self.col_size)
        self.dtype = self.out_hbm_ref.dtype
        self.core_index = lax.axis_index((self.core_axis_name, self.subcore_axis_name))

        self.in_hbm_i32 = self.in_hbm_ref.bitcast(jnp.int32)
        self.out_hbm_i32 = self.out_hbm_ref.bitcast(jnp.int32)
        self.num_phys_cols = self.col_size

    def col_loop(
        self,
        col_base: _Dynamic,
        gather_ref: _Dynamic,
        out_ref: _Dynamic,
        idx_rem: _Dynamic,
        unpack_col_chunk: _Dynamic,
    ) -> _Dynamic:
        """Execute col loop logic."""
        col_slice = pl.ds(col_base, unpack_col_chunk)
        if self.packing == 1:
            gather_dt = gather_ref.bitcast(self.dtype)
            out_dt = out_ref.bitcast(self.dtype)
            out_dt[:, col_slice] = gather_dt[:, col_slice]
        else:
            mask = (1 << self.dtype_bits) - 1
            shift_multiplier = self.dtype_bits.bit_length() - 1
            for i in range(self.num_simd_lanes // self.packing):
                packed_row = jnp.zeros((1, unpack_col_chunk), dtype=jnp.int32)
                for j in range(self.packing):
                    k = i * self.packing + j
                    dynamic_shift = jnp.left_shift(idx_rem[k], shift_multiplier)
                    val = jnp.bitwise_right_shift(
                        gather_ref[pl.ds(k, 1), col_slice], dynamic_shift
                    )
                    val = jnp.bitwise_and(val, mask)
                    pack_shift = j * self.dtype_bits
                    packed_row = jnp.bitwise_or(
                        packed_row, jnp.left_shift(val, pack_shift)
                    )
                out_ref[pl.ds(i, 1), col_slice] = packed_row

    def inner_pipeline(
        self,
        gather_ref: _Dynamic,
        out_ref: _Dynamic,
        idx_ref: _Dynamic,
        unpack_col_chunk: _Dynamic,
    ) -> _Dynamic:
        """Execute inner pipeline logic."""
        row_slice = pl.ds(
            pl.program_id(0) * self.row_subchunk_size, self.row_subchunk_size
        )
        subchunk_idxs = idx_ref[row_slice]
        if self.packing > 1:
            idx_rem = jnp.bitwise_and(subchunk_idxs, self.packing - 1)
        else:
            idx_rem = jnp.zeros_like(subchunk_idxs)

        col_loop_fn = functools.partial(
            self.col_loop,
            gather_ref=gather_ref,
            out_ref=out_ref,
            idx_rem=idx_rem,
            unpack_col_chunk=unpack_col_chunk,
        )
        plsc.parallel_loop(0, self.num_phys_cols, step=unpack_col_chunk)(col_loop_fn)

    def outer_pipeline(self, idx_ref: _Dynamic) -> _Dynamic:
        """Execute outer pipeline logic."""
        b = pl.program_id(0)
        b_global = b + self.block_start

        unpack_col_chunk = 128
        if self.num_phys_cols % unpack_col_chunk != 0:
            _err_msg = "Assertion failed: num_phys_cols % unpack_col_chunk == 0"
            raise AssertionError(_err_msg)
        shift_amount = self.packing.bit_length() - 1
        pltpu.emit_pipeline(
            functools.partial(
                self.inner_pipeline, idx_ref=idx_ref, unpack_col_chunk=unpack_col_chunk
            ),
            grid=(self.num_row_subchunks, self.num_cols),
            in_specs=pl.BlockSpec(
                (pl.Indirect(self.row_subchunk_size), self.num_phys_cols),
                lambda r, col_id: (
                    jnp.bitwise_right_shift(
                        idx_ref[
                            pl.ds(r * self.row_subchunk_size, self.row_subchunk_size)
                        ],
                        shift_amount,
                    ),
                    col_id,
                ),
            ),
            out_specs=pl.BlockSpec(
                (self.row_subchunk_size // self.packing, self.num_phys_cols),
                lambda r, col_id: (
                    (b_global * self.num_cores + self.core_index)
                    * self.num_row_subchunks
                    + r,
                    col_id,
                ),
            ),
        )(self.in_hbm_i32, self.out_hbm_i32)

    def run(self) -> _Dynamic:
        """Execute run logic."""
        pltpu.emit_pipeline(
            self.outer_pipeline,
            grid=(self.num_blocks,),
            in_specs=pl.BlockSpec(
                (self.row_chunk_size,),
                lambda b: ((b + self.block_start) * self.num_cores + self.core_index,),
            ),
        )(self.indices_hbm_ref)


def main_kernel_v2(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Execute main kernel v2 logic."""
    return _MainKernelV2Runner(*_p_args, **_p_kw).run()


@jax.jit
def ragged_gather_v2(
    x: jax.Array, indices: jax.Array, start: jax.Array, end: jax.Array
) -> jax.Array:
    """Perform gather on indices within dynamic array start and end using BlockSpec."""
    if x.ndim != _CONST_2:
        _err_msg = "Ragged gather only supports 2d inputs."
        raise AssertionError(_err_msg)
    if indices.ndim != 1:
        _err_msg = "Ragged gather only supports 1d indices."
        raise AssertionError(_err_msg)

    if jnp.isscalar(start):
        start = start[None]
    if jnp.isscalar(end):
        end = end[None]

    dtype = x.dtype
    if dtype not in (jnp.bfloat16, jnp.float32, jnp.int8, jnp.int4):
        _err_msg = f"dtype must be f32, bf16, int8, or int4, but got {dtype}"
        raise ValueError(_err_msg)

    sc_info = pltpu.get_tpu_info().sparse_core
    if sc_info is None:
        return x[indices]

    hidden_size = x.shape[-1]
    out_size = indices.size

    dtype_bits = jax.dtypes.itemsize_bits(dtype)
    packing = 32 // dtype_bits
    col_size = calculate_col_size(hidden_size, packing)

    aligned_hidden_size = ((hidden_size + col_size - 1) // col_size) * col_size

    num_simd_lanes = sc_info.num_lanes
    num_cores = sc_info.num_cores * sc_info.num_subcores
    base_block_size = num_simd_lanes * num_cores

    # Calculate ideal num_row_subchunks to avoid too much padding overhead.
    num_row_subchunks = max(
        1, min(4, (out_size + base_block_size - 1) // base_block_size)
    )

    row_subchunk_size = num_simd_lanes
    row_chunk_size = row_subchunk_size * num_row_subchunks
    block_size = row_chunk_size * num_cores

    out_pad_size = ((out_size + block_size - 1) // block_size) * block_size - out_size
    indices = jnp.pad(indices, ((0, out_pad_size)))

    vector_mesh = plsc.VectorSubcoreMesh(
        num_cores=sc_info.num_cores,
        num_subcores=sc_info.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    return core_map_kernel(
        functools.partial(
            main_kernel_v2,
            core_axis_name=vector_mesh.core_axis_name,
            subcore_axis_name=vector_mesh.subcore_axis_name,
            num_row_subchunks=num_row_subchunks,
        ),
        out_type=jax.ShapeDtypeStruct(
            (out_size + out_pad_size, aligned_hidden_size), dtype
        ),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=True,
            needs_layout_passes=True,
            disable_bounds_checks=True,
        ),
        scratch_types=[
            pltpu.VMEM((16,), jnp.int32),
            pltpu.VMEM((16,), jnp.int32),
            pltpu.SemaphoreType.DMA((1,)),
        ],
        mesh=vector_mesh,
        # Without an estimate the scheduler treats the call as free and puts
        # its done right after its start, leaving the TensorCore idle.
        cost_estimate=pl.CostEstimate(
            flops=0,
            transcendentals=0,
            bytes_accessed=2 * out_size * hidden_size * jnp.dtype(dtype).itemsize,
        ),
        name="sc_ragged_gather_v2",
    )(start, end, x, indices)[:out_size, :hidden_size]


# ============================================================================
# SparseCore ragged gather-reduce
# ============================================================================
#
# The routed MoE's return to token order (``ragged_gather_reduce``): each core
# scatter-adds its destination block's valid routes into FP32 VMEM.


_TOKEN_SUBCHUNK = 16
_NUM_TOKEN_SUBCHUNKS = 4
_TOKEN_BLOCK = _TOKEN_SUBCHUNK * _NUM_TOKEN_SUBCHUNKS
_ROUTE_INDEX_BITS = 20
_ROUTE_INDEX_MASK = (1 << _ROUTE_INDEX_BITS) - 1


def _align_to(value: int, alignment: int) -> int:
    """Execute align to logic."""
    return pl.cdiv(value, alignment) * alignment


def _fallback_implementation(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
) -> jax.Array:
    """Execute fallback implementation logic."""
    out = x[indices] * topk_weights[:, None].astype(jnp.float32)
    out = jnp.where(valid_rows_mask[:, None], out, 0)
    out = out.reshape(-1, reduce_group_size, out.shape[-1])
    return jnp.sum(out, axis=1).astype(x.dtype)


def _calculate_num_column_partitions(
    hidden_size: int, num_cores: int, num_lanes: int
) -> int:
    """Execute calculate num column partitions logic."""
    preferred_num_stages = 4
    num_column_partitions = 1
    while (
        num_cores % (num_column_partitions * 2) == 0
        and hidden_size % (num_lanes * num_column_partitions * 2) == 0
        and hidden_size // (num_column_partitions * 2 * num_lanes)
        >= preferred_num_stages
    ):
        num_column_partitions *= 2
    # Destination-major accumulation keeps a dense FP32 token block in VMEM.
    # For columns up to 1024, using one fewer column partition amortizes the
    # per-block metadata, clear, and output-DMA overhead while still leaving
    # enough room for the double-buffered indirect gather.  Wider columns stay
    # with the conservative partitioning selected above.
    if num_column_partitions >= _CONST_2 and hidden_size <= 1024 * (
        num_column_partitions // 2
    ):
        num_column_partitions //= 2
    return num_column_partitions


def partition_counts(hidden_size: int, tpu_info: _Dynamic) -> tuple[int, int]:
    """Column and row partition counts of the SparseCore subcores for a width."""
    sc_info = tpu_info.sparse_core
    num_cores = sc_info.num_cores * sc_info.num_subcores
    num_column_partitions = _calculate_num_column_partitions(
        hidden_size, num_cores, tpu_info.num_lanes
    )
    return num_column_partitions, num_cores // num_column_partitions


class _TokenBlockPipelineRunner:
    """Runner for SparseCore token block gather-reduce pipeline."""

    def __getattr__(self, name: _Dynamic) -> _Dynamic:
        """Dynamic attribute access."""
        raise AttributeError(name)

    def __init__(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Initialize token block pipeline runner."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            self.x_hbm_ref,
            self.route_metadata_hbm_ref,
            self.route_weights_hbm_ref,
            self.route_counts_hbm_ref,
            self.out_hbm_ref,
            self.accum_vmem_ref,
            self.out_tile_vmem_ref,
            self.sem_ref,
        ) = (
            _p_next(_p_iter, _p_get("x_hbm_ref")),
            _p_next(_p_iter, _p_get("route_metadata_hbm_ref")),
            _p_next(_p_iter, _p_get("route_weights_hbm_ref")),
            _p_next(_p_iter, _p_get("route_counts_hbm_ref")),
            _p_next(_p_iter, _p_get("out_hbm_ref")),
            _p_next(_p_iter, _p_get("accum_vmem_ref")),
            _p_next(_p_iter, _p_get("out_tile_vmem_ref")),
            _p_next(_p_iter, _p_get("sem_ref")),
        )
        (
            self.core_axis_name,
            self.subcore_axis_name,
            _,
            self.num_column_partitions,
            self.blocks_per_row_partition,
            self.topk,
        ) = (
            _p_get("core_axis_name"),
            _p_get("subcore_axis_name"),
            _p_get("num_row_partitions"),
            _p_get("num_column_partitions"),
            _p_get("blocks_per_row_partition"),
            _p_get("topk"),
        )
        sc_info = pltpu.get_tpu_info().sparse_core
        if not (sc_info is not None):
            _err_msg = "Assertion failed: sc_info is not None"
            raise AssertionError(_err_msg)
        num_simd_lanes = sc_info.num_lanes
        if num_simd_lanes != _TOKEN_SUBCHUNK:
            _err_msg = "Assertion failed: num_simd_lanes == _TOKEN_SUBCHUNK"
            raise AssertionError(_err_msg)

        core_id = jax.lax.axis_index((self.core_axis_name, self.subcore_axis_name))
        self.row_partition_id = core_id // self.num_column_partitions
        self.col_partition_id = core_id % self.num_column_partitions
        self.col_size = self.accum_vmem_ref.shape[-1]
        self.col_start = self.col_partition_id * self.col_size
        self.metadata_block_size = self.topk * _TOKEN_BLOCK
        self.x_32b_hbm_ref = self.x_hbm_ref.bitcast(jnp.int32)
        self.out_32b_hbm_ref = self.out_hbm_ref.bitcast(jnp.uint32)

    def metadata_block_index(self, block_id: _Dynamic) -> tuple[_Dynamic]:
        """Execute metadata block index logic."""
        return (self.row_partition_id * self.blocks_per_row_partition + block_id,)

    def zero_column(self, col_offset: _Dynamic) -> None:
        """Execute zero column logic."""
        col_slice = pl.ds(col_offset, _TOKEN_SUBCHUNK)
        zero = jnp.zeros((_TOKEN_SUBCHUNK,), jnp.float32)
        for token_row in range(_TOKEN_BLOCK):
            self.accum_vmem_ref[token_row, col_slice] = zero

    def route_indices_slice(
        self, route_chunk: _Dynamic, route_metadata_ref: _Dynamic
    ) -> _Dynamic:
        """Execute route indices slice logic."""
        start = route_chunk * _TOKEN_SUBCHUNK
        metadata = route_metadata_ref[pl.ds(start, _TOKEN_SUBCHUNK)]
        return jnp.bitwise_and(metadata, _ROUTE_INDEX_MASK)

    def fixed_two_route_column_loop(
        self,
        col_offset: _Dynamic,
        gather_ref: _Dynamic,
        source_indices: _Dynamic,
        route_destinations: _Dynamic,
        route_weights: _Dynamic,
    ) -> None:
        """Execute fixed two route column loop logic."""
        col_slice = pl.ds(col_offset, _TOKEN_SUBCHUNK)
        feature_indices = col_offset + jnp.arange(_TOKEN_SUBCHUNK, dtype=jnp.int32)
        route_batch_size = 16
        for route_batch_start in range(0, _TOKEN_SUBCHUNK, route_batch_size):
            weighted_values = []
            for route_lane in range(
                route_batch_start, route_batch_start + route_batch_size
            ):
                value_i32 = gather_ref[route_lane, col_slice]
                shift = jnp.where(
                    jnp.bitwise_and(source_indices[route_lane], 1) == 0, 16, 0
                )
                shifted = jnp.bitwise_and(
                    jnp.left_shift(value_i32, shift), jnp.int32(-65536)
                )
                value_f32 = (
                    plsc.bitcast(shifted, jnp.float32) * route_weights[route_lane]
                )
                weighted_values.append(value_f32)

            for pair_start in range(0, route_batch_size, 2):
                route_lane = route_batch_start + pair_start
                destination_indices = jnp.full_like(
                    feature_indices, route_destinations[route_lane]
                )
                plsc.addupdate_scatter(
                    self.accum_vmem_ref,
                    (destination_indices, feature_indices),
                    weighted_values[pair_start] + weighted_values[pair_start + 1],
                )

    def generic_column_loop(
        self,
        col_offset: _Dynamic,
        gather_ref: _Dynamic,
        routes: tuple[_Dynamic, _Dynamic, _Dynamic],
        metadata_start: _Dynamic,
        valid_route_count: _Dynamic,
    ) -> None:
        """Execute generic column loop logic."""
        source_indices, route_destinations, route_weights = routes
        col_slice = pl.ds(col_offset, _TOKEN_SUBCHUNK)
        feature_indices = col_offset + jnp.arange(_TOKEN_SUBCHUNK, dtype=jnp.int32)
        route_batch_size = 4
        for route_batch_start in range(0, _TOKEN_SUBCHUNK, route_batch_size):
            weighted_values = []
            for route_lane in range(
                route_batch_start, route_batch_start + route_batch_size
            ):
                value_i32 = gather_ref[route_lane, col_slice]
                shift = jnp.where(
                    jnp.bitwise_and(source_indices[route_lane], 1) == 0, 16, 0
                )
                shifted = jnp.bitwise_and(
                    jnp.left_shift(value_i32, shift), jnp.int32(-65536)
                )
                value_f32 = (
                    plsc.bitcast(shifted, jnp.float32) * route_weights[route_lane]
                )
                weighted_values.append(value_f32)

            for route_batch_lane in range(route_batch_size):
                route_lane = route_batch_start + route_batch_lane
                destination_indices = jnp.full_like(
                    feature_indices, route_destinations[route_lane]
                )
                route_valid = metadata_start + route_lane < valid_route_count
                scatter_mask = jnp.broadcast_to(route_valid, (_TOKEN_SUBCHUNK,))
                plsc.addupdate_scatter(
                    self.accum_vmem_ref,
                    (destination_indices, feature_indices),
                    weighted_values[route_batch_lane],
                    mask=scatter_mask,
                )

    def route_pipeline(
        self,
        gather_ref: _Dynamic,
        route_metadata_ref: _Dynamic,
        route_weights_ref: _Dynamic,
        valid_route_count: _Dynamic,
        fixed_two_routes: _Dynamic,
    ) -> None:
        """Execute route pipeline logic."""
        route_chunk = pl.program_id(0)
        metadata_start = route_chunk * _TOKEN_SUBCHUNK
        metadata_slice = pl.ds(metadata_start, _TOKEN_SUBCHUNK)
        route_metadata = route_metadata_ref[metadata_slice]
        source_indices = jnp.bitwise_and(route_metadata, _ROUTE_INDEX_MASK)
        route_destinations = jnp.bitwise_right_shift(route_metadata, _ROUTE_INDEX_BITS)
        route_weights = route_weights_ref[metadata_slice]

        routes = (source_indices, route_destinations, route_weights)
        pl.when(fixed_two_routes)(
            lambda: plsc.parallel_loop(0, self.col_size, step=_TOKEN_SUBCHUNK)(
                lambda col: self.fixed_two_route_column_loop(
                    col, gather_ref, source_indices, route_destinations, route_weights
                )
            )
        )
        pl.when(jnp.logical_not(fixed_two_routes))(
            lambda: plsc.parallel_loop(0, self.col_size, step=_TOKEN_SUBCHUNK)(
                lambda col: self.generic_column_loop(
                    col, gather_ref, routes, metadata_start, valid_route_count
                )
            )
        )

    def cast_column(
        self, col_offset: _Dynamic, accum_row_start: _Dynamic, output_buffer: _Dynamic
    ) -> None:
        """Execute cast column logic."""
        col_slice = pl.ds(col_offset, _TOKEN_SUBCHUNK)
        for token_lane in range(0, _TOKEN_SUBCHUNK, 2):
            accum_row = accum_row_start + token_lane
            packed_bf16 = plsc.pack(
                self.accum_vmem_ref[accum_row, col_slice],
                self.accum_vmem_ref[accum_row + 1, col_slice],
                format=plsc.PackFormat.INTERLEAVED,
            )
            self.out_tile_vmem_ref[output_buffer, token_lane // 2, col_slice] = (
                plsc.bitcast(packed_bf16, jnp.uint32)
            )

    def token_block_pipeline(
        self,
        route_metadata_ref: _Dynamic,
        route_weights_ref: _Dynamic,
        route_counts_ref: _Dynamic,
        output_sem_ref: _Dynamic,
    ) -> None:
        """Execute token block pipeline logic."""
        block_id = pl.program_id(0)
        global_block_id = (
            self.row_partition_id * self.blocks_per_row_partition + block_id
        )
        valid_route_count = route_counts_ref[...][0]
        fixed_two_routes = valid_route_count == 2 * _TOKEN_BLOCK
        num_route_chunks = pl.cdiv(valid_route_count, _TOKEN_SUBCHUNK)

        plsc.parallel_loop(0, self.col_size, step=_TOKEN_SUBCHUNK)(self.zero_column)

        pltpu.emit_pipeline(
            lambda g_ref: self.route_pipeline(
                g_ref,
                route_metadata_ref,
                route_weights_ref,
                valid_route_count,
                fixed_two_routes,
            ),
            grid=(num_route_chunks,),
            in_specs=pl.BlockSpec(
                (pl.Indirect(_TOKEN_SUBCHUNK), self.col_size),
                lambda route_chunk: (
                    jnp.bitwise_right_shift(
                        self.route_indices_slice(route_chunk, route_metadata_ref), 1
                    ),
                    self.col_partition_id,
                ),
            ),
            out_specs=(),
        )(self.x_32b_hbm_ref)

        for token_subchunk_pair in range(_NUM_TOKEN_SUBCHUNKS // 2):
            copies = []
            for output_buffer in range(2):
                token_subchunk = token_subchunk_pair * 2 + output_buffer
                accum_row_start = token_subchunk * _TOKEN_SUBCHUNK
                plsc.parallel_loop(0, self.col_size, step=_TOKEN_SUBCHUNK)(
                    lambda col_offset,
                    ars=accum_row_start,
                    ob=output_buffer: self.cast_column(col_offset, ars, ob)
                )

                output_row = (
                    global_block_id * _TOKEN_BLOCK + token_subchunk * _TOKEN_SUBCHUNK
                )
                output_row_packed = pl.multiple_of(
                    output_row // 2, _TOKEN_SUBCHUNK // 2
                )
                copy = pltpu.make_async_copy(
                    self.out_tile_vmem_ref.at[output_buffer],
                    self.out_32b_hbm_ref.at[
                        pl.ds(output_row_packed, _TOKEN_SUBCHUNK // 2),
                        pl.ds(self.col_start, self.col_size),
                    ],
                    output_sem_ref.at[output_buffer],
                )
                copy.start()
                copies.append(copy)
            for copy in copies:
                copy.wait()

    def run(self) -> _Dynamic:
        """Execute run logic."""
        pltpu.emit_pipeline(
            self.token_block_pipeline,
            grid=(self.blocks_per_row_partition,),
            in_specs=(
                pl.BlockSpec((self.metadata_block_size,), self.metadata_block_index),
                pl.BlockSpec((self.metadata_block_size,), self.metadata_block_index),
                pl.BlockSpec((_TOKEN_SUBCHUNK,), self.metadata_block_index),
            ),
            out_specs=(),
        )(
            self.route_metadata_hbm_ref,
            self.route_weights_hbm_ref,
            self.route_counts_hbm_ref,
            scratches=(self.sem_ref,),
        )


def _main_kernel(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Execute main kernel logic."""
    return _TokenBlockPipelineRunner(*_p_args, **_p_kw).run()


def _prepare_gather_metadata(
    tensors: tuple[jax.Array, jax.Array, jax.Array],
    num_tokens: int,
    reduce_group_size: int,
    *,
    token_padding: int = 0,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Prepare and sort gather metadata for SparseCore execution."""
    indices, topk_weights, valid_rows_mask = tensors
    indices_2d = indices.reshape(num_tokens, reduce_group_size)
    weights_2d = topk_weights.reshape(num_tokens, reduce_group_size)
    valid_2d = valid_rows_mask.reshape(num_tokens, reduce_group_size)
    if token_padding:
        indices_2d = jnp.pad(indices_2d, ((0, token_padding), (0, 0)))
        weights_2d = jnp.pad(weights_2d, ((0, token_padding), (0, 0)))
        valid_2d = jnp.pad(valid_2d, ((0, token_padding), (0, 0)))

    padded_num_tokens = num_tokens + token_padding
    num_blocks = padded_num_tokens // _TOKEN_BLOCK
    metadata_block_size = _TOKEN_BLOCK * reduce_group_size
    indices_blocks = indices_2d.reshape(num_blocks, metadata_block_size)
    valid_blocks = valid_2d.reshape(num_blocks, metadata_block_size)
    weights_blocks = weights_2d.reshape(num_blocks, metadata_block_size)
    destination_template = jnp.repeat(
        jnp.arange(_TOKEN_BLOCK, dtype=jnp.int32), reduce_group_size
    )
    destination_blocks = jnp.broadcast_to(destination_template, indices_blocks.shape)

    route_order = jnp.argsort(~valid_blocks, axis=1, stable=True)
    compact_valid = jnp.take_along_axis(valid_blocks, route_order, axis=1)
    route_indices = jnp.take_along_axis(indices_blocks, route_order, axis=1)
    route_destinations = jnp.take_along_axis(destination_blocks, route_order, axis=1)
    route_weights = jnp.take_along_axis(weights_blocks, route_order, axis=1)
    return route_indices, route_destinations, route_weights, compact_valid


@functools.partial(jax.jit, static_argnames=("reduce_group_size",))
def ragged_gather_reduce(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
) -> jax.Array:
    """Destination-major SparseCore gather-reduce: the routed MoE's combine."""
    sc_info = pltpu.get_tpu_info().sparse_core
    if (
        sc_info is None
        or x.dtype != jnp.bfloat16
        or x.shape[-1] % pltpu.get_tpu_info().num_lanes != 0
    ):
        return _fallback_implementation(
            x,
            indices,
            topk_weights,
            valid_rows_mask,
            reduce_group_size,
        )
    dtype_bytes = jax.dtypes.itemsize_bits(x.dtype) // 8
    if jnp.size(x) * dtype_bytes * 2 < pltpu.get_tpu_info().vmem_capacity_bytes * 0.6:
        return _fallback_implementation(
            x,
            indices,
            topk_weights,
            valid_rows_mask,
            reduce_group_size,
        )

    input_size = indices.size
    if input_size % reduce_group_size != 0:
        _err_msg = (
            f"input_size={input_size!r} must be divisible by "
            f"reduce_group_size={reduce_group_size!r}"
        )
        raise ValueError(_err_msg)
    if x.shape[0] % 2:
        x = jnp.pad(x, ((0, 1), (0, 0)))
    num_tokens = input_size // reduce_group_size
    hidden_size = x.shape[-1]
    if x.shape[0] > 1 << _ROUTE_INDEX_BITS:
        _err_msg = (
            f"destination-major gather-reduce supports at most "
            f"{1 << _ROUTE_INDEX_BITS} source rows, got {x.shape[0]}"
        )
        raise ValueError(_err_msg)
    num_column_partitions, num_row_partitions = partition_counts(
        hidden_size, pltpu.get_tpu_info()
    )

    token_alignment = num_row_partitions * _TOKEN_BLOCK
    padded_num_tokens = _align_to(num_tokens, token_alignment)
    token_padding = padded_num_tokens - num_tokens

    (
        route_indices,
        route_destinations,
        route_weights,
        compact_valid,
    ) = _prepare_gather_metadata(
        (indices, topk_weights, valid_rows_mask),
        num_tokens,
        reduce_group_size,
        token_padding=token_padding,
    )

    num_blocks = padded_num_tokens // _TOKEN_BLOCK
    metadata_block_size = _TOKEN_BLOCK * reduce_group_size
    route_metadata = jnp.bitwise_or(
        jnp.left_shift(route_destinations, _ROUTE_INDEX_BITS), route_indices
    )
    route_counts = jnp.sum(compact_valid, axis=1, dtype=jnp.int32)
    route_counts = jnp.pad(route_counts[:, None], ((0, 0), (0, _TOKEN_SUBCHUNK - 1)))

    out = jnp.zeros((padded_num_tokens, hidden_size), dtype=x.dtype)
    blocks_per_row_partition = num_blocks // num_row_partitions
    column_partition_size = hidden_size // num_column_partitions

    kernel_fn = functools.partial(
        _main_kernel,
        core_axis_name="core",
        subcore_axis_name="subcore",
        num_row_partitions=num_row_partitions,
        num_column_partitions=num_column_partitions,
        blocks_per_row_partition=blocks_per_row_partition,
        topk=reduce_group_size,
    )

    sc_grid = (num_row_partitions, num_column_partitions)
    out = pl.pallas_call(
        kernel_fn,
        out_shape=jax.ShapeDtypeStruct(out.shape, out.dtype),
        grid_spec=operator.attrgetter("SparseCoreGridSpec")(pltpu)(
            grid=sc_grid,
            num_subcores=pltpu.get_tpu_info().sparse_core.num_subcores,
            dimension_semantics=("parallel", "parallel"),
        ),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(
                (num_blocks * metadata_block_size,),
                lambda _r, _c: (0,),
                memory_space=pltpu.HBM,
            ),
            pl.BlockSpec(
                (num_blocks * metadata_block_size,),
                lambda _r, _c: (0,),
                memory_space=pltpu.HBM,
            ),
            pl.BlockSpec(
                (num_blocks, _TOKEN_SUBCHUNK),
                lambda _r, _c: (0, 0),
                memory_space=pltpu.HBM,
            ),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
        scratch_shapes=[
            pltpu.VMEM((_TOKEN_BLOCK, column_partition_size), jnp.float32),
            pltpu.VMEM((2, _TOKEN_SUBCHUNK // 2, column_partition_size), jnp.uint32),
            pltpu.SemaphoreType.DMA((2,)),
        ],
    )(x, route_metadata.reshape(-1), route_weights.reshape(-1), route_counts)

    if token_padding:
        out = out[:num_tokens]
    return out


CYCLES = (
    (0, 1, 3, 2, 6, 7, 5, 4, 12, 13, 15, 14, 10, 11, 9, 8),
    (0, 2, 6, 4, 5, 7, 3, 1, 9, 11, 15, 13, 12, 14, 10, 8),
    (0, 4, 12, 8, 9, 13, 5, 1, 3, 7, 15, 11, 10, 14, 6, 2),
)


def ring_tables(devices: _Dynamic, token_ranks: _Dynamic) -> _Dynamic:
    """Build [family,position/peer/token-owner,mesh-index] lookup tables."""
    if len(devices) != _CONST_32 or sorted(token_ranks) != list(range(32)):
        _err_msg = "K3 rings require 32 devices and a token-owner permutation"
        raise ValueError(_err_msg)
    if [d.id for d in devices] != sorted(d.id for d in devices):
        _err_msg = "Production collective mesh must use ascending device IDs"
        raise ValueError(_err_msg)
    coords = [sorted({d.coords[a] for d in devices}) for a in range(3)]
    if tuple(map(len, coords)) != (2, 2, 4):
        _err_msg = "K3 rings require a physical 2x2x4 slice"
        raise ValueError(_err_msg)
    physical_to_mesh = {}
    for i, device in enumerate(devices):
        x, y, z = [coords[a].index(device.coords[a]) for a in range(3)]
        if device.core_on_chip not in (0, 1):
            _err_msg = "Expected two cores per chip"
            raise ValueError(_err_msg)
        physical = ((x * 2 + y) * 4 + z) * 2 + device.core_on_chip
        physical_to_mesh[physical] = i
    if len(physical_to_mesh) != _CONST_32:
        _err_msg = "Duplicate physical cores"
        raise ValueError(_err_msg)
    result = []
    for cycle in CYCLES:
        peers = tuple(
            physical_to_mesh[chip * 2 + core] for chip in cycle for core in range(2)
        )
        positions = tuple(peers.index(i) for i in range(32))
        result.append((positions, peers, tuple(token_ranks)))
    return np.asarray(result, np.int32)


def _regions(rows: _Dynamic, width: _Dynamic) -> _Dynamic:
    """The ring streams' ``(row, rows, column, columns)`` regions: two groups.

    x three cycles x two directions. From 64 rows a stream carries a quarter
    of the rows and its cycle's column third; below that a quarter would be
    under a BF16 tile (16 rows), so each stream carries every row and a
    twelfth of the columns.
    """
    if rows % 16 or width % 128 or width < _CONST_384:
        _err_msg = "K3 rings need 16-row ownership and 128-column alignment"
        raise ValueError(_err_msg)
    blocks = width // 128
    if rows < _CONST_64:
        if blocks < _CONST_12:
            _err_msg = "K3 rings under 64 rows need at least 12 column blocks"
            raise ValueError(_err_msg)
        widths = [128 * (blocks // 12 + (i < blocks % 12)) for i in range(12)]
        cols = np.cumsum([0, *widths[:-1]])
        return tuple(
            (0, rows, int(col), w) for col, w in zip(cols, widths, strict=True)
        )
    if rows % 64:
        _err_msg = "K3 rings need 64-row ownership from 64 rows"
        raise ValueError(_err_msg)
    widths = [128 * (blocks // 3 + (i < blocks % 3)) for i in range(3)]
    cols = [0, widths[0], widths[0] + widths[1]]
    return tuple(
        ((2 * group + direction) * (rows // 4), rows // 4, cols[family], widths[family])
        for group in range(2)
        for family in range(3)
        for direction in range(2)
    )


# Sequence-parallel ownership: a rank holds ``stripe`` rows of every
# ``32 * stripe`` tokens. Local row i of rank r is window token
# (i // stripe) * 32 stripe + r * stripe + i % stripe. The stripe is STRIPE
# rows; a window under 32 STRIPE tokens (512, 1024) is one block of
# window / 32-row stripes (``window_stripe``).
STRIPE = 64
SP_BLOCK = 32 * STRIPE
SMALL_WINDOWS = (512, 1024)


def check_window(seq_len: _Dynamic, what: _Dynamic) -> _Dynamic:
    """Reject a window the sequence-parallel kernels can't run: 512, 1024,.

    or a multiple of 2048 up to 8192.
    """
    if seq_len not in SMALL_WINDOWS and (
        seq_len % SP_BLOCK or not SP_BLOCK <= seq_len <= _CONST_8192
    ):
        _err_msg = (
            f"{what} needs T in {SMALL_WINDOWS} or 2048..8192 (a multiple of 2048)"
        )
        raise ValueError(_err_msg)


def window_stripe(local_rows: _Dynamic) -> _Dynamic:
    """The stripe of a window with ``local_rows`` rows per rank."""
    return min(STRIPE, local_rows)


def sp_token_index(rank: _Dynamic, local: _Dynamic, stripe: _Dynamic) -> _Dynamic:
    """Window token index of local row(s) ``local`` on rank ``rank``."""
    return (local // stripe) * 32 * stripe + rank * stripe + local % stripe


def sp_order(window: _Dynamic) -> _Dynamic:
    """The window token indices in rank-major ownership order: rank r's.

    local rows are ``sp_order(window)[r * window // 32 : (r + 1) * window // 32]``.
    """
    stripe = window_stripe(window // 32)
    return np.arange(window).reshape(-1, 32, stripe).transpose(1, 0, 2).reshape(-1)


def _stripe_pieces(begin: _Dynamic, size: _Dynamic, stripe: _Dynamic) -> _Dynamic:
    """Split the static local rows [begin, begin + size) at stripe ends.

    (offset in the range, local row, rows) pieces, each contiguous in the
    window.
    """
    pieces, row = [], begin
    while row < begin + size:
        rows = min(begin + size, (row // stripe + 1) * stripe) - row
        pieces.append((row - begin, row, rows))
        row += rows
    return pieces


def _ring_enter(tables: _Dynamic, rank: _Dynamic) -> None:
    """Meet the six ring neighbours (both directions of the three cycles).

    before the first remote copy: each has entered the kernel.
    """
    positions = [tables[f, 0, rank] for f in range(3)]
    peers = [tables[f, 1, (positions[f] + d) % 32] for f in range(3) for d in (-1, 1)]
    barrier = pltpu.get_barrier_semaphore()
    for p in peers:
        pl.semaphore_signal(
            barrier, 1, device_id=(p,), device_id_type=pl.DeviceIdType.MESH
        )
    pl.semaphore_wait(barrier, len(peers))

    def _sync_ready(ready: _Dynamic) -> None:
        """Signal and wait on ready semaphore."""
        for p in peers:
            pl.semaphore_signal(
                ready, 1, device_id=(p,), device_id_type=pl.DeviceIdType.MESH
            )
        pl.semaphore_wait(ready, len(peers))

    pl.run_scoped(_sync_ready, ready=pltpu.SemaphoreType.REGULAR)


class _ReduceScatterRunner:
    """Runner for reduce scatter ring communication."""

    def __getattr__(self, name: _Dynamic) -> _Dynamic:
        """Dynamic attribute access."""
        raise AttributeError(name)

    def __init__(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Initialize reduce scatter runner."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            self.tables_ref,
            self.x_ref,
            self.o_ref,
            self.local_sems,
            self.send_sems,
            self.recv_sems,
            self.sem_credits,
            self.store_sem,
            self.local_buf,
            self.payload,
            self.received,
        ) = (
            _p_next(_p_iter, _p_get("tables_ref")),
            _p_next(_p_iter, _p_get("x_ref")),
            _p_next(_p_iter, _p_get("o_ref")),
            _p_next(_p_iter, _p_get("local_sems")),
            _p_next(_p_iter, _p_get("send_sems")),
            _p_next(_p_iter, _p_get("recv_sems")),
            _p_next(_p_iter, _p_get("credits")),
            _p_next(_p_iter, _p_get("store_sem")),
            _p_next(_p_iter, _p_get("local_buf")),
            _p_next(_p_iter, _p_get("payload")),
            _p_next(_p_iter, _p_get("received")),
        )
        self.streams = _p_get("streams")
        self.buffers = _p_get("buffers")
        self.tile = pl.program_id(0)
        self.step = pl.program_id(1)
        self.rank = lax.axis_index("k3_sp")
        self.positions = [self.tables_ref[f, 0, self.rank] for f in range(3)]

    def ring_coords(self, pos: _Dynamic, family: _Dynamic) -> tuple[_Dynamic]:
        """Execute ring coords logic."""
        peer = self.tables_ref[family, 1, pos]
        return (peer,)

    def remote(
        self,
        index: _Dynamic,
        stream: int,
        target: _Dynamic,
        rows: _Dynamic,
        cols: _Dynamic,
    ) -> _Dynamic:
        """Execute remote copy logic."""
        return pltpu.make_async_remote_copy(
            self.payload.at[rows, cols],
            self.received.at[index % self.buffers, rows, cols],
            self.send_sems.at[stream],
            self.recv_sems.at[index % self.buffers, stream],
            device_id=target,
            device_id_type=pl.DeviceIdType.MESH,
        )

    def step_seed(self, rows: _Dynamic, cols: _Dynamic) -> None:
        """Execute seed logic."""
        self.payload.at[rows, cols][...] = self.local_buf.at[rows, cols][...].astype(
            self.payload.dtype
        )

    def step_add(
        self, rows: _Dynamic, cols: _Dynamic, stream: int, source: _Dynamic
    ) -> None:
        """Execute add logic."""
        self.payload.at[rows, cols][...] = (
            self.local_buf.at[rows, cols][...].astype(self.payload.dtype)
            + self.received.at[(self.step - 1) % self.buffers, rows, cols][...]
        )
        pl.semaphore_signal(
            self.sem_credits.at[stream],
            1,
            device_id=source,
            device_id_type=pl.DeviceIdType.MESH,
        )

    def step_forward(
        self, stream: int, target: _Dynamic, rows: _Dynamic, cols: _Dynamic
    ) -> None:
        """Execute forward logic."""
        pl.when(self.step >= self.buffers)(
            lambda: pl.semaphore_wait(self.sem_credits.at[stream], 1)
        )
        self.remote(self.step, stream, target, rows, cols).start()

    def step_finish(self, stream: int, rows: _Dynamic, cols: _Dynamic) -> None:
        """Execute finish logic."""
        pl.semaphore_wait(self.sem_credits.at[stream], self.buffers)
        output = pltpu.make_async_copy(
            self.payload.at[rows, cols],
            self.o_ref.at[
                pl.ds(self.tile * self.payload.shape[0] + rows.start, rows.size), cols
            ],
            self.store_sem,
        )
        output.start()
        output.wait()

    def _carry_partial(self, real_count: _Dynamic, window: _Dynamic) -> None:
        """Execute carry partial logic for causal conv."""
        rows_idx = jnp.arange(self.tile + 3, dtype=jnp.int32)
        picked = [
            jnp.sum(_row_where(rows_idx == real_count + i, window), axis=0)
            for i in range(3)
        ]
        self.conv_hist_vmem[...] = jnp.stack(picked)

    def _load_seg_state(self, current: _Dynamic, position: _Dynamic) -> None:
        """Load state for a sequence segment."""
        self.copy_now(
            (
                (self.rec_state_in_ref.at[current], self.rec_state_vmem),
                (self.conv_hist_in_ref.at[current], self.conv_hist_vmem),
            ),
            7,
        )
        pl.when(position == 0)(lambda: self.write_checkpoint(current))

    def _store_seg_state(self, current: _Dynamic) -> None:
        """Store state for a sequence segment."""
        self.copy_now(
            (
                (self.rec_state_vmem, self.rec_state_out_ref.at[current]),
                (self.conv_hist_vmem, self.conv_hist_out_ref.at[current]),
            ),
            9,
        )

    def _drain_block(self, t: _Dynamic, buf: _Dynamic) -> None:
        """Drain stores at block boundary."""
        self.store_tile(t, buf).wait()
        if self.block_tiles > 1:
            self.store_tile(t - 1, 1 - buf).wait()

    def _load_weights(self, weight_copies: tuple[_Dynamic, ...]) -> None:
        """Load layer weights into VMEM."""
        for c in weight_copies:
            c.start()
        for c in weight_copies:
            c.wait()

    def run(self) -> None:
        """Execute reduce scatter step."""
        pl.when(self.step == 0)(lambda: _ring_enter(self.tables_ref, self.rank))

        descriptors = []
        reads = []
        for stream, region in enumerate(self.streams):
            begin, size = region[:2]
            col_begin, width = (
                region[2:] if len(region) == _CONST_4 else (0, self.o_ref.shape[1])
            )
            cols = pl.ds(col_begin, width)
            family = (stream // 2) % 3
            direction = -1 if stream % 2 == 0 else 1
            position = self.positions[family]
            target = self.ring_coords((position + direction) % 32, jnp.int32(family))
            source = self.ring_coords((position - direction) % 32, jnp.int32(family))
            owner_mesh = self.ring_coords(
                (position - direction * (self.step + 1)) % 32, jnp.int32(family)
            )[0]
            owner = self.tables_ref[family, 2, owner_mesh]
            rows = pl.ds(begin, size)
            read = pltpu.make_async_copy(
                self.x_ref.at[
                    pl.ds(
                        owner * self.o_ref.shape[0]
                        + self.tile * self.payload.shape[0]
                        + begin,
                        size,
                    ),
                    cols,
                ],
                self.local_buf.at[rows, cols],
                self.local_sems.at[stream],
            )
            read.start()
            reads.append(read)
            descriptors.append((target, source, rows, cols))

        for stream, (target, source, rows, cols) in enumerate(descriptors):
            pl.when(self.step > 0)(
                lambda s=stream, t=target, r=rows, c=cols: self.remote(
                    self.step - 1, s, t, r, c
                ).wait()
            )
            reads[stream].wait()
            pl.when(self.step == 0)(lambda r=rows, c=cols: self.step_seed(r, c))
            pl.when(self.step > 0)(
                lambda s=stream, src=source, r=rows, c=cols: self.step_add(r, c, s, src)
            )
            pl.when(self.step < _CONST_31)(
                lambda s=stream, t=target, r=rows, c=cols: self.step_forward(s, t, r, c)
            )
            pl.when(self.step == _CONST_31)(
                lambda s=stream, r=rows, c=cols: self.step_finish(s, r, c)
            )


def _reduce_scatter_kernel(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Execute reduce scatter kernel logic."""
    return _ReduceScatterRunner(*_p_args, **_p_kw).run()


class _AllGatherRunner:
    """Runner for all-gather ring communication."""

    def __getattr__(self, name: _Dynamic) -> _Dynamic:
        """Dynamic attribute access."""
        raise AttributeError(name)

    def __init__(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Initialize all gather runner."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            self.tables,
            self.x,
            self.out,
            self.load_sems,
            self.send_sems,
            self.recv_sems,
            self.sem_credits,
            self.store_sems,
            self.payload,
            self.received,
        ) = (
            _p_next(_p_iter, _p_get("tables")),
            _p_next(_p_iter, _p_get("x")),
            _p_next(_p_iter, _p_get("out")),
            _p_next(_p_iter, _p_get("load_sems")),
            _p_next(_p_iter, _p_get("send_sems")),
            _p_next(_p_iter, _p_get("recv_sems")),
            _p_next(_p_iter, _p_get("credits")),
            _p_next(_p_iter, _p_get("store_sems")),
            _p_next(_p_iter, _p_get("payload")),
            _p_next(_p_iter, _p_get("received")),
        )
        self.regions = _dyn(_p_kw.get("regions"))
        self.buffers = _dyn(_p_get("buffers"))
        self.striped = _p_kw.get("striped", False)
        self.tile = pl.program_id(0)
        self.step = pl.program_id(1)
        self.rank = lax.axis_index("k3_sp")
        self.positions = [self.tables[f, 0, self.rank] for f in range(3)]
        self.local_first = self.tile * self.payload.shape[0]
        self.stripe = window_stripe(self.x.shape[0])

    def owner(self, pos: _Dynamic, family: _Dynamic) -> _Dynamic:
        """Execute owner logic."""
        return self.tables[family, 1, pos % 32]

    def peer(self, pos: _Dynamic, family: _Dynamic) -> tuple[_Dynamic]:
        """Execute peer logic."""
        return (self.owner(pos, family),)

    def remote(
        self, index: _Dynamic, rows: _Dynamic, cols: _Dynamic, s: int, target: _Dynamic
    ) -> _Dynamic:
        """Execute remote copy logic."""
        return pltpu.make_async_remote_copy(
            self.payload.at[rows, cols],
            self.received.at[index % self.buffers, rows, cols],
            self.send_sems.at[s],
            self.recv_sems.at[index % self.buffers, s],
            device_id=target,
            device_id_type=pl.DeviceIdType.MESH,
        )

    def store(
        self,
        index: _Dynamic,
        spec: tuple[_Dynamic, _Dynamic, _Dynamic, int, int, _Dynamic, int],
    ) -> list[_Dynamic]:
        """Execute store logic."""
        begin, size, cols, s, family, pos, direction = spec
        owner_mesh = self.owner(pos - direction * index, jnp.int32(family))
        dest = self.tables[family, 2, owner_mesh]
        if not self.striped:
            return [
                pltpu.make_async_copy(
                    self.payload.at[pl.ds(begin, size), cols],
                    self.out.at[
                        pl.ds(
                            dest * self.x.shape[0]
                            + self.tile * self.payload.shape[0]
                            + begin,
                            size,
                        ),
                        cols,
                    ],
                    self.store_sems.at[s],
                )
            ]
        return [
            pltpu.make_async_copy(
                self.payload.at[pl.ds(row, count), cols],
                self.out.at[pl.ds(sp_token_index(dest, row, self.stripe), count), cols],
                self.store_sems.at[s],
            )
            for _, row, count in _stripe_pieces(begin, size, self.stripe)
        ]

    def _do_load_all(self) -> None:
        """Start async copies for initial load."""
        for s, (begin, size, col, width) in enumerate(self.regions):
            rows, cols = pl.ds(begin, size), pl.ds(col, width)
            pltpu.make_async_copy(
                self.x.at[pl.ds(self.local_first + begin, size), cols],
                self.payload.at[rows, cols],
                self.load_sems.at[s],
            ).start()

    def _do_receive_step(
        self,
        stream: int,
        endpoints: tuple[_Dynamic, _Dynamic],
        bounds: tuple[_Dynamic, _Dynamic],
        spec: tuple[_Dynamic, _Dynamic, _Dynamic, int, int, _Dynamic, int],
    ) -> None:
        """Execute receive step for a stream."""
        target, source = endpoints
        rows, cols = bounds
        self.remote(self.step - 1, rows, cols, stream, target).wait()
        for copy in self.store(self.step - 1, spec):
            copy.wait()
        self.payload.at[rows, cols][...] = self.received.at[
            (self.step - 1) % self.buffers, rows, cols
        ][...]
        pl.semaphore_signal(
            self.sem_credits.at[stream],
            1,
            device_id=source,
            device_id_type=pl.DeviceIdType.MESH,
        )

    def _do_forward_step(
        self, stream: int, target: _Dynamic, rows: _Dynamic, cols: _Dynamic
    ) -> None:
        """Execute forward step for a stream."""
        pl.when(self.step >= self.buffers)(
            lambda: pl.semaphore_wait(self.sem_credits.at[stream], 1)
        )
        self.remote(self.step, rows, cols, stream, target).start()

    def _do_finish_step(
        self,
        stream: int,
        spec: tuple[_Dynamic, _Dynamic, _Dynamic, int, int, _Dynamic, int],
    ) -> None:
        """Execute finish step for a stream."""
        for copy in self.store(self.step, spec):
            copy.wait()
        pl.semaphore_wait(self.sem_credits.at[stream], self.buffers)

    def run(self) -> None:
        """Execute all gather step."""
        pl.when(self.step == 0)(lambda: _ring_enter(self.tables, self.rank))
        pl.when(self.step == 0)(self._do_load_all)

        for s, (begin, size, col, width) in enumerate(self.regions):
            family = (s // 2) % 3
            direction = -1 if s % 2 == 0 else 1
            pos = self.positions[family]
            target = self.peer(pos + direction, jnp.int32(family))
            source = self.peer(pos - direction, jnp.int32(family))
            rows, cols = pl.ds(begin, size), pl.ds(col, width)
            spec = (begin, size, cols, s, family, pos, direction)

            pl.when(self.step == 0)(
                lambda b=begin,
                sz=size,
                cl=cols,
                rw=rows,
                stream=s: pltpu.make_async_copy(
                    self.x.at[pl.ds(self.local_first + b, sz), cl],
                    self.payload.at[rw, cl],
                    self.load_sems.at[stream],
                ).wait()
            )

            pl.when(self.step > 0)(
                lambda st=s, tgt=target, src=source, rw=rows, cl=cols, sp=spec: (
                    self._do_receive_step(st, (tgt, src), (rw, cl), sp)
                )
            )

            for copy in self.store(self.step, spec):
                copy.start()

            pl.when(self.step < _CONST_31)(
                lambda st=s, tgt=target, rw=rows, cl=cols: (
                    self._do_forward_step(st, tgt, rw, cl)
                )
            )

            pl.when(self.step == _CONST_31)(
                lambda st=s, sp=spec: self._do_finish_step(st, sp)
            )


def _all_gather_kernel(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Execute all gather kernel logic."""
    return _AllGatherRunner(*_p_args, **_p_kw).run()


class _ProjectRingRunner:
    """Runner for project ring communication."""

    def __getattr__(self, name: _Dynamic) -> _Dynamic:
        """Dynamic attribute access."""
        raise AttributeError(name)

    def __init__(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Initialize project ring runner."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            self.tables_ref,
            self.x_ref,
            self.w_ref,
            self.o_ref,
            self.local_sems,
            self.send_sems,
            self.recv_sems,
            self.sem_credits,
            self.store_sem,
            self.local_buf,
            self.payload,
            self.received,
            self.weights,
            self.inputs,
            self.weight_sem,
        ) = (
            _p_next(_p_iter, _p_get("tables_ref")),
            _p_next(_p_iter, _p_get("x_ref")),
            _p_next(_p_iter, _p_get("w_ref")),
            _p_next(_p_iter, _p_get("o_ref")),
            _p_next(_p_iter, _p_get("local_sems")),
            _p_next(_p_iter, _p_get("send_sems")),
            _p_next(_p_iter, _p_get("recv_sems")),
            _p_next(_p_iter, _p_get("credits")),
            _p_next(_p_iter, _p_get("store_sem")),
            _p_next(_p_iter, _p_get("local_buf")),
            _p_next(_p_iter, _p_get("payload")),
            _p_next(_p_iter, _p_get("received")),
            _p_next(_p_iter, _p_get("weights")),
            _p_next(_p_iter, _p_get("inputs")),
            _p_next(_p_iter, _p_get("weight_sem")),
        )
        self.streams = _p_get("streams")
        self.buffers = _p_get("buffers")
        self.tile = pl.program_id(0)
        self.step = pl.program_id(1)
        self.rank = lax.axis_index("k3_sp")
        self.positions = [self.tables_ref[f, 0, self.rank] for f in range(3)]
        self.stripe = window_stripe(self.o_ref.shape[0])

    def ring_coords(self, pos: _Dynamic, family: _Dynamic) -> tuple[_Dynamic]:
        """Execute ring coords logic."""
        peer = self.tables_ref[family, 1, pos]
        return (peer,)

    def remote(
        self,
        index: _Dynamic,
        rows: _Dynamic,
        cols: _Dynamic,
        stream: int,
        target: _Dynamic,
    ) -> _Dynamic:
        """Execute remote copy logic."""
        return pltpu.make_async_remote_copy(
            self.payload.at[rows, cols],
            self.received.at[index % self.buffers, rows, cols],
            self.send_sems.at[stream],
            self.recv_sems.at[index % self.buffers, stream],
            device_id=target,
            device_id_type=pl.DeviceIdType.MESH,
        )

    def _do_add_step(
        self, stream: int, source: _Dynamic, rows: _Dynamic, cols: _Dynamic
    ) -> None:
        """Execute add step logic."""
        self.payload.at[rows, cols][...] = (
            self.local_buf.at[rows, cols][...].astype(self.payload.dtype)
            + self.received.at[(self.step - 1) % self.buffers, rows, cols][...]
        )
        pl.semaphore_signal(
            self.sem_credits.at[stream],
            1,
            device_id=source,
            device_id_type=pl.DeviceIdType.MESH,
        )

    def _do_forward_step(
        self, stream: int, target: _Dynamic, rows: _Dynamic, cols: _Dynamic
    ) -> None:
        """Execute forward step logic."""
        pl.when(self.step >= self.buffers)(
            lambda: pl.semaphore_wait(self.sem_credits.at[stream], 1)
        )
        self.remote(self.step, rows, cols, stream, target).start()

    def _do_finish_step(self, stream: int, rows: _Dynamic, cols: _Dynamic) -> None:
        """Execute finish step logic."""
        pl.semaphore_wait(self.sem_credits.at[stream], self.buffers)
        self.local_buf.at[rows, cols][...] = self.payload.at[rows, cols][...].astype(
            self.local_buf.dtype
        )
        output = pltpu.make_async_copy(
            self.local_buf.at[rows, cols],
            self.o_ref.at[
                pl.ds(self.tile * self.payload.shape[0] + rows.start, rows.size), cols
            ],
            self.store_sem,
        )
        output.start()
        output.wait()

    def run(self) -> None:
        """Execute project ring step."""
        pl.when(self.step == 0)(lambda: _ring_enter(self.tables_ref, self.rank))
        pl.when(self.step == 0)(
            lambda: pltpu.async_copy(self.w_ref, self.weights, self.weight_sem).wait()
        )

        descriptors = []
        reads = []
        for stream, region in enumerate(self.streams):
            begin, size = region[:2]
            col_begin, width = (
                region[2:] if len(region) == _CONST_4 else (0, self.o_ref.shape[1])
            )
            cols = pl.ds(col_begin, width)
            family = (stream // 2) % 3
            direction = -1 if stream % 2 == 0 else 1
            position = self.positions[family]
            target = self.ring_coords((position + direction) % 32, jnp.int32(family))
            source = self.ring_coords((position - direction) % 32, jnp.int32(family))
            peer = self.tables_ref[
                family, 1, (position - direction * (self.step + 1)) % 32
            ]
            owner = self.tables_ref[family, 2, peer]
            rows = pl.ds(begin, size)
            read = [
                pltpu.make_async_copy(
                    self.x_ref.at[
                        pl.ds(sp_token_index(owner, row, self.stripe), count), :
                    ],
                    self.inputs.at[stream, pl.ds(offset, count)],
                    self.local_sems.at[stream],
                )
                for offset, row, count in _stripe_pieces(begin, size, self.stripe)
            ]
            for copy in read:
                copy.start()
            reads.append(read)
            descriptors.append((target, source, rows, cols))

        for stream, (target, source, rows, cols) in enumerate(descriptors):
            pl.when(self.step > 0)(
                lambda s=stream, r=rows, c=cols, t=target: self.remote(
                    self.step - 1, r, c, s, t
                ).wait()
            )
            for copy in reads[stream]:
                copy.wait()
            self.local_buf.at[rows, cols][...] = jnp.dot(
                self.inputs[stream],
                self.weights[:, cols],
                preferred_element_type=jnp.float32,
            ).astype(jnp.bfloat16)

            pl.when(self.step == 0)(
                lambda r=rows, c=cols: self.payload.at[r, c].__setitem__(
                    slice(None), self.local_buf.at[r, c][...].astype(self.payload.dtype)
                )
            )

            pl.when(self.step > 0)(
                lambda s=stream, src=source, r=rows, c=cols: self._do_add_step(
                    s, src, r, c
                )
            )

            pl.when(self.step < _CONST_31)(
                lambda s=stream, t=target, r=rows, c=cols: self._do_forward_step(
                    s, t, r, c
                )
            )

            pl.when(self.step == _CONST_31)(
                lambda s=stream, r=rows, c=cols: self._do_finish_step(s, r, c)
            )


def _project_ring_kernel(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Execute project ring kernel logic."""
    return _ProjectRingRunner(*_p_args, **_p_kw).run()


def project_reduce_scatter(
    x: _Dynamic,
    w: _Dynamic,
    mesh: _Dynamic,
    tables: _Dynamic,
    *,
    collective_id: _Dynamic,
) -> _Dynamic:
    """Fuse a three-head BF16 attention projection with SP32 RS."""
    if (
        x.shape[0] != _CONST_32
        or x.shape[1] % 32
        or x.shape[2] != _CONST_384
        or w.shape != (32, 384, 7168)
    ):
        _err_msg = "K3 projection/RS requires SP32 and 3 local heads"
        raise ValueError(_err_msg)
    if x.dtype != jnp.bfloat16 or w.dtype != jnp.bfloat16:
        _err_msg = "K3 projection/RS requires BF16 operands"
        raise ValueError(_err_msg)
    tokens = x.shape[1]
    rows = tokens // 32
    if rows > _CONST_256:
        _err_msg = "K3 projection/RS needs at most 256 rows per rank"
        raise ValueError(_err_msg)
    regions = _regions(rows, 7168)
    streams = len(regions)
    chunk = max(size for _, size, _, _ in regions)

    def local(x: _Dynamic, w: _Dynamic) -> _Dynamic:
        """Execute local logic."""
        x = x.reshape(tokens, 384)
        w = w.reshape(384, 7168)
        hbm = pl.BlockSpec(memory_space=pltpu.HBM)
        scratch = (
            pltpu.SemaphoreType.DMA((streams,)),
            pltpu.SemaphoreType.DMA((streams,)),
            pltpu.SemaphoreType.DMA((2, streams)),
            pltpu.SemaphoreType.REGULAR((streams,)),
            pltpu.SemaphoreType.DMA,
            pltpu.VMEM((rows, 7168), jnp.bfloat16),
            pltpu.VMEM((rows, 7168), jnp.bfloat16),
            pltpu.VMEM((2, rows, 7168), jnp.bfloat16),
            pltpu.VMEM((384, 7168), jnp.bfloat16),
            pltpu.VMEM((streams, chunk, 384), jnp.bfloat16),
            pltpu.SemaphoreType.DMA,
        )
        return pl.pallas_call(
            functools.partial(_project_ring_kernel, streams=regions, buffers=2),
            out_shape=jax.ShapeDtypeStruct((rows, 7168), jnp.bfloat16),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                in_specs=(hbm, hbm),
                out_specs=hbm,
                scratch_shapes=scratch,
                grid=(1, 32),
            ),
            compiler_params=pltpu.CompilerParams(
                collective_id=collective_id, vmem_limit_bytes=40 * 1024**2
            ),
            name="kimi_attention_project_ring_rs",
        )(jnp.asarray(tables, jnp.int32), x, w)

    return jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(P("k3_sp", None, None), P("k3_sp", None, None)),
        out_specs=P("k3_sp", None),
        check_vma=False,
    )(x, w)


# ============================================================================
# Fused grouped matmul (expert FFN)
# ============================================================================
#
# The grouped matmul adapted from ``kernels/megablox/gmm_v2.py``: the routed
# experts' gate/up (fused SiTU) and down projections, MXFP4 weights.


def situ_and_mul(
    gate: jax.Array, up: jax.Array, beta: float, linear_beta: float | None
) -> jax.Array:
    """Execute situ and mul logic."""
    gate = beta * jnp.tanh(gate / beta) * jax.nn.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * jnp.tanh(up / linear_beta)
    return gate * up


def interleave_lane(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
    """Interleaves two arrays along lane dim at zero-cost."""
    if lhs.shape != rhs.shape:
        _err_msg = "Assertion failed: lhs.shape == rhs.shape"
        raise AssertionError(_err_msg)
    chunk_size = pltpu.get_tpu_info().num_lanes
    num_chunks = lhs.shape[-1] // chunk_size
    lhs_chunks = jnp.split(lhs, num_chunks, axis=-1)
    rhs_chunks = jnp.split(rhs, num_chunks, axis=-1)
    interleaved = []
    for i in range(num_chunks):
        interleaved += [lhs_chunks[i], rhs_chunks[i]]
    return jnp.concat(interleaved, axis=-1)


def deinterleave_lane(val: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Deinterleaves an array along lane dim at zero-cost."""
    chunk_size = pltpu.get_tpu_info().num_lanes
    num_chunks = val.shape[-1] // chunk_size
    chunks = jnp.split(val, num_chunks, axis=-1)
    lhs = jnp.concat(chunks[0::2], axis=-1)
    rhs = jnp.concat(chunks[1::2], axis=-1)
    return lhs, rhs


def apply_act_fn(acc: jax.Array, fuse_act: str | None) -> _Dynamic:
    """Applies a fused activation function to the accumulator.

    This function is used when an activation function is fused with the matrix
    multiplication. The input accumulator `acc` is expected to contain
    concatenated results for both the 'gate' and 'up' projections.

    Args:
        acc: The accumulator array, with the last dimension being 2 * tile_n.
        fuse_act: None, or ``"situ:<beta>:<linear_beta>"`` (``linear_beta``
            may be ``none``) for the SiTU gated activation.

    Returns:
        The result of applying the activation function.

    Raises:
        NotImplementedError: If an unsupported `fuse_act` is provided.
    """
    if fuse_act is None:
        return acc

    acc_gate, acc_up = deinterleave_lane(acc)
    match fuse_act:
        case str() if fuse_act.startswith("situ:"):
            _, beta, linear_beta = fuse_act.split(":")
            linear_beta = None if linear_beta == "none" else float(linear_beta)
            return situ_and_mul(acc_gate, acc_up, float(beta), linear_beta)
        case _:
            _err_msg = f"Unsupported activation function: {fuse_act}"
            raise NotImplementedError(_err_msg)


def align_to(x: _Dynamic, a: _Dynamic) -> _Dynamic:
    """Execute align to logic."""
    return pl.cdiv(x, a) * a


def _get_lhs_sublane_size(dtype: jnp.dtype, size_m: int) -> int:
    """Sublane block size the kernel processes lhs rows in."""
    size_lhs_sublane = pltpu.get_tpu_info().get_sublane_tiling(dtype)
    return min(size_lhs_sublane, size_m)


def get_packing_factor(storage_dtype: jnp.dtype, quant_dtype: jnp.dtype | None) -> int:
    """Retrieve get packing factor logic."""
    if quant_dtype is None or quant_dtype == storage_dtype:
        return 1
    storage_bits = jax.dtypes.itemsize_bits(storage_dtype)
    quant_bits = jax.dtypes.itemsize_bits(quant_dtype)
    packing_factor, remainder = divmod(storage_bits, quant_bits)
    if remainder != 0:
        _err_msg = (
            f"Storage dtype {storage_dtype} is not divisible by quant "
            f"dtype {quant_dtype}"
        )
        raise ValueError(_err_msg)
    return packing_factor


class RhsRef(ABC):
    """Abstract class that defines interfaces for rhs values."""

    @abstractmethod
    def get_weight(self) -> jax.Array:
        """Retrieve weight array."""
        ...

    @abstractmethod
    def get_scale(self, replicate_size: int | None = None) -> jax.Array:
        """Returns scale array, optionally replicated across sublanes."""
        ...


@dataclasses.dataclass(frozen=True)
class WeightsRef(RhsRef):
    """Dataclass for a single weights."""

    weight: _Dynamic
    scale: _Dynamic | None

    def __post_init__(self) -> None:
        """Execute post init logic."""
        _ensure_dataclasses_registered()

    def get_weight(self) -> jax.Array:
        """Retrieve get weight logic."""
        return self.weight[...]

    def get_scale(self, replicate_size: int | None = None) -> jax.Array:
        """Retrieve get scale logic."""
        if not (self.scale is not None):
            _err_msg = "Assertion failed: self.scale is not None"
            raise AssertionError(_err_msg)
        if self.scale.dtype == jnp.uint8:  # E8M0 exponents [blocks, n]
            exponents = self.scale[...]
            powers = e8m0_scale(exponents)[:, None, :]
            return jnp.broadcast_to(
                powers, (exponents.shape[0], replicate_size or 1, exponents.shape[1])
            )
        if replicate_size is not None:
            # Perform zero-stride load for efficient broadcasting across sublanes.
            return self.scale[:, pl.ds(0, replicate_size, 0), :]
        return self.scale[...]


@dataclasses.dataclass(frozen=True)
class FusedWeightsRef(RhsRef):
    """Dataclass for gate and up weights used in fused activation."""

    gate: WeightsRef
    up: WeightsRef

    def __post_init__(self) -> None:
        """Execute post init logic."""
        _ensure_dataclasses_registered()

    def get_weight(self) -> jax.Array:
        """Retrieve get weight logic."""
        w_gate = self.gate.get_weight()
        w_up = self.up.get_weight()
        return interleave_lane(w_gate, w_up)

    def get_scale(self, replicate_size: int | None = None) -> jax.Array:
        """Retrieve get scale logic."""
        s_gate = self.gate.get_scale(replicate_size)
        s_up = self.up.get_scale(replicate_size)
        return interleave_lane(s_gate, s_up)


@dataclasses.dataclass(frozen=True)
class MetadataRef:
    """Manage metadata ref instance and state."""

    gm_id_to_group_id: jax.Array
    gm_id_to_m_offset: jax.Array

    if TYPE_CHECKING:

        def __init__(
            self,
            gm_id_to_group_id: jax.Array | pl.MemoryRef,
            gm_id_to_m_offset: jax.Array | pl.MemoryRef,
        ) -> None: ...

        def get_array_aval(self) -> jax_core.AbstractValue: ...

        def get_ref_aval(self) -> AbstractRef | TransformedRef: ...

    def __post_init__(self) -> None:
        """Execute post init logic."""
        _ensure_dataclasses_registered()


_DATACLASSES_REGISTERED_BOX = [False]


def _ensure_dataclasses_registered() -> None:
    """Execute ensure dataclasses registered logic."""
    if _DATACLASSES_REGISTERED_BOX[0]:
        return
    _DATACLASSES_REGISTERED_BOX[0] = True
    jax.tree_util.register_dataclass(WeightsRef)
    jax.tree_util.register_dataclass(FusedWeightsRef)
    jax.tree_util.register_dataclass(MetadataRef)


@dataclasses.dataclass(frozen=True)
class TileSizes:
    """Manage tile sizes instance and state."""

    tile_m: int
    tile_k: int
    tile_n: int
    bucket_base: int


@dataclasses.dataclass(frozen=True)
class Dimensions:
    """Manage dimensions instance and state."""

    size_m: int
    size_k: int
    size_n: int
    size_group: int
    size_lhs_group: int
    size_lhs_sublane: int


@dataclasses.dataclass(frozen=True)
class InputConfigs:
    """Manage input configs instance and state."""

    quant_dtype: jnp.dtype | None
    quant_block_size: int | None
    dtype: jnp.dtype
    has_scale: bool = False
    # Scales as raw MXFP4 E8M0 exponent bytes ([group, blocks, n] uint8),
    # converted per tile in VMEM, instead of [group, blocks, 1, n] FP32.
    scale_e8m0: bool = False

    @property
    def should_unpack(self) -> bool:
        """True if rhs weights are sub-byte (e.g. INT4) packed in.

        INT32/UINT32 carriers.
        """
        return get_packing_factor(self.dtype, self.quant_dtype) > 1

    @property
    def should_dequantize_before_matmul(self) -> bool:
        """Dequantize rhs before matmul if block size limits MXU utilization."""
        if not self.has_scale:
            return False
        if not (self.quant_block_size is not None):
            _err_msg = "Assertion failed: self.quant_block_size is not None"
            raise AssertionError(_err_msg)
        mxu_size = pltpu.get_tpu_info().mxu_column_size
        return self.quant_block_size < mxu_size

    @property
    def should_dequantize_after_matmul(self) -> bool:
        """Execute should dequantize after matmul logic."""
        return self.has_scale and not self.should_dequantize_before_matmul


@dataclasses.dataclass(frozen=True)
class GmmConfigs:
    """Manage gmm configs instance and state."""

    tiles: TileSizes
    dims: Dimensions
    lhs_cfgs: InputConfigs
    rhs_cfgs: InputConfigs
    out_dtype: jnp.dtype
    acc_dtype: jnp.dtype
    fuse_act: str | None

    @property
    def num_quant_blocks_per_tile_k(self) -> int:
        """Execute num quant blocks per tile k logic."""
        if self.rhs_cfgs.quant_block_size is None:
            _msg = "Missing quant_block_size"
            raise ValueError(_msg)
        return pl.cdiv(self.tiles.tile_k, self.rhs_cfgs.quant_block_size)

    @property
    def out_size_n(self) -> int:
        """Execute out size n logic."""
        if self.fuse_act is None:
            return self.dims.size_n
        return self.dims.size_n // 2


TileFn = Callable[[Dimensions, InputConfigs, InputConfigs, int, str | None], TileSizes]


class IndexMaps:
    """Index maps for GMM kernel."""

    def __init__(self, metadata_ref: MetadataRef, cfgs: GmmConfigs) -> None:
        """Execute init logic."""
        self.metadata_ref = metadata_ref
        self.cfgs = cfgs

    def lhs_index_map(
        self, _: jax.Array, gm_id: jax.Array, k_id: jax.Array
    ) -> _Dynamic:
        """Execute lhs index map logic."""
        m_start = self.metadata_ref.gm_id_to_m_offset[gm_id]
        m_end = self.metadata_ref.gm_id_to_m_offset[gm_id + 1]

        row_start = m_start // self.cfgs.dims.size_lhs_sublane
        row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
        row_size = row_end - row_start

        return (pl.ds(row_start, row_size), 0, k_id)

    def rhs_weight_index_map(
        self, n_id: jax.Array, gm_id: jax.Array, k_id: jax.Array
    ) -> _Dynamic:
        """Execute rhs weight index map logic."""
        group_id = self.metadata_ref.gm_id_to_group_id[gm_id]
        return (group_id, k_id, n_id)

    def rhs_scale_index_map(
        self, n_id: jax.Array, gm_id: jax.Array, k_id: jax.Array
    ) -> _Dynamic:
        """Execute rhs scale index map logic."""
        group_id = self.metadata_ref.gm_id_to_group_id[gm_id]
        # Simply multiplying k_id by num_quant_blocks_per_tile_k will not work
        # since a single quant block could be shared along multiple k tile.
        k_row = k_id * self.cfgs.tiles.tile_k
        if self.cfgs.rhs_cfgs.quant_block_size is None:
            _msg = "Missing quant_block_size"
            raise ValueError(_msg)
        b_row = k_row // self.cfgs.rhs_cfgs.quant_block_size
        b_tile_id = b_row // self.cfgs.num_quant_blocks_per_tile_k
        if self.cfgs.rhs_cfgs.scale_e8m0:
            return (group_id, b_tile_id, n_id)
        return (group_id, b_tile_id, 0, n_id)

    def out_index_map(
        self, n_id: jax.Array, gm_id: jax.Array, _: jax.Array
    ) -> _Dynamic:
        """Execute out index map logic."""
        is_last_gm = gm_id == (pl.num_programs(1) - 1)
        m_start = self.metadata_ref.gm_id_to_m_offset[gm_id]
        m_end = self.metadata_ref.gm_id_to_m_offset[gm_id + 1]

        row_start = m_start // self.cfgs.dims.size_lhs_sublane
        capped_row_end = m_end // self.cfgs.dims.size_lhs_sublane
        last_row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
        row_end = jnp.where(is_last_gm, last_row_end, capped_row_end)
        row_size = row_end - row_start

        return (pl.ds(row_start, row_size), 0, n_id)


def generate_block_specs(
    metadata_ref: MetadataRef, cfgs: GmmConfigs
) -> tuple[tuple[pl.BlockSpec, WeightsRef], pl.BlockSpec]:
    """Generates block specs for the given lhs, rhs, and out refs."""
    index_map = IndexMaps(metadata_ref, cfgs)
    bounded_slice_gm = pl.BoundedSlice(cfgs.tiles.tile_m // cfgs.dims.size_lhs_sublane)

    lhs_block_spec = pl.BlockSpec(
        (bounded_slice_gm, cfgs.dims.size_lhs_sublane, cfgs.tiles.tile_k),
        index_map.lhs_index_map,
    )

    rhs_weight_spec = pl.BlockSpec(
        (None, cfgs.tiles.tile_k, cfgs.tiles.tile_n),
        index_map.rhs_weight_index_map,
        pipeline_mode=pl.Buffered(buffer_count=3),
    )
    rhs_scale_block_spec = None
    if cfgs.rhs_cfgs.has_scale:
        rhs_scale_block_spec = pl.BlockSpec(
            (None, cfgs.num_quant_blocks_per_tile_k, cfgs.tiles.tile_n)
            if cfgs.rhs_cfgs.scale_e8m0
            else (None, cfgs.num_quant_blocks_per_tile_k, 1, cfgs.tiles.tile_n),
            index_map.rhs_scale_index_map,
        )

    rhs_block_spec = WeightsRef(
        weight=rhs_weight_spec,
        scale=rhs_scale_block_spec,
    )

    out_block_spec = pl.BlockSpec(
        (bounded_slice_gm, cfgs.dims.size_lhs_sublane, cfgs.tiles.tile_n),
        index_map.out_index_map,
    )

    return (lhs_block_spec, rhs_block_spec), out_block_spec


class _GmmInnerKernelRunner:
    """Runner for inner GMM kernel matmul pipeline."""

    def __getattr__(self, name: _Dynamic) -> _Dynamic:
        """Dynamic attribute access."""
        raise AttributeError(name)

    def __init__(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Initialize runner with arguments."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            self.tiled_lhs_ref,
            self.tiled_rhs_ref,
            self.tiled_out_ref,
            self.partial_out_ref,
            self.acc_ref,
            self.metadata_ref,
        ) = (
            _p_next(_p_iter, _p_get("tiled_lhs_ref")),
            _p_next(_p_iter, _p_get("tiled_rhs_ref")),
            _p_next(_p_iter, _p_get("tiled_out_ref")),
            _p_next(_p_iter, _p_get("partial_out_ref")),
            _p_next(_p_iter, _p_get("acc_ref")),
            _p_next(_p_iter, _p_get("metadata_ref")),
        )
        self.cfgs = _p_get("cfgs")
        self.gm_id = pl.program_id(1)
        m_start = self.metadata_ref.gm_id_to_m_offset[self.gm_id]
        m_end = self.metadata_ref.gm_id_to_m_offset[self.gm_id + 1]
        m_offset = m_start - m_start % self.cfgs.dims.size_lhs_sublane
        self.m_start_local = m_start - m_offset
        self.m_end_local = m_end - m_offset

    def _prepare_rhs(
        self, tiled_rhs: _Dynamic, is_last_k_step: _Dynamic
    ) -> tuple[_Dynamic, int]:
        """Prepare and dequantize RHS tile if required."""
        rhs_tile_n = tiled_rhs.shape[1]
        rhs_qbs = self.cfgs.rhs_cfgs.quant_block_size or self.cfgs.tiles.tile_k
        if self.cfgs.rhs_cfgs.should_dequantize_before_matmul:
            tiled_rhs_scale = self.tiled_rhs_ref.get_scale(
                replicate_size=rhs_qbs
            ).astype(self.cfgs.lhs_cfgs.dtype)
            num_blocks = self.cfgs.num_quant_blocks_per_tile_k
            tiled_rhs_dequant = tiled_rhs.astype(self.cfgs.lhs_cfgs.dtype).reshape(
                num_blocks, rhs_qbs, rhs_tile_n
            )
            tiled_rhs_dequant = tiled_rhs_dequant * tiled_rhs_scale
            tiled_rhs = tiled_rhs_dequant.reshape(self.cfgs.tiles.tile_k, rhs_tile_n)
            rhs_qbs = self.cfgs.tiles.tile_k

        valid_k = self.cfgs.dims.size_k % self.cfgs.tiles.tile_k
        if is_last_k_step and valid_k != 0:
            mask_rhs = lax.broadcasted_iota(jnp.int32, tiled_rhs.shape, 0) < valid_k
            tiled_rhs = jnp.where(mask_rhs, tiled_rhs, 0)
        return tiled_rhs, rhs_qbs

    def _compute_matmul(
        self,
        tiled_lhs: _Dynamic,
        tiled_rhs: _Dynamic,
        bucket_m: int,
        rhs_qbs: int,
    ) -> _Dynamic:
        """Compute block matrix multiplication."""
        tpu_info = pltpu.get_tpu_info()
        mxu_size = tpu_info.mxu_column_size
        rhs_tile_n = tiled_rhs.shape[1]
        acc_list = []
        for start_n in range(0, rhs_tile_n, mxu_size):
            end_n = min(rhs_tile_n, start_n + mxu_size)
            col_size = end_n - start_n
            acc_n = jnp.zeros((bucket_m, col_size), dtype=self.acc_ref.dtype)
            for start_k in range(0, self.cfgs.tiles.tile_k, rhs_qbs):
                end_k = min(self.cfgs.tiles.tile_k, start_k + rhs_qbs)
                block_acc = jnp.matmul(
                    tiled_lhs[:, start_k:end_k],
                    tiled_rhs[start_k:end_k, start_n:end_n],
                    preferred_element_type=jnp.float32,
                ).astype(self.acc_ref.dtype)
                if self.cfgs.rhs_cfgs.should_dequantize_after_matmul:
                    b_id = start_k // rhs_qbs
                    rhs_scale_replicated = self.tiled_rhs_ref.get_scale(
                        replicate_size=bucket_m
                    )[b_id, :, start_n : start_n + col_size]
                    block_acc *= rhs_scale_replicated.astype(self.acc_ref.dtype)
                acc_n += block_acc
            acc_list.append(acc_n)
        return jnp.concatenate(acc_list, axis=1)

    def _postprocess_output(
        self,
        acc: _Dynamic,
        is_first_k_step: _Dynamic,
        is_last_k_step: _Dynamic,
        bucket_m: int,
    ) -> None:
        """Postprocess accumulator output and store result."""
        if not is_first_k_step:
            acc = jnp.pad(acc, ((self.cfgs.tiles.tile_m - bucket_m, 0), (0, 0)))
            acc += self.acc_ref[...]
        acc_m = acc.shape[0]

        if is_last_k_step:
            acc = apply_act_fn(acc, self.cfgs.fuse_act)
            iota = lax.broadcasted_iota(jnp.int32, acc.shape, 0)
            mask = jnp.logical_and(self.m_start_local <= iota, iota < self.m_end_local)
            acc_masked = jnp.where(mask, acc, 0)
            tiled_out_2d_ref = self.tiled_out_ref.reshape(-1, self.cfgs.tiles.tile_n)
            tiled_out_2d_ref[:acc_m] = acc_masked.astype(self.tiled_out_ref.dtype)
            partial_out_zeros = jnp.zeros_like(self.partial_out_ref)
            self.tiled_out_ref[0] += jnp.where(
                self.gm_id == 0, partial_out_zeros, self.partial_out_ref[...]
            )
            last_row = self.m_end_local // self.cfgs.dims.size_lhs_sublane
            self.partial_out_ref[...] = jnp.where(
                self.m_end_local % self.cfgs.dims.size_lhs_sublane == 0,
                partial_out_zeros,
                self.tiled_out_ref[last_row],
            )
        else:
            self.acc_ref[:acc_m] = acc

    def step_matmul(
        self, is_first_k_step: _Dynamic, is_last_k_step: _Dynamic, bucket_m: int
    ) -> None:
        """Execute one matmul step."""
        tiled_lhs = self.tiled_lhs_ref.reshape(-1, self.cfgs.tiles.tile_k)[:bucket_m]
        tiled_rhs = self.tiled_rhs_ref.get_weight()
        tiled_rhs, rhs_qbs = self._prepare_rhs(tiled_rhs, is_last_k_step)
        acc = self._compute_matmul(tiled_lhs, tiled_rhs, bucket_m, rhs_qbs)
        self._postprocess_output(acc, is_first_k_step, is_last_k_step, bucket_m)

    def run_matmul_step(self, bucket_idx: int) -> None:
        """Execute run matmul step logic."""
        bucket_m = self.cfgs.tiles.bucket_base * (bucket_idx + 1)
        num_k = pl.num_programs(2)
        k_id = pl.program_id(2)
        is_first_k_step = k_id == 0
        is_last_k_step = k_id == (num_k - 1)

        if bucket_m == self.cfgs.tiles.tile_m:
            lax.cond(
                is_first_k_step,
                lambda: lax.cond(
                    is_last_k_step,
                    lambda: self.step_matmul(
                        is_first_k_step=True, is_last_k_step=True, bucket_m=bucket_m
                    ),
                    lambda: self.step_matmul(
                        is_first_k_step=True, is_last_k_step=False, bucket_m=bucket_m
                    ),
                ),
                lambda: lax.cond(
                    is_last_k_step,
                    lambda: self.step_matmul(
                        is_first_k_step=False, is_last_k_step=True, bucket_m=bucket_m
                    ),
                    lambda: self.step_matmul(
                        is_first_k_step=False, is_last_k_step=False, bucket_m=bucket_m
                    ),
                ),
            )
        else:
            lax.cond(
                is_first_k_step,
                lambda: self.step_matmul(
                    is_first_k_step=True, is_last_k_step=True, bucket_m=bucket_m
                ),
                lambda: self.step_matmul(
                    is_first_k_step=False, is_last_k_step=True, bucket_m=bucket_m
                ),
            )

    def run(self) -> _Dynamic:
        """Run GMM inner kernel."""
        num_buckets = self.cfgs.tiles.tile_m // self.cfgs.tiles.bucket_base
        branches = [
            functools.partial(self.run_matmul_step, bucket_idx=bucket_idx)
            for bucket_idx in range(num_buckets)
        ]
        bucket_idx = self.m_end_local // self.cfgs.tiles.bucket_base
        lax.switch(bucket_idx, branches)


def inner_kernel(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Inner kernel invoked by emit_pipeline to perform matmul."""
    return _GmmInnerKernelRunner(*_p_args, **_p_kw).run()


def fill_metadata(
    lhs_group_sizes_ref: jax.Array,  # int32[size_lhs_group]
    group_offset_ref: jax.Array,  # int32[1]
    metadata_ref: MetadataRef,
    *,
    cfgs: GmmConfigs,
) -> jax.Array:
    """Fills the metadata for the given lhs group sizes and group offset.

    Iterates over the lhs group sizes and if the group id is valid, determines
    the number of gm tiles that are needed to process the current group. Then,
    it fills starting and ending offset (gm_id_to_m_offset), and the group id
    (gm_id_to_group_id) for each gm tile.

    Args:
        lhs_group_sizes_ref: The group sizes of lhs.
        group_offset_ref: Offset of the first group to process.
        metadata_ref: Metadata that is used to determine the group id and m offsets
            for each gmm tile.
        cfgs: GmmConfigs.

    Returns:
        The number of gm tiles to process lhs with given group offset.
    """
    group_offset = group_offset_ref[0]
    max_num_group = group_offset + cfgs.dims.size_group
    metadata_ref.gm_id_to_m_offset[0] = 0

    @jax.named_scope("inner_tm_loop")
    def inner_tm_loop(
        tm_id: _Dynamic,
        curr_m_offset: _Dynamic,
        *,
        end_m_offset: _Dynamic,
        group_id: _Dynamic,
    ) -> _Dynamic:
        """Execute inner tm loop logic."""
        local_offset = curr_m_offset % cfgs.dims.size_lhs_sublane
        tm_size = jnp.minimum(
            cfgs.tiles.tile_m - local_offset, end_m_offset - curr_m_offset
        )

        metadata_ref.gm_id_to_group_id[tm_id] = group_id

        next_m_offset = curr_m_offset + tm_size
        metadata_ref.gm_id_to_m_offset[tm_id] = curr_m_offset
        metadata_ref.gm_id_to_m_offset[tm_id + 1] = next_m_offset

        return next_m_offset

    @jax.named_scope("outer_group_loop")
    def outer_group_loop(lhs_group_id: _Dynamic, carry: _Dynamic) -> _Dynamic:
        """Execute outer group loop logic."""
        num_gm, start_m_offset = carry

        group_id = lhs_group_id - group_offset
        group_size = lhs_group_sizes_ref[lhs_group_id]
        end_m_offset = start_m_offset + group_size

        # Assume following arguments:
        # - size_lhs_sublane & tile_m = 4
        # - group_size = 3
        # - start_m_offset = 7
        #
        # If we visualize it, it will look like this where:
        # - |: denotes boundaries between sublanes
        # - 0: denotes values for other groups
        # - 1: denotes values for the current group
        # | 0 0 0 0 | 0 0 0 1 | 1 1 0 0 |
        #
        # In this example, we see that we require processing 2 m tiles.
        # But, performing a naive cdiv(group_size, tile_m) will return 1.
        # Instead, adding local_offset will give us the correct value.
        local_offset = start_m_offset % cfgs.dims.size_lhs_sublane
        aligned_group_size = group_size + local_offset
        curr_num_gm = pl.cdiv(aligned_group_size, cfgs.tiles.tile_m)

        # We need to handle cases where we should not process the group.
        # 1. Even if group_size is 0, if local_offset is not 0, cdiv will return 1.
        # 2. If group comes before the group_offset, we should not process it.
        should_process = jnp.logical_and(group_size > 0, group_id >= 0)
        curr_num_gm = jnp.where(should_process, curr_num_gm, 0)
        next_num_gm = num_gm + curr_num_gm

        tm_loop_fn = functools.partial(
            inner_tm_loop,
            end_m_offset=end_m_offset,
            group_id=group_id,
        )
        lax.fori_loop(num_gm, next_num_gm, tm_loop_fn, start_m_offset)

        return next_num_gm, end_m_offset

    num_gm, _ = lax.fori_loop(0, max_num_group, outer_group_loop, (0, 0))
    return num_gm


def kernel_main(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Entry point for GMM kernel.

    Computes metadata to determine which rows of lhs needs processing and how
    they will be tiled. And then, invoke inner kernel using metadata.

    Uses the following notation:
    - g: rhs group dimension
    - m: Batch dimension
    - gm: Batch tiling dimension. Aligned to size_lhs_sublane and has tile size
        of tile_m. Skips over empty groups and accounts for revisited tiles.
    - k: in dimension
    - n: out dimension

    Args:
        lhs_group_sizes_ref: Reference to the group sizes of lhs.
        group_offset_ref: Reference to the group offset.
        lhs_ref: Reference to the lhs.
        rhs_ref: Reference to the rhs.
        out_ref: Reference to the out.
        partial_out_ref: Reference to the partial output.
        acc_ref: Reference to the accumulator.
        metadata_ref: Reference to the metadata.
        cfgs: GmmConfigs.
    """
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        lhs_group_sizes_ref,
        group_offset_ref,
        rhs_layer_ref,
        lhs_ref,
        rhs_ref,
        out_ref,
        partial_out_ref,
        acc_ref,
        metadata_ref,
    ) = (
        _p_next(_p_iter, _p_get("lhs_group_sizes_ref")),
        _p_next(_p_iter, _p_get("group_offset_ref")),
        _p_next(_p_iter, _p_get("rhs_layer_ref")),
        _p_next(_p_iter, _p_get("lhs_ref")),
        _p_next(_p_iter, _p_get("rhs_ref")),
        _p_next(_p_iter, _p_get("out_ref")),
        _p_next(_p_iter, _p_get("partial_out_ref")),
        _p_next(_p_iter, _p_get("acc_ref")),
        _p_next(_p_iter, _p_get("metadata_ref")),
    )
    cfgs = _p_get("cfgs")
    num_k = pl.cdiv(cfgs.dims.size_k, cfgs.tiles.tile_k)
    num_n = pl.cdiv(cfgs.out_size_n, cfgs.tiles.tile_n)

    if rhs_ref.weight.ndim == _CONST_4:
        # Select the HBM view before the pipeline loads a tile. An array-level
        # dynamic slice would copy every expert in this layer into a new buffer.
        rhs_ref = dataclasses.replace(
            rhs_ref,
            weight=rhs_ref.weight.at[rhs_layer_ref[0]],
        )
        if (
            rhs_ref.scale is not None
            and rhs_ref.scale.dtype == jnp.uint8
            and rhs_ref.scale.ndim == _CONST_4
        ):
            rhs_ref = dataclasses.replace(
                rhs_ref, scale=rhs_ref.scale.at[rhs_layer_ref[0]]
            )

    if cfgs.rhs_cfgs.should_unpack:
        rhs_weight = rhs_ref.weight.bitcast(cfgs.rhs_cfgs.quant_dtype)
        rhs_ref = dataclasses.replace(rhs_ref, weight=rhs_weight)

    # Fill metadata buffer and return number of group & m iterations.
    num_gm = fill_metadata(
        lhs_group_sizes_ref,
        group_offset_ref,
        metadata_ref,
        cfgs=cfgs,
    )

    (lhs_spec, rhs_spec), out_spec = generate_block_specs(metadata_ref, cfgs)

    if cfgs.fuse_act is not None:
        rhs_up_ref = jax.tree.map(lambda x: x.at[..., cfgs.out_size_n :], rhs_ref)
        fused_rhs_ref: _Dynamic = FusedWeightsRef(gate=rhs_ref, up=rhs_up_ref)
        fused_rhs_spec: _Dynamic = FusedWeightsRef(
            gate=rhs_spec,
            up=rhs_spec,
        )
    else:
        fused_rhs_ref = rhs_ref
        fused_rhs_spec = rhs_spec

    # Execute the inner kernel.
    pipeline_fn = pltpu.emit_pipeline(
        functools.partial(inner_kernel, cfgs=cfgs),
        grid=(num_n, num_gm, num_k),
        in_specs=(lhs_spec, fused_rhs_spec),
        out_specs=out_spec,
    )

    # Bounded slice requires second last dim to be aligned to the sublane size.
    # rhs_ref uses static tiling thus reshape is not needed.
    lhs_in = lhs_ref.reshape(-1, cfgs.dims.size_lhs_sublane, lhs_ref.shape[-1])
    out_in = out_ref.reshape(-1, cfgs.dims.size_lhs_sublane, out_ref.shape[-1])
    scratches = [partial_out_ref, acc_ref, metadata_ref]
    pipeline_fn(lhs_in, fused_rhs_ref, out_in, scratches=scratches)


def _is_tk_quant_block_compatible(tk: int, quant_block_size: int | None) -> bool:
    """Execute is tile k quant block compatible logic."""
    if quant_block_size is None:
        return False
    return tk % quant_block_size == 0 or quant_block_size % tk == 0


def _estimate_gmm_vmem(
    tiles: tuple[int, int, int],
    lhs_cfgs: InputConfigs,
    rhs_cfgs: InputConfigs,
    *,
    fuse_act_factor: int = 1,
) -> int:
    """Execute gmm vmem estimate logic."""
    tm, tn, tk = tiles
    lhs_bits = jax.dtypes.itemsize_bits(lhs_cfgs.dtype)
    rhs_bits = jax.dtypes.itemsize_bits(rhs_cfgs.quant_dtype or rhs_cfgs.dtype)
    lhs_tile_bytes = lhs_bits // 8
    lhs_vmem = 2 * tm * tk * lhs_tile_bytes

    rhs_weight_vmem = tk * tn * rhs_bits // 8
    rhs_scale_vmem = 0
    if rhs_cfgs.has_scale and rhs_cfgs.quant_block_size is not None:
        num_quant_blocks_per_tile_k = pl.cdiv(tk, rhs_cfgs.quant_block_size)
        rhs_scale_vmem = num_quant_blocks_per_tile_k * tn * 4
    rhs_vmem = fuse_act_factor * (3 * rhs_weight_vmem + 2 * rhs_scale_vmem)

    acc_cols = fuse_act_factor * tn
    acc_vmem = tm * acc_cols * 4

    out_dtype_bytes = jax.dtypes.itemsize_bits(lhs_cfgs.dtype) // 8
    out_vmem = 2 * tm * tn * out_dtype_bytes

    return lhs_vmem + rhs_vmem + acc_vmem + out_vmem


def calculate_tiling(
    dims: Dimensions,
    lhs_cfgs: InputConfigs,
    rhs_cfgs: InputConfigs,
    vmem_limit_bytes: int,
    fuse_act: str | None = None,
) -> TileSizes:
    """Calculate optimal tile sizes for GMM kernel."""
    rhs_dtype = rhs_cfgs.quant_dtype or rhs_cfgs.dtype
    rhs_bits = jax.dtypes.itemsize_bits(rhs_dtype)

    bf16_bf16_tile_m = 128
    rhs_mod = min(pl.cdiv(16, rhs_bits), 2)
    tile_m = min(bf16_bf16_tile_m // rhs_mod, dims.size_m)

    tile_n_limit = min(pltpu.get_tpu_info().mxu_column_size * 2, dims.size_n)

    size_n_per_rhs = dims.size_n
    fuse_act_factor = 1
    if fuse_act is not None:
        fuse_act_factor = 2
        size_n_per_rhs //= fuse_act_factor
        tile_n_limit //= fuse_act_factor

    num_k_tiles = num_n_tiles = 1
    num_lanes = pltpu.get_tpu_info().num_lanes
    tile_k = align_to(dims.size_k, num_lanes)
    tile_n = align_to(size_n_per_rhs, num_lanes)

    def vmem_est(m: int, n: int, k: int) -> int:
        """Estimate VMEM usage for given tile dimensions."""
        return _estimate_gmm_vmem(
            (m, n, k), lhs_cfgs, rhs_cfgs, fuse_act_factor=fuse_act_factor
        )

    while vmem_est(tile_m, tile_n, tile_k) > vmem_limit_bytes and tile_n > tile_n_limit:
        num_n_tiles += 1
        tile_n = align_to(size_n_per_rhs, num_n_tiles * num_lanes) // num_n_tiles

    if tile_n < tile_n_limit:
        num_n_tiles -= 1
        tile_n = align_to(size_n_per_rhs, num_n_tiles * num_lanes) // num_n_tiles

        while (
            vmem_est(tile_m, tile_n, tile_k) > vmem_limit_bytes
            or not _is_tk_quant_block_compatible(tile_k, rhs_cfgs.quant_block_size)
        ) and tile_k > num_lanes:
            num_k_tiles += 1
            tile_k = align_to(dims.size_k, num_k_tiles * num_lanes) // num_k_tiles

    if tile_n == 0 or tile_k == 0:
        final_estimate = vmem_est(tile_m, tile_n, tile_k)
        _err_msg = (
            f"Could not find valid tile sizes for dims={dims!r} and "
            f"final_estimate={final_estimate!r} (limit: "
            f"{vmem_limit_bytes})."
        )
        raise ValueError(_err_msg)

    max_num_buckets = 4
    bucket_base = tile_m
    for _ in range(1, max_num_buckets):
        new_tile_m = tile_m + bucket_base
        if (
            new_tile_m > dims.size_m
            or vmem_est(new_tile_m, tile_n, tile_k) > vmem_limit_bytes
        ):
            break
        tile_m = new_tile_m

    return TileSizes(
        tile_m=tile_m, tile_k=tile_k, tile_n=tile_n, bucket_base=bucket_base
    )


def _validate_rhs_scale_spec(
    rhs_scale: _Dynamic, size_group: int, size_n: int, size_k: int
) -> None:
    """Validate RHS scale tensor dimensions."""
    num_quant_blocks = rhs_scale.shape[1]
    if rhs_scale.dtype == jnp.uint8:
        if rhs_scale.shape != (size_group, num_quant_blocks, size_n):
            _err_msg = (
                "Assertion failed: rhs_scale.shape == (size_group, "
                "num_quant_blocks, size_n)"
            )
            raise AssertionError(_err_msg)
    elif rhs_scale.shape != (size_group, num_quant_blocks, 1, size_n):
        _err_msg = (
            "Assertion failed: rhs_scale.shape == (size_group, "
            "num_quant_blocks, 1, size_n)"
        )
        raise AssertionError(_err_msg)
    if size_k % num_quant_blocks != 0:
        _err_msg = "Assertion failed: size_k % num_quant_blocks == 0"
        raise AssertionError(_err_msg)


def validate_inputs(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Validates the inputs for the GMM kernel."""
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        lhs,
        rhs,
        rhs_scale,
        group_sizes,
        group_offset,
        fuse_act,
        packing_factor,
    ) = (
        _p_next(_p_iter, _p_get("lhs")),
        _p_next(_p_iter, _p_get("rhs")),
        _p_next(_p_iter, _p_get("rhs_scale")),
        _p_next(_p_iter, _p_get("group_sizes")),
        _p_next(_p_iter, _p_get("group_offset")),
        _p_next(_p_iter, _p_get("fuse_act")),
        _p_next(_p_iter, _p_get("packing_factor", 1)),
    )
    size_m = lhs.shape[0]
    size_group, size_k, size_n = rhs.shape
    size_k *= packing_factor
    size_lhs_group = group_sizes.shape[0]

    if not (size_group <= size_lhs_group):
        _err_msg = "Assertion failed: size_group <= size_lhs_group"
        raise AssertionError(_err_msg)
    if lhs.shape != (size_m, size_k):
        _err_msg = "Assertion failed: lhs.shape == (size_m, size_k)"
        raise AssertionError(_err_msg)
    if rhs.shape != (size_group, size_k // packing_factor, size_n):
        _err_msg = (
            "Assertion failed: rhs.shape == (size_group, size_k // "
            "packing_factor, size_n)"
        )
        raise AssertionError(_err_msg)
    if rhs_scale is not None:
        _validate_rhs_scale_spec(rhs_scale, size_group, size_n, size_k)

    if group_offset.shape != (1,):
        _err_msg = "Assertion failed: group_offset.shape == (1,)"
        raise AssertionError(_err_msg)

    size_lhs_sublane = _get_lhs_sublane_size(lhs.dtype, size_m)
    if fuse_act is not None:
        num_lanes = pltpu.get_tpu_info().num_lanes
        if size_n % (2 * num_lanes) != 0:
            _err_msg = (
                f"size_n={size_n!r} should be divisible by 2 * num_lanes when "
                f"fuse_act is enabled since we need to split n dimension for "
                f"gate and up."
            )
            raise ValueError(_err_msg)

    return Dimensions(
        size_m=size_m,
        size_k=size_k,
        size_n=size_n,
        size_group=size_group,
        size_lhs_group=size_lhs_group,
        size_lhs_sublane=size_lhs_sublane,
    )


def get_cost_estimate(cfgs: GmmConfigs) -> _Dynamic:
    """Returns the cost estimate for the GMM kernel."""
    dims = cfgs.dims
    lhs_dtype = cfgs.lhs_cfgs.dtype
    rhs_dtype = cfgs.rhs_cfgs.quant_dtype or cfgs.rhs_cfgs.dtype

    # We use bits for rhs since it could be a sub-byte dtype like int4.
    rhs_bits = jax.dtypes.itemsize_bits(rhs_dtype)
    fp32_bytes = jnp.dtype(jnp.float32).itemsize

    # NOTE(kyuyeunk): Add compute flops for dequant.
    flops = 2 * dims.size_m * dims.size_k * dims.size_n

    lhs_bytes = dims.size_m * dims.size_k * lhs_dtype.itemsize

    rhs_size = dims.size_group * dims.size_k * dims.size_n
    rhs_bytes = rhs_size * rhs_bits // 8
    if cfgs.rhs_cfgs.has_scale:
        if cfgs.rhs_cfgs.quant_block_size is None:
            _msg = "Missing quant_block_size"
            raise ValueError(_msg)
        num_quant_blocks = pl.cdiv(dims.size_k, cfgs.rhs_cfgs.quant_block_size)
        rhs_bytes += dims.size_group * num_quant_blocks * dims.size_n * fp32_bytes

    out_bytes = dims.size_m * cfgs.out_size_n * cfgs.out_dtype.itemsize

    total_bytes = lhs_bytes + rhs_bytes + out_bytes

    return pl.CostEstimate(
        flops=flops,
        bytes_accessed=total_bytes,
        transcendentals=0,
    )


def get_scope_name(cfgs: GmmConfigs) -> str:
    """Retrieve get scope name logic."""
    dims = cfgs.dims
    tiles = cfgs.tiles
    return (
        f"gmm_v2-g_{dims.size_group}-m_{dims.size_m}-k_{dims.size_k}-act_{cfgs.fuse_act}"
        f"-n_{dims.size_n}-tm_{tiles.tile_m}-tk_{tiles.tile_k}-tn_{tiles.tile_n}"
    )


def make_gmm_configs(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Fills the GMM config for the GMM kernel."""
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        lhs,
        rhs,
        rhs_scale,
        group_sizes,
        group_offset,
    ) = (
        _p_next(_p_iter, _p_get("lhs")),
        _p_next(_p_iter, _p_get("rhs")),
        _p_next(_p_iter, _p_get("rhs_scale")),
        _p_next(_p_iter, _p_get("group_sizes")),
        _p_next(_p_iter, _p_get("group_offset")),
    )
    (
        tile_info,
        vmem_limit_bytes,
        out_dtype,
        acc_dtype,
        fuse_act,
        rhs_quant_dtype,
    ) = (
        _p_get("tile_info"),
        _p_get("vmem_limit_bytes"),
        _p_get("out_dtype"),
        _p_get("acc_dtype"),
        _p_get("fuse_act"),
        _p_get("rhs_quant_dtype"),
    )
    packing_factor = get_packing_factor(rhs.dtype, rhs_quant_dtype)

    dims = validate_inputs(
        lhs,
        rhs,
        rhs_scale,
        group_sizes,
        group_offset,
        fuse_act,
        packing_factor,
    )

    if rhs_scale is not None:
        has_scale = True
        rhs_quant_dtype = rhs_quant_dtype or rhs.dtype
        num_blocks = rhs_scale.shape[1]
        block_size = dims.size_k // num_blocks
    else:
        has_scale = False
        block_size = dims.size_k

    rhs_cfgs = InputConfigs(
        quant_dtype=rhs_quant_dtype,
        quant_block_size=block_size,
        dtype=rhs.dtype,
        has_scale=has_scale,
        scale_e8m0=rhs_scale is not None and rhs_scale.dtype == jnp.uint8,
    )

    # The lhs is never quantized: the matmul runs in its own dtype.
    lhs_cfgs = InputConfigs(quant_dtype=None, quant_block_size=None, dtype=lhs.dtype)

    if out_dtype is None:
        out_dtype = lhs.dtype

    if acc_dtype is None:
        acc_dtype = jnp.float32.dtype

    if isinstance(tile_info, TileSizes):
        tiles = tile_info
    else:
        tiles = tile_info(
            dims, lhs_cfgs, rhs_cfgs, vmem_limit_bytes or (40 * 1024**2), fuse_act
        )

    return GmmConfigs(
        dims=dims,
        tiles=tiles,
        lhs_cfgs=lhs_cfgs,
        rhs_cfgs=rhs_cfgs,
        out_dtype=jnp.dtype(out_dtype),
        acc_dtype=jnp.dtype(acc_dtype),
        fuse_act=fuse_act,
    )


def get_metadata(cfgs: GmmConfigs) -> dict[str, str]:
    """Retrieve get metadata logic."""
    cfgs_dict = dataclasses.asdict(cfgs)
    ret: dict[str, str] = {}
    for path, val in jax.tree_util.tree_leaves_with_path(cfgs_dict):
        key = jax.tree_util.keystr(path, simple=True, separator=".")
        ret[key] = str(val)
    return ret


@jax.jit(
    static_argnames=[
        "tile_info",
        "vmem_limit_bytes",
        "precision",
        "preferred_element_type",
        "acc_dtype",
        "fuse_act",
        "rhs_quant_dtype",
    ]
)
def gmm_v2(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """GMM kernel implemented with emit_pipeline.

    Dynamically calculate offset lhs/out tiles to reduce redundant computations.
    Additionally, it adjusts dma size based on number of valid rows and utilize
    triple buffering on weights to better utilize memory.

    Args:
        lhs: lhs with shape [size_m, size_k].
        rhs: Native [size_group, size_k, size_n] or packed
            [size_group, size_k/8, size_n]. A leading layer axis is allowed
            when rhs_layer is supplied; the kernel selects it without copying.
        rhs_layer: Optional int32[1] index into a resident RHS weight family.
            Scales still describe the selected layer.
        group_sizes: The group sizes of lhs rows of shape [size_lhs_group,].
        rhs_scale: The rhs scale of shape [size_group, num_blocks, 1, out_size],
            or E8M0 uint8 [size_group, num_blocks, out_size].
        group_offset: Optional. The group offset of shape [1,].
        tile_info: The tile sizes or tile function to use.
        vmem_limit_bytes: Optional vmem limit in bytes.
        precision: Unused. Exists for compatibility reasons.
        preferred_element_type: Optional jnp.dtype for the output matrix.
        acc_dtype: Optional jnp.dtype for the accumulator.
        fuse_act: Activation function to fuse with GMM (``"situ:..."``), None
            if no fusion. Output rows outside every group are not written.
        rhs_quant_dtype: Quantized dtype encoded by a pre-packed RHS carrier,
            packed along k.

    Returns:
        Output of shape [size_m, size_n].
    """
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        lhs,
        rhs,
        group_sizes,
        rhs_scale,
        group_offset,
    ) = (
        _p_next(_p_iter, _p_get("lhs")),
        _p_next(_p_iter, _p_get("rhs")),
        _p_next(_p_iter, _p_get("group_sizes")),
        _p_next(_p_iter, _p_get("rhs_scale")),
        _p_next(_p_iter, _p_get("group_offset")),
    )
    (
        rhs_layer,
        tile_info,
        vmem_limit_bytes,
        precision,
        preferred_element_type,
        acc_dtype,
        fuse_act,
        rhs_quant_dtype,
    ) = (
        _p_get("rhs_layer"),
        _p_get("tile_info", calculate_tiling),
        _p_get("vmem_limit_bytes"),
        _p_get("precision", jax.lax.Precision.DEFAULT),
        _p_get("preferred_element_type"),
        _p_get("acc_dtype"),
        _p_get("fuse_act"),
        _p_get("rhs_quant_dtype"),
    )
    del precision

    # The kernel reshapes lhs by its sublane block, so pad the rows up to it.
    # The extra rows sit past every group, so they add no work, and they are
    # sliced back off the output below.
    size_m = lhs.shape[0]
    size_lhs_sublane = _get_lhs_sublane_size(lhs.dtype, size_m)
    padded_size_m = align_to(size_m, size_lhs_sublane)
    if padded_size_m != size_m:
        lhs = jnp.pad(lhs, ((0, padded_size_m - size_m), (0, 0)))

    if group_offset is None:
        group_offset = jnp.array([0], dtype=jnp.int32)
    elif jnp.isscalar(group_offset):
        group_offset = group_offset[None]

    if vmem_limit_bytes is None:
        vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)

    family = rhs.ndim == _CONST_4
    if family != (rhs_layer is not None):
        _err_msg = "A family RHS requires a runtime layer index"
        raise ValueError(_err_msg)
    # E8M0 scales may also keep the family's layer axis ([L, E, blocks, N]).
    scale_family = (
        family
        and rhs_scale is not None
        and rhs_scale.dtype == jnp.uint8
        and rhs_scale.ndim == _CONST_4
    )
    if rhs_layer is None:
        rhs_layer = jnp.zeros((1,), jnp.int32)
    elif rhs_layer.shape != (1,) or rhs_layer.dtype != jnp.int32:
        _err_msg = "rhs_layer must have shape (1,) and dtype int32"
        raise ValueError(_err_msg)
    cfgs = make_gmm_configs(
        lhs,
        jax.ShapeDtypeStruct(rhs.shape[1:], rhs.dtype) if family else rhs,
        (
            jax.ShapeDtypeStruct(rhs_scale.shape[1:], rhs_scale.dtype)
            if (scale_family and rhs_scale is not None)
            else rhs_scale
        ),
        group_sizes,
        group_offset,
        tile_info=tile_info,
        vmem_limit_bytes=vmem_limit_bytes,
        out_dtype=preferred_element_type,
        acc_dtype=acc_dtype,
        fuse_act=fuse_act,
        rhs_quant_dtype=rhs_quant_dtype,
    )
    dims = cfgs.dims
    tiles = cfgs.tiles

    rhs_scale_spec = None
    if rhs_scale is not None:
        if rhs_scale.dtype != jnp.uint8:  # E8M0 exponents stay bytes
            rhs_scale = rhs_scale.astype(jnp.float32)
        rhs_scale_spec = pl.BlockSpec(memory_space=pltpu.HBM)

    max_num_gm = dims.size_group + pl.cdiv(dims.size_m, tiles.tile_m) - 1
    acc_cols = 2 * tiles.tile_n if cfgs.fuse_act is not None else tiles.tile_n
    scratch_shapes = [
        # partial_out_ref
        pltpu.VMEM((dims.size_lhs_sublane, tiles.tile_n), cfgs.out_dtype),
        # acc_ref
        pltpu.VMEM((tiles.tile_m, acc_cols), cfgs.acc_dtype),
        # metadata_ref
        MetadataRef(
            gm_id_to_group_id=pltpu.SMEM((max_num_gm,), jnp.int32),
            gm_id_to_m_offset=pltpu.SMEM((max_num_gm + 1,), jnp.int32),
        ),
    ]

    num_lanes = pltpu.get_tpu_info().num_lanes
    aligned_n = align_to(cfgs.out_size_n, num_lanes)
    out_init = jax.ShapeDtypeStruct((dims.size_m, aligned_n), cfgs.out_dtype)
    rhs_weights = WeightsRef(weight=rhs, scale=rhs_scale)

    return pl.pallas_call(
        functools.partial(kernel_main, cfgs=cfgs),
        out_shape=out_init,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                WeightsRef(
                    weight=pl.BlockSpec(memory_space=pltpu.HBM),
                    scale=rhs_scale_spec,
                ),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit_bytes,
            disable_bounds_checks=True,
        ),
        name=get_scope_name(cfgs),
        cost_estimate=get_cost_estimate(cfgs),
        metadata=get_metadata(cfgs),
    )(group_sizes, group_offset, rhs_layer, lhs, rhs_weights)[
        :size_m, : cfgs.out_size_n
    ]


# ============================================================================
# Prefill building blocks
# ============================================================================
#
# Chunked KDA math, the MLA projections and the logits' residual mixture.


# Structural constants of the K3 per-rank (TP32) layout.
_HEADS = 3  # attention heads per rank (96 total)
_CHANNELS = 128
_KDA_CHUNK = 64
_KDA_SUB_BLOCK = 16


def _dot(x: _Dynamic, w: _Dynamic) -> _Dynamic:
    """[rows, K] @ [K, N] with an FP32 accumulator; rows must be 8-aligned."""
    if x.shape[0] % 8:
        _err_msg = "prefill dots use native MXU row tiles (multiples of 8)"
        raise ValueError(_err_msg)
    return jax.lax.dot_general(
        x, w, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
    )


dot: _Dynamic = _dot


def _dot3(x: _Dynamic, y: _Dynamic, contracting: _Dynamic = 2) -> _Dynamic:
    """Batched head-major dot: [H, M, K] @ [H, K, N] -> [H, M, N] in FP32."""
    return jax.lax.dot_general(
        x,
        y,
        (((contracting,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )


def _row_where(mask: _Dynamic, x: _Dynamic, fill: _Dynamic = 0) -> _Dynamic:
    """``jnp.where(mask[..., None...], x, fill)`` with bool-safe broadcasting."""
    m = mask.astype(jnp.int32)  # numeric while reshaping/broadcasting
    while len(m.shape) < len(x.shape):
        m = m[..., None]
    m = jnp.broadcast_to(m, x.shape) != 0
    return jnp.where(m, x, fill)


def _mask_where(mask: _Dynamic, x: _Dynamic, fill: _Dynamic) -> _Dynamic:
    """jnp.where with a broadcasting bool mask, lowered through int32 so the.

    Mosaic i1-reshape limitation is not hit.
    """
    mask = jnp.broadcast_to(mask.astype(jnp.int32), x.shape) != 0
    return jnp.where(mask, x, fill)


# ---------------------------------------------------------------------------
# Chunked KDA math (pure functions; also called on ref reads in the kernel).
# ---------------------------------------------------------------------------


def _inclusive_cumsum_rows(x: _Dynamic) -> _Dynamic:
    """Inclusive FP32 cumsum over axis -2 via exact log-shift doubling."""
    shift = 1
    rows = x.shape[-2]
    while shift < rows:
        x = x + jnp.pad(x, ((0, 0), (shift, 0), (0, 0)))[..., :rows, :]
        shift *= 2
    return x


def _invert_delta_rule_two_level(low: _Dynamic, block: _Dynamic = 8) -> _Dynamic:
    """(I + L)^{-1} via a block-diagonal / off-diagonal factorization.

    Splits L = L_diag + F into the block-diagonal part (sub-blocks of ``block``
    rows) and the off-block-diagonal remainder, so (I + L)^{-1} =
    (I + G)^{-1} (I + L_diag)^{-1} with G = (I + L_diag)^{-1} F. Each factor is
    inverted by exact nilpotent Neumann doubling, but the largest power formed
    is the 4th (vs M^(C/2) single-level), which stays well conditioned under
    reduced-precision MXU matmul. Mirrors the vllm-torchtpu chunk_kda solve.
    """
    chunk_size = low.shape[-1]
    nc_inv = chunk_size // block
    eye = jnp.eye(chunk_size, dtype=jnp.float32)[None]
    idx = jnp.arange(chunk_size, dtype=jnp.int32)
    same_block = (idx[:, None] // block == idx[None, :] // block).astype(jnp.float32)
    l_diag = low * same_block[None]
    f = low - l_diag
    # (I + L_diag)^{-1}: within each block x block strictly-lower block L^block = 0.
    neg_ld = -l_diag
    p = eye + neg_ld
    mk = neg_ld
    for _ in range(block.bit_length() - 2):  # block=8 -> 2 doublings
        mk = _dot3(mk, mk)
        p = p + _dot3(p, mk)
    # G = P @ F; (I + G)^{-1}, G^nc_inv = 0 (block strictly lower triangular).
    g = _dot3(p, f)
    h = -g
    inv_ig = eye + h
    hk = h
    for _ in range(nc_inv.bit_length() - 2):  # nc_inv=8 -> 2 doublings
        hk = _dot3(hk, hk)
        inv_ig = inv_ig + _dot3(inv_ig, hk)
    return _dot3(inv_ig, p)


def _kda_chunk_core(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """One chunk of the Kimi delta rule in parallel (chunked) form.

    ``q``, ``k``: [H, C, K] FP32, with ``q`` already L2-normalized and scaled
    by K**-0.5 and ``k`` L2-normalized. ``v``: [H, C, V] FP32. ``gate``:
    [H, C, K] FP32 log-decay per channel (<= 0, natural log). ``beta``:
    [H, C] FP32 in [0, 1]. ``state``: [H, K, V] FP32. ``invert`` picks the
    (I + L)^{-1} routine (default `_invert_delta_rule_two_level`).

    Returns (out [H, C, V] FP32, new_state [H, K, V] FP32, aux), aux the
    intermediates gc, aqk, low, solve and v_new.

    The intra-chunk decay factors use the sub-block anchor factorization
    exp(g_r - g_t) = exp(g_r - a_i) * exp(a_i - g_t), ``a_i`` the gate
    cumsum at the middle row of r's sub-block, so each sub-block's rows
    (its diagonal block included) are one matmul. For the bounded K3 gate
    (|gate| <= 5) the row factors stay within exp(+-40); a column factor
    underflows only where the true product is below exp(-47), and products
    above the diagonal (possibly inf) are masked out afterwards. The matmul
    runs at HIGHEST precision: the factors' spread would otherwise cost the
    diagonal blocks their FP32 accuracy.
    """
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        q,
        k,
        v,
        gate,
        beta,
        state,
    ) = (
        _p_next(_p_iter, _p_get("q")),
        _p_next(_p_iter, _p_get("k")),
        _p_next(_p_iter, _p_get("v")),
        _p_next(_p_iter, _p_get("gate")),
        _p_next(_p_iter, _p_get("beta")),
        _p_next(_p_iter, _p_get("state")),
    )
    (
        sub_block,
        invert,
    ) = (
        _p_get("sub_block", _KDA_SUB_BLOCK),
        _p_get("invert"),
    )
    if invert is None:
        invert = _invert_delta_rule_two_level
    _head_count, chunk_size, _channels = q.shape
    value_channels = v.shape[-1]
    if chunk_size % sub_block:
        _err_msg = "chunk size must be a multiple of the sub-block"
        raise ValueError(_err_msg)
    nc = chunk_size // sub_block
    gc = _inclusive_cumsum_rows(gate)  # [H, C, K], inclusive over rows
    beta_k = beta[..., None].astype(jnp.float32)

    rows = jax.lax.broadcasted_iota(jnp.int32, (chunk_size, chunk_size), 0)
    cols = jax.lax.broadcasted_iota(jnp.int32, (chunk_size, chunk_size), 1)
    causal = (cols <= rows)[None]  # [1, C, C]
    strictly_causal = (cols < rows)[None]

    aqk_parts = []
    akk_parts = []
    col_iota = jax.lax.broadcasted_iota(jnp.int32, (chunk_size,), 0)  # [C]
    for i in range(nc):
        i_s = i * sub_block
        anchor = gc[:, i_s + sub_block // 2, :][:, None, :]  # [H, 1, K]
        row_scale = jnp.exp(gc[:, i_s : i_s + sub_block, :] - anchor)  # [H, BC, K]
        # Columns up to the sub-block's end (later ones are above the
        # diagonal for all its rows: zeroed rather than overflowing).
        through = (col_iota < i_s + sub_block).astype(jnp.float32)[None, :, None]
        k_scaled = k * jnp.exp(jnp.minimum(anchor - gc, 80.0)) * through  # [H, C, K]
        both = jax.lax.dot_general(
            jnp.concatenate(
                (
                    q[:, i_s : i_s + sub_block] * row_scale,
                    k[:, i_s : i_s + sub_block] * row_scale,
                ),
                axis=1,
            ),  # [H, 2*BC, K]
            k_scaled.transpose(0, 2, 1),  # [H, K, C]
            (((2,), (1,)), ((0,), (0,))),
            preferred_element_type=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
        )  # [H, 2*BC, C]
        aqk_parts.append(both[:, :sub_block])
        akk_parts.append(both[:, sub_block:])
    aqk = jnp.concatenate(aqk_parts, axis=1)
    akk = jnp.concatenate(akk_parts, axis=1)
    aqk = _mask_where(causal, aqk, jnp.float32(0))
    # L is strictly causal, and beta scales rows: delta_r = beta_r * (v_r -
    # k_r @ state_{r-1}), so L[r, t] = beta_r * (k_r . k_t) * exp(gc_r-gc_t).
    low = _mask_where(strictly_causal, akk * beta[:, :, None], jnp.float32(0))

    # (I + L)^{-1} by exact nilpotent Neumann inversion (L^C = 0 terminates it).
    solve = invert(low)

    rhs = jnp.concatenate(
        (
            v * beta_k,
            k * jnp.exp(gc) * beta_k,
        ),
        axis=-1,
    )  # [H, C, V + K]
    solved = _dot3(solve, rhs)
    u, w = solved[..., :value_channels], solved[..., value_channels:]
    v_new = u - _dot3(w, state)  # [H, C, V]

    anchor0 = gc[:, 0, :]  # [H, K]
    out = _dot3(
        q * jnp.exp(gc - anchor0[:, None, :]),
        state * jnp.exp(anchor0)[:, :, None],
    )
    out = out + _dot3(aqk, v_new)

    g_last = gc[:, -1, :]  # [H, K]
    kg = k * jnp.exp(g_last[:, None, :] - gc)
    new_state = state * jnp.exp(g_last)[:, :, None] + _dot3(kg, v_new, contracting=1)
    aux = {"gc": gc, "aqk": aqk, "low": low, "solve": solve, "v_new": v_new}
    return out, new_state, aux


def kda_chunk(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """One chunk of the Kimi delta rule; returns (out, new_state).

    See `_kda_chunk_core` for the full docstring. ``invert`` picks the
    (I + L)^{-1} routine (default `_invert_delta_rule_two_level`).
    """
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        q,
        k,
        v,
        gate,
        beta,
        state,
    ) = (
        _p_next(_p_iter, _p_get("q")),
        _p_next(_p_iter, _p_get("k")),
        _p_next(_p_iter, _p_get("v")),
        _p_next(_p_iter, _p_get("gate")),
        _p_next(_p_iter, _p_get("beta")),
        _p_next(_p_iter, _p_get("state")),
    )
    (
        sub_block,
        invert,
    ) = (
        _p_get("sub_block", _KDA_SUB_BLOCK),
        _p_get("invert"),
    )
    out, new_state, _ = _kda_chunk_core(
        q, k, v, gate, beta, state, sub_block=sub_block, invert=invert
    )
    return out, new_state


# ---------------------------------------------------------------------------
# MLA projections
# ---------------------------------------------------------------------------


def mla_projections(
    hidden: _Dynamic,
    weights: _Dynamic,
    *,
    epsilon: _Dynamic = 1e-5,
    return_latent: _Dynamic = False,
    query_scale: _Dynamic = 1.0,
) -> _Dynamic:
    """MLA Q/K/V projections for a token window (per-rank layout).

    ``hidden``: [T, hidden] BF16. Returns (query [T,3,256] times
    ``query_scale`` before its BF16 rounding, new_key [T,3,256],
    new_value [T,3,128], output_gate [T,384]), all BF16; with
    ``return_latent`` also the compressed KV that vLLM's MLA cache stores,
    ``[T, 576]`` BF16: the normalized 512-lane latent then the 64 rope lanes
    (K3's MLA is NoPE; the rope lanes are used unrotated). It is the same on
    every rank.
    """
    tokens = hidden.shape[0]
    query_latent = _dot(hidden, weights.query_down_projection)
    query_latent = (
        query_latent
        * jax.lax.rsqrt(
            jnp.mean(query_latent * query_latent, axis=1, keepdims=True) + epsilon
        )
        * weights.query_normalization
    ).astype(jnp.bfloat16)
    query = (_dot(query_latent, weights.query_up_projection) * query_scale).astype(
        jnp.bfloat16
    )
    query = query.reshape(tokens, _HEADS, 256)

    compressed = _dot(hidden, weights.key_value_down_projection).astype(jnp.bfloat16)
    kv_latent = compressed[:, :512].astype(jnp.float32)
    kv_latent = (
        kv_latent
        * jax.lax.rsqrt(
            jnp.mean(kv_latent * kv_latent, axis=1, keepdims=True) + epsilon
        )
        * weights.key_value_normalization
    ).astype(jnp.bfloat16)
    expanded = _dot(kv_latent, weights.key_value_up_projection).astype(jnp.bfloat16)
    expanded = expanded.reshape(tokens, _HEADS, 256)
    new_key = jnp.concatenate(
        (
            expanded[:, :, :128],
            jnp.broadcast_to(compressed[:, None, 512:576], (tokens, _HEADS, 64)),
            jnp.zeros((tokens, _HEADS, 64), jnp.bfloat16),
        ),
        axis=2,
    )
    new_value = expanded[:, :, 128:]
    output_gate = _dot(hidden, weights.output_gate_projection).astype(jnp.bfloat16)
    if return_latent:
        latent = jnp.concatenate((kv_latent, compressed[:, 512:576]), axis=1)
        return query, new_key, new_value, output_gate, latent
    return query, new_key, new_value, output_gate


# ---------------------------------------------------------------------------
# Residual mixture (dense form, used by the logits; the kernel streams it)
# ---------------------------------------------------------------------------


def residual_mixture(
    blocks: _Dynamic,
    prefix: _Dynamic,
    folded: _Dynamic,
    active_block_count: _Dynamic,
    *,
    epsilon: _Dynamic = 1e-5,
) -> _Dynamic:
    """Learned residual mixture over [8, T, H] blocks + [T, H] prefix.

    Returns [T, H] BF16.
    """
    values = jnp.concatenate(
        (blocks.astype(jnp.float32), prefix[None].astype(jnp.float32)), axis=0
    )
    scores = jnp.sum(values * folded.astype(jnp.float32), axis=-1)
    scores *= jax.lax.rsqrt(jnp.mean(values * values, axis=-1) + epsilon)
    index = jax.lax.broadcasted_iota(jnp.int32, (9, 1), 0)
    active = (index < active_block_count) | (index == _CONST_8)
    probabilities = jax.nn.softmax(jnp.where(active, scores, -jnp.inf), axis=0)
    mixed = jnp.sum(values * probabilities[..., None], axis=0)
    return mixed.astype(jnp.bfloat16)


# ============================================================================
# Residual mixture and RMSNorm (HBM-tiled)
# ============================================================================
#
# An online softmax: one read of each active block per tile, one block in VMEM.


class _MixtureNormRunner:
    """Runner for tiled residual mixture and RMSNorm."""

    def __getattr__(self, name: _Dynamic) -> _Dynamic:
        """Dynamic attribute access."""
        raise AttributeError(name)

    def __init__(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Initialize runner with arguments."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            self.active_ref,
            self.prefix_ref,
            self.blocks_ref,
            self.folded_ref,
            self.norm_ref,
            self.output_ref,
            self.prefix_bufs,
            self.block_buf,
            self.output_bufs,
            self.numerator_buf,
            self.row_max_buf,
            self.row_sum_buf,
            self.folded_buf,
            self.norm_buf,
            self.prefix_sems,
            self.block_sem,
            self.output_sems,
            self.weight_sems,
        ) = (
            _p_next(_p_iter, _p_get("active_ref")),
            _p_next(_p_iter, _p_get("prefix_ref")),
            _p_next(_p_iter, _p_get("blocks_ref")),
            _p_next(_p_iter, _p_get("folded_ref")),
            _p_next(_p_iter, _p_get("norm_ref")),
            _p_next(_p_iter, _p_get("output_ref")),
            _p_next(_p_iter, _p_get("prefix_bufs")),
            _p_next(_p_iter, _p_get("block_buf")),
            _p_next(_p_iter, _p_get("output_bufs")),
            _p_next(_p_iter, _p_get("numerator_buf")),
            _p_next(_p_iter, _p_get("row_max_buf")),
            _p_next(_p_iter, _p_get("row_sum_buf")),
            _p_next(_p_iter, _p_get("folded_buf")),
            _p_next(_p_iter, _p_get("norm_buf")),
            _p_next(_p_iter, _p_get("prefix_sems")),
            _p_next(_p_iter, _p_get("block_sem")),
            _p_next(_p_iter, _p_get("output_sems")),
            _p_next(_p_iter, _p_get("weight_sems")),
        )
        self.tile = _p_get("tile")
        self.eps = _p_get("eps")
        tokens, _hidden = self.prefix_ref.shape
        self.n_tiles = tokens // self.tile
        self.active_count = self.active_ref[0]

    def load_prefix(self, t: _Dynamic, slot: _Dynamic) -> _Dynamic:
        """Execute load prefix logic."""
        return pltpu.make_async_copy(
            self.prefix_ref.at[pl.ds(t * self.tile, self.tile), :],
            self.prefix_bufs.at[slot],
            self.prefix_sems.at[slot],
        )

    def store_output(self, t: _Dynamic, slot: _Dynamic) -> _Dynamic:
        """Execute store output logic."""
        return pltpu.make_async_copy(
            self.output_bufs.at[slot],
            self.output_ref.at[pl.ds(t * self.tile, self.tile), :],
            self.output_sems.at[slot],
        )

    def _process_active_block(self, block: int, t: _Dynamic, folded: _Dynamic) -> None:
        """Process one active block for the tile."""
        copy = pltpu.make_async_copy(
            self.blocks_ref.at[block, pl.ds(t * self.tile, self.tile), :],
            self.block_buf,
            self.block_sem.at[0],
        )
        copy.start()
        copy.wait()
        values = self.block_buf[...].astype(jnp.float32)
        score = jnp.sum(values * folded, axis=1)
        score *= jax.lax.rsqrt(jnp.mean(values * values, axis=1) + self.eps)
        old_max = self.row_max_buf[...]
        new_max = jnp.maximum(old_max, score[:, None])
        old_scale = jnp.exp(old_max - new_max)
        new_scale = jnp.exp(score[:, None] - new_max)
        self.numerator_buf[...] = (
            self.numerator_buf[...] * old_scale + values * new_scale
        )
        self.row_sum_buf[...] = self.row_sum_buf[...] * old_scale + new_scale
        self.row_max_buf[...] = new_max

    def tile_body(self, t: _Dynamic, folded: _Dynamic, norm: _Dynamic) -> None:
        """Execute tile body logic."""
        slot = t % 2
        self.load_prefix(t, slot).wait()

        pl.when(t >= _CONST_2)(lambda: self.store_output(t - 2, slot).wait())
        pl.when(t + 1 < self.n_tiles)(lambda: self.load_prefix(t + 1, 1 - slot).start())

        prefix = self.prefix_bufs.at[slot][...].astype(jnp.float32)
        prefix_score = jnp.sum(prefix * folded, axis=1)
        prefix_score *= jax.lax.rsqrt(jnp.mean(prefix * prefix, axis=1) + self.eps)
        self.numerator_buf[...] = prefix
        self.row_max_buf[...] = prefix_score[:, None]
        self.row_sum_buf[...] = jnp.ones((self.tile, 1), jnp.float32)

        for block in range(8):
            pl.when(block < self.active_count)(
                lambda b=block: self._process_active_block(b, t, folded)
            )

        mixed = (self.numerator_buf[...] / self.row_sum_buf[...]).astype(jnp.bfloat16)
        mixed = mixed.astype(jnp.float32)
        normalized = (
            mixed
            * jax.lax.rsqrt(jnp.mean(mixed * mixed, axis=1, keepdims=True) + self.eps)
            * norm
        ).astype(jnp.bfloat16)
        self.output_bufs.at[slot][...] = normalized
        self.store_output(t, slot).start()

    def run(self) -> None:
        """Run mixture norm tiled body."""
        weights = (
            pltpu.make_async_copy(
                self.folded_ref, self.folded_buf, self.weight_sems.at[0]
            ),
            pltpu.make_async_copy(self.norm_ref, self.norm_buf, self.weight_sems.at[1]),
        )
        for copy in weights:
            copy.start()
        for copy in weights:
            copy.wait()
        folded = self.folded_buf[...]
        norm = self.norm_buf[...]

        self.load_prefix(0, 0).start()
        pl.loop(0, self.n_tiles)(lambda t: self.tile_body(t, folded, norm))
        self.store_output(self.n_tiles - 1, (self.n_tiles - 1) % 2).wait()
        if self.n_tiles > 1:
            self.store_output(self.n_tiles - 2, (self.n_tiles - 2) % 2).wait()


def mixture_norm_tiled_body(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Run residual mixture plus RMSNorm with caller-owned refs and scratch."""
    return _MixtureNormRunner(*_p_args, **_p_kw).run()


def mixture_norm_tiled_scratch_shapes(
    hidden_size: _Dynamic, tile: _Dynamic
) -> _Dynamic:
    """Scratch specs in the order expected by ``mixture_norm_tiled_body``."""
    return (
        pltpu.VMEM((2, tile, hidden_size), jnp.bfloat16),
        pltpu.VMEM((tile, hidden_size), jnp.bfloat16),
        pltpu.VMEM((2, tile, hidden_size), jnp.bfloat16),
        pltpu.VMEM((tile, hidden_size), jnp.float32),
        pltpu.VMEM((tile, 1), jnp.float32),
        pltpu.VMEM((tile, 1), jnp.float32),
        pltpu.VMEM((1, hidden_size), jnp.float32),
        pltpu.VMEM((1, hidden_size), jnp.bfloat16),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA((1,)),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA((2,)),
    )


def make_mixture_norm_tiled(
    *,
    seq_len: _Dynamic,
    tile: _Dynamic = 256,
    eps: _Dynamic = 1e-5,
    interpret: _Dynamic = False,
) -> _Dynamic:
    """Build ``(prefix, blocks, folded, norm, active) -> normalized``.

    ``active`` is an int32 array of shape [1] with a value from 0 to 8.
    The sequence length and tile must be positive, and seq_len must be a
    multiple of tile. Inputs and output are HBM-backed in the Pallas body.
    """
    if seq_len <= 0 or tile <= 0 or seq_len % tile:
        _err_msg = "seq_len must be a positive multiple of tile"
        raise ValueError(_err_msg)

    def run(
        prefix: _Dynamic,
        blocks: _Dynamic,
        folded: _Dynamic,
        norm: _Dynamic,
        active: _Dynamic,
    ) -> _Dynamic:
        """Execute run logic."""
        if len(prefix.shape) != _CONST_2 or prefix.shape[0] != seq_len:
            _err_msg = "prefix must have shape [seq_len, hidden]"
            raise ValueError(_err_msg)
        hidden = prefix.shape[1]
        if hidden % 128 or prefix.dtype != jnp.bfloat16:
            _err_msg = "prefix must be BF16 with a 128-aligned width"
            raise ValueError(_err_msg)
        if blocks.shape != (8, seq_len, hidden) or blocks.dtype != jnp.bfloat16:
            _err_msg = "blocks must be BF16 with shape [8, seq_len, hidden]"
            raise ValueError(_err_msg)
        if folded.shape != (1, hidden) or folded.dtype != jnp.float32:
            _err_msg = "folded must be FP32 with shape [1, hidden]"
            raise ValueError(_err_msg)
        if norm.shape != (1, hidden) or norm.dtype != jnp.bfloat16:
            _err_msg = "norm must be BF16 with shape [1, hidden]"
            raise ValueError(_err_msg)
        if active.shape != (1,) or active.dtype != jnp.int32:
            _err_msg = "active must be int32 with shape [1]"
            raise ValueError(_err_msg)
        call = pl.pallas_call(
            functools.partial(mixture_norm_tiled_body, tile=tile, eps=eps),
            out_shape=jax.ShapeDtypeStruct(prefix.shape, jnp.bfloat16),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                in_specs=[pl.BlockSpec(memory_space=pltpu.HBM) for _ in range(4)],
                out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
                scratch_shapes=mixture_norm_tiled_scratch_shapes(hidden, tile),
            ),
            name="mixture_norm_tiled",
            interpret=interpret,
            compiler_params=pltpu.CompilerParams(vmem_limit_bytes=64 * 1024**2),
        )
        return call(active, prefix, blocks, folded, norm)

    return run


def make_mixture_norm_sp(
    mesh: _Dynamic, *, seq_len: _Dynamic, eps: _Dynamic = 1e-5
) -> _Dynamic:
    """Apply the tiled mixture/RMSNorm to each rank's T/32 local rows."""
    if mesh.size != _CONST_32 or len(mesh.axis_names) != 1 or seq_len % 32:
        _err_msg = "mixture SP requires a flat TP32 mesh and T divisible by 32"
        raise ValueError(_err_msg)
    axis_name = mesh.axis_names[0]
    local_tokens = seq_len // 32
    tile = min(256, local_tokens)
    if local_tokens % tile:
        _err_msg = "local token count must be divisible by the mixture tile"
        raise ValueError(_err_msg)
    op = make_mixture_norm_tiled(seq_len=local_tokens, tile=tile, eps=eps)

    def local(
        prefix: _Dynamic,
        blocks: _Dynamic,
        folded: _Dynamic,
        norm: _Dynamic,
        active: _Dynamic,
    ) -> _Dynamic:
        """Execute local logic."""
        return op(prefix[0], blocks[0], folded[0], norm[0], active)[None]

    return jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(P(axis_name), P(axis_name), P(axis_name), P(axis_name), P()),
            out_specs=P(axis_name),
            check_vma=False,
        )
    )


# ============================================================================
# Sequence-tiled KDA pipeline.
# ============================================================================
#
# The window streams through VMEM in tiles; the KDA state carries across them.


class _KdaPrefillRunner:
    """Runner for KDA prefill tiled execution over packed sequences."""

    def __getattr__(self, name: _Dynamic) -> _Dynamic:
        """Dynamic attribute access."""
        raise AttributeError(name)

    def __init__(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Initialize KDA prefill runner."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            self.hidden_ref,
            self.gate_w_ref,
            self.fb_w_ref,
            self.proj_w_ref,
            self.conv_w_ref,
            self.alog_ref,
            self.dt_ref,
            self.norm_ref,
            self.rec_state_in_ref,
            self.conv_hist_in_ref,
            self.out_ref,
            self.rec_state_out_ref,
            self.conv_hist_out_ref,
            self.hidden_bufs,
            self.out_bufs,
            self.gate_w_vmem,
            self.fb_w_vmem,
            self.proj_w_vmem,
            self.conv_w_vmem,
            self.alog_vmem,
            self.dt_vmem,
            self.norm_vmem,
            self.rec_state_vmem,
            self.conv_hist_vmem,
            self.hidden_sems,
            self.out_sems,
            self.weight_sems,
        ) = (
            _p_next(_p_iter, _p_get("hidden_ref")),
            _p_next(_p_iter, _p_get("gate_w_ref")),
            _p_next(_p_iter, _p_get("fb_w_ref")),
            _p_next(_p_iter, _p_get("proj_w_ref")),
            _p_next(_p_iter, _p_get("conv_w_ref")),
            _p_next(_p_iter, _p_get("alog_ref")),
            _p_next(_p_iter, _p_get("dt_ref")),
            _p_next(_p_iter, _p_get("norm_ref")),
            _p_next(_p_iter, _p_get("rec_state_in_ref")),
            _p_next(_p_iter, _p_get("conv_hist_in_ref")),
            _p_next(_p_iter, _p_get("out_ref")),
            _p_next(_p_iter, _p_get("rec_state_out_ref")),
            _p_next(_p_iter, _p_get("conv_hist_out_ref")),
            _p_next(_p_iter, _p_get("hidden_bufs")),
            _p_next(_p_iter, _p_get("out_bufs")),
            _p_next(_p_iter, _p_get("gate_w_vmem")),
            _p_next(_p_iter, _p_get("fb_w_vmem")),
            _p_next(_p_iter, _p_get("proj_w_vmem")),
            _p_next(_p_iter, _p_get("conv_w_vmem")),
            _p_next(_p_iter, _p_get("alog_vmem")),
            _p_next(_p_iter, _p_get("dt_vmem")),
            _p_next(_p_iter, _p_get("norm_vmem")),
            _p_next(_p_iter, _p_get("rec_state_vmem")),
            _p_next(_p_iter, _p_get("conv_hist_vmem")),
            _p_next(_p_iter, _p_get("hidden_sems")),
            _p_next(_p_iter, _p_get("out_sems")),
            _p_next(_p_iter, _p_get("weight_sems")),
        )
        (
            self.tile,
            self.decay_lower_bound,
            self.epsilon,
            self.checkpoint,
            self.segments,
            self.tiles,
            self.block_tiles,
            self.projection,
        ) = (
            _p_get("tile"),
            _p_get("decay_lower_bound"),
            _p_get("epsilon"),
            _p_get("checkpoint"),
            _p_get("segments"),
            _p_get("tiles"),
            _p_get("block_tiles"),
            _p_get("projection"),
        )
        self.head_count, self.channels = _HEADS, _CHANNELS
        self.width = self.head_count * self.channels
        self.segment_ref, self.count = self.segments
        self.rec_checkpoint_ref, self.hist_checkpoint_ref = self.checkpoint
        (
            self.weight_ref,
            self.weight_vmem,
            self.partial_ref,
            self.partial_bufs,
            self.partial_sems,
        ) = self.projection
        self.first_tile, self.end_tile, self.first, self.last = self.tiles

    def field(self, index: _Dynamic, s: _Dynamic) -> _Dynamic:
        """Fetch descriptor field index for sequence s."""
        return self.segment_ref[index * self.count + s]

    def copy_now(self, pairs: _Dynamic, first_semaphore: _Dynamic) -> None:
        """Execute synchronous async copy sequence."""
        copies = [
            pltpu.make_async_copy(
                source, destination, self.weight_sems.at[first_semaphore + i]
            )
            for i, (source, destination) in enumerate(pairs)
        ]
        for copy in copies:
            copy.start()
        for copy in copies:
            copy.wait()

    def write_checkpoint(self, segment: _Dynamic) -> None:
        """Store write checkpoint state."""
        self.copy_now(
            (
                (self.rec_state_vmem, self.rec_checkpoint_ref.at[segment]),
                (self.conv_hist_vmem, self.hist_checkpoint_ref.at[segment]),
            ),
            11,
        )

    def unused_copies(self, s: _Dynamic) -> list[_Dynamic]:
        """Construct copies for unused sequence slots."""
        pairs = [
            (self.rec_state_in_ref.at[s], self.rec_state_out_ref.at[s]),
            (self.conv_hist_in_ref.at[s], self.conv_hist_out_ref.at[s]),
            (self.rec_state_in_ref.at[s], self.rec_checkpoint_ref.at[s]),
            (self.conv_hist_in_ref.at[s], self.hist_checkpoint_ref.at[s]),
        ]
        return [
            pltpu.make_async_copy(source, target, self.weight_sems.at[14 + i % 2])
            for i, (source, target) in enumerate(pairs)
        ]

    def each_unused(self, action: _Dynamic) -> None:
        """Execute action over unused sequence slots."""
        for s in range(self.count):

            def _apply_unused(slot_idx: int = s) -> None:
                """Apply action to all unused copies for one slot."""
                for copy in self.unused_copies(slot_idx):
                    action(copy)

            pl.when(self.field(1, s) == 0)(_apply_unused)

    def load_tile(self, t: _Dynamic, buf: _Dynamic) -> _Dynamic:
        """Execute load tile logic."""
        return pltpu.make_async_copy(
            self.hidden_ref.at[pl.ds(t * self.tile, self.tile), :],
            self.hidden_bufs.at[buf],
            self.hidden_sems.at[buf],
        )

    def store_tile(self, t: _Dynamic, buf: _Dynamic) -> _Copies:
        """Execute store tile logic."""
        return _Copies(
            [
                pltpu.make_async_copy(
                    self.out_bufs.at[buf],
                    self.out_ref.at[pl.ds(t * self.tile, self.tile), :],
                    self.out_sems.at[buf],
                ),
                pltpu.make_async_copy(
                    self.partial_bufs.at[buf],
                    self.partial_ref.at[pl.ds(t * self.tile, self.tile), :],
                    self.partial_sems.at[buf],
                ),
            ]
        )

    def _find_segment(
        self, first_row: _Dynamic
    ) -> tuple[_Dynamic, _Dynamic, _Dynamic, _Dynamic, _Dynamic]:
        """Identify sequence segment and positions for first row."""
        segment, segment_end = -1, 0
        for s in range(self.count):
            start = self.field(0, s)
            end = start + self.field(1, s)
            inside = (first_row >= start) & (first_row < end)
            segment = jnp.where(inside, s, segment)
            segment_end = jnp.where(inside, end, segment_end)
        current = jnp.maximum(segment, 0)
        offset, position = self.field(0, current), self.field(2, current)
        return segment, segment_end, current, offset, position

    def _run_causal_conv(
        self,
        real_rows: _Dynamic,
        qkv: _Dynamic,
        conv_filter: _Dynamic,
        row_end: _Dynamic,
        t: _Dynamic,
    ) -> _Dynamic:
        """Execute 4-tap causal convolution."""
        window = jnp.concatenate((self.conv_hist_vmem[...], qkv), axis=0)
        convolved = jnp.zeros(
            (self.tile, 3, self.head_count, self.channels), jnp.float32
        )
        for tap in range(4):
            convolved = (
                convolved
                + window[tap : tap + self.tile].astype(jnp.float32) * conv_filter[tap]
            )
        filtered = jax.nn.silu(convolved).astype(jnp.bfloat16)
        filtered = _row_where(real_rows, filtered)

        real_count = jnp.clip(row_end - t * self.tile, 0, self.tile)
        pl.when(real_count == self.tile)(
            lambda: self.conv_hist_vmem.__setitem__(
                slice(None), window[self.tile : self.tile + 3]
            )
        )
        pl.when((real_count > 0) & (real_count < self.tile))(
            lambda: self._carry_partial(real_count, window)
        )
        return filtered

    def _run_chunked_kda(
        self,
        filtered: _Dynamic,
        decay: _Dynamic,
        beta: _Dynamic,
    ) -> _Dynamic:
        """Run chunked KDA scan over 64-token chunks."""
        query = filtered[:, 0].astype(jnp.float32)
        key = filtered[:, 1].astype(jnp.float32)
        value = filtered[:, 2].astype(jnp.float32)
        query, key = kda_normalize_query_key(query, key, self.channels)

        outputs = []
        for c in range(self.tile // _KDA_CHUNK):
            s0, s1 = c * _KDA_CHUNK, (c + 1) * _KDA_CHUNK
            chunk_out, new_state = kda_chunk(
                query[s0:s1].transpose(1, 0, 2),
                key[s0:s1].transpose(1, 0, 2),
                value[s0:s1].transpose(1, 0, 2),
                decay[s0:s1].transpose(1, 0, 2),
                beta[s0:s1].transpose(1, 0),
                self.rec_state_vmem[...],
            )
            self.rec_state_vmem[...] = new_state
            outputs.append(chunk_out.transpose(1, 0, 2))
        return jnp.concatenate(outputs, axis=0)

    def tile_body(self, t: _Dynamic) -> None:
        """Execute tile body logic."""
        buf = t % 2
        in_block = t % self.block_tiles

        pl.when(in_block == 0)(lambda: self.load_tile(t, buf).start())
        self.load_tile(t, buf).wait()

        first_row = t * self.tile
        segment, segment_end, current, offset, position = self._find_segment(first_row)

        pl.when((segment >= 0) & (first_row == offset))(
            lambda: self._load_seg_state(current, position)
        )
        row_end = segment_end

        pl.when(in_block >= _CONST_2)(lambda: self.store_tile(t - 2, buf).wait())
        pl.when(in_block + 1 < self.block_tiles)(
            lambda: self.load_tile(t + 1, 1 - buf).start()
        )

        conv_filter = self.conv_w_vmem[...].astype(jnp.float32)
        a_log = jnp.exp(self.alog_vmem[...])
        dt_bias = self.dt_vmem[...]
        norm = self.norm_vmem[0]
        hidden_tile = self.hidden_bufs.at[buf][...]

        gate_features = _dot(hidden_tile, self.gate_w_vmem[...]).astype(jnp.bfloat16)
        raw_gate = (
            _dot(gate_features, self.fb_w_vmem[...])
            .astype(jnp.bfloat16)
            .reshape(self.tile, self.head_count, self.channels)
        )
        projected = _dot(hidden_tile, self.proj_w_vmem[...]).astype(jnp.bfloat16)
        qkv = projected[:, : 3 * self.width].reshape(
            self.tile, 3, self.head_count, self.channels
        )
        output_gate = projected[:, 3 * self.width : 4 * self.width].reshape(
            self.tile, self.head_count, self.channels
        )
        update_rate = projected[:, 4 * self.width : 4 * self.width + self.head_count]
        real_rows = jnp.arange(self.tile, dtype=jnp.int32) + t * self.tile < row_end
        raw_gate = _row_where(real_rows, raw_gate, -1e30)
        update_rate = _row_where(real_rows, update_rate, -1e30)

        filtered = self._run_causal_conv(real_rows, qkv, conv_filter, row_end, t)
        decay = kda_decay(self.decay_lower_bound, a_log[None], raw_gate, dt_bias[None])
        beta = jax.nn.sigmoid(update_rate.astype(jnp.float32))

        attended = self._run_chunked_kda(filtered, decay, beta)
        attended = attended.astype(jnp.bfloat16).astype(jnp.float32)
        attended *= jax.lax.rsqrt(
            jnp.mean(attended * attended, axis=-1, keepdims=True) + self.epsilon
        )
        attended = (attended * norm).astype(jnp.bfloat16) * jax.nn.sigmoid(
            output_gate.astype(jnp.float32)
        ).astype(jnp.bfloat16)
        attended = attended.reshape(self.tile, self.width)
        self.out_bufs.at[buf][...] = attended

        columns = self.weight_vmem.shape[1] // 4
        for c in range(4):
            self.partial_bufs.at[buf, :, pl.ds(c * columns, columns)][...] = _dot(
                attended, self.weight_vmem[:, pl.ds(c * columns, columns)]
            ).astype(jnp.bfloat16)

        pl.when(
            (segment >= 0) & (position > 0) & ((t + 1) * self.tile == offset + position)
        )(lambda: self.write_checkpoint(current))

        pl.when((segment >= 0) & (t == (segment_end - 1) // self.tile))(
            lambda: self._store_seg_state(current)
        )

        self.store_tile(t, buf).start()
        pl.when(in_block == self.block_tiles - 1)(lambda: self._drain_block(t, buf))

    def run(self) -> None:
        """Run KDA prefill tiled kernel."""
        weight_copies = (
            pltpu.make_async_copy(
                self.gate_w_ref, self.gate_w_vmem, self.weight_sems.at[0]
            ),
            pltpu.make_async_copy(
                self.fb_w_ref, self.fb_w_vmem, self.weight_sems.at[1]
            ),
            pltpu.make_async_copy(
                self.proj_w_ref, self.proj_w_vmem, self.weight_sems.at[2]
            ),
            pltpu.make_async_copy(
                self.conv_w_ref, self.conv_w_vmem, self.weight_sems.at[3]
            ),
            pltpu.make_async_copy(
                self.alog_ref, self.alog_vmem, self.weight_sems.at[4]
            ),
            pltpu.make_async_copy(self.dt_ref, self.dt_vmem, self.weight_sems.at[5]),
            pltpu.make_async_copy(
                self.norm_ref, self.norm_vmem, self.weight_sems.at[6]
            ),
            pltpu.make_async_copy(
                self.weight_ref, self.weight_vmem, self.weight_sems.at[13]
            ),
        )

        pl.when(self.first)(lambda: self._load_weights(weight_copies))
        pl.when(self.first)(lambda: self.each_unused(lambda copy: copy.start()))

        pl.loop(self.first_tile, self.end_tile)(self.tile_body)
        pl.when(self.last)(lambda: self.each_unused(lambda copy: copy.wait()))


def kda_prefill_tiled_body(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Run one KDA layer over a packed window, reading and writing tiled."""
    return _KdaPrefillRunner(*_p_args, **_p_kw).run()


class _Copies:
    """Several async copies started and awaited together."""

    def __getattr__(self, name: _Dynamic) -> _Dynamic:
        """Dynamic attribute access."""
        raise AttributeError(name)

    def __init__(self, copies: _Dynamic) -> None:
        """Execute init logic."""
        self.copies = copies

    def start(self) -> _Dynamic:
        """Execute start logic."""
        for copy in self.copies:
            copy.start()

    def wait(self) -> _Dynamic:
        """Execute wait logic."""
        for copy in self.copies:
            copy.wait()


def kda_prefill_tiled_scratch_shapes(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Scratch specs, in the order expected by ``kda_prefill_tiled_body``."""
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        hidden_size,
        tile,
        gate_w,
        fb_w,
        proj_w,
        conv_w,
        alog,
        dt,
        norm,
    ) = (
        _p_next(_p_iter, _p_get("hidden_size")),
        _p_next(_p_iter, _p_get("tile")),
        _p_next(_p_iter, _p_get("gate_w")),
        _p_next(_p_iter, _p_get("fb_w")),
        _p_next(_p_iter, _p_get("proj_w")),
        _p_next(_p_iter, _p_get("conv_w")),
        _p_next(_p_iter, _p_get("alog")),
        _p_next(_p_iter, _p_get("dt")),
        _p_next(_p_iter, _p_get("norm")),
    )
    width = _HEADS * _CHANNELS
    return (
        pltpu.VMEM((2, tile, hidden_size), jnp.bfloat16),
        pltpu.VMEM((2, tile, width), jnp.bfloat16),
        pltpu.VMEM(gate_w.shape, gate_w.dtype),
        pltpu.VMEM(fb_w.shape, fb_w.dtype),
        pltpu.VMEM(proj_w.shape, proj_w.dtype),
        pltpu.VMEM(conv_w.shape, conv_w.dtype),
        pltpu.VMEM(alog.shape, alog.dtype),
        pltpu.VMEM(dt.shape, dt.dtype),
        pltpu.VMEM(norm.shape, norm.dtype),
        pltpu.VMEM((_HEADS, _CHANNELS, _CHANNELS), jnp.float32),
        pltpu.VMEM((3, 3, _HEADS, _CHANNELS), jnp.bfloat16),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA((16,)),
    )


# ============================================================================
# MLA flash attention (long windows)
# ============================================================================
#
# Each rank's 3 heads over the packed window and every rank's cached prefixes.


TILE = 128
SCALE = 192.0**-0.5
# The online softmax runs in base 2 on queries pre-scaled by SCALE * log2(e)
# (``mla_projections(query_scale=SCALE_LOG2)``, before their BF16 rounding),
# so scores leave the MXU ready for exp2.
SCALE_LOG2 = SCALE * 1.4426950408889634


class _FlashKernelPackedRunner:
    """Runner for causal MLA flash attention over a packed window."""

    def __getattr__(self, name: _Dynamic) -> _Dynamic:
        """Dynamic attribute access."""
        raise AttributeError(name)

    def __init__(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Initialize runner with arguments."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            self.length_ref,
            self.segment_ref,
        ) = (
            _p_next(_p_iter, _p_get("length_ref")),
            _p_next(_p_iter, _p_get("segment_ref")),
        )
        refs = tuple(_p_iter)
        (
            self.tile,
            self.q_tile,
            self.count,
            self.prefixed,
        ) = (
            _p_get("tile"),
            _p_get("q_tile"),
            _p_get("count"),
            _p_kw.get("prefixed", False),
        )
        if self.prefixed:
            (
                self.layer_ref,
                self.q_ref,
                self.k_hbm,
                self.v_hbm,
                self.prefix_k_hbm,
                self.prefix_v_hbm,
                self.out_ref,
                self.k_buf,
                self.v_buf,
                self.sems,
            ) = refs
        else:
            (
                self.q_ref,
                self.k_hbm,
                self.v_hbm,
                self.out_ref,
                self.k_buf,
                self.v_buf,
                self.sems,
            ) = refs
            self.layer_ref = self.prefix_k_hbm = self.prefix_v_hbm = None

        self.head, self.query_block = pl.program_id(0), pl.program_id(1)
        self.length = self.length_ref[0]
        self.offsets = [self.segment_ref[s] for s in range(self.count)]
        self.ends = [
            self.segment_ref[s] + self.segment_ref[self.count + s]
            for s in range(self.count)
        ]
        self.first_row = self.query_block * self.q_tile
        self.rows = self.first_row + jnp.arange(self.q_tile)
        self.q = self.q_ref[0]

    def _compute_bounds(self) -> tuple[_Dynamic, _Dynamic, _Dynamic, _Dynamic]:
        """Compute sequence bounds and key counts."""
        row_start = self.rows + 1
        lowest = jnp.int32(1 << 30)
        interior = jnp.int32(1 << 30)
        for s in range(self.count):
            inside = (self.rows >= self.offsets[s]) & (self.rows < self.ends[s])
            row_start = jnp.where(inside, self.offsets[s], row_start)
            meets = (
                (self.offsets[s] < self.first_row + self.q_tile)
                & (self.ends[s] > self.first_row)
                & (self.ends[s] > self.offsets[s])
            )
            lowest = jnp.where(meets, jnp.minimum(lowest, self.offsets[s]), lowest)
            holds = (self.offsets[s] <= self.first_row) & (
                self.first_row + self.q_tile <= self.ends[s]
            )
            interior = jnp.where(
                holds, (self.offsets[s] + self.tile - 1) // self.tile, interior
            )
        first_block = jnp.where(lowest < (1 << 30), lowest // self.tile, 0)
        last_query_key = (
            (self.query_block + 1) * self.q_tile + self.tile - 1
        ) // self.tile
        last_real_key = (self.length + self.tile - 1) // self.tile
        count_blocks = jnp.where(
            (self.first_row < self.length) & (lowest < (1 << 30)),
            jnp.minimum(last_query_key, last_real_key),
            0,
        )
        return row_start, first_block, count_blocks, interior

    def real(self, indices: _Dynamic) -> _Dynamic:
        """Execute real logic."""
        inside = indices < 0
        for s in range(self.count):
            inside = inside | ((indices >= self.offsets[s]) & (indices < self.ends[s]))
        return inside

    def window_mask(
        self, key_rows: _Dynamic, sl: _Dynamic, row_start: _Dynamic
    ) -> _Dynamic:
        """Execute window mask logic."""
        return (key_rows <= self.rows[None, sl]) & (key_rows >= row_start[None, sl])

    def run(self) -> _Dynamic:
        """Run packed flash attention kernel."""
        row_start, first_block, count_blocks, interior = self._compute_bounds()
        carry = (
            jnp.full((1, self.q_tile), -jnp.inf, jnp.float32),
            jnp.zeros((128 + _ONES, self.q_tile), jnp.float32),
        )
        if self.prefixed:
            if not (self.layer_ref is not None):
                _err_msg = "Assertion failed: layer_ref is not None"
                raise AssertionError(_err_msg)
            carry = _prefix_phase(
                carry,
                self.q,
                self.segment_ref,
                self.layer_ref[0],
                self.head,
                self.first_row,
                self.rows,
                self.first_row < self.length,
                self.prefix_k_hbm,
                self.prefix_v_hbm,
                self.k_buf,
                self.v_buf,
                self.sems,
                tile=self.tile,
                q_tile=self.q_tile,
                count=self.count,
            )

        b = jnp.minimum(count_blocks, self.first_row // self.tile)
        a = jnp.maximum(first_block, interior)
        a, b = (
            jnp.where(a < b, a, count_blocks),
            jnp.where(a < b, b, count_blocks),
        )
        loop = functools.partial(
            _key_loop,
            q=self.q,
            layer=0,
            head=self.head,
            k_hbm=self.k_hbm,
            v_hbm=self.v_hbm,
            k_buf=self.k_buf,
            v_buf=self.v_buf,
            sems=self.sems,
            tile=self.tile,
        )
        self.row_start = row_start
        carry = loop(carry, first_block, a, mask=self.window_mask, value_mask=self.real)
        carry = loop(
            carry,
            a,
            b,
            mask=None,
            value_mask=self.real,
            streamed=self.first_row < self.length,
        )
        carry = loop(
            carry, b, count_blocks, mask=self.window_mask, value_mask=self.real
        )
        _, accumulator = carry
        denominator = accumulator[128:129]
        safe = jnp.where(denominator > 0, denominator, 1.0)
        self.out_ref[0] = (accumulator[:128] / safe).T.astype(jnp.bfloat16)


def _flash_kernel_packed(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Causal MLA flash attention over a packed window of count sequences."""
    return _FlashKernelPackedRunner(*_p_args, **_p_kw).run()


def _prefix_phase(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """The cached-prefix part of ``_flash_kernel_packed``: fold each.

    sequence's arena rows into the online softmax ``carry`` of the query
    tile ``q`` (window ``rows``, from ``first_row``), once per distinct
    sequence of the tile: the key blocks wholly inside its rows stream
    unmasked over the whole tile and only its rows keep the result; the
    partial blocks at either end are masked.
    """
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        carry,
        q,
        segment_ref,
        layer,
        head,
        first_row,
        rows,
        live,
        k_hbm,
        v_hbm,
        k_buf,
        v_buf,
        sems,
    ) = (
        _p_next(_p_iter, _p_get("carry")),
        _p_next(_p_iter, _p_get("q")),
        _p_next(_p_iter, _p_get("segment_ref")),
        _p_next(_p_iter, _p_get("layer")),
        _p_next(_p_iter, _p_get("head")),
        _p_next(_p_iter, _p_get("first_row")),
        _p_next(_p_iter, _p_get("rows")),
        _p_next(_p_iter, _p_get("live")),
        _p_next(_p_iter, _p_get("k_hbm")),
        _p_next(_p_iter, _p_get("v_hbm")),
        _p_next(_p_iter, _p_get("k_buf")),
        _p_next(_p_iter, _p_get("v_buf")),
        _p_next(_p_iter, _p_get("sems")),
    )
    (
        tile,
        q_tile,
        count,
    ) = (
        _p_get("tile"),
        _p_get("q_tile"),
        _p_get("count"),
    )
    offsets = [segment_ref[s] for s in range(count)]
    ends = [segment_ref[s] + segment_ref[count + s] for s in range(count)]
    row_sequence = jnp.full(rows.shape, -1, jnp.int32)
    for s in range(count):
        row_sequence = jnp.where(
            (rows >= offsets[s]) & (rows < ends[s]), s, row_sequence
        )

    # A rolled sequence loop avoids emitting a full attention body for every
    # 256-row query slice, including slices that repeat the same sequence.
    def visit(sequence: _Dynamic, carry: _Dynamic) -> _Dynamic:
        """Execute visit logic."""
        offset = segment_ref[sequence]
        end = offset + segment_ref[count + sequence]
        base = segment_ref[3 * count + sequence]
        size = jnp.where(
            live & (offset < first_row + q_tile) & (end > first_row),
            segment_ref[4 * count + sequence],
            0,
        )
        lo_row, hi_row = base, base + size
        lo = base // tile
        hi = jnp.where(size > 0, (base + size + tile - 1) // tile, lo)

        def mask(key_rows: _Dynamic, sl: _Dynamic) -> _Dynamic:
            """Execute mask logic."""
            return (
                (row_sequence[None, sl] == sequence)
                & (key_rows >= lo_row)
                & (key_rows < hi_row)
            )

        def value_mask(value_rows: _Dynamic) -> _Dynamic:
            """Execute value mask logic."""
            return (value_rows >= lo_row) & (value_rows < hi_row)

        loop = functools.partial(
            _key_loop,
            q=q,
            layer=layer,
            head=head,
            k_hbm=k_hbm,
            v_hbm=v_hbm,
            k_buf=k_buf,
            v_buf=v_buf,
            sems=sems,
            tile=tile,
        )
        # Whole blocks [a, b); without one, [lo, hi) is all edge.
        a, b = (lo_row + tile - 1) // tile, hi_row // tile
        a, b = jnp.where(a < b, a, hi), jnp.where(a < b, b, hi)

        # Both edges, [lo, a) and [b, hi), through one rolled masked body.
        def edge(i: _Dynamic, carry: _Dynamic) -> _Dynamic:
            """Execute edge logic."""
            return loop(
                carry,
                jnp.where(i == 0, lo, b),
                jnp.where(i == 0, a, hi),
                mask=mask,
                value_mask=value_mask,
            )

        carry = jax.lax.fori_loop(0, 2, edge, carry)
        streamed = loop(carry, a, b, streamed=True)
        own = row_sequence[None] == sequence
        return tuple(_mask_where(own, new, old) for new, old in zip(streamed, carry))

    # Visit only the sequences that meet the tile (contiguous: offsets
    # ascend); any other visit is a no-op, but one per slot and query tile
    # costs ~0.6 ms a layer at 16 slots.
    first, end = jnp.int32(count), jnp.int32(0)
    for s in range(count):
        meets = live & (offsets[s] < first_row + q_tile) & (ends[s] > first_row)
        first = jnp.where(meets, jnp.minimum(first, s), first)
        end = jnp.where(meets, jnp.maximum(end, s + 1), end)
    return jax.lax.fori_loop(first, jnp.maximum(first, end), visit, carry)


# The online softmax runs transposed, keys by queries (S^T = K Q^T,
# out^T = [V^T; ones] P^T): per-query statistics reduce over sublanes, and PV
# streams only the value rows (plus _ONES rows of ones for the denominators)
# against the probabilities as MXU weights. QK and softmax for FRAGMENT-query
# fragment i overlap PV for i - 1 (separate physical/query-compute tiling and
# delayed PV follow the scheduling idea in heavyball-research/flywheel-tpu
# (Apache-2.0), f646686, pallas/fwd_pipeline.py).
FRAGMENT = 512
_ONES = 8
# Streamed blocks exponentiate against the maximum of the earlier key
# blocks; probabilities reach at most 2**STALE_LIMIT before the
# end-of-block rescale, far inside FP32/BF16 range even summed over a block.
STALE_LIMIT = 64.0


class _KeyLoopRunner:
    """Runner for key block folding in transposed online softmax."""

    def __getattr__(self, name: _Dynamic) -> _Dynamic:
        """Dynamic attribute access."""
        raise AttributeError(name)

    def __init__(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Initialize key loop runner."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            self.carry,
            self.lo,
            self.hi,
        ) = (
            _p_next(_p_iter, _p_get("carry")),
            _p_next(_p_iter, _p_get("lo")),
            _p_next(_p_iter, _p_get("hi")),
        )
        (
            self.q,
            self.layer,
            self.head,
            self.k_hbm,
            self.v_hbm,
            self.k_buf,
            self.v_buf,
            self.sems,
            self.tile,
            self.mask,
            self.value_mask,
            self.streamed,
        ) = (
            _p_get("q"),
            _p_get("layer"),
            _p_get("head"),
            _p_get("k_hbm"),
            _p_get("v_hbm"),
            _p_get("k_buf"),
            _p_get("v_buf"),
            _p_get("sems"),
            _p_get("tile"),
            _p_get("mask"),
            _p_get("value_mask"),
            _p_kw.get("streamed", False),
        )
        self.slots = self.k_buf.shape[0]

    def copies(self, key_block: _Dynamic, slot: _Dynamic) -> tuple[_Dynamic, _Dynamic]:
        """Execute copies logic."""
        return (
            pltpu.make_async_copy(
                self.k_hbm.at[
                    self.layer, self.head, pl.ds(key_block * self.tile, self.tile), :
                ],
                self.k_buf.at[slot],
                self.sems.at[0, slot],
            ),
            pltpu.make_async_copy(
                self.v_hbm.at[
                    self.layer, self.head, :, pl.ds(key_block * self.tile, self.tile)
                ],
                self.v_buf.at[slot],
                self.sems.at[1, slot],
            ),
        )

    def prime(self) -> None:
        """Start async copies for initial key block."""
        for copy in self.copies(self.lo, self.lo % self.slots):
            copy.start()

    def block(self, key_block: _Dynamic) -> tuple[_Dynamic, _Dynamic]:
        """Wait for this block's key/value tile and prefetch the next."""
        slot = key_block % self.slots
        for copy in self.copies(key_block, slot):
            copy.wait()

        @pl.when(key_block + 1 < self.hi)
        def _prefetch_next() -> None:
            """Start async copies for the next key block."""
            for c in self.copies(key_block + 1, (key_block + 1) % self.slots):
                c.start()

        k, v = self.k_buf[slot], self.v_buf[slot]
        if self.value_mask is not None:
            value_keys = key_block * self.tile + jax.lax.broadcasted_iota(
                jnp.int32, (1, self.tile), 1
            )
            v = jnp.where(self.value_mask(value_keys), v, jnp.zeros((), v.dtype))
        return k, jnp.concatenate((v, jnp.ones((_ONES, self.tile), v.dtype)), axis=0)

    def step(self, key_block: _Dynamic, carry: _Dynamic) -> _Dynamic:
        """Execute step logic."""
        k, v = self.block(key_block)
        key_rows = key_block * self.tile + jax.lax.broadcasted_iota(
            jnp.int32, (self.tile, 1), 0
        )
        return _exact_step(
            self.q,
            k,
            v,
            carry,
            None if self.mask is None else functools.partial(self.mask, key_rows),
        )

    def exact(self, c: _Dynamic) -> _Dynamic:
        """Execute exact logic."""
        pl.when(self.lo < self.hi)(self.prime)
        return jax.lax.fori_loop(self.lo, self.hi, self.step, c)

    def stream_body(
        self, key_block: _Dynamic, state: _Dynamic
    ) -> tuple[_Dynamic, _Dynamic]:
        """Execute stream loop body."""
        c, worst = state
        bk, bv = self.block(key_block)
        c, gap = _stale_step(self.q, bk, bv, c)
        return c, jnp.maximum(worst, gap)

    def stream(self, c: _Dynamic) -> _Dynamic:
        """Execute stream logic."""
        self.prime()
        first = self.step(self.lo, c)
        worst = jnp.full(c[0].shape, -jnp.inf, jnp.float32)
        result, worst = jax.lax.fori_loop(
            self.lo + 1, self.hi, self.stream_body, (first, worst)
        )
        return jax.lax.cond(
            jnp.max(worst) > STALE_LIMIT, self.exact, lambda _: result, c
        )

    def run(self) -> _Dynamic:
        """Run key loop."""
        if not self.streamed:
            return self.exact(self.carry)
        return jax.lax.cond(self.lo < self.hi, self.stream, lambda c: c, self.carry)


def _key_loop(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Fold key blocks into transposed online softmax carry."""
    return _KeyLoopRunner(*_p_args, **_p_kw).run()


def _exact_step(
    q: _Dynamic, k: _Dynamic, v: _Dynamic, carry: _Dynamic, mask: _Dynamic
) -> _Dynamic:
    """One key block of the transposed online softmax (``_key_loop``), each.

    query's maximum updated before exponentiating (a query may start with
    no key: a -inf maximum). ``v`` is ``[V^T; ones]``.
    """
    maximum, accumulator = carry
    fragment = min(FRAGMENT, q.shape[0])
    maxima, accumulators = [], []
    previous = None

    def fold(sl: _Dynamic, probabilities: _Dynamic, correction: _Dynamic) -> _Dynamic:
        """Execute fold logic."""
        weighted = jax.lax.dot_general(
            v,
            probabilities,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        accumulators.append(accumulator[:, sl] * correction + weighted)

    for qi in range(q.shape[0] // fragment):
        sl = slice(qi * fragment, (qi + 1) * fragment)
        scores = jax.lax.dot_general(
            k, q[sl], (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
        )  # [keys, queries]
        if mask is not None:
            scores = jnp.where(mask(sl), scores, -jnp.inf)
        new_maximum = jnp.maximum(
            maximum[:, sl], jnp.max(scores, axis=0, keepdims=True)
        )
        # A query with no key yet keeps a -inf maximum; exponentiate against 0.
        safe_maximum = jnp.where(new_maximum == -jnp.inf, 0.0, new_maximum)
        probabilities = jnp.exp2(scores - safe_maximum).astype(jnp.bfloat16)
        maxima.append(new_maximum)
        if previous is not None:
            fold(*previous)
        previous = sl, probabilities, jnp.exp2(maximum[:, sl] - safe_maximum)
    if not (previous is not None):
        _err_msg = "Assertion failed: previous is not None"
        raise AssertionError(_err_msg)
    fold(*previous)
    return jnp.concatenate(maxima, axis=1), jnp.concatenate(accumulators, axis=1)


def _stale_step(q: _Dynamic, k: _Dynamic, v: _Dynamic, carry: _Dynamic) -> _Dynamic:
    """One unmasked key block with every query's running maximum finite.

    exponentiate the block's scores against the carried maximum (not the
    block's own), so each score feeds the max and the exponent straight
    from the MXU instead of being stored and read back, and rescale once
    the block is folded in. The carry means the same as after an exact
    step. Returns (carry, each query's block maximum less the carried
    one, ``[1, Q]``).
    """
    maximum, accumulator = carry
    fragment = min(FRAGMENT, q.shape[0])
    maxima, accumulators, gaps = [], [], []
    previous = None

    def fold(
        sl: _Dynamic,
        probabilities: _Dynamic,
        reference: _Dynamic,
        new_maximum: _Dynamic,
    ) -> _Dynamic:
        """Execute fold logic."""
        weighted = jax.lax.dot_general(
            v,
            probabilities,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        accumulators.append(
            (accumulator[:, sl] + weighted) * jnp.exp2(reference - new_maximum)
        )

    for qi in range(q.shape[0] // fragment):
        sl = slice(qi * fragment, (qi + 1) * fragment)
        scores = jax.lax.dot_general(
            k, q[sl], (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
        )  # [keys, queries]
        reference = maximum[:, sl]
        block_maximum = jnp.max(scores, axis=0, keepdims=True)
        probabilities = jnp.exp2(scores - reference).astype(jnp.bfloat16)
        new_maximum = jnp.maximum(reference, block_maximum)
        maxima.append(new_maximum)
        gaps.append(block_maximum - reference)
        if previous is not None:
            fold(*previous)
        previous = sl, probabilities, reference, new_maximum
    if not (previous is not None):
        _err_msg = "Assertion failed: previous is not None"
        raise AssertionError(_err_msg)
    fold(*previous)
    carry = jnp.concatenate(maxima, axis=1), jnp.concatenate(accumulators, axis=1)
    return carry, jnp.concatenate(gaps, axis=1)


class CpPrefix(NamedTuple):
    """Context parallel prefixes (vLLM decode context parallelism): each.

    rank's pool holds only its share of every sequence's pages (page p on
    rank p % 32); each MLA layer all-gathers every rank's pages and attends
    its 3 heads over all of them (see make_mla_attention_layer_sp_ring).
    ``pools``: the op's MLA layers' unified pools, in order (rank-sharded);
    ``ids``: ``[32, 1 + P]`` int32, rank-sharded -- each rank's own prefix
    page count then its pages, sequences end to end; ``local``: ``[32, 2 *
    count]`` int32, rank-sharded -- each sequence's base in that rank's
    arena and its token count there.
    """

    pools: tuple
    ids: _Dynamic
    local: _Dynamic

    def split(self) -> _Dynamic:
        """Execute split logic."""
        return tuple(CpPrefix((pool,), self.ids, self.local) for pool in self.pools)


GATHER_STREAMS = 6  # the three physical ring cycles, both directions
GATHER_PARTS = 4  # chunks per page (32 of its 128 rows each)


class _PrefixPageGatherRunner:
    """Runner for all-gather of valid prefix pages over the six TP32 rings."""

    def __getattr__(self, name: _Dynamic) -> _Dynamic:
        """Dynamic attribute access."""
        raise AttributeError(name)

    def __init__(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Initialize prefix page gather runner."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            self.tables,
            self.counts_ref,
            self.ids_ref,
            self.pool_hbm,
            self.out_hbm,
            self.local_sem,
            self.send_sems,
            self.recv_sems,
            self.done,
        ) = (
            _p_next(_p_iter, _p_get("tables")),
            _p_next(_p_iter, _p_get("counts_ref")),
            _p_next(_p_iter, _p_get("ids_ref")),
            _p_next(_p_iter, _p_get("pool_hbm")),
            _p_next(_p_iter, _p_get("out_hbm")),
            _p_next(_p_iter, _p_get("local_sem")),
            _p_next(_p_iter, _p_get("send_sems")),
            _p_next(_p_iter, _p_get("recv_sems")),
            _p_next(_p_iter, _p_get("done")),
        )
        self.slots = _p_get("slots")
        self.ranks = 32
        self.rows = self.out_hbm.shape[1] // GATHER_PARTS
        self.rank = lax.axis_index("k3_sp")
        self.positions = [self.tables[f, 0, self.rank] for f in range(3)]
        self.neighbours = [
            self.ring(f, self.positions[f] + d) for f in range(3) for d in (-1, 1)
        ]

    def ring(self, family: _Dynamic, position: _Dynamic) -> _Dynamic:
        """Fetch ring peer rank."""
        return self.tables[family, 1, position % self.ranks]

    def chunks(self, owner: _Dynamic, stream: _Dynamic) -> _Dynamic:
        """Calculate chunk count for owner on stream."""
        return (
            jnp.maximum(
                self.counts_ref[owner] * GATHER_PARTS - stream + GATHER_STREAMS - 1, 0
            )
            // GATHER_STREAMS
        )

    def chunk(self, owner: _Dynamic, stream: _Dynamic, i: _Dynamic) -> _Dynamic:
        """Construct chunk reference."""
        c = stream + GATHER_STREAMS * i
        return self.out_hbm.at[
            owner * self.slots + 1 + c // GATHER_PARTS,
            pl.ds((c % GATHER_PARTS) * self.rows, self.rows),
        ]

    def remote(
        self, src: _Dynamic, stream: _Dynamic, step: _Dynamic, peer: _Dynamic
    ) -> _Dynamic:
        """Create async remote copy."""
        return pltpu.make_async_remote_copy(
            src,
            src,
            self.send_sems.at[stream],
            self.recv_sems.at[stream, step],
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )

    def local(self, q: _Dynamic) -> _Dynamic:
        """Create local pool copy."""
        return pltpu.make_async_copy(
            self.pool_hbm.at[self.ids_ref[1 + q]],
            self.out_hbm.at[self.rank * self.slots + 1 + q],
            self.local_sem,
        )

    def _sync_local_copies(self) -> None:
        """Execute local copies and neighbor barrier."""
        count = self.counts_ref[self.rank]
        pl.loop(0, count)(lambda q: self.local(q).start())
        barrier = pltpu.get_barrier_semaphore()
        for peer in self.neighbours:
            pl.semaphore_signal(
                barrier, 1, device_id=(peer,), device_id_type=pl.DeviceIdType.MESH
            )
        pl.semaphore_wait(barrier, len(self.neighbours))
        pl.loop(0, count)(lambda q: self.local(q).wait())

    def _step_stream(
        self,
        k: _Dynamic,
        stream: _Dynamic,
        family: _Dynamic,
        direction: _Dynamic,
        downstream: _Dynamic,
    ) -> None:
        """Process one stream at ring step k."""
        owner = self.ring(family, self.positions[family] - direction * k)
        size = self.chunks(owner, stream)

        pl.loop(0, size)(
            lambda i: self.remote(
                self.chunk(owner, stream, i), stream, k, self.rank
            ).wait_recv()
        )

        def _forward() -> None:
            """Forward chunks downstream."""
            pl.loop(0, size)(
                lambda i: self.remote(
                    self.chunk(owner, stream, i), stream, k + 1, downstream
                ).start()
            )

        pl.when(k < self.ranks - 1)(_forward)

    def run(self) -> None:
        """Run prefix page gather kernel."""
        self._sync_local_copies()

        streams = []
        for stream in range(GATHER_STREAMS):
            family = stream // 2 % 3
            direction = -1 if stream % 2 == 0 else 1
            downstream = self.ring(family, self.positions[family] + direction)
            streams.append((stream, family, direction, downstream))

        for stream, _, _, downstream in streams:
            pl.loop(0, self.chunks(self.rank, stream))(
                lambda i, st=stream, ds=downstream: self.remote(
                    self.chunk(self.rank, st, i), st, 1, ds
                ).start()
            )

        def _step_k(k: _Dynamic) -> None:
            """Execute step k for all streams."""
            for stream, family, direction, downstream in streams:
                self._step_stream(k, stream, family, direction, downstream)

        pl.loop(1, self.ranks)(_step_k)

        def _drain_k(k: _Dynamic) -> None:
            """Drain step k for all streams."""
            for stream, family, direction, downstream in streams:
                owner = self.ring(family, self.positions[family] - direction * k)
                pl.loop(0, self.chunks(owner, stream))(
                    lambda i, st=stream, ow=owner, ds=downstream: self.remote(
                        self.chunk(ow, st, i), st, k + 1, ds
                    ).wait_send()
                )

        pl.loop(0, self.ranks - 1)(_drain_k)

        for peer in self.neighbours:
            pl.semaphore_signal(
                self.done, 1, device_id=(peer,), device_id_type=pl.DeviceIdType.MESH
            )
        pl.semaphore_wait(self.done, len(self.neighbours))


def _prefix_page_gather_kernel(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """All-gather every rank's valid prefix pages over the six TP32 rings."""
    return _PrefixPageGatherRunner(*_p_args, **_p_kw).run()


def gathered_prefix_pages(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Every rank's cached prefix pages, gathered (inside ``shard_map``).

    ``(pages, order)`` for ``expand_prefix_pages``. ``pool``/``ids``/``local``
    are this rank's view (see ``CpPrefix``). ``pages`` holds every rank's
    ``1 + P`` page slots in mesh order (``[32 (1 + P), ...]``; only valid
    pages are written); ``order`` is ``prefix_page_order``'s, ``capacity``
    the most prefix tokens a window holds.
    """
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        pool,
        ids,
        local,
        tables,
        axis_name,
    ) = (
        _p_next(_p_iter, _p_get("pool")),
        _p_next(_p_iter, _p_get("ids")),
        _p_next(_p_iter, _p_get("local")),
        _p_next(_p_iter, _p_get("tables")),
        _p_next(_p_iter, _p_get("axis_name")),
    )
    (
        count,
        capacity,
    ) = (
        _p_get("count"),
        _p_get("capacity"),
    )
    page_tokens = pool.shape[-3] * pool.shape[-2]
    slots = ids.shape[0]  # 1 + P: slot 0 holds the count (never referenced)
    every = jax.lax.all_gather(local, axis_name)  # [32, 2 * count]: bases, token counts
    ranks = every.shape[0]
    counts = jnp.sum(every[:, count:], axis=1) // page_tokens  # pages per rank
    pages = pl.pallas_call(
        functools.partial(_prefix_page_gather_kernel, slots=slots),
        out_shape=jax.ShapeDtypeStruct((ranks * slots, *pool.shape[1:]), pool.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=(
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA((GATHER_STREAMS,)),
                pltpu.SemaphoreType.DMA((GATHER_STREAMS, ranks)),
                pltpu.SemaphoreType.REGULAR,
            ),
        ),
        compiler_params=pltpu.CompilerParams(collective_id=44),
        name="kimi_prefill_prefix_page_gather",
    )(
        jnp.asarray(tables, jnp.int32),
        counts.astype(jnp.int32),
        ids.astype(jnp.int32),
        pool,
    )
    return pages, prefix_page_order(
        every, count=count, page_tokens=page_tokens, slots=slots, capacity=capacity
    )


def prefix_page_order(
    every: _Dynamic,
    *,
    count: _Dynamic,
    page_tokens: _Dynamic,
    slots: _Dynamic,
    capacity: _Dynamic,
) -> _Dynamic:
    """The arena order of ``gathered_prefix_pages``: ``[1 + capacity / page]``.

    int32, the total page count, then the gathered slot of each arena page.
    ``every`` is every rank's ``local`` (``[ranks, 2 * count]``: each
    sequence's base and token count in that rank's pages); rank r's local
    page q sits at gathered slot ``r * slots + 1 + q``. Arena runs go
    sequence-major, then mesh rank, so each sequence is contiguous at its
    global base (descriptor ``[3 * count:5 * count]``); attention over a
    prefix does not depend on its key order.
    """
    ranks = every.shape[0]
    starts = (every[:, :count] // page_tokens).T.reshape(
        -1
    )  # [count * ranks] local first pages
    sizes = (every[:, count:] // page_tokens).T.reshape(-1)
    run_rank = jnp.tile(jnp.arange(ranks, dtype=jnp.int32), count)
    ends = jnp.cumsum(sizes)
    if capacity % page_tokens:
        _err_msg = f"the prefix capacity must be whole {page_tokens}-token pages"
        raise ValueError(_err_msg)
    arena = jnp.arange(capacity // page_tokens, dtype=jnp.int32)
    run = jnp.minimum(
        jnp.sum(arena[:, None] >= ends[None, :], axis=1), sizes.shape[0] - 1
    )
    within = arena - (ends[run] - sizes[run])
    source = run_rank[run] * slots + 1 + starts[run] + within
    total = ends[-1]
    order = jnp.where(arena < total, source, 0).astype(jnp.int32)
    return jnp.concatenate((total[None].astype(jnp.int32), order))


EXPAND_PAGES = 4  # pages per expansion step


class _ExpandPagesRunner:
    """Runner for expand_prefix_pages double-buffered reads and projections."""

    def __getattr__(self, name: _Dynamic) -> _Dynamic:
        """Dynamic attribute access."""
        raise AttributeError(name)

    def __init__(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Initialize expand pages runner."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            self.order_ref,
            self.pages_hbm,
            self.up_ref,
            self.keys_hbm,
            self.values_hbm,
            self.buf,
            self.kbuf,
            self.vbuf,
            self.sems,
        ) = (
            _p_next(_p_iter, _p_get("order_ref")),
            _p_next(_p_iter, _p_get("pages_hbm")),
            _p_next(_p_iter, _p_get("up_ref")),
            _p_next(_p_iter, _p_get("keys_hbm")),
            _p_next(_p_iter, _p_get("values_hbm")),
            _p_next(_p_iter, _p_get("buf")),
            _p_next(_p_iter, _p_get("kbuf")),
            _p_next(_p_iter, _p_get("vbuf")),
            _p_next(_p_iter, _p_get("sems")),
        )
        self.heads = _p_get("heads")
        self.total = self.order_ref[0]
        _, self.pages, self.page_rows, self.packing, self.lanes = self.buf.shape
        self.page_tokens = self.page_rows * self.packing
        self.span = self.pages * self.page_tokens
        self.steps = (self.total + self.pages - 1) // self.pages
        self.words = self.buf.bitcast(jnp.uint32)

    def reads(self, c: _Dynamic, slot: _Dynamic) -> list[_Dynamic]:
        """Execute reads logic."""
        return [
            pltpu.make_async_copy(
                self.pages_hbm.at[self.order_ref[1 + c * self.pages + j]],
                self.buf.at[slot, j],
                self.sems.at[0, slot],
            )
            for j in range(self.pages)
        ]

    def writes(self, c: _Dynamic, slot: _Dynamic) -> list[_Dynamic]:
        """Execute writes logic."""
        rows = pl.ds(c * self.span, self.span)
        return [
            pltpu.make_async_copy(
                self.kbuf.at[slot], self.keys_hbm.at[:, rows], self.sems.at[1, slot]
            ),
            pltpu.make_async_copy(
                self.vbuf.at[slot],
                self.values_hbm.at[:, :, rows],
                self.sems.at[2, slot],
            ),
        ]

    def _expand_tokens(self, c: _Dynamic, slot: _Dynamic) -> None:
        """Expand tokens for current step."""
        for j in range(self.pages):
            x = pltpu.bitcast(
                self.words.at[slot, j].reshape(self.page_rows, self.lanes)[...],
                self.buf.dtype,
            ).astype(jnp.bfloat16)
            x = jnp.where(c * self.pages + j < self.total, x, jnp.zeros((), x.dtype))
            rows = pl.ds(j * self.page_tokens, self.page_tokens)
            rope = x[:, 512:576]
            for h in range(self.heads):
                expanded = jnp.dot(
                    x[:, :512],
                    self.up_ref[:, h * 256 : (h + 1) * 256],
                    preferred_element_type=jnp.float32,
                ).astype(jnp.bfloat16)
                self.kbuf[slot, h, rows, :] = jnp.concatenate(
                    (
                        expanded[:, :128],
                        rope,
                        jnp.zeros((self.page_tokens, 64), jnp.bfloat16),
                    ),
                    axis=-1,
                )
                self.vbuf[slot, h, :, rows] = expanded[:, 128:].T

    def step(self, c: _Dynamic, carry: _Dynamic) -> _Dynamic:
        """Execute step logic."""
        slot = c % 2
        for copy in self.reads(c, slot):
            copy.wait()

        @pl.when(c + 1 < self.steps)
        def _prefetch_reads() -> None:
            """Start read copies for the next step."""
            for copy in self.reads(c + 1, 1 - slot):
                copy.start()

        @pl.when(c >= _CONST_2)
        def _wait_prev_writes() -> None:
            """Wait for write copies from two steps ago."""
            for copy in self.writes(c - 2, slot):
                copy.wait()

        self._expand_tokens(c, slot)

        for copy in self.writes(c, slot):
            copy.start()
        return carry

    def run(self) -> None:
        """Run expand pages kernel."""

        @pl.when(self.steps > 0)
        def _prime_reads() -> None:
            """Start initial read copies."""
            for copy in self.reads(0, 0):
                copy.start()

        jax.lax.fori_loop(0, self.steps, self.step, 0)

        @pl.when(self.steps >= _CONST_2)
        def _drain_second_last_writes() -> None:
            """Wait for second-to-last step write copies."""
            for copy in self.writes(self.steps - 2, (self.steps - 2) % 2):
                copy.wait()

        @pl.when(self.steps >= 1)
        def _drain_last_writes() -> None:
            """Wait for final step write copies."""
            for copy in self.writes(self.steps - 1, (self.steps - 1) % 2):
                copy.wait()


def _expand_pages_kernel(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Expand prefix pages into head-major arena."""
    return _ExpandPagesRunner(*_p_args, **_p_kw).run()


def expand_prefix_pages(
    pages: _Dynamic, order: _Dynamic, key_value_up: _Dynamic
) -> _Dynamic:
    """One MLA layer's cached prefixes as a one-layer arena for.

    ``mla_flash_prefill_packed``: keys ``[1, H, C, 256]`` and values
    transposed, ``[1, H, 128, C]``, with H = ``key_value_up``'s heads (this
    rank's 3) and C = ``(len(order) - 1) * page_tokens``. Arena page i is
    gathered page ``order[1 + i]`` (unified-pool layout
    ``[page_rows, packing, 640]``, lanes 0-575 the latent), up-projected
    exactly as ``mla_projections`` makes a new token's key and value. Rows
    past the ``order[0]`` pages (rounded up to ``EXPAND_PAGES``) are never
    written: the flash kernel reads only each sequence's rows.
    """
    heads = key_value_up.shape[1] // 256
    page_rows, packing, lanes = pages.shape[1:]
    page_tokens = page_rows * packing
    span = EXPAND_PAGES * page_tokens
    capacity = (order.shape[0] - 1) * page_tokens
    if capacity % span:
        _err_msg = f"the prefix capacity must be a multiple of {span} tokens"
        raise ValueError(_err_msg)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    keys, values = pl.pallas_call(
        functools.partial(_expand_pages_kernel, heads=heads),
        out_shape=(
            jax.ShapeDtypeStruct((heads, capacity, 256), jnp.bfloat16),
            jax.ShapeDtypeStruct((heads, 128, capacity), jnp.bfloat16),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[hbm, pl.BlockSpec(memory_space=pltpu.VMEM)],
            out_specs=(hbm, hbm),
            scratch_shapes=(
                pltpu.VMEM((2, EXPAND_PAGES, page_rows, packing, lanes), pages.dtype),
                pltpu.VMEM((2, heads, span, 256), jnp.bfloat16),
                pltpu.VMEM((2, heads, 128, span), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((3, 2)),
            ),
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=64 * 1024**2),
        name="kimi_prefill_expand_prefix_pages",
    )(order.astype(jnp.int32), pages, key_value_up)
    return keys[None], values[None]


def mla_flash_prefill_packed(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Causal MLA over a packed window of ``count`` sequences, per head.

    Args:
        query, key, value: The window's head-major ``[3, T, 256]``,
            ``[3, T, 256]`` and ``[3, T, 128]`` BF16 rows, queries
            pre-scaled by ``SCALE_LOG2``.
        length: ``[1]``, the end of the last sequence.
        segments: The int32 descriptor (``_flash_kernel_packed``),
            ``[3 * count]``, or ``[5 * count]`` with ``arena``.
        arena: Optional ``([L, 3, C, 256], [L, 3, 128, C])`` cached-prefix
            keys and transposed values, read in place from HBM.
        layer: The arena layer, a ``[1]`` or scalar int32.

    Returns:
        ``[3, T, 128]`` BF16; padded rows are zero.
    """
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        query,
        key,
        value,
        length,
        segments,
    ) = (
        _p_next(_p_iter, _p_get("query")),
        _p_next(_p_iter, _p_get("key")),
        _p_next(_p_iter, _p_get("value")),
        _p_next(_p_iter, _p_get("length")),
        _p_next(_p_iter, _p_get("segments")),
    )
    (
        count,
        tile,
        arena,
        layer,
    ) = (
        _p_get("count"),
        _p_get("tile", TILE),
        _p_get("arena"),
        _p_get("layer", 0),
    )
    if query.ndim != _CONST_3 or query.shape[2] != _CONST_256:
        _err_msg = "query must be [H,T,256]"
        raise ValueError(_err_msg)
    heads, tokens = query.shape[:2]
    if heads != _CONST_3:
        _err_msg = "the window phase takes this rank's 3 heads"
        raise ValueError(_err_msg)
    if key.shape != (3, tokens, 256) or value.shape != (3, tokens, 128):
        _err_msg = "packed MLA keys/values are the window's rows"
        raise ValueError(_err_msg)
    fields = 3 if arena is None else 5
    if segments.shape != (fields * count,):
        _err_msg = f"segments must be [{fields} * count]"
        raise ValueError(_err_msg)
    length = jnp.asarray(length, jnp.int32).reshape((1,))
    q_tile, k_tile = _tiles(tokens, tokens, tile)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    scalars = (length, segments.astype(jnp.int32))
    operands = (
        query,
        key[None],
        jnp.swapaxes(value, 1, 2)[None],
    )  # the window as a one-layer arena
    in_specs = (pl.BlockSpec((1, q_tile, 256), lambda h, i, *_: (h, i, 0)), hbm, hbm)
    if arena is not None:
        arena_keys, arena_values = arena
        if (
            arena_keys.ndim != _CONST_4
            or arena_keys.shape[1:] != (heads, arena_keys.shape[2], 256)
            or arena_values.shape != (*arena_keys.shape[:2], 128, arena_keys.shape[2])
            or arena_keys.shape[2] % k_tile
        ):
            _err_msg = (
                f"arena must be [L,{heads},C,256] / [L,{heads},128,C] with C a "
                f"multiple of {k_tile}"
            )
            raise ValueError(_err_msg)
        scalars += (jnp.asarray(layer, jnp.int32).reshape((1,)),)
        operands += (arena_keys, arena_values)
        in_specs += (hbm, hbm)
    name = "kimi_mla_flash_prefill_packed" + ("_prefixed" if arena is not None else "")
    return pl.pallas_call(
        functools.partial(
            _flash_kernel_packed,
            tile=k_tile,
            q_tile=q_tile,
            count=count,
            prefixed=arena is not None,
        ),
        out_shape=jax.ShapeDtypeStruct((heads, tokens, 128), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalars),
            grid=(heads, tokens // q_tile),
            in_specs=in_specs,
            out_specs=pl.BlockSpec((1, q_tile, 128), lambda h, i, *_: (h, i, 0)),
            scratch_shapes=(
                pltpu.VMEM((2, k_tile, 256), jnp.bfloat16),
                pltpu.VMEM((2, 128, k_tile), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((2, 2)),
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
            vmem_limit_bytes=64 * 1024**2,
        ),
        name=name,
    )(*scalars, *operands)


def _tiles(tokens: _Dynamic, capacity: _Dynamic, tile: _Dynamic) -> _Dynamic:
    """(query tile, key tile). Default 1024-row query and 2048-row key.

    tiles; ``KernelConfig.mla_flash_qtile`` / ``mla_flash_ktile`` override.
    """
    q_tile = _config().mla_flash_qtile
    k_tile = _config().mla_flash_ktile
    q_tile = q_tile if tokens % q_tile == 0 else tile
    k_tile = (
        k_tile if capacity % k_tile == 0 else (512 if capacity % 512 == 0 else tile)
    )
    return q_tile, k_tile


# ============================================================================
# SP block half rings
# ============================================================================
#
# The KDA layer's window all-gather and reduce-scatter, one SP block at a time.


# Half rings for one SP block. Each of the three cycles carries a column
# third; on each cycle a rank's stripe travels all its rows both
# ways, HALF_HOPS[+1] = 16 hops forward and 15 back, so a ring takes 17
# steps instead of the full ring's 32 at the same bytes per link.
HALF_HOPS = {1: 16, -1: 15}
HALF_STEPS = 17


def _half_ring_streams(width: _Dynamic) -> _Dynamic:
    """Execute half ring streams logic."""
    blocks = width // 128
    widths = [128 * (blocks // 3 + (i < blocks % 3)) for i in range(3)]
    cols = [0, widths[0], widths[0] + widths[1]]
    return tuple(
        (family, direction, cols[family], widths[family])
        for family in range(3)
        for direction in (1, -1)
    )


def _half_gather_step(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Step ``step`` (0..16) of SP block ``block``'s window all-gather on.

    half rings: copies go HBM to the peer's HBM, each owner's rows straight
    to their window rows (forwarded from there), so no buffer is reused and
    no credits are needed. ``local_ref``: the rank's rows; ``window_ref``:
    the window.
    """
    _p_iter = iter(_p_args)
    (
        tables_ref,
        local_ref,
        window_ref,
        send_sems,
        recv_sems,
        own_sem,
        step,
        block,
    ) = (
        _dyn(next(_p_iter, _p_kw.get("tables_ref"))),
        _dyn(next(_p_iter, _p_kw.get("local_ref"))),
        _dyn(next(_p_iter, _p_kw.get("window_ref"))),
        _dyn(next(_p_iter, _p_kw.get("send_sems"))),
        _dyn(next(_p_iter, _p_kw.get("recv_sems"))),
        _dyn(next(_p_iter, _p_kw.get("own_sem"))),
        _dyn(_p_kw.get("step")),
        _dyn(_p_kw.get("block")),
    )
    rank = jax.lax.axis_index("k3_sp")
    positions = [tables_ref[f, 0, rank] for f in range(3)]
    stripe = window_stripe(local_ref.shape[0])
    local_rows = pl.ds(block * stripe, stripe)

    def window_rows(family: _Dynamic, position: _Dynamic) -> _Dynamic:
        """Execute window rows logic."""
        owner = tables_ref[family, 2, tables_ref[family, 1, position % 32]]
        return pl.ds(block * 32 * stripe + owner * stripe, stripe)

    @pl.when(step == 0)
    def own_rows() -> _Dynamic:
        """Execute own rows logic."""
        copy = pltpu.make_async_copy(
            local_ref.at[local_rows],
            window_ref.at[window_rows(0, positions[0])],
            own_sem,
        )
        copy.start()
        if local_ref.shape[0] not in (16, 32, 64, 128, 256):
            copy.wait()

    for stream, (family, direction, col, width) in enumerate(
        _half_ring_streams(local_ref.shape[1])
    ):
        cols, hops, position = (
            pl.ds(col, width),
            HALF_HOPS[direction],
            positions[family],
        )
        target = (tables_ref[family, 1, (position + direction) % 32],)

        _st_send_5040 = (family, direction, cols, position, target, stream)

        def send(
            *_p_args: _Dynamic, _st: _Dynamic = _st_send_5040, **_p_kw: _Dynamic
        ) -> _Dynamic:
            """Execute send logic."""
            _p_iter = iter(_p_args)
            _p_get: _Dynamic = _p_kw.get
            _p_next: _Dynamic = next
            family, direction, cols, position, target, stream = _st
            index = _p_next(_p_iter, _p_get("index"))
            source = _p_next(_p_iter, _p_get("source"))
            family = _p_next(_p_iter, _p_get("family", family))
            direction = _p_next(_p_iter, _p_get("direction", direction))
            cols = _p_next(_p_iter, _p_get("cols", cols))
            position = _p_next(_p_iter, _p_get("position", position))
            target = _p_next(_p_iter, _p_get("target", target))
            stream = _p_next(_p_iter, _p_get("stream", stream))
            rows = window_rows(family, position - direction * index)
            return pltpu.make_async_remote_copy(
                source if source is not None else window_ref.at[rows, cols],
                window_ref.at[rows, cols],
                send_sems.at[stream],
                recv_sems.at[stream],
                device_id=target,
                device_id_type=pl.DeviceIdType.MESH,
            )

        own = local_ref.at[local_rows, cols]
        pl.when(step == 0)(lambda send=send, own=own: send(0, own).start())

        @pl.when((step >= 1) & (step <= hops))
        def arrived(send: _Dynamic = send) -> _Dynamic:
            """Execute arrived logic."""
            copy = send(step - 1, None)
            copy.wait_recv()
            copy.wait_send()

        @pl.when((step >= 1) & (step < hops))
        def forward(send: _Dynamic = send) -> _Dynamic:
            """Execute forward logic."""
            send(step, None).start()

    # The local destination is disjoint from every remote stripe. Let the
    # copy run while the half rings progress, then join before KDA consumes
    # this block and before own_sem is reused for the next block.
    if local_ref.shape[0] in (16, 32, 64, 128, 256):

        @pl.when(step == HALF_STEPS - 1)
        def finish_local() -> _Dynamic:
            """Execute finish local logic."""
            pltpu.make_async_copy(
                local_ref.at[local_rows],
                window_ref.at[window_rows(0, positions[0])],
                own_sem,
            ).wait()


class _HalfReduceStepRunner:
    """Runner for one step of half-ring reduce-scatter."""

    def __getattr__(self, name: _Dynamic) -> _Dynamic:
        """Dynamic attribute access."""
        raise AttributeError(name)

    def __init__(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Initialize half reduce step runner."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            self.tables_ref,
            self.length_ref,
            self.partial_ref,
            self.prefix_ref,
            self.output_ref,
            self.read_sems,
            self.send_sems,
            self.recv_sems,
            self.sem_credits,
            self.store_sem,
            self.local_bufs,
            self.payload,
            self.received,
        ) = (
            _p_next(_p_iter, _p_get("tables_ref")),
            _p_next(_p_iter, _p_get("length_ref")),
            _p_next(_p_iter, _p_get("partial_ref")),
            _p_next(_p_iter, _p_get("prefix_ref")),
            _p_next(_p_iter, _p_get("output_ref")),
            _p_next(_p_iter, _p_get("read_sems")),
            _p_next(_p_iter, _p_get("send_sems")),
            _p_next(_p_iter, _p_get("recv_sems")),
            _p_next(_p_iter, _p_get("credits")),
            _p_next(_p_iter, _p_get("store_sem")),
            _p_next(_p_iter, _p_get("local_bufs")),
            _p_next(_p_iter, _p_get("payload")),
            _p_next(_p_iter, _p_get("received")),
        )
        (
            self.step,
            self.block,
            self.replace,
        ) = (
            _p_get("step"),
            _p_get("block"),
            _p_get("replace"),
        )
        self.rank = jax.lax.axis_index("k3_sp")
        self.token_owner = self.tables_ref[0, 2, self.rank]
        self.positions = [self.tables_ref[f, 0, self.rank] for f in range(3)]
        self.stripe = self.payload.shape[1]
        self.local_first = self.block * self.stripe
        self.window_first = self.block * 32 * self.stripe
        self.streams = _half_ring_streams(self.partial_ref.shape[1])

    def make_read(
        self,
        spec: tuple[_Dynamic, ...],
        index: _Dynamic,
    ) -> _Dynamic:
        """Construct read async copy."""
        stream, family, direction, cols, side, hops, position = spec
        peer = self.tables_ref[family, 1, (position + direction * (hops - index)) % 32]
        owner = self.tables_ref[family, 2, peer]
        return pltpu.make_async_copy(
            self.partial_ref.at[
                pl.ds(self.window_first + owner * self.stripe, self.stripe), cols
            ],
            self.local_bufs.at[index % 2, side, :, cols],
            self.read_sems.at[index % 2, stream],
        )

    def make_remote(
        self, stream: int, target: _Dynamic, cols: _Dynamic, side: int, index: _Dynamic
    ) -> _Dynamic:
        """Construct remote async copy."""
        return pltpu.make_async_remote_copy(
            self.payload.at[side, :, cols],
            self.received.at[index % 2, side, :, cols],
            self.send_sems.at[stream],
            self.recv_sems.at[index % 2, stream],
            device_id=target,
            device_id_type=pl.DeviceIdType.MESH,
        )

    def _accumulate_stream(
        self, stream: int, family: int, direction: int, col: int, width: int
    ) -> None:
        """Accumulate logic for one stream."""
        cols = pl.ds(col, width)
        hops = HALF_HOPS[direction]
        position = self.positions[family]
        side = 0 if direction == 1 else 1
        spec = (stream, family, direction, cols, side, hops, position)
        target = (self.tables_ref[family, 1, (position + direction) % 32],)
        source = (self.tables_ref[family, 1, (position - direction) % 32],)

        pl.when(self.step == 0)(lambda: self.make_read(spec, 0).start())

        def _do_accumulate() -> None:
            """Accumulate read and remote copy."""
            self.make_read(spec, self.step).wait()
            pl.when(self.step < hops)(
                lambda: self.make_read(spec, self.step + 1).start()
            )
            local = self.local_bufs.at[self.step % 2, side, :, cols][...]

            pl.when(self.step == 0)(
                lambda: self.payload.at[side, :, cols].__setitem__(slice(None), local)
            )

            def _do_add() -> None:
                """Add remote copy to payload."""
                self.make_remote(stream, target, cols, side, self.step - 1).wait()
                self.payload.at[side, :, cols][...] = (
                    local + self.received.at[(self.step - 1) % 2, side, :, cols][...]
                )
                pl.semaphore_signal(
                    self.sem_credits.at[stream],
                    1,
                    device_id=source,
                    device_id_type=pl.DeviceIdType.MESH,
                )

            pl.when((self.step > 0) & (self.step < hops))(_do_add)

            def _do_forward() -> None:
                """Forward payload to downstream peer."""
                pl.when(self.step >= _CONST_2)(
                    lambda: pl.semaphore_wait(self.sem_credits.at[stream], 1)
                )
                self.make_remote(stream, target, cols, side, self.step).start()

            pl.when(self.step < hops)(_do_forward)
            pl.when(self.step == hops)(
                lambda: self.make_remote(
                    stream, target, cols, side, self.step - 1
                ).wait()
            )

        pl.when(self.step <= hops)(_do_accumulate)

    def _finish(self) -> None:
        """Execute finish step for all half-ring streams."""
        for stream, (family, direction, col, width) in enumerate(self.streams):
            if direction != 1:
                continue
            cols = pl.ds(col, width)
            staging = self.local_bufs.at[1, 1, :, cols]
            prefix_copy = pltpu.make_async_copy(
                self.prefix_ref.at[pl.ds(self.local_first, self.stripe), cols],
                staging,
                self.store_sem,
            )
            prefix_copy.start()
            prefix_copy.wait()
            valid = (
                sp_token_index(
                    self.token_owner,
                    self.local_first
                    + jax.lax.broadcasted_iota(jnp.int32, (self.stripe, width), 0),
                    self.stripe,
                )
                < self.length_ref[0]
            )
            projection = (
                self.local_bufs.at[0, 0, :, cols][...].astype(jnp.float32)
                + self.received.at[1, 0, :, cols][...].astype(jnp.float32)
                + self.received.at[0, 1, :, cols][...].astype(jnp.float32)
            )
            residual = staging[...].astype(jnp.float32)
            residual = jnp.where(self.replace, jnp.zeros_like(residual), residual)
            staging[...] = (residual + jnp.where(valid, projection, 0)).astype(
                jnp.bfloat16
            )
            output = pltpu.make_async_copy(
                staging,
                self.output_ref.at[pl.ds(self.local_first, self.stripe), cols],
                self.store_sem,
            )
            output.start()
            output.wait()

        for stream in range(len(self.streams)):
            pl.semaphore_signal(
                self.sem_credits.at[stream],
                1,
                device_id=(
                    self.tables_ref[
                        self.streams[stream][0],
                        1,
                        (
                            self.positions[self.streams[stream][0]]
                            - self.streams[stream][1]
                        )
                        % 32,
                    ],
                ),
                device_id_type=pl.DeviceIdType.MESH,
            )
        for stream in range(len(self.streams)):
            pl.semaphore_wait(self.sem_credits.at[stream], 2)

    def run(self) -> None:
        """Run half reduce step."""
        for stream, (family, direction, col, width) in enumerate(self.streams):
            self._accumulate_stream(stream, family, direction, col, width)
        pl.when(self.step == HALF_STEPS - 1)(self._finish)


def _half_reduce_step(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Step (0..16) of SP block's reduce-scatter of partials on half rings."""
    return _HalfReduceStepRunner(*_p_args, **_p_kw).run()


# ============================================================================
# MLA attention layer (sequence-parallel)
# ============================================================================
#
# Projections, flash attention, output gate, projection ring, residual add.


_PROJECTION_NAMES = (
    "query_down_projection",
    "query_up_projection",
    "key_value_down_projection",
    "key_value_up_projection",
    "output_gate_projection",
    "query_normalization",
    "key_value_normalization",
)


class ProjectionWeights(NamedTuple):
    """Projection weights for MLA attention layer."""

    query_down_projection: _Dynamic
    query_up_projection: _Dynamic
    key_value_down_projection: _Dynamic
    key_value_up_projection: _Dynamic
    output_gate_projection: _Dynamic
    query_normalization: _Dynamic
    key_value_normalization: _Dynamic


def make_mla_attention_layer_sp_ring(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """One MLA layer over a packed window of ``segments`` sequences.

    ``run(hidden, prefix, weights, segment_desc, length, arena)``:
    ``hidden`` is the gathered window ([T, 7168] on every rank), ``prefix``
    the rank's residual rows, ``segment_desc`` the ``[5 * segments]``
    descriptor (see ``_flash_kernel_packed``) and ``arena`` the layer's
    ``CpPrefix`` (at most ``prefix_capacity`` prefix tokens). Returns the
    new residual rows and the rank-leading ``[32, T, 576]`` compressed KV
    (identical on every rank). ``output_projection_repacked`` is [384,7168]
    per rank: the rank's attention features over the full hidden width, built
    from the chip pair's decode shards at load time
    (``rank_loader._sequence_parallel_output``).
    """
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    mesh = _p_next(_p_iter, _p_get("mesh"))
    (
        seq_len,
        segments,
        prefix_capacity,
        token_ranks,
        collective_id,
        flash_tile,
    ) = (
        _p_get("seq_len"),
        _p_get("segments"),
        _p_get("prefix_capacity"),
        _p_get("token_ranks"),
        _p_get("collective_id", 31),
        _p_get("flash_tile", 128),
    )
    if not segments:
        _err_msg = "the MLA layer runs packed windows"
        raise ValueError(_err_msg)
    if mesh.size != _CONST_32 or tuple(mesh.axis_names) != ("k3_sp",):
        _err_msg = "ring MLA requires a flat k3_sp TP32 mesh"
        raise ValueError(_err_msg)
    check_window(seq_len, "ring MLA")
    if token_ranks is None:
        token_ranks = tuple(range(32))
    tables = ring_tables(list(mesh.devices.flat), token_ranks)
    local_tokens = seq_len // 32

    def local_packed_cp(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
        """One rank's view: this rank's 3 heads over every rank's prefix.

        pages and the window.
        """
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            full_hidden,
            weights,
            segment_desc,
            length,
            pool,
            ids,
            local,
        ) = (
            _p_next(_p_iter, _p_get("full_hidden")),
            _p_next(_p_iter, _p_get("weights")),
            _p_next(_p_iter, _p_get("segment_desc")),
            _p_next(_p_iter, _p_get("length")),
            _p_next(_p_iter, _p_get("pool")),
            _p_next(_p_iter, _p_get("ids")),
            _p_next(_p_iter, _p_get("local")),
        )
        local_weights = jax.tree.map(lambda value: value[0], weights)
        projections = ProjectionWeights(
            *(local_weights[name] for name in _PROJECTION_NAMES)
        )
        query, new_key, new_value, output_gate, latent = mla_projections(
            full_hidden, projections, return_latent=True, query_scale=SCALE_LOG2
        )
        tokens = query.shape[0]
        real = (jnp.arange(tokens) < length.reshape(()))[None, :, None]
        new_key = jnp.where(real, new_key.transpose(1, 0, 2), 0)
        new_value = jnp.where(real, new_value.transpose(1, 0, 2), 0)
        query = query.transpose(1, 0, 2)  # [3, T, 256]
        pages, order = gathered_prefix_pages(
            pool,
            ids[0],
            local[0],
            tables,
            "k3_sp",
            count=segments,
            capacity=prefix_capacity,
        )
        arena = expand_prefix_pages(pages, order, projections.key_value_up_projection)
        attended = mla_flash_prefill_packed(
            query,
            new_key,
            new_value,
            length,
            segment_desc[: 5 * segments],
            count=segments,
            tile=flash_tile,
            arena=arena,
        )
        attended = attended.transpose(1, 0, 2).reshape(tokens, 384)
        gate = jax.nn.sigmoid(output_gate.astype(jnp.float32)).astype(jnp.bfloat16)
        return (attended * gate).astype(jnp.bfloat16)[None], latent[None]

    def run(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
        """Execute run logic."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            hidden,
            prefix,
            weights,
            segment_desc,
            length,
            arena,
        ) = (
            _p_next(_p_iter, _p_get("hidden")),
            _p_next(_p_iter, _p_get("prefix")),
            _p_next(_p_iter, _p_get("weights")),
            _p_next(_p_iter, _p_get("segment_desc")),
            _p_next(_p_iter, _p_get("length")),
            _p_next(_p_iter, _p_get("arena")),
        )
        (pool,) = arena.pools
        gated, latent = jax.shard_map(
            local_packed_cp,
            mesh=mesh,
            in_specs=(P(), P("k3_sp"), P(), P(), P("k3_sp"), P("k3_sp"), P("k3_sp")),
            out_specs=(P("k3_sp"), P("k3_sp")),
            check_vma=False,
        )(hidden, weights, segment_desc, length, pool, arena.ids, arena.local)
        projected = project_reduce_scatter(
            gated,
            weights["output_projection_repacked"],
            mesh,
            tables,
            collective_id=collective_id,
        ).reshape(32, local_tokens, 7168)

        # Masked residual add inside a shard_map: an auto-sharded iota over the
        # rank axis needs the partitioner's partition-id, which runtimes that
        # only accept manual SPMD programs (torch_tpu's jax_op) reject.
        def residual(
            prefix: _Dynamic, projected: _Dynamic, length: _Dynamic
        ) -> _Dynamic:
            """Execute residual logic."""
            rank = jax.lax.axis_index("k3_sp")
            real = (
                sp_token_index(
                    rank, jnp.arange(local_tokens), window_stripe(local_tokens)
                )
                < length[0]
            )
            return (prefix[0] + jnp.where(real[:, None], projected[0], 0))[None]

        output = jax.shard_map(
            residual,
            mesh=mesh,
            in_specs=(P("k3_sp"), P("k3_sp"), P()),
            out_specs=P("k3_sp"),
            check_vma=False,
        )(prefix, projected, length)
        return output.astype(jnp.bfloat16), latent

    return jax.jit(run)


# ============================================================================
# MoE reduce-scatter ring
# ============================================================================
#
# The ``kernels/kimi_k3/collectives.py`` ring (whose wrapper is BF16 only) with
# an FP32 or BF16 running sum; FP32 avoids rounding the partials at every hop.


def ring_reduce_scatter_fp32(
    partials: _Dynamic,
    mesh: _Dynamic,
    tables: _Dynamic,
    *,
    collective_id: _Dynamic = 43,
    dtype: _Dynamic = jnp.float32,
) -> _Dynamic:
    """Reduce [32,T,D] partials (BF16 or FP32) to [32,T/32,D] token shards;.

    the running sum is ``dtype`` (FP32, or BF16) at every hop.
    """
    if mesh.size != _CONST_32 or tuple(mesh.axis_names) != ("k3_sp",):
        _err_msg = "FP32 ring requires a flat k3_sp TP32 mesh"
        raise ValueError(_err_msg)
    if (
        partials.ndim != _CONST_3
        or partials.shape[0] != _CONST_32
        or partials.dtype not in (jnp.float32, jnp.bfloat16)
    ):
        _err_msg = "FP32 ring needs [32,T,D] BF16 or FP32 partials"
        raise ValueError(_err_msg)
    tokens, width = partials.shape[1:]
    if tokens % 32 or tokens // 32 > _CONST_256:
        _err_msg = "FP32 ring needs at most 256 local rows"
        raise ValueError(_err_msg)
    rows = tokens // 32
    regions = _regions(rows, width)
    streams, buffers = len(regions), 2
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)

    def local(part: _Dynamic) -> _Dynamic:
        """Execute local logic."""
        scratch = (
            pltpu.SemaphoreType.DMA((streams,)),
            pltpu.SemaphoreType.DMA((streams,)),
            pltpu.SemaphoreType.DMA((buffers, streams)),
            pltpu.SemaphoreType.REGULAR((streams,)),
            pltpu.SemaphoreType.DMA,
            pltpu.VMEM((rows, width), partials.dtype),  # local rows, as given
            pltpu.VMEM((rows, width), dtype),
            pltpu.VMEM((buffers, rows, width), dtype),
        )
        call = pl.pallas_call(
            functools.partial(
                _reduce_scatter_kernel,
                streams=regions,
                buffers=buffers,
            ),
            out_shape=jax.ShapeDtypeStruct((rows, width), dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                grid=(1, 32),
                in_specs=(hbm,),
                out_specs=hbm,
                scratch_shapes=scratch,
            ),
            compiler_params=pltpu.CompilerParams(
                collective_id=collective_id,
                vmem_limit_bytes=4 * rows * width * 4 + 8 * 1024**2,
            ),
            name="kimi_prefill_fp32_ring_rs",
        )
        return call(jnp.asarray(tables, jnp.int32), part[0])

    return jax.shard_map(
        local,
        mesh=mesh,
        in_specs=P("k3_sp", None, None),
        out_specs=P("k3_sp", None),
        check_vma=False,
    )(partials)


# ============================================================================
# Routed MoE (expert-major)
# ============================================================================
#
# Each EP shard runs its experts over the whole window; a ring sums the partials.


def _counting_sort_positions(keys: _Dynamic, buckets: _Dynamic) -> _Dynamic:
    """Stable pair-to-expert-major positions using the MXU rank pass."""
    size = keys.shape[0]
    block = min(512, size)
    if buckets > block:
        _err_msg = "counting-sort bucket count exceeds its MXU block"
        raise ValueError(_err_msg)
    pad = -size % block
    if pad:
        keys = jnp.pad(keys, (0, pad), constant_values=buckets - 1)
    onehot = jax.nn.one_hot(keys.reshape(-1, block), buckets, dtype=jnp.bfloat16)
    lower = jnp.concatenate(
        (
            jnp.ones((1, block), jnp.bfloat16),
            jnp.tril(jnp.ones((block, block), jnp.bfloat16), -1),
        ),
        axis=0,
    )
    both = jnp.einsum("ij,bje->bie", lower, onehot, preferred_element_type=jnp.float32)
    hist = both[:, 0, :]
    rank = both[:, 1:, :]
    counts = hist.sum(axis=0)
    base = jnp.cumsum(hist, axis=0) - hist + jnp.cumsum(counts) - counts
    positions = ((base[:, None, :] + rank) * onehot).sum(axis=2)
    return positions.reshape(-1)[:size].astype(jnp.int32), counts.astype(jnp.int32)


def expert_family_slot(family_count: _Dynamic, slot: _Dynamic) -> _Dynamic:
    """Map a global MoE slot (0..91) to its index in a resident expert bank.

    Expert weights are banked by execution group (``_EXPERT_BANK_SIZES`` in
    ``kimi_k3/prefill.py``): eight layers per regular group (slot ``3 + 8 i +
    j`` -> ``j``) and one for the final layer (-> 0). Any other count, the
    first bank (slots 0..2) or a complete 92-layer family, takes the global
    slot. Dense (non-expert) families are not banked.
    """
    if family_count == _CONST_8:
        return (slot - 3) % 8
    if family_count == 1:
        return jnp.zeros_like(slot)
    return slot


def routed_experts(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Compute one EP shard's routed contribution for all tokens.

    `latent` is [T, K] BF16. `topk_ids` and normalized probabilities are
    [T, topk]. Expert tensors are the K3 K-major packed MXFP4 layout. Every
    selected pair occupies one sorted row; nonlocal pairs sort to the end and
    the GMM skips them using dynamic group sizes. The result is [T, N] BF16.
    If expert_layer is supplied, packed weights keep a leading layer axis,
    and so may their E8M0 scales. No capacity limit or route drop is applied.
    The SparseCore moves the rows: ``ragged_gather_v2`` into expert-major
    order, ``ragged_gather_reduce`` back, weighting and summing each token's
    top-k contributions in FP32.
    """
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        latent,
        topk_ids,
        topk_probabilities,
        expert_gate_up,
        expert_gate_up_scales,
        expert_down,
        expert_down_scales,
    ) = (
        _p_next(_p_iter, _p_get("latent")),
        _p_next(_p_iter, _p_get("topk_ids")),
        _p_next(_p_iter, _p_get("topk_probabilities")),
        _p_next(_p_iter, _p_get("expert_gate_up")),
        _p_next(_p_iter, _p_get("expert_gate_up_scales")),
        _p_next(_p_iter, _p_get("expert_down")),
        _p_next(_p_iter, _p_get("expert_down_scales")),
    )
    (
        first_expert,
        expert_layer,
    ) = (
        _p_get("first_expert"),
        _p_get("expert_layer"),
    )
    if expert_layer is not None:
        expert_layer = expert_family_slot(expert_gate_up.shape[0], expert_layer)
    tokens, input_width = latent.shape
    local_experts = expert_gate_up.shape[-3]
    if topk_ids.shape != topk_probabilities.shape or topk_ids.shape[0] != tokens:
        _err_msg = "topk ids and probabilities must have shape [T, topk]"
        raise ValueError(_err_msg)
    if expert_gate_up.shape[-2] * 8 != input_width:
        _err_msg = "latent width must match the packed gate/up contraction"
        raise ValueError(_err_msg)
    if expert_gate_up.shape[-1] != 2 * expert_down.shape[-2] * 8:
        _err_msg = "gate/up and down intermediate widths disagree"
        raise ValueError(_err_msg)

    topk = topk_ids.shape[1]
    local_ids = topk_ids.astype(jnp.int32) - jnp.asarray(first_expert, jnp.int32)
    valid = (local_ids >= 0) & (local_ids < local_experts) & (topk_probabilities != 0)
    keys = jnp.where(valid, local_ids, local_experts).reshape(-1)
    token_id = jnp.broadcast_to(
        jnp.arange(tokens, dtype=jnp.int32)[:, None], (tokens, topk)
    ).reshape(-1)
    token_bits = max(1, (tokens - 1).bit_length())
    if (local_experts + 1).bit_length() + token_bits >= _CONST_31:
        _err_msg = "expert and token ids do not fit the packed sort key"
        raise ValueError(_err_msg)
    # Sorting one packed integer gives both an expert-major order and its
    # source token without gathering a second large permutation vector.
    packed = (keys << token_bits) | token_id
    order = jnp.argsort(packed, stable=False)
    sorted_tokens = token_id[order]
    inverse, all_counts = _counting_sort_positions(keys, local_experts + 1)
    counts = all_counts[:local_experts]
    x = ragged_gather_v2(
        latent,
        sorted_tokens,
        jnp.array([0], jnp.int32),
        jnp.sum(counts, dtype=jnp.int32)[None],
    )

    # The E8M0 exponent bytes go to the GMM as they are (it converts each
    # tile in VMEM): an FP32 [E, blocks, 1, N] copy pads 8x in HBM.
    gu_scale = expert_gate_up_scales.astype(jnp.uint8)
    down_scale = expert_down_scales.astype(jnp.uint8)
    # gmm_v2 splits [gate, up] and interleaves lanes for the fused SiTU
    # activation (expert_activation).
    middle = gmm_v2(
        x,
        expert_gate_up,
        counts,
        rhs_scale=gu_scale,
        rhs_layer=expert_layer,
        rhs_quant_dtype=jnp.float4_e2m1fn,
        fuse_act="situ:4:25",
    )
    down = gmm_v2(
        middle,
        expert_down,
        counts,
        rhs_scale=down_scale,
        rhs_layer=expert_layer,
        rhs_quant_dtype=jnp.float4_e2m1fn,
    )

    # Restore the token-major pair order, mask the unwritten nonlocal GMM rows,
    # and reduce the top-k contributions per token in FP32.
    return ragged_gather_reduce(
        down,
        inverse,
        topk_probabilities.reshape(-1),
        valid.reshape(-1),
        reduce_group_size=topk,
    )


def _moe_latent_all_gather(latent: _Dynamic, tables: _Dynamic) -> _Dynamic:
    """Gather BF16 SP latents in mesh-rank order using the TP32 ring.

    The gather's owners are mesh ranks, independent of the routed-output
    ring's token ownership permutation.
    """
    rows, width = latent.shape
    regions = _regions(rows, width)
    streams, buffers = len(regions), 2
    return pl.pallas_call(
        functools.partial(_all_gather_kernel, regions=regions, buffers=buffers),
        out_shape=jax.ShapeDtypeStruct((32 * rows, width), latent.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(1, 32),
            in_specs=(pl.BlockSpec(memory_space=pltpu.HBM),),
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=(
                pltpu.SemaphoreType.DMA((streams,)),
                pltpu.SemaphoreType.DMA((streams,)),
                pltpu.SemaphoreType.DMA((buffers, streams)),
                pltpu.SemaphoreType.REGULAR((streams,)),
                pltpu.SemaphoreType.DMA((streams,)),
                pltpu.VMEM((rows, width), latent.dtype),
                pltpu.VMEM((buffers, rows, width), latent.dtype),
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=41,
            vmem_limit_bytes=3 * rows * width * 2 + 8 * 1024**2,
        ),
        name="kimi_prefill_moe_latent_ag",
    )(jnp.asarray(tables, jnp.int32), latent)


def moe_local_routed(
    hidden_local: _Dynamic,
    length: _Dynamic,
    weights: _Dynamic,
    *,
    axis_name: _Dynamic,
    gather_tables: _Dynamic,
) -> _Dynamic:
    """This rank's routed-expert partial for every token of the window, and.

    its FP32 half of the TP2 shared expert over its chip's rows (reduced
    across the chip pair in the tail).
    Routing is sequence-parallel (each rank routes its own ``T/32`` rows);
    the latent, ids and probabilities are then gathered so that each EP
    shard runs its experts over all tokens (``routed_experts``). The caller's
    reduce-scatter ring sums the partials. ``latent_down`` is [7168,4096] on
    every rank; 56 local experts (EP16/TP2).
    """
    rank = jax.lax.axis_index(axis_name)
    local_tokens, hidden_width = hidden_local.shape
    if hidden_width != _CONST_7168:
        _err_msg = "K3 sequence-parallel MoE expects [T/32,7168]"
        raise ValueError(_err_msg)
    if weights["latent_down"].shape[-2:] != (7168, 4096):
        _err_msg = "latent projection weights must use replicated SP layout"
        raise ValueError(_err_msg)
    local_experts = weights["expert_gu"].shape[-3]
    if local_experts != _CONST_56:
        _err_msg = "K3 SP MoE requires EP16/TP2 expert weights"
        raise ValueError(_err_msg)
    expert_tp = 32 * local_experts // 896
    tokens = local_tokens * 32

    logits = jax.lax.dot_general(
        hidden_local,
        weights["router"],
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    raw = jax.nn.sigmoid(logits)
    _, local_ids = jax.lax.top_k(raw + weights["router_bias"], 16)
    local_probs = jnp.take_along_axis(raw, local_ids, axis=1)
    local_probs /= jnp.maximum(local_probs.sum(axis=1, keepdims=True), 1e-30)
    valid_rows = sp_token_index(
        rank, jnp.arange(local_tokens), window_stripe(local_tokens)
    ) < length.reshape(())
    local_probs = jnp.where(valid_rows[:, None], local_probs, 0)

    if (
        weights["latent_down"].ndim == _CONST_3
    ):  # the resident family: select the layer in the kernel
        latent_local = gmm_v2(
            hidden_local,
            weights["latent_down"][:, None],
            jnp.full((1,), local_tokens, jnp.int32),
            rhs_layer=weights["expert_layer"],
        )
    else:
        latent_local = hidden_local @ weights["latent_down"]
    latent_all = _moe_latent_all_gather(latent_local[:, :3584], gather_tables)
    # One all-gather for ids and probability bits.
    routes = jnp.concatenate(
        (
            local_ids.astype(jnp.int32),
            jax.lax.bitcast_convert_type(local_probs, jnp.int32),
        ),
        axis=1,
    )
    routes = jax.lax.all_gather(routes, axis_name).reshape(tokens, 32)
    ids_all = routes[:, :16]
    probs_all = jax.lax.bitcast_convert_type(routes[:, 16:], jnp.float32)
    routed = routed_experts(
        latent_all,
        ids_all,
        probs_all,
        weights["expert_gu"],
        weights["expert_gus"],
        weights["expert_down"],
        weights["expert_ds"],
        first_expert=(rank // expert_tp) * local_experts,
        expert_layer=weights.get("expert_layer"),
    )
    shared = _moe_sp_shared_partial(
        _moe_sp_shared_rows(hidden_local, axis_name), weights
    )
    return routed, shared


SHARED_PAIRS = tuple((2 * i, 2 * i + 1) for i in range(16))  # the chip-local TP2 groups


def _moe_sp_shared_rows(hidden_local: _Dynamic, axis_name: _Dynamic) -> _Dynamic:
    """Both cores' token rows of this chip, for the TP2 shared expert."""
    local_tokens = hidden_local.shape[0]
    return jax.lax.all_gather(
        hidden_local, axis_name, axis_index_groups=SHARED_PAIRS
    ).reshape(2 * local_tokens, 7168)


def _moe_sp_shared_partial(chip_rows: _Dynamic, weights: _Dynamic) -> _Dynamic:
    """This core's half of the shared expert over the chip's rows: FP32.

    partials for _moe_sp_shared_reduce. Pure matmuls (no collectives), so
    it can overlap SparseCore work.
    """
    # One-group GMMs: the shared expert's M (the chip's rows) is small, and
    # the GMM keeps each weight tile on the MXU for all of it, with the
    # activation fused on the FP32 accumulator.
    rows = jnp.full((1,), chip_rows.shape[0], jnp.int32)
    # Resident layer families ([L, ...]): the kernel selects the layer (a
    # sliced operand would be copied).
    family = weights["shared_gu"].ndim == _CONST_3
    layer = {"rhs_layer": weights["expert_layer"]} if family else {}

    def one_group(w: _Dynamic) -> _Dynamic:
        """Execute one group logic."""
        return w[:, None] if family else w[None]

    shared_mid = gmm_v2(
        chip_rows, one_group(weights["shared_gu"]), rows, fuse_act="situ:4:25", **layer
    )
    return gmm_v2(
        shared_mid,
        one_group(weights["shared_down"]),
        rows,
        preferred_element_type=jnp.float32,
        **layer,
    )


def _moe_sp_shared_reduce(partial: _Dynamic, axis_name: _Dynamic) -> _Dynamic:
    """Execute moe sp shared reduce logic."""
    return jax.lax.psum_scatter(
        partial,
        axis_name,
        scatter_dimension=0,
        tiled=True,
        axis_index_groups=SHARED_PAIRS,
    ).astype(jnp.bfloat16)


def _moe_sp_tail(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Finish the routed branch from local reduced MoE rows and add the.

    shared expert's rows.
    """
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        routed,
        shared,
        prefix_local,
        length,
        weights,
    ) = (
        _p_next(_p_iter, _p_get("routed")),
        _p_next(_p_iter, _p_get("shared")),
        _p_next(_p_iter, _p_get("prefix_local")),
        _p_next(_p_iter, _p_get("length")),
        _p_next(_p_iter, _p_get("weights")),
    )
    (
        axis_name,
        eps,
        shared_reduced,
    ) = (
        _p_get("axis_name"),
        _p_get("eps"),
        _p_kw.get("shared_reduced", False),
    )
    rank = jax.lax.axis_index(axis_name)
    local_tokens = prefix_local.shape[0]
    valid_rows = sp_token_index(
        rank, jnp.arange(local_tokens), window_stripe(local_tokens)
    ) < length.reshape(())
    routed = routed.astype(jnp.float32)
    routed *= jax.lax.rsqrt(jnp.mean(routed * routed, axis=1, keepdims=True) + eps)
    routed = (routed * weights["latent_norm"][:, :3584]).astype(jnp.bfloat16)
    up = routed @ weights["latent_up"]
    if not shared_reduced:
        shared = _moe_sp_shared_reduce(shared, axis_name)
    output = (up + shared).astype(jnp.bfloat16)
    output = jnp.where(valid_rows[:, None], output, 0)
    return (prefix_local + output).astype(jnp.bfloat16)


def make_moe_fused_sp_repacked(
    mesh: _Dynamic,
    *,
    seq_len: _Dynamic,
    eps: _Dynamic = 1e-5,
    collective_id: _Dynamic = 42,
    token_ranks: _Dynamic = None,
) -> _Dynamic:
    """Build the TP32 MoE layer for repacked projection weights: the local.

    routed partials, their reduce-scatter ring, then the latent-up and
    shared-expert tail on each rank's rows.
    """
    if tuple(mesh.axis_names) != ("k3_sp",):
        _err_msg = "MoE Pallas ring reduction needs k3_sp"
        raise ValueError(_err_msg)
    check_window(seq_len, "MoE Pallas ring reduction")
    axis_name = mesh.axis_names[0]
    if token_ranks is None:
        token_ranks = tuple(range(32))
    tables = ring_tables(list(mesh.devices.flat), token_ranks)
    gather_tables = ring_tables(list(mesh.devices.flat), tuple(range(32)))
    local_tokens = seq_len // 32

    def local_partial(
        hidden: _Dynamic, _prefix: _Dynamic, weights: _Dynamic, length: _Dynamic
    ) -> _Dynamic:
        """Execute local partial logic."""
        local_weights = jax.tree.map(lambda value: value[0], weights)
        routed, shared = moe_local_routed(
            hidden[0],
            length,
            local_weights,
            axis_name=axis_name,
            gather_tables=gather_tables,
        )
        return routed[None], shared[None]

    partial_op = jax.jit(
        jax.shard_map(
            local_partial,
            mesh=mesh,
            in_specs=(P(axis_name), P(axis_name), P(axis_name), P()),
            out_specs=(P(axis_name), P(axis_name)),
            check_vma=False,
        )
    )

    def local_tail(
        reduced: _Dynamic,
        shared: _Dynamic,
        prefix: _Dynamic,
        weights: _Dynamic,
        length: _Dynamic,
    ) -> _Dynamic:
        """Execute local tail logic."""
        local_weights = jax.tree.map(lambda value: value[0], weights)
        return _moe_sp_tail(
            reduced[0],
            shared[0],
            prefix[0],
            length,
            local_weights,
            axis_name=axis_name,
            eps=eps,
        )[None]

    tail_op = jax.jit(
        jax.shard_map(
            local_tail,
            mesh=mesh,
            in_specs=(P(axis_name), P(axis_name), P(axis_name), P(axis_name), P()),
            out_specs=P(axis_name),
            check_vma=False,
        )
    )

    def run_ring(
        hidden: _Dynamic, prefix: _Dynamic, weights: _Dynamic, length: _Dynamic
    ) -> _Dynamic:
        """Execute run ring logic."""
        partials, shared = partial_op(hidden, prefix, weights, length)
        reduced = ring_reduce_scatter_fp32(
            partials,
            mesh,
            tables,
            collective_id=collective_id,
            dtype=jnp.dtype(_config().moe_reduce_dtype),
        )
        reduced = reduced.reshape(32, local_tokens, 3584)
        return tail_op(reduced, shared, prefix, weights, length)

    return jax.jit(run_ring)


# ============================================================================
# MLA weight families
# ============================================================================
#
# Each family holds every MLA layer's copy; a runtime slot picks the layer.


MLA_FAMILIES = (
    "m_qa",
    "m_qb",
    "m_ka",
    "m_kb",
    "m_gate",
    "m_qnorm",
    "m_knorm",
    "m_o",
)
MLA_FIELD_NAMES = (
    "query_down_projection",
    "query_up_projection",
    "key_value_down_projection",
    "key_value_up_projection",
    "output_gate_projection",
    "query_normalization",
    "key_value_normalization",
    "output_projection_repacked",
)


def make_indexed_mla_layer_sp_ring(
    mesh: _Dynamic,
    *,
    seq_len: _Dynamic = 8192,
    segments: _Dynamic,
    prefix_capacity: _Dynamic,
    flash_tile: _Dynamic = 256,
) -> _Dynamic:
    """The MLA layer at MLA slot ``slot`` of the family-major weights.

    ``run(hidden, prefix, families, segment_desc, length, slot, arena)``
    (``arena``: the ``CpPrefix`` of the slot's pool).
    """
    base = make_mla_attention_layer_sp_ring(
        mesh,
        seq_len=seq_len,
        flash_tile=flash_tile,
        segments=segments,
        prefix_capacity=prefix_capacity,
    )

    def select(families: _Dynamic, slot: _Dynamic) -> _Dynamic:
        """Execute select logic."""
        return {
            field: jax.lax.dynamic_index_in_dim(
                families[name],
                slot[0],
                axis=1,
                keepdims=False,
            )
            for name, field in zip(MLA_FAMILIES, MLA_FIELD_NAMES)
        }

    def run(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
        """Execute run logic."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            hidden,
            prefix,
            families,
            segment_desc,
            length,
            slot,
            arena,
        ) = (
            _p_next(_p_iter, _p_get("hidden")),
            _p_next(_p_iter, _p_get("prefix")),
            _p_next(_p_iter, _p_get("families")),
            _p_next(_p_iter, _p_get("segment_desc")),
            _p_next(_p_iter, _p_get("length")),
            _p_next(_p_iter, _p_get("slot")),
            _p_next(_p_iter, _p_get("arena")),
        )
        return base(hidden, prefix, select(families, slot), segment_desc, length, arena)

    return jax.jit(run)


# ============================================================================
# MoE weight families
# ============================================================================
#
# Each family holds every MoE layer's copy; a runtime slot picks the layer.


MOE_FAMILIES = (
    "router",
    "router_bias",
    "latent_down",
    "latent_up",
    "latent_norm",
    "shared_gu",
    "shared_down",
    "expert_gu",
    "expert_gus",
    "expert_down",
    "expert_ds",
)


def make_indexed_moe_sp_ring(mesh: _Dynamic, *, seq_len: _Dynamic = 8192) -> _Dynamic:
    """Select a MoE layer, keeping packed expert families resident for tile DMA."""
    if seq_len in (512, 1024, 2048, 4096, 8192):
        return fused.make_indexed_moe_sp_ring(mesh, seq_len=seq_len)
    return _make_indexed_moe_sp_ring_reference(mesh, seq_len=seq_len)


def _make_indexed_moe_sp_ring_reference(
    mesh: _Dynamic, *, seq_len: _Dynamic = 8192
) -> _Dynamic:
    """SparseCore gather/combine path, for the windows the fused and packed.

    MoEs do not cover (BF16 reduction below 4K).
    """
    base = make_moe_fused_sp_repacked(mesh, seq_len=seq_len)

    def run(
        hidden: _Dynamic,
        prefix: _Dynamic,
        families: _Dynamic,
        length: _Dynamic,
        slot: _Dynamic,
    ) -> _Dynamic:
        """Execute run logic."""
        selected = {
            name: (
                families[name]
                if name
                in (
                    "expert_gu",
                    "expert_down",
                    "expert_gus",
                    "expert_ds",
                    "latent_down",
                    "shared_gu",
                    "shared_down",
                )
                else jax.lax.dynamic_index_in_dim(
                    families[name],
                    slot[0],
                    axis=1,
                    keepdims=False,
                )
            )
            for name in MOE_FAMILIES
        }
        selected["expert_layer"] = jnp.broadcast_to(slot[None], (mesh.size, 1))
        return base(hidden, prefix, selected, length)

    return jax.jit(run)


# ============================================================================
# Norm + all-gather + KDA
# ============================================================================
#
# Mixture, RMSNorm and window all-gather in one Pallas call, optionally with KDA.


def _norm_gather_body(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Execute norm gather body logic."""
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        active_ref,
        tables_ref,
        prefix_ref,
        blocks_ref,
        folded_ref,
        norm_ref,
        normalized_ref,
        gathered_ref,
    ) = (
        _p_next(_p_iter, _p_get("active_ref")),
        _p_next(_p_iter, _p_get("tables_ref")),
        _p_next(_p_iter, _p_get("prefix_ref")),
        _p_next(_p_iter, _p_get("blocks_ref")),
        _p_next(_p_iter, _p_get("folded_ref")),
        _p_next(_p_iter, _p_get("norm_ref")),
        _p_next(_p_iter, _p_get("normalized_ref")),
        _p_next(_p_iter, _p_get("gathered_ref")),
    )
    scratch = tuple(_p_iter)
    (
        regions,
        eps,
        mix_tile,
    ) = (
        _p_get("regions"),
        _p_get("eps"),
        _p_get("mix_tile"),
    )
    mix_scratch = scratch[:12]
    ring_scratch = scratch[12:]

    @pl.when(pl.program_id(1) == 0)
    def normalize() -> _Dynamic:
        """Execute normalize logic."""
        mixture_norm_tiled_body(
            active_ref,
            prefix_ref,
            blocks_ref,
            folded_ref,
            norm_ref,
            normalized_ref,
            *mix_scratch,
            tile=mix_tile,
            eps=eps,
        )

    _all_gather_kernel(
        tables_ref,
        normalized_ref,
        gathered_ref,
        *ring_scratch,
        regions=regions,
        buffers=2,
        striped=True,
    )


def make_norm_gather_sp(
    mesh: _Dynamic,
    *,
    seq_len: _Dynamic,
    token_ranks: _Dynamic = None,
    collective_id: _Dynamic = 37,
    eps: _Dynamic = 1e-5,
) -> _Dynamic:
    """Return ``(normalized [32,T/32,H], gathered [T,H])``.

    The first result is token-sharded, and the second is replicated. The
    caller supplies rank-leading prefix, snapshots, and norm weights.
    """
    if mesh.size != _CONST_32 or tuple(mesh.axis_names) != ("k3_sp",):
        _err_msg = "norm/gather requires a flat k3_sp TP32 mesh"
        raise ValueError(_err_msg)
    check_window(seq_len, "fused norm/gather")
    rows, width = seq_len // 32, 7168
    if token_ranks is None:
        token_ranks = tuple(range(32))
    tables = jnp.asarray(
        ring_tables(list(mesh.devices.flat), token_ranks),
        jnp.int32,
    )
    regions = _regions(rows, width)
    streams = len(regions)
    mix_scratch = mixture_norm_tiled_scratch_shapes(width, rows)
    ring_scratch = (
        pltpu.SemaphoreType.DMA((streams,)),
        pltpu.SemaphoreType.DMA((streams,)),
        pltpu.SemaphoreType.DMA((2, streams)),
        pltpu.SemaphoreType.REGULAR((streams,)),
        pltpu.SemaphoreType.DMA((streams,)),
        pltpu.VMEM((rows, width), jnp.bfloat16),
        pltpu.VMEM((2, rows, width), jnp.bfloat16),
    )
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    call = pl.pallas_call(
        functools.partial(_norm_gather_body, regions=regions, eps=eps, mix_tile=rows),
        out_shape=(
            jax.ShapeDtypeStruct((rows, width), jnp.bfloat16),
            jax.ShapeDtypeStruct((seq_len, width), jnp.bfloat16),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(1, 32),
            in_specs=(hbm, hbm, hbm, hbm),
            out_specs=(hbm, hbm),
            scratch_shapes=(*mix_scratch, *ring_scratch),
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=collective_id,
            vmem_limit_bytes=64 * 1024**2,
        ),
        name="kimi_prefill_norm_gather_sp",
    )

    def local(
        prefix: _Dynamic,
        blocks: _Dynamic,
        folded: _Dynamic,
        norm: _Dynamic,
        active: _Dynamic,
    ) -> _Dynamic:
        """Execute local logic."""
        normalized, gathered = call(
            active,
            tables,
            prefix[0],
            blocks[0],
            folded[0],
            norm[0],
        )
        return normalized[None], gathered

    return jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(P("k3_sp"), P("k3_sp"), P("k3_sp"), P("k3_sp"), P()),
            out_specs=(P("k3_sp"), P()),
            check_vma=False,
        )
    )


def _kda_layer_body(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """One KDA layer of the family-major weights (``slot_ref``) over a.

    packed window of ``count`` sequences, pipelined over its ``blocks`` SP
    blocks on half rings: grid step ``17 i + j`` runs ring step j of block
    i's window all-gather, a share of block ``i - 1``'s KDA tiles (with
    their output-projection partials) and ring step j of block ``i - 2``'s
    reduce-scatter, so both rings run under the delta rule. Steps b < blocks
    first apply the mixture and RMSNorm to block b's local rows.
    ``segment_ref`` is the SMEM ``[3 * count]`` descriptor (see
    ``kda_prefill_tiled_body``). KDA slot ``9k`` is layer ``12k``, the first
    layer of a residual block, whose attention output replaces the partial
    sum instead of adding to it.
    """
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        slot_ref,
        segment_ref,
        active_ref,
        tables_ref,
        length_ref,
        prefix_ref,
        blocks_ref,
        folded_ref,
        norm_ref,
        gate_family_ref,
        fb_family_ref,
        proj_family_ref,
        conv_family_ref,
        alog_family_ref,
        dt_family_ref,
        kda_norm_family_ref,
        rec_in_ref,
        hist_in_ref,
        output_family_ref,
        normalized_ref,
        gathered_ref,
        attended_ref,
        rec_out_ref,
        hist_out_ref,
        final_ref,
        rec_checkpoint_ref,
        hist_checkpoint_ref,
        partial_ref,
        pool,
    ) = (
        _p_next(_p_iter, _p_get("slot_ref")),
        _p_next(_p_iter, _p_get("segment_ref")),
        _p_next(_p_iter, _p_get("active_ref")),
        _p_next(_p_iter, _p_get("tables_ref")),
        _p_next(_p_iter, _p_get("length_ref")),
        _p_next(_p_iter, _p_get("prefix_ref")),
        _p_next(_p_iter, _p_get("blocks_ref")),
        _p_next(_p_iter, _p_get("folded_ref")),
        _p_next(_p_iter, _p_get("norm_ref")),
        _p_next(_p_iter, _p_get("gate_family_ref")),
        _p_next(_p_iter, _p_get("fb_family_ref")),
        _p_next(_p_iter, _p_get("proj_family_ref")),
        _p_next(_p_iter, _p_get("conv_family_ref")),
        _p_next(_p_iter, _p_get("alog_family_ref")),
        _p_next(_p_iter, _p_get("dt_family_ref")),
        _p_next(_p_iter, _p_get("kda_norm_family_ref")),
        _p_next(_p_iter, _p_get("rec_in_ref")),
        _p_next(_p_iter, _p_get("hist_in_ref")),
        _p_next(_p_iter, _p_get("output_family_ref")),
        _p_next(_p_iter, _p_get("normalized_ref")),
        _p_next(_p_iter, _p_get("gathered_ref")),
        _p_next(_p_iter, _p_get("attended_ref")),
        _p_next(_p_iter, _p_get("rec_out_ref")),
        _p_next(_p_iter, _p_get("hist_out_ref")),
        _p_next(_p_iter, _p_get("final_ref")),
        _p_next(_p_iter, _p_get("rec_checkpoint_ref")),
        _p_next(_p_iter, _p_get("hist_checkpoint_ref")),
        _p_next(_p_iter, _p_get("partial_ref")),
        _p_next(_p_iter, _p_get("pool")),
    )
    scratch = tuple(_p_iter)
    (
        eps,
        count,
        blocks,
    ) = (
        _p_get("eps"),
        _p_get("count"),
        _p_get("blocks"),
    )
    slot = slot_ref[0]
    step = pl.program_id(1)
    phase, ring_step = step // HALF_STEPS, step % HALF_STEPS
    stripe = window_stripe(prefix_ref.shape[0])
    mix_small, gather_sems = scratch[:8], scratch[8:11]
    reduce_sems, projection_scratch = scratch[11:16], scratch[16:19]
    kda_scratch = scratch[19:]

    def views(
        shapes: _Dynamic,
    ) -> _Dynamic:  # the shared pool: the mixture at step 0, then the projection ring
        """Execute views logic."""
        offset, result = 0, []
        for shape, dtype in shapes:
            size = pool_alias.storage_size(shape, jnp.dtype(dtype))
            result.append(
                pool_alias.view(
                    pool.at[pl.ds(offset // 128, size // 128), :], shape, dtype
                )
            )
            offset += size
        return result

    pl.when(step == 0)(lambda: _ring_enter(tables_ref, jax.lax.axis_index("k3_sp")))

    @pl.when(step < blocks)  # only block 0's mixture delays the first ring
    def normalize() -> _Dynamic:
        """Execute normalize logic."""
        rows = pl.ds(step * stripe, stripe)
        mixture_norm_tiled_body(
            active_ref,
            prefix_ref.at[rows],
            blocks_ref.at[:, rows],
            folded_ref,
            norm_ref,
            normalized_ref.at[rows],
            *views(_kda_mix_shapes(stripe)),
            *mix_small,
            tile=stripe,
            eps=eps,
        )

    @pl.when(phase < blocks)
    def gather() -> _Dynamic:
        """Execute gather logic."""
        _half_gather_step(
            tables_ref,
            normalized_ref,
            gathered_ref,
            *gather_sems,
            step=ring_step,
            block=phase,
        )

    @pl.when(phase >= _CONST_2)
    def reduce() -> _Dynamic:
        """Execute reduce logic."""
        _half_reduce_step(
            tables_ref,
            length_ref,
            partial_ref,
            prefix_ref,
            final_ref,
            *reduce_sems,
            *views(_kda_reduce_shapes(stripe)),
            step=ring_step,
            block=phase - 2,
            replace=slot % 9 == 0,
        )

    # One call site for the KDA tile: inlined twice, the layer would no
    # longer fit the instruction memory.
    block_tiles = 32 * stripe // 128
    attending = (phase >= 1) & (phase <= blocks)
    first_tile = jnp.where(
        attending, (phase - 1) * block_tiles + ring_step * block_tiles // HALF_STEPS, 0
    )
    end_tile = jnp.where(
        attending,
        (phase - 1) * block_tiles + (ring_step + 1) * block_tiles // HALF_STEPS,
        0,
    )
    kda_prefill_tiled_body(
        gathered_ref,
        gate_family_ref.at[slot],
        fb_family_ref.at[slot],
        proj_family_ref.at[slot],
        conv_family_ref.at[slot],
        alog_family_ref.at[slot],
        dt_family_ref.at[slot],
        kda_norm_family_ref.at[slot],
        rec_in_ref,
        hist_in_ref,
        attended_ref,
        rec_out_ref,
        hist_out_ref,
        *kda_scratch,
        tile=128,
        decay_lower_bound=-5.0,
        epsilon=eps,
        checkpoint=(rec_checkpoint_ref, hist_checkpoint_ref),
        segments=(segment_ref, count),
        tiles=(
            first_tile,
            end_tile,
            step == HALF_STEPS,
            step == HALF_STEPS * (blocks + 1) - 1,
        ),
        block_tiles=block_tiles,
        projection=(
            output_family_ref.at[slot],
            *projection_scratch[:1],
            partial_ref,
            *projection_scratch[1:],
        ),
    )


def _kda_mix_shapes(mix_tile: _Dynamic) -> _Dynamic:
    """Execute kda mix shapes logic."""
    return (
        ((2, mix_tile, 7168), jnp.bfloat16),
        ((mix_tile, 7168), jnp.bfloat16),
        ((2, mix_tile, 7168), jnp.bfloat16),
        ((mix_tile, 7168), jnp.float32),
    )


def _kda_reduce_shapes(stripe: _Dynamic) -> _Dynamic:
    """Execute kda reduce shapes logic."""
    return (
        ((2, 2, stripe, 7168), jnp.bfloat16),  # the local partials (slot, direction)
        ((2, stripe, 7168), jnp.bfloat16),  # payload (direction)
        ((2, 2, stripe, 7168), jnp.bfloat16),  # received (slot, direction)
    )


def make_norm_gather_kda_sp(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """One KDA layer as one Pallas call (``_kda_layer_body``): the local.

    residual mixture and RMSNorm, then per SP block the 32-rank all-gather
    of the window, KDA over the packed window (``segments`` sequences, their
    states leading with the sequence: ``[1, segments, ...]`` per rank) and
    the output projection ring with the residual add, pipelined so that the
    rings run under the delta rule.
    ``run(prefix, blocks, folded, norm, families, rec, hist, active, length,
    slot, segment_desc)`` returns ``(normalized, gathered, attended, rec,
    hist, output, rec_at, hist_at)``: the ``*_at`` states are each
    sequence's at its checkpoint position.
    """
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    mesh = _p_next(_p_iter, _p_get("mesh"))
    (
        seq_len,
        segments,
        token_ranks,
        collective_id,
        eps,
    ) = (
        _p_get("seq_len"),
        _p_get("segments"),
        _p_get("token_ranks"),
        _p_get("collective_id", 38),
        _p_get("eps", 1e-05),
    )
    if mesh.size != _CONST_32 or tuple(mesh.axis_names) != ("k3_sp",):
        _err_msg = "norm/gather/KDA requires a flat k3_sp TP32 mesh"
        raise ValueError(_err_msg)
    check_window(seq_len, "fused norm/gather/KDA")
    if not segments:
        _err_msg = "the KDA layer runs packed windows"
        raise ValueError(_err_msg)
    state_lead = (segments,)
    rows, width = seq_len // 32, 7168
    stripe = window_stripe(rows)
    blocks = rows // stripe
    if token_ranks is None:
        token_ranks = tuple(range(32))
    tables = jnp.asarray(
        ring_tables(list(mesh.devices.flat), token_ranks),
        jnp.int32,
    )
    streams = len(_half_ring_streams(width))
    pool_bytes = max(
        sum(
            pool_alias.storage_size(shape, jnp.dtype(dtype))
            for shape, dtype in _kda_mix_shapes(stripe)
        ),
        sum(
            pool_alias.storage_size(shape, jnp.dtype(dtype))
            for shape, dtype in _kda_reduce_shapes(stripe)
        ),
    )
    shapes = (
        jax.ShapeDtypeStruct((width, 128), jnp.bfloat16),
        jax.ShapeDtypeStruct((128, 384), jnp.bfloat16),
        jax.ShapeDtypeStruct((width, 1664), jnp.bfloat16),
        jax.ShapeDtypeStruct((4, 3, 3, 128), jnp.float32),
        jax.ShapeDtypeStruct((3, 1), jnp.float32),
        jax.ShapeDtypeStruct((3, 128), jnp.float32),
        jax.ShapeDtypeStruct((8, 128), jnp.float32),
    )
    scratch = (
        pltpu.VMEM((pool_bytes // 128, 128), jnp.uint8),
        *mixture_norm_tiled_scratch_shapes(width, stripe)[4:],
        pltpu.SemaphoreType.DMA((streams,)),  # the window all-gather
        pltpu.SemaphoreType.DMA((streams,)),
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA((2, streams)),  # the reduce-scatter
        pltpu.SemaphoreType.DMA((streams,)),
        pltpu.SemaphoreType.DMA((2, streams)),
        pltpu.SemaphoreType.REGULAR((streams,)),
        pltpu.SemaphoreType.DMA,
        pltpu.VMEM((384, width), jnp.bfloat16),  # the output projection
        pltpu.VMEM((2, 128, width), jnp.bfloat16),  # its partials
        pltpu.SemaphoreType.DMA((2,)),
        *kda_prefill_tiled_scratch_shapes(width, 128, *shapes),
    )
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    output_shapes = (
        jax.ShapeDtypeStruct((rows, width), jnp.bfloat16),
        jax.ShapeDtypeStruct((seq_len, width), jnp.bfloat16),
        jax.ShapeDtypeStruct((seq_len, 384), jnp.bfloat16),
        jax.ShapeDtypeStruct((*state_lead, 3, 128, 128), jnp.float32),
        jax.ShapeDtypeStruct((*state_lead, 3, 3, 3, 128), jnp.bfloat16),
        jax.ShapeDtypeStruct((rows, width), jnp.bfloat16),
        jax.ShapeDtypeStruct((*state_lead, 3, 128, 128), jnp.float32),
        jax.ShapeDtypeStruct((*state_lead, 3, 3, 3, 128), jnp.bfloat16),
        jax.ShapeDtypeStruct((seq_len, width), jnp.bfloat16),  # projection partials
    )
    call = pl.pallas_call(
        functools.partial(
            _kda_layer_body,
            count=segments,
            eps=eps,
            blocks=blocks,
        ),
        out_shape=output_shapes,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=5,
            grid=(1, HALF_STEPS * (blocks + 2)),
            in_specs=(hbm,) * 14,
            out_specs=(hbm,) * len(output_shapes),
            scratch_shapes=scratch,
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=collective_id,
            vmem_limit_bytes=64 * 1024**2,
        ),
        name="kimi_prefill_norm_gather_kda_sp",
    )

    def local(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
        """Execute local logic."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            prefix,
            blocks,
            folded,
            norm,
            weights,
            rec,
            hist,
            active,
            length,
            slot,
            segment_desc,
        ) = (
            _p_next(_p_iter, _p_get("prefix")),
            _p_next(_p_iter, _p_get("blocks")),
            _p_next(_p_iter, _p_get("folded")),
            _p_next(_p_iter, _p_get("norm")),
            _p_next(_p_iter, _p_get("weights")),
            _p_next(_p_iter, _p_get("rec")),
            _p_next(_p_iter, _p_get("hist")),
            _p_next(_p_iter, _p_get("active")),
            _p_next(_p_iter, _p_get("length")),
            _p_next(_p_iter, _p_get("slot")),
            _p_next(_p_iter, _p_get("segment_desc")),
        )
        w = jax.tree.map(lambda value: value[0], weights)
        outputs = call(
            slot,
            segment_desc,
            active,
            tables,
            length,
            prefix[0],
            blocks[0],
            folded[0],
            norm[0],
            w["k_gate"],
            w["k_fb"],
            w["k_projection"],
            w["k_conv"],
            w["k_a_log"],
            w["k_dt"],
            w["k_norm"],
            rec[0],
            hist[0],
            w["k_o"],
        )
        normalized, gathered, attended, new_rec, new_hist, final, rec_at, hist_at, _ = (
            outputs
        )
        return (
            normalized[None],
            gathered,
            attended[None],
            new_rec[None],
            new_hist[None],
            final[None],
            rec_at[None],
            hist_at[None],
        )

    return jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(P("k3_sp"),) * 7 + (P(), P(), P(), P()),
            out_specs=(
                P("k3_sp"),
                P(),
                P("k3_sp"),
                P("k3_sp"),
                P("k3_sp"),
                P("k3_sp"),
                P("k3_sp"),
                P("k3_sp"),
            ),
            check_vma=False,
        )
    )


# ============================================================================
# Dense FFN layer
# ============================================================================
#
# Layer 0's dense FFN, TP32 over the intermediate dimension.


def make_dense_ffn_sp(mesh: _Dynamic, *, seq_len: _Dynamic) -> _Dynamic:
    """Layer-0 dense FFN over local rows with a TP32 output reduction."""
    if mesh.size != _CONST_32 or len(mesh.axis_names) != 1 or seq_len % 32:
        _err_msg = "dense FFN SP requires flat TP32 and T divisible by 32"
        raise ValueError(_err_msg)
    axis_name = mesh.axis_names[0]
    local_tokens = seq_len // 32

    def local(
        hidden: _Dynamic,
        prefix: _Dynamic,
        gate_up: _Dynamic,
        down: _Dynamic,
        length: _Dynamic,
    ) -> _Dynamic:
        """Execute local logic."""
        x = hidden[0]
        if (
            x.shape != (local_tokens, 7168)
            or gate_up[0].shape != (7168, 2304)
            or down[0].shape != (1152, 7168)
        ):
            _err_msg = "dense FFN weights must use K3 TP32 shapes"
            raise ValueError(_err_msg)
        # Rows are sequence-parallel (``sp_token_index`` stripes) while the
        # weights are TP32 over the intermediate dimension, so every rank
        # applies its shard to every rank's rows (gathered in rank order) and
        # the partial outputs are reduce-scattered back to their owners.
        x = jax.lax.all_gather(x, axis_name, tiled=True)
        gu = jax.lax.dot_general(
            x,
            gate_up[0],
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        ).astype(jnp.bfloat16)
        gate, value = jnp.split(gu, 2, axis=1)
        gate, value = gate.astype(jnp.float32), value.astype(jnp.float32)
        middle = expert_activation(gate, value).astype(jnp.bfloat16)
        partial = jax.lax.dot_general(
            middle,
            down[0],
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        reduced = jax.lax.psum_scatter(
            partial,
            axis_name,
            scatter_dimension=0,
            tiled=True,
        ).astype(jnp.bfloat16)
        rank = jax.lax.axis_index(axis_name)
        valid = (
            sp_token_index(rank, jnp.arange(local_tokens), window_stripe(local_tokens))
            < length[0]
        )
        return (prefix[0] + jnp.where(valid[:, None], reduced, 0))[None]

    return jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(P(axis_name), P(axis_name), P(axis_name), P(axis_name), P()),
            out_specs=P(axis_name),
            check_vma=False,
        )
    )


# ============================================================================
# Four-layer groups
# ============================================================================
#
# One straight-line group program reused through a runtime group index:
# SparseCore row movement does not lower inside a dynamic JAX loop.


class _RegularGroup4Runner:
    """Runner for one runtime-indexed group at layers 4g..4g+3."""

    def __getattr__(self, name: _Dynamic) -> _Dynamic:
        """Dynamic attribute access."""
        raise AttributeError(name)

    def __init__(
        self,
        mesh: _Dynamic,
        *,
        seq_len: _Dynamic = 8192,
        segments: _Dynamic,
        prefix_capacity: _Dynamic,
        mla_flash_tile: _Dynamic = 256,
    ) -> None:
        """Initialize regular group 4 runner."""
        if mesh.size != _CONST_32 or tuple(mesh.axis_names) != ("k3_sp",):
            _err_msg = "indexed four-layer group requires flat k3_sp TP32"
            raise ValueError(_err_msg)
        check_window(seq_len, "indexed four-layer group")
        self.sharding = NamedSharding(mesh, P("k3_sp"))
        self.mixture = make_mixture_norm_sp(mesh, seq_len=seq_len)
        self.norm_gather = make_norm_gather_sp(mesh, seq_len=seq_len)
        self.kda = make_norm_gather_kda_sp(mesh, seq_len=seq_len, segments=segments)
        self.kda_outputs = (3, 4, 6, 7)
        self.mla = make_indexed_mla_layer_sp_ring(
            mesh,
            seq_len=seq_len,
            flash_tile=mla_flash_tile,
            segments=segments,
            prefix_capacity=prefix_capacity,
        )
        self.moe = make_indexed_moe_sp_ring(mesh, seq_len=seq_len)

    def choose(self, family: _Dynamic, slot: _Dynamic) -> _Dynamic:
        """Execute choose logic."""
        return jax.lax.dynamic_index_in_dim(family, slot, axis=1, keepdims=False)

    def put_state(self, family: _Dynamic, state: _Dynamic, slot: _Dynamic) -> _Dynamic:
        """Execute put state logic."""
        return jax.lax.dynamic_update_slice(
            family, state[:, None], (0, slot, *([0] * (family.ndim - 2)))
        )

    def scalar(self, value: _Dynamic) -> _Dynamic:
        """Execute scalar logic."""
        return jnp.asarray([value], jnp.int32)

    def run(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
        """Execute regular group 4 layer logic."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            prefix,
            blocks,
            common,
            kda_families,
            mla_families,
            moe_families,
            kda_state,
            mla_state,
            length,
            group,
            segment_desc,
            arena,
        ) = (
            _p_next(_p_iter, _p_get("prefix")),
            _p_next(_p_iter, _p_get("blocks")),
            _p_next(_p_iter, _p_get("common")),
            _p_next(_p_iter, _p_get("kda_families")),
            _p_next(_p_iter, _p_get("mla_families")),
            _p_next(_p_iter, _p_get("moe_families")),
            _p_next(_p_iter, _p_get("kda_state")),
            _p_next(_p_iter, _p_get("mla_state")),
            _p_next(_p_iter, _p_get("length")),
            _p_next(_p_iter, _p_get("group")),
            _p_next(_p_iter, _p_get("segment_desc")),
            _p_next(_p_iter, _p_get("arena")),
        )
        base = group[0] * 4
        kda_states = tuple(kda_state)
        if len(kda_states) != len(self.kda_outputs):
            _err_msg = "kda_state is (rec, hist, rec_at, hist_at)"
            raise ValueError(_err_msg)
        mla_cache = tuple(mla_state)
        if len(mla_cache) != 1:
            _err_msg = "mla_state is (latents,)"
            raise ValueError(_err_msg)
        blocks = jax.lax.cond(
            base % 12 == 0,
            lambda x: x.at[:, base // 12].set(prefix),
            lambda x: x,
            blocks,
        )
        tap = prefix
        for offset in range(4):
            index = base + offset
            if offset == _CONST_2:
                tap = prefix
            attn_res = self.choose(common["attn_res"], index)
            attn_norm = self.choose(common["attn_norm"], index)
            attn_active = self.scalar((index - 1) // 12 + 1)
            if offset < _CONST_3:
                slot = base - group[0] + offset
                result = self.kda(
                    prefix,
                    blocks,
                    attn_res,
                    attn_norm,
                    kda_families,
                    self.choose(kda_states[0], slot),
                    self.choose(kda_states[1], slot),
                    attn_active,
                    length,
                    self.scalar(slot),
                    segment_desc,
                )
                after_attention = result[5]
                kda_states = tuple(
                    self.put_state(family, result[i], slot)
                    for family, i in zip(kda_states, self.kda_outputs, strict=True)
                )
            else:
                slot = group[0]
                _, gathered = self.norm_gather(
                    prefix,
                    blocks,
                    attn_res,
                    attn_norm,
                    attn_active,
                )
                after_attention, *updates = self.mla(
                    gathered,
                    prefix,
                    mla_families,
                    segment_desc,
                    length,
                    self.scalar(slot),
                    arena,
                )
                mla_cache = tuple(
                    self.put_state(family, update, slot)
                    for family, update in zip(mla_cache, updates, strict=True)
                )
            ffn_input = self.mixture(
                prefix,
                blocks,
                self.choose(common["ffn_res"], index),
                self.choose(common["ffn_norm"], index),
                attn_active,
            )
            moe_slot = index - 1
            after_ffn = self.moe(
                ffn_input,
                moe_families,
                self.choose(common["router_bias"], moe_slot),
                self.choose(common["shared_gate"], moe_slot),
                self.scalar(moe_slot),
            )
            prefix = after_attention + after_ffn

        return prefix, blocks, kda_states, mla_cache, tap


def make_regular_group4_sp_indexed(
    mesh: _Dynamic,
    *,
    seq_len: _Dynamic = 8192,
    segments: _Dynamic,
    prefix_capacity: _Dynamic,
    mla_flash_tile: _Dynamic = 256,
) -> _Dynamic:
    """One runtime-indexed group at layers 4g..4g+3 (g = 1..22) over a packed window."""
    runner = _RegularGroup4Runner(
        mesh,
        seq_len=seq_len,
        segments=segments,
        prefix_capacity=prefix_capacity,
        mla_flash_tile=mla_flash_tile,
    )
    return jax.named_call(
        runner.run,
        name="regular_group4_sp_indexed",
    )


def make_regular_group8_sp_indexed(
    mesh: _Dynamic,
    *,
    seq_len: _Dynamic = 8192,
    segments: _Dynamic,
    prefix_capacity: _Dynamic,
    mla_flash_tile: _Dynamic = 256,
) -> _Dynamic:
    """Two consecutive regular four-layer groups as one executable.

    (``pair`` p runs groups 2p + 1 and 2p + 2; ``arena`` carries both
    groups' MLA pools, in order). Also returns the prefix between the two
    groups: the residual stream entering layer 8p + 12 (pair 10's is the
    DSpark draft's aux layer 88).
    """
    group4 = make_regular_group4_sp_indexed(
        mesh,
        seq_len=seq_len,
        mla_flash_tile=mla_flash_tile,
        segments=segments,
        prefix_capacity=prefix_capacity,
    )
    sharding = NamedSharding(mesh, P("k3_sp"))

    def run(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
        """Execute run logic."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            prefix,
            blocks,
            common,
            kda_families,
            mla_families,
            moe_families,
            kda_state,
            mla_state,
            length,
            pair,
            segment_desc,
            arena,
        ) = (
            _p_next(_p_iter, _p_get("prefix")),
            _p_next(_p_iter, _p_get("blocks")),
            _p_next(_p_iter, _p_get("common")),
            _p_next(_p_iter, _p_get("kda_families")),
            _p_next(_p_iter, _p_get("mla_families")),
            _p_next(_p_iter, _p_get("moe_families")),
            _p_next(_p_iter, _p_get("kda_state")),
            _p_next(_p_iter, _p_get("mla_state")),
            _p_next(_p_iter, _p_get("length")),
            _p_next(_p_iter, _p_get("pair")),
            _p_next(_p_iter, _p_get("segment_desc")),
            _p_next(_p_iter, _p_get("arena")),
        )
        first = jnp.asarray([2 * pair[0] + 1], jnp.int32)
        second = first + 1
        arenas = arena.split()
        middle, blocks, kda_state, mla_state, _ = group4(
            prefix,
            blocks,
            common,
            kda_families,
            mla_families,
            moe_families,
            kda_state,
            mla_state,
            length,
            first,
            segment_desc,
            arenas[0],
        )
        prefix, blocks, kda_state, mla_state, tap = group4(
            middle,
            blocks,
            common,
            kda_families,
            mla_families,
            moe_families,
            kda_state,
            mla_state,
            length,
            second,
            segment_desc,
            arenas[1],
        )
        return prefix, blocks, kda_state, mla_state, middle, tap

    return jax.jit(
        run,
        out_shardings=(
            sharding,
            sharding,
            (sharding,) * 4,
            (sharding,),
            sharding,
            sharding,
        ),
    )


def make_final_mla_sp_indexed(
    mesh: _Dynamic,
    *,
    seq_len: _Dynamic = 8192,
    segments: _Dynamic,
    prefix_capacity: _Dynamic,
    mla_flash_tile: _Dynamic = 256,
) -> _Dynamic:
    """K3's last layer (92: MLA, then MoE) over a packed window.

    ``run(prefix, blocks, common, mla_families, moe_families, mla_state,
    length, segment_desc, arena)``; see make_prefill_stack_sp_indexed.
    """
    if mesh.size != _CONST_32 or tuple(mesh.axis_names) != ("k3_sp",):
        _err_msg = "final MLA layer requires flat k3_sp TP32"
        raise ValueError(_err_msg)
    check_window(seq_len, "final MLA layer")
    sharding = NamedSharding(mesh, P("k3_sp"))
    mixture = make_mixture_norm_sp(mesh, seq_len=seq_len)
    norm_gather = make_norm_gather_sp(mesh, seq_len=seq_len)
    mla = make_indexed_mla_layer_sp_ring(
        mesh,
        seq_len=seq_len,
        flash_tile=mla_flash_tile,
        segments=segments,
        prefix_capacity=prefix_capacity,
    )
    moe = make_indexed_moe_sp_ring(mesh, seq_len=seq_len)

    def run(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
        """Execute run logic."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            prefix,
            blocks,
            common,
            mla_families,
            moe_families,
            mla_state,
            length,
            segment_desc,
            arena,
        ) = (
            _p_next(_p_iter, _p_get("prefix")),
            _p_next(_p_iter, _p_get("blocks")),
            _p_next(_p_iter, _p_get("common")),
            _p_next(_p_iter, _p_get("mla_families")),
            _p_next(_p_iter, _p_get("moe_families")),
            _p_next(_p_iter, _p_get("mla_state")),
            _p_next(_p_iter, _p_get("length")),
            _p_next(_p_iter, _p_get("segment_desc")),
            _p_next(_p_iter, _p_get("arena")),
        )
        if len(mla_state) != 1:
            _err_msg = "mla_state is (latents,)"
            raise ValueError(_err_msg)
        attn_res = common["attn_res"][:, 92]
        attn_norm = common["attn_norm"][:, 92]
        active = jnp.asarray([8], jnp.int32)
        _, gathered = norm_gather(
            prefix,
            blocks,
            attn_res,
            attn_norm,
            active,
        )
        after_attention, *updates = mla(
            gathered,
            prefix,
            mla_families,
            segment_desc,
            length,
            jnp.asarray([23], jnp.int32),
            arena,
        )
        mla_cache = tuple(
            jax.lax.dynamic_update_slice(
                family, update[:, None], (0, 23, *([0] * (family.ndim - 2)))
            )
            for family, update in zip(mla_state, updates, strict=True)
        )
        ffn_input = mixture(
            after_attention,
            blocks,
            common["mlp_res"][:, 92],
            common["ffn_norm"][:, 92],
            active,
        )
        prefix = moe(
            ffn_input,
            after_attention,
            moe_families,
            length,
            jnp.asarray([91], jnp.int32),
        )
        return prefix, blocks, mla_cache

    return jax.jit(
        run,
        out_shardings=(
            sharding,
            sharding,
            (sharding,),
        ),
    )


# ============================================================================
# Layer stack
# ============================================================================
#
# Layer 0's KDA and dense FFN, then a ``fori_loop`` picking KDA or MLA per layer.


class _PrefillStackRunner:
    """Runner for layers 0..depth-1 over a packed window."""

    def __getattr__(self, name: _Dynamic) -> _Dynamic:
        """Dynamic attribute access."""
        raise AttributeError(name)

    def __init__(
        self,
        mesh: _Dynamic,
        *_args: _Dynamic,
        **_kw: _Dynamic,
    ) -> None:
        """Initialize prefill stack runner."""
        seq_len = _kw.get("seq_len", 8192)
        depth = _kw.get("depth", 93)
        segments = _kw.get("segments")
        prefix_capacity = _kw.get("prefix_capacity")
        mla_flash_tile = _kw.get("mla_flash_tile", 256)
        tap_layer = _kw.get("tap_layer")
        if mesh.size != _CONST_32 or tuple(mesh.axis_names) != ("k3_sp",):
            _err_msg = "indexed prefill stack requires flat k3_sp TP32 mesh"
            raise ValueError(_err_msg)
        if tap_layer is not None and not 1 <= tap_layer < depth:
            _err_msg = "tap_layer must be one of the stack's later layers"
            raise ValueError(_err_msg)
        check_window(seq_len, "indexed prefill stack")
        if not _CONST_2 <= depth <= _CONST_93:
            _err_msg = "indexed prefill stack needs depth 2..93"
            raise ValueError(_err_msg)
        self.depth = depth
        self.tap_layer = tap_layer
        self.sharding = NamedSharding(mesh, P("k3_sp"))
        self.mixture = make_mixture_norm_sp(mesh, seq_len=seq_len)
        self.norm_gather = make_norm_gather_sp(mesh, seq_len=seq_len)
        self.kda = make_norm_gather_kda_sp(mesh, seq_len=seq_len, segments=segments)
        self.mla = make_indexed_mla_layer_sp_ring(
            mesh,
            seq_len=seq_len,
            flash_tile=mla_flash_tile,
            segments=segments,
            prefix_capacity=prefix_capacity,
        )
        self.moe = make_indexed_moe_sp_ring(mesh, seq_len=seq_len)
        self.dense = make_dense_ffn_sp(mesh, seq_len=seq_len)
        self.kda_outputs = (3, 4, 6, 7)

    def choose(self, family: _Dynamic, slot: _Dynamic) -> _Dynamic:
        """Execute choose logic."""
        return jax.lax.dynamic_index_in_dim(family, slot, axis=1, keepdims=False)

    def put_state(self, family: _Dynamic, state: _Dynamic, slot: _Dynamic) -> _Dynamic:
        """Execute put state logic."""
        return jax.lax.dynamic_update_slice(
            family, state[:, None], (0, slot, *([0] * (family.ndim - 2)))
        )

    def active(self, value: _Dynamic) -> _Dynamic:
        """Execute active logic."""
        return jnp.asarray([value], jnp.int32)

    def _run_kda_step(
        self,
        tokens: tuple[_Dynamic, _Dynamic],
        shared: tuple[_Dynamic, _Dynamic, _Dynamic],
        kda_state: tuple[_Dynamic, ...],
        kda_families: _Dynamic,
        slots: tuple[_Dynamic, _Dynamic],
    ) -> tuple[_Dynamic, tuple[_Dynamic, ...]]:
        """Run KDA attention sublayer."""
        prefix, blocks = tokens
        common, length, segment_desc = shared
        layer, kda_slot = slots
        active_blocks = self.active((layer - 1) // 12 + 1)
        result = self.kda(
            prefix,
            blocks,
            self.choose(common["attn_res"], layer),
            self.choose(common["attn_norm"], layer),
            kda_families,
            self.choose(kda_state[0], kda_slot),
            self.choose(kda_state[1], kda_slot),
            active_blocks,
            length,
            self.active(kda_slot),
            segment_desc,
        )
        new_kda = tuple(
            self.put_state(family, result[i], kda_slot)
            for family, i in zip(kda_state, self.kda_outputs, strict=True)
        )
        return result[5], new_kda

    def _run_mla_step(
        self,
        tokens: tuple[_Dynamic, _Dynamic],
        shared: tuple[_Dynamic, _Dynamic, _Dynamic, _Dynamic],
        mla_state: tuple[_Dynamic, ...],
        mla_families: _Dynamic,
        slots: tuple[_Dynamic, _Dynamic],
    ) -> tuple[_Dynamic, tuple[_Dynamic, ...]]:
        """Run MLA attention sublayer."""
        prefix, blocks = tokens
        common, length, segment_desc, arena = shared
        layer, mla_slot = slots
        active_blocks = self.active((layer - 1) // 12 + 1)
        _, gathered = self.norm_gather(
            prefix,
            blocks,
            self.choose(common["attn_res"], layer),
            self.choose(common["attn_norm"], layer),
            active_blocks,
        )
        after_attention, *updates = self.mla(
            gathered,
            prefix,
            mla_families,
            segment_desc,
            length,
            self.active(mla_slot),
            arena,
        )
        new_mla = tuple(
            self.put_state(family, update, mla_slot)
            for family, update in zip(mla_state, updates, strict=True)
        )
        return after_attention, new_mla

    def run(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
        """Run prefill stack execution."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            hidden,
            common,
            kda_families,
            mla_families,
            moe_families,
            kda_state,
            mla_state,
            length,
            segment_desc,
            arena,
        ) = (
            _p_next(_p_iter, _p_get("hidden")),
            _p_next(_p_iter, _p_get("common")),
            _p_next(_p_iter, _p_get("kda_families")),
            _p_next(_p_iter, _p_get("mla_families")),
            _p_next(_p_iter, _p_get("moe_families")),
            _p_next(_p_iter, _p_get("kda_state")),
            _p_next(_p_iter, _p_get("mla_state")),
            _p_next(_p_iter, _p_get("length")),
            _p_next(_p_iter, _p_get("segment_desc")),
            _p_next(_p_iter, _p_get("arena")),
        )
        kda_state = tuple(kda_state)
        if len(kda_state) != len(self.kda_outputs):
            _err_msg = "kda_state is (rec, hist, rec_at, hist_at)"
            raise ValueError(_err_msg)
        if len(mla_state) != 1:
            _err_msg = "mla_state is (latents,)"
            raise ValueError(_err_msg)

        seq_len, hidden_size = hidden.shape
        blocks = jnp.zeros((8, seq_len, hidden_size), dtype=hidden.dtype)

        # Layer 0: KDA followed by dense FFN
        result0 = self.kda(
            hidden,
            blocks,
            common["attn_res"][:, 0],
            common["attn_norm"][:, 0],
            kda_families,
            kda_state[0][:, 0],
            kda_state[1][:, 0],
            self.active(0),
            length,
            self.active(0),
            segment_desc,
        )
        kda_state = tuple(
            self.put_state(family, result0[i], 0)
            for family, i in zip(kda_state, self.kda_outputs, strict=True)
        )
        ffn_in0 = self.mixture(
            hidden,
            blocks,
            common["ffn_res"][:, 0],
            common["ffn_norm"][:, 0],
            self.active(0),
        )
        prefix = result0[5] + self.dense(ffn_in0)

        tap_state = prefix

        def layer_step(layer: _Dynamic, carry: _Dynamic) -> _Dynamic:
            """Perform one layer step."""
            p, b, ks, ms, ts = carry
            ts = jax.lax.cond(
                layer == (self.tap_layer if self.tap_layer is not None else -1),
                lambda: p,
                lambda: ts,
            )
            b = jax.lax.cond(
                layer % 12 == 0,
                lambda x: x.at[:, layer // 12].set(p),
                lambda x: x,
                b,
            )
            is_mla = layer % 4 == _CONST_3
            kda_slot = layer - layer // 4
            mla_slot = layer // 4

            def do_mla() -> _Dynamic:
                """Run MLA step."""
                return self._run_mla_step(
                    (p, b),
                    (common, length, segment_desc, arena),
                    ms,
                    mla_families,
                    (layer, mla_slot),
                )

            def do_kda() -> _Dynamic:
                """Run KDA step."""
                return self._run_kda_step(
                    (p, b),
                    (common, length, segment_desc),
                    ks,
                    kda_families,
                    (layer, kda_slot),
                )

            after_attn, (new_ks, new_ms) = jax.lax.cond(
                is_mla,
                lambda: (do_mla()[0], (ks, do_mla()[1])),
                lambda: (do_kda()[0], (do_kda()[1], ms)),
            )

            active_blocks = self.active((layer - 1) // 12 + 1)
            ffn_in = self.mixture(
                p,
                b,
                self.choose(common["ffn_res"], layer),
                self.choose(common["ffn_norm"], layer),
                active_blocks,
            )
            moe_slot = layer - 1
            after_ffn = self.moe(
                ffn_in,
                moe_families,
                self.choose(common["router_bias"], moe_slot),
                self.choose(common["shared_gate"], moe_slot),
                self.active(moe_slot),
            )
            p = after_attn + after_ffn
            return p, b, new_ks, new_ms, ts

        initial = (prefix, blocks, kda_state, tuple(mla_state), tap_state)
        prefix, blocks, kda_state, mla_state, tap_state = jax.lax.fori_loop(
            1, self.depth, layer_step, initial
        )
        if self.tap_layer is not None:
            return prefix, kda_state, mla_state, tap_state
        return prefix, kda_state, mla_state


def make_prefill_stack_sp_indexed(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Layers 0..depth-1 over a packed window of segments sequences."""
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    mesh = _p_next(_p_iter, _p_get("mesh"))
    (
        seq_len,
        depth,
        segments,
        prefix_capacity,
        mla_flash_tile,
        unroll,
        tap_layer,
    ) = (
        _p_get("seq_len", 8192),
        _p_get("depth", 93),
        _p_get("segments"),
        _p_get("prefix_capacity"),
        _p_get("mla_flash_tile", 256),
        _p_kw.get("unroll", False),
        _p_get("tap_layer"),
    )
    runner = _PrefillStackRunner(
        mesh,
        seq_len=seq_len,
        depth=depth,
        segments=segments,
        prefix_capacity=prefix_capacity,
        mla_flash_tile=mla_flash_tile,
        unroll=unroll,
        tap_layer=tap_layer,
    )
    return runner.run
