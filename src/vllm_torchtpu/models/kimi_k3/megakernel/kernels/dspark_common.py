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
"""Shared non-SWA constants and helpers for DSpark speculative decoding."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import tpu

_Dynamic = Any


ROWS = 8  # maximum block width and padded public ABI (anchor + 7)
RANKS = 32
LM_CHUNK = 512  # LM head columns per chunk (the gate/up buffer width)
MLP_PADDED = 512  # 448 MLP channels per rank padded to whole lane tiles
LM_CHUNKS = 10  # 10 * 512 = 5120 vocabulary rows per rank
FC_IN = 35840 // 4  # aux features contracted per host (8960)
CHUNK = 7168 // 8  # 896-column chunk of the hidden vector per host lane
ROW_PACKED = 7168 // 256  # 28 FP32 rows hold one BF16 [1, 7168] row
NEG = -3.0e38

# Output vector layout (int32 [1, 128]): drafts, this step's verify targets,
# accepted count, bonus token (the next anchor), top-2 margins as FP32 bits.
OUT_DRAFTS, OUT_TARGETS, OUT_COUNT, OUT_BONUS, OUT_MARGINS = 0, 8, 16, 17, 24

# Semaphore index ranges (per slot) so a fast peer's next phase can never
# satisfy a wait of the previous phase.
SEM_RS, SEM_HOST, SEM_GATHER, SEM_TINY, SEM_BCAST, SEMS_PER_SLOT = 0, 8, 16, 32, 64, 96

VMEM_NAMES = (
    "ln1",
    "ln2",
    "qnorm",
    "kvnorm",
    "hidden_norm",
    "final_norm",
    "mask_embed",
    "rot_q",
    "rot_kv",
    "head_ones",
    "kv_ones",
    "head_select",
    "head_place",
)

RING_MAGIC = 0x44535231  # "DSR1"
RING_BLOCK_ROWS = 15  # 69 * 15 rows hold a ring and its header
RING_ROW_LANES = 1280  # bf16 per 2560-byte pool row


def bf16(x: _Dynamic) -> _Dynamic:
    """Execute bf16 logic."""
    return x.astype(jnp.bfloat16)


def f32(x: _Dynamic) -> _Dynamic:
    """Execute f32 logic."""
    return x.astype(jnp.float32)


def select_rows(block32: _Dynamic, offset: _Dynamic, rows: _Dynamic) -> _Dynamic:
    """Rows ``offset .. offset + rows`` of a ``[16, W]`` value as ``[rows, W]``."""
    row_index = jax.lax.broadcasted_iota(jnp.int32, block32.shape, 0)
    return jnp.concatenate(
        [
            jnp.sum(
                jnp.where(row_index == offset + k, block32, 0.0),
                axis=0,
                keepdims=True,
            )
            for k in range(rows)
        ],
        axis=0,
    )


def rms_norm(x_bf16: _Dynamic, weight_bf16: _Dynamic, eps: _Dynamic) -> _Dynamic:
    """Qwen3RMSNorm: FP32 stats, BF16 cast, BF16 scale."""
    x32 = f32(x_bf16)
    normed = x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    return bf16(f32(bf16(normed)) * f32(weight_bf16))


def pack(value: _Dynamic) -> _Dynamic:
    """``[rows, width]`` -> FP32 ``[rows * width / 256, 128]`` words."""
    rows, width = value.shape
    lanes = (
        value.astype(jnp.float32).reshape(rows * width // 128, 128).astype(jnp.bfloat16)
    )
    return jax.lax.bitcast_convert_type(tpu.bitcast(lanes, jnp.uint32), jnp.float32)


def unpack(words: _Dynamic, rows: _Dynamic, width: _Dynamic) -> _Dynamic:
    """Inverse of :func:`pack`, returned as FP32 ``[rows, width]``."""
    values = tpu.bitcast(jax.lax.bitcast_convert_type(words, jnp.uint32), jnp.bfloat16)
    return values.astype(jnp.float32).reshape(rows, width)


def _header_bytes(values: _Dynamic) -> _Dynamic:
    """int32 ``[..., k]`` -> bf16 ``[..., 4 k]``: each byte as an exact small.

    number. (Bit-cast halves would include bf16 subnormals, which the TPU
    flushes to zero when it moves them.)
    """
    shifts = jnp.asarray([0, 8, 16, 24], jnp.int32)
    parts = jnp.right_shift(values[..., None], shifts) & 0xFF
    return parts.astype(jnp.bfloat16).reshape(*values.shape[:-1], -1)


def _header_ints(parts: _Dynamic) -> _Dynamic:
    """Inverse of :func:`_header_bytes` on ``[..., k, 4]``."""
    shifts = jnp.asarray([0, 8, 16, 24], jnp.int32)
    return jnp.sum(jnp.left_shift(parts.astype(jnp.int32) & 0xFF, shifts), axis=-1)


def ring_rows(c: _Dynamic) -> int:
    """Pool rows a ring's K | V values take (``[layers, window, 2 head_dim]`` bf16)."""
    return -(-c.layers * c.window * 2 * c.head_dim // RING_ROW_LANES)


def pack_ring(ring: _Dynamic, meta: _Dynamic, blocks: int) -> _Dynamic:
    """``ring`` ``[n, layers, window, 2 head_dim]`` bf16 and ``meta`` ``[n, 2]``.

    int32 (length, first position) -> ``[n, blocks, RING_BLOCK_ROWS,
    RING_ROW_LANES]`` bf16: the ring's rows, then a header row, zero padded.
    """
    n, size = ring.shape[0], int(np.prod(ring.shape[1:]))
    flat = jnp.pad(ring.reshape(n, size), ((0, 0), (0, -size % RING_ROW_LANES)))
    rows = flat.reshape(n, -1, RING_ROW_LANES)
    header = jnp.concatenate(
        (jnp.full((n, 1), RING_MAGIC, jnp.int32), meta.astype(jnp.int32)),
        axis=1,
    )
    header = _header_bytes(header).reshape(n, 1, -1)
    header = jnp.pad(header, ((0, 0), (0, 0), (0, RING_ROW_LANES - header.shape[-1])))
    total = blocks * RING_BLOCK_ROWS
    if rows.shape[1] + 1 > total:
        _err_msg = (
            f"a ring needs {rows.shape[1] + 1} rows, {blocks} blocks hold {total}"
        )
        raise ValueError(_err_msg)
    packed = jnp.concatenate((rows, header), axis=1)
    packed = jnp.pad(packed, ((0, 0), (0, total - packed.shape[1]), (0, 0)))
    packed = jax.lax.optimization_barrier(packed)
    return packed.reshape(n, blocks, RING_BLOCK_ROWS, RING_ROW_LANES)


def unpack_ring(
    c: _Dynamic, packed: _Dynamic, length: _Dynamic, cap: _Dynamic = None
) -> _Dynamic:
    """Inverse of :func:`pack_ring` for one ring read back from the pool:.

    ``packed`` ``[blocks, RING_BLOCK_ROWS, RING_ROW_LANES]`` bf16 -> ``(ring
    [layers, window, 2 head_dim], meta [2])``. A ring whose header is not
    ``RING_MAGIC`` at ``length`` is empty: meta ``(length, length)``.
    ``cap``: raise the ring's first position to at least this (a prefix hit
    keeps what the ring held before it).
    """
    rows = packed.reshape(-1, RING_ROW_LANES)
    count = ring_rows(c)
    shape = (c.layers, c.window, 2 * c.head_dim)
    ring = rows[:count].reshape(-1)[: int(np.prod(shape))].reshape(shape)
    header = _header_ints(rows[count, : 4 * 3].reshape(3, 4))
    valid = (header[0] == RING_MAGIC) & (header[1] == length)
    first = jnp.where(valid, header[2], length)
    if cap is not None:
        first = jnp.maximum(first, cap)
    return ring, jnp.stack((jnp.asarray(length, jnp.int32), first.astype(jnp.int32)))


def ring_positions(c: _Dynamic, meta: _Dynamic) -> _Dynamic:
    """The ring's slot positions ``[window]`` int32 (``-1`` empty) from its meta."""
    length, first = meta[0], meta[1]
    window = c.window
    slot = jnp.arange(window, dtype=jnp.int32)
    position = length - 1 - jnp.mod(length - 1 - slot, window)
    return jnp.where(
        (position >= 0) & (position >= first) & (position >= length - window),
        position,
        -1,
    )


def synthetic_count(rates: _Dynamic, seed: _Dynamic, position: _Dynamic) -> _Dynamic:
    """Execute synthetic count logic."""
    uniform = jax.random.uniform(jax.random.fold_in(jax.random.PRNGKey(seed), position))
    return jnp.sum(uniform < jnp.asarray(rates, jnp.float32)).astype(jnp.int32)


def nucleus_accept(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Execute nucleus accept logic."""
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        values,
        ids,
        lse,
        drafts,
        forced_count,
        top_p,
        uniforms,
    ) = (
        _p_next(_p_iter, _p_get("values")),
        _p_next(_p_iter, _p_get("ids")),
        _p_next(_p_iter, _p_get("lse")),
        _p_next(_p_iter, _p_get("drafts")),
        _p_next(_p_iter, _p_get("forced_count")),
        _p_next(_p_iter, _p_get("top_p")),
        _p_next(_p_iter, _p_get("uniforms")),
    )
    values, ids, lse, drafts, uniforms = (
        jnp.asarray(x) for x in (values, ids, lse, drafts, uniforms)
    )
    rows, n = values.shape
    order = jnp.argsort(-values, axis=1)
    sorted_values = jnp.take_along_axis(values, order, axis=1)
    sorted_ids = jnp.take_along_axis(ids, order, axis=1)
    probs = jnp.exp(sorted_values - lse[:, None])
    cumulative = jnp.cumsum(probs, axis=1)
    keep = ((cumulative - probs) < top_p) | (jnp.arange(n)[None, :] == 0)
    kept = jnp.where(keep, probs, 0.0)
    nucleus = kept / jnp.sum(kept, axis=1, keepdims=True)
    draft_mask = sorted_ids == drafts[:, None]
    draft_prob = jnp.sum(jnp.where(draft_mask, nucleus, 0.0), axis=1)
    k = jnp.arange(rows, dtype=jnp.int32)
    drafted = drafts >= 0
    accept = drafted & (uniforms[:, 0] < draft_prob)
    first_reject = jnp.min(jnp.where(~accept & (k < rows - 1), k, rows - 1))
    count = jnp.where(forced_count >= 0, forced_count, first_reject).astype(jnp.int32)
    row = nucleus[count]
    remove = (forced_count < 0) & drafted[count]
    residual = jnp.where(remove & draft_mask[count], 0.0, row)
    cumulative = jnp.cumsum(residual)
    total = cumulative[-1]
    pick = jnp.argmax(cumulative > uniforms[count, 1] * total)
    bonus = jnp.where(total > 0, sorted_ids[count, pick], sorted_ids[count, 0])
    return count, bonus.astype(jnp.int32)
