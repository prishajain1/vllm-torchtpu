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

"""Decode context parallelism (DCP) for Kimi-K3 MLA.

Source provenance: Adapted from inferact/k3-megakernel@30c9ba966b.

Without DCP every TP32 rank holds its 3 heads' expanded K|V for every token.
Under DCP the ``G`` ranks of a group (consecutive ranks, ``member = rank %
G``) each hold the compressed latent (576 of 640 lanes) of a disjoint 1/G of
the tokens and attend *all* the group's heads over them in latent space
(``q_abs = [q_nope @ W_UK^T | q_rope | 0]``, score scale folded in):

1. **q all-gather.** Each rank writes its ``[B, heads, 640]`` absorbed
   queries into slot ``member`` of every group peer's gather buffer.
2. **Local attention.** For each row, over the rank's own pages (global page
   ``j G + member`` is local page ``j``; tokens ``< position``), ``span``
   pages per step: ``Q @ C^T`` with each member's ``slots`` head slots
   (``packet_layout``) on sublanes, online softmax in FP32,
   ``O += P @ C[:, :512]``.
3. **Combine.** Each rank normalizes its partial (``o = O / l``, ``lse = m +
   log l``) and sends member ``d`` a BF16 ``[slots B, 640]`` packet of ``d``'s
   heads: row ``slots b + h``, ``o`` in lanes 0:512, ``lse`` as a high and a
   low BF16 part in lanes 512 and 513.
4. **Merge** on the owner: the log-sum-exp merge of the G partials plus the
   new token (replicated, so in no rank's pages yet), giving the latent-space
   output ``[B, heads, 512]`` (the caller applies W_UV).

Under attention DP (``heads`` = 6 for TP16 x DP2) the DCP group is the DP
group of ``32 / dp`` ranks, over that group's rows. Global page ``g`` lives on
member ``g % G``. Block tables are the same on every rank, in one of two
conventions (``KernelConfig.dcp_tables``):

* **local** (the default; vLLM's ``cp_kv_cache_interleave_size`` of one page):
  entry ``j`` covers G pages and every rank keeps its page in its block of that
  id; pools shrink by G, blocks are shared per G pages.
* **global**: entry ``g`` is page ``g``'s block, exactly vLLM's non-DCP table;
  only its owner fills that block, so every rank's pool has all the block ids.

The cache is vLLM's unified pools ``[blocks, 128, packing, 640]`` (packing 2 in
BF16, 4 in FP8); ``paged.commit_latent_rows`` commits new rows on their owners.
In-kernel pieces: :class:`DcpCache` (pool and table), :class:`DcpWorkspace`
(:func:`persistent_scratch`, which peers write into, so it must not alias other
scoped VMEM, and :func:`scoped_scratch`) and :class:`DcpAttention` (``prime``
the first page DMAs early, ``attend`` later).
"""

from __future__ import annotations

import contextlib
import dataclasses
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.models.kimi_k3.megakernel.config import active as _config

_Dynamic = Any
_CONST_1024 = 1024
_CONST_128 = 128
_CONST_16 = 16
_CONST_2 = 2
_CONST_2048 = 2048
_CONST_24 = 24
_CONST_3 = 3
_CONST_32 = 32
_CONST_4 = 4
_CONST_48 = 48
_CONST_512 = 512
_CONST_576 = 576
_CONST_580 = 580
_CONST_6 = 6
_CONST_64 = 64
_CONST_8 = 8


PAGE_TOKENS = 256
LANES = 640  # latent 512 | rope 64 | zero 64
LATENT = 512
HEADS_PER_RANK = 3
NEG = -1e30  # finite "no tokens" log-sum-exp
# batch rows of an unchunked call; a larger batch runs in chunks
# (see _attend_chunked)
MAX_ROWS = 8
HOST_LANES = 8  # DCP ranks per host (rank = host * 8 + lane)
DP4_HEADS = 12  # attention DP4's held heads per rank (four TP32 shards)


def _lookup_rows(batch: _Dynamic, heads: _Dynamic = HEADS_PER_RANK) -> _Dynamic:
    """Calls whose page steps find their row by a binary search over SMEM.

    bounds (and stage rows one by one): 32, 48 and 64 rows; attention DP4's
    16, 24 and 32 group rows.
    """
    if heads == DP4_HEADS:
        return batch in (16, 24, 32)
    return batch in (32, 48, 64)


def chunk_layout(
    batch: int, heads: int = HEADS_PER_RANK, *, chunk_rows: int | None = None
) -> tuple[int, int]:
    """Get (chunk rows, ring slots) of a chunked call (``batch > MAX_ROWS``).

    8-row chunks at B16/B32/B48/B64 and attention DP4's 24 group rows
    (amortizing exchanges), else 4 (also attention DP2's B128 and DP4's: 64
    group rows of six heads, 32 of twelve, for VMEM); ring slots
    4 at B32 with three heads per rank, 3 for 8-row chunks, else 2
    (``persistent_scratch`` uses 4 for FP8 query packets). ``chunk_rows``
    overrides the width and keeps at least three slots for a staged combine.
    """
    wide = batch <= _CONST_16 or (
        batch in (32, 48, 64) and not (batch == _CONST_64 and heads == _CONST_6)
    )
    if heads == DP4_HEADS:
        wide = batch <= _CONST_24
    rows = chunk_rows if chunk_rows is not None else 8 if wide else 4
    ring = (
        4
        if batch == _CONST_32 and heads == HEADS_PER_RANK
        else 3
        if wide or chunk_rows is not None
        else 2
    )
    if rows not in (4, 8) or batch % rows or rows > MAX_ROWS or ring not in (2, 3, 4):
        _err_msg = (
            f"DCP chunks: {rows} rows must divide the batch {batch} "
            f"(<= {MAX_ROWS}), ring 2, 3 or 4"
        )
        raise ValueError(_err_msg)
    return rows, ring


GROUP = _config().dcp_group
GLOBAL_PAGES = _config().dcp_tables == "global"
SPAN = 4  # pages per step
BUFFERS = 4  # steps in flight


def page_plan(
    batch: int,
    heads: int = HEADS_PER_RANK,
    sequence_rows: int = 1,
    cache_dtype: _Dynamic = jnp.bfloat16,
    *,
    force_chunks: _Dynamic = False,
) -> dict:
    """The decoder's page buffers for ``batch`` rows of ``heads`` heads per.

    rank (``sequence_rows`` per sequence): ``span`` pages per step and
    ``buffers`` steps in flight, sized to fit VMEM. Up to four rows and plain
    decode: ``SPAN`` x ``BUFFERS`` (3 buffers chunked, except attention DP2's
    and DP4's FP8 B128, whose long-context page reads outrun three; FP8 B64 and B96
    stage twice the span, see ``staged_folds``). Verifies above four rows:
    half the span, twice the buffers (4 at B64 and attention DP's B32), 3/4
    chunked.
    """
    if batch <= _CONST_4:
        return {"span": SPAN, "buffers": BUFFERS}
    if sequence_rows == 1:
        deep = (
            (batch == _CONST_64 and heads == _CONST_6)
            or (batch == _CONST_32 and heads == DP4_HEADS)
        ) and (cache_dtype == jnp.float8_e4m3fn)
        return {"span": SPAN, "buffers": 3 if chunked(batch) and not deep else BUFFERS}
    span = max(1, SPAN // 2)
    buffers = (
        4
        if batch == _CONST_64 or (batch == _CONST_32 and heads != HEADS_PER_RANK)
        else BUFFERS * SPAN // span
    )
    return {
        "span": span,
        "buffers": buffers
        if not chunked(batch, force_chunks)
        else max(2, buffers * 3 // 4),
    }


MAX_GROUPS = 2  # row groups of a call (see packet_layout)


def packet_layout(batch: int, heads: int = HEADS_PER_RANK) -> tuple[int, int]:
    """``(head slots, row groups)`` of a call with ``batch`` rows of ``heads``.

    heads per rank. Head slots are the packet rows per rank and batch row.
    B2 runs two groups whose exchanges overlap the other group's attention
    (used at B2 only; above it the DMA count doubles); a group's packet slice
    must be a multiple of 8 rows unless it has 4-row slots, so B2 pads its
    three heads with a zero slot. Otherwise one group of a slot per head.
    """
    return (4, 2) if batch == _CONST_2 and heads == HEADS_PER_RANK else (heads, 1)


def owner_index(table: _Dynamic) -> _Dynamic:
    """This rank's page-owner index from a static per-mesh-position table.

    (``table[p]`` for mesh position ``p``, e.g. its vLLM TP rank); None: the
    mesh position. Works in a kernel or at the XLA level.
    """
    rank = jax.lax.axis_index("tp")
    if table is None:
        return rank
    owner = 0
    for position, value in enumerate(table):
        owner = jnp.where(rank == position, value, owner)
    return owner


@dataclasses.dataclass(frozen=True)
class DcpCache:
    """This rank's latent pages in HBM: ``pool_ref`` the per-MLA-layer tuple.

    of vLLM's unified pools (``[blocks, 128, packing, 640]``, token ``t`` of a
    block at ``[t // packing, t % packing]``), picked by the traced layer index
    ``layer``. ``table_ref`` is the flat ``[rows * pages]`` int32 block table
    in SMEM (``pages`` entries per row), ``group`` the DCP group size,
    ``global_pages`` the table convention (see the module docstring).
    vLLM's pages hold stale bytes past a request's tokens and in lanes
    576:640 (freed blocks, KDA state that reads as NaN in bf16); the
    attention zeroes both before they reach the MXU.
    """

    pool_ref: _Dynamic
    table_ref: _Dynamic
    pages: int
    layer: _Dynamic = None
    group: int = GROUP
    global_pages: bool = GLOBAL_PAGES
    # Which pages this rank holds: global page g is on the rank whose owner
    # index is g % group. Default: its member index (mesh order); a P/D
    # decoder uses its vLLM TP rank (see owner_index), the order the KV
    # transfer's destination units follow.
    owner: _Dynamic = None
    owner_table: _Dynamic = None
    # The table row of attention row ``b`` (default ``b``): attention DP
    # attends its group's rows of the batch-wide table.
    table_row: Callable[[_Dynamic], _Dynamic] | None = None

    @property
    def dtype(self) -> _Dynamic:
        """Execute dtype logic."""
        return self.pool_ref[0].dtype

    @property
    def page_tokens(self) -> _Dynamic:
        """Execute page tokens logic."""
        ref = self.pool_ref[0]
        return ref.shape[-3] * ref.shape[-2]

    def start_page_copy(
        self,
        row: _Dynamic,
        local: _Dynamic,
        member: _Dynamic,
        destination: _Dynamic,
        semaphore: _Dynamic,
    ) -> _Dynamic:
        """Start the DMA of ``row``'s local page ``local`` on ``member``."""
        entry = local * self.group + member if self.global_pages else local
        if self.table_row is not None:
            row = self.table_row(row)
        block = self.table_ref[row * self.pages + entry]
        for index, ref in enumerate(self.pool_ref):

            @pl.when(self.layer == index)
            def start(ref: _Dynamic = ref) -> _Dynamic:
                """Execute start logic."""
                tpu.make_async_copy(ref.at[block], destination, semaphore).start()

    def start_native_copies(
        self,
        row: _Dynamic,
        locals_: _Dynamic,
        member: _Dynamic,
        destinations: _Dynamic,
        semaphore: _Dynamic,
    ) -> _Dynamic:
        """Start the DMAs of ``row``'s local pages ``locals_`` on ``member``.

        into ``[page_tokens, 640]`` ``destinations`` (see ``native_pages``):
        each DMA retiles a pool block into the buffer's native tile. Its
        descriptor setup is not predicated, so the layer's pool is picked by
        one binary tree of branches for all the pages, not per pool.
        """
        if self.table_row is not None:
            row = self.table_row(row)
        blocks = [
            self.table_ref[
                row * self.pages
                + (local * self.group + member if self.global_pages else local)
            ]
            for local in locals_
        ]

        def pick(first: _Dynamic, end: _Dynamic) -> _Dynamic:
            """Execute pick logic."""
            if end - first == 1:
                for block, destination in zip(blocks, destinations):
                    tpu.make_async_copy(
                        self.pool_ref[first].at[block].reshape(self.page_tokens, LANES),
                        destination,
                        semaphore,
                    ).start()
                return
            middle = (first + end) // 2
            pl.when(self.layer < middle)(lambda: pick(first, middle))
            pl.when(self.layer >= middle)(lambda: pick(middle, end))

        pick(0, len(self.pool_ref))


@dataclasses.dataclass(frozen=True)
class DcpWorkspace:
    # Persistent: peers' remote copies land here.
    """Manage dcp workspace instance and state."""

    gathered: _Dynamic  # BF16 queries, or FP8 query/scale packets as uint8
    send: _Dynamic  # [G, slots B, 640] bf16
    receive: _Dynamic  # [2, G, slots B, 640] bf16
    gather_semaphores: (
        Any  # DMA [2 (send, receive), 2 (parity), MAX_GROUPS (row group)]
    )
    combine_semaphores: _Dynamic  # DMA [2, 2, MAX_GROUPS]
    # Scoped to one MLA invocation.
    queries: _Dynamic  # [B, head_rows, 640], cache dtype; head slots padded to 16
    pages: _Dynamic  # [buffers, span * page_rows, packing, 640], cache dtype
    page_semaphores: _Dynamic  # DMA [buffers]
    maximum: _Dynamic  # [B, head_rows, 1] f32 (per head slot)
    denominator: _Dynamic  # [B, head_rows, 1] f32
    accumulator: _Dynamic  # [B, head_rows, 512] f32
    own_rows: _Dynamic  # [slots B, 640] f32 (chunked calls: chunk_rows_shape)
    new_rows: _Dynamic  # [slots B, 640] f32 (chunked calls: chunk_rows_shape)
    merged: (
        Any  # chunked calls: chunk_rows_shape(512) merged outputs (else a placeholder)
    )
    lookup_bounds: _Dynamic
    lookup_counts: _Dynamic
    query_scale: _Dynamic = None  # [held, heads, 1] FP32 dequantization scale for FP8 Q
    hosts: _Dynamic = None


def chunked(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> bool:
    """Whether a call of ``batch`` rows runs as chunks of ``MAX_ROWS``.

    (``force``: even a single chunk, e.g. for the two-stage combine).
    """
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    batch = _p_next(_p_iter, _p_get("batch"))
    force = _p_next(_p_iter, _p_kw.get("force", False))
    return batch > MAX_ROWS or force


GROUPED_SPEC_SHAPES = (
    (4, 4),
    (8, 4),
    (8, 8),
    (16, 8),
    (32, 8),
    (32, 4),
    (64, 8),
    (64, 4),
)


def _grouped_spec(
    batch: _Dynamic, group: _Dynamic, heads: _Dynamic, sequence_rows: _Dynamic
) -> _Dynamic:
    """Verifier shapes (rows per call, rows per sequence) supported by grouped.

    FP8 QK/PV and query transport: DCP32 with three heads per rank, or
    attention DP2's DCP16 with six (the same 96-row score tile).
    """
    if group == _CONST_16 and heads == _CONST_6:
        return (batch, sequence_rows) in (*GROUPED_SPEC_SHAPES, (16, 4))
    return (
        group == _CONST_32
        and heads == HEADS_PER_RANK
        and (batch, sequence_rows) in GROUPED_SPEC_SHAPES
    )


def persistent_scratch(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Execute persistent scratch logic."""
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        batch,
        group,
        heads,
        cache_dtype,
        sequence_rows,
        fp8_wire,
        force_chunks,
        chunk_rows,
    ) = (
        _p_next(_p_iter, _p_get("batch")),
        _p_next(_p_iter, _p_get("group", GROUP)),
        _p_next(_p_iter, _p_get("heads", HEADS_PER_RANK)),
        _p_get("cache_dtype", jnp.bfloat16),
        _p_get("sequence_rows", 1),
        _p_kw.get("fp8_wire", False),
        _p_kw.get("force_chunks", False),
        _p_get("chunk_rows"),
    )
    # Send the receiver's identical FP8 operand and lossless FP32 scale
    # (``fp8_wire``: for any shape, e.g. the MLA DSpark draft's two heads).
    # Plain attention DP2 holds six heads for B/2 rows, DP4 twelve for B/4.
    # Speculative attention
    # retains full-precision owner queries for its causal new-token fold.
    """Execute persistent scratch logic."""
    plain = sequence_rows == 1 and (
        (batch in (32, 64) and heads == HEADS_PER_RANK)
        or (group == _CONST_16 and batch in (16, 32, 48, 64) and heads == _CONST_6)
        or (group == _CONST_8 and batch in (16, 24, 32) and heads == DP4_HEADS)
    )
    query_dtype = (
        jnp.uint8
        if cache_dtype == jnp.float8_e4m3fn
        and (plain or fp8_wire or _grouped_spec(batch, group, heads, sequence_rows))
        else jnp.bfloat16
    )
    if chunked(batch, force_chunks):
        chunk, ring = chunk_layout(batch, heads, chunk_rows=chunk_rows)
        rows = packet_layout(batch, heads)[0] * chunk
        if query_dtype == jnp.uint8 and not (group == _CONST_16 and sequence_rows > 1):
            # The smaller query buffers make two-chunk lookahead fit at B64.
            # Attention DP's verifies keep one chunk: two change their outputs
            # at 128 rows.
            ring = 4
        return (
            tpu.VMEM((ring, group, rows, LANES), query_dtype),
            tpu.VMEM((group, rows, LANES), jnp.bfloat16),
            tpu.VMEM((ring, group, rows, LANES), jnp.bfloat16),
            tpu.SemaphoreType.DMA((2, ring)),
            tpu.SemaphoreType.DMA((2, ring)),
        )
    rows = packet_layout(batch, heads)[0] * batch
    return (
        tpu.VMEM((2, group, rows, LANES), query_dtype),
        tpu.VMEM((group, rows, LANES), jnp.bfloat16),
        tpu.VMEM((2, group, rows, LANES), jnp.bfloat16),
        tpu.SemaphoreType.DMA((2, 2, MAX_GROUPS)),
        tpu.SemaphoreType.DMA((2, 2, MAX_GROUPS)),
    )


PERSISTENT_NAMES = (
    "dcp gathered queries",
    "dcp send packets",
    "dcp received packets",
    "dcp gather semaphores",
    "dcp combine semaphores",
)


def native_pages(
    batch: _Dynamic,
    heads: _Dynamic = HEADS_PER_RANK,
    sequence_rows: _Dynamic = 1,
    cache_dtype: _Dynamic = jnp.bfloat16,
) -> bool:
    """Whether a call stages pages as ``[span * page_tokens, 640]`` in the.

    MXU's native tile (retiled by the DMA) rather than as the pool's
    ``[span * 128, packing, 640]`` blocks, whose loads relayout every tile:
    attention DP2's plain B96 and B128 (48 and 64 group rows of six heads)
    and FP8 B64 (32; not under the opt-in ``mla_decode_ktile=2048``);
    attention DP4's plain B96 and B128 and FP8 B64 (24, 32 and 16 rows).
    """
    if heads == DP4_HEADS:
        return sequence_rows == 1 and (
            batch in (24, 32)
            or (batch == _CONST_16 and cache_dtype == jnp.float8_e4m3fn)
        )
    return (
        heads == _CONST_6
        and sequence_rows == 1
        and (
            batch in (48, 64)
            or (
                batch == _CONST_32
                and cache_dtype == jnp.float8_e4m3fn
                and _config().mla_decode_ktile == _CONST_1024
            )
        )
    )


def staged_folds(
    batch: _Dynamic,
    heads: _Dynamic = HEADS_PER_RANK,
    sequence_rows: _Dynamic = 1,
    cache_dtype: _Dynamic = jnp.bfloat16,
) -> int:
    """Softmax folds per page step. Attention DP2's FP8 B64 and B96 (32 and.

    48 group rows) and DP4's (16 and 24) stage two: one page wait, row
    lookup, pool pick and QK per 2048 tokens, folded as two 1024-token parts
    in order (the normalization and FP8 probability boundaries of 1024-token
    steps, so outputs are unchanged). Their rows end with a static tail
    (``DcpAttention._staged_tail``), which B128 has no VMEM for.
    """
    if heads == DP4_HEADS:
        return (
            2
            if (
                native_pages(batch, heads, sequence_rows, cache_dtype)
                and batch in (16, 24)
                and cache_dtype == jnp.float8_e4m3fn
            )
            else 1
        )
    return (
        2
        if (
            native_pages(batch, heads, sequence_rows, cache_dtype)
            and batch in (32, 48)
            and cache_dtype == jnp.float8_e4m3fn
            and _config().mla_decode_ktile == _CONST_1024
        )
        else 1
    )


def scoped_scratch(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Execute scoped scratch logic."""
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        batch,
        span,
        buffers,
        heads,
        group,
        cache_dtype,
        page_tokens,
        sequence_rows,
        force_chunks,
        chunk_rows,
    ) = (
        _p_next(_p_iter, _p_get("batch")),
        _p_next(_p_iter, _p_get("span", SPAN)),
        _p_next(_p_iter, _p_get("buffers", BUFFERS)),
        _p_next(_p_iter, _p_get("heads", HEADS_PER_RANK)),
        _p_next(_p_iter, _p_get("group", GROUP)),
        _p_next(_p_iter, _p_get("cache_dtype", jnp.bfloat16)),
        _p_next(_p_iter, _p_get("page_tokens", PAGE_TOKENS)),
        _p_get("sequence_rows", 1),
        _p_kw.get("force_chunks", False),
        _p_get("chunk_rows"),
    )
    """Execute scoped scratch logic."""
    rows = packet_layout(batch, heads)[0] * batch
    tile = head_rows(batch, group, heads)
    packing = 4 // jnp.dtype(cache_dtype).itemsize
    # FP8 doubles tokens per physical page. The opt-in mla_decode_ktile=2048
    # folds 2048 tokens per step for plain DP2 B64 (32 local rows).
    wide = _config().mla_decode_ktile == _CONST_2048
    if wide and (
        group != _CONST_16
        or heads != _CONST_6
        or cache_dtype != jnp.float8_e4m3fn
        or page_tokens != _CONST_512
    ):
        _err_msg = (
            "mla_decode_ktile=2048 requires DCP16, six heads per rank "
            "and FP8 e4m3fn KV with 512-token pages"
        )
        raise ValueError(_err_msg)
    if cache_dtype == jnp.float8_e4m3fn and batch in (8, 16, 32, 64):
        span = (
            4
            if wide and batch == _CONST_32 and sequence_rows == 1
            else min(span, SPAN * PAGE_TOKENS // page_tokens)
        )
    else:
        span = span * PAGE_TOKENS // page_tokens
    span = max(1, span) * staged_folds(batch, heads, sequence_rows, cache_dtype)
    pages = (
        (buffers, span * page_tokens, LANES)
        if native_pages(batch, heads, sequence_rows, cache_dtype)
        else (buffers, span * page_tokens // packing, packing, LANES)
    )
    held = (
        chunk_layout(batch, heads, chunk_rows=chunk_rows)[0]
        if chunked(batch, force_chunks)
        else batch
    )  # rows whose attention state is live at once
    return (
        tpu.VMEM((held, tile, LANES), cache_dtype),
        tpu.VMEM(pages, cache_dtype),
        tpu.SemaphoreType.DMA((buffers,)),
        tpu.VMEM((held, tile, 1), jnp.float32),
        tpu.VMEM((held, tile, 1), jnp.float32),
        tpu.VMEM((held, tile, LATENT), jnp.float32),
        *(
            (
                tpu.VMEM(
                    chunk_rows_shape(batch, LANES, heads, chunk_rows=chunk_rows),
                    jnp.float32,
                ),
                tpu.VMEM(
                    chunk_rows_shape(batch, LANES, heads, chunk_rows=chunk_rows),
                    jnp.float32,
                ),
                tpu.VMEM(
                    chunk_rows_shape(batch, LATENT, heads, chunk_rows=chunk_rows),
                    jnp.float32,
                ),
            )
            if chunked(batch, force_chunks)
            else (
                tpu.VMEM((rows, LANES), jnp.float32),
                tpu.VMEM((rows, LANES), jnp.float32),
                tpu.VMEM((8, 128), jnp.float32),
            )
        ),
        tpu.SMEM((batch if _lookup_rows(batch, heads) else 1,), jnp.int32),
        tpu.SMEM((batch if _lookup_rows(batch, heads) else 1,), jnp.int32),
        tpu.VMEM((held, tile, 1), jnp.float32),
    )


def head_rows(
    batch: _Dynamic, group: _Dynamic = GROUP, heads: _Dynamic = HEADS_PER_RANK
) -> _Dynamic:
    """Rows of a step's score tile: the group's head slots, rounded up to 16."""
    slots = packet_layout(batch, heads)[0] * group
    return -(-slots // 16) * 16


def chunk_rows_shape(
    batch: _Dynamic,
    lanes: _Dynamic,
    heads: _Dynamic = HEADS_PER_RANK,
    *,
    chunk_rows: _Dynamic = None,
) -> _Dynamic:
    """Chunked calls: per-chunk row buffers ``[chunks, rows padded to 8,.

    lanes]`` (a chunk's packet rows, ``slots x chunk``, need not be a
    multiple of the f32 tile; chunks are indexed on the leading dim).
    """
    chunk = chunk_layout(batch, heads, chunk_rows=chunk_rows)[0]
    rows = packet_layout(batch, heads)[0] * chunk
    return (batch // chunk, -(-rows // 8) * 8, lanes)


SCOPED_NAMES = (
    "dcp query columns",
    "dcp latent pages",
    "dcp page semaphores",
    "dcp maximum",
    "dcp denominator",
    "dcp accumulator",
    "dcp own rows",
    "dcp new rows",
    "dcp merged",
    "dcp lookup bounds",
    "dcp lookup counts",
    "dcp query scales",
)


class DcpAttention:
    """One DCP attention invocation for ``batch`` rows at ``position_ref``."""

    _attend_chunked: Any = None

    def _init_buffers(self, skip: _Dynamic) -> None:
        """Initialize workspace buffer geometry and rank indexing."""
        workspace = self.ws
        cache = self.cache
        self.buffers = workspace.pages.shape[0]
        self.page_tokens = cache.page_tokens
        self.fp8 = cache.dtype == jnp.float8_e4m3fn
        self.packing = 4 // jnp.dtype(cache.dtype).itemsize
        self.page_rows = self.page_tokens // self.packing
        self.native = workspace.pages.ndim == _CONST_3
        self.span = workspace.pages.shape[1] // (
            self.page_tokens if self.native else self.page_rows
        )
        self.folds = staged_folds(
            self.batch, self.heads, self.sequence_rows, cache.dtype
        )
        self.fold_tokens = self.span * self.page_tokens // self.folds
        self.skip = set(skip)
        rank = jax.lax.axis_index("tp")
        self.member = rank % self.group
        self.base = rank - self.member
        self.owner = self.member if cache.owner is None else cache.owner

    def _init_counts(self) -> None:
        """Initialize page counts and step offsets."""
        self.counts, starts = [], [0]
        for b in range(self.batch):
            limit = self._limit(b) + (
                self.sequence_rows - 1 if self.cache_verify_rows else 0
            )
            needed = (limit + self.page_tokens - 1) // self.page_tokens
            count = jnp.maximum(needed - self.owner + self.group - 1, 0) // self.group
            self.counts.append(
                count if not self.shared or b % self.sequence_rows == 0 else 0
            )
            starts.append(starts[-1] + (self.counts[-1] + self.span - 1) // self.span)
        self.starts = tuple(starts)
        self.total = starts[-1]
        if _lookup_rows(self.batch, self.heads):
            for b in range(self.batch):
                self.ws.lookup_bounds[b] = starts[b]
                self.ws.lookup_counts[b] = self.counts[b]

    def __init__(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Execute init logic."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            cache,
            workspace,
            position_ref,
            batch,
            skip,
            sequence_rows,
            heads,
            chunk_rows,
            cache_verify_rows,
        ) = (
            _p_next(_p_iter, _p_get("cache")),
            _p_next(_p_iter, _p_get("workspace")),
            _p_next(_p_iter, _p_get("position_ref")),
            _p_next(_p_iter, _p_get("batch")),
            _p_get("skip", ()),
            _p_get("sequence_rows", 1),
            _p_get("heads", HEADS_PER_RANK),
            _p_get("chunk_rows"),
            _p_kw.get("cache_verify_rows", False),
        )
        if batch > MAX_ROWS:
            chunk = chunk_layout(batch, heads, chunk_rows=chunk_rows)[0]
            if chunk % sequence_rows:
                _err_msg = (
                    f"DCP chunks of {chunk} rows must hold whole "
                    f"{sequence_rows}-row sequences"
                )
                raise ValueError(_err_msg)
        if sequence_rows < 1 or batch % sequence_rows:
            _err_msg = "sequence_rows must divide the batch"
            raise ValueError(_err_msg)
        if cache.group not in (8, 16, 32):
            _err_msg = "DCP groups are 8, 16 or 32 consecutive ranks"
            raise ValueError(_err_msg)
        self.cache, self.ws, self.position_ref, self.batch = (
            cache,
            workspace,
            position_ref,
            batch,
        )
        self.sequence_rows = sequence_rows
        self.chunk_rows = chunk_rows
        self.cache_verify_rows = cache_verify_rows
        self.heads = heads
        self.slots, self.groups = packet_layout(batch, heads)
        self.group = cache.group
        self.tile = workspace.queries.shape[1]
        if self.group * self.slots > self.tile:
            _err_msg = (
                f"{self.group} ranks x {self.slots} head slots exceed {self.tile} rows"
            )
            raise ValueError(_err_msg)
        self._init_buffers(skip)
        self.shared = sequence_rows > 1 and all(
            first % sequence_rows == 0 and end % sequence_rows == 0
            for first, end in self.row_groups()
        )
        if cache_verify_rows and not (self.shared and self.fp8):
            _err_msg = "cached verify rows need shared FP8 sequences"
            raise ValueError(_err_msg)
        self._init_counts()

    def _limit(self, row: _Dynamic) -> _Dynamic:
        """Pool tokens ``row`` may see: those below its position, or below.

        its sequence's first position (``sequence_rows > 1``); ``row`` may be
        traced.
        """
        if self.sequence_rows == 1:
            return self.position_ref[row]
        return self.position_ref[row // self.sequence_rows * self.sequence_rows]

    def _locate(self, step: _Dynamic) -> _Dynamic:
        """Execute locate logic."""
        if _lookup_rows(self.batch, self.heads):
            row = jnp.int32(0)
            for bit in range((self.batch - 1).bit_length() - 1, -1, -1):
                candidate = jnp.minimum(row + (1 << bit), self.batch - 1)
                row = jnp.where(
                    step >= self.ws.lookup_bounds[candidate], candidate, row
                )
            return row, step - self.ws.lookup_bounds[row], self.ws.lookup_counts[row]
        row, first, count = 0, 0, self.counts[0]
        for b in range(1, self.batch):
            later = step >= self.starts[b]
            row = jnp.where(later, b, row)
            first = jnp.where(later, self.starts[b], first)
            count = jnp.where(later, self.counts[b], count)
        return row, step - first, count

    def _start_step(self, step: _Dynamic) -> _Dynamic:
        """The step's ``span`` page DMAs; pages past the row's last re-read its.

        last page (masked by position).
        """
        row, local_step, count = self._locate(step)
        slot = step % self.buffers
        if self.native:
            self.cache.start_native_copies(
                row,
                [
                    jnp.minimum(local_step * self.span + index, count - 1)
                    for index in range(self.span)
                ],
                self.owner,
                [
                    self.ws.pages.at[
                        slot, pl.ds(index * self.page_tokens, self.page_tokens)
                    ]
                    for index in range(self.span)
                ],
                self.ws.page_semaphores.at[slot],
            )
            return
        for index in range(self.span):
            local = jnp.minimum(local_step * self.span + index, count - 1)
            self.cache.start_page_copy(
                row,
                local,
                self.owner,
                self.ws.pages.at[slot, pl.ds(index * self.page_rows, self.page_rows)],
                self.ws.page_semaphores.at[slot],
            )

    def _wait_step(self, step: _Dynamic) -> _Dynamic:
        """Execute wait step logic."""
        slot = step % self.buffers
        tpu.make_async_copy(
            self.ws.pages.at[slot],
            self.ws.pages.at[slot],
            self.ws.page_semaphores.at[slot],
        ).wait()

    def _attend_shared_step(
        self, row: _Dynamic, base: _Dynamic, visible: _Dynamic, load: _Dynamic
    ) -> None:
        """Handle shared FP8 step folding."""
        if self.group_fold and self.sequence_rows in (4, 8):
            active = self.sequence_rows - (1 if self.lse_in_padding else 0)
            first_rows = min(4, active)
            first = self._group_scores(load, visible, row - base, first_rows)
            if active > _CONST_4:
                second = self._group_scores(load, visible, row - base + 4, active - 4)
                self._group_fold(first, load, row - base, first_rows)
                self._group_fold(second, load, row - base + 4, active - 4)
            else:
                self._group_fold(first, load, row - base, first_rows)
        elif _grouped_spec(self.batch, self.group, self.heads, self.sequence_rows):

            def fold_group(g: _Dynamic, score: _Dynamic) -> _Dynamic:
                """Execute fold group logic."""
                upcoming = self._group_scores(
                    load, visible, row - base + (g + 1) * 4, 4
                )
                self._group_fold(score, load, row - base + g * 4, 4)
                return upcoming

            final = jax.lax.fori_loop(
                0,
                self.sequence_rows // 4 - 1,
                fold_group,
                self._group_scores(load, visible, row - base, 4),
            )
            self._group_fold(
                final,
                load,
                row - base + self.sequence_rows - 4,
                4,
            )

    def _attend_step(self, step: _Dynamic, base: _Dynamic = 0) -> _Dynamic:
        """One step of the local attention: wait for its pages, prefetch.

        ahead, and fold the pages' scores into its row's running softmax
        (held at row - ``base``: the chunk's first row).
        """
        self._wait_step(step)

        @pl.when(step + self.buffers - 1 < self.total)
        def prefetch() -> _Dynamic:
            """Execute prefetch logic."""
            self._start_step(step + self.buffers - 1)

        if "pagecompute" in self.skip:
            return None
        row, visible, load = self._step_pages(step)
        if self.cache_verify_rows:
            self._attend_cached_verify(step, row, load, base)
        elif self.shared and self.fp8:
            self._attend_shared_step(row, base, visible, load)
        elif self.shared and (self.fp8 or self.batch in (16, 32)):

            def fold_row(r: _Dynamic, score: _Dynamic) -> _Dynamic:
                """Execute fold row logic."""
                upcoming = self._scores(load, visible, row - base + r + 1)
                self._fold_scores(score, load, row - base + r)
                return upcoming

            active = self.sequence_rows - (1 if self.lse_in_padding else 0)
            final = jax.lax.fori_loop(
                0, active - 1, fold_row, self._scores(load, visible, row - base)
            )
            self._fold_scores(final, load, row - base + active - 1)
        elif self.shared:

            @pl.loop(0, self.sequence_rows - (1 if self.lse_in_padding else 0))
            def fold_row(r: _Dynamic) -> _Dynamic:
                """Execute fold row logic."""
                self._fold_pages(load, visible, row - base + r)
        else:
            self._fold_pages(load, visible, row - base)
        return None

    def _cache_verify_row(
        self, slot: _Dynamic, local_step: _Dynamic, row: _Dynamic
    ) -> _Dynamic:
        """Execute cache verify row logic."""
        position = self.position_ref[row]
        page = position // self.page_tokens
        local_page = page // self.group - local_step * self.span

        @pl.when(
            (page % self.group == self.owner)
            & (local_page >= 0)
            & (local_page < self.span)
        )
        def write() -> _Dynamic:
            """Execute write logic."""
            if chunked(self.batch, self.force_chunked):
                chunk = chunk_layout(
                    self.batch, self.heads, chunk_rows=self.chunk_rows
                )[0]
                latent = self.ws.new_rows[row // chunk, self.slots * (row % chunk)]
            else:
                latent = self.ws.new_rows[self.slots * row]
            offset = local_page * self.page_tokens + position % self.page_tokens
            tile_tokens = 8 * self.packing
            first = pl.multiple_of(offset // tile_tokens * 8, 8)
            words = (
                self.ws.pages.bitcast(jnp.uint32).at[slot]
                if self.native
                else self.ws.pages.bitcast(jnp.uint32)
                .at[slot]
                .reshape(self.span * self.page_rows, LANES)
            )
            block = pltpu.bitcast(words[pl.ds(first, 8)], self.cache.dtype)
            token = jax.lax.broadcasted_iota(jnp.int32, block.shape, 0)
            block = jnp.where(
                token == offset % tile_tokens,
                latent[None, :].astype(self.cache.dtype),
                block,
            )
            words[pl.ds(first, 8)] = pltpu.bitcast(block, jnp.uint32)

    def _attend_cached_verify(
        self, step: _Dynamic, row: _Dynamic, load: _Dynamic, base: _Dynamic
    ) -> _Dynamic:
        # Earlier verify tokens use the cache's FP8 QK/PV, like sequential decode.
        """Execute attend cached verify logic."""
        _, local_step, _ = self._locate(step)

        @pl.loop(0, self.sequence_rows - 1)
        def cache(r: _Dynamic) -> _Dynamic:
            """Execute cache logic."""
            self._cache_verify_row(step % self.buffers, local_step, row + r)

        offset = jax.lax.broadcasted_iota(
            jnp.int32, (1, self.span * self.page_tokens), 1
        )
        page = (
            local_step * self.span + offset // self.page_tokens
        ) * self.group + self.owner
        token = page * self.page_tokens + offset % self.page_tokens
        if _grouped_spec(self.batch, self.group, self.heads, self.sequence_rows):
            for first in range(0, self.sequence_rows, 4):
                query = (
                    jax.lax.broadcasted_iota(jnp.int32, (4 * self.tile, 1), 0)
                    // self.tile
                )
                visible = token < self.position_ref[row] + first + query
                score = self._group_scores(load, visible, row - base + first, 4)
                self._group_fold(score, load, row - base + first, 4)
        else:

            @pl.loop(0, self.sequence_rows)
            def query(r: _Dynamic) -> _Dynamic:
                """Execute query logic."""
                self._fold_pages(
                    load, token < self.position_ref[row + r], row - base + r
                )

    def _step_pages(self, step: _Dynamic) -> _Dynamic:
        """Step ``step``'s row, the mask of its visible tokens and a reader of.

        its pages (their stale bytes zeroed first).
        """
        row, local_step, _ = self._locate(step)
        offset = jax.lax.broadcasted_iota(
            jnp.int32, (1, self.span * self.page_tokens), 1
        )  # tokens on lanes
        global_page = (
            local_step * self.span + offset // self.page_tokens
        ) * self.group + self.owner
        token = global_page * self.page_tokens + offset % self.page_tokens
        visible = token < self._limit(row)
        slot = step % self.buffers
        self._clean_step(slot, row, local_step)
        return row, visible, self._pages(slot)

    def _pages(self, slot: _Dynamic) -> _Dynamic:
        """``load(lanes, tokens=None)``: buffer ``slot``'s pages in the cache.

        dtype (``tokens``, a range of them: native buffers only).
        """
        if self.native:

            def native(lanes: _Dynamic, tokens: _Dynamic = None) -> _Dynamic:
                """Execute native logic."""
                return self.ws.pages[
                    slot, slice(None) if tokens is None else tokens, lanes
                ]

            return native
        words = (
            self.ws.pages.bitcast(jnp.uint32)
            .at[slot]
            .reshape(self.span * self.page_rows, LANES)
        )

        def packed(lanes: _Dynamic, tokens: _Dynamic = None) -> _Dynamic:
            """Execute packed logic."""
            if tokens is not None:
                _err_msg = "token ranges need native pages"
                raise AssertionError(_err_msg)
            return pltpu.bitcast(words[:, lanes], self.cache.dtype)

        return packed

    def _attend_range_split(self, chunk: _Dynamic, base: _Dynamic) -> None:
        """Row-by-row attention for split ranges."""

        @pl.loop(0, chunk)
        def row_loop(j: _Dynamic) -> None:
            """Execute row loop logic."""
            row = base + j
            begin = self.ws.lookup_bounds[row]
            count = self.ws.lookup_counts[row]
            full_end = begin + count // self.span
            self._attend_pipeline(begin, full_end, base, peel=True)
            if self.folds > 1:
                pl.when(count % self.span != 0)(
                    lambda: self._staged_tail(full_end, row, count, base)
                )
                return

            for tail_pages in range(1, self.span):

                @pl.when(count % self.span == tail_pages)
                def tail(tail_pages: _Dynamic = tail_pages) -> None:
                    """Execute tail logic."""
                    self._wait_step(full_end)

                    @pl.when(full_end + self.buffers - 1 < self.total)
                    def prefetch() -> None:
                        """Execute prefetch logic."""
                        self._start_step(full_end + self.buffers - 1)

                    _, visible, load = self._step_pages(full_end)
                    score = self._scores(
                        load,
                        visible,
                        row - base,
                        tail_tokens=tail_pages * self.page_tokens,
                    )
                    self._fold_scores(
                        score,
                        load,
                        row - base,
                        tail_tokens=tail_pages * self.page_tokens,
                    )

    def _attend_range(self, first: _Dynamic, end: _Dynamic, base: _Dynamic) -> _Dynamic:
        """Execute attend range logic."""
        skip_attend = self.folds == 1 and (
            self.sequence_rows != 1
            or (
                self.heads != HEADS_PER_RANK
                and not (
                    self.group == _CONST_16
                    and self.heads == _CONST_6
                    and self.batch == _CONST_32
                )
            )
            or self.batch not in (32, 64)
            or self.ws.gathered.dtype != jnp.uint8
            or (
                self.span != _CONST_2
                and not (
                    _config().mla_decode_ktile == _CONST_2048 and self.span == _CONST_4
                )
            )
        )
        if skip_attend or "attend" in self.skip or "pagecompute" in self.skip:
            self._attend_pipeline(first, end, base)
            return None
        chunk = chunk_layout(self.batch, self.heads, chunk_rows=self.chunk_rows)[0]
        if self.folds > 1:
            self._attend_range_split(chunk, base)
            return None
        remainder = jnp.int32(0)
        for j in range(chunk):
            remainder = remainder | (self.ws.lookup_counts[base + j] % self.span)

        @pl.when(remainder == 0)
        def whole() -> None:
            """Execute whole logic."""
            self._attend_pipeline(first, end, base)

        pl.when(remainder != 0)(lambda: self._attend_range_split(chunk, base))
        return None

    def _staged_tail(
        self, step: _Dynamic, row: _Dynamic, count: _Dynamic, base: _Dynamic
    ) -> _Dynamic:
        """A staged row's partial last step ``step`` (its first.

        ``count % span`` pages): a whole 1024-token part if it has one, then a
        one-page remainder, each folded as a 1024-token step folds it (two
        static shapes for the 1 .. span - 1 pages).
        """
        self._wait_step(step)

        @pl.when(step + self.buffers - 1 < self.total)
        def prefetch() -> _Dynamic:
            """Execute prefetch logic."""
            self._start_step(step + self.buffers - 1)

        held, slot = row - base, step % self.buffers
        local_step, pages = count // self.span, count % self.span
        part_pages = self.span // self.folds
        self._clean_step(
            slot,
            row,
            local_step,
            (self.span - 1) * self.page_tokens,
            end=pages * self.page_tokens,
        )
        load = self._pages(slot)
        limit = self._limit(row)

        def visible(first_page: _Dynamic, tokens: _Dynamic) -> _Dynamic:
            """Execute visible logic."""
            offset = jax.lax.broadcasted_iota(jnp.int32, (1, tokens), 1)
            global_page = (
                local_step * self.span + first_page + offset // self.page_tokens
            ) * self.group + self.owner
            return global_page * self.page_tokens + offset % self.page_tokens < limit

        @pl.when(pages >= part_pages)
        def whole() -> _Dynamic:
            """Execute whole logic."""
            tokens = pl.ds(0, self.fold_tokens)
            score = self._scores(
                lambda lanes: load(lanes, tokens), visible(0, self.fold_tokens), held
            )
            self._fold_part(score, lambda: load(pl.ds(0, LATENT), tokens), held)

        @pl.when(pages % part_pages != 0)
        def remainder() -> _Dynamic:
            """Execute remainder logic."""
            first = pages - 1
            tokens = pl.ds(
                pl.multiple_of(first * self.page_tokens, self.page_tokens),
                self.page_tokens,
            )
            score = self._scores(
                lambda lanes: load(lanes, tokens),
                visible(first, self.page_tokens),
                held,
            )
            score = jnp.pad(
                score,
                ((0, 0), (0, self.fold_tokens - self.page_tokens)),
                constant_values=-jnp.inf,
            )
            self._fold_part(
                score, lambda: load(pl.ds(0, LATENT), tokens), held, self.page_tokens
            )

    def _attend_pipeline(
        self, first: _Dynamic, end: _Dynamic, base: _Dynamic, *, peel: _Dynamic = False
    ) -> _Dynamic:
        """Steps ``first .. end - 1`` of a chunk whose state rows start at row.

        ``base``. Plain decode pipelines them: step s + 1's scores are issued
        before step s's softmax and P V, so the next QK overlaps them, and a
        step's buffer is refilled only after its P V. Normally the last step
        recomputes its own scores (unused), avoiding a branch before QK.
        ``peel`` is static: the split-tail path finishes PV outside the loop
        instead, before issuing the separate half-width QK/PV.
        """
        if self.sequence_rows > 1 or "pagecompute" in self.skip:
            pl.loop(first, end)(lambda step: self._attend_step(step, base=base))
            return

        def scores(step: _Dynamic) -> _Dynamic:
            """Execute scores logic."""
            row, visible, load = self._step_pages(step)
            return self._scores(load, visible, row - base), row

        @pl.when(first < end)
        def attend() -> _Dynamic:
            """Execute attend logic."""
            self._wait_step(first)

            @pl.when(first + self.buffers - 1 < self.total)
            def prefetch_first() -> _Dynamic:
                """Execute prefetch first logic."""
                self._start_step(first + self.buffers - 1)

            def fold(step: _Dynamic, carry: _Dynamic) -> _Dynamic:
                """Execute fold logic."""
                score, row = carry
                more = step + 1 < end
                if peel:
                    self._wait_step(step + 1)
                    upcoming = scores(step + 1)
                else:

                    @pl.when(more)
                    def wait_next() -> _Dynamic:
                        """Execute wait next logic."""
                        self._wait_step(step + 1)

                    upcoming = scores(jnp.minimum(step + 1, end - 1))
                self._fold_scores(score, self._pages(step % self.buffers), row - base)

                @pl.when((True if peel else more) & (step + self.buffers < self.total))
                def prefetch() -> _Dynamic:
                    """Execute prefetch logic."""
                    self._start_step(step + self.buffers)

                return upcoming

            carry = jax.lax.fori_loop(first, end - int(peel), fold, scores(first))
            if peel:
                score, row = carry
                self._fold_scores(
                    score, self._pages((end - 1) % self.buffers), row - base
                )

    def _store_query(self, held: _Dynamic, query: _Dynamic) -> _Dynamic:
        """Quantize once per query, outside the streamed page loop.

        Scale each absorbed head independently; its latent/RoPE columns share
        one scale, so the QK dot needs only one FP32 multiply per score row.
        Padding heads are zero and get scale one. KV uses scale one.
        """
        if self.fp8:
            maximum = jnp.max(jnp.abs(query), axis=1, keepdims=True)
            scale = jnp.where(maximum > 0, maximum / 448.0, 1.0)
            self.ws.queries[held] = (query / scale).astype(jnp.float8_e4m3fn)
            self.ws.query_scale[held] = scale
        else:
            self.ws.queries[held] = query.astype(jnp.bfloat16)

    def _group_scores(
        self, load: _Dynamic, visible: _Dynamic, held: _Dynamic, count: _Dynamic
    ) -> _Dynamic:
        """QK for adjacent queries, with independent head rows and scales."""
        query = jnp.concatenate(
            [self.ws.queries[held + i] for i in range(count)], axis=0
        )
        scales = jnp.concatenate(
            [self.ws.query_scale[held + i] for i in range(count)], axis=0
        )
        latent = load(pl.ds(0, LANES))
        score = jax.lax.dot_general(
            query, latent, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
        )
        return jnp.where(visible, score * scales, -jnp.inf)

    def _group_fold(
        self, score: _Dynamic, load: _Dynamic, held: _Dynamic, count: _Dynamic
    ) -> _Dynamic:
        """Fold grouped PV back into each query's own online state."""
        ws = self.ws
        previous = jnp.concatenate([ws.maximum[held + i] for i in range(count)], axis=0)
        new_maximum = jnp.maximum(previous, jnp.max(score, axis=1, keepdims=True))
        scale = jnp.exp(previous - new_maximum)
        probabilities = jnp.exp(score - new_maximum)
        totals = jnp.sum(probabilities, axis=1, keepdims=True)
        high = (probabilities * 256).astype(jnp.float8_e4m3fn)
        values = load(pl.ds(0, LATENT))
        weighted = jax.lax.dot_general(
            high, values, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
        ) * (1.0 / 256)
        for i in range(count):
            rows = slice(i * self.tile, (i + 1) * self.tile)
            ws.denominator[held + i] = (
                ws.denominator[held + i] * scale[rows] + totals[rows]
            )
            ws.maximum[held + i] = new_maximum[rows]
            ws.accumulator[held + i] = (
                ws.accumulator[held + i] * scale[rows] + weighted[rows]
            )

    def _fold_pages(
        self, load: _Dynamic, visible: _Dynamic, held: _Dynamic
    ) -> _Dynamic:
        """Fold a step's pages (``load(lanes)`` reads them) into the running.

        softmax of state row ``held``.
        """
        self._fold_scores(self._scores(load, visible, held), load, held)

    def _scores(
        self,
        load: _Dynamic,
        visible: _Dynamic,
        held: _Dynamic,
        *,
        tail_tokens: _Dynamic = 0,
    ) -> _Dynamic:
        """State row ``held``'s masked scores of a step's pages."""
        # Heads-major (flash-attention layout): scores [head_rows head slots,
        # span * page_tokens tokens] = Q C^T, so the probabilities feed P V as is.
        latent = load(pl.ds(0, LANES))
        if tail_tokens:
            latent = latent[:tail_tokens]
        score = jax.lax.dot_general(
            self.ws.queries[held],
            latent,
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        if self.fp8:
            score = score * self.ws.query_scale[held]
        if tail_tokens:
            score = jnp.pad(
                score,
                ((0, 0), (0, self.span * self.page_tokens - tail_tokens)),
                constant_values=-jnp.inf,
            )
        return jnp.where(visible, score, -jnp.inf)

    def _fold_scores(
        self,
        score: _Dynamic,
        load: _Dynamic,
        held: _Dynamic,
        *,
        tail_tokens: _Dynamic = 0,
    ) -> _Dynamic:
        """Fold a step's ``score`` into the running softmax of state row.

        ``held``; the values are loaded after the softmax (a load, not a
        live 320-vreg value across it). A staged step folds its parts in
        order.
        """
        if self.folds == 1:
            self._fold_part(score, lambda: load(pl.ds(0, LATENT)), held, tail_tokens)
            return
        for first in range(0, self.span * self.page_tokens, self.fold_tokens):
            tokens = pl.ds(first, self.fold_tokens)
            self._fold_part(
                score[:, first : first + self.fold_tokens],
                lambda tokens=tokens: load(pl.ds(0, LATENT), tokens),
                held,
            )

    def _fold_part(
        self,
        score: _Dynamic,
        values: _Dynamic,
        held: _Dynamic,
        tail_tokens: _Dynamic = 0,
    ) -> _Dynamic:
        """``_fold_scores`` of one fold; ``values()`` loads its values."""
        ws = self.ws
        previous = ws.maximum[held]  # [head_rows, 1]
        new_maximum = jnp.maximum(previous, jnp.max(score, axis=1, keepdims=True))
        scale = jnp.exp(previous - new_maximum)
        probabilities = jnp.exp(score - new_maximum)  # [head_rows, span * page_tokens]
        ws.denominator[held] = ws.denominator[held] * scale + jnp.sum(
            probabilities, axis=1, keepdims=True
        )
        ws.maximum[held] = new_maximum
        # Keep FP32 probabilities for the denominator; use scaled E4M3 for PV.
        high = (
            (probabilities * 256).astype(jnp.float8_e4m3fn)
            if self.fp8
            else probabilities.astype(jnp.bfloat16)
        )
        values = values()
        if tail_tokens:
            values = values[:tail_tokens]
            high = high[:, :tail_tokens]
        weighted = jax.lax.dot_general(
            high, values, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
        )
        if self.fp8:
            weighted = weighted * (1.0 / 256)
        ws.accumulator[held] = ws.accumulator[held] * scale + weighted

    def _clean_step(
        self,
        slot: _Dynamic,
        row: _Dynamic,
        local_step: _Dynamic,
        tokens: _Dynamic = None,
        end: _Dynamic = None,
    ) -> _Dynamic:
        """A step's pages as 32-bit rows ``[span * page_rows, 640]`` (a word.

        packs ``packing`` tokens), with vLLM's stale bytes zeroed in place:
        lanes 576:640 of every token and every token at or past the row's
        limit (a step's visible offsets are a prefix). Stores into the buffer,
        not a masked copy of the whole tile as a value (which spills).
        ``tokens`` (static) and ``end`` (traced, at most ``tokens``) bound the
        zeroing to the tokens a tail reads.
        """
        ws, rows = self.ws, tokens or self.span * self.page_tokens
        words = (
            ws.pages.bitcast(jnp.uint32).at[slot]
            if self.native
            else ws.pages.bitcast(jnp.uint32)
            .at[slot]
            .reshape(self.span * self.page_rows, LANES)
        )
        if (
            self.batch in (16, 32, 48, 64)
            or self.heads != HEADS_PER_RANK
            or (self.batch, self.sequence_rows) == (8, 1)
        ):
            pad_shape = (rows // self.packing, LANES - LATENT)
            pad_lane = jax.lax.broadcasted_iota(jnp.int32, pad_shape, 1)
            tpu.store(
                words.at[:, pl.ds(LATENT, LANES - LATENT)]
                if tokens is None
                else words.at[
                    pl.ds(0, rows // self.packing), pl.ds(LATENT, LANES - LATENT)
                ],
                jnp.zeros(pad_shape, jnp.uint32),
                mask=pad_lane >= _CONST_64,
            )
        else:
            pad = words[
                :, pl.ds(LATENT, LANES - LATENT)
            ]  # a word's two tokens share its lane
            pad_lane = jax.lax.broadcasted_iota(jnp.int32, pad.shape, 1)
            words[:, pl.ds(LATENT, LANES - LATENT)] = jnp.where(
                pad_lane < _CONST_64, pad, jnp.zeros((), pad.dtype)
            )
        limit = self._limit(row)
        cut: _Dynamic = jnp.int32(0)  # visible offsets
        for k in range(rows // self.page_tokens):
            first_token = (
                (local_step * self.span + k) * self.group + self.owner
            ) * self.page_tokens
            cut = cut + jnp.clip(limit - first_token, 0, self.page_tokens)
        tile_tokens = 8 * self.packing
        tile = cut // tile_tokens  # one native eight-word-row tile

        @pl.when(tile < rows // tile_tokens)
        def boundary() -> _Dynamic:  # the tile holding the cut
            """Execute boundary logic."""
            at = pl.ds(pl.multiple_of(tile * 8, 8), 8)
            block = pltpu.bitcast(words[at, :], self.cache.dtype)
            offset = tile * tile_tokens + jax.lax.broadcasted_iota(
                jnp.int32, block.shape, 0
            )
            words[at, :] = pltpu.bitcast(
                jnp.where(offset < cut, block, jnp.zeros((), block.dtype)), jnp.uint32
            )

        @pl.loop(tile + 1, (rows if end is None else end) // tile_tokens)
        def invisible(t: _Dynamic) -> _Dynamic:
            """Execute invisible logic."""
            words[pl.ds(pl.multiple_of(t * 8, 8), 8), :] = jnp.zeros(
                (8, LANES), jnp.uint32
            )

        return words

    def _pack_partials(
        self, first: _Dynamic, end: _Dynamic, *, chunk: _Dynamic = None
    ) -> _Dynamic:
        """Rows ``first..end``: normalized outputs and log-sum-exps into each.

        peer's send packet (BF16; the lse as a high and a low part).
        """
        _ = chunk
        ws, group = self.ws, self.group
        lses = {}
        for held in range(first, end):
            total_weight = ws.denominator[held]  # [head_rows, 1]
            row_maximum = ws.maximum[held]
            has = total_weight > 0
            normalized = jnp.where(
                has, ws.accumulator[held] / jnp.where(has, total_weight, 1.0), 0.0
            )
            lse = jnp.where(
                has, row_maximum + jnp.log(jnp.where(has, total_weight, 1.0)), NEG
            )
            target = pl.ds(self.slots * held, self.slots)
            lse_high = lse.astype(jnp.bfloat16).astype(jnp.float32)
            lane = jax.lax.broadcasted_iota(
                jnp.int32, (lse.shape[0], LANES - LATENT), 1
            )
            lse_lanes = jnp.where(
                lane == 0, lse_high, jnp.where(lane == 1, lse - lse_high, 0.0)
            )
            lses[held] = (lse_high, lse - lse_high)
            for peer in range(group):
                heads_of_peer = slice(self.slots * peer, self.slots * (peer + 1))
                ws.send[self._send_index(peer), target, :] = jnp.concatenate(
                    (normalized[heads_of_peer], lse_lanes[heads_of_peer]), axis=1
                ).astype(jnp.bfloat16)
        if self.lse_in_padding:
            self._pack_lse_rows(lses, first, end)

    def _pack_lse_rows(
        self, lses: _Dynamic, first: _Dynamic, end: _Dynamic
    ) -> _Dynamic:
        """``lse_in_padding``: each whole sequence's log-sum-exps (rows but its.

        last, a padding row) into that last row's two packet rows: packet row
        ``r = slots j + h`` of the sequence at lane ``r``, the high parts in
        the first, the low parts in the second (BF16, as lanes 512:514 hold
        them). Each peer's [1, 128] lane vector is a 0/1 product, exact.
        """
        ws, group, slots, rows = self.ws, self.group, self.slots, self.sequence_rows
        if slots < _CONST_2 or slots * (rows - 1) > _CONST_128 or (end - first) % rows:
            _err_msg = (
                "lse_in_padding takes two or more head slots, "
                "one lane each, and whole sequences"
            )
            raise ValueError(_err_msg)
        head_rows = lses[first][0].shape[0]
        slot_row = jax.lax.broadcasted_iota(jnp.int32, (head_rows, 128), 0)
        lane = jax.lax.broadcasted_iota(jnp.int32, (head_rows, 128), 1)
        index = jax.lax.broadcasted_iota(jnp.int32, (group, head_rows), 0)  # send slot
        if self._lane_major:
            hosts = group // HOST_LANES
            index = (index % hosts) * HOST_LANES + index // hosts  # the slot's peer
        owner = (
            jax.lax.broadcasted_iota(jnp.int32, (group, head_rows), 1) // slots == index
        ).astype(jnp.bfloat16)
        for start in range(first, end, rows):
            for part in range(2):  # high, low
                spread = jnp.zeros((head_rows, 128), jnp.float32)
                for j in range(rows - 1):
                    value = lses[start + j][part]  # [head_rows, 1]
                    spread = jnp.where(
                        lane == slots * j + slot_row % slots, value, spread
                    )
                vectors = jax.lax.dot_general(
                    owner,
                    spread.astype(jnp.bfloat16),
                    (((1,), (0,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )  # [group, 128]
                row = slots * (start + rows - 1) + part
                ws.send[:, pl.ds(row, 1), pl.ds(0, 128)] = vectors[:, None, :].astype(
                    jnp.bfloat16
                )

    # Profiling: named scopes inside attend (``trace`` their prefix; None: none).
    trace: str | None = None
    # Explicit-keys sequences whose last row is padding (the MLA DSpark
    # draft): partials carry their log-sum-exps in that row, and the combine
    # moves only the 512 latent lanes (a fifth fewer bytes).
    lse_in_padding: bool = False

    def _combine_lanes(self) -> _Dynamic:
        """Execute combine lanes logic."""
        return pl.ds(0, LATENT if self.lse_in_padding else LANES)

    # Chunked explicit-keys calls (lse_in_padding, DCP32): the combine in two
    # stages, (stage-2 receive [ring, hosts, rows, 640], stage-2 send [hosts,
    # rows, 640], DMA semaphores [2, ring]) or None. Stage one stays on the
    # host: each rank sends a host-mate the partials of the owners on its
    # lane (one per host) and merges the eight it receives per owner; stage
    # two sends each merged partial to its owner across hosts, which merges
    # four. Cross-host combine bytes drop eightfold (the cross-host links
    # bound the exchange); the merged partials round to BF16 once more.
    host_combine: _Dynamic = None
    force_chunked = (
        False  # run even one chunk through _attend_chunked (scratch: force_chunks)
    )
    # FP8 explicit-keys sequences: QK/PV for several rows in one
    # dot (the grouped path's arithmetic, per row as the ungrouped fold).
    group_fold = False
    # packing for host_combine: send packet of peer p at (p % 8) hosts + p // 8
    _lane_major = False

    def _send_index(self, peer: _Dynamic) -> _Dynamic:
        """Execute send index logic."""
        if not self._lane_major:
            return peer
        hosts = self.group // HOST_LANES
        return (peer % HOST_LANES) * hosts + peer // HOST_LANES

    def _scope(self, name: _Dynamic) -> _Dynamic:
        """Execute scope logic."""
        return (
            jax.named_scope(self.trace + name)
            if self.trace
            else contextlib.nullcontext()
        )

    def prime(self) -> _Dynamic:
        """Issue the first page DMAs (call as early as the positions allow)."""
        for step in range(self.buffers - 1):

            @pl.when(step < self.total)
            def prime(step: _Dynamic = step) -> _Dynamic:
                """Execute prime logic."""
                self._start_step(step)

    def row_groups(self) -> _Dynamic:
        """``[(first row, end row)]``: the batch in ``packet_layout``'s groups."""
        groups = self.groups
        bounds = [self.batch * g // groups for g in range(groups + 1)]
        return [(bounds[g], bounds[g + 1]) for g in range(groups)]

    def _init_attend_buffers(self, queries: _Dynamic, new_latents: _Dynamic) -> None:
        """Initialize attention query and latent buffers."""
        ws, batch = self.ws, self.batch
        ws.own_rows[...] = jnp.zeros(ws.own_rows.shape, jnp.float32)
        ws.new_rows[...] = jnp.zeros(ws.new_rows.shape, jnp.float32)
        for b in range(batch):
            ws.own_rows[pl.ds(self.slots * b, self.heads), :] = queries[b]
            ws.new_rows[pl.ds(self.slots * b, self.heads), :] = jnp.broadcast_to(
                new_latents[b][None, :], (self.heads, LANES)
            )

    def _send_gather_queries(self, parity: _Dynamic, groups: _Dynamic) -> None:
        """All-gather queries into peer slots."""
        ws, group, member, base = self.ws, self.group, self.member, self.base
        ws.gathered[parity, member] = (
            _pack_fp8_query(ws.own_rows[...])
            if ws.gathered.dtype == jnp.uint8
            else ws.own_rows[...].astype(jnp.bfloat16)
        )
        if "gather" in self.skip:
            return
        for g in range(len(groups)):
            first, end = groups[g]
            row_slice = pl.ds(self.slots * first, self.slots * (end - first))
            for peer in range(group):

                @pl.when(peer != member)
                def send_query(
                    peer: _Dynamic = peer,
                    row_slice: _Dynamic = row_slice,
                    g: _Dynamic = g,
                ) -> None:
                    """Execute send query logic."""
                    tpu.make_async_remote_copy(
                        ws.gathered.at[parity, member, row_slice],
                        ws.gathered.at[parity, member, row_slice],
                        ws.gather_semaphores.at[0, parity, g],
                        ws.gather_semaphores.at[1, parity, g],
                        device_id=(base + peer,),
                        device_id_type=pl.DeviceIdType.MESH,
                    ).start()

    def _attend_single_group(
        self,
        g: int,
        first: int,
        end: int,
        parity: _Dynamic,
    ) -> None:
        """Wait for gather, run local attention over pages, and send partials."""
        ws, group, member, base = self.ws, self.group, self.member, self.base
        heads = group * self.slots
        row_slice = pl.ds(self.slots * first, self.slots * (end - first))
        others = ws.gathered.at[parity, pl.ds(0, group - 1), row_slice]
        if "gather" not in self.skip:
            with self._scope("gather_wait"):
                tpu.make_async_copy(
                    others, others, ws.gather_semaphores.at[0, parity, g]
                ).wait()
                tpu.make_async_copy(
                    others, others, ws.gather_semaphores.at[1, parity, g]
                ).wait()
        local_scope = self._scope("local")
        local_scope.__enter__()
        for b in range(first, end):
            gathered = ws.gathered[
                parity, :, pl.ds(self.slots * b, self.slots), :
            ].reshape(heads, LANES)
            padding = ((0, self.tile - heads), (0, 0))
            if ws.gathered.dtype == jnp.uint8:
                query, scale = _unpack_fp8_query(gathered)
                ws.queries[b] = jnp.pad(query, padding)
                ws.query_scale[b] = jnp.pad(scale, padding, constant_values=1)
            else:
                self._store_query(b, jnp.pad(gathered.astype(jnp.float32), padding))

        if self.heads != HEADS_PER_RANK:
            self._attend_pipeline(
                self.starts[first],
                self.starts[first] if "attend" in self.skip else self.starts[end],
                0,
            )
        else:

            @pl.loop(
                self.starts[first],
                self.starts[first] if "attend" in self.skip else self.starts[end],
            )
            def attend_loop(step: _Dynamic) -> None:
                """Execute attend loop logic."""
                self._attend_step(step)

        self._pack_partials(first, end)
        for peer in range(0 if "combine" in self.skip else group):

            @pl.when(peer != member)
            def send_partial(peer: _Dynamic = peer, g: _Dynamic = g) -> None:
                """Execute send partial logic."""
                tpu.make_async_remote_copy(
                    ws.send.at[peer, row_slice, self._combine_lanes()],
                    ws.receive.at[parity, member, row_slice, self._combine_lanes()],
                    ws.combine_semaphores.at[0, parity, g],
                    ws.combine_semaphores.at[1, parity, g],
                    device_id=(base + peer,),
                    device_id_type=pl.DeviceIdType.MESH,
                ).start()

        ws.receive[parity, member, row_slice] = ws.send[member, row_slice]
        local_scope.__exit__(None, None, None)

    def _wait_and_merge_partials(
        self,
        groups: _Dynamic,
        parity: _Dynamic,
        keys: _Dynamic,
        key_valid: _Dynamic,
        new_latents: _Dynamic = None,
    ) -> _Dynamic:
        """Wait for partials and perform LSE merge."""
        ws, group, batch = self.ws, self.group, self.batch
        with self._scope("combine_wait"):
            for g in range(len(groups)):
                first, end = groups[g]
                row_slice = pl.ds(self.slots * first, self.slots * (end - first))
                others = ws.receive.at[
                    parity, pl.ds(0, group - 1), row_slice, self._combine_lanes()
                ]
                if "combine" not in self.skip:
                    tpu.make_async_copy(
                        others, others, ws.combine_semaphores.at[0, parity, g]
                    ).wait()
                    tpu.make_async_copy(
                        others, others, ws.combine_semaphores.at[1, parity, g]
                    ).wait()
        merge_scope = self._scope("merge")
        merge_scope.__enter__()

        partials = ws.receive[parity].astype(jnp.float32)
        lse = (
            partials[:, :, LATENT : LATENT + 1]
            + partials[:, :, LATENT + 1 : LATENT + 2]
        )
        if keys is not None:
            if not (key_valid is not None):
                _err_msg = "Assertion failed: key_valid is not None"
                raise AssertionError(_err_msg)
            sequences = batch // self.sequence_rows
            rows = self.slots * self.sequence_rows
            merged = jnp.concatenate(
                [
                    _merge_received_keys(
                        ws.receive.at[parity, :, pl.ds(rows * s, rows)],
                        ws.own_rows[pl.ds(rows * s, rows), :],
                        keys[s],
                        key_valid[s],
                        group,
                        lse_rows=self.slots if self.lse_in_padding else 0,
                    )
                    for s in range(sequences)
                ],
                axis=0,
            )
        elif self.sequence_rows == 1 or self.cache_verify_rows:
            new_score = jnp.sum(
                ws.own_rows[...] * ws.new_rows[...], axis=1, keepdims=True
            )
            top = jnp.maximum(jnp.max(lse, axis=0), new_score)
            weights = jnp.exp(lse - top[None])
            new_weight = jnp.exp(new_score - top)
            merged = (
                jnp.sum(weights * partials[:, :, :LATENT], axis=0)
                + new_weight * ws.new_rows[:, :LATENT]
            ) / (jnp.sum(weights, axis=0) + new_weight)
        else:
            latents = new_latents
            scores = jax.lax.dot_general(
                ws.own_rows[...],
                latents,
                (((1,), (1,)), ((), ())),
                precision=jax.lax.Precision.HIGHEST,
                preferred_element_type=jnp.float32,
            )
            slot_row = (
                jax.lax.broadcasted_iota(jnp.int32, scores.shape, 0) // self.slots
            )
            key_row = jax.lax.broadcasted_iota(jnp.int32, scores.shape, 1)
            causal = (
                key_row // self.sequence_rows == slot_row // self.sequence_rows
            ) & (key_row <= slot_row)
            scores = jnp.where(causal, scores, -jnp.inf)
            top = jnp.maximum(
                jnp.max(lse, axis=0), jnp.max(scores, axis=1, keepdims=True)
            )
            weights = jnp.exp(lse - top[None])
            new_weights = jnp.exp(scores - top)
            new_values = jax.lax.dot_general(
                new_weights,
                latents[:, :LATENT],
                (((1,), (0,)), ((), ())),
                precision=jax.lax.Precision.HIGHEST,
                preferred_element_type=jnp.float32,
            )
            merged = (
                jnp.sum(weights * partials[:, :, :LATENT], axis=0) + new_values
            ) / (jnp.sum(weights, axis=0) + jnp.sum(new_weights, axis=1, keepdims=True))
        result = jnp.stack(
            [merged[self.slots * b : self.slots * b + self.heads] for b in range(batch)]
        )
        merge_scope.__exit__(None, None, None)
        return result

    def attend(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
        """``queries`` [B, heads, 640] FP32 absorbed (scale folded in),.

        ``new_latents`` [B, 640] FP32, ``parity`` alternating per invocation
        (MLA layer). ``between`` runs while the query gather is in flight.
        Returns the merged latent-space output [B, heads, 512] FP32.
        """
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            queries,
            new_latents,
            parity,
        ) = (
            _p_next(_p_iter, _p_get("queries")),
            _p_next(_p_iter, _p_get("new_latents")),
            _p_next(_p_iter, _p_get("parity")),
        )
        between = next(_p_iter, _p_kw.get("between"))
        keys = _p_kw.get("keys")
        key_valid = _p_kw.get("key_valid")
        if keys is not None and self.sequence_rows == 1:
            _err_msg = "explicit new keys need multi-row sequences"
            raise ValueError(_err_msg)
        if chunked(self.batch, self.force_chunked):
            return self._attend_chunked(
                queries, new_latents, between, keys=keys, key_valid=key_valid
            )
        groups = self.row_groups()
        self._init_attend_buffers(queries, new_latents)
        self._send_gather_queries(parity, groups)
        if between is not None:
            between()
        self.ws.maximum[...] = jnp.full(self.ws.maximum.shape, -jnp.inf, jnp.float32)
        self.ws.denominator[...] = jnp.zeros(self.ws.denominator.shape, jnp.float32)
        self.ws.accumulator[...] = jnp.zeros(self.ws.accumulator.shape, jnp.float32)
        for g, (first, end) in enumerate(groups):
            self._attend_single_group(g, first, end, parity)
        return self._wait_and_merge_partials(
            groups, parity, keys, key_valid, new_latents
        )

    def _chunked_first_step(self, first: _Dynamic, steps: list[_Dynamic]) -> _Dynamic:
        """Calculate first step offset for chunk."""
        total = jnp.int32(0)
        for b in range(self.batch):
            total = total + jnp.where(b < first, steps[b], 0)
        return total

    def _chunked_ring(self, c: _Dynamic, count: int, ring_slots: int) -> _Dynamic:
        """Ring slot for chunk."""
        layer = self.cache.layer if self.cache.layer is not None else 0
        return (layer * count + c) % ring_slots

    def _chunked_send_queries(self, c: _Dynamic, slot: _Dynamic, rows: int) -> None:
        """All-gather queries for chunk c."""
        ws, member, group, base = self.ws, self.member, self.group, self.base
        own = ws.own_rows[c, :rows, :]
        ws.gathered[slot, member] = (
            _pack_fp8_query(own)
            if ws.gathered.dtype == jnp.uint8
            else own.astype(jnp.bfloat16)
        )
        for peer in range(0 if "gather" in self.skip else group):

            @pl.when(peer != member)
            def _send_q(peer: _Dynamic = peer) -> None:
                """Send query slice to peer."""
                tpu.make_async_remote_copy(
                    ws.gathered.at[slot, member],
                    ws.gathered.at[slot, member],
                    ws.gather_semaphores.at[0, slot],
                    ws.gather_semaphores.at[1, slot],
                    device_id=(base + peer,),
                    device_id_type=pl.DeviceIdType.MESH,
                ).start()

    def _chunked_wait_gather(self, slot: _Dynamic) -> None:
        """Wait for chunk queries on ring slot."""
        if "gather" in self.skip:
            return
        ws, group = self.ws, self.group
        others = ws.gathered.at[slot, pl.ds(0, group - 1)]
        tpu.make_async_copy(others, others, ws.gather_semaphores.at[0, slot]).wait()
        tpu.make_async_copy(others, others, ws.gather_semaphores.at[1, slot]).wait()

    def _chunked_send_partials(
        self,
        slot: _Dynamic,
        hier: _Dynamic,
        hier_state: tuple[_Dynamic, ...],
    ) -> None:
        """Send chunk partials to peers."""
        ws, member, group, base = self.ws, self.member, self.group, self.base
        if hier:
            _, _, _, hosts, host, lane, lanes512 = hier_state
            for k in range(1, HOST_LANES):
                mate = (lane + k) % HOST_LANES
                tpu.make_async_remote_copy(
                    ws.send.at[pl.ds(mate * hosts, hosts), :, lanes512],
                    ws.receive.at[slot, pl.ds(lane * hosts, hosts), :, lanes512],
                    ws.combine_semaphores.at[0, slot],
                    ws.combine_semaphores.at[1, slot],
                    device_id=(base + host * HOST_LANES + mate,),
                    device_id_type=pl.DeviceIdType.MESH,
                ).start()
            ws.receive[slot, pl.ds(lane * hosts, hosts)] = ws.send[
                pl.ds(lane * hosts, hosts)
            ]
            return
        for peer in range(0 if "combine" in self.skip else group):

            @pl.when(peer != member)
            def _send_p(peer: _Dynamic = peer) -> None:
                """Send partials to peer."""
                tpu.make_async_remote_copy(
                    ws.send.at[peer, :, self._combine_lanes()],
                    ws.receive.at[slot, member, :, self._combine_lanes()],
                    ws.combine_semaphores.at[0, slot],
                    ws.combine_semaphores.at[1, slot],
                    device_id=(base + peer,),
                    device_id_type=pl.DeviceIdType.MESH,
                ).start()

        ws.receive[slot, member] = ws.send[member]

    def _chunked_wait_sent(self, slot: _Dynamic, peers_bytes: int) -> None:
        """Wait for partials sent on ring slot."""
        if "combine" in self.skip:
            return
        others = self.ws.receive.at[
            slot, pl.ds(0, peers_bytes), :, self._combine_lanes()
        ]
        tpu.make_async_copy(
            others, others, self.ws.combine_semaphores.at[0, slot]
        ).wait()

    def _chunked_wait_received(self, slot: _Dynamic, peers_bytes: int) -> None:
        """Wait for partials received on ring slot."""
        if "combine" in self.skip:
            return
        others = self.ws.receive.at[
            slot, pl.ds(0, peers_bytes), :, self._combine_lanes()
        ]
        tpu.make_async_copy(
            others, others, self.ws.combine_semaphores.at[1, slot]
        ).wait()

    def _chunked_host_merge(
        self,
        slot: _Dynamic,
        rows: int,
        hier_state: tuple[_Dynamic, ...],
    ) -> None:
        """Stage two of chunk: merge the host's eight partials of each owner."""
        stage2_send, stage2_recv, stage2_sems, hosts, host, lane, lanes512 = hier_state
        ws, group, base = self.ws, self.group, self.base
        pad = rows - self.slots
        tails = ws.receive[slot, :, pl.ds(pad, 2), pl.ds(0, 128)].astype(jnp.float32)
        vectors = tails[:, 0:1, :] + tails[:, 1:2, :]
        packet_row = jax.lax.broadcasted_iota(jnp.int32, (group, rows, 128), 1)
        packet_lane = jax.lax.broadcasted_iota(jnp.int32, (group, rows, 128), 2)
        lses = jnp.sum(
            jnp.where(packet_lane == packet_row, vectors, 0.0), axis=2, keepdims=True
        )
        lses = jnp.where(packet_row[:, :, :1] < pad, lses, NEG)
        for owner_host in range(hosts):
            sources = [source * hosts + owner_host for source in range(HOST_LANES)]
            top = lses[sources[0]]
            for index in sources[1:]:
                top = jnp.maximum(top, lses[index])
            total = jnp.zeros_like(top)
            accumulator: _Dynamic = None
            for index in sources:
                weight = jnp.exp(lses[index] - top)
                total = total + weight
                term = weight * ws.receive[slot, index, :, pl.ds(0, LATENT)].astype(
                    jnp.float32
                )
                accumulator = term if accumulator is None else accumulator + term
            stage2_send[owner_host] = _lse_packet(
                accumulator / total, top + jnp.log(total), self.slots
            )
        for k in range(1, hosts):
            owner_host = (host + k) % hosts
            tpu.make_async_remote_copy(
                stage2_send.at[owner_host, :, lanes512],
                stage2_recv.at[slot, host, :, lanes512],
                stage2_sems.at[0, slot],
                stage2_sems.at[1, slot],
                device_id=(base + owner_host * HOST_LANES + lane,),
                device_id_type=pl.DeviceIdType.MESH,
            ).start()
        stage2_recv[slot, host] = stage2_send[host]

    def _chunked_stage2_wait(
        self,
        slot: _Dynamic,
        which: int,
        hier_state: tuple[_Dynamic, ...],
    ) -> None:
        """Wait for stage 2 partials."""
        _, stage2_recv, stage2_sems, hosts, _, _, lanes512 = hier_state
        others = stage2_recv.at[slot, pl.ds(0, hosts - 1), :, lanes512]
        tpu.make_async_copy(others, others, stage2_sems.at[which, slot]).wait()

    def _chunked_merge(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Merge partials for chunk c."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            c,
            slot,
            chunk,
            rows,
            keys,
            key_valid,
            hier,
            hier_state,
        ) = (
            _p_next(_p_iter, _p_get("c")),
            _p_next(_p_iter, _p_get("slot")),
            _p_next(_p_iter, _p_get("chunk")),
            _p_next(_p_iter, _p_get("rows")),
            _p_next(_p_iter, _p_get("keys")),
            _p_next(_p_iter, _p_get("key_valid")),
            _p_next(_p_iter, _p_get("hier")),
            _p_next(_p_iter, _p_get("hier_state")),
        )
        if "merge" in self.skip:
            return
        ws, group, batch, slots = self.ws, self.group, self.batch, self.slots
        if self.sequence_rows == 1 or self.cache_verify_rows:
            unroll = (
                4
                if self.heads == _CONST_6
                and (batch == _CONST_64 or (batch == _CONST_48 and self.fp8))
                else 4
                if self.heads == DP4_HEADS
                and (batch == _CONST_32 or (batch == _CONST_24 and self.fp8))
                else 1
            )
            ws.merged[c, :rows, :] = _merge_received(
                ws.receive.at[slot],
                ws.own_rows[c, :rows, :],
                ws.new_rows[c, :rows, :],
                group,
                unroll=unroll,
            )
            return
        if keys is not None:
            if not (key_valid is not None):
                _err_msg = "Assertion failed: key_valid is not None"
                raise AssertionError(_err_msg)
            _, stage2_recv, _, hosts, _, _, _ = hier_state
            received, sources = (
                (stage2_recv.at[slot], hosts) if hier else (ws.receive.at[slot], group)
            )
            ws.merged[c, :rows, :] = _merge_received_keys(
                received,
                ws.own_rows[c, :rows, :],
                keys[c],
                key_valid[c],
                sources,
                lse_rows=self.slots if self.lse_in_padding else 0,
            )
            return
        latents = jnp.concatenate(
            [ws.new_rows[c, pl.ds(slots * j, 1), :] for j in range(chunk)], axis=0
        )
        ws.merged[c, :rows, :] = _merge_received_sequences(
            ws.receive.at[slot],
            ws.own_rows[c, :rows, :],
            latents,
            group,
            slots,
            self.sequence_rows,
            unroll=4
            if self.fp8 and _grouped_spec(batch, group, self.heads, self.sequence_rows)
            else 1,
        )

    def _chunked_local_attend(
        self,
        c: _Dynamic,
        chunk: _Dynamic,
        heads: _Dynamic,
        slot: _Dynamic,
        steps: _Dynamic,
    ) -> None:
        """Unpack gathered queries and run local attention for chunk c."""
        ws, slots = self.ws, self.slots
        local_scope = self._scope("local")
        local_scope.__enter__()
        for j in range(0 if "build" in self.skip else chunk):
            gathered = ws.gathered[slot, :, pl.ds(slots * j, slots), :].reshape(
                heads, LANES
            )
            padding = ((0, self.tile - heads), (0, 0))
            if ws.gathered.dtype == jnp.uint8:
                query, scale = _unpack_fp8_query(gathered)
                ws.queries[j] = jnp.pad(query, padding)
                ws.query_scale[j] = jnp.pad(scale, padding, constant_values=1)
            else:
                self._store_query(j, jnp.pad(gathered.astype(jnp.float32), padding))
        ws.maximum[...] = jnp.full(ws.maximum.shape, -jnp.inf, jnp.float32)
        ws.denominator[...] = jnp.zeros(ws.denominator.shape, jnp.float32)
        ws.accumulator[...] = jnp.zeros(ws.accumulator.shape, jnp.float32)
        first = chunk * c
        self._attend_range(
            self._chunked_first_step(first, steps),
            self._chunked_first_step(first, steps)
            if "attend" in self.skip
            else self._chunked_first_step(first + chunk, steps),
            first,
        )
        local_scope.__exit__(None, None, None)

    def _chunked_step(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Execute one chunk pipeline step."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            c,
            chunk,
            heads,
            ring_slots,
            count,
            lookahead,
            peers_bytes,
            steps,
            keys,
            key_valid,
            hier,
            hier_state,
        ) = (
            _p_next(_p_iter, _p_get("c")),
            _p_next(_p_iter, _p_get("chunk")),
            _p_next(_p_iter, _p_get("heads")),
            _p_next(_p_iter, _p_get("ring_slots")),
            _p_next(_p_iter, _p_get("count")),
            _p_next(_p_iter, _p_get("lookahead")),
            _p_next(_p_iter, _p_get("peers_bytes")),
            _p_next(_p_iter, _p_get("steps")),
            _p_next(_p_iter, _p_get("keys")),
            _p_next(_p_iter, _p_get("key_valid")),
            _p_next(_p_iter, _p_get("hier")),
            _p_next(_p_iter, _p_get("hier_state")),
        )
        slots = self.slots
        rows = slots * chunk
        done = c + lookahead - ring_slots

        @pl.when(done >= 0)
        def merge_done() -> None:
            """Merge completed chunk partials."""
            done_slot = self._chunked_ring(done, count, ring_slots)
            with self._scope("combine_wait"):
                if hier:
                    self._chunked_stage2_wait(done_slot, 1, hier_state)
                else:
                    self._chunked_wait_received(done_slot, peers_bytes)
            with self._scope("merge"):
                self._chunked_merge(
                    done,
                    done_slot,
                    chunk,
                    rows,
                    keys,
                    key_valid,
                    hier,
                    hier_state,
                )

        @pl.when(c + lookahead < count)
        def send_next() -> None:
            """Send query packet for lookahead chunk."""
            next_c = c + lookahead
            self._chunked_send_queries(
                next_c, self._chunked_ring(next_c, count, ring_slots), rows
            )

        slot = self._chunked_ring(c, count, ring_slots)
        with self._scope("gather_wait"):
            self._chunked_wait_gather(slot)
        self._chunked_local_attend(c, chunk, heads, slot, steps)

        @pl.when(c >= 1)
        def free_send_buffer() -> None:
            """Wait for previous chunk's send buffer to drain."""
            with self._scope("send_wait"):
                self._chunked_wait_sent(
                    self._chunked_ring(c - 1, count, ring_slots), peers_bytes
                )

        with self._scope("pack_send"):
            self._pack_partials(0, 0 if "pack" in self.skip else chunk)
            self._chunked_send_partials(slot, hier, hier_state)

        if hier:

            @pl.when(c >= 1)
            def host_stage() -> None:
                """Run stage-two host merge for previous chunk."""
                prev_slot = self._chunked_ring(c - 1, count, ring_slots)
                with self._scope("host_wait"):
                    self._chunked_wait_received(prev_slot, peers_bytes)

                    @pl.when(c >= _CONST_2)
                    def free_stage2() -> None:
                        """Wait for stage-two send buffer from c - 2."""
                        self._chunked_stage2_wait(
                            self._chunked_ring(c - 2, count, ring_slots),
                            0,
                            hier_state,
                        )

                with self._scope("host_merge"):
                    self._chunked_host_merge(prev_slot, rows, hier_state)

    def _chunked_epilogue(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
        """Drain remaining chunks after the pipelined chunk loop."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            chunk,
            rows,
            ring_slots,
            count,
            lookahead,
            peers_bytes,
            keys,
            key_valid,
            hier,
            hier_state,
        ) = (
            _p_next(_p_iter, _p_get("chunk")),
            _p_next(_p_iter, _p_get("rows")),
            _p_next(_p_iter, _p_get("ring_slots")),
            _p_next(_p_iter, _p_get("count")),
            _p_next(_p_iter, _p_get("lookahead")),
            _p_next(_p_iter, _p_get("peers_bytes")),
            _p_next(_p_iter, _p_get("keys")),
            _p_next(_p_iter, _p_get("key_valid")),
            _p_next(_p_iter, _p_get("hier")),
            _p_next(_p_iter, _p_get("hier_state")),
        )
        ws, batch, slots = self.ws, self.batch, self.slots
        last_slot = self._chunked_ring(count - 1, count, ring_slots)
        if hier:
            with self._scope("host_wait"):
                self._chunked_wait_received(last_slot, peers_bytes)
                if count >= _CONST_2:
                    self._chunked_stage2_wait(
                        self._chunked_ring(count - 2, count, ring_slots),
                        0,
                        hier_state,
                    )
            with self._scope("host_merge"):
                self._chunked_host_merge(last_slot, rows, hier_state)
            self._chunked_wait_sent(last_slot, peers_bytes)
            for c in range(max(count - ring_slots + lookahead, 0), count):
                c_slot = self._chunked_ring(c, count, ring_slots)
                with self._scope("combine_wait"):
                    self._chunked_stage2_wait(c_slot, 1, hier_state)
                with self._scope("merge"):
                    self._chunked_merge(
                        c, c_slot, chunk, rows, keys, key_valid, hier, hier_state
                    )
            self._chunked_stage2_wait(last_slot, 0, hier_state)
            self._lane_major = False
            return jnp.stack(
                [
                    ws.merged[b // chunk, pl.ds(slots * (b % chunk), self.heads), :]
                    for b in range(batch)
                ]
            )
        if batch != _CONST_32:
            self._chunked_wait_sent(last_slot, peers_bytes)
        for c in range(max(count - ring_slots + lookahead, 0), count):
            c_slot = self._chunked_ring(c, count, ring_slots)
            with self._scope("combine_wait"):
                self._chunked_wait_received(c_slot, peers_bytes)
            with self._scope("merge"):
                self._chunked_merge(
                    c, c_slot, chunk, rows, keys, key_valid, hier, hier_state
                )
        if batch == _CONST_32:
            self._chunked_wait_sent(last_slot, peers_bytes)
        return jnp.stack(
            [
                ws.merged[b // chunk, pl.ds(slots * (b % chunk), self.heads), :]
                for b in range(batch)
            ]
        )

    def _attend_chunked(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
        """Execute attend for a batch of C chunks pipelined."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            queries,
            new_latents,
            between,
        ) = (
            _p_next(_p_iter, _p_get("queries")),
            _p_next(_p_iter, _p_get("new_latents")),
            _p_next(_p_iter, _p_get("between")),
        )
        keys = _p_kw.get("keys")
        key_valid = _p_kw.get("key_valid")
        ws, group, batch, slots = self.ws, self.group, self.batch, self.slots
        chunk = chunk_layout(batch, self.heads, chunk_rows=self.chunk_rows)[0]
        if keys is not None and chunk != self.sequence_rows:
            _err_msg = "explicit new keys need one sequence per chunk"
            raise ValueError(_err_msg)
        ring_slots = ws.gathered.shape[0]
        count = batch // chunk
        hier = self.host_combine is not None
        lookahead = 2 if ring_slots == _CONST_4 and not hier else 1
        self._lane_major = hier
        hier_state: _Dynamic = (object(),) * 7
        if hier:
            if (
                group != 4 * HOST_LANES
                or not self.lse_in_padding
                or keys is None
                or "combine" in self.skip
            ):
                _err_msg = (
                    "host_combine: DCP32 explicit-keys sequences with lse_in_padding"
                )
                raise ValueError(_err_msg)
            stage2_recv, stage2_send, stage2_sems = self.host_combine
            hosts = group // HOST_LANES
            host, lane = self.member // HOST_LANES, self.member % HOST_LANES
            lanes512 = self._combine_lanes()
            hier_state = (
                stage2_send,
                stage2_recv,
                stage2_sems,
                hosts,
                host,
                lane,
                lanes512,
            )

        rows = slots * chunk
        heads = group * slots
        ws.own_rows[...] = jnp.zeros(ws.own_rows.shape, jnp.float32)
        ws.new_rows[...] = jnp.zeros(ws.new_rows.shape, jnp.float32)
        for b in range(batch):
            c_idx, j = divmod(b, chunk)
            ws.own_rows[c_idx, pl.ds(slots * j, self.heads), :] = queries[b]
            ws.new_rows[c_idx, pl.ds(slots * j, self.heads), :] = jnp.broadcast_to(
                new_latents[b][None, :], (self.heads, LANES)
            )

        steps = [(self.counts[b] + self.span - 1) // self.span for b in range(batch)]
        peers_bytes = (HOST_LANES - 1) * (group // HOST_LANES) if hier else group - 1
        for initial in range(min(lookahead, count)):
            self._chunked_send_queries(
                initial, self._chunked_ring(initial, count, ring_slots), rows
            )
        if between is not None:
            between()

        @pl.loop(0, count)
        def chunk_step(c: _Dynamic) -> None:
            """Run one chunk iteration."""
            self._chunked_step(
                c,
                chunk,
                heads,
                ring_slots,
                count,
                lookahead,
                peers_bytes,
                steps,
                keys,
                key_valid,
                hier,
                hier_state,
            )

        return self._chunked_epilogue(
            chunk,
            rows,
            ring_slots,
            count,
            lookahead,
            peers_bytes,
            keys,
            key_valid,
            hier,
            hier_state,
        )


def _merge_received(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Execute merge received logic."""
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        received,
        own_rows,
        new_rows,
        group,
    ) = (
        _p_next(_p_iter, _p_get("received")),
        _p_next(_p_iter, _p_get("own_rows")),
        _p_next(_p_iter, _p_get("new_rows")),
        _p_next(_p_iter, _p_get("group")),
    )
    _slots = _p_next(_p_iter, _p_get("slots", None))
    _fp8_query = _p_kw.get("fp8_query", False)
    unroll = _p_kw.get("unroll", 1)
    """Log-sum-exp merge of the ``group`` peers' partials in ``received``.

    [G, rows, 640] (a ref: normalized outputs, lse hi/lo in lanes 512:514)
    with the new token (``own_rows`` the rows' queries, ``new_rows`` its
    latent per head slot): [rows, 512]. Online over the peers in a loop
    (``unroll`` peers a step), so few peers' [rows, 640] are live at a time
    (bounds register pressure).
    """
    top = jnp.sum(
        own_rows * new_rows, axis=1, keepdims=True
    )  # [rows, 1]: the new token's score
    accumulator = new_rows[:, :LATENT]
    denominator = jnp.ones_like(top)

    def fold(peer: _Dynamic, carry: _Dynamic) -> _Dynamic:
        """Execute fold logic."""
        top, accumulator, denominator = carry
        partial = received[peer].astype(jnp.float32)  # [rows, 640]
        lse = partial[:, LATENT : LATENT + 1] + partial[:, LATENT + 1 : LATENT + 2]
        new_top = jnp.maximum(top, lse)
        keep, weight = jnp.exp(top - new_top), jnp.exp(lse - new_top)
        return (
            new_top,
            accumulator * keep + weight * partial[:, :LATENT],
            denominator * keep + weight,
        )

    _, accumulator, denominator = jax.lax.fori_loop(
        0, group, fold, (top, accumulator, denominator), unroll=unroll
    )
    return accumulator / denominator


def _merge_received_sequences(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """``_merge_received`` for a chunk of whole multi-row sequences: each."""
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        received,
        own_rows,
        latents,
        group,
        slots,
        sequence_rows,
    ) = (
        _p_next(_p_iter, _p_get("received")),
        _p_next(_p_iter, _p_get("own_rows")),
        _p_next(_p_iter, _p_get("latents")),
        _p_next(_p_iter, _p_get("group")),
        _p_next(_p_iter, _p_get("slots")),
        _p_next(_p_iter, _p_get("sequence_rows")),
    )
    unroll = _p_kw.get("unroll", 1)
    """``_merge_received`` for a chunk of whole multi-row sequences: each.

    row's head slots also fold in, causally, the new tokens of its
    sequence's earlier rows in the chunk (``latents`` [chunk, 640]: every
    row's new latent; every rank holds them all), then the peers' partials
    online as there.
    """
    scores = jax.lax.dot_general(
        own_rows,
        latents,
        (((1,), (1,)), ((), ())),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )  # [rows, chunk]
    slot_row = jax.lax.broadcasted_iota(jnp.int32, scores.shape, 0) // slots
    key_row = jax.lax.broadcasted_iota(jnp.int32, scores.shape, 1)
    causal = (key_row // sequence_rows == slot_row // sequence_rows) & (
        key_row <= slot_row
    )
    scores = jnp.where(causal, scores, -jnp.inf)
    top = jnp.max(scores, axis=1, keepdims=True)  # finite: a row sees its own token
    weights = jnp.exp(scores - top)
    accumulator = jax.lax.dot_general(
        weights,
        latents[:, :LATENT],
        (((1,), (0,)), ((), ())),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    denominator = jnp.sum(weights, axis=1, keepdims=True)

    def fold(peer: _Dynamic, carry: _Dynamic) -> _Dynamic:
        """Execute fold logic."""
        top, accumulator, denominator = carry
        partial = received[peer].astype(jnp.float32)  # [rows, 640]
        lse = partial[:, LATENT : LATENT + 1] + partial[:, LATENT + 1 : LATENT + 2]
        new_top = jnp.maximum(top, lse)
        keep, weight = jnp.exp(top - new_top), jnp.exp(lse - new_top)
        return (
            new_top,
            accumulator * keep + weight * partial[:, :LATENT],
            denominator * keep + weight,
        )

    # Bounded unrolling exposes the next peer load without changing the
    # online merge order or materializing all 32 peers in FP32.
    def fold_group(peer: _Dynamic, carry: _Dynamic) -> _Dynamic:
        """Execute fold group logic."""
        for i in range(unroll):
            carry = fold(peer * unroll + i, carry)
        return carry

    _, accumulator, denominator = jax.lax.fori_loop(
        0, group // unroll, fold_group, (top, accumulator, denominator)
    )
    return accumulator / denominator


def _lse_packet(normalized: _Dynamic, lse: _Dynamic, lse_rows: _Dynamic) -> _Dynamic:
    """``[rows, 640]`` BF16 ``lse_in_padding`` packet of a partial."""
    rows = normalized.shape[0]
    pad = rows - lse_rows
    high = lse.astype(jnp.bfloat16).astype(jnp.float32)
    low = (lse - high).astype(jnp.bfloat16).astype(jnp.float32)
    row = jax.lax.broadcasted_iota(jnp.int32, (rows, 128), 0)
    lane = jax.lax.broadcasted_iota(jnp.int32, (rows, 128), 1)
    high_vector = jnp.sum(
        jnp.where((lane == row) & (row < pad), high, 0.0), axis=0, keepdims=True
    )
    low_vector = jnp.sum(
        jnp.where((lane == row) & (row < pad), low, 0.0), axis=0, keepdims=True
    )
    packet = jnp.concatenate(
        (normalized, jnp.zeros((rows, LANES - LATENT), jnp.float32)), axis=1
    )
    row640 = jax.lax.broadcasted_iota(jnp.int32, (rows, LANES), 0)
    pad_lanes = jnp.zeros((1, LANES - 128), jnp.float32)
    packet = jnp.where(
        row640 == pad, jnp.concatenate((high_vector, pad_lanes), axis=1), packet
    )
    packet = jnp.where(
        row640 == pad + 1, jnp.concatenate((low_vector, pad_lanes), axis=1), packet
    )
    return packet.astype(jnp.bfloat16)


def _merge_received_keys(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """``_merge_received`` with explicit new keys: every row (head slot)."""
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        received,
        own_rows,
        keys,
        key_valid,
        group,
    ) = (
        _p_next(_p_iter, _p_get("received")),
        _p_next(_p_iter, _p_get("own_rows")),
        _p_next(_p_iter, _p_get("keys")),
        _p_next(_p_iter, _p_get("key_valid")),
        _p_next(_p_iter, _p_get("group")),
    )
    lse_rows = _p_kw.get("lse_rows", 0)
    """``_merge_received`` with explicit new keys: every row (head slot).

    attends the valid ones of ``keys`` [K, 640] (``key_valid`` [1, K] nonzero;
    at least one), and the peers' partials. Two passes rather than online:
    every peer's log-sum-exp (lanes 512:514) in one load, one maximum, then a
    weighted sum of the partials (no rescaling or exp per peer).
    """
    keys = keys.astype(jnp.float32)
    scores = jax.lax.dot_general(
        own_rows,
        keys,
        (((1,), (1,)), ((), ())),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )  # [rows, K]
    scores = jnp.where(key_valid != 0, scores, -jnp.inf)
    if lse_rows:
        # ``lse_in_padding``: the last ``lse_rows`` packet rows (a padding
        # row's head slots) hold the others' lse, packet row r at lane r.
        rows = received.shape[1]
        pad = rows - lse_rows
        vector = received[:, pad, pl.ds(0, 128)].astype(jnp.float32) + received[
            :, pad + 1, pl.ds(0, 128)
        ].astype(jnp.float32)  # [G, 128]
        row = jax.lax.broadcasted_iota(jnp.int32, (group, rows, 128), 1)
        lane = jax.lax.broadcasted_iota(jnp.int32, (group, rows, 128), 2)
        lse = jnp.sum(
            jnp.where(lane == row, vector[:, None, :], 0.0), axis=2, keepdims=True
        )  # [G, rows, 1]
        lse = jnp.where(row[:, :, :1] < pad, lse, NEG)  # the padding row: no peers
    else:
        tails = received[:, :, LATENT : LATENT + 128].astype(
            jnp.float32
        )  # [G, rows, 128]
        lse = tails[:, :, 0:1] + tails[:, :, 1:2]  # [G, rows, 1]
    top = jnp.maximum(
        jnp.max(scores, axis=1, keepdims=True), jnp.max(lse, axis=0)
    )  # [rows, 1]
    weights = jnp.exp(scores - top)
    peer_weights = jnp.exp(lse - top[None])  # [G, rows, 1]
    accumulator = jax.lax.dot_general(
        weights,
        keys[:, :LATENT],
        (((1,), (0,)), ((), ())),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    denominator = jnp.sum(weights, axis=1, keepdims=True) + jnp.sum(
        peer_weights, axis=0
    )
    for peer in range(group):
        accumulator = accumulator + peer_weights[peer] * received[
            peer, :, :LATENT
        ].astype(jnp.float32)
    return accumulator / denominator


def _pack_fp8_query(query: _Dynamic) -> _Dynamic:
    """FP8 query bytes plus a lossless FP32 scale in padding lanes 576:580.

    Round to BF16 first, matching the BF16 query path. Integer transport
    preserves every scale bit, including bytes that would encode NaNs if
    treated as FP8 values. The owner's FP32 query stays live separately for
    the new-token score.
    """
    query = query.astype(jnp.bfloat16).astype(jnp.float32)
    maximum = jnp.max(jnp.abs(query), axis=1, keepdims=True)
    scale = jnp.where(maximum > 0, maximum / 448.0, 1.0)
    # Perform masks/broadcasts in 32-bit lanes; packed-byte sublane broadcasts
    # are unsupported by Mosaic. Only the stored wire stays byte packed.
    payload = jax.lax.bitcast_convert_type(
        (query / scale).astype(jnp.float8_e4m3fn), jnp.uint8
    ).astype(jnp.uint32)
    bits = jax.lax.bitcast_convert_type(scale, jnp.uint32)
    lanes = jax.lax.broadcasted_iota(jnp.int32, query.shape, 1)
    scale_bytes = (bits >> (((lanes - 576) % 4) * 8)) & 255
    return jnp.where(lanes < _CONST_576, payload, scale_bytes).astype(jnp.uint8)


def _unpack_fp8_query(packet: _Dynamic) -> _Dynamic:
    """Recover the exact query/scale and clear metadata before the QK dot."""
    packet = packet.astype(jnp.uint32)
    lanes = jax.lax.broadcasted_iota(jnp.int32, packet.shape, 1)
    # Reassemble the scale with a lane reduction; narrow byte slices lose bits on TPU.
    words = packet << (((lanes - 576) % 4) * 8)
    bits = jnp.sum(
        jnp.where((lanes >= _CONST_576) & (lanes < _CONST_580), words, jnp.uint32(0)),
        axis=1,
        keepdims=True,
    )
    scale = jax.lax.bitcast_convert_type(bits, jnp.float32)
    payload = jnp.where(lanes < _CONST_576, packet, jnp.uint32(0)).astype(jnp.uint8)
    return jax.lax.bitcast_convert_type(payload, jnp.float8_e4m3fn), scale


def _attend_chunked(self: _Dynamic, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Execute attend for a batch of C chunks pipelined."""
    return self._attend_chunked(*_p_args, **_p_kw)


DcpAttention.attend_chunked = _attend_chunked
