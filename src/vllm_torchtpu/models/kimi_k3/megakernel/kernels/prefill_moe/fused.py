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

"""Fused EP16/TP2 prefill MoE for 512- through 8192-token windows.

Source provenance: Adapted from inferact/k3-megakernel@30c9ba966b.

Schedule:
  1. Router/latent/shared projections overlap the incremental Top16, route sort,
     route exchange, packed latent ring and shared-output reduction.
  2. Compact routing metadata moves to SMEM. GU/SiTU and the first down column
     share a pipeline; later down columns overlap token combine and reduction.
  3. The reduced VMEM shard is normalized, projected up and added to the shared
     expert and residual outputs.

The factory at the end is the only production entry point. The small-window, 4K
and 8K schedules are selected by window size; 8K skew overflow retains every route via
an uncapped fallback. Python generators split work at tracing time, allowing
independent vector/communication work between MXU panels without extra calls.
"""

from __future__ import annotations

import dataclasses
import functools
import operator
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.models.kimi_k3.megakernel.config import active as _config
from vllm_torchtpu.models.kimi_k3.megakernel.kernels import prefill_megakernel as k

from .indexed_ragged_gmm import PackedRows, indexed_gate_up
from .moe_down_valu import down_valu, down_valu_compact

_Dynamic = Any
_CONST_1024 = 1024
_CONST_12 = 12
_CONST_128 = 128
_CONST_14 = 14
_CONST_16 = 16
_CONST_1792 = 1792
_CONST_2 = 2
_CONST_256 = 256
_CONST_28 = 28
_CONST_30 = 30
_CONST_31 = 31
_CONST_32 = 32
_CONST_32768 = 32768
_CONST_4096 = 4096
_CONST_56 = 56
_CONST_72 = 72
_CONST_8192 = 8192
_CONST_896 = 896


def routed_experts_prepared(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Consume prepared expert-major metadata with the existing GU/down path."""
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        latent,
        sorted_tokens,
        sorted_probs,
        counts,
        expert_gate_up,
        expert_gate_up_scales,
        expert_down,
        expert_down_scales,
    ) = (
        _p_next(_p_iter, _p_get("latent")),
        _p_next(_p_iter, _p_get("sorted_tokens")),
        _p_next(_p_iter, _p_get("sorted_probs")),
        _p_next(_p_iter, _p_get("counts")),
        _p_next(_p_iter, _p_get("expert_gate_up")),
        _p_next(_p_iter, _p_get("expert_gate_up_scales")),
        _p_next(_p_iter, _p_get("expert_down")),
        _p_next(_p_iter, _p_get("expert_down_scales")),
    )
    (
        reduce_tables,
        expert_layer,
        global_route_max,
    ) = (
        _p_get("reduce_tables"),
        _p_get("expert_layer"),
        _p_get("global_route_max"),
    )
    if expert_layer is not None:
        expert_layer = k.expert_family_slot(expert_gate_up.shape[0], expert_layer)
    tokens = latent.shape[0]
    # The E8M0 exponent bytes go to the GMM as they are (it converts each
    # tile in VMEM): an FP32 [E, blocks, 1, N] copy pads 8x in HBM.
    gu_scale = expert_gate_up_scales.astype(jnp.uint8)
    down_scale = expert_down_scales.astype(jnp.uint8)
    # gmm_v2 splits [gate, up] and interleaves lanes for the fused SiTU
    # activation (expert_activation).
    middle = indexed_gate_up(
        latent, sorted_tokens, expert_gate_up, counts, gu_scale, expert_layer
    )
    options = {
        "tokens": tokens,
        "layer": expert_layer,
        "ring_dtype": jnp.dtype(_config().moe_reduce_dtype),
    }
    if tokens == _CONST_8192:
        return down_valu_compact(
            middle,
            expert_down,
            down_scale,
            counts,
            sorted_tokens,
            sorted_probs,
            reduce_tables,
            tile=256,
            columns=896,
            unroll=32,
            metadata_prefix_rows=32768,
            slice_input=False,
            global_route_max=global_route_max,
            **options,
        )
    return down_valu(
        middle,
        expert_down,
        down_scale,
        counts,
        sorted_tokens,
        sorted_probs,
        reduce_tables,
        tile=128,
        columns=1792,
        unroll=8,
        **options,
    )


class _SortRouteRunner:
    """In-place bitonic network runner for route sorting."""

    def __init__(self, keys: _Dynamic, probabilities: _Dynamic, size: _Dynamic) -> None:
        """Initialize sort runner."""
        self.keys = keys
        self.probabilities = probabilities
        self.size = size
        self.window, self.rows = 4096, 32
        if not (size >= self.window and size & size - 1 == 0):
            _err_msg = "Assertion failed: size >= window and size & size - 1 == 0"
            raise AssertionError(_err_msg)
        self.shape = (self.rows, 128)
        row = jax.lax.broadcasted_iota(jnp.int32, self.shape, 0)
        self.lane = jax.lax.broadcasted_iota(jnp.int32, self.shape, 1)
        self.positions = row * 128 + self.lane
        self.pending = []

    def _view(self, ref: _Dynamic, ordinal: _Dynamic) -> _Dynamic:
        """Execute view logic."""
        return ref.at[pl.ds(ordinal * self.rows, self.rows), :]

    def _partner(self, value: _Dynamic, distance: _Dynamic) -> _Dynamic:
        """Execute partner logic."""
        if distance >= _CONST_1024:
            return jnp.concatenate(
                [
                    value[
                        8 * (i ^ (distance // 1024)) : 8 * (i ^ (distance // 1024)) + 8
                    ]
                    for i in range(self.rows // 8)
                ],
                axis=0,
            )
        if distance >= _CONST_128:
            shape3 = (self.rows // 8, 8, 128)
            index = jax.lax.broadcasted_iota(jnp.int32, shape3, 1)
            return jnp.take_along_axis(
                value.reshape(shape3),
                index ^ (distance // 128),
                axis=1,
                mode="promise_in_bounds",
            ).reshape(self.shape)
        return jnp.take_along_axis(
            value, self.lane ^ distance, axis=1, mode="promise_in_bounds"
        )

    def _flush(self) -> _Dynamic:
        """Execute flush logic."""
        if not self.pending:
            return
        for ordinal in range(self.size // self.window):
            kr, pr = (
                self._view(self.keys, ordinal),
                self._view(self.probabilities, ordinal),
            )
            x, p = kr[...], pr[...]
            global_position = self.positions + ordinal * self.window
            for distance, group in self.pending:
                y, q = self._partner(x, distance), self._partner(p, distance)
                descending = (global_position & group) != 0
                upper = (global_position & distance) != 0
                take_min = ~(descending ^ upper)
                keep = ~((x < y) ^ take_min)
                x, p = jnp.where(keep, x, y), jnp.where(keep, p, q)
            kr[...], pr[...] = x, p
            yield
        self.pending.clear()

    def run(self) -> _Dynamic:
        """Run the bitonic network."""
        for stage in range(1, self.size.bit_length()):
            for shift in reversed(range(stage)):
                distance, group = 1 << shift, 1 << stage
                if distance < self.window:
                    self.pending.append((distance, group))
                    continue
                yield from self._flush()
                stride = distance // self.window
                for ordinal in range(self.size // (2 * self.window)):
                    low = ordinal // stride * (2 * stride) + ordinal % stride
                    high = low + stride
                    kl, kh = self._view(self.keys, low), self._view(self.keys, high)
                    plow, phigh = (
                        self._view(self.probabilities, low),
                        self._view(self.probabilities, high),
                    )
                    x, y, p, q = kl[...], kh[...], plow[...], phigh[...]
                    ascending = (low * self.window & group) == 0
                    keep = ~((x < y) ^ ascending)
                    kl[...], kh[...] = jnp.where(keep, x, y), jnp.where(keep, y, x)
                    plow[...], phigh[...] = jnp.where(keep, p, q), jnp.where(keep, q, p)
                    yield
        yield from self._flush()


def _sort_route_steps(
    keys: _Dynamic, probabilities: _Dynamic, *, size: _Dynamic
) -> _Dynamic:
    """Yield after each 4096-route window of the in-place bitonic network.

    Keys encode (local expert, original route position), so all live keys are
    unique and the result has deterministic expert-major/token-major order.
    Probability values follow exactly the same swaps as their keys.
    """
    yield from _SortRouteRunner(keys, probabilities, size).run()


class _Top16:
    """Incremental biased sigmoid Top16; ties choose the lowest expert ID."""

    def __init__(self, logits: _Dynamic, bias: _Dynamic) -> None:
        """Execute init logic."""
        rows, _experts = logits.shape
        self.raw = 1.0 / (1.0 + jnp.exp(-logits))
        self.scores = self.raw + bias
        self.experts = jax.lax.broadcasted_iota(jnp.int32, logits.shape, 1)
        self.slots = jax.lax.broadcasted_iota(jnp.int32, (rows, 16), 1)
        self.selected = jnp.zeros((rows, 16), jnp.int32)
        self.probabilities = jnp.zeros((rows, 16), jnp.float32)
        self.index = 0

    def step(self) -> _Dynamic:
        """Execute step logic."""
        best = jnp.max(self.scores, axis=1, keepdims=True)
        winner = jnp.min(
            jnp.where(self.scores == best, self.experts, self.scores.shape[1]),
            axis=1,
            keepdims=True,
        )
        chosen = self.experts == winner
        probability = jnp.sum(jnp.where(chosen, self.raw, 0.0), axis=1, keepdims=True)
        self.selected = jnp.where(self.slots == self.index, winner, self.selected)
        self.probabilities = jnp.where(
            self.slots == self.index, probability, self.probabilities
        )
        self.scores = jnp.where(chosen, -jnp.inf, self.scores)
        self.index += 1

    def finish(self, valid: _Dynamic) -> _Dynamic:
        """Execute finish logic."""
        while self.index < _CONST_16:
            self.step()
        denominator = jnp.maximum(
            jnp.sum(self.probabilities, axis=1, keepdims=True), 1e-30
        )
        return self.selected, jnp.where(valid, self.probabilities / denominator, 0.0)


def _step_metadata_chunk(
    chunk: int,
    wire: _Dynamic,
    keys: _Dynamic,
    probabilities: _Dynamic,
    cfg: tuple[_Dynamic, ...],
) -> tuple[_Dynamic, _Dynamic, _Dynamic]:
    """Process a single metadata chunk."""
    tokens, rank, bits, shape, rows, lanes = cfg
    """Process a single metadata chunk."""
    if tokens < _CONST_4096:
        records = wire[pl.ds(chunk * 32, 32), :].reshape(128, 32)
        ids = records[:, :16]
        probs = jax.lax.bitcast_convert_type(records[:, 16:], jnp.float32)
    else:
        chunks_per_rank = tokens // 4096
        rank_base = (chunk // chunks_per_rank) * (tokens // 128)
        chunk_base = (chunk % chunks_per_rank) * 16
        ids = wire[pl.ds(rank_base + chunk_base, 16), :]
        probs = jax.lax.bitcast_convert_type(
            wire[pl.ds(rank_base + tokens // 256 + chunk_base, 16), :],
            jnp.float32,
        )
    local = ids - (rank // 2 * 56)
    valid = (probs != 0) & (local >= 0) & (local < _CONST_56)
    local = jnp.where(valid, local, 56)
    original_position = chunk * 2048 + rows * shape[1] + lanes
    packed = (local << bits) | original_position
    keys[pl.ds(chunk * 16, 16), :] = packed.reshape(16, 128)
    probabilities[pl.ds(chunk * 16, 16), :] = probs.reshape(16, 128)
    return local.reshape(16, 128), ids.reshape(16, 128), probs.reshape(16, 128) != 0


def _extract_token_ids(
    chunk: _Dynamic, keys: _Dynamic, token_ids: _Dynamic, bits: int
) -> None:
    """Extract token IDs from sorted keys."""
    index = pl.ds(chunk * 32, 32)
    token_ids[index, :] = (keys[index, :] & ((1 << bits) - 1)) // 16


def _metadata_steps(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Build stable local routing metadata, yielding one prep/sort chunk at a time.

    The all-gathered wire contains 16 expert IDs followed by 16 probability bit
    patterns per token. Every rank counts all 16 expert groups at 8K, making the
    compact-capacity decision uniform before any conditional collective runs.
    """
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        wire,
        keys,
        probabilities,
        token_ids,
        counts,
        maximum,
    ) = (
        _p_next(_p_iter, _p_get("wire")),
        _p_next(_p_iter, _p_get("keys")),
        _p_next(_p_iter, _p_get("probabilities")),
        _p_next(_p_iter, _p_get("token_ids")),
        _p_next(_p_iter, _p_get("counts")),
        _p_next(_p_iter, _p_get("maximum")),
    )
    (
        tokens,
        rank,
        counts_ready,
    ) = (
        _p_get("tokens"),
        _p_get("rank"),
        _p_get("counts_ready"),
    )
    bits = (tokens * 16 - 1).bit_length()
    shape = (128, 16) if tokens < _CONST_4096 else (16, 128)
    expert_index = jnp.arange(56, dtype=jnp.int32)
    group_index = jnp.arange(16, dtype=jnp.int32)
    lanes = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
    rows = jax.lax.broadcasted_iota(jnp.int32, shape, 0)
    total_counts = jnp.zeros((56,), jnp.int32)
    total_groups = jnp.zeros((16,), jnp.int32)
    for chunk in range(tokens // 128):
        local_flat, ids_flat, live_flat = _step_metadata_chunk(
            chunk, wire, keys, probabilities, (tokens, rank, bits, shape, rows, lanes)
        )
        tile_counts = jnp.zeros((56,), jnp.int32)
        for expert in range(56):
            count = jnp.sum((local_flat == expert).astype(jnp.int32))
            tile_counts = jnp.where(expert_index == expert, count, tile_counts)
        total_counts += tile_counts
        if tokens == _CONST_8192:
            tile_groups = jnp.zeros((16,), jnp.int32)
            groups = ids_flat // 56
            for group in range(16):
                count = jnp.sum((live_flat & (groups == group)).astype(jnp.int32))
                tile_groups = jnp.where(group_index == group, count, tile_groups)
            total_groups += tile_groups
        yield
    counts[...] = total_counts
    if maximum is not None:
        maximum[...] = jnp.max(total_groups)[None]
    counts_ready()
    yield from _sort_route_steps(keys, probabilities, size=tokens * 16)

    def extract(chunk: _Dynamic) -> _Dynamic:
        """Execute extract logic."""
        _extract_token_ids(chunk, keys, token_ids, bits)

    pl.loop(0, tokens * 16 // 4096)(extract)


def _dense_panels(
    lhs: _Dynamic,
    weights: _Dynamic,
    buffers: _Dynamic,
    semaphores: _Dynamic,
    *,
    columns: _Dynamic,
) -> _Dynamic:
    """Stream 256-column BF16 panels, with the next weight DMA in flight.

    The yielded FP32 product is deliberately not stored here. The caller places
    independent routing/vector/communication work before consuming it so the
    compiler can schedule that work alongside the MXU product.
    """

    def source(panel: _Dynamic) -> _Dynamic:
        """Execute source logic."""
        return weights.at[:, pl.ds(panel * 256, 256)]

    pltpu.make_async_copy(source(0), buffers[0], semaphores[0]).start()
    for panel in range(columns // 256):
        bank = panel % 2
        pltpu.make_async_copy(source(panel), buffers[bank], semaphores[bank]).wait()
        if panel + 1 < columns // 256:
            pltpu.make_async_copy(
                source(panel + 1), buffers[1 - bank], semaphores[1 - bank]
            ).start()
        yield (
            panel * 256,
            jnp.matmul(lhs, buffers[bank][...], preferred_element_type=jnp.float32),
        )


def _latent_half_pair_panels(
    lhs: _Dynamic, weights: _Dynamic, buffers: _Dynamic, semaphores: _Dynamic
) -> _Dynamic:
    """Pair matching 256-column fragments of the two 1792-wide halves."""

    def source_column(panel: _Dynamic) -> _Dynamic:
        """Execute source column logic."""
        return (panel // 2) * 256 + (panel % 2) * 1792

    def copy(panel: _Dynamic) -> _Dynamic:
        """Execute copy logic."""
        return pltpu.make_async_copy(
            weights.at[:, pl.ds(source_column(panel), 256)],
            buffers[panel % 4],
            semaphores[panel % 4],
        )

    for panel in range(2):
        copy(panel).start()
    for panel in range(0, 14, 2):
        for j in range(2):
            copy(panel + j).wait()
        for j in range(2):
            if panel + 2 + j < _CONST_14:
                copy(panel + 2 + j).start()
        products = [
            jnp.matmul(
                lhs, buffers[(panel + j) % 4][...], preferred_element_type=jnp.float32
            )
            for j in range(2)
        ]
        for j, product in enumerate(products):
            yield source_column(panel + j), product


def _route_gather_steps(
    payload: _Dynamic,
    output: _Dynamic,
    send: _Dynamic,
    recv: _Dynamic,
    *,
    rank: _Dynamic,
) -> _Dynamic:
    """Five recursive-doubling VMEM exchanges; one completion counter per round.

    Output ranges are immutable once received. A call-entry collective barrier
    is required, and callers must drain this generator before consuming output.
    """
    rows = payload.shape[0]
    output[pl.ds(rank * rows, rows), :] = payload[...]
    for stage in range(5):
        peers = 1 << stage
        peer = rank ^ peers
        origin = rank // peers * (peers * rows)
        incoming = peer // peers * (peers * rows)
        source = output.at[pl.ds(origin, peers * rows), :]
        destination = output.at[pl.ds(origin, peers * rows), :]
        received = output.at[pl.ds(incoming, peers * rows), :]
        transfer = pltpu.make_async_remote_copy(
            source,
            destination,
            send.at[stage],
            recv.at[stage],
            device_id=peer,
            device_id_type=pl.DeviceIdType.LOGICAL,
        )
        transfer.start()
        yield stage
        transfer.wait_send()
        # The local receive address differs from the destination on our peer.
        pltpu.make_async_remote_copy(
            received,
            received,
            send.at[stage],
            recv.at[stage],
            device_id=peer,
            device_id_type=pl.DeviceIdType.LOGICAL,
        ).wait_recv()


def _enter_mesh() -> _Dynamic:
    """Match collective lifetimes before any peer can write another's scratch."""
    barrier = pltpu.get_barrier_semaphore()
    for peer in range(32):
        pl.semaphore_signal(
            barrier, 1, device_id=peer, device_id_type=pl.DeviceIdType.LOGICAL
        )
    pl.semaphore_wait(barrier, 32)


class _LatentGatherRunner:
    """Twelve directional streams, with two receive slots per stream."""

    def __init__(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Initialize latent gather runner."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            self.tables,
            self.payload,
            self.received,
            self.output,
            self.send,
            self.recv,
            self.sem_credits,
            self.stores,
        ) = (
            _p_next(_p_iter, _p_get("tables")),
            _p_next(_p_iter, _p_get("payload")),
            _p_next(_p_iter, _p_get("received")),
            _p_next(_p_iter, _p_get("output")),
            _p_next(_p_iter, _p_get("send")),
            _p_next(_p_iter, _p_get("recv")),
            _p_next(_p_iter, _p_get("credits")),
            _p_next(_p_iter, _p_get("stores")),
        )
        self.rank = _p_get("rank")
        self.rows = self.payload.shape[0]
        if self.rows not in (16, 32, 64, 128, 256):
            _err_msg = "Assertion failed: rows in (16, 32, 64, 128, 256)"
            raise AssertionError(_err_msg)
        self.regions = []
        begin = 0
        for stream in range(12):
            size = self.rows // 12 + (stream < self.rows % 12)
            family = stream // 2 % 3
            direction = -1 if stream % 2 == 0 else 1
            position = self.tables[family, 0, self.rank]
            target = self.tables[family, 1, (position + 32 + direction) % 32]
            source_peer = self.tables[family, 1, (position + 32 - direction) % 32]
            self.regions.append(
                (begin, size, family, direction, position, target, source_peer)
            )
            begin += size

    def _destination(self, region: _Dynamic, step: _Dynamic) -> _Dynamic:
        """Execute destination logic."""
        begin, size, family, direction, position, _, _ = region
        owner = self.tables[family, 1, (position + 64 - direction * step) % 32]
        return self.output.at[pl.ds(owner * self.rows + begin, size), :, :]

    def _remote(
        self,
        source: _Dynamic,
        target: _Dynamic,
        ss: _Dynamic,
        rs: _Dynamic,
        peer: _Dynamic,
    ) -> _Dynamic:
        """Execute remote logic."""
        return pltpu.make_async_remote_copy(
            source,
            target,
            ss,
            rs,
            device_id=peer,
            device_id_type=pl.DeviceIdType.LOGICAL,
        )

    def _run_small_step(self, step: int, stream: int, region: _Dynamic) -> None:
        """Execute small payload step for one stream."""
        begin, size, _, _, _, peer, source_peer = region
        piece = self.payload.at[pl.ds(begin, size), :, :]
        ss = self.send.at[stream]
        credit = self.sem_credits.at[stream]
        st = self.stores.at[stream]
        if step:
            old = self.received.at[(step - 1) % 2, pl.ds(begin, size), :, :]
            transfer = self._remote(
                piece, old, ss, self.recv.at[(step - 1) % 2, stream], peer
            )
            transfer.wait_send()
            transfer.wait_recv()
            pltpu.make_async_copy(piece, self._destination(region, step - 1), st).wait()
            piece[...] = old[...]
            pl.semaphore_signal(
                credit,
                1,
                device_id=source_peer,
                device_id_type=pl.DeviceIdType.LOGICAL,
            )
        dest = self._destination(region, step)
        store = pltpu.make_async_copy(piece, dest, st)
        store.start()
        if step < _CONST_31:
            if step >= _CONST_2:
                pl.semaphore_wait(credit, 1)
            future = self.received.at[step % 2, pl.ds(begin, size), :, :]
            self._remote(
                piece, future, ss, self.recv.at[step % 2, stream], peer
            ).start()
        else:
            store.wait()
            pl.semaphore_wait(credit, 2)

    def _run_small(self) -> _Dynamic:
        """Execute small gather loop."""
        for step in range(32):
            for stream, region in enumerate(self.regions):
                self._run_small_step(step, stream, region)
            yield

    def _run_large_step(
        self, step: int, stream: int, region: _Dynamic, pending: list[_Dynamic]
    ) -> None:
        """Execute large gather step for one stream."""
        begin, size, _, _, _, peer, source_peer = region
        if pending[stream] is not None:
            transfer, incoming, completed_step = pending[stream]
            transfer.wait_send()
            self._remote(
                incoming,
                incoming,
                self.send.at[stream],
                self.recv.at[completed_step % 2, stream],
                peer,
            ).wait_recv()
            if completed_step + 2 < _CONST_31:
                pl.semaphore_signal(
                    self.sem_credits.at[stream],
                    1,
                    device_id=source_peer,
                    device_id_type=pl.DeviceIdType.LOGICAL,
                )
        if step < _CONST_31:
            source = (
                self.payload.at[pl.ds(begin, size), :, :]
                if step == 0
                else self._destination(region, step)
            )
            dest = self._destination(region, step)
            incoming = self._destination(region, step + 1)
            if step >= _CONST_2:
                pl.semaphore_wait(self.sem_credits.at[stream], 1)
            transfer = self._remote(
                source, dest, self.send.at[stream], self.recv.at[step % 2, stream], peer
            )
            transfer.start()
            pending[stream] = transfer, incoming, step

    def _run_large(self) -> _Dynamic:
        """Execute large gather loop."""
        own = self.output.at[pl.ds(self.rank * self.rows, self.rows), :, :]
        seed = pltpu.make_async_copy(self.payload, own, self.stores)
        seed.start()
        pending: list[_Dynamic] = [None] * 12
        for step in range(32):
            for stream, region in enumerate(self.regions):
                self._run_large_step(step, stream, region, pending)
            yield
        seed.wait()

    def run(self) -> _Dynamic:
        """Run latent gather."""
        if self.rows <= _CONST_128:
            yield from self._run_small()
        else:
            yield from self._run_large()


def _latent_gather_steps(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Twelve directional streams, with two receive slots per stream.

    Windows up to 4K forward through VMEM, copying the completed receive
    slot into payload.
    8K forwards immutable final HBM ranges to avoid that copy. The latter still
    returns credits before reusing receive counters; immutable data alone does
    not make reusing an unconsumed semaphore safe.
    """
    yield from _LatentGatherRunner(*_p_args, **_p_kw).run()


def _latent_ready_init(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Seed twelve independently progressing streams of the 8K latent gather.

    Cache neighbor IDs and owner row offsets in SMEM once. Each stream tracks
    its hop and receive/forward/complete phase; state[0] counts active streams.
    """
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        tables,
        payload,
        output,
        send,
        recv,
        stores,
        state,
        meta,
    ) = (
        _p_next(_p_iter, _p_get("tables")),
        _p_next(_p_iter, _p_get("payload")),
        _p_next(_p_iter, _p_get("output")),
        _p_next(_p_iter, _p_get("send")),
        _p_next(_p_iter, _p_get("recv")),
        _p_next(_p_iter, _p_get("stores")),
        _p_next(_p_iter, _p_get("state")),
        _p_next(_p_iter, _p_get("meta")),
    )
    rank = _p_get("rank")
    rows = payload.shape[0]
    if rows != _CONST_256:
        _err_msg = "Assertion failed: rows == 256"
        raise AssertionError(_err_msg)
    for directed in range(6):
        family, direction = directed // 2, -1 if directed % 2 == 0 else 1
        position = tables[family, 0, rank]
        meta[directed * 34] = tables[family, 1, (position + direction) & 31]
        meta[directed * 34 + 1] = tables[family, 1, (position - direction) & 31]

        @pl.loop(0, 32)
        def owners(
            step: _Dynamic,
            *,
            family: _Dynamic = family,
            position: _Dynamic = position,
            direction: _Dynamic = direction,
            directed: _Dynamic = directed,
        ) -> _Dynamic:
            """Execute owners logic."""
            owner = tables[family, 1, (position - direction * step) & 31]
            meta[directed * 34 + 2 + step] = owner * rows

    pltpu.make_async_copy(
        payload, output.at[pl.ds(rank * rows, rows), :, :], stores
    ).start()
    state[0] = 12
    begin = 0
    for stream in range(12):
        size = rows // 12 + (stream < rows % 12)
        part = pl.ds(begin, size)
        pltpu.make_async_remote_copy(
            payload.at[part, :, :],
            output.at[pl.ds(rank * rows + begin, size), :, :],
            send.at[stream],
            recv.at[0, stream],
            device_id=meta[(stream % 6) * 34],
            device_id_type=pl.DeviceIdType.LOGICAL,
        ).start()
        state[1 + stream] = 1
        state[13 + stream] = 0
        begin += size


class _LatentReadyRunner:
    """Advance ready streams without blocking unrelated compute or streams."""

    def __init__(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Initialize latent ready runner."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            self.payload,
            self.output,
            self.send,
            self.recv,
            self.sem_credits,
            self.stores,
            self.state,
            self.meta,
        ) = (
            _p_next(_p_iter, _p_get("payload")),
            _p_next(_p_iter, _p_get("output")),
            _p_next(_p_iter, _p_get("send")),
            _p_next(_p_iter, _p_get("recv")),
            _p_next(_p_iter, _p_get("credits")),
            _p_next(_p_iter, _p_get("stores")),
            _p_next(_p_iter, _p_get("state")),
            _p_next(_p_iter, _p_get("meta")),
        )
        self.rank = _p_kw.get("rank")
        self.drain = _p_kw.get("drain", False)
        self.rows = self.payload.shape[0]

    def _receive_consume(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Consume incoming transfer and handle credits."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        stream = _p_next(_p_iter, _p_get("stream"))
        base = _p_next(_p_iter, _p_get("base"))
        start = _p_next(_p_iter, _p_get("start"))
        size = _p_next(_p_iter, _p_get("size"))
        step = _p_next(_p_iter, _p_get("step"))
        bank = _p_next(_p_iter, _p_get("bank"))
        incoming = self.output.at[pl.ds(self.meta[base + 2 + step] + start, size), :, :]
        transfer = pltpu.make_async_remote_copy(
            incoming,
            incoming,
            self.send.at[stream],
            self.recv.at[bank, stream],
            device_id=self.meta[base],
            device_id_type=pl.DeviceIdType.LOGICAL,
        )
        transfer.wait_send()
        transfer.wait_recv()
        pl.when(step < _CONST_30)(
            lambda: pl.semaphore_signal(
                self.sem_credits.at[stream],
                1,
                device_id=self.meta[base + 1],
                device_id_type=pl.DeviceIdType.LOGICAL,
            )
        )
        self.state[13 + stream] = jnp.where(step == _CONST_31, 2, 1)
        pl.when(step == _CONST_31)(lambda: self._complete())

    def _complete(self) -> None:
        """Complete a stream."""
        self.state[0] = self.state[0] - 1

    def _receive_step(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Check if receive is ready and consume."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        stream = _p_next(_p_iter, _p_get("stream"))
        base = _p_next(_p_iter, _p_get("base"))
        start = _p_next(_p_iter, _p_get("start"))
        size = _p_next(_p_iter, _p_get("size"))
        units = _p_next(_p_iter, _p_get("units"))
        step = _p_next(_p_iter, _p_get("step"))
        bank = (step - 1) & 1
        ready = (pl.semaphore_read(self.send.at[stream]) >= units) & (
            pl.semaphore_read(self.recv.at[bank, stream]) >= units
        )
        pl.when(ready)(
            lambda: self._receive_consume(stream, base, start, size, step, bank)
        )

    def _forward_issue(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Issue forward remote copy."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        stream = _p_next(_p_iter, _p_get("stream"))
        base = _p_next(_p_iter, _p_get("base"))
        start = _p_next(_p_iter, _p_get("start"))
        size = _p_next(_p_iter, _p_get("size"))
        step = _p_next(_p_iter, _p_get("step"))
        pl.when(step >= _CONST_2)(
            lambda: pl.semaphore_wait(self.sem_credits.at[stream], 1)
        )
        source = self.output.at[pl.ds(self.meta[base + 2 + step] + start, size), :, :]
        pltpu.make_async_remote_copy(
            source,
            source,
            self.send.at[stream],
            self.recv.at[step & 1, stream],
            device_id=self.meta[base],
            device_id_type=pl.DeviceIdType.LOGICAL,
        ).start()
        self.state[1 + stream] = step + 1
        self.state[13 + stream] = 0

    def _forward_step(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Check if forward can be issued."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        stream = _p_next(_p_iter, _p_get("stream"))
        base = _p_next(_p_iter, _p_get("base"))
        start = _p_next(_p_iter, _p_get("start"))
        size = _p_next(_p_iter, _p_get("size"))
        step = _p_next(_p_iter, _p_get("step"))
        available = (step < _CONST_2) | (
            pl.semaphore_read(self.sem_credits.at[stream]) >= 1
        )
        pl.when(available)(lambda: self._forward_issue(stream, base, start, size, step))

    def _sweep_stream(self, stream: int, start: int, size: int) -> None:
        """Sweep one stream."""
        base = (stream % 6) * 34
        units = size * 2 * 1792 * 2 // 32
        step = self.state[1 + stream]
        pl.when(self.state[13 + stream] == 0)(
            lambda: self._receive_step(stream, base, start, size, units, step)
        )
        pl.when(self.state[13 + stream] == 1)(
            lambda: self._forward_step(stream, base, start, size, step)
        )

    def _sweep(self) -> None:
        """Sweep all 12 streams."""
        begin = 0
        for stream in range(12):
            size = self.rows // 12 + (stream < self.rows % 12)
            start = begin
            begin += size
            self._sweep_stream(stream, start, size)

    def run(self) -> None:
        """Run latent ready progress."""
        if self.drain:
            jax.lax.while_loop(
                lambda _i: self.state[0] != 0,
                lambda i: (self._sweep(), i + 1)[1],
                jnp.int32(0),
            )
            pltpu.make_async_copy(
                self.payload,
                self.output.at[pl.ds(self.rank * self.rows, self.rows), :, :],
                self.stores,
            ).wait()
        else:
            pl.when(self.state[0] != 0)(lambda: self._sweep())


def _latent_ready_progress(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Advance ready streams without blocking unrelated compute or streams.

    A receive consumes both DMA completion counts. Credits protect the two
    alternating receive semaphores from reuse before the peer consumes them.
    The final drain completes every stream before routed experts read HBM.
    """
    return _LatentReadyRunner(*_p_args, **_p_kw).run()


def _stripe_validity(rows: _Dynamic, length: _Dynamic, rank: _Dynamic) -> _Dynamic:
    """Execute stripe validity logic."""
    row = jax.lax.broadcasted_iota(jnp.int32, (rows, 1), 0)
    stripe = min(rows, 64)
    return (row // stripe * (32 * stripe) + rank * stripe + row % stripe) < length


class _PrologueRunner:
    """Router, latent projection and shared experts, with incremental collectives."""

    def __init__(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Initialize prologue runner."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            self.tables,
            self.control,
            self.hidden,
            self.router,
            self.bias,
            self.latent_weight,
            self.shared_gu,
            self.shared_down,
            self.shared,
            self.latent_hbm,
            self.token_ids,
            self.probabilities,
            self.counts,
            self.maximum,
        ) = (
            _p_next(_p_iter, _p_get("tables")),
            _p_next(_p_iter, _p_get("control")),
            _p_next(_p_iter, _p_get("hidden")),
            _p_next(_p_iter, _p_get("router")),
            _p_next(_p_iter, _p_get("bias")),
            _p_next(_p_iter, _p_get("latent_weight")),
            _p_next(_p_iter, _p_get("shared_gu")),
            _p_next(_p_iter, _p_get("shared_down")),
            _p_next(_p_iter, _p_get("shared")),
            _p_next(_p_iter, _p_get("latent_hbm")),
            _p_next(_p_iter, _p_get("token_ids")),
            _p_next(_p_iter, _p_get("probabilities")),
            _p_next(_p_iter, _p_get("counts")),
            _p_next(_p_iter, _p_get("maximum")),
        )
        (
            self.counts_ready,
            self.metadata_ready,
        ) = (
            _p_get("counts_ready"),
            _p_get("metadata_ready"),
        )
        self.rows, self.hidden_size = self.hidden.shape
        self.rank = jax.lax.axis_index("k3_sp")
        self.parity, self.peer = self.rank % 2, self.rank ^ 1
        self.family = self.control[1]
        self.router = self.router.at[self.family]
        self.bias = self.bias.at[self.family]
        self.latent_weight = self.latent_weight.at[self.family]
        self.shared_gu = self.shared_gu.at[self.family]
        self.shared_down = self.shared_down.at[self.family]
        self.valid = _stripe_validity(self.rows, self.control[0], self.rank)
        self.route_steps = iter(())
        self.metadata_steps = iter(())
        self.route_started = False
        self.metadata_started = False
        self.latent_steps = iter(())
        self.latent_tick = 0

    def _start_metadata(self, s: _Dynamic) -> None:
        """Execute start metadata logic."""
        if not self.metadata_started:
            self.metadata_started = True
            self.metadata_steps = _metadata_steps(
                s["route_wire"],
                s["keys"],
                self.probabilities,
                self.token_ids,
                self.counts,
                self.maximum,
                tokens=self.rows * 32,
                rank=self.rank,
                counts_ready=self.counts_ready,
            )

    def _metadata_progress(self, _s: _Dynamic = None) -> None:
        """Execute metadata progress logic."""
        if self.metadata_started:
            for _ in range(2 if self.rows <= _CONST_128 else 6):
                next(self.metadata_steps, None)

    def _latent_progress(self, s: _Dynamic, *, drain: bool = False) -> None:
        """Execute latent progress logic."""
        if self.rows == _CONST_256:
            if self.latent_tick == 0:
                _latent_ready_init(
                    self.tables,
                    s["latent_payload"],
                    self.latent_hbm,
                    s["latent_send_sems"],
                    s["latent_recv_sems"],
                    s["latent_stores"],
                    s["latent_state"],
                    s["latent_meta"],
                    rank=self.rank,
                )
            else:
                _latent_ready_progress(
                    s["latent_payload"],
                    self.latent_hbm,
                    s["latent_send_sems"],
                    s["latent_recv_sems"],
                    s["latent_credits"],
                    s["latent_stores"],
                    s["latent_state"],
                    s["latent_meta"],
                    rank=self.rank,
                    drain=drain,
                )
            self.latent_tick += 1
        elif drain:
            for _ in self.latent_steps:
                pass
        else:
            next(self.latent_steps, None)

    def _gu_copy(
        self, panel: int, offset: int, buffers: _Dynamic, sems: _Dynamic
    ) -> _Dynamic:
        """Execute gu copy logic."""
        bank = panel % 2
        return pltpu.make_async_copy(
            self.shared_gu.at[:, pl.ds(offset + panel * 256, 256)],
            buffers[bank],
            sems[bank],
        )

    def _down_copy(self, s: _Dynamic, panel: int) -> _Dynamic:
        """Execute down copy logic."""
        return pltpu.make_async_copy(
            self.shared_down.at[:, pl.ds(panel * 256, 256)],
            s["down_buffers"][panel % 2],
            s["down_sems"][panel % 2],
        )

    def _finish_reduction(self, s: _Dynamic, pending: _Dynamic) -> None:
        """Execute finish reduction logic."""
        transfer, bank, panel = pending
        own = s["partial"][bank][pl.ds(self.parity * self.rows, self.rows), :]
        transfer.wait_send()
        transfer.wait_recv()
        self.shared[:, pl.ds(panel * 256, 256)] = (own + s["peer_y"][bank][...]).astype(
            jnp.bfloat16
        )

    def _start_route_steps(self, s: _Dynamic, top: _Dynamic) -> None:
        """Start route gather steps when top-16 finish."""
        ids, probs = top.finish(self.valid)
        if self.rows < _CONST_128:
            wire = jnp.concatenate(
                (ids, jax.lax.bitcast_convert_type(probs, jnp.int32)), axis=1
            )
        else:
            wire = jnp.concatenate(
                (
                    ids.reshape(self.rows // 8, 128),
                    jax.lax.bitcast_convert_type(probs, jnp.int32).reshape(
                        self.rows // 8, 128
                    ),
                ),
                axis=0,
            )
        s["route_payload"][...] = wire.reshape(self.rows // 4, 128)
        self.route_steps = _route_gather_steps(
            s["route_payload"],
            s["route_wire"],
            s["route_send"],
            s["route_recv"],
            rank=self.rank,
        )
        next(self.route_steps, None)
        self.route_started = True

    def _route_panel(
        self, s: _Dynamic, col: int, product: _Dynamic, top: _Dynamic
    ) -> None:
        """Process one latent product panel."""
        if self.route_started:
            stage = next(self.route_steps, None)
            if self.rows <= _CONST_128 and stage is None:
                self._start_metadata(s)
        if self.rows <= _CONST_128:
            self._metadata_progress()
        if self.rows == _CONST_256:
            fragment = product.astype(jnp.bfloat16)
            if col < _CONST_1792:
                self.first_half = fragment
            else:
                if self.first_half is None:
                    _msg = "Missing first_half"
                    raise ValueError(_msg)
                paired = jnp.concatenate((self.first_half, fragment), axis=1)
                s["latent_payload"][:, :, pl.ds(col - 1792, 256)] = paired.reshape(
                    self.rows, 2, 256
                )
        else:
            s["latent"][:, pl.ds(col, 256)] = product.astype(jnp.bfloat16)
        for _ in range(min(2, 16 - top.index)):
            top.step()
        if top.index == _CONST_16 and not self.route_started:
            self._start_route_steps(s, top)

    def _route_phase(
        self,
        s: _Dynamic,
        own: _Dynamic,
        gate_buffers: _Dynamic,
        gate_sems: _Dynamic,
        router_v: _Dynamic,
    ) -> None:
        """Execute routing and latent projection phase."""
        hx = own[...]
        logits = jnp.matmul(hx, router_v[...], preferred_element_type=jnp.float32)
        top = _Top16(logits, s["bias"][...])
        if self.rows == _CONST_256:
            latent_buffers = tuple(
                s["router_gu"].at[:, pl.ds(i * 256, 256)] for i in range(4)
            )
            latent_products = _latent_half_pair_panels(
                hx, self.latent_weight, latent_buffers, s["gu_sems"]
            )
        else:
            latent_products = _dense_panels(
                hx, self.latent_weight, gate_buffers, gate_sems, columns=4096
            )
        self.first_half = None
        for col, product in latent_products:
            self._route_panel(s, col, product, top)
        for _ in self.route_steps:
            pass
        self._start_metadata(s)
        if self.rows <= _CONST_128:
            s["latent_payload"][...] = s["latent"][:, :3584].reshape(self.rows, 2, 1792)
        self.latent_steps = _latent_gather_steps(
            self.tables,
            s["latent_payload"],
            s["latent_recv"],
            self.latent_hbm,
            s["latent_send_sems"],
            s["latent_recv_sems"],
            s["latent_credits"],
            s["latent_stores"],
            rank=self.rank,
        )

    def _gu_loop(
        self,
        s: _Dynamic,
        gate_buffers: _Dynamic,
        up_buffers: _Dynamic,
        gate_sems: _Dynamic,
        up_sems: _Dynamic,
    ) -> None:
        """Execute GU panels compute loop."""
        self._gu_copy(0, 0, gate_buffers, gate_sems).start()
        self._gu_copy(0, 3072, up_buffers, up_sems).start()
        if self.rows == _CONST_256:
            self._latent_progress(s)
        for panel in range(12):
            if self.rows <= _CONST_128:
                self._latent_progress(s)
            bank = panel % 2
            self._gu_copy(panel, 0, gate_buffers, gate_sems).wait()
            self._gu_copy(panel, 3072, up_buffers, up_sems).wait()
            if panel + 1 < _CONST_12:
                self._gu_copy(panel + 1, 0, gate_buffers, gate_sems).start()
                self._gu_copy(panel + 1, 3072, up_buffers, up_sems).start()
            for row in (0, self.rows):
                lhs = s["pair"][pl.ds(row, self.rows), :]
                gate = jnp.matmul(
                    lhs, gate_buffers[bank][...], preferred_element_type=jnp.float32
                )
                up = jnp.matmul(
                    lhs, up_buffers[bank][...], preferred_element_type=jnp.float32
                )
                self._metadata_progress(s)
                activated = (4.0 * jnp.tanh(gate / 4.0)) * (
                    1.0 / (1.0 + jnp.exp(-gate))
                )
                activated *= 25.0 * jnp.tanh(up / 25.0)
                s["middle"][pl.ds(row, self.rows), pl.ds(panel * 256, 256)] = (
                    activated.astype(jnp.bfloat16)
                )
                if self.rows == _CONST_256:
                    self._latent_progress(s)

    def _down_loop(self, s: _Dynamic) -> None:
        """Execute down projection compute loop."""
        self._down_copy(s, 0).start()
        pending = None
        for panel in range(28):
            bank = panel % 2
            self._down_copy(s, panel).wait()
            if panel + 1 < _CONST_28:
                self._down_copy(s, panel + 1).start()
            for row in (0, self.rows):
                product = jnp.matmul(
                    s["middle"][pl.ds(row, self.rows), :],
                    s["down_buffers"][bank][...],
                    preferred_element_type=jnp.float32,
                )
                self._metadata_progress(s)
                s["partial"][bank][pl.ds(row, self.rows), :] = product
                if row == self.rows:
                    if pending is not None:
                        self._finish_reduction(s, pending)
                    transfer = pltpu.make_async_remote_copy(
                        s["partial"][bank].at[
                            pl.ds((1 - self.parity) * self.rows, self.rows), :
                        ],
                        s["peer_y"][bank],
                        s["reduce_send"][bank],
                        s["reduce_recv"][bank],
                        device_id=self.peer,
                        device_id_type=pl.DeviceIdType.LOGICAL,
                    )
                    transfer.start()
                    pending = transfer, bank, panel
                if self.rows in (256, row):
                    self._latent_progress(s)
        self._finish_reduction(s, pending)

    def run_body(self, s: _Dynamic) -> None:
        """Execute prologue body logic."""
        own = s["pair"].at[pl.ds(self.parity * self.rows, self.rows), :]
        copy = pltpu.make_async_copy(self.hidden, own, s["input_sem"])
        copy.start()
        copy.wait()
        if self.rows <= _CONST_128:
            copy = pltpu.make_async_copy(self.router, s["router"], s["input_sem"])
            copy.start()
            copy.wait()
        copy = pltpu.make_async_copy(self.bias, s["bias"], s["input_sem"])
        copy.start()
        copy.wait()
        _enter_mesh()
        pl.semaphore_signal(
            s["peer_credit"],
            1,
            device_id=self.peer,
            device_id_type=pl.DeviceIdType.LOGICAL,
        )
        pl.semaphore_wait(s["peer_credit"], 1)
        peer_input = pltpu.make_async_remote_copy(
            own,
            own,
            s["peer_send"],
            s["peer_recv"],
            device_id=self.peer,
            device_id_type=pl.DeviceIdType.LOGICAL,
        )
        peer_input.start()
        if self.rows == _CONST_256:
            buffers = tuple(s["router_gu"].at[:, pl.ds(i * 256, 256)] for i in range(4))
            router_v = s["router_gu"].at[:, :896]
            copy = pltpu.make_async_copy(self.router, router_v, s["gu_sems"][0])
            copy.start()
            copy.wait()
        else:
            buffers, router_v = s["gu_buffers"], s["router"]
        gate_buffers, up_buffers = buffers[:2], buffers[2:]
        gate_sems, up_sems = s["gu_sems"][:2], s["gu_sems"][2:]

        self._route_phase(s, own, gate_buffers, gate_sems, router_v)
        peer_input.wait_send()
        peer_input.wait_recv()

        self._gu_loop(s, gate_buffers, up_buffers, gate_sems, up_sems)
        self._down_loop(s)

        if self.rows == _CONST_256:
            for _ in self.metadata_steps:
                self._latent_progress(s)
            self.metadata_ready()
            self._latent_progress(s, drain=True)
        else:
            self._latent_progress(s, drain=True)
            for _ in self.metadata_steps:
                pass

    def run(self) -> None:
        """Run scoped prologue."""
        vmem, dma = pltpu.VMEM, pltpu.SemaphoreType.DMA
        scratch = {
            "pair": vmem((2 * self.rows, self.hidden_size), jnp.bfloat16),
            "router": vmem((self.hidden_size, 896), jnp.bfloat16)
            if self.rows <= _CONST_128
            else None,
            "router_gu": vmem((self.hidden_size, 1024), jnp.bfloat16)
            if self.rows == _CONST_256
            else None,
            "gu_buffers": tuple(
                vmem((self.hidden_size, 256), jnp.bfloat16) for _ in range(4)
            )
            if self.rows <= _CONST_128
            else (),
            "gu_sems": tuple(dma for _ in range(4)),
            "bias": vmem((1, 896), jnp.float32),
            "input_sem": dma,
            "peer_send": dma,
            "peer_recv": dma,
            "peer_credit": pltpu.SemaphoreType.REGULAR,
            "latent": vmem((self.rows, 4096), jnp.bfloat16)
            if self.rows <= _CONST_128
            else None,
            "route_payload": vmem((self.rows // 4, 128), jnp.int32),
            "route_wire": vmem((self.rows * 8, 128), jnp.int32),
            "route_send": dma((5,)),
            "route_recv": dma((5,)),
            "keys": vmem((self.rows * 4, 128), jnp.int32),
            "latent_payload": vmem((self.rows, 2, 1792), jnp.bfloat16),
            "latent_recv": vmem((2, self.rows, 2, 1792), jnp.bfloat16)
            if self.rows <= _CONST_128
            else None,
            "latent_state": pltpu.SMEM((25,), jnp.int32)
            if self.rows == _CONST_256
            else None,
            "latent_meta": pltpu.SMEM((6 * 34,), jnp.int32)
            if self.rows == _CONST_256
            else None,
            "latent_send_sems": dma((12,)),
            "latent_recv_sems": dma((2, 12)),
            "latent_credits": pltpu.SemaphoreType.REGULAR((12,)),
            "latent_stores": dma((12,)) if self.rows <= _CONST_128 else dma,
            "middle": vmem((2 * self.rows, 3072), jnp.bfloat16),
            "down_buffers": tuple(vmem((3072, 256), jnp.bfloat16) for _ in range(2)),
            "down_sems": (dma, dma),
            "partial": tuple(vmem((2 * self.rows, 256), jnp.float32) for _ in range(2)),
            "peer_y": tuple(vmem((self.rows, 256), jnp.float32) for _ in range(2)),
            "reduce_send": (dma, dma),
            "reduce_recv": (dma, dma),
        }
        pl.run_scoped(self.run_body, scratch)


def _prologue(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Router, latent projection and shared experts, with incremental collectives.

    The caller retains shared/metadata outputs across this scope. All temporary
    projection and communication buffers die before routed-expert scratch is
    allocated. The initial HBM input load targets our half of the peer buffer;
    both the router and shared expert use this same copy.
    """
    return _PrologueRunner(*_p_args, **_p_kw).run()


def _buffering_spec(spec: _Dynamic) -> _Dynamic:
    """Execute buffering logic."""
    return dataclasses.replace(spec, pipeline_mode=pl.Buffered(buffer_count=2))


def _shifted_spec(spec: _Dynamic) -> _Dynamic:
    """Execute shifted logic."""
    index_map = spec.index_map
    return dataclasses.replace(spec, index_map=lambda n, g, k: index_map(n + 1, g, k))


def _is_buffer(value: _Dynamic) -> _Dynamic:
    """Execute is buffer logic."""
    return hasattr(value, "window_ref")


def _pad_buffer(buffer: _Dynamic) -> _Dynamic:
    """Execute pad logic."""
    stride = buffer.window_ref.spec.tiles.block_shape[0]
    return pl.run_scoped(
        lambda p: p,
        p=pltpu.VMEM((buffer.shape[0], stride + buffer.shape[1]), buffer.dtype),
    )


def _trim_buffer(buffer: _Dynamic, original: _Dynamic) -> _Dynamic:
    """Execute trim logic."""
    stride = original.window_ref.spec.tiles.block_shape[0]
    return buffer[:, stride:]


class _RoutedExpertsRunner:
    """Joint GU/SiTU/first-down pipeline followed by remaining columns."""

    def __init__(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Initialize the routed experts runner."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            self.latent,
            self.gu_weights,
            self.gu_scales,
            self.rhs,
            self.scales,
            self.counts,
            self.destinations,
            self.probabilities,
            self.tables,
            self.layer,
            self.offset,
            self.middle,
            self.out,
        ) = (
            _p_next(_p_iter, _p_get("latent")),
            _p_next(_p_iter, _p_get("gu_weights")),
            _p_next(_p_iter, _p_get("gu_scales")),
            _p_next(_p_iter, _p_get("rhs")),
            _p_next(_p_iter, _p_get("scales")),
            _p_next(_p_iter, _p_get("counts")),
            _p_next(_p_iter, _p_get("destinations")),
            _p_next(_p_iter, _p_get("probabilities")),
            _p_next(_p_iter, _p_get("tables")),
            _p_next(_p_iter, _p_get("layer")),
            _p_next(_p_iter, _p_get("offset")),
            _p_next(_p_iter, _p_get("middle")),
            _p_next(_p_iter, _p_get("out")),
        )
        (
            self.tokens,
            self.probabilities_ready,
        ) = (
            _p_get("tokens"),
            _p_get("probabilities_ready"),
        )
        self.columns = 1792 if self.tokens < _CONST_8192 else 896
        self.tile = 128 if self.tokens < _CONST_8192 else 256
        self.unroll = 8 if self.tokens < _CONST_8192 else 32
        self.gu_unroll = 4 if self.tokens < _CONST_8192 else 8
        self.first_down_unroll = 2 if self.tokens < _CONST_8192 else 8
        self.metadata_row_capacity = 0 if self.tokens < _CONST_8192 else 32768
        self.ring_dtype = jnp.float32
        if self.tokens not in (512, 1024, 2048, 4096, 8192):
            _err_msg = "Assertion failed: tokens in (512, 1024, 2048, 4096, 8192)"
            raise AssertionError(_err_msg)
        self.x = jax.ShapeDtypeStruct((self.tokens * 16, 1536), self.latent.dtype)
        self.rows, self.width = (
            self.metadata_row_capacity or self.x.shape[0],
            self.rhs.shape[-1],
        )
        if not (self.rows <= self.x.shape[0]):
            _err_msg = "Assertion failed: rows <= x.shape[0]"
            raise AssertionError(_err_msg)
        self.tm, self.tn, self.localrows = (
            self.tile,
            self.columns,
            self.tokens // 32,
        )
        self.packed_half = 512 if self.tokens == _CONST_8192 else self.tn // 2
        if self.width % self.tn != 0:
            _err_msg = "Assertion failed: width % tn == 0"
            raise AssertionError(_err_msg)
        self.limit = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.98)
        self.cfg = k.make_gmm_configs(
            jax.ShapeDtypeStruct((self.rows, self.x.shape[-1]), self.x.dtype),
            jax.ShapeDtypeStruct(self.rhs.shape[1:], self.rhs.dtype),
            jax.ShapeDtypeStruct(self.scales.shape[1:], self.scales.dtype),
            self.counts,
            self.offset,
            tile_info=k.calculate_tiling,
            vmem_limit_bytes=self.limit,
            out_dtype=None,
            acc_dtype=None,
            fuse_act=None,
            rhs_quant_dtype=jnp.float4_e2m1fn,
        )
        self.cfg = dataclasses.replace(
            self.cfg,
            tiles=dataclasses.replace(self.cfg.tiles, tile_m=self.tm, tile_n=self.tn),
        )
        self.sub = self.cfg.dims.size_lhs_sublane
        self.max_gm = self.cfg.dims.size_group + pl.cdiv(self.rows, self.tm) - 1
        self.regions = operator.attrgetter("_regions")(k)(self.localrows, self.tn)
        self.streams = len(self.regions)
        self.store_tile = 128
        self.gu_cfg = k.make_gmm_configs(
            jax.ShapeDtypeStruct((self.rows, 3584), self.latent.dtype),
            jax.ShapeDtypeStruct(self.gu_weights.shape[1:], self.gu_weights.dtype),
            jax.ShapeDtypeStruct(self.gu_scales.shape[1:], self.gu_scales.dtype),
            self.counts,
            self.offset,
            tile_info=k.calculate_tiling,
            vmem_limit_bytes=self.limit,
            out_dtype=None,
            acc_dtype=None,
            fuse_act="situ:4:25",
            rhs_quant_dtype=jnp.float4_e2m1fn,
        )
        self.gu_cfg = dataclasses.replace(
            self.gu_cfg,
            tiles=dataclasses.replace(
                self.gu_cfg.tiles, tile_m=self.tm, tile_n=1536, tile_k=3584
            ),
        )
        if self.gu_cfg.dims.size_lhs_sublane != self.sub:
            _err_msg = "Assertion failed: gu_cfg.dims.size_lhs_sublane == sub"
            raise AssertionError(_err_msg)
        for config in (self.cfg, self.gu_cfg):
            if not (config.rhs_cfgs.should_dequantize_before_matmul):
                _err_msg = (
                    "Assertion failed: config.rhs_cfgs.should_dequantize_before_matmul"
                )
                raise AssertionError(_err_msg)
            if config.rhs_cfgs.should_dequantize_after_matmul:
                _err_msg = (
                    "Assertion failed: not"
                    " config.rhs_cfgs.should_dequantize_after_matmul"
                )
                raise AssertionError(_err_msg)

    def run(self) -> _Dynamic:
        """Execute run logic."""
        return pl.run_scoped(
            self.run_scoped,
            metadata=k.MetadataRef(
                gm_id_to_group_id=pltpu.SMEM((self.max_gm,), jnp.int32),
                gm_id_to_m_offset=pltpu.SMEM((self.max_gm + 1,), jnp.int32),
            ),
            packed=pltpu.VMEM((2, self.tm, 2, self.packed_half), self.x.dtype),
            sums=pltpu.VMEM((self.tokens, 2, self.packed_half), jnp.float32),
            cast=pltpu.VMEM((self.store_tile, self.tn), self.x.dtype),
            sends=pltpu.SemaphoreType.DMA((self.streams,)),
            receives=pltpu.SemaphoreType.DMA((2, self.streams)),
            credits=pltpu.SemaphoreType.REGULAR((self.streams,)),
            stores=pltpu.SemaphoreType.DMA,
            payload=pltpu.VMEM((self.localrows, self.tn), self.ring_dtype),
            rx=pltpu.VMEM((2, self.localrows, self.tn), self.ring_dtype),
            state=pltpu.SMEM((1 + 2 * self.streams,), jnp.int32),
            ring_meta=pltpu.SMEM((6 * 34,), jnp.int32),
        )

    def run_scoped(self, **scratch: _Dynamic) -> _Dynamic:
        """Execute run scoped logic."""
        return self.kernel(
            self.counts,
            self.offset,
            self.layer,
            self.tables,
            self.destinations,
            self.probabilities,
            self.latent,
            k.WeightsRef(weight=self.gu_weights, scale=self.gu_scales),
            k.WeightsRef(weight=self.rhs, scale=self.scales),
            self.middle,
            self.out,
            **scratch,
        )

    def kernel(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
        """Execute kernel logic."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            counts,
            offset,
            layer,
            tables,
            destinations,
            probabilities,
            latent,
            gu,
            w,
            middle,
            out,
            metadata,
            packed,
            sums,
            cast,
            sends,
            receives,
            cred_sems,
            stores,
            payload,
            rx,
            state,
            ring_meta,
        ) = (
            _p_next(_p_iter, _p_get("counts")),
            _p_next(_p_iter, _p_get("offset")),
            _p_next(_p_iter, _p_get("layer")),
            _p_next(_p_iter, _p_get("tables")),
            _p_next(_p_iter, _p_get("destinations")),
            _p_next(_p_iter, _p_get("probabilities")),
            _p_next(_p_iter, _p_get("latent")),
            _p_next(_p_iter, _p_get("gu")),
            _p_next(_p_iter, _p_get("w")),
            _p_next(_p_iter, _p_get("middle")),
            _p_next(_p_iter, _p_get("out")),
            _p_next(_p_iter, _p_get("metadata")),
            _p_next(_p_iter, _p_get("packed")),
            _p_next(_p_iter, _p_get("sums")),
            _p_next(_p_iter, _p_get("cast")),
            _p_next(_p_iter, _p_get("sends")),
            _p_next(_p_iter, _p_get("receives")),
            _p_next(_p_iter, _p_get("credits")),
            _p_next(_p_iter, _p_get("stores")),
            _p_next(_p_iter, _p_get("payload")),
            _p_next(_p_iter, _p_get("rx")),
            _p_next(_p_iter, _p_get("state")),
            _p_next(_p_iter, _p_get("ring_meta")),
        )
        self.k_counts = counts
        self.k_offset = offset
        self.k_layer = layer
        self.k_tables = tables
        self.k_destinations = destinations
        self.k_probabilities = probabilities
        self.k_latent = latent
        self.k_middle = middle
        self.k_out = out
        self.metadata = metadata
        self.packed = packed
        self.sums = sums
        self.cast = cast
        self.sends = sends
        self.receives = receives
        self.credits = cred_sems
        self.stores = stores
        self.payload = payload
        self.rx = rx
        self.state = state
        self.ring_meta = ring_meta

        w = dataclasses.replace(
            w, weight=w.weight.at[layer[0]], scale=w.scale.at[layer[0]]
        )
        self.w = dataclasses.replace(w, weight=w.weight.bitcast(jnp.float4_e2m1fn))
        gu = dataclasses.replace(
            gu,
            weight=gu.weight.at[layer[0]].bitcast(jnp.float4_e2m1fn),
            scale=gu.scale.at[layer[0]],
        )
        self.gu = k.FusedWeightsRef(
            gate=gu, up=jax.tree.map(lambda r: r.at[..., 1536:], gu)
        )
        rank = lax.axis_index("k3_sp")
        self.init_rings(rank)

        active_ng = k.fill_metadata(counts, offset, metadata, cfgs=self.cfg)
        pl.when(active_ng == 0)(self._empty_metadata)
        self.ng = jnp.maximum(active_ng, 1)
        self.active_ng = active_ng

        self.gu_panels = pl.cdiv(3072, pltpu.get_tpu_info().mxu_column_size)
        self.moved_rows = self.gu_panels * self.gu_unroll

        (self.xs, self.ws), _ = k.generate_block_specs(metadata, self.cfg)
        self.state[0] = 0

        (_, gu_ws), self.gu_ys = k.generate_block_specs(metadata, self.gu_cfg)
        self.gu_ws = k.FusedWeightsRef(gate=gu_ws, up=gu_ws)

        pl.run_scoped(
            self.first_column,
            gu_partial=pltpu.VMEM((self.sub, 1536), self.gu_cfg.out_dtype),
            acts=pltpu.VMEM((2, self.tm, 2, 1792), latent.dtype),
            act_sems=pltpu.SemaphoreType.DMA((2,)),
        )

        pl.run_scoped(
            self.remaining_columns,
            subtotal_vmem=pltpu.VMEM((self.tokens, self.tn), self.x.dtype),
        )

    def init_rings(self, rank: _Dynamic) -> None:
        """Execute init rings logic."""
        for directed in range(6):
            family, direction = directed // 2, -1 if directed % 2 == 0 else 1
            position = self.k_tables[family, 0, rank]
            self.ring_meta[directed * 34] = self.k_tables[
                family, 1, (position + direction) & 31
            ]
            self.ring_meta[directed * 34 + 1] = self.k_tables[
                family, 1, (position - direction) & 31
            ]

            def _hop(
                step: _Dynamic,
                d: int = directed,
                f: int = family,
                p: _Dynamic = position,
                di: int = direction,
            ) -> None:
                """Execute hop step logic."""
                owner_mesh = self.k_tables[f, 1, (p - di * (step + 1)) & 31]
                owner = self.k_tables[f, 2, owner_mesh]
                self.ring_meta[d * 34 + 2 + step] = owner * self.localrows

            pl.loop(0, 32)(_hop)

    def _empty_metadata(self) -> None:
        """Execute empty metadata logic."""
        self.metadata.gm_id_to_group_id[0] = 0
        self.metadata.gm_id_to_m_offset[0] = 0
        self.metadata.gm_id_to_m_offset[1] = 0

    def advance(
        self,
        _stop: _Dynamic,
        block: _Dynamic,
        subtotal_vmem: _Dynamic,
        *,
        drain: _Dynamic = False,
        poll_round: _Dynamic = 0,
    ) -> _Dynamic:
        """Execute advance logic."""
        _ready_reduce(
            self.k_tables,
            subtotal_vmem,
            self.k_out.at[:, pl.ds(block * self.tn, self.tn)],
            self.sends,
            self.receives,
            self.credits,
            self.stores,
            self.payload,
            self.rx,
            streams=self.regions,
            buffers=2,
            state=self.state,
            ring_meta=self.ring_meta,
            drain=drain,
            poll_round=poll_round,
            sparse_poll=(self.ng >= _CONST_72),
        )

    def _emit_block(self, b: _Dynamic, subtotal_vmem: _Dynamic) -> None:
        """Execute emit block logic."""
        values = self.sums[pl.ds(b * self.store_tile, self.store_tile)].reshape(
            self.store_tile, 2 * self.packed_half
        )
        self.cast[...] = values[:, : self.tn].astype(jnp.bfloat16)
        subtotal_vmem.at[pl.ds(b * self.store_tile, self.store_tile)][...] = self.cast[
            ...
        ]

    def emit_subtotals(self, subtotal_vmem: _Dynamic) -> None:
        """Execute emit subtotals logic."""
        pl.loop(0, self.tokens // self.store_tile)(
            lambda b: self._emit_block(b, subtotal_vmem)
        )

    def route(self, i: _Dynamic) -> tuple[_Dynamic, _Dynamic]:
        """Execute route logic."""
        return (self.k_destinations[i], self.k_probabilities[i])

    def _consume_update(self, i: _Dynamic, begin: _Dynamic, source: _Dynamic) -> None:
        """Execute consume update logic."""
        dest, weight = self.route(i)
        value = source.at[pl.ds(i - begin, 1)][...].astype(jnp.float32)
        target = self.sums.at[pl.ds(dest, 1)]
        target[...] = target[...] + value * weight

    def consume(
        self,
        tile_g: _Dynamic,
        source: _Dynamic,
        first: _Dynamic = 0,
        *,
        loop_unroll: _Dynamic = None,
    ) -> None:
        """Execute consume logic."""
        if loop_unroll is None:
            loop_unroll = self.unroll
        start = self.metadata.gm_id_to_m_offset[tile_g]
        end = self.metadata.gm_id_to_m_offset[tile_g + 1]
        begin = start // self.sub * self.sub
        start = jnp.minimum(start + first, end)
        full = (end - start) // loop_unroll

        def _chunk(b: _Dynamic) -> None:
            """Execute chunk logic."""
            for u in range(loop_unroll):
                self._consume_update(start + b * loop_unroll + u, begin, source)

        pl.loop(0, full)(_chunk)
        pl.loop(start + full * loop_unroll, end)(
            lambda i: self._consume_update(i, begin, source)
        )

    def gu_work(self, g: _Dynamic, panel: _Dynamic) -> None:
        """Execute gu work logic."""
        previous_g = jnp.maximum(g - 1, 0)
        start = self.metadata.gm_id_to_m_offset[previous_g]
        end = self.metadata.gm_id_to_m_offset[previous_g + 1]
        begin = start // self.sub * self.sub
        for row in range(self.gu_unroll):
            offset = panel * self.gu_unroll + row
            i = jnp.minimum(start + offset, end - 1)
            dest, prob = self.route(i)
            weight = jnp.where((g > 0) & (start + offset < end), prob, 0)
            previous = self.packed.at[(g + 1) % 2]
            value = previous.at[pl.ds(i - begin, 1)][...].astype(jnp.float32)
            target = self.sums.at[pl.ds(dest, 1)]
            target[...] = target[...] + value * weight

    def _overlap_work(
        self,
        panel: _Dynamic,
        previous_g: _Dynamic,
        previous: _Dynamic,
        down_unroll: int,
        *,
        first: bool = False,
    ) -> None:
        """Execute overlap work logic."""
        start = self.metadata.gm_id_to_m_offset[previous_g]
        end = self.metadata.gm_id_to_m_offset[previous_g + 1]
        begin = start // self.sub * self.sub
        for row in range(down_unroll):
            offset = (self.moved_rows if first else 0) + panel * down_unroll + row
            i = jnp.minimum(start + offset, end - 1)
            dest, prob = self.route(i)
            g = pl.program_id(1)
            weight = jnp.where((g > 0) & (start + offset < end), prob, 0)
            value = previous.at[pl.ds(i - begin, 1)][...].astype(jnp.float32)
            target = self.sums.at[pl.ds(dest, 1)]
            target[...] = target[...] + value * weight

    def inner(
        self,
        x: _Dynamic,
        w: _Dynamic,
        metadata: _Dynamic,
        *,
        first: _Dynamic = False,
        subtotal_vmem: _Dynamic = None,
    ) -> _Dynamic:
        """Execute inner logic."""
        g, n = (pl.program_id(1), 0 if first else pl.program_id(0) + 1)
        if not first:
            pl.when(g == 0)(
                lambda: self.sums.__setitem__(
                    slice(None), jnp.zeros(self.sums.shape, jnp.float32)
                )
            )
            self.advance(
                jnp.minimum(32, (g + 1) * 32 // self.ng),
                n - 1,
                subtotal_vmem,
                poll_round=g,
            )
        current = self.packed.at[g % 2]
        previous = self.packed.at[(g + 1) % 2]
        previous_g = jnp.maximum(g - 1, 0)
        panel_count = pl.cdiv(self.tn, pltpu.get_tpu_info().mxu_column_size)
        down_unroll = self.first_down_unroll if first else self.unroll

        def _do_overlap(panel: _Dynamic) -> None:
            """Execute overlap work logic."""
            self._overlap_work(panel, previous_g, previous, down_unroll, first=first)

        if not first:
            pl.when(g == 0)(
                lambda: previous.__setitem__(
                    slice(None), jnp.zeros(previous.shape, previous.dtype)
                )
            )

        pl.when(self.active_ng > 0)(
            lambda: _down_inner(
                x,
                w,
                current,
                metadata,
                cfgs=self.cfg,
                overlap_work=_do_overlap,
            )
            if not first
            else None
        )
        if first:
            k.dot(
                current,
                x,
                w,
                metadata,
                cfgs=self.cfg,
                overlap_work=_do_overlap,
                unroll=down_unroll,
            )
        else:
            pl.when(g > 0)(
                lambda: self.consume(
                    previous_g,
                    previous,
                    (self.moved_rows if first else 0) + panel_count * down_unroll,
                    loop_unroll=8 if first else self.unroll,
                )
            )

            def _finish() -> None:
                """Execute finish logic."""
                self.consume(g, current, loop_unroll=8 if first else self.unroll)
                self.advance(jnp.int32(32), n - 1, subtotal_vmem, drain=True)
                self.emit_subtotals(subtotal_vmem)
                self.state[0] = 0
                pl.when(n == self.width // self.tn - 1)(
                    lambda: self.advance(jnp.int32(32), n, subtotal_vmem, drain=True)
                )

            pl.when(g == self.ng - 1)(_finish)

    def _bounds(self, g: _Dynamic) -> tuple[_Dynamic, _Dynamic]:
        """Execute bounds logic."""
        begin = self.metadata.gm_id_to_m_offset[g] // self.sub * self.sub
        end = pl.cdiv(self.metadata.gm_id_to_m_offset[g + 1], self.sub) * self.sub
        return (begin, end - begin)

    def _start(
        self, g: _Dynamic, bank: _Dynamic, acts: _Dynamic, act_sems: _Dynamic
    ) -> None:
        """Execute start logic."""
        begin, nrows = self._bounds(g)

        def _row(i: _Dynamic) -> None:
            """Execute start row logic."""
            source = self.k_destinations[begin + i]
            pltpu.make_async_copy(
                self.k_latent.at[pl.ds(source, 1)],
                acts.at[bank, pl.ds(i, 1)],
                act_sems.at[bank],
            ).start()

        pl.loop(0, nrows)(_row)

    def joint(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Execute joint logic."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        gw = _p_next(_p_iter, _p_get("gw"))
        dw = _p_next(_p_iter, _p_get("dw"))
        gy = _p_next(_p_iter, _p_get("gy"))
        metadata = _p_next(_p_iter, _p_get("metadata"))
        gu_partial = _p_next(_p_iter, _p_get("gu_partial"))
        acts = _p_next(_p_iter, _p_get("acts"))
        act_sems = _p_next(_p_iter, _p_get("act_sems"))

        g = pl.program_id(1)
        bank = g % 2

        def _init_acc() -> None:
            """Execute accumulation initialization logic."""
            self.probabilities_ready()
            self.sums[...] = jnp.zeros(self.sums.shape, jnp.float32)
            self.packed.at[1][...] = jnp.zeros(
                self.packed.at[1].shape, self.packed.dtype
            )

        pl.when(g == 0)(_init_acc)

        def _gate_up() -> None:
            """Execute gate up logic."""
            pl.when(g == 0)(lambda: self._start(g, bank, acts, act_sems))
            _, nrows = self._bounds(g)
            active = acts.at[bank, pl.ds(0, nrows)]
            pltpu.make_async_copy(active, active, act_sems.at[bank]).wait()
            pl.when(g + 1 < self.ng)(
                lambda: self._start(g + 1, 1 - bank, acts, act_sems)
            )
            _gu_inner(
                PackedRows(acts.at[bank]),
                gw,
                gy,
                gu_partial,
                metadata,
                cfgs=self.gu_cfg,
                overlap_work=lambda panel: self.gu_work(g, panel),
            )

        pl.when(self.active_ng > 0)(_gate_up)
        self.inner(gy, dw, metadata, first=True)

    def first_column(
        self, gu_partial: _Dynamic, acts: _Dynamic, act_sems: _Dynamic
    ) -> _Dynamic:
        """Execute first column logic."""
        joint_gu_ws, joint_down_ws = jax.tree.map(
            _buffering_spec,
            (self.gu_ws, self.ws),
            is_leaf=lambda v: isinstance(v, pl.BlockSpec),
        )

        def _joint_step(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
            """Execute joint step logic."""
            _p_iter = iter(_p_args)
            _p_get: _Dynamic = _p_kw.get
            _p_next: _Dynamic = next
            gw = _p_next(_p_iter, _p_get("gw"))
            dw = _p_next(_p_iter, _p_get("dw"))
            gy = _p_next(_p_iter, _p_get("gy"))
            meta = _p_next(_p_iter, _p_get("metadata"))
            gup = _p_next(_p_iter, _p_get("gu_partial"))
            ac = _p_next(_p_iter, _p_get("acts"))
            asems = _p_next(_p_iter, _p_get("act_sems"))
            self.joint(gw, dw, gy, meta, gup, ac, asems)

        specs = {
            "grid": (1, self.ng, 1),
            "in_specs": (joint_gu_ws, joint_down_ws),
            "out_specs": self.gu_ys,
        }
        refs = (self.gu, self.w, self.k_middle.reshape(-1, self.sub, 1536))
        scratches = (self.metadata, gu_partial, acts, act_sems)
        if self.tokens == _CONST_8192:
            pltpu.emit_pipeline(_joint_step, **specs)(*refs, scratches=scratches)
            return

        pipeline, allocate = pltpu.emit_pipeline_with_allocations(_joint_step, **specs)
        allocations = allocate(*refs)
        logical_gu = allocations[0]
        padded_gu = jax.tree.map(_pad_buffer, logical_gu, is_leaf=_is_buffer)

        def run_pipeline(buffers: _Dynamic) -> None:
            """Execute pipeline run logic."""
            trimmed_gu = jax.tree.map(
                _trim_buffer, buffers[0], logical_gu, is_leaf=_is_buffer
            )
            pipeline(trimmed_gu, *buffers[1:])

        pl.run_scoped(run_pipeline, (padded_gu, *allocations[1:]))

    def remaining_columns(self, subtotal_vmem: _Dynamic) -> _Dynamic:
        """Execute remaining columns logic."""
        self.emit_subtotals(subtotal_vmem)
        self.state[0] = 0

        shifted_ws = jax.tree.map(
            _shifted_spec, self.ws, is_leaf=lambda v: isinstance(v, pl.BlockSpec)
        )

        def remaining(x: _Dynamic, w: _Dynamic, metadata: _Dynamic) -> _Dynamic:
            """Execute remaining pipeline logic."""
            self.inner(x, w, metadata, subtotal_vmem=subtotal_vmem)

        pltpu.emit_pipeline(
            remaining,
            grid=(self.width // self.tn - 1, self.ng, 1),
            in_specs=(self.xs, shifted_ws),
            out_specs=(),
        )(self.k_middle.reshape(-1, self.sub, 1536), self.w, scratches=(self.metadata,))


def _routed_experts(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Joint GU/SiTU/first-down pipeline followed by remaining columns."""
    return _RoutedExpertsRunner(*_p_args, **_p_kw).run()


def _slice_ref(ref: _Dynamic, first: _Dynamic, size: _Dynamic) -> _Dynamic:
    """Execute slice ref logic."""
    if not (ref.scale is not None):
        _err_msg = "Assertion failed: ref.scale is not None"
        raise AssertionError(_err_msg)
    return dataclasses.replace(
        ref,
        weight=ref.weight.at[:, pl.ds(first, size)],
        scale=ref.scale.at[..., pl.ds(first, size)],
    )


def _column_rhs(rhs: _Dynamic, first: _Dynamic, size: _Dynamic) -> _Dynamic:
    """Slice references before dequantization to bound panel temporaries."""
    if isinstance(rhs, k.FusedWeightsRef):
        pair = 2 * pltpu.get_tpu_info().num_lanes
        if not (first % pair == 0 and size % pair == 0):
            _err_msg = "Assertion failed: first % pair == 0 and size % pair == 0"
            raise AssertionError(_err_msg)
        return k.FusedWeightsRef(
            _slice_ref(rhs.gate, first // 2, size // 2),
            _slice_ref(rhs.up, first // 2, size // 2),
        )
    return _slice_ref(rhs, first, size)


def _expert_product(
    lhs: _Dynamic,
    rhs: _Dynamic,
    *,
    cfgs: _Dynamic,
    bucket_rows: _Dynamic,
    overlap_work: _Dynamic,
) -> _Dynamic:
    """Dequantize MXFP4 weights, panel-locally for the 8K row tile."""
    if cfgs.dims.size_k != cfgs.tiles.tile_k:
        _err_msg = "Assertion failed: cfgs.dims.size_k == cfgs.tiles.tile_k"
        raise AssertionError(_err_msg)
    columns = pltpu.get_tpu_info().mxu_column_size
    activations = lhs.reshape(-1, cfgs.tiles.tile_k)[:bucket_rows]
    width = (
        2 * rhs.gate.weight.shape[-1]
        if isinstance(rhs, k.FusedWeightsRef)
        else rhs.weight.shape[-1]
    )
    block = cfgs.rhs_cfgs.quant_block_size
    panel_local = cfgs.tiles.tile_m == _CONST_256
    weights = None
    if not panel_local:
        weights = rhs.get_weight()
        scales = rhs.get_scale(replicate_size=block).astype(cfgs.lhs_cfgs.dtype)
        weights = weights.astype(cfgs.lhs_cfgs.dtype).reshape(
            cfgs.num_quant_blocks_per_tile_k, block, width
        )
        weights = (weights * scales).reshape(cfgs.tiles.tile_k, width)
    panels = []
    for first in range(0, width, columns):
        panel_width = min(columns, width - first)
        if panel_local:
            panel_rhs = _column_rhs(rhs, first, panel_width)
            panel_weights = (
                panel_rhs.get_weight()
                .astype(cfgs.lhs_cfgs.dtype)
                .reshape(cfgs.num_quant_blocks_per_tile_k, block, panel_width)
            )
            scales = panel_rhs.get_scale(replicate_size=block).astype(
                cfgs.lhs_cfgs.dtype
            )
            panel_weights = (panel_weights * scales).reshape(
                cfgs.tiles.tile_k, panel_width
            )
        else:
            if not (weights is not None):
                _err_msg = "Assertion failed: weights is not None"
                raise AssertionError(_err_msg)
            panel_weights = weights[:, first : first + panel_width]
        accumulator = jnp.zeros((bucket_rows, panel_width), cfgs.acc_dtype)
        product = jnp.matmul(
            activations, panel_weights, preferred_element_type=jnp.float32
        ).astype(cfgs.acc_dtype)
        overlap_work(first // columns)
        accumulator += product
        panels.append(accumulator)
    return jnp.concatenate(panels, axis=1)


def _gu_inner(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """GU/SiTU with the production row buckets and adjacent-expert carry.

    Expert boundaries need not align with the 16-row BF16 sublane. Mask outside
    the active expert, then preserve the preceding expert's partial first row.
    """
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        lhs,
        rhs,
        output,
        partial,
        metadata,
    ) = (
        _p_next(_p_iter, _p_get("lhs")),
        _p_next(_p_iter, _p_get("rhs")),
        _p_next(_p_iter, _p_get("output")),
        _p_next(_p_iter, _p_get("partial")),
        _p_next(_p_iter, _p_get("metadata")),
    )
    (
        cfgs,
        overlap_work,
    ) = (
        _p_get("cfgs"),
        _p_get("overlap_work"),
    )
    group = pl.program_id(1)
    start = metadata.gm_id_to_m_offset[group]
    end = metadata.gm_id_to_m_offset[group + 1]
    aligned = start - start % cfgs.dims.size_lhs_sublane
    first, last = start - aligned, end - aligned

    def product(bucket_index: _Dynamic) -> _Dynamic:
        """Execute product logic."""
        rows = cfgs.tiles.bucket_base * (bucket_index + 1)
        value = _expert_product(
            lhs, rhs, cfgs=cfgs, bucket_rows=rows, overlap_work=overlap_work
        )
        value = k.apply_act_fn(value, cfgs.fuse_act)
        row = lax.broadcasted_iota(jnp.int32, value.shape, 0)
        value = jnp.where((first <= row) & (row < last), value, 0)
        output.reshape(-1, cfgs.tiles.tile_n)[:rows] = value.astype(output.dtype)
        zeros = jnp.zeros_like(partial)
        output[0] += jnp.where(group == 0, zeros, partial[...])
        last_row = last // cfgs.dims.size_lhs_sublane
        partial[...] = jnp.where(
            last % cfgs.dims.size_lhs_sublane == 0, zeros, output[last_row]
        )

    # At 8K, share the 128-row product for both small buckets.
    # Keep route order, trading padded work for fewer instruction overlays.
    first_bucket = 1 if cfgs.tiles.tile_m == _CONST_256 else 0
    branches = [
        functools.partial(product, bucket_index=i)
        for i in range(first_bucket, cfgs.tiles.tile_m // cfgs.tiles.bucket_base)
    ]
    lax.switch(jnp.maximum(last // cfgs.tiles.bucket_base - first_bucket, 0), branches)


def _down_inner(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Down projection stored as packed rows for token-wise VALU combine.

    Each valid expert row is consumed individually, so this output needs no
    adjacent-expert carry. BF16 rounding precedes FP32 routing weighting.
    """
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        lhs,
        rhs,
        output,
        metadata,
    ) = (
        _p_next(_p_iter, _p_get("lhs")),
        _p_next(_p_iter, _p_get("rhs")),
        _p_next(_p_iter, _p_get("output")),
        _p_next(_p_iter, _p_get("metadata")),
    )
    (
        cfgs,
        overlap_work,
    ) = (
        _p_get("cfgs"),
        _p_get("overlap_work"),
    )
    if cfgs.fuse_act is not None:
        _err_msg = "Assertion failed: cfgs.fuse_act is None"
        raise AssertionError(_err_msg)
    group = pl.program_id(1)
    start = metadata.gm_id_to_m_offset[group]
    end = metadata.gm_id_to_m_offset[group + 1]
    aligned = start - start % cfgs.dims.size_lhs_sublane
    last = end - aligned

    # GU already masks inactive rows; consume() visits only live routes.
    def product(bucket_index: _Dynamic) -> _Dynamic:
        """Execute product logic."""
        rows = cfgs.tiles.bucket_base * (bucket_index + 1)
        with jax.named_scope(f"bm{rows}_fragment"):
            value = _expert_product(
                lhs, rhs, cfgs=cfgs, bucket_rows=rows, overlap_work=overlap_work
            )
            if cfgs.tiles.tile_n == _CONST_896:
                # Keep the half boundary on a complete 128-lane block.
                # Only the final 128 elements are padding; the consumer trims
                # them once per subtotal before communicating the 896 columns.
                value = value.astype(output.dtype)
                padded = jnp.concatenate(
                    (value, jnp.zeros((rows, 128), output.dtype)), axis=1
                )
                output[:rows] = padded.reshape(rows, 2, 512)
            else:
                output[:rows] = value.astype(output.dtype).reshape(
                    rows, 2, cfgs.tiles.tile_n // 2
                )

    branches = [
        functools.partial(product, bucket_index=i)
        for i in range(cfgs.tiles.tile_m // cfgs.tiles.bucket_base)
    ]
    lax.switch(last // cfgs.tiles.bucket_base, branches)


class _ReadyReduceRunner:
    """Same 12-stream protocol with cached descriptors and cheap failed polls."""

    def __init__(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Initialize ready reduce runner."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            _,
            self.x_ref,
            self.o_ref,
            self.send_sems,
            self.recv_sems,
            self.sem_credits,
            self.store_sem,
            self.payload,
            self.received,
        ) = (
            _p_next(_p_iter, _p_get("tables_ref")),
            _p_next(_p_iter, _p_get("x_ref")),
            _p_next(_p_iter, _p_get("o_ref")),
            _p_next(_p_iter, _p_get("send_sems")),
            _p_next(_p_iter, _p_get("recv_sems")),
            _p_next(_p_iter, _p_get("credits")),
            _p_next(_p_iter, _p_get("store_sem")),
            _p_next(_p_iter, _p_get("payload")),
            _p_next(_p_iter, _p_get("received")),
        )
        (
            self.streams,
            self.buffers,
            self.state,
            self.ring_meta,
            self.drain,
            self.poll_round,
            self.sparse_poll,
        ) = (
            _p_get("streams"),
            _p_get("buffers"),
            _p_get("state"),
            _p_get("ring_meta"),
            _p_kw.get("drain", False),
            _p_get("poll_round", 0),
            _p_kw.get("sparse_poll", True),
        )
        if self.buffers != _CONST_2:
            _err_msg = "Assertion failed: buffers == 2"
            raise AssertionError(_err_msg)
        self.count = len(self.streams)

    def _remote(
        self,
        stream: _Dynamic,
        rows: _Dynamic,
        cols: _Dynamic,
        target: _Dynamic,
        bank: _Dynamic,
    ) -> _Dynamic:
        """Execute remote logic."""
        return pltpu.make_async_remote_copy(
            self.payload.at[rows, cols],
            self.received.at[bank, rows, cols],
            self.send_sems.at[stream],
            self.recv_sems.at[bank, stream],
            device_id=target,
            device_id_type=pl.DeviceIdType.MESH,
        )

    def _initialize(self) -> None:
        """Execute initialize logic."""
        peers = [(self.ring_meta[d * 34],) for d in range(6)]
        sem = pltpu.get_barrier_semaphore()
        for peer in peers:
            pl.semaphore_signal(
                sem, 1, device_id=peer, device_id_type=pl.DeviceIdType.MESH
            )
        pl.semaphore_wait(sem, len(peers))

        @functools.partial(pl.run_scoped, ready=pltpu.SemaphoreType.REGULAR)
        def initialized(ready: _Dynamic) -> _Dynamic:
            """Execute initialized logic."""
            for peer in peers:
                pl.semaphore_signal(
                    ready, 1, device_id=peer, device_id_type=pl.DeviceIdType.MESH
                )
            pl.semaphore_wait(ready, len(peers))

        for stream, region in enumerate(self.streams):
            begin, size, col_begin, width = region
            base = (stream % 6) * 34
            rows, cols = pl.ds(begin, size), pl.ds(col_begin, width)
            source_rows = pl.ds(
                pl.multiple_of(self.ring_meta[base + 2], self.o_ref.shape[0]) + begin,
                size,
            )
            target = (self.ring_meta[base],)
            self.payload.at[rows, cols][...] = self.x_ref.at[source_rows, cols][
                ...
            ].astype(self.payload.dtype)
            self._remote(stream, rows, cols, target, jnp.int32(0)).start()
            self.state[1 + stream] = 1
            self.state[1 + self.count + stream] = 0
        self.state[0] = self.count + 1

    def _receive_reduce(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Execute reduce logic."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        stream = _p_next(_p_iter, _p_get("stream"))
        base = _p_next(_p_iter, _p_get("base"))
        rows = _p_next(_p_iter, _p_get("rows"))
        cols = _p_next(_p_iter, _p_get("cols"))
        begin = _p_next(_p_iter, _p_get("begin"))
        size = _p_next(_p_iter, _p_get("size"))
        step = _p_next(_p_iter, _p_get("step"))
        bank = _p_next(_p_iter, _p_get("bank"))
        target = (self.ring_meta[base],)
        self._remote(stream, rows, cols, target, bank).wait()
        source_rows = pl.ds(
            pl.multiple_of(self.ring_meta[base + 2 + step], self.o_ref.shape[0])
            + begin,
            size,
        )
        self.payload.at[rows, cols][...] = (
            self.x_ref.at[source_rows, cols][...].astype(self.payload.dtype)
            + self.received.at[bank, rows, cols][...]
        )
        pl.semaphore_signal(
            self.sem_credits.at[stream],
            1,
            device_id=(self.ring_meta[base + 1],),
            device_id_type=pl.DeviceIdType.MESH,
        )
        self.state[1 + self.count + stream] = jnp.where(step == _CONST_31, 2, 1)

    def _receive_step(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Execute receive step check."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        stream = _p_next(_p_iter, _p_get("stream"))
        base = _p_next(_p_iter, _p_get("base"))
        rows = _p_next(_p_iter, _p_get("rows"))
        cols = _p_next(_p_iter, _p_get("cols"))
        begin = _p_next(_p_iter, _p_get("begin"))
        size = _p_next(_p_iter, _p_get("size"))
        units = _p_next(_p_iter, _p_get("units"))
        step = _p_next(_p_iter, _p_get("step"))
        """Execute receive step check."""
        bank = (step - 1) & 1
        ready = (pl.semaphore_read(self.send_sems.at[stream]) >= units) & (
            pl.semaphore_read(self.recv_sems.at[bank, stream]) >= units
        )
        pl.when(ready)(
            lambda: self._receive_reduce(
                stream, base, rows, cols, begin, size, step, bank
            )
        )

    def _forward_send(
        self, stream: int, base: int, rows: _Dynamic, cols: _Dynamic, step: _Dynamic
    ) -> None:
        """Execute forward send logic."""
        pl.when(step >= self.buffers)(
            lambda: pl.semaphore_wait(self.sem_credits.at[stream], 1)
        )
        self._remote(stream, rows, cols, (self.ring_meta[base],), step & 1).start()
        self.state[1 + stream] = step + 1
        self.state[1 + self.count + stream] = 0

    def _forward_step(
        self, stream: int, base: int, rows: _Dynamic, cols: _Dynamic, step: _Dynamic
    ) -> None:
        """Execute forward step check."""
        available = (step < self.buffers) | (
            pl.semaphore_read(self.sem_credits.at[stream]) >= 1
        )
        pl.when(available)(lambda: self._forward_send(stream, base, rows, cols, step))

    def _finish_step(self, stream: int) -> None:
        """Execute finish step check."""
        available = pl.semaphore_read(self.sem_credits.at[stream]) >= self.buffers
        pl.when(available)(lambda: self._release_stream(stream))

    def _release_stream(self, stream: int) -> None:
        """Release semaphore credits and decrement active state."""
        pl.semaphore_wait(self.sem_credits.at[stream], self.buffers)
        self.state[1 + self.count + stream] = 3
        self.state[0] = self.state[0] - 1

    def _sweep_stream(self, stream: int, region: tuple[int, int, int, int]) -> None:
        """Sweep one reduce stream."""
        begin, size, col_begin, width = region
        base = (stream % 6) * 34
        rows, cols = pl.ds(begin, size), pl.ds(col_begin, width)
        units = size * width * 4 // 32
        step = self.state[1 + stream]
        pl.when(self.state[1 + self.count + stream] == 0)(
            lambda: self._receive_step(
                stream, base, rows, cols, begin, size, units, step
            )
        )
        pl.when(self.state[1 + self.count + stream] == 1)(
            lambda: self._forward_step(stream, base, rows, cols, step)
        )
        pl.when(self.state[1 + self.count + stream] == _CONST_2)(
            lambda: self._finish_step(stream)
        )

    def _sweep(self, selected: _Dynamic = None) -> None:
        """Execute sweep logic."""
        for stream, region in enumerate(self.streams):
            if selected is not None and (stream // 6) != selected:
                continue
            self._sweep_stream(stream, region)

    def run(self) -> None:
        """Execute ready reduce loop."""
        pl.when(self.state[0] == 0)(lambda: self._initialize())
        if self.drain:
            jax.lax.while_loop(
                lambda _i: self.state[0] != 1,
                lambda i: (self._sweep(), i + 1)[1],
                jnp.int32(0),
            )
            output = pltpu.make_async_copy(self.payload, self.o_ref, self.store_sem)
            output.start()
            output.wait()
        else:
            pl.when(self.state[0] != 1)(lambda: self._run_progress())

    def _run_progress(self) -> None:
        """Run progress for active streams."""
        for group in range(2):
            cond = ((self.poll_round & 1) == group) | ~self.sparse_poll
            pl.when(cond)(lambda g=group: self._sweep(g))


def _ready_reduce(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Same 12-stream protocol with cached descriptors and cheap failed polls.

    ring_meta has six directed rings, each with neighbor mesh IDs followed by
    32 source-row bases. SMEM offsets, not physical VMEM pointers, are cached.
    Neither a failed receive nor a failed credit poll needs a payload address.
    """
    return _ReadyReduceRunner(*_p_args, **_p_kw).run()


def _epilogue(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Normalize routed output and project up with three weight/prefix banks.

    Reduced FP32 rows remain in VMEM. Preserve the BF16 boundaries after
    normalization, after latent-up, and after adding the shared expert.
    """
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        control,
        reduced,
        shared,
        norm_weight,
        up_weight,
        prefix,
        output,
    ) = (
        _p_next(_p_iter, _p_get("control")),
        _p_next(_p_iter, _p_get("reduced")),
        _p_next(_p_iter, _p_get("shared")),
        _p_next(_p_iter, _p_get("norm_weight")),
        _p_next(_p_iter, _p_get("up_weight")),
        _p_next(_p_iter, _p_get("prefix")),
        _p_next(_p_iter, _p_get("output")),
    )
    rows, width = reduced.shape
    family = control[1]
    norm_weight = norm_weight.at[family, :, :width]
    up_weight = up_weight.at[family]
    valid = _stripe_validity(rows, control[0], jax.lax.axis_index("k3_sp"))
    vmem, dma = pltpu.VMEM, pltpu.SemaphoreType.DMA

    def body(s: _Dynamic) -> _Dynamic:
        """Execute body logic."""
        copy = pltpu.make_async_copy(norm_weight, s["norm"], s["norm_sem"])
        copy.start()
        copy.wait()
        panels = 7168 // 512

        def weight_copy(panel: _Dynamic) -> _Dynamic:
            """Execute weight copy logic."""
            return pltpu.make_async_copy(
                up_weight.at[:, pl.ds(panel * 512, 512)],
                s["weights"][panel % 3],
                s["weight_sems"][panel % 3],
            )

        def prefix_copy(panel: _Dynamic) -> _Dynamic:
            """Execute prefix copy logic."""
            return pltpu.make_async_copy(
                prefix.at[:, pl.ds(panel * 512, 512)],
                s["tiles"][panel % 3],
                s["prefix_sems"][panel % 3],
            )

        def output_copy(panel: _Dynamic) -> _Dynamic:
            """Execute output copy logic."""
            return pltpu.make_async_copy(
                s["tiles"][panel % 3],
                output.at[:, pl.ds(panel * 512, 512)],
                s["output_sems"][panel % 3],
            )

        # Hide the first weight/residual loads behind the RMS reduction.
        for panel in (0, 1):
            weight_copy(panel).start()
            prefix_copy(panel).start()
        routed = reduced[...]
        mean = jnp.sum(routed * routed, axis=1, keepdims=True) / width
        inverse = jax.lax.rsqrt(mean + 1e-5)
        s["normalized"][...] = ((routed * inverse) * s["norm"][...]).astype(
            jnp.bfloat16
        )
        for panel in range(panels):
            bank = panel % 3
            weight_copy(panel).wait()
            prefix_copy(panel).wait()
            if panel + 2 < panels:
                if panel >= 1:
                    output_copy(panel - 1).wait()
                weight_copy(panel + 2).start()
                prefix_copy(panel + 2).start()
            product = jnp.matmul(
                s["normalized"][...],
                s["weights"][bank][...],
                preferred_element_type=jnp.float32,
            ).astype(jnp.bfloat16)
            value = product + shared[:, pl.ds(panel * 512, 512)]
            value = jnp.where(valid, value, 0)
            s["tiles"][bank][...] = s["tiles"][bank][...] + value
            output_copy(panel).start()
        for panel in range(panels - 3, panels):
            output_copy(panel).wait()

    pl.run_scoped(
        body,
        {
            "norm": vmem((1, width), jnp.float32),
            "norm_sem": dma,
            "normalized": vmem((rows, width), jnp.bfloat16),
            "weights": tuple(vmem((width, 512), jnp.bfloat16) for _ in range(3)),
            "tiles": tuple(vmem((rows, 512), jnp.bfloat16) for _ in range(3)),
            "weight_sems": (dma, dma, dma),
            "prefix_sems": (dma, dma, dma),
            "output_sems": (dma, dma, dma),
        },
    )


class _FlatMetadata:
    """Scalar expert-row access to the 2D SMEM layout used by metadata DMA.

    Keeping DMA source and destination two-dimensional avoids an unsupported
    1D VMEM memref reshape. Only scalar reads are needed by expert pipelines.
    """

    def __init__(self, ref: _Dynamic) -> None:
        # Both views remain 2D; scalar consumers use one contiguous index.
        """Execute init logic."""
        self.ref = ref.reshape(1, ref.shape[0] * ref.shape[1])

    def __getitem__(self, index: _Dynamic) -> _Dynamic:
        """Execute getitem logic."""
        return self.ref[0, index]


class _LayerKernelRunner:
    """One launch owns all compute, metadata and communication on the fast path."""

    def __init__(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Initialize layer kernel runner."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            self.tables,
            self.control,
            self.hidden,
            self.router,
            self.bias,
            self.latent_down,
            self.shared_gu,
            self.shared_down,
            self.expert_gu,
            self.gu_scales,
            self.expert_down,
            self.down_scales,
            self.norm,
            self.latent_up,
            self.prefix,
            self.output,
            self.latent,
            self.middle,
            self.fallback_ids,
            self.fallback_probs,
            self.fallback_counts,
            self.fallback_shared,
            self.maximum,
        ) = (
            _p_next(_p_iter, _p_get("tables")),
            _p_next(_p_iter, _p_get("control")),
            _p_next(_p_iter, _p_get("hidden")),
            _p_next(_p_iter, _p_get("router")),
            _p_next(_p_iter, _p_get("bias")),
            _p_next(_p_iter, _p_get("latent_down")),
            _p_next(_p_iter, _p_get("shared_gu")),
            _p_next(_p_iter, _p_get("shared_down")),
            _p_next(_p_iter, _p_get("expert_gu")),
            _p_next(_p_iter, _p_get("gu_scales")),
            _p_next(_p_iter, _p_get("expert_down")),
            _p_next(_p_iter, _p_get("down_scales")),
            _p_next(_p_iter, _p_get("norm")),
            _p_next(_p_iter, _p_get("latent_up")),
            _p_next(_p_iter, _p_get("prefix")),
            _p_next(_p_iter, _p_get("output")),
            _p_next(_p_iter, _p_get("latent")),
            _p_next(_p_iter, _p_get("middle")),
            _p_next(_p_iter, _p_get("fallback_ids")),
            _p_next(_p_iter, _p_get("fallback_probs")),
            _p_next(_p_iter, _p_get("fallback_counts")),
            _p_next(_p_iter, _p_get("fallback_shared")),
            _p_next(_p_iter, _p_get("maximum")),
        )
        self.rows = self.hidden.shape[0]
        self.tokens = self.rows * 32
        self.capacity = self.tokens * 16 if self.tokens < _CONST_8192 else 32768
        self.shape = (self.tokens // 8, 128)

    def _count_copy(self, s: _Dynamic) -> _Dynamic:
        """Execute count copy logic."""
        return pltpu.make_async_copy(s["counts_v"], s["counts"], s["count_sem"])

    def _maximum_copy(self, s: _Dynamic) -> _Dynamic:
        """Execute maximum copy logic."""
        return pltpu.make_async_copy(self.maximum, s["maximum"], s["max_sem"])

    def _counts_ready(self, s: _Dynamic) -> None:
        """Execute counts ready logic."""
        self._count_copy(s).start()
        if self.tokens == _CONST_8192:
            self._maximum_copy(s).start()

    def _probability_copy(self, s: _Dynamic, probs_v: _Dynamic) -> _Dynamic:
        """Execute probability copy logic."""
        return pltpu.make_async_copy(
            probs_v.at[: self.capacity // 128, :],
            s["probs"],
            s["prob_sem"],
        )

    def _export_overflow(self, s: _Dynamic, ids_v: _Dynamic, probs_v: _Dynamic) -> None:
        """Execute export overflow logic."""
        for source, dest in (
            (ids_v, self.fallback_ids),
            (probs_v, self.fallback_probs),
            (s["counts_v"], self.fallback_counts),
            (s["shared"], self.fallback_shared),
        ):
            copy = pltpu.make_async_copy(source, dest, s["export_sem"])
            copy.start()
            copy.wait()

    def _metadata_ready(
        self, s: _Dynamic, indices: _Dynamic, probs_v: _Dynamic
    ) -> None:
        """Execute metadata ready logic."""
        indices.start()
        self._probability_copy(s, probs_v).start()

    def _prepare(self, s: _Dynamic, ids_v: _Dynamic, probs_v: _Dynamic) -> None:
        """Execute prepare logic."""
        indices = pltpu.make_async_copy(
            ids_v.at[: self.capacity // 128, :], s["ids"], s["index_sem"]
        )
        _prologue(
            self.tables,
            self.control,
            self.hidden,
            self.router,
            self.bias,
            self.latent_down,
            self.shared_gu,
            self.shared_down,
            s["shared"],
            self.latent,
            ids_v,
            probs_v,
            s["counts_v"],
            self.maximum,
            counts_ready=lambda: self._counts_ready(s),
            metadata_ready=lambda: self._metadata_ready(s, indices, probs_v),
        )
        self._count_copy(s).wait()
        if self.tokens == _CONST_8192:
            self._maximum_copy(s).wait()
        if self.tokens != _CONST_8192:
            self._metadata_ready(s, indices, probs_v)
        if self.tokens == _CONST_8192:
            pl.when(s["maximum"][0] > self.capacity)(
                lambda: self._export_overflow(s, ids_v, probs_v)
            )
        indices.wait()
        if self.tokens == _CONST_8192:
            self._probability_copy(s, probs_v).wait()

    def _probabilities_ready(self, s: _Dynamic) -> None:
        """Execute probabilities ready logic."""
        if self.tokens < _CONST_8192:
            self._probability_copy(s, s["probs_v"]).wait()

    def _fast(self, s: _Dynamic) -> None:
        """Execute fast logic."""
        _routed_experts(
            self.latent,
            self.expert_gu,
            self.gu_scales,
            self.expert_down,
            self.down_scales,
            s["counts"],
            _FlatMetadata(s["ids"]),
            _FlatMetadata(s["probs"]),
            self.tables,
            s["family"],
            s["offset"],
            self.middle,
            s["reduced"],
            tokens=self.tokens,
            probabilities_ready=lambda: self._probabilities_ready(s),
        )
        _epilogue(
            self.control,
            s["reduced"],
            s["shared"],
            self.norm,
            self.latent_up,
            self.prefix,
            self.output,
        )

    def run_body(self, s: _Dynamic) -> None:
        """Execute layer kernel body logic."""
        s["offset"][0] = 0
        family = k.expert_family_slot(self.expert_gu.shape[0], self.control[1])
        s["family"][0] = family
        vmem = pltpu.VMEM

        if self.tokens == _CONST_8192:
            pl.run_scoped(
                lambda ids, pr: self._prepare(s, ids, pr),
                vmem(self.shape, jnp.int32),
                vmem(self.shape, jnp.float32),
            )
        else:
            pl.run_scoped(
                lambda ids: self._prepare(s, ids, s["probs_v"]),
                vmem(self.shape, jnp.int32),
            )

        if self.tokens < _CONST_8192:
            self._fast(s)
        else:
            pl.when(s["maximum"][0] <= self.capacity)(lambda: self._fast(s))

    def run(self) -> None:
        """Run scoped layer kernel."""
        vmem, smem, dma = pltpu.VMEM, pltpu.SMEM, pltpu.SemaphoreType.DMA
        pl.run_scoped(
            self.run_body,
            {
                "shared": vmem((self.rows, 7168), jnp.bfloat16),
                "reduced": vmem((self.rows, 3584), jnp.float32),
                "probs_v": vmem(self.shape, jnp.float32)
                if self.tokens < _CONST_8192
                else None,
                "counts_v": vmem((56,), jnp.int32),
                "counts": smem((56,), jnp.int32),
                "maximum": smem((1,), jnp.int32)
                if self.tokens == _CONST_8192
                else None,
                "ids": smem((self.capacity // 128, 128), jnp.int32),
                "probs": smem((self.capacity // 128, 128), jnp.float32),
                "family": smem((1,), jnp.int32),
                "offset": smem((1,), jnp.int32),
                "count_sem": dma,
                "max_sem": dma if self.tokens == _CONST_8192 else None,
                "index_sem": dma,
                "prob_sem": dma,
                "export_sem": dma,
            },
        )


def _layer_kernel(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """One launch owns all compute, metadata and communication on the fast path.

    Large projection scratch has a shorter lifetime than expert scratch. Only
    shared output and the scalar routing metadata survive that boundary. At 8K
    all ranks take the same overflow branch; no routes are dropped to fit SMEM.
    """
    return _LayerKernelRunner(*_p_args, **_p_kw).run()


def make_indexed_moe_sp_ring(mesh: _Dynamic, *, seq_len: _Dynamic = 8192) -> _Dynamic:
    """Build the fused EP16/TP2 prefill MoE for a supported serving window.

    Inputs follow the prefill layer interface: rank-sharded normalized hidden
    states, residual prefix, resident weight families, valid length and family
    index. All fast-path operations execute in one Pallas kernel. Rare 8K route
    overflow uses the uncapped production expert path with the same metadata.
    """
    if seq_len not in (512, 1024, 2048, 4096, 8192):
        _err_msg = (
            "Fused prefill MoE supports 512, 1024, 2048, 4096 and 8192 token windows"
        )
        raise ValueError(_err_msg)
    if operator.attrgetter("_config")(k)().moe_reduce_dtype != "float32":
        _err_msg = (
            "Fused prefill MoE requires FP32 reduction; use the dispatcher for BF16"
        )
        raise ValueError(_err_msg)
    if mesh.size != _CONST_32 or tuple(mesh.axis_names) != ("k3_sp",):
        _err_msg = "Fused prefill MoE requires a flat 32-rank k3_sp mesh"
        raise ValueError(_err_msg)

    rows = seq_len // 32
    tables = k.ring_tables(list(mesh.devices.flat), tuple(range(32)))
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    shape = (seq_len // 8, 128)
    bf16 = jnp.bfloat16
    workspace_shapes = [
        ((rows, 7168), bf16),
        ((seq_len, 2, 1792), bf16),
        ((seq_len * 16, 1536), bf16),
    ]
    output_specs = [hbm] * 3
    if seq_len == _CONST_8192:
        # Only 8K can overflow compact SMEM routing capacity. At <=4K every
        # route fits, so omit the unused overflow outputs and maximum DMA.
        workspace_shapes += [
            (shape, jnp.int32),
            (shape, jnp.float32),
            ((56,), jnp.int32),
            ((rows, 7168), bf16),
            ((1,), jnp.int32),
        ]
        output_specs += [hbm] * 4 + [pl.BlockSpec()]
    kernel = pl.pallas_call(
        _layer_kernel,
        out_shape=tuple(jax.ShapeDtypeStruct(s, d) for s, d in workspace_shapes),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=(hbm,) * 13,
            out_specs=tuple(output_specs),
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=0,
            vmem_limit_bytes=64 * 1024**2,
            disable_bounds_checks=True,
            shape_invariant_numerics=False,
        ),
        name=f"fused_moe_pallas_{seq_len}",
    )

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(P("k3_sp"), P("k3_sp"), P("k3_sp"), P(), P()),
        out_specs=P("k3_sp"),
        check_vma=False,
    )
    def local(
        hidden: _Dynamic,
        prefix: _Dynamic,
        families: _Dynamic,
        length: _Dynamic,
        slot: _Dynamic,
    ) -> _Dynamic:
        """Execute local logic."""
        w = {name: value[0] for name, value in families.items()}
        control = jnp.concatenate((length.reshape(1), slot.reshape(1)))
        outputs = kernel(
            jnp.asarray(tables),
            control,
            hidden[0],
            w["router"],
            w["router_bias"],
            w["latent_down"],
            w["shared_gu"],
            w["shared_down"],
            w["expert_gu"],
            w["expert_gus"],
            w["expert_down"],
            w["expert_ds"],
            w["latent_norm"],
            w["latent_up"],
            prefix[0],
        )
        if seq_len < _CONST_8192:
            return outputs[0][None]
        final, latent, _, ids, probs, counts, shared, maximum = outputs

        def overflow() -> _Dynamic:
            """Execute overflow logic."""
            reduced = routed_experts_prepared(
                latent,
                ids.reshape(-1),
                probs.reshape(-1),
                counts,
                w["expert_gu"],
                w["expert_gus"],
                w["expert_down"],
                w["expert_ds"],
                reduce_tables=tables,
                expert_layer=slot,
                global_route_max=maximum[0],
            )
            selected = {n: w[n][slot[0]] for n in ("latent_norm", "latent_up")}
            return operator.attrgetter("_moe_sp_tail")(k)(
                reduced,
                shared,
                prefix[0],
                length,
                selected,
                axis_name="k3_sp",
                eps=1e-5,
                shared_reduced=True,
            )

        return jax.lax.cond(maximum[0] <= _CONST_32768, lambda: final, overflow)[None]

    return jax.jit(local)
