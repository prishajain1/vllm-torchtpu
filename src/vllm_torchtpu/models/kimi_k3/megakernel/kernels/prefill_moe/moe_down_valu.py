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

"""MXFP4 down, weighted token accumulation, and pipelined reduce-scatter.

Source provenance: Adapted from inferact/k3-megakernel@30c9ba966b and
vllm_torchtpu/kernels/megablox/gmm_v2.py@644426d851.

While the MXU produces tile n, VALU accumulates tile n-1 in packed token
layout. Completed column-block subtotals feed the ring directly from VMEM;
the previous block's ring advances between tiles of the next down block.
Products and local subtotals round to BF16; token weighting uses FP32 and
the ring uses ``ring_dtype``.
"""

from __future__ import annotations

#
# Portions of this file are modified copies of code Copyright 2026 Google LLC,
# licensed under the Apache License, Version 2.0, from
# vllm_torchtpu/kernels/megablox/gmm_v2.py.
import dataclasses
import functools
import operator
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.models.kimi_k3.megakernel.kernels import prefill_megakernel as k

_Dynamic = Any
_CONST_31 = 31
_CONST_4 = 4


def _ring_coords(
    tables_ref: _Dynamic, pos: _Dynamic, family: _Dynamic
) -> tuple[_Dynamic]:
    """Execute ring coords logic."""
    peer = tables_ref[family, 1, pos]
    return (peer,)


def _build_ring_descriptors(
    tables_ref: _Dynamic,
    positions: list[_Dynamic],
    streams: _Dynamic,
    o_ref: _Dynamic,
    step: _Dynamic,
) -> list[tuple[_Dynamic, ...]]:
    """Construct descriptors for ring communication streams."""
    descriptors = []
    for stream, region in enumerate(streams):
        begin, size = region[:2]
        col_begin, width = (
            region[2:] if len(region) == _CONST_4 else (0, o_ref.shape[1])
        )
        cols = pl.ds(col_begin, width)
        family = stream // 2 % 3
        direction = -1 if stream % 2 == 0 else 1
        position = positions[family]
        target = _ring_coords(
            tables_ref, (position + direction) % 32, jnp.int32(family)
        )
        source = _ring_coords(
            tables_ref, (position - direction) % 32, jnp.int32(family)
        )
        owner_mesh = _ring_coords(
            tables_ref, (position - direction * (step + 1)) % 32, jnp.int32(family)
        )[0]
        owner = tables_ref[family, 2, owner_mesh]
        rows = pl.ds(begin, size)
        source_rows = pl.ds(owner * o_ref.shape[0] + begin, size)
        descriptors.append((target, source, rows, cols, source_rows))
    return descriptors


def _ring_stream_exchange(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
    """Execute stream send/recv/add step in ring."""
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        target,
        source,
        rows,
        cols,
        source_rows,
        stream,
        x_ref,
        o_ref,
        payload,
        received,
        buffers,
        send_sems,
        recv_sems,
        step,
        credit_tokens,
    ) = (
        _p_next(_p_iter, _p_get("target")),
        _p_next(_p_iter, _p_get("source")),
        _p_next(_p_iter, _p_get("rows")),
        _p_next(_p_iter, _p_get("cols")),
        _p_next(_p_iter, _p_get("source_rows")),
        _p_next(_p_iter, _p_get("stream")),
        _p_next(_p_iter, _p_get("x_ref")),
        _p_next(_p_iter, _p_get("o_ref")),
        _p_next(_p_iter, _p_get("payload")),
        _p_next(_p_iter, _p_get("received")),
        _p_next(_p_iter, _p_get("buffers")),
        _p_next(_p_iter, _p_get("send_sems")),
        _p_next(_p_iter, _p_get("recv_sems")),
        _p_next(_p_iter, _p_get("step")),
        _p_next(_p_iter, _p_get("credit_tokens")),
    )
    val = (
        x_ref.reshape(x_ref.shape[0], o_ref.shape[1])
        .at[source_rows, cols][...]
        .astype(payload.dtype)
    )
    pl.when(step > 0)(
        lambda: pltpu.make_async_remote_copy(
            payload.at[rows, cols],
            received.at[(step - 1) % buffers, rows, cols],
            send_sems.at[stream],
            recv_sems.at[(step - 1) % buffers, stream],
            device_id=target,
            device_id_type=pl.DeviceIdType.MESH,
        ).wait()
    )

    @pl.when(step == 0)
    def _seed_payload() -> None:
        """Seed initial ring payload."""
        payload.at[rows, cols][...] = val

    @pl.when(step > 0)
    def _recv_and_signal() -> None:
        """Accumulate received ring payload and signal credit."""
        payload.at[rows, cols][...] = (
            val + received.at[(step - 1) % buffers, rows, cols][...]
        )
        pl.semaphore_signal(
            credit_tokens.at[stream],
            1,
            device_id=source,
            device_id_type=pl.DeviceIdType.MESH,
        )

    pl.when(step >= buffers)(lambda: pl.semaphore_wait(credit_tokens.at[stream], 1))
    pl.when(step < _CONST_31)(
        lambda: pltpu.make_async_remote_copy(
            payload.at[rows, cols],
            received.at[step % buffers, rows, cols],
            send_sems.at[stream],
            recv_sems.at[step % buffers, stream],
            device_id=target,
            device_id_type=pl.DeviceIdType.MESH,
        ).start()
    )
    pl.when(step == _CONST_31)(
        lambda: pl.semaphore_wait(credit_tokens.at[stream], buffers)
    )


def ring_step(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Execute ring step logic."""
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        tables_ref,
        x_ref,
        o_ref,
        send_sems,
        recv_sems,
        credit_tokens,
        store_sem,
        payload,
        received,
    ) = (
        _p_next(_p_iter, _p_get("tables_ref")),
        _p_next(_p_iter, _p_get("x_ref")),
        _p_next(_p_iter, _p_get("o_ref")),
        _p_next(_p_iter, _p_get("send_sems")),
        _p_next(_p_iter, _p_get("recv_sems")),
        _p_next(_p_iter, _p_get("credit_tokens")),
        _p_next(_p_iter, _p_get("store_sem")),
        _p_next(_p_iter, _p_get("payload")),
        _p_next(_p_iter, _p_get("received")),
    )
    (
        streams,
        buffers,
        step,
    ) = (
        _p_get("streams"),
        _p_get("buffers"),
        _p_get("step"),
    )
    rank = lax.axis_index("k3_sp")
    positions = [tables_ref[f, 0, rank] for f in range(3)]
    pl.when(step == 0)(lambda: operator.attrgetter("_ring_enter")(k)(tables_ref, rank))
    descriptors = _build_ring_descriptors(tables_ref, positions, streams, o_ref, step)
    for stream, (target, source, rows, cols, source_rows) in enumerate(descriptors):
        _ring_stream_exchange(
            target,
            source,
            rows,
            cols,
            source_rows,
            stream,
            x_ref,
            o_ref,
            payload,
            received,
            buffers,
            send_sems,
            recv_sems,
            step,
            credit_tokens,
        )

    @pl.when(step == _CONST_31)
    def _final_store() -> None:
        """Start and wait for the final ring output DMA store."""
        store_copy = pltpu.make_async_copy(payload, o_ref, store_sem)
        store_copy.start()
        store_copy.wait()


class _DownValuKernel:
    """Prefill MoE down VALU Pallas kernel."""

    def __init__(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Initialize down VALU kernel parameters."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            self.cfg,
            self.family,
            self.streams,
            self.regions,
            self.tn,
            self.tm,
            self.store_tile,
            self.unroll,
            self.sub,
            self.route_chunk,
            self.route_span,
            self.stream_metadata,
            self.metadata_cache_rows,
            self.width,
            self.localrows,
            self.tokens,
        ) = (
            _p_next(_p_iter, _p_get("cfg")),
            _p_next(_p_iter, _p_get("family")),
            _p_next(_p_iter, _p_get("streams")),
            _p_next(_p_iter, _p_get("regions")),
            _p_next(_p_iter, _p_get("tn")),
            _p_next(_p_iter, _p_get("tm")),
            _p_next(_p_iter, _p_get("store_tile")),
            _p_next(_p_iter, _p_get("unroll")),
            _p_next(_p_iter, _p_get("sub")),
            _p_next(_p_iter, _p_get("route_chunk")),
            _p_next(_p_iter, _p_get("route_span")),
            _p_next(_p_iter, _p_get("stream_metadata")),
            _p_next(_p_iter, _p_get("metadata_cache_rows")),
            _p_next(_p_iter, _p_get("width")),
            _p_next(_p_iter, _p_get("localrows")),
            _p_next(_p_iter, _p_get("tokens")),
        )

    def empty_metadata(self, metadata: _Dynamic) -> None:
        """Initialize empty metadata."""
        metadata.gm_id_to_group_id[0] = 0
        metadata.gm_id_to_m_offset[0] = 0
        metadata.gm_id_to_m_offset[1] = 0

    def advance(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Advance ring hop step."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            stop,
            block,
            state,
            tables,
            subtotal_vmem,
            out,
            sends,
            receives,
            credit_tokens,
            stores,
            payload,
            rx,
        ) = (
            _p_next(_p_iter, _p_get("stop")),
            _p_next(_p_iter, _p_get("block")),
            _p_next(_p_iter, _p_get("state")),
            _p_next(_p_iter, _p_get("tables")),
            _p_next(_p_iter, _p_get("subtotal_vmem")),
            _p_next(_p_iter, _p_get("out")),
            _p_next(_p_iter, _p_get("sends")),
            _p_next(_p_iter, _p_get("receives")),
            _p_next(_p_iter, _p_get("credit_tokens")),
            _p_next(_p_iter, _p_get("stores")),
            _p_next(_p_iter, _p_get("payload")),
            _p_next(_p_iter, _p_get("rx")),
        )

        @pl.loop(state[0], stop)
        def _advance_step(step_idx: _Dynamic) -> None:
            """Advance one ring step."""
            ring_step(
                tables,
                subtotal_vmem,
                out.at[:, pl.ds(block * self.tn, self.tn)],
                sends,
                receives,
                credit_tokens,
                stores,
                payload,
                rx,
                streams=self.regions,
                buffers=2,
                step=step_idx,
            )

        state[0] = stop

    def emit_subtotals(
        self, _n: _Dynamic, subtotal_vmem: _Dynamic, sums: _Dynamic, cast: _Dynamic
    ) -> None:
        """Emit subtotals to VMEM."""

        @pl.loop(0, self.tokens // self.store_tile)
        def _emit_block(b: _Dynamic) -> None:
            """Cast and store one subtotal tile."""
            cast[...] = (
                sums[pl.ds(b * self.store_tile, self.store_tile)]
                .reshape(self.store_tile, self.tn)
                .astype(jnp.bfloat16)
            )
            subtotal_vmem.at[pl.ds(b * self.store_tile, self.store_tile)][...] = cast[
                ...
            ]

    def route(
        self,
        *_p_args: _Dynamic,
        **_p_kw: _Dynamic,
    ) -> tuple[_Dynamic, _Dynamic]:
        """Compute route destination and probability."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (g, i, metadata, dest_tiles, prob_tiles, destinations, probabilities) = (
            _p_next(_p_iter, _p_get("g")),
            _p_next(_p_iter, _p_get("i")),
            _p_next(_p_iter, _p_get("metadata")),
            _p_next(_p_iter, _p_get("dest_tiles")),
            _p_next(_p_iter, _p_get("prob_tiles")),
            _p_next(_p_iter, _p_get("destinations")),
            _p_next(_p_iter, _p_get("probabilities")),
        )
        if not self.stream_metadata:
            return destinations[i], probabilities[i]
        chunk = metadata.gm_id_to_m_offset[g] // self.route_chunk
        begin = chunk * self.route_chunk
        return dest_tiles[chunk % 2, i - begin], prob_tiles[chunk % 2, i - begin]

    def consume(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Consume routes into accumulator."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            g,
            tile,
            metadata,
            sums,
            dest_tiles,
            prob_tiles,
            destinations,
            probabilities,
            offset,
        ) = (
            _p_next(_p_iter, _p_get("g")),
            _p_next(_p_iter, _p_get("tile")),
            _p_next(_p_iter, _p_get("metadata")),
            _p_next(_p_iter, _p_get("sums")),
            _p_next(_p_iter, _p_get("dest_tiles")),
            _p_next(_p_iter, _p_get("prob_tiles")),
            _p_next(_p_iter, _p_get("destinations")),
            _p_next(_p_iter, _p_get("probabilities")),
            _p_next(_p_iter, _p_get("offset", 0)),
        )
        start = metadata.gm_id_to_m_offset[g]
        end = metadata.gm_id_to_m_offset[g + 1]
        begin = start // self.sub * self.sub
        start = jnp.minimum(start + offset, end)

        def _update(i_idx: _Dynamic) -> None:
            """Accumulate one routed token row."""
            dest, prob = self.route(
                g,
                i_idx,
                metadata,
                dest_tiles,
                prob_tiles,
                destinations,
                probabilities,
            )
            value = tile.at[pl.ds(i_idx - begin, 1)][...].astype(jnp.float32)
            target = sums.at[pl.ds(dest, 1)]
            target[...] = target[...] + value * prob

        full = (end - start) // self.unroll

        @pl.loop(0, full)
        def _chunk(b: _Dynamic) -> None:
            """Unroll full route chunks."""
            for u in range(self.unroll):
                _update(start + b * self.unroll + u)

        @pl.loop(start + full * self.unroll, end)
        def _tail(i_idx: _Dynamic) -> None:
            """Accumulate remaining route rows."""
            _update(i_idx)

    def load_routes(
        self,
        *_p_args: _Dynamic,
        **_p_kw: _Dynamic,
    ) -> None:
        """Load streaming routes for chunk."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            chunk,
            dest_tiles,
            prob_tiles,
            route_sem,
            route_tags,
            destinations,
            probabilities,
        ) = (
            _p_next(_p_iter, _p_get("chunk")),
            _p_next(_p_iter, _p_get("dest_tiles")),
            _p_next(_p_iter, _p_get("prob_tiles")),
            _p_next(_p_iter, _p_get("route_sem")),
            _p_next(_p_iter, _p_get("route_tags")),
            _p_next(_p_iter, _p_get("destinations")),
            _p_next(_p_iter, _p_get("probabilities")),
        )
        begin = chunk * self.route_chunk
        copies = (
            pltpu.make_async_copy(
                destinations.at[pl.ds(begin, self.route_span)],
                dest_tiles.at[chunk % 2],
                route_sem,
            ),
            pltpu.make_async_copy(
                probabilities.at[pl.ds(begin, self.route_span)],
                prob_tiles.at[chunk % 2],
                route_sem,
            ),
        )
        for cp in copies:
            cp.start()
        for cp in copies:
            cp.wait()
        if self.metadata_cache_rows:
            route_tags[chunk % 2] = chunk

    def overlap_work(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Execute overlap work between MXU panel issue/use."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            panel,
            previous_g,
            metadata,
            dest_tiles,
            prob_tiles,
            destinations,
            probabilities,
            previous,
            sums,
            g,
        ) = (
            _p_next(_p_iter, _p_get("panel")),
            _p_next(_p_iter, _p_get("previous_g")),
            _p_next(_p_iter, _p_get("metadata")),
            _p_next(_p_iter, _p_get("dest_tiles")),
            _p_next(_p_iter, _p_get("prob_tiles")),
            _p_next(_p_iter, _p_get("destinations")),
            _p_next(_p_iter, _p_get("probabilities")),
            _p_next(_p_iter, _p_get("previous")),
            _p_next(_p_iter, _p_get("sums")),
            _p_next(_p_iter, _p_get("g")),
        )
        start = metadata.gm_id_to_m_offset[previous_g]
        end = metadata.gm_id_to_m_offset[previous_g + 1]
        begin = start // self.sub * self.sub
        for row in range(self.unroll):
            off = panel * self.unroll + row
            i = jnp.minimum(start + off, end - 1)
            dest, prob = self.route(
                previous_g,
                i,
                metadata,
                dest_tiles,
                prob_tiles,
                destinations,
                probabilities,
            )
            weight = jnp.where((g > 0) & (start + off < end), prob, 0)
            value = previous.at[pl.ds(i - begin, 1)][...].astype(jnp.float32)
            target = sums.at[pl.ds(dest, 1)]
            target[...] = target[...] + value * weight

    def inner_step(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Execute single pipeline inner step."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (x, w, partial, acc, metadata, ctx) = (
            _p_next(_p_iter, _p_get("x")),
            _p_next(_p_iter, _p_get("w")),
            _p_next(_p_iter, _p_get("partial")),
            _p_next(_p_iter, _p_get("acc")),
            _p_next(_p_iter, _p_get("metadata")),
            _p_next(_p_iter, _p_get("ctx")),
        )
        g, n = pl.program_id(1), pl.program_id(0)
        if self.stream_metadata:
            chunk = metadata.gm_id_to_m_offset[g] // self.route_chunk
            prev_chunk = (
                metadata.gm_id_to_m_offset[jnp.maximum(g - 1, 0)] // self.route_chunk
            )
            missing = (
                ctx["route_tags"][chunk % 2] != chunk
                if self.metadata_cache_rows
                else (g == 0) | (chunk != prev_chunk)
            )
            pl.when(missing)(
                lambda: self.load_routes(
                    chunk,
                    ctx["dest_tiles"],
                    ctx["prob_tiles"],
                    ctx["route_sem"],
                    ctx["route_tags"],
                    ctx["destinations"],
                    ctx["probabilities"],
                )
            )

        @pl.when(g == 0)
        def _clear_sums() -> None:
            """Zero subtotal accumulator at start of column block."""
            ctx["sums"][...] = jnp.zeros(ctx["sums"].shape, jnp.float32)

        pl.when(n > 0)(
            lambda: self.advance(
                jnp.minimum(32, (g + 1) * 32 // ctx["ng"]),
                n - 1,
                ctx["state"],
                ctx["tables"],
                ctx["subtotal_vmem"],
                ctx["out"],
                ctx["sends"],
                ctx["receives"],
                ctx["credit_tokens"],
                ctx["stores"],
                ctx["payload"],
                ctx["rx"],
            )
        )
        current = ctx["packed"].at[g % 2]
        previous = ctx["packed"].at[(g + 1) % 2]
        previous_g = jnp.maximum(g - 1, 0)
        panel_count = pl.cdiv(self.tn, pltpu.get_tpu_info().mxu_column_size)

        def overlap(p: _Dynamic) -> _Dynamic:
            """Execute overlap work between MXU panel issue/use."""
            return self.overlap_work(
                p,
                previous_g,
                metadata,
                ctx["dest_tiles"],
                ctx["prob_tiles"],
                ctx["destinations"],
                ctx["probabilities"],
                previous,
                ctx["sums"],
                g,
            )

        @pl.when(g == 0)
        def _init_previous() -> None:
            """Zero previous tile buffer at start of column block."""
            previous[...] = jnp.zeros(previous.shape, previous.dtype)

        pl.when(ctx["active_ng"] > 0)(
            lambda: packed_inner_kernel(
                x,
                w,
                current,
                partial,
                acc,
                metadata,
                cfgs=self.cfg,
                overlap_work=overlap,
            )
        )
        pl.when(g > 0)(
            lambda: self.consume(
                previous_g,
                previous,
                metadata,
                ctx["sums"],
                ctx["dest_tiles"],
                ctx["prob_tiles"],
                ctx["destinations"],
                ctx["probabilities"],
                offset=panel_count * self.unroll,
            )
        )

        def finish_g() -> None:
            """Complete reduction for current group."""
            self.consume(
                g,
                current,
                metadata,
                ctx["sums"],
                ctx["dest_tiles"],
                ctx["prob_tiles"],
                ctx["destinations"],
                ctx["probabilities"],
            )
            pl.when(n > 0)(
                lambda: self.advance(
                    jnp.int32(32),
                    n - 1,
                    ctx["state"],
                    ctx["tables"],
                    ctx["subtotal_vmem"],
                    ctx["out"],
                    ctx["sends"],
                    ctx["receives"],
                    ctx["credit_tokens"],
                    ctx["stores"],
                    ctx["payload"],
                    ctx["rx"],
                )
            )
            self.emit_subtotals(n, ctx["subtotal_vmem"], ctx["sums"], ctx["cast"])
            ctx["state"][0] = 0
            pl.when(n == self.width // self.tn - 1)(
                lambda: self.advance(
                    jnp.int32(32),
                    n,
                    ctx["state"],
                    ctx["tables"],
                    ctx["subtotal_vmem"],
                    ctx["out"],
                    ctx["sends"],
                    ctx["receives"],
                    ctx["credit_tokens"],
                    ctx["stores"],
                    ctx["payload"],
                    ctx["rx"],
                )
            )

        pl.when(g == ctx["ng"] - 1)(finish_g)

    def __call__(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
        """Execute kernel."""
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
            x,
            w,
            out,
            partial,
            acc,
            metadata,
            packed,
            sums,
            cast,
            sends,
            receives,
            credit_tokens,
            stores,
            payload,
            rx,
            state,
            subtotal_vmem,
            dest_tiles,
            prob_tiles,
            route_sem,
            route_tags,
        ) = (
            _p_next(_p_iter, _p_get("counts")),
            _p_next(_p_iter, _p_get("offset")),
            _p_next(_p_iter, _p_get("layer")),
            _p_next(_p_iter, _p_get("tables")),
            _p_next(_p_iter, _p_get("destinations")),
            _p_next(_p_iter, _p_get("probabilities")),
            _p_next(_p_iter, _p_get("x")),
            _p_next(_p_iter, _p_get("w")),
            _p_next(_p_iter, _p_get("out")),
            _p_next(_p_iter, _p_get("partial")),
            _p_next(_p_iter, _p_get("acc")),
            _p_next(_p_iter, _p_get("metadata")),
            _p_next(_p_iter, _p_get("packed")),
            _p_next(_p_iter, _p_get("sums")),
            _p_next(_p_iter, _p_get("cast")),
            _p_next(_p_iter, _p_get("sends")),
            _p_next(_p_iter, _p_get("receives")),
            _p_next(_p_iter, _p_get("credit_tokens")),
            _p_next(_p_iter, _p_get("stores")),
            _p_next(_p_iter, _p_get("payload")),
            _p_next(_p_iter, _p_get("rx")),
            _p_next(_p_iter, _p_get("state")),
            _p_next(_p_iter, _p_get("subtotal_vmem")),
            _p_next(_p_iter, _p_get("dest_tiles")),
            _p_next(_p_iter, _p_get("prob_tiles")),
            _p_next(_p_iter, _p_get("route_sem")),
            _p_next(_p_iter, _p_get("route_tags")),
        )
        if self.family:
            w = dataclasses.replace(
                w, weight=w.weight.at[layer[0]], scale=w.scale.at[layer[0]]
            )
        w = dataclasses.replace(w, weight=w.weight.bitcast(jnp.float4_e2m1fn))
        active_ng = k.fill_metadata(counts, offset, metadata, cfgs=self.cfg)
        pl.when(active_ng == 0)(lambda: self.empty_metadata(metadata))
        ng = jnp.maximum(active_ng, 1)
        (xs, ws), _ = k.generate_block_specs(metadata, self.cfg)
        state[0] = 0
        if self.metadata_cache_rows:
            route_tags[0] = -1
            route_tags[1] = -1
        ctx = {
            "active_ng": active_ng,
            "ng": ng,
            "state": state,
            "tables": tables,
            "subtotal_vmem": subtotal_vmem,
            "out": out,
            "sends": sends,
            "receives": receives,
            "credit_tokens": credit_tokens,
            "stores": stores,
            "payload": payload,
            "rx": rx,
            "sums": sums,
            "cast": cast,
            "dest_tiles": dest_tiles,
            "prob_tiles": prob_tiles,
            "route_sem": route_sem,
            "route_tags": route_tags,
            "destinations": destinations,
            "probabilities": probabilities,
            "packed": packed,
        }

        def step_fn(
            x_in: _Dynamic,
            w_in: _Dynamic,
            partial_in: _Dynamic,
            acc_in: _Dynamic,
            meta_in: _Dynamic,
        ) -> None:
            """Execute pipeline step."""
            self.inner_step(x_in, w_in, partial_in, acc_in, meta_in, ctx=ctx)

        pltpu.emit_pipeline(
            step_fn,
            grid=(self.width // self.tn, ng, 1),
            in_specs=(xs, ws),
            out_specs=(),
        )(x.reshape(-1, self.sub, x.shape[-1]), w, scratches=(partial, acc, metadata))


def down_valu(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Complete BF16 products are combined with routing weights."""
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        x,
        rhs,
        scales,
        counts,
        destinations,
        probabilities,
        tables,
    ) = (
        _p_next(_p_iter, _p_get("x")),
        _p_next(_p_iter, _p_get("rhs")),
        _p_next(_p_iter, _p_get("scales")),
        _p_next(_p_iter, _p_get("counts")),
        _p_next(_p_iter, _p_get("destinations")),
        _p_next(_p_iter, _p_get("probabilities")),
        _p_next(_p_iter, _p_get("tables")),
    )
    (
        tokens,
        columns,
        tile,
        unroll,
        layer,
        stream_metadata,
        metadata_cache_rows,
        metadata_row_capacity,
        ring_dtype,
    ) = (
        _p_get("tokens"),
        _p_get("columns", 1792),
        _p_get("tile", 128),
        _p_get("unroll", 8),
        _p_get("layer"),
        _p_kw.get("stream_metadata", False),
        _p_get("metadata_cache_rows", 0),
        _p_get("metadata_row_capacity", 0),
        _p_get("ring_dtype", jnp.float32),
    )
    if not (tile in (64, 128, 256) and unroll > 0):
        _err_tile = "Assertion failed: tile in (64, 128, 256) and unroll > 0"
        raise AssertionError(_err_tile)
    if tokens not in (4096, 8192):
        _err_tokens = "Assertion failed: tokens in (4096, 8192)"
        raise AssertionError(_err_tokens)
    rows, width = metadata_row_capacity or x.shape[0], rhs.shape[-1]
    if not (rows <= x.shape[0]):
        _err_rows = "Assertion failed: rows <= x.shape[0]"
        raise AssertionError(_err_rows)
    tm, tn, localrows = tile, columns, tokens // 32
    if width % tn != 0:
        _err_width = "Assertion failed: width % tn == 0"
        raise AssertionError(_err_width)
    limit = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.98)
    offset = jnp.zeros((1,), jnp.int32)
    family = rhs.ndim == _CONST_4
    if family != (layer is not None):
        _err_layer = "Resident weights require a layer index"
        raise ValueError(_err_layer)
    if layer is None:
        layer = offset
    cfg = k.make_gmm_configs(
        jax.ShapeDtypeStruct((rows, x.shape[-1]), x.dtype),
        jax.ShapeDtypeStruct(rhs.shape[1:] if family else rhs.shape, rhs.dtype),
        jax.ShapeDtypeStruct(
            scales.shape[1:] if family else scales.shape, scales.dtype
        ),
        counts,
        offset,
        tile_info=k.calculate_tiling,
        vmem_limit_bytes=limit,
        out_dtype=None,
        acc_dtype=None,
        fuse_act=None,
        rhs_quant_dtype=jnp.float4_e2m1fn,
    )
    cfg = dataclasses.replace(
        cfg, tiles=dataclasses.replace(cfg.tiles, tile_m=tm, tile_n=tn)
    )
    sub = cfg.dims.size_lhs_sublane
    max_gm = cfg.dims.size_group + pl.cdiv(rows, tm) - 1
    regions = operator.attrgetter("_regions")(k)(localrows, tn)
    streams = len(regions)
    store_tile = 128
    if metadata_cache_rows and not (stream_metadata and metadata_cache_rows % 128 == 0):
        _err_cache = (
            "Assertion failed: stream_metadata and metadata_cache_rows % 128 == 0"
        )
        raise AssertionError(_err_cache)
    route_chunk = min(rows, metadata_cache_rows or 8192)
    route_span = route_chunk + tm
    kernel = _DownValuKernel(
        cfg,
        family,
        streams,
        regions,
        tn,
        tm,
        store_tile,
        unroll,
        sub,
        route_chunk,
        route_span,
        stream_metadata,
        metadata_cache_rows,
        width,
        localrows,
        tokens,
    )
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    scratch_shapes: _Dynamic = (
        None,
        pltpu.VMEM((tm, tn), jnp.float32),
        k.MetadataRef(
            gm_id_to_group_id=pltpu.SMEM((max_gm,), jnp.int32),
            gm_id_to_m_offset=pltpu.SMEM((max_gm + 1,), jnp.int32),
        ),
        pltpu.VMEM((2, tm, 2, tn // 2), x.dtype),
        pltpu.VMEM((tokens, 2, tn // 2), jnp.float32),
        pltpu.VMEM((store_tile, tn), x.dtype),
        pltpu.SemaphoreType.DMA((streams,)),
        pltpu.SemaphoreType.DMA((2, streams)),
        pltpu.SemaphoreType.REGULAR((streams,)),
        pltpu.SemaphoreType.DMA,
        pltpu.VMEM((localrows, tn), ring_dtype),
        pltpu.VMEM((2, localrows, tn), ring_dtype),
        pltpu.SMEM((1,), jnp.int32),
        pltpu.VMEM((tokens, tn), x.dtype),
        pltpu.SMEM((2, route_span), jnp.int32) if stream_metadata else None,
        pltpu.SMEM((2, route_span), jnp.float32) if stream_metadata else None,
        pltpu.SemaphoreType.DMA if stream_metadata else None,
        pltpu.SMEM((2,), jnp.int32) if metadata_cache_rows else None,
    )
    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((localrows, width), ring_dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4 if stream_metadata else 6,
            in_specs=((hbm, hbm) if stream_metadata else ())
            + (hbm, k.WeightsRef(weight=hbm, scale=hbm)),
            out_specs=hbm,
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(collective_id=77, vmem_limit_bytes=limit),
        name=f"down_valu_m{tm}_n{tn}_u{unroll}_mc{metadata_cache_rows}",
    )
    return call(
        counts,
        offset,
        layer,
        jnp.asarray(tables, jnp.int32),
        jnp.pad(destinations, (0, route_span)) if stream_metadata else destinations,
        jnp.pad(probabilities, (0, route_span)) if stream_metadata else probabilities,
        x,
        k.WeightsRef(weight=rhs, scale=scales),
    )


def _packed_inner_matmul(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
    """Execute inner tile matmul for packed inner kernel."""
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        bucket_m,
        tiled_lhs_ref,
        tiled_rhs_ref,
        tiled_out_ref,
        acc_ref,
        cfgs,
        overlap_work,
        m_start_local,
        m_end_local,
    ) = (
        _p_next(_p_iter, _p_get("bucket_m")),
        _p_next(_p_iter, _p_get("tiled_lhs_ref")),
        _p_next(_p_iter, _p_get("tiled_rhs_ref")),
        _p_next(_p_iter, _p_get("tiled_out_ref")),
        _p_next(_p_iter, _p_get("acc_ref")),
        _p_next(_p_iter, _p_get("cfgs")),
        _p_next(_p_iter, _p_get("overlap_work")),
        _p_next(_p_iter, _p_get("m_start_local")),
        _p_next(_p_iter, _p_get("m_end_local")),
    )
    mxu_size = pltpu.get_tpu_info().mxu_column_size
    tiled_lhs = tiled_lhs_ref.reshape(-1, cfgs.tiles.tile_k)[:bucket_m]
    tiled_rhs = tiled_rhs_ref.get_weight()
    rhs_tile_n = tiled_rhs.shape[1]
    rhs_qbs = cfgs.rhs_cfgs.quant_block_size
    if not (rhs_qbs is not None):
        _err_msg = "Assertion failed: rhs_qbs is not None"
        raise AssertionError(_err_msg)
    if cfgs.rhs_cfgs.should_dequantize_before_matmul:
        tiled_rhs_scale = tiled_rhs_ref.get_scale(replicate_size=rhs_qbs).astype(
            cfgs.lhs_cfgs.dtype
        )
        num_blocks = cfgs.num_quant_blocks_per_tile_k
        tiled_rhs_dequant = tiled_rhs.astype(cfgs.lhs_cfgs.dtype).reshape(
            num_blocks, rhs_qbs, rhs_tile_n
        )
        tiled_rhs_dequant = tiled_rhs_dequant * tiled_rhs_scale
        tiled_rhs = tiled_rhs_dequant.reshape(cfgs.tiles.tile_k, rhs_tile_n)
        rhs_qbs = cfgs.tiles.tile_k
    acc_list = []
    for start_n in range(0, rhs_tile_n, mxu_size):
        end_n = min(rhs_tile_n, start_n + mxu_size)
        col_size = end_n - start_n
        acc_n = jnp.zeros((bucket_m, col_size), dtype=acc_ref.dtype)
        for start_k in range(0, cfgs.tiles.tile_k, rhs_qbs):
            end_k = min(cfgs.tiles.tile_k, start_k + rhs_qbs)
            block_acc = jnp.matmul(
                tiled_lhs[:, start_k:end_k],
                tiled_rhs[start_k:end_k, start_n:end_n],
                preferred_element_type=jnp.float32,
            ).astype(acc_ref.dtype)
            if end_k == cfgs.tiles.tile_k:
                overlap_work(start_n // mxu_size)
            if cfgs.rhs_cfgs.should_dequantize_after_matmul:
                b_id = start_k // rhs_qbs
                rhs_scale_replicated = tiled_rhs_ref.get_scale(replicate_size=bucket_m)[
                    b_id, :, start_n : start_n + col_size
                ]
                block_acc *= rhs_scale_replicated.astype(acc_ref.dtype)
            acc_n += block_acc
        acc_list.append(acc_n)
    acc = jnp.concatenate(acc_list, axis=1)
    acc_m = acc.shape[0]
    iota = lax.broadcasted_iota(jnp.int32, acc.shape, 0)
    mask = jnp.logical_and(m_start_local <= iota, iota < m_end_local)
    acc_masked = jnp.where(mask, acc, 0)
    tiled_out_ref[:acc_m] = acc_masked.astype(tiled_out_ref.dtype).reshape(
        acc_m, 2, cfgs.tiles.tile_n // 2
    )


def packed_inner_kernel(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Variant of inner_kernel with a direct packed final store."""
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        tiled_lhs_ref,
        tiled_rhs_ref,
        tiled_out_ref,
    ) = (
        _p_next(_p_iter, _p_get("tiled_lhs_ref")),
        _p_next(_p_iter, _p_get("tiled_rhs_ref")),
        _p_next(_p_iter, _p_get("tiled_out_ref")),
    )
    next(_p_iter, None)
    acc_ref = _p_next(_p_iter, _p_get("acc_ref"))
    metadata_ref = _p_next(_p_iter, _p_get("metadata_ref"))
    (
        cfgs,
        overlap_work,
    ) = (
        _p_get("cfgs"),
        _p_get("overlap_work"),
    )
    if cfgs.dims.size_k != cfgs.tiles.tile_k:
        _err_msg = "Assertion failed: cfgs.dims.size_k == cfgs.tiles.tile_k"
        raise AssertionError(_err_msg)
    if cfgs.fuse_act is not None:
        _err_msg = "Assertion failed: cfgs.fuse_act is None"
        raise AssertionError(_err_msg)
    gm_id = pl.program_id(1)
    m_start = metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = metadata_ref.gm_id_to_m_offset[gm_id + 1]
    m_offset = m_start - m_start % cfgs.dims.size_lhs_sublane
    m_start_local = m_start - m_offset
    m_end_local = m_end - m_offset

    def run_matmul_step(bucket_idx: int) -> _Dynamic:
        """Execute run matmul step logic."""
        bucket_m = cfgs.tiles.bucket_base * (bucket_idx + 1)
        with jax.named_scope(f"bm{bucket_m}_fragment"):
            _packed_inner_matmul(
                bucket_m,
                tiled_lhs_ref,
                tiled_rhs_ref,
                tiled_out_ref,
                acc_ref,
                cfgs,
                overlap_work,
                m_start_local,
                m_end_local,
            )

    branches = [
        functools.partial(run_matmul_step, bucket_idx=bucket_idx)
        for bucket_idx in range(cfgs.tiles.tile_m // cfgs.tiles.bucket_base)
    ]
    bucket_idx = m_end_local // cfgs.tiles.bucket_base
    lax.switch(bucket_idx, branches)


def down_valu_compact(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Uniform compact-prefix fast path with an exact, uncapped overflow path.

    All ranks must select the same collective-containing branch. The caller
    supplies the maximum derived from globally replicated route counts to
    avoid an additional collective. The full activation buffer stays in HBM;
    only the scalar metadata prefix is bounded. Overflow never drops routes.
    """
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        x,
        rhs,
        scales,
        counts,
        destinations,
        probabilities,
        tables,
    ) = (
        _p_next(_p_iter, _p_get("x")),
        _p_next(_p_iter, _p_get("rhs")),
        _p_next(_p_iter, _p_get("scales")),
        _p_next(_p_iter, _p_get("counts")),
        _p_next(_p_iter, _p_get("destinations")),
        _p_next(_p_iter, _p_get("probabilities")),
        _p_next(_p_iter, _p_get("tables")),
    )
    (
        metadata_prefix_rows,
        slice_input,
        global_route_max,
    ) = (
        _p_get("metadata_prefix_rows"),
        _p_kw.get("slice_input", True),
        _p_get("global_route_max"),
    )
    kwargs = _p_kw
    capacity = metadata_prefix_rows
    peak = (
        lax.pmax(counts.sum(), "k3_sp")
        if global_route_max is None
        else global_route_max
    )
    fits = peak <= capacity
    compact_args = dict(kwargs, stream_metadata=False, metadata_cache_rows=0)
    overflow_args = dict(kwargs, stream_metadata=True, metadata_cache_rows=8192)
    return lax.cond(
        fits,
        lambda: down_valu(
            x[:capacity] if slice_input else x,
            rhs,
            scales,
            counts,
            destinations[:capacity],
            probabilities[:capacity],
            tables,
            metadata_row_capacity=capacity,
            **compact_args,
        ),
        lambda: down_valu(
            x, rhs, scales, counts, destinations, probabilities, tables, **overflow_args
        ),
    )
