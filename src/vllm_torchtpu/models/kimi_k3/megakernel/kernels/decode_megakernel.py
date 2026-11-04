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

"""Kimi K3 decode megakernel components.

Source provenance: Adapted from inferact/k3-megakernel@30c9ba966b.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import math
import operator
import sys
import warnings
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
from jax import typing as jax_typing
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

from vllm_torchtpu.models.kimi_k3.megakernel import kernels as kimi
from vllm_torchtpu.models.kimi_k3.megakernel.common import collectives32, pool_alias
from vllm_torchtpu.models.kimi_k3.megakernel.common.vmem_pool import VmemPool
from vllm_torchtpu.models.kimi_k3.megakernel.config import active as _config
from vllm_torchtpu.models.kimi_k3.megakernel.layout import CONV_HEAD_ROWS, cache_layout

from . import mla_dcp

_Dynamic = Any


def _dyn(val: _Dynamic) -> _Dynamic:
    """Coerce value to dynamic type for static analyzers."""
    return val


_CONST_1024 = 1024
_CONST_12 = 12
_CONST_128 = 128
_CONST_16 = 16
_CONST_1664 = 1664
_CONST_2 = 2
_CONST_3 = 3
_CONST_32 = 32
_CONST_384 = 384
_CONST_4 = 4
_CONST_64 = 64
_CONST_7 = 7
_CONST_7168 = 7168
_CONST_768 = 768
_CONST_8 = 8
_CONST_8192 = 8192
_CONST_896 = 896
_CONST_92 = 92
_CONST_96 = 96


@dataclasses.dataclass(frozen=True, slots=True)
class _PaddedBf16Rows(pool_alias.Reinterpret):
    """Expose active rows while preserving the full native row capacity (a.

    multiple of 16 rows).
    """

    def __getattr__(self, name: _Dynamic) -> _Dynamic:
        """Dynamic attribute access."""
        raise AttributeError(name)

    def _validate_shape(self, input_shape: _Dynamic) -> _Dynamic:
        """Execute validate shape logic."""
        rows, width = self.shape
        backing_rows = math.prod(input_shape) // width
        if not (
            len(self.shape) == _CONST_2
            and width % 128 == 0
            and (1 <= rows <= backing_rows)
        ):
            _err_msg = (
                "Assertion failed: len(self.shape) == 2 and width % 128 == 0 "
                "and (1 <= rows <= backing_rows)"
            )
            raise AssertionError(_err_msg)
        if not (
            backing_rows * width == math.prod(input_shape) and backing_rows % 16 == 0
        ):
            _err_msg = (input_shape, self.shape)
            raise AssertionError(_err_msg)


# Named phase scopes for XProf custom-call tracing (KernelConfig.phase_scopes=True).
PHASE_SCOPES = _config().phase_scopes


def _phase(name: _Dynamic) -> _Dynamic:
    """Execute phase logic."""
    return jax.named_scope(name) if PHASE_SCOPES else contextlib.nullcontext()


def _phase_start(name: _Dynamic) -> _Dynamic:
    """Execute phase start logic."""
    scope = _phase(name)
    scope.__enter__()
    return scope


def _phase_end(scope: _Dynamic) -> _Dynamic:
    """Execute phase end logic."""
    scope.__exit__(None, None, None)


def _scoped(name: _Dynamic) -> _Dynamic:
    """Decorator form of _phase for pl.when bodies."""

    def wrap(function: _Dynamic) -> _Dynamic:
        """Execute wrap logic."""

        @functools.wraps(function)
        def scoped(*args: _Dynamic, **kwargs: _Dynamic) -> _Dynamic:
            """Execute scoped logic."""
            with _phase(name):
                return function(*args, **kwargs)

        return scoped

    return wrap


# Largest supported decode batch; SCATTER_BATCHES lists the token-scatter batches.
MAX_BATCH = 128
SCATTER_BATCHES = (8, 12, 16, 20, 24, 28, 32, 64, 96, 128)


def _bf16_rows_view(ref: _Dynamic, rows: _Dynamic, width: _Dynamic) -> _Dynamic:
    """Execute bf16 rows view logic."""
    operator.attrgetter("_ensure_patched")(pool_alias)()
    ref = ref.bitcast(jnp.bfloat16)
    return pool_alias.types.TransformedRef(
        ref.ref,
        (*ref.transforms, _PaddedBf16Rows((rows, width))),
    )


def _dot(x: _Dynamic, w: _Dynamic) -> _Dynamic:
    """Execute dot logic."""
    batch_size = x.shape[0]
    if not 1 <= batch_size <= MAX_BATCH:
        _err_msg = f"_dot supports 1 to {MAX_BATCH} rows"
        raise ValueError(_err_msg)
    if batch_size == 1:
        # B1 broadcasts into one 16-row tile instead of zero-padding.
        tiled_x = jnp.broadcast_to(x, (16, x.shape[-1]))
    elif batch_size % 16:
        # Whole native 16-row MXU tiles.
        tiled_x = jnp.pad(x, ((0, -batch_size % 16), (0, 0)))
    else:
        tiled_x = x
    return jax.lax.dot_general(
        tiled_x,
        w,
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )[:batch_size]


def _attention_scores(queries: _Dynamic, keys: _Dynamic) -> _Dynamic:
    """Multiply partial query rows by a transposed key tile in FP32."""
    row_count = queries.shape[0]
    padded_queries = jnp.pad(queries, ((0, -row_count % 16), (0, 0)))
    return jax.lax.dot_general(
        padded_queries,
        keys,
        (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )[:row_count]


def _gated_activation(gate: _Dynamic, value: _Dynamic) -> _Dynamic:
    """Kimi K3 shared/routed expert activation, rounded to BF16."""
    return (
        4 * jnp.tanh(gate / 4) * jax.nn.sigmoid(gate) * 25 * jnp.tanh(value / 25)
    ).astype(jnp.bfloat16)


def _route_hidden_states(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Project and select local expert routes through one shared B1/B2 path.

    ``routing_scores_vmem`` is [batch_size, 1024]. Paired ranks exchange their
    512-score halves directly in that padded workspace, then every top-k
    iteration advances all batch rows together. Selected local routes are
    returned as [batch_size, top_k] values for scalar SMEM materialization by
    the caller.
    """
    _p_iter = iter(_p_args)
    (
        hidden_states,
        routing_projection_vmem,
        routing_bias_ref,
        routing_scores_vmem,
        send_semaphore,
        receive_semaphore,
        selected_expert_count,
        local_expert_count,
        expert_tensor_parallel,
        on_local_selection,
        after_selection,
        after_scores,
        after_score_send,
        live_rows,
    ) = (
        _dyn(next(_p_iter, _p_kw.get("hidden_states"))),
        _dyn(next(_p_iter, _p_kw.get("routing_projection_vmem"))),
        _dyn(next(_p_iter, _p_kw.get("routing_bias_ref"))),
        _dyn(next(_p_iter, _p_kw.get("routing_scores_vmem"))),
        _dyn(next(_p_iter, _p_kw.get("send_semaphore"))),
        _dyn(next(_p_iter, _p_kw.get("receive_semaphore"))),
        _dyn(_p_kw.get("selected_expert_count")),
        _dyn(_p_kw.get("local_expert_count")),
        _dyn(_p_kw.get("expert_tensor_parallel")),
        _dyn(_p_kw.get("on_local_selection")),
        _dyn(_p_kw.get("after_selection")),
        _dyn(_p_kw.get("after_scores")),
        _dyn(_p_kw.get("after_score_send")),
        _dyn(_p_kw.get("live_rows")),
    )
    rank = jax.lax.axis_index("tp")
    batch_size = hidden_states.shape[0]
    expert_count = routing_bias_ref.shape[-1]
    if expert_count != _CONST_896 or hidden_states.shape[-1] > _CONST_7168:
        _err_msg = "Host-shared router requires 896 expert scores"
        raise ValueError(_err_msg)
    if routing_scores_vmem.shape[-1] != _CONST_1024:
        _err_msg = "Routing score workspace must have padded width 1024"
        raise ValueError(_err_msg)

    router_lane = rank % 2
    local_scores = _dot(hidden_states, routing_projection_vmem[0]).reshape(
        batch_size, 512
    )
    # Lane 1's last 128 router columns are padding; in the shared arena they
    # hold stale bytes, so zero their scores.
    column = jax.lax.broadcasted_iota(jnp.int32, local_scores.shape, 1)
    local_scores = jnp.where(
        (router_lane == 0) | (column < _CONST_384), local_scores, 0.0
    )
    score_half = routing_scores_vmem.at[:, pl.ds(router_lane * 512, 512)]
    score_half[...] = local_scores
    if after_scores is not None:
        after_scores()  # the router projection is consumed
    copy = tpu.make_async_remote_copy(
        score_half,
        score_half,
        send_semaphore,
        receive_semaphore,
        device_id=(rank ^ 1,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    copy.start()
    if after_score_send is not None:
        after_score_send()
    copy.wait()

    raw_scores = routing_scores_vmem[:, :expert_count]
    uncorrected_probabilities = jax.nn.sigmoid(raw_scores)
    routing_scores_vmem[:, :expert_count] = (
        uncorrected_probabilities + routing_bias_ref[...]
    )
    selected_experts = jnp.zeros((batch_size, selected_expert_count), jnp.int32)
    selected_probabilities = jnp.zeros((batch_size, selected_expert_count), jnp.float32)
    local_route_counts = jnp.zeros((batch_size,), jnp.int32)
    probability_totals = jnp.zeros((batch_size,), jnp.float32)
    expert_ids = jnp.arange(expert_count)[None, :]
    route_slots = jnp.arange(selected_expert_count)[None, :]
    # Only top-k rank is sequential. Every score reduction and ownership
    # operation carries the leading batch dimension with no per-row control.
    for selection in range(selected_expert_count):
        winners = jnp.argmax(routing_scores_vmem[:, :expert_count], axis=1).astype(
            jnp.int32
        )
        winner_mask = expert_ids == winners[:, None]
        probabilities = jnp.sum(
            jnp.where(winner_mask, uncorrected_probabilities, 0), axis=1
        )
        probability_totals += probabilities
        routing_scores_vmem[:, :expert_count] = jnp.where(
            winner_mask, -jnp.inf, routing_scores_vmem[:, :expert_count]
        )

        counts_before_selection = local_route_counts
        owned_locally = winners // local_expert_count == rank // expert_tensor_parallel
        if live_rows is not None:
            owned_locally &= live_rows
        destination_mask = owned_locally[:, None] & (
            route_slots == counts_before_selection[:, None]
        )
        selected_experts = jnp.where(
            destination_mask,
            (winners % local_expert_count)[:, None],
            selected_experts,
        )
        selected_probabilities = jnp.where(
            destination_mask,
            probabilities[:, None],
            selected_probabilities,
        )
        local_route_counts = counts_before_selection + owned_locally.astype(jnp.int32)
        on_local_selection(
            winners, owned_locally, counts_before_selection, selected_experts
        )
        after_selection(selection)

    return (
        selected_experts,
        selected_probabilities,
        local_route_counts,
        probability_totals
        if live_rows is None
        else jnp.where(live_rows, probability_totals, 1.0),
    )


class _RouteOwnedRowsRunner:
    """Runner for sharded routing of owned rows."""

    def __getattr__(self, name: _Dynamic) -> _Dynamic:
        """Dynamic attribute access."""
        raise AttributeError(name)

    def __init__(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Initialize route owned rows runner."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            self.owned_hidden_ref,
            self.batch_size,
            self.routing_projection_vmem,
            self.routing_bias_ref,
            self.routing_scores_vmem,
            self.selection_source_vmem,
            self.selections_vmem,
            self.send_semaphores,
            self.receive_semaphores,
        ) = (
            _p_next(_p_iter, _p_get("owned_hidden_ref")),
            _p_next(_p_iter, _p_get("batch_size")),
            _p_next(_p_iter, _p_get("routing_projection_vmem")),
            _p_next(_p_iter, _p_get("routing_bias_ref")),
            _p_next(_p_iter, _p_get("routing_scores_vmem")),
            _p_next(_p_iter, _p_get("selection_source_vmem")),
            _p_next(_p_iter, _p_get("selections_vmem")),
            _p_next(_p_iter, _p_get("send_semaphores")),
            _p_next(_p_iter, _p_get("receive_semaphores")),
        )
        (
            self.selected_expert_count,
            self.local_expert_count,
            self.expert_tensor_parallel,
            self.on_local_selection,
            self.after_selection,
            self.after_scores,
            self.live_rows,
        ) = (
            _p_get("selected_expert_count"),
            _p_get("local_expert_count"),
            _p_get("expert_tensor_parallel"),
            _p_get("on_local_selection"),
            _p_get("after_selection"),
            _p_get("after_scores"),
            _p_get("live_rows"),
        )
        self.rank = jax.lax.axis_index("tp")
        self.rows = self.owned_hidden_ref.shape[0]
        self.pair = (self.rank % 8) // 2
        self.expert_count = self.routing_bias_ref.shape[-1]
        if self.rows * 4 != self.batch_size or self.expert_count != _CONST_896:
            _err_msg = "Sharded routing needs one row block per local rank pair"
            raise ValueError(_err_msg)

    def _compute_local_scores(self) -> None:
        """Compute local router scores."""
        router_lane = self.rank % 2
        local_scores = _dot(
            self.owned_hidden_ref[...], self.routing_projection_vmem[0]
        ).reshape(self.rows, 512)
        column = jax.lax.broadcasted_iota(jnp.int32, local_scores.shape, 1)
        local_scores = jnp.where(
            (router_lane == 0) | (column < _CONST_384), local_scores, 0.0
        )
        score_half = self.routing_scores_vmem.at[
            pl.ds(0, self.rows), pl.ds(router_lane * 512, 512)
        ]
        score_half[...] = local_scores
        if self.after_scores is not None:
            self.after_scores()
        copy = tpu.make_async_remote_copy(
            score_half,
            score_half,
            self.send_semaphores.at[1],
            self.receive_semaphores.at[1],
            device_id=(self.rank ^ 1,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        copy.wait()

    def _select_top_k(self) -> tuple[list[_Dynamic], list[_Dynamic], _Dynamic]:
        """Perform top-k expert selection."""
        scores = self.routing_scores_vmem.at[
            pl.ds(0, self.rows), pl.ds(0, self.expert_count)
        ]
        uncorrected_probabilities = jax.nn.sigmoid(scores[...])
        scores[...] = uncorrected_probabilities + self.routing_bias_ref[...]
        expert_ids = jnp.arange(self.expert_count)[None, :]
        winners_by_slot, probabilities_by_slot = [], []
        probability_totals = jnp.zeros((self.rows,), jnp.float32)
        for selection in range(self.selected_expert_count):
            winners = jnp.argmax(scores[...], axis=1).astype(jnp.int32)
            winner_mask = expert_ids == winners[:, None]
            probabilities = jnp.sum(
                jnp.where(winner_mask, uncorrected_probabilities, 0), axis=1
            )
            probability_totals += probabilities
            scores[...] = jnp.where(winner_mask, -jnp.inf, scores[...])
            winners_by_slot.append(winners)
            probabilities_by_slot.append(probabilities)
            self.after_selection(selection)
        return winners_by_slot, probabilities_by_slot, probability_totals

    def _gather_remote_selections(self, owner: int) -> None:
        """Gather selection data from remote peers."""
        destination = self.selections_vmem.at[pl.ds(owner * self.rows, self.rows)]
        destination[...] = self.selection_source_vmem[...]
        copies = []
        for offset in range(3, 0, -1):
            copy = tpu.make_async_remote_copy(
                self.selection_source_vmem,
                destination,
                self.send_semaphores.at[1 + offset],
                self.receive_semaphores.at[1 + offset],
                device_id=(self.rank ^ (2 * offset),),
                device_id_type=pl.DeviceIdType.MESH,
            )
            copy.start()
            copies.append(copy)
        for copy in copies:
            copy.wait()

    def _distribute_selections(
        self,
        winners_by_slot: list[_Dynamic],
        probabilities_by_slot: list[_Dynamic],
        probability_totals: _Dynamic,
    ) -> None:
        """Distribute selection records to all rank pairs."""
        k = self.selected_expert_count
        self.selection_source_vmem[...] = jnp.concatenate(
            (
                jnp.stack(winners_by_slot, axis=1).astype(jnp.float32),
                jnp.stack(probabilities_by_slot, axis=1),
                probability_totals[:, None],
                jnp.zeros((self.rows, 128 - 2 * k - 1), jnp.float32),
            ),
            axis=1,
        )
        for owner in range(4):
            pl.when(self.pair == owner)(
                lambda ow=owner: self._gather_remote_selections(ow)
            )

    def _derive_local_routes(self) -> _Dynamic:
        """Derive local route assignments from gathered selections."""
        k = self.selected_expert_count
        gathered = self.selections_vmem[...]
        all_winners = gathered[:, :k].astype(jnp.int32)
        all_probabilities = gathered[:, k : 2 * k]
        selected_experts = jnp.zeros((self.batch_size, k), jnp.int32)
        selected_probabilities = jnp.zeros((self.batch_size, k), jnp.float32)
        local_route_counts = jnp.zeros((self.batch_size,), jnp.int32)
        route_slots = jnp.arange(k)[None, :]
        for selection in range(k):
            winners = all_winners[:, selection]
            probabilities = all_probabilities[:, selection]
            counts_before_selection = local_route_counts
            owned_locally = (
                winners // self.local_expert_count
                == self.rank // self.expert_tensor_parallel
            )
            if self.live_rows is not None:
                owned_locally &= self.live_rows
            destination_mask = owned_locally[:, None] & (
                route_slots == counts_before_selection[:, None]
            )
            selected_experts = jnp.where(
                destination_mask,
                (winners % self.local_expert_count)[:, None],
                selected_experts,
            )
            selected_probabilities = jnp.where(
                destination_mask, probabilities[:, None], selected_probabilities
            )
            local_route_counts = counts_before_selection + owned_locally.astype(
                jnp.int32
            )
            self.on_local_selection(
                winners, owned_locally, counts_before_selection, selected_experts
            )
        return (
            selected_experts,
            selected_probabilities,
            local_route_counts,
            gathered[:, 2 * k]
            if self.live_rows is None
            else jnp.where(self.live_rows, gathered[:, 2 * k], 1.0),
        )

    def run(self) -> _Dynamic:
        """Run sharded route execution."""
        self._compute_local_scores()
        winners_by_slot, probabilities_by_slot, probability_totals = (
            self._select_top_k()
        )
        self._distribute_selections(
            winners_by_slot, probabilities_by_slot, probability_totals
        )
        return self._derive_local_routes()


def _route_owned_rows(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """``_route_hidden_states`` with the top-k sharded over the local rank pairs."""
    return _RouteOwnedRowsRunner(*_p_args, **_p_kw).run()


def _start_initial_expert_weight_copies(
    *_p_args: _Dynamic, **_p_kw: _Dynamic
) -> _Dynamic:
    """Prefetch the first routed experts in their execution order.

    This uses the same cross-row ordering as the expert stream, so it works for
    both shared and divergent routes without specializing on batch size.
    """
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        selected_experts,
        local_route_counts,
        copy_expert,
    ) = (
        _p_next(_p_iter, _p_get("selected_experts")),
        _p_next(_p_iter, _p_get("local_route_counts")),
        _p_next(_p_iter, _p_get("copy_expert")),
    )
    (
        slot_count,
        already_started,
        group_pending_routes,
    ) = (
        _p_get("slot_count"),
        _p_get("already_started"),
        _p_get("group_pending_routes"),
    )
    selected_expert_count = selected_experts.shape[1]
    route_slots = jnp.arange(selected_expert_count)[None, :]
    processed = jnp.zeros(selected_experts.shape, jnp.bool_)

    def next_expert(completed: _Dynamic) -> _Dynamic:
        """Execute next expert logic."""
        pending = (route_slots < local_route_counts[:, None]) & ~completed
        if group_pending_routes:
            minimum_expert = jnp.min(
                jnp.where(
                    pending,
                    selected_experts.astype(jnp.float32),
                    jnp.inf,
                )
            )
            has_expert = jnp.any(pending)
            expert = jnp.where(has_expert, minimum_expert, 0).astype(jnp.int32)
            consumed = pending & (selected_experts == expert)
        else:
            rows_with_routes = jnp.any(pending, axis=1)
            current_slots = jnp.sum(completed.astype(jnp.int32), axis=1)
            current_slot_mask = route_slots == current_slots[:, None]
            current_experts = jnp.sum(
                jnp.where(current_slot_mask, selected_experts, 0), axis=1
            )
            minimum_expert = jnp.min(
                jnp.where(
                    rows_with_routes,
                    current_experts.astype(jnp.float32),
                    jnp.inf,
                )
            )
            has_expert = jnp.any(rows_with_routes)
            expert = jnp.where(has_expert, minimum_expert, 0).astype(jnp.int32)
            consumed = (rows_with_routes & (current_experts == expert))[
                :, None
            ] & current_slot_mask
        return expert, consumed, has_expert

    for slot in range(slot_count):
        expert, consumed, has_expert = next_expert(processed)

        @pl.when(has_expert & (slot >= already_started))
        def start_copy(expert: _Dynamic = expert, slot: _Dynamic = slot) -> _Dynamic:
            """Execute start copy logic."""
            _start_async_copies(copy_expert(expert, slot))

        processed |= consumed


class _QuantizedExpertDotRunner:
    """Runner for resident packed MXFP4 matrix multiplication."""

    def __getattr__(self, name: _Dynamic) -> _Dynamic:
        """Dynamic attribute access."""
        raise AttributeError(name)

    def __init__(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Initialize quantized expert dot runner."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            self.source,
            self.packed,
            self.scales,
            self.accumulator,
            self.contraction_size,
            _,
        ) = (
            _p_next(_p_iter, _p_get("source")),
            _p_next(_p_iter, _p_get("packed")),
            _p_next(_p_iter, _p_get("scales")),
            _p_next(_p_iter, _p_get("accumulator")),
            _p_next(_p_iter, _p_get("contraction_size")),
            _p_next(_p_iter, _p_get("output_size")),
        )
        (
            self.block_major,
            dot_tile,
        ) = (
            _p_kw.get("block_major", False),
            _p_get("dot_tile"),
        )
        self.dot_tile = (MXFP4_DOT_TILE or 0) if dot_tile is None else dot_tile
        self.accumulator[...] = jnp.zeros(self.accumulator.shape, jnp.float32)

        if self.packed.dtype != jnp.uint32:
            _err_msg = "Expert weights must use packed MXFP4"
            raise ValueError(_err_msg)

        self.split_small_contraction = (
            self.contraction_size == _CONST_768 and self.block_major
        )
        self.block_size = (
            256
            if self.split_small_contraction
            else min(MXFP4_BLOCK, self.contraction_size)
        )
        self.static_tiles = MXFP4_STATIC_TILE > 0 and not self.split_small_contraction

    def _tile_values(self, start: _Dynamic, size: _Dynamic) -> _Dynamic:
        """Extract tiled values."""
        if self.static_tiles:
            return tpu.bitcast(
                self.packed[start // 8 : (start + size) // 8, :], jnp.float4_e2m1fn
            ).astype(jnp.float8_e4m3fn)
        return tpu.bitcast(
            self.packed[pl.ds(pl.multiple_of(start // 8, 8), size // 8), :],
            jnp.float4_e2m1fn,
        ).astype(jnp.float8_e4m3fn)

    def _tile_scale(self, start: _Dynamic, size: _Dynamic) -> _Dynamic:
        """Extract tiled exponents and scale."""
        if self.static_tiles or self.split_small_contraction:
            exponents = self.scales[...][start // 32 : (start + size) // 32, :].astype(
                jnp.uint32
            )
        else:
            exponents = self.scales[
                pl.ds(pl.multiple_of(start // 32, 32), size // 32), :
            ].astype(jnp.uint32)
        return jax.lax.bitcast_convert_type(exponents << 23, jnp.float32)

    def _tile_activation(self, start: _Dynamic, size: _Dynamic) -> _Dynamic:
        """Extract tiled activation."""
        activation_alignment = 128 if self.split_small_contraction else 512
        if self.static_tiles:
            return self.source[:, start : start + size].astype(jnp.bfloat16)
        return self.source[
            :, pl.ds(pl.multiple_of(start, activation_alignment), size)
        ].astype(jnp.bfloat16)

    def _dot_tiled_partials(
        self, activation: _Dynamic, values: _Dynamic, scale: _Dynamic, size: _Dynamic
    ) -> None:
        """Compute dot partials with sub-tile splitting."""
        partials = []
        for offset in range(0, size, self.dot_tile):
            width = min(self.dot_tile, size - offset)
            mask = jnp.arange(width // 32)[:, None] == jnp.arange(width)[None, :] // 32
            left = jnp.where(
                mask[None],
                activation[:, None, offset : offset + width],
                jnp.bfloat16(0),
            )
            partials.append(
                jax.lax.dot_general(
                    left,
                    values[offset : offset + width],
                    (((2,), (0,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )
            )
        partial = jnp.concatenate(partials, axis=1)
        self.accumulator[...] += jnp.sum(partial * scale[None], axis=1)

    def _accumulate_tile(self, start: _Dynamic, size: _Dynamic) -> None:
        """Execute one tile accumulation."""
        values = self._tile_values(start, size)
        scale = self._tile_scale(start, size)
        activation = self._tile_activation(start, size)

        if self.dot_tile and not self.split_small_contraction and size > self.dot_tile:
            self._dot_tiled_partials(activation, values, scale, size)
            return

        scale_block_mask = (
            jnp.arange(size // 32)[:, None] == jnp.arange(size)[None, :] // 32
        )
        if self.split_small_contraction and activation.shape[0] == 1:
            left = jnp.where(scale_block_mask, activation, jnp.bfloat16(0))
            scaled = (
                jax.lax.dot_general(
                    left,
                    values,
                    (((1,), (0,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )
                * scale
            )
            total = scaled[0:1]
            for block in range(1, size // 32):
                total = total + scaled[block : block + 1]
            self.accumulator[...] += total
        elif self.split_small_contraction:
            left = jnp.where(
                scale_block_mask[:, None, :],
                activation[None, :, :],
                jnp.bfloat16(0),
            )
            partial = jax.lax.dot_general(
                left,
                values,
                (((2,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            self.accumulator[...] += jnp.sum(partial * scale[:, None, :], axis=0)
        else:
            left = jnp.where(
                scale_block_mask[None], activation[:, None, :], jnp.bfloat16(0)
            )
            partial = jax.lax.dot_general(
                left,
                values,
                (((2,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            self.accumulator[...] += jnp.sum(partial * scale[None], axis=1)

    def run(self) -> None:
        """Run quantized expert dot accumulation."""
        if self.static_tiles:
            for start in range(0, self.contraction_size, MXFP4_STATIC_TILE):
                self._accumulate_tile(
                    start, min(MXFP4_STATIC_TILE, self.contraction_size - start)
                )
        elif self.split_small_contraction:
            for start in range(0, self.contraction_size, self.block_size):
                self._accumulate_tile(start, self.block_size)
        else:
            pl.loop(0, self.contraction_size // self.block_size)(
                lambda index: self._accumulate_tile(
                    index * self.block_size, self.block_size
                )
            )
            if self.contraction_size % self.block_size:
                self._accumulate_tile(
                    self.contraction_size // self.block_size * self.block_size,
                    self.contraction_size % self.block_size,
                )


def _quantized_expert_dot(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Multiply by one resident packed MXFP4 matrix."""
    return _QuantizedExpertDotRunner(*_p_args, **_p_kw).run()


def _paired_expert_gate_up(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Run two resident expert gate/up projections.

    With separate compact MXFP4 sources, only the matching row/weight
    products are consumed. Keep cross-expert entries zero while preserving
    each diagonal's scale reduction and tile accumulation order.
    """
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        source,
        gate_up_vmem,
        gate_up_scale_vmem,
        accumulator,
        first_slot,
        second_slot,
        contraction_size,
    ) = (
        _p_next(_p_iter, _p_get("source")),
        _p_next(_p_iter, _p_get("gate_up_vmem")),
        _p_next(_p_iter, _p_get("gate_up_scale_vmem")),
        _p_next(_p_iter, _p_get("accumulator")),
        _p_next(_p_iter, _p_get("first_slot")),
        _p_next(_p_iter, _p_get("second_slot")),
        _p_next(_p_iter, _p_get("contraction_size")),
    )
    (
        second_source,
        active_capacity,
    ) = (
        _p_get("second_source"),
        _p_get("active_capacity"),
    )
    accumulator[...] = jnp.zeros(accumulator.shape, jnp.float32)

    def tile(start: _Dynamic, size: _Dynamic) -> _Dynamic:
        """Execute tile logic."""
        packed = jnp.concatenate(
            (
                gate_up_vmem[first_slot, pl.ds(start // 8, size // 8), :],
                gate_up_vmem[second_slot, pl.ds(start // 8, size // 8), :],
            ),
            axis=1,
        )
        values = tpu.bitcast(packed, jnp.float4_e2m1fn).astype(jnp.float8_e4m3fn)
        exponents = jnp.concatenate(
            (
                gate_up_scale_vmem[first_slot, pl.ds(start // 32, size // 32), :],
                gate_up_scale_vmem[second_slot, pl.ds(start // 32, size // 32), :],
            ),
            axis=1,
        ).astype(jnp.uint32)
        scale = jax.lax.bitcast_convert_type(exponents << 23, jnp.float32)
        activation = source[:, pl.ds(start, size)].astype(jnp.bfloat16)
        if second_source is not None:
            activation = jnp.concatenate(
                (
                    activation,
                    second_source[:, pl.ds(start, size)].astype(jnp.bfloat16),
                ),
                axis=0,
            )
        scale_block_mask = (
            jnp.arange(size // 32)[:, None] == jnp.arange(size)[None, :] // 32
        )
        left = jnp.where(
            scale_block_mask[None], activation[:, None, :], jnp.bfloat16(0)
        )
        if second_source is not None:
            rows = source.shape[0]
            active = rows if active_capacity is None else active_capacity
            columns = values.shape[1] // 2
            for half in range(2):
                partial = jax.lax.dot_general(
                    left[half * rows : half * rows + active],
                    values[:, half * columns : (half + 1) * columns],
                    (((2,), (0,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )
                result = jnp.sum(
                    partial * scale[None, :, half * columns : (half + 1) * columns],
                    axis=1,
                )
                accumulator[
                    half * rows : (half + 1) * rows,
                    half * columns : (half + 1) * columns,
                ] += (
                    result
                    if active_capacity is None
                    else jnp.pad(result, ((0, rows - active), (0, 0)))
                )
        else:
            partial = jax.lax.dot_general(
                left,
                values,
                (((2,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            accumulator[...] += jnp.sum(partial * scale[None], axis=1)

    # Narrow scaled partials (MXFP4_PAIR_TILE) shorten register live ranges.
    tile_size = MXFP4_PAIR_TILE
    for start in range(0, contraction_size, tile_size):
        tile(start, min(tile_size, contraction_size - start))


def _paired_order_expert_gate_up(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Project one expert in the paired gate/up's FP32 accumulation order.

    Pair membership changes when null routes disappear. Tail and overflow
    projections must retain the paired tile's scale reduction and accumulation
    order so an unrelated row cannot change a live expert's rounded result.
    """
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        source,
        packed,
        scales,
        accumulator,
        contraction_size,
        tile_size,
    ) = (
        _p_next(_p_iter, _p_get("source")),
        _p_next(_p_iter, _p_get("packed")),
        _p_next(_p_iter, _p_get("scales")),
        _p_next(_p_iter, _p_get("accumulator")),
        _p_next(_p_iter, _p_get("contraction_size")),
        _p_next(_p_iter, _p_get("tile_size")),
    )
    accumulator[...] = jnp.zeros(accumulator.shape, jnp.float32)
    for start in range(0, contraction_size, tile_size):
        size = min(tile_size, contraction_size - start)
        values = tpu.bitcast(
            packed[pl.ds(start // 8, size // 8), :], jnp.float4_e2m1fn
        ).astype(jnp.float8_e4m3fn)
        exponents = scales[pl.ds(start // 32, size // 32), :].astype(jnp.uint32)
        scale = jax.lax.bitcast_convert_type(exponents << 23, jnp.float32)
        activation = source[:, pl.ds(start, size)].astype(jnp.bfloat16)
        mask = jnp.arange(size // 32)[:, None] == jnp.arange(size)[None, :] // 32
        left = jnp.where(mask[None], activation[:, None, :], jnp.bfloat16(0))
        partial = jax.lax.dot_general(
            left, values, (((2,), (0,)), ((), ())), preferred_element_type=jnp.float32
        )
        accumulator[...] += jnp.sum(partial * scale[None], axis=1)


def _dot_scoped(function: _Dynamic, name_of: _Dynamic) -> _Dynamic:
    """Profiling: trace each routed-expert product as its own e_dot_* phase."""

    @functools.wraps(function)
    def scoped(*args: _Dynamic, **kwargs: _Dynamic) -> _Dynamic:
        """Execute scoped logic."""
        with _phase(name_of(*args, **kwargs)):
            return function(*args, **kwargs)

    return scoped


if PHASE_SCOPES:
    _quantized_expert_dot = _dot_scoped(
        _quantized_expert_dot, lambda *_args, **_kwargs: "e_dot_mxfp4"
    )
    _paired_expert_gate_up = _dot_scoped(
        _paired_expert_gate_up, lambda *_args, **_kwargs: "e_dot_pair"
    )


class _DivergentRoutesRunner:
    """Runner for divergent routed expert stream loop."""

    def __getattr__(self, name: _Dynamic) -> _Dynamic:
        """Dynamic attribute access."""
        raise AttributeError(name)

    def __init__(self, stream: _Dynamic) -> None:
        """Initialize divergent routes runner."""
        self.stream = stream

    def pack(self, choice: _Dynamic) -> _Dynamic:
        """Execute pack logic."""
        return tuple(value.astype(jnp.int32) for value in choice)

    def unpack(self, choice: _Dynamic) -> _Dynamic:
        """Execute unpack logic."""
        return (choice[0], *(value != 0 for value in choice[1:]))

    def prefetch_third(self, third_copies: _Dynamic) -> None:
        """Execute prefetch third logic."""
        _start_async_copies(third_copies)

    def prefetch_fourth(self, fourth_copies: _Dynamic) -> None:
        """Execute prefetch fourth logic."""
        _start_async_copies(fourth_copies)

    def execute_full_pair(self, s: _Dynamic) -> None:
        """Execute full pair logic."""
        (
            source,
            expert_gate_up_vmem,
            expert_gate_up_scale_vmem,
            gate_up_accumulator,
            expert_input_size,
            share_full_finish,
            finish_expert,
            expert_width,
        ) = (
            self.stream.source,
            self.stream.expert_gate_up_vmem,
            self.stream.expert_gate_up_scale_vmem,
            self.stream.gate_up_accumulator,
            self.stream.expert_input_size,
            self.stream.share_full_finish,
            self.stream.finish_expert,
            self.stream.expert_width,
        )
        s.first_copies.gate_up.wait()
        s.first_copies.gate_up_scales.wait()
        s.second_copies.gate_up.wait()
        s.second_copies.gate_up_scales.wait()
        _paired_expert_gate_up(
            source,
            expert_gate_up_vmem,
            expert_gate_up_scale_vmem,
            gate_up_accumulator,
            s.current_base_slot,
            s.current_base_slot + 1,
            expert_input_size,
        )
        if not share_full_finish:
            finish_expert(
                s.first_rows,
                s.first_slot_mask,
                s.first_copies,
                0,
                s.current_base_slot,
            )
            finish_expert(
                s.second_rows,
                s.second_slot_mask,
                s.second_copies,
                2 * expert_width,
                s.current_base_slot + 1,
            )

    def execute_full_tail(self, s: _Dynamic) -> None:
        """Execute full tail logic."""
        (
            source,
            expert_gate_up_vmem,
            expert_gate_up_scale_vmem,
            gate_up_accumulator,
            expert_input_size,
            share_full_finish,
            finish_expert,
            expert_width,
            expert_dot,
        ) = (
            self.stream.source,
            self.stream.expert_gate_up_vmem,
            self.stream.expert_gate_up_scale_vmem,
            self.stream.gate_up_accumulator,
            self.stream.expert_input_size,
            self.stream.share_full_finish,
            self.stream.finish_expert,
            self.stream.expert_width,
            self.stream.expert_dot,
        )
        s.first_copies.gate_up.wait()
        s.first_copies.gate_up_scales.wait()
        expert_dot(
            source,
            expert_gate_up_vmem.at[s.current_base_slot],
            expert_gate_up_scale_vmem.at[s.current_base_slot],
            gate_up_accumulator.at[:, : 2 * expert_width],
            expert_input_size,
            2 * expert_width,
        )
        if not share_full_finish:
            finish_expert(
                s.first_rows,
                s.first_slot_mask,
                s.first_copies,
                0,
                s.current_base_slot,
            )

    def execute_compact_pair(self, s: _Dynamic) -> None:
        """Execute compact pair logic."""
        (
            expert_gate_up_scale_vmem,
            expert_gate_up_vmem,
            expert_input_size,
            expert_width,
            finish_singleton_expert,
            first_single_source,
            gate_up_accumulator,
            second_single_source,
            stage_singleton_source,
        ) = (
            self.stream.expert_gate_up_scale_vmem,
            self.stream.expert_gate_up_vmem,
            self.stream.expert_input_size,
            self.stream.expert_width,
            self.stream.finish_singleton_expert,
            self.stream.first_single_source,
            self.stream.gate_up_accumulator,
            self.stream.second_single_source,
            self.stream.stage_singleton_source,
        )
        s.first_copies.gate_up.wait()
        s.first_copies.gate_up_scales.wait()
        s.second_copies.gate_up.wait()
        s.second_copies.gate_up_scales.wait()
        stage_singleton_source(s.first_rows, first_single_source)
        stage_singleton_source(s.second_rows, second_single_source)
        _paired_expert_gate_up(
            first_single_source,
            expert_gate_up_vmem,
            expert_gate_up_scale_vmem,
            gate_up_accumulator.at[:2],
            s.current_base_slot,
            s.current_base_slot + 1,
            expert_input_size,
            second_source=second_single_source,
        )
        finish_singleton_expert(
            s.first_rows,
            s.first_slot_mask,
            s.first_copies,
            0,
            0,
            s.current_base_slot,
        )
        finish_singleton_expert(
            s.second_rows,
            s.second_slot_mask,
            s.second_copies,
            1,
            2 * expert_width,
            s.current_base_slot + 1,
        )

    def execute_compact_multi_pair(self, s: _Dynamic) -> None:
        """Execute compact multi pair logic."""
        (
            compact_capacity,
            expert_gate_up_scale_vmem,
            expert_gate_up_vmem,
            expert_input_size,
            expert_width,
            finish_compact_expert,
            first_compact_source_vmem,
            gate_up_accumulator,
            second_compact_source_vmem,
            share_compact_finish,
            stage_compact_rows,
        ) = (
            self.stream.compact_capacity,
            self.stream.expert_gate_up_scale_vmem,
            self.stream.expert_gate_up_vmem,
            self.stream.expert_input_size,
            self.stream.expert_width,
            self.stream.finish_compact_expert,
            self.stream.first_compact_source_vmem,
            self.stream.gate_up_accumulator,
            self.stream.second_compact_source_vmem,
            self.stream.share_compact_finish,
            self.stream.stage_compact_rows,
        )
        s.first_copies.gate_up.wait()
        s.first_copies.gate_up_scales.wait()
        s.second_copies.gate_up.wait()
        s.second_copies.gate_up_scales.wait()
        stage_compact_rows(
            s.first_rows,
            first_compact_source_vmem,
            row_plan=s.first_plan,
        )
        stage_compact_rows(
            s.second_rows,
            second_compact_source_vmem,
            row_plan=s.second_plan,
        )

        def project_pair(capacity: _Dynamic = None) -> None:
            """Execute project pair logic."""
            _paired_expert_gate_up(
                first_compact_source_vmem,
                expert_gate_up_vmem,
                expert_gate_up_scale_vmem,
                gate_up_accumulator.at[: 2 * compact_capacity],
                s.current_base_slot,
                s.current_base_slot + 1,
                expert_input_size,
                second_source=second_compact_source_vmem,
                active_capacity=capacity,
            )

        if share_compact_finish:
            largest = jnp.maximum(s.first_row_count, s.second_row_count)
            previous = 1
            for capacity in (2, 4, 8):

                @pl.when((largest > previous) & (largest <= capacity))
                @_scoped(f"e_capacity_{capacity}")
                def execute_capacity(capacity: _Dynamic = capacity) -> None:
                    """Execute execute capacity logic."""
                    project_pair(capacity)

                previous = capacity
        else:
            project_pair()
            finish_compact_expert(
                s.first_rows,
                s.first_slot_mask,
                s.first_copies,
                0,
                0,
                s.current_base_slot,
            )
            finish_compact_expert(
                s.second_rows,
                s.second_slot_mask,
                s.second_copies,
                compact_capacity,
                2 * expert_width,
                s.current_base_slot + 1,
            )

    def execute_noncompact_pair(self, s: _Dynamic) -> None:
        """Execute noncompact pair logic."""
        (
            compact_only,
            expert_width,
            run_expert_in_passes,
            selected_probabilities,
        ) = (
            self.stream.compact_only,
            self.stream.expert_width,
            self.stream.run_expert_in_passes,
            self.stream.selected_probabilities,
        )
        if compact_only:
            s.first_copies.gate_up.wait()
            s.first_copies.gate_up_scales.wait()
            s.second_copies.gate_up.wait()
            s.second_copies.gate_up_scales.wait()
            run_expert_in_passes(
                s.first_rows,
                s.first_slot_mask,
                s.first_copies,
                selected_probabilities,
                0,
                s.current_base_slot,
            )
            run_expert_in_passes(
                s.second_rows,
                s.second_slot_mask,
                s.second_copies,
                selected_probabilities,
                2 * expert_width,
                s.current_base_slot + 1,
            )
        else:
            self.execute_full_pair(s)

    def execute_compact_tail(self, s: _Dynamic) -> None:
        """Execute compact tail logic."""
        (
            expert_dot,
            expert_gate_up_scale_vmem,
            expert_gate_up_vmem,
            expert_input_size,
            expert_width,
            finish_singleton_expert,
            first_single_source,
            gate_up_accumulator,
            stage_singleton_source,
        ) = (
            self.stream.expert_dot,
            self.stream.expert_gate_up_scale_vmem,
            self.stream.expert_gate_up_vmem,
            self.stream.expert_input_size,
            self.stream.expert_width,
            self.stream.finish_singleton_expert,
            self.stream.first_single_source,
            self.stream.gate_up_accumulator,
            self.stream.stage_singleton_source,
        )
        s.first_copies.gate_up.wait()
        s.first_copies.gate_up_scales.wait()
        stage_singleton_source(s.first_rows, first_single_source)
        expert_dot(
            first_single_source,
            expert_gate_up_vmem.at[s.current_base_slot],
            expert_gate_up_scale_vmem.at[s.current_base_slot],
            gate_up_accumulator.at[:1, : 2 * expert_width],
            expert_input_size,
            2 * expert_width,
        )
        finish_singleton_expert(
            s.first_rows,
            s.first_slot_mask,
            s.first_copies,
            0,
            0,
            s.current_base_slot,
        )

    def execute_compact_multi_tail(self, s: _Dynamic) -> None:
        """Execute compact multi tail logic."""
        (
            compact_capacity,
            expert_dot,
            expert_gate_up_scale_vmem,
            expert_gate_up_vmem,
            expert_input_size,
            expert_width,
            finish_compact_expert,
            first_compact_source_vmem,
            gate_up_accumulator,
            share_compact_finish,
            stage_compact_rows,
        ) = (
            self.stream.compact_capacity,
            self.stream.expert_dot,
            self.stream.expert_gate_up_scale_vmem,
            self.stream.expert_gate_up_vmem,
            self.stream.expert_input_size,
            self.stream.expert_width,
            self.stream.finish_compact_expert,
            self.stream.first_compact_source_vmem,
            self.stream.gate_up_accumulator,
            self.stream.share_compact_finish,
            self.stream.stage_compact_rows,
        )
        s.first_copies.gate_up.wait()
        s.first_copies.gate_up_scales.wait()
        stage_compact_rows(
            s.first_rows,
            first_compact_source_vmem,
            row_plan=s.first_plan,
        )
        expert_dot(
            first_compact_source_vmem,
            expert_gate_up_vmem.at[s.current_base_slot],
            expert_gate_up_scale_vmem.at[s.current_base_slot],
            gate_up_accumulator.at[:compact_capacity, : 2 * expert_width],
            expert_input_size,
            2 * expert_width,
            active_rows=s.first_row_count if share_compact_finish else None,
        )
        if not share_compact_finish:
            finish_compact_expert(
                s.first_rows,
                s.first_slot_mask,
                s.first_copies,
                0,
                0,
                s.current_base_slot,
            )

    def execute_noncompact_tail(self, s: _Dynamic) -> None:
        """Execute noncompact tail logic."""
        (
            compact_only,
            run_expert_in_passes,
            selected_probabilities,
        ) = (
            self.stream.compact_only,
            self.stream.run_expert_in_passes,
            self.stream.selected_probabilities,
        )
        if compact_only:
            s.first_copies.gate_up.wait()
            s.first_copies.gate_up_scales.wait()
            run_expert_in_passes(
                s.first_rows,
                s.first_slot_mask,
                s.first_copies,
                selected_probabilities,
                0,
                s.current_base_slot,
            )
        else:
            self.execute_full_tail(s)

    def execute_full_finish(self, s: _Dynamic) -> None:
        """Execute full finish logic."""
        finish_expert, selected_probabilities = (
            self.stream.finish_expert,
            self.stream.selected_probabilities,
        )

        def route_weights(route_slot: _Dynamic) -> _Dynamic:
            """Execute route weights logic."""
            return jnp.where(
                (s.first_slot_mask == route_slot)[:, None],
                selected_probabilities[:, route_slot : route_slot + 1],
                selected_probabilities[:, route_slot + 1 : route_slot + 2],
            )

        def selected_expert(index: _Dynamic) -> _Dynamic:
            """Execute selected expert logic."""
            return jnp.where(index == 0, s.first_expert, s.second_expert)

        finish_expert(
            s.first_rows | s.second_rows,
            route_weights,
            (s.first_copies, s.second_copies),
            0,
            s.current_base_slot,
            selected_expert=selected_expert,
        )

    def execute_compact_finish(self, s: _Dynamic) -> None:
        """Execute compact finish logic."""
        compact_route_data, finish_compact_expert = (
            self.stream.compact_route_data,
            self.stream.finish_compact_expert,
        )
        active_rows = s.first_rows | s.second_rows
        route_weights, expert_indices = compact_route_data(
            s.first_rows,
            s.first_slot_mask,
            offset=0,
            row_plan=s.first_plan,
        )

        def finish_selected() -> None:
            """Execute finish selected logic."""
            finish_compact_expert(
                active_rows,
                route_weights,
                (s.first_copies, s.second_copies),
                0,
                0,
                s.current_base_slot,
                expert_indices=expert_indices,
            )

        finish_selected()

    def execute_pair(self, s: _Dynamic) -> None:
        """Execute pair logic."""
        self.execute_full_pair(s)

    def execute_tail(self, s: _Dynamic) -> None:
        """Execute tail logic."""
        self.execute_full_tail(s)

    def _prefetch(
        self,
        iteration: _Dynamic,
        has_third: _Dynamic,
        has_fourth: _Dynamic,
        third_copies: _Dynamic,
        fourth_copies: _Dynamic,
    ) -> None:
        """Execute prefetch logic."""
        should_prefetch_third = has_third & (iteration != 0)
        should_prefetch_fourth = has_fourth & (iteration != 0)

        @pl.when(should_prefetch_third)
        def prefetch_third() -> None:
            """Execute prefetch third logic."""
            self.prefetch_third(third_copies)

        @pl.when(should_prefetch_fourth)
        def prefetch_fourth() -> None:
            """Execute prefetch fourth logic."""
            self.prefetch_fourth(fourth_copies)

    def _dispatch_small(
        self, s: _Dynamic, has_first: _Dynamic, has_second: _Dynamic
    ) -> None:
        """Execute dispatch small logic."""

        @pl.when(has_first & has_second)
        def run_pair() -> None:
            """Execute run pair logic."""
            self.execute_pair(s)

        @pl.when(has_first & ~has_second)
        def run_tail() -> None:
            """Execute run tail logic."""
            self.execute_tail(s)

    def _dispatch_batched_pair(
        self, s: _Dynamic, has_first: _Dynamic, has_second: _Dynamic
    ) -> None:
        """Execute dispatch batched pair logic."""
        compact_capacity = self.stream.compact_capacity
        compact_pair = (
            has_first
            & has_second
            & (s.first_row_count == 1)
            & (s.second_row_count == 1)
        )
        s.compact_pair = compact_pair

        @pl.when(compact_pair)
        @_scoped("e_single_pair")
        def run_compact_pair() -> None:
            """Execute run compact pair logic."""
            self.execute_compact_pair(s)

        if compact_capacity > 1:
            multi_pair = (
                has_first
                & has_second
                & ~compact_pair
                & (s.first_row_count <= compact_capacity)
                & (s.second_row_count <= compact_capacity)
            )

            @pl.when(multi_pair)
            @_scoped("e_multi_pair")
            def run_compact_multi_pair() -> None:
                """Execute run compact multi pair logic."""
                self.execute_compact_multi_pair(s)
        else:
            multi_pair = jnp.bool_(bool(0))
        s.multi_pair = multi_pair

        @pl.when(has_first & has_second & ~compact_pair & ~multi_pair)
        @_scoped("e_pass_pair")
        def run_noncompact_pair() -> None:
            """Execute run noncompact pair logic."""
            self.execute_noncompact_pair(s)

    def _dispatch_batched_tail(
        self, s: _Dynamic, has_first: _Dynamic, has_second: _Dynamic
    ) -> None:
        """Execute dispatch batched tail logic."""
        compact_capacity = self.stream.compact_capacity
        compact_tail = has_first & ~has_second & (s.first_row_count == 1)
        s.compact_tail = compact_tail

        @pl.when(compact_tail)
        @_scoped("e_single_tail")
        def run_compact_tail() -> None:
            """Execute run compact tail logic."""
            self.execute_compact_tail(s)

        if compact_capacity > 1:
            multi_tail = (
                has_first
                & ~has_second
                & ~compact_tail
                & (s.first_row_count <= compact_capacity)
            )

            @pl.when(multi_tail)
            @_scoped("e_multi_tail")
            def run_compact_multi_tail() -> None:
                """Execute run compact multi tail logic."""
                self.execute_compact_multi_tail(s)
        else:
            multi_tail = jnp.bool_(bool(0))
        s.multi_tail = multi_tail

        @pl.when(has_first & ~has_second & ~compact_tail & ~multi_tail)
        @_scoped("e_pass_tail")
        def run_noncompact_tail() -> None:
            """Execute run noncompact tail logic."""
            self.execute_noncompact_tail(s)

    def _dispatch_finish(
        self, s: _Dynamic, has_first: _Dynamic, has_second: _Dynamic
    ) -> None:
        """Execute dispatch finish logic."""
        share_compact_finish, share_full_finish = (
            self.stream.share_compact_finish,
            self.stream.share_full_finish,
        )
        if share_full_finish:

            @pl.when(has_first & has_second & ~s.compact_pair & ~s.multi_pair)
            @_scoped("e_pair_finish")
            def run_full_finish() -> None:
                """Execute run full finish logic."""
                self.execute_full_finish(s)

        if share_compact_finish:

            @pl.when(
                (s.compact_pair | s.multi_pair)
                | ((s.compact_tail | s.multi_tail) & (s.first_row_count > 1))
            )
            @_scoped("e_compact_finish")
            def run_compact_finish() -> None:
                """Execute run compact finish logic."""
                self.execute_compact_finish(s)

    def run_next_expert_pair(
        self, iteration: _Dynamic, carry: _Dynamic = None
    ) -> _Dynamic:
        """Execute run next expert pair logic."""
        (
            batch_size,
            carry_routes,
            compact_row_plan,
            copy_expert,
            next_expert,
            processed_routes_vmem,
            progress,
            reuse_row_plan,
            selected_expert_count,
        ) = (
            self.stream.batch_size,
            self.stream.carry_routes,
            self.stream.compact_row_plan,
            self.stream.copy_expert,
            self.stream.next_expert,
            self.stream.processed_routes_vmem,
            self.stream.progress,
            self.stream.reuse_row_plan,
            self.stream.selected_expert_count,
        )
        if progress is not None:
            progress(iteration)
        with _phase("e_iteration"):
            select_scope = _phase_start("e_select")
            processed = processed_routes_vmem[:, :selected_expert_count] != 0
            (
                first_expert,
                first_rows,
                first_slot_mask,
                first_consumed,
                has_first,
            ) = self.unpack(carry[0]) if carry_routes else next_expert(processed)
            (
                second_expert,
                second_rows,
                second_slot_mask,
                second_consumed,
                has_second,
            ) = (
                self.unpack(carry[1])
                if carry_routes
                else next_expert(processed | first_consumed)
            )
            processed_after_pair = processed | first_consumed | second_consumed
            (
                third_expert,
                third_rows,
                third_slot_mask,
                third_consumed,
                has_third,
            ) = next_expert(processed_after_pair)
            (
                fourth_expert,
                fourth_rows,
                fourth_slot_mask,
                fourth_consumed,
                has_fourth,
            ) = next_expert(processed_after_pair | third_consumed)

            current_base_slot = (iteration % 2) * 2
            next_base_slot = ((iteration + 1) % 2) * 2
            first_copies = copy_expert(first_expert, current_base_slot)
            second_copies = copy_expert(second_expert, current_base_slot + 1)
            third_copies = copy_expert(third_expert, next_base_slot)
            fourth_copies = copy_expert(fourth_expert, next_base_slot + 1)

            self._prefetch(
                iteration,
                has_third,
                has_fourth,
                third_copies,
                fourth_copies,
            )
            _phase_end(select_scope)

            class _PairState:
                """Pair state."""

                def __init__(self) -> None:
                    """Initialize pair state."""
                    self.first_copies = first_copies
                    self.second_copies = second_copies
                    self.first_rows = first_rows
                    self.second_rows = second_rows
                    self.first_slot_mask = first_slot_mask
                    self.second_slot_mask = second_slot_mask
                    self.first_expert = first_expert
                    self.second_expert = second_expert
                    self.current_base_slot = current_base_slot
                    self.first_plan = None
                    self.second_plan = None
                    self.first_row_count = None
                    self.second_row_count = None
                    self.compact_pair = None
                    self.multi_pair = None
                    self.compact_tail = None
                    self.multi_tail = None

            s = _PairState()

            if batch_size >= _CONST_8:
                s.first_row_count = jnp.sum(first_rows.astype(jnp.int32))
                s.second_row_count = jnp.sum(second_rows.astype(jnp.int32))
                s.first_plan = compact_row_plan(first_rows) if reuse_row_plan else None
                s.second_plan = (
                    compact_row_plan(second_rows) if reuse_row_plan else None
                )
                self._dispatch_batched_pair(s, has_first, has_second)
                self._dispatch_batched_tail(s, has_first, has_second)
                self._dispatch_finish(s, has_first, has_second)
            else:
                self._dispatch_small(s, has_first, has_second)

            if carry_routes:
                return (
                    self.pack(
                        (
                            third_expert,
                            third_rows,
                            third_slot_mask,
                            third_consumed,
                            has_third,
                        )
                    ),
                    self.pack(
                        (
                            fourth_expert,
                            fourth_rows,
                            fourth_slot_mask,
                            fourth_consumed,
                            has_fourth,
                        )
                    ),
                )
            return None

    def run(self) -> None:
        """Execute run logic."""
        (
            carry_routes,
            next_expert,
            pair_count,
            selected_experts,
        ) = (
            self.stream.carry_routes,
            self.stream.next_expert,
            self.stream.pair_count,
            self.stream.selected_experts,
        )
        runner = self
        initial_first = initial_second = None
        if carry_routes:
            initial_processed = jnp.zeros(selected_experts.shape, jnp.bool_)
            initial_first = next_expert(initial_processed)
            initial_second = next_expert(initial_processed | initial_first[3])
            jax.lax.fori_loop(
                0,
                pair_count,
                runner.run_next_expert_pair,
                (runner.pack(initial_first), runner.pack(initial_second)),
            )
        else:

            @pl.loop(0, pair_count)
            def run_original_expert_pair(iteration: _Dynamic) -> None:
                """Execute run original expert pair logic."""
                runner.run_next_expert_pair(iteration)


class _RoutedExpertStreamRunner:
    """Runner for routed expert stream execution."""

    def __getattr__(self, name: _Dynamic) -> _Dynamic:
        """Dynamic attribute access."""
        raise AttributeError(name)

    def __init__(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Initialize routed expert stream runner."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            self.source,
            self.selected_experts,
            self.selected_probabilities,
            self.local_route_counts,
            self.probability_totals,
            self.expert_gate_up_vmem,
            self.expert_gate_up_scale_vmem,
            self.expert_down_vmem,
            self.expert_down_scale_vmem,
            self.expert_output,
            self.gate_up_accumulator,
            self.down_accumulator,
            self.processed_routes_vmem,
            self.first_compact_source_vmem,
            self.second_compact_source_vmem,
            self.copy_expert,
        ) = (
            _p_next(_p_iter, _p_get("source")),
            _p_next(_p_iter, _p_get("selected_experts")),
            _p_next(_p_iter, _p_get("selected_probabilities")),
            _p_next(_p_iter, _p_get("local_route_counts")),
            _p_next(_p_iter, _p_get("probability_totals")),
            _p_next(_p_iter, _p_get("expert_gate_up_vmem")),
            _p_next(_p_iter, _p_get("expert_gate_up_scale_vmem")),
            _p_next(_p_iter, _p_get("expert_down_vmem")),
            _p_next(_p_iter, _p_get("expert_down_scale_vmem")),
            _p_next(_p_iter, _p_get("expert_output")),
            _p_next(_p_iter, _p_get("gate_up_accumulator")),
            _p_next(_p_iter, _p_get("down_accumulator")),
            _p_next(_p_iter, _p_get("processed_routes_vmem")),
            _p_next(_p_iter, _p_get("first_compact_source_vmem")),
            _p_next(_p_iter, _p_get("second_compact_source_vmem")),
            _p_next(_p_iter, _p_get("copy_expert")),
        )
        (
            self.expert_input_size,
            self.expert_width,
            self.expert_output_size,
            self.group_pending_routes,
            self.progress,
        ) = (
            _p_get("expert_input_size"),
            _p_get("expert_width"),
            _p_get("expert_output_size"),
            _p_get("group_pending_routes"),
            _p_get("progress"),
        )
        self.batch_size, self.selected_expert_count = self.selected_experts.shape
        self.dot_tile = (
            MXFP4_DOT_TILE
            if MXFP4_DOT_TILE is not None
            else 1024
            if self.batch_size == _CONST_96
            else 512
            if self.batch_size in (1, 2, 4, 8, 16, 32, 64, 128)
            else 0
        )
        self.singleton_block = (
            SINGLETON_BLOCK
            if SINGLETON_BLOCK is not None
            else self.batch_size in (16, 32, 64, 96, 128)
        )
        self.stage_blocks = (
            STAGE_BLOCKS
            if STAGE_BLOCKS is not None
            else self.batch_size in (16, 32, 64, 96, 128)
        )
        if PHASE_SCOPES:
            self.expert_dot = _dot_scoped(
                self.expert_dot, lambda *_args, **_kwargs: "e_dot"
            )
            unscoped_copy_expert = self.copy_expert

            def copy_expert_scoped(
                expert: _Dynamic, slot: _Dynamic, **options: _Dynamic
            ) -> _Dynamic:
                """Execute copy expert scoped logic."""
                copies = unscoped_copy_expert(expert, slot, **options)
                return SimpleNamespace(
                    **{
                        name: None if copy is None else _ScopedWait(copy)
                        for name, copy in vars(copies).items()
                    }
                )

            self.copy_expert = copy_expert_scoped

        self.route_slots = jnp.arange(self.selected_expert_count)[None, :]
        self.processed_routes_vmem[...] = jnp.zeros(
            self.processed_routes_vmem.shape, jnp.int32
        )
        self.expert_output[...] = jnp.zeros(self.expert_output.shape, jnp.float32)

        self.compact_capacity = self.first_compact_source_vmem.shape[0]
        if self.compact_capacity > 1:
            self.first_single_source = self.first_compact_source_vmem.at[:1]
            self.second_single_source = self.second_compact_source_vmem.at[:1]
        else:
            self.first_single_source = self.first_compact_source_vmem
            self.second_single_source = self.second_compact_source_vmem

        self.reuse_row_plan = self.stage_blocks and self.batch_size in (32, 64, 96, 128)
        self.compact_overflow = _use_compact_overflow(
            self.batch_size, self.group_pending_routes
        )
        self.compact_only = (
            self.gate_up_accumulator.shape[0] < self.batch_size or self.compact_overflow
        )
        self.share_compact_finish = (
            SHARED_EXPERT_FINISHERS
            and self.batch_size in (16, 32, 64, 96, 128)
            and not self.compact_only
        )
        self.share_full_finish = self.share_compact_finish and (not self.compact_only)

        self.valid_routes = self.route_slots < self.local_route_counts[:, None]
        self.routes_are_shared = (
            (not self.group_pending_routes)
            & jnp.all(self.local_route_counts == self.local_route_counts[0])
            & jnp.all(
                ~self.valid_routes
                | (self.selected_experts == self.selected_experts[0:1])
            )
        )

        if self.group_pending_routes:
            used = jnp.zeros((1, 128), jnp.bool_)
            for route_slot in range(self.selected_expert_count):
                used |= jnp.any(
                    self.valid_routes[:, route_slot : route_slot + 1]
                    & (
                        self.selected_experts[:, route_slot : route_slot + 1]
                        == jnp.arange(128)[None, :]
                    ),
                    axis=0,
                    keepdims=True,
                )
            self.pair_count = (jnp.sum(used.astype(jnp.int32)) + 1) // 2
        else:
            self.pair_count = (jnp.sum(self.local_route_counts) + 1) // 2

        self.carry_routes = (
            self.batch_size in (2, 4) and not self.group_pending_routes
        ) or (self.batch_size in (8, 16, 32, 64, 96, 128) and self.group_pending_routes)

    def expert_dot(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
        """Execute expert dot logic."""
        (
            batch_size,
            dot_tile,
            expert_input_size,
        ) = (
            self.batch_size,
            self.dot_tile,
            self.expert_input_size,
        )
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            source,
            packed,
            scales,
            accumulator,
            contraction_size,
            output_size,
        ) = (
            _p_next(_p_iter, _p_get("source")),
            _p_next(_p_iter, _p_get("packed")),
            _p_next(_p_iter, _p_get("scales")),
            _p_next(_p_iter, _p_get("accumulator")),
            _p_next(_p_iter, _p_get("contraction_size")),
            _p_next(_p_iter, _p_get("output_size")),
        )
        options = _p_kw
        if batch_size > 1 and contraction_size == expert_input_size:
            # A live expert may switch between a pair and the unpaired tail
            # as other rows enter or leave. Match the paired gate's scale
            # reduction tree, including its final partial tile.
            return _paired_order_expert_gate_up(
                source, packed, scales, accumulator, contraction_size, MXFP4_PAIR_TILE
            )
        return _quantized_expert_dot(
            source,
            packed,
            scales,
            accumulator,
            contraction_size,
            output_size,
            dot_tile=dot_tile,
            **options,
        )

    def next_expert(self, processed: _Dynamic) -> _Dynamic:
        """Execute next expert logic."""
        (
            group_pending_routes,
            local_route_counts,
            route_slots,
            selected_experts,
        ) = (
            self.group_pending_routes,
            self.local_route_counts,
            self.route_slots,
            self.selected_experts,
        )
        pending = (route_slots < local_route_counts[:, None]) & ~processed
        if group_pending_routes:
            minimum_expert = jnp.min(
                jnp.where(
                    pending,
                    selected_experts.astype(jnp.float32),
                    jnp.inf,
                )
            )
            has_expert = jnp.any(pending)
            expert = jnp.where(has_expert, minimum_expert, 0).astype(jnp.int32)
            selected_slot_mask = pending & (selected_experts == expert)
            active_rows = jnp.any(selected_slot_mask, axis=1)
            return (
                expert,
                active_rows,
                selected_slot_mask,
                selected_slot_mask,
                has_expert,
            )
        rows_with_routes = jnp.any(pending, axis=1)
        current_slots = jnp.sum(processed.astype(jnp.int32), axis=1)
        current_slot_mask = route_slots == current_slots[:, None]
        current_experts = jnp.sum(
            jnp.where(current_slot_mask, selected_experts, 0), axis=1
        )
        minimum_expert = jnp.min(
            jnp.where(
                rows_with_routes,
                current_experts.astype(jnp.float32),
                jnp.inf,
            )
        )
        has_expert = jnp.any(rows_with_routes)
        expert = jnp.where(has_expert, minimum_expert, 0).astype(jnp.int32)
        active_rows = rows_with_routes & (current_experts == expert)
        consumed = active_rows[:, None] & current_slot_mask
        return expert, active_rows, current_slot_mask, consumed, has_expert

    def finish_expert(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
        """Execute finish expert logic."""
        (
            batch_size,
            down_accumulator,
            expert_dot,
            expert_down_scale_vmem,
            expert_down_vmem,
            expert_output,
            expert_output_size,
            expert_width,
            gate_up_accumulator,
            probability_totals,
            selected_probabilities,
            share_full_finish,
        ) = (
            self.batch_size,
            self.down_accumulator,
            self.expert_dot,
            self.expert_down_scale_vmem,
            self.expert_down_vmem,
            self.expert_output,
            self.expert_output_size,
            self.expert_width,
            self.gate_up_accumulator,
            self.probability_totals,
            self.selected_probabilities,
            self.share_full_finish,
        )
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            active_rows,
            current_slot_mask,
            copies,
            column,
            slot,
            output_weights,
        ) = (
            _p_next(_p_iter, _p_get("active_rows")),
            _p_next(_p_iter, _p_get("current_slot_mask")),
            _p_next(_p_iter, _p_get("copies")),
            _p_next(_p_iter, _p_get("column")),
            _p_next(_p_iter, _p_get("slot")),
            _p_next(_p_iter, _p_get("output_weights")),
        )
        gate_up = (
            gate_up_accumulator[
                :,
                (
                    pl.ds(pl.multiple_of(column, 128), 2 * expert_width)
                    if share_full_finish
                    else slice(column, column + 2 * expert_width)
                ),
            ]
            .astype(jnp.bfloat16)
            .astype(jnp.float32)
        )
        middle = _gated_activation(gate_up[:, :expert_width], gate_up[:, expert_width:])
        gate_up_accumulator[:, :expert_width] = middle.astype(jnp.float32)
        copies.down.wait()
        copies.down_scales.wait()
        expert_dot(
            gate_up_accumulator.at[:, :expert_width],
            expert_down_vmem.at[slot],
            expert_down_scale_vmem.at[slot],
            down_accumulator,
            expert_width,
            expert_output_size,
            # Keep the down projection's scale reductions independent of
            # how many rows happen to route to this expert.
            block_major=batch_size >= _CONST_8,
        )
        if output_weights is None:
            route_probabilities = (
                jnp.sum(
                    jnp.where(
                        current_slot_mask,
                        selected_probabilities,
                        jnp.float32(0),
                    ),
                    axis=1,
                )
                / probability_totals
            )
            output_weights = jnp.where(active_rows, route_probabilities, jnp.float32(0))
        expert_output[:, :expert_output_size] += (
            down_accumulator[...].astype(jnp.bfloat16).astype(jnp.float32)
            * output_weights[:, None]
        )

    def finish_singleton_expert(
        self,
        *_p_args: _Dynamic,
        **_p_kw: _Dynamic,
    ) -> _Dynamic:
        """Finish one expert whose route is active for exactly one batch row."""
        (
            batch_size,
            down_accumulator,
            expert_dot,
            expert_down_scale_vmem,
            expert_down_vmem,
            expert_output,
            expert_output_size,
            expert_width,
            gate_up_accumulator,
            probability_totals,
            selected_probabilities,
        ) = (
            self.batch_size,
            self.down_accumulator,
            self.expert_dot,
            self.expert_down_scale_vmem,
            self.expert_down_vmem,
            self.expert_output,
            self.expert_output_size,
            self.expert_width,
            self.gate_up_accumulator,
            self.probability_totals,
            self.selected_probabilities,
        )
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            active_rows,
            current_slot_mask,
            copies,
            compact_row,
            column,
            slot,
        ) = (
            _p_next(_p_iter, _p_get("active_rows")),
            _p_next(_p_iter, _p_get("current_slot_mask")),
            _p_next(_p_iter, _p_get("copies")),
            _p_next(_p_iter, _p_get("compact_row")),
            _p_next(_p_iter, _p_get("column")),
            _p_next(_p_iter, _p_get("slot")),
        )
        gate_up = (
            gate_up_accumulator[
                compact_row : compact_row + 1,
                column : column + 2 * expert_width,
            ]
            .astype(jnp.bfloat16)
            .astype(jnp.float32)
        )
        middle = _gated_activation(gate_up[:, :expert_width], gate_up[:, expert_width:])
        gate_up_accumulator[compact_row : compact_row + 1, :expert_width] = (
            middle.astype(jnp.float32)
        )
        copies.down.wait()
        copies.down_scales.wait()
        expert_dot(
            gate_up_accumulator.at[compact_row : compact_row + 1, :expert_width],
            expert_down_vmem.at[slot],
            expert_down_scale_vmem.at[slot],
            down_accumulator.at[:1],
            expert_width,
            expert_output_size,
            # Use the 256-wide block-major accumulation of the compact/full
            # paths so a singleton row rounds the same as in a group.
            block_major=True,
        )
        route_probabilities = (
            jnp.sum(
                jnp.where(
                    current_slot_mask,
                    selected_probabilities,
                    jnp.float32(0),
                ),
                axis=1,
            )
            / probability_totals
        )
        singleton_output = down_accumulator[:1].astype(jnp.bfloat16).astype(jnp.float32)
        row = jnp.sum(jnp.where(active_rows, jnp.arange(batch_size), 0))
        probability = jnp.sum(
            jnp.where(active_rows, route_probabilities, jnp.float32(0))
        )
        output_row = expert_output.at[pl.ds(row, 1), :expert_output_size]
        output_row[...] += singleton_output * probability

    def stage_singleton_source(
        self,
        active_rows: _Dynamic,
        destination: _Dynamic,
    ) -> _Dynamic:
        """Execute stage singleton source logic."""
        (
            batch_size,
            expert_input_size,
            singleton_block,
            source,
        ) = (
            self.batch_size,
            self.expert_input_size,
            self.singleton_block,
            self.source,
        )
        if singleton_block and batch_size % 16 == 0:
            active_index = jnp.sum(jnp.where(active_rows, jnp.arange(batch_size), 0))
            block = source[
                pl.ds(pl.multiple_of(active_index // 16 * 16, 16), 16),
                :expert_input_size,
            ].astype(jnp.float32)
            within = (
                jax.lax.broadcasted_iota(jnp.int32, block.shape, 0) == active_index % 16
            )
            destination[...] = jnp.sum(
                jnp.where(within, block, 0.0), axis=0, keepdims=True
            ).astype(destination.dtype)
            return
        active_index = jnp.sum(jnp.where(active_rows, jnp.arange(batch_size), 0))
        for row in range(batch_size):

            @pl.when(active_index == row)
            def copy_indexed_row(row: _Dynamic = row) -> _Dynamic:
                """Execute copy indexed row logic."""
                destination[...] = source[row : row + 1, :expert_input_size].astype(
                    destination.dtype
                )

    def row_bitmask_words(self, active_rows: _Dynamic) -> _Dynamic:
        """The active rows as scalar bit words (32 rows per word)."""
        batch_size = self.batch_size
        index = jnp.arange(batch_size)
        bits = active_rows.astype(jnp.int32) << (index % 32)
        return [
            jnp.sum(jnp.where(index // 32 == word, bits, 0))
            for word in range(-(-batch_size // 32))
        ]

    def compact_row_positions(self, active_rows: _Dynamic) -> _Dynamic:
        # Counts and row indices are < 128, hence exact in FP32. This avoids
        # the split reductions used for unrestricted int32.
        """Execute compact row positions logic."""
        batch_size, reuse_row_plan = self.batch_size, self.reuse_row_plan
        count_dtype = jnp.float32 if reuse_row_plan else jnp.int32
        index = jnp.arange(batch_size)
        return jnp.sum(
            jnp.where(
                index[None, :] < index[:, None],
                active_rows[None, :].astype(count_dtype),
                0,
            ),
            axis=1,
        ).astype(jnp.int32)

    def compact_row_plan(
        self,
        active_rows: _Dynamic,
        offset: _Dynamic = 0,
        before: _Dynamic = None,
    ) -> _Dynamic:
        """Execute compact row plan logic."""
        (
            batch_size,
            compact_capacity,
            compact_row_positions,
        ) = (
            self.batch_size,
            self.compact_capacity,
            self.compact_row_positions,
        )
        if before is None:
            before = compact_row_positions(active_rows)
        # One ordered index list serves both input staging and output
        # selection. Invalid trailing entries must not select source row 0.
        selected = active_rows[None, :] & (
            before[None, :] == offset + jnp.arange(compact_capacity)[:, None]
        )
        indices = jnp.sum(
            jnp.where(
                selected, jnp.arange(batch_size).astype(jnp.float32)[None, :], 0.0
            ),
            axis=1,
        ).astype(jnp.int32)
        return indices, jnp.any(selected, axis=1)

    def stage_planned_rows(self, row_plan: _Dynamic, destination: _Dynamic) -> _Dynamic:
        """Execute stage planned rows logic."""
        (
            batch_size,
            compact_capacity,
            expert_input_size,
            source,
        ) = (
            self.batch_size,
            self.compact_capacity,
            self.expert_input_size,
            self.source,
        )
        indices, valid = row_plan
        if batch_size in (64, 96, 128):
            # A one-hot product selects the planned rows (exact: one BF16
            # term per entry, zero rows past the plan) without a scalar
            # read and a gather per row.
            one_hot = (
                (indices[:, None] == jnp.arange(batch_size)[None, :]) & valid[:, None]
            ).astype(jnp.bfloat16)
            destination[...] = _dot(one_hot, source[:, :expert_input_size])
            return
        destination[...] = jnp.zeros(destination.shape, destination.dtype)

        if batch_size == _CONST_32:

            @pl.loop(0, batch_size // 16)
            def stage_block(block: _Dynamic) -> _Dynamic:
                """Execute stage block logic."""
                in_block = valid & (indices // 16 == block)

                @pl.when(jnp.any(in_block))
                def copy_block() -> _Dynamic:
                    """Execute copy block logic."""
                    rows = source[
                        pl.ds(pl.multiple_of(block * 16, 16), 16), :expert_input_size
                    ]
                    gather_indices = jnp.broadcast_to(
                        (indices % 16).astype(jnp.int16)[:, None],
                        (compact_capacity, expert_input_size),
                    )
                    selected_rows = jnp.take_along_axis(
                        rows,
                        gather_indices,
                        axis=0,
                        mode="promise_in_bounds",
                    ).astype(jnp.float32)
                    destination[...] = jnp.where(
                        in_block[:, None], selected_rows, destination[...]
                    )

            return

        # Each gather stays inside one native 16-row BF16 tile. Matching
        # int16 indices avoid unsupported mixed-width Mosaic gathers.
        @pl.loop(0, jnp.sum(valid.astype(jnp.float32)).astype(jnp.int32))
        def copy_row(position: _Dynamic) -> _Dynamic:
            """Execute copy row logic."""
            index = jnp.sum(
                jnp.where(
                    jnp.arange(compact_capacity) == position,
                    indices.astype(jnp.float32),
                    0.0,
                )
            ).astype(jnp.int32)
            rows = source[
                pl.ds(pl.multiple_of(index // 16 * 16, 16), 16), :expert_input_size
            ]
            gather_indices = jnp.full((1, expert_input_size), index % 16, jnp.int16)
            selected_row = jnp.take_along_axis(
                rows,
                gather_indices,
                axis=0,
                mode="promise_in_bounds",
            ).astype(jnp.float32)
            destination[pl.ds(position, 1), :] = selected_row

    def _stage_compact_b16(
        self,
        active_rows: _Dynamic,
        destination: _Dynamic,
        offset: _Dynamic,
    ) -> None:
        """Stage compact rows for B16 with scalar scan over row bitmask."""
        (
            batch_size,
            compact_capacity,
            expert_input_size,
            row_bitmask_words,
            source,
        ) = (
            self.batch_size,
            self.compact_capacity,
            self.expert_input_size,
            self.row_bitmask_words,
            self.source,
        )
        (word,) = row_bitmask_words(active_rows)

        def stage_block(block: _Dynamic, position: _Dynamic) -> _Dynamic:
            """Execute stage block logic."""
            first = pl.multiple_of(block * 16, 16)
            bits = (word >> (block * 16)) & 65535

            def stage_nonempty(position: _Dynamic) -> _Dynamic:
                """Execute stage nonempty logic."""
                for row in range(16):
                    bit = (bits >> row) & 1
                    take = (
                        (bit != 0)
                        & (position >= offset)
                        & (position < offset + compact_capacity)
                    )

                    @pl.when(take)
                    def copy_block_row(
                        row: _Dynamic = row, position: _Dynamic = position
                    ) -> _Dynamic:
                        """Execute copy block row logic."""
                        rows = source[pl.ds(first, 16), :expert_input_size]
                        destination[pl.ds(position - offset, 1), :] = rows[
                            row : row + 1
                        ].astype(jnp.float32)

                    position = position + bit
                return position

            return stage_nonempty(position)

        jax.lax.fori_loop(0, batch_size // 16, stage_block, jnp.int32(0))

    def _stage_compact_mod16(
        self,
        active_rows: _Dynamic,
        destination: _Dynamic,
        offset: _Dynamic,
    ) -> None:
        """Stage compact rows for multiples of 16."""
        (
            batch_size,
            compact_capacity,
            expert_input_size,
            source,
        ) = (
            self.batch_size,
            self.compact_capacity,
            self.expert_input_size,
            self.source,
        )
        index = jnp.arange(batch_size)
        active = active_rows.astype(jnp.int32)

        @pl.loop(0, batch_size // 16)
        def stage_block(block: _Dynamic) -> _Dynamic:
            """Execute stage block logic."""
            first = pl.multiple_of(block * 16, 16)
            bits = jnp.sum(jnp.where(index // 16 == block, active << (index % 16), 0))
            position = jnp.sum(jnp.where(index < block * 16, active, 0))
            for row in range(16):
                bit = (bits >> row) & 1
                take = (
                    (bit != 0)
                    & (position >= offset)
                    & (position < offset + compact_capacity)
                )

                @pl.when(take)
                def copy_block_row(
                    row: _Dynamic = row, position: _Dynamic = position
                ) -> _Dynamic:
                    """Execute copy block row logic."""
                    rows = source[pl.ds(first, 16), :expert_input_size]
                    destination[pl.ds(position - offset, 1), :] = rows[
                        row : row + 1
                    ].astype(jnp.float32)

                position = position + bit

    def _stage_compact_fallback(
        self,
        active_rows: _Dynamic,
        destination: _Dynamic,
        offset: _Dynamic,
    ) -> None:
        """Stage compact rows fallback per-row loop."""
        (
            batch_size,
            compact_capacity,
            expert_input_size,
            row_bitmask_words,
            source,
        ) = (
            self.batch_size,
            self.compact_capacity,
            self.expert_input_size,
            self.row_bitmask_words,
            self.source,
        )
        words = row_bitmask_words(active_rows)
        position = jnp.int32(0)
        for row in range(batch_size):
            bit = (words[row // 32] >> (row % 32)) & 1
            if isinstance(offset, int) and offset == 0:  # static first pass
                take = bit != 0
            else:
                take = (
                    (bit != 0)
                    & (position >= offset)
                    & (position < offset + compact_capacity)
                )

            @pl.when(take)
            def copy_active_row(
                row: _Dynamic = row, position: _Dynamic = position
            ) -> _Dynamic:
                """Execute copy active row logic."""
                destination[pl.ds(position - offset, 1), :] = source[
                    row : row + 1, :expert_input_size
                ].astype(jnp.float32)

            position = position + bit

    def stage_compact_rows(
        self,
        active_rows: _Dynamic,
        destination: _Dynamic,
        offset: _Dynamic = 0,
        row_plan: _Dynamic = None,
    ) -> _Dynamic:
        """Copy active rows offset .. offset + C - 1 (in row order) into.

        ``destination`` [C, K] FP32 (the rest zero).
        """
        (
            _stage_compact_b16,
            _stage_compact_fallback,
            _stage_compact_mod16,
            batch_size,
            compact_row_plan,
            reuse_row_plan,
            stage_blocks,
            stage_planned_rows,
        ) = (
            self._stage_compact_b16,
            self._stage_compact_fallback,
            self._stage_compact_mod16,
            self.batch_size,
            self.compact_row_plan,
            self.reuse_row_plan,
            self.stage_blocks,
            self.stage_planned_rows,
        )
        if reuse_row_plan:
            if row_plan is None:
                row_plan = compact_row_plan(active_rows, offset)
            stage_planned_rows(row_plan, destination)
            return
        destination[...] = jnp.zeros(destination.shape, destination.dtype)
        if stage_blocks and batch_size == _CONST_16:
            _stage_compact_b16(active_rows, destination, offset)
            return
        if stage_blocks and batch_size % 16 == 0:
            _stage_compact_mod16(active_rows, destination, offset)
            return
        _stage_compact_fallback(active_rows, destination, offset)

    def compact_route_data(
        self,
        active_rows: _Dynamic,
        current_slot_mask: _Dynamic,
        offset: _Dynamic = 0,
        row_plan: _Dynamic = None,
    ) -> _Dynamic:
        """Execute compact route data logic."""
        (
            batch_size,
            compact_capacity,
            compact_row_plan,
            compact_row_positions,
            probability_totals,
            reuse_row_plan,
            selected_probabilities,
        ) = (
            self.batch_size,
            self.compact_capacity,
            self.compact_row_plan,
            self.compact_row_positions,
            self.probability_totals,
            self.reuse_row_plan,
            self.selected_probabilities,
        )
        route_probabilities = (
            jnp.sum(
                jnp.where(current_slot_mask, selected_probabilities, jnp.float32(0)),
                axis=1,
            )
            / probability_totals
        )
        if reuse_row_plan:
            if row_plan is None:
                row_plan = compact_row_plan(active_rows, offset)
            indices, valid = row_plan
            selected = (jnp.arange(batch_size)[:, None] == indices[None, :]) & valid[
                None, :
            ]
            selector = selected.astype(jnp.bfloat16)
            in_pass = jnp.any(selected, axis=1)
        else:
            before = compact_row_positions(active_rows)
            selector = (
                (before[:, None] - offset == jnp.arange(compact_capacity)[None, :])
                & active_rows[:, None]
            ).astype(jnp.bfloat16)
            in_pass = (
                active_rows & (before >= offset) & (before < offset + compact_capacity)
            )
        return selector, jnp.where(in_pass, route_probabilities, jnp.float32(0))

    def finish_compact_expert(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
        """Finish one expert whose active rows offset .. offset + C - 1 sit.

        compacted in accumulator rows [first_row, first_row + C).
        """
        (
            compact_capacity,
            compact_route_data,
            down_accumulator,
            expert_dot,
            expert_down_scale_vmem,
            expert_down_vmem,
            expert_output,
            expert_output_size,
            expert_width,
            gate_up_accumulator,
            share_compact_finish,
        ) = (
            self.compact_capacity,
            self.compact_route_data,
            self.down_accumulator,
            self.expert_dot,
            self.expert_down_scale_vmem,
            self.expert_down_vmem,
            self.expert_output,
            self.expert_output_size,
            self.expert_width,
            self.gate_up_accumulator,
            self.share_compact_finish,
        )
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            active_rows,
            current_slot_mask,
            copies,
            first_row,
            column,
            slot,
            offset,
            wait_down,
            route_data,
        ) = (
            _p_next(_p_iter, _p_get("active_rows")),
            _p_next(_p_iter, _p_get("current_slot_mask")),
            _p_next(_p_iter, _p_get("copies")),
            _p_next(_p_iter, _p_get("first_row")),
            _p_next(_p_iter, _p_get("column")),
            _p_next(_p_iter, _p_get("slot")),
            _p_next(_p_iter, _p_get("offset", 0)),
            _p_next(_p_iter, _p_kw.get("wait_down", True)),
            _p_next(_p_iter, _p_get("route_data")),
        )
        rows = (
            pl.ds(pl.multiple_of(first_row, 8), compact_capacity)
            if share_compact_finish
            else slice(first_row, first_row + compact_capacity)
        )
        gate_up = (
            gate_up_accumulator[
                rows,
                (
                    pl.ds(pl.multiple_of(column, 128), 2 * expert_width)
                    if share_compact_finish
                    else slice(column, column + 2 * expert_width)
                ),
            ]
            .astype(jnp.bfloat16)
            .astype(jnp.float32)
        )
        middle = _gated_activation(gate_up[:, :expert_width], gate_up[:, expert_width:])
        gate_up_accumulator[rows, :expert_width] = middle.astype(jnp.float32)
        if wait_down:
            copies.down.wait()
            copies.down_scales.wait()
        expert_dot(
            gate_up_accumulator.at[rows, :expert_width],
            expert_down_vmem.at[slot],
            expert_down_scale_vmem.at[slot],
            down_accumulator.at[:compact_capacity],
            expert_width,
            expert_output_size,
            block_major=True,
        )
        if route_data is None:
            route_data = compact_route_data(active_rows, current_slot_mask, offset)
        selector, output_weights = route_data
        selected = _dot(
            selector,
            down_accumulator[:compact_capacity].astype(jnp.bfloat16),
        )
        expert_output[:, :expert_output_size] += selected * output_weights[:, None]

    def run_expert_in_passes(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
        """One expert with more than C rows, C rows per pass."""
        (
            compact_capacity,
            compact_route_data,
            compact_row_plan,
            compact_row_positions,
            expert_dot,
            expert_gate_up_scale_vmem,
            expert_gate_up_vmem,
            expert_input_size,
            expert_width,
            finish_compact_expert,
            first_compact_source_vmem,
            gate_up_accumulator,
            reuse_row_plan,
            stage_compact_rows,
        ) = (
            self.compact_capacity,
            self.compact_route_data,
            self.compact_row_plan,
            self.compact_row_positions,
            self.expert_dot,
            self.expert_gate_up_scale_vmem,
            self.expert_gate_up_vmem,
            self.expert_input_size,
            self.expert_width,
            self.finish_compact_expert,
            self.first_compact_source_vmem,
            self.gate_up_accumulator,
            self.reuse_row_plan,
            self.stage_compact_rows,
        )
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            active_rows,
            current_slot_mask,
            copies,
            row_count,
            slot,
            paired_gate,
        ) = (
            _p_next(_p_iter, _p_get("active_rows")),
            _p_next(_p_iter, _p_get("current_slot_mask")),
            _p_next(_p_iter, _p_get("copies")),
            _p_next(_p_iter, _p_get("row_count")),
            _p_next(_p_iter, _p_get("slot")),
            _p_next(_p_iter, _p_kw.get("paired_gate", False)),
        )
        copies.gate_up.wait()
        copies.gate_up_scales.wait()
        copies.down.wait()
        copies.down_scales.wait()

        before = compact_row_positions(active_rows) if reuse_row_plan else None

        @pl.loop(0, (row_count + compact_capacity - 1) // compact_capacity)
        def run_pass(index: _Dynamic) -> _Dynamic:
            """Execute run pass logic."""
            offset = index * compact_capacity
            row_plan = (
                compact_row_plan(active_rows, offset, before)
                if reuse_row_plan
                else None
            )
            stage_compact_rows(active_rows, first_compact_source_vmem, offset, row_plan)
            if paired_gate:
                _paired_order_expert_gate_up(
                    first_compact_source_vmem,
                    expert_gate_up_vmem.at[slot],
                    expert_gate_up_scale_vmem.at[slot],
                    gate_up_accumulator.at[:compact_capacity, : 2 * expert_width],
                    expert_input_size,
                    MXFP4_PAIR_TILE,
                )
            else:
                expert_dot(
                    first_compact_source_vmem,
                    expert_gate_up_vmem.at[slot],
                    expert_gate_up_scale_vmem.at[slot],
                    gate_up_accumulator.at[:compact_capacity, : 2 * expert_width],
                    expert_input_size,
                    2 * expert_width,
                )
            # Every pass owns its rows; finish it before reusing the compact
            # accumulator. Weight completions are consumed once, above.
            finish_compact_expert(
                active_rows,
                current_slot_mask,
                copies,
                0,
                0,
                slot,
                offset=offset,
                wait_down=False,
                route_data=compact_route_data(
                    active_rows, current_slot_mask, offset, row_plan
                ),
            )

    def shared_expert_at(self, index: _Dynamic) -> _Dynamic:
        """Execute shared expert at logic."""
        return jnp.sum(
            jnp.where(
                self.route_slots[0] == index,
                self.selected_experts[0],
                jnp.int32(0),
            )
        ).astype(jnp.int32)

    def shared_probabilities_at(self, index: _Dynamic) -> _Dynamic:
        """Execute shared probabilities at logic."""
        return jnp.sum(
            jnp.where(
                self.route_slots == index,
                self.selected_probabilities,
                jnp.float32(0),
            ),
            axis=1,
        )

    def wait_for_gate_up(self, index: _Dynamic, slot: _Dynamic) -> _Dynamic:
        """Execute wait for gate up logic."""
        copies = self.copy_expert(self.shared_expert_at(index), slot, down=False)
        copies.gate_up.wait()
        copies.gate_up_scales.wait()

    def wait_for_down(self, index: _Dynamic, slot: _Dynamic) -> _Dynamic:
        """Execute wait for down logic."""
        copies = self.copy_expert(self.shared_expert_at(index), slot, gate_up=False)
        copies.down.wait()
        copies.down_scales.wait()

    def finish_shared_expert(
        self, index: _Dynamic, slot: _Dynamic, column: _Dynamic
    ) -> _Dynamic:
        """Execute finish shared expert logic."""
        (
            gate_up_accumulator,
            expert_width,
            expert_dot,
            expert_down_vmem,
            expert_down_scale_vmem,
            down_accumulator,
            expert_output_size,
            batch_size,
            probability_totals,
            expert_output,
            local_route_counts,
        ) = (
            self.gate_up_accumulator,
            self.expert_width,
            self.expert_dot,
            self.expert_down_vmem,
            self.expert_down_scale_vmem,
            self.down_accumulator,
            self.expert_output_size,
            self.batch_size,
            self.probability_totals,
            self.expert_output,
            self.local_route_counts,
        )
        gate_up = (
            gate_up_accumulator[:, column : column + 2 * expert_width]
            .astype(jnp.bfloat16)
            .astype(jnp.float32)
        )
        middle = _gated_activation(gate_up[:, :expert_width], gate_up[:, expert_width:])
        gate_up_accumulator[:, :expert_width] = middle.astype(jnp.float32)
        self.wait_for_down(index, slot)
        expert_dot(
            gate_up_accumulator.at[:, :expert_width],
            expert_down_vmem.at[slot],
            expert_down_scale_vmem.at[slot],
            down_accumulator,
            expert_width,
            expert_output_size,
            block_major=batch_size >= _CONST_8,
        )
        route_weights = (self.shared_probabilities_at(index) / probability_totals)[
            :, None
        ]
        expert_output[:, :expert_output_size] += (
            down_accumulator[...].astype(jnp.bfloat16).astype(jnp.float32)
            * route_weights
        )

        @pl.when(index + 4 < local_route_counts[0])
        def refill_expert() -> _Dynamic:
            """Execute refill expert logic."""
            _start_async_copies(
                self.copy_expert(self.shared_expert_at(index + 4), slot)
            )

    def run_shared_pair(self, pair: _Dynamic) -> _Dynamic:
        """Execute run shared pair logic."""
        first = 2 * pair
        first_slot = first % 4
        second_slot = (first + 1) % 4
        self.wait_for_gate_up(first, first_slot)
        self.wait_for_gate_up(first + 1, second_slot)
        _paired_expert_gate_up(
            self.source,
            self.expert_gate_up_vmem,
            self.expert_gate_up_scale_vmem,
            self.gate_up_accumulator,
            first_slot,
            second_slot,
            self.expert_input_size,
        )
        self.finish_shared_expert(first, first_slot, 0)
        self.finish_shared_expert(first + 1, second_slot, 2 * self.expert_width)

    def run_shared_tail(self) -> _Dynamic:
        """Execute run shared tail logic."""
        index = self.local_route_counts[0] - 1
        slot = index % 4
        self.wait_for_gate_up(index, slot)
        self.expert_dot(
            self.source,
            self.expert_gate_up_vmem.at[slot],
            self.expert_gate_up_scale_vmem.at[slot],
            self.gate_up_accumulator.at[:, : 2 * self.expert_width],
            self.expert_input_size,
            2 * self.expert_width,
        )
        self.finish_shared_expert(index, slot, 0)

    def run_shared_routes_impl(self) -> _Dynamic:
        """Execute run shared routes logic."""
        if self.compact_only:
            return None

        @pl.loop(0, self.local_route_counts[0] // 2)
        def run_shared_pair(pair: _Dynamic) -> _Dynamic:
            """Execute run shared pair logic."""
            self.run_shared_pair(pair)

        @pl.when(self.local_route_counts[0] % 2 == 1)
        def run_shared_tail() -> _Dynamic:
            """Execute run shared tail logic."""
            self.run_shared_tail()

        return None

    def run_divergent_routes_impl(self) -> _Dynamic:
        """Execute run divergent routes logic."""
        _DivergentRoutesRunner(self).run()

    def run(self) -> None:
        """Execute run logic."""
        runner = self

        @pl.when(runner.routes_are_shared & (not runner.compact_only))
        def run_shared_routes() -> _Dynamic:
            """Execute run shared routes logic."""
            runner.run_shared_routes_impl()

        @pl.when(~runner.routes_are_shared)
        def run_divergent_routes() -> _Dynamic:
            """Execute run divergent routes logic."""
            runner.run_divergent_routes_impl()


def _run_routed_expert_stream(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Execute the next route shared by any ready batch rows.

    Rows retain their own routing order. When several rows have the same next
    expert, its weights are loaded once and its projections carry the batch
    dimension. The loop is bounded by the distinct experts (grouped routes)
    or the route count, never by batch size.
    """
    return _RoutedExpertStreamRunner(*_p_args, **_p_kw).run()


def _staging_slot_rows(payload_rows: int) -> int:
    """A staged per-pair payload's slot: whole 16-row BF16 tiles."""
    return -(-payload_rows // 16) * 16


def projection_staging_scratch(
    batch_size: int, attention_dp: int = 1
) -> tuple[_Dynamic, ...]:
    """Above B32: send/receive staging of the attention projection's four.

    per-pair payloads ([2, 4 * slot, 128] BF16).
    """
    if batch_size <= _CONST_32:
        return ()
    payload_rows = (batch_size // attention_dp // 4) * 3584 // 128
    return (tpu.VMEM((2, 4 * _staging_slot_rows(payload_rows), 128), jnp.bfloat16),)


def routed_send_scratch(batch_size: int) -> tuple[_Dynamic, ...]:
    """Above B32: the routed reduce-scatter's send side (eight lanes' tokens)."""
    if batch_size <= _CONST_32:
        return ()
    return (tpu.VMEM((8 * 14 * -(-batch_size // 8), 128), jnp.float32),)


def communication_scratch(
    batch_size: int = 1, attention_dp: int = 1
) -> tuple[_Dynamic, ...]:
    """Shared for the entire fused stack, including the reduction sequence."""
    # FP32 transport rows per token: 28 for shared outputs and attention
    # projections, 14 for routed outputs, in separate buffers. Above B8 only
    # layer 0's dense all-reduce carries the whole batch; it runs in 8-row
    # halves, so B16 keeps the B8 capacity.
    wide_reduction_rows = max(
        64, 28 * (min(batch_size, 8) if batch_size <= _CONST_16 else 16)
    )
    # Above B32 the host exchange grows with the attention projection's
    # per-pair payloads (staged in projection_staging_scratch).
    host_reduction_rows = max(
        wide_reduction_rows,
        7 * batch_size // attention_dp if batch_size > _CONST_32 else 0,
    )
    # The shared-expert reduce-scatter receives 4 rows per token in each lane
    # and host stage.
    wide_reduction_rows = max(wide_reduction_rows, 4 * batch_size)
    host_reduction_rows = max(host_reduction_rows, 4 * batch_size)
    if attention_dp > 1:
        # The DP MoE-input gather's 4 * DP row wires use lanes 4..7.
        wide_reduction_rows = max(wide_reduction_rows, 7 * batch_size)
    # The eight-lane routed reduce-scatter pads the batch to whole tokens per lane.
    padded_tokens = (
        8 * -(-batch_size // 8) if batch_size in (6, *SCATTER_BATCHES) else batch_size
    )
    expert_reduction_rows = max(32, 14 * padded_tokens)
    expert_host_rows = expert_reduction_rows
    if batch_size > _CONST_32:
        # One lane's tokens per source lane (send side: routed_send_scratch);
        # the host buffer also carries the shared-expert shard sends.
        expert_reduction_rows = 14 * -(-batch_size // 8)
        expert_host_rows = max(expert_reduction_rows, 8 * batch_size)
    # Whole 8-row FP32 tiles: padded strides make the reductions' ref reshapes
    # fail to compile ("reshape on a non-contiguous memref").
    (
        wide_reduction_rows,
        expert_reduction_rows,
        host_reduction_rows,
        expert_host_rows,
    ) = (
        -(-rows // 8) * 8
        for rows in (
            wide_reduction_rows,
            expert_reduction_rows,
            host_reduction_rows,
            expert_host_rows,
        )
    )
    return (
        tpu.VMEM((8, wide_reduction_rows, 128), jnp.float32),
        tpu.VMEM((8, expert_reduction_rows, 128), jnp.float32),
        tpu.VMEM((4, host_reduction_rows, 128), jnp.float32),
        tpu.VMEM((4, expert_host_rows, 128), jnp.float32),
        # Attention's group-gather features of every held shard, in native
        # 16-row backing (active rows via _bf16_rows_view).
        tpu.VMEM(
            (max(16, -(-batch_size // 16) * 16), 1024 * attention_dp), jnp.bfloat16
        ),
        # Above B32 (always token-scattered) attention writes only each
        # pair's batch / 4 rows here.
        tpu.VMEM(
            (
                max(
                    16,
                    -(
                        -(batch_size // 4 if batch_size >= _CONST_64 else batch_size)
                        // 16
                    )
                    * 16,
                ),
                7168,
            ),
            jnp.bfloat16,
        ),
        tpu.SemaphoreType.DMA((16,)),
        tpu.SemaphoreType.DMA((16,)),
        tpu.SMEM((1,), jnp.int32),
        tpu.VMEM((batch_size, 4096), jnp.bfloat16),
    )


TensorLike = jax_typing.ArrayLike | jax.ShapeDtypeStruct


@dataclasses.dataclass(frozen=True, slots=True)
class AttentionCommunicationWorkspace:
    """Collective transport storage shared by attention and MoE layers."""

    shared_local_reduction_vmem: (
        Any  # [8, rows, 128] FP32; rows: see communication_scratch
    )
    expert_local_reduction_vmem: _Dynamic  # [8, rows, 128] FP32

    shared_host_reduction_vmem: _Dynamic  # [4, rows, 128] FP32
    expert_host_reduction_vmem: _Dynamic  # [4, rows, 128] FP32
    attention_features_vmem: _Dynamic  # [16 * n, 1024 * attention_dp] BF16
    attention_output_vmem: _Dynamic  # [16 * n, 7168] BF16
    send_semaphores: _Dynamic  # [16]
    receive_semaphores: _Dynamic  # [16]
    reduction_phase: _Dynamic  # [1]
    latent_gather_vmem: _Dynamic  # [batch_size, 4096]
    projection_staging_vmem: _Dynamic = None  # above B32: [2, 4 * payload, 128] BF16
    routed_send_vmem: _Dynamic = None  # above B32: [8 * lane payload, 128] FP32
    attention_gather_vmem: _Dynamic = None  # B4: private final-gather packets
    routed_consumed: _Dynamic = (
        None  # routed reduce-scatter: host-mates done reading their slots
    )
    pair_semaphores: _Dynamic = (
        None  # DP MoE-input gather: the rank pair's half-wire swap
    )


class _ScopedWait:
    """An async copy whose wait is traced as the e_wait phase."""

    def __init__(self, copy: _Dynamic) -> None:
        """Execute init logic."""
        self.copy = copy

    def start(self) -> _Dynamic:
        """Execute start logic."""
        self.copy.start()

    def wait(self) -> _Dynamic:
        """Execute wait logic."""
        with _phase("e_wait"):
            self.copy.wait()


def _start_async_copies(copies: SimpleNamespace) -> None:
    """Execute start async copies logic."""
    for copy in vars(copies).values():
        if copy is not None:
            copy.start()


def _gather_output_shards(
    output_vmem: _Dynamic,
    send_semaphores: _Dynamic,
    receive_semaphores: _Dynamic,
    semaphore_slot: _Dynamic,
) -> None:
    """Execute gather output shards logic."""
    rank = jax.lax.axis_index("tp")
    local_shard = output_vmem.at[:, pl.ds((rank % 8) * 1024, 1024)]
    for offset in (7, 6, 5, 3, 4, 2, 1):
        tpu.make_async_remote_copy(
            local_shard,
            local_shard,
            send_semaphores.at[semaphore_slot],
            receive_semaphores.at[semaphore_slot],
            device_id=(rank ^ offset,),
            device_id_type=pl.DeviceIdType.MESH,
        ).start()
    received_shards = output_vmem.at[:, : 7 * 1024]
    tpu.make_async_copy(
        received_shards,
        received_shards,
        send_semaphores.at[semaphore_slot],
    ).wait()
    tpu.make_async_copy(
        received_shards,
        received_shards,
        receive_semaphores.at[semaphore_slot],
    ).wait()


@dataclasses.dataclass(frozen=True, slots=True)
class MixtureOfExpertsWeights:
    """MoE parameters and persistent resources for asynchronous weight copies."""

    routing_projection: _Dynamic  # [hidden_size, 896]
    routing_projection_vmem: _Dynamic  # [1, hidden_size, 512]
    routing_copy_semaphores: _Dynamic  # [2]
    routing_bias: TensorLike  # [1, 896]

    latent_input_projection: TensorLike  # [hidden_size, 128]
    latent_output_projection: TensorLike  # [expert_output_size, 1024]
    latent_output_projection_vmem: _Dynamic  # [expert_output_size, 1024]
    latent_output_copy_semaphore: _Dynamic  # scalar DMA semaphore
    latent_normalization: TensorLike  # [1, padded_latent_size]
    shared_gate_up_projection: TensorLike  # [hidden_size, 2 * shared_width]
    shared_down_projection: TensorLike  # [shared_width, hidden_size]
    # Packed MXFP4: [local_experts, expert_input_size / 8, 2 * expert_width].
    expert_gate_up_projection: _Dynamic
    expert_gate_up_projection_vmem: _Dynamic  # [4, ...]
    expert_gate_up_scales: _Dynamic
    expert_gate_up_scales_vmem: _Dynamic  # [4, ...]
    # Packed MXFP4: [local_experts, expert_width / 8, expert_output_size].
    expert_down_projection: _Dynamic
    expert_down_projection_vmem: _Dynamic  # [4, ...]
    expert_down_scales: _Dynamic
    expert_down_scales_vmem: _Dynamic  # [4, ...]
    expert_copy_semaphores: _Dynamic  # [4, 4]

    def _copy_routing_projection(
        self,
        *,
        wait: bool,
        initialize_padding: bool = True,
    ) -> None:
        """Execute copy routing projection logic."""
        lane = jax.lax.axis_index("tp") % 2

        @pl.when(lane == 0)
        def copy_full_half() -> _Dynamic:
            """Execute copy full half logic."""
            copy = tpu.make_async_copy(
                self.routing_projection.at[:, :512],
                self.routing_projection_vmem.at[0],
                self.routing_copy_semaphores.at[0],
            )
            copy.wait() if wait else copy.start()

        @pl.when(lane == 1)
        def copy_tail() -> _Dynamic:
            """Execute copy tail logic."""
            if not wait and initialize_padding:
                self.routing_projection_vmem[0, :, 384:] = jnp.zeros(
                    (self.routing_projection.shape[0], 128),
                    jnp.bfloat16,
                )
            copy = tpu.make_async_copy(
                self.routing_projection.at[:, pl.ds(512, 384)],
                self.routing_projection_vmem.at[0, :, :384],
                self.routing_copy_semaphores.at[0],
            )
            copy.wait() if wait else copy.start()

    def start_routing_projection_copy(self, *, initialize_padding: bool = True) -> None:
        """Execute start routing projection copy logic."""
        self._copy_routing_projection(
            wait=False,
            initialize_padding=initialize_padding,
        )

    def wait_for_routing_projection_copy(self) -> None:
        """Execute wait for routing projection copy logic."""
        self._copy_routing_projection(wait=True)

    def make_latent_output_copy(self) -> _Dynamic:
        """Construct make latent output copy logic."""
        return tpu.make_async_copy(
            self.latent_output_projection,
            self.latent_output_projection_vmem,
            self.latent_output_copy_semaphore,
        )

    def make_expert_copies(
        self,
        expert: _Dynamic,
        slot: _Dynamic,
        *,
        gate_up: bool = True,
        down: bool = True,
    ) -> SimpleNamespace:
        """Construct make expert copies logic."""
        return SimpleNamespace(
            gate_up=(
                tpu.make_async_copy(
                    self.expert_gate_up_projection.at[expert],
                    self.expert_gate_up_projection_vmem.at[slot],
                    self.expert_copy_semaphores.at[slot, 0],
                )
                if gate_up
                else None
            ),
            gate_up_scales=(
                tpu.make_async_copy(
                    self.expert_gate_up_scales.at[expert],
                    self.expert_gate_up_scales_vmem.at[slot],
                    self.expert_copy_semaphores.at[slot, 1],
                )
                if gate_up
                else None
            ),
            down=(
                tpu.make_async_copy(
                    self.expert_down_projection.at[expert],
                    self.expert_down_projection_vmem.at[slot],
                    self.expert_copy_semaphores.at[slot, 2],
                )
                if down
                else None
            ),
            down_scales=(
                tpu.make_async_copy(
                    self.expert_down_scales.at[expert],
                    self.expert_down_scales_vmem.at[slot],
                    self.expert_copy_semaphores.at[slot, 3],
                )
                if down
                else None
            ),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class MixtureOfExpertsWorkspace:
    """Persistent VMEM and semaphores used by one MoE invocation."""

    selected_experts_vmem: _Dynamic  # [batch_size, 128]
    selected_probabilities_vmem: _Dynamic  # [batch_size, 128]
    local_route_counts_vmem: _Dynamic  # [batch_size, 128]
    probability_totals_vmem: _Dynamic  # [batch_size, 128]

    routing_scores_vmem: _Dynamic  # [batch_size, 1024]
    shared_outputs_vmem: _Dynamic  # [batch_size, hidden_size]
    routed_outputs_vmem: _Dynamic  # [batch_size, padded_latent_size]
    gate_up_accumulator_vmem: (
        Any  # [rows, 4 * expert_width]; rows: _expert_accumulator_rows
    )
    down_accumulator_vmem: _Dynamic  # [rows, expert_output_size]
    processed_routes_vmem: _Dynamic  # [batch_size, 128]
    first_compact_source_vmem: _Dynamic  # [max(1, _compact_rows), expert_input_size]
    second_compact_source_vmem: _Dynamic  # [max(1, _compact_rows), expert_input_size]
    shared_send_semaphores: _Dynamic  # [10]
    shared_receive_semaphores: _Dynamic  # [10]
    output_vmem: _Dynamic  # [batch_size, 8192] or token-scatter transport shape
    output_send_semaphores: _Dynamic  # [2] or [7]
    output_receive_semaphores: _Dynamic  # [2] or [7]
    latent_send_semaphores: _Dynamic  # [2]
    latent_receive_semaphores: _Dynamic  # [2]
    # Sharded routing only: this pair's selection rows and everyone's.
    selection_source_vmem: _Dynamic = None  # [batch_size / 4, 128] FP32
    selections_vmem: _Dynamic = None  # [batch_size, 128] FP32


class _MoeRunner:
    """Runner executing one mixture-of-experts layer inside a Pallas program."""

    def __getattr__(self, name: _Dynamic) -> _Dynamic:
        """Dynamic attribute access."""
        raise AttributeError(name)

    def __init__(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Initialize MoE runner."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        self.hidden_ref = _p_next(_p_iter, _p_get("hidden_ref"))
        self.output_ref = _p_next(_p_iter, _p_get("output_ref"))
        self.weights = _p_get("weights")
        self.workspace = _p_get("workspace")
        self.communication_workspace = _p_get("communication_workspace")
        self.selected_expert_count = _p_get("selected_expert_count", 16)
        self.epsilon = _p_get("epsilon", 1e-05)
        self.latent_feature_count = _p_get("latent_feature_count", 3584)
        self.synchronize_devices = _p_kw.get("synchronize_devices", True)
        self.routing_projection_is_prefetched = _p_kw.get(
            "routing_projection_is_prefetched", False
        )
        self.group_pending_expert_routes = _p_kw.get(
            "group_pending_expert_routes", False
        )
        self.token_scatter_output = _p_kw.get("token_scatter_output", False)
        self.routed_reduce_scatter = _p_kw.get("routed_reduce_scatter", False)
        self.after_experts = _p_kw.get("after_experts")
        self.after_routing_scores = _p_get("after_routing_scores")
        self.overlap_router_latent = _p_kw.get("overlap_router_latent", False)
        self.overlap_shared_reduction = _p_kw.get("overlap_shared_reduction", False)
        self.prepare_weights_vmem = _p_get("prepare_weights_vmem")
        self.shared_gate_up_stream = _p_get("shared_gate_up_stream")
        self.token_scatter_owners = _p_get("token_scatter_owners", 4)
        self.owned_hidden_ref = _p_get("owned_hidden_ref")
        self.live_rows = _p_get("live_rows")

        self.batch_size, self.hidden_size = self.hidden_ref.shape
        self.expert_count = self.weights.routing_projection.shape[1]
        self.latent_size = self.weights.latent_normalization.shape[-1]
        self.expert_output_size = self.weights.expert_down_projection.shape[-1]
        self.local_expert_count = self.weights.expert_gate_up_projection.shape[0]
        self.expert_tensor_parallel = 32 * self.local_expert_count // self.expert_count
        self.expert_width = self.weights.expert_down_projection.shape[1] * 8
        self.expert_input_size = self.weights.expert_gate_up_projection.shape[1] * 8
        self.shared_width = self.weights.shared_down_projection.shape[0]
        self.expert_prefetch_slot_count = 4

        self.prefix_state: list[_Dynamic] = [jnp.int32(1), jnp.int32(0)]
        self.dense_copies: _Dynamic = None
        self.latent_input_copy: _Dynamic = None
        self.latent_input_projection_vmem: _Dynamic = None
        self.shared_gate_up_projection_vmem: _Dynamic = None
        self.shared_down_projection_vmem: _Dynamic = None

        self._validate_dimensions()
        self._validate_workspaces()
        self._init_workspace_views()
        self._init_communication_state()

    def _validate_dimensions(self) -> None:
        """Validate input and weight dimensions."""
        if not 1 <= self.batch_size <= MAX_BATCH:
            _err_msg = f"MoE supports 1 to {MAX_BATCH} rows"
            raise ValueError(_err_msg)
        if (
            self.token_scatter_output
            and self.batch_size not in SCATTER_BATCHES
            and not (self.batch_size in (2, 4) and self.token_scatter_owners == 1)
        ):
            _err_msg = (
                "MoE token-scattered output requires batch size 8 or 16, or "
                "B2/B4 to one owner"
            )
            raise ValueError(_err_msg)
        if self.routed_reduce_scatter and self.batch_size not in (
            6,
            *SCATTER_BATCHES,
        ):
            _err_msg = "MoE routed reduce-scatter requires batch size 6, 8 or 16"
            raise ValueError(_err_msg)
        if (
            self.weights.expert_gate_up_projection.dtype != jnp.uint32
            or self.weights.expert_down_projection.dtype != jnp.uint32
        ):
            _err_msg = "Routed-expert weights must be packed MXFP4"
            raise ValueError(_err_msg)
        if self.expert_tensor_parallel not in (1, 2, 4):
            _err_msg = "Expert tensor parallelism must be 1, 2, or 4"
            raise ValueError(_err_msg)
        if (
            self.hidden_size % 128
            or self.expert_count % 128
            or self.latent_size % 256
            or self.expert_width % 256
            or self.hidden_size > _CONST_8192
            or self.latent_size > _CONST_8192
        ):
            _err_msg = "Unsupported MoE tile dimensions"
            raise ValueError(_err_msg)
        self._validate_weight_shapes()

    def _validate_weight_shapes(self) -> None:
        """Validate shapes of projection weights and scales."""
        if self.weights.latent_input_projection.shape != (
            self.hidden_size,
            128,
        ) or self.weights.latent_output_projection.shape != (
            self.expert_output_size,
            1024,
        ):
            _err_msg = "Latent projections require padded TP32 shards"
            raise ValueError(_err_msg)
        if (
            self.expert_count * self.expert_tensor_parallel
            != 32 * self.local_expert_count
            or self.expert_output_size > self.latent_size
            or self.expert_input_size > self.latent_size
        ):
            _err_msg = "Expert ownership or latent padding is inconsistent"
            raise ValueError(_err_msg)
        if self.expert_input_size % 512 or self.expert_output_size % 128:
            _err_msg = "Expert dimensions must satisfy MXFP4 alignment"
            raise ValueError(_err_msg)
        if (
            self.weights.expert_gate_up_projection.shape[1:]
            != (self.expert_input_size // 8, 2 * self.expert_width)
            or self.weights.expert_gate_up_scales.shape[1:]
            != (self.expert_input_size // 32, 2 * self.expert_width)
            or self.weights.expert_gate_up_scales.dtype != jnp.uint8
        ):
            _err_msg = "MXFP4 gate/up weights have inconsistent shapes"
            raise ValueError(_err_msg)
        if (
            self.weights.expert_down_projection.shape[1] * 8 != self.expert_width
            or self.weights.expert_down_scales.dtype != jnp.uint8
        ):
            _err_msg = "MXFP4 down weights or scales have inconsistent shapes"
            raise ValueError(_err_msg)

    def _validate_workspaces(self) -> None:
        """Validate shapes of workspace buffers."""
        expected_output_shape = (
            (self.batch_size // self.token_scatter_owners, self.hidden_size)
            if self.token_scatter_output
            else self.hidden_ref.shape
        )
        if self.output_ref.shape != expected_output_shape:
            _err_msg = "MoE output has the wrong shape for its communication mode"
            raise ValueError(_err_msg)

        accumulator_rows = _expert_accumulator_rows(
            self.batch_size,
            group_pending_routes=self.group_pending_expert_routes,
        )
        expected_workspace_shapes = (
            (self.workspace.selected_experts_vmem.shape, (self.batch_size, 128)),
            (
                self.workspace.selected_probabilities_vmem.shape,
                (self.batch_size, 128),
            ),
            (self.workspace.local_route_counts_vmem.shape, (self.batch_size, 128)),
            (self.workspace.probability_totals_vmem.shape, (self.batch_size, 128)),
            (self.workspace.routing_scores_vmem.shape, (self.batch_size, 1024)),
            (
                self.workspace.shared_outputs_vmem.shape,
                (
                    self.batch_size,
                    1024 if _shared_shards(self.batch_size) else self.hidden_size,
                ),
            ),
            (
                self.workspace.routed_outputs_vmem.shape,
                (self.batch_size, self.latent_size),
            ),
            (
                self.workspace.gate_up_accumulator_vmem.shape,
                (accumulator_rows[0], 4 * self.expert_width),
            ),
            (
                self.workspace.down_accumulator_vmem.shape,
                (accumulator_rows[1], self.expert_output_size),
            ),
            (self.workspace.processed_routes_vmem.shape, (self.batch_size, 128)),
            (
                self.workspace.first_compact_source_vmem.shape,
                (max(1, _compact_rows(self.batch_size)), self.expert_input_size),
            ),
            (
                self.workspace.second_compact_source_vmem.shape,
                (max(1, _compact_rows(self.batch_size)), self.expert_input_size),
            ),
        )
        if any(actual != expected for actual, expected in expected_workspace_shapes):
            _err_msg = "MoE workspace has an invalid shape"
            raise ValueError(_err_msg)

        expected_output_vmem_shape = (
            (
                self.token_scatter_owners + 8,
                self.batch_size // self.token_scatter_owners * 1024 // 128,
                128,
            )
            if self.token_scatter_output
            else (self.batch_size, 8192)
        )
        if self.workspace.output_vmem.shape != expected_output_vmem_shape:
            _err_msg = "MoE output communication workspace has the wrong shape"
            raise ValueError(_err_msg)
        if self.communication_workspace.latent_gather_vmem.shape != (
            self.batch_size,
            4096,
        ):
            _err_msg = "MoE latent-gather workspace must match the batch size"
            raise ValueError(_err_msg)
        if self.weights.routing_projection_vmem.shape != (
            1,
            self.hidden_size,
            512,
        ):
            _err_msg = "MoE routing projection workspace has the wrong shape"
            raise ValueError(_err_msg)
        self._validate_expert_workspaces()

    def _validate_expert_workspaces(self) -> None:
        """Validate expert weight workspace shapes."""
        slots = self.expert_prefetch_slot_count
        expert_workspace_shapes = (
            (
                self.weights.expert_gate_up_projection_vmem.shape,
                (slots, *self.weights.expert_gate_up_projection.shape[1:]),
            ),
            (
                self.weights.expert_gate_up_scales_vmem.shape,
                (slots, *self.weights.expert_gate_up_scales.shape[1:]),
            ),
            (
                self.weights.expert_down_projection_vmem.shape,
                (slots, *self.weights.expert_down_projection.shape[1:]),
            ),
            (
                self.weights.expert_down_scales_vmem.shape,
                (slots, *self.weights.expert_down_scales.shape[1:]),
            ),
        )
        if any(actual != expected for actual, expected in expert_workspace_shapes):
            _err_msg = "MoE expert weight workspace has an invalid shape"
            raise ValueError(_err_msg)

    def _init_workspace_views(self) -> None:
        """Initialize local workspace buffer references."""
        self.expert_gate_up_vmem = self.weights.expert_gate_up_projection_vmem
        self.expert_gate_up_scale_vmem = self.weights.expert_gate_up_scales_vmem
        self.expert_down_vmem = self.weights.expert_down_projection_vmem
        self.expert_down_scale_vmem = self.weights.expert_down_scales_vmem
        self.selected_experts_vmem = self.workspace.selected_experts_vmem
        self.selected_probabilities_vmem = self.workspace.selected_probabilities_vmem
        self.local_route_counts_vmem = self.workspace.local_route_counts_vmem
        self.probability_totals_vmem = self.workspace.probability_totals_vmem
        self.shared_outputs_vmem = self.workspace.shared_outputs_vmem
        self.routed_outputs_vmem = self.workspace.routed_outputs_vmem
        self.shared_sharded = self.shared_outputs_vmem.shape[1] < self.hidden_size
        self.shared_down_weights = (
            self.prepare_weights_vmem[2]
            if self.prepare_weights_vmem is not None
            else None
        )
        if not (not self.shared_sharded or self.shared_down_weights is not None):
            _err_msg = "shared shards need pooled weights"
            raise AssertionError(_err_msg)
        self.gate_up_accumulator_vmem = self.workspace.gate_up_accumulator_vmem
        self.down_accumulator_vmem = self.workspace.down_accumulator_vmem
        self.processed_routes_vmem = self.workspace.processed_routes_vmem
        self.first_compact_source_vmem = self.workspace.first_compact_source_vmem
        self.second_compact_source_vmem = self.workspace.second_compact_source_vmem

    def _init_communication_state(self) -> None:
        """Initialize communication buffers, semaphores, and payloads."""
        self.shared_send_semaphores = self.workspace.shared_send_semaphores
        self.shared_receive_semaphores = self.workspace.shared_receive_semaphores
        self.output_vmem = self.workspace.output_vmem
        self.output_send_semaphores = self.workspace.output_send_semaphores
        self.output_receive_semaphores = self.workspace.output_receive_semaphores
        self.latent_send_semaphores = self.workspace.latent_send_semaphores
        self.latent_receive_semaphores = self.workspace.latent_receive_semaphores
        self.shared_local_reduction_vmem = (
            self.communication_workspace.shared_local_reduction_vmem
        )
        self.expert_local_reduction_vmem = (
            self.communication_workspace.expert_local_reduction_vmem
        )
        self.shared_host_reduction_vmem = (
            self.communication_workspace.shared_host_reduction_vmem
        )
        self.expert_host_reduction_vmem = (
            self.communication_workspace.expert_host_reduction_vmem
        )
        self.routed_send_vmem = self.communication_workspace.routed_send_vmem
        self.routed_consumed = self.communication_workspace.routed_consumed
        self.send_semaphores = self.communication_workspace.send_semaphores
        self.receive_semaphores = self.communication_workspace.receive_semaphores
        self.reduction_phase = self.communication_workspace.reduction_phase
        self.latent_vmem = self.communication_workspace.latent_gather_vmem
        self.rank = jax.lax.axis_index("tp")
        if self.synchronize_devices:
            collectives32.barrier()
            self.reduction_phase[0] = 0
        self.hidden = self.hidden_ref[...]

        if not self.routing_projection_is_prefetched:
            self.weights.start_routing_projection_copy()
        self.latent_output_copy = self.weights.make_latent_output_copy()
        self.overlap_latent_gather = self.selected_expert_count >= _CONST_12
        self.overlap_router_latent = (
            self.overlap_router_latent
            and self.overlap_latent_gather
            and self.owned_hidden_ref is None
        )
        self.overlap_shared_routing = self.overlap_router_latent and self.shared_sharded
        self.latent_slot = self.reduction_phase[0] % 2

        self.shared_rows = ((self.hidden_size + 1023) // 1024) * 8
        self.shared_payload = self.shared_rows // 2
        self.expert_rows = self.expert_output_size // 128
        self.expert_payload = self.expert_rows // 2
        self.tokens_per_lane = -(-self.batch_size // 8)
        self.lane_payload = self.tokens_per_lane * self.expert_payload
        self.reduce_scatter_shared_output = (
            self.batch_size in (6, *SCATTER_BATCHES)
            and self.group_pending_expert_routes
        )
        self.reduce_scatter_routed_output = self.routed_reduce_scatter
        self.shared_shard_width = 1024
        self.shared_shard_rows = self.shared_shard_width // 128
        self.shared_shard_payload = self.shared_shard_rows // 2

    def copy_expert(
        self,
        expert: _Dynamic,
        slot: _Dynamic,
        *,
        gate_up: _Dynamic = True,
        down: _Dynamic = True,
    ) -> _Dynamic:
        """Execute copy expert logic."""
        return self.weights.make_expert_copies(
            expert,
            slot,
            gate_up=gate_up,
            down=down,
        )

    def latent_copy(self, offset: _Dynamic, slot: _Dynamic) -> _Dynamic:
        """Execute latent copy logic."""
        return tpu.make_async_remote_copy(
            self.latent_vmem.at[:, pl.ds(offset * 128, 128)],
            self.latent_vmem.at[:, pl.ds(slot * 128, 128)],
            self.latent_send_semaphores.at[offset - 1],
            self.latent_receive_semaphores.at[offset - 1],
            device_id=(self.rank ^ offset,),
            device_id_type=pl.DeviceIdType.MESH,
        )

    def wait_for_latent_gather(self) -> None:
        """Wait for latent gather copies from peers."""
        with _phase("t0_latent_gather_wait"):
            for offset in range(1, 32):
                self.latent_copy(offset, self.rank).wait()

    def start_latent_gather(
        self,
        latent_input_projection_vmem: _Dynamic,
        latent_input_copy: _Dynamic,
    ) -> None:
        """Start async gather of projected latent inputs."""
        with _phase("t0_latent_gather"):
            latent_input_copy.wait()
            self.latent_vmem[:, pl.ds(self.rank * 128, 128)] = _dot(
                self.hidden, latent_input_projection_vmem[...]
            )
            # Rank-to-rank transpose fills all 32 latent slots in one exchange.
            copies = []
            for offset in range(1, 32):
                copy = self.latent_copy(offset, offset ^ self.rank)
                copy.start()
                copies.append(copy)
            if not self.overlap_latent_gather:
                for copy in copies:
                    copy.wait()

    def _start_prepare_batch_copies(
        self,
        latent_input_copy_semaphore: _Dynamic,
        dense_copy_semaphores: _Dynamic,
    ) -> None:
        """Start asynchronous weight copies for prepare_batch."""
        self.latent_input_copy = tpu.make_async_copy(
            self.weights.latent_input_projection,
            self.latent_input_projection_vmem,
            latent_input_copy_semaphore,
        )
        self.latent_input_copy.start()
        self.dense_copies = SimpleNamespace(
            gate_up=None
            if self.shared_gate_up_stream is not None
            else tpu.make_async_copy(
                self.weights.shared_gate_up_projection,
                self.shared_gate_up_projection_vmem,
                dense_copy_semaphores.at[0],
            ),
            down=tpu.make_async_copy(
                self.weights.shared_down_projection,
                self.shared_down_projection_vmem,
                dense_copy_semaphores.at[1],
            ),
        )
        _start_async_copies(self.dense_copies)
        if self.shared_gate_up_stream is not None:
            self.shared_gate_up_stream.start()
        self.weights.wait_for_routing_projection_copy()

    def _prepare_shared_compute(self) -> None:
        """Execute shared expert computation."""
        if self.shared_gate_up_stream is not None:
            shared_gate_up = (
                self.shared_gate_up_stream.dot(
                    self.hidden,
                    self.communication_workspace.reduction_phase[0] >= 0,
                )
                .astype(jnp.bfloat16)
                .astype(jnp.float32)
            )
        else:
            self.dense_copies.gate_up.wait()
            shared_gate_up = (
                _dot(self.hidden, self.shared_gate_up_projection_vmem[...])
                .astype(jnp.bfloat16)
                .astype(jnp.float32)
            )
        middle = _gated_activation(
            shared_gate_up[:, : self.shared_width],
            shared_gate_up[:, self.shared_width :],
        )
        if not self.overlap_shared_routing:
            self.dense_copies.down.wait()
        if self.shared_sharded:
            self.routed_outputs_vmem[:, : self.shared_width] = middle.astype(
                jnp.float32
            )
        else:
            self.shared_outputs_vmem[...] = _dot(
                middle, self.shared_down_projection_vmem[...]
            )

    def _start_expert_prefetch(self, winners: _Dynamic, route_counts: _Dynamic) -> None:
        """Start DMA copy for prefetched expert."""
        _start_async_copies(
            self.copy_expert(
                winners[0] % self.local_expert_count,
                route_counts[0],
            )
        )

    def _on_local_selection(
        self,
        winners: _Dynamic,
        owned_locally: _Dynamic,
        route_counts: _Dynamic,
        unused_selected_experts: _Dynamic,
    ) -> None:
        """Prefetch shared route prefix experts."""
        if self.group_pending_expert_routes:
            return
        del unused_selected_experts
        self.prefix_state[0] *= jnp.all(winners == winners[0]).astype(jnp.int32)
        should_prefetch = (
            self.prefix_state[0]
            * owned_locally.astype(jnp.int32)[0]
            * (route_counts[0] < self.expert_prefetch_slot_count).astype(jnp.int32)
        )

        @pl.when(should_prefetch != 0)
        def start_expert_copy() -> None:
            """Trigger async copy for prefetched expert."""
            self._start_expert_prefetch(winners, route_counts)

        self.prefix_state[1] += should_prefetch

    def _after_selection(self, selection: _Dynamic) -> None:
        """Overlap routing steps with latent gather or shared compute."""
        if self.overlap_shared_routing and selection == _CONST_7:
            self._prepare_shared_compute()
        if (
            selection == _CONST_3
            and self.overlap_latent_gather
            and not self.overlap_router_latent
        ):
            self.start_latent_gather(
                self.latent_input_projection_vmem,
                self.latent_input_copy,
            )

    def _after_score_send(self) -> None:
        """Gather latent inputs after score transmission."""
        self.start_latent_gather(
            self.latent_input_projection_vmem, self.latent_input_copy
        )

    def _route_batch(self) -> tuple[_Dynamic, _Dynamic, _Dynamic, _Dynamic]:
        """Execute routing projection and selection."""
        with _phase("p3_route"):
            if self.owned_hidden_ref is not None:
                return _route_owned_rows(
                    self.owned_hidden_ref,
                    self.batch_size,
                    self.weights.routing_projection_vmem,
                    self.weights.routing_bias,
                    self.workspace.routing_scores_vmem,
                    self.workspace.selection_source_vmem,
                    self.workspace.selections_vmem,
                    self.send_semaphores,
                    self.receive_semaphores,
                    selected_expert_count=self.selected_expert_count,
                    local_expert_count=self.local_expert_count,
                    expert_tensor_parallel=self.expert_tensor_parallel,
                    on_local_selection=self._on_local_selection,
                    after_selection=self._after_selection,
                    after_scores=self.after_routing_scores,
                    live_rows=self.live_rows,
                )
            return _route_hidden_states(
                self.hidden,
                self.weights.routing_projection_vmem,
                self.weights.routing_bias,
                self.workspace.routing_scores_vmem,
                self.send_semaphores.at[0],
                self.receive_semaphores.at[0],
                selected_expert_count=self.selected_expert_count,
                local_expert_count=self.local_expert_count,
                expert_tensor_parallel=self.expert_tensor_parallel,
                on_local_selection=self._on_local_selection,
                after_selection=self._after_selection,
                after_scores=self.after_routing_scores,
                after_score_send=(
                    self._after_score_send if self.overlap_router_latent else None
                ),
                live_rows=self.live_rows,
            )

    def prepare_batch(
        self,
        latent_input_projection_vmem: _Dynamic,
        latent_input_copy_semaphore: _Dynamic,
        shared_gate_up_projection_vmem: _Dynamic,
        shared_down_projection_vmem: _Dynamic,
        dense_copy_semaphores: _Dynamic,
    ) -> None:
        """Execute prepare batch logic."""
        self.latent_input_projection_vmem = latent_input_projection_vmem
        self.shared_gate_up_projection_vmem = shared_gate_up_projection_vmem
        self.shared_down_projection_vmem = shared_down_projection_vmem
        self._start_prepare_batch_copies(
            latent_input_copy_semaphore,
            dense_copy_semaphores,
        )
        self.prefix_state = [jnp.int32(1), jnp.int32(0)]
        (
            selected_experts,
            selected_probabilities,
            local_route_counts,
            selected_probability_totals,
        ) = self._route_batch()
        self.selected_experts_vmem[...] = jnp.pad(
            selected_experts,
            ((0, 0), (0, 128 - self.selected_expert_count)),
        )
        self.selected_probabilities_vmem[...] = jnp.pad(
            selected_probabilities,
            ((0, 0), (0, 128 - self.selected_expert_count)),
        )
        self.local_route_counts_vmem[...] = jnp.broadcast_to(
            local_route_counts[:, None], self.local_route_counts_vmem.shape
        )
        self.probability_totals_vmem[...] = jnp.broadcast_to(
            selected_probability_totals[:, None],
            self.probability_totals_vmem.shape,
        )
        _start_initial_expert_weight_copies(
            selected_experts,
            local_route_counts,
            self.copy_expert,
            slot_count=self.expert_prefetch_slot_count,
            already_started=self.prefix_state[1],
            group_pending_routes=self.group_pending_expert_routes,
        )
        if not self.overlap_latent_gather:
            self.start_latent_gather(
                self.latent_input_projection_vmem,
                self.latent_input_copy,
            )
            self.wait_for_latent_gather()
        if self.overlap_shared_routing:
            self.dense_copies.down.wait()
        else:
            self._prepare_shared_compute()

    def encode_shared(self, value: _Dynamic, destination: _Dynamic) -> None:
        """Execute encode shared logic."""
        transport_batch_size = value.shape[0]
        packed = tpu.bitcast(
            value.reshape(transport_batch_size * self.shared_rows, 128).astype(
                jnp.bfloat16
            ),
            jnp.float32,
        )
        destination[: transport_batch_size * self.shared_payload, :] = packed.reshape(
            transport_batch_size * self.shared_payload, 256
        )[:, ::2]

    def decode_shared(self, value: _Dynamic) -> _Dynamic:
        """Execute decode shared logic."""
        transport_batch_size = value.shape[1] // self.shared_payload
        unpacked = tpu.bitcast(value, jnp.bfloat16).reshape(
            -1, transport_batch_size, self.shared_rows, 128
        )
        return jnp.sum(unpacked.astype(jnp.float32), axis=0).reshape(
            transport_batch_size, self.hidden_size
        )

    def encode_shared_shard(self, value: _Dynamic, destination: _Dynamic) -> None:
        """Execute encode shared shard logic."""
        transport_batch_size = value.shape[0]
        packed = tpu.bitcast(
            value.reshape(transport_batch_size * self.shared_shard_rows, 128).astype(
                jnp.bfloat16
            ),
            jnp.float32,
        )
        destination[: transport_batch_size * self.shared_shard_payload, :] = (
            packed.reshape(transport_batch_size * self.shared_shard_payload, 256)[
                :, ::2
            ]
        )

    def decode_shared_shard(self, value: _Dynamic) -> _Dynamic:
        """Execute decode shared shard logic."""
        transport_batch_size = value.shape[1] // self.shared_shard_payload
        unpacked = tpu.bitcast(value, jnp.bfloat16).reshape(
            -1, transport_batch_size, self.shared_shard_rows, 128
        )
        return jnp.sum(unpacked.astype(jnp.float32), axis=0).reshape(
            transport_batch_size, self.shared_shard_width
        )

    def encode_expert(self, value: _Dynamic, destination: _Dynamic) -> None:
        """Execute encode expert logic."""
        transport_batch_size = value.shape[0]
        packed = tpu.bitcast(
            value[:, : self.expert_output_size]
            .reshape(transport_batch_size * self.expert_rows, 128)
            .astype(jnp.bfloat16),
            jnp.float32,
        )
        destination[: transport_batch_size * self.expert_payload, :] = packed.reshape(
            transport_batch_size * self.expert_payload, 256
        )[:, ::2]

    def decode_expert(self, value: _Dynamic) -> _Dynamic:
        """Execute decode expert logic."""
        transport_batch_size = value.shape[1] // self.expert_payload
        unpacked = tpu.bitcast(value, jnp.bfloat16).reshape(
            -1, transport_batch_size, self.expert_rows, 128
        )
        return jnp.sum(unpacked.astype(jnp.float32), axis=0).reshape(
            transport_batch_size, self.expert_output_size
        )

    def decode_gathered_experts(self, value: _Dynamic) -> _Dynamic:
        """Execute decode gathered experts logic."""
        unpacked = tpu.bitcast(value, jnp.bfloat16).reshape(
            -1, self.tokens_per_lane, self.expert_rows, 128
        )
        return unpacked.astype(jnp.float32).reshape(-1, self.expert_output_size)[
            : self.batch_size, :
        ]

    def shared_copy(self, payload: _Dynamic, offset: _Dynamic) -> _Dynamic:
        """Execute shared copy logic."""
        return tpu.make_async_remote_copy(
            self.shared_local_reduction_vmem.at[self.rank % 8, :payload, :],
            self.shared_local_reduction_vmem.at[(self.rank ^ offset) % 8, :payload, :],
            self.shared_send_semaphores.at[offset - 1],
            self.shared_receive_semaphores.at[offset - 1],
            device_id=(self.rank ^ offset,),
            device_id_type=pl.DeviceIdType.MESH,
        )

    def shared_shard_send_view(self, target_lane: _Dynamic) -> _Dynamic:
        """Execute shared shard send view logic."""
        if self.routed_send_vmem is not None:
            return self.routed_send_vmem.reshape(8, self.shared_shard_payload, 128).at[
                target_lane
            ]
        return (
            self.shared_local_reduction_vmem.at[
                self.rank % 8, : 8 * self.shared_shard_payload, :
            ]
            .reshape(8, self.shared_shard_payload, 128)
            .at[target_lane]
        )

    def shared_shard_receive_view(self, source_lane: _Dynamic) -> _Dynamic:
        """Execute shared shard receive view logic."""
        return self.shared_local_reduction_vmem.at[
            source_lane, : self.shared_shard_payload, :
        ]

    def shared_shard_copy(self, offset: _Dynamic) -> _Dynamic:
        """Execute shared shard copy logic."""
        target_lane = (self.rank ^ offset) % 8
        source_lane = target_lane
        return tpu.make_async_remote_copy(
            self.shared_shard_send_view(target_lane),
            self.shared_shard_receive_view(source_lane),
            self.shared_send_semaphores.at[offset - 1],
            self.shared_receive_semaphores.at[offset - 1],
            device_id=(self.rank ^ offset,),
            device_id_type=pl.DeviceIdType.MESH,
        )

    def expert_copy(
        self, payload: _Dynamic, offset: _Dynamic, *, host: _Dynamic = False
    ) -> _Dynamic:
        """Execute expert copy logic."""
        if host:
            stride = 8
            send_vmem = self.expert_host_reduction_vmem
            receive_vmem = self.expert_host_reduction_vmem
            slot = self.rank // stride
        else:
            stride = 1
            send_vmem = self.expert_local_reduction_vmem
            receive_vmem = self.expert_local_reduction_vmem
            slot = self.rank % 8
        return tpu.make_async_remote_copy(
            send_vmem.at[slot, :payload, :],
            receive_vmem.at[(self.rank ^ (offset * stride)) // stride % 8, :payload, :],
            self.send_semaphores.at[self.reduction_phase[0] + offset - 1],
            self.receive_semaphores.at[self.reduction_phase[0] + offset - 1],
            device_id=(self.rank ^ (offset * stride),),
            device_id_type=pl.DeviceIdType.MESH,
        )

    def routed_send_storage(self) -> _Dynamic:
        """Return storage buffer for routed scatter send."""
        if self.routed_send_vmem is not None:
            return self.routed_send_vmem.at[: 8 * self.lane_payload, :]
        return self.expert_local_reduction_vmem.at[
            self.rank % 8, : 8 * self.lane_payload, :
        ]

    def expert_shard_send_view(self, target_lane: _Dynamic) -> _Dynamic:
        """Execute expert shard send view logic."""
        return (
            self.routed_send_storage()
            .reshape(8, self.lane_payload, 128)
            .at[target_lane]
        )

    def expert_shard_receive_view(self, source_lane: _Dynamic) -> _Dynamic:
        """Execute expert shard receive view logic."""
        return self.expert_local_reduction_vmem.at[source_lane, : self.lane_payload, :]

    def expert_shard_copy(self, offset: _Dynamic) -> _Dynamic:
        """Execute expert shard copy logic."""
        target_lane = (self.rank ^ offset) % 8
        source_lane = target_lane
        return tpu.make_async_remote_copy(
            self.expert_shard_send_view(target_lane),
            self.expert_shard_receive_view(source_lane),
            self.send_semaphores.at[self.reduction_phase[0] + offset - 1],
            self.receive_semaphores.at[self.reduction_phase[0] + offset - 1],
            device_id=(self.rank ^ offset,),
            device_id_type=pl.DeviceIdType.MESH,
        )

    def begin_shared_reduction(self, local_shared_outputs: _Dynamic) -> None:
        """Execute begin shared reduction logic."""
        if self.reduce_scatter_shared_output:
            if self.shared_sharded:
                if self.shared_down_weights is not None:
                    dense_down = self.shared_down_weights[...]
                else:
                    dense_down = self.weights.shared_down_projection_vmem[...]
                shards = _dot(local_shared_outputs, dense_down).reshape(
                    8, self.shared_shard_rows, 128
                )
            else:
                shards = local_shared_outputs.reshape(8, self.shared_shard_rows, 128)
            self.encode_shared_shard(shards, self.shared_shard_send_view(self.rank % 8))
            for offset in range(7, 0, -1):
                self.shared_shard_copy(offset).start()
            return

        payload = local_shared_outputs.shape[0] * self.shared_payload
        self.encode_shared(
            local_shared_outputs,
            self.shared_local_reduction_vmem.at[self.rank % 8],
        )
        for offset in range(7, 0, -1):
            self.shared_copy(payload, offset).start()

    def finish_shared_reduction(self, payload: _Dynamic) -> None:
        """Execute finish shared reduction logic."""
        if self.reduce_scatter_shared_output:
            lane = self.rank % 8
            for offset in range(7, 0, -1):
                self.shared_shard_copy(offset).wait()
            owned_shard = self.shared_shard_send_view(lane)[...]
            self.shared_shard_receive_view(lane)[...] = owned_shard
            self.encode_shared_shard(
                self.decode_shared_shard(
                    self.shared_local_reduction_vmem[:, : self.shared_shard_payload, :]
                ),
                self.shared_host_reduction_vmem.at[self.rank // 8],
            )
            return
        for offset in range(7, 0, -1):
            self.shared_copy(payload, offset).wait()
        self.encode_shared(
            self.decode_shared(self.shared_local_reduction_vmem[:, :payload, :]),
            self.shared_host_reduction_vmem.at[self.rank // 8],
        )

    def finish_shared_host_reduction(
        self, payload: _Dynamic, reduced_shared_outputs: _Dynamic
    ) -> None:
        """Execute finish shared host reduction logic."""
        if self.reduce_scatter_shared_output:
            reduced_shared_outputs[...] = jnp.zeros(
                reduced_shared_outputs.shape, jnp.float32
            )
            reduced_shared_outputs[...] = self.decode_shared_shard(
                self.shared_host_reduction_vmem[:, : self.shared_shard_payload, :]
            )
            return

        reduced_shared_outputs[...] = jnp.zeros(
            reduced_shared_outputs.shape, jnp.float32
        )
        reduced_shared_outputs[:, : self.hidden_size] = self.decode_shared(
            self.shared_host_reduction_vmem[:, :payload, :]
        )

    def _validate_reduction_payloads(
        self, shared_transport_payload: int, expert_transport_payload: int
    ) -> None:
        """Validate reduction buffer capacities."""
        if not (shared_transport_payload <= self.shared_local_reduction_vmem.shape[1]):
            _err_msg = (
                "Assertion failed: shared_transport_payload <= "
                "shared_local_reduction_vmem.shape[1]"
            )
            raise AssertionError(_err_msg)
        if not (
            self.reduce_scatter_routed_output
            or expert_transport_payload <= self.expert_local_reduction_vmem.shape[1]
        ):
            _err_msg = (
                "Assertion failed: reduce_scatter_routed_output or "
                "expert_transport_payload <= "
                "expert_local_reduction_vmem.shape[1]"
            )
            raise AssertionError(_err_msg)
        if not (self.routed_send_vmem is None or self.reduce_scatter_routed_output):
            _err_msg = (
                "Assertion failed: routed_send_vmem is None or "
                "reduce_scatter_routed_output"
            )
            raise AssertionError(_err_msg)
        if not (
            self.routed_consumed is not None or not self.reduce_scatter_routed_output
        ):
            _err_msg = (
                "Assertion failed: routed_consumed is not None or not "
                "reduce_scatter_routed_output"
            )
            raise AssertionError(_err_msg)

    def _routed_scatter_send(self, local_expert_outputs: _Dynamic) -> None:
        """Send local routed expert outputs for scatter reduction."""
        t1 = _phase_start("t1_routed_send")
        routed_scatter_storage = self.routed_send_storage()
        if self.batch_size < 8 * self.tokens_per_lane:
            routed_scatter_storage[...] = jnp.zeros(
                routed_scatter_storage.shape, jnp.float32
            )
        self.encode_expert(local_expert_outputs, routed_scatter_storage)
        for offset in range(7, 0, -1):
            self.expert_shard_copy(offset).start()
        _phase_end(t1)

    def _routed_scatter_local_exchange(self, lane: _Dynamic) -> None:
        """Execute lane-to-lane exchange and host reduction for routed scatter."""
        t3 = _phase_start("t3_routed_local")
        for offset in range(7, 0, -1):
            self.expert_shard_copy(offset).wait()
        owned_shard = self.expert_shard_send_view(lane)[...]
        self.expert_shard_receive_view(lane)[...] = owned_shard
        self.encode_expert(
            self.decode_expert(
                self.expert_local_reduction_vmem[:, : self.lane_payload, :]
            ),
            self.expert_host_reduction_vmem.at[self.rank // 8],
        )
        for offset in range(1, 8):
            pl.semaphore_signal(
                self.routed_consumed,
                1,
                device_id=(self.rank ^ offset,),
                device_id_type=pl.DeviceIdType.MESH,
            )
        _phase_end(t3)

        t4 = _phase_start("t4_routed_host")
        for offset in range(3, 0, -1):
            self.expert_copy(self.lane_payload, offset, host=bool(1)).start()
        for offset in range(3, 0, -1):
            self.expert_copy(self.lane_payload, offset, host=bool(1)).wait()
        _phase_end(t4)

    def _routed_scatter_gather(
        self,
        lane: _Dynamic,
        reduced_expert_outputs: _Dynamic,
        reduced_shared_outputs: _Dynamic,
        shared_transport_payload: _Dynamic,
    ) -> None:
        """Gather reduced routed expert outputs across lanes."""
        t5 = _phase_start("t5_routed_gather")
        reduced_token = self.decode_expert(
            self.expert_host_reduction_vmem[:, : self.lane_payload, :]
        )
        self.encode_expert(
            reduced_token,
            self.expert_local_reduction_vmem.at[lane],
        )
        pl.semaphore_wait(self.routed_consumed, 7)
        for offset in range(7, 0, -1):
            self.expert_copy(self.lane_payload, offset).start()
        if self.overlap_shared_reduction:
            with _phase("t6_shared_during_routed_gather"):
                self.finish_shared_host_reduction(
                    shared_transport_payload, reduced_shared_outputs
                )
        for offset in range(7, 0, -1):
            self.expert_copy(self.lane_payload, offset).wait()
        reduced_expert_outputs[:, : self.expert_output_size] = (
            self.decode_gathered_experts(
                self.expert_local_reduction_vmem[:, : self.lane_payload, :]
            )
        )
        _phase_end(t5)

        if not self.overlap_shared_reduction:
            t6 = _phase_start("t6_shared_host")
            self.finish_shared_host_reduction(
                shared_transport_payload, reduced_shared_outputs
            )
            _phase_end(t6)

    def _finish_routed_scatter_reductions(
        self,
        local_expert_outputs: _Dynamic,
        reduced_expert_outputs: _Dynamic,
        reduced_shared_outputs: _Dynamic,
        shared_transport_payload: _Dynamic,
    ) -> None:
        """Finish reduce-scatter and gather reductions for routed experts."""
        self._routed_scatter_send(local_expert_outputs)
        if self.after_experts is not None:
            self.after_experts()

        t2 = _phase_start("t2_shared_local")
        if not self.overlap_shared_reduction:
            self.finish_shared_reduction(shared_transport_payload)
        _phase_end(t2)

        lane = self.rank % 8
        self._routed_scatter_local_exchange(lane)
        self._routed_scatter_gather(
            lane,
            reduced_expert_outputs,
            reduced_shared_outputs,
            shared_transport_payload,
        )

    def _finish_routed_allgather_reductions(
        self,
        local_expert_outputs: _Dynamic,
        reduced_expert_outputs: _Dynamic,
        reduced_shared_outputs: _Dynamic,
        shared_transport_payload: _Dynamic,
        expert_transport_payload: _Dynamic,
    ) -> None:
        """Finish full all-gather reductions for routed experts."""
        self.encode_expert(
            local_expert_outputs,
            self.expert_local_reduction_vmem.at[self.rank % 8],
        )
        for offset in range(7, 0, -1):
            self.expert_copy(expert_transport_payload, offset).start()

        if self.after_experts is not None:
            self.after_experts()

        t2 = _phase_start("t2_shared_local")
        if not self.overlap_shared_reduction:
            self.finish_shared_reduction(shared_transport_payload)
        _phase_end(t2)

        for offset in range(7, 0, -1):
            self.expert_copy(expert_transport_payload, offset).wait()
        self.encode_expert(
            self.decode_expert(
                self.expert_local_reduction_vmem[:, :expert_transport_payload, :]
            ),
            self.expert_host_reduction_vmem.at[self.rank // 8],
        )
        for offset in range(3, 0, -1):
            self.expert_copy(expert_transport_payload, offset, host=bool(1)).start()
        for offset in range(3, 0, -1):
            self.expert_copy(expert_transport_payload, offset, host=bool(1)).wait()
        reduced_expert_outputs[...] = jnp.zeros(
            reduced_expert_outputs.shape, jnp.float32
        )
        reduced_expert_outputs[:, : self.expert_output_size] = self.decode_expert(
            self.expert_host_reduction_vmem[:, :expert_transport_payload, :]
        )
        self.finish_shared_host_reduction(
            shared_transport_payload, reduced_shared_outputs
        )

    def finish_reductions(
        self,
        local_expert_outputs: _Dynamic,
        reduced_expert_outputs: _Dynamic,
        reduced_shared_outputs: _Dynamic,
    ) -> None:
        """Execute finish reductions logic."""
        transport_batch_size = local_expert_outputs.shape[0]
        shared_transport_payload = transport_batch_size * (
            self.shared_shard_payload
            if self.reduce_scatter_shared_output
            else self.shared_payload
        )
        expert_transport_payload = transport_batch_size * self.expert_payload
        self._validate_reduction_payloads(
            shared_transport_payload, expert_transport_payload
        )

        self.latent_output_copy.start()
        if self.reduce_scatter_routed_output:
            self._finish_routed_scatter_reductions(
                local_expert_outputs,
                reduced_expert_outputs,
                reduced_shared_outputs,
                shared_transport_payload,
            )
        else:
            self._finish_routed_allgather_reductions(
                local_expert_outputs,
                reduced_expert_outputs,
                reduced_shared_outputs,
                shared_transport_payload,
                expert_transport_payload,
            )
        self.reduction_phase[0] += 2

    def _poll_shared(
        self,
        progress_state: _Dynamic,
        expected: _Dynamic,
        payload: _Dynamic,
        iteration: _Dynamic,
    ) -> None:
        """Poll shared DMA transfers for progress during expert loop."""

        @pl.when((progress_state[0] == 0) & (iteration % 4 == 0))
        def try_progress() -> None:
            """Check readiness of all shared reduction semaphores."""
            ready = jnp.bool_(bool(1))
            for peer in range(7):
                ready = ready & (
                    pl.semaphore_read(self.shared_send_semaphores.at[peer]) >= expected
                )
                ready = ready & (
                    pl.semaphore_read(self.shared_receive_semaphores.at[peer])
                    >= expected
                )

            @pl.when(ready)
            def advance() -> None:
                """Advance shared reduction when semaphores are ready."""
                with _phase("t2_shared_local_polled"):
                    self.finish_shared_reduction(payload)
                progress_state[0] = 1

    def _run_with_shared_progress(self, progress_state: _Dynamic) -> None:
        """Execute run with shared progress logic."""
        progress_state[0] = 0
        payload = self.batch_size * (
            self.shared_shard_payload
            if self.reduce_scatter_shared_output
            else self.shared_payload
        )
        expected = payload * 128 * 4 // 32

        def poll_shared(iteration: _Dynamic) -> None:
            """Poll shared progress callback."""
            self._poll_shared(progress_state, expected, payload, iteration)

        with _phase("p4_experts"):
            _run_routed_expert_stream(
                self.latent_vmem,
                self.selected_experts_vmem[:, : self.selected_expert_count],
                self.selected_probabilities_vmem[:, : self.selected_expert_count],
                self.local_route_counts_vmem[:, 0],
                self.probability_totals_vmem[:, 0],
                self.expert_gate_up_vmem,
                self.expert_gate_up_scale_vmem,
                self.expert_down_vmem,
                self.expert_down_scale_vmem,
                self.routed_outputs_vmem,
                self.gate_up_accumulator_vmem,
                self.down_accumulator_vmem,
                self.processed_routes_vmem,
                self.first_compact_source_vmem,
                self.second_compact_source_vmem,
                self.copy_expert,
                expert_input_size=self.expert_input_size,
                expert_width=self.expert_width,
                expert_output_size=self.expert_output_size,
                group_pending_routes=self.group_pending_expert_routes,
                progress=poll_shared,
            )

        @pl.when(progress_state[0] == 0)
        def drain_shared() -> None:
            """Drain shared reduction on completion of expert loop."""
            with _phase("t2_shared_local_drain"):
                self.finish_shared_reduction(payload)
            progress_state[0] = 1

    def _run_expert_stream(self) -> None:
        """Dispatch routed expert stream with or without shared progress."""
        if self.overlap_shared_reduction:
            pl.run_scoped(
                self._run_with_shared_progress,
                progress_state=tpu.SMEM((1,), jnp.int32),
            )
        else:
            with _phase("p4_experts"):
                _run_routed_expert_stream(
                    self.latent_vmem,
                    self.selected_experts_vmem[:, : self.selected_expert_count],
                    self.selected_probabilities_vmem[:, : self.selected_expert_count],
                    self.local_route_counts_vmem[:, 0],
                    self.probability_totals_vmem[:, 0],
                    self.expert_gate_up_vmem,
                    self.expert_gate_up_scale_vmem,
                    self.expert_down_vmem,
                    self.expert_down_scale_vmem,
                    self.routed_outputs_vmem,
                    self.gate_up_accumulator_vmem,
                    self.down_accumulator_vmem,
                    self.processed_routes_vmem,
                    self.first_compact_source_vmem,
                    self.second_compact_source_vmem,
                    self.copy_expert,
                    expert_input_size=self.expert_input_size,
                    expert_width=self.expert_width,
                    expert_output_size=self.expert_output_size,
                    group_pending_routes=self.group_pending_expert_routes,
                    progress=None,
                )

    def _finish_latent_and_output(self) -> None:
        """Execute latent up-projection, residual add, and output transport."""
        t7 = _phase_start("t7_latent_up")
        routed = self.routed_outputs_vmem[...].astype(jnp.bfloat16).astype(jnp.float32)
        normalized = (
            routed
            * jax.lax.rsqrt(
                jnp.sum(routed * routed, axis=1, keepdims=True)
                / self.latent_feature_count
                + self.epsilon
            )
            * self.weights.latent_normalization[...]
        ).astype(jnp.bfloat16)
        self.latent_output_copy.wait()
        up = _dot(
            normalized[:, : self.expert_output_size],
            self.weights.latent_output_projection_vmem[...],
        )

        local_output = (
            up
            + (
                self.shared_outputs_vmem[...]
                if self.shared_sharded
                else self.shared_outputs_vmem[:, pl.ds((self.rank % 8) * 1024, 1024)]
            )
        ).astype(jnp.bfloat16)
        _phase_end(t7)

        t8 = _phase_start("t8_output_transport")
        if self.token_scatter_output:
            self._transport_output_scatter(local_output)
        else:
            self._transport_output_gather(local_output)
        _phase_end(t8)
        self.reduction_phase[0] += 1

    def _transport_output_scatter(self, local_output: _Dynamic) -> None:
        """Scatter local outputs across peer owners."""
        owners = self.token_scatter_owners
        tokens_per_pair = self.batch_size // owners
        payload_rows = tokens_per_pair * 1024 // 128
        lane = self.rank % 8
        pair = lane // (8 // owners)
        send = self.output_vmem.at[pl.ds(0, owners)]
        receive = self.output_vmem.at[pl.ds(owners, 8)]
        for target_pair in range(owners):
            row_start = target_pair * tokens_per_pair
            send[target_pair] = local_output[
                row_start : row_start + tokens_per_pair
            ].reshape(payload_rows, 128)
        receive[lane] = send[pair]
        copies = []
        for offset in range(7, 0, -1):
            target_lane = lane ^ offset
            target_pair = target_lane // (8 // owners)
            copy = tpu.make_async_remote_copy(
                send.at[target_pair],
                receive.at[lane],
                self.output_send_semaphores.at[offset - 1],
                self.output_receive_semaphores.at[offset - 1],
                device_id=(self.rank ^ offset,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            copy.start()
            copies.append(copy)
        for copy in copies:
            copy.wait()

        for source_lane in range(7):
            block = receive[source_lane].reshape(tokens_per_pair, 1024)
            self.output_ref[:, pl.ds(source_lane * 1024, 1024)] = block

    def _transport_output_gather(self, local_output: _Dynamic) -> None:
        """Gather local output shards into output_ref."""
        self.output_vmem[:, pl.ds((self.rank % 8) * 1024, 1024)] = local_output
        _gather_output_shards(
            self.output_vmem,
            self.output_send_semaphores,
            self.output_receive_semaphores,
            self.reduction_phase[0] % 2,
        )
        self.output_ref[...] = self.output_vmem[
            : self.batch_size, : self.hidden_size
        ].astype(self.hidden_ref.dtype)

    def run(self) -> None:
        """Execute MoE layer pipeline."""
        if self.prepare_weights_vmem is not None:
            latent_down, shared_gate_up, shared_down = self.prepare_weights_vmem
            pl.run_scoped(
                lambda latent_sem, dense_sems: self.prepare_batch(
                    latent_down,
                    latent_sem,
                    shared_gate_up,
                    shared_down,
                    dense_sems,
                ),
                tpu.SemaphoreType.DMA,
                tpu.SemaphoreType.DMA((2,)),
            )
        else:
            pl.run_scoped(
                self.prepare_batch,
                tpu.VMEM(
                    self.weights.latent_input_projection.shape,
                    self.weights.latent_input_projection.dtype,
                ),
                tpu.SemaphoreType.DMA,
                tpu.VMEM(
                    self.weights.shared_gate_up_projection.shape,
                    self.weights.shared_gate_up_projection.dtype,
                ),
                tpu.VMEM(
                    self.weights.shared_down_projection.shape,
                    self.weights.shared_down_projection.dtype,
                ),
                tpu.SemaphoreType.DMA((2,)),
            )

        if self.shared_sharded:
            if not self.reduce_scatter_shared_output:
                _err_msg = "Assertion failed: reduce_scatter_shared_output"
                raise AssertionError(_err_msg)
            self.begin_shared_reduction(
                self.routed_outputs_vmem[:, : self.shared_width].astype(jnp.bfloat16)
            )
        else:
            self.begin_shared_reduction(self.shared_outputs_vmem[...])
        if self.overlap_latent_gather:
            self.wait_for_latent_gather()

        self._run_expert_stream()
        self.finish_reductions(
            self.routed_outputs_vmem[...],
            self.routed_outputs_vmem,
            self.shared_outputs_vmem,
        )
        self._finish_latent_and_output()


def moe(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Execute one mixture-of-experts layer inside a Pallas program.

    ``token_scatter_output`` leaves each of the ``token_scatter_owners`` local
    row owners (4: rank pairs; 8: ranks) with its batch // owners rows.
    ``owned_hidden_ref`` (this rank pair's rows of ``hidden_ref``) shards the
    router's top-k over the rank pairs (_route_owned_rows).

    ``live_rows`` masks null serving slots before route prefetch/compaction;
    the router scores and shared expert still use the compiled batch shape.

    ``prepare_weights_vmem`` optionally supplies caller-owned VMEM for the
    latent-down, shared gate/up and shared down projections instead of a
    scoped allocation.
    """
    return _MoeRunner(*_p_args, **_p_kw).run()


# Weight streaming. Every large head-sharded attention weight is copied
# through a ChunkedWeightStream; KernelConfig.weight_streams (JSON {name: [chunks,
# depth]}) splits one into chunks through a ring of ``depth`` slots. The
# default, one chunk, is exactly the whole-weight copy and product.
STREAMED_WEIGHTS = {
    # name: (axis, semaphore base in the stream semaphores when chunked)
    "k_projection": ("columns", 0),
    "k_o": ("rows", 4),
    "m_qb": ("columns", 0),
    "m_kb": ("columns", 4),
    "m_gate": ("columns", 8),
    "m_o": ("rows", 12),
    "m_qa": ("columns", 16),
    "shared_gu": ("columns", 20),
}
WEIGHT_STREAMS = {name: tuple(value) for name, value in _config().weight_streams}
if set(WEIGHT_STREAMS) - set(STREAMED_WEIGHTS):
    unknown = sorted(set(WEIGHT_STREAMS) - set(STREAMED_WEIGHTS))
    _err_msg = f"KernelConfig.weight_streams names unknown weights {unknown}"
    raise ValueError(_err_msg)


# Rings deeper than four slots (attention DP's small batches) signal from
# here, past every weight's own four: a KDA and an MLA layer's streams never
# overlap in time.
DEEP_STREAM_BASE = 24


def _stream_base(name: _Dynamic, depth: _Dynamic) -> _Dynamic:
    """Execute stream base logic."""
    return STREAMED_WEIGHTS[name][1] if depth <= _CONST_4 else DEEP_STREAM_BASE


def _stream_semaphore_count(streams: _Dynamic) -> int:
    """Stream semaphores for a configuration (each weight owns 4 from its.

    base). A minimum of 8 keeps the scalar schedule stable.
    """
    return max(
        [8]
        + [
            _stream_base(name, depth) + depth
            for name, (chunks, depth) in streams.items()
            if chunks > 1
        ]
    )


# B32/B64 paged DCP needs these chunks to leave room for communication scratch.
# These do not alter arithmetic or the resident weight representation.
DCP_B32_WEIGHT_STREAMS = {
    "m_qa": (12, 4),
    "k_projection": (13, 4),
    "shared_gu": (3, 3),
    "m_gate": (3, 2),
}
# Plain decode's input-projection rings fill the arena that layer 0's dense
# slabs size.
DCP_B32_DECODE_RINGS = {"m_qa": (12, 7), "k_projection": (13, 7)}


# Attention data parallelism: with DP groups of 32 / dp ranks, every rank holds
# the logical TP32 shards t, t + 32 / dp, ... (t = rank % (32 / dp)) of each
# head-sharded attention weight, concatenated along the axis it streams on
# ("concat", per-layer axis) or stacked on a new leading axis ("stack").
# Replicated weights (k_gate, m_qa, m_ka, norms) are unchanged.
ATTENTION_DP_WEIGHTS = {
    "k_projection": ("concat", 1),
    "k_o": ("concat", 0),
    "k_fb": ("concat", 1),
    "k_conv": ("stack", None),
    "k_a_log": ("stack", None),
    "k_dt": ("stack", None),
    "m_qb": ("concat", 1),
    "m_kb": ("concat", 1),
    "m_gate": ("concat", 1),
    "m_o": ("concat", 0),
}
# Stream defaults per attention-DP degree: shard-sized chunks through one slot,
# output projections as rank-pair halves (own halves first) through two slots.
DP_WEIGHT_STREAMS = {
    2: {
        # Whole shards through one slot; half-shard chunks through two slots
        # would need a wider slot.
        "k_projection": (2, 1),
        "k_o": (4, 2),
        # Shard chunks through one slot keep MLA's region-B share (q/kv-up
        # and gate) under the MoE shared-expert weights it alternates with.
        "m_qb": (2, 1),
        "m_kb": (2, 1),
        "m_gate": (2, 1),
        "m_o": (4, 2),
    },
    4: {
        # A TP8 KDA input projection is 91 MiB. Keep only one TP32
        # head-shard (22.75 MiB) resident and refill after each shard.
        "k_projection": (4, 1),
        "k_o": (8, 2),
        "m_qb": (4, 1),
        "m_kb": (4, 1),
        "m_gate": (4, 1),
        "m_o": (8, 2),
    },
}


def _dp_dcp_weight_streams(dp: int, group_rows: int) -> dict[str, tuple[int, int]]:
    """Attention DP under DCP: the KDA input projection (13 chunks per held.

    shard) and MLA q-down (12) stream so the DCP pages and packets fit, while
    key-value-up stays resident. Ring depths shrink as the group rows grow;
    with one group row (DP2 B2) MLA query-up and gate are also copied whole.
    """
    if dp != _CONST_2:
        depths = {8: (13, 12), 16: (7, 7)}.get(group_rows, (6, 4))
    elif group_rows <= _CONST_8:
        depths = (13, 12)
    else:
        depths = {16: (10, 9), 32: (7, 7)}.get(group_rows, (4, 4))
    streams = {
        "k_projection": (13 * dp, depths[0]),
        "m_qa": (12, depths[1]),
        "m_kb": (dp, dp),
    }
    if dp == _CONST_2 and group_rows == 1:
        streams.update(m_qb=(dp, dp), m_gate=(dp, dp))
    return streams


def _attention_projection_streams(dp: int, schedule: str) -> dict[str, tuple[int, int]]:
    """Static stream overrides for the opt-in attention-DP projection schedule.

    ``mla_up_double`` double-buffers MLA query-up and KV-up without changing
    the projection dot shapes; the gate projection stays single-buffered.
    """
    choices = ("default", "mla_up_double")
    if schedule not in choices:
        _err_msg = f"attention_projection_schedule must be one of {choices}"
        raise ValueError(_err_msg)
    if schedule == "default":
        return {}
    if dp not in (2, 4):
        _err_msg = "Projection schedule experiments require attention DP2/DP4"
        raise ValueError(_err_msg)
    return dict.fromkeys(("m_qb", "m_kb"), (dp, 2))


def _chunk_sizes(extent: int, chunks: int, unit: int) -> tuple[int, ...]:
    """Split ``extent`` into ``chunks`` whole ``unit`` blocks, larger first."""
    blocks, remainder = divmod(extent // unit, chunks)
    if extent % unit or not 1 <= chunks <= extent // unit:
        _err_msg = f"Cannot split {extent} into {chunks} chunks of whole {unit}-blocks"
        raise ValueError(_err_msg)
    return tuple(unit * (blocks + (index < remainder)) for index in range(chunks))


class ChunkedWeightStream:
    """A weight copied from HBM in chunks, through a ring of VMEM slots.

    ``source`` [K, N] is split along ``axis`` ("columns": output columns;
    "rows": the contraction) into ``sizes``. Position i of the stream holds
    chunk ``order(i)`` (default i; may be a traced value when all chunks have
    one size), in slot i % depth, signalling semaphore ``base + i % depth``.
    With ``depth`` equal to the chunk count the slots are the chunks' own
    parts of ``full`` (the weight's whole VMEM view); a smaller depth reuses
    slots inside ``full``'s pool range (``pool`` from byte ``offset``).

    ``start`` issues the first ``depth`` positions. Consumers ``wait`` for a
    position, read ``view``, then ``release`` it, which issues the position
    ``depth`` ahead into the freed slot; ``dot`` does this for column streams.
    Chunked products are not bit-identical to the whole-weight product (the
    compiler's MXU accumulation split depends on the shape).
    """

    def __getattr__(self, name: _Dynamic) -> _Dynamic:
        """Dynamic attribute access."""
        raise AttributeError(name)

    def __init__(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Execute init logic."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            source,
            full,
            semaphores,
        ) = (
            _p_next(_p_iter, _p_get("source")),
            _p_next(_p_iter, _p_get("full")),
            _p_next(_p_iter, _p_get("semaphores")),
        )
        (
            axis,
            sizes,
            depth,
            base,
            pool,
            offset,
            order,
        ) = (
            _p_get("axis"),
            _p_get("sizes"),
            _p_get("depth"),
            _p_get("base", 0),
            _p_get("pool"),
            _p_get("offset"),
            _p_get("order"),
        )
        if axis not in ("columns", "rows"):
            _err_msg = "Stream axis must be 'columns' or 'rows'"
            raise ValueError(_err_msg)
        self.axis, self.sizes, self.depth, self.chunks = (
            axis,
            tuple(sizes),
            depth,
            len(sizes),
        )
        self.shards = 1
        extent = source.shape[1 if axis == "columns" else 0]
        if sum(self.sizes) != extent:
            _err_msg = "Chunk sizes must cover the streamed axis"
            raise ValueError(_err_msg)
        if not 1 <= depth <= self.chunks or base + depth > semaphores.shape[0]:
            _err_msg = "Stream depth must fit the chunks and the semaphores"
            raise ValueError(_err_msg)
        if order is not None and len(set(self.sizes)) != 1:
            _err_msg = "A reordered stream needs equal chunk sizes"
            raise ValueError(_err_msg)
        self.source, self.full, self.semaphores, self.base, self.order = (
            source,
            full,
            semaphores,
            base,
            order,
        )
        self.slots = None
        if depth < self.chunks:
            if pool is None:
                _err_msg = "A ring stream needs its weight's pool range"
                raise ValueError(_err_msg)
            size = max(self.sizes)
            shape = (
                (full.shape[0], size) if axis == "columns" else (size, full.shape[1])
            )
            slot_bytes = pool_alias.storage_size(shape, full.dtype)
            if depth * slot_bytes > pool_alias.storage_size(full.shape, full.dtype):
                _err_msg = "Ring slots exceed the weight's pool range"
                raise ValueError(_err_msg)
            self.slots = tuple(
                pool_alias.view(
                    pool.at[
                        pl.ds((offset + slot * slot_bytes) // 128, slot_bytes // 128), :
                    ],
                    shape,
                    full.dtype,
                )
                for slot in range(depth)
            )

    def _chunk(self, position: _Dynamic) -> _Dynamic:
        """Execute chunk logic."""
        return position if self.order is None else self.order(position)

    def _extent(self, position: _Dynamic) -> _Dynamic:
        """Execute extent logic."""
        chunk = self._chunk(position)
        if self.order is None:
            return sum(self.sizes[:chunk]), self.sizes[chunk]
        return chunk * self.sizes[0], self.sizes[0]

    def _part(self, ref: _Dynamic, start: _Dynamic, size: _Dynamic) -> _Dynamic:
        """Execute part logic."""
        if self.axis == "columns":
            return ref.at[:, pl.ds(start, size)]
        return ref.at[pl.ds(start, size)]

    def view(self, position: _Dynamic) -> _Dynamic:
        """Execute view logic."""
        start, size = self._extent(position)
        if self.chunks == 1:
            return self.full
        if self.slots is None:
            return self._part(self.full, start, size)
        return self._part(self.slots[position % self.depth], 0, size)

    def _copy(self, position: _Dynamic) -> _Dynamic:
        """Execute copy logic."""
        start, size = self._extent(position)
        source = (
            self.source if self.chunks == 1 else self._part(self.source, start, size)
        )
        return tpu.make_async_copy(
            source,
            self.view(position),
            self.semaphores.at[self.base + position % self.depth],
        )

    def start(self) -> _Dynamic:
        """Execute start logic."""
        for position in range(self.depth):
            self._copy(position).start()

    def wait(self, position: _Dynamic) -> _Dynamic:
        """Execute wait logic."""
        self._copy(position).wait()

    def wait_all(self) -> _Dynamic:
        """Execute wait all logic."""
        if self.slots is not None:
            _err_msg = "Only a stream with a slot per chunk can wait for all chunks"
            raise ValueError(_err_msg)
        for position in range(self.chunks):
            self.wait(position)

    def release(self, position: _Dynamic, guard: _Dynamic) -> _Dynamic:
        """Refill the slot freed by ``position``. ``guard`` is a runtime-true.

        condition that keeps the refill in its own region, after the loads of
        the chunk it overwrites.
        """
        if position + self.depth < self.chunks:

            @pl.when(guard)
            def refill() -> _Dynamic:
                """Execute refill logic."""
                self._copy(position + self.depth).start()

    def dot_position(
        self, x: _Dynamic, position: _Dynamic, guard: _Dynamic
    ) -> _Dynamic:
        """``x`` times one column chunk (FP32): wait, multiply, release."""
        if self.axis != "columns":
            _err_msg = "dot_position is for column streams"
            raise ValueError(_err_msg)
        self.wait(position)
        result = _dot(x, self.view(position)[...])
        self.release(position, guard)
        return result

    def dot(self, x: _Dynamic, guard: _Dynamic) -> _Dynamic:
        """``x @ source`` in FP32 for a column stream."""
        if self.axis != "columns":
            _err_msg = "dot is for column streams"
            raise ValueError(_err_msg)
        if self.slots is None:
            # A DMA wait is a scheduling boundary: with a slot per chunk,
            # resolve every transfer before the products so they overlap.
            self.wait_all()
        parts = []
        for position in range(self.chunks):
            if self.slots is not None:
                self.wait(position)
            parts.append(_dot(x, self.view(position)[...]))
            self.release(position, guard)
        return parts[0] if self.chunks == 1 else jnp.concatenate(parts, axis=1)


class KdaStateStream:
    """KDA state streamed through three rolling or four private VMEM buffers.

    Normally buffer c % 3 is reused only after its write has landed. B32's
    unrolled path uses four buffers: prime chunks 0/1 before projection,
    then start chunks 2/3 at recurrence entry. Final writes drain on exit.
    With slots, read each sequence's initial state directly from the bank;
    subsequent verifier rows are produced by the recurrence before use.
    """

    def __getattr__(self, name: _Dynamic) -> _Dynamic:
        """Dynamic attribute access."""
        raise AttributeError(name)

    def __init__(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
        """Execute init logic."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            self,
            history_hbm,
            state_hbm,
            history_output_hbm,
            state_output_hbm,
            history_buffers,
            state_buffers,
            read_semaphores,
            write_semaphores,
        ) = (
            _p_next(_p_iter, _p_get("self")),
            _p_next(_p_iter, _p_get("history_hbm")),
            _p_next(_p_iter, _p_get("state_hbm")),
            _p_next(_p_iter, _p_get("history_output_hbm")),
            _p_next(_p_iter, _p_get("state_output_hbm")),
            _p_next(_p_iter, _p_get("history_buffers")),
            _p_next(_p_iter, _p_get("state_buffers")),
            _p_next(_p_iter, _p_get("read_semaphores")),
            _p_next(_p_iter, _p_get("write_semaphores")),
        )
        (
            chunk,
            chunks,
            slots,
            sequence_rows,
        ) = (
            _p_get("chunk"),
            _p_get("chunks"),
            _p_get("slots"),
            _p_get("sequence_rows", 1),
        )
        self.history_hbm, self.state_hbm = history_hbm, state_hbm
        self.history_output_hbm, self.state_output_hbm = (
            history_output_hbm,
            state_output_hbm,
        )
        self.history_buffers, self.state_buffers = history_buffers, state_buffers
        self.read_semaphores, self.write_semaphores = read_semaphores, write_semaphores
        self.chunk, self.chunks = chunk, chunks
        self.slots = slots
        self.sequence_rows = sequence_rows
        self.pending_writes = []
        self.buffer_count = history_buffers.shape[0]
        self.four_buffers = self.buffer_count == _CONST_4
        if not (
            not self.four_buffers
            or (self.chunks == _CONST_4 and self.chunk == _CONST_8 and (not KDA_STAGED))
        ):
            _err_msg = (
                "Assertion failed: not self.four_buffers or (self.chunks == 4 "
                "and self.chunk == 8 and (not KDA_STAGED))"
            )
            raise AssertionError(_err_msg)

    def _reads(self, index: _Dynamic) -> _Dynamic:
        """Execute reads logic."""
        rows, slot = pl.ds(index * self.chunk, self.chunk), index % self.buffer_count
        if self.slots is not None:
            return tuple(
                tpu.make_async_copy(
                    source.at[pl.ds(self.slots[index * self.chunk + row], 1)],
                    buffer.at[slot, pl.ds(row, 1)],
                    self.read_semaphores.at[slot, kind],
                )
                for kind, (source, buffer) in enumerate(
                    (
                        (self.history_hbm, self.history_buffers),
                        (self.state_hbm, self.state_buffers),
                    )
                )
                for row in range(0, self.chunk, self.sequence_rows)
            )
        return (
            tpu.make_async_copy(
                self.history_hbm.at[rows],
                self.history_buffers.at[slot],
                self.read_semaphores.at[slot, 0],
            ),
            tpu.make_async_copy(
                self.state_hbm.at[rows],
                self.state_buffers.at[slot],
                self.read_semaphores.at[slot, 1],
            ),
        )

    def _writes(self, index: _Dynamic) -> _Dynamic:
        """Execute writes logic."""
        rows, slot = pl.ds(index * self.chunk, self.chunk), index % self.buffer_count
        return (
            tpu.make_async_copy(
                self.history_buffers.at[slot],
                self.history_output_hbm.at[rows],
                self.write_semaphores.at[slot, 0],
            ),
            tpu.make_async_copy(
                self.state_buffers.at[slot],
                self.state_output_hbm.at[rows],
                self.write_semaphores.at[slot, 1],
            ),
        )

    def prime(self) -> _Dynamic:
        """Execute prime logic."""
        for index in range(min(2, self.chunks)):
            for copy in self._reads(index):
                copy.start()

    def acquire(self, index: _Dynamic) -> _Dynamic:
        """Execute acquire logic."""
        if self.four_buffers and index == 0:
            for tail in range(2, self.chunks):
                for copy in self._reads(tail):
                    copy.start()
        for copy in self._reads(index):
            copy.wait()
        slot = index % self.buffer_count
        return self.history_buffers.at[slot], self.state_buffers.at[slot]

    def release(self, index: _Dynamic) -> _Dynamic:
        """Execute release logic."""
        for copy in self._writes(index):
            copy.start()
        self.pending_writes.append(index)
        if self.four_buffers:
            return
        ahead = index + 2
        if ahead < self.chunks:
            if ahead >= _CONST_3:
                # Buffer ahead % 3 last held chunk ahead - 3.
                for copy in self._writes(ahead - 3):
                    copy.wait()
                self.pending_writes.remove(ahead - 3)
            for copy in self._reads(ahead):
                copy.start()

    def finish(self) -> _Dynamic:
        """Execute finish logic."""
        for index in list(self.pending_writes):
            for copy in self._writes(index):
                copy.wait()
        self.pending_writes.clear()

    # Traced-index variants for a rolled chunk loop: chunk c's writes are
    # drained when chunk c + 3 is read into its buffer, the last three in
    # finish_rolled.
    def acquire_rolled(self, index: _Dynamic) -> _Dynamic:
        """Execute acquire rolled logic."""
        for copy in self._reads(index):
            copy.wait()
        slot = index % 3
        return self.history_buffers.at[slot], self.state_buffers.at[slot]

    def release_rolled(self, index: _Dynamic) -> _Dynamic:
        """Execute release rolled logic."""
        for copy in self._writes(index):
            copy.start()
        ahead = index + 2

        @pl.when(ahead < self.chunks)
        def refill() -> _Dynamic:
            """Execute refill logic."""

            @pl.when(ahead >= _CONST_3)
            def drain() -> _Dynamic:
                """Execute drain logic."""
                for copy in self._writes(ahead - 3):
                    copy.wait()

            for copy in self._reads(ahead):
                copy.start()

    def finish_rolled(self) -> _Dynamic:
        """Execute finish rolled logic."""
        for index in range(max(0, self.chunks - 3), self.chunks):
            for copy in self._writes(index):
                copy.wait()


class KdaStateStreamSlice:
    """One head shard's rows in a shared, flattened KDA state stream.

    The caller primes the parent once and drains it after all head shards. Local
    KDA calls use shard-local chunk indices and finish calls, while the
    parent keeps its lookahead and buffer ownership across shard boundaries.
    """

    def __getattr__(self, name: _Dynamic) -> _Dynamic:
        """Dynamic attribute access."""
        raise AttributeError(name)

    def __init__(self, parent: _Dynamic, *, first_chunk: int, chunks: int) -> None:
        """Execute init logic."""
        if isinstance(first_chunk, int) and (
            first_chunk < 0 or chunks < 1 or first_chunk + chunks > parent.chunks
        ):
            _err_msg = "A state-stream slice must fit within its parent"
            raise ValueError(_err_msg)
        self.parent = parent
        self.first_chunk, self.chunks = first_chunk, chunks
        self.chunk, self.slots = parent.chunk, parent.slots

    def acquire(self, index: _Dynamic) -> _Dynamic:
        """Execute acquire logic."""
        with _phase("t1_kda_state_read_wait"):
            return self.parent.acquire(self.first_chunk + index)

    def release(self, index: _Dynamic) -> _Dynamic:
        """Execute release logic."""
        self.parent.release(self.first_chunk + index)

    def acquire_rolled(self, index: _Dynamic) -> _Dynamic:
        """Execute acquire rolled logic."""
        with _phase("t1_kda_state_read_wait"):
            return self.parent.acquire_rolled(self.first_chunk + index)

    def release_rolled(self, index: _Dynamic) -> _Dynamic:
        """Execute release rolled logic."""
        self.parent.release_rolled(self.first_chunk + index)

    def finish(self) -> _Dynamic:
        # The next shard can already have reads in flight in these buffers.
        # Only the owner of the flattened stream can drain its final writes.
        """Execute finish logic."""

    def finish_rolled(self) -> _Dynamic:
        """Execute finish rolled logic."""


def _shard_dot(
    stream: _Dynamic, x: _Dynamic, shard: _Dynamic, guard: _Dynamic
) -> _Dynamic:
    """``x`` times one held shard's columns of a column stream (FP32): the.

    shard's chunks in order, concatenated.
    """
    per_shard = stream.chunks // stream.shards
    parts = [
        stream.dot_position(x, shard * per_shard + index, guard)
        for index in range(per_shard)
    ]
    return parts[0] if per_shard == 1 else jnp.concatenate(parts, axis=1)


def _traced_dot_position(
    stream: _Dynamic,
    x_ref: _Dynamic,
    position: _Dynamic,
    guard: _Dynamic,
    out_ref: _Dynamic,
) -> _Dynamic:
    """``dot_position`` at a traced ``position`` of a ring column stream with.

    equal chunks: wait, multiply ``x_ref`` by the chunk into ``out_ref``
    (BF16), refill the slot ``depth`` positions ahead.
    """
    if stream.slots is None or stream.order is not None or len(set(stream.sizes)) != 1:
        _err_msg = "A traced stream position needs a ring of equal chunks"
        raise ValueError(_err_msg)
    size = stream.sizes[0]
    slot = position % stream.depth

    def copy(at: _Dynamic, destination: _Dynamic, semaphore: _Dynamic) -> _Dynamic:
        """Execute copy logic."""
        source = stream.source.at[:, pl.ds(pl.multiple_of(at * size, 128), size)]
        return tpu.make_async_copy(source, destination, semaphore)

    for index, destination in enumerate(stream.slots):

        @pl.when(slot == index)
        def consume(
            destination: _Dynamic = destination,
            semaphore: _Dynamic = stream.semaphores.at[stream.base + index],
        ) -> _Dynamic:
            """Execute consume logic."""
            copy(position, destination, semaphore).wait()
            out_ref[...] = _dot(x_ref[...], destination[...]).astype(jnp.bfloat16)

            @pl.when(guard & (position + stream.depth < stream.chunks))
            def refill() -> _Dynamic:
                """Execute refill logic."""
                copy(position + stream.depth, destination, semaphore).start()


@dataclasses.dataclass(frozen=True, slots=True)
class KimiDeltaAttentionWeights:
    """KDA parameters and their persistent projection-copy resources."""

    gate_projection: TensorLike  # [hidden_size, 128]
    gate_projection_vmem: _Dynamic  # [hidden_size, 128]
    input_projection: TensorLike  # [hidden_size, 1664]
    input_projection_vmem: _Dynamic  # [hidden_size, 1664]

    feature_gate_projection: TensorLike  # [128, 384]
    feature_gate_projection_vmem: _Dynamic  # [128, 384]
    # [384, hidden_size], or [768, hidden_size / 2] after paired-rank repartitioning.
    output_projection: TensorLike
    output_projection_vmem: _Dynamic  # [384, hidden_size] or [768, hidden_size / 2]
    convolution_filter: TensorLike  # [4, 3, 3, 128]
    exp_log_decay_rate: TensorLike  # [3, 1]
    time_step_bias: TensorLike  # [3, 128]
    output_normalization: TensorLike  # [128] or [8, 128]
    projection_copy_semaphores: _Dynamic  # [4]

    def __post_init__(self) -> None:
        """Execute post init logic."""
        _ensure_dataclasses_registered()

    def make_projection_copies(self) -> SimpleNamespace:
        """Create the asynchronous projection-weight copies."""
        return SimpleNamespace(
            gate=tpu.make_async_copy(
                self.gate_projection,
                self.gate_projection_vmem,
                self.projection_copy_semaphores.at[0],
            ),
            feature_gate=tpu.make_async_copy(
                self.feature_gate_projection,
                self.feature_gate_projection_vmem,
                self.projection_copy_semaphores.at[1],
            ),
            input=tpu.make_async_copy(
                self.input_projection,
                self.input_projection_vmem,
                self.projection_copy_semaphores.at[2],
            ),
            output=tpu.make_async_copy(
                self.output_projection,
                self.output_projection_vmem,
                self.projection_copy_semaphores.at[3],
            ),
        )


def _kda_row_chunk(batch_size: int, attention_dp: int = 1) -> int:
    """Rows per KDA recurrence chunk above B8 (smaller batches run whole).

    TP32 B128 and above use four: eight-row chunks exceed the compiler's VMEM
    although _vmem_budget accepts them. Attention DP's B128 peaks outside KDA
    and fits eight.
    """
    return (
        8
        if batch_size < _CONST_128 or (attention_dp > 1 and batch_size == _CONST_128)
        else 4
    )


class _KdaRunner:
    """Runner for Kimi Delta Attention layer execution."""

    def __getattr__(self, name: _Dynamic) -> _Dynamic:
        """Dynamic attribute access."""
        raise AttributeError(name)

    def __init__(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Initialize KDA runner."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            self.hidden_ref,
            self.output_ref,
        ) = (
            _p_next(_p_iter, _p_get("hidden_ref")),
            _p_next(_p_iter, _p_get("output_ref")),
        )
        self.updated_convolution_history_ref = next(
            _p_iter, _p_get("updated_convolution_history_ref")
        )
        self.updated_recurrent_state_ref = next(
            _p_iter, _p_get("updated_recurrent_state_ref")
        )
        (
            self.weights,
            self.convolution_history_ref,
            self.recurrent_state_ref,
            self.communication_workspace,
            self.epsilon,
            self.decay_lower_bound,
            self.synchronize_devices,
            self.weights_are_prefetched,
            self.token_scatter_output,
            self.sequence_rows,
            self.wait_for_state,
            self.on_state_updated,
            streams,
            self.shard_index,
            self.attended_out,
            self.gate_features_vmem,
            self.state_stream,
            stage_projections,
            self.projected_rows,
            first_call,
        ) = (
            _p_get("weights"),
            _p_get("convolution_history_ref"),
            _p_get("recurrent_state_ref"),
            _p_get("communication_workspace"),
            _p_get("epsilon", 1e-05),
            _p_get("decay_lower_bound", -5.0),
            _p_kw.get("synchronize_devices", True),
            _p_kw.get("weights_are_prefetched", False),
            _p_kw.get("token_scatter_output", False),
            _p_get("sequence_rows", 1),
            _p_get("wait_for_state"),
            _p_get("on_state_updated"),
            _p_get("streams"),
            _p_get("shard_index"),
            _p_get("attended_out"),
            _p_get("gate_features_vmem"),
            _p_get("state_stream"),
            _p_kw.get("stage_projections", False),
            _p_get("projected_rows"),
            _p_get("first_call"),
        )
        self.streams = streams or {}
        self.first_shard = (
            self.shard_index in (None, 0) if first_call is None else first_call
        )
        self.traced_shard = self.shard_index is not None and not isinstance(
            self.shard_index, int
        )
        if (self.shard_index is None) != (self.attended_out is None):
            _err_msg = "Per-shard KDA calls must take attended_out"
            raise ValueError(_err_msg)
        self.batch_size, self.hidden_size = self.hidden_ref.shape
        self.head_count, self.channels_per_head = 3, 128
        self.attention_width = self.head_count * self.channels_per_head
        self._validate_shapes()

        if self.synchronize_devices:
            collectives32.barrier()
            self.communication_workspace.reduction_phase[0] = 0

        self.staged = (
            (stage_projections or (KDA_STAGED and self.shard_index is None))
            and self.batch_size > _CONST_8
            and self.sequence_rows == 1
        )
        self.projection_width = (
            1664
            if self.shard_index is not None
            else self.weights.input_projection.shape[1]
        )
        self.rolled = self.staged and self.state_stream is not None
        self.row_chunk = (
            self.state_stream.chunk
            if self.state_stream is not None
            else _kda_row_chunk(self.batch_size)
        )
        self.chunk_count = self.batch_size // self.row_chunk

    def _validate_shapes(self) -> None:
        """Validate input and state tensor shapes."""
        state_batch = (
            self.convolution_history_ref.shape[0]
            if self.state_stream is not None and self.state_stream.slots is not None
            else self.batch_size
        )
        state_shape = (
            state_batch,
            self.head_count,
            self.channels_per_head,
            self.channels_per_head,
        )
        history_shape = (state_batch, 3, 3, CONV_HEAD_ROWS, self.channels_per_head)
        w_in_ok = (
            self.shard_index is not None
            or self.weights.input_projection.shape[1] == _CONST_1664
        )
        if (
            self.output_ref.shape != self.hidden_ref.shape
            or self.weights.gate_projection.shape != (self.hidden_size, 128)
            or self.weights.input_projection.shape[0] != self.hidden_size
            or self.weights.input_projection.shape[1] % 1664
            or not w_in_ok
            or self.weights.exp_log_decay_rate.shape != (3, 1)
            or self.convolution_history_ref.shape != history_shape
            or self.recurrent_state_ref.shape != state_shape
        ):
            _err_msg = "KDA inputs and state must match the input batch"
            raise ValueError(_err_msg)
        if self.batch_size % self.sequence_rows:
            _err_msg = "sequence_rows must divide the KDA batch"
            raise ValueError(_err_msg)

    def _start_weight_copies(self) -> _Dynamic:
        """Create and initiate asynchronous copies for weight matrices."""
        weight_copies = self.weights.make_projection_copies()
        if not self.weights_are_prefetched:
            weight_copies.gate.start()
            weight_copies.feature_gate.start()
            for name in ("input", "output"):
                if name in self.streams:
                    self.streams[name].start()
                else:
                    getattr(weight_copies, name).start()
        return weight_copies

    def _compute_gate_features(self, weight_copies: _Dynamic) -> _Dynamic:
        """Compute gate features projection."""
        if self.first_shard:
            weight_copies.gate.wait()
        if self.first_shard or self.gate_features_vmem is None:
            gate_features = _dot(
                self.hidden_ref[...], self.weights.gate_projection_vmem[...]
            ).astype(jnp.bfloat16)
            if self.gate_features_vmem is not None:
                self.gate_features_vmem[:, :128] = gate_features
        else:
            gate_features = self.gate_features_vmem[:, :128]
        return gate_features

    def _compute_feature_gate(
        self,
        gate_features: _Dynamic,
        gate_vmem: _Dynamic,
        weight_copies: _Dynamic,
    ) -> _Dynamic:
        """Compute feature gate projections into VMEM or register array."""
        if self.first_shard:
            weight_copies.feature_gate.wait()
        feature_gate_vmem = self.weights.feature_gate_projection_vmem
        if self.shard_index is not None and not self.traced_shard:
            feature_gate_vmem = feature_gate_vmem.at[
                :, pl.ds(self.shard_index * self.attention_width, self.attention_width)
            ]
        if (
            self.rolled
            and self.traced_shard
            and feature_gate_vmem.shape[1] > 2 * self.attention_width
        ):
            raw_gate = _dot(gate_features, feature_gate_vmem[...])
            selected = raw_gate[:, : self.attention_width]
            for held in range(1, feature_gate_vmem.shape[1] // self.attention_width):
                selected = jnp.where(
                    self.shard_index == held,
                    raw_gate[
                        :,
                        held * self.attention_width : (held + 1) * self.attention_width,
                    ],
                    selected,
                )
            gate_vmem[...] = selected.astype(jnp.bfloat16).reshape(
                self.chunk_count, self.row_chunk, -1
            )
            return None
        if self.rolled:
            gate_vmem[...] = (
                _dot(gate_features, feature_gate_vmem[...])
                .astype(jnp.bfloat16)
                .reshape(self.chunk_count, self.row_chunk, -1)
            )
            return None
        if self.staged:
            gate_vmem[...] = _dot(gate_features, feature_gate_vmem[...]).astype(
                jnp.bfloat16
            )
            return None
        raw_recurrent_gate = _dot(gate_features, feature_gate_vmem[...])
        if self.traced_shard:
            raw_recurrent_gate = jnp.where(
                self.shard_index == 0,
                raw_recurrent_gate[:, : self.attention_width],
                raw_recurrent_gate[:, self.attention_width :],
            )
        return (
            raw_recurrent_gate.astype(jnp.bfloat16)
            .reshape(self.batch_size, self.head_count, self.channels_per_head)
            .astype(jnp.float32)
        )

    def _compute_projected_input(self, weight_copies: _Dynamic) -> _Dynamic:
        """Compute input projections."""
        if self.projected_rows is not None:
            return self.projected_rows
        if self.shard_index is not None:
            return _shard_dot(
                self.streams["input"],
                self.hidden_ref[...],
                self.shard_index,
                self.communication_workspace.reduction_phase[0] >= 0,
            ).astype(jnp.bfloat16)
        if "input" in self.streams:
            return (
                self.streams["input"]
                .dot(
                    self.hidden_ref[...],
                    self.communication_workspace.reduction_phase[0] >= 0,
                )
                .astype(jnp.bfloat16)
            )
        weight_copies.input.wait()
        return _dot(
            self.hidden_ref[...], self.weights.input_projection_vmem[...]
        ).astype(jnp.bfloat16)

    def _setup_projected_fns(
        self,
        projected: _Dynamic,
        projected_vmem: _Dynamic,
        gate_vmem: _Dynamic,
        output_gate_vmem: _Dynamic,
        raw_recurrent_gate: _Dynamic,
    ) -> tuple[_Dynamic, _Dynamic, _Dynamic, _Dynamic]:
        """Construct row accessors for queries, keys, values, update rate, and decay."""
        if self.rolled:
            projected_vmem[...] = projected.reshape(
                self.chunk_count, self.row_chunk, -1
            )
            output_gate_vmem[...] = projected[
                :, 3 * self.attention_width : 4 * self.attention_width
            ]
            qkv = _RowFunction(
                lambda index: projected_vmem[index][
                    :, : 3 * self.attention_width
                ].reshape(self.row_chunk, 3, self.head_count, self.channels_per_head)
            )
            update_rate = _RowFunction(
                lambda index: projected_vmem[index][
                    :,
                    4 * self.attention_width : 4 * self.attention_width
                    + self.head_count,
                ]
                .reshape(self.row_chunk, self.head_count, 1)
                .astype(jnp.float32)
            )
            decay = _RowFunction(
                lambda index: self.decay_lower_bound
                * jax.nn.sigmoid(
                    jnp.exp(self.weights.exp_log_decay_rate[...])[None]
                    * (
                        gate_vmem[index]
                        .reshape(
                            self.row_chunk, self.head_count, self.channels_per_head
                        )
                        .astype(jnp.float32)
                        + self.weights.time_step_bias[...][None]
                    )
                )
            )
            if self.wait_for_state is not None:
                self.wait_for_state()
            return qkv, update_rate, decay, None

        def _row_span(rows: _Dynamic) -> _Dynamic:
            """Compute row count for slice or batch size."""
            if (
                isinstance(rows, slice)
                and rows.start is not None
                and rows.stop is not None
            ):
                return rows.stop - rows.start
            return self.batch_size

        if self.staged:
            projected_vmem[...] = projected
            qkv = _RowFunction(
                lambda rows: projected_vmem[rows, : 3 * self.attention_width].reshape(
                    _row_span(rows),
                    3,
                    self.head_count,
                    self.channels_per_head,
                )
            )
            update_rate = _RowFunction(
                lambda rows: projected_vmem[
                    rows,
                    4 * self.attention_width : 4 * self.attention_width
                    + self.head_count,
                ]
                .reshape(_row_span(rows), self.head_count, 1)
                .astype(jnp.float32)
            )
            decay = _RowFunction(
                lambda rows: self.decay_lower_bound
                * jax.nn.sigmoid(
                    jnp.exp(self.weights.exp_log_decay_rate[...])[None]
                    * (
                        gate_vmem[rows]
                        .reshape(
                            _row_span(rows),
                            self.head_count,
                            self.channels_per_head,
                        )
                        .astype(jnp.float32)
                        + self.weights.time_step_bias[...][None]
                    )
                )
            )
            if self.wait_for_state is not None:
                self.wait_for_state()
            return qkv, update_rate, decay, None

        qkv = _RowFunction(
            lambda rows: projected[rows, : 3 * self.attention_width].reshape(
                _row_span(rows),
                3,
                self.head_count,
                self.channels_per_head,
            )
        )
        projected_tail = projected[:, 3 * self.attention_width :]
        output_gate = projected_tail[:, : self.attention_width].reshape(
            self.batch_size, self.head_count, self.channels_per_head
        )
        aw = self.attention_width
        update_rate = (
            projected_tail[:, aw : aw + self.head_count]
            .reshape(self.batch_size, self.head_count, 1)
            .astype(jnp.float32)
        )
        if self.wait_for_state is not None:
            self.wait_for_state()
        if not (raw_recurrent_gate is not None):
            _err_msg = "Assertion failed: raw_recurrent_gate is not None"
            raise AssertionError(_err_msg)
        decay = self.decay_lower_bound * jax.nn.sigmoid(
            jnp.exp(self.weights.exp_log_decay_rate[...])[None]
            * (raw_recurrent_gate + self.weights.time_step_bias[...][None])
        )
        return qkv, update_rate, decay, output_gate

    def _step_recurrence(
        self,
        rows: _Dynamic,
        *args: _Dynamic,
        **kwargs: _Dynamic,
    ) -> _Dynamic:
        """Run one chunk of recurrence for single sequence_rows."""
        if len(args) == 1 and isinstance(args[0], tuple):
            qkv, update_rate, decay = args[0]
            history_ref = kwargs.get("history_ref")
            state_ref = kwargs.get("state_ref")
        elif len(args) >= _CONST_3:
            qkv, update_rate, decay = args[0], args[1], args[2]
            history_ref = (
                args[_CONST_3] if len(args) > _CONST_3 else kwargs.get("history_ref")
            )
            state_ref = (
                args[_CONST_4] if len(args) > _CONST_4 else kwargs.get("state_ref")
            )
        else:
            qkv = kwargs.get("qkv")
            update_rate = kwargs.get("update_rate")
            decay = kwargs.get("decay")
            history_ref = kwargs.get("history_ref")
            state_ref = kwargs.get("state_ref")
        qkv, update_rate, decay, state_ref = (
            _dyn(qkv),
            _dyn(update_rate),
            _dyn(decay),
            _dyn(state_ref),
        )
        if history_ref is None:
            convolution_window = self.convolution_history_ref[
                rows, :, :, : self.head_count
            ]
            recurrent_state = self.recurrent_state_ref[rows].astype(jnp.float32)
        else:
            convolution_window = history_ref[:, :, :, : self.head_count]
            recurrent_state = state_ref[...].astype(jnp.float32)
        concatenated_window = jnp.concatenate(
            (convolution_window, qkv[rows][:, None]), axis=1
        )
        filtered_qkv = jax.nn.silu(
            jnp.sum(
                concatenated_window.astype(jnp.float32)
                * self.weights.convolution_filter[...][None, :, : self.head_count],
                axis=1,
            )
        ).astype(jnp.bfloat16)
        query = filtered_qkv[:, 0].astype(jnp.float32)
        key = filtered_qkv[:, 1].astype(jnp.float32)
        value = filtered_qkv[:, 2].astype(jnp.float32)
        query *= (
            jax.lax.rsqrt(jnp.sum(query * query, axis=-1, keepdims=True) + 1e-6)
            * self.channels_per_head**-0.5
        )
        key *= jax.lax.rsqrt(jnp.sum(key * key, axis=-1, keepdims=True) + 1e-6)
        next_recurrent_state = recurrent_state * jnp.exp(decay[rows])[..., None]
        predicted_value = jnp.sum(key[..., None] * next_recurrent_state, axis=2)
        state_update = jax.nn.sigmoid(update_rate[rows]) * (value - predicted_value)
        next_recurrent_state += key[..., None] * state_update[..., None, :]
        attended_rows = jnp.sum(query[..., None] * next_recurrent_state, axis=2)
        if history_ref is None:
            self.updated_convolution_history_ref[rows, :, :, : self.head_count] = (
                concatenated_window[:, 1:]
            )
            self.updated_recurrent_state_ref[rows] = next_recurrent_state.astype(
                self.updated_recurrent_state_ref.dtype
            )
        else:
            history_ref[:, :, :, : self.head_count] = concatenated_window[:, 1:]
            state_ref[...] = next_recurrent_state.astype(state_ref.dtype)
        return attended_rows

    def _run_chunked_recurrence(
        self,
        qkv: _Dynamic,
        update_rate: _Dynamic,
        decay: _Dynamic,
        attended_vmem: _Dynamic,
    ) -> None:
        """Run chunked recurrence across multiple batch regions."""
        chunk = self.row_chunk
        if self.rolled:

            @pl.loop(0, self.chunk_count)
            def run_rolled_chunk(index: _Dynamic) -> _Dynamic:
                """Execute run rolled chunk logic."""
                history_chunk, state_chunk = self.state_stream.acquire_rolled(index)
                attended_vmem[pl.ds(pl.multiple_of(index * chunk, chunk), chunk)] = (
                    self._step_recurrence(
                        index, qkv, update_rate, decay, history_chunk, state_chunk
                    )
                )
                self.state_stream.release_rolled(index)

            self.state_stream.finish_rolled()
            return

        for index, first in enumerate(range(0, self.batch_size, chunk)):

            @pl.when(self.communication_workspace.reduction_phase[0] >= 0)
            def run_chunk(index: _Dynamic = index, first: _Dynamic = first) -> _Dynamic:
                """Execute run chunk logic."""
                if self.state_stream is None:
                    attended_vmem[first : first + chunk] = self._step_recurrence(
                        slice(first, first + chunk), qkv, update_rate, decay
                    )
                else:
                    history_chunk, state_chunk = self.state_stream.acquire(index)
                    attended_vmem[first : first + chunk] = self._step_recurrence(
                        slice(first, first + chunk),
                        qkv,
                        update_rate,
                        decay,
                        history_chunk,
                        state_chunk,
                    )
                    self.state_stream.release(index)

    def _run_recurrence_loop(
        self,
        qkv: _Dynamic,
        update_rate: _Dynamic,
        decay: _Dynamic,
        attended_vmem: _Dynamic,
    ) -> _Dynamic:
        """Execute recurrence over batch chunks."""
        if self.batch_size <= _CONST_8:
            if self.state_stream is None:
                return self._step_recurrence(slice(None), (qkv, update_rate, decay))
            if not (
                self.state_stream.chunks == 1
                and self.state_stream.chunk == self.batch_size
            ):
                _err_msg = (
                    "Assertion failed: state_stream.chunks == 1 and "
                    "state_stream.chunk == batch_size"
                )
                raise AssertionError(_err_msg)
            history_chunk, state_chunk = self.state_stream.acquire(0)
            attended = self._step_recurrence(
                slice(None),
                qkv,
                update_rate,
                decay,
                history_chunk,
                state_chunk,
            )
            self.state_stream.release(0)
            self.state_stream.finish()
            return attended

        self._run_chunked_recurrence(qkv, update_rate, decay, attended_vmem)

        if (
            self.state_stream is not None
            and not self.rolled
            and self.batch_size != _CONST_32
        ):
            if not (self.state_stream is not None):
                _err_msg = "Assertion failed: state_stream is not None"
                raise AssertionError(_err_msg)
            self.state_stream.finish()
        return attended_vmem[...]

    def _step_sequence(
        self,
        first: _Dynamic,
        state_pair: tuple[_Dynamic, _Dynamic],
        store: _Dynamic,
        recurrence_data: tuple[_Dynamic, _Dynamic, _Dynamic, _Dynamic],
    ) -> list[_Dynamic]:
        """Run one sequence of tokens for multiple sequence_rows."""
        history, state = state_pair
        qkv, update_rate, decay, conv_filter = recurrence_data
        """Run one sequence of tokens for multiple sequence_rows."""
        window = jnp.concatenate(
            (history, qkv[first : first + self.sequence_rows]),
            axis=0,
        )
        state = state.astype(jnp.float32)
        rows = []
        for offset in range(self.sequence_rows):
            row = first + offset
            filtered = jax.nn.silu(
                jnp.sum(
                    window[offset : offset + 4].astype(jnp.float32) * conv_filter,
                    axis=0,
                )
            ).astype(jnp.bfloat16)
            query = filtered[0].astype(jnp.float32)
            key = filtered[1].astype(jnp.float32)
            value = filtered[2].astype(jnp.float32)
            query *= (
                jax.lax.rsqrt(jnp.sum(query * query, axis=-1, keepdims=True) + 1e-6)
                * self.channels_per_head**-0.5
            )
            key *= jax.lax.rsqrt(jnp.sum(key * key, axis=-1, keepdims=True) + 1e-6)
            state = state * jnp.exp(decay[row])[..., None]
            predicted_value = jnp.sum(key[..., None] * state, axis=1)
            state_update = jax.nn.sigmoid(update_rate[row]) * (value - predicted_value)
            state = state + key[..., None] * state_update[:, None, :]
            rows.append(jnp.sum(query[..., None] * state, axis=1)[None])
            saved_state = state.astype(self.recurrent_state_ref.dtype)
            store(offset, window[offset + 1 : offset + 4], saved_state)
            state = saved_state.astype(jnp.float32)
        return rows

    def _run_sequence_loop(
        self,
        qkv: _Dynamic,
        update_rate: _Dynamic,
        decay: _Dynamic,
        attended_vmem: _Dynamic,
    ) -> _Dynamic:
        """Execute sequence recurrence over sequences and row chunks."""
        conv_filter = self.weights.convolution_filter[...].astype(jnp.float32)
        if self.state_stream is None:
            attended_rows = []
            for first in range(0, self.batch_size, self.sequence_rows):

                def store(
                    offset: _Dynamic,
                    history: _Dynamic,
                    state: _Dynamic,
                    first: _Dynamic = first,
                ) -> _Dynamic:
                    """Execute store logic."""
                    self.updated_convolution_history_ref[
                        first + offset, :, :, : self.head_count
                    ] = history
                    self.updated_recurrent_state_ref[first + offset] = state

                attended_rows += self._step_sequence(
                    first,
                    (
                        self.convolution_history_ref[first, :, :, : self.head_count],
                        self.recurrent_state_ref[first],
                    ),
                    store,
                    (qkv, update_rate, decay, conv_filter),
                )
            return jnp.concatenate(attended_rows, axis=0)

        chunk = self.state_stream.chunk
        if chunk % self.sequence_rows:
            _err_msg = "Streamed KDA chunks must hold whole sequences"
            raise ValueError(_err_msg)
        for index, base in enumerate(range(0, self.batch_size, chunk)):

            @pl.when(self.communication_workspace.reduction_phase[0] >= 0)
            def run_sequence_chunk(
                index: _Dynamic = index, base: _Dynamic = base
            ) -> _Dynamic:
                """Execute run sequence chunk logic."""
                history_chunk, state_chunk = self.state_stream.acquire(index)
                rows = []
                for local in range(0, chunk, self.sequence_rows):

                    def store(
                        offset: _Dynamic,
                        history: _Dynamic,
                        state: _Dynamic,
                        local: _Dynamic = local,
                    ) -> _Dynamic:
                        """Execute store logic."""
                        history_chunk[local + offset, :, :, : self.head_count] = history
                        state_chunk[local + offset] = state

                    rows += self._step_sequence(
                        base + local,
                        (
                            history_chunk[local, :, :, : self.head_count],
                            state_chunk[local],
                        ),
                        store,
                        (qkv, update_rate, decay, conv_filter),
                    )
                attended_vmem[base : base + chunk] = jnp.concatenate(rows, axis=0)
                self.state_stream.release(index)

        if self.batch_size != _CONST_32:
            self.state_stream.finish()
        return attended_vmem[...]

    def _apply_output_gate(
        self,
        attended: _Dynamic,
        output_gate: _Dynamic,
        output_gate_vmem: _Dynamic,
        projected_vmem: _Dynamic,
    ) -> _Dynamic:
        """Normalize attended features and multiply with sigmoid output gate."""
        attended = attended.astype(jnp.bfloat16).astype(jnp.float32)
        attended *= jax.lax.rsqrt(
            jnp.mean(attended * attended, axis=-1, keepdims=True) + self.epsilon
        )
        normalization = (
            self.weights.output_normalization[...]
            if len(self.weights.output_normalization.shape) == 1
            else self.weights.output_normalization[0]
        )
        if self.rolled:
            output_gate = output_gate_vmem[...].reshape(
                self.batch_size, self.head_count, self.channels_per_head
            )
        elif self.staged:
            output_gate = projected_vmem[
                :, 3 * self.attention_width : 4 * self.attention_width
            ].reshape(self.batch_size, self.head_count, self.channels_per_head)
        if not (output_gate is not None):
            _err_msg = "Assertion failed: output_gate is not None"
            raise AssertionError(_err_msg)
        attended = (attended * normalization).astype(jnp.bfloat16) * jax.nn.sigmoid(
            output_gate.astype(jnp.float32)
        ).astype(jnp.bfloat16)
        if self.on_state_updated is not None:
            self.on_state_updated()
        return attended

    def _project_output(
        self,
        attended: _Dynamic,
        reduction_vmem: _Dynamic,
        weight_copies: _Dynamic,
    ) -> None:
        """Project attended features to output reference."""
        if self.attended_out is not None:
            self.attended_out[...] = attended.reshape(
                self.batch_size, self.attention_width
            )
            return
        output_stream = self.streams.get("output")
        if output_stream is None:
            weight_copies.output.wait()
        elif output_stream.chunks == 1:
            output_stream.wait_all()
            output_stream = None
        with _phase("p2_attn_project_reduce"):
            _project_attention_2d(
                attended.reshape(self.batch_size, self.attention_width),
                self.weights.output_projection_vmem,
                self.output_ref,
                reduction_vmem,
                self.communication_workspace.shared_local_reduction_vmem,
                self.communication_workspace.shared_host_reduction_vmem,
                self.communication_workspace.attention_features_vmem,
                self.communication_workspace.attention_output_vmem,
                self.communication_workspace.send_semaphores,
                self.communication_workspace.receive_semaphores,
                self.communication_workspace.reduction_phase,
                _dot,
                token_scatter_output=self.token_scatter_output,
                weight_stream=output_stream,
                staging=self.communication_workspace.projection_staging_vmem,
                gather_wire=self.communication_workspace.attention_gather_vmem,
            )

    def _run_scoped_body(
        self,
        reduction_vmem: _Dynamic,
        attended_vmem: _Dynamic = None,
        projected_vmem: _Dynamic = None,
        gate_vmem: _Dynamic = None,
        output_gate_vmem: _Dynamic = None,
    ) -> _Dynamic:
        """Execute run logic within scoped VMEM allocations."""
        kda_scope = [_phase_start("q3_kda_projections")]
        weight_copies = self._start_weight_copies()
        gate_features = self._compute_gate_features(weight_copies)
        raw_recurrent_gate = self._compute_feature_gate(
            gate_features, gate_vmem, weight_copies
        )
        projected = self._compute_projected_input(weight_copies)
        qkv, update_rate, decay, output_gate = self._setup_projected_fns(
            projected,
            projected_vmem,
            gate_vmem,
            output_gate_vmem,
            raw_recurrent_gate,
        )
        del projected

        _phase_end(kda_scope.pop())
        kda_scope.append(_phase_start("q4_kda_recurrence"))
        if self.sequence_rows == 1:
            attended = self._run_recurrence_loop(qkv, update_rate, decay, attended_vmem)
        else:
            attended = self._run_sequence_loop(qkv, update_rate, decay, attended_vmem)
        _phase_end(kda_scope.pop())
        kda_scope.append(_phase_start("q5_kda_output_gate"))
        attended = self._apply_output_gate(
            attended, output_gate, output_gate_vmem, projected_vmem
        )
        _phase_end(kda_scope.pop())

        self._project_output(attended, reduction_vmem, weight_copies)

    def run(self) -> _Dynamic:
        """Execute KDA layer with scoped VMEM buffers."""
        pl.run_scoped(
            self._run_scoped_body,
            tpu.VMEM(
                (self.batch_size, 3584 if self.batch_size > _CONST_32 else 8192),
                jnp.float32,
            ),
            *(
                (
                    tpu.VMEM(
                        (
                            self.batch_size,
                            self.head_count,
                            self.channels_per_head,
                        ),
                        jnp.float32,
                    ),
                )
                if self.batch_size > _CONST_8
                and (self.sequence_rows == 1 or self.state_stream is not None)
                else ()
            ),
            *(
                (
                    tpu.VMEM(
                        (self.chunk_count, self.row_chunk, self.projection_width),
                        jnp.bfloat16,
                    ),
                    tpu.VMEM(
                        (self.chunk_count, self.row_chunk, self.attention_width),
                        jnp.bfloat16,
                    ),
                    tpu.VMEM(
                        (self.batch_size, self.attention_width),
                        jnp.bfloat16,
                    ),
                )
                if self.rolled
                else (
                    tpu.VMEM((self.batch_size, self.projection_width), jnp.bfloat16),
                    tpu.VMEM(
                        (self.batch_size, self.attention_width),
                        jnp.bfloat16,
                    ),
                )
                if self.staged
                else ()
            ),
        )
        if (
            self.state_stream is not None
            and not self.rolled
            and self.batch_size == _CONST_32
        ):
            if not (self.state_stream is not None):
                _err_msg = "Assertion failed: state_stream is not None"
                raise AssertionError(_err_msg)
            self.state_stream.finish()


def kda(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Execute one Kimi Delta Attention layer inside a Pallas program."""
    return _KdaRunner(*_p_args, **_p_kw).run()


@dataclasses.dataclass(frozen=True, slots=True)
class MultiHeadLatentAttentionWeights:
    """MLA parameters and their persistent projection-copy resources."""

    query_down_projection: TensorLike  # [hidden_size, 1536]
    query_down_projection_vmem: _Dynamic  # [hidden_size, 1536]
    key_value_down_projection: TensorLike  # [hidden_size, 640]
    key_value_down_projection_vmem: _Dynamic  # [hidden_size, 640]

    query_up_projection: TensorLike  # [1536, 768]
    query_up_projection_vmem: _Dynamic  # [1536, 768]
    key_value_up_projection: TensorLike  # [512, 768]
    key_value_up_projection_vmem: _Dynamic  # [512, 768]
    output_gate_projection: TensorLike  # [hidden_size, 384]
    output_gate_projection_vmem: _Dynamic  # [hidden_size, 384]
    # [384, hidden_size], or [768, hidden_size / 2] after paired-rank repartitioning.
    output_projection: TensorLike
    output_projection_vmem: _Dynamic  # [384, hidden_size] or [768, hidden_size / 2]
    query_normalization: TensorLike  # [1, 1536]
    key_value_normalization: TensorLike  # [1, 512]
    projection_copy_semaphores: _Dynamic  # [6]

    def make_projection_copies(self) -> SimpleNamespace:
        """Create the asynchronous projection-weight copies."""
        return SimpleNamespace(
            query_down=tpu.make_async_copy(
                self.query_down_projection,
                self.query_down_projection_vmem,
                self.projection_copy_semaphores.at[0],
            ),
            key_value_down=tpu.make_async_copy(
                self.key_value_down_projection,
                self.key_value_down_projection_vmem,
                self.projection_copy_semaphores.at[1],
            ),
            query_up=tpu.make_async_copy(
                self.query_up_projection,
                self.query_up_projection_vmem,
                self.projection_copy_semaphores.at[2],
            ),
            key_value_up=tpu.make_async_copy(
                self.key_value_up_projection,
                self.key_value_up_projection_vmem,
                self.projection_copy_semaphores.at[3],
            ),
            output_gate=tpu.make_async_copy(
                self.output_gate_projection,
                self.output_gate_projection_vmem,
                self.projection_copy_semaphores.at[4],
            ),
            output=tpu.make_async_copy(
                self.output_projection,
                self.output_projection_vmem,
                self.projection_copy_semaphores.at[5],
            ),
        )


PAGE_TOKENS = 256  # tokens per paged MLA cache page (two 128-token read tiles)
MLA_LAYERS = 24


@dataclasses.dataclass(frozen=True, slots=True)
class MultiHeadLatentAttentionWorkspace:
    """Persistent VMEM and semaphores used by one MLA invocation."""

    # Unused placeholders ([8, 128] and one DMA semaphore): the DCP attention
    # keeps its pages in mla_dcp's workspace.
    key_cache_vmem: _Dynamic
    value_cache_vmem: _Dynamic
    cache_copy_semaphores: _Dynamic
    query_vmem: _Dynamic  # [batch_size, 3, 256]
    output_gate_values_vmem: _Dynamic  # [batch_size, 384]
    new_key_vmem: _Dynamic  # [batch_size, 3, 256]
    new_value_vmem: _Dynamic  # [batch_size, 3, 128]
    maximum_vmem: _Dynamic  # [batch_size, 3, 1]
    denominator_vmem: _Dynamic  # [batch_size, 3, 1]
    accumulator_vmem: _Dynamic  # [batch_size, 3, 128]


_DATACLASSES_REGISTERED_BOX = [False]


def _ensure_dataclasses_registered() -> None:
    """Execute ensure dataclasses registered logic."""
    if _DATACLASSES_REGISTERED_BOX[0]:
        return
    _DATACLASSES_REGISTERED_BOX[0] = True
    jax.tree_util.register_dataclass(_PaddedBf16Rows)
    jax.tree_util.register_dataclass(AttentionCommunicationWorkspace)
    jax.tree_util.register_dataclass(MixtureOfExpertsWeights)
    jax.tree_util.register_dataclass(MixtureOfExpertsWorkspace)
    jax.tree_util.register_dataclass(KimiDeltaAttentionWeights)
    jax.tree_util.register_dataclass(MultiHeadLatentAttentionWeights)
    jax.tree_util.register_dataclass(MultiHeadLatentAttentionWorkspace)


_ensure_dataclasses_registered()


class _MlaRunner:
    """Runner for multi-head latent-attention layer execution."""

    def __getattr__(self, name: _Dynamic) -> _Dynamic:
        """Dynamic attribute access."""
        raise AttributeError(name)

    def __init__(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Initialize MLA runner."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            self.position_ref,
            self.hidden_ref,
            self.output_ref,
            self.new_key_ref,
            self.new_value_ref,
        ) = (
            _p_next(_p_iter, _p_get("position_ref")),
            _p_next(_p_iter, _p_get("hidden_ref")),
            _p_next(_p_iter, _p_get("output_ref")),
            _p_next(_p_iter, _p_get("new_key_ref")),
            _p_next(_p_iter, _p_get("new_value_ref")),
        )
        (
            self.weights,
            self.workspace,
            self.communication_workspace,
            self.dcp,
            self.epsilon,
            self.synchronize_devices,
            self.weights_are_prefetched,
            self.token_scatter_output,
            self.sequence_rows,
        ) = (
            _p_get("weights"),
            _p_get("workspace"),
            _p_get("communication_workspace"),
            _p_get("dcp"),
            _p_get("epsilon", 1e-05),
            _p_kw.get("synchronize_devices", True),
            _p_kw.get("weights_are_prefetched", False),
            _p_kw.get("token_scatter_output", False),
            _p_get("sequence_rows", 1),
        )
        self.output_projection_after_query_down = _p_kw.get(
            "output_projection_after_query_down", False
        )
        self.streams = _p_get("streams") or {}
        self.batch_size, self.hidden_size = self.hidden_ref.shape
        self._validate_shapes()
        self.weight_copies = self.weights.make_projection_copies()

    def _validate_shapes(self) -> None:
        """Validate MLA tensor shapes and configurations."""
        context_length = (
            self.dcp[0].page_tokens
            * self.dcp[0].pages
            * (1 if self.dcp[0].global_pages else self.dcp[0].group)
        )
        if not 1 <= self.batch_size <= MAX_BATCH:
            _err_msg = f"MLA supports 1 to {MAX_BATCH} batch rows"
            raise ValueError(_err_msg)
        if self.output_ref.shape != self.hidden_ref.shape:
            _err_msg = "MLA output must match the hidden-state shape"
            raise ValueError(_err_msg)
        if (
            not 1 <= self.sequence_rows <= self.batch_size
            or self.batch_size % self.sequence_rows
        ):
            _err_msg = "sequence_rows must divide the MLA batch"
            raise ValueError(_err_msg)
        if self.position_ref.shape != (self.batch_size,):
            _err_msg = "Positions must match the MLA batch size"
            raise ValueError(_err_msg)
        invalid_weights = (
            self.weights.query_down_projection.shape != (self.hidden_size, 1536)
            or self.weights.key_value_down_projection.shape != (self.hidden_size, 640)
            or self.weights.query_up_projection.shape != (1536, 768)
            or self.weights.key_value_up_projection.shape != (512, 768)
            or self.weights.output_gate_projection.shape != (self.hidden_size, 384)
            or self.weights.query_normalization.shape != (1, 1536)
            or self.weights.key_value_normalization.shape != (1, 512)
            or context_length < _CONST_384
            or context_length % 128
        )
        if invalid_weights:
            _err_msg = "MLA projection and normalization shapes are invalid"
            raise ValueError(_err_msg)

    def start_copy(self, name: _Dynamic) -> None:
        """Start async copy for named projection."""
        if name in self.streams:
            self.streams[name].start()
        else:
            getattr(self.weight_copies, name).start()

    def projection(self, name: _Dynamic, x: _Dynamic) -> _Dynamic:
        """Multiply x by named projection in FP32."""
        guard = self.communication_workspace.reduction_phase[0] >= 0
        if name in self.streams:
            return self.streams[name].dot(x, guard)
        getattr(self.weight_copies, name).wait()
        return _dot(x, getattr(self.weights, f"{name}_projection_vmem")[...])

    def _prepare_projections(self) -> tuple[_Dynamic, _Dynamic]:
        """Compute query and key/value latents."""
        if not self.weights_are_prefetched:
            self.weight_copies.query_down.start()
            self.weight_copies.key_value_down.start()
        self.start_copy("query_up")
        self.start_copy("key_value_up")
        self.start_copy("output_gate")
        if not self.output_projection_after_query_down:
            self.start_copy("output")

        guard = self.communication_workspace.reduction_phase[0] >= 0
        if "query_down" in self.streams:
            query_latent = (
                self.streams["query_down"]
                .dot(self.hidden_ref[...], guard)
                .astype(jnp.bfloat16)
                .astype(jnp.float32)
            )
        else:
            self.weight_copies.query_down.wait()
            query_latent = (
                _dot(self.hidden_ref[...], self.weights.query_down_projection_vmem[...])
                .astype(jnp.bfloat16)
                .astype(jnp.float32)
            )

        pl.when(guard & self.output_projection_after_query_down)(
            lambda: self.start_copy("output")
        )

        query_latent = (
            query_latent
            * jax.lax.rsqrt(
                jnp.mean(query_latent * query_latent, axis=1, keepdims=True)
                + self.epsilon
            )
            * self.weights.query_normalization[...]
        ).astype(jnp.bfloat16)
        if "query_up" in self.streams:
            self.workspace.query_vmem[...] = (
                self.streams["query_up"].dot(query_latent, guard).astype(jnp.bfloat16)
            )
        else:
            self.weight_copies.query_up.wait()
            self.workspace.query_vmem[...] = _dot(
                query_latent, self.weights.query_up_projection_vmem[...]
            ).astype(jnp.bfloat16)

        self.weight_copies.key_value_down.wait()
        compressed = _dot(
            self.hidden_ref[...], self.weights.key_value_down_projection_vmem[...]
        ).astype(jnp.bfloat16)
        key_value_latent = compressed[:, :512].astype(jnp.float32)
        key_value_latent = (
            key_value_latent
            * jax.lax.rsqrt(
                jnp.mean(key_value_latent * key_value_latent, axis=1, keepdims=True)
                + self.epsilon
            )
            * self.weights.key_value_normalization[...]
        ).astype(jnp.bfloat16)
        new_latent = jnp.concatenate(
            (
                key_value_latent,
                compressed[:, 512:576],
                jnp.zeros((self.batch_size, 64), jnp.bfloat16),
            ),
            axis=1,
        )
        self.new_key_ref[:, 0] = new_latent[:, :256]
        self.new_key_ref[:, 1] = new_latent[:, 256:512]
        self.new_key_ref[:, 2] = jnp.concatenate(
            (new_latent[:, 512:], jnp.zeros((self.batch_size, 128), jnp.bfloat16)),
            axis=1,
        )
        self.new_value_ref[:, :3] = jnp.zeros((self.batch_size, 3, 128), jnp.bfloat16)
        return query_latent, new_latent

    def _attend(self, new_latent: _Dynamic) -> _Dynamic:
        """Run DCP attention over latent-space KV cache."""
        dcp_skip = _config().dcp_skip
        dcp_cache, dcp_workspace = self.dcp[0], self.dcp[1]
        dcp_attention = mla_dcp.DcpAttention(
            dcp_cache,
            dcp_workspace,
            self.position_ref,
            self.batch_size,
            sequence_rows=self.sequence_rows,
            skip=dcp_skip,
            cache_verify_rows=self.sequence_rows > 1
            and self.dcp[0].dtype == jnp.float8_e4m3fn,
        )
        dcp_attention.prime()

        self.workspace.maximum_vmem[...] = jnp.full(
            (self.batch_size, 3, 1), -jnp.inf, jnp.float32
        )
        self.workspace.denominator_vmem[...] = jnp.zeros(
            (self.batch_size, 3, 1), jnp.float32
        )
        self.workspace.accumulator_vmem[...] = jnp.zeros(
            (self.batch_size, 3, 128), jnp.float32
        )

        if "key_value_up" in self.streams:
            self.streams["key_value_up"].wait_all()
        else:
            self.weight_copies.key_value_up.wait()
        up = self.weights.key_value_up_projection_vmem
        query = self.workspace.query_vmem[...]
        scale = 192**-0.5
        absorbed = []
        for head in range(3):
            latent_query = jax.lax.dot_general(
                query[:, head, :128],
                up[:, head * 256 : head * 256 + 128],
                (((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            absorbed.append(
                jnp.concatenate(
                    (
                        latent_query,
                        query[:, head, 128:192].astype(jnp.float32),
                        jnp.zeros((self.batch_size, 64), jnp.float32),
                    ),
                    axis=1,
                )
                * scale
            )

        def output_gate() -> None:
            """Project output gate."""
            self.workspace.output_gate_values_vmem[...] = self.projection(
                "output_gate", self.hidden_ref[...]
            ).astype(jnp.bfloat16)

        merged = dcp_attention.attend(
            jnp.stack(absorbed, axis=1),
            new_latent.astype(jnp.float32),
            self.dcp[0].layer % 2,
            between=output_gate,
        )
        outputs = []
        for head in range(3):
            latent = merged[:, head]
            high = latent.astype(jnp.bfloat16)
            low = (latent - high.astype(jnp.float32)).astype(jnp.bfloat16)
            outputs.append(
                sum(
                    jax.lax.dot_general(
                        part,
                        up[:, head * 256 + 128 : head * 256 + 256],
                        (((1,), (0,)), ((), ())),
                        preferred_element_type=jnp.float32,
                    )
                    for part in (high, low)
                )
            )
        gate = jax.nn.sigmoid(
            self.workspace.output_gate_values_vmem[...].astype(jnp.float32)
        ).astype(jnp.bfloat16)
        return (
            jnp.stack(outputs, axis=1)
            .reshape(self.batch_size, 384)
            .astype(jnp.bfloat16)
            * gate
        )

    def _project_output(self, attention: _Dynamic) -> None:
        """Project attended heads to output ref."""
        output_stream = self.streams.get("output")
        if output_stream is None:
            self.weight_copies.output.wait()
        elif output_stream.chunks == 1:
            output_stream.wait_all()
            output_stream = None

        def project_output(reduction_vmem: _Dynamic) -> None:
            """Project attention output to destination."""
            with _phase("p2_attn_project_reduce"):
                _project_attention_2d(
                    attention,
                    self.weights.output_projection_vmem,
                    self.output_ref,
                    reduction_vmem,
                    self.communication_workspace.shared_local_reduction_vmem,
                    self.communication_workspace.shared_host_reduction_vmem,
                    self.communication_workspace.attention_features_vmem,
                    self.communication_workspace.attention_output_vmem,
                    self.communication_workspace.send_semaphores,
                    self.communication_workspace.receive_semaphores,
                    self.communication_workspace.reduction_phase,
                    _dot,
                    token_scatter_output=self.token_scatter_output,
                    weight_stream=output_stream,
                    staging=self.communication_workspace.projection_staging_vmem,
                    gather_wire=self.communication_workspace.attention_gather_vmem,
                )

        pl.run_scoped(
            project_output,
            tpu.VMEM(
                (self.batch_size, 3584 if self.batch_size > _CONST_32 else 8192),
                jnp.float32,
            ),
        )

    def run(self) -> None:
        """Run full MLA layer."""
        scope_projections = _phase_start("q6_mla_projections")
        _, new_latent = self._prepare_projections()
        _phase_end(scope_projections)

        scope_attend = _phase_start("q7_mla_attend")
        attention = self._attend(new_latent)
        _phase_end(scope_attend)

        self._project_output(attention)


def mla(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Execute one multi-head latent-attention layer inside a Pallas program."""
    return _MlaRunner(*_p_args, **_p_kw).run()


class _MlaDcpShardsRunner:
    """Runner for one MLA layer of attention DP under DCP."""

    def __getattr__(self, name: _Dynamic) -> _Dynamic:
        """Dynamic attribute access."""
        raise AttributeError(name)

    def __init__(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Initialize MLA DCP shards runner."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            self.position_ref,
            self.hidden_ref,
            self.new_key_ref,
            self.new_value_ref,
            self.attended_out,
        ) = (
            _p_next(_p_iter, _p_get("position_ref")),
            _p_next(_p_iter, _p_get("hidden_ref")),
            _p_next(_p_iter, _p_get("new_key_ref")),
            _p_next(_p_iter, _p_get("new_value_ref")),
            _p_next(_p_iter, _p_get("attended_out")),
        )
        (
            self.weights,
            self.streams,
            self.communication_workspace,
            self.dcp,
            self.epsilon,
            self.sequence_rows,
        ) = (
            _p_get("weights"),
            _p_get("streams"),
            _p_get("communication_workspace"),
            _p_get("dcp"),
            _p_get("epsilon", 1e-05),
            _p_get("sequence_rows", 1),
        )
        self.rows = self.hidden_ref.shape[0]
        self.shards = self.attended_out.shape[0]
        self.up_stream = self.streams["key_value_up"]
        if self.up_stream.slots is not None or self.up_stream.chunks != self.shards:
            _err_msg = (
                "DCP attention DP needs every shard's key-value-up columns resident"
            )
            raise ValueError(_err_msg)
        self.weight_copies = self.weights.make_projection_copies()
        self.guard = self.communication_workspace.reduction_phase[0] >= 0

    def start_copy(self, name: _Dynamic) -> None:
        """Start async copy for named projection."""
        if name in self.streams:
            self.streams[name].start()
        else:
            getattr(self.weight_copies, name).start()

    def _prepare_projections(self) -> tuple[_Dynamic, list[_Dynamic]]:
        """Compute query and KV latents and absorbed queries."""
        for name in ("query_up", "key_value_up", "output_gate"):
            self.start_copy(name)

        if "query_down" in self.streams:
            query_latent = self.streams["query_down"].dot(
                self.hidden_ref[...], self.guard
            )
        else:
            self.weight_copies.query_down.wait()
            query_latent = _dot(
                self.hidden_ref[...], self.weights.query_down_projection_vmem[...]
            )
        query_latent = query_latent.astype(jnp.bfloat16).astype(jnp.float32)

        pl.when(self.guard)(lambda: self.start_copy("output"))

        query_latent = (
            query_latent
            * jax.lax.rsqrt(
                jnp.mean(query_latent * query_latent, axis=1, keepdims=True)
                + self.epsilon
            )
            * self.weights.query_normalization[...]
        ).astype(jnp.bfloat16)

        self.weight_copies.key_value_down.wait()
        compressed = _dot(
            self.hidden_ref[...], self.weights.key_value_down_projection_vmem[...]
        ).astype(jnp.bfloat16)
        key_value_latent = compressed[:, :512].astype(jnp.float32)
        key_value_latent = (
            key_value_latent
            * jax.lax.rsqrt(
                jnp.mean(key_value_latent * key_value_latent, axis=1, keepdims=True)
                + self.epsilon
            )
            * self.weights.key_value_normalization[...]
        ).astype(jnp.bfloat16)
        new_latent = jnp.concatenate(
            (
                key_value_latent,
                compressed[:, 512:576],
                jnp.zeros((self.rows, 64), jnp.bfloat16),
            ),
            axis=1,
        )
        self.new_key_ref[:, 0] = new_latent[:, :256]
        self.new_key_ref[:, 1] = new_latent[:, 256:512]
        self.new_key_ref[:, 2] = jnp.concatenate(
            (new_latent[:, 512:], jnp.zeros((self.rows, 128), jnp.bfloat16)), axis=1
        )
        self.new_value_ref[:, :3] = jnp.zeros((self.rows, 3, 128), jnp.bfloat16)
        self.up_stream.wait_all()
        scale = 192**-0.5
        absorbed = []
        for shard in range(self.shards):
            query = (
                _shard_dot(self.streams["query_up"], query_latent, shard, self.guard)
                .astype(jnp.bfloat16)
                .reshape(self.rows, 3, 256)
            )
            up = self.up_stream.view(shard)
            for head in range(3):
                latent_query = jax.lax.dot_general(
                    query[:, head, :128],
                    up[:, head * 256 : head * 256 + 128],
                    (((1,), (1,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )
                absorbed.append(
                    jnp.concatenate(
                        (
                            latent_query,
                            query[:, head, 128:192].astype(jnp.float32),
                            jnp.zeros((self.rows, 64), jnp.float32),
                        ),
                        axis=1,
                    )
                    * scale
                )
        return new_latent, absorbed

    def _attend(self, new_latent: _Dynamic, absorbed: list[_Dynamic]) -> None:
        """Run DCP attention across shards."""
        dcp_cache, dcp_workspace = self.dcp[0], self.dcp[1]
        dcp_attention = mla_dcp.DcpAttention(
            dcp_cache,
            dcp_workspace,
            self.position_ref,
            self.rows,
            skip=_config().dcp_skip,
            heads=3 * self.shards,
            sequence_rows=self.sequence_rows,
        )
        dcp_attention.prime()

        def output_gates() -> None:
            """Project output gates across shards."""
            for shard in range(self.shards):
                self.attended_out[shard] = _shard_dot(
                    self.streams["output_gate"], self.hidden_ref[...], shard, self.guard
                ).astype(jnp.bfloat16)

        merged = dcp_attention.attend(
            jnp.stack(absorbed, axis=1),
            new_latent.astype(jnp.float32),
            self.dcp[0].layer % 2,
            between=output_gates,
        )
        for shard in range(self.shards):
            up = self.up_stream.view(shard)
            outputs = []
            for head in range(3):
                latent = merged[:, 3 * shard + head]
                high = latent.astype(jnp.bfloat16)
                low = (latent - high.astype(jnp.float32)).astype(jnp.bfloat16)
                outputs.append(
                    sum(
                        jax.lax.dot_general(
                            part,
                            up[:, head * 256 + 128 : head * 256 + 256],
                            (((1,), (0,)), ((), ())),
                            preferred_element_type=jnp.float32,
                        )
                        for part in (high, low)
                    )
                )
            self.attended_out[shard] = jnp.stack(outputs, axis=1).reshape(
                self.rows, 384
            ).astype(jnp.bfloat16) * jax.nn.sigmoid(
                self.attended_out[shard].astype(jnp.float32)
            ).astype(jnp.bfloat16)

    def run(self) -> None:
        """Run full MLA DCP shards execution."""
        scope_projections = _phase_start("q6_mla_projections")
        new_latent, absorbed = self._prepare_projections()
        _phase_end(scope_projections)

        scope_attend = _phase_start("q7_mla_attend")
        self._attend(new_latent, absorbed)
        _phase_end(scope_attend)


def mla_dcp_shards(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """One MLA layer of attention DP under DCP."""
    return _MlaDcpShardsRunner(*_p_args, **_p_kw).run()


# At B4, scatter the projection rows and gather the completed output once.
# Keep all 16 BF16 rank-pair partials until the final FP32 sum: intermediate
# host sums can change long-context logits. The final gather has private
# persistent storage so adjacent phases cannot overwrite unread packets.
B4_ATTENTION_GATHER = _config().b4_attention_gather


class _ProjectAttention2dRunner:
    """Runner for 2D attention output projection."""

    def __getattr__(self, name: _Dynamic) -> _Dynamic:
        """Dynamic attribute access."""
        raise AttributeError(name)

    def __init__(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Initialize 2D attention projection runner."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            self.attended,
            self.weight,
            self.out,
            self.comm,
            self.local,
            self.host_reduction,
            self.feature_backing,
            self.output_backing,
            self.sends,
            self.recvs,
            self.phase,
            self.dot,
        ) = (
            _p_next(_p_iter, _p_get("attended")),
            _p_next(_p_iter, _p_get("weight")),
            _p_next(_p_iter, _p_get("out")),
            _p_next(_p_iter, _p_get("comm")),
            _p_next(_p_iter, _p_get("local")),
            _p_next(_p_iter, _p_get("host_reduction")),
            _p_next(_p_iter, _p_get("feature_backing")),
            _p_next(_p_iter, _p_get("output_backing")),
            _p_next(_p_iter, _p_get("sends")),
            _p_next(_p_iter, _p_get("recvs")),
            _p_next(_p_iter, _p_get("phase")),
            _p_next(_p_iter, _p_get("dot")),
        )
        (
            self.token_scatter_output,
            self.weight_stream,
            self.staging,
            self.gather_wire,
        ) = (
            _p_kw.get("token_scatter_output", False),
            _p_get("weight_stream"),
            _p_get("staging"),
            _p_get("gather_wire"),
        )
        self.rank = jax.lax.axis_index("tp")
        self.slot = self.phase[0] % 2
        self.batch_size = self.attended.shape[0]
        self.ordered_full_output = (
            B4_ATTENTION_GATHER
            and self.batch_size == _CONST_4
            and not self.token_scatter_output
        )
        self.width = self.attended.shape[-1]
        self.g = self.weight.shape[0] // self.width
        if self.g != _CONST_2:
            _err_msg = "Assertion failed: g == 2"
            raise AssertionError(_err_msg)
        valid_batch = (
            1 <= self.batch_size <= MAX_BATCH
            and self.out.shape[0] == self.batch_size
            and (self.comm.shape[0] == self.batch_size)
        )
        if not valid_batch:
            _err_msg = "Assertion failed: batch_size and shapes valid"
            raise AssertionError(_err_msg)
        self.h = self.out.shape[-1]
        self.half = self.h // self.g
        self.nr = 32 // self.g

    def _init_compute(self, features: _Dynamic) -> None:
        """Initialize attention compute and exchange features."""
        view = features.at[:, pl.ds((self.rank % self.g) * self.width, self.width)]
        view[...] = self.attended
        for offset in range(self.g - 1, 0, -1):
            tpu.make_async_remote_copy(
                view,
                view,
                self.sends.at[10 + self.slot],
                self.recvs.at[10 + self.slot],
                device_id=(self.rank ^ offset,),
                device_id_type=pl.DeviceIdType.MESH,
            ).start()
        self.comm[...] = jnp.zeros(self.comm.shape, jnp.float32)
        if self.weight_stream is None:
            own_weight = self.weight[pl.ds((self.rank % 2) * self.width, self.width), :]
        else:
            self.weight_stream.wait(0)
            own_weight = self.weight_stream.view(0)[...]
        self.comm[:, : self.half] = self.dot(self.attended, own_weight)
        if self.weight_stream is not None:
            self.weight_stream.release(0, self.phase[0] >= 0)
        count = features.at[:, : (self.g - 1) * self.width]
        tpu.make_async_copy(count, count, self.sends.at[10 + self.slot]).wait()
        tpu.make_async_copy(count, count, self.recvs.at[10 + self.slot]).wait()
        peer_lane = (self.rank % 2) ^ 1
        if self.weight_stream is None:
            peer_weight = self.weight[pl.ds(peer_lane * self.width, self.width), :]
        else:
            self.weight_stream.wait(1)
            peer_weight = self.weight_stream.view(1)[...]
        self.comm[:, : self.half] += self.dot(
            features[:, pl.ds(peer_lane * self.width, self.width)], peer_weight
        )

    def _scatter_local_exchange(
        self, tokens_per_pair: int, payload_rows: int, pair: _Dynamic
    ) -> _Dynamic:
        """Perform local pair reduction copies."""
        if not (
            self.staging is not None or 2 * self.local.shape[1] >= 4 * payload_rows
        ):
            _err_msg = "wide reduction rows too small for the payloads"
            raise AssertionError(_err_msg)
        local_send = (
            (
                self.local.at[0].bitcast(jnp.bfloat16)
                if self.staging is None
                else self.staging.at[0]
            )
            .at[pl.ds(0, 4 * payload_rows), :]
            .reshape(4, payload_rows, 128)
        )
        local_receive = (
            (
                self.local.at[1].bitcast(jnp.bfloat16)
                if self.staging is None
                else self.staging.at[1]
            )
            .at[pl.ds(0, 4 * payload_rows), :]
            .reshape(4, payload_rows, 128)
        )
        for target_pair in range(4):
            if tokens_per_pair == 1:
                partial = self.comm[target_pair, : self.half][None, :]
            else:
                partial = self.comm[
                    pl.ds(target_pair * tokens_per_pair, tokens_per_pair),
                    : self.half,
                ]
            local_send[target_pair] = partial.astype(jnp.bfloat16).reshape(
                payload_rows, 128
            )
        local_receive[pair] = local_send[pair]
        local_copies = []
        for offset in range(3, 0, -1):
            target_pair = pair ^ offset
            copy = tpu.make_async_remote_copy(
                local_send.at[target_pair],
                local_receive.at[pair],
                self.sends.at[offset - 1],
                self.recvs.at[offset - 1],
                device_id=(self.rank ^ (2 * offset),),
                device_id_type=pl.DeviceIdType.MESH,
            )
            copy.start()
            local_copies.append(copy)
        for copy in local_copies:
            copy.wait()
        return local_receive

    def _scatter_host_exchange(
        self, local_receive: _Dynamic, payload_rows: int, host: _Dynamic
    ) -> _Dynamic:
        """Perform host-level reduction copies."""
        if self.ordered_full_output:
            host_wire_rows = 4 * payload_rows
            if not (host_wire_rows <= 2 * self.host_reduction.shape[1]):
                _err_msg = (
                    "Assertion failed: host_wire_rows <= 2 * host_reduction.shape[1]"
                )
                raise AssertionError(_err_msg)
            host_receive = self.host_reduction.bitcast(jnp.bfloat16).at[
                :, pl.ds(0, host_wire_rows), :
            ]
            host_receive[host] = local_receive[...].reshape(host_wire_rows, 128)
        else:
            local_sum = jnp.sum(local_receive[...].astype(jnp.float32), axis=0)
            host_receive = self.host_reduction.at[:, pl.ds(0, payload_rows), :]
            host_receive[host] = local_sum
        host_copies = []
        host_view = host_receive.at[host]
        for offset in range(3, 0, -1):
            copy = tpu.make_async_remote_copy(
                host_view,
                host_view,
                self.sends.at[3 + offset - 1],
                self.recvs.at[3 + offset - 1],
                device_id=(self.rank ^ (8 * offset),),
                device_id_type=pl.DeviceIdType.MESH,
            )
            copy.start()
            host_copies.append(copy)
        for copy in host_copies:
            copy.wait()
        if self.ordered_full_output:
            rank_partials = (
                host_receive[...].reshape(16, payload_rows, 128).astype(jnp.float32)
            )
            return jnp.sum(rank_partials, axis=0).astype(jnp.bfloat16)
        return jnp.sum(host_receive[...], axis=0).astype(jnp.bfloat16)

    def _compute_scatter(self) -> None:
        """Compute token-scatter attention output."""
        if self.batch_size not in (4, *SCATTER_BATCHES):
            _err_msg = "Assertion failed: batch_size in (4, *SCATTER_BATCHES)"
            raise AssertionError(_err_msg)
        tokens_per_pair = self.batch_size // 4
        payload_rows = tokens_per_pair * self.half // 128
        host = self.rank // 8
        pair = (self.rank % 8) // 2
        parity = self.rank % 2

        local_receive = self._scatter_local_exchange(
            tokens_per_pair, payload_rows, pair
        )
        reduced_half = self._scatter_host_exchange(local_receive, payload_rows, host)

        owned_output = _bf16_rows_view(self.output_backing, tokens_per_pair, self.h)
        pair_send = self.local.at[2].bitcast(jnp.bfloat16).at[pl.ds(0, payload_rows), :]
        pair_receive = (
            self.local.at[3].bitcast(jnp.bfloat16).at[pl.ds(0, payload_rows), :]
        )
        pair_send[...] = reduced_half
        pair_copy = tpu.make_async_remote_copy(
            pair_send,
            pair_receive,
            self.sends.at[6],
            self.recvs.at[6],
            device_id=(self.rank ^ 1,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        pair_copy.start()
        pair_copy.wait()
        sent_half = pair_send[...].reshape(tokens_per_pair, self.half)
        received_half = pair_receive[...].reshape(tokens_per_pair, self.half)
        owned_output[:, : self.half] = jnp.where(parity == 0, sent_half, received_half)
        owned_output[:, self.half :] = jnp.where(parity == 0, received_half, sent_half)
        if self.ordered_full_output:
            if not (self.gather_wire is not None):
                _err_msg = "Assertion failed: gather_wire is not None"
                raise AssertionError(_err_msg)
            _gather_local_pair_rows(
                owned_output,
                self.out,
                self.sends,
                self.recvs,
                self.gather_wire,
                owners=4,
            )

    def _compute_regular(self, output: _Dynamic) -> None:
        """Compute regular attention output without scatter."""
        rows = ((self.half + 255) // 256) * 2
        payload = rows // 2
        batched_payload = self.batch_size * payload
        receive = self.local.reshape(self.nr, self.local.shape[1] * 8 // self.nr, 128)
        if not (batched_payload <= receive.shape[1]):
            _err_msg = "Assertion failed: batched_payload <= receive.shape[1]"
            raise AssertionError(_err_msg)
        bits = tpu.bitcast(
            self.comm[:, : rows * 128]
            .reshape(self.batch_size * rows, 128)
            .astype(jnp.bfloat16),
            jnp.uint32,
        )
        receive[self.rank // self.g, :batched_payload, :] = (
            jax.lax.bitcast_convert_type(bits, jnp.float32)
        )
        view = receive.at[self.rank // self.g, :batched_payload, :]
        for offset in range(self.nr - 1, 0, -1):
            tpu.make_async_remote_copy(
                view,
                view,
                self.sends.at[14 + self.slot],
                self.recvs.at[14 + self.slot],
                device_id=(self.rank ^ (self.g * offset),),
                device_id_type=pl.DeviceIdType.MESH,
            ).start()
        count = receive.at[: self.nr - 1, :batched_payload, :]
        tpu.make_async_copy(count, count, self.sends.at[14 + self.slot]).wait()
        tpu.make_async_copy(count, count, self.recvs.at[14 + self.slot]).wait()
        values = tpu.bitcast(
            jax.lax.bitcast_convert_type(receive[:, :batched_payload, :], jnp.uint32),
            jnp.bfloat16,
        ).astype(jnp.float32)
        self.comm[:, : rows * 128] = jnp.sum(values, axis=0).reshape(
            self.batch_size, rows * 128
        )
        view = output.at[:, pl.ds((self.rank % self.g) * self.half, self.half)]
        view[...] = self.comm[:, : self.half].astype(jnp.bfloat16)
        for offset in range(self.g - 1, 0, -1):
            tpu.make_async_remote_copy(
                view,
                view,
                self.sends.at[12 + self.slot],
                self.recvs.at[12 + self.slot],
                device_id=(self.rank ^ offset,),
                device_id_type=pl.DeviceIdType.MESH,
            ).start()
        count = output.at[:, : (self.g - 1) * self.half]
        tpu.make_async_copy(count, count, self.sends.at[12 + self.slot]).wait()
        tpu.make_async_copy(count, count, self.recvs.at[12 + self.slot]).wait()
        self.out[...] = output[...]

    def run(self) -> None:
        """Run 2D attention projection."""
        if not (self.g * self.width <= _CONST_1024 and self.h == _CONST_7168):
            _err_msg = "Assertion failed: g * width <= 1024 and h == 7168"
            raise AssertionError(_err_msg)
        feature_store = _bf16_rows_view(self.feature_backing, self.batch_size, 1024)
        output_store = _bf16_rows_view(self.output_backing, self.batch_size, 7168)
        features = feature_store.at[:, : self.g * self.width]
        self._init_compute(features)
        if self.token_scatter_output or self.ordered_full_output:
            self._compute_scatter()
        else:
            self._compute_regular(output_store)
        self.phase[0] += 1


def _project_attention_2d(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """``weight_stream`` (optional) is a row ChunkedWeightStream of ``weight``."""
    return _ProjectAttention2dRunner(*_p_args, **_p_kw).run()


class _ProjectAttentionDpRunner:
    """Runner for projecting DP2/TP16 or DP4/TP8 attention back to DP row ownership."""

    def __getattr__(self, name: _Dynamic) -> _Dynamic:
        """Dynamic attribute access."""
        raise AttributeError(name)

    def __init__(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Initialize attention DP projection runner."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            self.attended,
            self.weight,
            self.out,
            self.comm,
            self.local,
            self.host_reduction,
            self.feature_backing,
            self.sends,
            self.recvs,
            self.phase,
            self.dot,
        ) = (
            _p_next(_p_iter, _p_get("attended")),
            _p_next(_p_iter, _p_get("weight")),
            _p_next(_p_iter, _p_get("out")),
            _p_next(_p_iter, _p_get("comm")),
            _p_next(_p_iter, _p_get("local")),
            _p_next(_p_iter, _p_get("host_reduction")),
            _p_next(_p_iter, _p_get("feature_backing")),
            _p_next(_p_iter, _p_get("sends")),
            _p_next(_p_iter, _p_get("recvs")),
            _p_next(_p_iter, _p_get("phase")),
            _p_next(_p_iter, _p_get("dot")),
        )
        (
            self.weight_stream,
            self.staging,
            self.owners,
        ) = (
            _p_get("weight_stream"),
            _p_get("staging"),
            _p_get("owners", 4),
        )
        self.rank = jax.lax.axis_index("tp")
        self.slot = self.phase[0] % 2
        self.shards, self.rows, self.width = self.attended.shape
        self.hidden = self.out.shape[-1]
        self.half = self.hidden // 2
        valid_shards = (
            self.shards in (2, 4)
            and self.owners in (1, 4)
            and (self.rows % self.owners == 0)
        )
        if not valid_shards:
            _err_msg = "Assertion failed: valid shards and owners"
            raise AssertionError(_err_msg)
        if not (self.owners == _CONST_4 or self.shards == _CONST_2):
            _err_msg = "Assertion failed: owners == 4 or shards == 2"
            raise AssertionError(_err_msg)
        self.lane, self.pair, self.host = (
            self.rank % 2,
            (self.rank % 8) // 2,
            self.rank // 8,
        )
        self.rows_per_pair = self.rows // self.owners
        valid_weight = self.weight_stream is not None or self.weight.shape == (
            self.shards * 2 * self.width,
            self.half,
        )
        if not valid_weight:
            _err_msg = "Assertion failed: valid weight shape"
            raise AssertionError(_err_msg)
        self.guard = self.phase[0] >= 0

    def weight_rows(self, position: _Dynamic) -> _Dynamic:
        """Extract weight rows for given position."""
        shard, peer = position % self.shards, position // self.shards
        if self.weight_stream is not None:
            self.weight_stream.wait(position)
            return self.weight_stream.view(position)[...]
        return self.weight[
            pl.ds((shard * 2 + (self.lane ^ peer)) * self.width, self.width), :
        ]

    def _init_features_and_comm(self, features: _Dynamic) -> None:
        """Initialize and exchange features across lanes."""
        for shard in range(self.shards):
            view = features.at[
                :, pl.ds(shard * 2 * self.width + self.lane * self.width, self.width)
            ]
            view[...] = self.attended[shard]
            tpu.make_async_remote_copy(
                view,
                view,
                self.sends.at[10 + self.slot],
                self.recvs.at[10 + self.slot],
                device_id=(self.rank ^ 1,),
                device_id_type=pl.DeviceIdType.MESH,
            ).start()
        self.comm[...] = jnp.zeros(self.comm.shape, jnp.float32)
        for shard in range(self.shards):
            self.comm[:, : self.half] += self.dot(
                self.attended[shard], self.weight_rows(shard)
            )
            if self.weight_stream is not None:
                self.weight_stream.release(shard, self.guard)
        received = features.at[:, pl.ds(0, self.shards * self.width)]
        tpu.make_async_copy(received, received, self.sends.at[10 + self.slot]).wait()
        tpu.make_async_copy(received, received, self.recvs.at[10 + self.slot]).wait()
        peer_lane = self.lane ^ 1
        for shard in range(self.shards):
            peer_features = features[
                :, pl.ds(shard * 2 * self.width + peer_lane * self.width, self.width)
            ]
            self.comm[:, : self.half] += self.dot(
                peer_features, self.weight_rows(self.shards + shard)
            )
            if self.weight_stream is not None:
                self.weight_stream.release(self.shards + shard, self.guard)

    def _reduce_single_owner(self, payload_rows: int) -> tuple[_Dynamic, int, int]:
        """Reduce partials when owners == 1."""
        host_slots = self.host_reduction.bitcast(jnp.bfloat16)

        def pair_partials(source_host: _Dynamic) -> _Dynamic:
            """Extract pair partials."""
            return (
                host_slots.at[source_host]
                .at[pl.ds(0, 4 * payload_rows), :]
                .reshape(4, payload_rows, 128)
            )

        own_partial = pair_partials(self.host).at[self.pair]
        own_partial[...] = (
            self.comm[:, : self.half].astype(jnp.bfloat16).reshape(payload_rows, 128)
        )
        copies = []
        for offset in range(15, 0, -1):
            copy = tpu.make_async_remote_copy(
                own_partial,
                own_partial,
                self.sends.at[12],
                self.recvs.at[12],
                device_id=(self.rank ^ (2 * offset),),
                device_id_type=pl.DeviceIdType.MESH,
            )
            copy.start()
            copies.append(copy)
        for copy in copies:
            copy.wait()
        host_sums = [
            jnp.sum(pair_partials(source_host)[...].astype(jnp.float32), axis=0)
            for source_host in range(4)
        ]
        reduced_half = jnp.concatenate(
            [host_sums[2 * group] + host_sums[2 * group + 1] for group in range(2)],
            axis=0,
        ).astype(jnp.bfloat16)
        return reduced_half, payload_rows * 2, self.rows_per_pair * 2

    def _reduce_multi_owner(self, payload_rows: int) -> _Dynamic:
        """Reduce partials when owners > 1."""
        slot_rows = (
            payload_rows if self.staging is None else _staging_slot_rows(payload_rows)
        )
        local_send = (
            (
                self.local.at[0].bitcast(jnp.bfloat16)
                if self.staging is None
                else self.staging.at[0]
            )
            .at[pl.ds(0, 4 * slot_rows), :]
            .reshape(4, slot_rows, 128)
        )
        local_receive = (
            (
                self.local.at[1].bitcast(jnp.bfloat16)
                if self.staging is None
                else self.staging.at[1]
            )
            .at[pl.ds(0, 4 * slot_rows), :]
            .reshape(4, slot_rows, 128)
        )
        for target_pair in range(4):
            wire = (
                self.comm[
                    pl.ds(
                        target_pair * self.rows_per_pair
                        if self.owners == _CONST_4
                        else 0,
                        self.rows_per_pair,
                    ),
                    : self.half,
                ]
                .astype(jnp.bfloat16)
                .reshape(payload_rows, 128)
            )
            if slot_rows != payload_rows:
                wire = jnp.concatenate(
                    [wire, jnp.zeros((slot_rows - payload_rows, 128), jnp.bfloat16)]
                )
            local_send[target_pair] = wire
        local_receive[self.pair] = local_send[self.pair]
        copies = []
        for offset in range(3, 0, -1):
            copy = tpu.make_async_remote_copy(
                local_send.at[self.pair ^ offset],
                local_receive.at[self.pair],
                self.sends.at[offset - 1],
                self.recvs.at[offset - 1],
                device_id=(self.rank ^ (2 * offset),),
                device_id_type=pl.DeviceIdType.MESH,
            )
            copy.start()
            copies.append(copy)
        for copy in copies:
            copy.wait()
        local_sum = jnp.sum(local_receive[...].astype(jnp.float32), axis=0)
        if slot_rows != payload_rows:
            local_sum = local_sum[:payload_rows]

        if self.shards == _CONST_2:
            host_receive = self.host_reduction.at[:2, pl.ds(0, payload_rows), :]
            host_receive[self.host % 2] = local_sum
            host_view = host_receive.at[self.host % 2]
            copy = tpu.make_async_remote_copy(
                host_view,
                host_view,
                self.sends.at[3],
                self.recvs.at[3],
                device_id=(self.rank ^ 8,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            copy.start()
            copy.wait()
            return jnp.sum(host_receive[...], axis=0).astype(jnp.bfloat16)
        return local_sum.astype(jnp.bfloat16)

    def run(self) -> None:
        """Run attention DP projection."""
        features = _bf16_rows_view(
            self.feature_backing, self.rows, self.feature_backing.shape[-1]
        )
        if not (features.shape[-1] >= 2 * self.width * self.shards):
            _err_msg = "Assertion failed: features.shape[-1] >= 2 * width * shards"
            raise AssertionError(_err_msg)
        self._init_features_and_comm(features)
        payload_rows = self.rows_per_pair * self.half // 128
        rows_per_pair = self.rows_per_pair
        if self.owners == 1:
            reduced_half, payload_rows, rows_per_pair = self._reduce_single_owner(
                payload_rows
            )
        else:
            reduced_half = self._reduce_multi_owner(payload_rows)

        pair_send = self.local.at[2].bitcast(jnp.bfloat16).at[pl.ds(0, payload_rows), :]
        pair_receive = (
            self.local.at[3].bitcast(jnp.bfloat16).at[pl.ds(0, payload_rows), :]
        )
        pair_send[...] = reduced_half
        copy = tpu.make_async_remote_copy(
            pair_send,
            pair_receive,
            self.sends.at[6],
            self.recvs.at[6],
            device_id=(self.rank ^ 1,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        copy.wait()
        sent = pair_send[...].reshape(rows_per_pair, self.half)
        received_half = pair_receive[...].reshape(rows_per_pair, self.half)
        row = jnp.concatenate(
            (
                jnp.where(self.lane == 0, sent, received_half),
                jnp.where(self.lane == 0, received_half, sent),
            ),
            axis=1,
        )
        if self.out.shape != (rows_per_pair, self.hidden):
            _err_msg = "Assertion failed: out.shape == (rows_per_pair, hidden)"
            raise AssertionError(_err_msg)
        self.out[...] = row
        self.phase[0] += 1


def _project_attention_dp(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Project DP2/TP16 or DP4/TP8 attention back to DP row ownership."""
    return _ProjectAttentionDpRunner(*_p_args, **_p_kw).run()


class _GatherDpMoeRowsRunner:
    """Runner for gathering DP groups' rows and local pairs' rows."""

    def __getattr__(self, name: _Dynamic) -> _Dynamic:
        """Dynamic attribute access."""
        raise AttributeError(name)

    def __init__(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Initialize gather DP MoE rows runner."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            self.row,
            self.owned_out,
            self.out,
            self.local,
            self.sends,
            self.recvs,
            self.shards,
            self.pair_semaphores,
        ) = (
            _p_next(_p_iter, _p_get("row")),
            _p_next(_p_iter, _p_get("owned_out")),
            _p_next(_p_iter, _p_get("out")),
            _p_next(_p_iter, _p_get("local")),
            _p_next(_p_iter, _p_get("sends")),
            _p_next(_p_iter, _p_get("recvs")),
            _p_next(_p_iter, _p_get("shards")),
            _p_next(_p_iter, _p_get("pair_semaphores")),
        )
        self.rank = jax.lax.axis_index("tp")
        self.group_size = 32 // self.shards
        self.group, self.pair = self.rank // self.group_size, (self.rank % 8) // 2
        self.rows, self.hidden = self.row.shape
        self.payload = self.rows * self.hidden // 128
        self.stride = -(-self.payload // 16) * 16
        self.slot_count = 2 * self.local.shape[0] * self.local.shape[1] // self.stride
        self.base = self.slot_count // 2
        if not (self.base >= 4 * self.shards):
            _err_msg = "Assertion failed: base >= 4 * shards"
            raise AssertionError(_err_msg)
        self.wires = pool_alias.view(
            self.local, (self.slot_count, self.stride, 128), jnp.bfloat16
        )
        if self.pair_semaphores is not None:
            self.half = -(-self.payload // 32) * 16
            self.start = pl.multiple_of(
                (self.rank % 2) * (self.payload - self.half), 16
            )

    def wire(self, o: _Dynamic, k: _Dynamic, *, whole: _Dynamic = False) -> _Dynamic:
        """Execute wire logic."""
        if self.pair_semaphores is None or whole:
            return self.wires.at[self.base + 4 * o + k, pl.ds(0, self.payload)]
        return self.wires.at[self.base + 4 * o + k, pl.ds(self.start, self.half)]

    def send(
        self, source: _Dynamic, o: _Dynamic, k: _Dynamic, semaphore: _Dynamic
    ) -> _Dynamic:
        """Execute send logic."""
        copy = tpu.make_async_remote_copy(
            source,
            self.wire(o, k),
            self.sends.at[semaphore],
            self.recvs.at[semaphore],
            device_id=(self.rank ^ (self.group_size * o) ^ (2 * k),),
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        return copy

    def _run_short_wires(self) -> None:
        """Run short direct wire copies."""
        direct = [
            self.send(self.wire(0, 0), o, k, 7 if o == 0 else 8)
            for o in range(self.shards)
            for k in range(4)
            if o or k
        ]
        for copy in direct:
            copy.wait()

    def _run_forwarded_wires(self) -> None:
        """Run forwarded wire copies."""
        local_copies = [self.send(self.wire(0, 0), 0, k, 7) for k in range(1, 4)]
        group_copies = [
            self.send(self.wire(0, 0), o, 0, (8, 10, 11)[o - 1])
            for o in range(1, self.shards)
        ]
        forwarded = []
        for o, copy in enumerate(group_copies, 1):
            copy.wait_recv()
            for k in range(1, 4):
                forwarded.append(
                    tpu.make_async_remote_copy(
                        self.wire(o, 0),
                        self.wire(o, k),
                        self.sends.at[9],
                        self.recvs.at[9],
                        device_id=(self.rank ^ (2 * k),),
                        device_id_type=pl.DeviceIdType.MESH,
                    )
                )
                forwarded[-1].start()
        for copy in group_copies:
            copy.wait_send()
        for copy in (*local_copies, *forwarded):
            copy.wait()

    def _run_swaps_and_unpack(self) -> None:
        """Run wire swaps and output unpacking."""
        if self.pair_semaphores is not None:
            swaps = []
            for slot in range(1, 4 * self.shards):
                o, k = divmod(slot, 4)
                swaps.append(
                    tpu.make_async_remote_copy(
                        self.wire(o, k),
                        self.wire(o, k),
                        *self.pair_semaphores,
                        device_id=(self.rank ^ 1,),
                        device_id_type=pl.DeviceIdType.MESH,
                    )
                )
                swaps[-1].start()
            for copy in swaps:
                copy.wait()
        for d in range(self.shards):
            for p in range(4):
                slot = self.base + 4 * (self.group ^ d) + (self.pair ^ p)
                first = (p * self.shards + d) * self.rows
                self.out[first : first + self.rows, :] = self.wires[
                    slot, pl.ds(0, self.payload)
                ].reshape(self.rows, self.hidden)
            self.owned_out[d * self.rows : (d + 1) * self.rows, :] = self.wires[
                self.base + 4 * (self.group ^ d), pl.ds(0, self.payload)
            ].reshape(self.rows, self.hidden)

    def run(self) -> None:
        """Run gather DP MoE rows execution."""
        self.wire(0, 0, whole=True)[...] = self.row.reshape(self.payload, 128)
        if self.rows <= _CONST_2 and self.shards == _CONST_2:
            self._run_short_wires()
        else:
            self._run_forwarded_wires()
        self._run_swaps_and_unpack()


def _gather_dp_moe_rows(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Gather the DP groups' rows and the local pairs' rows, overlapped."""
    return _GatherDpMoeRowsRunner(*_p_args, **_p_kw).run()


def _router_copy(
    weight: _Dynamic,
    buffer: _Dynamic,
    sems: _Dynamic,
    wait: _Dynamic,
    *,
    initialize_padding: _Dynamic = True,
) -> _Dynamic:
    """Execute router copy logic."""
    lane = jax.lax.axis_index("tp") % 2

    @pl.when(lane < 1)
    def full() -> _Dynamic:
        """Execute full logic."""
        copy = tpu.make_async_copy(
            weight.at[:, pl.ds(lane * 512, 512)], buffer.at[0], sems.at[0]
        )
        copy.wait() if wait else copy.start()

    @pl.when(lane == 1)
    def tail() -> _Dynamic:
        """Execute tail logic."""
        if not wait and initialize_padding:
            buffer[0, :, 384:] = jnp.zeros((weight.shape[0], 128), jnp.bfloat16)
        copy = tpu.make_async_copy(
            weight.at[:, pl.ds(512, 384)], buffer.at[0, :, :384], sems.at[0]
        )
        copy.wait() if wait else copy.start()


class _DenseScatterRowsRunner:
    """Runner for reducing dense layer directly to token owners."""

    def __getattr__(self, name: _Dynamic) -> _Dynamic:
        """Dynamic attribute access."""
        raise AttributeError(name)

    def __init__(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Initialize dense scatter rows runner."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            self.partial,
            self.output,
            self.local,
            self.hosts,
            self.sends,
            self.recvs,
            self.owners,
        ) = (
            _p_next(_p_iter, _p_get("partial")),
            _p_next(_p_iter, _p_get("output")),
            _p_next(_p_iter, _p_get("local")),
            _p_next(_p_iter, _p_get("hosts")),
            _p_next(_p_iter, _p_get("sends")),
            _p_next(_p_iter, _p_get("recvs")),
            _p_next(_p_iter, _p_get("owners")),
        )
        self.rank = jax.lax.axis_index("tp")
        self.lane, self.host = self.rank % 8, self.rank // 8
        self.owned_rows = self.partial.shape[0] // self.owners
        self.hidden = self.partial.shape[1]
        self.wire_rows = 8 * self.hidden // 128
        self.payload = self.wire_rows // 2
        if not (self.owned_rows % 8 == 0 and 8 % self.owners == 0):
            _err_msg = "Assertion failed: owned_rows % 8 == 0 and 8 % owners == 0"
            raise AssertionError(_err_msg)
        if not (
            self.payload <= self.local.shape[1] and self.payload <= self.hosts.shape[1]
        ):
            _err_msg = "Assertion failed: payload <= local.shape[1] and hosts.shape[1]"
            raise AssertionError(_err_msg)

    def _encode(self, value: _Dynamic) -> _Dynamic:
        """Encode BF16 wire rows."""
        packed = tpu.bitcast(
            value.reshape(self.wire_rows, 128).astype(jnp.bfloat16), jnp.uint32
        )
        return jax.lax.bitcast_convert_type(packed, jnp.float32)

    def _decode(self, value: _Dynamic) -> _Dynamic:
        """Decode BF16 wire rows."""
        unpacked = tpu.bitcast(
            jax.lax.bitcast_convert_type(value, jnp.uint32), jnp.bfloat16
        ).astype(jnp.float32)
        return jnp.sum(unpacked, axis=0).reshape(8, self.hidden)

    def _reduce_owned(self, wires: _Dynamic) -> None:
        """Execute reduce owned logic across hosts and lanes."""
        for first in range(0, self.owned_rows, 8):
            if first:
                collectives32.barrier()
            for owner in range(self.owners):
                start = owner * self.owned_rows + first
                wires[owner] = self._encode(self.partial[start : start + 8])
            self.local[self.lane, : self.payload] = wires[
                self.lane // (8 // self.owners)
            ]
            copies = []
            for offset in range(7, 0, -1):
                copy = tpu.make_async_remote_copy(
                    wires.at[(self.lane ^ offset) // (8 // self.owners)],
                    self.local.at[self.lane, : self.payload],
                    self.sends.at[offset - 1],
                    self.recvs.at[offset - 1],
                    device_id=(self.rank ^ offset,),
                    device_id_type=pl.DeviceIdType.MESH,
                )
                copy.start()
                copies.append(copy)
            for copy in copies:
                copy.wait()
            self.hosts[self.host, : self.payload] = self._encode(
                self._decode(self.local[:, : self.payload])
            )
            copies = []
            for offset in range(3, 0, -1):
                view = self.hosts.at[self.host, : self.payload]
                copy = tpu.make_async_remote_copy(
                    view,
                    view,
                    self.sends.at[6 + offset],
                    self.recvs.at[6 + offset],
                    device_id=(self.rank ^ (8 * offset),),
                    device_id_type=pl.DeviceIdType.MESH,
                )
                copy.start()
                copies.append(copy)
            for copy in copies:
                copy.wait()
            self.output[first : first + 8] = self._decode(
                self.hosts[:, : self.payload]
            ).astype(jnp.bfloat16)

    def run(self) -> None:
        """Run dense scatter rows reduction."""
        pl.run_scoped(
            self._reduce_owned, tpu.VMEM((self.owners, self.payload, 128), jnp.float32)
        )


def _dense_scatter_rows(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Reduce the dense layer directly to its token owners."""
    return _DenseScatterRowsRunner(*_p_args, **_p_kw).run()


def _all_reduce_rows(
    partial: _Dynamic,
    local: _Dynamic,
    hosts: _Dynamic,
    sends: _Dynamic,
    recvs: _Dynamic,
) -> _Dynamic:
    """TP32 sum of an FP32 ``[batch, hidden]`` partial on every rank.

    The shared-expert transport of the mixture-of-experts operation: the
    partial is rounded to BF16 and packed in pairs into 32-bit words, each
    lane sends its packet to the seven other local lanes, the local sum is
    packed again and exchanged among the four hosts. ``local`` and ``hosts``
    are the wide reduction buffers of the communication workspace
    (``[8, rows, 128]`` and ``[4, rows, 128]`` FP32); semaphores 0..6 and
    7..9 are used and left idle.
    """
    rank = jax.lax.axis_index("tp")
    lane, host = rank % 8, rank // 8
    batch_size, hidden = partial.shape
    rows = batch_size * hidden // 128
    payload = rows // 2
    if not (
        rows % 2 == 0 and payload <= local.shape[1] and (payload <= hosts.shape[1])
    ):
        _err_msg = (
            "Assertion failed: rows % 2 == 0 and payload <= "
            "local.shape[1] and (payload <= hosts.shape[1])"
        )
        raise AssertionError(_err_msg)

    def encode(value: _Dynamic, destination: _Dynamic) -> _Dynamic:
        """Execute encode logic."""
        packed = tpu.bitcast(value.reshape(rows, 128).astype(jnp.bfloat16), jnp.uint32)
        destination[:payload, :] = jax.lax.bitcast_convert_type(packed, jnp.float32)

    def decode(buffer: _Dynamic) -> _Dynamic:
        """Execute decode logic."""
        decoded = tpu.bitcast(
            jax.lax.bitcast_convert_type(buffer, jnp.uint32), jnp.bfloat16
        ).astype(jnp.float32)
        return jnp.sum(decoded, axis=0).reshape(batch_size, hidden)

    def exchange(
        buffer: _Dynamic,
        index: _Dynamic,
        offsets: _Dynamic,
        stride: _Dynamic,
        semaphore_base: _Dynamic,
    ) -> _Dynamic:
        """Execute exchange logic."""
        view = buffer.at[index, :payload, :]
        copies = []
        for offset in offsets:
            copy = tpu.make_async_remote_copy(
                view,
                view,
                sends.at[semaphore_base + offset],
                recvs.at[semaphore_base + offset],
                device_id=(rank ^ (stride * offset),),
                device_id_type=pl.DeviceIdType.MESH,
            )
            copy.start()
            copies.append(copy)
        for copy in copies:
            copy.wait()

    encode(partial, local.at[lane])
    exchange(local, lane, range(7, 0, -1), 1, -1)
    encode(decode(local[:, :payload, :]), hosts.at[host])
    exchange(hosts, host, range(3, 0, -1), 8, 6)
    return decode(hosts[:, :payload, :])


def _gather_sp8_local_rows(
    row: _Dynamic, out: _Dynamic, local: _Dynamic, sends: _Dynamic, recvs: _Dynamic
) -> _Dynamic:
    """Eight owners directly feed the existing pre-attention all-gather."""
    rank = jax.lax.axis_index("tp")
    owner = rank % 8
    rows, hidden = row.shape
    payload = rows * hidden // 128
    if not (rows == _CONST_8 and payload % 16 == 0):
        _err_msg = "Assertion failed: rows == 8 and payload % 16 == 0"
        raise AssertionError(_err_msg)
    slots = 2 * local.shape[0] * local.shape[1] // payload
    if not (slots >= _CONST_32):
        _err_msg = "Assertion failed: slots >= 32"
        raise AssertionError(_err_msg)
    wires = pool_alias.view(local, (slots, payload, 128), jnp.bfloat16)
    source = wires.at[owner]
    source[...] = row.reshape(payload, 128)
    copies = []
    for offset in range(7, 0, -1):
        copy = tpu.make_async_remote_copy(
            source,
            source,
            sends.at[offset - 1],
            recvs.at[offset - 1],
            device_id=(rank ^ offset,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        copies.append(copy)
    for copy in copies:
        copy.wait()
    for pair in range(4):
        out[pair * 16 : (pair + 1) * 16] = jnp.concatenate(
            (
                wires[2 * pair].reshape(rows, hidden),
                wires[2 * pair + 1].reshape(rows, hidden),
            ),
            axis=0,
        )


def _gather_sp8_dp_moe_rows(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """DP2 MoE-input all-gather, eight owners, existing pair-major ABI out.

    Payload slots are relative (DP group XOR, local core XOR). The upper
    half of the existing transport allocation is disjoint from pre gather.
    No separate pair reconstruction collective is needed.
    """
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        row,
        owned_out,
        out,
        local,
        sends,
        recvs,
    ) = (
        _p_next(_p_iter, _p_get("row")),
        _p_next(_p_iter, _p_get("owned_out")),
        _p_next(_p_iter, _p_get("out")),
        _p_next(_p_iter, _p_get("local")),
        _p_next(_p_iter, _p_get("sends")),
        _p_next(_p_iter, _p_get("recvs")),
    )
    rank = jax.lax.axis_index("tp")
    group, owner = rank // 16, rank % 8
    rows, hidden = row.shape
    payload = rows * hidden // 128
    if not (rows == _CONST_8 and payload % 16 == 0):
        _err_msg = "Assertion failed: rows == 8 and payload % 16 == 0"
        raise AssertionError(_err_msg)
    slots = 2 * local.shape[0] * local.shape[1] // payload
    base = slots // 2
    if not (base >= _CONST_16):
        _err_msg = "Assertion failed: base >= 16"
        raise AssertionError(_err_msg)
    wires = pool_alias.view(local, (slots, payload, 128), jnp.bfloat16)

    def wire(o: _Dynamic, k: _Dynamic) -> _Dynamic:
        """Execute wire logic."""
        return wires.at[base + 8 * o + k]

    def send(
        source: _Dynamic,
        o: _Dynamic,
        k: _Dynamic,
        semaphore: _Dynamic,
        *,
        local_only: _Dynamic = False,
    ) -> _Dynamic:
        """Execute send logic."""
        copy = tpu.make_async_remote_copy(
            source,
            wire(o, k),
            sends.at[semaphore],
            recvs.at[semaphore],
            device_id=(rank ^ (0 if local_only else 16 * o) ^ k,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        return copy

    wire(0, 0)[...] = row.reshape(payload, 128)
    local_copies = [send(wire(0, 0), 0, k, 7) for k in range(1, 8)]
    group_copy = send(wire(0, 0), 1, 0, 8)
    group_copy.wait_recv()
    forwarded = [send(wire(1, 0), 1, k, 9, local_only=True) for k in range(1, 8)]
    group_copy.wait_send()
    for copy in (*local_copies, *forwarded):
        copy.wait()
    for d in range(2):
        for p in range(4):
            even = wire(group ^ d, owner ^ (2 * p))[...].reshape(rows, hidden)
            odd = wire(group ^ d, owner ^ (2 * p + 1))[...].reshape(rows, hidden)
            first = (2 * p + d) * 16
            out[first : first + 16] = jnp.concatenate((even, odd), axis=0)
        own = wire(group ^ d, 0)[...].reshape(rows, hidden)
        peer = wire(group ^ d, 1)[...].reshape(rows, hidden)
        owned_out[d * 16 : (d + 1) * 16] = jnp.where(
            owner % 2 == 0,
            jnp.concatenate((own, peer), axis=0),
            jnp.concatenate((peer, own), axis=0),
        )


def _gather_local_pair_rows(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """All-gather the token shards of the ``owners`` local row owners.

    Owner o (4: the local rank pairs; 8: the local ranks) holds rows
    o * r .. o * r + r - 1 in ``source`` [r, hidden]; every rank ends with all
    of them in ``destination``. Each rank sends to its counterpart in every
    other owner (rank ^ k * 8 / owners). ``packed`` stages through flat
    transport wires when a source row slice crosses a native BF16 tile.
    """
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        source,
        destination,
        sends,
        recvs,
        scratch,
        owners,
    ) = (
        _p_next(_p_iter, _p_get("source")),
        _p_next(_p_iter, _p_get("destination")),
        _p_next(_p_iter, _p_get("sends")),
        _p_next(_p_iter, _p_get("recvs")),
        _p_next(_p_iter, _p_get("scratch")),
        _p_next(_p_iter, _p_get("owners", 4)),
    )
    packed = _p_kw.get("packed", False)
    rank = jax.lax.axis_index("tp")
    stride = 8 // owners
    owner = (rank % 8) // stride
    rows_per_owner = source.shape[0]
    # One row, or a row count whose slice is not tile-aligned for a DMA
    # (3, 5, 6, 7 rows at B12..B28): send a flattened wire through the
    # transport scratch and assemble the destination from values.
    if packed or rows_per_owner == 1 or rows_per_owner & (rows_per_owner - 1):
        if not (scratch is not None):
            _err_msg = "Assertion failed: scratch is not None"
            raise AssertionError(_err_msg)
        payload_rows = rows_per_owner * source.shape[1] // 128
        source_wire = (
            scratch.at[owner].bitcast(jnp.bfloat16).at[pl.ds(0, payload_rows), :]
        )
        source_wire[...] = source[...].reshape(payload_rows, 128)
        copies = []
        for offset in range(owners - 1, 0, -1):
            copy = tpu.make_async_remote_copy(
                source_wire,
                source_wire,
                sends.at[offset - 1],
                recvs.at[offset - 1],
                device_id=(rank ^ (stride * offset),),
                device_id_type=pl.DeviceIdType.MESH,
            )
            copy.start()
            copies.append(copy)
        for copy in copies:
            copy.wait()
        wires = tuple(
            scratch.at[source_owner].bitcast(jnp.bfloat16)[:payload_rows]
            for source_owner in range(owners)
        )
        if rows_per_owner == 1:
            destination[...] = jnp.stack(
                tuple(wire.reshape(source.shape[1]) for wire in wires)
            )
        else:
            destination[...] = jnp.concatenate(
                tuple(wire.reshape(rows_per_owner, source.shape[1]) for wire in wires),
                axis=0,
            )
        return
    for owner_index in range(owners):

        @pl.when(owner == owner_index)
        def gather_owner(owner_index: _Dynamic = owner_index) -> _Dynamic:
            """Execute gather owner logic."""
            row_slice = pl.ds(owner_index * rows_per_owner, rows_per_owner)
            destination[row_slice] = source[...]
            copies = []
            for offset in range(owners - 1, 0, -1):
                copy = tpu.make_async_remote_copy(
                    source,
                    destination.at[row_slice],
                    sends.at[offset - 1],
                    recvs.at[offset - 1],
                    device_id=(rank ^ (stride * offset),),
                    device_id_type=pl.DeviceIdType.MESH,
                )
                copy.start()
                copies.append(copy)
            for copy in copies:
                copy.wait()


# Repeating-stack composition.

MOE = (
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
KDA = (
    "k_gate",
    "k_projection",
    "k_fb",
    "k_o",
    "k_conv",
    "k_a_log",
    "k_dt",
    "k_norm",
)
MLA = ("m_qa", "m_ka", "m_qb", "m_kb", "m_gate", "m_o", "m_qnorm", "m_knorm")
NORMS = ("attn_res", "mlp_res", "attn_norm", "ffn_norm")
EPILOGUE = ("output_res", "final_norm")
DENSE = (
    "dense_gu",
    "dense_down",
)  # layer 0's MLP, used when the stack includes layer 0
NAMES = MOE + KDA + MLA + NORMS


def _residual_mixture_row(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Compute one row of the learned residual mixture."""
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        block_values,
        prefix_value,
        folded,
        active_block_count,
    ) = (
        _p_next(_p_iter, _p_get("block_values")),
        _p_next(_p_iter, _p_get("prefix_value")),
        _p_next(_p_iter, _p_get("folded")),
        _p_next(_p_iter, _p_get("active_block_count")),
    )
    (
        epsilon,
        relaxed_reduction_order,
    ) = (
        _p_get("epsilon"),
        _p_kw.get("relaxed_reduction_order", False),
    )

    def normalized_scores(values: _Dynamic) -> _Dynamic:
        """Execute normalized scores logic."""
        scores = jnp.sum(values * folded[...], axis=1, keepdims=True)
        return scores * jax.lax.rsqrt(
            jnp.mean(values * values, axis=1, keepdims=True) + epsilon
        )

    if relaxed_reduction_order:
        block_values = block_values.astype(jnp.float32)
        prefix_value = prefix_value.astype(jnp.float32)

        # Keeping the hidden-sized prefix separate avoids materializing and
        # reducing a nine-row tensor. This changes FP32 association and is
        # therefore intentionally restricted to the relaxed mode.
        scores = jnp.concatenate(
            (
                normalized_scores(block_values),
                normalized_scores(prefix_value),
            ),
            axis=0,
        )
        values: _Dynamic = None
    else:
        values = jnp.concatenate(
            (
                block_values,
                prefix_value,
            ),
            axis=0,
        ).astype(jnp.float32)
        scores = jnp.sum(values * folded[...], axis=1, keepdims=True)
        scores *= jax.lax.rsqrt(
            jnp.mean(values * values, axis=1, keepdims=True) + epsilon
        )

    residual_indices = jax.lax.broadcasted_iota(jnp.int32, (9, 1), 0)
    active = (residual_indices < active_block_count) | (residual_indices == _CONST_8)
    probabilities = jax.nn.softmax(jnp.where(active, scores, -jnp.inf), axis=0)
    if relaxed_reduction_order:
        mixed = (
            jnp.sum(block_values * probabilities[:8], axis=0, keepdims=True)
            + prefix_value * probabilities[8:9]
        )
    else:
        mixed = jnp.sum(values * probabilities, axis=0, keepdims=True)
    return mixed.astype(jnp.bfloat16)


def _residual_mixture_by_row(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Compute a residual mixture for a batch-leading input."""
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        blocks,
        prefix,
        destination,
        folded,
        active_block_count,
    ) = (
        _p_next(_p_iter, _p_get("blocks")),
        _p_next(_p_iter, _p_get("prefix")),
        _p_next(_p_iter, _p_get("destination")),
        _p_next(_p_iter, _p_get("folded")),
        _p_next(_p_iter, _p_get("active_block_count")),
    )
    (
        batch_size,
        epsilon,
        relaxed_reduction_order,
    ) = (
        _p_get("batch_size"),
        _p_get("epsilon"),
        _p_kw.get("relaxed_reduction_order", False),
    )
    for batch_index in range(batch_size):
        destination[batch_index : batch_index + 1] = _residual_mixture_row(
            blocks[batch_index],
            prefix[batch_index : batch_index + 1],
            folded,
            active_block_count,
            epsilon=epsilon,
            relaxed_reduction_order=relaxed_reduction_order,
        )


# Stage fixed-size layer weights in the persistent projection/expert pool
# (KernelConfig.pooled_weights=False: scoped VMEM buffers).
POOLED_WEIGHTS = _config().pooled_weights
# From B32 a routed expert used by 2..COMPACT_ROWS rows runs gate/up and down on
# a compact tile of those rows, not the whole batch (the FP8 path's
# per-scale-block accumulation scales with it). C=4 does not compile.
COMPACT_ROWS = 8
# A routed expert used by one row (from B8) stages it via scalar compares and
# updates only that output row. KernelConfig.singleton_block (batch a multiple
# of 16) selects it from its aligned 16-row block instead of one branch per row;
# it and stage_blocks default to on at B16-B128.
SINGLETON_BLOCK = _config().singleton_block
STAGE_BLOCKS = _config().stage_blocks
# MXFP4 tiles: single-expert contraction block (multiple of 1024: scale rows
# stay 32-aligned), paired gate/up tile, and split MXU dot tile (multiple of
# 128; 0 = unsplit); split tiles keep the scale-reduction order.
MXFP4_DOT_TILE = _config().mxfp4_dot_tile
if MXFP4_DOT_TILE is not None and (MXFP4_DOT_TILE < 0 or MXFP4_DOT_TILE % 128):
    _err_msg = "KernelConfig.mxfp4_dot_tile must be zero or a positive multiple of 128"
    raise ValueError(_err_msg)
MXFP4_BLOCK = _config().mxfp4_block
MXFP4_PAIR_TILE = _config().mxfp4_pair_tile
# KernelConfig.mxfp4_static_tile=T: unroll the single-expert MXFP4 product over
# static T-row tiles instead of a dynamic loop, keeping the unpacked tile small.
MXFP4_STATIC_TILE = _config().mxfp4_static_tile


# B16 runs routed experts on compact 8-row tiles.
B16_COMPACT_EXPERTS = _config().b16_compact_experts


def _compact_rows(batch_size: _Dynamic) -> _Dynamic:
    """Execute compact rows logic."""
    if batch_size == _CONST_16 and B16_COMPACT_EXPERTS and not COMPACT_ONLY:
        return 8
    return COMPACT_ROWS if batch_size >= _CONST_32 else 0


COMPACT_ONLY = _config().compact_only

# Shared finishers keep the B16-B128 expert loop and its successor within
# IMEM (decode runs these batches without deduplicated HLO calls).
SHARED_EXPERT_FINISHERS = _config().shared_expert_finishers
# Grouped MXFP4 routes at B16-B128 keep the 2/4/8-row gate/up pairs but
# finish overflowing experts in eight-row passes; without the full-batch
# fallback the expert loop fits in IMEM next to its predecessor.
COMPACT_OVERFLOW = _config().compact_overflow
SHARED_SHARDS = _config().shared_shards
# KernelConfig.kda_staged=True (above B8, no attention DP): stage the KDA input
# and feature-gate projections in VMEM; with a state stream (above B16) the
# chunk loop is rolled into one pl.loop.
# NOTE: derive separate staging/rolling rules for B32/B64 with 8-row chunks.
KDA_STAGED = _config().kda_staged
# Diagnostic: skip named parts of each layer; results are invalid (timing/hang
# bisection only). Every rank skips the same parts, so collectives stay paired.
DEBUG_SKIP = set(_config().debug_skip)


class _RowFunction:
    """``f[rows]`` evaluates ``f(rows)``: a per-chunk value computed from refs."""

    def __init__(self, function: _Dynamic) -> None:
        """Execute init logic."""
        self.function = function

    def __getitem__(self, rows: _Dynamic) -> _Dynamic:
        """Execute getitem logic."""
        return self.function(rows)


def _shared_shards(batch_size: _Dynamic) -> _Dynamic:
    # B128 keeps only each lane's shared-expert output shard (VMEM).
    """Execute shared shards logic."""
    if SHARED_SHARDS is None:
        return batch_size == _CONST_128
    return SHARED_SHARDS and batch_size > _CONST_32


def _use_compact_overflow(
    batch_size: _Dynamic, group_pending_routes: _Dynamic = None
) -> _Dynamic:
    """Execute use compact overflow logic."""
    return (
        COMPACT_OVERFLOW
        and not COMPACT_ONLY
        and SHARED_EXPERT_FINISHERS
        and (
            batch_size in (32, 64, 96, 128)
            or (batch_size == _CONST_16 and B16_COMPACT_EXPERTS)
        )
        and COMPACT_ROWS == _CONST_8
        and group_pending_routes
    )


def _expert_accumulator_rows(
    batch_size: _Dynamic, *, group_pending_routes: _Dynamic = False
) -> _Dynamic:
    """(gate/up, down) accumulator rows: the batch, or only the compact tiles."""
    capacity = _compact_rows(batch_size)
    if capacity and (
        COMPACT_ONLY
        or _use_compact_overflow(batch_size, group_pending_routes=group_pending_routes)
    ):
        return 2 * capacity, capacity
    return batch_size, batch_size


MOE_PREPARE_WEIGHTS = ("latent_down", "shared_gu", "shared_down")
MLA_EXTRA_WEIGHTS = ("m_qb", "m_kb", "m_gate")


def _pool_layout(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Byte offsets of every view in the projection/expert pool.

    Region A (from 0) holds, one phase at a time: the next layer's attention
    projections (KDA k_gate..k_o, or MLA q/kv-down), prefetched after the
    experts finish; the MLA output projection over the consumed q-down
    projection; the four routed-expert slots during the MoE. Region B (after
    A) holds the MLA q/kv-up and gate projections during attention, the
    latent-down and shared-expert weights during routing, and latent-up in the
    MoE tail (which overlaps the next layer's attention prefetch into A).
    """
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    local_weights = _p_next(_p_iter, _p_get("local_weights"))
    (
        alias_expert_scales,
        pooled,
        managed,
        dense_bytes,
        arena_extra,
    ) = (
        _p_get("alias_expert_scales"),
        _p_get("pooled"),
        _p_kw.get("managed", False),
        _p_get("dense_bytes", 0),
        _p_get("arena_extra", 0),
    )

    def storage(name: _Dynamic, experts: _Dynamic = None) -> _Dynamic:
        """Execute storage logic."""
        if experts is None:
            experts = bool(0)
        value = local_weights[name]
        shape = (4, *value.shape[1:]) if experts else value.shape
        return pool_alias.storage_size(shape, value.dtype)

    expert_names = (
        ("expert_gu", "expert_gus", "expert_down", "expert_ds")
        if alias_expert_scales
        else ("expert_gu", "expert_down")
    )
    expert_bytes = sum(storage(name, experts=True) for name in expert_names)
    attention_bytes = max(
        sum(
            math.prod(local_weights[name].shape) * local_weights[name].dtype.itemsize
            for name in names
        )
        for names in (KDA[:4], MLA[:2])
    )
    layout = {"pool_bytes": max(expert_bytes, attention_bytes)}
    if not pooled:
        return layout
    region_b = -(-layout["pool_bytes"] // 4096) * 4096
    # One layer's attention input projections, as laid out by pool_views.
    layout["attention_bytes"] = max(
        sum(storage(name) for name in names) for names in (KDA[:4], MLA[:2])
    )
    if managed:
        # Region A is the arena the manager places the router, attention
        # projections, expert slots and dense slabs in.
        region_b = (
            -(-max(expert_bytes, layout["attention_bytes"], dense_bytes) // 4096) * 4096
        )
        region_b += arena_extra
        layout["arena_bytes"] = region_b
    end = region_b
    # Three disjoint lifetimes share region B: MLA attention, MoE routing and
    # shared expert, MoE tail.
    for names in (MLA_EXTRA_WEIGHTS, MOE_PREPARE_WEIGHTS, ("latent_up",)):
        offset = region_b
        for name in names:
            layout[name] = offset
            offset += storage(name)
        end = max(end, offset)
    layout["m_o"] = 0
    if storage("m_o") > storage("m_qa"):
        _err_msg = "The MLA output projection must fit over the q-down projection"
        raise ValueError(_err_msg)
    layout["pool_bytes"] = end
    return layout


def _vmem_bytes(spec: _Dynamic) -> int:
    """Tiled VMEM bytes of a scratch spec (0 for semaphores and SMEM)."""
    if str(getattr(spec, "memory_space", "")) != "vmem":
        return 0
    shape, itemsize = tuple(spec.shape), jnp.dtype(spec.dtype).itemsize
    if len(shape) < _CONST_2:
        return -(-math.prod(shape) * itemsize // 512) * 512
    packing = max(1, 4 // itemsize)
    rows, tile_rows = shape[-2], 8 * packing
    if rows < tile_rows:
        # Short second-minor dimensions use smaller native tiles (1 row for
        # 32-bit types, 2 or 4 for BF16), as in the compiler's reports.
        padded = max(packing, 1 << (rows - 1).bit_length())
    else:
        padded = -(-rows // tile_rows) * tile_rows
    return math.prod(shape[:-2]) * padded * (-(-shape[-1] // 128) * 128) * itemsize


VMEM_LIMIT_BYTES = _config().vmem_limit_mb * 1024**2
# Register-allocator spill slots are one program-wide VMEM area the compiler
# sizes; the budget only warns against this reserve.
VMEM_SPILL_RESERVE_BYTES = int(_config().vmem_spill_reserve_mb * 1024**2)


def _vmem_budget(
    persistent: _Dynamic,
    regions: _Dynamic,
    *,
    limit: _Dynamic = None,
    spill_reserve: _Dynamic = None,
) -> _Dynamic:
    """Check a static VMEM plan and render it as a table.

    ``persistent`` is [(name, spec)], live for the whole kernel. ``regions`` is
    {region: ([(name, spec)] held for the region, [[(name, spec)], ...] nested
    alternatives)}; regions never overlap each other, and a region's nested
    alternatives never overlap each other. Returns (report, peak bytes).
    """
    limit = VMEM_LIMIT_BYTES if limit is None else limit
    spill_reserve = VMEM_SPILL_RESERVE_BYTES if spill_reserve is None else spill_reserve

    def mib(value: _Dynamic) -> _Dynamic:
        """Execute mib logic."""
        return f"{value / 2**20:8.2f}"

    lines = ["persistent VMEM (MiB):"]
    persistent_bytes = 0
    for name, spec in persistent:
        size = _vmem_bytes(spec)
        persistent_bytes += size
        if size >= 64 * 1024:
            spec_desc = f"{tuple(spec.shape)} {jnp.dtype(spec.dtype).name}"
            lines.append(f"  {mib(size)}  {name} {spec_desc}")
    lines.append(f"  {mib(persistent_bytes)}  total (entries under 64 KiB included)")
    peaks = {}
    for region, (held, nested) in regions.items():
        held_bytes = sum(_vmem_bytes(spec) for _, spec in held)
        nested_bytes = [
            sum(_vmem_bytes(spec) for _, spec in alternative) for alternative in nested
        ]
        peaks[region] = held_bytes + max(nested_bytes, default=0)
        lines.append(f"scoped region {region}: peak {mib(peaks[region]).strip()} MiB")
        for name, spec in held:
            if _vmem_bytes(spec) >= 64 * 1024:
                lines.append(f"  {mib(_vmem_bytes(spec))}  {name}")
        for alternative, size in zip(nested, nested_bytes):
            if size:
                alt_str = ", ".join(name for name, _ in alternative)
                lines.append(f"  {mib(size)}  nested: {alt_str}")
    worst = max(peaks, key=lambda k: peaks[k]) if peaks else None
    peak = persistent_bytes + (peaks[worst] if worst else 0)
    lines.append(
        f"peak {mib(peak).strip()} MiB (persistent + {worst}) + spill reserve "
        f"{mib(spill_reserve).strip()} of {mib(limit).strip()} MiB"
    )
    return "\n".join(lines), peak


def _mla_scoped_scratch(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Transient MLA resources allocated once around each layer invocation.

    (placeholders for the workspace's cache fields, ``mla_dcp.scoped_scratch``
    last; ``dcp_heads`` per rank).
    """
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    local_weights = _p_next(_p_iter, _p_get("local_weights"))
    (
        batch_size,
        sequence_rows,
        dcp_heads,
        dcp_dtype,
        dcp_page_tokens,
    ) = (
        _p_get("batch_size"),
        _p_get("sequence_rows"),
        _p_get("dcp_heads", 3),
        _p_get("dcp_dtype", jnp.bfloat16),
        _p_get("dcp_page_tokens", 256),
    )
    cache = (
        tpu.VMEM((8, 128), jnp.bfloat16),
        tpu.VMEM((8, 128), jnp.bfloat16),
        tpu.SemaphoreType.DMA((1,)),
    )
    return (
        *(
            ()
            if POOLED_WEIGHTS
            else (
                tpu.VMEM(local_weights[name].shape, local_weights[name].dtype)
                for name in MLA[2:6]
            )
        ),
        *cache,
        tpu.VMEM((batch_size, 3, 256), jnp.bfloat16),
        tpu.VMEM((batch_size, 384), jnp.bfloat16),
        tpu.VMEM((batch_size, 3, 256), jnp.bfloat16),
        tpu.VMEM((batch_size, 3, 128), jnp.bfloat16),
        tpu.VMEM((batch_size, 3, 1), jnp.float32),
        tpu.VMEM((batch_size, 3, 1), jnp.float32),
        tpu.VMEM((batch_size, 3, 128), jnp.float32),
        # The DCP page buffers by batch (mla_dcp.page_plan; DcpAttention
        # reads the span and the buffers from the shape).
        *mla_dcp.scoped_scratch(
            batch_size,
            heads=dcp_heads,
            cache_dtype=dcp_dtype,
            page_tokens=dcp_page_tokens,
            sequence_rows=sequence_rows,
            **mla_dcp.page_plan(batch_size, dcp_heads, sequence_rows, dcp_dtype),
        ),
    )


def _moe_scoped_scratch(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Transient MoE resources not held in the shared projection/expert pool."""
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    local_weights = _p_next(_p_iter, _p_get("local_weights"))
    (
        batch_size,
        alias_expert_scales,
        token_scatter_output,
        token_scatter_owners,
        shard_routing,
        group_pending_routes,
    ) = (
        _p_get("batch_size"),
        _p_get("alias_expert_scales"),
        _p_get("token_scatter_output"),
        _p_get("token_scatter_owners", 4),
        _p_kw.get("shard_routing", False),
        _p_kw.get("group_pending_routes", False),
    )
    gate_up = local_weights["expert_gu"]
    down = local_weights["expert_down"]
    latent_size = local_weights["latent_norm"].shape[-1]
    expert_input_size = gate_up.shape[1] * 8
    expert_width = down.shape[1] * 8
    expert_output_size = down.shape[-1]
    accumulator_rows = _expert_accumulator_rows(
        batch_size, group_pending_routes=group_pending_routes
    )
    scale_scratch = (
        ()
        if alias_expert_scales
        else (
            tpu.VMEM(
                (4, *local_weights["expert_gus"].shape[1:]),
                local_weights["expert_gus"].dtype,
            ),
            tpu.VMEM(
                (4, *local_weights["expert_ds"].shape[1:]),
                local_weights["expert_ds"].dtype,
            ),
        )
    )
    owners = token_scatter_owners
    output_shape = (
        (owners + 8, batch_size // owners * 1024 // 128, 128)
        if token_scatter_output
        else (batch_size, 8192)
    )
    return (
        *(
            ()
            if POOLED_WEIGHTS
            else (
                tpu.VMEM(
                    local_weights["latent_up"].shape, local_weights["latent_up"].dtype
                ),
            )
        ),
        tpu.SemaphoreType.DMA,
        tpu.SemaphoreType.DMA((4, 4)),
        *scale_scratch,
        tpu.VMEM((batch_size, 128), jnp.int32),
        tpu.VMEM((batch_size, 128), jnp.float32),
        tpu.VMEM((batch_size, 128), jnp.int32),
        tpu.VMEM((batch_size, 128), jnp.float32),
        tpu.VMEM((batch_size, 1024), jnp.float32),
        tpu.VMEM(
            (batch_size, 1024 if _shared_shards(batch_size) else 7168), jnp.float32
        ),
        tpu.VMEM((batch_size, latent_size), jnp.float32),
        tpu.VMEM((accumulator_rows[0], 4 * expert_width), jnp.float32),
        tpu.VMEM((accumulator_rows[1], expert_output_size), jnp.float32),
        tpu.VMEM((batch_size, 128), jnp.int32),
        *(
            (tpu.VMEM((_compact_rows(batch_size), expert_input_size), jnp.float32),) * 2
            if _compact_rows(batch_size)
            else (tpu.VMEM((1, expert_input_size), jnp.bfloat16),) * 2
        ),
        tpu.SemaphoreType.DMA((10,)),
        tpu.SemaphoreType.DMA((10,)),
        tpu.VMEM(output_shape, jnp.bfloat16),
        tpu.SemaphoreType.DMA((7,) if token_scatter_output else (2,)),
        tpu.SemaphoreType.DMA((7,) if token_scatter_output else (2,)),
        tpu.SemaphoreType.DMA((2,)),
        tpu.SemaphoreType.DMA((2,)),
        *(
            (
                tpu.VMEM((batch_size // 4, 128), jnp.float32),
                tpu.VMEM((batch_size, 128), jnp.float32),
            )
            if shard_routing
            else ()
        ),
    )


# Shard the MoE router's top-k over the local rank pairs in the four-owner
# token-scatter layout (KernelConfig.shard_routing=True).
SHARD_ROUTING = _config().shard_routing


class _DeltaAttentionRunner:
    """Runner for Kimi delta attention (KDA) recurrence in decoder layers."""

    def __getattr__(self, name: _Dynamic) -> _Dynamic:
        """Dynamic attribute access."""
        raise AttributeError(name)

    def __init__(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Initialize delta attention runner."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            self.layer_index,
            self.layer_weights,
            self.auxiliary_vmem,
            self.auxiliary_semaphores,
            self.convolution_output_ref,
            self.recurrent_output_ref,
            self.convolution_ref,
            self.recurrent_ref,
            self.convolution_history_vmem,
            self.recurrent_state_vmem,
            self.states,
            self.attention_dp,
            self.attention_base,
            self.kda_streams,
            self.swap_kda_projections,
            self.kda_shard_swap,
            self.group_rows,
            self.group_input_vmem,
            self.kda_projection_vmem,
            self.pool_semaphores,
            self.communication_workspace,
            self.eps,
            self.overlap_kda_state_reads,
            self.overlap_kda_state_writes,
            self.sequence_rows,
            self.dp_attended_vmem,
            self.dp_compressed_vmem,
            self.project_attention_dp,
            self.normalized_vmem,
            self.attention_vmem,
            self.attention_token_scatter,
            self.persist_kda_state,
            self.stream_kda_state,
            self.kda_state_source,
            self.kda_state_destination,
            self.kda_state_reads,
            self.kda_row_chunk,
            self.slots_ref,
            self.slot_reads,
            self.batch_size,
        ) = (
            _p_next(_p_iter, _p_get("layer_index")),
            _p_next(_p_iter, _p_get("layer_weights")),
            _p_next(_p_iter, _p_get("auxiliary_vmem")),
            _p_next(_p_iter, _p_get("auxiliary_semaphores")),
            _p_next(_p_iter, _p_get("convolution_output_ref")),
            _p_next(_p_iter, _p_get("recurrent_output_ref")),
            _p_next(_p_iter, _p_get("convolution_ref")),
            _p_next(_p_iter, _p_get("recurrent_ref")),
            _p_next(_p_iter, _p_get("convolution_history_vmem")),
            _p_next(_p_iter, _p_get("recurrent_state_vmem")),
            _p_next(_p_iter, _p_get("states")),
            _p_next(_p_iter, _p_get("attention_dp")),
            _p_next(_p_iter, _p_get("attention_base")),
            _p_next(_p_iter, _p_get("kda_streams")),
            _p_next(_p_iter, _p_get("swap_kda_projections")),
            _p_next(_p_iter, _p_get("kda_shard_swap")),
            _p_next(_p_iter, _p_get("group_rows")),
            _p_next(_p_iter, _p_get("group_input_vmem")),
            _p_next(_p_iter, _p_get("kda_projection_vmem")),
            _p_next(_p_iter, _p_get("pool_semaphores")),
            _p_next(_p_iter, _p_get("communication_workspace")),
            _p_next(_p_iter, _p_get("eps")),
            _p_next(_p_iter, _p_get("overlap_kda_state_reads")),
            _p_next(_p_iter, _p_get("overlap_kda_state_writes")),
            _p_next(_p_iter, _p_get("sequence_rows")),
            _p_next(_p_iter, _p_get("dp_attended_vmem")),
            _p_next(_p_iter, _p_get("dp_compressed_vmem")),
            _p_next(_p_iter, _p_get("project_attention_dp")),
            _p_next(_p_iter, _p_get("normalized_vmem")),
            _p_next(_p_iter, _p_get("attention_vmem")),
            _p_next(_p_iter, _p_get("attention_token_scatter")),
            _p_next(_p_iter, _p_get("persist_kda_state")),
            _p_next(_p_iter, _p_get("stream_kda_state")),
            _p_next(_p_iter, _p_get("kda_state_source")),
            _p_next(_p_iter, _p_get("kda_state_destination")),
            _p_next(_p_iter, _p_get("kda_state_reads")),
            _p_next(_p_iter, _p_get("kda_row_chunk")),
            _p_next(_p_iter, _p_get("slots_ref")),
            _p_next(_p_iter, _p_get("slot_reads")),
            _p_next(_p_iter, _p_get("batch_size")),
        )
        self.state_index = self.layer_index - self.layer_index // 4
        self.auxiliary_sources = tuple(
            self.layer_weights[name].at[self.state_index] for name in KDA[4:]
        )
        self.streams = self.kda_streams(self.state_index, self.attention_base)

    def _shard_weights(self, shard: _Dynamic) -> _Dynamic:
        """Execute shard weights logic."""
        return KimiDeltaAttentionWeights(
            self.layer_weights["k_gate"].at[self.state_index],
            self.kda_projection_vmem[0],
            self.layer_weights["k_projection"].at[self.state_index],
            self.kda_projection_vmem[1],
            self.layer_weights["k_fb"].at[self.state_index],
            self.kda_projection_vmem[2],
            self.layer_weights["k_o"].at[self.state_index],
            self.kda_projection_vmem[3],
            self.auxiliary_vmem[0].at[shard],
            self.auxiliary_vmem[1].at[shard],
            self.auxiliary_vmem[2].at[shard],
            self.auxiliary_vmem[3],
            self.pool_semaphores,
        )

    def _run_rolled_shards(
        self,
        projection: _Dynamic,
        shared_state_stream: _Dynamic,
        input_stream: _Dynamic,
        per_shard: int,
        chunks_per_shard: int,
    ) -> None:
        """One KDA body for every held shard (traced index)."""
        copies = self._shard_weights(0).make_projection_copies()
        copies.gate.wait()
        copies.feature_gate.wait()
        self.dp_compressed_vmem[:, :128] = _dot(
            self.group_input_vmem[...], self.kda_projection_vmem[0][...]
        ).astype(jnp.bfloat16)
        guard = self.communication_workspace.reduction_phase[0] >= 0

        @pl.loop(0, self.attention_dp)
        def shard_loop(shard: _Dynamic) -> _Dynamic:
            """Execute shard loop logic."""

            @pl.loop(0, per_shard)
            def project(position: _Dynamic) -> _Dynamic:
                """Execute project logic."""
                _traced_dot_position(
                    input_stream,
                    self.group_input_vmem,
                    shard * per_shard + position,
                    guard,
                    projection.at[position],
                )

            rows = pl.ds(pl.multiple_of(shard * self.group_rows, 8), self.group_rows)
            kda(
                self.group_input_vmem,
                self.group_input_vmem,
                self.convolution_output_ref.at[:, self.state_index].at[rows],
                self.recurrent_output_ref.at[:, self.state_index].at[rows],
                weights=self._shard_weights(shard),
                convolution_history_ref=self.convolution_ref.at[:, self.state_index],
                recurrent_state_ref=self.recurrent_ref.at[:, self.state_index],
                communication_workspace=self.communication_workspace,
                epsilon=self.eps,
                synchronize_devices=False,
                weights_are_prefetched=True,
                sequence_rows=self.sequence_rows,
                streams=self.streams,
                shard_index=shard,
                attended_out=self.dp_attended_vmem.at[shard],
                gate_features_vmem=self.dp_compressed_vmem,
                stage_projections=True,
                state_stream=KdaStateStreamSlice(
                    shared_state_stream,
                    first_chunk=shard * chunks_per_shard,
                    chunks=chunks_per_shard,
                ),
                projected_rows=jnp.concatenate(
                    [projection[c] for c in range(per_shard)],
                    axis=1,
                ),
                first_call=False,
            )

    def _execute_streamed_dp(
        self,
        history_buffers: _Dynamic,
        state_buffers: _Dynamic,
        read_semaphores: _Dynamic,
        write_semaphores: _Dynamic,
    ) -> _Dynamic:
        """Execute streamed DP KDA logic."""
        chunks_per_shard = self.group_rows // self.kda_row_chunk
        shared_state_stream = KdaStateStream(
            self.convolution_ref.at[:, self.state_index],
            self.recurrent_ref.at[:, self.state_index],
            self.convolution_output_ref.at[:, self.state_index],
            self.recurrent_output_ref.at[:, self.state_index],
            history_buffers,
            state_buffers,
            read_semaphores,
            write_semaphores,
            chunk=self.kda_row_chunk,
            chunks=self.attention_dp * chunks_per_shard,
            slots=self.slots_ref if self.slot_reads else None,
            sequence_rows=self.sequence_rows,
        )
        shared_state_stream.prime()
        input_stream = self.streams.get("input")
        rolled_shards = (
            self.attention_dp == _CONST_4
            and self.group_rows in (24, 32)
            and input_stream is not None
            and input_stream.slots is not None
            and self.slot_reads
            and "kda" not in DEBUG_SKIP
        )
        per_shard = chunk_columns = 0
        if rolled_shards:
            if not (input_stream is not None):
                _err_msg = "Assertion failed: input_stream is not None"
                raise AssertionError(_err_msg)
            per_shard = input_stream.chunks // self.attention_dp
            chunk_columns = input_stream.sizes[0]
            pl.run_scoped(
                lambda proj: self._run_rolled_shards(
                    proj,
                    shared_state_stream,
                    input_stream,
                    per_shard,
                    chunks_per_shard,
                ),
                tpu.VMEM((per_shard, self.group_rows, chunk_columns), jnp.bfloat16),
            )
        for shard in range(0 if rolled_shards else self.attention_dp):
            rows = pl.ds(shard * self.group_rows, self.group_rows)
            history_rows = self.convolution_ref.at[:, self.state_index]
            state_rows = self.recurrent_ref.at[:, self.state_index]
            if not self.slot_reads:
                history_rows = history_rows.at[rows]
                state_rows = state_rows.at[rows]
            state_stream = KdaStateStreamSlice(
                shared_state_stream,
                first_chunk=shard * chunks_per_shard,
                chunks=chunks_per_shard,
            )
            if "kda" in DEBUG_SKIP:
                for index in range(state_stream.chunks):
                    state_stream.acquire(index)
                    state_stream.release(index)
                state_stream.finish()
                continue
            kda(
                self.group_input_vmem,
                self.group_input_vmem,
                self.convolution_output_ref.at[:, self.state_index].at[rows],
                self.recurrent_output_ref.at[:, self.state_index].at[rows],
                weights=self._shard_weights(shard),
                convolution_history_ref=history_rows,
                recurrent_state_ref=state_rows,
                communication_workspace=self.communication_workspace,
                epsilon=self.eps,
                synchronize_devices=False,
                weights_are_prefetched=True,
                sequence_rows=self.sequence_rows,
                streams=self.streams,
                shard_index=shard,
                attended_out=self.dp_attended_vmem.at[shard],
                gate_features_vmem=self.dp_compressed_vmem,
                stage_projections=True,
                state_stream=state_stream,
            )
        if self.group_rows > _CONST_8 and "kda" not in DEBUG_SKIP:
            shared_state_stream.finish_rolled()
        else:
            shared_state_stream.finish()
        if "project" not in DEBUG_SKIP:
            self.project_attention_dp(
                self.kda_projection_vmem[3], self.streams.get("output")
            )

    def _execute_streamed(
        self,
        history_buffers: _Dynamic,
        state_buffers: _Dynamic,
        read_semaphores: _Dynamic,
        write_semaphores: _Dynamic,
    ) -> _Dynamic:
        """Execute streamed KDA logic."""
        state_stream = KdaStateStream(
            self.convolution_ref.at[:, self.state_index],
            self.recurrent_ref.at[:, self.state_index],
            self.convolution_output_ref.at[:, self.state_index],
            self.recurrent_output_ref.at[:, self.state_index],
            history_buffers,
            state_buffers,
            read_semaphores,
            write_semaphores,
            chunk=self.kda_row_chunk,
            chunks=self.batch_size // self.kda_row_chunk,
            slots=self.slots_ref if self.slot_reads else None,
            sequence_rows=self.sequence_rows,
        )
        state_stream.prime()
        kda(
            self.normalized_vmem,
            self.attention_vmem,
            self.convolution_output_ref.at[:, self.state_index],
            self.recurrent_output_ref.at[:, self.state_index],
            weights=KimiDeltaAttentionWeights(
                self.layer_weights["k_gate"].at[self.state_index],
                self.kda_projection_vmem[0],
                self.layer_weights["k_projection"].at[self.state_index],
                self.kda_projection_vmem[1],
                self.layer_weights["k_fb"].at[self.state_index],
                self.kda_projection_vmem[2],
                self.layer_weights["k_o"].at[self.state_index],
                self.kda_projection_vmem[3],
                self.auxiliary_vmem[0],
                self.auxiliary_vmem[1],
                self.auxiliary_vmem[2],
                self.auxiliary_vmem[3],
                self.pool_semaphores,
            ),
            convolution_history_ref=self.convolution_ref.at[:, self.state_index],
            recurrent_state_ref=self.recurrent_ref.at[:, self.state_index],
            communication_workspace=self.communication_workspace,
            epsilon=self.eps,
            synchronize_devices=False,
            weights_are_prefetched=True,
            token_scatter_output=self.attention_token_scatter,
            sequence_rows=self.sequence_rows,
            streams=self.streams,
            state_stream=state_stream,
        )

    def _execute_scoped(
        self,
        history_vmem: _Dynamic,
        state_vmem: _Dynamic,
        state_semaphores: _Dynamic,
    ) -> _Dynamic:
        """Execute execute scoped logic."""
        state_copies = (
            *self.kda_state_reads(
                self.convolution_ref,
                history_vmem,
                state_semaphores.at[0],
                self.state_index,
            ),
            *self.kda_state_reads(
                self.recurrent_ref,
                state_vmem,
                state_semaphores.at[1],
                self.state_index,
            ),
        )
        for copy in state_copies:
            copy.start()
        if not self.overlap_kda_state_reads:
            for copy in state_copies:
                copy.wait()
        self._execute(history_vmem, state_vmem, state_semaphores, state_copies)

    def _execute_dp(
        self,
        history_vmem: _Dynamic,
        state_vmem: _Dynamic,
        wait_for_state: _Dynamic,
        start_state_writes: _Dynamic,
    ) -> None:
        """Execute DP KDA shards."""
        swapped = (
            self.swap_kda_projections(self.streams["input"])
            if self.kda_shard_swap
            else None
        )
        for order in range(self.attention_dp):
            shard = order
            if self.kda_shard_swap:
                own_shard = jax.lax.axis_index("tp") // (32 // self.attention_dp)
                shard = (own_shard + order) % self.attention_dp
            rows = pl.ds(shard * self.group_rows, self.group_rows)
            kda(
                self.group_input_vmem,
                self.group_input_vmem,
                history_vmem.at[rows],
                state_vmem.at[rows],
                weights=self._shard_weights(shard),
                convolution_history_ref=history_vmem.at[rows],
                recurrent_state_ref=state_vmem.at[rows],
                communication_workspace=self.communication_workspace,
                epsilon=self.eps,
                synchronize_devices=False,
                weights_are_prefetched=True,
                wait_for_state=(
                    wait_for_state
                    if self.overlap_kda_state_reads and order == 0
                    else None
                ),
                on_state_updated=(
                    start_state_writes
                    if self.overlap_kda_state_writes and order == self.attention_dp - 1
                    else None
                ),
                sequence_rows=self.sequence_rows,
                streams=self.streams,
                shard_index=shard,
                attended_out=self.dp_attended_vmem.at[shard],
                gate_features_vmem=self.dp_compressed_vmem,
                stage_projections=True,
                projected_rows=None if swapped is None else swapped(order),
                first_call=order == 0,
            )
        self.project_attention_dp(
            self.kda_projection_vmem[3], self.streams.get("output")
        )

    def _execute_single(
        self,
        history_vmem: _Dynamic,
        state_vmem: _Dynamic,
        wait_for_state: _Dynamic,
        start_state_writes: _Dynamic,
    ) -> None:
        """Execute single shard KDA."""
        kda(
            self.normalized_vmem,
            self.attention_vmem,
            history_vmem,
            state_vmem,
            weights=KimiDeltaAttentionWeights(
                self.layer_weights["k_gate"].at[self.state_index],
                self.kda_projection_vmem[0],
                self.layer_weights["k_projection"].at[self.state_index],
                self.kda_projection_vmem[1],
                self.layer_weights["k_fb"].at[self.state_index],
                self.kda_projection_vmem[2],
                self.layer_weights["k_o"].at[self.state_index],
                self.kda_projection_vmem[3],
                self.auxiliary_vmem[0],
                self.auxiliary_vmem[1],
                self.auxiliary_vmem[2],
                self.auxiliary_vmem[3],
                self.pool_semaphores,
            ),
            convolution_history_ref=history_vmem,
            recurrent_state_ref=state_vmem,
            communication_workspace=self.communication_workspace,
            epsilon=self.eps,
            synchronize_devices=False,
            weights_are_prefetched=True,
            token_scatter_output=self.attention_token_scatter,
            sequence_rows=self.sequence_rows,
            wait_for_state=(wait_for_state if self.overlap_kda_state_reads else None),
            on_state_updated=(
                start_state_writes if self.overlap_kda_state_writes else None
            ),
            streams=self.streams,
        )

    def _execute(
        self,
        history_vmem: _Dynamic,
        state_vmem: _Dynamic,
        state_semaphores: _Dynamic,
        state_reads: _Dynamic = (),
    ) -> _Dynamic:
        """Execute execute logic."""
        state_writes = (
            tpu.make_async_copy(
                history_vmem,
                self.convolution_output_ref.at[:, self.state_index],
                state_semaphores.at[0],
            ),
            tpu.make_async_copy(
                state_vmem,
                self.recurrent_output_ref.at[:, self.state_index],
                state_semaphores.at[1],
            ),
        )

        def wait_for_state() -> _Dynamic:
            """Execute wait for state logic."""
            for copy in state_reads:
                copy.wait()

        def start_state_writes() -> _Dynamic:
            """Execute start state writes logic."""
            for copy in state_writes:
                copy.start()

        if self.attention_dp > 1:
            self._execute_dp(
                history_vmem, state_vmem, wait_for_state, start_state_writes
            )
        else:
            self._execute_single(
                history_vmem, state_vmem, wait_for_state, start_state_writes
            )
        if not self.overlap_kda_state_writes:
            start_state_writes()
        if not self.persist_kda_state:
            for copy in state_writes:
                copy.wait()

    def run(self) -> None:
        """Execute delta attention logic."""
        pending = tuple(
            tpu.make_async_copy(
                source,
                destination,
                self.auxiliary_semaphores.at[index],
            )
            for index, (source, destination) in enumerate(
                zip(self.auxiliary_sources, self.auxiliary_vmem[:4], strict=True)
            )
        )
        for copy in pending:
            copy.wait()

        if self.persist_kda_state:
            state_sources = (
                self.kda_state_source(self.convolution_ref, self.state_index),
                self.kda_state_source(self.recurrent_ref, self.state_index),
            )
            state_pending = tuple(
                tpu.make_async_copy(
                    source,
                    destination,
                    self.auxiliary_semaphores.at[index + 4],
                )
                for index, (source, destination) in enumerate(
                    zip(
                        state_sources,
                        (
                            self.kda_state_destination(self.convolution_history_vmem),
                            self.kda_state_destination(self.recurrent_state_vmem),
                        ),
                        strict=True,
                    )
                )
            )
            for copy in state_pending:
                copy.wait()
            self._execute(
                self.convolution_history_vmem,
                self.recurrent_state_vmem,
                self.auxiliary_semaphores,
            )
        elif self.stream_kda_state:
            state_buffer_count = (
                4
                if self.batch_size == _CONST_32
                and self.attention_dp == 1
                and not KDA_STAGED
                else 3
            )
            pl.run_scoped(
                self._execute_streamed_dp
                if self.attention_dp > 1
                else self._execute_streamed,
                tpu.VMEM(
                    (
                        state_buffer_count,
                        self.kda_row_chunk,
                        *self.states[0].shape[2:],
                    ),
                    self.states[0].dtype,
                ),
                tpu.VMEM(
                    (
                        state_buffer_count,
                        self.kda_row_chunk,
                        *self.states[1].shape[2:],
                    ),
                    self.states[1].dtype,
                ),
                tpu.SemaphoreType.DMA((state_buffer_count, 2)),
                tpu.SemaphoreType.DMA((state_buffer_count, 2)),
            )
        else:
            pl.run_scoped(
                self._execute_scoped,
                tpu.VMEM(
                    (self.batch_size, *self.states[0].shape[2:]),
                    self.states[0].dtype,
                ),
                tpu.VMEM(
                    (self.batch_size, *self.states[1].shape[2:]),
                    self.states[1].dtype,
                ),
                tpu.SemaphoreType.DMA((2,)),
            )


class _LatentAttentionRunner:
    """Runner for multi-head latent attention (MLA) in decoder layers."""

    def __getattr__(self, name: _Dynamic) -> _Dynamic:
        """Dynamic attribute access."""
        raise AttributeError(name)

    def __init__(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Initialize latent attention runner."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            self.layer_index,
            self.query_norms_vmem,
            self.key_value_norms_vmem,
            self.metadata_semaphores,
            self.attention_dp,
            self.attention_base,
            self.pool_view,
            self.weight_stream,
            self.layer_weights,
            self.pool_layout,
            self.query_down_weight_stream,
            self.mla_projection_vmem,
            self.pool_semaphores,
            self.communication_workspace,
            self.eps,
            self.attention_token_scatter,
            self.sequence_rows,
            self.group_rows_per_pair,
            self.tokens_per_pair,
            self.group_rows,
            self.group_positions_smem,
            self.group_input_vmem,
            self.dp_attended_vmem,
            self.pool_ref,
            self.table_ref,
            self.paged_pages,
            self.dcp_owner_table,
            self.dcp_persistent,
            self.position_ref,
            self.normalized_vmem,
            self.attention_vmem,
            self.key_output_ref,
            self.value_output_ref,
            self.batch_size,
            self.cache_handoff_head_count,
            self.mla_scoped_scratch,
            self.project_attention_dp,
        ) = (
            _p_next(_p_iter, _p_get("layer_index")),
            _p_next(_p_iter, _p_get("query_norms_vmem")),
            _p_next(_p_iter, _p_get("key_value_norms_vmem")),
            _p_next(_p_iter, _p_get("metadata_semaphores")),
            _p_next(_p_iter, _p_get("attention_dp")),
            _p_next(_p_iter, _p_get("attention_base")),
            _p_next(_p_iter, _p_get("pool_view")),
            _p_next(_p_iter, _p_get("weight_stream")),
            _p_next(_p_iter, _p_get("layer_weights")),
            _p_next(_p_iter, _p_get("pool_layout")),
            _p_next(_p_iter, _p_get("query_down_weight_stream")),
            _p_next(_p_iter, _p_get("mla_projection_vmem")),
            _p_next(_p_iter, _p_get("pool_semaphores")),
            _p_next(_p_iter, _p_get("communication_workspace")),
            _p_next(_p_iter, _p_get("eps")),
            _p_next(_p_iter, _p_get("attention_token_scatter")),
            _p_next(_p_iter, _p_get("sequence_rows")),
            _p_next(_p_iter, _p_get("group_rows_per_pair")),
            _p_next(_p_iter, _p_get("tokens_per_pair")),
            _p_next(_p_iter, _p_get("group_rows")),
            _p_next(_p_iter, _p_get("group_positions_smem")),
            _p_next(_p_iter, _p_get("group_input_vmem")),
            _p_next(_p_iter, _p_get("dp_attended_vmem")),
            _p_next(_p_iter, _p_get("pool_ref")),
            _p_next(_p_iter, _p_get("table_ref")),
            _p_next(_p_iter, _p_get("paged_pages")),
            _p_next(_p_iter, _p_get("dcp_owner_table")),
            _p_next(_p_iter, _p_get("dcp_persistent")),
            _p_next(_p_iter, _p_get("position_ref")),
            _p_next(_p_iter, _p_get("normalized_vmem")),
            _p_next(_p_iter, _p_get("attention_vmem")),
            _p_next(_p_iter, _p_get("key_output_ref")),
            _p_next(_p_iter, _p_get("value_output_ref")),
            _p_next(_p_iter, _p_get("batch_size")),
            _p_next(_p_iter, _p_get("cache_handoff_head_count")),
            _p_next(_p_iter, _p_get("mla_scoped_scratch")),
            _p_next(_p_iter, _p_get("project_attention_dp")),
        )
        self.state_index = self.layer_index // 4
        self.buffer_slot = self.layer_index % 2
        self.query_norm_vmem = self.query_norms_vmem.at[self.buffer_slot]
        self.key_value_norm_vmem = self.key_value_norms_vmem.at[self.buffer_slot]
        self.layer_metadata_semaphores = self.metadata_semaphores.at[self.buffer_slot]
        self.dp_output = None
        if self.attention_dp > 1:
            output_view = self.pool_view("m_o", base=self.attention_base)
            self.dp_output = (
                output_view,
                self.weight_stream(
                    "m_o",
                    self.layer_weights["m_o"].at[self.state_index],
                    output_view,
                    offset=self.pool_layout["m_o"] + self.attention_base,
                ),
            )

    def _hand_out_rows(
        self,
        semaphores: _Dynamic,
        new_key_vmem: _Dynamic,
        new_value_vmem: _Dynamic,
    ) -> None:
        """Execute hand out rows logic."""
        copies = (
            tpu.make_async_copy(
                new_key_vmem,
                self.key_output_ref.at[:, self.state_index],
                semaphores.at[0],
            ),
            tpu.make_async_copy(
                new_value_vmem,
                self.value_output_ref.at[:, self.state_index],
                semaphores.at[1],
            ),
        )
        for copy in copies:
            copy.start()
        for copy in copies:
            copy.wait()

    def _group_table_row(self, row: _Dynamic, data_group: _Dynamic) -> _Dynamic:
        """Execute group table row logic."""
        return (
            row // self.group_rows_per_pair * self.tokens_per_pair
            + data_group * self.group_rows_per_pair
            + row % self.group_rows_per_pair
        )

    def _compute(
        self,
        new_key_vmem: _Dynamic,
        new_value_vmem: _Dynamic,
        *workspace: _Dynamic,
    ) -> _Dynamic:
        """Execute compute logic."""
        count = len(mla_dcp.SCOPED_NAMES)
        workspace, dcp_scoped = workspace[:-count], workspace[-count:]
        metadata_copies = (
            tpu.make_async_copy(
                self.layer_weights["m_qnorm"].at[self.state_index],
                self.query_norm_vmem,
                self.layer_metadata_semaphores.at[0],
            ),
            tpu.make_async_copy(
                self.layer_weights["m_knorm"].at[self.state_index],
                self.key_value_norm_vmem,
                self.layer_metadata_semaphores.at[1],
            ),
        )
        for copy in metadata_copies:
            copy.wait()
        if POOLED_WEIGHTS:
            projection_vmem = tuple(
                self.pool_view(
                    name,
                    base=self.attention_base if name == "m_o" else 0,
                )
                for name in MLA[2:6]
            )
        else:
            projection_vmem, workspace = workspace[:4], workspace[4:]
        mla_streams = {
            key: stream
            for key, name, view in zip(
                ("query_up", "key_value_up", "output_gate", "output"),
                MLA[2:6],
                projection_vmem,
                strict=True,
            )
            if (
                stream := self.dp_output[1]
                if self.dp_output and name == "m_o"
                else self.weight_stream(
                    name,
                    self.layer_weights[name].at[self.state_index],
                    view,
                    offset=(
                        self.pool_layout[name]
                        + (self.attention_base if name == "m_o" else 0)
                        if POOLED_WEIGHTS
                        else None
                    ),
                )
            )
            is not None
        }
        query_down_stream = self.query_down_weight_stream(
            self.state_index, self.attention_base
        )
        if query_down_stream is not None:
            mla_streams["query_down"] = query_down_stream
        mla_options = {
            "weights": MultiHeadLatentAttentionWeights(
                self.layer_weights["m_qa"].at[self.state_index],
                self.mla_projection_vmem[0],
                self.layer_weights["m_ka"].at[self.state_index],
                self.mla_projection_vmem[1],
                self.layer_weights["m_qb"].at[self.state_index],
                projection_vmem[0],
                self.layer_weights["m_kb"].at[self.state_index],
                projection_vmem[1],
                self.layer_weights["m_gate"].at[self.state_index],
                projection_vmem[2],
                self.layer_weights["m_o"].at[self.state_index],
                projection_vmem[3],
                self.query_norm_vmem,
                self.key_value_norm_vmem,
                self.pool_semaphores,
            ),
            "workspace": MultiHeadLatentAttentionWorkspace(*workspace),
            "output_projection_after_query_down": POOLED_WEIGHTS,
            "streams": mla_streams,
            "communication_workspace": self.communication_workspace,
            "epsilon": self.eps,
            "synchronize_devices": False,
            "weights_are_prefetched": True,
            "token_scatter_output": self.attention_token_scatter,
            "sequence_rows": self.sequence_rows,
        }
        if self.attention_dp > 1:
            data_group = jax.lax.axis_index("tp") // (32 // self.attention_dp)
            group_handoff = pl.ds(0, self.group_rows)
            mla_dcp_shards(
                self.group_positions_smem,
                self.group_input_vmem,
                new_key_vmem.at[group_handoff],
                new_value_vmem.at[group_handoff],
                self.dp_attended_vmem,
                weights=mla_options["weights"],
                streams=mla_streams,
                communication_workspace=self.communication_workspace,
                dcp=(
                    mla_dcp.DcpCache(
                        self.pool_ref,
                        self.table_ref,
                        self.paged_pages,
                        self.state_index,
                        owner=(
                            None
                            if self.dcp_owner_table is None
                            else mla_dcp.owner_index(self.dcp_owner_table)
                            % mla_dcp.GROUP
                        ),
                        table_row=lambda row: self._group_table_row(row, data_group),
                    ),
                    mla_dcp.DcpWorkspace(*self.dcp_persistent, *dcp_scoped),
                ),
                epsilon=self.eps,
                sequence_rows=self.sequence_rows,
            )
        else:
            mla(
                self.position_ref,
                self.normalized_vmem,
                self.attention_vmem,
                new_key_vmem,
                new_value_vmem,
                dcp=(
                    mla_dcp.DcpCache(
                        self.pool_ref,
                        self.table_ref,
                        self.paged_pages,
                        self.state_index,
                        owner=(
                            None
                            if self.dcp_owner_table is None
                            else mla_dcp.owner_index(self.dcp_owner_table)
                        ),
                    ),
                    mla_dcp.DcpWorkspace(*self.dcp_persistent, *dcp_scoped),
                ),
                **mla_options,
            )

        commit_scope = _phase_start("q8_mla_commit")
        pl.run_scoped(
            lambda semaphores: self._hand_out_rows(
                semaphores, new_key_vmem, new_value_vmem
            ),
            tpu.SemaphoreType.DMA((2,)),
        )
        _phase_end(commit_scope)

    def run(self) -> None:
        """Execute MLA logic."""
        pl.run_scoped(
            self._compute,
            tpu.VMEM(
                (self.batch_size, self.cache_handoff_head_count, 256),
                jnp.bfloat16,
            ),
            tpu.VMEM(
                (self.batch_size, self.cache_handoff_head_count, 128),
                jnp.bfloat16,
            ),
            *self.mla_scoped_scratch,
        )
        if self.dp_output is not None:
            self.project_attention_dp(*self.dp_output)


class _DenseMlpRunner:
    """Runner for layer 0's dense MLP with weights streamed through the pool."""

    def __getattr__(self, name: _Dynamic) -> _Dynamic:
        """Dynamic attribute access."""
        raise AttributeError(name)

    def __init__(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Initialize dense MLP runner."""
        _p_iter = iter(_p_args)
        (
            self.dense_refs,
            self.dense_slabs,
            self.dense_semaphores,
            self.projection_expert_pool_vmem,
            self.pool,
            self.pool_alias,
            self.normalized_vmem,
            self.after_experts,
            self.communication_workspace,
            self.owned_rows_vmem,
            self.mixture_vmem,
            self.moe_token_scatter_output,
            self.attention_dp,
            self.sp_owners,
            self.batch_size,
            self.sequence_rows,
            self.hidden_size,
            self.owned_rows,
            self.collectives32,
        ) = (
            _dyn(next(_p_iter, _p_kw.get("dense_refs"))),
            _dyn(next(_p_iter, _p_kw.get("dense_slabs"))),
            _dyn(next(_p_iter, _p_kw.get("dense_semaphores"))),
            _dyn(next(_p_iter, _p_kw.get("projection_expert_pool_vmem"))),
            _dyn(next(_p_iter, _p_kw.get("pool"))),
            _dyn(next(_p_iter, _p_kw.get("pool_alias"))),
            _dyn(next(_p_iter, _p_kw.get("normalized_vmem"))),
            _dyn(next(_p_iter, _p_kw.get("after_experts"))),
            _dyn(next(_p_iter, _p_kw.get("communication_workspace"))),
            _dyn(next(_p_iter, _p_kw.get("owned_rows_vmem"))),
            _dyn(next(_p_iter, _p_kw.get("mixture_vmem"))),
            _dyn(next(_p_iter, _p_kw.get("moe_token_scatter_output"))),
            _dyn(next(_p_iter, _p_kw.get("attention_dp"))),
            _dyn(next(_p_iter, _p_kw.get("sp_owners"))),
            _dyn(next(_p_iter, _p_kw.get("batch_size"))),
            _dyn(next(_p_iter, _p_kw.get("sequence_rows"))),
            _dyn(next(_p_iter, _p_kw.get("hidden_size"))),
            _dyn(next(_p_iter, _p_kw.get("owned_rows"))),
            _dyn(next(_p_iter, _p_kw.get("collectives32"))),
        )

        gu_ref = self.dense_refs["dense_gu"]
        down_ref = self.dense_refs["dense_down"]
        self.gu_ref = gu_ref
        self.down_ref = down_ref
        self.k_block = 1024
        self.down_block = 384
        self.gu_width = gu_ref.shape[1]
        self.gu_blocks = self.hidden_size // self.k_block
        self.down_blocks = down_ref.shape[0] // self.down_block
        if gu_ref.shape != (self.gu_blocks * self.k_block, self.gu_width):
            _err_msg = (
                "Assertion failed: gu_ref.shape == (gu_blocks * k_block, gu_width)"
            )
            raise AssertionError(_err_msg)
        if down_ref.shape != (self.down_blocks * self.down_block, self.hidden_size):
            _err_msg = (
                "Assertion failed: down_ref.shape == (down_blocks * down_block, "
                "hidden_size)"
            )
            raise AssertionError(_err_msg)
        if self.gu_width != 2 * self.down_blocks * self.down_block:
            _err_msg = "Assertion failed: gu_width == 2 * down_blocks * down_block"
            raise AssertionError(_err_msg)
        self.slab_rows = (
            max(
                self.pool_alias.storage_size(
                    (self.k_block, self.gu_width), gu_ref.dtype
                ),
                self.pool_alias.storage_size(
                    (self.down_block, self.hidden_size), down_ref.dtype
                ),
            )
            // 128
        )
        self.slabs = self.dense_slabs
        if not (
            self.slabs * self.slab_rows <= self.projection_expert_pool_vmem.shape[0]
        ):
            _err_msg = (
                "Assertion failed: slabs * slab_rows <= "
                "projection_expert_pool_vmem.shape[0]"
            )
            raise AssertionError(_err_msg)
        self.total = self.gu_blocks + self.down_blocks
        self.slab_base = 0
        if self.pool is not None:
            self.slab_base = (
                self.pool.allocate("dense", self.slabs * self.slab_rows * 128) // 128
            )
            self.pool.acquire("dense")

    def _slab(self, index: _Dynamic, shape: _Dynamic, dtype: _Dynamic) -> _Dynamic:
        """Execute slab logic."""
        return self.pool_alias.view(
            self.projection_expert_pool_vmem.at[
                pl.ds(
                    self.slab_base + index * self.slab_rows,
                    self.pool_alias.storage_size(shape, dtype) // 128,
                ),
                :,
            ],
            shape,
            dtype,
        )

    def _block_copy(self, k: _Dynamic) -> _Dynamic:
        """Execute block copy logic."""
        s = k % self.slabs
        if k < self.gu_blocks:
            return tpu.make_async_copy(
                self.gu_ref.at[pl.ds(k * self.k_block, self.k_block), :],
                self._slab(s, (self.k_block, self.gu_width), self.gu_ref.dtype),
                self.dense_semaphores.at[s],
            )
        r = k - self.gu_blocks
        return tpu.make_async_copy(
            self.down_ref.at[pl.ds(r * self.down_block, self.down_block), :],
            self._slab(s, (self.down_block, self.hidden_size), self.down_ref.dtype),
            self.dense_semaphores.at[s],
        )

    def _step_gu_block(
        self,
        k: _Dynamic,
        s: _Dynamic,
        hidden_in: _Dynamic,
        proj_state: tuple[_Dynamic, _Dynamic],
    ) -> tuple[_Dynamic, _Dynamic]:
        """Accumulate gate/up projection block."""
        projection_vmem, projection = proj_state
        part = _dot(
            hidden_in[:, k * self.k_block : (k + 1) * self.k_block],
            self._slab(s, (self.k_block, self.gu_width), self.gu_ref.dtype)[...],
        )
        if projection_vmem is not None:
            if k == 0:
                projection_vmem[...] = part
            else:
                projection_vmem[...] += part
        else:
            projection = part if projection is None else projection + part
        activation = None
        if k == self.gu_blocks - 1:
            if projection_vmem is not None:
                projection = projection_vmem[...]
            half = self.gu_width // 2
            if not (projection is not None):
                _err_msg = "Assertion failed: projection is not None"
                raise AssertionError(_err_msg)
            activation = _gated_activation(
                projection[:, :half].astype(jnp.bfloat16),
                projection[:, half:].astype(jnp.bfloat16),
            )
            if projection_vmem is not None:
                projection_vmem[:, :half] = activation.astype(jnp.float32)
        return projection, activation

    def _step_down_block(
        self,
        k: _Dynamic,
        s: _Dynamic,
        activation: _Dynamic,
        buffers: tuple[_Dynamic, _Dynamic, _Dynamic],
    ) -> _Dynamic:
        """Accumulate down projection block."""
        projection_vmem, accumulator_vmem, accumulator = buffers
        r = k - self.gu_blocks
        if not (activation is not None):
            _err_msg = "Assertion failed: activation is not None"
            raise AssertionError(_err_msg)
        if projection_vmem is not None:
            activation_block = projection_vmem[
                :, r * self.down_block : (r + 1) * self.down_block
            ].astype(activation.dtype)
        else:
            activation_block = activation[
                :, r * self.down_block : (r + 1) * self.down_block
            ]
        down_slab = self._slab(
            s, (self.down_block, self.hidden_size), self.down_ref.dtype
        )
        part = _dot(activation_block, down_slab[...])
        if accumulator_vmem is not None:
            if r == 0:
                accumulator_vmem[...] = part
            else:
                accumulator_vmem[...] += part
        else:
            accumulator = part if accumulator is None else accumulator + part
        return accumulator

    def _stream_blocks(
        self,
        projection_vmem: _Dynamic = None,
        accumulator_vmem: _Dynamic = None,
    ) -> _Dynamic:
        """Execute stream blocks logic."""
        hidden_in = self.normalized_vmem[...]
        projection, activation, accumulator = None, None, None
        for k in range(self.total):
            s = k % self.slabs
            self._block_copy(k).wait()
            if k < self.gu_blocks:
                projection, act = self._step_gu_block(
                    k, s, hidden_in, (projection_vmem, projection)
                )
                if act is not None:
                    activation = act
            else:
                accumulator = self._step_down_block(
                    k,
                    s,
                    activation,
                    (projection_vmem, accumulator_vmem, accumulator),
                )
            if k + self.slabs < self.total:
                self._block_copy(k + self.slabs).start()
        return accumulator

    def _reduce_half(self, accumulator: _Dynamic, first: _Dynamic) -> _Dynamic:
        """Execute reduce half logic."""
        if first:
            self.collectives32.barrier()
        return _all_reduce_rows(
            accumulator[first : first + 8],
            self.communication_workspace.shared_local_reduction_vmem,
            self.communication_workspace.shared_host_reduction_vmem,
            self.communication_workspace.send_semaphores,
            self.communication_workspace.receive_semaphores,
        )

    def _finish_output(self, accumulator: _Dynamic) -> None:
        """Finish output reduction and scatter."""
        if (
            self.moe_token_scatter_output
            and self.attention_dp == _CONST_2
            and self.sp_owners == _CONST_4
            and self.batch_size in (64, 128)
            and self.sequence_rows == 1
        ):
            _dense_scatter_rows(
                accumulator,
                self.owned_rows_vmem,
                self.communication_workspace.shared_local_reduction_vmem,
                self.communication_workspace.shared_host_reduction_vmem,
                self.communication_workspace.send_semaphores,
                self.communication_workspace.receive_semaphores,
                self.sp_owners,
            )
        else:
            reduced = (
                jnp.concatenate(
                    [
                        self._reduce_half(accumulator, first)
                        for first in range(0, self.batch_size, 8)
                    ],
                    axis=0,
                )
                if self.batch_size > _CONST_8
                else _all_reduce_rows(
                    accumulator,
                    self.communication_workspace.shared_local_reduction_vmem,
                    self.communication_workspace.shared_host_reduction_vmem,
                    self.communication_workspace.send_semaphores,
                    self.communication_workspace.receive_semaphores,
                )
            )
            if self.moe_token_scatter_output:
                rank = jax.lax.axis_index("tp")
                pair = (rank % 8) // (8 // self.sp_owners)
                for owner_pair in range(self.sp_owners):

                    @pl.when(pair == owner_pair)
                    def store_owned_dense_rows(
                        *, owner_pair: _Dynamic = owner_pair
                    ) -> _Dynamic:
                        """Execute store owned dense rows logic."""
                        self.owned_rows_vmem[...] = reduced[
                            owner_pair * self.owned_rows : (owner_pair + 1)
                            * self.owned_rows
                        ].astype(jnp.bfloat16)
            else:
                self.mixture_vmem[...] = reduced.astype(jnp.bfloat16)

    def run(self) -> _Dynamic:
        """Execute dense MLP run logic."""
        for k in range(min(self.slabs, self.total)):
            self._block_copy(k).start()

        if self.batch_size <= _CONST_8:
            accumulator = self._stream_blocks()
        else:
            accumulator = pl.run_scoped(
                lambda projection_vmem, accumulator_vmem: (
                    self._stream_blocks(projection_vmem, accumulator_vmem),
                    accumulator_vmem[...],
                )[1],
                tpu.VMEM((self.batch_size, self.gu_width), jnp.float32),
                tpu.VMEM((self.batch_size, self.hidden_size), jnp.float32),
            )
        if self.pool is not None:
            self.pool.release("dense")
        self.after_experts()
        self._finish_output(accumulator)
        return accumulator


class _LayerRunner:
    """Runner for a single decoder layer within the megakernel loop."""

    def __getattr__(self, name: _Dynamic) -> _Dynamic:
        """Dynamic attribute access."""
        raise AttributeError(name)

    def __init__(self, ctx: _Dynamic) -> None:
        """Initialize layer runner with context."""
        self.ctx = ctx

    def capture_aux_hidden(
        self, source: _Dynamic = None, gather_owned_rows: _Dynamic = None
    ) -> _Dynamic:
        """Execute capture aux hidden logic."""
        layer_index = getattr(self, "layer_index", 0)
        (
            attention_dp_ownership,
            aux_hidden_layers,
            aux_output_ref,
            aux_semaphore,
            communication_workspace,
            gather_dp_rows,
            normalized_vmem,
            owned_rows_vmem,
            prefix_vmem,
            sp_owners,
        ) = (
            getattr(self.ctx, a)
            for a in (
                "attention_dp_ownership",
                "aux_hidden_layers",
                "aux_output_ref",
                "aux_semaphore",
                "communication_workspace",
                "gather_dp_rows",
                "normalized_vmem",
                "owned_rows_vmem",
                "prefix_vmem",
                "sp_owners",
            )
        )
        source = prefix_vmem if source is None else source
        if gather_owned_rows and attention_dp_ownership:
            # DP ranks hold only their own prefix rows.
            source = normalized_vmem
        for aux_slot, aux_layer in enumerate(aux_hidden_layers):
            # Layer id L names the residual stream entering layer L,
            # which is the prefix after layer L - 1 completes (or, with
            # aux_capture="attention", after its attention).
            # The rows go to HBM so the capture costs no VMEM.
            @pl.when(layer_index == aux_layer - 1)
            def capture(aux_slot: _Dynamic = aux_slot) -> _Dynamic:
                """Execute capture logic."""
                if gather_owned_rows:
                    # Sequence-parallel layers keep only this pair's
                    # prefix rows; complete them for the capture.
                    if attention_dp_ownership:
                        gather_dp_rows(normalized_vmem)
                    else:
                        _gather_local_pair_rows(
                            owned_rows_vmem,
                            prefix_vmem,
                            communication_workspace.send_semaphores,
                            communication_workspace.receive_semaphores,
                            communication_workspace.shared_local_reduction_vmem,
                            owners=sp_owners,
                        )
                aux_copy = tpu.make_async_copy(
                    source, aux_output_ref.at[aux_slot], aux_semaphore.at[0]
                )
                aux_copy.start()
                aux_copy.wait()

    def check_layer(self, layer_index: _Dynamic) -> _Dynamic:
        """Check if layer should be processed."""
        if not self.ctx.include_layer_zero:
            return layer_index > 0
        return True

    def start_norm_copies(
        self, layer_index: _Dynamic, layer_weights: _Dynamic
    ) -> list[_Dynamic]:
        """Start async copies for layer normalization weights."""
        (
            attention_norm_vmem,
            attention_residual_vmem,
            ffn_norm_vmem,
            layer_index,
            layer_weights,
            mlp_residual_vmem,
            norm_semaphores,
        ) = (
            getattr(self.ctx, a)
            for a in (
                "attention_norm_vmem",
                "attention_residual_vmem",
                "ffn_norm_vmem",
                "layer_index",
                "layer_weights",
                "mlp_residual_vmem",
                "norm_semaphores",
            )
        )
        norm_copies = [
            tpu.make_async_copy(
                layer_weights[name].at[layer_index],
                destination,
                norm_semaphores.at[index],
            )
            for index, (name, destination) in enumerate(
                (
                    ("attn_res", attention_residual_vmem),
                    ("mlp_res", mlp_residual_vmem),
                    ("attn_norm", attention_norm_vmem),
                    ("ffn_norm", ffn_norm_vmem),
                )
            )
        ]
        norm_copies[0].wait()
        norm_copies[2].wait()

        return norm_copies

    def pre_norm(self, layer_index: _Dynamic) -> None:
        """Execute pre-norm logic."""
        layer_scope = getattr(self, "layer_scope", [])
        (
            attention_dp_ownership,
            attention_residual_vmem,
            block_rows,
            block_rows_held,
            blocks_block_major,
            blocks_vmem,
            eps,
            layer_index,
            mix_dp_rows,
            normalized_vmem,
            owned_rows,
            owned_rows_vmem,
            owner_block_row,
            owner_indices,
            owner_prefix_row,
            owner_start,
            owner_when,
            prefix_vmem,
            residual,
            residual_reduction_order,
            sequence_parallel_pre_attention,
            sp_owners,
            store_mixed_rows,
        ) = (
            getattr(self.ctx, a)
            for a in (
                "attention_dp_ownership",
                "attention_residual_vmem",
                "block_rows",
                "block_rows_held",
                "blocks_block_major",
                "blocks_vmem",
                "eps",
                "layer_index",
                "mix_dp_rows",
                "normalized_vmem",
                "owned_rows",
                "owned_rows_vmem",
                "owner_block_row",
                "owner_indices",
                "owner_prefix_row",
                "owner_start",
                "owner_when",
                "prefix_vmem",
                "residual",
                "residual_reduction_order",
                "sequence_parallel_pre_attention",
                "sp_owners",
                "store_mixed_rows",
            )
        )
        if attention_dp_ownership:
            mix_dp_rows(
                attention_residual_vmem,
                (layer_index - 1) // 12 + 1,
                layer_index=layer_index,
                start_block=True,
            )
        elif sequence_parallel_pre_attention:
            rank = jax.lax.axis_index("tp")
            pair = (rank % 8) // (8 // sp_owners)
            for owner_pair in owner_indices(pair):

                @owner_when(pair, owner_pair)
                def prepare_owned_attention_rows(
                    *, owner_pair: _Dynamic = owner_pair
                ) -> _Dynamic:
                    """Execute prepare owned attention rows logic."""
                    row_offset = owner_pair * owned_rows
                    row_slice = pl.ds(owner_start(row_offset), owned_rows)

                    def mix_row(local_row: _Dynamic) -> _Dynamic:
                        """Execute mix row logic."""
                        global_row = row_offset + local_row
                        return _residual_mixture_row(
                            block_rows(owner_block_row(global_row, local_row)),
                            owner_prefix_row(global_row, row_slice, local_row),
                            attention_residual_vmem,
                            (layer_index - 1) // 12 + 1,
                            epsilon=eps,
                            relaxed_reduction_order=(
                                residual_reduction_order == "relaxed"
                            ),
                        )

                    store_mixed_rows(mix_row, owned_rows_vmem)

                    @pl.when((layer_index % 12 == 0) if blocks_block_major else False)
                    def start_owned_residual_block_major() -> _Dynamic:
                        """Execute start owned residual block major logic."""
                        for local_row in range(owned_rows):
                            global_row = row_offset + local_row
                            row = (
                                owner_block_row(global_row, local_row) % block_rows_held
                            )
                            half = blocks_vmem.at[row // 2, pl.ds((row % 2) * 8, 8)]
                            block_mask = (
                                jax.lax.broadcasted_iota(jnp.int32, half.shape, 0)
                                == layer_index // 12
                            )
                            half[...] = jnp.where(
                                block_mask,
                                owner_prefix_row(global_row, row_slice, local_row),
                                half[...],
                            )

                    @pl.when(False if blocks_block_major else (layer_index % 12 == 0))
                    def start_owned_residual_block() -> _Dynamic:
                        """Execute start owned residual block logic."""
                        block_mask = (
                            jax.lax.broadcasted_iota(
                                jnp.int32,
                                blocks_vmem[row_slice].shape,
                                1,
                            )
                            == layer_index // 12
                        )
                        blocks_vmem[row_slice] = jnp.where(
                            block_mask,
                            prefix_vmem[row_slice][:, None, :].astype(jnp.float32),
                            blocks_vmem[row_slice],
                        )
        else:
            residual(
                normalized_vmem,
                attention_residual_vmem,
                (layer_index - 1) // 12 + 1,
            )

            @pl.when(layer_index % 12 == 0)
            def start_residual_block() -> _Dynamic:
                """Execute start residual block logic."""
                block_mask = (
                    jax.lax.broadcasted_iota(jnp.int32, blocks_vmem.shape, 1)
                    == layer_index // 12
                )
                blocks_vmem[...] = jnp.where(
                    block_mask,
                    prefix_vmem[...][:, None, :].astype(jnp.float32),
                    blocks_vmem[...],
                )

        _phase_end(layer_scope.pop())

        self.layer_scope = layer_scope

    def pre_norm_gather(self) -> None:
        """Execute pre-norm gather logic."""
        layer_scope = getattr(self, "layer_scope", [])
        (
            attention_dp,
            attention_dp_ownership,
            attention_norm_vmem,
            communication_workspace,
            dp_owned_vmem,
            group_input_vmem,
            group_rows,
            kda_shard_swap,
            normalize,
            normalize_dp_rows,
            normalized_vmem,
            owned_rows_vmem,
            sequence_parallel_pre_attention,
            sp8_owned_value,
            sp8_residual,
            sp_owners,
        ) = (
            getattr(self.ctx, a)
            for a in (
                "attention_dp",
                "attention_dp_ownership",
                "attention_norm_vmem",
                "communication_workspace",
                "dp_owned_vmem",
                "group_input_vmem",
                "group_rows",
                "kda_shard_swap",
                "normalize",
                "normalize_dp_rows",
                "normalized_vmem",
                "owned_rows_vmem",
                "sequence_parallel_pre_attention",
                "sp8_owned_value",
                "sp8_residual",
                "sp_owners",
            )
        )
        if kda_shard_swap:
            # Every row is the swapped projection's input; attention reads
            # the group's.
            normalize(normalized_vmem, attention_norm_vmem)
            data_group = jax.lax.axis_index("tp") // (32 // attention_dp)
            rows = normalized_vmem[...]
            group_input_vmem[...] = jnp.where(
                data_group == 0, rows[:group_rows], rows[group_rows:]
            )
        elif attention_dp_ownership:
            normalize_dp_rows(attention_norm_vmem, split=sp8_residual)
            with _phase("t0_dp_group_gather"):
                if sp8_residual:
                    _gather_sp8_local_rows(
                        sp8_owned_value(),
                        group_input_vmem,
                        communication_workspace.shared_local_reduction_vmem,
                        communication_workspace.send_semaphores,
                        communication_workspace.receive_semaphores,
                    )
                else:
                    _gather_local_pair_rows(
                        dp_owned_vmem,
                        group_input_vmem,
                        communication_workspace.send_semaphores,
                        communication_workspace.receive_semaphores,
                        communication_workspace.shared_local_reduction_vmem,
                        packed=True,
                    )
        elif sequence_parallel_pre_attention:
            normalize(owned_rows_vmem, attention_norm_vmem)
            _gather_local_pair_rows(
                owned_rows_vmem,
                normalized_vmem,
                communication_workspace.send_semaphores,
                communication_workspace.receive_semaphores,
                communication_workspace.shared_local_reduction_vmem,
                owners=sp_owners,
            )
        else:
            normalize(normalized_vmem, attention_norm_vmem)

        _phase_end(layer_scope.pop())

        self.layer_scope = layer_scope

    def attention(self, layer_index: _Dynamic) -> None:
        """Execute attention logic."""
        (attention_base, kda_projection_vmem, mla_projection_vmem) = (
            getattr(self, "attention_base", 0),
            getattr(self, "kda_projection_vmem", None),
            getattr(self, "mla_projection_vmem", None),
        )
        (
            attention_dp,
            attention_token_scatter,
            attention_vmem,
            auxiliary_semaphores,
            auxiliary_vmem,
            batch_size,
            cache_handoff_head_count,
            communication_workspace,
            convolution_history_vmem,
            convolution_output_ref,
            convolution_ref,
            dcp_owner_table,
            dcp_persistent,
            dp_attended_vmem,
            dp_compressed_vmem,
            eps,
            group_input_vmem,
            group_positions_smem,
            group_rows,
            group_rows_per_pair,
            kda_row_chunk,
            kda_shard_swap,
            kda_state_destination,
            kda_state_reads,
            kda_state_source,
            kda_streams,
            key_output_ref,
            key_value_norms_vmem,
            layer_index,
            layer_weights,
            metadata_semaphores,
            mla_scoped_scratch,
            normalized_vmem,
            overlap_kda_state_reads,
            overlap_kda_state_writes,
            paged_pages,
            persist_kda_state,
            pool,
            pool_layout,
            pool_ref,
            pool_semaphores,
            pool_view,
            position_ref,
            project_attention_dp,
            query_down_weight_stream,
            query_norms_vmem,
            recurrent_output_ref,
            recurrent_ref,
            recurrent_state_vmem,
            sequence_rows,
            slot_reads,
            slots_ref,
            states,
            stream_kda_state,
            swap_kda_projections,
            table_ref,
            tokens_per_pair,
            value_output_ref,
            weight_stream,
        ) = (
            getattr(self.ctx, a)
            for a in (
                "attention_dp",
                "attention_token_scatter",
                "attention_vmem",
                "auxiliary_semaphores",
                "auxiliary_vmem",
                "batch_size",
                "cache_handoff_head_count",
                "communication_workspace",
                "convolution_history_vmem",
                "convolution_output_ref",
                "convolution_ref",
                "dcp_owner_table",
                "dcp_persistent",
                "dp_attended_vmem",
                "dp_compressed_vmem",
                "eps",
                "group_input_vmem",
                "group_positions_smem",
                "group_rows",
                "group_rows_per_pair",
                "kda_row_chunk",
                "kda_shard_swap",
                "kda_state_destination",
                "kda_state_reads",
                "kda_state_source",
                "kda_streams",
                "key_output_ref",
                "key_value_norms_vmem",
                "layer_index",
                "layer_weights",
                "metadata_semaphores",
                "mla_scoped_scratch",
                "normalized_vmem",
                "overlap_kda_state_reads",
                "overlap_kda_state_writes",
                "paged_pages",
                "persist_kda_state",
                "pool",
                "pool_layout",
                "pool_ref",
                "pool_semaphores",
                "pool_view",
                "position_ref",
                "project_attention_dp",
                "query_down_weight_stream",
                "query_norms_vmem",
                "recurrent_output_ref",
                "recurrent_ref",
                "recurrent_state_vmem",
                "sequence_rows",
                "slot_reads",
                "slots_ref",
                "states",
                "stream_kda_state",
                "swap_kda_projections",
                "table_ref",
                "tokens_per_pair",
                "value_output_ref",
                "weight_stream",
            )
        )

        def run_delta_attention() -> _Dynamic:
            """Execute run delta attention logic."""
            return _DeltaAttentionRunner(
                layer_index,
                layer_weights,
                auxiliary_vmem,
                auxiliary_semaphores,
                convolution_output_ref,
                recurrent_output_ref,
                convolution_ref,
                recurrent_ref,
                convolution_history_vmem,
                recurrent_state_vmem,
                states,
                attention_dp,
                attention_base,
                kda_streams,
                swap_kda_projections,
                kda_shard_swap,
                group_rows,
                group_input_vmem,
                kda_projection_vmem,
                pool_semaphores,
                communication_workspace,
                eps,
                overlap_kda_state_reads,
                overlap_kda_state_writes,
                sequence_rows,
                dp_attended_vmem,
                dp_compressed_vmem,
                project_attention_dp,
                normalized_vmem,
                attention_vmem,
                attention_token_scatter,
                persist_kda_state,
                stream_kda_state,
                kda_state_source,
                kda_state_destination,
                kda_state_reads,
                kda_row_chunk,
                slots_ref,
                slot_reads,
                batch_size,
            ).run()

        def run_latent_attention() -> _Dynamic:
            """Execute run latent attention logic."""
            return _LatentAttentionRunner(
                layer_index,
                query_norms_vmem,
                key_value_norms_vmem,
                metadata_semaphores,
                attention_dp,
                attention_base,
                pool_view,
                weight_stream,
                layer_weights,
                pool_layout,
                query_down_weight_stream,
                mla_projection_vmem,
                pool_semaphores,
                communication_workspace,
                eps,
                attention_token_scatter,
                sequence_rows,
                group_rows_per_pair,
                tokens_per_pair,
                group_rows,
                group_positions_smem,
                group_input_vmem,
                dp_attended_vmem,
                pool_ref,
                table_ref,
                paged_pages,
                dcp_owner_table,
                dcp_persistent,
                position_ref,
                normalized_vmem,
                attention_vmem,
                key_output_ref,
                value_output_ref,
                batch_size,
                cache_handoff_head_count,
                mla_scoped_scratch,
                project_attention_dp,
            ).run()

        if pool is not None:
            pool.acquire("attention")  # the attention code waits for its copies
        with _phase("p1_attention"):
            jax.lax.cond(
                (layer_index % 4 == _CONST_3) | (layer_index == _CONST_92),
                run_latent_attention,
                run_delta_attention,
            )
        if pool is not None:
            pool.release("attention")

    def post_mix_init(self) -> None:
        """Initialize post-mix."""
        norm_copies = getattr(self, "norm_copies", None)
        (
            attention_dp_ownership,
            ffn_norm_vmem,
            gather_dp_rows,
            layer_index,
            mix_dp_rows,
            mlp_residual_vmem,
            normalize_dp_rows,
            normalized_vmem,
            sp8_residual,
        ) = (
            getattr(self.ctx, a)
            for a in (
                "attention_dp_ownership",
                "ffn_norm_vmem",
                "gather_dp_rows",
                "layer_index",
                "mix_dp_rows",
                "mlp_residual_vmem",
                "normalize_dp_rows",
                "normalized_vmem",
                "sp8_residual",
            )
        )
        post_scope = _phase_start("q9_post_mix")
        with _phase("q9a_post_norm_wait"):
            if norm_copies is not None:
                norm_copies[1].wait()
                norm_copies[3].wait()
        if attention_dp_ownership:
            with _phase("q9b_post_rows"):
                mix_dp_rows(
                    mlp_residual_vmem,
                    layer_index // 12 + 1,
                    layer_index=layer_index,
                    attention_output=True,
                )
            normalize_dp_rows(ffn_norm_vmem, split=sp8_residual)
            _phase_end(post_scope)
            with _phase("t10_moe_input_gather"):
                gather_dp_rows(normalized_vmem, split=sp8_residual)

        self.post_scope = post_scope

    def post_mix_scatter(self, pair: _Dynamic) -> None:
        """Scatter post-mix."""
        post_scope = getattr(self, "post_scope", None)
        (
            block_rows,
            communication_workspace,
            eps,
            ffn_norm_vmem,
            hidden_size,
            layer_index,
            mlp_residual_vmem,
            normalize,
            normalized_vmem,
            owned_rows,
            owned_rows_vmem,
            owner_block_row,
            owner_indices,
            owner_prefix_row,
            owner_start,
            owner_when,
            prefix_vmem,
            residual_reduction_order,
            sp_owners,
            store_mixed_rows,
            tokens_per_pair,
        ) = (
            getattr(self.ctx, a)
            for a in (
                "block_rows",
                "communication_workspace",
                "eps",
                "ffn_norm_vmem",
                "hidden_size",
                "layer_index",
                "mlp_residual_vmem",
                "normalize",
                "normalized_vmem",
                "owned_rows",
                "owned_rows_vmem",
                "owner_block_row",
                "owner_indices",
                "owner_prefix_row",
                "owner_start",
                "owner_when",
                "prefix_vmem",
                "residual_reduction_order",
                "sp_owners",
                "store_mixed_rows",
                "tokens_per_pair",
            )
        )
        rank = jax.lax.axis_index("tp")
        pair = (rank % 8) // (8 // sp_owners)
        owned_attention_vmem = _bf16_rows_view(
            communication_workspace.attention_output_vmem,
            tokens_per_pair,
            hidden_size,
        )
        owned_normalized_vmem = owned_rows_vmem
        # With one owner per rank each rank keeps its half of the
        # pair's attention rows.
        pair_split = sp_owners // 4

        # Each row owner holds ``owned_rows`` rows. Static branches keep
        # source slices aligned while calculations use a purpose-sized
        # workspace.
        if owned_rows == 1:
            prefix_values = prefix_vmem[...]
        for owner_pair in owner_indices(pair):

            @owner_when(pair, owner_pair)
            def prepare_owned_rows(
                owner_pair: _Dynamic = owner_pair,
            ) -> _Dynamic:
                """Execute prepare owned rows logic."""
                attention_rows = (
                    owned_attention_vmem
                    if pair_split == 1
                    else owned_attention_vmem.at[
                        pl.ds(
                            owner_start((owner_pair % pair_split) * owned_rows),
                            owned_rows,
                        )
                    ]
                )
                row_offset = owner_pair * owned_rows
                row_slice = None
                if owned_rows == 1:
                    prefix_rows = prefix_values[owner_pair : owner_pair + 1]
                else:
                    row_slice = pl.ds(owner_start(row_offset), owned_rows)
                    prefix_rows = prefix_vmem[row_slice]
                updated_prefix = jnp.where(
                    layer_index % 12 == 0,
                    attention_rows[...],
                    prefix_rows + attention_rows[...],
                ).astype(jnp.bfloat16)
                if owned_rows == 1:
                    row_indices = jax.lax.broadcasted_iota(
                        jnp.int32, prefix_vmem.shape, 0
                    )
                    prefix_vmem[...] = jnp.where(
                        row_indices == owner_pair,
                        updated_prefix,
                        prefix_values,
                    )
                else:
                    prefix_vmem[row_slice] = updated_prefix

                def mix_row(local_row: _Dynamic) -> _Dynamic:
                    """Execute mix row logic."""
                    global_row = row_offset + local_row
                    return _residual_mixture_row(
                        block_rows(owner_block_row(global_row, local_row)),
                        (
                            updated_prefix[local_row : local_row + 1]
                            if owned_rows <= _CONST_2
                            # stored above; re-read inside the row region
                            else owner_prefix_row(global_row, row_slice, local_row)
                        ),
                        mlp_residual_vmem,
                        layer_index // 12 + 1,
                        epsilon=eps,
                        relaxed_reduction_order=(residual_reduction_order == "relaxed"),
                    )

                with _phase("q9b_post_rows"):
                    store_mixed_rows(mix_row, owned_normalized_vmem)

        normalize(owned_normalized_vmem, ffn_norm_vmem)
        _phase_end(post_scope)
        t10 = _phase_start("t10_moe_input_gather")
        _gather_local_pair_rows(
            owned_normalized_vmem,
            normalized_vmem,
            communication_workspace.send_semaphores,
            communication_workspace.receive_semaphores,
            communication_workspace.shared_local_reduction_vmem,
            owners=sp_owners,
        )
        _phase_end(t10)

    def post_mix_finish(self, layer_index: _Dynamic) -> None:
        """Finish post-mix."""
        capture_aux_hidden = self.capture_aux_hidden
        post_scope = getattr(self, "post_scope", None)
        (
            attention_vmem,
            aux_capture,
            ffn_norm_vmem,
            layer_index,
            mlp_residual_vmem,
            normalize,
            normalized_vmem,
            prefix_vmem,
            residual,
        ) = (
            getattr(self.ctx, a)
            for a in (
                "attention_vmem",
                "aux_capture",
                "ffn_norm_vmem",
                "layer_index",
                "mlp_residual_vmem",
                "normalize",
                "normalized_vmem",
                "prefix_vmem",
                "residual",
            )
        )
        prefix_vmem[...] = jnp.where(
            layer_index % 12 == 0,
            attention_vmem[...],
            prefix_vmem[...] + attention_vmem[...],
        )
        if aux_capture == "attention":
            capture_aux_hidden()
        residual(
            normalized_vmem,
            mlp_residual_vmem,
            layer_index // 12 + 1,
        )
        normalize(normalized_vmem, ffn_norm_vmem)
        _phase_end(post_scope)

    def post_mix(self, layer_index: _Dynamic, pair: _Dynamic) -> None:
        """Execute post-mix logic."""
        self.post_mix_init()
        self.post_mix_scatter(pair)
        self.post_mix_finish(layer_index)

    def after_experts(self) -> _Dynamic:
        """Execute after experts logic."""
        dense_layer = getattr(self, "dense_layer", False)
        (
            auxiliary_semaphores,
            convolution_history_vmem,
            convolution_output_ref,
            layer_index,
            layers,
            persist_kda_state,
            pool,
            prefetch_attention,
            prefetch_auxiliary,
            prefetch_metadata,
            prefetch_norms,
            prefetch_router,
            recurrent_output_ref,
            recurrent_state_vmem,
            request_attention,
            request_router,
        ) = (
            getattr(self.ctx, a)
            for a in (
                "auxiliary_semaphores",
                "convolution_history_vmem",
                "convolution_output_ref",
                "layer_index",
                "layers",
                "persist_kda_state",
                "pool",
                "prefetch_attention",
                "prefetch_auxiliary",
                "prefetch_metadata",
                "prefetch_norms",
                "prefetch_router",
                "recurrent_output_ref",
                "recurrent_state_vmem",
                "request_attention",
                "request_router",
            )
        )
        if persist_kda_state:

            @pl.when((layer_index % 4 != _CONST_3) & (layer_index != _CONST_92))
            def drain_state_writes() -> _Dynamic:
                """Execute drain state writes logic."""
                state_index = layer_index - layer_index // 4
                state_writes = (
                    tpu.make_async_copy(
                        convolution_history_vmem,
                        convolution_output_ref.at[:, state_index],
                        auxiliary_semaphores.at[0],
                    ),
                    tpu.make_async_copy(
                        recurrent_state_vmem,
                        recurrent_output_ref.at[:, state_index],
                        auxiliary_semaphores.at[1],
                    ),
                )
                for copy in state_writes:
                    copy.wait()

        if pool is not None:
            # The expert slots are free: queue the next layer's weights
            # (issued now if the arena has room, else by the release
            # that makes it).
            if not dense_layer:
                pool.release("experts")
            has_next = layer_index + 1 < layers
            pl.when(has_next)(lambda: prefetch_norms(layer_index + 1))
            request_attention(layer_index + 1, predicate=has_next)

            @pl.when(has_next)
            def prefetch_next_layer_metadata() -> _Dynamic:
                """Execute prefetch next layer metadata logic."""
                prefetch_auxiliary(layer_index + 1)
                prefetch_metadata(layer_index + 1)

            request_router(layer_index + 1, predicate=has_next)
            return

        @pl.when(layer_index + 1 < layers)
        def prefetch_next_layer() -> _Dynamic:
            """Execute prefetch next layer logic."""
            prefetch_norms(layer_index + 1)
            prefetch_attention(layer_index + 1)
            prefetch_auxiliary(layer_index + 1)
            prefetch_metadata(layer_index + 1)
            prefetch_router(layer_index + 1)

    def run_mixture(self, *resources: _Dynamic) -> _Dynamic:
        """Execute run mixture logic."""
        layer_index = getattr(self, "layer_index", 0)
        after_experts = self.after_experts
        (
            active_rows,
            alias_expert_scales,
            arena_router,
            attention_dp,
            batch_size,
            communication_workspace,
            eps,
            expert_pool_bytes,
            expert_views,
            latent_norms_vmem,
            layer_weights,
            metadata_semaphores,
            mixture_vmem,
            moe_routed_reduce_scatter,
            moe_token_scatter_output,
            normalized_vmem,
            owned_rows_vmem,
            pool,
            pool_layout,
            pool_view,
            residual_reduction_order,
            router_semaphores,
            router_vmem,
            routing_biases_vmem,
            sequence_rows,
            shard_routing,
            sp_owners,
            weight_stream,
        ) = (
            getattr(self.ctx, a)
            for a in (
                "active_rows",
                "alias_expert_scales",
                "arena_router",
                "attention_dp",
                "batch_size",
                "communication_workspace",
                "eps",
                "expert_pool_bytes",
                "expert_views",
                "latent_norms_vmem",
                "layer_weights",
                "metadata_semaphores",
                "mixture_vmem",
                "moe_routed_reduce_scatter",
                "moe_token_scatter_output",
                "normalized_vmem",
                "owned_rows_vmem",
                "pool",
                "pool_layout",
                "pool_view",
                "residual_reduction_order",
                "router_semaphores",
                "router_vmem",
                "routing_biases_vmem",
                "sequence_rows",
                "shard_routing",
                "sp_owners",
                "weight_stream",
            )
        )
        buffer_slot = layer_index % 2
        bias_vmem = routing_biases_vmem.at[buffer_slot]
        latent_norm_vmem = latent_norms_vmem.at[buffer_slot]
        metadata_copies = (
            tpu.make_async_copy(
                layer_weights["router_bias"].at[layer_index - 1],
                bias_vmem,
                metadata_semaphores.at[buffer_slot, 2],
            ),
            tpu.make_async_copy(
                layer_weights["latent_norm"].at[layer_index - 1],
                latent_norm_vmem,
                metadata_semaphores.at[buffer_slot, 3],
            ),
        )
        for copy in metadata_copies:
            copy.wait()

        if POOLED_WEIGHTS:
            latent_output_vmem = pool_view("latent_up")
            resources = (latent_output_vmem, *resources)
        latent_output_vmem, latent_output_semaphore, expert_semaphores = resources[:3]
        layer_router_vmem, layer_expert_vmem = router_vmem, expert_views()
        after_routing_scores = None
        if pool is not None:
            layer_router_vmem = arena_router(pool.acquire("router"))  # moe waits
            # The expert slots may overlap the router: its scores are
            # computed before the first expert copy starts.
            layer_expert_vmem = expert_views(
                pool.allocate("experts", expert_pool_bytes, replaces=("router",))
            )

            def after_routing_scores() -> _Dynamic:
                """Execute after routing scores logic."""
                pool.release("router")
                pool.acquire("experts")

        if alias_expert_scales:
            gate_up_vmem, gate_up_scales_vmem, down_vmem, down_scales_vmem = (
                layer_expert_vmem
            )
            workspace_resources = resources[3:]
        else:
            gate_up_vmem, down_vmem = layer_expert_vmem
            gate_up_scales_vmem, down_scales_vmem = resources[3:5]
            workspace_resources = resources[5:]
        mixture_output_vmem = (
            owned_rows_vmem if moe_token_scatter_output else mixture_vmem
        )
        moe(
            normalized_vmem,
            mixture_output_vmem,
            weights=MixtureOfExpertsWeights(
                layer_weights["router"].at[layer_index - 1],
                layer_router_vmem,
                router_semaphores,
                bias_vmem,
                layer_weights["latent_down"].at[layer_index - 1],
                layer_weights["latent_up"].at[layer_index - 1],
                latent_output_vmem,
                latent_output_semaphore,
                latent_norm_vmem,
                layer_weights["shared_gu"].at[layer_index - 1],
                layer_weights["shared_down"].at[layer_index - 1],
                layer_weights["expert_gu"].at[layer_index - 1],
                gate_up_vmem,
                layer_weights["expert_gus"].at[layer_index - 1],
                gate_up_scales_vmem,
                layer_weights["expert_down"].at[layer_index - 1],
                down_vmem,
                layer_weights["expert_ds"].at[layer_index - 1],
                down_scales_vmem,
                expert_semaphores,
            ),
            workspace=MixtureOfExpertsWorkspace(*workspace_resources),
            communication_workspace=communication_workspace,
            epsilon=eps,
            synchronize_devices=False,
            routing_projection_is_prefetched=True,
            group_pending_expert_routes=(
                batch_size in (6, *SCATTER_BATCHES)
                and residual_reduction_order == "relaxed"
            ),
            after_experts=after_experts,
            after_routing_scores=after_routing_scores,
            overlap_router_latent=(
                attention_dp == _CONST_2
                and batch_size == _CONST_128
                and sequence_rows == 1
            ),
            overlap_shared_reduction=(
                attention_dp == _CONST_2
                and batch_size in (64, 128)
                and sequence_rows == 1
            ),
            token_scatter_output=moe_token_scatter_output,
            routed_reduce_scatter=moe_routed_reduce_scatter,
            prepare_weights_vmem=(
                tuple(pool_view(name) for name in MOE_PREPARE_WEIGHTS)
                if POOLED_WEIGHTS
                else None
            ),
            shared_gate_up_stream=(
                weight_stream(
                    "shared_gu",
                    layer_weights["shared_gu"].at[layer_index - 1],
                    pool_view("shared_gu"),
                    offset=pool_layout["shared_gu"],
                )
                if POOLED_WEIGHTS
                else None
            ),
            token_scatter_owners=sp_owners,
            owned_hidden_ref=owned_rows_vmem if shard_routing else None,
            live_rows=active_rows,
        )

    def run_dense_mlp(self) -> _Dynamic:
        """Execute run dense mlp logic."""
        after_experts = self.after_experts
        (
            attention_dp,
            batch_size,
            communication_workspace,
            dense_refs,
            dense_semaphores,
            dense_slabs,
            hidden_size,
            mixture_vmem,
            moe_token_scatter_output,
            normalized_vmem,
            owned_rows,
            owned_rows_vmem,
            pool,
            projection_expert_pool_vmem,
            sequence_rows,
            sp_owners,
        ) = (
            getattr(self.ctx, a)
            for a in (
                "attention_dp",
                "batch_size",
                "communication_workspace",
                "dense_refs",
                "dense_semaphores",
                "dense_slabs",
                "hidden_size",
                "mixture_vmem",
                "moe_token_scatter_output",
                "normalized_vmem",
                "owned_rows",
                "owned_rows_vmem",
                "pool",
                "projection_expert_pool_vmem",
                "sequence_rows",
                "sp_owners",
            )
        )
        return _DenseMlpRunner(
            dense_refs=dense_refs,
            dense_slabs=dense_slabs,
            dense_semaphores=dense_semaphores,
            projection_expert_pool_vmem=projection_expert_pool_vmem,
            pool=pool,
            pool_alias=pool_alias,
            normalized_vmem=normalized_vmem,
            after_experts=after_experts,
            communication_workspace=communication_workspace,
            owned_rows_vmem=owned_rows_vmem,
            mixture_vmem=mixture_vmem,
            moe_token_scatter_output=moe_token_scatter_output,
            attention_dp=attention_dp,
            sp_owners=sp_owners,
            batch_size=batch_size,
            sequence_rows=sequence_rows,
            hidden_size=hidden_size,
            owned_rows=owned_rows,
            collectives32=collectives32,
        ).run()

    def ffn(self, *, dense_layer: _Dynamic = False) -> None:
        """Execute FFN layer logic."""
        capture_aux_hidden = self.capture_aux_hidden
        run_dense_mlp = self.run_dense_mlp
        run_mixture = self.run_mixture
        (aux_capture, moe_scoped_scratch, normalized_vmem) = (
            getattr(self.ctx, a)
            for a in ("aux_capture", "moe_scoped_scratch", "normalized_vmem")
        )
        if aux_capture == "mixture_input":
            capture_aux_hidden(normalized_vmem)
        if dense_layer:
            if "dense" not in DEBUG_SKIP:
                run_dense_mlp()
        else:
            with _phase("p0_moe"):
                if "moe" not in DEBUG_SKIP:
                    pl.run_scoped(run_mixture, *moe_scoped_scratch)

    def _finish_sp_rows(self, pair: _Dynamic) -> None:
        """Finish sequence parallel rows."""
        (
            communication_workspace,
            hidden_size,
            mixture_vmem,
            moe_token_scatter_output,
            owned_rows,
            owned_rows_vmem,
            owner_indices,
            owner_start,
            owner_when,
            prefix_vmem,
            sequence_parallel_pre_attention,
            sp_owners,
        ) = (
            getattr(self.ctx, a)
            for a in (
                "communication_workspace",
                "hidden_size",
                "mixture_vmem",
                "moe_token_scatter_output",
                "owned_rows",
                "owned_rows_vmem",
                "owner_indices",
                "owner_start",
                "owner_when",
                "prefix_vmem",
                "sequence_parallel_pre_attention",
                "sp_owners",
            )
        )
        rank = jax.lax.axis_index("tp")
        pair = (rank % 8) // (8 // sp_owners)
        owned_prefix_vmem = owned_rows_vmem
        scattered_mixture = owned_rows_vmem[...] if moe_token_scatter_output else None
        if owned_rows == 1:
            prefix_values = prefix_vmem[...]
            if not moe_token_scatter_output:
                mixture_values = mixture_vmem[...]
        for owner_pair in owner_indices(pair):

            @owner_when(pair, owner_pair)
            def finish_owned_rows(*, owner_pair: _Dynamic = owner_pair) -> _Dynamic:
                """Execute finish owned rows logic."""
                row_offset = owner_pair * owned_rows
                if owned_rows == 1:
                    finished_prefix = (
                        prefix_values[owner_pair : owner_pair + 1]
                        + (
                            # The MoE delivered this owner's row.
                            scattered_mixture
                            if moe_token_scatter_output
                            else mixture_values[owner_pair : owner_pair + 1]
                        )
                    ).astype(jnp.bfloat16)
                else:
                    row_slice = pl.ds(owner_start(row_offset), owned_rows)
                    finished_prefix = (
                        prefix_vmem[row_slice]
                        + (
                            scattered_mixture
                            if moe_token_scatter_output
                            else mixture_vmem[row_slice]
                        )
                    ).astype(jnp.bfloat16)
                owned_prefix_vmem[:, : hidden_size // 2] = finished_prefix[
                    :, : hidden_size // 2
                ]
                owned_prefix_vmem[:, hidden_size // 2 :] = finished_prefix[
                    :, hidden_size // 2 :
                ]

        if sequence_parallel_pre_attention:
            for owner_pair in owner_indices(pair):

                @owner_when(pair, owner_pair)
                def store_owned_prefix_rows(
                    *, owner_pair: _Dynamic = owner_pair
                ) -> _Dynamic:
                    """Execute store owned prefix rows logic."""
                    row_offset = owner_pair * owned_rows
                    row_slice = pl.ds(owner_start(row_offset), owned_rows)
                    prefix_vmem[row_slice] = owned_prefix_vmem[...]
        else:
            _gather_local_pair_rows(
                owned_prefix_vmem,
                prefix_vmem,
                communication_workspace.send_semaphores,
                communication_workspace.receive_semaphores,
                communication_workspace.shared_local_reduction_vmem,
                owners=sp_owners,
            )

    def finish_layer_rows(self, pair: _Dynamic) -> None:
        """Finish layer rows."""
        capture_aux_hidden = self.capture_aux_hidden
        (
            attention_dp,
            attention_dp_ownership,
            attention_token_scatter,
            aux_capture,
            dp_held_rows,
            dp_owned_vmem,
            group_rows_per_pair,
            mixture_vmem,
            owned_rows_vmem,
            prefix_vmem,
            sequence_parallel_pre_attention,
            sp_owners,
        ) = (
            getattr(self.ctx, a)
            for a in (
                "attention_dp",
                "attention_dp_ownership",
                "attention_token_scatter",
                "aux_capture",
                "dp_held_rows",
                "dp_owned_vmem",
                "group_rows_per_pair",
                "mixture_vmem",
                "owned_rows_vmem",
                "prefix_vmem",
                "sequence_parallel_pre_attention",
                "sp_owners",
            )
        )
        finish_scope = _phase_start("q0_finish_rows")
        if attention_dp_ownership:
            # The MoE delivered the pair's rows of every DP group (with
            # one owner, every row).
            group = jax.lax.axis_index("tp") // (32 // attention_dp)
            mixture = owned_rows_vmem[:dp_held_rows]
            for data_group in range(1, attention_dp if sp_owners > 1 else 1):
                first = data_group * group_rows_per_pair
                mixture = jnp.where(
                    group == data_group,
                    owned_rows_vmem[first : first + group_rows_per_pair],
                    mixture,
                )
            finished = (prefix_vmem[:dp_held_rows] + mixture).astype(jnp.bfloat16)
            prefix_vmem[:dp_held_rows] = finished
            if not (dp_owned_vmem is not None):
                _err_msg = "Assertion failed: dp_owned_vmem is not None"
                raise AssertionError(_err_msg)
            dp_owned_vmem[...] = finished
        elif attention_token_scatter:
            self._finish_sp_rows(pair)
        else:
            prefix_vmem[...] += mixture_vmem[...]
        _phase_end(finish_scope)
        if aux_capture == "layer":
            capture_aux_hidden(gather_owned_rows=sequence_parallel_pre_attention)

    def run_layer(
        self,
        layer_index: _Dynamic,
        layer_weights: _Dynamic,
        pair: _Dynamic,
        *,
        dense_layer: _Dynamic = False,
    ) -> None:
        """Execute full layer sequence."""
        self.layer_index = layer_index
        self.dense_layer = dense_layer
        pool = self.ctx.pool
        attention_base = pool.offset("attention") if pool is not None else 0
        self.attention_base = attention_base
        self.kda_projection_vmem, self.mla_projection_vmem = self.ctx.attention_views(
            attention_base
        )
        norm_copies = self.start_norm_copies(layer_index, layer_weights)
        self.norm_copies = norm_copies
        self.pre_norm(layer_index)
        self.pre_norm_gather()
        self.attention(layer_index)
        self.post_mix(layer_index, pair)
        self.ffn(dense_layer=dense_layer)
        self.finish_layer_rows(pair)

    def layer(
        self, layer_index: _Dynamic, *, dense_layer: _Dynamic = False
    ) -> _Dynamic:
        """Execute layer logic."""
        pair = (
            jax.lax.axis_index("tp") // 2
            if self.ctx.sequence_parallel_size == _CONST_4
            else jax.lax.axis_index("tp")
        )
        return self.run_layer(
            layer_index, self.ctx.layer_weights, pair, dense_layer=dense_layer
        )


class _BodyHelpers:
    """Helper methods for the megakernel body function."""

    def __getattr__(self, name: _Dynamic) -> _Dynamic:
        """Dynamic attribute access."""
        raise AttributeError(name)

    def __init__(self, ctx: _Dynamic) -> None:
        """Initialize body helpers with context."""
        self.ctx = ctx

    def _finish_owned_decoder_rows(self, owner_pair: _Dynamic) -> None:
        """Finish owned decoder rows."""
        final_block_count = (self.ctx.layers - 1) // 12 + 1
        (
            attention_residual_vmem,
            block_rows,
            eps,
            owned_rows,
            owned_rows_vmem,
            owner_block_row,
            owner_prefix_row,
            owner_start,
            store_mixed_rows,
        ) = (
            getattr(self.ctx, a)
            for a in (
                "attention_residual_vmem",
                "block_rows",
                "eps",
                "owned_rows",
                "owned_rows_vmem",
                "owner_block_row",
                "owner_prefix_row",
                "owner_start",
                "store_mixed_rows",
            )
        )
        row_offset = owner_pair * owned_rows
        row_slice = pl.ds(owner_start(row_offset), owned_rows)

        def mix_row(local_row: _Dynamic) -> _Dynamic:
            """Execute mix row logic."""
            global_row = row_offset + local_row
            values = jnp.concatenate(
                (
                    block_rows(
                        owner_block_row(global_row, local_row),
                        final_block_count,
                    ),
                    owner_prefix_row(global_row, row_slice, local_row),
                ),
                axis=0,
            ).astype(jnp.float32)
            scores = jnp.sum(values * attention_residual_vmem[0], axis=1)
            scores *= jax.lax.rsqrt(jnp.mean(values * values, axis=1) + eps)
            return jnp.sum(
                jax.nn.softmax(scores)[:, None] * values,
                axis=0,
                keepdims=True,
            ).astype(jnp.bfloat16)

        store_mixed_rows(mix_row, owned_rows_vmem)

    def finish_sp_outputs(self) -> None:
        """Finish sequence parallel outputs."""
        (
            attention_dp_ownership,
            attention_norm_vmem,
            attention_residual_vmem,
            block_rows,
            blocks_output_ref,
            communication_workspace,
            dp_owned_vmem,
            eps,
            final_normalization_ref,
            gather_dp_rows,
            layers,
            norm_semaphores,
            normalize,
            normalize_dp_rows,
            normalized_vmem,
            output_ref,
            output_residual_ref,
            owned_rows_vmem,
            owner_indices,
            owner_when,
            prefix_vmem,
            sp_owners,
            store_mixed_rows,
        ) = (
            getattr(self.ctx, a)
            for a in (
                "attention_dp_ownership",
                "attention_norm_vmem",
                "attention_residual_vmem",
                "block_rows",
                "blocks_output_ref",
                "communication_workspace",
                "dp_owned_vmem",
                "eps",
                "final_normalization_ref",
                "gather_dp_rows",
                "layers",
                "norm_semaphores",
                "normalize",
                "normalize_dp_rows",
                "normalized_vmem",
                "output_ref",
                "output_residual_ref",
                "owned_rows_vmem",
                "owner_indices",
                "owner_when",
                "prefix_vmem",
                "sp_owners",
                "store_mixed_rows",
            )
        )
        final_weight_copies = (
            tpu.make_async_copy(
                output_residual_ref,
                attention_residual_vmem,
                norm_semaphores.at[0],
            ),
            tpu.make_async_copy(
                final_normalization_ref,
                attention_norm_vmem,
                norm_semaphores.at[2],
            ),
        )
        for copy in final_weight_copies:
            copy.start()
        for copy in final_weight_copies:
            copy.wait()

        if attention_dp_ownership:
            final_block_count = (layers - 1) // 12 + 1

            def mix_row(row: _Dynamic) -> _Dynamic:
                """Execute mix row logic."""
                values = jnp.concatenate(
                    (
                        block_rows(row, final_block_count),
                        prefix_vmem[row : row + 1],
                    ),
                    axis=0,
                ).astype(jnp.float32)
                scores = jnp.sum(values * attention_residual_vmem[0], axis=1)
                scores *= jax.lax.rsqrt(jnp.mean(values * values, axis=1) + eps)
                return jnp.sum(
                    jax.nn.softmax(scores)[:, None] * values,
                    axis=0,
                    keepdims=True,
                ).astype(jnp.bfloat16)

            store_mixed_rows(mix_row, dp_owned_vmem)
            normalize_dp_rows(attention_norm_vmem)
            gather_dp_rows(normalized_vmem)
        else:
            rank = jax.lax.axis_index("tp")
            pair = (rank % 8) // (8 // sp_owners)
            final_block_count = (layers - 1) // 12 + 1
            for owner_pair in owner_indices(pair):

                @owner_when(pair, owner_pair)
                def finish_owned_decoder_rows(
                    *, owner_pair: _Dynamic = owner_pair
                ) -> _Dynamic:
                    """Execute finish owned decoder rows logic."""
                    self._finish_owned_decoder_rows(owner_pair)

            normalize(owned_rows_vmem, attention_norm_vmem)
            _gather_local_pair_rows(
                owned_rows_vmem,
                normalized_vmem,
                communication_workspace.send_semaphores,
                communication_workspace.receive_semaphores,
                communication_workspace.shared_local_reduction_vmem,
                owners=sp_owners,
            )

        def store_output(semaphore: _Dynamic) -> _Dynamic:
            """Execute store output logic."""
            copy = tpu.make_async_copy(normalized_vmem, output_ref, semaphore)
            copy.start()
            copy.wait()

        pl.run_scoped(store_output, tpu.SemaphoreType.DMA)
        blocks_output_ref[...] = jnp.zeros(blocks_output_ref.shape, jnp.uint8)

    def finish_dense_outputs(self) -> None:
        """Finish dense outputs."""
        (blocks_output_ref, blocks_vmem, output_ref, prefix_vmem) = (
            getattr(self.ctx, a)
            for a in ("blocks_output_ref", "blocks_vmem", "output_ref", "prefix_vmem")
        )

        def store_output(semaphore: _Dynamic) -> _Dynamic:
            """Execute store output logic."""
            copy = tpu.make_async_copy(prefix_vmem, output_ref, semaphore)
            copy.start()
            copy.wait()

        pl.run_scoped(store_output, tpu.SemaphoreType.DMA)
        blocks_output_ref[...] = blocks_vmem[...].astype(jnp.bfloat16)

    def finish_outputs(self) -> None:
        """Finish output logic."""
        if self.ctx.sequence_parallel_pre_attention:
            self.finish_sp_outputs()
        else:
            self.finish_dense_outputs()


class _DecodeStackRunner:
    """Runner for fused decoder stack execution."""

    def __getattr__(self, name: _Dynamic) -> _Dynamic:
        """Dynamic attribute access."""
        raise AttributeError(name)

    def __init__(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Initialize stack runner."""
        (
            self.prefix,
            self.embedding,
            self.weights,
            self.states,
            self.positions,
            self.batch_size,
            self.layers,
            self.eps,
            self.residual_reduction_order,
            self.mla_cache_batch_tile_size,
            self.alias_expert_scales,
            self.attention_token_scatter,
            self.moe_token_scatter_output,
            self.moe_routed_reduce_scatter,
            self.sequence_parallel_pre_attention,
            self._,
            self.sequence_rows,
            self.aux_hidden_layers,
            self.aux_capture,
            self.include_layer_zero,
            self.state_row,
            self.attention_dp,
            self.attention_projection_schedule,
            self.mla_commit_tile_size,
            self.sequence_parallel_size,
            self.shard_routing,
            self.block_tables,
            self.dcp_pool_map,
            self.dcp_owner_table,
            self.state_slots,
            self.live_rows,
        ) = self._parse_inputs(_p_args, _p_kw)
        self._setup_layout_batch_and_states()
        self._setup_layout_pools_and_tables()
        self._setup_layout_dp_and_sequence()
        self._setup_layout_aux_and_scatter()
        self._setup_layout_parallelism_rules()
        self._setup_layout_part_2a()
        self._setup_layout_part_2b()
        self._setup_pool_vars()
        self.ctx = self

    def _parse_inputs(
        self, _p_args: tuple[_Dynamic, ...], _p_kw: dict[str, _Dynamic]
    ) -> tuple[_Dynamic, ...]:
        """Parse positional and keyword inputs."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            prefix,
            embedding,
            weights,
            states,
            positions,
        ) = (
            _p_next(_p_iter, _p_get("prefix")),
            _p_next(_p_iter, _p_get("embedding")),
            _p_next(_p_iter, _p_get("weights")),
            _p_next(_p_iter, _p_get("states")),
            _p_next(_p_iter, _p_get("positions")),
        )
        (
            batch_size,
            layers,
            eps,
            residual_reduction_order,
            mla_cache_batch_tile_size,
            alias_expert_scales,
            attention_token_scatter,
            moe_token_scatter_output,
            moe_routed_reduce_scatter,
        ) = (
            _p_get("batch_size"),
            _p_get("layers", 93),
            _p_get("eps", 1e-05),
            _p_get("residual_reduction_order", "reference"),
            _p_get("mla_cache_batch_tile_size"),
            _p_kw.get("alias_expert_scales", False),
            _p_kw.get("attention_token_scatter", False),
            _p_kw.get("moe_token_scatter_output", False),
            _p_kw.get("moe_routed_reduce_scatter", False),
        )
        sequence_parallel_pre_attention = _p_kw.get(
            "sequence_parallel_pre_attention", False
        )
        (
            _,
            sequence_rows,
            aux_hidden_layers,
            aux_capture,
            include_layer_zero,
            state_row,
            attention_dp,
        ) = (
            _p_kw.get("gate_first_kda_projection", True),
            _p_get("sequence_rows", 1),
            _p_get("aux_hidden_layers", ()),
            _p_get("aux_capture", "layer"),
            _p_kw.get("include_layer_zero", False),
            _p_get("state_row"),
            _p_get("attention_dp", 1),
        )
        attention_projection_schedule = _p_get(
            "attention_projection_schedule", "default"
        )
        (
            mla_commit_tile_size,
            sequence_parallel_size,
            shard_routing,
            block_tables,
            dcp_pool_map,
            dcp_owner_table,
            state_slots,
            live_rows,
        ) = (
            _p_get("mla_commit_tile_size"),
            _p_get("sequence_parallel_size", 4),
            _p_get("shard_routing"),
            _p_get("block_tables"),
            _p_get("dcp_pool_map"),
            _p_get("dcp_owner_table"),
            _p_get("state_slots"),
            _p_get("live_rows"),
        )

        return (
            prefix,
            embedding,
            weights,
            states,
            positions,
            batch_size,
            layers,
            eps,
            residual_reduction_order,
            mla_cache_batch_tile_size,
            alias_expert_scales,
            attention_token_scatter,
            moe_token_scatter_output,
            moe_routed_reduce_scatter,
            sequence_parallel_pre_attention,
            _,
            sequence_rows,
            aux_hidden_layers,
            aux_capture,
            include_layer_zero,
            state_row,
            attention_dp,
            attention_projection_schedule,
            mla_commit_tile_size,
            sequence_parallel_size,
            shard_routing,
            block_tables,
            dcp_pool_map,
            dcp_owner_table,
            state_slots,
            live_rows,
        )

    def _setup_layout_batch_and_states(self) -> None:
        """Validate batch and states, compute select_state_row and pool_map."""
        (
            batch_size,
            block_tables,
            dcp_pool_map,
            include_layer_zero,
            state_row,
            states,
            weights,
        ) = (
            getattr(self, a)
            for a in (
                "batch_size",
                "block_tables",
                "dcp_pool_map",
                "include_layer_zero",
                "state_row",
                "states",
                "weights",
            )
        )
        if not 1 <= batch_size <= MAX_BATCH:
            _err_msg = (
                f"The fused decoder stack supports batch sizes from 1 to {MAX_BATCH}"
            )
            raise ValueError(_err_msg)
        if include_layer_zero and any(name not in weights for name in DENSE):
            _err_msg = "include_layer_zero needs the dense MLP weights"
            raise ValueError(_err_msg)
        select_state_row = state_row is not None
        if block_tables is None or dcp_pool_map is None:
            _err_msg = (
                "The stack takes block_tables and dcp_pool_map (vLLM's unified pools)"
            )
            raise ValueError(_err_msg)
        pool_map = tuple(dcp_pool_map)
        if len(dcp_pool_map) != MLA_LAYERS or len(states) != 2 + 1 + max(dcp_pool_map):
            _err_msg = (
                "The stack takes (conv, recurrent, *pools) and a pool per MLA layer"
            )
            raise ValueError(_err_msg)

        self.select_state_row = select_state_row
        self.pool_map = pool_map

    def _setup_layout_pools_and_tables(self) -> None:
        """Validate pool layout and block tables."""
        (batch_size, block_tables, states) = (
            getattr(self, a) for a in ("batch_size", "block_tables", "states")
        )
        layout = cache_layout(str(states[2].dtype))
        if any(
            pool.shape[1:] != layout.block_shape or pool.dtype != states[2].dtype
            for pool in states[2:]
        ):
            _err_msg = f"unified pools must match {layout.block_shape} {layout.dtype}"
            raise ValueError(_err_msg)
        if block_tables.shape[0] != batch_size or block_tables.ndim != _CONST_2:
            _err_msg = "block_tables must be [batch_size, pages]"
            raise ValueError(_err_msg)

        self.layout = layout

    def _setup_layout_dp_and_sequence(self) -> None:
        """Validate attention DP and sequence rows."""
        (
            attention_dp,
            batch_size,
            block_tables,
            include_layer_zero,
            select_state_row,
            sequence_rows,
        ) = (
            getattr(self, a)
            for a in (
                "attention_dp",
                "batch_size",
                "block_tables",
                "include_layer_zero",
                "select_state_row",
                "sequence_rows",
            )
        )
        if attention_dp != 1 and mla_dcp.GROUP * attention_dp != _CONST_32:
            _err_msg = (
                "Attention DP needs DCP over each DP group "
                "(KernelConfig.dcp_group = 32 / dp)"
            )
            raise ValueError(_err_msg)
        # Attention DP always runs with DP row ownership: a rank's residual stream
        # holds only its own rows of its DP group.
        attention_dp_ownership = attention_dp > 1
        paged_pages = int(block_tables.shape[1])
        if select_state_row and sequence_rows != batch_size:
            _err_msg = "state_row selection needs one sequence per tile"
            raise ValueError(_err_msg)
        first_layer = 0 if include_layer_zero else 1
        dense_names = DENSE if include_layer_zero else ()
        if not 1 <= sequence_rows <= batch_size or batch_size % sequence_rows:
            _err_msg = "sequence_rows must divide the batch"
            raise ValueError(_err_msg)
        mla_cache_batch_tile_size = self.mla_cache_batch_tile_size
        if sequence_rows > 1:
            # Rows of one sequence share one MLA cache entry, so the cache is never
            # tiled by rows; the other batch options act on rows and stay valid.
            mla_cache_batch_tile_size = batch_size

        (
            self.attention_dp_ownership,
            self.paged_pages,
            self.first_layer,
            self.dense_names,
            self.mla_cache_batch_tile_size,
        ) = (
            attention_dp_ownership,
            paged_pages,
            first_layer,
            dense_names,
            mla_cache_batch_tile_size,
        )

    def _setup_layout_aux_and_scatter(self) -> None:
        """Validate aux layers and attention token scatter."""
        (
            attention_token_scatter,
            aux_capture,
            aux_hidden_layers,
            batch_size,
            layers,
            mla_cache_batch_tile_size,
            residual_reduction_order,
        ) = (
            getattr(self, a)
            for a in (
                "attention_token_scatter",
                "aux_capture",
                "aux_hidden_layers",
                "batch_size",
                "layers",
                "mla_cache_batch_tile_size",
                "residual_reduction_order",
            )
        )
        aux_hidden_layers = tuple(aux_hidden_layers)
        if any(not 1 <= index <= layers - 1 for index in aux_hidden_layers) or len(
            set(aux_hidden_layers)
        ) != len(aux_hidden_layers):
            _err_msg = (
                "aux_hidden_layers are distinct layer ids in 1..layers-1; id "
                "L captures the residual stream entering layer L"
            )
            raise ValueError(_err_msg)
        if aux_capture not in ("layer", "attention", "mixture_input"):
            _err_msg = "aux_capture must be 'layer', 'attention' or 'mixture_input'"
            raise ValueError(_err_msg)
        if aux_capture != "layer" and attention_token_scatter:
            _err_msg = "Attention-point aux capture needs complete prefix rows"
            raise ValueError(_err_msg)
        b4_attention_gather = (
            B4_ATTENTION_GATHER
            and batch_size == _CONST_4
            and not attention_token_scatter
        )
        if residual_reduction_order not in ("reference", "row", "relaxed"):
            _err_msg = (
                "residual_reduction_order must be 'reference', 'row', or 'relaxed'"
            )
            raise ValueError(_err_msg)
        mla_cache_batch_tile_size = mla_cache_batch_tile_size or batch_size
        if not 1 <= mla_cache_batch_tile_size <= batch_size:
            _err_msg = "MLA cache batch tile size must be within the batch"
            raise ValueError(_err_msg)
        if batch_size % mla_cache_batch_tile_size:
            _err_msg = "MLA cache batch tile size must divide the batch"
            raise ValueError(_err_msg)

        (
            self.aux_hidden_layers,
            self.b4_attention_gather,
            self.mla_cache_batch_tile_size,
        ) = (
            aux_hidden_layers,
            b4_attention_gather,
            mla_cache_batch_tile_size,
        )

    def _setup_layout_parallelism_rules(self) -> None:
        """Validate parallelism constraints."""
        (
            attention_dp,
            attention_token_scatter,
            batch_size,
            moe_routed_reduce_scatter,
            moe_token_scatter_output,
            sequence_parallel_pre_attention,
            sequence_parallel_size,
        ) = (
            getattr(self, a)
            for a in (
                "attention_dp",
                "attention_token_scatter",
                "batch_size",
                "moe_routed_reduce_scatter",
                "moe_token_scatter_output",
                "sequence_parallel_pre_attention",
                "sequence_parallel_size",
            )
        )
        if (
            attention_token_scatter
            and batch_size not in (4, *SCATTER_BATCHES)
            and not (attention_dp > 1 and batch_size == _CONST_2)
        ):
            _err_msg = (
                "Attention token scatter requires batch size 4, 8 or 16, or "
                "attention DP's B2"
            )
            raise ValueError(_err_msg)
        if moe_token_scatter_output and not (
            attention_token_scatter
            and (batch_size in SCATTER_BATCHES or sequence_parallel_size < _CONST_4)
        ):
            _err_msg = "MoE token-scattered output requires B8 attention token scatter"
            raise ValueError(_err_msg)
        if moe_routed_reduce_scatter and batch_size not in (6, *SCATTER_BATCHES):
            _err_msg = "MoE routed reduce-scatter requires batch size 6, 8 or 16"
            raise ValueError(_err_msg)
        if sequence_parallel_pre_attention and not moe_token_scatter_output:
            _err_msg = "Sequence-parallel pre-attention requires B8 MoE token scatter"
            raise ValueError(_err_msg)

    def _setup_layout_part_2a(self) -> None:
        """Compute row ownership and commit rows."""
        (
            attention_dp,
            attention_token_scatter,
            batch_size,
            mla_cache_batch_tile_size,
            mla_commit_tile_size,
            moe_token_scatter_output,
            select_state_row,
            sequence_parallel_pre_attention,
            sequence_parallel_size,
            sequence_rows,
            shard_routing,
        ) = (
            getattr(self, a)
            for a in (
                "attention_dp",
                "attention_token_scatter",
                "batch_size",
                "mla_cache_batch_tile_size",
                "mla_commit_tile_size",
                "moe_token_scatter_output",
                "select_state_row",
                "sequence_parallel_pre_attention",
                "sequence_parallel_size",
                "sequence_rows",
                "shard_routing",
            )
        )
        sp_owners = sequence_parallel_size
        # Rows of each local row owner of attention DP's layout (a rank pair, or
        # at DP2 B2/B4 every rank: all rows).
        tokens_per_pair = (
            batch_size // (sp_owners if attention_dp > 1 else 4)
            if attention_token_scatter
            else 0
        )
        if sp_owners not in (1, 4, 8):
            _err_msg = (
                "sequence_parallel_size must be 1 (attention DP2 B2/B4), 4 "
                "(rank pairs) or 8 (ranks)"
            )
            raise ValueError(_err_msg)
        if sp_owners == 1 and not (attention_dp == _CONST_2 and batch_size in (2, 4)):
            _err_msg = "One row owner is attention DP2's B2/B4 layout"
            raise ValueError(_err_msg)
        if sp_owners == _CONST_8 and not (
            attention_token_scatter and batch_size % 8 == 0 and attention_dp == 1
        ):
            _err_msg = (
                "Eight row owners need the token-scatter layout, a multiple "
                "of 8 rows and no attention DP"
            )
            raise ValueError(_err_msg)
        owned_rows = batch_size // sp_owners if attention_token_scatter else 0
        shard_routing = SHARD_ROUTING if shard_routing is None else shard_routing
        # The router splits its columns over each rank pair, so the top-k shards
        # over pairs: four row owners.
        shard_routing = (
            shard_routing and moe_token_scatter_output and sp_owners == _CONST_4
        )
        if attention_dp not in (1, 2, 4):
            _err_msg = "Attention DP supports dp=1, dp=2 or dp=4"
            raise ValueError(_err_msg)
        if attention_dp > 1 and not (
            batch_size in (2, 4, 8, 16, 32, 64, 96, 128)
            and batch_size % (sp_owners * attention_dp) == 0
            and sequence_parallel_pre_attention
            and (batch_size // attention_dp) % sequence_rows == 0
            and not select_state_row
            and POOLED_WEIGHTS
        ):
            _err_msg = (
                "Attention DP needs a supported batch divisible by 4*DP, the "
                "sequence-parallel token-scatter layout, whole sequences in "
                "each group's rows and pooled weights"
            )
            raise ValueError(_err_msg)
        if attention_dp > 1 and (
            batch_size not in (32, 64, 96, 128)
            and not (attention_dp == _CONST_2 and batch_size in (2, 4, 8, 16))
        ):
            _err_msg = "Attention DP supports DP2 at B2..B128 or DP4 at B32..B128"
            raise ValueError(_err_msg)
        # The new-row hand-out's padded head count (cache_handoff_head_count)
        # follows these tile sizes.
        commit_rows = mla_commit_tile_size or mla_cache_batch_tile_size
        if batch_size % commit_rows:
            _err_msg = "mla_commit_tile_size must divide the batch"
            raise ValueError(_err_msg)

        (
            self.sp_owners,
            self.tokens_per_pair,
            self.owned_rows,
            self.shard_routing,
            self.commit_rows,
        ) = (
            sp_owners,
            tokens_per_pair,
            owned_rows,
            shard_routing,
            commit_rows,
        )

    def _setup_layout_part_2b(self) -> None:
        """Compute group rows and head counts."""
        (
            attention_dp,
            attention_dp_ownership,
            attention_projection_schedule,
            batch_size,
            commit_rows,
            mla_cache_batch_tile_size,
            prefix,
            sequence_rows,
            sp_owners,
        ) = (
            getattr(self, a)
            for a in (
                "attention_dp",
                "attention_dp_ownership",
                "attention_projection_schedule",
                "batch_size",
                "commit_rows",
                "mla_cache_batch_tile_size",
                "prefix",
                "sequence_rows",
                "sp_owners",
            )
        )
        group_rows = batch_size // attention_dp
        # Each local row owner holds tokens_per_pair rows; group d attends r of them.
        group_rows_per_pair = group_rows // sp_owners
        # The residual-stream rows a rank holds under DP row ownership: its own r,
        # or with one owner every row.
        dp_held_rows = batch_size if sp_owners == 1 else group_rows_per_pair
        sp8_residual = (
            attention_dp_ownership
            and attention_dp == _CONST_2
            and batch_size == _CONST_128
            and sequence_rows == 1
        )
        # Plain decode above two rows per rank pair: its two ranks split the
        # MoE-input gather (DP2 B128's eight residual owners gather their own).
        pair_split_gather = (
            attention_dp_ownership
            and not sp8_residual
            and sp_owners != 1
            and dp_held_rows > _CONST_2
            and sequence_rows == 1
        )
        default_streams = (
            {
                **DCP_B32_WEIGHT_STREAMS,
                **(DCP_B32_DECODE_RINGS if sequence_rows == 1 else {}),
            }
            if (
                batch_size in (32, 64)
                and attention_dp == 1
                and not _config().weight_streams
            )
            else {}
        )
        projection_streams = _attention_projection_streams(
            attention_dp,
            attention_projection_schedule,
        )
        conflicts = projection_streams.keys() & WEIGHT_STREAMS.keys()
        if conflicts:
            _err_msg = (
                f"Projection schedule conflicts with "
                f"KernelConfig.weight_streams: {sorted(conflicts)}"
            )
            raise ValueError(_err_msg)
        # DP2 B2/B4 (one row owner): each rank projects only its own group's held
        # shard of the KDA input projection, for every row (``swap_kda_projections``).
        kda_shard_swap = attention_dp_ownership and sp_owners == 1
        stream_config = {
            **DP_WEIGHT_STREAMS.get(attention_dp, {}),
            **(
                _dp_dcp_weight_streams(attention_dp, group_rows)
                if attention_dp > 1
                else {}
            ),
            **({"k_projection": (attention_dp, 1)} if kda_shard_swap else {}),
            **default_streams,
            **WEIGHT_STREAMS,
            **projection_streams,
        }
        cache_handoff_head_count = max(
            3, 8 // math.gcd(mla_cache_batch_tile_size, commit_rows, 8)
        )
        hidden_size = prefix.shape[-1]

        (
            self.group_rows,
            self.group_rows_per_pair,
            self.dp_held_rows,
            self.sp8_residual,
            self.pair_split_gather,
            self.default_streams,
            self.projection_streams,
            self.kda_shard_swap,
            self.stream_config,
            self.cache_handoff_head_count,
            self.hidden_size,
        ) = (
            group_rows,
            group_rows_per_pair,
            dp_held_rows,
            sp8_residual,
            pair_split_gather,
            default_streams,
            projection_streams,
            kda_shard_swap,
            stream_config,
            cache_handoff_head_count,
            hidden_size,
        )

    def _setup_pool_vars(self) -> None:
        """Compute pool weights and scoped scratch buffers."""
        (
            alias_expert_scales,
            attention_dp,
            attention_dp_ownership,
            attention_token_scatter,
            batch_size,
            dp_held_rows,
            moe_token_scatter_output,
            owned_rows,
            residual_reduction_order,
            select_state_row,
            sequence_parallel_pre_attention,
            sequence_rows,
            shard_routing,
            sp_owners,
            state_slots,
            states,
            stream_config,
            weights,
        ) = (
            getattr(self, a)
            for a in (
                "alias_expert_scales",
                "attention_dp",
                "attention_dp_ownership",
                "attention_token_scatter",
                "batch_size",
                "dp_held_rows",
                "moe_token_scatter_output",
                "owned_rows",
                "residual_reduction_order",
                "select_state_row",
                "sequence_parallel_pre_attention",
                "sequence_rows",
                "shard_routing",
                "sp_owners",
                "state_slots",
                "states",
                "stream_config",
                "weights",
            )
        )
        # DP row ownership keeps its rows' snapshots block-major at every batch.
        blocks_block_major = batch_size > _CONST_8 or attention_dp_ownership
        # Above B16 each row owner keeps only its own rows' snapshots.
        blocks_owned = batch_size > _CONST_16
        # Above B16 with token-scattered attention and MoE outputs, the
        # whole-batch attention and mixture outputs are never written.
        compact_outputs = (
            batch_size > _CONST_16
            and attention_token_scatter
            and moe_token_scatter_output
        )
        # DP row ownership: a rank's residual stream (prefix and snapshots) is
        # only its own group_rows_per_pair rows, held from row 0.
        block_rows_held = (
            dp_held_rows
            if attention_dp_ownership
            else owned_rows
            if blocks_owned
            else batch_size
        )
        # One copy of each per-owner row region with a traced owner index (program
        # size O(1) in the owner count). BF16 row slices at a traced offset must be
        # provably 16-aligned, so owners need a multiple of 16 rows (B64 at sp=4,
        # B128 at sp=8); smaller owners use unrolled copies.
        owner_generic = blocks_owned and owned_rows % 16 == 0
        # The trace-time arena manager (``VmemPool``) places the router, attention
        # input projections, routed-expert slots and layer 0's dense slabs; under
        # DP the router reuses the drained stream slots.
        pool_managed = POOLED_WEIGHTS
        if blocks_block_major and not sequence_parallel_pre_attention:
            _err_msg = (
                "Above eight rows the decoder needs sequence-parallel pre-attention"
            )
            raise ValueError(_err_msg)
        persist_kda_state = batch_size <= _CONST_4 and attention_dp == 1
        # Above B16 the KDA state streams through row-chunk buffers.
        stream_kda_state = batch_size > _CONST_16
        kda_row_chunk = _kda_row_chunk(batch_size, attention_dp)
        slot_reads = state_slots is not None
        if slot_reads and not (
            (attention_dp == 1 and (select_state_row or not persist_kda_state))
            or (
                attention_dp == _CONST_2
                and batch_size in (2, 4, 8, 16, 32, 64, 96, 128)
            )
            or (attention_dp == _CONST_4 and batch_size in (32, 64, 96, 128))
        ):
            _err_msg = (
                "state_slots needs DP1 with state_row or batch > 4, DP2 at "
                "B2..B128 or DP4 at B32..B128"
            )
            raise ValueError(_err_msg)
        overlap_kda_state_reads = (
            batch_size in (6, *SCATTER_BATCHES) or attention_dp > 1
        )
        overlap_kda_state_writes = (
            batch_size in (6, *SCATTER_BATCHES) or attention_dp > 1
        )

        def shape(value: _Dynamic) -> _Dynamic:
            """Execute shape logic."""
            return jax.ShapeDtypeStruct(value.shape, value.dtype)

        local_weights = {
            name: jax.ShapeDtypeStruct(value.shape[1:], value.dtype)
            for name, value in weights.items()
        }

        def pooled_shape(name: _Dynamic) -> _Dynamic:
            """Pool footprint of a layer weight: a ring stream's slot region.

            (depth x its largest chunk along the streamed axis), else the weight.
            """
            shape = local_weights[name].shape
            if name not in STREAMED_WEIGHTS or name not in stream_config:
                return shape
            chunks, depth = stream_config[name]
            if chunks == depth:
                return shape
            axis = 1 if STREAMED_WEIGHTS[name][0] == "columns" else 0
            largest = _chunk_sizes(shape[axis], chunks, 128 if axis == 1 else 16)[0]
            return tuple(
                depth * largest if index == axis else size
                for index, size in enumerate(shape)
            )

        pool_weights = {
            name: jax.ShapeDtypeStruct(pooled_shape(name), value.dtype)
            for name, value in local_weights.items()
        }
        mla_scoped_scratch = _mla_scoped_scratch(
            local_weights,
            # Attention DP runs MLA on the group's rows, one shard at a time.
            batch_size=batch_size // attention_dp,
            sequence_rows=sequence_rows,
            dcp_dtype=states[2].dtype,
            dcp_page_tokens=states[2].shape[-3] * states[2].shape[-2],
            dcp_heads=3 * attention_dp,
        )
        moe_scoped_scratch = _moe_scoped_scratch(
            local_weights,
            batch_size=batch_size,
            alias_expert_scales=alias_expert_scales,
            token_scatter_output=moe_token_scatter_output,
            token_scatter_owners=sp_owners,
            shard_routing=shard_routing,
            group_pending_routes=(
                batch_size in (6, *SCATTER_BATCHES)
                and residual_reduction_order == "relaxed"
            ),
        )

        (
            self.blocks_block_major,
            self.blocks_owned,
            self.compact_outputs,
            self.block_rows_held,
            self.owner_generic,
            self.pool_managed,
            self.persist_kda_state,
            self.stream_kda_state,
            self.kda_row_chunk,
            self.slot_reads,
            self.overlap_kda_state_reads,
            self.overlap_kda_state_writes,
            self.local_weights,
            self.pool_weights,
            self.mla_scoped_scratch,
            self.moe_scoped_scratch,
        ) = (
            blocks_block_major,
            blocks_owned,
            compact_outputs,
            block_rows_held,
            owner_generic,
            pool_managed,
            persist_kda_state,
            stream_kda_state,
            kda_row_chunk,
            slot_reads,
            overlap_kda_state_reads,
            overlap_kda_state_writes,
            local_weights,
            pool_weights,
            mla_scoped_scratch,
            moe_scoped_scratch,
        )

    def _setup_body_refs(self, refs: _Dynamic) -> None:
        """Unpack kernel input references and layer weights."""
        (batch_size, dense_names, live_rows, pool_map, slot_reads, states) = (
            getattr(self, a)
            for a in (
                "batch_size",
                "dense_names",
                "live_rows",
                "pool_map",
                "slot_reads",
                "states",
            )
        )
        remaining_refs = list(refs)
        table_ref = remaining_refs.pop(0)
        slots_ref = None
        if slot_reads:
            slots_ref = remaining_refs.pop(0)
        active_rows = None
        if live_rows is not None:
            live_ref = remaining_refs.pop(0)
            # Stack scalar integers before comparison: Mosaic cannot concatenate
            # scalar predicates into a vector through bitcast_vreg.
            active_rows = jnp.stack([live_ref[b] for b in range(batch_size)]) != 0
        prefix_ref = remaining_refs.pop(0)
        embedding_ref = remaining_refs.pop(0)
        refs = tuple(remaining_refs)
        layer_weights = dict(zip(NAMES, refs[: len(NAMES)], strict=True))
        output_residual_ref, final_normalization_ref = refs[
            len(NAMES) : len(NAMES) + len(EPILOGUE)
        ]
        dense_start = len(NAMES) + len(EPILOGUE)
        dense_refs = dict(
            zip(
                dense_names,
                refs[dense_start : dense_start + len(dense_names)],
                strict=True,
            )
        )
        state_start = dense_start + len(dense_names)
        state_count = len(states)
        convolution_ref, recurrent_ref = refs[state_start : state_start + 2]
        raw_pool_refs = refs[state_start + 2 : state_start + state_count]
        pool_ref = tuple(raw_pool_refs[index] for index in pool_map)
        output_refs = refs[state_start + state_count : state_start + state_count + 7]
        (
            output_ref,
            blocks_output_ref,
            convolution_output_ref,
            recurrent_output_ref,
            key_output_ref,
            value_output_ref,
            aux_output_ref,
        ) = output_refs

        (
            self.table_ref,
            self.slots_ref,
            self.active_rows,
            self.prefix_ref,
            self.embedding_ref,
            self.layer_weights,
            self.output_residual_ref,
            self.final_normalization_ref,
            self.dense_refs,
            self.convolution_ref,
            self.recurrent_ref,
            self.pool_ref,
            self.output_ref,
            self.blocks_output_ref,
            self.convolution_output_ref,
            self.recurrent_output_ref,
            self.key_output_ref,
            self.value_output_ref,
            self.aux_output_ref,
            self.body_common_refs,
        ) = (
            table_ref,
            slots_ref,
            active_rows,
            prefix_ref,
            embedding_ref,
            layer_weights,
            output_residual_ref,
            final_normalization_ref,
            dense_refs,
            convolution_ref,
            recurrent_ref,
            pool_ref,
            output_ref,
            blocks_output_ref,
            convolution_output_ref,
            recurrent_output_ref,
            key_output_ref,
            value_output_ref,
            aux_output_ref,
            refs[state_start + state_count + 7 :],
        )

    def _setup_body_vmem(self) -> None:
        """Unpack references and initialize scratch VMEM and workspace."""
        common = self.body_common_refs
        (
            attention_dp,
            b4_attention_gather,
            compact_outputs,
            moe_routed_reduce_scatter,
            pair_split_gather,
            select_state_row,
            state_row_ref,
        ) = (
            getattr(self, a)
            for a in (
                "attention_dp",
                "b4_attention_gather",
                "compact_outputs",
                "moe_routed_reduce_scatter",
                "pair_split_gather",
                "select_state_row",
                "state_row_ref",
            )
        )
        (
            prefix_vmem,
            blocks_vmem,
            normalized_vmem,
            attention_vmem,
            mixture_vmem,
            attention_residual_vmem,
            mlp_residual_vmem,
            attention_norm_vmem,
            ffn_norm_vmem,
            router_vmem,
            router_semaphores,
            projection_expert_pool_vmem,
            pool_semaphores,
            norm_semaphores,
            convolution_filter_vmem,
            exp_log_decay_rate_vmem,
            time_step_bias_vmem,
            output_normalization_vmem,
            convolution_history_vmem,
            recurrent_state_vmem,
            auxiliary_semaphores,
            query_norms_vmem,
            key_value_norms_vmem,
            routing_biases_vmem,
            latent_norms_vmem,
            metadata_semaphores,
            owned_rows_vmem,
            aux_semaphore,
            dense_semaphores,
            stream_semaphores,
            *common,
        ) = common
        group_positions_smem = None
        if attention_dp > 1:
            (
                _group_input_vmem,
                _dp_attended_vmem,
                group_positions_smem,
                _dp_query_latent_vmem,
                _dp_compressed_vmem,
            ) = common[:5]
            common = common[5:]
        self.dcp_persistent = common[: len(mla_dcp.PERSISTENT_NAMES)]
        common = common[len(mla_dcp.PERSISTENT_NAMES) :]
        if _config().vmem_pad_kb:
            headroom_pad = common[-1]
            headroom_pad[...] = jnp.zeros(headroom_pad.shape, jnp.float32)
            common = common[:-1]
        else:
            headroom_pad = None
        gather_wire = None
        if b4_attention_gather:
            *common, gather_wire = common
        pair_semaphores = None
        if pair_split_gather:
            *common, pair_send, pair_receive = common
            pair_semaphores = (pair_send.at[0], pair_receive.at[0])
        routed_consumed = None
        if moe_routed_reduce_scatter:
            *common, routed_consumed = common
        communication_workspace = AttentionCommunicationWorkspace(
            *common,
            attention_gather_vmem=gather_wire,
            routed_consumed=routed_consumed,
            pair_semaphores=pair_semaphores,
        )
        if compact_outputs:
            # Attention's whole-batch output argument is shape-checked but
            # unused with token scatter; the normalized input has its shape.
            attention_vmem = normalized_vmem

        # In sequence-row mode the chained recurrence reads only the first
        # row's input state; with a state row the accepted row of the
        # previous step is read directly into that slot.
        selected_state_row = state_row_ref[0] if select_state_row else None

        (
            self.prefix_vmem,
            self.blocks_vmem,
            self.normalized_vmem,
            self.attention_vmem,
            self.mixture_vmem,
            self.attention_residual_vmem,
            self.mlp_residual_vmem,
            self.attention_norm_vmem,
            self.ffn_norm_vmem,
            self.router_vmem,
            self.router_semaphores,
            self.projection_expert_pool_vmem,
            self.pool_semaphores,
            self.norm_semaphores,
            self.convolution_filter_vmem,
            self.exp_log_decay_rate_vmem,
            self.time_step_bias_vmem,
            self.output_normalization_vmem,
            self.convolution_history_vmem,
            self.recurrent_state_vmem,
            self.auxiliary_semaphores,
            self.query_norms_vmem,
            self.key_value_norms_vmem,
            self.routing_biases_vmem,
            self.latent_norms_vmem,
            self.metadata_semaphores,
            self.owned_rows_vmem,
            self.aux_semaphore,
            self.dense_semaphores,
            self.stream_semaphores,
            self.group_positions_smem,
            self.headroom_pad,
            self.communication_workspace,
            self.selected_state_row,
        ) = (
            prefix_vmem,
            blocks_vmem,
            normalized_vmem,
            attention_vmem,
            mixture_vmem,
            attention_residual_vmem,
            mlp_residual_vmem,
            attention_norm_vmem,
            ffn_norm_vmem,
            router_vmem,
            router_semaphores,
            projection_expert_pool_vmem,
            pool_semaphores,
            norm_semaphores,
            convolution_filter_vmem,
            exp_log_decay_rate_vmem,
            time_step_bias_vmem,
            output_normalization_vmem,
            convolution_history_vmem,
            recurrent_state_vmem,
            auxiliary_semaphores,
            query_norms_vmem,
            key_value_norms_vmem,
            routing_biases_vmem,
            latent_norms_vmem,
            metadata_semaphores,
            owned_rows_vmem,
            aux_semaphore,
            dense_semaphores,
            stream_semaphores,
            group_positions_smem,
            headroom_pad,
            communication_workspace,
            selected_state_row,
        )

    def kda_state_source(self, ref: _Dynamic, state_index: _Dynamic) -> _Dynamic:
        """Execute kda state source logic."""
        select_state_row, selected_state_row, slot_reads, slots_ref = (
            getattr(self, a)
            for a in (
                "select_state_row",
                "selected_state_row",
                "slot_reads",
                "slots_ref",
            )
        )
        if select_state_row:
            row = (
                slots_ref[selected_state_row]
                if slot_reads and slots_ref is not None
                else selected_state_row
            )
            return ref.at[pl.ds(row, 1), state_index]
        return ref.at[:, state_index]

    def kda_state_reads(
        self, ref: _Dynamic, vmem: _Dynamic, semaphore: _Dynamic, state_index: _Dynamic
    ) -> _Dynamic:
        """The copies of a layer's input KDA states into ``vmem``: one,.

        or one per row from its slot (``state_slots``).
        """
        (
            batch_size,
            kda_state_destination,
            kda_state_source,
            select_state_row,
            slot_reads,
            slots_ref,
        ) = (
            getattr(self, a)
            for a in (
                "batch_size",
                "kda_state_destination",
                "kda_state_source",
                "select_state_row",
                "slot_reads",
                "slots_ref",
            )
        )
        if not slot_reads or select_state_row:
            return (
                tpu.make_async_copy(
                    kda_state_source(ref, state_index),
                    kda_state_destination(vmem),
                    semaphore,
                ),
            )
        return tuple(
            tpu.make_async_copy(
                ref.at[pl.ds(slots_ref[b], 1), state_index],
                vmem.at[pl.ds(b, 1)],
                semaphore,
            )
            for b in range(batch_size)
        )

    def kda_state_destination(self, vmem: _Dynamic) -> _Dynamic:
        """Execute kda state destination logic."""
        select_state_row = self.select_state_row
        return vmem.at[pl.ds(0, 1)] if select_state_row else vmem

    def pool_view(self, name: _Dynamic, base: _Dynamic = 0) -> _Dynamic:
        """The pooled view of one layer weight at its _pool_layout offset.

        (from ``base``: the MLA output projection is placed relative to its
        layer's attention weights).
        """
        pool_layout, pool_weights, projection_expert_pool_vmem = (
            getattr(self, a)
            for a in ("pool_layout", "pool_weights", "projection_expert_pool_vmem")
        )
        value = pool_weights[name]
        size = math.prod(value.shape) * value.dtype.itemsize
        return pool_alias.view(
            projection_expert_pool_vmem.at[
                pl.ds((base + pool_layout[name]) // 128, size // 128), :
            ],
            value.shape,
            value.dtype,
        )

    def pool_views(
        self, names: _Dynamic, *, experts: _Dynamic = False, base: _Dynamic = 0
    ) -> _Dynamic:
        """Execute pool views logic."""
        pool_weights, projection_expert_pool_vmem = (
            getattr(self, a) for a in ("pool_weights", "projection_expert_pool_vmem")
        )
        views = []
        offset = base
        for name in names:
            value = pool_weights[name]
            logical_shape = value.shape[1:] if experts else value.shape
            view_shape = (4, *logical_shape) if experts else logical_shape
            size = math.prod(view_shape) * value.dtype.itemsize
            storage_size = pool_alias.storage_size(view_shape, value.dtype)
            views.append(
                pool_alias.view(
                    projection_expert_pool_vmem.at[
                        pl.ds(offset // 128, size // 128), :
                    ],
                    view_shape,
                    value.dtype,
                )
            )
            offset += storage_size
        return views

    def attention_views(self, base: _Dynamic = 0) -> _Dynamic:
        """The KDA and MLA input projections of one layer from ``base``."""
        pool_views = self.pool_views
        return (pool_views(KDA[:4], base=base), pool_views(MLA[:2], base=base))

    def expert_views(self, base: _Dynamic = 0) -> _Dynamic:
        """Execute expert views logic."""
        alias_expert_scales, pool_views = (
            getattr(self, a) for a in ("alias_expert_scales", "pool_views")
        )
        if alias_expert_scales:
            views = pool_views(
                ("expert_gu", "expert_down", "expert_gus", "expert_ds"),
                experts=True,
                base=base,
            )
            return (views[0], views[2], views[1], views[3])
        return pool_views(("expert_gu", "expert_down"), experts=True, base=base)

    def arena_router(self, offset: _Dynamic) -> _Dynamic:
        """Execute arena router logic."""
        hidden_size, pool = (getattr(self, a) for a in ("hidden_size", "pool"))
        if not pool is not None:
            _err_msg = "Assertion failed: pool is not None"
            raise AssertionError(_err_msg)
        return pool.view(offset, (1, hidden_size, 512), jnp.dtype(jnp.bfloat16))

    def request_router(self, index: _Dynamic, predicate: _Dynamic = None) -> _Dynamic:
        """Queue layer ``index``'s router copy into the arena."""
        arena_router, hidden_size, layer_weights, pool, router_semaphores = (
            getattr(self, a)
            for a in (
                "arena_router",
                "hidden_size",
                "layer_weights",
                "pool",
                "router_semaphores",
            )
        )
        if not pool is not None:
            _err_msg = "Assertion failed: pool is not None"
            raise AssertionError(_err_msg)
        pool.request(
            "router",
            pool_alias.storage_size((1, hidden_size, 512), jnp.dtype(jnp.bfloat16)),
            lambda offset, index: _router_copy(
                layer_weights["router"].at[index - 1],
                arena_router(offset),
                router_semaphores,
                bool(0),
                initialize_padding=False,
            ),
            index=index,
            predicate=predicate,
        )

    def request_attention(
        self, index: _Dynamic, predicate: _Dynamic = None
    ) -> _Dynamic:
        """Queue layer ``index``'s attention input projections."""
        pool, pool_layout, prefetch_attention = (
            getattr(self, a) for a in ("pool", "pool_layout", "prefetch_attention")
        )
        if not pool is not None:
            _err_msg = "Assertion failed: pool is not None"
            raise AssertionError(_err_msg)
        pool.request(
            "attention",
            pool_layout["attention_bytes"],
            lambda offset, index: prefetch_attention(index, base=offset),
            index=index,
            predicate=predicate,
        )

    def for_dp_owned_rows(self, run: _Dynamic) -> _Dynamic:
        """``run(first)`` with this rank's first batch row (static per.

        row owner and DP group).
        """
        attention_dp, group_rows_per_pair, sp_owners, tokens_per_pair = (
            getattr(self, a)
            for a in (
                "attention_dp",
                "group_rows_per_pair",
                "sp_owners",
                "tokens_per_pair",
            )
        )
        if sp_owners == 1:
            run(0)
            return
        rank = jax.lax.axis_index("tp")
        pair, group = (rank % 8 // (8 // sp_owners), rank // (32 // attention_dp))
        for owner in range(sp_owners):
            for data_group in range(attention_dp):

                @pl.when((pair == owner) & (group == data_group))
                def owned_region(
                    owner: _Dynamic = owner, data_group: _Dynamic = data_group
                ) -> _Dynamic:
                    """Execute owned region logic."""
                    run(owner * tokens_per_pair + data_group * group_rows_per_pair)

    def load_prefix(self, staged: _Dynamic, semaphore: _Dynamic) -> _Dynamic:
        """Execute load prefix logic."""
        (
            attention_dp_ownership,
            dp_held_rows,
            for_dp_owned_rows,
            prefix_ref,
            prefix_vmem,
        ) = (
            getattr(self, a)
            for a in (
                "attention_dp_ownership",
                "dp_held_rows",
                "for_dp_owned_rows",
                "prefix_ref",
                "prefix_vmem",
            )
        )
        copy = tpu.make_async_copy(prefix_ref, staged, semaphore)
        copy.start()
        copy.wait()
        if attention_dp_ownership:

            def take_owned_rows(first: _Dynamic) -> _Dynamic:
                """Execute take owned rows logic."""
                prefix_vmem[:dp_held_rows] = staged[first : first + dp_held_rows]

            for_dp_owned_rows(take_owned_rows)

    def initialize_blocks(self, embedding_ref: _Dynamic) -> _Dynamic:
        """Execute initialize blocks logic."""
        (
            attention_dp_ownership,
            batch_size,
            blocks_block_major,
            blocks_owned,
            blocks_vmem,
            dp_held_rows,
            for_dp_owned_rows,
            owned_rows,
            sp_owners,
        ) = (
            getattr(self, a)
            for a in (
                "attention_dp_ownership",
                "batch_size",
                "blocks_block_major",
                "blocks_owned",
                "blocks_vmem",
                "dp_held_rows",
                "for_dp_owned_rows",
                "owned_rows",
                "sp_owners",
            )
        )
        if attention_dp_ownership:

            def initialize_owned_blocks(first: _Dynamic) -> _Dynamic:
                """Execute initialize owned blocks logic."""
                for local in range(dp_held_rows):
                    blocks_vmem[local // 2, pl.ds(local % 2 * 8, 1)] = embedding_ref[
                        pl.ds(first + local, 1)
                    ].astype(jnp.bfloat16)

            for_dp_owned_rows(initialize_owned_blocks)
        elif blocks_owned:
            owner_of_rank = jax.lax.axis_index("tp") % 8 // (8 // sp_owners)
            for owner_index in range(sp_owners):

                @pl.when(owner_of_rank == owner_index)
                def initialize_owned_blocks(
                    owner_index: _Dynamic = owner_index,
                ) -> _Dynamic:
                    """Execute initialize owned blocks logic."""
                    for local in range(owned_rows):
                        blocks_vmem[local // 2, pl.ds(local % 2 * 8, 1)] = (
                            embedding_ref[
                                pl.ds(owner_index * owned_rows + local, 1)
                            ].astype(jnp.bfloat16)
                        )
        elif blocks_block_major:
            for row in range(batch_size):
                blocks_vmem[row // 2, pl.ds(row % 2 * 8, 1)] = embedding_ref[
                    pl.ds(row, 1)
                ].astype(jnp.bfloat16)
        else:
            blocks_vmem[:, 0] = embedding_ref[...].astype(jnp.float32)

    def staged_blocks(self, embedding_vmem: _Dynamic, semaphore: _Dynamic) -> _Dynamic:
        """Execute staged blocks logic."""
        embedding_ref, initialize_blocks = (
            getattr(self, a) for a in ("embedding_ref", "initialize_blocks")
        )
        copy = tpu.make_async_copy(embedding_ref, embedding_vmem, semaphore)
        copy.start()
        copy.wait()
        initialize_blocks(embedding_vmem)

    def owner_indices(self, pair: _Dynamic) -> _Dynamic:
        """Execute owner indices logic."""
        owner_generic, sp_owners = (
            getattr(self, a) for a in ("owner_generic", "sp_owners")
        )
        return (pair,) if owner_generic else range(sp_owners)

    def owner_when(self, pair: _Dynamic, owner: _Dynamic) -> _Dynamic:
        """Execute owner when logic."""
        owner_generic = self.owner_generic
        if owner_generic:
            return lambda function: function()
        return pl.when(pair == owner)

    def owner_start(self, start: _Dynamic) -> _Dynamic:
        """Execute owner start logic."""
        owner_generic = self.owner_generic
        return pl.multiple_of(start, 16) if owner_generic else start

    def owner_block_row(self, global_row: _Dynamic, local_row: _Dynamic) -> _Dynamic:
        """Execute owner block row logic."""
        owner_generic = self.owner_generic
        return local_row if owner_generic else global_row

    def owner_prefix_row(
        self, global_row: _Dynamic, row_slice: _Dynamic, local_row: _Dynamic
    ) -> _Dynamic:
        """Execute owner prefix row logic."""
        owner_generic, prefix_vmem = (
            getattr(self, a) for a in ("owner_generic", "prefix_vmem")
        )
        if owner_generic:
            return prefix_vmem[row_slice][local_row : local_row + 1]
        return prefix_vmem[pl.ds(global_row, 1)]

    def block_rows(self, row: _Dynamic, count: _Dynamic = 8) -> _Dynamic:
        """The residual snapshots of one batch row, ``[count, hidden]``."""
        block_rows_held, blocks_block_major, blocks_vmem = (
            getattr(self, a)
            for a in ("block_rows_held", "blocks_block_major", "blocks_vmem")
        )
        if blocks_block_major:
            row = row % block_rows_held
            return blocks_vmem[row // 2, pl.ds(row % 2 * 8, count)]
        return blocks_vmem[row] if count == _CONST_8 else blocks_vmem[row, :count]

    def residual(
        self, destination: _Dynamic, folded: _Dynamic, active_block_count: _Dynamic
    ) -> _Dynamic:
        """Execute residual logic."""
        batch_size, blocks_vmem, eps, prefix_vmem, residual_reduction_order = (
            getattr(self, a)
            for a in (
                "batch_size",
                "blocks_vmem",
                "eps",
                "prefix_vmem",
                "residual_reduction_order",
            )
        )
        _residual_mixture_by_row(
            blocks_vmem[...],
            prefix_vmem[...],
            destination,
            folded,
            active_block_count,
            batch_size=batch_size,
            epsilon=eps,
            relaxed_reduction_order=residual_reduction_order == "relaxed",
        )

    def store_mixed_rows(
        self, compute_row: _Dynamic, destination: _Dynamic
    ) -> _Dynamic:
        """Write the destination's owned rows. Above two rows per pair.

        (B16) each row runs in its own region: unrolled rows would keep
        every row's 8 x hidden FP32 blocks live and spill. DP2 B4's four
        rows (one owner) stay unrolled.
        """
        communication_workspace, sp_owners = (
            getattr(self, a) for a in ("communication_workspace", "sp_owners")
        )
        rows = destination.shape[0]
        if rows <= _CONST_2 or sp_owners == 1:
            destination[...] = jnp.concatenate(
                [compute_row(local_row) for local_row in range(rows)], axis=0
            )
            return
        for local_row in range(rows):

            @pl.when(communication_workspace.reduction_phase[0] >= 0)
            def store_row(local_row: _Dynamic = local_row) -> _Dynamic:
                """Execute store row logic."""
                destination[pl.ds(local_row, 1)] = compute_row(local_row)

    def normalize(self, destination: _Dynamic, scale: _Dynamic) -> _Dynamic:
        """Execute normalize logic."""
        eps = self.eps
        value = destination[...].astype(jnp.float32)
        destination[...] = (
            value
            * jax.lax.rsqrt(jnp.mean(value * value, axis=1, keepdims=True) + eps)
            * scale[...]
        ).astype(jnp.bfloat16)

    def sp8_owned_value(self) -> _Dynamic:
        """Execute sp8 owned value logic."""
        sp8_rows_vmem = self.sp8_rows_vmem
        if not sp8_rows_vmem is not None:
            _err_msg = "Assertion failed: sp8_rows_vmem is not None"
            raise AssertionError(_err_msg)
        value = sp8_rows_vmem[...]
        return jnp.where(jax.lax.axis_index("tp") % 2 == 0, value[:8], value[8:16])

    def _mix_sp8_residual(self, mix_row: _Dynamic) -> None:
        """Execute sp8 residual row mixing."""
        communication_workspace, hidden_size, sp8_rows_vmem = (
            getattr(self, a)
            for a in ("communication_workspace", "hidden_size", "sp8_rows_vmem")
        )
        if not sp8_rows_vmem is not None:
            _err_msg = "Assertion failed: sp8_rows_vmem is not None"
            raise AssertionError(_err_msg)
        sp8_rows_vmem[...] = jnp.zeros((16, hidden_size), jnp.bfloat16)
        for lane in range(2):

            @pl.when(jax.lax.axis_index("tp") % 2 == lane)
            def mix_half(lane: _Dynamic = lane) -> _Dynamic:
                """Execute mix half logic."""
                for local_row in range(8):

                    @pl.when(communication_workspace.reduction_phase[0] >= 0)
                    def store_row(local_row: _Dynamic = local_row) -> _Dynamic:
                        """Execute store row logic."""
                        row = lane * 8 + local_row
                        if not sp8_rows_vmem is not None:
                            _err_msg = "Assertion failed: sp8_rows_vmem is not None"
                            raise AssertionError(_err_msg)
                        sp8_rows_vmem[pl.ds(row, 1)] = mix_row(row)

    def _snapshot_blocks(self, layer_index: _Dynamic) -> None:
        """Execute snapshot of blocks."""
        blocks_vmem, dp_held_rows, prefix_vmem = (
            getattr(self, a) for a in ("blocks_vmem", "dp_held_rows", "prefix_vmem")
        )
        for row in range(dp_held_rows):
            half = blocks_vmem.at[row // 2, pl.ds(row % 2 * 8, 8)]
            mask = jax.lax.broadcasted_iota(jnp.int32, half.shape, 0)
            half[...] = jnp.where(
                mask == layer_index // 12, prefix_vmem[row : row + 1], half[...]
            )

    def mix_dp_rows(
        self,
        folded: _Dynamic,
        active_blocks: _Dynamic,
        *,
        layer_index: _Dynamic,
        attention_output: _Dynamic = False,
        start_block: _Dynamic = False,
    ) -> _Dynamic:
        """Execute mix dp rows logic."""
        (
            _mix_sp8_residual,
            _snapshot_blocks,
            block_rows,
            dp_held_rows,
            dp_owned_vmem,
            eps,
            kda_shard_swap,
            normalized_vmem,
            prefix_vmem,
            residual_reduction_order,
            sp8_residual,
            store_mixed_rows,
        ) = (
            getattr(self, a)
            for a in (
                "_mix_sp8_residual",
                "_snapshot_blocks",
                "block_rows",
                "dp_held_rows",
                "dp_owned_vmem",
                "eps",
                "kda_shard_swap",
                "normalized_vmem",
                "prefix_vmem",
                "residual_reduction_order",
                "sp8_residual",
                "store_mixed_rows",
            )
        )
        rows = slice(0, dp_held_rows)
        if attention_output:
            if not dp_owned_vmem is not None:
                _err_msg = "Assertion failed: dp_owned_vmem is not None"
                raise AssertionError(_err_msg)
            prefix_vmem[rows] = jnp.where(
                layer_index % 12 == 0,
                dp_owned_vmem[...],
                prefix_vmem[rows] + dp_owned_vmem[...],
            ).astype(jnp.bfloat16)

        def mix_row(row: _Dynamic) -> _Dynamic:
            """Execute mix row logic."""
            return _residual_mixture_row(
                block_rows(row),
                prefix_vmem[row : row + 1],
                folded,
                active_blocks,
                epsilon=eps,
                relaxed_reduction_order=residual_reduction_order == "relaxed",
            )

        if sp8_residual:
            _mix_sp8_residual(mix_row)
        else:
            store_mixed_rows(
                mix_row,
                normalized_vmem if start_block and kda_shard_swap else dp_owned_vmem,
            )
        if start_block:

            @pl.when(layer_index % 12 == 0)
            def snapshot() -> _Dynamic:
                """Execute snapshot logic."""
                _snapshot_blocks(layer_index)

    def _stage_native_rows(self, scale: _Dynamic) -> None:
        """Stage and normalize native rows for batch size 64."""
        (
            attention_dp,
            dp_owned_vmem,
            group_rows_per_pair,
            hidden_size,
            normalize,
            owned_rows_vmem,
        ) = (
            getattr(self, a)
            for a in (
                "attention_dp",
                "dp_owned_vmem",
                "group_rows_per_pair",
                "hidden_size",
                "normalize",
                "owned_rows_vmem",
            )
        )
        group = jax.lax.axis_index("tp") // (32 // attention_dp)
        for data_group in range(attention_dp):

            @pl.when(group == data_group)
            def stage_native_group(data_group: _Dynamic = data_group) -> _Dynamic:
                """Execute stage native group logic."""
                first = data_group * group_rows_per_pair
                parts = []
                if first:
                    parts.append(jnp.zeros((first, hidden_size), jnp.bfloat16))
                if not dp_owned_vmem is not None:
                    _err_msg = "Assertion failed: dp_owned_vmem is not None"
                    raise AssertionError(_err_msg)
                parts.append(dp_owned_vmem[...])
                tail = 16 - first - group_rows_per_pair
                if tail:
                    parts.append(jnp.zeros((tail, hidden_size), jnp.bfloat16))
                owned_rows_vmem[...] = jnp.concatenate(tuple(parts), axis=0)

        normalize(owned_rows_vmem, scale)
        native_rows = owned_rows_vmem[...]
        for data_group in range(attention_dp):

            @pl.when(group == data_group)
            def select_normalized_rows(data_group: _Dynamic = data_group) -> _Dynamic:
                """Execute select normalized rows logic."""
                first = data_group * group_rows_per_pair
                if not dp_owned_vmem is not None:
                    _err_msg = "Assertion failed: dp_owned_vmem is not None"
                    raise AssertionError(_err_msg)
                dp_owned_vmem[...] = native_rows[first : first + group_rows_per_pair]

    def normalize_dp_rows(
        self, scale: _Dynamic, *, split: _Dynamic = False
    ) -> _Dynamic:
        """Execute normalize dp rows logic."""
        _stage_native_rows, batch_size, dp_owned_vmem, normalize, sp8_rows_vmem = (
            getattr(self, a)
            for a in (
                "_stage_native_rows",
                "batch_size",
                "dp_owned_vmem",
                "normalize",
                "sp8_rows_vmem",
            )
        )
        if split:
            normalize(sp8_rows_vmem, scale)
            return
        if batch_size != _CONST_64:
            normalize(dp_owned_vmem, scale)
            return
        _stage_native_rows(scale)

    def swap_kda_projections(self, stream: _Dynamic) -> _Dynamic:
        """DP2 B2/B4's shard swap: this rank projects its own group's held.

        shard for every row, sends the other group's rows to its
        counterpart there (which projects the other shard) and gets this
        group's rows of that shard back. Returns ``rows(order)``: the
        group's rows of the own shard's input projection (order 0), then,
        after the exchange, of the other shard's (order 1).
        """
        attention_dp, communication_workspace, group_rows, normalized_vmem = (
            getattr(self, a)
            for a in (
                "attention_dp",
                "communication_workspace",
                "group_rows",
                "normalized_vmem",
            )
        )
        rank = jax.lax.axis_index("tp")
        group = rank // (32 // attention_dp)
        projected = stream.dot(
            normalized_vmem[...], communication_workspace.reduction_phase[0] >= 0
        )
        projected = projected.astype(jnp.bfloat16)
        own = jnp.where(group == 0, projected[:group_rows], projected[group_rows:])
        other = jnp.where(group == 0, projected[group_rows:], projected[:group_rows])
        used = group_rows * projected.shape[1] // 128
        payload = used + used % 2
        local = communication_workspace.shared_local_reduction_vmem
        send = local.at[4].bitcast(jnp.bfloat16).at[pl.ds(0, payload), :]
        receive = local.at[5].bitcast(jnp.bfloat16).at[pl.ds(0, payload), :]
        wire = other.reshape(used, 128)
        if payload != used:
            wire = jnp.concatenate(
                [wire, jnp.zeros((payload - used, 128), jnp.bfloat16)]
            )
        send[...] = wire
        copy = tpu.make_async_remote_copy(
            send,
            receive,
            communication_workspace.send_semaphores.at[14],
            communication_workspace.receive_semaphores.at[14],
            device_id=(rank ^ 32 // attention_dp,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()

        def rows(order: _Dynamic) -> _Dynamic:
            """Execute rows logic."""
            if order == 0:
                return own
            copy.wait()
            return receive[...][:used].reshape(own.shape)

        return rows

    def gather_dp_rows(
        self, destination: _Dynamic, *, split: _Dynamic = False
    ) -> _Dynamic:
        """Execute gather dp rows logic."""
        (
            attention_dp,
            communication_workspace,
            dp_owned_vmem,
            owned_rows_vmem,
            sp8_owned_value,
            sp_owners,
        ) = (
            getattr(self, a)
            for a in (
                "attention_dp",
                "communication_workspace",
                "dp_owned_vmem",
                "owned_rows_vmem",
                "sp8_owned_value",
                "sp_owners",
            )
        )
        if split:
            _gather_sp8_dp_moe_rows(
                sp8_owned_value(),
                owned_rows_vmem,
                destination,
                communication_workspace.shared_local_reduction_vmem,
                communication_workspace.send_semaphores,
                communication_workspace.receive_semaphores,
            )
            return
        if sp_owners == 1:
            if not dp_owned_vmem is not None:
                _err_msg = "Assertion failed: dp_owned_vmem is not None"
                raise AssertionError(_err_msg)
            destination[...] = dp_owned_vmem[...]
            return
        if not dp_owned_vmem is not None:
            _err_msg = "Assertion failed: dp_owned_vmem is not None"
            raise AssertionError(_err_msg)
        _gather_dp_moe_rows(
            dp_owned_vmem[...],
            owned_rows_vmem,
            destination,
            communication_workspace.shared_local_reduction_vmem,
            communication_workspace.send_semaphores,
            communication_workspace.receive_semaphores,
            attention_dp,
            pair_semaphores=communication_workspace.pair_semaphores,
        )

    def prefetch_norms(self, index: _Dynamic) -> _Dynamic:
        """Execute prefetch norms logic."""
        (
            attention_norm_vmem,
            attention_residual_vmem,
            ffn_norm_vmem,
            layer_weights,
            mlp_residual_vmem,
            norm_semaphores,
        ) = (
            getattr(self, a)
            for a in (
                "attention_norm_vmem",
                "attention_residual_vmem",
                "ffn_norm_vmem",
                "layer_weights",
                "mlp_residual_vmem",
                "norm_semaphores",
            )
        )
        destinations = (
            attention_residual_vmem,
            mlp_residual_vmem,
            attention_norm_vmem,
            ffn_norm_vmem,
        )
        for slot, (name, destination) in enumerate(
            zip(NORMS, destinations, strict=True)
        ):
            tpu.make_async_copy(
                layer_weights[name].at[index], destination, norm_semaphores.at[slot]
            ).start()

    def storage(self, name: _Dynamic) -> _Dynamic:
        """Execute storage logic."""
        pool_weights = self.pool_weights
        value = pool_weights[name]
        return pool_alias.storage_size(value.shape, value.dtype)

    def weight_stream(
        self,
        name: _Dynamic,
        source: _Dynamic,
        full: _Dynamic,
        *,
        offset: _Dynamic = None,
    ) -> _Dynamic:
        """The stream that owns ``name``'s copy, or None when the weight is.

        one chunk (plain whole-weight copy).
        """
        attention_dp, projection_expert_pool_vmem, stream_config, stream_semaphores = (
            getattr(self, a)
            for a in (
                "attention_dp",
                "projection_expert_pool_vmem",
                "stream_config",
                "stream_semaphores",
            )
        )
        axis = STREAMED_WEIGHTS[name][0]
        chunks, depth = stream_config.get(name, (1, 1))
        if chunks == 1:
            return None
        base = _stream_base(name, depth)
        order = None
        if axis == "rows":
            if chunks != 2 * attention_dp:
                _err_msg = f"{name} streams as its {2 * attention_dp} pair halves"
                raise ValueError(_err_msg)
            lane = jax.lax.axis_index("tp") % 2

            def order(position: _Dynamic) -> _Dynamic:
                """Execute order logic."""
                return position % attention_dp * 2 + (lane ^ position // attention_dp)

        extent = source.shape[1 if axis == "columns" else 0]
        unit = 128 if axis == "columns" else 16
        if axis == "columns" and attention_dp > 1:
            if chunks % attention_dp:
                _err_msg = (
                    f"{name}: chunks must be a multiple of the attention DP degree"
                )
                raise ValueError(_err_msg)
            sizes = (
                _chunk_sizes(extent // attention_dp, chunks // attention_dp, unit)
                * attention_dp
            )
        else:
            sizes = _chunk_sizes(extent, chunks, unit)
        stream = ChunkedWeightStream(
            source,
            full,
            stream_semaphores,
            axis=axis,
            sizes=sizes,
            depth=depth,
            base=base,
            order=order,
            pool=None if offset is None else projection_expert_pool_vmem,
            offset=offset,
        )
        stream.shards = attention_dp
        return stream

    @_scoped("t9_dp_project")
    def project_attention_dp(self, weight_view: _Dynamic, stream: _Dynamic) -> _Dynamic:
        """Attention DP: project every shard's gated features into this.

        rank's owned rows (the token-scatter attention output).
        """
        (
            communication_workspace,
            dp_attended_vmem,
            dp_held_rows,
            group_rows,
            hidden_size,
            sp_owners,
        ) = (
            getattr(self, a)
            for a in (
                "communication_workspace",
                "dp_attended_vmem",
                "dp_held_rows",
                "group_rows",
                "hidden_size",
                "sp_owners",
            )
        )
        owned = _bf16_rows_view(
            communication_workspace.attention_output_vmem, dp_held_rows, hidden_size
        )
        pl.run_scoped(
            lambda comm: _project_attention_dp(
                dp_attended_vmem,
                weight_view,
                owned,
                comm,
                communication_workspace.shared_local_reduction_vmem,
                communication_workspace.shared_host_reduction_vmem,
                communication_workspace.attention_features_vmem,
                communication_workspace.send_semaphores,
                communication_workspace.receive_semaphores,
                communication_workspace.reduction_phase,
                _dot,
                weight_stream=stream,
                staging=communication_workspace.projection_staging_vmem,
                owners=sp_owners,
            ),
            tpu.VMEM((group_rows, 8192), jnp.float32),
        )

    def query_down_weight_stream(
        self, state_index: _Dynamic, base: _Dynamic = 0
    ) -> _Dynamic:
        """MLA's q-down projection (at the attention weights' ``base``) as a.

        column stream, if configured.
        """
        attention_views, layer_weights, weight_stream = (
            getattr(self, a)
            for a in ("attention_views", "layer_weights", "weight_stream")
        )
        return weight_stream(
            "m_qa",
            layer_weights["m_qa"].at[state_index],
            attention_views(base)[1][0],
            offset=base if POOLED_WEIGHTS else None,
        )

    def kda_streams(self, state_index: _Dynamic, base: _Dynamic = 0) -> _Dynamic:
        """Execute kda streams logic."""
        (
            attention_dp,
            attention_views,
            kda_shard_swap,
            layer_weights,
            storage,
            stream_semaphores,
            weight_stream,
        ) = (
            getattr(self, a)
            for a in (
                "attention_dp",
                "attention_views",
                "kda_shard_swap",
                "layer_weights",
                "storage",
                "stream_semaphores",
                "weight_stream",
            )
        )
        kda_views = attention_views(base)[0]
        source = layer_weights["k_projection"].at[state_index]
        if kda_shard_swap:
            width = source.shape[1] // attention_dp
            group = jax.lax.axis_index("tp") // (32 // attention_dp)
            input_stream = ChunkedWeightStream(
                source.at[:, pl.ds(pl.multiple_of(group * width, 128), width)],
                kda_views[1],
                stream_semaphores,
                axis="columns",
                sizes=(width,),
                depth=1,
            )
        else:
            input_stream = weight_stream(
                "k_projection", source, kda_views[1], offset=base + storage("k_gate")
            )
        streams = {
            "input": input_stream,
            "output": weight_stream(
                "k_o",
                layer_weights["k_o"].at[state_index],
                kda_views[3],
                offset=base
                + storage("k_gate")
                + storage("k_projection")
                + storage("k_fb"),
            ),
        }
        return {key: stream for key, stream in streams.items() if stream is not None}

    def prefetch_attention(self, index: _Dynamic, base: _Dynamic = 0) -> _Dynamic:
        """Execute prefetch attention logic."""
        (
            attention_views,
            kda_streams,
            layer_weights,
            pool_semaphores,
            query_down_weight_stream,
        ) = (
            getattr(self, a)
            for a in (
                "attention_views",
                "kda_streams",
                "layer_weights",
                "pool_semaphores",
                "query_down_weight_stream",
            )
        )
        kda_projection_vmem, mla_projection_vmem = attention_views(base)

        def prefetch_kda() -> _Dynamic:
            """Execute prefetch kda logic."""
            state_index = index - index // 4
            projection_indices = (0, 2, 1, 3)
            streams = kda_streams(state_index, base)
            streamed = {
                name: streams[key]
                for key, name in (("input", "k_projection"), ("output", "k_o"))
                if key in streams
            }
            for slot, projection_index in enumerate(projection_indices):
                if KDA[projection_index] in streamed:
                    streamed[KDA[projection_index]].start()
                    continue
                tpu.make_async_copy(
                    layer_weights[KDA[projection_index]].at[state_index],
                    kda_projection_vmem[projection_index],
                    pool_semaphores.at[slot],
                ).start()

        def prefetch_mla() -> _Dynamic:
            """Execute prefetch mla logic."""
            state_index = index // 4
            query_down_stream = query_down_weight_stream(state_index, base)
            for slot, (name, destination) in enumerate(
                zip(MLA[:2], mla_projection_vmem, strict=True)
            ):
                if name == "m_qa" and query_down_stream is not None:
                    query_down_stream.start()
                    continue
                tpu.make_async_copy(
                    layer_weights[name].at[state_index],
                    destination,
                    pool_semaphores.at[slot],
                ).start()

        jax.lax.cond(
            (index % 4 == _CONST_3) | (index == _CONST_92), prefetch_mla, prefetch_kda
        )

    def prefetch_auxiliary(self, index: _Dynamic) -> _Dynamic:
        """Execute prefetch auxiliary logic."""
        (
            auxiliary_semaphores,
            auxiliary_vmem,
            convolution_ref,
            kda_state_destination,
            kda_state_source,
            layer_weights,
            persist_kda_state,
            recurrent_ref,
        ) = (
            getattr(self, a)
            for a in (
                "auxiliary_semaphores",
                "auxiliary_vmem",
                "convolution_ref",
                "kda_state_destination",
                "kda_state_source",
                "layer_weights",
                "persist_kda_state",
                "recurrent_ref",
            )
        )

        @pl.when((index % 4 != _CONST_3) & (index != _CONST_92))
        def prefetch_kda_auxiliary() -> _Dynamic:
            """Execute prefetch kda auxiliary logic."""
            state_index = index - index // 4
            sources = tuple(layer_weights[name].at[state_index] for name in KDA[4:])
            destinations = list(auxiliary_vmem[: len(sources)])
            if persist_kda_state:
                sources += (
                    kda_state_source(convolution_ref, state_index),
                    kda_state_source(recurrent_ref, state_index),
                )
                destinations += [
                    kda_state_destination(auxiliary_vmem[4]),
                    kda_state_destination(auxiliary_vmem[5]),
                ]
            for slot, (source, destination) in enumerate(
                zip(sources, destinations, strict=True)
            ):
                tpu.make_async_copy(
                    source, destination, auxiliary_semaphores.at[slot]
                ).start()

    def prefetch_metadata(
        self, index: _Dynamic, *, mixture_metadata: _Dynamic = True
    ) -> _Dynamic:
        """Execute prefetch metadata logic."""
        layer_weights, metadata_semaphores, metadata_vmem = (
            getattr(self, a)
            for a in ("layer_weights", "metadata_semaphores", "metadata_vmem")
        )
        buffer_slot = index % 2

        @pl.when((index % 4 == _CONST_3) | (index == _CONST_92))
        def prefetch_mla_norms() -> _Dynamic:
            """Execute prefetch mla norms logic."""
            for slot, name in enumerate(("m_qnorm", "m_knorm")):
                tpu.make_async_copy(
                    layer_weights[name].at[index // 4],
                    metadata_vmem[slot].at[buffer_slot],
                    metadata_semaphores.at[buffer_slot, slot],
                ).start()

        if not mixture_metadata:
            return
        for slot, name in enumerate(("router_bias", "latent_norm"), 2):
            tpu.make_async_copy(
                layer_weights[name].at[index - 1],
                metadata_vmem[slot].at[buffer_slot],
                metadata_semaphores.at[buffer_slot, slot],
            ).start()

    def prefetch_router(
        self, index: _Dynamic, *, initialize_padding: _Dynamic = False
    ) -> _Dynamic:
        """Execute prefetch router logic."""
        layer_weights, pool_managed, router_semaphores, router_vmem = (
            getattr(self, a)
            for a in (
                "layer_weights",
                "pool_managed",
                "router_semaphores",
                "router_vmem",
            )
        )
        if pool_managed:
            return
        _router_copy(
            layer_weights["router"].at[index - 1],
            router_vmem,
            router_semaphores,
            bool(0),
            initialize_padding=initialize_padding,
        )

    def _init_body_buffers_and_positions(self) -> None:
        """Initialize body scratch buffers, barriers, and group positions."""
        self.pool = (
            VmemPool(
                self.projection_expert_pool_vmem,
                self.pool_layout["arena_bytes"],
                report=_config().pool_report,
            )
            if self.pool_managed
            else None
        )
        self.auxiliary_vmem = (
            self.convolution_filter_vmem,
            self.exp_log_decay_rate_vmem,
            self.time_step_bias_vmem,
            self.output_normalization_vmem,
            self.convolution_history_vmem,
            self.recurrent_state_vmem,
        )
        self.metadata_vmem = (
            self.query_norms_vmem,
            self.key_value_norms_vmem,
            self.routing_biases_vmem,
            self.latent_norms_vmem,
        )
        collectives32.barrier()
        self.communication_workspace.reduction_phase[0] = 0
        if self.attention_dp > 1:
            if not self.group_positions_smem is not None:
                _err_msg = "Assertion failed: group_positions_smem is not None"
                raise AssertionError(_err_msg)
            data_group = jax.lax.axis_index("tp") // (32 // self.attention_dp)
            for row in range(self.group_rows):
                pair_index, within = divmod(row, self.group_rows_per_pair)
                self.group_positions_smem[row] = self.position_ref[
                    self.tokens_per_pair * pair_index
                    + data_group * self.group_rows_per_pair
                    + within
                ]

    def _init_body_prefix_and_prefetch(self) -> None:
        """Initialize prefix blocks and async prefetch for initial layer."""
        if self.attention_dp_ownership:
            pl.run_scoped(
                self.load_prefix,
                tpu.VMEM(self.prefix_ref.shape, self.prefix_ref.dtype),
                tpu.SemaphoreType.DMA,
            )
        else:
            pl.run_scoped(
                functools.partial(self.load_prefix, self.prefix_vmem),
                tpu.SemaphoreType.DMA,
            )
        self.blocks_vmem[...] = jnp.zeros(
            self.blocks_vmem.shape, self.blocks_vmem.dtype
        )
        if not (
            self.include_layer_zero
            and self.attention_dp == _CONST_2
            and (self.batch_size in (64, 128))
            and (self.sequence_rows == 1)
        ):
            pl.run_scoped(
                self.staged_blocks,
                tpu.VMEM(self.embedding_ref.shape, self.embedding_ref.dtype),
                tpu.SemaphoreType.DMA,
            )
        self.dp_owned_vmem = (
            _bf16_rows_view(
                self.communication_workspace.attention_output_vmem,
                self.dp_held_rows,
                self.hidden_size,
            )
            if self.attention_dp_ownership
            else None
        )
        self.sp8_rows_vmem = self.dp_owned_vmem if self.sp8_residual else None
        if self.include_layer_zero:
            if not self.pool_managed:

                @pl.when(jax.lax.axis_index("tp") % 2 == 1)
                def initialize_router_padding() -> _Dynamic:
                    """Execute initialize router padding logic."""
                    self.router_vmem[0, :, 384:] = jnp.zeros(
                        (self.router_vmem.shape[1], 128), jnp.bfloat16
                    )
        else:
            self.prefetch_router(jnp.int32(1), initialize_padding=True)
        self.prefetch_norms(jnp.int32(self.first_layer))
        if self.pool is not None:
            self.request_attention(jnp.int32(self.first_layer))
        else:
            self.prefetch_attention(jnp.int32(self.first_layer))
        self.prefetch_auxiliary(jnp.int32(self.first_layer))
        self.prefetch_metadata(
            jnp.int32(self.first_layer), mixture_metadata=not self.include_layer_zero
        )
        if self.pool is not None and not self.include_layer_zero:
            self.request_router(jnp.int32(1))

    def body(
        self,
        position_ref: _Dynamic,
        state_row_ref: _Dynamic,
        *refs: _Dynamic,
    ) -> None:
        """Pallas grid body function."""
        self.position_ref = position_ref
        self.state_row_ref = state_row_ref
        self._setup_body_refs(refs)
        self._setup_body_vmem()
        self._init_body_buffers_and_positions()
        self._init_body_prefix_and_prefetch()

        runner = _LayerRunner(self)
        if self.include_layer_zero:
            runner.layer(jnp.int32(0), dense_layer=True)
        if self.pool is not None:
            self.pool.loop(1, self.layers)(runner.layer)
            if self.pool.report:
                sys.stdout.write(f"{self.pool.timeline()}\n")
                sys.stdout.flush()
        else:
            pl.loop(1, self.layers)(runner.layer)

        if self.headroom_pad is not None:
            self.prefix_vmem[0:8, 0:128] += jnp.sum(
                self.headroom_pad[...].reshape(-1, 8, 128), axis=0
            ).astype(jnp.bfloat16)

        _BodyHelpers(self).finish_outputs()

    def _build_kernel_inputs_and_shapes(self) -> None:
        """Prepare kernel args, outputs and expert pool dimensions."""

        def shape(value: _Dynamic) -> _Dynamic:
            """Execute shape logic."""
            return jax.ShapeDtypeStruct(value.shape, value.dtype)

        (
            alias_expert_scales,
            aux_hidden_layers,
            batch_size,
            cache_handoff_head_count,
            dense_names,
            embedding,
            hidden_size,
            include_layer_zero,
            local_weights,
            prefix,
            select_state_row,
            sequence_parallel_pre_attention,
            state_row,
            states,
            weights,
        ) = (
            getattr(self, a)
            for a in (
                "alias_expert_scales",
                "aux_hidden_layers",
                "batch_size",
                "cache_handoff_head_count",
                "dense_names",
                "embedding",
                "hidden_size",
                "include_layer_zero",
                "local_weights",
                "prefix",
                "select_state_row",
                "sequence_parallel_pre_attention",
                "state_row",
                "states",
                "weights",
            )
        )
        args = (
            prefix,
            embedding,
            *(weights[name] for name in NAMES + EPILOGUE + dense_names),
            *states,
        )
        state_row_scalar = (
            jnp.asarray(state_row, jnp.int32).reshape(1)
            if select_state_row
            else jnp.zeros((1,), jnp.int32)
        )
        # (With state_slots the inputs are the whole slot arrays; the updated
        # states are per row.)
        state_output_shapes = (
            jax.ShapeDtypeStruct((batch_size, *states[0].shape[1:]), states[0].dtype),
            jax.ShapeDtypeStruct((batch_size, *states[1].shape[1:]), states[1].dtype),
            jax.ShapeDtypeStruct(
                (batch_size, MLA_LAYERS, cache_handoff_head_count, 256), jnp.bfloat16
            ),
            jax.ShapeDtypeStruct(
                (batch_size, MLA_LAYERS, cache_handoff_head_count, 128), jnp.bfloat16
            ),
        )
        output_shapes = (
            shape(prefix),
            (
                jax.ShapeDtypeStruct((8, 128), jnp.uint8)
                if sequence_parallel_pre_attention
                else jax.ShapeDtypeStruct((batch_size, 8, hidden_size), jnp.bfloat16)
            ),
            *state_output_shapes,
            jax.ShapeDtypeStruct(
                (max(1, len(aux_hidden_layers)), batch_size, hidden_size), jnp.bfloat16
            ),
        )
        pooled_expert_names = (
            ("expert_gu", "expert_gus", "expert_down", "expert_ds")
            if alias_expert_scales
            else ("expert_gu", "expert_down")
        )
        expert_pool_bytes = sum(
            pool_alias.storage_size(
                (4, *local_weights[name].shape[1:]), local_weights[name].dtype
            )
            for name in pooled_expert_names
        )
        # Above B64 three slabs keep the arena at the expert slots' size (VMEM).
        dense_slabs = 4 if batch_size <= _CONST_64 else 3
        dense_slab_bytes = (
            max(
                pool_alias.storage_size(
                    (1024, weights["dense_gu"].shape[1]), weights["dense_gu"].dtype
                ),
                pool_alias.storage_size(
                    (384, hidden_size), weights["dense_down"].dtype
                ),
            )
            if include_layer_zero
            else 0
        )

        (
            self.args,
            self.state_row_scalar,
            self.state_output_shapes,
            self.output_shapes,
            self.pooled_expert_names,
            self.expert_pool_bytes,
            self.dense_slabs,
            self.dense_slab_bytes,
        ) = (
            args,
            state_row_scalar,
            state_output_shapes,
            output_shapes,
            pooled_expert_names,
            expert_pool_bytes,
            dense_slabs,
            dense_slab_bytes,
        )

    def _compute_pool_layout(self) -> None:
        """Compute pool layout and verify expert pool footprint."""
        (
            alias_expert_scales,
            dense_slab_bytes,
            dense_slabs,
            expert_pool_bytes,
            pool_managed,
            pool_weights,
            pooled_expert_names,
        ) = (
            getattr(self, a)
            for a in (
                "alias_expert_scales",
                "dense_slab_bytes",
                "dense_slabs",
                "expert_pool_bytes",
                "pool_managed",
                "pool_weights",
                "pooled_expert_names",
            )
        )
        pool_layout = _pool_layout(
            pool_weights,
            alias_expert_scales=alias_expert_scales,
            pooled=POOLED_WEIGHTS,
            managed=pool_managed,
            dense_bytes=dense_slabs * dense_slab_bytes,
        )
        pool_bytes = pool_layout["pool_bytes"]
        if _config().vmem_report:

            def _mib(name: _Dynamic, experts: _Dynamic = None) -> _Dynamic:
                """Execute mib logic."""
                if experts is None:
                    experts = bool(0)
                value = pool_weights[name]
                shape = (4, *value.shape[1:]) if experts else value.shape
                return pool_alias.storage_size(shape, value.dtype) / 2**20

            region_a = {
                "kda": sum(_mib(n) for n in KDA[:4]),
                "mla": sum(_mib(n) for n in MLA[:2]),
                "experts": sum(_mib(n, bool(1)) for n in pooled_expert_names),
            }
            region_b = {
                "mla_extra": sum(_mib(n) for n in MLA_EXTRA_WEIGHTS),
                "moe_prepare": sum(_mib(n) for n in MOE_PREPARE_WEIGHTS),
                "latent_up": _mib("latent_up"),
            }
            region_a_mib = {k: round(v, 2) for k, v in region_a.items()}
            sys.stdout.write(f"pool region A (MiB): {region_a_mib}\n")
            sys.stdout.flush()
            if pool_managed:
                ab = pool_layout["arena_bytes"] / 2**20
                attb = pool_layout["attention_bytes"] / 2**20
                dsb = dense_slabs * dense_slab_bytes / 2**20
                sys.stdout.write(
                    f"arena (managed region A): {ab:.2f} MiB, "
                    f"attention {attb:.2f}, dense slabs {dsb:.2f}\n"
                )
                sys.stdout.flush()
            b_breakdown = {
                n: round(_mib(n), 2)
                for n in (*MLA_EXTRA_WEIGHTS, *MOE_PREPARE_WEIGHTS, "latent_up")
            }
            reg_b_str = {k: round(v, 2) for k, v in region_b.items()}
            sys.stdout.write(f"pool region B (MiB): {reg_b_str} {b_breakdown}\n")
            sys.stdout.flush()
        if not (pool_bytes >= expert_pool_bytes and pool_bytes % 128 == 0):
            _err_msg = (
                "Assertion failed: pool_bytes >= expert_pool_bytes and "
                "pool_bytes % 128 == 0"
            )
            raise AssertionError(_err_msg)

        self.pool_layout = pool_layout
        self.pool_bytes = pool_bytes

    def _compute_scratch_buffers_a(self) -> tuple[_Dynamic, ...]:
        """Compute core VMEM scratch allocations."""
        (
            attention_dp_ownership,
            batch_size,
            block_rows_held,
            blocks_block_major,
            compact_outputs,
            hidden_size,
            local_weights,
            persist_kda_state,
            pool_bytes,
            pool_managed,
            prefix,
            states,
            weights,
        ) = (
            getattr(self, a)
            for a in (
                "attention_dp_ownership",
                "batch_size",
                "block_rows_held",
                "blocks_block_major",
                "compact_outputs",
                "hidden_size",
                "local_weights",
                "persist_kda_state",
                "pool_bytes",
                "pool_managed",
                "prefix",
                "states",
                "weights",
            )
        )

        return (
            tpu.VMEM(
                (-(-block_rows_held // 16) * 16, hidden_size)
                if attention_dp_ownership
                else prefix.shape,
                jnp.bfloat16,
            ),
            # Residual snapshots are exact BF16 values. Up to B8: FP32
            # [batch, 8, hidden], no unpack per mix (a two-row allocation for the
            # owned rows miscompiles across loop iterations). Above B8 or with DP
            # row ownership: BF16, two rows per native (16, 128) tile.
            tpu.VMEM(
                (-(-block_rows_held // 2), 16, hidden_size)
                if blocks_block_major
                else (batch_size, 8, hidden_size),
                jnp.bfloat16 if blocks_block_major else jnp.float32,
            ),
            tpu.VMEM(prefix.shape, jnp.bfloat16),
            *(
                (tpu.VMEM((16, 128), jnp.bfloat16), tpu.VMEM((16, 128), jnp.bfloat16))
                if compact_outputs
                else (
                    tpu.VMEM(prefix.shape, jnp.bfloat16),
                    tpu.VMEM(prefix.shape, jnp.bfloat16),
                )
            ),
            *(tpu.VMEM((1, hidden_size), weights[name].dtype) for name in NORMS),
            # With the router in the arena this persistent slot is a stub.
            tpu.VMEM((8, 128) if pool_managed else (1, hidden_size, 512), jnp.bfloat16),
            tpu.SemaphoreType.DMA((2,)),
            tpu.VMEM((pool_bytes // 128, 128), jnp.uint8),
            tpu.SemaphoreType.DMA((6,)),
            tpu.SemaphoreType.DMA((4,)),
            *(
                tpu.VMEM(local_weights[name].shape, local_weights[name].dtype)
                for name in KDA[4:]
            ),
            *(
                (
                    tpu.VMEM((batch_size, *states[0].shape[2:]), states[0].dtype),
                    tpu.VMEM((batch_size, *states[1].shape[2:]), states[1].dtype),
                )
                if persist_kda_state
                else (
                    tpu.VMEM((1,), jnp.uint8),
                    tpu.VMEM((1,), jnp.uint8),
                )
            ),
            tpu.SemaphoreType.DMA((6,)),
            *(
                tpu.VMEM((2, *local_weights[name].shape), local_weights[name].dtype)
                for name in ("m_qnorm", "m_knorm", "router_bias", "latent_norm")
            ),
        )

    def _compute_scratch_buffers_b(self) -> tuple[_Dynamic, ...]:
        """Compute view and communication VMEM scratch allocations."""
        (
            attention_dp,
            attention_token_scatter,
            b4_attention_gather,
            batch_size,
            group_rows,
            hidden_size,
            moe_routed_reduce_scatter,
            owned_rows,
            pair_split_gather,
            sequence_rows,
            states,
            stream_config,
        ) = (
            getattr(self, a)
            for a in (
                "attention_dp",
                "attention_token_scatter",
                "b4_attention_gather",
                "batch_size",
                "group_rows",
                "hidden_size",
                "moe_routed_reduce_scatter",
                "owned_rows",
                "pair_split_gather",
                "sequence_rows",
                "states",
                "stream_config",
            )
        )

        return (
            tpu.SemaphoreType.DMA((2, 4)),
            tpu.VMEM(
                (owned_rows, hidden_size) if attention_token_scatter else (1,),
                jnp.bfloat16,
            ),
            tpu.SemaphoreType.DMA((1,)),
            tpu.SemaphoreType.DMA((4,)),
            tpu.SemaphoreType.DMA((_stream_semaphore_count(stream_config),)),
            *(
                (
                    # Attention DP: the group's attention input rows, each shard's
                    # gated attention output, the group rows' positions, and the
                    # MLA latents shared by the shards.
                    tpu.VMEM((group_rows, hidden_size), jnp.bfloat16),
                    tpu.VMEM((attention_dp, group_rows, 384), jnp.bfloat16),
                    tpu.SMEM((group_rows,), jnp.int32),
                    tpu.VMEM((group_rows, 1536), jnp.bfloat16),
                    tpu.VMEM((group_rows, 640), jnp.bfloat16),
                )
                if attention_dp > 1
                else ()
            ),
            # DCP: buffers group peers' remote copies land in.
            *mla_dcp.persistent_scratch(
                group_rows,
                heads=3 * attention_dp,
                cache_dtype=states[2].dtype,
                sequence_rows=sequence_rows,
            ),
            *communication_scratch(batch_size, attention_dp)[:-1],
            tpu.VMEM((batch_size, 4096), jnp.bfloat16),
            *projection_staging_scratch(batch_size, attention_dp),
            *routed_send_scratch(batch_size),
            *((tpu.SemaphoreType.REGULAR,) if moe_routed_reduce_scatter else ()),
            *(
                (tpu.SemaphoreType.DMA((1,)), tpu.SemaphoreType.DMA((1,)))
                if pair_split_gather
                else ()
            ),
            # Four 7168-element BF16 wires, padded to 64 BF16 transport rows.
            # Private storage (see B4_ATTENTION_GATHER).
            *((tpu.VMEM((4, 32, 128), jnp.float32),) if b4_attention_gather else ()),
            # Diagnostic: vmem_pad_kb reserves unused VMEM to probe headroom.
            *(
                (tpu.VMEM((_config().vmem_pad_kb * 2, 128), jnp.float32),)
                if _config().vmem_pad_kb
                else ()
            ),
        )

    def _compute_scratch(self) -> None:
        """Combine scratch buffers and names."""
        self.scratch = (
            *self._compute_scratch_buffers_a(),
            *self._compute_scratch_buffers_b(),
        )
        (
            attention_dp,
            b4_attention_gather,
            batch_size,
            moe_routed_reduce_scatter,
            pair_split_gather,
        ) = (
            getattr(self, a)
            for a in (
                "attention_dp",
                "b4_attention_gather",
                "batch_size",
                "moe_routed_reduce_scatter",
                "pair_split_gather",
            )
        )
        scratch_names = (
            "prefix",
            "blocks",
            "normalized",
            "attention",
            "mixture",
            *NORMS,
            "router",
            "router_semaphores",
            "projection_expert_pool",
            "pool_semaphores",
            "norm_semaphores",
            *KDA[4:],
            "kda_history",
            "kda_state",
            "auxiliary_semaphores",
            "m_qnorm",
            "m_knorm",
            "router_bias",
            "latent_norm",
            "metadata_semaphores",
            "owned_rows",
            "aux_semaphore",
            "dense_semaphores",
            "stream_semaphores",
            *(
                (
                    "dp_group_input",
                    "dp_attended",
                    "dp_group_positions",
                    "dp_query_latent",
                    "dp_compressed",
                )
                if attention_dp > 1
                else ()
            ),
            *mla_dcp.PERSISTENT_NAMES,
            "shared_local_reduction",
            "expert_local_reduction",
            "shared_host_reduction",
            "expert_host_reduction",
            "attention_features",
            "attention_output",
            "send_semaphores",
            "receive_semaphores",
            "reduction_phase",
            "latent_gather",
            *(("projection_staging", "routed_send") if batch_size > _CONST_32 else ()),
            *(("routed_consumed",) if moe_routed_reduce_scatter else ()),
            *(
                ("pair_send_semaphore", "pair_receive_semaphore")
                if pair_split_gather
                else ()
            ),
            *(("attention_gather",) if b4_attention_gather else ()),
            "headroom_pad",
        )

        self.scratch_names = scratch_names

    def _compute_vmem_regions(self) -> None:
        """Compute VMEM persistent and phase regions."""
        (
            alias_expert_scales,
            attention_dp,
            batch_size,
            cache_handoff_head_count,
            group_rows,
            hidden_size,
            include_layer_zero,
            kda_row_chunk,
            local_weights,
            mla_scoped_scratch,
            moe_scoped_scratch,
            output_shapes,
            persist_kda_state,
            scratch,
            scratch_names,
            shard_routing,
            states,
            stream_kda_state,
            weights,
        ) = (
            getattr(self, a)
            for a in (
                "alias_expert_scales",
                "attention_dp",
                "batch_size",
                "cache_handoff_head_count",
                "group_rows",
                "hidden_size",
                "include_layer_zero",
                "kda_row_chunk",
                "local_weights",
                "mla_scoped_scratch",
                "moe_scoped_scratch",
                "output_shapes",
                "persist_kda_state",
                "scratch",
                "scratch_names",
                "shard_routing",
                "states",
                "stream_kda_state",
                "weights",
            )
        )
        persistent = [
            (
                "output window 1",
                tpu.VMEM(output_shapes[1].shape, output_shapes[1].dtype),
            ),
            *zip(scratch_names, scratch),
        ]
        # Attention DP projects the group's rows (after the MLA workspace).
        projection_reduction = (
            "output-projection reduction",
            tpu.VMEM((group_rows, 8192), jnp.float32)
            if attention_dp > 1
            else tpu.VMEM(
                (batch_size, 3584 if batch_size > _CONST_32 else 8192), jnp.float32
            ),
        )
        regions = {
            "MLA": (
                [
                    (
                        "new-key handoff",
                        tpu.VMEM(
                            (batch_size, cache_handoff_head_count, 256), jnp.bfloat16
                        ),
                    ),
                    (
                        "new-value handoff",
                        tpu.VMEM(
                            (batch_size, cache_handoff_head_count, 128), jnp.bfloat16
                        ),
                    ),
                    *zip(
                        (
                            *(() if POOLED_WEIGHTS else MLA[2:6]),
                            "key cache tiles",
                            "value cache tiles",
                            "cache semaphores",
                            "query",
                            "output gate values",
                            "new key",
                            "new value",
                            "softmax maximum",
                            "softmax denominator",
                            "accumulator",
                            *mla_dcp.SCOPED_NAMES,
                        ),
                        mla_scoped_scratch,
                        strict=True,
                    ),
                ],
                [
                    *([] if attention_dp > 1 else [[projection_reduction]]),
                    [("new-row hand-out semaphores", tpu.SemaphoreType.DMA((2,)))],
                ],
            ),
            **(
                {"DP projection": ([projection_reduction], [])}
                if attention_dp > 1
                else {}
            ),
            "KDA": (
                []
                if persist_kda_state
                else [
                    (
                        "kda history chunks",
                        tpu.VMEM(
                            (3, kda_row_chunk, *states[0].shape[2:]), states[0].dtype
                        ),
                    ),
                    (
                        "kda state chunks",
                        tpu.VMEM(
                            (3, kda_row_chunk, *states[1].shape[2:]), states[1].dtype
                        ),
                    ),
                ]
                if stream_kda_state
                else [
                    (
                        "kda history",
                        tpu.VMEM((batch_size, *states[0].shape[2:]), states[0].dtype),
                    ),
                    (
                        "kda state",
                        tpu.VMEM((batch_size, *states[1].shape[2:]), states[1].dtype),
                    ),
                ],
                [
                    [
                        projection_reduction,
                        *(
                            (
                                (
                                    "DP staged projection",
                                    tpu.VMEM(
                                        (
                                            group_rows // kda_row_chunk,
                                            kda_row_chunk,
                                            1664,
                                        ),
                                        jnp.bfloat16,
                                    ),
                                ),
                                (
                                    "DP staged recurrence gate",
                                    tpu.VMEM(
                                        (
                                            group_rows // kda_row_chunk,
                                            kda_row_chunk,
                                            384,
                                        ),
                                        jnp.bfloat16,
                                    ),
                                ),
                                (
                                    "DP staged output gate",
                                    tpu.VMEM((group_rows, 384), jnp.bfloat16),
                                ),
                            )
                            if attention_dp > 1 and group_rows > _CONST_8
                            else ()
                        ),
                        *(
                            (
                                (
                                    "chunked attended",
                                    tpu.VMEM((batch_size, 3, 128), jnp.float32),
                                ),
                            )
                            if batch_size > _CONST_8
                            else ()
                        ),
                    ]
                ],
            ),
            "MoE": (
                list(
                    zip(
                        (
                            *(() if POOLED_WEIGHTS else ("latent_up",)),
                            "latent-up semaphore",
                            "expert semaphores",
                            *(
                                ()
                                if alias_expert_scales
                                else ("gate/up scale slots", "down scale slots")
                            ),
                            "selected experts",
                            "selected probabilities",
                            "local route counts",
                            "probability totals",
                            "routing scores",
                            "shared outputs",
                            "routed outputs",
                            "gate/up accumulator",
                            "down accumulator",
                            "processed routes",
                            "compact source 1",
                            "compact source 2",
                            "shared send semaphores",
                            "shared receive semaphores",
                            "output transport",
                            "output send semaphores",
                            "output receive semaphores",
                            "latent send semaphores",
                            "latent receive semaphores",
                            *(
                                ("routing selection source", "routing selections")
                                if shard_routing
                                else ()
                            ),
                        ),
                        moe_scoped_scratch,
                        strict=True,
                    )
                ),
                []
                if POOLED_WEIGHTS
                else [
                    [
                        (
                            name,
                            tpu.VMEM(
                                local_weights[name].shape, local_weights[name].dtype
                            ),
                        )
                        for name in MOE_PREPARE_WEIGHTS
                    ]
                ],
            ),
        }
        if include_layer_zero and batch_size > _CONST_8:
            regions["dense layer 0"] = (
                [
                    (
                        "gate/up accumulator",
                        tpu.VMEM(
                            (batch_size, weights["dense_gu"].shape[1]), jnp.float32
                        ),
                    ),
                    (
                        "down accumulator",
                        tpu.VMEM((batch_size, hidden_size), jnp.float32),
                    ),
                ],
                [],
            )

        self.persistent = persistent
        self.regions = regions

    def _check_vmem_budget(self) -> None:
        """Verify VMEM peak budget against hardware limits."""
        (
            alias_expert_scales,
            batch_size,
            dense_slab_bytes,
            dense_slabs,
            hidden_size,
            persistent,
            pool_layout,
            pool_managed,
            pool_weights,
            regions,
            scratch,
            scratch_names,
        ) = (
            getattr(self, a)
            for a in (
                "alias_expert_scales",
                "batch_size",
                "dense_slab_bytes",
                "dense_slabs",
                "hidden_size",
                "persistent",
                "pool_layout",
                "pool_managed",
                "pool_weights",
                "regions",
                "scratch",
                "scratch_names",
            )
        )
        vmem_report, vmem_peak = _vmem_budget(persistent, regions)
        if pool_managed:
            # Grow the arena, all or nothing, into VMEM above the spill reserve so
            # the next layer's attention projections and router (queued together
            # once the experts finish) fit without waiting for a release.
            # NOTE: derive the demand from a dry-run trace of the requests.
            demand = pool_layout["attention_bytes"] + pool_alias.storage_size(
                (1, hidden_size, 512), jnp.dtype(jnp.bfloat16)
            )
            shortfall = max(0, -(-(demand - pool_layout["arena_bytes"]) // 4096) * 4096)
            available = (
                (VMEM_LIMIT_BYTES - VMEM_SPILL_RESERVE_BYTES - vmem_peak) // 4096 * 4096
            )
            # B128's register spills exceed the reserve: its arena does not grow.
            spare = (
                shortfall if shortfall <= available and batch_size < _CONST_128 else 0
            )
            if spare > 0:
                pool_layout = _pool_layout(
                    pool_weights,
                    alias_expert_scales=alias_expert_scales,
                    pooled=POOLED_WEIGHTS,
                    managed=True,
                    dense_bytes=dense_slabs * dense_slab_bytes,
                    arena_extra=spare,
                )
                pool_bytes = pool_layout["pool_bytes"]
                pool_spec = tpu.VMEM((pool_bytes // 128, 128), jnp.uint8)
                index = scratch_names.index("projection_expert_pool")
                scratch = (*scratch[:index], pool_spec, *scratch[index + 1 :])
                persistent = [
                    (name, pool_spec if name == "projection_expert_pool" else spec)
                    for name, spec in persistent
                ]
                vmem_report, vmem_peak = _vmem_budget(persistent, regions)
            if _config().vmem_report:
                spare_mib = pool_layout["arena_bytes"] / 2**20
                sys.stdout.write(f"arena with spare VMEM: {spare_mib:.2f} MiB\n")
                sys.stdout.flush()
        if _config().vmem_report:
            sys.stdout.write(f"VMEM plan, batch {batch_size}:\n{vmem_report}\n")
            sys.stdout.flush()
        if vmem_peak > VMEM_LIMIT_BYTES:
            _err_msg = (
                f"The decoder's static VMEM plan exceeds the limit:\n{vmem_report}"
            )
            raise ValueError(_err_msg)
        if vmem_peak + VMEM_SPILL_RESERVE_BYTES > VMEM_LIMIT_BYTES:
            warn_msg = (
                "The decoder's VMEM plan leaves less than the spill reserve:\n"
                f"{vmem_report}"
            )
            warnings.warn(warn_msg, stacklevel=2)

        self.vmem_report = vmem_report
        self.vmem_peak = vmem_peak
        self.pool_layout = pool_layout
        self.pool_bytes = getattr(self, "pool_bytes", pool_layout["pool_bytes"])
        self.scratch = scratch
        self.persistent = persistent

    def _invoke_pallas_call(self) -> _Dynamic:
        """Invoke pallas call."""
        body = self.body
        (
            args,
            batch_size,
            block_tables,
            live_rows,
            output_shapes,
            positions,
            scratch,
            slot_reads,
            state_row_scalar,
            state_slots,
        ) = (
            getattr(self, a)
            for a in (
                "args",
                "batch_size",
                "block_tables",
                "live_rows",
                "output_shapes",
                "positions",
                "scratch",
                "slot_reads",
                "state_row_scalar",
                "state_slots",
            )
        )
        return pl.pallas_call(
            body,
            name="kimi_decoder_stack",
            out_shape=output_shapes,
            input_output_aliases={},
            grid_spec=tpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=3 + int(slot_reads) + int(live_rows is not None),
                in_specs=[pl.BlockSpec(memory_space=tpu.HBM) for _ in range(len(args))],
                out_specs=[
                    pl.BlockSpec(memory_space=tpu.HBM),
                    pl.BlockSpec(memory_space=tpu.VMEM),
                    *(pl.BlockSpec(memory_space=tpu.HBM) for _ in range(5)),
                ],
                scratch_shapes=scratch,
            ),
            compiler_params=tpu.CompilerParams(
                collective_id=21,
                vmem_limit_bytes=_config().vmem_limit_mb * 1024**2,
                disable_bounds_checks=not _config().bounds_checks,
                shape_invariant_numerics=True,
                internal_scratch_in_bytes=(
                    _config().internal_scratch_kb * 1024
                    if _config().internal_scratch_kb
                    else None
                ),
            ),
        )(
            positions,
            state_row_scalar,
            jnp.asarray(block_tables, jnp.int32).reshape(-1),
            *(
                (jnp.asarray(state_slots, jnp.int32).reshape(batch_size),)
                if slot_reads
                else ()
            ),
            *(
                (jnp.asarray(live_rows, jnp.int32).reshape(batch_size),)
                if live_rows is not None
                else ()
            ),
            *args,
        )

    def run(self) -> _Dynamic:
        """Compile and execute the fused megakernel stack."""
        self._build_kernel_inputs_and_shapes()
        self._compute_pool_layout()
        self._compute_scratch()
        self._compute_vmem_regions()
        self._check_vmem_budget()
        return self._invoke_pallas_call()


def stack(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Fuse the decoder layers while keeping a native batch tile in VMEM.

    The MLA cache is vLLM's unified pools, read in place with decode context
    parallelism (``mla_dcp``): ``states`` is ``(conv, recurrent, *pools)``,
    MLA layer ``m`` in ``pools[dcp_pool_map[m]]``, ``block_tables`` (int32
    ``[batch_size, pages]``) the rows' tables. The pools stay read-only: in
    place of an updated cache the kernel returns each row's new latent for
    every MLA layer, ``[batch, 24, H, 256]`` / ``[batch, 24, H, 128]`` (``H >=
    3`` padded heads; ``paged.commit_latent_rows`` commits them). Nothing is
    aliased: XLA copies every operand a communicating kernel updates in
    place, so the KDA states come back as fresh outputs too.

    With the token-scatter / sequence-parallel layout, ``sequence_parallel_size``
    local row owners (4: rank pairs; 8: single ranks; 1: every rank, at
    attention DP2's B2 and B4), replicated across the hosts, each own batch_size
    size rows for the per-token work.

    Layers 1..L-1 by default; with ``include_layer_zero`` the loop starts at
    layer 0 (``prefix`` is then the embedding; its KDA runs on the common
    attention path and a dense MLP streamed through the weight pool replaces
    the mixture of experts), which needs ``weights["dense_gu"]`` and
    ``weights["dense_down"]``. ``state_row`` (a traced int32 scalar, one
    sequence per tile only) selects the row of the KDA input states to read
    for the sequence's first row instead of a pre-selected state copy.

    ``state_slots`` (int32 ``[batch_size]``): ``states[:2]`` are the
    whole KDA slot arrays ``[slots, 69, ...]`` and row ``b`` reads its input
    state from slot ``state_slots[b]`` (DP2 indices include the head-shard
    bank offset and follow shard-major group rows; with ``state_row``, slot
    ``state_slots[state_row]``), instead of the caller gathering the rows
    first. The updated states are still returned per row. Needs
    ``state_row`` or ``batch_size`` > 4 with DP1, B2..B128 with DP2 or
    B32..B128 with DP4.
    """
    return _DecodeStackRunner(*_p_args, **_p_kw).run()


class _DecodeLocalRunner:
    """Runner for decoding one native batch tile with local TP32 weights and caches."""

    def __getattr__(self, name: _Dynamic) -> _Dynamic:
        """Dynamic attribute access."""
        raise AttributeError(name)

    def __init__(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Initialize decode local runner."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            self.weights,
            self.states,
            self.tokens,
            self.positions,
        ) = (
            _p_next(_p_iter, _p_get("weights")),
            _p_next(_p_iter, _p_get("states")),
            _p_next(_p_iter, _p_get("tokens")),
            _p_next(_p_iter, _p_get("positions")),
        )
        (
            self.batch_size,
            self.layers,
            self.eps,
            self.residual_reduction_order,
            self.mla_cache_batch_tile_size,
            self.alias_expert_scales,
            self.attention_token_scatter,
            self.moe_routed_reduce_scatter,
            self.moe_token_scatter_output,
        ) = (
            _p_get("batch_size"),
            _p_get("layers", 93),
            _p_get("eps", 1e-05),
            _p_get("residual_reduction_order", "reference"),
            _p_get("mla_cache_batch_tile_size"),
            _p_kw.get("alias_expert_scales", False),
            _p_kw.get("attention_token_scatter", False),
            _p_kw.get("moe_routed_reduce_scatter", False),
            _p_kw.get("moe_token_scatter_output", False),
        )
        self.sequence_parallel_pre_attention = _p_kw.get(
            "sequence_parallel_pre_attention", False
        )
        (
            gate_first_kda_projection,
            batched_decoder_boundaries,
            self.sequence_rows,
            aux_hidden_layers,
            self.aux_capture,
            self.state_row,
            self.attention_dp,
        ) = (
            _p_get("gate_first_kda_projection"),
            _p_get("batched_decoder_boundaries"),
            _p_get("sequence_rows", 1),
            _p_get("aux_hidden_layers", ()),
            _p_get("aux_capture", "layer"),
            _p_get("state_row"),
            _p_get("attention_dp", 1),
        )
        self.attention_projection_schedule = _p_get(
            "attention_projection_schedule", "default"
        )
        (
            self.mla_commit_tile_size,
            self.sequence_parallel_size,
            self.block_tables,
            self.dcp_pool_map,
            self.dcp_owner_table,
            self.state_slots,
            self.live_rows,
        ) = (
            _p_get("mla_commit_tile_size"),
            _p_get("sequence_parallel_size", 4),
            _p_get("block_tables"),
            _p_get("dcp_pool_map"),
            _p_get("dcp_owner_table"),
            _p_get("state_slots"),
            _p_get("live_rows"),
        )
        if gate_first_kda_projection is False:
            _err_msg = "The clean decoder always uses the split KDA gate projection"
            raise ValueError(_err_msg)
        self.gate_first_kda_projection = True
        if batched_decoder_boundaries is None:
            batched_decoder_boundaries = (
                self.batch_size in SCATTER_BATCHES or self.attention_dp > 1
            )
        self.batched_decoder_boundaries = batched_decoder_boundaries
        if not 1 <= self.batch_size <= MAX_BATCH:
            _err_msg = f"The native decoder supports batch sizes from 1 to {MAX_BATCH}"
            raise ValueError(_err_msg)
        self.aux_hidden_layers = tuple(aux_hidden_layers)
        self.rank = jax.lax.axis_index("tp")
        self.vocabulary = self.weights["embedding"].shape[0]

    def _prepare_embedding(self) -> _Dynamic:
        """Prepare embedding matrix for decoder."""
        if self.batched_decoder_boundaries:
            local_ids = self.tokens - self.rank * self.vocabulary
            local_embeddings = self.weights["embedding"][
                jnp.clip(local_ids, 0, self.vocabulary - 1)
            ]
            return kimi.tp_sum(
                jnp.where(
                    ((local_ids >= 0) & (local_ids < self.vocabulary))[:, None],
                    local_embeddings.astype(jnp.float32),
                    0,
                )
            ).astype(jnp.bfloat16)

        def prepare_first_layer_row(_: _Dynamic, token: _Dynamic) -> _Dynamic:
            """Execute prepare first layer row logic."""
            local_id = token - self.rank * self.vocabulary
            embedding_row = self.weights["embedding"][
                jnp.clip(local_id, 0, self.vocabulary - 1)
            ][None]
            embedding_row = kimi.tp_sum(
                jnp.where(
                    (local_id >= 0) & (local_id < self.vocabulary),
                    embedding_row.astype(jnp.float32),
                    0,
                )
            ).astype(jnp.bfloat16)
            return (), embedding_row[0]

        _, embedding = jax.lax.scan(prepare_first_layer_row, (), self.tokens)
        return embedding

    def _project_logits(self, prefix: _Dynamic, blocks: _Dynamic) -> _Dynamic:
        """Project final hidden states to logits."""
        if self.sequence_parallel_pre_attention and self.batched_decoder_boundaries:
            return jnp.dot(
                prefix, self.weights["lm_head"], preferred_element_type=jnp.float32
            )
        if self.sequence_parallel_pre_attention:

            def project_decoder_row(_: _Dynamic, hidden_row: _Dynamic) -> _Dynamic:
                """Execute project decoder row logic."""
                logits_row = jnp.dot(
                    hidden_row[None],
                    self.weights["lm_head"],
                    preferred_element_type=jnp.float32,
                )
                return (), logits_row[0]

            _, logits = jax.lax.scan(project_decoder_row, (), prefix)
            return logits

        active_block_count = (self.layers - 1) // 12 + 1

        def finish_decoder(_: _Dynamic, inputs: _Dynamic) -> _Dynamic:
            """Execute finish decoder logic."""
            prefix_row, block_rows = inputs
            hidden_row = kimi.attention_residual(
                prefix_row,
                block_rows[:active_block_count],
                self.weights["output_res"][0],
                eps=self.eps,
            )[None]
            hidden_row = kimi.rms(hidden_row, self.weights["final_norm"], self.eps)
            logits_row = jnp.dot(
                hidden_row,
                self.weights["lm_head"],
                preferred_element_type=jnp.float32,
            )
            return (), logits_row[0]

        _, logits = jax.lax.scan(finish_decoder, (), (prefix, blocks))
        return logits

    def run(self) -> _Dynamic:
        """Run decode local execution."""
        embedding = self._prepare_embedding()
        if self.state_row is not None and self.sequence_rows != self.batch_size:
            _err_msg = "state_row needs one sequence per tile"
            raise ValueError(_err_msg)
        prefix = embedding
        convolution_states, recurrent_states = self.states[:2]
        cache_states = tuple(self.states[2:])
        (
            prefix,
            blocks,
            convolution_states,
            recurrent_states,
            new_keys,
            new_values,
            aux_hidden,
        ) = stack(
            prefix,
            embedding,
            self.weights,
            (
                convolution_states,
                recurrent_states,
                *cache_states,
            ),
            self.positions,
            batch_size=self.batch_size,
            layers=self.layers,
            eps=self.eps,
            residual_reduction_order=self.residual_reduction_order,
            mla_cache_batch_tile_size=self.mla_cache_batch_tile_size,
            alias_expert_scales=self.alias_expert_scales,
            attention_token_scatter=self.attention_token_scatter,
            moe_token_scatter_output=self.moe_token_scatter_output,
            moe_routed_reduce_scatter=self.moe_routed_reduce_scatter,
            sequence_parallel_pre_attention=self.sequence_parallel_pre_attention,
            gate_first_kda_projection=self.gate_first_kda_projection,
            sequence_rows=self.sequence_rows,
            aux_hidden_layers=self.aux_hidden_layers,
            aux_capture=self.aux_capture,
            include_layer_zero=True,
            state_row=self.state_row,
            attention_dp=self.attention_dp,
            attention_projection_schedule=self.attention_projection_schedule,
            mla_commit_tile_size=self.mla_commit_tile_size,
            sequence_parallel_size=self.sequence_parallel_size,
            block_tables=self.block_tables,
            dcp_pool_map=self.dcp_pool_map,
            dcp_owner_table=self.dcp_owner_table,
            state_slots=self.state_slots,
            live_rows=self.live_rows,
        )
        logits = self._project_logits(prefix, blocks)
        new_states = (
            convolution_states,
            recurrent_states,
            new_keys,
            new_values,
        )
        if self.aux_hidden_layers:
            return logits[None], aux_hidden, new_states
        return logits[None], new_states


def decode_local(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Decode one native batch tile with local TP32 weights and caches."""
    return _DecodeLocalRunner(*_p_args, **_p_kw).run()
