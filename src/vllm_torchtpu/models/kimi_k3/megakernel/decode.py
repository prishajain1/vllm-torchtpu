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

"""Kimi-K3 decode: the decode role's program over vLLM's unified pools.

:class:`KimiDecoder` runs on one TP32 kernel mesh (all devices in ascending id;
a worker's kernel rank is its mesh position, not its TP rank) over
:class:`DecodeOps` and, when speculating, a :class:`DsparkDraft`. MLA is decode
context parallel, reading and committing latent pages in place where the KV
connector landed them; KDA state lives in private slots (``pools.SlotMap``),
imported with the prefill's DSpark ring on a request's first step. Every op
reads and returns a replicated control vector (:class:`Control`) holding each
continuing request's next input, so nothing is uploaded per step while the
host's tokens lag (async scheduling). A step is one of:

* one request, seven drafts: :meth:`DecodeOps.verify_packed`, then
  :meth:`DsparkDraft.step_single`;
* three drafts alone, or several requests: :meth:`DecodeOps.verify_batched`,
  then :meth:`DsparkDraft.step_batched` (each request keeps its own ring);
* otherwise a plain step (:meth:`DecodeOps.plain_step`), also whenever a
  structured-output request decodes (grammar masks: the verify ops do not
  mask, so the whole batch steps plain, without drafts).

The verify ops decide acceptance and commit only the selected KDA post-state.
Under attention DP a request's pages and KDA state live on the DP group its MLA
block ids name (``megakernel.dp_blocks``), and it takes one of that group's
batch rows (``_group_rows``); a speculating request takes one of its group's
half of the batched verify's requests (a single request too).
"""

from __future__ import annotations

import operator
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np
import torch
from jax.sharding import PartitionSpec as P
from torch_tpu._internal.sync import synchronize

from vllm_torchtpu.logger import init_logger

from .decode_ops import (
    GREEDY,
    Control,
    DecodeOps,
    dp_row_owners,
)
from .pools import (
    KDA_LAYERS,
    KdaImporter,
    SlotMap,
    draft_raw_pools,
    kda_blocks_at_length,
    tp_rank_permutation,
)
from .program import (
    DP_SPEC_BATCHES,
    SPEC_BATCHES,
    VOCAB,
    kernel_modules,
    spec_tokens_per_batch,
    supports_spec_tokens,
)
from .runtime_shim.bridge import hbm, materialize, register_global_op
from .runtime_shim.compilation import compile_ops
from .runtime_shim.config import MegakernelEngineConfig
from .runtime_shim.export_cache import prepare_distributed
from .runtime_shim.program import (
    KVGroup,
    RequestInput,
    StepInput,
    StepOutput,
)

_ = (Iterable, Mapping, Sequence, MegakernelEngineConfig)

_Dynamic = Any
_CONST_2 = 2
_CONST_3 = 3


logger = init_logger(__name__)


def _compute_spec_shapes(
    spec_tokens: Sequence[int],
    groups: int,
    max_num_seqs: int,
    batch_sizes: Sequence[int],
) -> tuple[tuple[int, int], ...]:
    """Compute compiled speculation shapes for the given schedule."""
    shapes = set()
    for b, k in enumerate(spec_tokens):
        if k and groups > 1:
            fuller = (-(-b // 2), min(b, max_num_seqs // groups))
            for compiled in DP_SPEC_BATCHES:
                if (
                    2 * fuller[0] <= compiled
                    and compiled // 4 < fuller[1]
                    and compiled <= max(batch_sizes)
                    and supports_spec_tokens(compiled, k, groups)
                ):
                    shapes.add((compiled, k + 1))
        elif k and (b >= _CONST_2 or k == _CONST_3):
            compiled = next((s for s in SPEC_BATCHES if s >= b), 0)
            if not compiled or compiled > max(batch_sizes):
                _err_msg = (
                    f"K3 decode: no compiled speculation batch holds {b} requests "
                    f"(batches {SPEC_BATCHES}, up to {max(batch_sizes)})"
                )
                raise ValueError(_err_msg)
            shapes.add((compiled, k + 1))
    return tuple(sorted(shapes))


class KimiDecoder:
    """The decode role's :class:`MegakernelProgram` over ``ops`` and, when.

    ``config`` speculates, the DSpark ``draft`` (None: plain steps only). Slot 0
    is the null slot padding rows use; slots below ``sub_base`` are the regular
    ones (live requests plus those whose release is in flight). ``spec_shapes``
    holds the compiled (padded request count, verify rows per request) pairs.
    """

    def __getattr__(self, name: _Dynamic) -> _Dynamic:
        """Allow dynamic attribute access."""
        raise AttributeError(name)

    WIDTH = DecodeOps.VERIFY_ROWS  # tokens per row of a step's result

    def __init__(
        self,
        config: MegakernelEngineConfig,
        ops: DecodeOps,
        draft: _Dynamic = None,
        *,
        tp_rank: int,
    ) -> None:
        """Execute init logic."""
        self.config, self.ops, self.draft = config, ops, draft
        self.tp_rank, self.rank = tp_rank, ops.rank
        self.sub_base = 2 * config.max_num_seqs + 1
        needed = Control.total_slots(self.sub_base) if draft else self.sub_base
        # Under attention DP each group's share of the banks follows the ops'
        # slot count (pools.group_slots), so it must be this decoder's.
        if ops.num_slots < needed or (ops.attention_dp > 1 and ops.num_slots != needed):
            _err_msg = (
                f"K3 decode: the ops hold {ops.num_slots} KDA slots; "
                f"--max-num-seqs {config.max_num_seqs} needs {needed}"
            )
            raise ValueError(_err_msg)
        self.slot_map = SlotMap(self.sub_base, int(ops.attention_dp))
        self.layout: Control | None = None
        self.control: torch.Tensor | None = None
        self.device_plan: np.ndarray | None = None
        self.tracked: set[str] = set()  # requests whose next input the control holds
        self.spec_req: str | None = None
        self.draft_state = None
        self.next_ids: torch.Tensor | None = None
        # Rings the prefill computed, per request, for its first speculation.
        self._offered: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        # With a draft, the schedule (vLLM's per-batch-size draft count,
        # validated in kimi_k3.program) says which batch sizes speculate; every
        # compiled verifier/drafter shape is derived from it.
        self.spec_tokens = (spec_tokens_per_batch(config) or ()) if draft else ()
        self.groups = int(ops.attention_dp)
        shapes = _compute_spec_shapes(
            self.spec_tokens, self.groups, config.max_num_seqs, ops.batch_sizes
        )
        self.spec_batches = tuple(sorted({b for b, _ in shapes}))
        # One sequence's 1x8 verify cannot span the DP groups.
        self.spec_alone = (
            self.groups == 1
            and len(self.spec_tokens) > 1
            and self.spec_tokens[1] == self.WIDTH - 1
        )
        # Batched speculation: each request's draft state between its steps,
        # and the verify rows its draft vector's drafts were made for (only
        # while they still follow its next token).
        self._drafts: dict[str, _Dynamic] = {}
        self._fresh: dict[str, int] = {}
        self._mode_rows: dict[tuple[int, ...], torch.Tensor] = {}
        self._pads: list[_Dynamic] = []  # draft states of padding groups
        self._compiled_layout: tuple | None = None
        self._zero: torch.Tensor | None = None
        self._modes: dict[int, torch.Tensor] = {}
        self._live_batches: set[int] = set()
        self.stats = {"plans": 0, "enters": 0, "chained": 0, "plain": 0, "batched": 0}
        self._group: dict[str, tuple[int, int]] = {}  # request -> (DP group, context)
        # Import each transferred request's DSpark ring: with a DSpark
        # speculative config the transfer carries one (backend.draft_ring_rows).
        # (The MLA draft keeps its context in its own pools: no ring.)
        self._ring = draft is not None and getattr(draft, "imports_ring", bool(1))
        self._ring_reports = 4
        self._importer: KdaImporter | None = None
        self.destination_ranks: tuple[int, ...] | None = None
        self._saw_request = False
        self._pages: int | None = None
        self._num_blocks = 0
        self._mla_group = 0
        self._kda_groups: list[tuple[int, int]] = []

    def load_weights(self) -> None:
        """Execute load weights logic."""
        started = time.perf_counter()
        self.ops.load_weights()
        if self.draft is not None and not self.draft.weights:
            self.draft.load_weights()
        if self.rank == 0:
            logger.info(
                "K3 decode: weights resident in %.1f s",
                time.perf_counter() - started,
            )

    def bind_kv_caches(
        self, caches: Mapping[str, torch.Tensor], groups: Sequence[KVGroup]
    ) -> None:
        """Decode from vLLM's unified pools; ``groups`` say which block-table.

        row the decoder reads (the MLA layers') and where each KDA layer's
        state blocks are (for a request's import).
        """
        self.load_weights()
        # A device's KV transfer ordinal is its mesh position: the connector
        # lists workers in mesh order (``destination_ranks``: TP rank per position).
        perm = tp_rank_permutation(self.tp_rank, self.rank)
        self.destination_ranks = (
            None if perm is None else tuple(q for _, q in sorted(perm))
        )
        # One importer per ops (it registers torch ops, whose names are
        # process-global), shared by the decoders over them.
        self._importer = self.ops.cached_op(
            ("kda_importer", None),
            lambda: KdaImporter(
                self.ops.mesh, None, self.ops.attention_dp, self.ops.num_slots
            ),
        )
        num_blocks = self._importer.bind(caches, self.config.kv_cache_dtype)
        self._num_blocks = num_blocks
        self.ops.use_unified_pools(self._importer.raw_pools, self._importer.mla_raw)
        if self.draft is not None and hasattr(self.draft, "bind_pools"):
            if not (self.ops.pool_map is not None):
                _err_msg = "Assertion failed: self.ops.pool_map is not None"
                raise AssertionError(_err_msg)
            # The MLA draft: the target's LM head and embedding, and pools
            # shaped like the target's MLA pools (same tables and owners).
            self.draft.attach_target(self.ops.weights)
            self.draft.bind_pools(
                self.ops.unified_pools[self.ops.pool_map[0]],
                self.ops.owner_table,
                draft_raw_pools(caches),
            )
        of = {name: g for g, group in enumerate(groups) for name in group.layer_names}
        mla_groups = {of[n] for n in self._importer.mla_layer_names if n in of}
        if len(mla_groups) != 1:
            _err_msg = (
                f"K3 decode: the MLA layers are in KV cache groups {sorted(mla_groups)}"
            )
            raise RuntimeError(_err_msg)
        self._mla_group = mla_groups.pop()
        missing = [n for n in self._importer.kda_layer_names if n not in of]
        if missing:
            _err_msg = (
                f"K3 decode: KDA layers {missing[:3]}... are in no KV cache group"
            )
            raise RuntimeError(_err_msg)
        self._kda_groups = [
            (of[n], int(groups[of[n]].block_size))
            for n in self._importer.kda_layer_names
        ]
        self._pages = int(groups[self._mla_group].table_width)
        if self.rank == 0:
            logger.info(
                "K3 decode: DCP decoder on vLLM's %d unified pools (%d blocks); HBM %s",
                len(self._importer.raw_pools),
                num_blocks,
                hbm(self.rank),
            )

    def warmup(self) -> None:
        """Execute warmup logic."""
        if self._pages is None:
            _err_msg = "K3 decode: bind the KV caches before warming up"
            raise RuntimeError(_err_msg)
        started = time.perf_counter()
        self._warm_importer()
        self._warm_steps()
        self._warm_importer()  # an import after the steps (another operand history)
        self._warm_steps()
        if self.rank == 0:
            logger.info("K3 decode: ops warm in %.1f s", time.perf_counter() - started)

    def step(self, step_input: StepInput) -> StepOutput:
        """One decode step (see the module docstring)."""
        if self._pages is None:
            _err_msg = "K3 decode: a request arrived before the KV pools were bound"
            raise RuntimeError(_err_msg)
        if step_input.copies:
            _err_msg = (
                "K3 decode: copy-on-write block copies (prefix hits inside a block)"
            )
            raise NotImplementedError(_err_msg)
        mask = step_input.mask
        if mask is not None and mask.shape != (len(step_input.requests), VOCAB // 32):
            _err_msg = (
                f"K3 decode: grammar masks {mask.shape} do not cover "
                f"{len(step_input.requests)} requests of the {VOCAB}-token "
                f"vocabulary"
            )
            raise ValueError(_err_msg)
        self.forget(step_input.finished)
        return StepOutput(self._decode(step_input.requests, mask))

    def _admit(self, request: RequestInput, slot: int) -> None:
        """A request's first step: import its KDA state after the tokens.

        computed before this step into ``slot``, and its DSpark ring. A
        request with no state block to import (none computed, or only the
        null block) starts from a clean state.
        """
        # Each KDA layer's state after the computed tokens: the block holding
        # position computed - 1, where the transfer committed it. (When the
        # computed tokens end on a block boundary the request already holds
        # the next block too; vLLM's runners would copy the state into it.)
        kda = kda_blocks_at_length(
            request.block_tables, self._kda_groups, request.computed
        )
        if self.groups > 1 and int(kda.max()) >= self._num_blocks:
            _err_msg = (
                f"K3 decode: KDA state blocks {kda.tolist()} are not shared blocks"
            )
            raise RuntimeError(_err_msg)
        # The decoder has no position-0 reset. Under attention DP the importer
        # clears the slot (from the null blocks) on the request's group only.
        if not kda.any() and self.groups == 1:
            for state in self.ops.slots:
                state[0, slot].zero_()
            return
        if not (self._importer is not None):
            _err_msg = "Assertion failed: self._importer is not None"
            raise AssertionError(_err_msg)
        started = time.perf_counter()
        kda_blocks = torch.from_numpy(kda).to("tpu")
        conv_slot, recurrent_slot = self.ops.slots[0], self.ops.slots[1]
        conv, recurrent = self._importer.import_state(
            conv_slot, recurrent_slot, kda_blocks, slot
        )
        self.ops.slots = [conv, recurrent]
        if self._ring:
            # The DSpark ring the prefill wrote beside the state (cold unless
            # it wrote one for exactly these tokens): the request's first
            # speculation starts from it instead of a cold ring.
            cache, positions = self._importer.import_ring(
                self.draft.config, kda_blocks, request.computed
            )
            self._offered[request.req_id] = (cache, positions)
            if self._ring_reports:  # the first few imports only (a host read each)
                self._ring_reports -= 1
                held = int((positions >= 0).sum().cpu())
                if self.rank == 0:
                    logger.info(
                        "K3 decode: request %s's draft ring holds %d positions",
                        request.req_id,
                        held,
                    )
        if self.rank == 0:
            logger.info(
                "K3 decode: imported request %s (%d tokens) into slot %d in %.1f ms",
                request.req_id,
                request.computed,
                slot,
                1e3 * (time.perf_counter() - started),
            )
            if not self._saw_request:
                self._saw_request = True
                logger.info("K3 decode: first real request; HBM %s", hbm(self.rank))

    def _null_ring(self) -> tuple[torch.Tensor, torch.Tensor]:
        """An imported ring from the null block (for warming up)."""
        if not (self.draft is not None and self._importer is not None):
            _err_msg = (
                "Assertion failed: self.draft is not None and self._importer "
                "is not None"
            )
            raise AssertionError(_err_msg)
        return self._importer.import_ring(
            self.draft.config, torch.zeros(KDA_LAYERS, dtype=torch.int32).to("tpu"), 0
        )

    def _warm_importer(self) -> None:
        """Compile the KDA import on the null slot and block."""
        if not (self._importer is not None):
            _err_msg = "Assertion failed: self._importer is not None"
            raise AssertionError(_err_msg)
        started = time.perf_counter()
        conv_slot, recurrent_slot = self.ops.slots[0], self.ops.slots[1]
        conv, recurrent = self._importer.import_state(
            conv_slot,
            recurrent_slot,
            torch.zeros(KDA_LAYERS, dtype=torch.int32).to("tpu"),
            0,
        )
        self.ops.slots = [conv, recurrent]
        materialize(recurrent)
        if self.rank == 0:
            logger.info(
                "K3 decode: warmed the KDA import in %.1f s",
                time.perf_counter() - started,
            )

    # ------------------------------------------------------------ state and helper ops
    def _prepare(self, pages: int) -> Control:
        """Execute prepare logic."""
        if self.layout is None or self.layout.pages != pages:
            self.layout = Control(self.sub_base, max(self.ops.batch_sizes), pages)
            self.control = None
        if self.control is None:
            self.control = torch.zeros(self.layout.size, dtype=torch.int32).to("tpu")
            self.device_plan = None
        if self._zero is None:
            self._zero = torch.zeros(1, dtype=torch.int32).to("tpu")
            self._modes = {
                0: torch.zeros(1, dtype=torch.int32).to("tpu"),
                1: torch.ones(1, dtype=torch.int32).to("tpu"),
            }
        return self.layout

    def _splice(self) -> _Dynamic:
        """``(control, plan) -> control`` with the plan part replaced."""
        layout = self.layout
        if not (layout is not None):
            _err_msg = "Assertion failed: layout is not None"
            raise AssertionError(_err_msg)
        key = ("splice", layout.key())

        def build() -> _Dynamic:
            """Execute build logic."""

            def fn(control: _Dynamic, plan: _Dynamic) -> _Dynamic:
                """Execute fn logic."""
                return (jnp.concatenate((control[: layout.plan_start], plan)),)

            return register_global_op(
                "pallas::k3_decode_control_splice",
                self.ops.mesh,
                fn,
                (P(), P()),
                (P(),),
                export_key=repr(key),
            )

        return self.ops.cached_op(key, build)

    def _enter(self) -> _Dynamic:
        """``(control, vector, keep) -> ids``: the verify operand of plan row.

        0's request from the control: its next token and after it placeholder
        drafts (``keep`` 0) or the last draft vector's drafts (``keep`` 1:
        the same request, a new plan), at its next positions; every row its
        slot; its state row; the plan's table.
        """
        layout, sub_base = self.layout, self.sub_base
        if not (layout is not None):
            _err_msg = "Assertion failed: layout is not None"
            raise AssertionError(_err_msg)
        key = ("enter", layout.key(), sub_base)

        def build() -> _Dynamic:
            """Execute build logic."""
            rows = self.WIDTH
            (dspark,) = kernel_modules("dspark_common")

            def fn(control: _Dynamic, vector: _Dynamic, keep: _Dynamic) -> _Dynamic:
                """Execute fn logic."""
                slot, token, position, row, _, tables = layout.inputs(control, 1)
                drafts = vector[0, dspark.OUT_DRAFTS : dspark.OUT_DRAFTS + rows - 1]
                tokens = jnp.where(
                    keep[0] != 0,
                    jnp.concatenate((token, drafts)),
                    jnp.full((rows,), token[0]),
                )
                sub = jnp.concatenate(
                    (
                        slot,
                        layout.state_slot(slot, jnp.arange(1, rows, dtype=jnp.int32)),
                    )
                )
                ids = jnp.concatenate(
                    (
                        tokens,
                        position[0] + jnp.arange(rows, dtype=jnp.int32),
                        sub,
                        row,
                        jnp.tile(tables[0], rows),
                    )
                )
                return (ids.astype(jnp.int32),)

            return register_global_op(
                "pallas::k3_decode_dspark_enter",
                self.ops.mesh,
                fn,
                (P(), P(), P()),
                (P(),),
                export_key=repr(key),
            )

        return self.ops.cached_op(key, build)

    def _enter_batched(self, requests: int, rows_per: int) -> _Dynamic:
        """``(control, vector_0.., modes) -> ids``: the batched verify operand.

        (``DecodeOps.verify_batched``) of plan rows 0..requests-1: per
        request its next token, then the first drafts of its draft vector
        (``modes`` 1) or placeholders (0), at its next positions; every row
        its request's slot (``indices``: each row's logical entry); one table
        per request.
        """
        layout, k = self.layout, rows_per
        if not (layout is not None):
            _err_msg = "Assertion failed: layout is not None"
            raise AssertionError(_err_msg)
        key = ("enter_batched", layout.key(), requests, k)

        def build() -> _Dynamic:
            """Execute build logic."""
            (dspark,) = kernel_modules("dspark_common")
            r = requests

            def fn(control: _Dynamic, *rest: _Dynamic) -> _Dynamic:
                """Execute fn logic."""
                vectors, modes = rest[:r], rest[r]
                slot, token, position, row, _, tables = layout.inputs(control, r)
                others = jnp.arange(1, layout.BANK + 1, dtype=jnp.int32)
                tokens, positions, slots, indices = [], [], [], []
                for j in range(r):
                    drafts = vectors[j][
                        0, dspark.OUT_DRAFTS : dspark.OUT_DRAFTS + k - 1
                    ]
                    tokens.append(
                        jnp.where(
                            modes[j] != 0,
                            jnp.concatenate((token[j : j + 1], drafts)),
                            jnp.full((k,), token[j]),
                        )
                    )
                    positions.append(position[j] + jnp.arange(k, dtype=jnp.int32))
                    rest_entries = jnp.where(others <= row[j], others - 1, others)
                    entry = jnp.concatenate((row[j : j + 1], rest_entries[: k - 1]))
                    indices.append(entry)
                    slots.append(layout.state_slot(slot[j], entry))
                ids = jnp.concatenate(
                    (*tokens, *positions, *slots, *indices, tables.reshape(-1))
                )
                return (ids.astype(jnp.int32),)

            return register_global_op(
                f"pallas::k3_decode_dspark_enter{requests}x{k}",
                self.ops.mesh,
                fn,
                (P(),) * (r + 2),
                (P(),),
                export_key=repr(key),
            )

        return self.ops.cached_op(key, build)

    def _spec_batch(self, requests: int, rows_per: int, groups: list[int]) -> int:
        """The compiled batched-speculation size holding ``requests``.

        decoding requests of ``rows_per`` verify rows (padded with null ones;
        under attention DP each of ``groups``' requests in its group's half),
        or 0: none.
        """
        if self.groups > 1:
            fuller = max(groups.count(g) for g in range(self.groups))
            return next(
                (
                    b
                    for b, rows in self.spec_shapes
                    if rows == rows_per and b // self.groups >= fuller
                ),
                0,
            )
        if requests < 1 or (requests == 1 and 1 not in self.spec_batches):
            return 0
        return next((b for b in self.spec_batches if b >= requests), 0)

    def _modes_for(self, modes: tuple[int, ...]) -> torch.Tensor:
        """Execute modes for logic."""
        if modes not in self._mode_rows:
            self._mode_rows[modes] = torch.tensor(modes, dtype=torch.int32).to("tpu")
        return self._mode_rows[modes]

    def _take_state(self, req_id: str) -> _Dynamic:
        """``req_id``'s draft state: kept from its last speculation, else the.

        ring the prefill computed, else a cold one.
        """
        if not (self.draft is not None):
            _err_msg = "Assertion failed: self.draft is not None"
            raise AssertionError(_err_msg)
        state = self._drafts.pop(req_id, None)
        if state is not None:
            return state
        offered = self._offered.pop(req_id, None)
        if offered is not None:
            return self.draft.from_ring(*offered, self._zero)
        return self.draft.new_state(self._zero)

    def _park_single(self) -> None:
        """The single-request speculation's state back with its request."""
        if self.spec_req is not None and self.draft_state is not None:
            self._drafts[self.spec_req] = self.draft_state
        self.spec_req, self.draft_state, self.next_ids = None, None, None

    def _sync_plan(self, entries: _Dynamic, tables: _Dynamic) -> bool:
        """Upload the plan for ``entries`` unless the device holds it; returns.

        whether it uploaded (the verify operand's table may have changed).
        """
        layout = self.layout
        if not (layout is not None):
            _err_msg = "Assertion failed: layout is not None"
            raise AssertionError(_err_msg)
        plan = layout.plan(entries, tables)
        if self.device_plan is not None and np.array_equal(plan, self.device_plan):
            return False
        (self.control,) = self._splice()(self.control, torch.from_numpy(plan).to("tpu"))
        self.stats["plans"] += 1
        self.device_plan = layout.consumed(plan, len(entries))
        return True

    def _replay(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Tokens computed on this role (a recompute after a preemption):.

        one batch-one plain step per token (on a row of its DP ``group``),
        their outputs discarded.
        """
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            slot,
            tokens,
            start,
            table,
            group,
        ) = (
            _p_next(_p_iter, _p_get("slot")),
            _p_next(_p_iter, _p_get("tokens")),
            _p_next(_p_iter, _p_get("start")),
            _p_next(_p_iter, _p_get("table")),
            _p_next(_p_iter, _p_get("group", 0)),
        )
        batch = self.ops.batch_for(1)
        for offset, token in enumerate(tokens):
            entries = [(slot, bool(1), int(token), start + offset, GREEDY)]
            if self.groups > 1:
                _, entries, tables, _ = self._place(
                    [group], entries, [table], len(table)
                )
                self._sync_plan(entries, tables)
            else:
                self._sync_plan(entries, [table])
            _, self.control = self.ops.plain_step(
                self.control, batch, self.layout, self.sub_base, bool(0)
            )

    def forget(self, finished: Iterable[str]) -> None:
        """Execute forget logic."""
        finished = set(finished or ())
        if not finished:
            return
        self.slot_map.release(finished)
        self.tracked -= finished
        for req in finished:
            self._group.pop(req, None)
            self._offered.pop(req, None)
            self._drafts.pop(req, None)
            self._fresh.pop(req, None)
        if self.spec_req in finished:
            self.spec_req, self.draft_state, self.next_ids = None, None, None

    # ------------------------------------------------------------ attention DP
    def _mla_table(self, request: RequestInput) -> _Dynamic:
        """``request``'s MLA block table as pool rows. Under attention DP.

        the scheduler's ids are ``g N + r``: DP group ``g``'s row ``r``
        (``megakernel.dp_blocks``); the request keeps that group.
        """
        table = request.block_tables[self._mla_group]
        if self.groups > 1:
            group = int(table.max()) // self._num_blocks
            self._group[request.req_id] = (group, request.end)
            table = table % self._num_blocks
        return table

    def _group_of(self, request: RequestInput) -> int:
        """``request``'s DP group: the scheduler's, else chosen on its.

        first step: the group with the fewest requests, then the least
        context.
        """
        if request.req_id not in self._group:
            count = [0] * self.groups
            context = [0] * self.groups
            for group, length in self._group.values():
                count[group] += 1
                context[group] += length
            group = min(range(self.groups), key=lambda g: (count[g], context[g]))
            self._group[request.req_id] = (group, request.end)
        return self._group[request.req_id][0]

    def _group_rows(self, batch: int, group: int) -> list[int]:
        """The batch rows DP group ``group`` attends: ``p * B / O + group * r + j``.

        (``O`` local row owners, ``dp_row_owners``; ``r = B / (O dp)``).
        """
        owners = dp_row_owners(batch)
        per = batch // self.groups // owners
        return [
            p * (batch // owners) + group * per + j
            for p in range(owners)
            for j in range(per)
        ]

    def _place(
        self, groups: list[int], entries: _Dynamic, tables: _Dynamic, pages: int
    ) -> _Dynamic:
        """Plan rows for requests of DP ``groups`` (in order): ``(batch,.

        entries, tables, plan row of each request)``, the other rows null.
        """
        load = [groups.count(g) for g in range(self.groups)]
        batch = self.ops.batch_for(self.groups * max(load))
        free = [self._group_rows(batch, g) for g in range(self.groups)]
        null_entry, null_table = (0, bool(1), 0, 0, GREEDY), np.zeros(pages, np.int32)
        placed_entries, placed_tables = [null_entry] * batch, [null_table] * batch
        rows = []
        for group, entry, table in zip(groups, entries, tables, strict=True):
            row = free[group].pop(0)
            placed_entries[row], placed_tables[row] = entry, table
            rows.append(row)
        return batch, placed_entries, placed_tables, rows

    # ------------------------------------------------------------ the step
    def _prepare_requests_for_decode(
        self, requests: Sequence[RequestInput]
    ) -> tuple[list, list, list]:
        """Extract entries, tables, and decoding requests from input requests."""
        entries, tables, decoding = [], [], []
        for i, request in enumerate(requests):
            if request.scheduled <= 0:
                continue
            table = self._mla_table(request)
            group = self._group_of(request) if self.groups > 1 else 0
            if self.slot_map.group(request.req_id) not in (None, group):
                self.slot_map.release([request.req_id])
            fresh = request.req_id not in self.slot_map
            slot = self.slot_map.allocate(request.req_id, group)
            if fresh:
                self.tracked.discard(request.req_id)
                self._admit(request, slot)
            sampling = request.sampling.as_tuple()
            if request.req_id in self.tracked:
                entries.append((slot, bool(0), 0, 0, sampling))
            else:
                known = min(request.end, request.num_tokens)
                tokens = request.token_ids[request.computed : known]
                replay = len(tokens) - (0 if request.partial else 1)
                if replay:
                    self._replay(
                        slot,
                        tokens[:replay],
                        request.computed,
                        table,
                        self._group_of(request) if self.groups > 1 else 0,
                    )
                if request.partial:
                    continue
                entries.append((slot, bool(1), int(tokens[-1]), known - 1, sampling))
                self.tracked.add(request.req_id)
            tables.append(table)
            decoding.append((i, request))
        return entries, tables, decoding

    def _execute_plain_decode_step(
        self, *_p_args: _Dynamic, **_p_kw: _Dynamic
    ) -> tuple[torch.Tensor, list | None]:
        """Execute non-speculative plain decode step."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (decoding, entries, tables, sampled, mask, pages, groups) = (
            _p_next(_p_iter, _p_get("decoding")),
            _p_next(_p_iter, _p_get("entries")),
            _p_next(_p_iter, _p_get("tables")),
            _p_next(_p_iter, _p_get("sampled")),
            _p_next(_p_iter, _p_get("mask")),
            _p_next(_p_iter, _p_get("pages")),
            _p_next(_p_iter, _p_get("groups")),
        )
        self._park_single()
        for _, request in decoding:
            self._fresh.pop(request.req_id, None)
        placed = None
        if self.groups > 1:
            batch, entries, tables, placed = self._place(groups, entries, tables, pages)
        else:
            batch = self.ops.batch_for(len(decoding))
        if batch not in self._live_batches:
            self._live_batches.add(batch)
            if self.rank == 0:
                logger.info(
                    "K3 decode: first live step at batch size %d (%d rows)",
                    batch,
                    len(decoding),
                )
        self._sync_plan(entries, tables)
        self.stats["plain"] += 1
        mask_operand = None
        if mask is not None:
            mask_operand = np.full((batch, VOCAB // 32), -1, np.int32)
            rows = placed if placed is not None else range(len(decoding))
            for plan_row, (i, _) in zip(rows, decoding, strict=True):
                mask_operand[plan_row] = mask[i]
            mask_operand = torch.from_numpy(mask_operand).to("tpu")
        emitted, self.control = self.ops.plain_step(
            self.control, batch, self.layout, self.sub_base, sampled, mask_operand
        )
        return emitted, placed

    def _execute_decode_step(
        self, *_p_args: _Dynamic, **_p_kw: _Dynamic
    ) -> tuple[torch.Tensor | None, list | None]:
        """Dispatch either single-speculate, batched-speculate, or plain step."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (decoding, entries, tables, sampled, mask, pages) = (
            _p_next(_p_iter, _p_get("decoding")),
            _p_next(_p_iter, _p_get("entries")),
            _p_next(_p_iter, _p_get("tables")),
            _p_next(_p_iter, _p_get("sampled")),
            _p_next(_p_iter, _p_get("mask")),
            _p_next(_p_iter, _p_get("pages")),
        )
        if not decoding:
            return None, None
        groups = [self._group_of(r) for _, r in decoding] if self.groups > 1 else []
        k = (
            self.spec_tokens[len(decoding)]
            if len(decoding) < len(self.spec_tokens)
            else 0
        )
        if mask is not None:
            k = 0
        compiled = self._spec_batch(len(decoding), k + 1, groups) if k else 0
        if self.spec_alone and len(decoding) == 1 and k == self.WIDTH - 1:
            return (
                self._speculate(decoding[0][1], entries, tables, _sampled=sampled),
                None,
            )
        if compiled and (
            len(decoding) == 1 or any(r.scheduled > 1 for _, r in decoding)
        ):
            return self._speculate_batched(
                decoding, entries, tables, sampled, compiled, groups
            )
        return self._execute_plain_decode_step(
            decoding, entries, tables, sampled, mask, pages, groups
        )

    def _pack_decode_tokens(
        self,
        requests: Sequence[RequestInput],
        decoding: list,
        emitted: torch.Tensor,
        placed: list | None,
    ) -> torch.Tensor:
        """Map emitted step tokens back to the caller's request ordering."""
        tokens = torch.full((len(requests), 8), -1, dtype=torch.int32)
        if self.groups > 1 and placed is not None:
            for (i, _), plan_row in zip(decoding, placed, strict=True):
                tokens[i] = emitted[plan_row]
        else:
            for row, (i, _) in enumerate(decoding):
                tokens[i] = emitted[row]
        return tokens

    # ------------------------------------------------------------ the step
    def _decode(
        self, requests: Sequence[RequestInput], mask: np.ndarray | None = None
    ) -> torch.Tensor:
        """One decode step for ``requests`` (the runner's batch order)."""
        pages = self._pages
        if not (pages is not None):
            _err_msg = "Assertion failed: pages is not None"
            raise AssertionError(_err_msg)
        self._prepare(pages)
        entries, tables, decoding = self._prepare_requests_for_decode(requests)
        sampled = any(entry[4][0] > 0 for entry in entries)
        emitted, placed = self._execute_decode_step(
            decoding, entries, tables, sampled, mask, pages
        )
        if emitted is None:
            return torch.full((len(requests), 8), -1, dtype=torch.int32)
        return self._pack_decode_tokens(requests, decoding, emitted, placed)

    # ------------------------------------------------------------ warmup
    def _warm_plain_step(
        self,
        batch: int,
        sampled: _Dynamic,
        masked: _Dynamic = None,
        *,
        null: np.ndarray,
    ) -> torch.Tensor:
        """Compile and execute a plain warmup step."""
        if masked is None:
            masked = bool(0)
        params = (0.7, 0.9, 20, 1) if sampled else GREEDY
        self._sync_plan([(0, bool(1), 0, 0, params)] * batch, [null] * batch)
        mask = None
        if masked:
            mask = torch.full((batch, VOCAB // 32), -1, dtype=torch.int32, device="tpu")
        emitted, self.control = self.ops.plain_step(
            self.control, batch, self.layout, self.sub_base, sampled, mask
        )
        return emitted

    def _warm_plain(
        self, sampled: _Dynamic, masked: _Dynamic = None, *, null: np.ndarray
    ) -> None:
        """Execute plain warmup steps across all batch sizes."""
        if masked is None:
            masked = bool(0)
        outs = [
            self._warm_plain_step(batch, sampled, masked, null=null)
            for batch in self.ops.batch_sizes
        ]
        for emitted in outs:
            materialize(emitted)

    def _warm_speculate(
        self,
        sampled: _Dynamic,
        *,
        null: np.ndarray,
        offered: _Dynamic = None,
        state: _Dynamic = None,
    ) -> None:
        """Warm up single-request speculative decoding."""
        params = (0.7, 0.9, 20, 1) if sampled else GREEDY
        self.device_plan = None
        self._sync_plan([(0, bool(1), 0, 0, params)], [null])
        if state is not None:
            self.draft_state = state
        elif offered is None:
            self.draft_state = self.draft.new_state(self._zero)
        else:
            self.draft_state = self.draft.from_ring(*offered, self._zero)
        (ids,) = self._enter()(self.control, self.draft_state.vector, self._modes[0])
        emitted = None
        for step in range(4):
            if step == _CONST_2:
                self.device_plan = None
                self._sync_plan([(0, bool(1), 0, 0, params)], [null])
                (ids,) = self._enter()(
                    self.control, self.draft_state.vector, self._modes[1]
                )
            sharded, aux, decisions = self.ops.verify_packed(
                ids,
                self.draft.aux_layers,
                self.layout,
                self.control,
                self._modes[min(step, 1)],
                operator.attrgetter("_synthetic_rates")(self.draft)(self.WIDTH - 1),
            )
            self.draft_state, emitted, ids, self.control = self.draft.step_single(
                self.draft_state,
                aux,
                sharded,
                ids,
                self.control,
                decisions,
                self.layout,
            )
        if emitted is not None:
            materialize(emitted)

    def _warm_speculate_batched(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> list:
        """Warm up batched speculative decoding."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (sampled, requests, rows_per, null, offered, states) = (
            _p_next(_p_iter, _p_get("sampled")),
            _p_next(_p_iter, _p_get("requests")),
            _p_next(_p_iter, _p_get("rows_per")),
            _p_next(_p_iter, _p_get("null")),
            _p_next(_p_iter, _p_get("offered")),
            _p_next(_p_iter, _p_get("states")),
        )
        params = (0.7, 0.9, 20, 1) if sampled else GREEDY
        if not (self.draft is not None):
            _err_msg = "Assertion failed: self.draft is not None"
            raise AssertionError(_err_msg)
        if states is None:
            states = [self.draft.new_state(self._zero) for _ in range(requests)]
        if offered is not None:
            states[0] = self.draft.from_ring(*offered, self._zero)
        emitted = None
        for step in range(3):
            self.device_plan = None
            self._sync_plan(
                [(0, step == 0, 0, 0, params)] * requests, [null] * requests
            )
            modes = self._modes_for((min(step, 1),) * requests)
            (ids,) = self._enter_batched(requests, rows_per)(
                self.control, *(s.vector for s in states), modes
            )
            sharded, aux, decisions = self.ops.verify_batched(
                ids,
                self.draft.aux_layers,
                requests,
                rows_per,
                self.layout,
                self.control,
                modes,
                operator.attrgetter("_synthetic_rates")(self.draft)(rows_per - 1),
            )
            states, emitted, self.control = self.draft.step_batched(
                states,
                aux,
                sharded,
                ids,
                self.control,
                decisions,
                self.layout,
                rows_per,
            )
        if emitted is not None:
            materialize(emitted)
        return states

    def _warm_transitions(self, sampled: _Dynamic, *, null: np.ndarray) -> None:
        """Warm up transitions between single and batched speculation."""
        if not (self.draft is not None):
            _err_msg = "Assertion failed: self.draft is not None"
            raise AssertionError(_err_msg)
        for requests, rows_per in self.spec_shapes:
            self._warm_speculate(sampled, null=null)
            states = [self.draft_state] + [
                self.draft.new_state(self._zero) for _ in range(requests - 1)
            ]
            states = self._warm_speculate_batched(
                sampled, requests, rows_per, null=null, states=states
            )
            self._warm_speculate(sampled, null=null, state=states[0])

    def _warm_precompile_layout(self, null: np.ndarray, started: float) -> None:
        """Precompile layout kernels and graph calls."""
        first = self._warm_plain_step(self.ops.batch_sizes[0], bool(0), null=null)
        synchronize(
            [first, self.control, *self.ops.slots, *self.ops.unified_pools],
            wait=False,
        )
        calls = self.ops.compilation_calls(
            self.layout,
            self.sub_base,
            self.draft.aux_layers if self.draft is not None else (),
            self.spec_alone,
            self.spec_shapes,
            operator.attrgetter("_synthetic_rates")(self.draft)
            if self.draft is not None
            else None,
            masked_ops=True,
        )
        if self.draft is not None:
            calls.extend(
                self.draft.compilation_calls(
                    self.layout, self.spec_alone, self.spec_shapes
                )
            )
        calls.sort(key=lambda call: call.priority, reverse=True)
        prepare_distributed(calls)
        compile_ops(calls)
        materialize(first)
        if self.layout is not None:
            self._compiled_layout = self.layout.key()
        if self.rank == 0:
            logger.info(
                "K3 decode: precompiled %d decode/draft programs in %.1f s",
                len(calls),
                time.perf_counter() - started,
            )

    def _warm_all_steps(
        self, *, speculating: _Dynamic, null: np.ndarray, ring: _Dynamic
    ) -> None:
        """Warm up plain and speculative step programs."""
        for sampled in (bool(0), bool(1)):
            self._warm_plain(sampled, null=null)
            if speculating:
                if self.spec_alone:
                    self._warm_speculate(sampled, null=null)
                for requests, rows_per in self.spec_shapes:
                    self._warm_speculate_batched(sampled, requests, rows_per, null=null)
                if self.spec_alone and self.spec_batches:
                    self._warm_transitions(sampled, null=null)
        if speculating and self.spec_alone and ring:
            self._warm_speculate(bool(0), null=null, offered=ring)

    def _warm_steps(self) -> None:
        """Compile the step ops on the null slot and page, in serving order."""
        started = time.perf_counter()
        pages = self._pages
        if not (pages is not None):
            _err_msg = "Assertion failed: pages is not None"
            raise AssertionError(_err_msg)
        ring = self._null_ring if self._ring else None
        layout = self._prepare(pages)
        null = np.zeros(pages, np.int32)
        if layout is not None and self._compiled_layout != layout.key():
            self._warm_precompile_layout(null, started)

        speculating = self.draft is not None and (self.spec_alone or self.spec_batches)
        self._warm_all_steps(speculating=speculating, null=null, ring=ring)
        if self.control is not None:
            self.control = torch.zeros_like(self.control)
        self._fresh.clear()
        self.tracked.clear()
        self.device_plan = None
        self.spec_req = None
        self.draft_state = None
        self.next_ids = None
        self._pads.clear()
        self.slot_map = SlotMap(self.sub_base, self.groups)
        if self.rank == 0:
            logger.info(
                "K3 decode: warmup complete in %.1f s", time.perf_counter() - started
            )
