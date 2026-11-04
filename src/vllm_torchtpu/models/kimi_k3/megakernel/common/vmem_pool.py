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

"""Trace-time manager for the kernel's shared VMEM weight arena.

Source provenance: Adapted from inferact/k3-megakernel@30c9ba966b.

Kernel code asks for weights by name and never places them itself:

    pool.request("router", nbytes, start, index=i)  # queue a load (FIFO)
    offset = pool.acquire("router", wait)   # wait for it; its arena offset
    pool.release("router")                  # its bytes are free again

A request is issued (its DMAs started) as soon as the arena has a free range
for it and every earlier request has been issued; otherwise it waits in the
queue and is issued by the release that makes room. Where and when each
weight loads therefore follows from the arena size and the order of
requests and releases, not from hand-placed offsets.

``allocate`` claims a range for a region its owner fills itself (the routed
expert slots); ``replaces`` lets it overlap loads the owner releases before
its first write (the router, consumed before the first expert copy).

Everything here runs while the kernel is traced: offsets, issue points and
semaphores are Python constants, and the emitted code is the same DMA starts
and waits as hand-placed calls. The layer loop goes through ``loop``, which
checks that its body ends in the arena state it started in, so a static
offset is valid on every iteration.

``start(offset, index)`` gets the loop index the load was requested for, which a
closure would misread for a load issued after the loop back-edge.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from jax.experimental import pallas as pl

from . import pool_alias

_Dynamic = Any


ALIGN = 4096


@dataclasses.dataclass
class _Entry:
    """Manage entry instance and state."""

    name: str
    nbytes: int
    start: object = (
        None  # start(offset, index) emits the DMA starts; None for allocate()
    )
    index: object = None  # the loop index the load was requested for
    predicate: object = None  # traced bool guarding the starts, or None
    offset: int | None = None
    state: str = "queued"  # queued | in_flight | acquired | allocated
    replaces: tuple = ()

    def signature(self) -> _Dynamic:
        """Execute signature logic."""
        return (self.name, self.nbytes, self.offset, self.state, self.replaces)


class PoolError(ValueError):
    """Manage pool error instance and state."""


class VmemPool:
    """Manage vmem pool instance and state."""

    def __init__(
        self, arena: _Dynamic, capacity: _Dynamic, *, report: _Dynamic = None
    ) -> None:
        """``arena``: [capacity / 128, 128] uint8 VMEM ref (None: dry run)."""
        if capacity % ALIGN:
            _err_msg = "The arena capacity must be a multiple of 4 KiB"
            raise PoolError(_err_msg)
        self.arena, self.capacity = arena, capacity
        self.entries = {}  # live, in request order
        self.queue = []  # names of queued requests, FIFO
        self.events = []
        self.report = bool(report)
        self.phase = ""
        self.finished = False

    # -- placement -----------------------------------------------------------

    def _occupied(self, ignore: _Dynamic = ()) -> _Dynamic:
        """Execute occupied logic."""
        return sorted(
            (entry.offset, entry.offset + _aligned(entry.nbytes))
            for entry in self.entries.values()
            if entry.offset is not None and entry.name not in ignore
        )

    def _fit(self, nbytes: _Dynamic, ignore: _Dynamic = ()) -> _Dynamic:
        """Lowest free offset for ``nbytes`` (first fit), or None."""
        size = _aligned(nbytes)
        cursor = 0
        for start, end in self._occupied(ignore):
            if start - cursor >= size:
                return cursor
            cursor = max(cursor, end)
        return cursor if self.capacity - cursor >= size else None

    def _issue_queued(self) -> _Dynamic:
        """Issue queued requests in FIFO order while the head fits."""
        while self.queue:
            entry = self.entries[self.queue[0]]
            offset = self._fit(entry.nbytes)
            if offset is None:
                return
            self.queue.pop(0)
            entry.offset, entry.state = offset, "in_flight"
            self._log("issue", entry)
            index = entry.index
            if entry.predicate is None:
                entry.start(offset, index)
            else:
                pl.when(entry.predicate)(
                    lambda entry=entry, offset=offset, index=index: entry.start(
                        offset, index
                    )
                )

    # -- API -----------------------------------------------------------------

    def request(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
        """Queue the load ``start(offset, index)`` of ``nbytes``;.

        issue it if it fits.
        """
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            name,
            nbytes,
            start,
        ) = (
            _p_next(_p_iter, _p_get("name")),
            _p_next(_p_iter, _p_get("nbytes")),
            _p_next(_p_iter, _p_get("start")),
        )
        (
            index,
            predicate,
        ) = (
            _p_get("index"),
            _p_get("predicate"),
        )
        self._check_open(name)
        if _aligned(nbytes) > self.capacity:
            _err_msg = f"{name}: {nbytes} bytes exceed the {self.capacity}-byte arena"
            raise PoolError(_err_msg)
        entry = _Entry(name, nbytes, start, index, predicate)
        self.entries[name] = entry
        self.queue.append(name)
        self._log("request", entry)
        self._issue_queued()

    def allocate(
        self, name: _Dynamic, nbytes: _Dynamic, *, replaces: _Dynamic = ()
    ) -> _Dynamic:
        """Claim a range now for a region its owner fills; it may overlap the.

        live entries in ``replaces``, which must be released before use.
        """
        self._check_open(name)
        for other in replaces:
            if other not in self.entries:
                _err_msg = f"{name}: replaces {other}, which is not live"
                raise PoolError(_err_msg)
        offset = self._fit(nbytes, ignore=replaces)
        if offset is None:
            _err_msg = f"allocate({name}): {self._describe(nbytes)}"
            raise PoolError(_err_msg)
        entry = _Entry(
            name, nbytes, offset=offset, state="allocated", replaces=tuple(replaces)
        )
        self.entries[name] = entry
        self._log("allocate", entry)
        return offset

    def acquire(self, name: _Dynamic, wait: _Dynamic = None) -> _Dynamic:
        """Wait for a requested load (``wait(offset)`` emits the waits; None.

        when the consumer waits itself). Returns its offset.
        """
        entry = self._live(name)
        if entry.state == "allocated":
            pending = [other for other in entry.replaces if other in self.entries]
            if pending:
                _err_msg = f"acquire({name}): {pending} not released yet"
                raise PoolError(_err_msg)
            return entry.offset
        if entry.state == "queued":
            _err_msg = (
                f"acquire({name}) before its load could be issued: "
                f"{self._describe(entry.nbytes)}"
            )
            raise PoolError(_err_msg)
        if entry.state == "in_flight":
            entry.state = "acquired"
            self._log("acquire", entry)
            if wait is not None:
                wait(entry.offset)
        return entry.offset

    def offset(self, name: _Dynamic) -> _Dynamic:
        """The arena offset of a live, issued entry (to build views)."""
        entry = self._live(name)
        if entry.offset is None:
            _err_msg = f"{name} has no offset yet: {self._describe(entry.nbytes)}"
            raise PoolError(_err_msg)
        return entry.offset

    def release(self, name: _Dynamic) -> _Dynamic:
        """Execute release logic."""
        entry = self._live(name)
        if entry.state == "queued":
            _err_msg = f"release({name}) of a load that was never issued"
            raise PoolError(_err_msg)
        if entry.state == "in_flight":
            _err_msg = f"release({name}) of a load that was never waited on"
            raise PoolError(_err_msg)
        del self.entries[name]
        self._log("release", entry)
        self._issue_queued()

    def view(self, offset: _Dynamic, shape: _Dynamic, dtype: _Dynamic) -> _Dynamic:
        """A [shape] dtype view of the arena from byte ``offset``."""
        nbytes = pool_alias.storage_size(shape, dtype)
        return pool_alias.view(
            self.arena.at[pl.ds(offset // 128, nbytes // 128), :], shape, dtype
        )

    # -- control flow --------------------------------------------------------

    def signature(self) -> _Dynamic:
        """Execute signature logic."""
        return (
            tuple(entry.signature() for entry in self.entries.values()),
            tuple(self.queue),
        )

    def loop(self, lower: _Dynamic, upper: _Dynamic) -> _Dynamic:
        """``pl.loop`` decorator; the body must end in the state it began in.

        Loads carried in from the previous iteration run unconditionally
        (the iteration exists), so their predicates are dropped.
        """

        def decorator(body: _Dynamic) -> _Dynamic:
            """Execute decorator logic."""
            entry_signature = self.signature()

            def traced(index: _Dynamic) -> _Dynamic:
                """Execute traced logic."""
                for entry in self.entries.values():
                    entry.predicate = None
                    if entry.state == "queued":
                        entry.index = index  # requested for this iteration
                self.phase_push("loop")
                body(index)
                self.phase_pop()
                if self.signature() != entry_signature:
                    _err_msg = (
                        f"The loop body does not end in the arena state it began in:\n "
                        f" entry: {entry_signature}\n  exit:  {self.signature()}"
                    )
                    raise PoolError(_err_msg)
                for entry in self.entries.values():
                    entry.predicate = None

            pl.loop(lower, upper)(traced)
            # The last iteration's requests were for a layer that does not
            # exist; nothing may wait for them.
            self.finished = True

        return decorator

    # -- reporting -----------------------------------------------------------

    def phase_push(self, label: _Dynamic) -> _Dynamic:
        """Execute phase push logic."""
        self.phase = f"{self.phase}/{label}" if self.phase else label

    def phase_pop(self) -> _Dynamic:
        """Execute phase pop logic."""
        self.phase = self.phase.rpartition("/")[0]

    def _log(self, op: _Dynamic, entry: _Dynamic) -> _Dynamic:
        """Execute log logic."""
        used = sum(
            _aligned(other.nbytes)
            for other in self.entries.values()
            if other.offset is not None
        )
        event = (self.phase, op, entry.name, entry.offset, entry.nbytes, used)
        self.events.append(event)

    def timeline(self) -> _Dynamic:
        """Execute timeline logic."""
        lines = [f"VMEM pool: {self.capacity / 2**20:.2f} MiB arena"]
        for phase, op, name, offset, nbytes, used in self.events:
            place = "" if offset is None else f"@{offset / 2**20:6.2f}"
            lines.append(
                f"  {phase:28s} {op:9s} {name:12s} {nbytes / 2**20:6.2f} MiB {place:8s}"
                f" in use {used / 2**20:6.2f}"
            )
        return "\n".join(lines)

    # -- helpers -------------------------------------------------------------

    def _check_open(self, name: _Dynamic) -> _Dynamic:
        """Execute check open logic."""
        if self.finished:
            _err_msg = f"{name}: the pool is closed"
            raise PoolError(_err_msg)
        if name in self.entries:
            _err_msg = f"{name} is already live"
            raise PoolError(_err_msg)

    def _live(self, name: _Dynamic) -> _Dynamic:
        """Execute live logic."""
        if self.finished and name in self.entries:
            _err_msg = f"{name}: requested in the last loop iteration, never issued"
            raise PoolError(_err_msg)
        if name not in self.entries:
            _err_msg = f"{name} is not live"
            raise PoolError(_err_msg)
        return self.entries[name]

    def _describe(self, nbytes: _Dynamic) -> _Dynamic:
        """Execute describe logic."""
        held = ", ".join(
            f"{entry.name} [{entry.offset / 2**20:.2f}, "
            f"+{entry.nbytes / 2**20:.2f}) {entry.state}"
            for entry in self.entries.values()
            if entry.offset is not None
        )
        queued = ", ".join(self.queue)
        return (
            f"needs {nbytes / 2**20:.2f} MiB of a "
            f"{self.capacity / 2**20:.2f} MiB arena; "
            f"held: {held or 'nothing'}; queued: {queued or 'nothing'}"
        )


def _aligned(nbytes: _Dynamic) -> _Dynamic:
    """Execute aligned logic."""
    return -(-nbytes // ALIGN) * ALIGN
