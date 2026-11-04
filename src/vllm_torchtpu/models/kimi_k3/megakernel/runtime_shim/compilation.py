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

"""Compile independent torch_tpu graphs before their ordered warmup execution."""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from torch_tpu._internal import execution_mode
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.compiler_options import custom_compiler_options

_ = (Iterable,)

_Dynamic = Any
_CONST_8 = 8


@dataclass(frozen=True)
class CompileCall:
    """One tuple-returning op and the exact eager compiler options it uses."""

    op: _Dynamic
    operands: tuple
    options: dict[str, str]
    priority: int = 0  # Larger independent graphs are submitted first.


def _finish_one(pending: list) -> None:
    """Wait for at least one pending compilation to finish."""
    while True:
        for index, (handle, _inputs, _outputs) in enumerate(pending):
            if handle.is_ready():
                entry = pending.pop(index)
                entry[0].get()
                return
        time.sleep(0.01)


def _drain_pending(
    pending: list, failure: BaseException | None
) -> BaseException | None:
    """Drain remaining pending compiles."""
    while pending:
        try:
            _finish_one(pending)
        except (
            RuntimeError,
            ValueError,
            TypeError,
            OSError,
            KeyboardInterrupt,
            SystemExit,
        ) as exc:
            if failure is None:
                failure = exc
    return failure


def compile_ops(calls: Iterable[CompileCall]) -> None:
    """Populate the eager executable cache with at most eight pending compiles.

    Graphs are built on the caller's thread from fresh placeholders, so
    compiling cannot consume donated buffers or run collectives. This does not
    replace the warmup, which executes the graphs and the programs left out of
    this schedule. The bound also caps compiler memory on hosts shared by eight
    workers. Submitted work drains before a failure propagates.
    """
    pending: list[tuple[_Dynamic, tuple, _Dynamic]] = []
    failure: BaseException | None = None
    try:
        for call in calls:
            if len(pending) == _CONST_8:
                _finish_one(pending)
            inputs = tuple(tpu_torch_compile.placeholder_like(t) for t in call.operands)
            eager_mode = (
                getattr(
                    execution_mode.EagerMode,
                    "INTERNAL_COMPILE_FX_GRAPH",
                    None,
                )
                or execution_mode.EagerMode._INTERNAL_COMPILE_FX_GRAPH
            )
            with execution_mode.set_eager_mode(eager_mode):
                outputs = call.op(*inputs)
            with custom_compiler_options(call.options):
                handle = tpu_torch_compile.async_traverse_and_compile(
                    list(outputs), list(inputs), fast_compile=True
                )
            pending.append((handle, inputs, outputs))
    except (
        RuntimeError,
        ValueError,
        TypeError,
        OSError,
        KeyboardInterrupt,
        SystemExit,
    ) as exc:
        failure = exc
    finally:
        failure = _drain_pending(pending, failure)
    if failure is not None:
        raise failure
