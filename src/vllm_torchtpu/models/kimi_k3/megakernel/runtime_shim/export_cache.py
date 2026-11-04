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

"""A persistent export cache for the megakernel's torch ops.

torch_tpu exports (traces and lowers) every op on its first call in a process
just to fingerprint the module and find the compiled executable. With
VLLM_XLA_CACHE_PATH set, ops built with an ``export_key`` (``megakernel.bridge``)
persist that call's serialized module, fingerprint and output placeholders
under ``<VLLM_XLA_CACHE_PATH>/k3_export`` (keyed in ``_ExportCache._path``),
and later processes register the module straight from disk.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import operator
import os
import pathlib
import pickle
import time
from collections.abc import Callable, Sequence
from typing import Any

import jax
import torch
import vllm.envs as vllm_envs
from torch_tpu._internal import execution_mode
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.pallas import pallas as pallas_impl

import vllm_torchtpu
from vllm_torchtpu.distributed import sharded_jax_op
from vllm_torchtpu.logger import init_logger

_ = (
    Callable,
    Sequence,
)

_Dynamic = Any


logger = init_logger(__name__)

_EXPORT_ENV_PREFIXES = ("JAX_",)
_SOURCE_FINGERPRINT_BOX: list[str | None] = [None]


def _source_fingerprint() -> str:
    """Execute source fingerprint logic."""
    if _SOURCE_FINGERPRINT_BOX[0] is None:
        digest = hashlib.blake2b(digest_size=16)
        root = pathlib.Path(vllm_torchtpu.__file__).resolve().parent
        for path in sorted(root.rglob("*.py")):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
        for package in ("jax", "jaxlib", "torch_tpu", "libtpu"):
            try:
                version = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                version = "-"
            digest.update(f"{package}={version};".encode())
        _SOURCE_FINGERPRINT_BOX[0] = digest.hexdigest()
    return _SOURCE_FINGERPRINT_BOX[0]


def _process_rank() -> int:
    """Execute process rank logic."""
    with contextlib.suppress(Exception):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank())
    return int(jax.process_index())


class _ExportCache:
    """The persistent export cache of one op (see the module docstring)."""

    def __init__(self, root: pathlib.Path, export_key: str) -> None:
        """Execute init logic."""
        self.root, self.export_key = root, export_key
        self.shared_rank: int | None = None

    def _path(
        self, op: _Dynamic, args: _Dynamic, kwargs: _Dynamic, *, rank: int | None = None
    ) -> pathlib.Path:
        """Execute path logic."""
        if rank is None:
            rank = self.shared_rank
        if rank is None:
            rank = _process_rank()
        signature: list[object] = []
        for index, arg in enumerate(args):
            if index in op.static_argnums:
                signature.append(("static", repr(arg)))
            elif arg is None:
                signature.append(None)
            else:
                signature.append((tuple(arg.shape), str(arg.dtype)))
        env = sorted(
            (k, v) for k, v in os.environ.items() if k.startswith(_EXPORT_ENV_PREFIXES)
        )
        mesh = (
            (tuple(int(d.id) for d in op.mesh.devices.flat), tuple(op.mesh.axis_names))
            if op.mesh
            else None
        )
        key = repr(
            (
                op.name,
                self.export_key,
                signature,
                sorted((k, repr(v)) for k, v in kwargs.items()),
                repr(op.input_partition_specs),
                repr(op.output_partition_specs),
                tuple(op.donate_argnums),
                op.static_argnums,
                mesh,
                rank,
                _source_fingerprint(),
                env,
            )
        )
        digest = hashlib.blake2b(key.encode(), digest_size=16).hexdigest()
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in op.name)[:80]
        return self.root / f"{safe}-{digest}.pkl"

    def call(
        self, op: _Dynamic, call: _Dynamic, args: _Dynamic, kwargs: _Dynamic
    ) -> _Dynamic:
        """Execute call logic."""
        kernel_key = operator.attrgetter("_get_kernel_invocation_key")(pallas_impl)(
            op.trace_key, args, kwargs, op.static_argnums
        )
        if kernel_key in op.output_shapes:
            return call(op, *args, **kwargs)
        path = self._path(op, args, kwargs)
        if self._load(op, kernel_key, path):
            return call(op, *args, **kwargs)
        vars(op)["_recorded_export"] = None
        result = call(op, *args, **kwargs)
        lowered = operator.attrgetter("_recorded_export")(op)
        fingerprint = op.kernel_key_to_mlir_fingerprint.get(kernel_key)
        if lowered is not None and fingerprint is not None:
            try:
                self._store(path, fingerprint, lowered)
            except OSError as exc:
                logger.warning("export cache: could not store %s: %s", path, exc)
        return result

    def _load(self, op: _Dynamic, kernel_key: _Dynamic, path: pathlib.Path) -> bool:
        """Execute load logic."""
        try:
            with path.open("rb") as handle:
                entry = vars(pickle)["load"](handle)
            module = (self.root / f"module-{entry['fingerprint']}.mlirbc").read_bytes()
        except (OSError, EOFError, pickle.UnpicklingError, KeyError):
            return False
        started = time.perf_counter()
        pallas_impl.tpu_torch_pallas.register_custom_kernel(
            op.name, entry["fingerprint"], serialized_mlir_module=module
        )
        outputs = [
            operator.attrgetter("_placeholder")(sharded_jax_op)(
                jax.core.ShapedArray(shape, dtype), spec, op.mesh
            )
            if array
            else pallas_impl.torch_placeholder(
                jax.core.ShapedArray(shape, dtype), mesh=op.mesh
            )
            for array, shape, dtype, spec in entry["outputs"]
        ]
        op.kernel_key_to_mlir_fingerprint[kernel_key] = entry["fingerprint"]
        op.output_shapes[kernel_key] = (outputs, entry["out_tree"])
        logger.debug(
            "export cache: %s from %s (%.2f s)",
            op.name,
            path.name,
            time.perf_counter() - started,
        )
        return True

    def _store(self, path: pathlib.Path, fingerprint: str, lowered: _Dynamic) -> None:
        """Execute store logic."""
        self.root.mkdir(parents=True, exist_ok=True)
        shardings = getattr(lowered, "_out_named_shardings", None)
        if shardings is None or len(shardings) != len(lowered.out_avals):
            shardings = (None,) * len(lowered.out_avals)
        outputs = []
        for aval, sharding in zip(lowered.out_avals, shardings):
            spec = getattr(sharding, "spec", None)
            outputs.append(
                (
                    spec is not None and isinstance(aval, jax.core.ShapedArray),
                    tuple(aval.shape),
                    aval.dtype,
                    spec,
                )
            )
        module_path = self.root / f"module-{fingerprint}.mlirbc"
        if not module_path.exists():
            _atomic_write(module_path, bytes(lowered.mlir_module_serialized))
        _atomic_write(
            path,
            pickle.dumps(
                {
                    "fingerprint": fingerprint,
                    "outputs": outputs,
                    "out_tree": lowered.out_tree,
                }
            ),
        )


def prepare_distributed(calls: Sequence[_Dynamic]) -> None:
    """Export rank-invariant programs on different workers.

    Every worker passes the same ordered programs, whose bodies and export keys
    do not depend on the rank. Each program's owner traces it on fresh
    placeholders without compiling or executing it; the others later load its
    entry from the export cache. Without the cache, fall back to per-worker
    export. All workers see export errors before any proceeds to compilation.
    """
    if not calls or not torch.distributed.is_initialized():
        return

    world = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    # CustomOpDef keeps the registered JaxCallable as its initial function;
    # ops without the export cache fall back to per-worker export.
    ops: list[_Dynamic] = [getattr(call.op, "_init_fn", None) for call in calls]
    eligible = all(
        getattr(op, "_export_cache", None) is not None
        and op.mesh is not None
        and op.mesh.size == world
        for op in ops
    )
    signatures = (
        [
            (
                operator.attrgetter("_path")(operator.attrgetter("_export_cache")(op))(
                    op, call.operands, {}, rank=index % world
                ).name,
                sorted(call.options.items()),
            )
            for index, (op, call) in enumerate(zip(ops, calls, strict=True))
        ]
        if eligible
        else None
    )
    gathered: list[_Dynamic] = [None] * world
    torch.distributed.all_gather_object(gathered, signatures)
    if any(signature is None for signature in gathered):
        return
    if any(signature != signatures for signature in gathered):
        _err_msg = "distributed export requires identical program signatures"
        raise ValueError(_err_msg)

    for index, op in enumerate(ops):
        operator.attrgetter("_export_cache")(op).shared_rank = index % world
    failure: Exception | None = None
    try:
        for index, call in enumerate(calls):
            if index % world != rank:
                continue
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
                call.op(*inputs)
    except (RuntimeError, ValueError, TypeError, OSError) as exc:
        failure = exc
    torch.distributed.all_gather_object(
        gathered, None if failure is None else f"rank {rank}: {failure!r}"
    )
    if any(error is not None for error in gathered):
        _err_msg = f"distributed export failed: {gathered}"
        raise RuntimeError(_err_msg)


def _atomic_write(path: pathlib.Path, data: bytes) -> None:
    """Execute atomic write logic."""
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


_CACHED_CLASSES: dict[type, type] = {}


def _cached_class(cls: type) -> type:
    """``cls`` whose ``__call__`` goes through the op's export cache."""
    cached = _CACHED_CLASSES.get(cls)
    if cached is None:
        call = cls.__call__

        def _call(self: _Dynamic, *args: _Dynamic, **kwargs: _Dynamic) -> _Dynamic:
            """Execute call logic."""
            return self._export_cache.call(self, call, args, kwargs)

        cached = type(f"{cls.__name__}WithExportCache", (cls,), {"__call__": _call})
        _CACHED_CLASSES[cls] = cached
    return cached


def hook(export_key: str | None) -> Callable[[_Dynamic], None] | None:
    """The ``sharded_jax_op(configure_callable=...)`` hook that puts an op.

    with ``export_key`` on the cache, or None when there is no key or no
    cache path (only an explicitly set, non-empty VLLM_XLA_CACHE_PATH: its
    default is a home directory; empty turns the cache off).
    """
    if (
        not export_key
        or not vllm_envs.is_set("VLLM_XLA_CACHE_PATH")
        or not vllm_envs.VLLM_XLA_CACHE_PATH
    ):
        return None
    root = pathlib.Path(vllm_envs.VLLM_XLA_CACHE_PATH) / "k3_export"

    def configure(op: _Dynamic) -> None:
        """Execute configure logic."""
        exported = op.exported

        def recording(*args: _Dynamic, **kwargs: _Dynamic) -> _Dynamic:
            """Execute recording logic."""
            lowered = exported(*args, **kwargs)
            vars(op)["_recorded_export"] = lowered
            return lowered

        op.exported = recording
        vars(op)["_export_cache"] = _ExportCache(root, export_key)
        op.__class__ = _cached_class(type(op))

    return configure
