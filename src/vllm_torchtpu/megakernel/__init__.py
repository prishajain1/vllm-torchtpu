# Copyright 2026 Google LLC
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
"""Core declarations, execution config, VMEM pool aliasing, and numeric primitives."""

import importlib

from . import advisor, errors
from .errors import MegakernelConfigError, MegakernelLayoutError

_LAZY_SUBMODULES: frozenset[str] = frozenset(
    {
        "decls",
        "exec_spec",
        "module",
        "module_options",
        "ops",
        "planner",
        "pool_alias",
        "profiles",
        "spec",
    }
)

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "DebugOptions": ("exec_spec", "DebugOptions"),
    "ExecSpec": ("exec_spec", "ExecSpec"),
    "Tiles": ("exec_spec", "Tiles"),
    "default_tiles": ("exec_spec", "default_tiles"),
    "rup": ("ops", "rup"),
    "encode_ecopy": ("planner", "encode_ecopy"),
    "REQUIRED": ("spec", "REQUIRED"),
    "ExtraInput": ("spec", "ExtraInput"),
    "FrozenDict": ("spec", "FrozenDict"),
    "HFParamDecl": ("spec", "HFParamDecl"),
    "KVCacheDecl": ("spec", "KVCacheDecl"),
    "LayerType": ("spec", "LayerType"),
    "ModelSpec": ("spec", "ModelSpec"),
    "Module": ("spec", "Module"),
    "ModuleBinding": ("spec", "ModuleBinding"),
    "ParamTransform": ("spec", "ParamTransform"),
    "RecurrentCacheDecl": ("spec", "RecurrentCacheDecl"),
    "ResolvedModuleGeometry": ("spec", "ResolvedModuleGeometry"),
    "ScratchReq": ("spec", "ScratchReq"),
    "SmallField": ("spec", "SmallField"),
    "Spec": ("spec", "Spec"),
    "StateDecl": ("spec", "StateDecl"),
    "SwitchSlot": ("spec", "SwitchSlot"),
    "VLLMAlias": ("spec", "VLLMAlias"),
    "WeightDecl": ("spec", "WeightDecl"),
}


def __getattr__(name: str) -> object:
    """Lazily import JAX-dependent megakernel submodules and symbols on first access."""
    if name in _LAZY_SUBMODULES:
        mod = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    target = _LAZY_EXPORTS.get(name)
    if target is not None:
        submod_name, attr_name = target
        mod = importlib.import_module(f"{__name__}.{submod_name}")
        val: object = mod.__dict__[attr_name]
        globals()[name] = val
        return val
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    """Return module attributes including lazily exported megakernel symbols."""
    return sorted({*globals(), *__all__})


__all__ = [
    "REQUIRED",
    "DebugOptions",
    "ExecSpec",
    "ExtraInput",
    "FrozenDict",
    "HFParamDecl",
    "KVCacheDecl",
    "LayerType",
    "MegakernelConfigError",
    "MegakernelLayoutError",
    "ModelSpec",
    "Module",
    "ModuleBinding",
    "ParamTransform",
    "RecurrentCacheDecl",
    "ResolvedModuleGeometry",
    "ScratchReq",
    "SmallField",
    "Spec",
    "StateDecl",
    "SwitchSlot",
    "Tiles",
    "VLLMAlias",
    "WeightDecl",
    "advisor",
    "decls",
    "default_tiles",
    "encode_ecopy",
    "errors",
    "exec_spec",
    "module",
    "module_options",
    "ops",
    "planner",
    "pool_alias",
    "profiles",
    "rup",
    "spec",
]
