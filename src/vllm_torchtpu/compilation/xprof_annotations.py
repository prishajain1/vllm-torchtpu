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
"""Hierarchical XProf / StableHLO debug location annotations for TorchTPU.

By default, TorchTPU's FX-to-MLIR exporter emits flat operator names (such as
``loc("mm/mm"(#loc))``), includes only the single innermost Python frame from
each FX node (alongside internal ``export.py`` / ``compiler.py`` /
``split_compiler.py`` frames), and leaves XLA module names unset unless a
``module_name`` is passed explicitly.

When ``VLLM_TPU_XPROF_ANNOTATIONS=1`` (default), this module enriches compile
time StableHLO metadata with zero runtime dispatch overhead:

1. **Framework Name Scope & Framework Ops**: Extracts the ``nn_module_stack``
   hierarchy from FX nodes (propagating across decomposed/helper FX nodes) so
   XProf renders nested module bars such as
   ``layers.0[Gemma4DecoderLayer]/mlp[Gemma4MLP]/gate_up_proj[MergedColumnParallelLinear]``
   above clean leaf ops (``mm``, ``mean``, ``ragged_paged_attention``).
2. **Source Code & Call Stack**: Synthesizes full ``callsite(...)`` location
   chains from ``node.stack_trace`` (e.g. ``gemma4.py`` down to leaf kernels)
   while stripping internal compiler trampoline frames.
3. **XLA Modules**: Derives deterministic, human-readable XLA module names
   (e.g. ``Gemma4TextModel_T16384_L00_to_L59_submod_0``) that remain identical
   across tensor-parallel ranks so collective handshakes and compile caches
   stay aligned.
"""

from __future__ import annotations

import contextlib
import dataclasses
import re
import threading
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import torch
import torch.fx as fx
import torch.utils._pytree as pytree

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_SKIPPED_WRAPPER_CLASSES = frozenset(
    {
        "OptimizedModule",
        "GraphModule",
        "_WrapperModule",
        "ModuleList",
        "ModuleDict",
        "Sequential",
    }
)

_NOISY_STACK_PATH_FRAGMENTS = (
    "torch/nn/modules/module.py",
    "torch/nn/modules/container.py",
    "torch/_dynamo/",
    "torch/_functorch/",
    "torch/_inductor/",
    "torch/_library/",
    "torch/_compile.py",
    "torch/fx/",
    "torch_tpu/_internal/export/",
    "torch_tpu/_internal/compile/",
    "vllm/compilation/",
    "vllm/tracing/",
    "vllm_torchtpu/compilation/",
)

_COLLECTIVE_OP_NAMES = frozenset(
    {
        "all_reduce",
        "all_reduce_",
        "reduce_scatter_tensor",
        "all_gather_into_tensor",
        "all_to_all_single",
        "wait_tensor",
        "getitem",
    }
)

_GETATTR_RE = re.compile(r"getattr\(([^,()]+),\s*['\"]([^'\"]+)['\"]\)")
_MODULES_BRACKET_RE = re.compile(r"\._modules\[['\"]([^'\"]+)['\"]\]")
_SELF_PREFIX_RE = re.compile(r"^(?:L\[['\"]self['\"]\]|self)(?:\.|$)")
_BRACKET_KEY_RE = re.compile(r"\[['\"]?([^'\"\]]+)['\"]?\]")
_MULTI_DOT_RE = re.compile(r"\.{2,}")
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.([A-Za-z0-9_]+))?")
_TRACEBACK_FRAME_RE = re.compile(r'File "([^"]+)", line (\d+), in ([^\n]+)')
_MARKER_FILE_RE = re.compile(r"^__vllm_xprof_fx_(\d+)__$")

_LOC_DEF_RE = re.compile(r"^(#loc[A-Za-z0-9_]*)\s*=\s*loc\((.*)\)\s*$")
_FILE_LOC_RE = re.compile(r'^"((?:[^"\\]|\\.)*)"\s*:\s*(\d+)\s*:\s*(\d+)')
_NAMELOC_CHILD_RE = re.compile(r'^"((?:[^"\\]|\\.)*)"\((#loc[A-Za-z0-9_]*)\)$')
_NAMELOC_BARE_RE = re.compile(r'^"((?:[^"\\]|\\.)*)"$')
_CALLSITE_RE = re.compile(
    r"^callsite\((#loc[A-Za-z0-9_]*)\s+at\s+(#loc[A-Za-z0-9_]*)\)$"
)
_LOC_REF_RE = re.compile(r"#loc[A-Za-z0-9_]*(?![A-Za-z0-9_])")
_BODY_LOC_RE = re.compile(r"loc\((#loc[A-Za-z0-9_]*)\)")
_OP_TYPE_WRAP_RE = re.compile(r"^([A-Za-z0-9_.-]+)\((.+)\)$")

_compile_context = threading.local()
_fallback_token_bucket: int | None = None
_hooks_lock = threading.Lock()
_hooks_installed = False


@dataclasses.dataclass(frozen=True)
class NodeAnnotation:
    """Extracted XProf metadata for a single FX node execution."""

    scope_path: str
    op_name: str
    frames: tuple[tuple[str, int, str], ...]


@contextlib.contextmanager
def compilation_context(*, token_bucket: int | None = None) -> Iterator[None]:
    """Attach compile-range metadata (such as padded token bucket) during compile."""
    global _fallback_token_bucket
    prev_bucket = getattr(_compile_context, "token_bucket", None)
    prev_fallback = _fallback_token_bucket
    _compile_context.token_bucket = token_bucket
    _fallback_token_bucket = token_bucket
    try:
        yield
    finally:
        _compile_context.token_bucket = prev_bucket
        _fallback_token_bucket = prev_fallback


def sanitize_mlir_symbol(name: str, max_len: int = 80) -> str:
    """Sanitize `name` into a valid MLIR bare symbol identifier."""
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", name)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "vllm_tpu_module"
    if not (cleaned[0].isalpha() or cleaned[0] == "_"):
        cleaned = f"m_{cleaned}"
    return cleaned[:max_len].rstrip("_") or "vllm_tpu_module"


def _escape_mlir_string(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", " ")
        .replace("\r", "")
    )


def _unescape_mlir_string(text: str) -> str:
    return text.replace('\\"', '"').replace("\\\\", "\\")


def _sanitize_scope_segment(segment: str) -> str:
    """Sanitize a NameLoc path segment so XLA exports the full op_name intact.

    XLA's `location_exporter` truncates `op_name` at `@`, splits `op_name:op_type`
    on `:`, and treats `foo(bar)` (ending in `)`) as `op_type(op_name)`. Replacing
    `@`/`:` with `_` and parentheses with brackets preserves the full `/`
    hierarchy cleanly in XProf's `Framework Name Scope` and `Framework Ops` rows.
    """
    s = segment.strip()
    s = (
        s.replace("@", "_")
        .replace(":", "_")
        .replace("(", "[")
        .replace(")", "]")
        .replace('"', "")
        .replace("\\", "_")
        .replace("\n", " ")
        .replace("\r", "")
    )
    return s.strip()


def _normalize_fqn(raw_fqn: str) -> str:
    """Normalize Dynamo / FX module paths into clean dot-separated FQNs."""
    s = raw_fqn.strip()
    if not s:
        return ""
    for _ in range(8):
        updated = _GETATTR_RE.sub(r"\1.\2", s)
        if updated == s:
            break
        s = updated
    s = _MODULES_BRACKET_RE.sub(r".\1", s)
    s = _SELF_PREFIX_RE.sub("", s)
    s = _BRACKET_KEY_RE.sub(r".\1", s)
    s = _MULTI_DOT_RE.sub(".", s)
    return s.strip(".")


def _short_class_name(cls_or_name: Any) -> str:
    if isinstance(cls_or_name, type):
        return cls_or_name.__name__
    s = str(cls_or_name).strip()
    if s.startswith("<class ") and s.endswith(">"):
        s = s[7:-1].strip("'\" ")
    return s.rsplit(".", 1)[-1].strip("'\"<> ")


def _iter_nn_module_entries(nn_module_stack: Any) -> list[tuple[str, str]]:
    """Return normalized `(fqn, short_class_name)` pairs from `nn_module_stack`."""
    if not isinstance(nn_module_stack, Mapping):
        return []
    entries: list[tuple[str, str]] = []
    for item in nn_module_stack.values():
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            raw_fqn, raw_cls = item[0], item[1]
        else:
            continue
        fqn = _normalize_fqn(str(raw_fqn))
        cls_name = _short_class_name(raw_cls)
        if cls_name in _SKIPPED_WRAPPER_CLASSES:
            continue
        entries.append((fqn, cls_name))
    return entries


def extract_fx_node_scope(
    nn_module_stack: Any,
    fallback_args: Sequence[Any] = (),
) -> str:
    """Build a `/`-delimited XProf `Framework Name Scope` path for an FX node."""
    entries = _iter_nn_module_entries(nn_module_stack)

    # When deeper child modules exist, omit the root (`fqn == ""`) so Row 1 of
    # XProf's `Framework Name Scope` immediately displays `layers.0[...]`,
    # `layers.1[...]`, ..., `norm[...]` side-by-side instead of a single 100%-wide
    # model root bar. Keep the root class if an op executes directly on the root.
    non_root = [(fqn, cls_name) for fqn, cls_name in entries if fqn]
    active = non_root if non_root else entries

    segments: list[str] = []
    prev_fqn = ""
    for fqn, cls_name in active:
        if not fqn:
            seg = f"[{cls_name}]" if cls_name else ""
        else:
            if prev_fqn and fqn == prev_fqn:
                continue
            if prev_fqn and fqn.startswith(f"{prev_fqn}."):
                rel = fqn[len(prev_fqn) + 1 :]
            else:
                rel = fqn
            seg = f"{rel}[{cls_name}]" if cls_name else rel
            prev_fqn = fqn
        seg = _sanitize_scope_segment(seg)
        if seg:
            segments.append(seg)

    if segments:
        return "/".join(segments)

    # Fallback for opaque custom ops (e.g. unified_attention_with_output) that
    # carry a layer FQN string argument such as "model.layers.0.self_attn.attn".
    for arg in fallback_args:
        if isinstance(arg, str) and "layers." in arg:
            norm = _normalize_fqn(arg)
            if norm:
                parts: list[str] = []
                tokens = norm.split(".")
                i = 0
                while i < len(tokens):
                    if tokens[i] == "layers" and i + 1 < len(tokens) and tokens[i + 1].isdigit():
                        parts.append(_sanitize_scope_segment(f"layers.{tokens[i + 1]}"))
                        i += 2
                    else:
                        parts.append(_sanitize_scope_segment(tokens[i]))
                        i += 1
                return "/".join(p for p in parts if p)

    return ""


def _is_noisy_stack_file(filepath: str) -> bool:
    normalized = filepath.replace("\\", "/")
    return any(frag in normalized for frag in _NOISY_STACK_PATH_FRAGMENTS)


def parse_fx_stack_frames(
    stack_trace: str | None,
) -> list[tuple[str, int, str]]:
    """Extract clean `(filepath, lineno, funcname)` frames from `node.stack_trace`.

    Frames are returned from outermost caller to innermost leaf so they can be
    chained directly into MLIR `callsite(inner at outer)` locations.
    """
    if not stack_trace:
        return []

    all_frames: list[tuple[str, int, str]] = []
    filtered_frames: list[tuple[str, int, str]] = []

    for match in _TRACEBACK_FRAME_RE.finditer(stack_trace):
        filepath = match.group(1).strip()
        try:
            lineno = max(1, int(match.group(2)))
        except ValueError:
            lineno = 1
        funcname = match.group(3).strip() or "forward"
        frame = (filepath, lineno, funcname)
        if not all_frames or all_frames[-1][:2] != frame[:2]:
            all_frames.append(frame)
        if not _is_noisy_stack_file(filepath):
            if not filtered_frames or filtered_frames[-1][:2] != frame[:2]:
                filtered_frames.append(frame)

    if filtered_frames:
        return filtered_frames
    return all_frames[-1:] if all_frames else []


def _fx_node_leaf_op_name(node: fx.Node) -> str:
    if node.op == "call_method":
        return _sanitize_scope_segment(str(node.target)) or "call_method"
    if node.op == "call_module":
        return _sanitize_scope_segment(str(node.target)) or "call_module"
    target = node.target
    schema = getattr(target, "_schema", None)
    if schema is not None and getattr(schema, "name", None):
        raw = str(schema.name).split("::")[-1].split(".")[0]
        return _sanitize_scope_segment(raw) or "op"
    name = getattr(target, "__name__", None) or str(target)
    raw = str(name).split("::")[-1].split(".")[0]
    return _sanitize_scope_segment(raw) or "op"


def clean_op_name(raw_op: str, fallback: str = "op") -> str:
    """Normalize an MLIR `NameLoc` op string and deduplicate repeated ATen names."""
    s = raw_op.strip().rstrip(":")
    m = _OP_TYPE_WRAP_RE.match(s)
    if m is not None:
        op_type, inner = m.group(1), m.group(2)
        s = f"{inner}/{op_type}"

    raw_parts = [p.strip() for p in s.split("/") if p.strip()]
    non_jit = [p for p in raw_parts if not p.startswith(("jit(", "pjit("))]
    if non_jit:
        raw_parts = non_jit

    deduped: list[str] = []
    for part in raw_parts:
        clean_part = _sanitize_scope_segment(part)
        if not clean_part:
            continue
        if not deduped or deduped[-1] != clean_part:
            deduped.append(clean_part)

    if deduped:
        return "/".join(deduped)
    return _sanitize_scope_segment(fallback) or "op"


def build_propagated_node_annotations(
    gm: fx.GraphModule,
) -> dict[fx.Node, NodeAnnotation]:
    """Extract and propagate `(scope_path, frames)` across helper/view FX nodes."""
    compute_nodes = [
        n
        for n in gm.graph.nodes
        if n.op not in ("placeholder", "get_attr", "output")
    ]
    scopes: dict[fx.Node, str] = {}
    frames_map: dict[fx.Node, list[tuple[str, int, str]]] = {}

    for node in compute_nodes:
        flat_args = [a for a in pytree.tree_leaves(node.args) if isinstance(a, str)]
        scopes[node] = extract_fx_node_scope(
            node.meta.get("nn_module_stack"), fallback_args=flat_args
        )
        frames_map[node] = parse_fx_stack_frames(node.stack_trace)

    # Forward pass: inherit missing scope/stack from closest compute producer.
    for node in compute_nodes:
        if scopes[node] and frames_map[node]:
            continue
        producers = [
            arg
            for arg in pytree.tree_leaves((node.args, node.kwargs))
            if isinstance(arg, fx.Node) and arg in scopes
        ]
        for prod in reversed(producers):
            if not scopes[node] and scopes[prod]:
                scopes[node] = scopes[prod]
            if not frames_map[node] and frames_map[prod]:
                frames_map[node] = frames_map[prod]
            if scopes[node] and frames_map[node]:
                break

    # Backward pass: inherit missing scope/stack for input views/casts feeding a layer.
    for node in reversed(compute_nodes):
        if scopes[node] and frames_map[node]:
            continue
        for user in node.users:
            if user not in scopes:
                continue
            if not scopes[node] and scopes[user]:
                scopes[node] = scopes[user]
            if not frames_map[node] and frames_map[user]:
                frames_map[node] = frames_map[user]
            if scopes[node] and frames_map[node]:
                break

    return {
        node: NodeAnnotation(
            scope_path=scopes[node],
            op_name=_fx_node_leaf_op_name(node),
            frames=tuple(frames_map[node]),
        )
        for node in compute_nodes
    }


def derive_xla_module_name(
    gm: fx.GraphModule,
    args: Sequence[Any] = (),
    *,
    submod_name: str | None = None,
    token_bucket: int | None = None,
) -> str:
    """Derive a deterministic, rank-symmetric XLA module name for XProf."""
    del args  # Unused: keep naming strictly rank-symmetric and shape-heuristic-free.
    if token_bucket is None:
        token_bucket = getattr(_compile_context, "token_bucket", None)
    if token_bucket is None:
        token_bucket = _fallback_token_bucket

    root_cls = ""
    decoder_model_cls = ""
    first_cls = ""
    layer_ids: list[int] = []
    layer_subblocks: set[str] = set()
    top_blocks: list[str] = []
    op_names: list[str] = []

    for node in gm.graph.nodes:
        if node.op in ("placeholder", "get_attr", "output"):
            continue
        op_names.append(_fx_node_leaf_op_name(node))
        entries = _iter_nn_module_entries(node.meta.get("nn_module_stack"))
        if entries and not first_cls and entries[0][1]:
            first_cls = entries[0][1]
        for fqn, cls_name in entries:
            if not fqn:
                if cls_name and not root_cls:
                    root_cls = cls_name
                continue
            m = _LAYER_RE.search(fqn)
            if m is not None:
                layer_ids.append(int(m.group(1)))
                if m.group(2):
                    layer_subblocks.add(m.group(2))
                elif cls_name and not decoder_model_cls:
                    if cls_name.endswith("DecoderLayer") and len(cls_name) > len(
                        "DecoderLayer"
                    ):
                        decoder_model_cls = f"{cls_name[:-len('DecoderLayer')]}Model"
                    else:
                        decoder_model_cls = cls_name
            else:
                top_seg = fqn.split(".", 1)[0]
                if top_seg and top_seg not in ("model", "language_model"):
                    if top_seg not in top_blocks:
                        top_blocks.append(top_seg)

    model_cls = root_cls or decoder_model_cls or first_cls or "VllmTpuModel"
    parts: list[str] = [model_cls]
    if token_bucket is not None and token_bucket > 0:
        parts.append(f"T{token_bucket}")

    if layer_ids:
        min_l, max_l = min(layer_ids), max(layer_ids)
        if min_l == max_l:
            parts.append(f"L{min_l:02d}")
            if len(layer_subblocks) == 1:
                parts.append(next(iter(layer_subblocks)))
        else:
            parts.append(f"L{min_l:02d}_to_L{max_l:02d}")
    elif top_blocks:
        parts.append("_".join(top_blocks[:2]))

    if op_names and all(op in _COLLECTIVE_OP_NAMES for op in op_names):
        primary_collectives = [
            op for op in op_names if op not in ("wait_tensor", "getitem")
        ]
        if primary_collectives:
            parts.append(primary_collectives[0])

    if submod_name:
        parts.append(submod_name)

    return sanitize_mlir_symbol("_".join(parts))


class _SynthesizedLocBuilder:
    """Deduplicating builder for `#loc_vs_<N>` MLIR debug location aliases."""

    def __init__(self) -> None:
        self._counter = 0
        self._defs: list[str] = []
        self._intern: dict[str, str] = {}

    def _emit(self, rhs: str) -> str:
        existing = self._intern.get(rhs)
        if existing is not None:
            return existing
        loc_id = f"#loc_vs_{self._counter}"
        self._counter += 1
        self._intern[rhs] = loc_id
        self._defs.append(f"{loc_id} = loc({rhs})")
        return loc_id

    def build_stack(
        self,
        frames: Sequence[tuple[str, int, str]],
        unknown_loc_id: str,
    ) -> str:
        if not frames:
            return unknown_loc_id
        cur: str | None = None
        for filepath, lineno, funcname in frames:
            file_rhs = f'"{_escape_mlir_string(filepath)}":{lineno}:0'
            file_loc = self._emit(file_rhs)
            fn_rhs = f'"{_escape_mlir_string(funcname)}"({file_loc})'
            fn_loc = self._emit(fn_rhs)
            if cur is None:
                cur = fn_loc
            else:
                cur = self._emit(f"callsite({fn_loc} at {cur})")
        return cur if cur is not None else unknown_loc_id

    @property
    def lines(self) -> list[str]:
        return self._defs


def _apply_header_rewrites(
    body_lines: list[str],
    module_name: str | None,
    donated_inputs: Sequence[int] | None,
) -> None:
    if not body_lines:
        return

    if module_name:
        safe_mod = sanitize_mlir_symbol(module_name)
        body_lines[0] = re.sub(
            r'^(\s*module)(?:\s+@(?:[^\s"{]+|"(?:[^"\\]|\\.)*"))?(\s*(?:attributes\b.*)?\{)',
            rf"\1 @{safe_mod}\2",
            body_lines[0],
            count=1,
        )

    if not donated_inputs:
        return

    donor_indices = sorted({int(i) for i in donated_inputs if int(i) >= 0})
    if not donor_indices:
        return

    for idx, line in enumerate(body_lines):
        if "func.func" not in line or "@main(" not in line:
            continue
        updated = line
        for arg_idx in donor_indices:
            arg_pat = re.compile(
                rf"(%arg{arg_idx}:\s*tensor<[^>]+>)(\s*\{{[^}}]*\}})?(\s+loc\()"
            )

            def _inject_donor(match: re.Match[str]) -> str:
                prefix, attrs, loc_suffix = (
                    match.group(1),
                    match.group(2),
                    match.group(3),
                )
                if attrs:
                    if "jax.buffer_donor" in attrs:
                        return match.group(0)
                    merged = attrs[:-1].rstrip() + ", jax.buffer_donor = true}"
                    return f"{prefix}{merged}{loc_suffix}"
                return f"{prefix} {{jax.buffer_donor = true}}{loc_suffix}"

            updated = arg_pat.sub(_inject_donor, updated, count=1)
        body_lines[idx] = updated
        break


def rewrite_mlir_debug_locations(
    mlir_text: str,
    marker_table: Mapping[int, NodeAnnotation],
    *,
    module_name: str | None = None,
    donated_inputs: Sequence[int] | None = None,
) -> str:
    """Rewrite StableHLO debug locations with FX module scopes and user stacks."""
    trailing_newline = mlir_text.endswith("\n")
    lines = mlir_text.splitlines()
    if not lines:
        return mlir_text

    module_start = next(
        (i for i, line in enumerate(lines) if line.lstrip().startswith("module")),
        None,
    )
    if module_start is None:
        return mlir_text

    # Locate contiguous `#loc... = loc(...)` definitions at the footer.
    end_nonempty = len(lines)
    while end_nonempty > module_start and not lines[end_nonempty - 1].strip():
        end_nonempty -= 1

    footer_start = end_nonempty
    while (
        footer_start > module_start
        and _LOC_DEF_RE.match(lines[footer_start - 1]) is not None
    ):
        footer_start -= 1

    header_lines = list(lines[:module_start])
    body_lines = list(lines[module_start:footer_start])
    footer_lines = list(lines[footer_start:end_nonempty])

    _apply_header_rewrites(body_lines, module_name, donated_inputs)

    if not marker_table or not footer_lines:
        out = "\n".join(header_lines + body_lines + footer_lines)
        return f"{out}\n" if trailing_newline else out

    # Ensure an unknown location alias is available in the header.
    unknown_loc_id: str | None = None
    for line in header_lines:
        m = _LOC_DEF_RE.match(line)
        if m is not None and m.group(2).strip() == "unknown":
            unknown_loc_id = m.group(1)
            break
    if unknown_loc_id is None:
        unknown_loc_id = "#loc_vllm_unk"
        header_lines.insert(0, f"{unknown_loc_id} = loc(unknown)")

    loc_defs: list[tuple[str, str]] = []
    file_marker_fx: dict[str, int] = {}
    fn_marker_fx: dict[str, int] = {}
    noisy_file_locs: set[str] = set()
    noisy_fn_locs: set[str] = set()
    nameloc_info: dict[str, tuple[str, str | None]] = {}
    callsite_info: dict[str, tuple[str, str]] = {}
    used_in_callsites: set[str] = set()
    loc_to_fx_idx: dict[str, int] = {}

    for line in footer_lines:
        m_def = _LOC_DEF_RE.match(line)
        if m_def is None:
            continue
        loc_id, rhs = m_def.group(1), m_def.group(2).strip()
        loc_defs.append((loc_id, rhs))

        m_file = _FILE_LOC_RE.match(rhs)
        if m_file is not None:
            filename = _unescape_mlir_string(m_file.group(1))
            m_marker = _MARKER_FILE_RE.match(filename)
            if m_marker is not None:
                fx_idx = int(m_marker.group(1))
                file_marker_fx[loc_id] = fx_idx
                loc_to_fx_idx[loc_id] = fx_idx
            elif _is_noisy_stack_file(filename):
                noisy_file_locs.add(loc_id)
            continue

        m_nl_child = _NAMELOC_CHILD_RE.match(rhs)
        if m_nl_child is not None:
            raw_name = _unescape_mlir_string(m_nl_child.group(1))
            child_id = m_nl_child.group(2)
            nameloc_info[loc_id] = (raw_name, child_id)
            if child_id in file_marker_fx:
                fx_idx = file_marker_fx[child_id]
                fn_marker_fx[loc_id] = fx_idx
                loc_to_fx_idx[loc_id] = fx_idx
            elif child_id in noisy_file_locs:
                noisy_fn_locs.add(loc_id)
            elif child_id in loc_to_fx_idx:
                loc_to_fx_idx[loc_id] = loc_to_fx_idx[child_id]
            continue

        m_nl_bare = _NAMELOC_BARE_RE.match(rhs)
        if m_nl_bare is not None:
            nameloc_info[loc_id] = (_unescape_mlir_string(m_nl_bare.group(1)), None)
            continue

        m_cs = _CALLSITE_RE.match(rhs)
        if m_cs is not None:
            callee_id, caller_id = m_cs.group(1), m_cs.group(2)
            callsite_info[loc_id] = (callee_id, caller_id)
            used_in_callsites.add(callee_id)
            used_in_callsites.add(caller_id)
            if callee_id in loc_to_fx_idx:
                loc_to_fx_idx[loc_id] = loc_to_fx_idx[callee_id]
            elif caller_id in loc_to_fx_idx:
                loc_to_fx_idx[loc_id] = loc_to_fx_idx[caller_id]
            continue

        for ref_id in _LOC_REF_RE.findall(rhs):
            if ref_id in loc_to_fx_idx:
                loc_to_fx_idx[loc_id] = loc_to_fx_idx[ref_id]
                break

    if not loc_to_fx_idx:
        out = "\n".join(header_lines + body_lines + footer_lines)
        return f"{out}\n" if trailing_newline else out

    # Collect `#loc...` aliases directly attached to ops inside the module body.
    used_on_ops: set[str] = set()
    for line in body_lines:
        stripped = line.lstrip()
        if stripped.startswith(("module ", "module{", "func.func ", "return ", "}")):
            continue
        for m_op_loc in _BODY_LOC_RE.finditer(line):
            used_on_ops.add(m_op_loc.group(1))

    # Synthesize deduplicated user stack chains at the top of the footer so all
    # forward references from `#loc1..#locM` satisfy MLIR's topological order.
    synth_builder = _SynthesizedLocBuilder()
    fx_stack_loc: dict[int, str] = {}
    for fx_idx in sorted(set(loc_to_fx_idx.values())):
        ann = marker_table.get(fx_idx)
        frames = ann.frames if ann is not None else ()
        fx_stack_loc[fx_idx] = synth_builder.build_stack(frames, unknown_loc_id)

    remap: dict[str, str] = {}
    rewritten_defs: list[str] = []

    for loc_id, rhs in loc_defs:
        if loc_id in file_marker_fx:
            remap[loc_id] = fx_stack_loc.get(file_marker_fx[loc_id], unknown_loc_id)
            continue
        if loc_id in fn_marker_fx:
            remap[loc_id] = fx_stack_loc.get(fn_marker_fx[loc_id], unknown_loc_id)
            continue
        if loc_id in noisy_file_locs or loc_id in noisy_fn_locs:
            remap[loc_id] = unknown_loc_id
            continue

        if loc_id in callsite_info:
            callee_id, caller_id = callsite_info[loc_id]
            if callee_id in fn_marker_fx:
                remap[loc_id] = fx_stack_loc.get(
                    fn_marker_fx[callee_id], unknown_loc_id
                )
                continue
            new_callee = remap.get(callee_id, callee_id)
            new_caller = remap.get(caller_id, caller_id)
            if callee_id in noisy_fn_locs or new_callee == unknown_loc_id:
                remap[loc_id] = new_caller
                continue
            if new_caller == unknown_loc_id:
                remap[loc_id] = new_callee
                continue
            rewritten_defs.append(
                f"{loc_id} = loc(callsite({new_callee} at {new_caller}))"
            )
            continue

        if (
            loc_id in nameloc_info
            and loc_id in used_on_ops
            and loc_id not in used_in_callsites
        ):
            orig_name, child_id = nameloc_info[loc_id]
            new_child = (
                remap.get(child_id, child_id)
                if child_id is not None
                else unknown_loc_id
            )
            fx_idx_opt = loc_to_fx_idx.get(loc_id)
            ann = marker_table.get(fx_idx_opt) if fx_idx_opt is not None else None
            if ann is not None:
                leaf_op = clean_op_name(orig_name, fallback=ann.op_name)
                scoped_name = (
                    f"{ann.scope_path}/{leaf_op}" if ann.scope_path else leaf_op
                )
            else:
                scoped_name = clean_op_name(orig_name, fallback=orig_name)
            escaped_scoped = _escape_mlir_string(scoped_name)
            if new_child == unknown_loc_id:
                rewritten_defs.append(f'{loc_id} = loc("{escaped_scoped}")')
            else:
                rewritten_defs.append(
                    f'{loc_id} = loc("{escaped_scoped}"({new_child}))'
                )
            continue

        if loc_id in nameloc_info:
            orig_name, child_id = nameloc_info[loc_id]
            new_child = (
                remap.get(child_id, child_id)
                if child_id is not None
                else unknown_loc_id
            )
            escaped_orig = _escape_mlir_string(orig_name)
            if new_child == unknown_loc_id:
                rewritten_defs.append(f'{loc_id} = loc("{escaped_orig}")')
            else:
                rewritten_defs.append(
                    f'{loc_id} = loc("{escaped_orig}"({new_child}))'
                )
            continue

        updated_rhs = _LOC_REF_RE.sub(
            lambda m: remap.get(m.group(0), m.group(0)), rhs
        )
        rewritten_defs.append(f"{loc_id} = loc({updated_rhs})")

    # Any op in `body_lines` whose location was not a standalone NameLoc (for
    # example, a custom op pointing directly at a callsite) gets a fresh scoped
    # NameLoc wrapper appended at the end of the footer.
    wrapped_op_locs: dict[str, str] = {}
    extra_op_defs: list[str] = []
    op_counter = 0

    for loc_id in sorted(used_on_ops):
        is_standalone_nameloc = (
            loc_id in nameloc_info and loc_id not in used_in_callsites
        )
        if is_standalone_nameloc:
            continue
        fx_idx_opt = loc_to_fx_idx.get(loc_id)
        ann = marker_table.get(fx_idx_opt) if fx_idx_opt is not None else None
        resolved_target = remap.get(loc_id, loc_id)
        if ann is not None and ann.scope_path:
            scoped_name = _escape_mlir_string(f"{ann.scope_path}/{ann.op_name}")
            new_op_loc = f"#loc_vop_{op_counter}"
            op_counter += 1
            if resolved_target == unknown_loc_id:
                extra_op_defs.append(f'{new_op_loc} = loc("{scoped_name}")')
            else:
                extra_op_defs.append(
                    f'{new_op_loc} = loc("{scoped_name}"({resolved_target}))'
                )
            wrapped_op_locs[loc_id] = new_op_loc
        elif resolved_target != loc_id:
            wrapped_op_locs[loc_id] = resolved_target

    new_body_lines: list[str] = []
    for line in body_lines:
        stripped = line.lstrip()
        if stripped.startswith(("module ", "module{", "func.func ", "return ", "}")):
            if remap:
                line = _BODY_LOC_RE.sub(
                    lambda m: f"loc({remap.get(m.group(1), m.group(1))})", line
                )
            new_body_lines.append(line)
            continue
        if wrapped_op_locs or remap:
            line = _BODY_LOC_RE.sub(
                lambda m: f"loc({wrapped_op_locs.get(m.group(1), remap.get(m.group(1), m.group(1)))})",
                line,
            )
        new_body_lines.append(line)

    all_lines = (
        header_lines
        + new_body_lines
        + synth_builder.lines
        + rewritten_defs
        + extra_op_defs
    )
    out = "\n".join(all_lines)
    return f"{out}\n" if trailing_newline else out


def install_xprof_annotation_hooks() -> None:
    """Idempotently install TorchTPU XProf annotation hooks."""
    global _hooks_installed
    if _hooks_installed:
        return

    with _hooks_lock:
        if _hooks_installed:
            return

        try:
            import torch_tpu._internal.export as tt_export_pkg
            from torch_tpu._internal import execution_mode
            from torch_tpu._internal.benchmarks import xprof_adapter
            from torch_tpu._internal.compile import split_compiler, tpu_torch_compile
            from torch_tpu._internal.export import export as tt_export_mod
        except ImportError:
            logger.debug(
                "[xprof_annotations] torch_tpu internals unavailable; skipping hooks."
            )
            return

        orig_fx_to_mlir = tt_export_mod.fx_to_mlir
        orig_compile_submod = split_compiler._SubmodCompiler.compile_submod

        class _AnnotatedFxInterpreter(tt_export_mod.EagerLikeFxInterpreter):
            def __init__(self, module: fx.GraphModule) -> None:
                super().__init__(module)
                self._node_annotations = build_propagated_node_annotations(module)
                self.marker_table: dict[int, NodeAnnotation] = {}
                self._next_idx = 0

            def run_node(self, node: fx.Node) -> Any:
                if node.op in ("placeholder", "get_attr", "output"):
                    return super(
                        tt_export_mod.EagerLikeFxInterpreter, self
                    ).run_node(node)

                idx = self._next_idx
                self._next_idx += 1
                ann = self._node_annotations.get(node)
                if ann is not None:
                    self.marker_table[idx] = ann

                def dispatch_fx_node(n: fx.Node) -> Any:
                    return super(
                        tt_export_mod.EagerLikeFxInterpreter, self
                    ).run_node(n)

                try:
                    dispatch_fx_node.__code__ = dispatch_fx_node.__code__.replace(
                        co_filename=f"__vllm_xprof_fx_{idx}__",
                        co_firstlineno=1,
                        co_name=f"fx_{idx}",
                    )
                except Exception as exc:
                    logger.debug(
                        "[xprof_annotations] Could not tag FX node %s: %s",
                        node.name,
                        exc,
                    )

                return dispatch_fx_node(node)

        def _annotated_compile_submod(
            self: Any,
            input_mod: fx.GraphModule,
            args: list[torch.Tensor],
            kwargs: Any,
            submod_name: str | None = None,
        ) -> Any:
            if not envs.VLLM_TPU_XPROF_ANNOTATIONS or self.base_module_name is not None:
                return orig_compile_submod(
                    self, input_mod, args, kwargs, submod_name=submod_name
                )
            derived_base = derive_xla_module_name(input_mod, args, submod_name=None)
            self.base_module_name = derived_base
            try:
                return orig_compile_submod(
                    self, input_mod, args, kwargs, submod_name=submod_name
                )
            finally:
                self.base_module_name = None

        def _annotated_fx_to_mlir(
            module: fx.GraphModule,
            args: list[torch.Tensor | Any],
            build_mlir_module: bool = True,
            use_stablehlo_bounds: bool = False,
            argument_layouts: list[list[int]] | None = None,
            dynamic_outputs: Sequence[bool] | None = None,
            donated_inputs: Sequence[int] | None = None,
            module_name: str | None = None,
        ) -> Any:
            if not envs.VLLM_TPU_XPROF_ANNOTATIONS:
                return orig_fx_to_mlir(
                    module,
                    args,
                    build_mlir_module=build_mlir_module,
                    use_stablehlo_bounds=use_stablehlo_bounds,
                    argument_layouts=argument_layouts,
                    dynamic_outputs=dynamic_outputs,
                    donated_inputs=donated_inputs,
                    module_name=module_name,
                )

            resolved_module_name = module_name or derive_xla_module_name(
                module, args, submod_name=None
            )

            with xprof_adapter.TraceMe("export.fx_to_mlir.annotated"):
                argument_tensors = [a for a in args if isinstance(a, torch.Tensor)]
                if argument_layouts is not None:
                    assert len(argument_layouts) == len(argument_tensors), (
                        f"argument_layouts size mismatch: expected "
                        f"{len(argument_tensors)}, got {len(argument_layouts)}"
                    )
                    internal_layouts = list(argument_layouts)
                else:
                    internal_layouts = []

                with tt_export_mod._fx_to_mlir_lock:
                    device = torch.accelerator.current_accelerator()
                    device_module = getattr(torch, device.type)
                    default_device_index = device.index or 0
                    default_generator = device_module.default_generators[
                        default_device_index
                    ]
                    argument_generators = [
                        arg for arg in args if isinstance(arg, torch.Generator)
                    ]
                    has_implicit_default_generator = (
                        default_generator not in argument_generators
                    )
                    if has_implicit_default_generator:
                        argument_generators.append(default_generator)

                    export_mode_generators = {
                        gen: tpu_torch_compile.make_export_mode_device_generator(gen)
                        for gen in argument_generators
                    }
                    tpu_torch_compile.set_export_mode_default_device_generator(
                        export_mode_generators[default_generator]
                    )

                    for gen in argument_generators:
                        argument_tensors.append(
                            tpu_torch_compile.get_device_state_tensor(
                                export_mode_generators[gen]
                            )
                        )
                        if internal_layouts:
                            internal_layouts.append([])

                    tpu_torch_compile.push_enable_tracebacks(True)
                    try:
                        with execution_mode.set_eager_mode(
                            execution_mode.EagerMode._INTERNAL_COMPILE_FX_GRAPH
                        ):
                            mapped_args = [
                                export_mode_generators[x]
                                if isinstance(x, torch.Generator)
                                else (x.clone() if isinstance(x, torch.Tensor) else x)
                                for x in args
                            ]
                            interpreter = _AnnotatedFxInterpreter(module)
                            with xprof_adapter.TraceMe("EagerLikeFxInterpreter.run"):
                                fx_outputs = interpreter.run(*mapped_args)

                            (
                                result_tensors,
                                reconstruct_fx_outputs_fn,
                                deduped_dynamic_outputs,
                                unique_output_indices,
                            ) = tt_export_mod._process_fx_outputs(
                                module, fx_outputs, dynamic_outputs=dynamic_outputs
                            )

                            result_tensors.extend(
                                tpu_torch_compile.get_device_state_tensor(
                                    export_mode_generators[gen]
                                )
                                for gen in argument_generators
                            )

                            if has_implicit_default_generator:
                                begin_default_state_tensor = argument_tensors[-1]
                                end_default_state_tensor = result_tensors[-1]
                                updates_default_generator_state = (
                                    begin_default_state_tensor.data_ptr()
                                    != end_default_state_tensor.data_ptr()
                                )
                                if not updates_default_generator_state:
                                    if len(result_tensors) > 1:
                                        argument_tensors.pop()
                                        result_tensors.pop()
                                        if internal_layouts:
                                            internal_layouts.pop()
                                    else:
                                        return tt_export_mod.ExportedMlir(
                                            module=None,
                                            executable=None,
                                            mlir_result_tensors=[],
                                            reconstruct_fx_outputs_fn=reconstruct_fx_outputs_fn,
                                            updates_default_generator_state=False,
                                            is_noop=True,
                                            dynamic_outputs=[],
                                            unique_output_indices=[],
                                        )
                            else:
                                updates_default_generator_state = False

                            raw_mod = tpu_torch_compile.build_mlir(
                                result_tensors,
                                argument_tensors,
                                use_stablehlo_bounds,
                            )
                            raw_mlir_text = tpu_torch_compile.serialize_mlir_text(
                                raw_mod, enable_debug_info=True
                            )
                    finally:
                        tpu_torch_compile.pop_enable_tracebacks()
                        tpu_torch_compile.clear_export_mode_default_device_generators()

                target_mod = raw_mod
                try:
                    annotated_mlir_text = rewrite_mlir_debug_locations(
                        raw_mlir_text,
                        interpreter.marker_table,
                        module_name=resolved_module_name,
                        donated_inputs=donated_inputs,
                    )
                    with execution_mode.set_eager_mode(
                        execution_mode.EagerMode._INTERNAL_COMPILE_FX_GRAPH
                    ):
                        target_mod = tpu_torch_compile.parse_mlir_text(
                            annotated_mlir_text
                        )
                except Exception as exc:
                    logger.warning(
                        "[xprof_annotations] Falling back to unannotated MLIR "
                        "for module %s: %s",
                        resolved_module_name,
                        exc,
                    )
                    return orig_fx_to_mlir(
                        module,
                        args,
                        build_mlir_module=build_mlir_module,
                        use_stablehlo_bounds=use_stablehlo_bounds,
                        argument_layouts=argument_layouts,
                        dynamic_outputs=dynamic_outputs,
                        donated_inputs=donated_inputs,
                        module_name=resolved_module_name,
                    )

                with execution_mode.set_eager_mode(
                    execution_mode.EagerMode._INTERNAL_COMPILE_FX_GRAPH
                ):
                    compiled_exe = tpu_torch_compile.compile_mlir(
                        target_mod, use_stablehlo_bounds, internal_layouts
                    )

                return tt_export_mod.ExportedMlir(
                    module=target_mod if build_mlir_module else None,
                    executable=compiled_exe,
                    mlir_result_tensors=result_tensors,
                    reconstruct_fx_outputs_fn=reconstruct_fx_outputs_fn,
                    updates_default_generator_state=updates_default_generator_state,
                    dynamic_outputs=deduped_dynamic_outputs,
                    unique_output_indices=unique_output_indices,
                )

        split_compiler._SubmodCompiler.compile_submod = _annotated_compile_submod
        tt_export_mod.fx_to_mlir = _annotated_fx_to_mlir
        tt_export_pkg.fx_to_mlir = _annotated_fx_to_mlir
        _hooks_installed = True
        logger.info("[xprof_annotations] Installed TorchTPU XProf annotation hooks.")
