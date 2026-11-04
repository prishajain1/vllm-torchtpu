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

"""Read Kimi K3 checkpoint assets in the TP32 megakernel layout.

This module reads the tokenizer and Hugging Face checkpoint tensors and
describes the kernel's per-rank weight layout: ``Checkpoint`` slices each
family for a rank, which the pre-shard tool (``..tools preshard``)
writes out and ``rank_loader`` streams onto the TPU.
"""

from __future__ import annotations

import ast
import base64
import dataclasses
import importlib
import json
import os
import struct
import typing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

ml_dtypes: Any = importlib.import_module("ml_dtypes")

_Dynamic = Any
_CONST_2 = 2
_CONST_255 = 255
_CONST_28 = 28
_CONST_93 = 93


@dataclasses.dataclass(frozen=True)
class Config:
    """Manage config instance and state."""

    dim: int = 7168
    latent: int = 3584
    expert_hidden: int = 3072
    shared_hidden: int = 6144
    dense_hidden: int = 33792
    experts: int = 896
    top_k: int = 16
    layers: int = 93
    heads: int = 96
    head_dim: int = 128
    q_rank: int = 1536
    kv_rank: int = 512
    qk_extra: int = 64
    vocab: int = 163840
    conv_size: int = 4
    residual_block: int = 12
    eps: float = 1e-5
    situ_beta: float = 4.0
    situ_linear_beta: float = 25.0
    gate_lower_bound: float = -5.0
    full_attention: tuple[int, ...] = (*tuple(range(3, 92, 4)), 92)

    @classmethod
    def from_checkpoint(cls, directory: _Dynamic) -> _Dynamic:
        """Execute from checkpoint logic."""
        raw = json.loads((Path(directory) / "config.json").read_text())
        c = raw.get("text_config", raw)
        a = c["linear_attn_config"]
        required = {
            "mla_use_nope": True,
            "mla_use_output_gate": True,
            "latent_moe_use_norm": True,
            "moe_renormalize": True,
            "moe_router_activation_func": "sigmoid",
            "hidden_act": "situ",
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "num_expert_group": 1,
            "routed_scaling_factor": 1.0,
        }
        for name, expected in required.items():
            if c[name] != expected:
                _err_msg = f"Unsupported K3 setting {name}={c[name]!r}"
                raise ValueError(_err_msg)
        if not a["use_full_rank_gate"]:
            _err_msg = "K3 requires the full-rank output gate"
            raise ValueError(_err_msg)
        return cls(
            dim=c["hidden_size"],
            latent=c["routed_expert_hidden_size"],
            expert_hidden=c["moe_intermediate_size"],
            shared_hidden=c["moe_intermediate_size"] * c["num_shared_experts"],
            dense_hidden=c["intermediate_size"],
            experts=c["num_experts"],
            top_k=c["num_experts_per_token"],
            layers=c["num_hidden_layers"],
            heads=c["num_attention_heads"],
            head_dim=a["head_dim"],
            q_rank=c["q_lora_rank"],
            kv_rank=c["kv_lora_rank"],
            qk_extra=c["qk_rope_head_dim"],
            vocab=c["vocab_size"],
            conv_size=a["short_conv_kernel_size"],
            residual_block=c["attn_res_block_size"],
            eps=c["rms_norm_eps"],
            situ_beta=c["activation_situ_beta"],
            situ_linear_beta=c["activation_situ_linear_beta"],
            gate_lower_bound=a["gate_lower_bound"],
            full_attention=tuple(i - 1 for i in a["full_attn_layers"]),
        )


BF16 = np.dtype(ml_dtypes.bfloat16)
PREFIX = "language_model.model."
ALL = slice(None)  # a whole axis


def load_tokenizer(directory: _Dynamic) -> _Dynamic:
    """Read the checkpoint's tiktoken ranks, regex, and reserved-token mapping.

    Parse only the literal regex list from its tokenizer source; loading a
    tokenizer never executes checkpoint Python code or imports Transformers.
    """
    tiktoken: Any = importlib.import_module("tiktoken")

    directory = Path(directory)
    config = json.loads((directory / "tokenizer_config.json").read_text())
    source = ast.parse((directory / "tokenization_kimi.py").read_text())
    cls = next(
        n
        for n in source.body
        if isinstance(n, ast.ClassDef) and n.name == "TikTokenTokenizer"
    )
    assignment = next(
        n
        for n in cls.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "pat_str" for t in n.targets)
    )
    value = assignment.value
    if not (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and isinstance(value.func.value, ast.Constant)
        and value.func.value.value == "|"
        and value.func.attr == "join"
        and len(value.args) == 1
    ):
        _err_msg = "Unsupported checkpoint tokenizer pattern declaration"
        raise ValueError(_err_msg)
    pattern = "|".join(ast.literal_eval(value.args[0]))
    ranks = {
        base64.b64decode(token): int(rank)
        for token, rank in (
            line.split()
            for line in (directory / "tiktoken.model").read_bytes().splitlines()
        )
    }
    added = config["added_tokens_decoder"]
    special = {
        added.get(str(i), {}).get("content", f"<|reserved_token_{i}|>"): i
        for i in range(len(ranks), len(ranks) + 256)
    }
    return tiktoken.Encoding(
        name="kimi-k3-checkpoint",
        pat_str=pattern,
        mergeable_ranks=ranks,
        special_tokens=special,
    )


def pack_expert_bytes(value: _Dynamic) -> _Dynamic:
    """Checkpoint [N,K/2] bytes -> kernel [K/8,N] little-endian nibble groups."""
    if value.dtype != np.uint8 or value.ndim != _CONST_2 or value.shape[1] % 4:
        _err_msg = "Expected whole uint32 groups of packed FP4 bytes"
        raise ValueError(_err_msg)
    return np.ascontiguousarray(value).view("<u4").T.copy()


class Checkpoint:
    """Manage checkpoint instance and state."""

    def __init__(self, path: _Dynamic) -> None:
        """Execute init logic."""
        self.path = Path(path)
        self.config = Config.from_checkpoint(path)
        if self.config != Config():
            _err_msg = "This loader requires the full 93-layer/896-expert K3 contract"
            raise ValueError(_err_msg)
        self.index = json.loads(
            (self.path / "model.safetensors.index.json").read_text()
        )["weight_map"]
        missing = [f for f in set(self.index.values()) if not (self.path / f).is_file()]
        if missing:
            _err_msg = f"Checkpoint is missing {len(missing)} tensor shards"
            raise FileNotFoundError(_err_msg)

        self._layouts = {}

    # Coalesced streaming reads: a rank's expert tensors are grouped into
    # contiguous ranges (merging gaps up to COALESCE_GAP) and each range is
    # streamed on its own handle, ranges in parallel: seeks and handles shared
    # across threads are slow on network storage.
    COALESCE_GAP = 16 << 20
    CHUNK = 16 << 20
    READ_THREADS = 8

    def _layout(self, filename: _Dynamic) -> _Dynamic:
        """(header dict, data start) of one shard, cached."""
        if filename not in self._layouts:
            with Path(self.path / filename).open("rb") as f:
                size = struct.unpack("<Q", f.read(8))[0]
                self._layouts[filename] = (json.loads(f.read(size)), 8 + size)
        return self._layouts[filename]

    def read_many(self, keys: _Dynamic) -> _Dynamic:
        """Whole tensors for ``keys`` as uint8-viewable numpy arrays,.

        via coalesced reads.
        Returns ``{key: array}`` with the header dtype/shape applied. Used for
        the expert families.
        """
        by_file = {}
        for key in keys:
            by_file.setdefault(self.index[key], []).append(key)
        out = {}
        for filename, file_keys in by_file.items():
            header, base = self._layout(filename)
            spans = sorted(
                (header[k]["data_offsets"][0], header[k]["data_offsets"][1], k)
                for k in file_keys
            )
            ranges = []
            for start, end, key in spans:
                if ranges and start - ranges[-1][1] <= self.COALESCE_GAP:
                    ranges[-1][1] = max(ranges[-1][1], end)
                    ranges[-1][2].append(key)
                else:
                    ranges.append([start, end, [key]])
            path = self.path / filename

            def stream(
                bounds: _Dynamic, path: _Dynamic = path, base: _Dynamic = base
            ) -> _Dynamic:
                """Execute stream logic."""
                start, end, _ = bounds
                fd = os.open(path, os.O_RDONLY)
                try:
                    pieces = [
                        os.pread(fd, min(self.CHUNK, end - offset), base + offset)
                        for offset in range(start, end, self.CHUNK)
                    ]
                finally:
                    os.close(fd)
                return b"".join(pieces)

            with ThreadPoolExecutor(min(self.READ_THREADS, len(ranges))) as pool:
                buffers = dict(enumerate(pool.map(stream, ranges)))
            for r, (start, end, range_keys) in enumerate(ranges):
                buffer = np.frombuffer(buffers[r], np.uint8)
                for key in range_keys:
                    meta = header[key]
                    a, b = meta["data_offsets"]
                    out[key] = (
                        buffer[a - start : b - start]
                        .view(self._DTYPES[meta["dtype"]])
                        .reshape(meta["shape"])
                    )
        return out

    @staticmethod
    def _expert_parts(
        packed: _Dynamic,
        scale: _Dynamic,
        down: _Dynamic,
        part: _Dynamic,
        parts: _Dynamic,
    ) -> _Dynamic:
        """Execute expert parts logic."""
        if part is None:
            return packed, scale
        if part not in range(parts) or 3072 % parts:
            _err_msg = f"Expert TP part must be in 0..{parts - 1}"
            raise ValueError(_err_msg)
        width = 3072 // parts
        if down:
            return (
                packed[:, part * width // 2 : (part + 1) * width // 2],
                scale[:, part * width // 32 : (part + 1) * width // 32],
            )
        return packed[part * width : (part + 1) * width], scale[
            part * width : (part + 1) * width
        ]

    def experts(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
        """``(gate_up, gate_up_scales, down, down_scales)`` for ``count``.

        consecutive experts.
        Each element is a list over experts in kernel layout (gate/up packed
        ``[K/8, N]`` with E8M0 scales ``[K/32, N]``, see :func:`pack_expert_bytes`);
        all tensors of the layer's experts are fetched with :meth:`read_many`.
        """
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            layer,
            first,
            count,
        ) = (
            _p_next(_p_iter, _p_get("layer")),
            _p_next(_p_iter, _p_get("first")),
            _p_next(_p_iter, _p_get("count")),
        )
        (
            part,
            parts,
        ) = (
            _p_kw.get("part"),
            _p_kw.get("parts", 4),
        )
        base = f"{PREFIX}layers.{layer}.block_sparse_moe.experts."
        keys = [
            f"{base}{first + e}.{name}.{kind}"
            for e in range(count)
            for name in ("w1", "w2", "w3")
            for kind in ("weight_packed", "weight_scale")
        ]
        tensors = self.read_many(keys)
        gate_up, gate_up_scales, downs, down_scales = [], [], [], []
        for e in range(count):
            prefix = f"{base}{first + e}."
            weights, scales = [], []
            for name in ("w1", "w3"):
                packed, scale = self._expert_parts(
                    tensors[prefix + name + ".weight_packed"],
                    tensors[prefix + name + ".weight_scale"],
                    bool(0),
                    part,
                    parts,
                )
                if scale.dtype != np.uint8 or np.any(
                    (scale == 0) | (scale == _CONST_255)
                ):
                    _err_msg = f"{prefix}{name}: expected normal, finite E8M0 scales"
                    raise ValueError(_err_msg)
                weights.append(pack_expert_bytes(packed))
                scales.append(scale.T.copy())
            gate_up.append(np.concatenate(weights, axis=1))
            gate_up_scales.append(np.concatenate(scales, axis=1))
            packed, scale = self._expert_parts(
                tensors[prefix + "w2.weight_packed"],
                tensors[prefix + "w2.weight_scale"],
                bool(1),
                part,
                parts,
            )
            if scale.dtype != np.uint8 or np.any((scale == 0) | (scale == _CONST_255)):
                _err_msg = f"{prefix}w2: expected normal, finite E8M0 scales"
                raise ValueError(_err_msg)
            downs.append(pack_expert_bytes(packed))
            down_scales.append(scale.T.copy())
        return gate_up, gate_up_scales, downs, down_scales

    _DTYPES: typing.ClassVar[dict[str, _Dynamic]] = {
        "U8": np.uint8,
        "BF16": BF16,
        "F32": np.float32,
        "F16": np.float16,
    }

    def read(self, key: _Dynamic, selection: _Dynamic = None) -> _Dynamic:
        """One tensor, or ``tensor[selection]``, streamed from the shard.

        A leading-axis slice (``(rows, ...full...)`` or a bare slice) is a
        contiguous byte range and is streamed as such; any other selection
        reads the whole tensor and slices in memory. Both avoid small,
        non-contiguous reads, which can be slow on network-backed storage.
        """
        filename = self.index[key]
        header, base = self._layout(filename)
        meta = header[key]
        shape = list(meta["shape"])
        dtype = self._DTYPES[meta["dtype"]]
        start, end = meta["data_offsets"]
        rows = slice(None)
        rest = ()
        if selection is not None:
            parts = selection if isinstance(selection, tuple) else (selection,)
            leading, rest = parts[0], parts[1:]
            if isinstance(leading, slice) and all(
                isinstance(r, slice) and r == slice(None) for r in rest
            ):
                rows, rest = leading, ()
            else:
                rest = parts
        r0, r1, step = rows.indices(shape[0])
        if step != 1:
            _err_msg = "Only unit-step leading slices are supported"
            raise ValueError(_err_msg)
        row_bytes = (end - start) // shape[0] if shape else end - start
        offset = base + start + r0 * row_bytes
        length = (r1 - r0) * row_bytes if shape else end - start
        fd = os.open(self.path / filename, os.O_RDONLY)
        try:
            pieces = [
                os.pread(
                    fd, min(self.CHUNK, length - chunk_start), offset + chunk_start
                )
                for chunk_start in range(0, length, self.CHUNK)
            ]
        finally:
            os.close(fd)
        value = np.frombuffer(b"".join(pieces), dtype).reshape(
            [r1 - r0, *shape[1:]] if shape else []
        )
        return value[rest] if rest else value

    def dense(self, name: _Dynamic, layer: _Dynamic, rank: _Dynamic) -> _Dynamic:
        """One layer of a parameter family, with its exact global-rank slice."""
        base = f"{PREFIX}layers.{layer}."
        rows = slice(rank * 384, (rank + 1) * 384)
        heads = slice(rank * 3, (rank + 1) * 3)

        def read(suffix: _Dynamic, selection: _Dynamic = None) -> _Dynamic:
            """Execute read logic."""
            return self.read(base + suffix, selection)

        def matrix(
            suffix: _Dynamic, output: _Dynamic = ALL, contraction: _Dynamic = ALL
        ) -> _Dynamic:
            """Execute matrix logic."""
            return read(suffix + ".weight", (output, contraction)).T.copy()

        def folded(stem: _Dynamic, global_output: _Dynamic = None) -> _Dynamic:
            """Execute folded logic."""
            pre = PREFIX if bool(global_output) else base
            return self.read(pre + stem + "_norm.weight").astype(
                np.float32
            ) * self.read(pre + stem + "_proj.weight").astype(np.float32)

        def _latent(up: _Dynamic) -> _Dynamic:
            """Execute latent projection logic."""
            shape = (128, 7168) if up else (7168, 128)
            if rank >= _CONST_28:
                return np.zeros(shape, BF16)
            selection = slice(rank * 128, (rank + 1) * 128)
            stem = "block_sparse_moe.routed_expert_"
            return (
                matrix(stem + "up_proj", contraction=selection)
                if up
                else matrix(stem + "down_proj", selection)
            )

        def _gu(stem: str, real: int, padded: int) -> _Dynamic:
            """Execute gate/up projection logic."""
            selection = slice(rank * real, (rank + 1) * real)
            return np.concatenate(
                [
                    np.pad(
                        matrix(stem + "." + p + "_proj", selection),
                        ((0, 0), (0, padded - real)),
                    )
                    for p in ("gate", "up")
                ],
                axis=1,
            )

        readers: dict[str, _Dynamic] = {
            "embedding": lambda: self.read(
                PREFIX + "embed_tokens.weight",
                (slice(rank * 5120, (rank + 1) * 5120), slice(None)),
            ),
            "lm_head": lambda: self.read(
                "language_model.lm_head.weight",
                (slice(rank * 5120, (rank + 1) * 5120), slice(None)),
            ).T.copy(),
            "final_norm": lambda: self.read(PREFIX + "norm.weight")[None],
            "output_res": lambda: folded("output_attn_res", bool(1)),
            "attn_res": lambda: folded("self_attention_res"),
            "mlp_res": lambda: folded("mlp_res"),
            "attn_norm": lambda: read("input_layernorm.weight")[None],
            "ffn_norm": lambda: read("post_attention_layernorm.weight")[None],
            "dense_gu": lambda: _gu("mlp", 1056, 1152),
            "shared_gu": lambda: _gu("block_sparse_moe.shared_experts", 192, 256),
            "dense_down": lambda: np.pad(
                matrix(
                    "mlp.down_proj", contraction=slice(rank * 1056, (rank + 1) * 1056)
                ),
                ((0, 96), (0, 0)),
            ),
            "shared_down": lambda: np.pad(
                matrix(
                    "block_sparse_moe.shared_experts.down_proj",
                    contraction=slice(rank * 192, (rank + 1) * 192),
                ),
                ((0, 64), (0, 0)),
            ),
            "router": lambda: matrix("block_sparse_moe.gate"),
            "router_bias": lambda: read(
                "block_sparse_moe.gate.e_score_correction_bias"
            )[None],
            "latent_norm": lambda: np.pad(
                read("block_sparse_moe.routed_expert_norm.weight").astype(np.float32)[
                    None
                ],
                ((0, 0), (0, 512)),
            ),
            "latent_down": lambda: _latent(bool(0)),
            "latent_up": lambda: _latent(bool(1)),
            "k_gate": lambda: matrix("self_attn.f_a_proj"),
            "k_projection": lambda: np.concatenate(
                [matrix("self_attn." + p + "_proj", rows) for p in ("q", "k", "v")]
                + [matrix("self_attn.g_proj", rows)]
                + [np.pad(matrix("self_attn.b_proj", heads), ((0, 0), (0, 125)))],
                axis=1,
            ),
            "k_fb": lambda: matrix("self_attn.f_b_proj", rows),
            "k_o": lambda: matrix("self_attn.o_proj", contraction=rows),
            "m_o": lambda: matrix("self_attn.o_proj", contraction=rows),
            "k_conv": lambda: np.stack(
                [
                    read(
                        "self_attn." + p + "_conv1d.weight",
                        (rows, slice(None), slice(None)),
                    )[:, 0].T.reshape(4, 3, 128)
                    for p in ("q", "k", "v")
                ],
                axis=1,
            ),
            "k_a_log": lambda: read("self_attn.A_log", heads).reshape(3, 1),
            "k_dt": lambda: read("self_attn.dt_bias", rows).reshape(3, 128),
            "k_norm": lambda: np.pad(
                read("self_attn.o_norm.weight")[None], ((0, 7), (0, 0))
            ),
            "m_qa": lambda: matrix("self_attn.q_a_proj"),
            "m_ka": lambda: np.pad(
                matrix("self_attn.kv_a_proj_with_mqa"), ((0, 0), (0, 64))
            ),
            "m_qb": lambda: np.pad(
                matrix(
                    "self_attn.q_b_proj", slice(rank * 576, (rank + 1) * 576)
                ).reshape(1536, 3, 192),
                ((0, 0), (0, 0), (0, 64)),
            ).reshape(1536, 768),
            "m_kb": lambda: matrix(
                "self_attn.kv_b_proj", slice(rank * 768, (rank + 1) * 768)
            ),
            "m_gate": lambda: matrix("self_attn.g_proj", rows),
            "m_qnorm": lambda: read("self_attn.q_a_layernorm.weight")[None],
            "m_knorm": lambda: read("self_attn.kv_a_layernorm.weight")[None],
        }
        if name in readers:
            return readers[name]()
        raise KeyError(name)

    def dense_selected(
        self, name: _Dynamic, layer: _Dynamic, rank: _Dynamic
    ) -> _Dynamic:
        """One layer of a parameter family in the TP32 kernel layout.

        It differs from :meth:`dense` only for these families:

        * ``k_o``/``m_o``: paired-rank output projections. Rank ``r`` holds the
          contraction channels of ranks ``2p, 2p+1`` (``p = r // 2``, 768 rows)
          for output half ``r % 2`` -> ``[768, hidden / 2]``.
        * ``latent_up``: output-sharded over 8 blocks of 1024 hidden columns
          (block ``r % 8``; block 7 is zero padding past 7168), replicated
          across the four hosts, with the padded latent rows dropped ->
          ``[3584, 1024]``.
        * ``shared_gu``/``shared_down``: the rank's 192 shared channels without
          the 256 padding -> ``[hidden, 384]`` / ``[192, hidden]``.
        """
        base = f"{PREFIX}layers.{layer}."
        hidden = self.config.dim
        if name in ("k_o", "m_o"):
            pair, half = divmod(rank, 2)
            value = self.read(
                base + "self_attn.o_proj.weight",
                (
                    slice(half * hidden // 2, (half + 1) * hidden // 2),
                    slice(pair * 768, (pair + 1) * 768),
                ),
            )
            return value.T.copy()
        if name == "latent_up":
            block = rank % 8
            if (block + 1) * 1024 > hidden:
                return np.zeros((self.config.latent, 1024), BF16)
            value = self.read(
                base + "block_sparse_moe.routed_expert_up_proj.weight",
                (slice(block * 1024, (block + 1) * 1024), slice(None)),
            )
            return value.T.copy()
        if name == "shared_gu":
            rows = slice(rank * 192, (rank + 1) * 192)
            return np.concatenate(
                [
                    self.read(
                        base + f"block_sparse_moe.shared_experts.{p}_proj.weight",
                        (rows, slice(None)),
                    ).T
                    for p in ("gate", "up")
                ],
                axis=1,
            )
        if name == "shared_down":
            return self.read(
                base + "block_sparse_moe.shared_experts.down_proj.weight",
                (slice(None), slice(rank * 192, (rank + 1) * 192)),
            ).T.copy()
        return self.dense(name, layer, rank)


def selected_local_shapes(
    layers: _Dynamic = 93, hidden: _Dynamic = 7168, *, vocab: _Dynamic = 163840
) -> _Dynamic:
    """Per-rank shapes of the kernel layout.

    TP4/EP8 packed routed experts, output-sharded host-local ``latent_up``,
    compact shared experts, split KDA gate projection, and paired attention
    output projections.
    """
    if not _CONST_2 <= layers <= _CONST_93:
        _err_msg = "Kimi stack supports a prefix of 2..93 layers"
        raise ValueError(_err_msg)
    if vocab % 32:
        _err_msg = "Vocabulary must divide into 32 shards"
        raise ValueError(_err_msg)
    m = max(1, layers // 4 + (layers == _CONST_93))
    k = layers - (layers // 4 + (layers == _CONST_93))
    return {
        "router": (layers - 1, hidden, 896),
        "router_bias": (layers - 1, 1, 896),
        "latent_down": (layers - 1, hidden, 128),
        "latent_up": (layers - 1, 3584, 1024),
        "latent_norm": (layers - 1, 1, 4096),
        "shared_gu": (layers - 1, hidden, 384),
        "shared_down": (layers - 1, 192, hidden),
        "expert_gu": (layers - 1, 112, 448, 1536),
        "expert_gus": (layers - 1, 112, 112, 1536),
        "expert_down": (layers - 1, 112, 96, 3584),
        "expert_ds": (layers - 1, 112, 24, 3584),
        "k_gate": (k, hidden, 128),
        "k_projection": (k, hidden, 1664),
        "k_fb": (k, 128, 384),
        "k_o": (k, 768, hidden // 2),
        "k_conv": (k, 4, 3, 3, 128),
        "k_a_log": (k, 3, 1),
        "k_dt": (k, 3, 128),
        "k_norm": (k, 8, 128),
        "m_qa": (m, hidden, 1536),
        "m_ka": (m, hidden, 640),
        "m_qb": (m, 1536, 768),
        "m_kb": (m, 512, 768),
        "m_gate": (m, hidden, 384),
        "m_o": (m, 768, hidden // 2),
        "m_qnorm": (m, 1, 1536),
        "m_knorm": (m, 1, 512),
        "attn_res": (layers, 1, hidden),
        "mlp_res": (layers, 1, hidden),
        "attn_norm": (layers, 1, hidden),
        "ffn_norm": (layers, 1, hidden),
        "embedding": (vocab // 32, hidden),
        "lm_head": (hidden, vocab // 32),
        "dense_gu": (hidden, 2304),
        "dense_down": (1152, hidden),
        "output_res": (1, hidden),
        "final_norm": (1, hidden),
    }


def _family_layers(config: _Dynamic, name: _Dynamic, layers: _Dynamic) -> _Dynamic:
    """Checkpoint layer ids stored along a family's leading axis.

    (None: not per layer).
    """
    kda = {
        "k_gate",
        "k_projection",
        "k_fb",
        "k_o",
        "k_conv",
        "k_a_log",
        "k_dt",
        "k_norm",
    }
    mla = {"m_qa", "m_ka", "m_qb", "m_kb", "m_gate", "m_o", "m_qnorm", "m_knorm"}
    moe = {
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
    }
    norms = {"attn_res", "mlp_res", "attn_norm", "ffn_norm"}
    if name in kda:
        return [i for i in range(layers) if i not in config.full_attention]
    if name in mla:
        return [i for i in range(layers) if i in config.full_attention] or [3]
    if name in moe:
        return list(range(1, layers))
    if name in norms:
        return list(range(layers))
    return None
