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

"""Kimi-K3 fixtures for the standalone tools: stand-in unified pools laid out.

as vLLM's, their KV cache groups, the fixed prompt.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vllm_torchtpu.models.kimi_k3.megakernel import config, layout
from vllm_torchtpu.models.kimi_k3.megakernel.decode import KimiDecoder
from vllm_torchtpu.models.kimi_k3.megakernel.layout import cache_layout
from vllm_torchtpu.models.kimi_k3.megakernel.program import kernel_modules
from vllm_torchtpu.models.kimi_k3.megakernel.runtime_shim import program as mk
from vllm_torchtpu.models.kimi_k3.megakernel.runtime_shim.config import (
    MegakernelEngineConfig,
)

_Dynamic = Any


PAGE_BOX = [256]
DCP = 32
LOGICAL_BOX = [256 * 32]  # tokens per vLLM DCP block
CACHE_DTYPE_BOX = ["auto"]
PAGE: _Dynamic = 256
LOGICAL: _Dynamic = 256 * 32
CACHE_DTYPE: _Dynamic = "auto"


def configure_cache(dtype: str) -> None:
    """Configure standalone fixtures from the same cache option as serving."""
    PAGE_BOX[0] = cache_layout(dtype).tokens_per_block
    LOGICAL_BOX[0] = PAGE_BOX[0] * DCP
    CACHE_DTYPE_BOX[0] = dtype


def logical() -> int:
    """Tokens per DCP block under the installed kernel config's DCP group."""
    return PAGE_BOX[0] * config.active().dcp_group


def fixed_prompt(count: int, checkpoint: str) -> _Dynamic:
    """``(tokens, tokenizer)``: the first ``count`` tokens of a fixed text, the.

    model repository's own Python sources (``checkpoint``: the Kimi-K3
    checkpoint directory).
    """
    (load,) = kernel_modules("load")
    tokenizer = load.load_tokenizer(checkpoint)
    text = "".join(path.read_text() for path in sorted(Path(checkpoint).glob("*.py")))
    return tokenizer.encode(text, disallowed_special=())[:count], tokenizer


class Pools:
    """24 unified pools laid out per ``cache_layout(CACHE_DTYPE_BOX[0])`` (one MLA.

    layer and three KDA layers each, as vLLM allocates them) and the KV cache
    groups: the MLA layers' (``logical()``-token blocks, ``mla_width`` entries a
    row) and three KDA groups (a state column per ``PAGE`` tokens,
    ``state_width`` entries), KDA layer k in group k % 3. Block 0 is the null
    block.
    """

    def __init__(
        self,
        num_blocks: int,
        mla_width: int,
        state_width: int,
        *,
        pattern: bool = False,
    ) -> None:
        """Execute init logic."""
        cache = layout.cache_layout(CACHE_DTYPE_BOX[0])
        dtype = getattr(torch, cache.dtype)
        if pattern:  # synthetic latents: a random bf16 pattern (the rope lanes zero)
            # One seed on every rank: every page holds the same tokens, whatever
            # the DCP layout that places it.
            generator = torch.Generator().manual_seed(0)
            page = (
                torch.randn((1, 128, 2, 640), dtype=torch.float32, generator=generator)
                * 0.5
            )
            page[..., 576:] = 0
            page = page.to(torch.bfloat16)
            # Keep the same BF16 fixture across page geometries: a 512-token
            # FP8 page repeats the 256-token tile.
            page = (
                page.reshape(1, 256, 640)
                .repeat(1, cache.tokens_per_block // 256, 1)
                .reshape(1, *cache.block_shape)
                .to(dtype)
            )
            self.tensors = [
                page.to("tpu").repeat(num_blocks, 1, 1, 1) for _ in range(24)
            ]
        else:
            self.tensors = [
                torch.zeros((num_blocks, *cache.block_shape), dtype=dtype).to("tpu")
                for _ in range(24)
            ]
        mla_names = [
            f"model.layers.{layer}.self_attn.mla_attn.attn" for layer in layout.MLA
        ]
        kda_names = [f"model.layers.{layer}.self_attn" for layer in layout.KDA]
        self.caches = {name: self.tensors[m] for m, name in enumerate(mla_names)}
        self.caches.update(
            {name: self.tensors[k // 3] for k, name in enumerate(kda_names)}
        )
        self.groups = [mk.KVGroup(tuple(mla_names), logical(), mla_width)]
        self.groups += [
            mk.KVGroup(
                tuple(n for k, n in enumerate(kda_names) if k % 3 == g),
                PAGE_BOX[0],
                state_width,
            )
            for g in range(3)
        ]
        self.num_blocks = num_blocks
        self.mla_width, self.state_width = mla_width, state_width
        self._next = 1

    def scheduled(self, table: _Dynamic, index: int) -> np.ndarray:
        """MLA ``table`` as the decode scheduler issues it to the ``index``-th.

        request admitted: under attention DP that request is in DP group
        ``g = index % dp`` (fewest requests first) and its pages are ids
        ``g * num_blocks + row`` (``megakernel.dp_blocks``); the null block
        stays shared.
        """
        group = index % config.active().attention_dp
        table = np.asarray(table, np.int32)
        return np.where(table > 0, table + group * self.num_blocks, 0).astype(np.int32)

    def fresh(self, n: int) -> list[int]:
        """Execute fresh logic."""
        out = list(range(self._next, self._next + n))
        self._next += n
        return out

    def tables(
        self, mla: Sequence[int], kda: Sequence[Sequence[int] | np.ndarray]
    ) -> tuple[np.ndarray, ...]:
        """Execute tables logic."""

        def padded(table: Sequence[int] | np.ndarray, width: int) -> np.ndarray:
            """Execute padded logic."""
            return np.asarray(list(table) + [0] * (width - len(table)), np.int32)

        return (
            padded(mla, self.mla_width),
            *tuple(padded(t, self.state_width) for t in kda),
        )


def engine_config(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """The serving configuration's megakernel view, for the TP32 / DCP32.

    deployment (DCP over each group under the installed kernel config's
    attention DP) with the configured cache (``configure_cache``).
    """
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    role = _p_next(_p_iter, _p_get("role"))
    (
        max_num_seqs,
        max_model_len,
        max_num_batched_tokens,
        num_speculative_tokens,
    ) = (
        _p_get("max_num_seqs"),
        _p_get("max_model_len"),
        _p_get("max_num_batched_tokens"),
        _p_get("num_speculative_tokens", 0),
    )
    num_speculative_tokens_per_batch_size = _p_get(
        "num_speculative_tokens_per_batch_size"
    )
    dp = config.active().attention_dp
    return MegakernelEngineConfig(
        role=role,
        tensor_parallel_size=32,
        decode_context_parallel_size=DCP // dp,
        block_size=PAGE_BOX[0],
        kv_cache_dtype=CACHE_DTYPE_BOX[0],
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        num_speculative_tokens=num_speculative_tokens,
        num_speculative_tokens_per_batch_size=num_speculative_tokens_per_batch_size,
        attention_dp=dp,
    )


def decoder(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """A :class:`KimiDecoder` over shared ``ops`` (and ``draft``) and the.

    fixture ``pools``, bound and ready to warm. ``schedule``: vLLM's
    per-batch-size draft count (None: plain steps only). Pages are owned by
    mesh position, as the fixtures lay them out.
    """
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        ops,
        pools,
    ) = (
        _p_next(_p_iter, _p_get("ops")),
        _p_next(_p_iter, _p_get("pools")),
    )
    (
        max_num_seqs,
        max_model_len,
        draft,
        schedule,
    ) = (
        _p_get("max_num_seqs"),
        _p_get("max_model_len"),
        _p_get("draft"),
        _p_get("schedule"),
    )
    config = engine_config(
        "decode",
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        max_num_batched_tokens=512,
        num_speculative_tokens=7 if schedule else 0,
        num_speculative_tokens_per_batch_size=schedule,
    )
    decoder = KimiDecoder(config, ops, draft if schedule else None, tp_rank=ops.rank)
    decoder.bind_kv_caches(pools.caches, pools.groups)
    return decoder


def request(
    req_id: str,
    tokens: _Dynamic,
    table: np.ndarray,
    sampling: tuple[float, float, int, int] = (0.0, 1.0, 0, 0),
    computed: int = 0,
) -> _Dynamic:
    """A decode step's request scheduling ``tokens`` after ``computed`` ones,.

    MLA pages ``table``. Its KDA tables hold only the null block, so a
    decoder that has not seen it starts it from a clean state (no import).
    """
    tokens = np.asarray(tokens, np.int32)
    end = computed + len(tokens)
    token_ids = np.zeros(end, np.int32)
    token_ids[computed:] = tokens
    null = np.zeros(1, np.int32)
    return mk.RequestInput(
        req_id=req_id,
        computed=computed,
        scheduled=len(tokens),
        token_ids=token_ids,
        num_tokens=end,
        num_prompt_tokens=end,
        block_tables=(np.asarray(table, np.int32), null, null, null),
        sampling=mk.SamplingSpec(*sampling),
    )
