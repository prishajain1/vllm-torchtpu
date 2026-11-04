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

"""Kimi-K3 as a megakernel program: the prefill role's :class:`KimiPrefiller`.

or the decode role's :class:`KimiDecoder`, and what they support.
The K3 programs have no vLLM dependencies; ``megakernel.runner`` builds them
from vLLM's configuration and the standalone tools build them directly.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from .layout import cache_layout
from .runtime_shim.config import MegakernelEngineConfig
from .runtime_shim.program import Capabilities

_ = (MegakernelEngineConfig,)

_Dynamic = Any
_CONST_3 = 3
_CONST_8 = 8


TP_SIZE = 32
DCP_SIZE = 32
HIDDEN_SIZE = 7168
VOCAB = 163840  # tokens; vocabulary-sharded 5120 per TP rank
DRAFT_TOKENS = 7  # DSpark: the verify is the anchor and 7 drafts
SPEC_BATCHES = (1, 2, 4, 8, 16)  # the batched speculation's compiled request counts
# Attention DP2's compiled request counts: half of each are one group's.
DP_SPEC_BATCHES = (2, 4, 8, 16, 32)
MAX_STEP_TOKENS = 8192  # the prefill's largest window
# Sampling parameters the ops implement (beyond greedy): temperature, top-p,
# top-k and a per-request seed, on both roles.
SAMPLING = frozenset({"temperature", "top_p", "top_k", "seed"})

_SPEC_SUPPORT = (
    f"3 or {DRAFT_TOKENS} drafts at batches 1-8, and 3 at batches "
    f"9-{max(SPEC_BATCHES)} (attention DP2: 9-{max(DP_SPEC_BATCHES)}) through "
    f"the compiled speculation batches {SPEC_BATCHES} (attention DP2: "
    f"{DP_SPEC_BATCHES}; up to --max-num-seqs), 0 anywhere"
)


def kernel_modules(*names: str) -> list[_Dynamic]:
    """Import ``.kernels.<name>`` for each name."""
    return [
        importlib.import_module(
            f"vllm_torchtpu.models.kimi_k3.megakernel.kernels.{name}"
        )
        for name in names
    ]


def supports_spec_tokens(batch: int, drafts: int, attention_dp: int = 1) -> bool:
    """Widths supported by the single and batched verifier/drafter (under.

    attention DP2 three drafts up to 32 requests: a 128-row verify, drafted
    in parts).
    """
    largest = max(DP_SPEC_BATCHES if attention_dp > 1 else SPEC_BATCHES)
    return drafts == 0 or (
        1 <= batch <= largest
        and (drafts == _CONST_3 or (batch <= _CONST_8 and drafts == DRAFT_TOKENS))
    )


def _validate_schedule_entries(entries: list[tuple[int, int, int]]) -> None:
    """Validate that schedule ranges are ordered, non-overlapping, and valid."""
    previous_end = 0
    for lo, hi, k in entries:
        if lo <= 0 or hi < lo or k < 0:
            _err_msg = (
                f"num_speculative_tokens_per_batch_size entries are "
                f"(min_batch, max_batch, drafts) with 1 <= min <= max and "
                f"drafts >= 0, not ({lo}, {hi}, {k})"
            )
            raise NotImplementedError(_err_msg)
        if lo <= previous_end:
            _err_msg = (
                f"num_speculative_tokens_per_batch_size ranges must be "
                f"non-overlapping: ({lo}, {hi}, {k})"
            )
            raise NotImplementedError(_err_msg)
        previous_end = hi
    if entries[0][0] != 1:
        _err_msg = (
            f"num_speculative_tokens_per_batch_size's first range must "
            f"start at batch size 1: {entries[0]}"
        )
        raise NotImplementedError(_err_msg)


def _build_dense_schedule(
    schedule: _Dynamic,
    num_spec: int,
    max_seqs: int,
) -> list[int]:
    """Populate the dense draft table across batch sizes 1..max_seqs."""
    dense = [0] * (max_seqs + 1)
    if schedule is None:
        for b in range(1, max_seqs + 1):
            dense[b] = num_spec
        return dense
    entries = sorted((int(lo), int(hi), int(k)) for lo, hi, k in schedule)
    _validate_schedule_entries(entries)
    next_b, last_k = 1, 0
    for lo, hi, k in entries:
        for b in range(next_b, min(lo, max_seqs + 1)):
            dense[b] = min(num_spec, last_k)
        for b in range(max(lo, next_b), min(hi, max_seqs) + 1):
            dense[b] = min(num_spec, k)
        next_b, last_k = max(next_b, hi + 1), k
        if next_b > max_seqs:
            break
    for b in range(next_b, max_seqs + 1):
        dense[b] = min(num_spec, last_k)
    return dense


def _validate_dense_schedule(
    dense: list[int],
    schedule: _Dynamic,
    max_seqs: int,
    config: MegakernelEngineConfig,
) -> None:
    """Validate that every scheduled draft width is supported."""
    for b in range(1, max_seqs + 1):
        k = dense[b]
        if k == 0:
            continue
        supported = supports_spec_tokens(b, k, config.attention_dp)
        if b > 1 and supported and config.attention_dp == 1:
            supported = any(b <= s <= max_seqs for s in SPEC_BATCHES)
        if not supported:
            given = (
                "None: num_speculative_tokens everywhere"
                if schedule is None
                else str(schedule)
            )
            _err_msg = (
                f"DSpark on the K3 decoder supports {_SPEC_SUPPORT}; the "
                f"schedule gives {k} drafts at batch size {b} "
                f"(num_speculative_tokens_per_batch_size={given}; "
                f"--max-num-seqs {max_seqs})"
            )
            raise NotImplementedError(_err_msg)


def spec_tokens_per_batch(config: MegakernelEngineConfig) -> tuple[int, ...] | None:
    """The drafts vLLM schedules at every decoding batch size: a dense table.

    whose index ``b`` holds batch size ``b``'s count (``[0]`` unused), or
    None when the decoder does not speculate.
    vLLM's own schedule semantics (``build_dynamic_sd_schedule_lookup``): a
    range covers its batch sizes, a gap or the tail carries the previous
    range's count forward, every count clamps to ``num_speculative_tokens``,
    and without a schedule every batch size gets ``num_speculative_tokens``.
    Raises ``NotImplementedError`` for a schedule the kernels can't run.
    """
    if config.role != "decode" or not config.num_speculative_tokens:
        return None
    dense = _build_dense_schedule(
        config.num_speculative_tokens_per_batch_size,
        config.num_speculative_tokens,
        config.max_num_seqs,
    )
    _validate_dense_schedule(
        dense,
        config.num_speculative_tokens_per_batch_size,
        config.max_num_seqs,
        config,
    )
    return tuple(dense)


def recurrent_state_dtype(cache_dtype: str) -> str:
    """Decode bank storage; auto preserves FP32 independently of model dtype.

    Prefill and the connector retain their FP32 state format. The importer
    casts that state once to the decoder bank's dtype.
    """
    if cache_dtype not in ("auto", "float32", "bfloat16"):
        _err_msg = (
            f"K3 megakernel --mamba-ssm-cache-dtype supports auto, "
            f"float32, or bfloat16, not {cache_dtype!r}"
        )
        raise ValueError(_err_msg)
    return "float32" if cache_dtype == "auto" else cache_dtype


def _check_parallelism(config: MegakernelEngineConfig) -> None:
    """Validate role and parallelism constraints."""
    recurrent_state_dtype(config.mamba_ssm_cache_dtype)
    if config.role == "prefill" and config.mamba_ssm_cache_dtype == "bfloat16":
        _err_msg = (
            "K3 megakernel --mamba-ssm-cache-dtype bfloat16 is "
            "decode-only; the prefill worker requires auto or float32 for "
            "its FP32 state and transfer format"
        )
        raise NotImplementedError(_err_msg)
    if config.role not in ("prefill", "decode"):
        _err_msg = (
            f"the K3 megakernel serves one role of a P/D pair, not {config.role!r}"
        )
        raise NotImplementedError(_err_msg)
    if config.tensor_parallel_size != TP_SIZE:
        _err_msg = f"the K3 megakernel needs --tensor-parallel-size {TP_SIZE}"
        raise NotImplementedError(_err_msg)
    if config.attention_dp not in (1, 2) or (
        config.attention_dp > 1 and config.role != "decode"
    ):
        _err_msg = "the K3 decoder runs attention_dp 1 or 2; the prefill runs 1"
        raise NotImplementedError(_err_msg)
    dcp = DCP_SIZE // config.attention_dp
    if config.decode_context_parallel_size != dcp:
        _err_msg = f"the K3 megakernel needs --decode-context-parallel-size {dcp}" + (
            f" under attention_dp {config.attention_dp}" if dcp != DCP_SIZE else ""
        )
        raise NotImplementedError(_err_msg)


def _check_cache_and_speculation(config: MegakernelEngineConfig) -> None:
    """Validate block size, cache dtypes, and speculative settings."""
    layout = cache_layout(config.kv_cache_dtype)
    if config.block_size != layout.tokens_per_block:
        _err_msg = (
            f"the K3 megakernel with cache dtype {config.kv_cache_dtype} "
            f"requires --block-size {layout.tokens_per_block}"
        )
        raise NotImplementedError(_err_msg)
    if config.cp_kv_cache_interleave_size not in (None, layout.tokens_per_block):
        _err_msg = (
            f"the K3 megakernel requires --cp-kv-cache-interleave-size "
            f"{layout.tokens_per_block} with cache dtype "
            f"{config.kv_cache_dtype}"
        )
        raise NotImplementedError(_err_msg)
    if config.role == "prefill" and config.max_num_batched_tokens > MAX_STEP_TOKENS:
        _err_msg = (
            f"the K3 prefill takes at most {MAX_STEP_TOKENS} tokens per "
            f"step: --max-num-batched-tokens <= {MAX_STEP_TOKENS}"
        )
        raise NotImplementedError(_err_msg)
    if config.role == "decode" and config.num_speculative_tokens not in (
        0,
        3,
        DRAFT_TOKENS,
    ):
        _err_msg = (
            f"DSpark on the K3 decoder supports num_speculative_tokens=0, "
            f"3 or {DRAFT_TOKENS}"
        )
        raise NotImplementedError(_err_msg)
    dense = spec_tokens_per_batch(config)
    if config.synthetic_acceptance_rates_by_width is not None:
        by_width = dict(config.synthetic_acceptance_rates_by_width)
        missing = sorted({k for k in dense or () if k} - set(by_width))
        if missing:
            _err_msg = (
                f"synthetic_acceptance_lengths gives no length for the "
                f"{missing} draft widths the schedule runs (it has "
                f"{sorted(by_width)})"
            )
            raise NotImplementedError(_err_msg)
        own = by_width.get(config.num_speculative_tokens)
        if own is not None and own != config.synthetic_acceptance_rates:
            _err_msg = (
                f"synthetic_acceptance_lengths' "
                f"{config.num_speculative_tokens}-draft rates {list(own)} "
                f"differ from --speculative-config's "
                f"{list(config.synthetic_acceptance_rates or ())}"
            )
            raise NotImplementedError(_err_msg)
    if config.role == "decode" and not config.async_scheduling:
        _err_msg = (
            "the K3 decoder needs async scheduling (it keeps each "
            "request's next input on the device)"
        )
        raise NotImplementedError(_err_msg)


def check(config: MegakernelEngineConfig) -> None:
    """Raise ``NotImplementedError`` for a configuration the K3 programs.

    do not implement.
    """
    _check_parallelism(config)
    _check_cache_and_speculation(config)


def capabilities(config: MegakernelEngineConfig) -> Capabilities:
    """Execute capabilities logic."""
    check(config)
    dense = spec_tokens_per_batch(config)
    speculating = dense is not None and any(dense)
    return Capabilities(
        roles=frozenset({config.role}),
        sampling=SAMPLING,
        speculation="dspark" if speculating else None,
        num_speculative_tokens=config.num_speculative_tokens if speculating else 0,
        max_step_tokens=MAX_STEP_TOKENS
        if config.role == "prefill"
        else config.max_num_batched_tokens,
        page_aligned_chunks=config.role == "prefill",
        block_copies=config.role == "prefill",
        # The decoder's plain op masks its logits (the prefill discards its
        # first tokens, so it only accepts the requests). DSpark's verify ops
        # do not mask: a step with a structured request runs plain instead.
        structured_outputs=True,
    )


def build(
    config: MegakernelEngineConfig,
    presharded: Path,
    draft_checkpoint: Path | None = None,
    *,
    tp_rank: int,
) -> _Dynamic:
    """The program for ``config.role`` on TP rank ``tp_rank`` (the order the KV.

    connector pairs devices in; ``draft_checkpoint``: the DSpark speculator --
    the prefill computes its ring, the decoder drafts with it and imports the
    ring the prefill wrote).
    """
    check(config)
    if config.role == "prefill":
        prefill_mod = importlib.import_module(
            "vllm_torchtpu.models.kimi_k3.megakernel.prefill"
        )
        return prefill_mod.KimiPrefiller(
            config, presharded, draft_checkpoint, tp_rank=tp_rank
        )

    dense = spec_tokens_per_batch(config)
    draft = None
    if draft_checkpoint is not None and dense is not None and any(dense):
        draft_mod = importlib.import_module(
            "vllm_torchtpu.models.kimi_k3.megakernel.draft_mla"
        )
        draft = draft_mod.draft_class(draft_checkpoint)(
            max_position=config.max_model_len + 16,
            checkpoint=str(draft_checkpoint),
            presharded=Path(presharded),
            synthetic_rates=config.synthetic_acceptance_rates,
            synthetic_rates_by_width=dict(config.synthetic_acceptance_rates_by_width)
            if config.synthetic_acceptance_rates_by_width is not None
            else None,
        )
    # Speculative post-states are temporary; acceptance commits one state
    # into each regular request slot (decode_ops.Control).
    dec_ops_mod = importlib.import_module(
        "vllm_torchtpu.models.kimi_k3.megakernel.decode_ops"
    )
    dec_mod = importlib.import_module("vllm_torchtpu.models.kimi_k3.megakernel.decode")
    sub_base = 2 * config.max_num_seqs + 1
    ops = dec_ops_mod.DecodeOps(
        presharded,
        num_slots=dec_ops_mod.Control.total_slots(sub_base) if draft else sub_base,
        batch_sizes=dec_ops_mod.decode_batch_sizes(config.max_num_seqs),
        mamba_ssm_cache_dtype=config.mamba_ssm_cache_dtype,
    )
    return dec_mod.KimiDecoder(config, ops, draft, tp_rank=tp_rank)
