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

"""Kernel-only timing of the serving ops (``bench decode`` / ``bench prefill``).

**Decode** times ``KimiDecoder.step`` over vLLM's 24 unified pools for every
``--batches`` x ``--contexts`` case (``--mix``: one batch of rows at different
contexts), or with ``--dspark [CHECKPOINT]`` the DSpark verify + draft step
(serving's draft schedule: seven drafts through batch four, three through
eight; ``--acceptance`` fixes the accepted drafts per position so drafts
compare at equal acceptance). Each request starts at its context over
synthetic KV pages and a clean KDA state, with distinct tokens per row (they
set MoE routing); ``--steps`` steps are enqueued back to back and timed with
one host read, median of ``--repeats``. DSpark's own tokens per step on
synthetic contexts are not representative.

**Prefill** times :class:`KimiPrefiller.step` per ``--prompts`` case: ``LENxN``
is one step of ``N`` prompts of ``LEN`` new tokens, ``LENxN@PREFIX`` appends
them to a page-aligned cached prefix. Each line names the compiled windows the
case packs into.

``--trace DIR`` (both): rank 0 saves an xprof trace of each case's last repeat.
"""

from __future__ import annotations

import operator
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch_tpu._internal.profiler as tpu_profiler_mod

from vllm_torchtpu.models.kimi_k3.megakernel import config
from vllm_torchtpu.models.kimi_k3.megakernel import prefill as k3_prefill
from vllm_torchtpu.models.kimi_k3.megakernel import program as k3_program
from vllm_torchtpu.models.kimi_k3.megakernel.decode_ops import Control, DecodeOps
from vllm_torchtpu.models.kimi_k3.megakernel.draft_mla import draft_class
from vllm_torchtpu.models.kimi_k3.megakernel.runtime_shim import bridge
from vllm_torchtpu.models.kimi_k3.megakernel.runtime_shim import program as mk

from . import harness, k3

_Dynamic = Any
_CONST_4 = 4
_CONST_7 = 7
_CONST_8 = 8


def add_arguments(parser: _Dynamic, *, profile: bool = False) -> None:
    """Execute add arguments logic."""
    sub = parser.add_subparsers(dest="role", required=True)
    decode = sub.add_parser("decode", help="the decode ops")
    decode.add_argument("--batches", default="1,8")
    decode.add_argument(
        "--contexts", default="1152", help="per-request context (all rows of a batch)"
    )
    decode.add_argument(
        "--mix",
        default="",
        help="also one mixed batch, rows by context: e.g. 1152x4,131072x4",
    )
    decode.add_argument("--steps", type=int, default=40)
    decode.add_argument("--repeats", type=int, default=3)
    decode.add_argument("--temperature", type=float, default=0.0)
    decode.add_argument(
        "--max-num-seqs",
        type=int,
        default=0,
        help="slot layout as serving (default: max batch)",
    )
    decode.add_argument(
        "--dspark",
        default="",
        metavar="CHECKPOINT",
        help="time DSpark verify + draft steps with this speculator (batches "
        "up to eight); without the flag every step is plain",
    )
    decode.add_argument(
        "--drafts",
        type=int,
        choices=(3, 7),
        default=0,
        help="with --dspark: this many drafts at every batch up to eight "
        "(default: serving's schedule)",
    )
    decode.add_argument(
        "--acceptance",
        default="",
        help="with --dspark: synthetic per-position acceptance rates of the "
        "seven drafts (comma separated; three drafts take the first three), "
        "e.g. 0.92,0.83,0.73,0.65,0.57,0.50,0.43; default: the drafts' own",
    )
    decode.add_argument(
        "--spec-drafts",
        type=int,
        choices=(0, 3, 7),
        default=0,
        help="with --dspark: this many drafts at every batch, any size (e.g. "
        "attention DP2's batched verify + draft); default: --drafts or "
        "serving's schedule",
    )
    decode.add_argument(
        "--save-tokens",
        default="",
        help="file: rank 0 saves each case's timed greedy tokens [steps, batch] (npz)",
    )
    prefill = sub.add_parser("prefill", help="the prefill stack")
    prefill.add_argument(
        "--prompts",
        default="8192x1,2048x4",
        help="cases LENxN[@PREFIX]: N prompts of LEN new tokens per step, each "
        "appended to a cached prefix of PREFIX tokens (a page-aligned hit)",
    )
    prefill.add_argument("--repeats", type=int, default=3)
    prefill.add_argument(
        "--token-file",
        default="",
        help="a .npy of int32 token ids: each prompt is a random slice of it "
        "(real text) instead of uniform random ids",
    )
    for role in (decode, prefill):
        role.add_argument(
            "--trace",
            default="",
            help="directory: rank 0 saves an xprof trace of each case",
        )
        if profile:
            role.add_argument(
                "--llo", default="", help="directory for the compiled LLO of every op"
            )


def _prompt_tokens(rng: _Dynamic, corpus: _Dynamic, size: _Dynamic) -> _Dynamic:
    """``size`` token ids: uniform random, or a random slice of ``corpus``."""
    if corpus is None:
        return rng.integers(1000, 100000, size, dtype=np.int32)
    if size > corpus.size:
        _err_msg = f"--token-file holds {corpus.size} tokens, a prompt needs {size}"
        raise SystemExit(_err_msg)
    start = int(rng.integers(0, corpus.size - size + 1))
    return corpus[start : start + size]


@contextmanager
def _trace(directory: str, tag: str, *, last: bool) -> _Dynamic:
    """Rank 0 saves an xprof trace of the last repeat under ``directory``."""
    ranks = [0]
    if not (directory and last and harness.rank() in ranks):
        yield
        return

    profiler = operator.attrgetter("_impl")(tpu_profiler_mod)

    path = str(Path(directory) / tag)
    if ranks != [0]:
        path = str(Path(directory) / f"rank{harness.rank()}" / tag)
    profiler.start_trace(path)
    try:
        yield
    finally:
        profiler.stop_trace()
        harness.log(f"trace of {tag}: {path}")


def run(args: _Dynamic, records: harness.Records) -> None:
    """Execute run logic."""
    (run_decode if args.role == "decode" else run_prefill)(args, records)


def _get_draft_schedule(
    draft: _Dynamic,
    spec_drafts: int,
    drafts: int,
    max_num_seqs: int,
) -> _Dynamic:
    """Compute draft schedule spans for the decoder."""
    if draft is None:
        return None
    if spec_drafts:
        return ((1, max_num_seqs, spec_drafts),)
    if drafts:
        spans = [(1, min(8, max_num_seqs), drafts)]
        if max_num_seqs > _CONST_8:
            spans.append((9, max_num_seqs, 0))
        return tuple(spans)
    spans = [(1, min(4, max_num_seqs), 7)]
    if max_num_seqs > _CONST_4:
        spans.append((5, min(8, max_num_seqs), 3))
    if max_num_seqs > _CONST_8:
        spans.append((9, max_num_seqs, 0))
    return tuple(spans)


def _benchmark_decode_step(
    *_p_args: _Dynamic, **_p_kw: _Dynamic
) -> tuple[float, list, list]:
    """Execute warmup and timed step repeats for one decode case."""
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        decoder,
        pools,
        batch,
        context,
        tag,
        dp,
        pages,
        sampling,
        draft,
        drafts_count,
        args,
    ) = (
        _p_next(_p_iter, _p_get("decoder")),
        _p_next(_p_iter, _p_get("pools")),
        _p_next(_p_iter, _p_get("batch")),
        _p_next(_p_iter, _p_get("context")),
        _p_next(_p_iter, _p_get("tag")),
        _p_next(_p_iter, _p_get("dp")),
        _p_next(_p_iter, _p_get("pages")),
        _p_next(_p_iter, _p_get("sampling")),
        _p_next(_p_iter, _p_get("draft")),
        _p_next(_p_iter, _p_get("drafts_count")),
        _p_next(_p_iter, _p_get("args")),
    )

    def rows_for(*, first: bool = False) -> _Dynamic:
        """Construct StepInput requests for decode benchmark step."""
        per_row = context if isinstance(context, list) else [context] * batch
        out = []
        for r in range(batch):
            table = pools.scheduled([1 + r // dp * pages + p for p in range(pages)], r)
            speculating = draft is not None and drafts_count > 0 and not first
            query = drafts_count + 1 if speculating else 1
            tokens = [1000 + 37 * r] if first else np.zeros(query, np.int32)
            computed = per_row[r] - 1 if first else 0
            out.append(k3.request(f"{tag}-{r}", tokens, table, sampling, computed))
        return mk.StepInput(out)

    decoder.step(rows_for(first=True))
    bridge.materialize(decoder.step(rows_for(first=False)).tokens)
    times, outs = [], []
    for repeat in range(args.repeats):
        with _trace(args.trace, tag, last=(repeat == args.repeats - 1)):
            started = time.perf_counter()
            outs.extend(
                decoder.step(rows_for(first=False)).tokens for _ in range(args.steps)
            )
            bridge.materialize(outs[-1])
            times.append(1e3 * (time.perf_counter() - started) / args.steps)
    return statistics.median(times), times, outs


def _drafts_at(args: _Dynamic, batch: int) -> int:
    """Return number of draft tokens for the batch size."""
    if args.spec_drafts:
        return args.spec_drafts
    if args.drafts:
        return args.drafts if batch <= _CONST_8 else 0
    return 7 if batch <= _CONST_4 else 3 if batch <= _CONST_8 else 0


def _build_decoder_and_pools(
    args: _Dynamic,
    cases: list,
    max_num_seqs: int,
    max_context: int,
) -> tuple[_Dynamic, _Dynamic, _Dynamic, int, int]:
    """Instantiate DecodeOps, Pools, draft model, and decoder."""
    sub_base = 2 * max_num_seqs + 1
    pages = -(-(max_context + 8 * args.steps * args.repeats + 64) // k3.logical()) + 1
    dp = config.active().attention_dp
    rows = max(batch for batch, _ in cases)
    blocks = 1 + -(-rows // dp) * pages
    sizes = tuple(sorted({b for b, _ in cases if b > 1} | {1}))
    ops = DecodeOps(
        args.presharded,
        num_slots=Control.total_slots(sub_base) if args.dspark else sub_base,
        batch_sizes=sizes,
    )
    pools = k3.Pools(blocks, pages, 1, pattern=True)

    draft = None
    if args.dspark and any(_drafts_at(args, batch) for batch, _ in cases):
        rates = None
        if args.acceptance:
            seven = tuple(float(r) for r in args.acceptance.split(","))
            if len(seven) != _CONST_7:
                _err_msg = "--acceptance takes seven rates"
                raise SystemExit(_err_msg)
            rates = {7: seven, 3: seven[:3]}
        draft = draft_class(args.dspark)(
            max_position=max_context + 8 * args.steps * args.repeats + 256,
            checkpoint=args.dspark,
            presharded=args.presharded,
            synthetic_rates_by_width=dict(rates) if rates is not None else None,
        )
    schedule = _get_draft_schedule(draft, args.spec_drafts, args.drafts, max_num_seqs)
    decoder = k3.decoder(
        ops,
        pools,
        max_num_seqs=max_num_seqs,
        max_model_len=pages * k3.logical(),
        draft=draft,
        schedule=schedule,
    )
    harness.sync()
    decoder.warmup()
    harness.log(
        f"warm: {len(pools.tensors)} pools of {blocks} blocks, batch sizes {sizes}"
    )
    return decoder, pools, draft, dp, pages


def run_decode(args: _Dynamic, records: harness.Records) -> None:
    """Execute run decode logic."""
    batches = [int(b) for b in args.batches.split(",")]
    contexts = [int(c) for c in args.contexts.split(",")]
    mix = [
        int(part.split("x")[0])
        for part in filter(None, args.mix.split(","))
        for _ in range(int(part.split("x")[1]))
    ]
    cases = [(batch, context) for context in contexts for batch in batches] + (
        [(len(mix), mix)] if mix else []
    )
    max_context = max(contexts + mix)
    rows = max(batch for batch, _ in cases)
    max_num_seqs = args.max_num_seqs or rows
    decoder, pools, draft, dp, pages = _build_decoder_and_pools(
        args, cases, max_num_seqs, max_context
    )
    sampling = (args.temperature, 0.95 if args.temperature > 0 else 1.0, 0, 7)
    results, streams = [], {}
    for batch, context in cases:
        label = context if not isinstance(context, list) else "mix " + args.mix
        tag = f"b{batch}c{context if not isinstance(context, list) else 'mix'}"
        d_cnt = _drafts_at(args, batch)
        ms, times, outs = _benchmark_decode_step(
            decoder, pools, batch, context, tag, dp, pages, sampling, draft, d_cnt, args
        )
        dspark = draft is not None and d_cnt > 0
        if dspark:
            tokens = torch.cat([o[:batch] for o in outs]).cpu()
            per_step = float((tokens >= 0).sum()) / len(outs) / batch
            rate = 1e3 * per_step * batch / ms
            line = (
                f"B{batch} ctx {label}: DSpark x{d_cnt} "
                f"{ms:.2f} ms per step, "
                f"{per_step:.2f} tokens per request step -> {rate:.0f} tok/s"
            )
        else:
            rate = batch * 1e3 / ms
            line = f"B{batch} ctx {label}: {ms:.2f} ms per step, {rate:.1f} tok/s"
        harness.log(f"{line}  (repeats {[round(t, 2) for t in times]})")
        records.add(
            case=tag,
            batch=batch,
            context=label,
            dspark=dspark,
            temperature=args.temperature,
            ms=round(ms, 3),
            tokens_per_s=round(rate, 1),
            repeats=[round(t, 3) for t in times],
        )
        results.append((batch, label, ms))
        if args.save_tokens:
            streams[tag] = torch.stack([o[:batch, 0] for o in outs]).cpu().numpy()
        decoder.forget([f"{tag}-{r}" for r in range(batch)])
    if args.save_tokens and harness.rank() == 0:
        np.savez(args.save_tokens, **streams)
    harness.log("summary: " + "; ".join(f"B{b}@{c}={ms:.2f}ms" for b, c, ms in results))


def _parse_prefill_cases(prompts: str) -> list[tuple[int, int, int]]:
    """Parse comma-separated prompt cases."""
    cases = []
    for part in filter(None, prompts.split(",")):
        prompt, _, prefix = part.partition("@")
        length, count = (int(x) for x in prompt.split("x"))
        prefix = int(prefix) if prefix else 0
        if length * count > k3_program.MAX_STEP_TOKENS or count > k3_prefill.SLOTS:
            harness.fail(
                f"prefill case {part}: N * LEN must fit one 8192-token step, "
                f"N <= {k3_prefill.SLOTS}"
            )
        if prefix % k3.PAGE:
            harness.fail(
                f"prefill case {part}: a prefix-cache hit is page-aligned; "
                f"@PREFIX must be a multiple of {k3.PAGE}"
            )
        cases.append((length, count, prefix))
    return cases


def _build_prefill_pools_and_tables(
    cases: list[tuple[int, int, int]],
    max_seqs: int,
    max_model_len: int,
) -> tuple[_Dynamic, list]:
    """Compute slot layout and build prefill pools and block tables."""
    totals = [length + prefix for length, _, prefix in cases]
    slot_layout = []
    for slot in range(max_seqs):
        longest = max(
            total for total, (_, count, _) in zip(totals, cases) if count > slot
        )
        columns = sorted(
            {
                column
                for length, count, prefix in cases
                if count > slot
                for column in ([prefix // k3.PAGE - 1] if prefix else [])
                + [(prefix + length - 1) // k3.PAGE]
            }
        )
        slot_layout.append((-(-longest // k3.LOGICAL), columns))
    pools = k3.Pools(
        1 + sum(mla + 3 * len(columns) for mla, columns in slot_layout),
        max_model_len // k3.LOGICAL,
        max_model_len // k3.PAGE,
        pattern=True,
    )
    tables = []
    for mla, columns in slot_layout:
        kda = []
        for _ in range(3):
            row = np.zeros(max_model_len // k3.PAGE, np.int32)
            for column in columns:
                row[column] = pools.fresh(1)[0]
            kda.append(row.tolist())
        tables.append(pools.tables(pools.fresh(mla), kda))
    return pools, tables


def run_prefill(args: _Dynamic, records: harness.Records) -> None:
    """Execute run prefill logic."""
    cases = _parse_prefill_cases(args.prompts)
    max_seqs = max(count for _, count, _ in cases)
    totals = [length + prefix for length, _, prefix in cases]
    max_model_len = -(-max(totals) // k3.LOGICAL) * k3.LOGICAL
    prefiller = k3_program.build(
        k3.engine_config(
            "prefill",
            max_num_seqs=max_seqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=k3_program.MAX_STEP_TOKENS,
        ),
        args.presharded,
        tp_rank=torch.distributed.get_rank(),
    )
    prefiller.load_weights()
    pools, tables = _build_prefill_pools_and_tables(cases, max_seqs, max_model_len)
    prefiller.bind_kv_caches(pools.caches, pools.groups)
    prefiller.warmup()
    harness.sync()
    rng = np.random.default_rng(0)
    corpus = np.load(args.token_file).astype(np.int32) if args.token_file else None
    largest = prefiller.program.windows[-1]
    for length, count, prefix in cases:
        tag = f"p{length}x{count}" + (f"c{prefix}" if prefix else "")
        pieces = [
            piece
            for r in range(count)
            for piece in k3_prefill.plan_pieces(r, prefix, prefix + length, [], largest)
        ]
        windows = "+".join(
            str(window)
            for window, _ in k3_prefill.pack_windows(
                pieces,
                prefiller.program.windows,
                prefiller.program.slots,
                prefiller.program.capacity,
            )
        )
        times = []
        for repeat in range(args.repeats + 1):
            rows = [
                mk.RequestInput(
                    req_id=f"{tag}-{repeat}-{r}",
                    computed=prefix,
                    scheduled=length,
                    token_ids=_prompt_tokens(rng, corpus, prefix + length),
                    num_tokens=prefix + length,
                    num_prompt_tokens=prefix + length,
                    block_tables=tables[r],
                )
                for r in range(count)
            ]
            with _trace(args.trace, tag, last=(repeat == args.repeats)):
                started = time.perf_counter()
                bridge.materialize(prefiller.step(mk.StepInput(rows)).tokens)
                elapsed = 1e3 * (time.perf_counter() - started)
            if repeat:
                times.append(elapsed)
        ms = statistics.median(times)
        rate = length * count * 1e3 / ms
        on_prefix = f" on a {prefix} prefix" if prefix else ""
        harness.log(
            f"prefill {count} x {length} tokens{on_prefix} (window {windows}): "
            f"{ms:.1f} ms per step, {rate:.0f} tok/s  "
            f"(repeats {[round(t, 1) for t in times]})"
        )
        records.add(
            case=tag,
            length=length,
            count=count,
            context=prefix,
            windows=windows,
            ms=round(ms, 2),
            tokens_per_s=round(rate, 1),
            repeats=[round(t, 2) for t in times],
        )
