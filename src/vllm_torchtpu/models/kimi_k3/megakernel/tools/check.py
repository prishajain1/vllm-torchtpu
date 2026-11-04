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

"""Numerics checks of the K3 programs (``check spec`` / ``check handoff`` /.

``check packing`` / ``check greedy``).
**spec** drives the decoder (``KimiDecoder``) as the runner does, over stand-in
unified pools, and checks greedy DSpark against plain greedy decode (the streams
agree until a near-tie), requests joining and leaving across the single and
batched speculation widths (each stream against its plain reference), seeded
sampling (one seed reproduces, another differs) and that no step after warmup
compiles; it reports tokens per step and pipelined step times.
``--require-verifier-parity`` requires complete output equality against
one-token steps of the same verifier shape for every ``--large-batches`` case.
**handoff** runs the prefill program over chunked prompts and prefix-cache hits
into stand-in pools, binds the decode program to the same pools and decodes
each request from the prefill's first token, against a decode-only replay: the
first tokens must match (later ones may part at a near-tie). With ``--dspark``
each imported draft ring must hold the prompt's last positions, and decoding
from it is compared with a cold ring in tokens per step.
**packing** prefills sixteen prompts of mixed lengths, four continuing cached
prefixes, into separate pool blocks three times: each alone, packed at most
four to a window, and all in one window. Each packing's first tokens, latent
pages and final KDA states (with ``--dspark``, draft rings) are compared with
the prompts alone (relative L2). A packing places rows on other ranks, so the
reductions run in another order and the deep stack amplifies that: the
one-window packing must stay as close as the four-to-a-window reference.
**greedy** replays the fixed prompt and four short prompts on the decoder, then
writes each request's greedy tokens and text to ``--output`` (JSON): two kernel
configs' files compare a layout change (e.g. attention DP) on real text.
**masked** drives the decoder with grammar bitmasks (structured outputs): an
all-ones mask must reproduce the unmasked tokens exactly, random masks must
never yield a disallowed token, and a mask allowing a single token must force
it, greedy or sampled, wherever in the vocabulary the token lives.
"""

from __future__ import annotations

import dataclasses
import functools
import gc
import json
import math
import operator
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vllm_torchtpu.models.kimi_k3.megakernel import prefill as k3_prefill
from vllm_torchtpu.models.kimi_k3.megakernel import program as k3_program
from vllm_torchtpu.models.kimi_k3.megakernel.decode_ops import (
    Control,
    DecodeOps,
    decode_batch_sizes,
)
from vllm_torchtpu.models.kimi_k3.megakernel.draft_mla import (
    MlaDsparkDraft as DsparkDraft,
)
from vllm_torchtpu.models.kimi_k3.megakernel.draft_mla import draft_class
from vllm_torchtpu.models.kimi_k3.megakernel.pools import kda_blocks_at_length
from vllm_torchtpu.models.kimi_k3.megakernel.runtime_shim import bridge
from vllm_torchtpu.models.kimi_k3.megakernel.runtime_shim import program as mk

from . import harness, k3

_Dynamic = Any
_CONST_1000 = 1000
_CONST_2 = 2
_CONST_4 = 4


def add_arguments(parser: _Dynamic) -> None:
    """Execute add arguments logic."""
    sub = parser.add_subparsers(dest="check", required=True)
    spec = sub.add_parser(
        "spec", help="DSpark against plain decode, batch transitions, seeded sampling"
    )
    spec.add_argument("--prompt-tokens", type=int, default=4000)
    spec.add_argument("--generate", type=int, default=200)
    spec.add_argument("--single-drafts", type=int, choices=(3, 7), default=7)
    spec.add_argument("--small-batch-drafts", type=int, choices=(3, 7), default=7)
    spec.add_argument(
        "--large-batches",
        default="",
        help="also speculate with three drafts at these request counts (the "
        "fixture prompts reused), e.g. 16,32 (32: attention DP2 only), against "
        "plain steps of the same requests",
    )
    spec.add_argument(
        "--require-verifier-parity",
        action="store_true",
        help="require every generated token in --large-batches to match "
        "one-token steps of the same verifier shape",
    )
    spec.add_argument(
        "--dspark", required=True, help="the DSpark speculator checkpoint"
    )
    handoff = sub.add_parser(
        "handoff", help="prefill -> pools -> decode against a decode-only replay"
    )
    handoff.add_argument("--generate", type=int, default=32)
    handoff.add_argument("--single-drafts", type=int, choices=(3, 7), default=7)
    handoff.add_argument(
        "--dspark",
        default=None,
        help="the DSpark speculator checkpoint: hand the draft ring over too",
    )
    handoff.add_argument("--speculate-steps", type=int, default=64)
    packing = sub.add_parser(
        "packing", help="prompts packed into one prefill window against each alone"
    )
    packing.add_argument(
        "--dspark",
        default=None,
        help="the DSpark speculator checkpoint: compare the prefill's rings too",
    )
    greedy = sub.add_parser(
        "greedy", help="real prompts replayed on the decoder, then greedy tokens"
    )
    greedy.add_argument("--prompt-tokens", type=int, default=5000)
    greedy.add_argument("--generate", type=int, default=128)
    greedy.add_argument("--output", required=True)
    masked = sub.add_parser(
        "masked", help="grammar-masked (structured outputs) vs unmasked decode"
    )
    masked.add_argument("--prompt-tokens", type=int, default=4000)
    masked.add_argument("--generate", type=int, default=48)
    masked.add_argument(
        "--dspark",
        default=None,
        help="the DSpark speculator checkpoint: run every case on a speculating "
        "decoder (masked steps fall back to plain ones) and check speculation "
        "around a run of masked steps",
    )
    for check in (spec, handoff, packing, greedy, masked):
        check.add_argument(
            "--model",
            required=True,
            help="the Kimi-K3 checkpoint (the tokenizer and the fixed prompt's text)",
        )


def run(args: _Dynamic, records: harness.Records) -> None:
    """Execute run logic."""
    {
        "spec": run_spec,
        "handoff": run_handoff,
        "packing": run_packing,
        "greedy": run_greedy,
        "masked": run_masked,
    }[args.check](args, records)


def run_greedy(args: _Dynamic, records: harness.Records) -> None:
    """Execute run greedy logic."""
    all_ids, tokenizer = k3.fixed_prompt(args.prompt_tokens, args.model)
    prompts = [np.asarray(all_ids, np.int32)] + [
        np.asarray(tokenizer.encode(text, disallowed_special=()), np.int32)
        for text in (
            "Write a short story about a robot who learns to paint.",
            "Explain, step by step, how a transistor amplifies a signal.",
            "List the prime numbers below fifty and say why each is prime.",
            "Once upon a time, in a small village by the sea,",
        )
    ]
    pages = -(-(args.prompt_tokens + args.generate + 64) // k3.logical()) + 1
    # One batch size for every config (under attention DP the five prompts
    # need B8): the replay's single rows and the batch run the same kernel.
    ops = DecodeOps(args.presharded, num_slots=2 * 8 + 1, batch_sizes=(8,))
    pools = k3.Pools(len(prompts) * pages + 1, pages, 1)
    decoder = k3.decoder(
        ops,
        pools,
        max_num_seqs=8,
        max_model_len=pages * k3.logical(),
    )
    harness.sync()
    decoder.warmup()
    tables = [
        pools.scheduled(np.arange(1 + n * pages, 1 + (n + 1) * pages), n)
        for n in range(len(prompts))
    ]

    def step(requests: _Dynamic) -> _Dynamic:
        """Execute step logic."""
        return decoder.step(mk.StepInput(requests)).tokens

    started = time.perf_counter()
    outs = [
        step([k3.request(f"r{n}", ids, tables[n]) for n, ids in enumerate(prompts)])
    ]
    harness.log(f"prompts replayed in {time.perf_counter() - started:.1f} s")
    outs.extend(
        step([k3.request(f"r{n}", [0], tables[n]) for n in range(len(prompts))])
        for _ in range(args.generate - 1)
    )
    streams = torch.stack([o[: len(prompts), 0] for o in outs]).cpu().numpy().T
    result = [
        {
            "prompt_tokens": len(ids),
            "tokens": stream.tolist(),
            "text": tokenizer.decode(stream.tolist()),
        }
        for ids, stream in zip(prompts, streams, strict=True)
    ]
    if harness.rank() == 0:
        with Path(args.output).open("w") as stream:
            json.dump(result, stream, indent=1)
    for n, item in enumerate(result):
        harness.log(
            f"r{n} ({item['prompt_tokens']} prompt tokens): {item['text'][:160]!r}"
        )
    records.add(check="greedy", requests=len(prompts), generate=args.generate)


def _agree(a: list[int], b: list[int]) -> int:
    """Execute agree logic."""
    n = min(len(a), len(b))
    return next((i for i in range(n) if a[i] != b[i]), n)


def _run_masked_decode(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Execute masked decode steps for both prompts."""
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (decoder, names, prompts, tables, tag, steps, masks, sampling) = (
        _p_next(_p_iter, _p_get("decoder")),
        _p_next(_p_iter, _p_get("names")),
        _p_next(_p_iter, _p_get("prompts")),
        _p_next(_p_iter, _p_get("tables")),
        _p_next(_p_iter, _p_get("tag")),
        _p_next(_p_iter, _p_get("steps")),
        _p_next(_p_iter, _p_get("masks")),
        _p_next(_p_iter, _p_get("sampling", (0.0, 1.0, 0, 0))),
    )
    outs = [
        decoder.step(
            mk.StepInput(
                [
                    k3.request(tag + n, ids, table, sampling)
                    for n, ids, table in zip(names, prompts, tables, strict=True)
                ],
                mask=masks(0),
            )
        ).tokens[:2, :1]
    ]
    outs.extend(
        decoder.step(
            mk.StepInput(
                [
                    k3.request(tag + n, [0], table, sampling)
                    for n, table in zip(names, tables, strict=True)
                ],
                mask=masks(step),
            )
        ).tokens[:2, :1]
        for step in range(1, steps)
    )
    decoder.forget(tuple(tag + n for n in names))
    return torch.stack(outs).cpu().numpy().reshape(-1, 2)


def run_masked(args: _Dynamic, records: harness.Records) -> None:
    """Execute run masked logic."""
    all_ids, tokenizer = k3.fixed_prompt(args.prompt_tokens, args.model)
    prompts = [
        np.asarray(all_ids, np.int32),
        np.asarray(
            tokenizer.encode(
                "Write a short story about a robot who learns to paint.",
                disallowed_special=(),
            ),
            np.int32,
        ),
    ]
    words = k3_program.VOCAB // 32
    # Speculative steps take up to eight tokens each.
    pages = -(-(args.prompt_tokens + 8 * args.generate + 64) // k3.logical()) + 1
    ops = DecodeOps(args.presharded, num_slots=2 * 4 + 1, batch_sizes=(1, 2, 4))
    pools = k3.Pools(16 * pages + 1, pages, 1)
    draft = None
    if args.dspark:
        draft = DsparkDraft(
            max_position=args.prompt_tokens + 8 * args.generate + 64,
            checkpoint=args.dspark,
            presharded=args.presharded,
        )
    decoder = k3.decoder(
        ops,
        pools,
        max_num_seqs=4,
        max_model_len=pages * k3.logical(),
        draft=draft,
        # Seven drafts alone, three for two to four requests.
        schedule=((1, 1, 7), (2, 4, 3)) if draft else None,
    )
    harness.sync()
    decoder.warmup()
    tables = [
        pools.scheduled(np.arange(1 + n * pages, 1 + (n + 1) * pages), n)
        for n in range(len(prompts))
    ]
    names = tuple(f"m{n}" for n in range(len(prompts)))

    decode = functools.partial(_run_masked_decode, decoder, names, prompts, tables)

    steps = args.generate
    ones = np.full((2, words), -1, np.int32)
    # The plain reference. A speculating decoder would verify the requests'
    # first step (their anchors); an all-ones mask keeps it plain like the
    # masked runs', so masked and unmasked plain ops are compared alone.
    reference = decode(
        "u", steps, lambda s: ones if draft is not None and s == 0 else None
    )
    unmasked = decode("a", steps, lambda _: ones)
    same = int((unmasked == reference).sum())
    harness.log(
        f"all-ones masks: {same} of {reference.size} tokens identical to unmasked"
    )

    # Random masks: every token the op chooses must be allowed.
    rng = np.random.default_rng(0)
    random_masks = [
        rng.integers(-(2**31), 2**31, (2, words), np.int32) for _ in range(steps)
    ]
    chosen = decode("r", steps, lambda s: random_masks[s])
    violations = sum(
        not (random_masks[s][n][token // 32] >> (token % 32)) & 1
        for s in range(steps)
        for n, token in enumerate(chosen[s])
        if token >= 0
    )
    harness.log(f"random masks: {violations} disallowed tokens of {chosen.size}")

    # A mask allowing a single token forces it, greedy or sampled, wherever
    # the token lives in the (vocabulary-sharded) range.
    forced_ok = True
    for sampling, label in (
        ((0.0, 1.0, 0, 0), "greedy"),
        ((0.7, 0.9, 0, 42), "sampled"),
    ):
        for token in (17, 81920, k3_program.VOCAB - 13):
            mask = np.zeros((2, words), np.int32)
            mask[:, token // 32] = np.int32(1 << (token % 32))
            out = decode("f", 1, lambda _, mask=mask: mask, sampling)
            ok = bool((out[0] == token).all())
            forced_ok &= ok
            harness.log(
                f"forced token {token} ({label}): {out[0].tolist()} "
                f"({'ok' if ok else 'WRONG'})"
            )

    # A masked request must not move its unmasked batch-mate (row mapping).
    blocked = np.full((2, words), -1, np.int32)
    blocked[0, reference[0, 0] // 32] &= ~np.int32(1 << (reference[0, 0] % 32))
    mixed = decode("b", 1, lambda _: blocked)
    row_maps = int(mixed[0, 1]) == int(reference[0, 1])
    blocks = bool(
        int(mixed[0, 0]) != int(reference[0, 0])
        and (blocked[0, mixed[0, 0] // 32] >> (mixed[0, 0] % 32)) & 1
    )
    harness.log(
        f"mixed batch: unmasked row unchanged {row_maps}; masked row blocks "
        f"{int(reference[0, 0])} -> {int(mixed[0, 0])} ({'ok' if blocks else 'WRONG'})"
    )
    spec_ok = True
    if draft is not None:
        spec_ok = _masked_around_speculation(decoder, prompts, tables, names, reference)
    records.add(
        check="masked",
        dspark=bool(draft),
        generate=steps,
        all_ones_identical=[same, int(reference.size)],
        random_violations=int(violations),
        forced_ok=forced_ok,
        mixed_rows_ok=bool(row_maps and blocks),
        speculation_ok=spec_ok,
    )
    ok = same == reference.size and not violations and forced_ok and row_maps and blocks
    if not (ok and spec_ok):
        harness.fail("masked check")


def _masked_around_speculation(
    decoder: _Dynamic,
    prompts: _Dynamic,
    tables: _Dynamic,
    names: _Dynamic,
    reference: _Dynamic,
) -> bool:
    """Both prompts speculating (three drafts each, as vLLM schedules them),.

    then a run of masked steps (all-ones masks: the batch steps plain), then
    speculating again; and the same steps speculating throughout. The masked
    steps must not speculate, speculation must resume after them (a
    multi-token step), and the greedy streams must follow the plain
    reference (until a near-tie: the verify and plain ops reduce in
    different orders) about as far as the pure DSpark streams do.
    """
    ones = np.full((2, k3_program.VOCAB // 32), -1, np.int32)
    plain_ref = [[int(t) for t in reference[:, n]] for n in range(2)]

    def run(tag: _Dynamic, windows: _Dynamic) -> _Dynamic:
        """Execute run logic."""
        streams, bounds = [[], []], []
        first, resumed, stepped_ok = True, bool(0), True
        for window, (label, mask, steps) in enumerate(windows):
            before = dict(decoder.stats)
            for _ in range(steps):
                requests = [
                    k3.request(tag + n, ids if first else np.zeros(4, np.int32), table)
                    for n, ids, table in zip(names, prompts, tables, strict=True)
                ]
                rows = decoder.step(mk.StepInput(requests, mask=mask)).tokens[:2]
                rows = rows.cpu().numpy()
                for n in range(2):
                    tokens = [int(t) for t in rows[n] if t >= 0]
                    streams[n] += tokens
                    resumed |= window == _CONST_2 and len(tokens) > 1
                first = False
            bounds.append([len(st) for st in streams])
            speculated = decoder.stats["batched"] - before.get("batched", 0)
            plain = decoder.stats["plain"] - before.get("plain", 0)
            expected = plain == steps if label == "masked" else speculated > 0
            stepped_ok &= expected
            harness.log(
                f"dspark {tag}, {label} window: {speculated} speculative and "
                f"{plain} plain steps, streams end at {bounds[-1]} "
                f"({'ok' if expected else 'WRONG'})"
            )
        decoder.forget(tuple(tag + n for n in names))
        return streams, bounds, resumed, stepped_ok

    masked_windows = (
        ("speculate", None, 6),
        ("masked", ones, 6),
        ("speculate", None, 10),
    )
    pure_windows = (
        ("speculate", None, 6),
        ("speculate", None, 6),
        ("speculate", None, 10),
    )
    mixed, bounds, resumed, stepped_ok = run("s", masked_windows)
    pure, _, _, _ = run("n", pure_windows)
    vs_plain = [_agree(st, ref) for st, ref in zip(mixed, plain_ref, strict=True)]
    pure_vs_plain = [_agree(st, ref) for st, ref in zip(pure, plain_ref, strict=True)]
    vs_pure = [_agree(a, b) for a, b in zip(mixed, pure, strict=True)]
    harness.log(
        f"dspark, agreement (tokens): masked run vs plain {vs_plain}, pure DSpark "
        f"vs plain {pure_vs_plain}, masked run vs pure DSpark {vs_pure}; the "
        f"masked window spans tokens {bounds[0]} to {bounds[1]}; speculation "
        f"resumed after it: {resumed}"
    )
    return stepped_ok and resumed and all(a > 0 for a in vs_plain)


class _SpecRunner:
    """Orchestrates numerics spec check."""

    def __init__(self, args: _Dynamic, records: harness.Records) -> None:
        """Initialize runner state."""
        self.args = args
        self.records = records
        self.count = args.prompt_tokens
        self.generate = args.generate
        self.rows = args.single_drafts + 1
        self.large = sorted(int(n) for n in args.large_batches.split(",") if n)
        if args.require_verifier_parity and not self.large:
            harness.fail("--require-verifier-parity requires --large-batches")
        self.most = max([8, *self.large])
        self.all_ids, self.tokenizer = k3.fixed_prompt(self.count + 4000, args.model)
        self.ids = np.asarray(self.all_ids[: self.count], np.int32)
        self._init_prompts()
        self._init_models()

    def _encode(self, text: str) -> np.ndarray:
        """Encode text to token array."""
        return np.asarray(self.tokenizer.encode(text, disallowed_special=()), np.int32)

    def _init_prompts(self) -> None:
        """Initialize prompts dictionary."""
        self.prompts = {
            "A": self.ids,
            "B": self._encode("The history of the lighthouse begins"),
            "C": np.asarray(self.all_ids[self.count : self.count + 2000], np.int32),
            "D": self._encode("Write a short story about a robot who learns to paint."),
            "E": self._encode(
                "Explain, step by step, how a transistor amplifies a signal."
            ),
            "F": np.asarray(self.all_ids[self.count + 2000 :], np.int32),
            "G": self._encode(
                "List the prime numbers below fifty and say why each is prime."
            ),
            "H": self._encode("Once upon a time, in a small village by the sea,"),
        }
        self.longest_prompt = max(map(len, self.prompts.values()))
        self.pages = (
            -(-(self.longest_prompt + 4 * self.generate + 64) // k3.logical()) + 1
        )

    def _init_models(self) -> None:
        """Initialize ops, pools, and decoders."""
        self.ops = DecodeOps(
            self.args.presharded,
            num_slots=Control.total_slots(2 * self.most + 1),
            batch_sizes=decode_batch_sizes(self.most),
        )
        self.pools = k3.Pools(
            (10 + max([0, *self.large])) * self.pages + 1, self.pages, 1
        )
        self.draft = draft_class(self.args.dspark)(
            max_position=self.longest_prompt + 8 * self.generate + 64,
            checkpoint=self.args.dspark,
            presharded=self.args.presharded,
        )
        self.single = self._decoder(((1, 1, self.args.single_drafts), (2, 8, 0)))
        self.plain = self._decoder()
        self.batched = self._decoder(
            (
                (1, 1, self.args.single_drafts),
                (2, 4, self.args.small_batch_drafts),
                (5, 8, 3),
            )
        )
        harness.sync()
        self.tables = {
            name: self.pools.scheduled(
                np.arange(1 + n * self.pages, 1 + (n + 1) * self.pages, dtype=np.int32),
                n,
            )
            for n, name in enumerate(("A", "B", "ref", "C", "D", "E", "F", "G", "H"))
        }

    def _decoder(self, schedule: _Dynamic = None) -> _Dynamic:
        """Construct a decoder."""
        return k3.decoder(
            self.ops,
            self.pools,
            max_num_seqs=self.most,
            max_model_len=self.pages * k3.logical(),
            draft=self.draft,
            schedule=schedule,
        )

    def tokens_of(self, emitted: _Dynamic) -> list[int]:
        """Convert emitted tensors to token list."""
        return [int(t) for t in torch.cat(emitted).cpu().reshape(-1).tolist() if t >= 0]

    def row(
        self,
        req: _Dynamic,
        tokens: _Dynamic,
        table: _Dynamic,
        sampling: _Dynamic = (0.0, 1.0, 0, 0),
    ) -> _Dynamic:
        """Create request input."""
        return k3.request(req, tokens, table, sampling)

    def decode(
        self, decoder: _Dynamic, requests: _Dynamic, finished: _Dynamic = ()
    ) -> _Dynamic:
        """Execute single decode step."""
        return decoder.step(mk.StepInput(requests, finished)).tokens

    def plain_reference(self, prompt: _Dynamic, n: int) -> list[int]:
        """Generate plain reference sequence."""
        outs = [
            self.decode(self.plain, [self.row("ref", prompt, self.tables["ref"])])[:1]
        ]
        outs.extend(
            self.decode(self.plain, [self.row("ref", [0], self.tables["ref"])])[:1]
            for _ in range(n - 1)
        )
        self.plain.forget(("ref",))
        return self.tokens_of(outs)

    def run_alone(
        self, req: _Dynamic, sampling: _Dynamic, n_steps: int, *, first: bool = True
    ) -> list[_Dynamic]:
        """Run single request alone."""
        outs = []
        if first:
            outs.append(
                self.decode(
                    self.single, [self.row(req, self.ids, self.tables["A"], sampling)]
                )[:1]
            )
        outs.extend(
            self.decode(
                self.single,
                [self.row(req, np.zeros(self.rows), self.tables["A"], sampling)],
            )[:1]
            for _ in range(n_steps)
        )
        return outs

    def verify_rows(self, requests: int) -> int:
        """Return verification rows for batch size."""
        return int(self.batched.spec_tokens[requests] + 1)

    def _run_single_checks(
        self, reference: list[int], b_reference: list[int]
    ) -> tuple[_Dynamic, ...]:
        """Run single and batch-2 greedy checks."""
        a_tokens = self.tokens_of(self.run_alone("a", (0.0, 1.0, 0, 0), 60))
        same = _agree(a_tokens, reference)
        per_step = (len(a_tokens) - 1) / 60
        harness.log(
            f"greedy DSpark vs plain greedy: first {same} of "
            f"{min(len(a_tokens), len(reference))} agree; "
            f"{per_step:.2f} tokens per speculative step"
        )
        outs_a = [
            self.decode(
                self.single,
                [
                    self.row("a", [0], self.tables["A"]),
                    self.row("b", self.prompts["B"], self.tables["B"]),
                ],
            )[:1]
        ]
        outs_b = [
            self.decode(
                self.single,
                [
                    self.row("a", [0], self.tables["A"]),
                    self.row("b", self.prompts["B"], self.tables["B"]),
                ],
            )[1:2]
        ]
        for _ in range(23):
            out = self.decode(
                self.single,
                [
                    self.row("a", [0], self.tables["A"]),
                    self.row("b", [0], self.tables["B"]),
                ],
            )
            outs_a.append(out[:1])
            outs_b.append(out[1:2])
        b_tokens = self.tokens_of(outs_b)
        outs_a.append(
            self.decode(
                self.single,
                [self.row("a", np.zeros(self.rows), self.tables["A"])],
                ("b",),
            )[:1]
        )
        outs_a += self.run_alone("a", (0.0, 1.0, 0, 0), 20, first=False)
        a_more = self.tokens_of(outs_a)
        same_b = _agree(b_tokens, b_reference)
        harness.log(
            f"B (batch 2 with A) vs its plain reference: "
            f"first {same_b} of {len(b_tokens)} agree"
        )
        harness.log(f"A with B, then alone: {self.tokenizer.decode(a_more[:120])!r}")
        harness.log(f"B: {self.tokenizer.decode(b_tokens)!r}")
        self.single.forget(("a",))
        return a_tokens, same, per_step, b_tokens, same_b

    def _run_single_sampling_and_bench(self) -> tuple[bool, bool, float]:
        """Run sampling reproducibility and timing checks on single decoder."""
        runs = []
        for req, seed in (("s1", 7), ("s2", 7), ("s3", 8)):
            runs.append(self.tokens_of(self.run_alone(req, (0.7, 0.9, 50, seed), 12)))
            self.single.forget((req,))
        reproducible, varies = runs[0] == runs[1], runs[0] != runs[2]
        harness.log(
            f"sampled: same seed identical {reproducible}, "
            f"other seed differs {varies}; "
            f"{self.tokenizer.decode(runs[0][:60])!r}"
        )
        self.run_alone("t", (0.0, 1.0, 0, 0), 4)
        bridge.materialize(self.single.control)
        before = dict(self.single.stats)
        started, host = time.perf_counter(), 0.0
        for _ in range(40):
            t0 = time.perf_counter()
            out = self.decode(
                self.single, [self.row("t", np.zeros(self.rows), self.tables["A"])]
            )
            host += time.perf_counter() - t0
        bridge.materialize(out)
        step_ms = 1e3 * (time.perf_counter() - started) / 40
        harness.log(
            f"pipelined greedy DSpark step: {step_ms:.2f} ms "
            f"(host {1e3 * host / 40:.2f} ms per step); "
            f"ops over 40 steps: "
            f"{ {k: self.single.stats[k] - before[k] for k in before} }"
        )
        self.single.forget(("t",))
        return reproducible, varies, step_ms

    def churn(self) -> tuple[float, float]:
        """Measure join and steady step times under churn."""
        short = {n: self.prompts[n][:24] for n in self.prompts}
        order = [
            ("A",),
            ("A", "B"),
            tuple("ABCD"),
            tuple("ABCDEFGH"),
            tuple("ABC"),
            ("C",),
            ("C", "D"),
            tuple("CDEFG"),
            tuple("ABCDEFGH"),
            ("H",),
        ]
        live, worst_join, worst_steady, log = set(), 0.0, 0.0, []
        for names in order:
            leaving = tuple("c" + n for n in sorted(live - set(names)))
            times = []
            for step in range(4):
                group = []
                for n in names:
                    if n in live:
                        q = self.verify_rows(len(names))
                        group.append(self.row("c" + n, np.zeros(q), self.tables[n]))
                    else:
                        group.append(self.row("c" + n, short[n], self.tables[n]))
                t0 = time.perf_counter()
                out = self.decode(self.batched, group, leaving if step == 0 else ())
                bridge.materialize(out)
                times.append(1e3 * (time.perf_counter() - t0))
                live = set(names)
            worst_join = max(worst_join, times[0])
            worst_steady = max(worst_steady, *times[1:])
            log.append(f"{''.join(names)} " + "/".join(f"{t:.0f}" for t in times))
        self.batched.forget(tuple("c" + n for n in live))
        harness.log(
            "churn after warmup, ms per step (join/steady...): " + "; ".join(log)
        )
        return worst_join, worst_steady

    def run_group(
        self, names: tuple[str, ...], tag: str, sampling: _Dynamic, steps: int
    ) -> tuple[list[list[int]], list[float]]:
        """Run group of requests under batched speculation."""
        first = [
            self.row(tag + n, self.prompts[n], self.tables[n], sampling) for n in names
        ]
        k = self.verify_rows(len(names))
        following = [
            self.row(tag + n, np.zeros(k), self.tables[n], sampling) for n in names
        ]
        count_ = len(names)
        outs = [self.decode(self.batched, first)[:count_]]
        outs.extend(self.decode(self.batched, following)[:count_] for _ in range(steps))
        stacked = torch.stack(outs).cpu().numpy()
        streams = [
            [int(t) for t in stacked[:, r].reshape(-1) if t >= 0] for r in range(count_)
        ]
        rates = [float((stacked[1:, r] >= 0).sum(axis=1).mean()) for r in range(count_)]
        return streams, rates

    def report(
        self,
        label: str,
        names: tuple[str, ...],
        streams: list[list[int]],
        rates: list[float],
        references: dict[str, list[int]],
    ) -> list[int]:
        """Report agreement against plain references."""
        agree = [
            _agree(st, references[n]) for n, st in zip(names, streams, strict=True)
        ]
        harness.log(
            f"batched greedy, {label}: "
            + "; ".join(
                f"{n} agrees for {a} of {min(len(st), len(references[n]))} tokens at "
                f"{r:.2f} tokens per step"
                for n, a, st, r in zip(names, agree, streams, rates, strict=True)
            )
        )
        return agree

    def arriving(self, name: str, *, joining: bool) -> list[int]:
        """Check transferred request arrival."""
        n, req = len(self.prompts[name]), "t" + name
        if not joining:
            arrival = k3.request(
                req, self.prompts[name][-1:], self.tables[name], computed=n - 1
            )
            out = self.decode(self.batched, [arrival])[:1]
            self.batched.forget((req,))
            return self.tokens_of([out])
        self.decode(self.batched, [self.row("tA", self.prompts["A"], self.tables["A"])])
        for _ in range(4):
            self.decode(
                self.batched, [self.row("tA", np.zeros(self.rows), self.tables["A"])]
            )
        k = self.verify_rows(2)
        tokens = np.concatenate((self.prompts[name][-1:], np.full(k - 1, -1)))
        arrival = k3.request(req, tokens, self.tables[name], computed=n - 1)
        arrival = dataclasses.replace(arrival, num_tokens=n)
        out = self.decode(
            self.batched, [self.row("tA", np.zeros(k), self.tables["A"]), arrival]
        )[1:2]
        self.batched.forget(("tA", req))
        return self.tokens_of([out])

    def plain_group(
        self,
        names: tuple[str, ...],
        streams: list[list[int]],
        references: dict[str, list[int]],
        n: int = 112,
    ) -> tuple[list[int], list[int]]:
        """Run plain group reference."""
        first = [self.row("g" + m, self.prompts[m], self.tables[m]) for m in names]
        following = [self.row("g" + m, [0], self.tables[m]) for m in names]
        outs = [self.decode(self.plain, first)[: len(names)]]
        outs.extend(
            self.decode(self.plain, following)[: len(names)] for _ in range(n - 1)
        )
        self.plain.forget(tuple("g" + m for m in names))
        stacked = torch.stack(outs).cpu().numpy()
        tokens = [[int(t) for t in stacked[:, r, 0]] for r in range(len(names))]
        agree = [_agree(t, references[m]) for t, m in zip(tokens, names, strict=True)]
        follow = [_agree(st, t) for st, t in zip(streams, tokens, strict=True)]
        harness.log(
            f"plain batch-{len(names)} greedy: "
            + "; ".join(
                f"{m} agrees for {a} of {n} (the batched stream follows it for "
                f"{f} of {min(n, len(st))})"
                for m, a, f, st in zip(names, agree, follow, streams, strict=True)
            )
        )
        return agree, follow

    def pipelined(
        self,
        decoder: _Dynamic,
        requests: int,
        query: int,
        eight: tuple[str, ...],
        n_steps: int = 40,
    ) -> float:
        """Measure pipelined step latency."""
        names = eight[:requests]
        first = [
            self.row(f"q{r}", self.prompts[n], self.tables[n])
            for r, n in enumerate(names)
        ]
        self.decode(decoder, first)
        following = [
            self.row(f"q{r}", np.zeros(query), self.tables[n])
            for r, n in enumerate(names)
        ]
        for _ in range(4):
            out = self.decode(decoder, following)
        bridge.materialize(out)
        started = time.perf_counter()
        for _ in range(n_steps):
            out = self.decode(decoder, following)
        bridge.materialize(out)
        decoder.forget(tuple(f"q{r}" for r in range(requests)))
        return 1e3 * (time.perf_counter() - started) / n_steps

    def _group_streams(
        self, *_p_args: _Dynamic, **_p_kw: _Dynamic
    ) -> tuple[list[list[int]], list[float]]:
        """Collect group streams for large batch."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (dec, n, query, steps, tag, source, group_tables) = (
            _p_next(_p_iter, _p_get("dec")),
            _p_next(_p_iter, _p_get("n")),
            _p_next(_p_iter, _p_get("query")),
            _p_next(_p_iter, _p_get("steps")),
            _p_next(_p_iter, _p_get("tag")),
            _p_next(_p_iter, _p_get("source")),
            _p_next(_p_iter, _p_get("group_tables")),
        )
        first = [
            self.row(f"{tag}{i}", self.prompts[source[i]][:256], group_tables[i])
            for i in range(n)
        ]
        following = [
            self.row(f"{tag}{i}", np.zeros(query), group_tables[i]) for i in range(n)
        ]
        outs = [self.decode(dec, first)[:n]]
        outs.extend(self.decode(dec, following)[:n] for _ in range(steps))
        dec.forget(tuple(f"{tag}{i}" for i in range(n)))
        stacked = torch.stack(outs).cpu().numpy()
        streams = [
            [int(t) for t in stacked[:, r].reshape(-1) if t >= 0] for r in range(n)
        ]
        rates = [float((stacked[1:, r] >= 0).sum(axis=1).mean()) for r in range(n)]
        return streams, rates

    def _group_ms(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> float:
        """Measure group step latency for large batch."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (dec, n, query, source, group_tables, steps) = (
            _p_next(_p_iter, _p_get("dec")),
            _p_next(_p_iter, _p_get("n")),
            _p_next(_p_iter, _p_get("query")),
            _p_next(_p_iter, _p_get("source")),
            _p_next(_p_iter, _p_get("group_tables")),
            _p_next(_p_iter, _p_get("steps", 40)),
        )
        self.decode(
            dec,
            [
                self.row(f"w{i}", self.prompts[source[i]][:256], group_tables[i])
                for i in range(n)
            ],
        )
        following = [
            self.row(f"w{i}", np.zeros(query), group_tables[i]) for i in range(n)
        ]
        for _ in range(4):
            out = self.decode(dec, following)
        bridge.materialize(out)
        started = time.perf_counter()
        for _ in range(steps):
            out = self.decode(dec, following)
        bridge.materialize(out)
        dec.forget(tuple(f"w{i}" for i in range(n)))
        return 1e3 * (time.perf_counter() - started) / steps

    def _run_large_batch_case(
        self,
        n: int,
        source: list[str],
        group_tables: list[_Dynamic],
        large_results: dict[str, _Dynamic],
    ) -> None:
        """Run single large batch verification case."""
        k = self.batched.spec_tokens[n] + 1
        spec_streams, spec_rates = self._group_streams(
            self.batched, n, k, 32, f"s{n}_", source, group_tables
        )
        plain_streams, _ = self._group_streams(
            self.plain, n, 1, 112, f"p{n}_", source, group_tables
        )
        follow = [
            _agree(st, pt) for st, pt in zip(spec_streams, plain_streams, strict=True)
        ]
        verify_follow = None
        if self.args.require_verifier_parity:
            verify_streams, _ = self._group_streams(
                self.batched,
                n,
                k - 1,
                max(map(len, spec_streams)) - 1,
                f"v{n}_",
                source,
                group_tables,
            )
            expected = max(map(len, spec_streams))
            if any(len(stream) != expected for stream in verify_streams):
                harness.fail("the sequential verifier must emit one token per step")
            verify_follow = [
                _agree(st, vt)
                for st, vt in zip(spec_streams, verify_streams, strict=True)
            ]
            exact = all(
                agree == len(stream)
                for agree, stream in zip(verify_follow, spec_streams, strict=True)
            )
            harness.log(
                f"batch-{n} same-verifier sequential reference: "
                f"{sum(verify_follow)}/{sum(map(len, spec_streams))} tokens match; "
                f"exact {exact}"
            )
            if not exact:
                harness.fail(
                    f"batch-{n} speculative output differs from sequential "
                    f"steps of its verifier: {verify_follow} versus "
                    f"{list(map(len, spec_streams))} tokens"
                )
        spec_ms = self._group_ms(self.batched, n, k, source, group_tables)
        plain_ms = self._group_ms(self.plain, n, 1, source, group_tables)
        large_results[str(n)] = {
            "follow": follow,
            "verifier_follow": verify_follow,
            "generated_tokens": list(map(len, spec_streams)),
            "tokens_per_step": [round(r, 2) for r in spec_rates],
            "ms": [round(spec_ms, 3), round(plain_ms, 3)],
        }
        harness.log(
            f"batched greedy, {n} x {k} rows vs plain batch-{n}: the streams "
            f"follow for {sorted(follow)} tokens (of up to "
            f"{min(len(st) for st in spec_streams)}+), "
            f"{np.mean(spec_rates):.2f} tokens per request-step; step "
            f"{spec_ms:.2f} ms vs plain {plain_ms:.2f} ms; tokens per ms "
            f"{sum(spec_rates) / spec_ms:.3f} vs {n / plain_ms:.3f}"
        )

    def _record_results(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Record all spec outputs and verify assertions."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            same,
            a_tokens,
            reference,
            per_step,
            same_b,
            b_tokens,
            reproducible,
            varies,
            step_ms,
            same_pa,
            same_pb,
            pair_rates,
            same_more,
            four_agree,
            plain_four,
            follow_four,
            four_rates,
            three_agree,
            eight_agree,
            plain_eight,
            follow_eight,
            eight_rates,
            five_agree,
            b_reproducible,
            b_varies,
            timing,
            churn_join,
            churn_steady,
            arrivals,
            large_results,
        ) = (
            _p_next(_p_iter, _p_get("same")),
            _p_next(_p_iter, _p_get("a_tokens")),
            _p_next(_p_iter, _p_get("reference")),
            _p_next(_p_iter, _p_get("per_step")),
            _p_next(_p_iter, _p_get("same_b")),
            _p_next(_p_iter, _p_get("b_tokens")),
            _p_next(_p_iter, _p_get("reproducible")),
            _p_next(_p_iter, _p_get("varies")),
            _p_next(_p_iter, _p_get("step_ms")),
            _p_next(_p_iter, _p_get("same_pa")),
            _p_next(_p_iter, _p_get("same_pb")),
            _p_next(_p_iter, _p_get("pair_rates")),
            _p_next(_p_iter, _p_get("same_more")),
            _p_next(_p_iter, _p_get("four_agree")),
            _p_next(_p_iter, _p_get("plain_four")),
            _p_next(_p_iter, _p_get("follow_four")),
            _p_next(_p_iter, _p_get("four_rates")),
            _p_next(_p_iter, _p_get("three_agree")),
            _p_next(_p_iter, _p_get("eight_agree")),
            _p_next(_p_iter, _p_get("plain_eight")),
            _p_next(_p_iter, _p_get("follow_eight")),
            _p_next(_p_iter, _p_get("eight_rates")),
            _p_next(_p_iter, _p_get("five_agree")),
            _p_next(_p_iter, _p_get("b_reproducible")),
            _p_next(_p_iter, _p_get("b_varies")),
            _p_next(_p_iter, _p_get("timing")),
            _p_next(_p_iter, _p_get("churn_join")),
            _p_next(_p_iter, _p_get("churn_steady")),
            _p_next(_p_iter, _p_get("arrivals")),
            _p_next(_p_iter, _p_get("large_results")),
        )
        self.records.add(
            case="spec",
            single_drafts=self.args.single_drafts,
            small_batch_drafts=self.args.small_batch_drafts,
            dspark_vs_plain=[same, min(len(a_tokens), len(reference))],
            tokens_per_step=per_step,
            batch2_vs_plain=[same_b, len(b_tokens)],
            seed_reproducible=reproducible,
            seed_varies=varies,
            pipelined_ms=round(step_ms, 3),
            batched_pair_agree=[same_pa, same_pb],
            batched_pair_tokens_per_step=[round(r, 2) for r in pair_rates],
            batched_then_alone=same_more,
            batched_four_agree=four_agree,
            plain_four_agree=plain_four,
            batched_four_follow_plain=follow_four,
            batched_four_tokens_per_step=[round(r, 2) for r in four_rates],
            batched_three_agree=three_agree,
            batched_eight_agree=eight_agree,
            plain_eight_agree=plain_eight,
            batched_eight_follow_plain=follow_eight,
            batched_eight_tokens_per_step=[round(r, 2) for r in eight_rates],
            batched_five_agree=five_agree,
            batched_seed_reproducible=b_reproducible,
            batched_seed_varies=b_varies,
            batched_ms={str(b): t for b, t in timing.items()},
            churn_worst_ms=[round(churn_join), round(churn_steady)],
            transferred_arrival_first_tokens=arrivals,
            large_batches=large_results,
        )
        ok = reproducible and varies and same > 0
        ok = ok and all(
            a > 0 for r in large_results.values() for a in (r.get("follow") or ())
        )
        ok = ok and b_reproducible and b_varies and same_pa > 0 and same_pb > 0
        ok = ok and all(
            a > 0 for a in four_agree + three_agree + eight_agree + five_agree
        )
        ok = ok and all(a[:1] == j[:1] for a, j in arrivals.values())
        if churn_steady > _CONST_1000 or churn_join > _CONST_1000:
            harness.log("a step after warmup compiled (churn above)")
            ok = False
        if not ok:
            harness.fail("spec check")

    def _run_batched_suite(
        self, reference: list[int], references: dict[str, list[int]]
    ) -> dict[str, _Dynamic]:
        """Execute batched group checks."""
        churn_join, churn_steady = self.churn()
        pair_streams, pair_rates = self.run_group(("A", "B"), "p", (0.0, 1.0, 0, 0), 48)
        self.decode(self.batched, [], ("pB",))
        alone = [
            self.decode(
                self.batched, [self.row("pA", np.zeros(self.rows), self.tables["A"])]
            )[:1]
            for _ in range(20)
        ]
        pa_more = pair_streams[0] + self.tokens_of(alone)
        self.batched.forget(("pA",))
        same_pa, same_pb = self.report(
            f"2 x {self.verify_rows(2)} rows",
            ("A", "B"),
            pair_streams,
            pair_rates,
            references,
        )
        same_more = _agree(pa_more, reference)
        harness.log(f"A alone again after the pair: {same_more} agree")
        four = ("A", "B", "C", "D")
        four_streams, four_rates = self.run_group(four, "f", (0.0, 1.0, 0, 0), 32)
        self.batched.forget(tuple("f" + n for n in four))
        four_agree = self.report(
            f"4 x {self.verify_rows(4)} rows",
            four,
            four_streams,
            four_rates,
            references,
        )
        arrivals = {
            n: (self.arriving(n, joining=False), self.arriving(n, joining=True))
            for n in ("B", "D")
        }
        harness.log(
            "transferred arrival, first token alone vs joining a speculation "
            "with draft slots: "
            + "; ".join(f"{n} {a} vs {j}" for n, (a, j) in arrivals.items())
        )
        plain_four, follow_four = self.plain_group(four, four_streams, references)
        eight = tuple("ABCDEFGH")
        eight_streams, eight_rates = self.run_group(eight, "e", (0.0, 1.0, 0, 0), 32)
        self.batched.forget(tuple("e" + n for n in eight))
        eight_agree = self.report(
            f"8 x {self.verify_rows(8)} rows",
            eight,
            eight_streams,
            eight_rates,
            references,
        )
        plain_eight, follow_eight = self.plain_group(eight, eight_streams, references)
        five = tuple("ABCDE")
        five_streams, five_rates = self.run_group(five, "v", (0.0, 1.0, 0, 0), 16)
        self.batched.forget(tuple("v" + n for n in five))
        five_agree = self.report(
            f"5 (padded to 8) x {self.verify_rows(5)} rows",
            five,
            five_streams,
            five_rates,
            references,
        )
        three = ("A", "B", "C")
        three_streams, three_rates = self.run_group(three, "t", (0.0, 1.0, 0, 0), 16)
        self.batched.forget(tuple("t" + n for n in three))
        three_agree = self.report(
            f"3 (padded to 4) x {self.verify_rows(3)} rows",
            three,
            three_streams,
            three_rates,
            references,
        )
        sampled_runs = []
        for tag, seed in (("x", 7), ("y", 7), ("z", 8)):
            streams, _ = self.run_group(("A", "B"), tag, (0.7, 0.9, 50, seed), 12)
            self.batched.forget((f"{tag}A", f"{tag}B"))
            sampled_runs.append(streams)
        b_reproducible, b_varies = (
            sampled_runs[0] == sampled_runs[1],
            sampled_runs[0] != sampled_runs[2],
        )
        harness.log(
            f"batched sampled: same seed identical {b_reproducible}, "
            f"other seed differs {b_varies}; "
            f"A {self.tokenizer.decode(sampled_runs[0][0][:40])!r}"
        )
        timing = {}
        for requests, rates in ((2, pair_rates), (4, four_rates), (8, eight_rates)):
            k = self.verify_rows(requests)
            spec_ms, plain_ms = (
                self.pipelined(self.batched, requests, k, eight),
                self.pipelined(self.plain, requests, 1, eight),
            )
            timing[requests] = (round(spec_ms, 3), round(plain_ms, 3))
            harness.log(
                f"pipelined batched DSpark step ({requests} x {k} rows): "
                f"{spec_ms:.2f} ms; plain batch-{requests} step {plain_ms:.2f} ms; "
                f"tokens per ms at the greedy rates above "
                f"{sum(rates) / spec_ms:.3f} vs {requests / plain_ms:.3f}"
            )
        return {
            "churn_join": churn_join,
            "churn_steady": churn_steady,
            "pair_rates": pair_rates,
            "same_pa": same_pa,
            "same_pb": same_pb,
            "same_more": same_more,
            "four_agree": four_agree,
            "plain_four": plain_four,
            "follow_four": follow_four,
            "four_rates": four_rates,
            "arrivals": arrivals,
            "eight_agree": eight_agree,
            "plain_eight": plain_eight,
            "follow_eight": follow_eight,
            "eight_rates": eight_rates,
            "five_agree": five_agree,
            "three_agree": three_agree,
            "b_reproducible": b_reproducible,
            "b_varies": b_varies,
            "timing": timing,
        }

    def run(self) -> None:
        """Run all spec checks."""
        self.single.warmup()
        self.plain.warmup()
        reference = self.plain_reference(self.ids, self.generate)
        b_reference = self.plain_reference(self.prompts["B"], 24)
        harness.log("references done")
        a_tokens, same, per_step, b_tokens, same_b = self._run_single_checks(
            reference, b_reference
        )
        reproducible, varies, step_ms = self._run_single_sampling_and_bench()
        self.batched.warmup()
        references = {
            n: reference
            if n == "A"
            else self.plain_reference(self.prompts[n], self.generate)
            for n in self.prompts
        }
        b_res = self._run_batched_suite(reference, references)
        large_results = {}
        if self.large:
            letters = tuple(self.prompts)
            source = [letters[i % len(letters)] for i in range(self.most)]
            group_tables = [
                self.pools.scheduled(
                    np.arange(
                        1 + (9 + i) * self.pages,
                        1 + (10 + i) * self.pages,
                        dtype=np.int32,
                    ),
                    9 + i,
                )
                for i in range(self.most)
            ]
            for n in self.large:
                self._run_large_batch_case(n, source, group_tables, large_results)
        self._record_results(
            same,
            a_tokens,
            reference,
            per_step,
            same_b,
            b_tokens,
            reproducible,
            varies,
            step_ms,
            b_res["same_pa"],
            b_res["same_pb"],
            b_res["pair_rates"],
            b_res["same_more"],
            b_res["four_agree"],
            b_res["plain_four"],
            b_res["follow_four"],
            b_res["four_rates"],
            b_res["three_agree"],
            b_res["eight_agree"],
            b_res["plain_eight"],
            b_res["follow_eight"],
            b_res["eight_rates"],
            b_res["five_agree"],
            b_res["b_reproducible"],
            b_res["b_varies"],
            b_res["timing"],
            b_res["churn_join"],
            b_res["churn_steady"],
            b_res["arrivals"],
            large_results,
        )


def run_spec(args: _Dynamic, records: harness.Records) -> None:
    """Execute run spec logic."""
    _SpecRunner(args, records).run()


def _handoff_row(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> mk.RequestInput:
    """Construct RequestInput for handoff benchmark."""
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (pools, req_id, computed, scheduled, tokens, prompt_len, mla_table, kda_tables) = (
        _p_next(_p_iter, _p_get("pools")),
        _p_next(_p_iter, _p_get("req_id")),
        _p_next(_p_iter, _p_get("computed")),
        _p_next(_p_iter, _p_get("scheduled")),
        _p_next(_p_iter, _p_get("tokens")),
        _p_next(_p_iter, _p_get("prompt_len")),
        _p_next(_p_iter, _p_get("mla_table")),
        _p_next(_p_iter, _p_get("kda_tables")),
    )
    tokens = np.asarray(tokens, np.int32)
    return mk.RequestInput(
        req_id=req_id,
        computed=computed,
        scheduled=scheduled,
        token_ids=tokens,
        num_tokens=len(tokens),
        num_prompt_tokens=prompt_len,
        block_tables=pools.tables(mla_table, kda_tables),
    )


def _handoff_continue_from(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> list[int]:
    """Continue decode execution from a handoff prompt row."""
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (decoder, generate, req_id, first_row) = (
        _p_next(_p_iter, _p_get("decoder")),
        _p_next(_p_iter, _p_get("generate")),
        _p_next(_p_iter, _p_get("req_id")),
        _p_next(_p_iter, _p_get("first_row")),
    )
    outs = [decoder.step(mk.StepInput([first_row])).tokens[:1]]
    computed = first_row.end
    for _ in range(generate - 1):
        following = mk.RequestInput(
            req_id=req_id,
            computed=computed,
            scheduled=1,
            token_ids=np.zeros(computed + 1, np.int32),
            num_tokens=computed + 1,
            num_prompt_tokens=first_row.num_prompt_tokens,
            block_tables=first_row.block_tables,
        )
        outs.append(decoder.step(mk.StepInput([following])).tokens[:1])
        computed += 1
    decoder.step(mk.StepInput([], (req_id,)))
    return [int(t) for t in torch.cat(outs).cpu().reshape(-1).tolist() if t >= 0]


def _run_handoff_prefill_steps(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> dict:
    """Run prefiller steps across schedule and collect first tokens."""
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (prefiller, schedule, row, lengths, hit, prompts, logical, states, pools) = (
        _p_next(_p_iter, _p_get("prefiller")),
        _p_next(_p_iter, _p_get("schedule")),
        _p_next(_p_iter, _p_get("row")),
        _p_next(_p_iter, _p_get("lengths")),
        _p_next(_p_iter, _p_get("hit")),
        _p_next(_p_iter, _p_get("prompts")),
        _p_next(_p_iter, _p_get("logical")),
        _p_next(_p_iter, _p_get("states")),
        _p_next(_p_iter, _p_get("pools")),
    )
    done = {req: hit[req][1] if req in hit else 0 for req in lengths}
    first_token = {}
    for step in schedule:
        rows = [
            row(
                req,
                done[req],
                total - done[req],
                prompts[req],
                len(prompts[req]),
                logical[req],
                states[req],
            )
            for req, total in step
        ]
        tokens = (
            prefiller.step(mk.StepInput(rows)).tokens[: len(rows), 0].cpu().tolist()
        )
        for (req, total), token in zip(step, tokens, strict=True):
            done[req] = total
            if total == len(prompts[req]):
                first_token[req] = int(token)
    bridge.materialize(pools.tensors[0])
    harness.log(f"prefill done; first tokens {first_token}")
    prefiller.program.weights.clear()
    operator.attrgetter("_expert_banks")(prefiller.program).clear()
    if prefiller.ring is not None:
        prefiller.ring.weights = ()
    return first_token


def _run_handoff_decode_checks(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> bool:
    """Verify handoff decoder output against reference tables."""
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        lengths,
        first_token,
        continue_from,
        row,
        prompts,
        generate,
        logical,
        decode_states,
        reference_tables,
        tokenizer,
        hit,
        records,
    ) = (
        _p_next(_p_iter, _p_get("lengths")),
        _p_next(_p_iter, _p_get("first_token")),
        _p_next(_p_iter, _p_get("continue_from")),
        _p_next(_p_iter, _p_get("row")),
        _p_next(_p_iter, _p_get("prompts")),
        _p_next(_p_iter, _p_get("generate")),
        _p_next(_p_iter, _p_get("logical")),
        _p_next(_p_iter, _p_get("decode_states")),
        _p_next(_p_iter, _p_get("reference_tables")),
        _p_next(_p_iter, _p_get("tokenizer")),
        _p_next(_p_iter, _p_get("hit")),
        _p_next(_p_iter, _p_get("records")),
    )
    all_first_match = True
    for req, n in lengths.items():
        pd = [
            first_token[req],
            *continue_from(
                f"pd-{req}",
                row(
                    f"pd-{req}",
                    n,
                    1,
                    prompts[req] + [first_token[req]],
                    n,
                    logical[req],
                    decode_states[req],
                ),
            )[: generate - 1],
        ]
        ref = continue_from(
            f"ref-{req}",
            row(
                f"ref-{req}", 0, n, prompts[req], n, reference_tables[req], [[], [], []]
            ),
        )
        same = _agree(pd, ref)
        all_first_match &= pd[0] == ref[0]
        harness.log(
            f"{req} ({n} tokens{', hit ' + str(hit[req][1]) if req in hit else ''}): "
            f"first token {'matches' if pd[0] == ref[0] else 'DIFFERS'}; "
            f"first {same} of {len(ref)} tokens agree; "
            f"P/D {tokenizer.decode(pd[:24])!r}; "
            f"reference {tokenizer.decode(ref[:24])!r}"
        )
        records.add(
            case=f"handoff-{req}",
            prompt_tokens=n,
            first_token_matches=pd[0] == ref[0],
            agree=[same, len(ref)],
        )
    return all_first_match


def _init_handoff_prompts_and_schedule(
    ids: _Dynamic,
) -> tuple[_Dynamic, _Dynamic, _Dynamic, _Dynamic]:
    """Build prompts and schedule for handoff numerics check."""
    lengths = {
        "A": k3.LOGICAL + 3808,
        "B": k3.LOGICAL + 1500,
        "C": 700,
        "D": k3.PAGE,
        "E": 100,
        "F": k3.LOGICAL + 400,
    }
    hit = {"B": ("A", k3.LOGICAL), "F": ("A", k3.LOGICAL)}
    prompts, at = {}, 0
    for req, n in lengths.items():
        if req in hit:
            donor, cached = hit[req]
            prompts[req] = prompts[donor][:cached] + ids[at : at + n - cached]
            at += n - cached
        else:
            prompts[req] = ids[at : at + n]
            at += n
    schedule = [
        [("A", 4096), ("C", 700)],
        *[[("A", end)] for end in range(8192, k3.LOGICAL + 1, 4096)],
        [("A", lengths["A"])],
        [("B", lengths["B"]), ("D", k3.PAGE)],
        [("E", 100)],
        [("F", lengths["F"])],
    ]
    return lengths, hit, prompts, schedule


def run_handoff(args: _Dynamic, records: harness.Records) -> None:
    """Execute run handoff logic."""
    generate, max_model_len = args.generate, 32768
    rank = bridge.kernel_mesh("tp", k3_program.TP_SIZE)[1]
    ids, tokenizer = k3.fixed_prompt(24000, args.model)
    lengths, hit, prompts, schedule = _init_handoff_prompts_and_schedule(ids)
    decode_pages = max_model_len // k3.LOGICAL + 1
    pools = k3.Pools(
        1
        + 3 * decode_pages
        + sum(-(-n // k3.LOGICAL) + 3 * -(-n // k3.PAGE) for n in lengths.values())
        + 8,
        decode_pages,
        max_model_len // k3.PAGE,
    )
    logical, states = {}, {}
    for req, n in lengths.items():
        blocks = -(-n // k3.LOGICAL)
        if req in hit:
            donor, cached = hit[req]
            logical[req] = logical[donor][: cached // k3.LOGICAL] + pools.fresh(
                blocks - cached // k3.LOGICAL
            )
        else:
            logical[req] = pools.fresh(blocks)
        states[req] = [pools.fresh(-(-n // k3.PAGE)) for _ in range(3)]
    for req, (donor, cached) in hit.items():
        for g in range(3):
            states[req][g][cached // k3.PAGE - 1] = states[donor][g][
                cached // k3.PAGE - 1
            ]
    decode_states = {
        req: [s + pools.fresh(-(-(n + 1) // k3.PAGE) - len(s)) for s in states[req]]
        for req, n in lengths.items()
    }
    reference_tables = {req: pools.fresh(decode_pages) for req in lengths}
    row = functools.partial(_handoff_row, pools)

    draft = Path(args.dspark) if args.dspark else None
    prefiller = k3_program.build(
        k3.engine_config(
            "prefill",
            max_num_seqs=4,
            max_model_len=max_model_len,
            max_num_batched_tokens=4096,
        ),
        args.presharded,
        draft,
        tp_rank=rank,
    )
    prefiller.load_weights()
    prefiller.bind_kv_caches(pools.caches, pools.groups)
    harness.sync()
    first_token = _run_handoff_prefill_steps(
        prefiller, schedule, row, lengths, hit, prompts, logical, states, pools
    )
    del prefiller
    gc.collect()
    harness.sync()

    decoder = k3_program.build(
        k3.engine_config(
            "decode",
            max_num_seqs=2,
            max_model_len=max_model_len,
            max_num_batched_tokens=512,
            num_speculative_tokens=args.single_drafts if draft else 0,
            num_speculative_tokens_per_batch_size=(
                (1, 1, args.single_drafts),
                (2, 2, 0),
            )
            if draft
            else None,
        ),
        args.presharded,
        draft,
        tp_rank=rank,
    )
    decoder.load_weights()
    decoder.bind_kv_caches(pools.caches, pools.groups)
    harness.sync()

    continue_from = functools.partial(_handoff_continue_from, decoder, generate)
    all_first_match = _run_handoff_decode_checks(
        lengths,
        first_token,
        continue_from,
        row,
        prompts,
        generate,
        logical,
        decode_states,
        reference_tables,
        tokenizer,
        hit,
        records,
    )
    rings_ok = True
    if draft is not None:
        rings_ok = _handoff_rings(
            args,
            records,
            decoder,
            lengths,
            row,
            prompts,
            first_token,
            logical,
            decode_states,
        )
    if not all_first_match:
        harness.fail("a prefill first token differs from the decode-only reference")
    if not rings_ok:
        harness.fail("an imported draft ring does not hold the prompt's last positions")


def _handoff_rings(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> bool:
    """The DSpark half of ``check handoff`` (see the module docstring)."""
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    (
        args,
        records,
        decoder,
        lengths,
        row,
        prompts,
        first_token,
        logical,
        states,
    ) = (
        _p_next(_p_iter, _p_get("args")),
        _p_next(_p_iter, _p_get("records")),
        _p_next(_p_iter, _p_get("decoder")),
        _p_next(_p_iter, _p_get("lengths")),
        _p_next(_p_iter, _p_get("row")),
        _p_next(_p_iter, _p_get("prompts")),
        _p_next(_p_iter, _p_get("first_token")),
        _p_next(_p_iter, _p_get("logical")),
        _p_next(_p_iter, _p_get("states")),
    )
    window, steps, ok = decoder.draft.config.window, args.speculate_steps, True
    for req, n in lengths.items():
        first = row(
            f"spec-{req}",
            n,
            1,
            prompts[req] + [first_token[req]],
            n,
            logical[req],
            states[req],
        )
        blocks = kda_blocks_at_length(
            first.block_tables, operator.attrgetter("_kda_groups")(decoder), n
        )
        _, positions = operator.attrgetter("_importer")(decoder).import_ring(
            decoder.draft.config, torch.from_numpy(blocks).to("tpu"), n
        )
        held = int((positions >= 0).sum().cpu())
        ok &= held == min(n, window)
        counts = {}
        for arm, ring in (("ring", bool(1)), ("cold", bool(0))):
            vars(decoder)["_ring"] = ring
            req_id = f"spec-{arm}-{req}"
            outs = [
                decoder.step(
                    mk.StepInput([dataclasses.replace(first, req_id=req_id)])
                ).tokens[:1]
            ]
            following = mk.RequestInput(
                req_id=req_id,
                computed=n + 1,
                scheduled=args.single_drafts + 1,
                token_ids=np.zeros(n + args.single_drafts + 2, np.int32),
                num_tokens=n + args.single_drafts + 2,
                num_prompt_tokens=n,
                block_tables=first.block_tables,
            )
            outs.extend(
                decoder.step(mk.StepInput([following])).tokens[:1]
                for _ in range(steps - 1)
            )
            decoder.step(mk.StepInput([], (req_id,)))
            emitted = torch.cat(outs).cpu().numpy()
            counts[arm] = (emitted >= 0).sum(axis=1)
        vars(decoder)["_ring"] = True
        windows = [w for w in (16, 64) if w < steps] + [steps]
        rates = {
            arm: [round(float(counts[arm][:w].mean()), 2) for w in windows]
            for arm in counts
        }
        harness.log(
            f"{req} ({n} tokens): imported ring holds {held} positions "
            f"(expected {min(n, window)}); DSpark tokens per step ring / cold over "
            "the first "
            + "; ".join(
                f"{w} steps: {rates['ring'][i]} / {rates['cold'][i]}"
                for i, w in enumerate(windows)
            )
        )
        records.add(
            case=f"handoff-ring-{req}",
            prompt_tokens=n,
            ring_positions=held,
            tokens_per_step_ring=rates["ring"],
            tokens_per_step_cold=rates["cold"],
        )
    return ok


# Pages per ``check packing`` prompt when the window has two pages a sequence
# (BF16 pools); FP8's 512-token pages leave one page each.
_PACKING_PAGES = (1, 2, 1, 4, 1, 3, 1, 2, 1, 1, 5, 2, 1, 3, 2, 2)
_PACKING_PREFIX_PAGES = (2, 4, 8, 3)


def _packing_cases(page: int, window: int, slots: int) -> list[tuple[int, int]]:
    """``check packing``'s prompts, ``(new tokens, cached prefix tokens)``:.

    ``slots`` prompts whose page-padded rows fill one ``window`` exactly, the
    last four on cached prefixes; they end mid-page, on a page or (one) after
    a single token.
    """
    pages = list(_PACKING_PAGES)
    if len(pages) != slots or sum(pages) != window // page:
        pages = [window // page // slots] * slots
    cases = []
    for i, n in enumerate(pages):
        trim = 0 if i % 4 == 1 else (37 * i + 11) % page
        length = 1 if i == _CONST_4 else n * page - trim
        hit = i - (slots - len(_PACKING_PREFIX_PAGES))
        cases.append((length, _PACKING_PREFIX_PAGES[hit] * page if hit >= 0 else 0))
    return cases


class _PackingRunner:
    """Numerics packing runner."""

    def __init__(self, args: _Dynamic, records: harness.Records) -> None:
        """Initialize packing runner."""
        self.args = args
        self.records = records
        self.window = k3_program.MAX_STEP_TOKENS
        self.slots = k3_prefill.SLOTS
        self.max_model_len = 2 * self.window
        self.cases = _packing_cases(k3.PAGE, self.window, self.slots)
        self.hits = [i for i, (_, prefix) in enumerate(self.cases) if prefix]
        self.ids, self.tokenizer = k3.fixed_prompt(
            sum(n + p for n, p in self.cases), args.model
        )
        self.prompts = []
        at = 0
        for length, prefix in self.cases:
            self.prompts.append(self.ids[at : at + prefix + length])
            at += prefix + length
        self.mla_width = self.max_model_len // k3.LOGICAL
        self.state_width = self.max_model_len // k3.PAGE
        self.per_request = -(-self.max_model_len // k3.LOGICAL) + 3
        self.pools = k3.Pools(
            1 + (len(self.hits) + 3 * len(self.cases)) * self.per_request,
            self.mla_width,
            self.state_width,
        )
        rank = bridge.kernel_mesh("tp", k3_program.TP_SIZE)[1]
        draft = Path(args.dspark) if args.dspark else None
        self.prefiller = k3_program.build(
            k3.engine_config(
                "prefill",
                max_num_seqs=self.slots,
                max_model_len=self.max_model_len,
                max_num_batched_tokens=self.window,
            ),
            args.presharded,
            draft,
            tp_rank=rank,
        )
        self.prefiller.load_weights()
        self.prefiller.bind_kv_caches(self.pools.caches, self.pools.groups)
        self.prefiller.warmup()
        harness.sync()

    def tables(
        self, end: int, donor: _Dynamic = None, cached: int = 0
    ) -> tuple[list[int], list[list[int]]]:
        """Fresh blocks for a request end tokens long."""
        mla = self.pools.fresh(-(-end // k3.LOGICAL))
        kda = [[0] * self.state_width for _ in range(3)]
        for g in range(3):
            kda[g][(end - 1) // k3.PAGE] = self.pools.fresh(1)[0]
            if donor is not None:
                column = cached // k3.PAGE - 1
                kda[g][column] = donor[1][g][column]
        return mla, kda

    def row(
        self, req_id: str, tokens: _Dynamic, computed: int, table: _Dynamic
    ) -> mk.RequestInput:
        """Create RequestInput for packing step."""
        tokens = np.asarray(tokens, np.int32)
        return mk.RequestInput(
            req_id=req_id,
            computed=computed,
            scheduled=len(tokens) - computed,
            token_ids=tokens,
            num_tokens=len(tokens),
            num_prompt_tokens=len(tokens),
            block_tables=self.pools.tables(*table),
        )

    def step(self, rows: list[mk.RequestInput], copies: _Dynamic = ()) -> _Dynamic:
        """Execute prefill step."""
        return self.prefiller.step(mk.StepInput(rows, copies=tuple(copies))).tokens

    def device(self, values: _Dynamic, dtype: _Dynamic = torch.int32) -> torch.Tensor:
        """Move values to device."""
        return torch.from_numpy(np.asarray(values)).to(dtype).to("tpu")

    def squares(self, a: torch.Tensor, b: torch.Tensor) -> list[float]:
        """Compute sum of squared difference and norm."""
        a, b = a.to(torch.float32), b.to(torch.float32)
        return [float(((a - b) ** 2).sum().cpu()), float((a**2).sum().cpu())]

    def owned_pages(self, table: _Dynamic, prefix: int, length: int) -> list[int]:
        """Compute pages owned by local rank."""
        cp = k3.LOGICAL // k3.PAGE
        pages = range(prefix // k3.PAGE, (prefix + length) // k3.PAGE)
        return [table[0][k // cp] for k in pages if k % cp == self.prefiller.cp_rank]

    def compare(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> dict[str, _Dynamic]:
        """Compute comparison metrics for request."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (arm, i, length, prefix, runs, states, rings) = (
            _p_next(_p_iter, _p_get("arm")),
            _p_next(_p_iter, _p_get("i")),
            _p_next(_p_iter, _p_get("length")),
            _p_next(_p_iter, _p_get("prefix")),
            _p_next(_p_iter, _p_get("runs")),
            _p_next(_p_iter, _p_get("states")),
            _p_next(_p_iter, _p_get("rings")),
        )
        entry = {}
        pages = [
            self.owned_pages(runs[a][1][i], prefix, length) for a in ("alone", arm)
        ]
        if pages[0]:
            index = [self.device(p, torch.int64) for p in pages]
            per_layer = [
                self.squares(pool[index[0]], pool[index[1]])
                for pool in self.pools.tensors
            ]
            entry["latents"] = np.sum(per_layer, axis=0).tolist()
            entry["latents0"] = per_layer[0]
        for name, part in (("recurrent", 0), ("conv", 1)):
            entry[name] = self.squares(
                states["alone"][part][0, :, i], states[arm][part][0, :, i]
            )
        if rings:
            entry["ring"] = self.squares(rings["alone"][i], rings[arm][i])
        return entry

    def _execute_arm(
        self, arm: str, limit: int, donors: dict[int, _Dynamic]
    ) -> tuple[list[int], list[_Dynamic], list[_Dynamic]]:
        """Run single packing arm."""
        blocks = [
            self.tables(prefix + length, donors.get(i), prefix)
            for i, (length, prefix) in enumerate(self.cases)
        ]
        rows = [
            self.row(f"{arm}-{i}", self.prompts[i], prefix, blocks[i])
            for i, (_, prefix) in enumerate(self.cases)
        ]
        copies = {i: (donors[i][0][0], blocks[i][0][0]) for i in self.hits}
        if arm == "alone":
            first = []
            for i, one in enumerate(rows):
                tokens = self.step([one], [copies[i]] if i in copies else ())
                first.append(int(tokens[0, 0].cpu()))
        else:
            pieces = [
                k3_prefill.plan_pieces(i, prefix, prefix + n, [], self.window, k3.PAGE)[
                    0
                ]
                for i, (n, prefix) in enumerate(self.cases)
            ]
            plan = k3_prefill.pack_windows(
                pieces,
                self.prefiller.program.windows,
                limit,
                self.prefiller.program.capacity,
            )
            shapes = [f"{len(m)} sequences in a {w}-token window" for w, m in plan]
            harness.log(f"{arm}: " + "; ".join(shapes))
            if arm == "packed" and (
                len(plan) != 1 or len(plan[0][1]) != len(self.cases)
            ):
                harness.fail(
                    f"the {len(self.cases)} prompts did not pack into one window"
                )
            self.prefiller.slots = limit
            tokens = self.step(rows, list(copies.values()))
            self.prefiller.slots = self.slots
            first = [int(t) for t in tokens[: len(rows), 0].cpu().tolist()]
        finals = [
            kda_blocks_at_length(
                r.block_tables,
                operator.attrgetter("_kda_groups")(self.prefiller),
                r.end,
            )
            for r in rows
        ]
        return first, blocks, finals

    def run(self) -> None:
        """Run all packing checks."""
        donors = {i: self.tables(self.cases[i][1]) for i in self.hits}
        self.step(
            [
                self.row(
                    f"donor-{i}", self.prompts[i][: self.cases[i][1]], 0, donors[i]
                )
                for i in self.hits
            ]
        )
        runs = {}
        for arm, limit in (("alone", 1), ("packed4", 4), ("packed", self.slots)):
            runs[arm] = self._execute_arm(arm, limit, donors)
        bridge.materialize(self.pools.tensors[0])
        states = {
            arm: operator.attrgetter("_writer")(self.prefiller).read_states(
                self.device(np.concatenate(runs[arm][2]))
            )
            for arm in runs
        }
        rings = {}
        if self.prefiller.ring is not None:
            for arm in runs:
                rings[arm] = [
                    operator.attrgetter("_writer")(self.prefiller).read_rings(
                        self.prefiller.ring.config,
                        self.device(runs[arm][2][i]),
                        [prefix + length],
                    )[0][0]
                    for i, (length, prefix) in enumerate(self.cases)
                ]
        arms = ("packed4", "packed")
        sums = {
            arm: [
                self.compare(arm, i, n, prefix, runs, states, rings)
                for i, (n, prefix) in enumerate(self.cases)
            ]
            for arm in arms
        }
        gathered: list[dict[str, list[dict[str, tuple[float, float]]]]] = [
            {} for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather_object(gathered, sums)
        names = ("latents0", "latents", "recurrent", "conv", "ring")

        def relative(arm: str, i: int, name: str) -> float | None:
            """Calculate relative L2 error."""
            num = sum(g[arm][i].get(name, (0.0, 0.0))[0] for g in gathered)
            den = sum(g[arm][i].get(name, (0.0, 0.0))[1] for g in gathered)
            return math.sqrt(num / den) if den else None

        alone = runs["alone"][0]
        worst = {arm: dict.fromkeys(names, 0.0) for arm in arms}
        matches = {
            arm: sum(a == b for a, b in zip(alone, runs[arm][0])) for arm in arms
        }
        for i, (length, prefix) in enumerate(self.cases):
            line = []
            for arm in arms:
                errors = {n: relative(arm, i, n) for n in names}
                errors = {n: v for n, v in errors.items() if v is not None}
                for name, value in errors.items():
                    worst[arm][name] = max(worst[arm][name], value)
                token = runs[arm][0][i]
                line.append(
                    f"{arm} {'same' if token == alone[i] else 'DIFFERS'} "
                    f"{self.tokenizer.decode([token])!r} "
                    + " ".join(f"{k} {v:.1e}" for k, v in errors.items())
                )
                self.records.add(
                    case=f"packing-{arm}-{i}",
                    length=length,
                    prefix=prefix,
                    first_token_matches=token == alone[i],
                    **{f"rel_l2_{k}": round(v, 6) for k, v in errors.items()},
                )
            harness.log(
                f"{i:2d}: {length} tokens"
                + (f" on a {prefix}-token prefix" if prefix else "")
                + f", alone {self.tokenizer.decode([alone[i]])!r}: "
                + "; ".join(line)
            )
        for arm in arms:
            harness.log(
                f"{arm}: {matches[arm]} of {len(self.cases)} first tokens match alone; "
                "worst relative L2 "
                + ", ".join(f"{k} {v:.1e}" for k, v in worst[arm].items() if v)
            )
        if matches["packed"] < matches["packed4"] - 1 or any(
            worst["packed"][n] > 2 * worst["packed4"][n] + 0.02 for n in names
        ):
            harness.fail(
                "one-window packing departs further than the reference packing"
            )


def run_packing(args: _Dynamic, records: harness.Records) -> None:
    """Execute run packing logic."""
    _PackingRunner(args, records).run()
