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
"""DSpark MLA under attention DP: verify-row placement and parts drafting."""

from __future__ import annotations

import operator
import types
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.kimi_k3.megakernel.decode import KimiDecoder
from vllm_torchtpu.models.kimi_k3.megakernel.decode_ops import Control, dp_group_rows
from vllm_torchtpu.models.kimi_k3.megakernel.draft_mla import (
    MlaDsparkDraft,
)
from vllm_torchtpu.models.kimi_k3.megakernel.draft_mla import (
    _batched_fn as _mla_batched_fn,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kernels import dspark_common as dspark
from vllm_torchtpu.models.kimi_k3.megakernel.program import spec_tokens_per_batch
from vllm_torchtpu.models.kimi_k3.megakernel.runtime_shim.config import (
    MegakernelEngineConfig,
)

_Dynamic = Any
_CONST_8 = 8


pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("batch", [8, 16, 32, 64, 96, 128])
def test_verify_rows_are_the_rows_each_group_attends(batch: _Dynamic) -> _Dynamic:
    """Verify test verify rows are the rows each group attends logic."""
    rows = dp_group_rows(batch, 2)
    decoder = types.SimpleNamespace(groups=2)
    expected = [
        operator.attrgetter("_group_rows")(KimiDecoder)(decoder, batch, g)
        for g in range(2)
    ]
    assert rows.tolist() == expected[0] + expected[1]


def _stub_mla_fused(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Execute stub mla fused logic."""
    _p_iter = iter(_p_args)
    next(_p_iter, None)
    next(_p_iter, None)
    aux = next(_p_iter, _p_kw.get("aux"))
    logits = next(_p_iter, _p_kw.get("logits"))
    previous = next(_p_iter, _p_kw.get("previous"))
    bases = next(_p_iter, _p_kw.get("bases"))
    count = next(_p_iter, _p_kw.get("count"))
    anchor = next(_p_iter, _p_kw.get("anchor"))
    tables = next(_p_iter, _p_kw.get("tables"))
    out = jnp.zeros((bases.shape[0], 128), jnp.int32)
    out = out.at[:, dspark.OUT_COUNT].set(count).at[:, dspark.OUT_BONUS].set(anchor)
    out = out.at[:, 40].set(aux[0, :: logits.shape[1], 0].astype(jnp.int32))
    out = out.at[:, 41].set(logits[:, 0, 0].astype(jnp.int32))
    out = out.at[:, 42].set(tables[:, 0]).at[:, 43].set(previous[:, 0])
    rows = aux[0, :, :1].astype(jnp.float32)
    return out, jnp.stack([rows, rows + 1])


def _stub_commit(
    pools: _Dynamic, latents: _Dynamic, _tables: _Dynamic, positions: _Dynamic
) -> _Dynamic:
    """Execute stub commit logic."""
    return tuple(
        pool.at[positions].set(latents[layer, :, 0]) for layer, pool in enumerate(pools)
    )


@pytest.mark.parametrize(("verified", "dp"), [(16, 1), (16, 2), (32, 2)])
def test_mla_drafting_in_parts_matches_one_call(
    verified: _Dynamic, dp: _Dynamic
) -> _Dynamic:
    """Verify test mla drafting in parts matches one call logic."""
    rng = np.random.default_rng(verified + dp)
    rows_per, pages = 4, 3
    control = Control(slots=65, rows=64, pages=pages)
    slots = rng.permutation(np.arange(1, 65))[:verified]
    entries = [
        (int(s), i % 3 == 0, int(rng.integers(5, 999)), 0, (0.0, 1.0, 0, 0))
        for i, s in enumerate(slots)
    ]
    plan = control.plan(entries, [np.arange(pages, dtype=np.int32)] * verified)
    state = jnp.asarray(
        np.concatenate([rng.integers(0, 50, 3 * control.slots), plan]), jnp.int32
    )
    total = verified * rows_per
    bases = 100 * rng.permutation(verified)
    positions = (bases[:, None] + np.arange(rows_per)[None, :]).reshape(-1)
    ids = np.concatenate(
        [
            rng.integers(0, 1000, total),
            positions,
            rng.integers(0, 1000, 2 * total),
            rng.integers(0, 1000, verified * pages),
        ]
    )
    ids = jnp.asarray(ids, jnp.int32)
    aux = jnp.asarray(rng.integers(0, 99, (5, total, 16)), jnp.bfloat16)
    logits = jnp.asarray(rng.integers(0, 99, (1, total, 32)), jnp.float32)
    decisions = jnp.asarray(
        np.stack(
            [rng.integers(0, rows_per, verified), rng.integers(0, 999, verified)], 1
        ),
        jnp.int32,
    )
    previous = jnp.asarray(rng.integers(0, 999, (verified, 128)), jnp.int32)
    pools = (jnp.zeros(100 * verified, jnp.float32),) * 2

    def draft(index: _Dynamic, state: _Dynamic, pools: _Dynamic) -> _Dynamic:
        """Execute draft logic."""
        fn = _mla_batched_fn(
            dspark,
            _stub_mla_fused,
            _stub_commit,
            control,
            (),
            2,
            index,
            verified,
            rows_per,
        )
        r = len(index)
        out = fn(
            *pools,
            *(previous[j : j + 1] for j in index),
            aux,
            logits,
            ids,
            state,
            decisions,
        )
        return out[:r], out[r : r + 2], out[r + 2], out[r + 3]

    whole = draft(tuple(range(verified)), state, pools)
    vectors, emitted, order = [None] * verified, [], []
    part_state, part_pools = state, pools
    for index in MlaDsparkDraft.parts(verified, dp):
        assert len(index) <= _CONST_8
        assert [j // (verified // dp) for j in index] == sorted(
            list(range(dp)) * (len(index) // dp)
        )
        part_vectors, part_pools, part_emitted, part_state = draft(
            index, part_state, part_pools
        )
        for j, vector in zip(index, part_vectors, strict=True):
            vectors[j] = vector
        emitted.append(part_emitted)
        order.extend(index)
    assert sorted(order) == list(range(verified))
    np.testing.assert_array_equal(np.concatenate(vectors), np.concatenate(whole[0]))
    for part_pool, whole_pool in zip(part_pools, whole[1], strict=True):
        np.testing.assert_array_equal(part_pool, whole_pool)
    np.testing.assert_array_equal(np.concatenate(emitted)[np.argsort(order)], whole[2])
    np.testing.assert_array_equal(part_state, whole[3])


def _config(
    attention_dp: _Dynamic, schedule: _Dynamic, max_num_seqs: _Dynamic = 32
) -> _Dynamic:
    """Execute config logic."""
    return MegakernelEngineConfig(
        role="decode",
        tensor_parallel_size=32,
        decode_context_parallel_size=32 // attention_dp,
        block_size=512,
        kv_cache_dtype="fp8",
        max_num_seqs=max_num_seqs,
        max_model_len=1 << 17,
        max_num_batched_tokens=512,
        num_speculative_tokens=7,
        num_speculative_tokens_per_batch_size=schedule,
        attention_dp=attention_dp,
    )


def test_attention_dp2_speculates_up_to_32_requests() -> _Dynamic:
    """Verify test attention dp2 speculates up to 32 requests logic."""
    schedule = ((1, 2, 7), (3, 32, 3))
    dense = spec_tokens_per_batch(_config(2, schedule))
    assert dense[1:3] == (7, 7)
    assert set(dense[3:]) == {3}
    with pytest.raises(NotImplementedError):
        spec_tokens_per_batch(_config(1, schedule))
    with pytest.raises(NotImplementedError):
        spec_tokens_per_batch(_config(2, ((1, 16, 7),)))
