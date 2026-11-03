# SPDX-License-Identifier: Apache-2.0
"""Token-sharded decode indexer (VLLM_TPU_QSA_DECODE_TOKEN_SHARD).

On the replicated backbone each DCP pair indexes one contiguous 1/size of a
decode / verify step's rows (``token_shard_step``); a shard holds at most one
row per request slot, so it takes the shared-stream decode indexer. The
concatenated shards must select exactly what the whole step selects, for the
plain and the compressed-DCP (owned groups, local lists) indexers. Real
StreamIndex kernel in TPU interpret mode on CPU, a JAX top-k in place of the
SparseCore selector.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_qwen4_exp_indexer_shard import reference_kernel  # noqa: F401 (fixture)
from test_qwen4_exp_shared_stream_indexer import (  # noqa: F401 (fixture)
    BLOCK_TOPK,
    RATIO,
    inputs,
    interpret,
)

from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa

# (q_lens per request slot, context per slot, padded rows, shards). Verify
# steps (4 rows per request) with idle slots and an empty bucket tail; a
# shard boundary inside a request; one-token decodes.
CASES = {
    "verify_full": ([4] * 4 + [0] * 4, [3001, 2048, 1203, 517] + [0] * 4, 16, 4),
    "verify_idle_tail": ([4, 4, 0, 4, 0, 0, 0, 0], [999, 3500, 0, 64] + [0] * 4, 16, 4),
    "boundary_in_request": ([3, 4, 4, 1, 0, 0, 0, 0], [2999, 1022, 4000, 7] + [0] * 4, 16, 2),
    "decodes": ([1] * 8, [4000, 3, 64, 65, 999, 2048, 3001, 517], 8, 4),
}


def select(args, dcp=None):
    out = pooled_qsa.select_blocks(**args, ratio=RATIO, block_topk=BLOCK_TOPK, dcp=dcp)
    return tuple(np.asarray(x) for x in out) if dcp else (np.asarray(out),)


def token_sharded(args, size, dcp=None):
    rows = len(args["query"]) // size
    shards = []
    for s in range(size):
        base, starts, context = pooled_qsa.token_shard_step(
            args["starts"], args["context"], jnp.asarray([s], jnp.int32), rows
        )
        part = dict(args, starts=starts, context=context)
        for name in ("query", "lengths", "req"):
            part[name] = jax.lax.dynamic_slice_in_dim(args[name], base, rows)
        # As in the op: the shard is at most one row per slot, so the
        # max_query_len=1 routing keeps it on the decode indexer.
        assert rows <= len(args["context"])
        shards.append(select(part, dcp))
    return tuple(np.concatenate(parts) for parts in zip(*shards, strict=True))


@pytest.mark.parametrize("dcp_rank", [None, 0, 1])
@pytest.mark.parametrize("case", sorted(CASES))
def test_decode_token_shards_match_whole_step(
    monkeypatch, interpret, reference_kernel, case, dcp_rank
):
    monkeypatch.setenv("VLLM_TPU_QSA_DECODE_QUERY_BLOCK", "4")
    monkeypatch.setenv("VLLM_TPU_QSA_DECODE_KV_BLOCK", "256")
    monkeypatch.setattr(pooled_qsa, "decode_query_chunk_size", lambda pages: 1 << 20)
    q_lens, context, tokens, size = CASES[case]
    # Integer data: q @ k is exact in any order, so the CPU GEMM's blocking by
    # row count cannot differ between the shard and the whole step.
    args = inputs(len(case), q_lens, context, tokens, "integer")
    dcp = None if dcp_rank is None else (jnp.asarray([dcp_rank], jnp.int32), 2)
    expected = select(args, dcp)
    actual = token_sharded(args, size, dcp)
    for a, e in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, e)
