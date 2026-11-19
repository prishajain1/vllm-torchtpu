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
"""Model-independent sequence-parallel helpers for dense tensor-parallel layers.

Instead of all-reducing each row-parallel linear's partial sum and repeating
RMSNorms, residual adds, and FP8 quantization across all TP ranks, sequence
parallelism reduce-scatters the partial sums along the token axis, runs the
per-token ops on `num_tokens // tp_size` tokens per rank, and all-gathers the
1-byte FP8 activations and f32 row scales for the next column-parallel linear.
"""

from typing import Any

import torch
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.model_executor.layers.linear import LinearBase

from vllm_torchtpu import envs
from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.distributed.dcp import get_dcp_world_size
from vllm_torchtpu.distributed.mesh_utils import get_pipeline_stage
from vllm_torchtpu.distributed.pcp import get_pcp_world_size
from vllm_torchtpu.layers.adapter.custom_ops import fp8_linear_op

# Below 1024 padded tokens (measured on tpu7x at TP=8 with Gemma 4 31B),
# separate reduce-scatter and all-gather collectives cost more barrier latency
# than the per-token norms and quantization save.
MIN_SEQUENCE_PARALLEL_TOKENS = 1024


def validate_sequence_parallel_env() -> None:
    """Rejects sub-flags that require `TPU_DENSE_SEQUENCE_PARALLEL=1`."""
    if envs.TPU_DENSE_SP_HIER_RS != "off" and not envs.TPU_DENSE_SEQUENCE_PARALLEL:
        raise ValueError("TPU_DENSE_SP_HIER_RS requires TPU_DENSE_SEQUENCE_PARALLEL=1")


def validate_sequence_parallel_config(tp_size: int) -> None:
    """Validates parallelism topology and registers custom ops at model init."""
    if tp_size <= 1:
        raise ValueError("TPU_DENSE_SEQUENCE_PARALLEL requires TP > 1")
    if (
        get_pipeline_stage()[1] > 1
        or get_pcp_world_size() > 1
        or get_dcp_world_size() > 1
    ):
        raise NotImplementedError(
            "TPU_DENSE_SEQUENCE_PARALLEL does not support pipeline or context "
            "parallelism"
        )
    fp8_linear_op.register_fp8_linear_ops()


def validate_sharded_linear_config(linear: LinearBase) -> None:
    """Validates at layer init that `linear` supports sharded all-gather."""
    fp8_linear_op.validate_fp8_linear_config(linear)


def should_shard_tokens(
    num_tokens: Any,
    tp_size: int,
    min_tokens: int | None = None,
) -> bool:
    """Whether a padded `num_tokens` bucket runs on the sequence-parallel path."""
    threshold = MIN_SEQUENCE_PARALLEL_TOKENS if min_tokens is None else min_tokens
    # Only backbone shape_variants.warmup() traces each side of this branch into
    # its own executable; anything compiled outside warmup keeps all-reduce.
    return (
        (
            not torch.compiler.is_compiling()
            or shape_variants.aot_cache_tag() is not None
        )
        and num_tokens >= threshold
        and num_tokens % tp_size == 0
    )


def slice_token_shard(
    x: torch.Tensor, num_tokens: Any, tp_rank: int, tp_size: int
) -> torch.Tensor:
    """Slices `tp_rank`'s contiguous token shard from replicated `[num_tokens, ...]`."""
    shard_tokens = num_tokens // tp_size
    return x.narrow(0, tp_rank * shard_tokens, shard_tokens)


def all_gather_tokens(x_shard: torch.Tensor, num_tokens: Any) -> torch.Tensor:
    """All-gathers token shards across TP ranks, preserving symbolic `num_tokens`."""
    return tensor_model_parallel_all_gather(x_shard, 0).narrow(0, 0, num_tokens)


def all_gather_prequantized_fp8_linear(
    linear: LinearBase,
    x_q_shard: torch.Tensor,
    x_scale_shard: torch.Tensor,
    num_tokens: Any,
) -> torch.Tensor:
    """All-gathers pre-quantized FP8 shards and applies `linear` over all tokens."""
    fp8_linear_op.validate_prequantized_fp8_linear(linear)
    return fp8_linear_op.apply_prequantized_fp8_linear(
        linear,
        all_gather_tokens(x_q_shard, num_tokens),
        all_gather_tokens(x_scale_shard, num_tokens),
    )


def all_gather_linear(
    linear: LinearBase, x_shard: torch.Tensor, num_tokens: Any
) -> torch.Tensor:
    """Quantizes `x_shard`, all-gathers across TP, and applies `linear`."""
    x_q_shard, x_scale_shard = fp8_linear_op.quantize_fp8_activations(x_shard)
    return all_gather_prequantized_fp8_linear(
        linear, x_q_shard, x_scale_shard, num_tokens
    )
