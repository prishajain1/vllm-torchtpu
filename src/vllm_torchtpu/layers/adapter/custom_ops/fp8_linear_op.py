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
"""Custom-op bridges for dynamic per-row FP8 quantization and pre-quantized matmul.

Splitting `xla_quantized_matmul` at the quantized activation lets each TP rank
quantize only its own token shard and all-gather 1-byte FP8 activations plus
per-row f32 scales instead of 2-byte bf16 activations.
"""

from collections.abc import Callable
from threading import Lock
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas
from vllm.model_executor.layers.linear import LinearBase

from vllm_torchtpu.kernels.quantized_matmul.util import (
    quantize_tensor,
    xla_prequantized_matmul,
)
from vllm_torchtpu.layers.adapter.linear_common import (
    KEEP_VLLM_LAYOUT_ATTR,
    WEIGHT_FLIPPED_ATTR,
)
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU

if TYPE_CHECKING:
    from vllm_torchtpu.layers.adapter.quantization.configs import TpuLinear


def _quantize_fp8_activations_jax(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """`[m, k]` bf16 -> (`[m, k]` FP8 bits as uint8, `[m, 1]` f32 row scales).

    Returned as uint8 because torch collectives support uint8 rather than FP8.
    """
    x_q, x_scale = quantize_tensor(x, jnp.float8_e4m3fn)
    return jax.lax.bitcast_convert_type(x_q, jnp.uint8), x_scale


def _prequantized_fp8_matmul_jax(
    x_q: jax.Array, x_scale: jax.Array, w_q: jax.Array, w_scale: jax.Array
) -> jax.Array:
    x_q_fp8 = jax.lax.bitcast_convert_type(x_q, jnp.float8_e4m3fn)
    return xla_prequantized_matmul(
        x_q_fp8, x_scale, w_q, w_scale, out_dtype=jnp.bfloat16
    )


_ops: tuple[Callable, Callable] | None = None
_ops_lock = Lock()


def _get_ops() -> tuple[Callable, Callable]:
    global _ops
    if _ops is not None:
        return _ops

    with _ops_lock:
        if _ops is not None:
            return _ops

        quantize_op = pallas.jax_op(
            "pallas::fp8_quantize_activations", _quantize_fp8_activations_jax
        )

        def _fake_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return (
                x.new_empty(x.shape, dtype=torch.uint8),
                x.new_empty((x.shape[0], 1), dtype=torch.float32),
            )

        quantize_op.register_fake(_fake_quantize)

        matmul_op = pallas.jax_op(
            "pallas::fp8_prequantized_matmul", _prequantized_fp8_matmul_jax
        )

        def _fake_matmul(
            x_q: torch.Tensor,
            x_scale: torch.Tensor,
            w_q: torch.Tensor,
            w_scale: torch.Tensor,
        ) -> torch.Tensor:
            del x_scale, w_scale
            return x_q.new_empty((x_q.shape[0], w_q.shape[-1]), dtype=torch.bfloat16)

        matmul_op.register_fake(_fake_matmul)
        _ops = quantize_op, matmul_op
        return _ops


def register_fp8_linear_ops() -> None:
    """Registers the FP8 activation-quantization and matmul ops before tracing."""
    _get_ops()


def quantize_fp8_activations(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantizes `[m, k]` bf16 activations to (`[m, k]` uint8, `[m, 1]` f32 scales)."""
    if x.ndim != 2 or x.dtype != torch.bfloat16:
        raise NotImplementedError(
            f"quantize_fp8_activations expects 2-D bfloat16 activations, got "
            f"{x.dtype} {tuple(x.shape)}"
        )
    return _get_ops()[0](x)


def validate_fp8_linear_config(layer: LinearBase) -> None:
    """Raises at layer init unless `layer` is configured for bias-free W8A8 FP8."""
    if not isinstance(layer, LinearBase):
        raise NotImplementedError(
            f"Expected a vLLM LinearBase layer, got {type(layer).__name__}"
        )
    quant_method = layer.quant_method
    if (
        not isinstance(quant_method, VllmFp8LinearMethodTPU)
        or quant_method.w8a16_max_rows
        or getattr(layer, KEEP_VLLM_LAYOUT_ATTR, False)
        or layer.bias is not None
    ):
        raise NotImplementedError(
            f"{layer.prefix}: pre-quantized FP8 matmul requires a bias-free FP8 "
            "linear with [n_in, n_out] weights, 1-D per-channel weight_scale, and "
            "TPU_FP8_LINEAR_W8A16_MAX_ROWS=0."
        )


def validate_prequantized_fp8_linear(layer: LinearBase) -> None:
    """Raises unless `layer` runs per-channel W8A8 FP8 matmul on `(k, n)` weights."""
    validate_fp8_linear_config(layer)
    tpu_layer = cast("TpuLinear", layer)
    if (
        not getattr(layer, WEIGHT_FLIPPED_ATTR, False)
        or not hasattr(layer, "weight_scale")
        or tpu_layer.weight.dtype != torch.float8_e4m3fn
        or tpu_layer.weight_scale.dim() != 1
    ):
        # TODO: support blockwise FP8 scales (gmm_v2) and W8A16 small-row buckets.
        raise NotImplementedError(
            f"{layer.prefix}: pre-quantized FP8 matmul requires a bias-free FP8 "
            "linear with [n_in, n_out] weights, 1-D per-channel weight_scale, and "
            "TPU_FP8_LINEAR_W8A16_MAX_ROWS=0."
        )


def apply_prequantized_fp8_linear(
    layer: LinearBase, x_q: torch.Tensor, x_scale: torch.Tensor
) -> torch.Tensor:
    """Applies `layer`'s FP8 matmul to activations from `quantize_fp8_activations`."""
    tpu_layer = cast("TpuLinear", layer)
    return _get_ops()[1](x_q, x_scale, tpu_layer.weight, tpu_layer.weight_scale)
