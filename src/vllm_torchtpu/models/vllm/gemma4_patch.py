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
"""Gemma 4: sequence parallelism for dense tensor-parallel decoder layers.

When `should_shard_tokens` admits a padded token bucket (`>=
MIN_SEQUENCE_PARALLEL_TOKENS`, TP-divisible, during backbone warmup), each
sublayer reduce-scatters its `o_proj` / `down_proj` partial sum along the token
axis, runs RMSNorms, residual adds, and FP8 activation quantization on
`num_tokens // tp_size` tokens per rank, and all-gathers 1-byte FP8 rows plus
f32 row scales into `qkv_proj` / `gate_up_proj`. Layer 0 slices its token shard
from the replicated embedding output and the final layer all-gathers the
residual back to `[num_tokens, H]`.

When `TPU_DENSE_SP_HIER_RS=bf16|fp8` is set on an 8-device tpu7x 2x2x1 TP
group, eligible buckets fuse each reduce-scatter and its residual/RMSNorm/FP8
epilogue into a single hierarchical Pallas kernel call.

`_attention_partial`, `_replicated_forward`, `_sharded_forward`, and
`_fused_sharded_forward` are adapted from upstream vLLM
`vllm/model_executor/models/gemma4.py` (`Gemma4Attention` and
`Gemma4DecoderLayer` at `v0.33.0`) with these changes:
- `o_proj` and `down_proj` run with `reduce_results=False` so `_replicated_forward`
  issues explicit `all_reduce`s and the sharded forwards issue `reduce_scatter`s.
- `qkv_proj` and `gate_up_proj` consume pre-quantized FP8 rows gathered across TP
  ranks (`k_eq_v` global-attention layers load `K` weights into both `K` and `V`
  slots of `QKVParallelLinear` upstream, so `[q_size, kv_size, kv_size]` holds).
- Layer 0 applies `narrow(0, 0, num_tokens)` or `slice_token_shard` for uniform
  Dynamo symbolic inputs, and `_fused_sharded_forward` carries each non-final
  layer's un-reduced `down_proj` tail via `Gemma4Model`'s `residual` slot.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
from vllm.distributed import (
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.gemma4 import (
    Gemma4Attention,
    Gemma4DecoderLayer,
    Gemma4MLP,
    Gemma4Model,
)

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter import dense_sequence_parallel
from vllm_torchtpu.layers.adapter.custom_ops import fused_rs_epilogue_op
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)

_orig_layer_init = Gemma4DecoderLayer.__init__
_CarriedSublayerTail = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class _DenseSpLayerState:
    is_last_layer: bool
    streaming_collectives: fused_rs_epilogue_op.FusedSublayerCollectives | None = None


def _validate_layer_norms(layer: Gemma4DecoderLayer) -> None:
    norms = (
        layer.input_layernorm,
        layer.post_attention_layernorm,
        layer.pre_feedforward_layernorm,
        layer.post_feedforward_layernorm,
    )
    for norm in norms:
        if not isinstance(norm, RMSNorm):
            raise NotImplementedError(
                "TPU_DENSE_SEQUENCE_PARALLEL requires vLLM RMSNorm decoder norms, "
                f"got {type(norm).__name__}"
            )
    epsilons = {norm.variance_epsilon for norm in norms}
    if len(epsilons) != 1:
        raise NotImplementedError(
            "TPU_DENSE_SEQUENCE_PARALLEL requires matching variance_epsilon across "
            f"decoder norms, got {sorted(epsilons)}"
        )


def _configure_layer(
    layer: Gemma4DecoderLayer, num_hidden_layers: int, prefix: str = ""
) -> None:
    if prefix.startswith("draft_model."):
        raise NotImplementedError(
            "TPU_DENSE_SEQUENCE_PARALLEL does not support MTP draft_model layers"
        )
    dense_sequence_parallel.validate_sequence_parallel_config(
        layer.mlp.down_proj.tp_size
    )
    if (
        layer.enable_moe_block
        or layer.self_attn.is_kv_shared_layer
        or layer.per_layer_input_gate is not None
    ):
        # TODO: support MoE, KV-shared (q_proj-only), and per-layer-embedding layers.
        raise NotImplementedError(
            "TPU_DENSE_SEQUENCE_PARALLEL supports dense Gemma 4 layers without KV "
            "sharing or per-layer embeddings"
        )
    if layer.self_attn.o_proj.bias is not None or layer.mlp.down_proj.bias is not None:
        # With reduce_results=False, TpuRowParallelLinear falls back to upstream
        # RowParallelLinear, which adds bias on rank 0 only and breaks XLA
        # collective channel matching across TP ranks.
        raise NotImplementedError(
            "TPU_DENSE_SEQUENCE_PARALLEL requires bias-free o_proj and down_proj"
        )
    dense_sequence_parallel.validate_sharded_linear_config(layer.self_attn.qkv_proj)
    dense_sequence_parallel.validate_sharded_linear_config(layer.mlp.gate_up_proj)
    _validate_layer_norms(layer)
    cast("Any", layer.self_attn.o_proj).reduce_results = False
    cast("Any", layer.mlp.down_proj).reduce_results = False
    payload = envs.TPU_DENSE_SP_HIER_RS
    streaming_collectives = None
    if payload != "off":
        hidden = layer.input_layernorm.weight.shape[0]
        if fused_rs_epilogue_op.supports_streaming_sublayer(
            fused_rs_epilogue_op.MIN_STREAMING_RS_TOKENS, hidden
        ):
            streaming_collectives = (
                fused_rs_epilogue_op.build_streaming_sublayer_collectives(
                    layer.input_layernorm.variance_epsilon,
                    fp8_wire=payload == "fp8",
                )
            )
        else:
            logger.warning_once(
                "TPU_DENSE_SP_HIER_RS=%s is set, but hidden=%d does not fit "
                "the streaming epilogue; keeping XLA reduce-scatter.",
                payload,
                hidden,
            )
    state = _DenseSpLayerState(
        is_last_layer=layer.layer_idx == num_hidden_layers - 1,
        streaming_collectives=streaming_collectives,
    )
    cast("Any", layer)._tpu_dense_sp = state


def _layer_state(layer: Gemma4DecoderLayer) -> _DenseSpLayerState:
    return cast("_DenseSpLayerState", cast("Any", layer)._tpu_dense_sp)


def _patched_layer_init(
    self: Gemma4DecoderLayer,
    config: Any,
    *args: Any,
    **kwargs: Any,
) -> None:
    _orig_layer_init(self, config, *args, **kwargs)
    prefix = kwargs.get("prefix", args[2] if len(args) > 2 else "")
    _configure_layer(self, config.num_hidden_layers, prefix=str(prefix))


def _unsupported_set_aux_hidden_state_layers(
    self: Gemma4Model, layers: Sequence[int]
) -> None:
    del self
    if layers:
        raise NotImplementedError(
            "TPU_DENSE_SEQUENCE_PARALLEL does not support aux_hidden_state_layers "
            "(Eagle3 / DFlash speculative decoding)"
        )


def _attention_partial(
    attn: Gemma4Attention, positions: torch.Tensor, qkv: torch.Tensor
) -> torch.Tensor:
    # Copied from vLLM Gemma4Attention.forward (v0.33.0, non-KV-shared branch),
    # taking pre-projected `qkv` and returning unreduced `o_proj` output.
    q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)

    q = q.unflatten(-1, (attn.num_heads, attn.head_dim))
    q = attn.q_norm(q)
    q = q.flatten(-2, -1)

    k = k.unflatten(-1, (attn.num_kv_heads, attn.head_dim))
    k = attn.k_norm(k)
    k = k.flatten(-2, -1)
    q, k = attn.rotary_emb(positions, q, k)

    v = v.unflatten(-1, (attn.num_kv_heads, attn.head_dim))
    v = attn.v_norm(v)
    v = v.flatten(-2, -1)
    attn_output = attn.attn(q, k, v)

    output, _ = attn.o_proj(attn_output)
    return output


def _mlp_partial(mlp: Gemma4MLP, gate_up: torch.Tensor) -> torch.Tensor:
    output, _ = mlp.down_proj(mlp.act_fn(gate_up))
    return output


def _replicated_forward(
    layer: Gemma4DecoderLayer, positions: torch.Tensor, hidden_states: torch.Tensor
) -> torch.Tensor:
    # Copied from vLLM Gemma4DecoderLayer.forward (v0.33.0, dense branch) with
    # explicit all-reduces because o_proj/down_proj have reduce_results=False.
    if layer.layer_idx == 0:
        # Read positions.shape[0] in the same place as _sharded_forward so all
        # token buckets lift identical Dynamo graph inputs (shape_variants).
        hidden_states = hidden_states.narrow(0, 0, positions.shape[0])
    residual = hidden_states
    attn = layer.self_attn
    qkv, _ = attn.qkv_proj(layer.input_layernorm(residual))
    hidden_states = tensor_model_parallel_all_reduce(
        _attention_partial(attn, positions, qkv)
    )
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = hidden_states + residual
    residual = hidden_states

    gate_up, _ = layer.mlp.gate_up_proj(layer.pre_feedforward_layernorm(hidden_states))
    hidden_states = tensor_model_parallel_all_reduce(_mlp_partial(layer.mlp, gate_up))
    hidden_states = layer.post_feedforward_layernorm(hidden_states)
    hidden_states = hidden_states + residual
    return hidden_states * layer.layer_scalar


def _sharded_forward(
    layer: Gemma4DecoderLayer, positions: torch.Tensor, hidden_states: torch.Tensor
) -> torch.Tensor:
    num_tokens = positions.shape[0]
    attn = layer.self_attn
    if layer.layer_idx == 0:
        hidden_states = dense_sequence_parallel.slice_token_shard(
            hidden_states, num_tokens, attn.o_proj.tp_rank, attn.o_proj.tp_size
        )
    residual = hidden_states
    qkv = dense_sequence_parallel.all_gather_linear(
        attn.qkv_proj, layer.input_layernorm(residual), num_tokens
    )
    hidden_states = tensor_model_parallel_reduce_scatter(
        _attention_partial(attn, positions, qkv), 0
    )
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = hidden_states + residual
    residual = hidden_states

    gate_up = dense_sequence_parallel.all_gather_linear(
        layer.mlp.gate_up_proj,
        layer.pre_feedforward_layernorm(hidden_states),
        num_tokens,
    )
    hidden_states = tensor_model_parallel_reduce_scatter(
        _mlp_partial(layer.mlp, gate_up), 0
    )
    hidden_states = layer.post_feedforward_layernorm(hidden_states)
    hidden_states = hidden_states + residual
    hidden_states = hidden_states * layer.layer_scalar
    if _layer_state(layer).is_last_layer:
        hidden_states = dense_sequence_parallel.all_gather_tokens(
            hidden_states, num_tokens
        )
    return hidden_states


def _fused_sharded_forward(
    layer: Gemma4DecoderLayer,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    carried: _CarriedSublayerTail | None,
    collectives: fused_rs_epilogue_op.FusedSublayerCollectives,
    is_last_layer: bool,
) -> tuple[torch.Tensor, _CarriedSublayerTail | None]:
    """Sequence-parallel forward fusing each sublayer's RS + residual epilogue.

    Because `down_proj`'s fused epilogue consumes the next layer's
    `input_layernorm.weight`, non-final layers return the un-reduced `down_proj`
    partial sum plus `carried = (residual, post_feedforward_layernorm.weight,
    layer_scalar)` via `Gemma4Model`'s `residual` carry slot.
    """
    num_tokens = positions.shape[0]
    attn = layer.self_attn
    if carried is None:
        residual = dense_sequence_parallel.slice_token_shard(
            hidden_states, num_tokens, attn.o_proj.tp_rank, attn.o_proj.tp_size
        )
        qkv = dense_sequence_parallel.all_gather_linear(
            attn.qkv_proj, layer.input_layernorm(residual), num_tokens
        )
    else:
        residual, post_norm_weight, scale = carried
        residual, x_q, x_scale = collectives.norm_add_scale_next_norm(
            hidden_states,
            residual,
            post_norm_weight,
            scale,
            layer.input_layernorm.weight,
        )
        qkv = dense_sequence_parallel.all_gather_prequantized_fp8_linear(
            attn.qkv_proj, x_q, x_scale, num_tokens
        )

    residual, x_q, x_scale = collectives.norm_add_next_norm(
        _attention_partial(attn, positions, qkv),
        residual,
        layer.post_attention_layernorm.weight,
        layer.pre_feedforward_layernorm.weight,
    )
    gate_up = dense_sequence_parallel.all_gather_prequantized_fp8_linear(
        layer.mlp.gate_up_proj, x_q, x_scale, num_tokens
    )
    mlp_partial = _mlp_partial(layer.mlp, gate_up)
    post_norm_weight = layer.post_feedforward_layernorm.weight
    if not is_last_layer:
        return mlp_partial, (residual, post_norm_weight, layer.layer_scalar)
    residual = collectives.norm_add_scale(
        mlp_partial, residual, post_norm_weight, layer.layer_scalar
    )
    return dense_sequence_parallel.all_gather_tokens(residual, num_tokens), None


def _dense_sequence_parallel_forward(
    self: Gemma4DecoderLayer,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: _CarriedSublayerTail | None,
    per_layer_input: torch.Tensor | None = None,
    **kwargs: object,
) -> tuple[torch.Tensor, _CarriedSublayerTail | None]:
    del per_layer_input, kwargs
    num_tokens = positions.shape[0]
    if not dense_sequence_parallel.should_shard_tokens(
        num_tokens, self.mlp.down_proj.tp_size
    ):
        return _replicated_forward(self, positions, hidden_states), None
    state = _layer_state(self)
    if (
        state.streaming_collectives is not None
        and fused_rs_epilogue_op.supports_streaming_sublayer(
            num_tokens, hidden_states.shape[-1]
        )
    ):
        return _fused_sharded_forward(
            self,
            positions,
            hidden_states,
            residual,
            state.streaming_collectives,
            state.is_last_layer,
        )
    return _sharded_forward(self, positions, hidden_states), None


def maybe_patch_gemma4_dense_sequence_parallel(
    model_config: ModelConfig | None,
) -> bool | None:
    """Patches `Gemma4DecoderLayer` once a Gemma 4 `ModelConfig` is available."""
    dense_sequence_parallel.validate_sequence_parallel_env()
    if not envs.TPU_DENSE_SEQUENCE_PARALLEL:
        return None
    if model_config is None:
        return False
    if model_config.hf_text_config.model_type != "gemma4_text":
        logger.warning_once(
            "TPU_DENSE_SEQUENCE_PARALLEL=1 has no effect on model_type=%s "
            "(only gemma4_text is supported).",
            model_config.hf_text_config.model_type,
        )
        return False
    if getattr(model_config, "lora_config", None) is not None:
        raise NotImplementedError(
            "TPU_DENSE_SEQUENCE_PARALLEL does not support LoRA adapters"
        )
    Gemma4DecoderLayer.__init__ = _patched_layer_init
    # Upstream annotates return as tuple[Tensor, Tensor] while returning None.
    Gemma4DecoderLayer.forward = cast("Any", _dense_sequence_parallel_forward)
    Gemma4Model._set_aux_hidden_state_layers = cast(
        "Any", _unsupported_set_aux_hidden_state_layers
    )
    logger.info("Applied TPU patch: Gemma 4 dense sequence parallelism.")
    return None
