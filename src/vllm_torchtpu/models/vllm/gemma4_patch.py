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

`_attention_partial`, `_replicated_forward`, and `_sharded_forward` are adapted
from upstream vLLM `vllm/model_executor/models/gemma4.py` (`Gemma4Attention` and
`Gemma4DecoderLayer` at `v0.33.0`) with these changes:
- `o_proj` and `down_proj` run with `reduce_results=False` so `_replicated_forward`
  issues explicit `all_reduce`s and `_sharded_forward` issues `reduce_scatter`s.
- `_sharded_forward` feeds `qkv_proj` and `gate_up_proj` via `all_gather_linear`
  (quantizing each rank's shard to FP8 before the gather). Upstream `k_eq_v`
  global-attention layers load `K` weights into both `K` and `V` slots of
  `QKVParallelLinear`, so the `[q_size, kv_size, kv_size]` split is unchanged.
- Layer 0 applies `narrow(0, 0, num_tokens)` (replicated) or `slice_token_shard`
  (sharded) so both paths lift identical Dynamo symbolic inputs, and the final
  layer gathers the token shard back to `[num_tokens, H]`.
"""

from __future__ import annotations

from collections.abc import Sequence
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
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)

_orig_layer_init = Gemma4DecoderLayer.__init__
_IS_LAST_LAYER_ATTR = "_tpu_dense_sp_is_last_layer"


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
    setattr(layer, _IS_LAST_LAYER_ATTR, layer.layer_idx == num_hidden_layers - 1)


def _is_last_layer(layer: Gemma4DecoderLayer) -> bool:
    return bool(getattr(layer, _IS_LAST_LAYER_ATTR))


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
    if _is_last_layer(layer):
        hidden_states = dense_sequence_parallel.all_gather_tokens(
            hidden_states, num_tokens
        )
    return hidden_states


def _dense_sequence_parallel_forward(
    self: Gemma4DecoderLayer,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    per_layer_input: torch.Tensor | None = None,
    **kwargs: object,
) -> tuple[torch.Tensor, None]:
    del residual, per_layer_input, kwargs
    num_tokens = positions.shape[0]
    forward = (
        _sharded_forward
        if dense_sequence_parallel.should_shard_tokens(
            num_tokens, self.mlp.down_proj.tp_size
        )
        else _replicated_forward
    )
    return forward(self, positions, hidden_states), None


def maybe_patch_gemma4_dense_sequence_parallel(
    model_config: ModelConfig | None,
) -> bool | None:
    """Patches `Gemma4DecoderLayer` once a Gemma 4 `ModelConfig` is available."""
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
