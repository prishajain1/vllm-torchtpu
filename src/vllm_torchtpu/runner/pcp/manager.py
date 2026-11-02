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
"""PCP manager (`USE_TPU_PCP_MANAGER=1`), after upstream vLLM's MRV2
`PCPManager`.

`PcpManager` delegates to a stateful `PcpShardingPlan` (`sharding_plan.py`),
selected via `TPU_PCP_SHARDING_PLAN` (`headtail` or `interleaved`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.pcp import get_pcp_group
from vllm_torchtpu.layers.core.sequence_layout import SequenceLayoutDescriptor
from vllm_torchtpu.runner.pcp.head_tail_sharding_plan import HeadTailPcpShardingPlan
from vllm_torchtpu.runner.pcp.interleaved_sharding_plan import (
    InterleavedPcpShardingPlan,
    validate_interleaved_pcp_params,
)
from vllm_torchtpu.runner.pcp.sharding_plan import (
    PcpShardingPlan,
    grow_host_token_buffers,
)

if TYPE_CHECKING:
    from vllm_torchtpu.layers.core.pcp_metadata import PcpMetadata


def validate_pcp_manager_config(vllm_config: Any) -> str:
    plan_name = envs.TPU_PCP_SHARDING_PLAN.lower()
    if plan_name not in ("headtail", "interleaved"):
        raise ValueError(
            f"Unsupported TPU_PCP_SHARDING_PLAN: {envs.TPU_PCP_SHARDING_PLAN!r}."
        )
    parallel_config = vllm_config.parallel_config
    model_config = vllm_config.model_config
    pcp_size = int(parallel_config.prefill_context_parallel_size)
    dcp_size = int(parallel_config.decode_context_parallel_size)

    if plan_name == "headtail":
        tp_size = int(parallel_config.tensor_parallel_size)
        hf_config = getattr(model_config, "hf_text_config", None)
        sparse_mla = bool(getattr(model_config, "use_mla", False)) and hasattr(
            hf_config, "index_topk"
        )
        if not sparse_mla:
            raise NotImplementedError("The PCP manager only supports sparse MLA.")
        if pcp_size <= 1 or dcp_size <= 1:
            raise NotImplementedError(
                "USE_TPU_PCP_MANAGER needs prefill and decode context parallel "
                f"sizes > 1, got pcp={pcp_size}, dcp={dcp_size}."
            )
        if dcp_size != tp_size * pcp_size:
            raise NotImplementedError(
                "PCP x DCP requires decode_context_parallel_size == "
                "tensor_parallel_size * prefill_context_parallel_size, got "
                f"dcp={dcp_size}, tp={tp_size}, pcp={pcp_size}."
            )
        if getattr(model_config, "is_multimodal_model", False):
            raise NotImplementedError("PCP x DCP does not support multimodal inputs.")
        max_model_len = int(model_config.max_model_len)
        if max_model_len > envs.TPU_PCP_KV_WORKSPACE_TOKENS:
            raise ValueError(
                f"max_model_len ({max_model_len}) exceeds "
                f"TPU_PCP_KV_WORKSPACE_TOKENS ({envs.TPU_PCP_KV_WORKSPACE_TOKENS}); "
                "raise TPU_PCP_KV_WORKSPACE_TOKENS or lower --max-model-len."
            )
    else:
        validate_interleaved_pcp_params(
            pcp_size,
            dcp_size,
            int(parallel_config.cp_kv_cache_interleave_size),
        )
        if model_config is not None and getattr(
            model_config, "is_multimodal_model", False
        ):
            mm_config = getattr(model_config, "multimodal_config", None)
            mm_limits = getattr(mm_config, "limit_per_prompt", None)
            lang_only = bool(getattr(model_config, "language_model_only", False)) or (
                isinstance(mm_limits, dict)
                and bool(mm_limits)
                and all(v == 0 for v in mm_limits.values())
            )
            if not lang_only:
                raise NotImplementedError(
                    "Interleaved PCP manager does not support multimodal inputs; "
                    "pass --language-model-only for text-only execution."
                )

    if parallel_config.is_moe_model and not parallel_config.enable_expert_parallel:
        raise NotImplementedError(
            "PCP on an MoE model requires --enable-expert-parallel."
        )
    if int(getattr(parallel_config, "pipeline_parallel_size", 1)) != 1:
        raise NotImplementedError("PCP manager does not support pipeline parallelism.")
    if int(getattr(parallel_config, "data_parallel_size", 1)) != 1:
        raise NotImplementedError("PCP manager does not support data parallelism.")
    if vllm_config.speculative_config is not None:
        raise NotImplementedError(
            "PCP manager does not support speculative decoding yet."
        )
    if getattr(vllm_config, "lora_config", None) is not None:
        raise NotImplementedError("PCP manager does not support LoRA.")
    return plan_name


class PcpManager:
    """PCP manager owned by `TPUModelRunner`. The runner reads the prepared
    step's state from `sharding_plan`."""

    uses_selected_logits_hidden_states: bool = True

    def __init__(self, sharding_plan: PcpShardingPlan) -> None:
        self.sharding_plan = sharding_plan

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> PcpManager:
        plan_name = validate_pcp_manager_config(vllm_config)
        parallel_config = vllm_config.parallel_config
        group = get_pcp_group()
        pcp_rank = 0 if group is None else int(group.rank_in_group)
        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        if plan_name == "headtail":
            return cls(
                HeadTailPcpShardingPlan(
                    pcp_size=parallel_config.prefill_context_parallel_size,
                    pcp_rank=pcp_rank,
                    dcp_size=parallel_config.decode_context_parallel_size,
                    page_size=vllm_config.cache_config.block_size,
                    workspace_tokens=envs.TPU_PCP_KV_WORKSPACE_TOKENS,
                    max_num_batched_tokens=max_num_batched_tokens,
                    max_num_seqs=vllm_config.scheduler_config.max_num_seqs,
                )
            )
        return cls(
            InterleavedPcpShardingPlan(
                pcp_size=parallel_config.prefill_context_parallel_size,
                pcp_rank=pcp_rank,
                dcp_size=parallel_config.decode_context_parallel_size,
                interleave_size=parallel_config.cp_kv_cache_interleave_size,
                max_num_batched_tokens=max_num_batched_tokens,
            )
        )

    @property
    def local_reqs_per_req(self) -> int:
        return self.sharding_plan.local_reqs_per_req

    @property
    def max_local_num_tokens(self) -> int:
        return self.sharding_plan.max_local_num_tokens

    @property
    def sequence_layout_descriptor(self) -> SequenceLayoutDescriptor:
        return self.sharding_plan.sequence_layout_descriptor

    @property
    def requires_backend_preinit(self) -> bool:
        return self.sharding_plan.requires_backend_preinit

    @property
    def backend_preinit_world_size(self) -> int:
        return self.sharding_plan.pcp_size

    @property
    def local_compile_token_multiple(self) -> int:
        return self.sharding_plan.local_compile_token_multiple

    def num_reqs_in_chunk(
        self, is_prefilling: Any, seq_lens: Any, capacity: int
    ) -> int:
        return self.sharding_plan.num_reqs_in_chunk(is_prefilling, seq_lens, capacity)

    def set_page_size(self, page_size: int) -> None:
        self.sharding_plan.set_page_size(page_size)

    def configure_kv_cache(
        self, kv_cache_config: Any, kernel_block_size: int | None = None
    ) -> None:
        self.sharding_plan.configure_kv_cache(kv_cache_config, kernel_block_size)

    def reserve_host_token_capacity(
        self, runner: Any, required_num_tokens: int
    ) -> None:
        # Replication and per-rank padding can grow the packed layout.
        grow_host_token_buffers(
            runner, int(required_num_tokens) * self.sharding_plan.pcp_size
        )

    def prepare_real(self, **kwargs: Any) -> None:
        self.sharding_plan.prepare_real(**kwargs)

    def prepare_dummy(
        self, *, num_tokens: int, num_reqs: int, max_num_tokens: int
    ) -> None:
        self.sharding_plan.prepare_dummy(
            num_tokens=num_tokens, num_reqs=num_reqs, max_num_tokens=max_num_tokens
        )

    def build_metadata(self, **kwargs: Any) -> PcpMetadata | None:
        return self.sharding_plan.build_metadata(**kwargs)

    def maybe_select_logits_hidden_states(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor | None:
        return self.sharding_plan.maybe_select_logits_hidden_states(hidden_states)

    def finalize_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.sharding_plan.finalize_hidden_states(hidden_states)
