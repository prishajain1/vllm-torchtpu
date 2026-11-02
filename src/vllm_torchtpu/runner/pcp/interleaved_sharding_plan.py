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
"""Interleaved PCP sharding plan (`TPU_PCP_SHARDING_PLAN=interleaved`).

Deals each step's request-major tokens round-robin across PCP ranks in chunks
of `cp_kv_cache_interleave_size` tokens for streamed PCP attention and GDN.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_torchtpu.distributed.pcp import all_gather_equal_tokens
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.kernel import (
    PCP_STREAMING_RPA_LOCAL_COMPILE_TOKEN_MULTIPLE,
)
from vllm_torchtpu.layers.core.pcp_sequence_layout import (
    PCP_STREAMING_SEQUENCE_LAYOUT_DESCRIPTOR,
    PcpSequenceLayoutEligibility,
    prepare_pcp_sequence_layout,
)
from vllm_torchtpu.layers.core.sequence_layout import SequenceLayoutDescriptor
from vllm_torchtpu.runner.pcp.sharding_plan import PcpShardingPlan

if TYPE_CHECKING:
    from vllm_torchtpu.layers.core.pcp_metadata import PcpMetadata


def validate_interleaved_pcp_params(
    pcp_size: int,
    dcp_size: int,
    interleave_size: int,
) -> None:
    """Validate static topology and interleave parameters for interleaved PCP."""
    if pcp_size <= 1:
        raise NotImplementedError(
            "Interleaved PCP requires prefill_context_parallel_size > 1, "
            f"got {pcp_size}."
        )
    if dcp_size != 1:
        raise NotImplementedError(
            "Interleaved PCP does not support decode_context_parallel_size > 1, "
            f"got {dcp_size}."
        )
    if interleave_size <= 0:
        raise ValueError(
            f"cp_kv_cache_interleave_size must be positive, got {interleave_size}."
        )
    if PCP_STREAMING_RPA_LOCAL_COMPILE_TOKEN_MULTIPLE % interleave_size != 0:
        raise ValueError(
            f"cp_kv_cache_interleave_size ({interleave_size}) must divide "
            f"{PCP_STREAMING_RPA_LOCAL_COMPILE_TOKEN_MULTIPLE}."
        )


def interleaved_max_local_tokens(
    max_num_batched_tokens: int,
    pcp_size: int,
    interleave_size: int,
) -> int:
    """Upper bound on a rank's padded local token count for any step."""
    budget = max(1, int(max_num_batched_tokens))
    cycle = int(pcp_size) * int(interleave_size)
    rank0_max = (budget // cycle) * int(interleave_size) + min(
        budget % cycle, int(interleave_size)
    )
    multiple = PCP_STREAMING_RPA_LOCAL_COMPILE_TOKEN_MULTIPLE
    return cdiv(max(1, rank0_max), multiple) * multiple


def attention_kv_page_size(
    kv_cache_config: Any,
    kernel_block_size: int | None = None,
) -> int | None:
    """Resolve the attention kernel page size for interleaved PCP validation."""
    if kernel_block_size is not None:
        return int(kernel_block_size)
    sizes: set[int] = set()
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            spec = next(iter(spec.kv_cache_specs.values()))
        if isinstance(spec, MambaSpec):
            continue
        if isinstance(spec, AttentionSpec):
            sizes.add(int(spec.block_size))
    if not sizes:
        return None
    if len(sizes) > 1:
        raise ValueError(
            "Interleaved PCP requires one attention KV cache block size, "
            f"got {sorted(sizes)}."
        )
    return next(iter(sizes))


class InterleavedPcpShardingPlan(PcpShardingPlan):
    """Interleaved round-robin PCP token sharding plan (`dcp_size == 1`)."""

    def __init__(
        self,
        *,
        pcp_size: int,
        pcp_rank: int,
        interleave_size: int,
        max_num_batched_tokens: int,
        dcp_size: int = 1,
        page_size: int | None = None,
    ) -> None:
        pcp_size = int(pcp_size)
        pcp_rank = int(pcp_rank)
        dcp_size = int(dcp_size)
        interleave_size = int(interleave_size)
        validate_interleaved_pcp_params(pcp_size, dcp_size, interleave_size)
        if not 0 <= pcp_rank < pcp_size:
            raise ValueError(f"pcp_rank must be in [0, {pcp_size}), got {pcp_rank}.")
        super().__init__(pcp_size=pcp_size, pcp_rank=pcp_rank, dcp_size=dcp_size)
        self.interleave_size = interleave_size
        self._eligibility = PcpSequenceLayoutEligibility(
            pcp_size=pcp_size,
            interleave_size=interleave_size,
            dcp_size=dcp_size,
        )
        self._max_local_num_tokens = interleaved_max_local_tokens(
            max_num_batched_tokens, pcp_size, interleave_size
        )
        if page_size is not None:
            self.set_page_size(page_size)

    @property
    def local_reqs_per_req(self) -> int:
        return 1

    @property
    def max_local_num_tokens(self) -> int:
        return self._max_local_num_tokens

    @property
    def sequence_layout_descriptor(self) -> SequenceLayoutDescriptor:
        return PCP_STREAMING_SEQUENCE_LAYOUT_DESCRIPTOR

    @property
    def requires_backend_preinit(self) -> bool:
        return True

    @property
    def local_compile_token_multiple(self) -> int:
        return PCP_STREAMING_RPA_LOCAL_COMPILE_TOKEN_MULTIPLE

    def set_page_size(self, page_size: int) -> None:
        page_size = int(page_size)
        if page_size <= 0 or page_size % self.interleave_size != 0:
            raise ValueError(
                f"Attention KV page_size ({page_size}) must be a positive "
                f"multiple of cp_kv_cache_interleave_size ({self.interleave_size})."
            )
        super().set_page_size(page_size)

    def configure_kv_cache(
        self, kv_cache_config: Any, kernel_block_size: int | None = None
    ) -> None:
        page_size = attention_kv_page_size(kv_cache_config, kernel_block_size)
        if page_size is not None:
            self.set_page_size(page_size)

    def prepare_real(
        self,
        *,
        runner: Any,
        scheduler_output: Any,
        start_index: int,
        num_reqs: int,
        num_scheduled_tokens_per_req: Any,
        total_num_scheduled_tokens: int,
        use_max_model_len: bool,
        target_num_reqs: int,
        padded_num_reqs: int,
    ) -> None:
        self._reset_step_state()
        num_reqs = int(num_reqs)
        num_tokens = int(total_num_scheduled_tokens)
        if num_reqs <= 0 or num_tokens <= 0:
            return
        q_lens = np.asarray(num_scheduled_tokens_per_req[:num_reqs], dtype=np.int32)
        if int(q_lens.sum()) != num_tokens:
            raise ValueError(
                "total_num_scheduled_tokens must match "
                "sum(num_scheduled_tokens_per_req): "
                f"got {num_tokens} vs {int(q_lens.sum())}."
            )

        scheduled_encoder_inputs = getattr(
            scheduler_output, "scheduled_encoder_inputs", None
        )
        if scheduled_encoder_inputs and hasattr(runner.input_batch, "req_ids"):
            req_ids = runner.input_batch.req_ids[start_index : start_index + num_reqs]
            if any(
                req_id is not None and bool(scheduled_encoder_inputs.get(req_id))
                for req_id in req_ids
            ):
                raise NotImplementedError(
                    "Interleaved PCP manager does not support multimodal "
                    "encoder inputs."
                )

        decision = self._eligibility.evaluate_runner_chunk(
            input_batch=runner.input_batch,
            scheduler_output=scheduler_output,
            start_index=start_index,
            num_reqs=num_reqs,
            num_scheduled_tokens_per_req=q_lens,
            num_tokens_paddings=runner.num_tokens_paddings,
            max_num_tokens=getattr(runner, "max_num_tokens", None),
            dp_target_bucket=getattr(runner, "_dp_target_bucket", None),
        )
        prepared = prepare_pcp_sequence_layout(
            runner,
            scheduler_output,
            start_index,
            num_reqs,
            q_lens,
            num_tokens,
            use_max_model_len,
            target_num_reqs,
            padded_num_reqs,
            get_native_pcp_rank=lambda: self.pcp_rank,
            get_native_pcp_world_size=lambda: self.pcp_size,
            decision=decision,
        )

        local_start = prepared.local_token_slice.start
        local_end = prepared.local_token_slice.stop
        packed = prepared.request_major_to_packed_token_indices
        local_mask = (packed >= local_start) & (packed < local_end)
        request_major_to_local = np.full(num_tokens, -1, dtype=np.int32)
        request_major_to_local[local_mask] = (packed[local_mask] - local_start).astype(
            np.int32
        )

        self.token_slice = prepared.local_token_slice
        self.global_num_tokens = num_tokens
        self.global_padded_num_tokens = prepared.padded_total_num_scheduled_tokens
        self.local_num_tokens = prepared.local_total_num_scheduled_tokens
        self.local_padded_num_tokens = prepared.local_padded_total_num_scheduled_tokens
        self.local_num_reqs = num_reqs
        self.request_major_to_local = request_major_to_local
        self.logits_indices_cpu = prepared.logits_indices_cpu
        self.logits_local_indices_cpu = prepared.logits_local_indices_cpu
        self.logits_owner_mask_cpu = prepared.logits_owner_mask_cpu
        self.logits_all_local = False

    def prepare_dummy(
        self, *, num_tokens: int, num_reqs: int, max_num_tokens: int
    ) -> None:
        del max_num_tokens
        self._reset_step_state()
        num_tokens = int(num_tokens)
        self.token_slice = slice(0, num_tokens)
        self.global_num_tokens = num_tokens * self.pcp_size
        self.global_padded_num_tokens = num_tokens * self.pcp_size
        self.local_num_tokens = num_tokens
        self.local_padded_num_tokens = num_tokens
        self.local_num_reqs = int(num_reqs)

    def build_metadata(
        self,
        *,
        runner: Any,
        block_table_obj: Any,
        target_num_reqs: int,
        target_num_blocks: int,
        split_block_tables: torch.Tensor | None = None,
    ) -> PcpMetadata | None:
        del runner, block_table_obj, target_num_reqs
        del target_num_blocks, split_block_tables
        return None

    def finalize_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return all_gather_equal_tokens(hidden_states, dim=0)
