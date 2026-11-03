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
"""Unit tests for `InterleavedPcpShardingPlan` and `PcpManager` validation."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

from tests.runner.runner_input_test_utils import (
    _fake_tpu_events,
    _make_runner,
    _scheduler_output,
)
from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter.cp_mla_attention import dcp_spans_pcp
from vllm_torchtpu.layers.core.pcp_sequence_layout import (
    PCP_STREAMING_SEQUENCE_LAYOUT_DESCRIPTOR,
)
from vllm_torchtpu.platforms.pcp_validation import PcpStaticSupportValidator
from vllm_torchtpu.runner.pcp import (
    HeadTailPcpShardingPlan,
    InterleavedPcpShardingPlan,
    PcpManager,
    attention_kv_page_size,
    interleaved_max_local_tokens,
)
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def _vllm_config(*, pcp_size=4, dcp_size=1, tp_size=2, interleave_size=4, **overrides):
    model_kw = dict(
        use_mla=False,
        hf_text_config=SimpleNamespace(),
        max_model_len=32768,
        is_multimodal_model=False,
        language_model_only=False,
        multimodal_config=None,
    )
    parallel_kw = dict(
        prefill_context_parallel_size=pcp_size,
        decode_context_parallel_size=dcp_size,
        tensor_parallel_size=tp_size,
        cp_kv_cache_interleave_size=interleave_size,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        is_moe_model=True,
        enable_expert_parallel=True,
    )
    for key, val in list(overrides.items()):
        if key in model_kw:
            model_kw[key] = overrides.pop(key)
        elif key in parallel_kw:
            parallel_kw[key] = overrides.pop(key)
    return SimpleNamespace(
        model_config=SimpleNamespace(**model_kw),
        parallel_config=SimpleNamespace(**parallel_kw),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=16384, max_num_seqs=16, async_scheduling=False
        ),
        cache_config=SimpleNamespace(block_size=128),
        speculative_config=overrides.pop("speculative_config", None),
        lora_config=overrides.pop("lora_config", None),
        kv_transfer_config=None,
    )


def _attach_interleaved_manager(runner, *, pcp_size, pcp_rank, interleave_size):
    runner.parallel_config.prefill_context_parallel_size = pcp_size
    runner.parallel_config.cp_kv_cache_interleave_size = interleave_size
    runner.sequence_layout_planner = None
    runner.pcp_manager = PcpManager(
        InterleavedPcpShardingPlan(
            pcp_size=pcp_size,
            pcp_rank=pcp_rank,
            interleave_size=interleave_size,
            max_num_batched_tokens=max(runner.num_tokens_paddings) * pcp_size,
            page_size=runner.block_size,
        )
    )


@pytest.mark.parametrize(
    ("plan_env", "cfg_kw", "kv_layout", "expected_err", "match"),
    [
        ("zigzag", {}, "NHD", ValueError, "Unsupported TPU_PCP_SHARDING_PLAN"),
        ("interleaved", {"pcp_size": 1}, "NHD", NotImplementedError, "> 1"),
        ("interleaved", {"dcp_size": 2}, "NHD", NotImplementedError, "decode_context"),
        ("interleaved", {"interleave_size": 3}, "NHD", ValueError, "must divide 512"),
        ("interleaved", {"interleave_size": 4}, "HND", ValueError, "multiple of 128"),
        (
            "interleaved",
            {"enable_expert_parallel": False},
            "NHD",
            NotImplementedError,
            "expert-parallel",
        ),
        (
            "interleaved",
            {"is_multimodal_model": True},
            "NHD",
            NotImplementedError,
            "language-model-only",
        ),
        (
            "interleaved",
            {
                "speculative_config": SimpleNamespace(
                    method="mtp", num_speculative_tokens=1
                )
            },
            "NHD",
            NotImplementedError,
            "speculative",
        ),
    ],
)
def test_pcp_manager_static_validation_rejects_invalid_configs(
    monkeypatch, plan_env, cfg_kw, kv_layout, expected_err, match
):
    monkeypatch.setattr(envs, "USE_TPU_PCP_MANAGER", True)
    monkeypatch.setattr(envs, "TPU_PCP_SHARDING_PLAN", plan_env)
    with pytest.raises(expected_err, match=match):
        PcpStaticSupportValidator.validate_platform_config(
            _vllm_config(**cfg_kw), kv_cache_layout=kv_layout
        )


def test_pcp_manager_selection_page_size_and_dcp_flag(monkeypatch):
    monkeypatch.setattr(envs, "USE_TPU_PCP_MANAGER", True)
    monkeypatch.setattr(envs, "TPU_PCP_SHARDING_PLAN", "interleaved")
    assert dcp_spans_pcp() is False
    cfg = _vllm_config(is_multimodal_model=True, language_model_only=True)
    PcpStaticSupportValidator.validate_platform_config(cfg, kv_cache_layout="NHD")
    mgr = PcpManager.from_vllm_config(cfg)
    assert isinstance(mgr.sharding_plan, InterleavedPcpShardingPlan)
    assert mgr.sequence_layout_descriptor == PCP_STREAMING_SEQUENCE_LAYOUT_DESCRIPTOR
    assert mgr.requires_backend_preinit is True
    assert interleaved_max_local_tokens(16384, pcp_size=4, interleave_size=4) == 4096

    attn = FullAttentionSpec(
        block_size=128, num_kv_heads=2, head_size=128, dtype=torch.bfloat16
    )
    mamba = MambaSpec(shapes=((4, 8),), dtypes=(torch.bfloat16,), block_size=512)
    kv_cfg = SimpleNamespace(
        kv_cache_groups=[
            KVCacheGroupSpec(["a"], UniformTypeKVCacheSpecs(128, {"a": attn})),
            KVCacheGroupSpec(["m"], mamba),
        ]
    )
    assert attention_kv_page_size(kv_cfg) == 128
    mgr.configure_kv_cache(kv_cfg)
    assert mgr.sharding_plan.page_size == 128
    with pytest.raises(ValueError, match="multiple of cp_kv_cache_interleave_size"):
        mgr.set_page_size(6)

    monkeypatch.setattr(envs, "TPU_PCP_SHARDING_PLAN", "headtail")
    assert dcp_spans_pcp() is True
    ht_cfg = _vllm_config(
        pcp_size=2,
        dcp_size=8,
        tp_size=4,
        use_mla=True,
        hf_text_config=SimpleNamespace(index_topk=2048),
    )
    assert isinstance(
        PcpManager.from_vllm_config(ht_cfg).sharding_plan, HeadTailPcpShardingPlan
    )


@pytest.mark.parametrize(
    ("computed", "prompts", "scheduled", "interleave", "uses_mrope"),
    [
        ([0], [37], [37], 4, False),
        ([12], [32], [20], 4, False),
        ([40], [32], [1], 4, False),
        ([20, 31, 44, 15, 28, 39], [16] * 6, [1] * 6, 4, False),
        ([40, 8], [32, 48], [1, 23], 4, True),
        ([0, 12], [260, 300], [260, 270], 128, False),
    ],
)
def test_interleaved_plan_matches_legacy_sequence_layout(
    computed, prompts, scheduled, interleave, uses_mrope
):
    pcp_size = 4
    block_size = max(16, interleave)
    paddings = [512, 1024]
    for rank in range(pcp_size):
        ref = _make_runner(
            num_computed_tokens=computed,
            prompt_tokens=prompts,
            scheduled_tokens=scheduled,
            token_paddings=paddings,
            uses_mrope=uses_mrope,
            pcp_size=pcp_size,
            block_size=block_size,
        )
        ref.parallel_config.cp_kv_cache_interleave_size = interleave
        ref.sequence_layout_planner.eligibility.interleave_size = interleave

        mgr = _make_runner(
            num_computed_tokens=computed,
            prompt_tokens=prompts,
            scheduled_tokens=scheduled,
            token_paddings=paddings,
            uses_mrope=uses_mrope,
            pcp_size=pcp_size,
            block_size=block_size,
        )
        _attach_interleaved_manager(
            mgr, pcp_size=pcp_size, pcp_rank=rank, interleave_size=interleave
        )

        with (
            patch(
                "vllm_torchtpu.layers.core.pcp_sequence_layout._get_native_pcp_rank",
                return_value=rank,
            ),
            patch(
                "vllm_torchtpu.layers.core.pcp_sequence_layout._get_native_pcp_world_size",
                return_value=pcp_size,
            ),
        ):
            ref_attn, ref_logits, *_ = TPUModelRunner._prepare_inputs(
                ref, _scheduler_output(scheduled), 0, len(scheduled)
            )
        mgr_attn, mgr_logits, *_ = TPUModelRunner._prepare_inputs(
            mgr, _scheduler_output(scheduled), 0, len(scheduled)
        )

        assert torch.equal(mgr.input_ids, ref.input_ids)
        assert torch.equal(mgr.position_ids, ref.position_ids)
        assert torch.equal(mgr_logits, ref_logits)
        md_mgr, md_ref = mgr_attn["layer.0"], ref_attn["layer.0"]
        assert torch.equal(md_mgr.seq_lens, md_ref.seq_lens)
        assert torch.equal(md_mgr.query_start_loc, md_ref.query_start_loc)
        assert (
            md_mgr.sequence_layout_kind,
            md_mgr.sequence_layout_protocol,
        ) == ("partial", "pcp_streaming")


def test_interleaved_logits_selection_dummy_and_bucket_filtering(monkeypatch):
    _fake_tpu_events(monkeypatch)
    scheduled = [9, 6, 1]
    hidden_per_rank = []
    managers = []
    for rank in range(4):
        r = _make_runner(
            num_computed_tokens=[0, 4, 20],
            prompt_tokens=[9, 10, 16],
            scheduled_tokens=scheduled,
            token_paddings=[16, 64, 512, 1024, 2048],
            pcp_size=4,
            block_size=16,
        )
        _attach_interleaved_manager(r, pcp_size=4, pcp_rank=rank, interleave_size=4)
        assert TPUModelRunner._backbone_token_paddings(r) == [512, 1024, 2048]
        r.num_tokens_paddings = [512, 1024]
        r.max_num_tokens = 1024
        TPUModelRunner._prepare_inputs(r, _scheduler_output(scheduled), 0, 3)
        local_ids = r.input_ids.to(torch.float32).unsqueeze(-1)
        hidden_per_rank.append(local_ids)
        managers.append(r.pcp_manager)

    with patch(
        "vllm_torchtpu.runner.pcp.sharding_plan.all_reduce_sum", side_effect=lambda x: x
    ):
        summed = sum(
            m.maybe_select_logits_hidden_states(h)
            for m, h in zip(managers, hidden_per_rank)
        )
    # Last tokens: req0 (len 9 -> 108), req1 (len 6 -> 209), req2 (pos 20 -> 320).
    assert summed[:3, 0].to(torch.int32).tolist() == [108, 209, 320]
    assert managers[0].sharding_plan.local_index_for_request_major_token(0) == 0
    assert managers[0].sharding_plan.local_index_for_request_major_token(4) is None

    # Prompt/decode boundary crossing is rejected.
    bad = _make_runner(
        num_computed_tokens=[6],
        prompt_tokens=[8],
        scheduled_tokens=[4],
        token_paddings=[512],
        pcp_size=4,
        block_size=16,
    )
    _attach_interleaved_manager(bad, pcp_size=4, pcp_rank=0, interleave_size=4)
    with pytest.raises(NotImplementedError, match="boundary"):
        TPUModelRunner._prepare_inputs(bad, _scheduler_output([4]), 0, 1)

    # Dummy step + finalize_hidden_states all-gather.
    managers[0].prepare_dummy(num_tokens=512, num_reqs=8, max_num_tokens=1024)
    assert managers[0].sharding_plan.global_padded_num_tokens == 2048
    assert (
        managers[0].build_metadata(
            runner=bad, block_table_obj=None, target_num_reqs=8, target_num_blocks=4
        )
        is None
    )
    with patch(
        "vllm_torchtpu.runner.pcp.interleaved_sharding_plan.all_gather_equal_tokens",
        side_effect=lambda x, dim=0: x.repeat(4, 1),
    ):
        out = managers[0].finalize_hidden_states(torch.ones((512, 8)))
    assert out.shape == (2048, 8)
    _ = np.zeros(1)
