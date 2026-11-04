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

"""State-storage selection and verifier rounding at decode token boundaries."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

from vllm_torchtpu.models.kimi_k3.megakernel import decode, decode_ops, program
from vllm_torchtpu.models.kimi_k3.megakernel.decode_ops import DecodeOps
from vllm_torchtpu.models.kimi_k3.megakernel.kernels import decode_megakernel as kernel
from vllm_torchtpu.models.kimi_k3.megakernel.runtime_shim import backend
from vllm_torchtpu.models.kimi_k3.megakernel.runtime_shim.config import (
    MegakernelEngineConfig,
)

_Dynamic = Any


pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    ("setting", "expected"),
    [("auto", "float32"), ("float32", "float32"), ("bfloat16", "bfloat16")],
)
def test_engine_config_passes_state_dtype_to_decoder(
    monkeypatch: _Dynamic, setting: _Dynamic, expected: _Dynamic
) -> _Dynamic:
    """Verify test engine config passes state dtype to decoder logic."""
    monkeypatch.setattr(backend, "role", lambda _: "decode")
    monkeypatch.setattr(backend, "attention_dp", lambda _: 2)
    monkeypatch.setattr(backend, "synthetic_acceptance_lengths", lambda _: None)
    config = MegakernelEngineConfig.from_vllm_config(
        SimpleNamespace(
            parallel_config=SimpleNamespace(
                tensor_parallel_size=32,
                decode_context_parallel_size=16,
                cp_kv_cache_interleave_size=256,
            ),
            scheduler_config=SimpleNamespace(
                max_num_seqs=128, max_num_batched_tokens=128, async_scheduling=True
            ),
            model_config=SimpleNamespace(max_model_len=139264),
            cache_config=SimpleNamespace(
                block_size=256, cache_dtype="auto", mamba_ssm_cache_dtype=setting
            ),
            speculative_config=None,
        )
    )
    monkeypatch.setattr(
        decode_ops, "DecodeOps", lambda _path, **_kw: SimpleNamespace(**_kw)
    )
    monkeypatch.setattr(decode, "KimiDecoder", lambda _config, ops, _draft, **_kw: ops)
    ops = program.build(config, Path("unused"), tp_rank=0)
    assert ops.mamba_ssm_cache_dtype == setting
    assert program.recurrent_state_dtype(ops.mamba_ssm_cache_dtype) == expected


@pytest.mark.parametrize("setting", ["float16", "fp8", "invalid"])
def test_unsupported_state_dtype_rejected_before_allocation(
    setting: _Dynamic,
) -> _Dynamic:
    """Verify test unsupported state dtype rejected before allocation logic."""
    with pytest.raises(ValueError, match="mamba-ssm-cache-dtype"):
        DecodeOps(Path("unused"), num_slots=2, mamba_ssm_cache_dtype=setting)


def kda_recurrence_call(
    projected: _Dynamic,
    history: _Dynamic,
    state: _Dynamic,
    sequence_rows: _Dynamic,
    *,
    interpret: _Dynamic = True,
) -> _Dynamic:
    """Actual KDA function with fixed projections, excluding cross-rank work."""
    batch = projected.shape[0]

    def body(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
        """Execute body logic."""
        _p_iter = iter(_p_args)
        projected_ref = next(_p_iter, _p_kw.get("projected_ref"))
        history_ref = next(_p_iter, _p_kw.get("history_ref"))
        state_ref = next(_p_iter, _p_kw.get("state_ref"))
        feature_ref = next(_p_iter, _p_kw.get("feature_ref"))
        next_history = next(_p_iter, _p_kw.get("next_history"))
        next_state = next(_p_iter, _p_kw.get("next_state"))
        output = next(_p_iter, _p_kw.get("output"))
        no_copy = SimpleNamespace(wait=lambda: None)
        weights = SimpleNamespace(
            gate_projection=jnp.zeros((128, 128), jnp.bfloat16),
            input_projection=jnp.zeros((128, 1664), jnp.bfloat16),
            feature_gate_projection_vmem=feature_ref,
            exp_log_decay_rate=jnp.zeros((3, 1), jnp.float32),
            time_step_bias=jnp.full((3, 128), -3.0, jnp.float32),
            convolution_filter=jnp.full((4, 3, 3, 128), 0.25, jnp.bfloat16),
            output_normalization=jnp.ones((128,), jnp.bfloat16),
            make_projection_copies=lambda: SimpleNamespace(
                gate=no_copy, feature_gate=no_copy
            ),
        )
        kernel.kda(
            jnp.zeros((batch, 128), jnp.bfloat16),
            jnp.zeros((batch, 128), jnp.bfloat16),
            next_history,
            next_state,
            weights=weights,
            convolution_history_ref=history_ref,
            recurrent_state_ref=state_ref,
            communication_workspace=SimpleNamespace(
                reduction_phase=jnp.array([0], jnp.int32)
            ),
            synchronize_devices=False,
            weights_are_prefetched=True,
            sequence_rows=sequence_rows,
            shard_index=0,
            first_call=False,
            attended_out=output,
            gate_features_vmem=jnp.zeros((batch, 128), jnp.bfloat16),
            projected_rows=projected_ref[...],
        )

    return pl.pallas_call(
        body,
        out_shape=(
            jax.ShapeDtypeStruct(history.shape, history.dtype),
            jax.ShapeDtypeStruct(state.shape, state.dtype),
            jax.ShapeDtypeStruct((batch, 384), jnp.bfloat16),
        ),
        interpret=interpret,
    )(projected, history, state, jnp.zeros((128, 384), jnp.bfloat16))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("width", [4, 8])
def test_verifier_state_matches_sequential_decode(
    dtype: _Dynamic, width: _Dynamic
) -> _Dynamic:
    """Verify test verifier state matches sequential decode logic."""
    rng = np.random.default_rng(413)
    requests = 2
    projected = jnp.asarray(rng.normal(size=(requests, width, 1664)), jnp.bfloat16)
    history = jnp.asarray(rng.normal(size=(requests, 3, 3, 4, 128)), jnp.bfloat16)
    state = jnp.asarray(rng.normal(size=(requests, 3, 128, 128)), dtype)
    batched = kda_recurrence_call(
        projected.reshape(-1, 1664),
        jnp.repeat(history, width, axis=0),
        jnp.repeat(state, width, axis=0),
        width,
    )
    for offset in range(width):
        history, state, attended = kda_recurrence_call(
            projected[:, offset], history, state, 1
        )
        # Only the three real conv heads are written (the fourth is padding).
        np.testing.assert_array_equal(
            batched[0][offset::width, :, :, :3], history[:, :, :, :3]
        )
        np.testing.assert_array_equal(batched[1][offset::width], state)
        np.testing.assert_array_equal(batched[2][offset::width], attended)


def test_prefill_rejects_bf16_instead_of_ignoring_the_flag() -> _Dynamic:
    """Verify test prefill rejects bf16 instead of ignoring the flag logic."""
    config = MegakernelEngineConfig(
        role="prefill",
        tensor_parallel_size=32,
        decode_context_parallel_size=32,
        block_size=256,
        max_num_seqs=1,
        max_model_len=8192,
        max_num_batched_tokens=8192,
        mamba_ssm_cache_dtype="bfloat16",
    )
    with pytest.raises(NotImplementedError, match="decode-only"):
        program.check(config)
