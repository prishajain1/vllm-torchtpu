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
"""Tests for fused MoE dispatch."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vllm_torchtpu.models.kimi_k3.megakernel.kernels import (
    prefill_megakernel as kernels,
)

_Dynamic = Any


@pytest.mark.parametrize("window", [512, 1024, 2048, 4096, 8192])
def test_dispatch_by_window_unconditional_fused(window: _Dynamic) -> _Dynamic:
    """Verify test dispatch by window unconditional fused logic."""
    mock_mesh = MagicMock()
    with (
        patch(
            "vllm_torchtpu.models.kimi_k3.megakernel.kernels.prefill_moe.fused.make_indexed_moe_sp_ring"
        ) as mock_fused,
        patch(
            "vllm_torchtpu.models.kimi_k3.megakernel.kernels.prefill_megakernel._make_indexed_moe_sp_ring_reference"
        ) as mock_ref,
    ):
        mock_fused.return_value = ("fused", window)
        mock_ref.return_value = ("ref", window)

        res = kernels.make_indexed_moe_sp_ring(mock_mesh, seq_len=window)
        assert res == ("fused", window)
        mock_fused.assert_called_once_with(mock_mesh, seq_len=window)


@pytest.mark.parametrize("window", [256, 16384])
def test_dispatch_outside_window_reference_fallback(window: _Dynamic) -> _Dynamic:
    """Verify test dispatch outside window reference fallback logic."""
    mock_mesh = MagicMock()
    with (
        patch(
            "vllm_torchtpu.models.kimi_k3.megakernel.kernels.prefill_moe.fused.make_indexed_moe_sp_ring"
        ) as mock_fused,
        patch(
            "vllm_torchtpu.models.kimi_k3.megakernel.kernels.prefill_megakernel._make_indexed_moe_sp_ring_reference"
        ) as mock_ref,
    ):
        mock_fused.return_value = ("fused", window)
        mock_ref.return_value = ("ref", window)

        res = kernels.make_indexed_moe_sp_ring(mock_mesh, seq_len=window)
        assert res == ("ref", window)
        mock_ref.assert_called_once_with(mock_mesh, seq_len=window)
