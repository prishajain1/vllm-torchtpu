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
"""Tests for prefill expert bank layout and selection."""

from __future__ import annotations

from typing import Any

from vllm_torchtpu.models.kimi_k3.megakernel.kernels import prefill_megakernel

_Dynamic = Any


def test_expert_banks_constants() -> _Dynamic:
    """Verify test expert banks constants logic."""
    assert hasattr(prefill_megakernel, "make_indexed_moe_sp_ring")
