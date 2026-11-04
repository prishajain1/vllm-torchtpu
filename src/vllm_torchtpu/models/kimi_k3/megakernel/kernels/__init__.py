# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Kimi K3 checkpoint configuration and kernel primitives."""

from __future__ import annotations

from typing import Any

import jax

from vllm_torchtpu.kernels.kimi_k3.attention_residual import attention_residual

from .dspark_common import rms_norm as rms
from .load import Config

_Dynamic = Any


def tp_sum(x: _Dynamic) -> _Dynamic:
    """Sum tensor across TP axis."""
    return jax.lax.psum(x, "tp")


__all__ = ["Config", "attention_residual", "rms", "tp_sum"]
