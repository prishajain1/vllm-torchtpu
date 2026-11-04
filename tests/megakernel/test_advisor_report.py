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
"""Tests for HTML provenance badges, flag pills, and one-sided uncertainty whiskers."""

from __future__ import annotations

import re

from vllm_torchtpu.megakernel import advisor
from vllm_torchtpu.megakernel.advisor import chips
from vllm_torchtpu.megakernel.advisor import report as advisor_report

_WHISKER_RE = re.compile(r'<line\s+class="uncertainty-whisker"\s+([^/>]+)/>')
_ATTR_RE = re.compile(r'([a-zA-Z0-9_-]+)="([^"]*)"')


def test_html_renders_provenance_tags_flag_badges_and_onesided_bands() -> None:
    """Verify HTML report renders provenance badges, flag pills, and upward whiskers."""
    sweep = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    html_out = advisor.render_advisor_html(sweep)

    assert "<th>Provenance</th>" in html_out
    assert "<th>Flags</th>" in html_out
    assert '<details class="prov-details">' in html_out
    for tag in ("FACT", "DERIVED", "ASSUMED", "MEASURED"):
        assert f"prov-tag-{tag}" in html_out

    assert '<span class="flag-pill">' in html_out
    assert "coverage_assumed" in html_out

    multi_call_sweep = advisor.advise_regimes(
        "qwen3.5-397b", "tpu7x", vmem_budget_mib=14.0, op_overhead_us=50.0
    )
    assert "multi_call_unmodeled" in advisor.render_advisor_html(multi_call_sweep)

    unverified_cfg: dict[str, object] = {
        "model_type": "custom_causal_lm_v1",
        "hidden_size": 2048,
        "intermediate_size": 5632,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "vocab_size": 32000,
    }
    unverified_sweep = advisor.advise_regimes(unverified_cfg, "tpu7x")
    unverified_html = advisor.render_advisor_html(unverified_sweep)
    assert "architecture_unverified" in unverified_html

    band_svg = advisor_report.render_step_band_svg(sweep)
    assert band_svg in html_out
    whiskers = _WHISKER_RE.findall(band_svg)
    assert len(whiskers) == len(sweep.regimes)
    for raw_attrs in whiskers:
        attrs = dict(_ATTR_RE.findall(raw_attrs))
        x1 = float(attrs["x1"])
        x2 = float(attrs["x2"])
        floor_us = float(attrs["data-floor"])
        upper_us = float(attrs["data-upper"])
        assert x2 >= x1
        assert upper_us >= floor_us


def test_html_calibrated_provenance_promotes_measured_badge() -> None:
    """Verify calibrated measurements promote MEASURED badge counts in HTML output."""
    cal_data = chips.load_public_calibration("tpu7x")
    assert cal_data is not None
    sweep = advisor.advise_regimes(
        "Qwen/Qwen3-30B-A3B-FP8",
        "tpu7x",
        measurements=cal_data,
    )
    html_out = advisor.render_advisor_html(sweep)
    assert "MEASURED:" in html_out
    assert "MEASURED: 0" not in html_out
    assert "xplane:tpu7x_qwen3_30b_a3b_fp8_tp8_ep8" in html_out
