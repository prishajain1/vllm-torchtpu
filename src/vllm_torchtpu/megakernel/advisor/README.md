# TPU Megakernel Fusion-Scope Advisor

The `vllm_torchtpu.megakernel.advisor` package evaluates whether a model and serving workload benefit from fusing operators across a layer, layer group, or full step on TPU, using hardware specifications, model geometry, and optional trace calibration.

## Three-Step Workflow

### 1. Prior Estimate (`estimate`)

Evaluate HBM fit, VMEM token ceiling, roofline step floor, and uncertainty bands without a TPU or `jax` installation:

```bash
python -m vllm_torchtpu.megakernel.advisor estimate \
  --model qwen3.5-397b --chip tpu7x --phase decode --batch 16 --tp 8 \
  --explain --unresolved
```

### 2. Trace Calibration (`calibrate`)

Extract empirical host/device launch gaps, collective hop latencies, and fused-eligible step coverage into an `advisor.measurements.v1` JSON file from an XPlane profile directory or Perfetto JSON trace:

```bash
python examples/tpu_profiling.py \
  --model Qwen/Qwen3-30B-A3B-FP8 --tensor-parallel-size 8 \
  --enable-expert-parallel --profile-dir /tmp/tpu_profile
python -m vllm_torchtpu.megakernel.advisor calibrate \
  --trace /tmp/tpu_profile --chip tpu7x --model Qwen/Qwen3-30B-A3B-FP8 \
  --out measurements.json
```

### 3. Calibrated Evaluation (`--measurements`)

Re-run with `--measurements measurements.json` (or load bundled public priors via `chips.load_public_calibration("tpu7x")`) to promote `ASSUMED` inputs to `MEASURED` and settle `tie(measure: <input>)` verdicts:

```bash
python -m vllm_torchtpu.megakernel.advisor estimate \
  --model qwen3.5-397b --chip tpu7x --measurements measurements.json \
  --json report.json
```

## Provenance Tags

Every reported metric in `report.metrics` carries a provenance tag, unit, source string, and `[low, high]` band:

| Tag | Meaning |
| --- | --- |
| `FACT` | Hardware ceiling (`vmem_capacity_bytes`, `hbm_bw_bytes_per_s`, `bf16_flops_per_s`) or model config constant. |
| `DERIVED` | Closed-form function of `FACT` or `MEASURED` quantities (`tokens_per_core`, `onchip_ceiling`, `step_floor_us`). |
| `ASSUMED` | Uncalibrated prior bracket (`launch_gap_us`, `collective_latency_us`, `step_coverage`) with explicit `[low, high]` bounds. |
| `MEASURED` | Quantity populated from `--measurements` (`advisor.measurements.v1`). |

## Key CLI Flags

| Flag | Description |
| --- | --- |
| `--model` / `--config` | Registered preset name, HuggingFace repo ID, local directory, or `config.json` path. |
| `--chip` / `--chip-spec` | Built-in TPU target (`tpu7x`, `v6e`, `v5p`, `v5e`, `v4`, `auto`) or path to a custom JSON chip spec. |
| `--measurements` | Path to an `advisor.measurements.v1` JSON file. |
| `--objective` | `step` (minimize step latency), `throughput` (maximize `tok/s/chip`), or `both` (flag objective splits). |
| `--explain` / `--unresolved` | Print per-metric provenance tags or list unresolved `ASSUMED` inputs ranked by step-time sensitivity. |
| `--json [out.json]` | Write JSON output to stdout (when path omitted) or to `out.json`. |
| `--html out.html` | Write a self-contained inline-SVG HTML regime report to `out.html`. |

## Roofline Floor Contract

`step_floor_us` (`ScopeEstimate.step_bound_us`) is a hardware roofline lower bound:

- Optimistic floor (`band[0]`): peak hardware FLOP/s (FP8 peak on native-FP8 chips when weights are FP8) and lower-bound collective latency.
- Conservative upper bound (`band[1]` / `step_upper_us`): BF16 attention dot throughput on prefill plus upper-bound launch gap and collective hop latency.
- Whenever the top-2 feasible scope bands `[step_bound_us, step_upper_us]` overlap, `verdict` reports `tie(measure: <most_sensitive_input>)` rather than picking a winner from an uncalibrated assumption.
