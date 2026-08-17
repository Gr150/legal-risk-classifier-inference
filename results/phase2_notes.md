# Phase 2 — Trustworthy Benchmark Harness (Baseline)

Run: 2026-08-17, on DGX Spark (`greentrend-spark`), via
`scripts/phase2_benchmark_baseline.py`, raw output in `baseline_results.json`.

## Path
`Govardhan12345/legal-risk-classifier-lora` (Mistral-7B-Instruct-v0.3 + LoRA), BF16,
eager attention, batch size 1, single request at a time, on NVIDIA GB10.

## WorkloadSpec (frozen, hashed)
`sha256: dc1332ba2b6f65b13be17236a84668f10c11519ae081804c9f689068abd87f74`

- 6 fixed held-out clauses (same set verified in Phase 1), cycled 4x for 24 timed calls
- `max_new_tokens=300`, `temperature=0.1`, `seed=42+call_index` (varied but reproducible)
- Prompt template: Mistral `[INST]` wrapper matching the adapter's documented training format

## Anti-pattern guards applied
- **Warmup Bleed**: 3 untimed `generate()` calls before `torch.cuda.reset_peak_memory_stats()`
  and the timed region begins.
- **Stale Cache / Constant Output**: each timed call uses a genuinely different clause
  (cycling through all 6), not the same cached input repeated.
- **Cherry-picking**: all 24 raw per-call latencies saved in `per_call`, not just a best case.
- **Config Immutability**: workload + generation params hashed; any drift is detectable
  by re-hashing and comparing.
- **Setup Pre-computation**: correctness (valid JSON, required keys, valid risk_level) is
  checked on the actual timed `generate()` output, not a separate untimed pass.
- `torch.cuda.synchronize()` called immediately before and after each timed region, so
  latency reflects actual GPU completion, not just kernel dispatch.

## Baseline cost model
| Metric | Value |
|---|---|
| Median latency | 7.804s |
| P90 latency | 8.622s |
| Mean latency | 7.948s |
| Stdev | 0.325s (4.1% of mean — low run-to-run variance) |
| Min / Max | 7.679s / 8.643s |
| Peak GPU memory | 14.924 GB |
| Correctness | 24/24 (100%) |

Average output length ≈ 103 new tokens per call → **~13.0 tokens/sec** at batch size 1,
BF16, eager. This is the reference throughput number Phase 3's optimizations are measured
against.

## Evidence
Matches the plan's Phase 2 hypothesis: single-digit-to-low-double-digit-second latency,
dominated by autoregressive decode (300-token budget, ~100-112 tokens actually generated
per clause). Low stdev (4% of mean) indicates a stable, reproducible baseline — no thermal
throttling or contention observed during the run. 100% correctness on the timed path
confirms the Phase 1 prompt-template fix held under repeated sampling, not just the
single manual check.

## Boundary
- Single-unit DGX Spark, not clock-locked/isolated lab conditions — background OS/desktop
  processes were present (per Phase 0's `nvidia-smi` process list showing Xorg/gnome-shell).
- `do_sample=True` with `temperature=0.1` introduces minor sampling variance per call
  (mitigated by fixed per-call seeds for reproducibility, but not fully deterministic
  like greedy decoding would be).
- Only 6 distinct clause inputs cycled — a larger, more diverse held-out set would give a
  tighter estimate of true output-length variance across clause types.

Phase 2 status: **complete**.
