# Phase 4 — Unified Memory & Hardware Utilisation Investigation

Run: 2026-08-17, on DGX Spark (`greentrend-spark`), via `scripts/phase4_utilization.py`,
raw output in `phase4_utilization.json`. Profiles a single BF16 decode call and a single
INT4 decode call while sampling `nvidia-smi` every 100ms during generation.

## Path
Same batch-1, single-request decode workload as Phases 2–3, with a live `nvidia-smi`
sampler running in a background thread during the timed `generate()` call.

## Is decode compute-bound or bandwidth-bound?

### Evidence 1 — GPU "busy" but power draw stays low
| Precision | Mean GPU util | Mean power draw | SM clock |
|---|---|---|---|
| BF16 | 92% | 27.7 W | 2411 MHz (constant, no throttling) |
| INT4 | 84.6% | 37.2 W | 2411 MHz (constant, no throttling) |

`nvidia-smi`'s `utilization.gpu` reports the fraction of time *any* kernel was
running/scheduled on the SMs — it does not mean the ALUs are doing heavy FLOP work.
27.7W at 92% "utilization" is a classic memory-stall signature: the SM is occupied
issuing instructions and waiting on data from memory, not saturating its compute
pipelines (a genuinely compute-bound kernel at this clock would be expected to draw
meaningfully more power). INT4 draws *more* power (37.2W) at *lower* reported
utilization (84.6%) — consistent with INT4 doing real dequantization compute work per
weight read, whereas BF16 spends its cycles mostly waiting.

### Evidence 2 — Weight-byte reduction predicts the observed speedup
| Precision | Weight bytes (approx) | Time/token | Achieved effective bandwidth |
|---|---|---|---|
| BF16 | 14.4 GB | 80.1 ms | 179.7 GB/s |
| INT4 (NF4) | 3.6 GB | 31.0 ms | 116.0 GB/s |

Cutting weight bytes moved per decode step by **75%** (14.4GB → 3.6GB) reduced
per-token latency by **61%** (80.1ms → 31.0ms). A compute-bound kernel would show
latency essentially *unchanged* by a reduction in weight byte-size, since compute-bound
cost scales with FLOPs, not bytes moved. The fact that latency tracks weight size this
closely is direct, repeatable evidence that batch-1 decode on this model is
**memory-bandwidth-bound**, exactly as the plan hypothesized as the standard behavior
for autoregressive LLM decode generally.

(INT4's achieved bandwidth, 116 GB/s, is *lower* than BF16's 179.7 GB/s despite moving
fewer bytes — expected, since NF4 dequantization adds real per-weight compute overhead
that a pure bandwidth model doesn't capture. The 2.55x wall-clock speedup measured in
Phase 3 is the number that matters; the "achieved bandwidth" figures here are a
diagnostic, not a headline claim.)

**Note on the ~273 GB/s figure sometimes cited for DGX Spark's LPDDR5x unified memory**:
this project did not independently measure raw memory bandwidth (e.g. via a STREAM-style
microbenchmark) — the 179.7 GB/s BF16 figure above is inference workload's *achieved*
bandwidth, not peak, and a real end-to-end LLM decode loop is expected to land well
under any vendor peak number due to attention/kernel-launch overhead. No peak-bandwidth
claim is made here beyond what was actually measured.

## GB10 unified memory shows up in `nvidia-smi` reporting gaps, not just architecture docs
Every sample had `mem_util_pct = 0` and `mem_clock_mhz` / `mem_used_mib` = `[N/A]`,
regardless of precision or actual memory pressure. This confirms the Phase 0 finding
(`nvidia-smi`'s top-level `Memory-Usage` field also read "Not Supported" for GB10):
the tool's standard discrete-GPU memory-reporting fields are not wired up for the
Grace-Blackwell unified-memory architecture. Real GPU memory usage in this project was
tracked via `torch.cuda.memory_allocated()` / `max_memory_allocated()` instead (as done
in Phases 1–3), not `nvidia-smi`.

## NVLink-C2C / host-device transfer cost
No dedicated NVLink-C2C bandwidth microbenchmark (e.g. `nvbandwidth`, or the kind of
`nvlink_c2c_bandwidth_benchmark.py` referenced in the plan) was available/installed on
this system, and none was run — reported here rather than fabricated. What was directly
observed: across all of Phases 0–4, model loading via `AutoModelForCausalLM.from_pretrained(device_map="auto")`
showed no separate, measurable "host-to-device copy" phase distinct from weight
deserialization — unlike a discrete-GPU box where weights are read from disk/host RAM
and then explicitly copied over PCIe into VRAM as a distinct step. On GB10's coherent
unified memory, CPU and GPU address the same physical memory pool, so there is no
analogous transfer step for `device_map="auto"` to perform. This matches the plan's
hypothesis directly: **the "host-device transfer optimization" category the book
targets on discrete GPUs is not applicable on this hardware** — not because it was
tried and failed, but because the architecture removes the problem it targets. That
"doesn't apply here" is itself the finding, not a gap in the investigation.

## Conclusion
- Decode at batch size 1 is bandwidth-bound, not compute-bound — supported by both the
  low-power/high-occupancy signature and the near-linear latency-vs-weight-bytes
  relationship measured directly (Phase 3's INT4 result plus this phase's per-token
  instrumentation).
- `nvidia-smi`'s standard memory telemetry doesn't work on GB10's unified memory
  architecture — `torch.cuda` memory APIs are the reliable source in this project.
- Host-device transfer optimization (a discrete-GPU concern) doesn't apply here by
  construction — unified memory removes the transfer step, not just its cost.

## Boundary
- Single-call profiling per precision (not averaged across many runs like Phases 2–3),
  since the goal here was characterizing utilization *shape* during one representative
  call, not a statistically tight latency estimate — those numbers already exist from
  Phase 2/3.
- "Achieved effective bandwidth" is a derived estimate from `(approx weight bytes) /
  (measured time per token)`, not a direct hardware bandwidth-counter measurement — no
  Nsight/profiler tooling was available on this system to measure DRAM bytes moved
  directly.
- Single DGX Spark unit, not an isolated/clock-locked lab environment.

Phase 4 status: **complete**.
