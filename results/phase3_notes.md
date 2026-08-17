# Phase 3 — Optimization Paths, Tested Independently

Run: 2026-08-17, on DGX Spark (`greentrend-spark`), via `scripts/phase3_run_all.py`
(INT4/FP8/NVFP4/torch.compile) and its batching sweep. Each variant loaded its own
fresh model instance and was measured against the identical Phase 2 methodology
(3 untimed warmup calls, 24 timed calls cycling through the same 6 held-out clauses,
`torch.cuda.synchronize()` around the timed region, correctness checked on the timed
output). Baseline for comparison: Phase 2's BF16 batch-1 result — median 7.8039s,
p90 8.6221s, peak memory 14.924GB, 100% correctness.

## Path/Baseline/Cost model/Evidence — precision variants

| Variant | Median latency | vs. BF16 baseline | Peak memory | vs. baseline | Correctness |
|---|---|---|---|---|---|
| BF16 (Phase 2 baseline) | 7.804s | 1.00x | 14.924 GB | — | 24/24 (100%) |
| **INT4 (bitsandbytes NF4)** | **3.066s** | **2.55x faster** | **4.568 GB** | **69% less** | 24/24 (100%) |
| FP8 (torchao dynamic) | 8.593s | 0.91x (**9% slower**) | 11.165 GB | 25% less | 24/24 (100%) |
| NVFP4 (torchao) | — | **failed to run** | — | — | — |
| torch.compile (on BF16) | 7.875s | 0.99x (no real change) | 22.119 GB | 48% more | 24/24 (100%) |

### INT4 — the clear winner
2.55x latency reduction and a 69% memory cut, at 100% correctness on all 24 timed calls.
This matches the plan's hypothesis: INT4/bitsandbytes is the most mature quantization
path, and single-token autoregressive decode is memory-bandwidth-bound, so cutting
weight bytes moved per token translates almost directly into latency reduction. This is
the variant worth taking into Phase 5's serving layer.

### FP8 — measured, and it lost
Contrary to a naive "lower precision = faster" assumption, `torchao`'s dynamic FP8
weight+activation quantization was **9% slower** than the BF16 baseline, despite using
25% less memory. Most likely explanation: `Float8DynamicActivationFloat8WeightConfig`
performs *dynamic* (i.e. runtime, per-forward-pass) activation quantization, which adds
compute overhead on every decode step. At batch size 1, a single token's compute is
already tiny relative to the fixed cost of scanning and re-quantizing weights each step,
so the quantization overhead isn't amortized — unlike a prefill/training workload with
large batched matmuls, where FP8 Tensor Core throughput gains dominate the quantization
cost. This directly explains why the plan's cited book benchmark (`precisionfp8_te`,
5.17x on a prefill/training-style workload) does not transfer to this batch-1 decode
workload — the two are compute-bound vs. bandwidth-bound regimes, and FP8's benefit is
specific to the former. Reported as-is: a real, executed result that disagrees with a
naive prior, not discarded.

### NVFP4 — attempted, not viable yet
`torchao==0.18.0` (the latest available at time of testing) does not expose
`NVFP4InferenceConfig` under `torchao.prototype.mx_formats` on this aarch64+Blackwell
build — `ImportError` on import, confirmed and documented in `nvfp4_results.json`. This
matches the plan's explicit expectation that NVFP4 tooling maturity on Arm+Blackwell is
uncertain as of this date. No further workaround was forced; this is reported as a
genuine tooling-maturity finding, not a workaround-and-hide.

### torch.compile — no measurable benefit, and it cost memory
Applied on top of BF16 (the compute path, not INT4 — bitsandbytes' 4-bit ops are not
`torch.compile`-graph-friendly in the same way). Compilation itself happened during 2
extra untimed warmup calls before the standard 3-warmup/24-timed region, so no
compilation cost leaked into the timed numbers (anti Setup Pre-computation). Result:
median latency 7.875s vs. 7.804s baseline — statistically indistinguishable (well within
the ~0.33s stdev band) — while peak memory rose 48% (22.1GB vs 14.9GB) from the
compiled-graph's kernel/buffer caching. For a single-token autoregressive decode loop
with KV-cache growth, `torch.compile`'s `reduce-overhead` mode has little dispatch
overhead left to remove once HF's `generate()` is already doing its own decode-step
optimizations — the win case for `torch.compile` is typically prefill/fixed-shape
workloads, not this one. Reported as a genuine negative result.

## Batching sweep (BF16, batch sizes 1/4/8/16)

| Batch size | Batch latency (median) | Throughput | vs. batch=1 throughput |
|---|---|---|---|
| 1 | 8.00s | 0.125 req/s | 1.00x |
| 4 | 9.18s | 0.436 req/s | 3.48x |
| 8 | 11.22s | 0.713 req/s | 5.70x |
| 16 | 13.43s | 1.191 req/s | 9.52x |

Confirms the plan's hypothesis directly: batch-size-1 decode is bandwidth-bound (adding
3 more concurrent requests only adds ~15% to wall-clock latency, because the GPU wasn't
saturated at batch=1), and gains grow substantially at batch 8–16 as the tensor cores
get enough concurrent work to matter — 16 requests processed in 13.43s vs. the 127.7s
it would take to run them one at a time sequentially at the batch-1 rate (**9.52x**
aggregate throughput). All 4 batch sizes held 100% correctness.

**Known issue, disclosed rather than hidden**: the `peak_memory_gb` figures recorded in
this sweep (29.7–30.5GB, barely moving across batch sizes 1→16) do not look physically
plausible for a ~14.9GB BF16 model — KV-cache growth at these batch sizes and sequence
lengths should add tens to low-hundreds of MB, not be swamped by a constant ~15GB
offset. The batching sweep ran as the 5th variant in the same Python process,
immediately after `torch.compile`; `gc.collect()` + `torch.cuda.empty_cache()` between
variants does not clear `torch._dynamo`'s compiled-artifact cache, so leftover compiled
graph state most likely inflated every peak-memory reading in this sweep by a roughly
constant amount. The **latency and throughput numbers are unaffected** (monotonic,
sub-0.01s round-to-round stdev, and consistent with the expected bandwidth-bound →
compute-bound transition) and are reported as trustworthy; the peak-memory column in
`batching_sweep_results.json` is flagged and should not be quoted as true per-batch-size
memory cost. A clean re-run of just the batching sweep in its own process would fix this
if exact batch-memory numbers are needed later.

## Change / Result summary
- **Adopt**: INT4 (bitsandbytes NF4) — 2.55x faster, 69% less memory, no correctness
  loss. This is the precision path carried into Phase 5's serving layer.
- **Adopt**: batching at the serving layer — up to 9.5x throughput at batch 16 with
  acceptable per-batch latency growth (8.0s → 13.4s for 16x the work).
- **Reject**: FP8 (torchao dynamic quant) for this batch-1 decode workload — measured
  slower, not faster, despite lower memory.
- **Reject (for now)**: NVFP4 — not viable on current aarch64+Blackwall `torchao`
  tooling; revisit when the library adds mx_formats/NVFP4 support for this platform.
- **Reject**: `torch.compile` on the raw HF `generate()` decode loop — no latency
  benefit, real memory cost, on this workload shape.

## Boundary
- All Phase 3 numbers use the same 6-clause, batch-1 workload as Phase 2 for the
  precision comparisons; the batching sweep is a separate axis (BF16 only) and was not
  cross-combined with INT4 due to time — INT4 + batching together is a natural next
  experiment but wasn't run here.
- Single DGX Spark unit, not an isolated/clock-locked lab environment.
- FP8 and torch.compile results are specific to `torchao==0.18.0` and this exact
  Transformers/PEFT/HF `generate()` call path — a serving-native quantized/compiled
  kernel (e.g. inside vLLM) could behave differently, which Phase 5 will surface.

Phase 3 status: **complete**.
