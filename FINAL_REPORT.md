# Local Hosting & Optimization Case Study — Final Report

**legal-risk-classifier-lora** (Mistral-7B-Instruct-v0.3 + LoRA) on NVIDIA DGX Spark
(GB10 Grace Blackwell superchip, 128GB coherent unified memory)

All numbers below trace to a script in this repo that was actually executed on the
target hardware on 2026-08-17. Raw JSON for every phase is in [`results/`](results/).
Full per-phase writeups (this document is the consolidated summary) are in
`results/phase*_notes.md`.

## Executive summary

| Phase | Headline result |
|---|---|
| 0 — Environment | Ubuntu 24.04 aarch64, CUDA 13.0, GB10, 121GB unified memory. All deps installed cleanly — no wheel workarounds needed, contrary to expectation. |
| 1 — Model verification | Base + LoRA load correctly (14.5GB → 14.8GB). Caught and fixed a real prompt-template mismatch before it could invalidate later results. 5/6 held-out clauses correctly classified; the 1 miss reproduces a documented model weakness. |
| 2 — Baseline harness | BF16, batch 1: **7.80s median / 8.62s p90 latency**, 100% correctness, 14.9GB peak memory. Anti-pattern-guarded methodology (warmup isolation, hashed workload, fresh inputs per call). |
| 3 — Optimization paths | **INT4 (bitsandbytes): 2.55x faster, 69% less memory** — the clear winner. FP8: measured **9% slower**. NVFP4: **not viable** on this stack (ImportError). torch.compile: no measurable gain. Batching: up to **9.5x throughput** at batch 16. |
| 4 — Hardware utilisation | Decode confirmed **memory-bandwidth-bound**, not compute-bound (low power at high occupancy; latency tracks weight-bytes near-linearly). `nvidia-smi` memory telemetry doesn't work on unified memory. NVLink-C2C host-transfer optimization category confirmed inapplicable by architecture. |
| 5 — Serving layer | vLLM + FastAPI `/classify` endpoint. Served latency **beat raw in-process latency** even at concurrency 1. **15.4x throughput** at concurrency 16, latency improved under load, 100% correctness across 76 requests. |

## Path

`Govardhan12345/legal-risk-classifier-lora`, a LoRA adapter (rank 32, ~85M trainable
params) fine-tuned on `mistralai/Mistral-7B-Instruct-v0.3` for CUAD-style contract
clause risk classification, deployed locally on a single NVIDIA DGX Spark unit and
benchmarked end-to-end: environment → model verification → trustworthy baseline →
independently-tested optimizations → hardware root-cause analysis → production-style
serving layer.

## Baseline

BF16, eager attention, batch size 1, single request at a time — median 7.804s latency,
100% correctness (24/24 timed calls), 14.924GB peak GPU memory. Frozen and hashed as a
`WorkloadSpec` (`sha256: dc1332ba...`) so every later comparison is against a
byte-identical reference (`results/baseline_results.json`).

## Cost model

Per-request cost at baseline: ~7.8s wall time, ~13 tokens/sec generation rate,
dominated by autoregressive decode. Confirmed in Phase 4 to be memory-bandwidth-bound:
cutting weight bytes moved per token by 75% (BF16→INT4) cut per-token latency by 61% —
a near-linear relationship a compute-bound kernel would not show.

## Evidence

- `results/baseline_results.json` — Phase 2 baseline, 24 raw per-call latencies
- `results/int4_results.json`, `fp8_results.json`, `nvfp4_results.json`,
  `torch_compile_results.json`, `batching_sweep_results.json` — Phase 3, each variant
  measured independently against the same methodology
- `results/phase4_utilization.json` — live `nvidia-smi` sampling during generation,
  BF16 vs INT4
- `results/phase5_load_test.json` — served endpoint, sequential + concurrency sweep

## Change

Adopted for a production deployment on this hardware:
1. **INT4 quantization (bitsandbytes NF4)** as the model precision — 2.55x faster, 69%
   less memory, zero correctness cost.
2. **vLLM as the serving engine**, not a hand-rolled FastAPI+`generate()` loop — native
   LoRA support, continuous batching, PagedAttention, worked cleanly on aarch64+Blackwell
   with no fallback needed.
3. **Batching/concurrency at the serving layer** — the single largest throughput lever
   found in this entire project (15.4x), larger than any single precision change.

Rejected, with evidence:
- FP8 (torchao dynamic quantization) — measured slower than the BF16 baseline on this
  batch-1 decode workload; the quantization overhead doesn't amortize without large
  batched matmuls.
- NVFP4 — tooling not yet available for this platform (`torchao==0.18.0`,
  aarch64+Blackwell); a genuine "not ready yet" finding, not a workaround.
- `torch.compile` on the raw HF `generate()` loop — no latency benefit, real memory cost.

## Result

A running, load-tested `POST /classify` endpoint on real hardware, served through vLLM,
returning clause type / risk level / summary / reason / confidence for real contract
clause text — end to end, from a cold environment to a production-shaped API, with
every number in this document backed by a script and a JSON artifact in this repo.

## Boundary

- Single DGX Spark unit, not clock-locked/isolated lab conditions; background OS
  processes were present throughout.
- The held-out clause set is small (6 clauses) — sufficient to verify correctness and
  characterize latency distribution, but not a statistically comprehensive risk-model
  evaluation (that already exists in the adapter's own model card, 454 held-out
  examples).
- Phase 5's prefix-cache hit rate (90.6%) is inflated by the repeating 6-clause
  workload; real production traffic with unique clause text would see less benefit
  from that specific mechanism, though the underlying continuous-batching throughput
  gain is independent of it.
- INT4 and vLLM serving were validated independently (Phase 3 vs Phase 5) but not
  combined in this project — INT4-quantized serving through vLLM is a natural next
  experiment.
- FP8/NVFP4/torch.compile findings are specific to the exact library versions used
  (`torchao==0.18.0`, `transformers==5.15.0`, `vllm==0.27.1`) — all subject to change
  as this tooling matures on Arm+Blackwell.

## What's next
- Combine INT4 quantization with vLLM serving (not yet tested together).
- Re-test NVFP4 once `torchao` (or `nvidia-modelopt`) ships mx_formats/NVFP4 support
  for this platform.
- Push concurrency past 16 — GPU KV cache usage was only 0.3% at 16 concurrent
  requests, suggesting significant untested headroom.
- Broaden the held-out clause set beyond 6 examples for a tighter latency-distribution
  estimate across more clause-type/output-length combinations.

---
Code and full results: [github.com/Gr150/legal-risk-classifier-inference](https://github.com/Gr150/legal-risk-classifier-inference)
