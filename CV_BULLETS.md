# CV / Portfolio Bullets

Drawn directly from the executed results in this repo — see `FINAL_REPORT.md` for the
full write-up.

1. Benchmarked and optimized local inference for a fine-tuned Mistral-7B legal risk
   classifier on an NVIDIA DGX Spark (GB10 Grace Blackwell, 128GB unified memory),
   achieving a 2.55x latency reduction and 69% memory reduction via INT4 quantization,
   validated against a 5-precision comparison (INT4/FP8/NVFP4/torch.compile/BF16)
   with a warmup-isolated, hashed-workload benchmark harness.

2. Diagnosed autoregressive decode as memory-bandwidth-bound (not compute-bound)
   through live GPU utilization profiling and a controlled weight-bytes-vs-latency
   experiment, directly explaining why a naive "lower precision = faster" assumption
   failed for FP8 (measured 9% slower) while INT4 succeeded.

3. Deployed a production-shaped `POST /classify` API using vLLM's continuous batching
   and native LoRA serving on aarch64+Blackwell, achieving 15.4x throughput at 16
   concurrent requests with per-request latency holding steady, and served latency
   beating raw single-process inference even at concurrency 1.

4. Built and executed a benchmark methodology guarding against common validity
   failures (warmup bleed, stale-cache/constant-output, cherry-picking, config drift,
   setup pre-computation) across 6+ optimization variants, publishing every raw result
   — including two negative findings (FP8, NVFP4-not-yet-viable) — rather than only
   the wins.

5. Took a LoRA-adapted legal-clause-risk model (88.99% held-out accuracy, from Mistral
   7B's 67.36% zero-shot baseline) from a cold hardware environment through to a
   load-tested, benchmarked, publicly documented local deployment on NVIDIA's newest
   desktop AI hardware — end-to-end infra and optimization ownership, not just model
   training.

## Suggested second bullet for the existing legal-risk-classifier-lora project entry
Deployed and benchmarked the fine-tuned classifier for local inference on an NVIDIA
DGX Spark (Grace Blackwell), achieving 2.55x lower latency via INT4 quantization and
15.4x higher throughput under concurrent load via vLLM's continuous batching serving
layer — full methodology and raw results published publicly.
