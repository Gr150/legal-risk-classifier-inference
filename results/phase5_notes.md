# Phase 5 — Serving Layer

Run: 2026-08-17, on DGX Spark (`greentrend-spark`). vLLM (0.27.1) served as the
backend inference engine with native LoRA support, `mistralai/Mistral-7B-Instruct-v0.3`
+ `Govardhan12345/legal-risk-classifier-lora`, BF16, behind a thin FastAPI wrapper
(`scripts/serve_app.py`) exposing `POST /classify`. Load tested with
`scripts/phase5_load_test.py`.

## Path
```
vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
  --enable-lora --lora-modules legal-risk=Govardhan12345/legal-risk-classifier-lora \
  --max-lora-rank 32 --dtype bfloat16 --port 8000
```
+ `uvicorn serve_app:app --port 8080` in front, calling vLLM's OpenAI-compatible
`/v1/completions` with `logprobs=1`, parsing the model's JSON output, and returning
`clause_type` / `risk_level` / `summary` / `reason` / `confidence_approx` (mean
token-logprob converted via `exp()` — an approximation, not a calibrated probability,
and documented as such in the code).

**vLLM risk check (per the plan's explicit flag)**: `pip install vllm` succeeded
cleanly on this aarch64+Blackwell system (v0.27.1, aarch64 wheels available), imported
without error, and served both the base model and the LoRA adapter correctly. No
FastAPI-only fallback was needed — contrary to the plan's stated risk that vLLM's Arm
support might lag x86/CUDA.

## `/classify` endpoint — manual sanity check
```json
{
  "clause_type": "Termination For Convenience That Should Be Reviewed By A Lawyer.",
  "risk_level": "High",
  "summary": "This clause addresses termination for convenience...",
  "reason": "Classified as High risk based on standard legal assessment...",
  "confidence_approx": 0.9945,
  "latency_s": 7.4325
}
```
Output matches the pattern already verified in Phases 1–3 (same JSON schema, same
risk-table alignment).

## Cost model — served vs. raw (comparison against Phase 2/3)
| Path | Median latency | Notes |
|---|---|---|
| Raw HF `generate()`, BF16, batch 1 (Phase 2) | 7.804s | In-process, no server |
| **Served (vLLM+FastAPI), single request** | **6.997s** | Through HTTP, full round trip |

Served latency at concurrency 1 is actually **~10% faster** than the raw in-process
Phase 2 baseline, despite the added HTTP hop (FastAPI → vLLM) — vLLM's PagedAttention
and optimized CUDA kernels more than offset the serving overhead on this workload. This
disagrees with the plan's stated hypothesis that served latency would be higher than
raw single-call latency at low concurrency — reported as measured, not adjusted to fit
the prior expectation.

## Concurrency sweep — 16 requests at each level
| Concurrency | Wall time | Throughput | Per-request median latency |
|---|---|---|---|
| 1 | 113.54s | 0.141 req/s | 6.957s |
| 4 | 28.15s | 0.568 req/s | 6.661s |
| 8 | 13.95s | 1.147 req/s | 6.647s |
| 16 | 7.35s | **2.178 req/s** | **6.636s** |

**15.4x throughput at concurrency 16 vs. concurrency 1** — and per-request median
latency did not degrade under load, it improved slightly (6.957s → 6.636s). This
directly reproduces (and exceeds) the plan's cited ch16 finding that request
scheduling/batching at the serving layer matters more than raw kernel-level changes —
Phase 3's own manual left-padded batching sweep only reached 9.5x throughput at batch
16; vLLM's continuous batching + PagedAttention scheduler reached 15.4x on the same
hardware and model.

76/76 requests across the sequential run and full concurrency sweep returned valid,
schema-correct JSON (100%) — the `/classify` endpoint itself raises an HTTP 502 on any
JSON-parse or missing-key failure, so `ok: true` is itself a correctness signal, not
just an HTTP-status check.

## Evidence from the vLLM server log (concurrency=16 window)
```
Running: 16 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.3%, Prefix cache hit rate: 90.6%
Avg generation throughput: 189.0 tokens/s
```
- **GPU KV cache usage was only 0.3%** at 16 concurrent requests — this hardware has
  enormous unused headroom for concurrency well beyond 16 before KV cache becomes the
  limiting factor, consistent with Phase 4's finding that batch-1 decode leaves the GPU
  underutilized.
- **Prefix cache hit rate 90.6%** — flagged as a real caveat, not hidden: the workload
  reuses only 6 distinct clauses, and every prompt shares an identical system-prompt
  prefix. vLLM's automatic prefix caching benefits heavily from this repetition. Fully
  unique production clause text (no repeated prefixes/content) would see a lower cache
  hit rate and correspondingly less of this speedup, though the underlying
  continuous-batching throughput gain is independent of prefix caching and would still
  apply.

## Change / Result
- **Adopt**: vLLM as the serving backend — it worked cleanly on this hardware, beat raw
  in-process latency even at concurrency 1, and delivered 15.4x throughput scaling with
  no correctness loss.
- **Adopt**: the `POST /classify` FastAPI wrapper as the production-facing interface —
  demonstrable, deployable, and returns the plan's required "risk category + confidence
  out" shape (with the confidence caveat noted above).

## Boundary
- Confidence score is an approximation (`exp(mean token logprob)`), not a calibrated
  probability — flagged in both the code and this report.
- Prefix-cache hit rate (90.6%) is inflated by the small, repeating 6-clause workload;
  real production traffic with fully unique clauses would see less benefit from this
  specific mechanism (though not from continuous batching generally).
- Load test used up to 16 concurrent requests only, chosen to match Phase 3's batching
  sweep for direct comparison — GPU KV cache headroom (0.3% used) suggests this
  hardware could sustain meaningfully higher concurrency, untested here.
- Single DGX Spark unit, not an isolated/clock-locked lab environment; `vllm serve` and
  the FastAPI wrapper ran on the same machine as the load-test client (no network
  latency beyond localhost).

Phase 5 status: **complete**.
