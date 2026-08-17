# legal-risk-classifier-inference

Local hosting, benchmarking, and optimization case study for
[`Govardhan12345/legal-risk-classifier-lora`](https://huggingface.co/Govardhan12345/legal-risk-classifier-lora)
(Mistral-7B-Instruct-v0.3 + LoRA) on an NVIDIA DGX Spark (GB10 Grace Blackwell superchip,
128GB coherent unified memory).

Every number in this repo comes from a run that was actually executed on the target
hardware. **Start here: [FINAL_REPORT.md](FINAL_REPORT.md)** for the consolidated
write-up, or [environment.md](environment.md) for the exact environment. Raw JSON
results per phase are in [results/](results/); CV/portfolio bullets in
[CV_BULLETS.md](CV_BULLETS.md).

## Headline result
INT4 quantization: **2.55x faster, 69% less memory**, zero correctness loss. vLLM
serving: **15.4x throughput** at 16 concurrent requests, latency held steady under
load. Two documented negative results (FP8 slower than baseline; NVFP4 not yet viable
on this stack) reported alongside the wins.

## Phases
- Phase 0 — Environment setup ([environment.md](environment.md))
- Phase 1 — Load and verify the real model (`scripts/phase1_lora_verify.py`, [results/phase1_notes.md](results/phase1_notes.md))
- Phase 2 — Trustworthy benchmark harness (`scripts/phase2_benchmark_baseline.py`, [results/phase2_notes.md](results/phase2_notes.md))
- Phase 3 — Optimization paths: INT4 / FP8 / NVFP4 / torch.compile / batching (`scripts/phase3_run_all.py`, [results/phase3_notes.md](results/phase3_notes.md))
- Phase 4 — Unified memory & hardware utilisation investigation (`scripts/phase4_utilization.py`, [results/phase4_notes.md](results/phase4_notes.md))
- Phase 5 — Serving layer: vLLM + FastAPI `/classify` (`scripts/serve_app.py`, `scripts/phase5_load_test.py`, [results/phase5_notes.md](results/phase5_notes.md))
- Phase 6 — Write-up ([FINAL_REPORT.md](FINAL_REPORT.md), [CV_BULLETS.md](CV_BULLETS.md))
