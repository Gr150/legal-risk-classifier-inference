# legal-risk-classifier-inference

Local hosting, benchmarking, and optimization case study for
[`Govardhan12345/legal-risk-classifier-lora`](https://huggingface.co/Govardhan12345/legal-risk-classifier-lora)
(Mistral-7B-Instruct-v0.3 + LoRA) on an NVIDIA DGX Spark (GB10 Grace Blackwell superchip,
128GB coherent unified memory).

Every number in this repo comes from a run that was actually executed on the target
hardware. See [environment.md](environment.md) for the exact environment and [results/](results/)
for raw JSON output per phase.

## Phases
- Phase 0 — Environment setup ([environment.md](environment.md))
- Phase 1 — Load and verify the real model (`scripts/phase1_lora_verify.py`)
- Phase 2 — Trustworthy benchmark harness (in progress)
- Phase 3 — Optimization paths: INT4 / FP8 / NVFP4 / torch.compile / batching (planned)
- Phase 4 — Unified memory & hardware utilisation investigation (planned)
- Phase 5 — Serving layer (planned)
- Phase 6 — Write-up (planned)
