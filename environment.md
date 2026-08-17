# Environment — DGX Spark (greentrend-spark)

Recorded: 2026-08-17

## Hardware / OS
- Host: `greentrend-spark` (Tailscale-reachable)
- OS: Ubuntu 24.04.4 LTS (Noble Numbat), kernel `6.17.0-1029-nvidia`, `aarch64`
- GPU: NVIDIA GB10 (DGX Spark, Grace Blackwell superchip)
- Driver: 580.173.02
- CUDA (driver-reported): 13.0
- `nvcc`: release 13.0, V13.0.88 (build `cuda_13.0.r13.0/compiler.36424714_0`)
- Memory: 121Gi total unified (RAM), 62Gi free / 117Gi available at time of check, 15Gi swap
- Disk: `/dev/nvme0n1p2` 3.7T total, 551G used, 3.0T available, mounted at `/`
- Python: 3.12.3 at `/usr/bin/python3`
- `torch`: not installed yet

## Notes
- `nvidia-smi` reports GPU memory as "Not Supported" for GB10 — this is expected on the
  Grace-Blackwell unified-memory architecture; GPU memory usage should be read via unified
  memory / system RAM accounting instead of the usual discrete-GPU `Memory-Usage` field.
  This is itself a Phase 4-relevant finding, not a tooling bug.
- `aarch64` architecture confirmed — this is the actual constraint behind the "expect a
  wheel/source-build issue" hypothesis in the plan's Phase 0, since many PyPI wheels
  (bitsandbytes, prebuilt torch CUDA wheels) are x86_64-only by default.

## Network note
`pip`/`curl` initially failed with `Name or service not known` — general internet routing
worked (`ping 8.8.8.8` succeeded) but DNS resolution via the local `systemd-resolved` stub
was broken (`resolvectl query` errored with "Cannot assign requested address" on the WiFi
link `wlP9s9`, whose IPv6 DNS server was likely unroutable). Fixed by explicitly pinning
that link's DNS servers:
```
sudo resolvectl dns wlP9s9 8.8.8.8 1.1.1.1
sudo resolvectl domain wlP9s9 "~."
```
Not a Python/package issue — worth flagging since it looked like one at first.

## Installed package versions (venv)
Installed 2026-08-17, in `~/construction_local_inference/venv`:
- `torch` 2.13.0+cu130 — **default PyPI wheel worked out of the box** on aarch64 + CUDA 13,
  no NVIDIA-specific index or source build needed. This contradicts the plan's Phase 0
  hypothesis of expected wheel friction — worth reporting as a "cleaner than expected" finding.
- `transformers` 5.15.0
- `peft` 0.20.0
- `accelerate` 1.14.0
- `bitsandbytes` 0.50.1 — imported cleanly, no CUDA-op warnings at import time (real
  int4/int8 kernel behavior still to be verified in Phase 3)
- `numpy` 2.5.2 (not a torch dependency by default — had to be installed explicitly,
  otherwise torch prints a "Failed to initialize NumPy" warning)
- `huggingface_hub` 1.27.0 — note: `huggingface-cli` is deprecated in this version;
  use `hf auth login` (not `huggingface-cli login`, not `hf-cli login`)

## Model access
Both target repos confirmed reachable via `HfApi().model_info()` after `hf auth login`:
- `mistralai/Mistral-7B-Instruct-v0.3` — OK, 15 files
- `Govardhan12345/legal-risk-classifier-lora` — OK, 8 files

## Base model load + forward pass (Phase 0 validation)
Script: `scripts/phase0_forward_pass.py`. Loaded `mistralai/Mistral-7B-Instruct-v0.3` in
FP16 via `device_map="auto"`.
- Model load time: 189.5s (includes ~14.5GB weight download + reconstruction from cache
  format; first-run cost only)
- Forward pass: OK, logits shape `[1, 32, 32768]`, 1062.4ms (first call — includes CUDA
  context/kernel warmup, not a steady-state number; Phase 2 harness will isolate warmup)
- GPU memory allocated: 14.54 GB
- GPU memory reserved: 14.56 GB
- Matches the plan's expected ~14-16GB FP16 footprint hypothesis.

Phase 0 status: **complete**.
