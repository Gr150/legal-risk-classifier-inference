"""
Phase 4 — Unified memory & hardware utilisation investigation.

Runs a fixed generate() call while sampling `nvidia-smi` GPU/memory utilization in a
background thread, for both BF16 and INT4 precisions, to determine whether decode is
compute-bound or bandwidth-bound. Also computes achieved effective memory bandwidth
from (approximate weight bytes read per token) / (measured time per token), as a cross-
check against Phase 3's natural experiment (INT4 cut weight bytes ~75%, latency dropped
~61% — the near-proportional relationship is itself evidence of a bandwidth-bound
regime, since a compute-bound workload would not show that relationship).
"""
import csv
import json
import subprocess
import threading
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from bench_common import BASE_MODEL_ID, ADAPTER_ID, WORKLOAD, MAX_NEW_TOKENS, TEMPERATURE, SEED, build_prompt

NVIDIA_SMI_FIELDS = "utilization.gpu,utilization.memory,power.draw,clocks.current.sm,clocks.current.memory,memory.used"
SAMPLE_INTERVAL_MS = 100


class GpuSampler:
    def __init__(self):
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", f"--query-gpu={NVIDIA_SMI_FIELDS}",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2,
                )
                row = next(csv.reader([out.stdout.strip()]))
                self.samples.append({
                    "t": time.perf_counter(),
                    "gpu_util_pct": row[0].strip(),
                    "mem_util_pct": row[1].strip(),
                    "power_w": row[2].strip(),
                    "sm_clock_mhz": row[3].strip(),
                    "mem_clock_mhz": row[4].strip(),
                    "mem_used_mib": row[5].strip(),
                })
            except Exception as e:
                self.samples.append({"t": time.perf_counter(), "error": str(e)})
            time.sleep(SAMPLE_INTERVAL_MS / 1000)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)


def summarize_samples(samples):
    def to_float(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    gpu_utils = [to_float(s.get("gpu_util_pct")) for s in samples if "error" not in s]
    mem_utils = [to_float(s.get("mem_util_pct")) for s in samples if "error" not in s]
    powers = [to_float(s.get("power_w")) for s in samples if "error" not in s]
    gpu_utils = [v for v in gpu_utils if v is not None]
    mem_utils = [v for v in mem_utils if v is not None]
    powers = [v for v in powers if v is not None]

    def stats(vals):
        if not vals:
            return {"mean": None, "max": None, "n": 0}
        return {"mean": round(sum(vals) / len(vals), 2), "max": max(vals), "n": len(vals)}

    return {
        "n_samples": len(samples),
        "n_errors": sum(1 for s in samples if "error" in s),
        "gpu_util_pct": stats(gpu_utils),
        "mem_util_pct": stats(mem_utils),
        "power_w": stats(powers),
        "raw_first_sample": samples[0] if samples else None,
    }


def profile_variant(model, tok, variant_name, weight_bytes_approx):
    prompt = build_prompt(WORKLOAD[0]["text"])
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    # untimed warmup
    with torch.no_grad():
        torch.manual_seed(SEED)
        model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                        temperature=TEMPERATURE, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()

    sampler = GpuSampler()
    sampler.start()
    torch.manual_seed(SEED)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                                  temperature=TEMPERATURE, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    total_time_s = time.perf_counter() - t0
    sampler.stop()

    n_new_tokens = int(out_ids.shape[1] - inputs["input_ids"].shape[1])
    time_per_token_s = total_time_s / n_new_tokens
    achieved_bandwidth_gb_s = (weight_bytes_approx / 1e9) / time_per_token_s

    return {
        "variant": variant_name,
        "total_time_s": round(total_time_s, 4),
        "n_new_tokens": n_new_tokens,
        "time_per_token_ms": round(time_per_token_s * 1000, 3),
        "weight_bytes_approx_gb": round(weight_bytes_approx / 1e9, 3),
        "achieved_effective_bandwidth_gb_s": round(achieved_bandwidth_gb_s, 2),
        "nvidia_smi_samples_during_generate": summarize_samples(sampler.samples),
    }


def main():
    results = {}

    print("=== BF16 profiling ===")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
    model.eval()
    # ~7.2B params, BF16 = 2 bytes/param, decode reads full weight set per token at batch=1
    bf16_weight_bytes = 7.2e9 * 2
    results["bf16"] = profile_variant(model, tok, "bf16", bf16_weight_bytes)
    print(json.dumps({k: v for k, v in results["bf16"].items() if k != "nvidia_smi_samples_during_generate"}, indent=2))
    del model, base_model, tok
    torch.cuda.empty_cache()

    print("\n=== INT4 profiling ===")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto")
    model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
    model.eval()
    # ~7.2B params, NF4 = 0.5 bytes/param (plus small double-quant scale overhead, ignored here)
    int4_weight_bytes = 7.2e9 * 0.5
    results["int4"] = profile_variant(model, tok, "int4_nf4", int4_weight_bytes)
    print(json.dumps({k: v for k, v in results["int4"].items() if k != "nvidia_smi_samples_during_generate"}, indent=2))
    del model, base_model, tok
    torch.cuda.empty_cache()

    with open("results/phase4_utilization.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results/phase4_utilization.json")


if __name__ == "__main__":
    main()
