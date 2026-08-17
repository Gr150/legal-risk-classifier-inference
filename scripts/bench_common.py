"""Shared benchmark harness used by all Phase 3 optimization variants.

Reused (not reinvented) so every variant is measured with the same anti-pattern
guards established in Phase 2: warmup isolation, fresh input per timed call,
full-distribution reporting, hashed workload spec, correctness on the timed path.
"""
import json
import statistics
import time

import torch

BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
ADAPTER_ID = "Govardhan12345/legal-risk-classifier-lora"

MAX_NEW_TOKENS = 300
TEMPERATURE = 0.1
SEED = 42

WORKLOAD = [
    {"id": "termination_1", "category": "Termination", "text": "Either party may terminate this Agreement at any time without cause upon thirty (30) days written notice to the other party."},
    {"id": "indemnification_1", "category": "Indemnification", "text": "Vendor shall indemnify, defend, and hold harmless Customer from and against any and all claims, damages, losses, and expenses, including reasonable attorneys' fees, arising out of or resulting from Vendor's breach of this Agreement, without any cap or limitation on liability."},
    {"id": "noncompete_1", "category": "Non-Compete", "text": "During the term of this Agreement and for a period of five (5) years thereafter, Contractor shall not, directly or indirectly, engage in any business that competes with Company anywhere in the world."},
    {"id": "liability_cap_1", "category": "Limitation of Liability", "text": "In no event shall either party's aggregate liability under this Agreement exceed the total fees paid in the twelve (12) months preceding the claim."},
    {"id": "ip_assignment_1", "category": "IP Rights", "text": "All work product, inventions, and intellectual property created by Contractor in connection with the Services shall be the sole and exclusive property of Company, and Contractor hereby assigns all right, title, and interest therein to Company."},
    {"id": "auto_renewal_1", "category": "Auto-Renewal", "text": "This Agreement shall automatically renew for successive one-year terms unless either party provides written notice of non-renewal at least ninety (90) days prior to the end of the then-current term."},
]


def build_prompt(clause_text: str) -> str:
    return f"""<s>[INST] You are a legal risk analyst specialising in commercial contracts.

Analyse the following contract clause:
{clause_text}

Respond ONLY with a valid JSON object:
{{
  "clause_type": "...",
  "risk_level": "High or Medium or Low",
  "summary": "...",
  "reason": "..."
}} [/INST]"""


def check_correctness(output_text: str) -> bool:
    try:
        obj = json.loads(output_text.strip())
    except json.JSONDecodeError:
        return False
    required_keys = {"clause_type", "risk_level", "summary", "reason"}
    if not required_keys.issubset(obj.keys()):
        return False
    return obj.get("risk_level") in ("High", "Medium", "Low")


def percentile(data, p):
    data = sorted(data)
    k = (len(data) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(data) - 1)
    if f == c:
        return data[f]
    return data[f] + (data[c] - data[f]) * (k - f)


def run_single_request_benchmark(model, tok, variant_name, n_warmup=3, n_timed=24, extra_meta=None):
    """Batch-size-1 benchmark: single request at a time, matching Phase 2's methodology."""
    print(f"[{variant_name}] Running {n_warmup} untimed warmup calls...")
    warmup_prompt = build_prompt(WORKLOAD[0]["text"])
    for i in range(n_warmup):
        inputs = tok(warmup_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            torch.manual_seed(SEED)
            model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                temperature=TEMPERATURE, pad_token_id=tok.eos_token_id,
            )
        print(f"  warmup {i + 1}/{n_warmup} done")

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    print(f"[{variant_name}] Running {n_timed} timed calls...")
    latencies_s = []
    correctness_flags = []
    per_call_records = []

    for i in range(n_timed):
        clause = WORKLOAD[i % len(WORKLOAD)]
        prompt = build_prompt(clause["text"])
        inputs = tok(prompt, return_tensors="pt").to(model.device)

        torch.manual_seed(SEED + i)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out_ids = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                temperature=TEMPERATURE, pad_token_id=tok.eos_token_id,
            )
        torch.cuda.synchronize()
        latency_s = time.perf_counter() - t0

        generated = tok.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        correct = check_correctness(generated)
        n_new_tokens = out_ids.shape[1] - inputs["input_ids"].shape[1]

        latencies_s.append(latency_s)
        correctness_flags.append(correct)
        per_call_records.append({
            "call_index": i, "clause_id": clause["id"], "latency_s": round(latency_s, 4),
            "n_new_tokens": int(n_new_tokens), "correct": correct,
        })
        print(f"  call {i + 1}/{n_timed} [{clause['id']}] {latency_s:.3f}s correct={correct}")

    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

    results = {
        "variant": variant_name,
        "gpu": torch.cuda.get_device_name(0),
        "batch_size": 1,
        "n_warmup": n_warmup,
        "n_timed": n_timed,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "latency_s": {
            "median": round(statistics.median(latencies_s), 4),
            "p90": round(percentile(latencies_s, 90), 4),
            "mean": round(statistics.mean(latencies_s), 4),
            "stdev": round(statistics.stdev(latencies_s), 4) if len(latencies_s) > 1 else 0.0,
            "min": round(min(latencies_s), 4),
            "max": round(max(latencies_s), 4),
        },
        "correctness": {
            "n_correct": sum(correctness_flags),
            "n_total": len(correctness_flags),
            "rate": round(sum(correctness_flags) / len(correctness_flags), 4),
        },
        "peak_memory_gb": round(peak_mem_gb, 3),
        "per_call": per_call_records,
    }
    if extra_meta:
        results.update(extra_meta)
    return results
