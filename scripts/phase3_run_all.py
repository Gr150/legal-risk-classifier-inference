"""
Phase 3 — run all optimization variants in one pass: INT4, FP8, NVFP4,
torch.compile, and the batch-size sweep.

Each variant is measured independently (its own model load, its own results file)
so a failure in one (e.g. NVFP4 tooling not being ready) doesn't block the others,
and each result stands on its own as a Path/Baseline/Cost-model/Evidence artifact.
Between variants the model is deleted and torch.cuda.empty_cache() is called to
avoid one variant's memory footprint contaminating the next.
"""
import gc
import json
import os
import statistics
import time
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from bench_common import (
    BASE_MODEL_ID, ADAPTER_ID, WORKLOAD, MAX_NEW_TOKENS, TEMPERATURE, SEED,
    build_prompt, check_correctness, percentile, run_single_request_benchmark,
)

os.makedirs("results", exist_ok=True)


def cleanup(*objs):
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def save(name, results):
    path = f"results/{name}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[saved] {path}")


def run_int4():
    print("\n" + "=" * 60 + "\n[INT4] starting\n" + "=" * 60)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
    model.eval()
    results = run_single_request_benchmark(
        model, tok, variant_name="int4_nf4_bnb",
        extra_meta={"precision": "int4_nf4", "quant_lib": "bitsandbytes", "status": "ok"},
    )
    save("int4_results", results)
    cleanup(model, base_model, tok)
    return results


def run_fp8():
    print("\n" + "=" * 60 + "\n[FP8] starting\n" + "=" * 60)
    try:
        from torchao.quantization import quantize_
        from torchao.quantization.quant_api import Float8DynamicActivationFloat8WeightConfig
    except ImportError as e:
        result = {
            "variant": "fp8_torchao", "status": "failed",
            "stage": "import", "error": f"{type(e).__name__}: {e}",
            "note": "torchao not installed or Float8 quantization API unavailable in this build.",
        }
        save("fp8_results", result)
        return result

    try:
        tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID, dtype=torch.bfloat16, device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
        model.eval()
        quantize_(model, Float8DynamicActivationFloat8WeightConfig())
        results = run_single_request_benchmark(
            model, tok, variant_name="fp8_torchao",
            extra_meta={"precision": "fp8", "quant_lib": "torchao", "status": "ok"},
        )
        save("fp8_results", results)
        cleanup(model, base_model, tok)
        return results
    except Exception as e:
        result = {
            "variant": "fp8_torchao", "status": "failed",
            "stage": "quantize_or_generate",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }
        save("fp8_results", result)
        cleanup(*[o for o in [locals().get("model"), locals().get("base_model"), locals().get("tok")] if o is not None])
        return result


def run_nvfp4():
    print("\n" + "=" * 60 + "\n[NVFP4] starting\n" + "=" * 60)
    try:
        from torchao.prototype.mx_formats import NVFP4InferenceConfig
        from torchao.quantization import quantize_
    except ImportError as e:
        result = {
            "variant": "nvfp4_torchao", "status": "failed",
            "stage": "import", "error": f"{type(e).__name__}: {e}",
            "note": "NVFP4 tooling not available in this torchao build on this aarch64+Blackwell stack.",
        }
        save("nvfp4_results", result)
        return result

    try:
        tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID, dtype=torch.bfloat16, device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
        model.eval()
        quantize_(model, NVFP4InferenceConfig())
        results = run_single_request_benchmark(
            model, tok, variant_name="nvfp4_torchao",
            extra_meta={"precision": "nvfp4", "quant_lib": "torchao", "status": "ok"},
        )
        save("nvfp4_results", results)
        cleanup(model, base_model, tok)
        return results
    except Exception as e:
        result = {
            "variant": "nvfp4_torchao", "status": "failed",
            "stage": "quantize_or_generate",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }
        save("nvfp4_results", result)
        cleanup(*[o for o in [locals().get("model"), locals().get("base_model"), locals().get("tok")] if o is not None])
        return result


def run_torch_compile():
    print("\n" + "=" * 60 + "\n[torch.compile] starting (on top of BF16, the best mature path)\n" + "=" * 60)
    try:
        tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID, dtype=torch.bfloat16, device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
        model.eval()
        model.forward = torch.compile(model.forward, mode="reduce-overhead")

        # Extra warmup calls beyond the standard 3: compilation happens here,
        # NOT inside the timed region (anti Setup Pre-computation).
        print("[torch.compile] extra warmup to trigger graph compilation...")
        warmup_prompt = build_prompt(WORKLOAD[0]["text"])
        for i in range(2):
            inputs = tok(warmup_prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                    temperature=TEMPERATURE, pad_token_id=tok.eos_token_id,
                )
            print(f"  compile-warmup {i + 1}/2 done")

        results = run_single_request_benchmark(
            model, tok, variant_name="torch_compile_bf16",
            extra_meta={"precision": "bf16", "compiled": True, "status": "ok"},
        )
        save("torch_compile_results", results)
        cleanup(model, base_model, tok)
        return results
    except Exception as e:
        result = {
            "variant": "torch_compile_bf16", "status": "failed",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }
        save("torch_compile_results", result)
        cleanup(*[o for o in [locals().get("model"), locals().get("base_model"), locals().get("tok")] if o is not None])
        return result


def run_batching_sweep(batch_sizes=(1, 4, 8, 16), n_rounds=5, n_warmup_rounds=1):
    print("\n" + "=" * 60 + "\n[Batching sweep] starting (BF16)\n" + "=" * 60)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, dtype=torch.bfloat16, device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
    model.eval()

    sweep_results = {"variant": "batching_sweep_bf16", "status": "ok", "batch_sizes": {}}

    for bs in batch_sizes:
        print(f"\n[Batching sweep] batch_size={bs}")
        prompts = [build_prompt(WORKLOAD[i % len(WORKLOAD)]["text"]) for i in range(bs)]
        inputs = tok(prompts, return_tensors="pt", padding=True).to(model.device)

        for r in range(n_warmup_rounds):
            with torch.no_grad():
                torch.manual_seed(SEED)
                model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                    temperature=TEMPERATURE, pad_token_id=tok.eos_token_id,
                )
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        batch_latencies = []
        correctness_flags = []
        for r in range(n_rounds):
            torch.manual_seed(SEED + r)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                    temperature=TEMPERATURE, pad_token_id=tok.eos_token_id,
                )
            torch.cuda.synchronize()
            batch_latency_s = time.perf_counter() - t0
            batch_latencies.append(batch_latency_s)

            gen_only = out_ids[:, inputs["input_ids"].shape[1]:]
            for row in gen_only:
                text = tok.decode(row, skip_special_tokens=True)
                correctness_flags.append(check_correctness(text))
            print(f"  round {r + 1}/{n_rounds}: batch_latency={batch_latency_s:.3f}s "
                  f"throughput={bs / batch_latency_s:.2f} req/s")

        peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
        sweep_results["batch_sizes"][str(bs)] = {
            "batch_latency_s": {
                "median": round(statistics.median(batch_latencies), 4),
                "mean": round(statistics.mean(batch_latencies), 4),
                "stdev": round(statistics.stdev(batch_latencies), 4) if len(batch_latencies) > 1 else 0.0,
            },
            "throughput_req_per_s": round(bs / statistics.median(batch_latencies), 4),
            "per_request_latency_s_equiv": round(statistics.median(batch_latencies), 4),
            "peak_memory_gb": round(peak_mem_gb, 3),
            "correctness_rate": round(sum(correctness_flags) / len(correctness_flags), 4),
            "n_rounds": n_rounds,
        }

    save("batching_sweep_results", sweep_results)
    cleanup(model, base_model, tok)
    return sweep_results


def main():
    all_results = {}
    for name, fn in [
        ("int4", run_int4),
        ("fp8", run_fp8),
        ("nvfp4", run_nvfp4),
        ("torch_compile", run_torch_compile),
        ("batching_sweep", run_batching_sweep),
    ]:
        try:
            all_results[name] = fn()
        except Exception as e:
            print(f"[{name}] UNCAUGHT FAILURE: {e}")
            traceback.print_exc()
            all_results[name] = {"status": "failed", "error": str(e)}
            gc.collect()
            torch.cuda.empty_cache()

    print("\n" + "=" * 60 + "\nPhase 3 — ALL VARIANTS DONE\n" + "=" * 60)
    for name, r in all_results.items():
        status = r.get("status", "unknown")
        if status == "ok" and "latency_s" in r:
            print(f"{name}: OK, median={r['latency_s']['median']}s, correctness={r['correctness']['rate']}")
        elif status == "ok" and "batch_sizes" in r:
            print(f"{name}: OK, see results/batching_sweep_results.json")
        else:
            print(f"{name}: {status} — {r.get('error', r.get('note', ''))}")


if __name__ == "__main__":
    main()
