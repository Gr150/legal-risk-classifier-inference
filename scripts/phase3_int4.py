"""Phase 3 — INT4 quantization (bitsandbytes NF4), measured against the same
batch-1 harness as the Phase 2 BF16 baseline for an apples-to-apples comparison."""
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from bench_common import BASE_MODEL_ID, ADAPTER_ID, run_single_request_benchmark


def main():
    print("Loading tokenizer + INT4 (NF4) base model + LoRA adapter...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
    model.eval()

    results = run_single_request_benchmark(
        model, tok, variant_name="int4_nf4_bnb",
        extra_meta={"precision": "int4_nf4", "quant_lib": "bitsandbytes"},
    )

    with open("results/int4_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Summary ===")
    print(json.dumps({k: v for k, v in results.items() if k != "per_call"}, indent=2))


if __name__ == "__main__":
    main()
