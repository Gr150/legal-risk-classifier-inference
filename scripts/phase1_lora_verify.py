import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
ADAPTER_ID = "Govardhan12345/legal-risk-classifier-lora"

# Small held-out set of CUAD-style clause examples spanning common risk categories.
HELD_OUT_CLAUSES = [
    {
        "id": "termination_1",
        "category": "Termination",
        "text": "Either party may terminate this Agreement at any time without cause upon thirty (30) days written notice to the other party.",
    },
    {
        "id": "indemnification_1",
        "category": "Indemnification",
        "text": "Vendor shall indemnify, defend, and hold harmless Customer from and against any and all claims, damages, losses, and expenses, including reasonable attorneys' fees, arising out of or resulting from Vendor's breach of this Agreement, without any cap or limitation on liability.",
    },
    {
        "id": "noncompete_1",
        "category": "Non-Compete",
        "text": "During the term of this Agreement and for a period of five (5) years thereafter, Contractor shall not, directly or indirectly, engage in any business that competes with Company anywhere in the world.",
    },
    {
        "id": "liability_cap_1",
        "category": "Limitation of Liability",
        "text": "In no event shall either party's aggregate liability under this Agreement exceed the total fees paid in the twelve (12) months preceding the claim.",
    },
    {
        "id": "ip_assignment_1",
        "category": "IP Rights",
        "text": "All work product, inventions, and intellectual property created by Contractor in connection with the Services shall be the sole and exclusive property of Company, and Contractor hereby assigns all right, title, and interest therein to Company.",
    },
    {
        "id": "auto_renewal_1",
        "category": "Auto-Renewal",
        "text": "This Agreement shall automatically renew for successive one-year terms unless either party provides written notice of non-renewal at least ninety (90) days prior to the end of the then-current term.",
    },
]


def format_prompt(clause_text: str) -> str:
    # Matches the exact template documented in the adapter's model card
    # (Govardhan12345/legal-risk-classifier-lora "How to Use" section).
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


def gpu_mem_gb():
    return {
        "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 3),
    }


def main():
    results = {"idle_memory": None, "base_loaded_memory": None, "lora_loaded_memory": None, "outputs": []}

    torch.cuda.reset_peak_memory_stats()
    results["idle_memory"] = gpu_mem_gb()
    print("Idle GPU memory:", results["idle_memory"])

    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    print("Loading base model (fp16)...")
    t0 = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    base_load_time = time.time() - t0
    results["base_loaded_memory"] = gpu_mem_gb()
    results["base_load_time_s"] = round(base_load_time, 2)
    print(f"Base model loaded in {base_load_time:.1f}s, memory:", results["base_loaded_memory"])

    print("Loading LoRA adapter...")
    t0 = time.time()
    model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
    model.eval()
    lora_load_time = time.time() - t0
    results["lora_loaded_memory"] = gpu_mem_gb()
    results["lora_load_time_s"] = round(lora_load_time, 2)
    print(f"LoRA adapter loaded in {lora_load_time:.1f}s, memory:", results["lora_loaded_memory"])

    for clause in HELD_OUT_CLAUSES:
        prompt = format_prompt(clause["text"])
        inputs = tok(prompt, return_tensors="pt").to(model.device)

        t0 = time.time()
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.1,
                pad_token_id=tok.eos_token_id,
            )
        gen_time = time.time() - t0

        generated = tok.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        record = {
            "id": clause["id"],
            "expected_category": clause["category"],
            "clause_text": clause["text"],
            "model_output": generated.strip(),
            "gen_time_s": round(gen_time, 2),
        }
        results["outputs"].append(record)
        print(f"\n--- {clause['id']} ({clause['category']}) ---")
        print(f"Output: {generated.strip()}")
        print(f"Gen time: {gen_time:.2f}s")

    with open("results/phase1_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results to results/phase1_results.json")


if __name__ == "__main__":
    main()
