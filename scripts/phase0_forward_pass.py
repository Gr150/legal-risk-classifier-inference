import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL_ID)

print("Loading model (fp16, device_map=auto)...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map="auto",
)
load_time = time.time() - t0
print(f"Model loaded in {load_time:.1f}s")

model.eval()

prompt = "Classify the risk level of this clause: 'Either party may terminate this agreement at any time without cause upon 30 days written notice.'"
inputs = tok(prompt, return_tensors="pt").to(model.device)

t0 = time.time()
with torch.no_grad():
    out = model(**inputs)
fwd_time = time.time() - t0

print(f"Forward pass OK. logits shape: {out.logits.shape}, time: {fwd_time*1000:.1f}ms")
print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"GPU memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
