"""
Phase 5 — Serving layer.

Thin FastAPI wrapper exposing POST /classify in front of a vLLM OpenAI-compatible
server (started separately via `vllm serve ... --enable-lora`). Formats the clause
into the same [INST] prompt template verified in Phases 1-4, calls vLLM's
/v1/completions endpoint with logprobs enabled, parses the model's JSON output, and
returns clause_type/risk_level/summary/reason plus an approximate confidence score
derived from the mean token logprob (NOT a calibrated probability -- documented as
an approximation).

Run vLLM first (separate process):
  vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
    --enable-lora --lora-modules legal-risk=Govardhan12345/legal-risk-classifier-lora \
    --max-lora-rank 32 --dtype bfloat16 --port 8000

Then run this wrapper:
  uvicorn serve_app:app --port 8080
"""
import json
import math
import os
import time

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
LORA_NAME = os.environ.get("LORA_NAME", "legal-risk")
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.1

app = FastAPI(title="legal-risk-classifier-serving")


class ClassifyRequest(BaseModel):
    clause: str


class ClassifyResponse(BaseModel):
    clause_type: str
    risk_level: str
    summary: str
    reason: str
    confidence_approx: float
    latency_s: float


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


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    prompt = build_prompt(req.clause)

    payload = {
        "model": LORA_NAME,
        "prompt": prompt,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "logprobs": 1,
    }

    t0 = time.perf_counter()
    resp = requests.post(f"{VLLM_BASE_URL}/v1/completions", json=payload, timeout=60.0)
    latency_s = time.perf_counter() - t0

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"vLLM backend error: {resp.text}")

    data = resp.json()
    choice = data["choices"][0]
    text = choice["text"]

    logprobs_list = (choice.get("logprobs") or {}).get("token_logprobs") or []
    logprobs_list = [lp for lp in logprobs_list if lp is not None]
    mean_logprob = sum(logprobs_list) / len(logprobs_list) if logprobs_list else None
    confidence_approx = round(math.exp(mean_logprob), 4) if mean_logprob is not None else 0.0

    try:
        obj = json.loads(text.strip())
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail=f"Model did not return valid JSON: {text[:300]}")

    required = {"clause_type", "risk_level", "summary", "reason"}
    if not required.issubset(obj.keys()):
        raise HTTPException(status_code=502, detail=f"Missing required keys in model output: {obj}")

    return ClassifyResponse(
        clause_type=obj["clause_type"],
        risk_level=obj["risk_level"],
        summary=obj["summary"],
        reason=obj["reason"],
        confidence_approx=confidence_approx,
        latency_s=round(latency_s, 4),
    )
