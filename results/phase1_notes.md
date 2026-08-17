# Phase 1 — Load and Verify the Real Model

Run: 2026-08-17, on DGX Spark (`greentrend-spark`), via
`scripts/phase1_lora_verify.py`, raw output in `phase1_results.json`.

## Memory footprint
| State | Allocated | Reserved |
|---|---|---|
| Idle (before load) | 0.0 GB | 0.0 GB |
| Base model loaded (BF16) | 14.496 GB | 14.498 GB |
| + LoRA adapter loaded | 14.838 GB | 15.194 GB |

LoRA adapter adds ~0.34–0.7GB — small relative to the base model, consistent with the
adapter's documented ~150MB weight size plus generation-time overhead. Well inside the
128GB unified memory budget on this hardware, matching the plan's Phase 1 hypothesis.

Load times: base model 106.98s (from local HF cache, not counting the original download),
LoRA adapter 1.68s.

## Prompt-format correction (real finding, not a bug)
The first run used a generic instruction prompt (no `[INST]` wrapper, FP16, greedy
decoding, 120 max tokens) and produced malformed/truncated JSON — the model instead
echoed its own trained instruction template back before running out of tokens. Checking
the adapter's model card ("How to Use" section) showed it was trained with:
- Mistral's `<s>[INST] ... [/INST]` chat template
- **BF16** precision (not FP16)
- `temperature=0.1`, `max_new_tokens=300`

After matching that exact template, JSON output was well-formed and complete on all 6
held-out clauses (6/6 = 100%, matching the model card's claimed 100% JSON parse success
rate on its own held-out test set).

This is the kind of load-path issue Phase 1 exists to catch — logged here rather than
silently fixed, since it would have invalidated every later benchmark if it had gone
unnoticed.

## Sanity-check on outputs (6 held-out clauses)
| Clause | Expected category | Model `clause_type` | Model `risk_level` | Assessment |
|---|---|---|---|---|
| Termination for convenience | Termination | Termination For Convenience | High | Correct — matches model card's High-risk table |
| Indemnification (no liability cap) | Indemnification | **Cap On Liability** | Medium | **Misclassified** — see below |
| Non-compete (5yr, worldwide) | Non-Compete | Non-Compete | High | Correct |
| Liability cap (12mo fees) | Limitation of Liability | Cap On Liability | Medium | Correct |
| IP assignment | IP Rights | Ip Ownership Assignment | High | Correct |
| Auto-renewal (90-day notice) | Auto-Renewal | Notice Period To Terminate Renewal | Low | Correct |

**5/6 (83%) correctly categorized and risk-scored**, consistent with the model card's
overall 88.99% held-out accuracy.

**1/6 misclassification, and it's informative**: the indemnification clause explicitly
states liability is "without any cap or limitation on liability" (i.e. unlimited,
should be High risk per the model's own training table). The model instead classified
it as "Cap On Liability" / Medium — it appears to have keyed on the surface phrase "cap
or limitation on liability" rather than parsing the negation ("without any"). This
matches the model card's own disclosed limitation (High Risk recall is the weakest
category at 67.95%, vs. ~90%+ for Medium/Low) — the case study reproduced a real,
documented weakness rather than inventing a new one.

## Conclusion
Model + adapter load correctly, memory footprint matches hypothesis, and output quality
is broadly consistent with the adapter's published evaluation numbers. No load-path bug
found — the earlier bad output was a prompt-format mismatch on our side, now fixed and
matched to the adapter's documented template. Safe to proceed to Phase 2 (benchmark
harness) using this corrected prompt/precision configuration as the baseline.

Phase 1 status: **complete**.
