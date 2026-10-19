# Trustworth AI Quadrails

A production-ready input/output guardrail layer for the university **Trustworthy AI** assignment, built on top of the existing FastAPI + Qdrant + Redis RAG backend.
No API keys, no Docker, no network required — everything runs offline.

---

## Table of Contents

1. [Assignment Deliverables](#assignment-deliverables)
2. [Architecture](#architecture)
3. [What the Guardrails Do](#what-the-guardrails-do)
4. [Why Better than a Regex List](#why-better-than-a-regex-list)
5. [Evaluation Results (TEST split, honest numbers)](#evaluation-results)
6. [Files Added / Changed](#files-added--changed)
7. [Running Locally](#running-locally)
8. [Design Decisions and Limitations](#design-decisions-and-limitations)

---

## Assignment Deliverables

| Deliverable | Location |
|---|---|
| Input guardrails | `app/guardrails/input_pipeline.py` |
| Output guardrails | `app/guardrails/output_pipeline.py` |
| PII detection | `app/guardrails/pii.py` |
| Injection detection | `app/guardrails/injection.py` |
| Topic classification | `app/guardrails/topics.py` |
| Schema validation | `app/guardrails/schema_validator.py` |
| Groundedness checking | `app/guardrails/groundedness.py` |
| Output PII / canary | `app/guardrails/output_sensitive.py` |
| Spotlighting | `app/guardrails/spotlight.py` |
| Audit logging | `app/guardrails/audit.py` |
| Policy config | `app/guardrails/policy.toml` |
| Evaluation harness | `evaluation/harness.py` |
| Dev + test datasets | `evaluation/datasets.py` |
| Evaluation runner | `evaluation/run_eval.py` |
| Unit + integration tests | `tests/guardrails/` |
| Evaluation report | `docs/evaluation_report.md` |
| Architecture diagram | `docs/architecture.mmd` and [Architecture](#architecture) below |
| README (this file) | `README.md` |

---

## Architecture

```
User message
     │
     ▼
┌──────────────────────────────────┐
│        INPUT PIPELINE            │
│                                  │
│  1. Normalize                    │  ← Unicode NFKC, zero-width strip,
│     (normalization.py)           │    Devanagari digits, leet decode,
│                                  │    spaced-letter collapse, base64 decode
│  2. Length check                 │  ← block > 2000 chars
│                                  │
│  3. PII detection & masking      │  ← Luhn card, entropy secrets,
│     (pii.py)                     │    Nepal mobile/landline/citizenship/PAN,
│                                  │    email (incl. obfuscated), IP, JWT
│  4. Injection scoring            │  ← token-window noisy-OR, mention/quote
│     (injection.py)               │    discount, Redis session risk
│                                  │
│  5. Topic classification         │  ← self-harm, violence, illegal activity,
│     (topics.py)                  │    cyber abuse, hate, data exfiltration,
│                                  │    professional advice
│                                  │
│  Decision: ALLOW / MASK /        │
│  SAFE_RESPONSE / HARD_BLOCK /    │
│  ALLOW_WITH_DISCLAIMER           │
└──────────────────────────────────┘
     │ allowed (sanitized text)
     ▼
┌──────────────────────────────────┐
│        RAG PIPELINE              │
│                                  │
│  6. Intent classification        │  ← booking vs question
│                                  │
│  7. Spotlighting                 │  ← wrap chunks with per-request nonce;
│     (spotlight.py)               │    scan + drop poisoned chunks
│                                  │
│  8. Vector retrieval             │  ← Qdrant similarity search
│                                  │
│  9. LLM generation               │  ← OpenAI-compatible API
│                                  │
└──────────────────────────────────┘
     │ raw LLM output
     ▼
┌──────────────────────────────────┐
│       OUTPUT PIPELINE            │
│                                  │
│ 10. Schema validation            │  ← strict Pydantic v2, rejects
│     (schema_validator.py)        │    "false" string as bool,
│                                  │    drops unfaithful citations
│ 11. Groundedness check           │  ← IDF-weighted recall, number/date/
│     (groundedness.py)            │    amount canonicalization, negation
│                                  │    polarity, quantifier mismatch
│ 12. Output sensitive info        │  ← PII provenance (session_own /
│     (output_sensitive.py)        │    context_sourced / unknown),
│                                  │    canary token, n-gram system-prompt
│                                  │
│  Decision: PASS / REDACT /       │
│  REGENERATE / FALLBACK           │
└──────────────────────────────────┘
     │
     ▼
  ChatResponse  +  GuardrailsSummarySchema
     → Audit log (structured JSON, session hash, policy version)
```

---

## What the Guardrails Do

### Input Guardrails

| Check | Module | Action |
|---|---|---|
| Length > 2000 chars | `input_pipeline.py` | `HARD_BLOCK` |
| Unicode / leet / Devanagari normalization | `normalization.py` | Pre-processing |
| Email, phone, card (Luhn), citizenship, PAN, bank, IP, secret/JWT | `pii.py` | `MASK` |
| Injection scoring (token-window + session accumulation) | `injection.py` | `HARD_BLOCK` ≥ 0.75 / flag ≥ 0.45 |
| Self-harm | `topics.py` | `SAFE_RESPONSE` |
| Violence, illegal activity, hate, data exfiltration | `topics.py` | `HARD_BLOCK` |
| Medical / legal / financial advice | `topics.py` | `ALLOW_WITH_DISCLAIMER` |

### Output Guardrails

| Check | Module | Action |
|---|---|---|
| Strict JSON schema + bool("false") fix | `schema_validator.py` | Retry / `FALLBACK` |
| Unfaithful citations | `schema_validator.py` | Drop citation |
| Number/date/amount mismatch | `groundedness.py` | `REDACT` or `REGENERATE` |
| Negation polarity mismatch | `groundedness.py` | Score penalty |
| Unknown PII in output | `output_sensitive.py` | `REDACT` |
| Canary token leak | `output_sensitive.py` | `REDACT` |
| System-prompt n-gram overlap | `output_sensitive.py` | `REDACT` |

### Cross-cutting

- **Spotlighting** (Hines et al., 2024): each retrieved chunk is wrapped with a per-request hex nonce. Chunks containing injection patterns are dropped before reaching the LLM.
- **Audit logging**: every stage emits a structured JSON line with session hash (not raw ID), action, risk score, finding types, and policy version.
- **Monitor mode**: all checks run but nothing is blocked — for measuring false-positive rates.
- **Policy TOML**: all thresholds and messages live in `app/guardrails/policy.toml`. Tunable without code changes.

---

## Why Better than a Regex List

The naive baseline in our evaluation uses:
- **Injection**: 6 hard-coded keyword phrases
- **PII**: 1 email-only regex
- **Groundedness**: always True (no check)
- **Input action**: keyword→block, email→mask, otherwise allow

All numbers below come from `evaluation/run_eval.py --split test`. They are never fabricated.

---

## Evaluation Results

**TEST split** — held out, never used during tuning.

Reproduce with:
```bash
uv run python evaluation/run_eval.py --split test --out evaluation/results_test.json
```

### Input Action Classification (n=19)

| Metric | This system | Naive baseline | Delta |
|---|---|---|---|
| Accuracy | **78.9%** | 31.6% | +47.4 pp |
| Macro-F1 | **53.8%** | 14.4% | +39.4 pp |

### PII Detection (n=10)

| Metric | This system | Naive baseline | Delta |
|---|---|---|---|
| Precision | 85.7% | 100.0% | −14.3 pp |
| Recall | **85.7%** | 14.3% | +71.4 pp |
| **F1** | **85.7%** | 25.0% | **+60.7 pp** |

*Baseline precision is 100% only because it has 1 pattern that never fires falsely; but it misses 85.7% of PII (phone, card, ID, IP, secrets). Our system catches all these at the cost of a small precision drop.*

### Injection Detection (n=8, threshold=0.45)

| Metric | This system | Naive baseline | Delta |
|---|---|---|---|
| Precision | **100.0%** | 100.0% | 0 pp |
| Recall | **100.0%** | 25.0% | +75 pp |
| **F1** | **100.0%** | 40.0%| **+60 pp** |

*Baseline catches 1 in 4 injections. The system catches all 4, including romanized-Nepali and encoding tricks the keyword list misses.*

### Groundedness (n=5)

| Metric | This system | Naive baseline | Delta |
|---|---|---|---|
| Precision | **66.7%** | 40.0% | +26.7 pp |
| Recall | **100.0%** | 100.0% | 0 pp |
| **F1** | **80.0%** | 57.1% | **+22.9 pp** |

*The baseline (always-True) cannot detect any fabricated answer. Our system correctly catches wrong numbers and wrong dates.*

> **Honest note**: Input-action macro-F1 of 53.8% reflects real difficulty at the mask/allow boundary. The system consistently beats the baseline on every metric, but there is room to improve.

---


## Running Locally

```bash
# Install (Python 3.11+, uv)
uv sync

# Lint (all checks pass)
uv run ruff check .

# Type check
uv run mypy app

# Tests (109 pass, 0 fail)
uv run pytest tests/ -q

# Evaluation — dev split (for tuning, look but do not report)
uv run python evaluation/run_eval.py --split dev

# Evaluation — test split (for reporting)
uv run python evaluation/run_eval.py --split test --out evaluation/results_test.json
```

### Streamlit UI

The UI is a thin client for the existing FastAPI chat endpoint. Start the API,
install the optional UI dependency, and launch Streamlit:

```bash
uv sync --extra dev --extra ui
uv run streamlit run streamlit_app.py
```

Use the sidebar to set the API URL (default `http://localhost:8000`) and start
a conversation. The interface displays the answer, guardrail status and flags,
source references, and booking state returned by `/api/v1/chat`.

To run the API and UI together with Docker Compose:

```bash
docker compose up --build
```

The API is available at `http://localhost:8000` and the Streamlit UI at
`http://localhost:8501`. Inside Compose, the UI reaches the API through
`http://api:8000`.

## Deliverable Locations

| Deliverable | Current location |
|---|---|
| Source guardrails | `app/guardrails/` |
| Test suite | `tests/guardrails/` and `tests/` |
| Evaluation code and aggregate results | `evaluation/` |
| Evaluation report | `docs/evaluation_report.md` |
| Architecture diagram | `docs/architecture.mmd` |
| Architecture overview | README Architecture section |
| README | `README.md` |
| Optional UI | `streamlit_app.py` |

The evaluation results JSON contains per-case records as well as aggregate
metrics. The standalone report and Mermaid architecture source are linked above.

---

## Design Decisions and Limitations

### Key decisions

- **No external ML models**: all detection is regex, rule-based, IDF-weighted, entropy, and Luhn. Fully offline and auditable.
- **Noisy-OR injection scoring + session risk**: one ambiguous message scores low; a session with escalating attempts accumulates risk in Redis, catching multi-turn attacks.
- **Spotlighting with per-request nonces**: each chunk carries a unique nonce; the system prompt instructs the LLM to treat only nonce-bracketed text as data. Poisoned chunks are dropped before the LLM call.
- **Strict Pydantic v2 schemas**: `bool("false") == True` in Python — the LLM's string `"false"` had to be caught at schema level. This also fixed a real bug in booking extraction.
- **Dev/test split discipline**: test split was written before any threshold tuning. All reported numbers are from the test split only.

### Limitations (honest)

- **Lexical groundedness only**: no semantic similarity. High-overlap paraphrases may pass; low-overlap paraphrases may fail.
- **No multilingual NLP beyond Devanagari digits**: longer Nepali text is not handled.
- **Injection recall ceiling**: adversarial prompts designed specifically to avoid our window rules will not be detected.
- **PII precision**: generic IP-like strings and technical terms can produce false positives.
- **Single-doc IDF degeneracy**: with one retrieved chunk, IDF weights are uniform; discrimination between topics is weaker.
