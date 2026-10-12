# LLM Cost Autopilot — Case Study

## Problem

Every production LLM application faces the same hidden tax: a default model
is chosen upfront ("we'll use GPT-4 for everything"), and that choice is never
revisited. The result is that simple factual queries — "What is the capital of
France?" — cost the same as complex multi-step reasoning tasks, and the bill
quietly grows.

For teams running thousands of requests per day, 60–80% of those requests are
simple enough to be handled adequately by a model that costs 50–100x less than
the default.

---

## Approach

Rather than picking a different default model, LLM Cost Autopilot builds an
**intelligent routing layer** that lives in front of your existing LLM calls.

The key insight is that prompt complexity is predictable from the prompt itself,
before any model is called. A lightweight text classifier can sort prompts into
complexity tiers in < 5ms — fast enough to add no meaningful latency.

The system works in four phases:

1. **Classify** — A RandomForest classifier scores each prompt as simple,
   moderate, or complex using 14 text features (length, keyword signals,
   question type, code/URL detection).
2. **Route** — A configurable tier-to-model map sends simple prompts to free
   local Ollama models, moderate prompts to fast cheap cloud models (Groq),
   and complex prompts to the best available model.
3. **Verify** — After returning the answer to the caller, the system
   asynchronously re-runs the same prompt on a high-quality reference model
   and uses an LLM-as-judge to score the cheap model's response. Failures
   are escalated (within a 5-second window) and logged for retraining.
4. **Log and visualise** — Every request, its cost, tier, model, and
   verification result is logged to Postgres and displayed in a real-time
   Streamlit dashboard.

---

## Architecture

```
Client
  └──► POST /v1/completions (FastAPI)
         ├── classify_and_route()      ← Phase 2: ~2ms
         ├── send_request()            ← Phase 1: provider call
         └── Return response immediately
               ↓ (BackgroundTask — non-blocking)
         ├── log_request()             ← Phase 4: Postgres write
         └── run_verification_loop()   ← Phase 3: ref model + judge
               ├── save_escalation_outcome()
               └── log_verification()

Postgres ←── request_log, verification_log
  └──► Streamlit dashboard (/v1/stats, GET /health)
```

---

## Results

> **Note:** The numbers below are placeholders. Run the load test with real
> API keys to populate this table:
>
> ```bash
> python scripts/load_test.py --mode both --requests 500
> ```
>
> Then query the DB for quality metrics:
> ```bash
> python -c "from db.queries import get_quality_metrics; print(get_quality_metrics(days=365))"
> ```

| Metric | Smart Routing | Baseline (All-Premium) |
|---|---|---|
| Total cost (500 req) | `[load test]` | `[load test]` |
| Gross savings | `[load test]` | — |
| Net savings (after verification) | `[load test]` | — |
| Avg latency | `[load test]` | `[load test]` |
| p95 latency | `[load test]` | `[load test]` |
| Avg quality score | `[DB query]` | — |
| Escalation rate | `[DB query]` | — |
| Routing: simple tier | `[load test]` | 0% |
| Routing: moderate tier | `[load test]` | 0% |
| Routing: complex tier | `[load test]` | 100% |

---

## What I'd Improve Next

### Caching
A semantic cache (embedding-based deduplication) would eliminate redundant
calls for near-identical prompts entirely. For many enterprise use cases,
20–40% of prompts are near-duplicates. Redis + a vector index (FAISS or
pgvector) could make this a 10ms operation.

### Sampled verification
Running Phase 3 verification on 100% of requests doubles the reference model
cost. Sampling verification at 10–20% of requests (stratified by tier) would
reduce verification cost by 80–90% while still catching systematic routing
failures quickly.

### PII scrubbing
Before sending prompts to cloud providers, a PII detection pass (Presidio or
a lightweight regex + NER model) would reduce compliance risk. Local Ollama
models can handle PII-sensitive prompts entirely.

### Provider pricing refresh
Model pricing changes frequently. The registry hardcodes prices; an automated
weekly job that scrapes provider pricing pages and diffs against the registry
would keep cost estimates accurate.

### Streaming support
The current API returns full responses. Adding streaming (`text/event-stream`)
would improve perceived latency for long generations, especially for complex-tier
prompts routed to slower high-quality models.

### Classifier confidence threshold
The classifier currently always picks a tier. Adding a confidence threshold
(e.g. if the model is < 60% confident, route to the next tier up) would
reduce mis-classification risk at the cost of slightly higher average spend.

---

*Built as part of a portfolio project demonstrating production-grade LLM
engineering: async Python, multi-provider abstraction, ML-based routing,
quality verification loops, and real-time cost observability.*
