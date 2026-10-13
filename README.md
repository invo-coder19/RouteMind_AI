<div align="center">

# ⚡ RouteMind AI — LLM Cost Autopilot

**Stop paying frontier model prices for questions that don't need them.**

RouteMind AI is a production-grade intelligent routing layer that sits in front of any LLM stack.
It classifies every incoming prompt by complexity, dispatches it to the cheapest model that can
handle it well, then quietly double-checks the answer in the background — escalating and
self-correcting when cheaper models underperform.

One HTTP call. Automatic model selection. Real-time cost observability.

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-316192?logo=postgresql&logoColor=white)](https://postgresql.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://docker.com)

</div>

---

## The Problem

Every production LLM deployment hides the same quiet cost leak: a single default model handles
every request — from *"What is 2 + 2?"* to *"Design a distributed rate-limiter at 100k RPS."*
Both get billed the same frontier-model rate.

In practice, **60–80% of real traffic is simple enough** for a model that costs 50–100× less.
The rest genuinely needs the expensive one. But nobody has time to sort requests by hand.

RouteMind AI sorts them automatically, at < 5ms overhead.

---

## How It Works

```
                        ┌─────────────────────────────────────────────────────┐
                        │              RouteMind AI  (api/main.py)             │
                        │                                                       │
  Your App              │  ┌──────────────┐    ┌───────────────────────────┐  │
  ──────────            │  │  Complexity  │    │      Router Client        │  │
  POST /v1/  ──────────►│  │  Classifier  │───►│  send_request()           │  │
  completions           │  │ (Phase 2 ML) │    │  Phase 1 — 5 providers    │  │
                        │  └──────────────┘    └─────────┬─────────────────┘  │
                        │                                 │ LLMResponse        │
  ◄─────────────────────│◄────────────────────────────────┘  (immediate)       │
  Instant response      │                                                       │
                        │  ┌─────────────────────────────────────────────────┐ │
                        │  │   Background  (non-blocking asyncio.create_task) │ │
                        │  │                                                  │ │
                        │  │  Phase 3 Verification Loop                      │ │
                        │  │   ├── Re-run on reference (high-tier) model      │ │
                        │  │   ├── LLM-as-judge: quality score 0.0–1.0        │ │
                        │  │   ├── Escalation decision (5 s window)           │ │
                        │  │   └── Write to PostgreSQL                        │ │
                        │  │                                                  │ │
                        │  │  Phase 4 Logging                                 │ │
                        │  │   └── request_log + verification_log             │ │
                        │  └─────────────────────────────────────────────────┘ │
                        └─────────────────────────────────────────────────────┘
                                               │
                                    ┌──────────▼──────────┐
                                    │  Streamlit Dashboard │
                                    │  Cost savings · Quality trend
                                    │  Routing distribution · Escalations
                                    └─────────────────────┘
```

### The routing tiers

| Tier | Prompt type | Default model | Cost |
|---|---|---|---|
| **Simple** | Factual lookups, one-step answers | Ollama `llama3.2:3b` (local) | **$0.00** |
| **Moderate** | Reasoning, code generation, summaries | Groq `llama-3.1-8b-instant` | **~$0.00006 / 1K tok** |
| **Complex** | Multi-step analysis, architecture, research | Gemini `gemini-2.0-flash` | **~$0.00030 / 1K tok** |

Tiers and models are **fully configurable** — change them live via `PUT /v1/routing-config` without restarting anything.

---

## Feature Highlights

| Feature | Details |
|---|---|
| 🧠 **ML classifier** | RandomForest on 14 text features — classifies in < 5ms |
| 🔀 **5-provider support** | OpenRouter, Groq, Gemini, Mistral, Ollama (local) |
| 🔁 **Auto-fallback** | Primary model fails → automatically escalates to next tier |
| 🔍 **Async quality verification** | LLM-as-judge scoring, never blocks the response |
| ⚡ **Escalation in 5 seconds** | If quality fails within window, reference answer is delivered |
| 📚 **Self-retraining** | Weekly job re-labels failures and retrains the classifier |
| 📊 **Live dashboard** | Streamlit: savings, quality trend, routing distribution, escalations |
| 🔧 **Hot-reload routing** | Update tier→model config via API, zero downtime |
| 🐳 **One-command deploy** | `docker compose up --build` |
| 🧪 **Mocked test suite** | 7 pytest tests, zero real API spend |

---

## Architecture Diagram

```mermaid
flowchart TD
    subgraph Client
        A[Your Application]
    end

    subgraph API["⚡ RouteMind API  (FastAPI)"]
        B[POST /v1/completions]
        C[Complexity Classifier\nPhase 2 — RandomForest]
        D[Router Client\nPhase 1 — send_request]
        E[Immediate Response]
        F[BackgroundTask]
    end

    subgraph Providers["LLM Providers"]
        G[🟢 Ollama — local free]
        H[🔵 Groq — fast cheap]
        I[🟣 Gemini / Mistral]
        J[⚪ OpenRouter]
    end

    subgraph Background["Phase 3 — Async Verification"]
        K[Reference Model Call]
        L[LLM-as-Judge Scoring]
        M[Escalation Decision]
    end

    subgraph Storage["Phase 4 — PostgreSQL"]
        N[(request_log)]
        O[(verification_log)]
    end

    subgraph Viz["Phase 4 — Dashboard"]
        P[Streamlit\nlocalhost:8501]
    end

    A -->|POST prompt| B
    B --> C
    C -->|tier: simple/moderate/complex| D
    D --> G & H & I & J
    G & H & I & J -->|LLMResponse| E
    E -->|instant| A
    B -.->|non-blocking| F
    F --> K --> L --> M
    F --> N
    M --> O
    N & O --> P
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- [Ollama](https://ollama.ai) installed and running (`ollama pull llama3.2:3b`)
- At least one cloud provider API key — [Groq](https://console.groq.com/keys) is free

### Step 1 — Clone & install

```bash
git clone https://github.com/invo-coder19/RouteMind_AI.git
cd RouteMind_AI

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

### Step 2 — Configure environment

Edit `.env` with your API keys:

```bash
# Cloud providers (at least one required)
OPENROUTER_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
MISTRAL_API_KEY=your_key_here

# Local Ollama
OLLAMA_BASE_URL=http://localhost:11434

# PostgreSQL (matches docker-compose defaults — no change needed)
DATABASE_URL=postgresql+asyncpg://routemind:routemind@localhost:5432/routemind

# Optional: protect write endpoints with an API key
# API_KEY=your_secret_here
```

### Step 3 — Train the classifier

```bash
python data/split_dataset.py     # stratified 80/20 split
python classifier/train.py       # trains RandomForest → saves model.pkl
```

Expected output: `Test accuracy: XX.X% ✓` (target: > 80%)

### Step 4 — Launch the full stack

```bash
docker compose up --build
```

| Service | URL | Description |
|---|---|---|
| **API** | `http://localhost:8000` | FastAPI — your single integration point |
| **Docs** | `http://localhost:8000/docs` | Interactive Swagger UI |
| **Dashboard** | `http://localhost:8501` | Streamlit cost & quality dashboard |
| **Postgres** | `localhost:5432` | Auto-initialised with schema |

> **Linux users:** Uncomment `extra_hosts: host.docker.internal:host-gateway` in
> `docker-compose.yml` so the API container can reach Ollama running on your host.

### Step 5 — Verify everything is healthy

```bash
curl http://localhost:8000/health
```
```json
{
  "status": "ok",
  "classifier_loaded": true,
  "db_connected": true
}
```

---

## API Reference

Base URL: `http://localhost:8000` · Swagger UI: `/docs` · ReDoc: `/redoc`

### `POST /v1/completions` — Main endpoint

Route any prompt to the cheapest capable model automatically.

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain the CAP theorem in simple terms."}'
```

**Request body:**

| Field | Type | Required | Description |
|---|---|---|---|
| `prompt` | `string` | ✅ | The user's prompt |
| `system_prompt` | `string` | ❌ | Optional system instruction |
| `max_tokens` | `int` | ❌ | Max tokens to generate (default: `1024`) |
| `force_model` | `string` | ❌ | Skip the classifier; use a specific `model_id` |

**Response:**

```json
{
  "output_text": "The CAP theorem states that a distributed system can guarantee...",
  "model_used": "llama-3.1-8b-instant",
  "provider": "groq",
  "complexity_tier": "moderate",
  "input_tokens": 14,
  "output_tokens": 89,
  "cost_usd": 0.0000063,
  "latency_ms": 418.2,
  "request_id": "b3f2e4a1-...",
  "fallback_used": false
}
```

The `X-Request-ID` response header carries the same UUID for end-to-end tracing.

---

### `GET /v1/models`

List every registered model with provider, pricing, and quality tier.

```bash
curl http://localhost:8000/v1/models
```

### `GET /v1/stats`

Live headline metrics — total savings, avg quality score, escalation rate.

```bash
curl http://localhost:8000/v1/stats
```

### `GET /v1/routing-config`

See the current tier → model mapping.

```bash
curl http://localhost:8000/v1/routing-config
```

### `PUT /v1/routing-config` *(requires `X-API-Key` if `API_KEY` is set)*

Update and hot-reload the routing config with zero downtime. Every `model_id`
is validated against the registry before the file is written — returns `422`
with a clear error if an unknown model is passed.

```bash
curl -X PUT http://localhost:8000/v1/routing-config \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_secret_here" \
  -d '{
    "routing": {
      "simple":   "llama3.2:3b",
      "moderate": "llama-3.1-8b-instant",
      "complex":  "gemini-2.0-flash"
    }
  }'
```

### `GET /health`

Liveness + readiness probe — used by Docker HEALTHCHECK and uptime monitors.

---

## Dashboard

Open **`http://localhost:8501`** after `docker compose up`.

| Section | What it shows |
|---|---|
| 💰 **Cost Savings** | Total $ saved vs. routing everything to the premium model |
| 📈 **Cost Over Time** | Daily actual spend vs. all-premium baseline |
| 🗂️ **Routing Distribution** | % of requests per complexity tier and per model |
| 🎯 **Quality Trend** | Avg judge score + escalation rate over time (dual axis) |
| 🚨 **Recent Escalations** | Real examples where the system caught and corrected itself |

All charts respect the time-window filter at the top. Data refreshes every 2 minutes.

---

## Load Testing

Run the built-in load test to generate the savings report:

```bash
# Both runs: smart routing vs. all-premium baseline (adjust --requests as needed)
python scripts/load_test.py --mode both --requests 500
```

Outputs `load_test_report.md` with:
- Total cost — smart routing vs. baseline
- Gross and net savings (net accounts for verification reference-model cost)
- Avg and p95 latency comparison
- Routing distribution by tier

---

## Running Tests

```bash
pytest tests/ -v
```

All 7 tests mock provider calls — **zero real API spend**. Covers:
successful routing, `force_model` override, invalid model IDs, provider failure
with automatic fallback, all-providers-fail 502, invalid routing-config update,
and the health endpoint.

---

## Project Structure

```
RouteMind_AI/
├── api/                  # Phase 5 — FastAPI application
│   ├── main.py           #   App factory, lifespan, middleware
│   ├── dependencies.py   #   Auth, request-ID, app-state injection
│   └── routes/           #   completions · health · models · stats · routing_config
│
├── classifier/           # Phase 2 — Complexity classifier (RandomForest)
├── config/               #   routing_map.yaml  (tier → model mapping)
├── core/                 # Phase 1 — Unified provider clients & exceptions
├── dashboard/            # Phase 4 — Streamlit cost & quality dashboard
├── data/                 #   Labeled prompts · train/test CSVs · split script
├── db/                   # Phase 4 — Schema · ORM models · logger · query layer
├── docs/                 #   case_study.md
├── models/               #   ModelConfig · MODEL_REGISTRY (5 providers, 10+ models)
├── scripts/              #   retrain_classifier.py · load_test.py
├── tests/                #   pytest suite (7 tests, fully mocked)
├── verification/         # Phase 3 — Async quality verification & feedback loop
│
├── Dockerfile            #   Multi-stage build — non-root user, no build tools in image
├── docker-compose.yml    #   db + api + dashboard — one command
├── pyproject.toml        #   pytest config
└── requirements.txt      #   Pinned deps for Python 3.11+
```

---

## Headline Result

> **Run `python scripts/load_test.py --mode both --requests 500` to populate these numbers with real data.**

| Metric | Smart Routing | Baseline (All-Premium) |
|---|---|---|
| Total cost | `[run load test]` | `[run load test]` |
| Gross savings | `[run load test]` | — |
| Net savings (after verification) | `[run load test]` | — |
| Avg latency | `[run load test]` | `[run load test]` |
| p95 latency | `[run load test]` | `[run load test]` |
| Avg quality score | `[run DB query]` | — |
| Escalation rate | `[run DB query]` | — |

---

## Contributors

| Name | Role |
|---|---|
| **Soham Deshpande** | Core Architecture & ML Pipeline |
| **Riya Gandhi** | API Design & Quality Verification System |
| **Avdhut Giri** | Dashboard, Deployment & Portfolio Polish |
