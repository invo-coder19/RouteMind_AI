# LLM Cost Autopilot

> **Stop routing every request to your most expensive model.**  
> LLM Cost Autopilot is an intelligent routing layer that classifies each incoming prompt by complexity, dispatches it to the cheapest capable model, asynchronously verifies quality against a reference model, and self-corrects when cheaper models underperform — all behind a single OpenAI-compatible API endpoint.

---

## Architecture

```mermaid
flowchart LR
    Client -->|POST /v1/completions| API[FastAPI\napi/main.py]
    API --> Classifier[Phase 2\nComplexity Classifier\nRandomForest]
    Classifier -->|tier| Router[Phase 1\nRouter Client\nsend_request]
    Router -->|cheap model| LLM[(LLM Provider\nGroq / Ollama /\nGemini / Mistral)]
    LLM -->|LLMResponse| API
    API -->|immediate response| Client

    API -.->|BackgroundTask| VerifyLoop[Phase 3\nVerification Loop]
    VerifyLoop --> RefModel[(Reference Model\nhigh quality_tier)]
    VerifyLoop --> Judge[LLM-as-Judge\nquality score]
    VerifyLoop --> Escalation[Escalation\nDecision]
    VerifyLoop --> DB[(PostgreSQL\nrequest_log\nverification_log)]

    DB --> Dashboard[Phase 4\nStreamlit Dashboard\ndashboard/app.py]
```

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Docker & Docker Compose
- [Ollama](https://ollama.ai) running locally (`ollama pull llama3.2:3b`)
- API keys for at least one cloud provider (Groq is free)

### 2. Environment setup

```bash
# Clone and enter
git clone https://github.com/invo-coder19/RouteMind_AI.git
cd RouteMind_AI

# Create venv
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

### 3. Configure API keys

Copy `.env` and fill in your keys:

```bash
# .env
OPENROUTER_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
MISTRAL_API_KEY=your_key_here
OLLAMA_BASE_URL=http://localhost:11434
DATABASE_URL=postgresql+asyncpg://routemind:routemind@localhost:5432/routemind

# Optional: set this to require X-API-Key on write endpoints
# API_KEY=your_secret_key
```

### 4. Train the classifier (Phase 2)

```bash
python data/split_dataset.py
python classifier/train.py
```

### 5. Start the full stack

```bash
docker compose up --build
```

This starts:
- **Postgres** on port 5432 (schema applied automatically)
- **FastAPI API** on port 8000
- **Streamlit Dashboard** on port 8501

> **Linux users:** Uncomment `extra_hosts: host.docker.internal:host-gateway` in `docker-compose.yml` so the API container can reach Ollama on your host.

### 6. Verify

```bash
curl http://localhost:8000/health
# {"status":"ok","classifier_loaded":true,"db_connected":true}
```

---

## API Reference

Base URL: `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`

### `POST /v1/completions`

Route a prompt to the cheapest capable model.

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?"}'
```

**Request body:**

| Field | Type | Required | Description |
|---|---|---|---|
| `prompt` | string | ✅ | The user's prompt |
| `system_prompt` | string | ❌ | Optional system instruction |
| `max_tokens` | int | ❌ | Max tokens (default: 1024) |
| `force_model` | string | ❌ | Bypass classifier; use a specific model_id |

**Response:**

```json
{
  "output_text": "Paris.",
  "model_used": "llama3.2:3b",
  "provider": "ollama",
  "complexity_tier": "simple",
  "input_tokens": 8,
  "output_tokens": 4,
  "cost_usd": 0.0,
  "latency_ms": 312.5,
  "request_id": "b3f2e...",
  "fallback_used": false
}
```

### `GET /v1/models`

List all registered models with pricing.

```bash
curl http://localhost:8000/v1/models
```

### `GET /v1/stats`

Headline savings and quality metrics.

```bash
curl http://localhost:8000/v1/stats
```

### `GET /v1/routing-config`

Current tier → model_id mapping.

### `PUT /v1/routing-config`

Hot-reload routing without restarting the API.

```bash
curl -X PUT http://localhost:8000/v1/routing-config \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_key" \
  -d '{"routing": {"simple": "llama3.2:3b", "moderate": "llama-3.1-8b-instant", "complex": "gemini-2.0-flash"}}'
```

### `GET /health`

Liveness + readiness probe (used by Docker).

---

## Load Test

```bash
# Run both smart-routing and baseline (100 requests each — adjust as needed)
python scripts/load_test.py --mode both --requests 500
```

Results are written to `load_test_report.md`.

---

## Dashboard

Open `http://localhost:8501` after `docker compose up`.

Sections:
- 💰 **Cost savings** vs. all-premium baseline (headline metric)
- 📈 **Cost over time** — actual vs. baseline
- 🗂️ **Routing distribution** — tier and model breakdown
- 🎯 **Quality trend** — judge score + escalation rate
- 🚨 **Recent escalations** — system self-corrections, spot-checkable

---

## Project Structure

```
RouteMind_AI/
├── api/              # Phase 5 — FastAPI application
│   ├── main.py       # App + lifespan + middleware
│   ├── dependencies.py
│   └── routes/       # completions, health, models, stats, routing_config
├── classifier/       # Phase 2 — Complexity classifier
├── config/           # routing_map.yaml
├── core/             # Phase 1 — Provider clients, exceptions
├── dashboard/        # Phase 4 — Streamlit dashboard
├── data/             # Labeled prompts, train/test splits
├── db/               # Phase 4 — Postgres schema, ORM, logger, queries
├── docs/             # Case study
├── models/           # ModelConfig, MODEL_REGISTRY
├── scripts/          # retrain_classifier.py, load_test.py
├── tests/            # pytest suite
├── verification/     # Phase 3 — Quality verification & feedback loop
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Headline Result

<!-- TODO: Fill in after running `python scripts/load_test.py --mode both --requests 500` -->

| Metric | Value |
|---|---|
| Gross cost savings | `[run load test]` |
| Net savings (after verification) | `[run load test]` |
| Avg quality score | `[run load test]` |
| Escalation rate | `[run load test]` |

---

## Running Tests

```bash
pytest tests/ -v
```

All provider calls are mocked — no API spend during tests.

---

## License

MIT — see [LICENSE](LICENSE).
