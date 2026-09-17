# RouteMind AI
### Autonomous Multi-LLM Inference Routing & Cost Optimization Platform

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2-e92063.svg)](https://docs.pydantic.dev/)
[![License](https://img.shields.io/badge/License-MIT%20%2F%20Apache-green.svg)](LICENSE)

---

## Overview

<div align="justify">
The Autonomous Multi-LLM Inference Routing and Cost Optimization Platform (<strong>RouteMind AI</strong>) is an intelligent infrastructure solution designed to optimize Large Language Model (LLM) inference by autonomously selecting the most suitable model for each incoming request. The platform dramatically reduces operational expenditure while preserving response quality through autonomous inference routing, prompt complexity analysis, and continuous post-inference quality validation.
</div>

<br/>

<div align="justify">
Architected as a provider-agnostic routing layer, RouteMind AI abstracts the underlying APIs of major commercial cloud providers (OpenRouter, Groq, Google Gemini, Mistral AI) as well as locally hosted inference engines (Ollama). This delivers a single, unified interface that enables cost-effective resource utilization across diverse generative AI workloads.
</div>

---

## Problem Statement

<div align="justify">
Organizations increasingly rely on Large Language Models to power enterprise applications, search engines, code assistants, and automated workflows. However, the standard architectural pattern—routing every prompt to high-capability, premium frontier models—introduces substantial operational overhead, inflated inference latency, and wasteful computational allocation for mundane queries.
</div>

<br/>

<div align="justify">
Existing model gateways provide limited dynamic model selection based on prompt complexity and typically lack automated mechanisms for continuous quality validation and adaptive routing. An intelligent routing layer is required to autonomously identify the most cost-effective language model capable of satisfying user-defined quality thresholds, while continually refining its routing decisions through ongoing empirical feedback.
</div>

---

## Objectives

<div align="justify">
The core engineering and research objectives of RouteMind AI include:
</div>

<div align="justify">
<ul>
  <li><strong>Unified Provider Abstraction:</strong> Design and implement a provider-agnostic async client capable of dispatching inference requests seamlessly across heterogeneous cloud APIs and local inference engines.</li>
  <li><strong>Autonomous Inference Routing:</strong> Eliminate the manual selection of model endpoints by automatically routing prompts based on real-time task complexity and quality constraints.</li>
  <li><strong>Prompt Complexity Classification:</strong> Categorize incoming queries (simple factual lookups, structured summaries, classification, or multi-step reasoning) prior to model invocation.</li>
  <li><strong>Cost Optimization:</strong> Minimize unnecessary usage of expensive frontier models by delegating low- and mid-complexity queries to high-throughput, low-cost alternatives.</li>
  <li><strong>Automated Quality Validation:</strong> Routinely benchmark and evaluate model outputs to verify that routed responses maintain parity with frontier model baselines.</li>
  <li><strong>Adaptive Learning:</strong> Ingest latency, token consumption, and quality validation scores into continuous feedback loops to adaptively calibrate future routing thresholds.</li>
  <li><strong>Extensible Architecture:</strong> Provide an open, decoupled architecture allowing rapid integration of emerging model providers and custom fine-tunes with zero breaking changes.</li>
</ul>
</div>

---

## System Architecture

```
                                      [ Incoming User Request / Prompt ]
                                                      │
                                                      ▼
                                      ┌────────────────────────────────┐
                                      │   Prompt Complexity Analyzer   │  (Phase 2)
                                      │   (Simple / Moderate / Complex)│
                                      └───────────────┬────────────────┘
                                                      │
                                                      ▼
                                      ┌────────────────────────────────┐
                                      │   Autonomous Model Router      │  (Phase 3)
                                      │   (Tier Matching & Policies)   │
                                      └───────────────┬────────────────┘
                                                      │
                                                      ▼
                               ┌──────────────────────────────────────────────┐
                               │       core.router_client (send_request)      │  (Phase 1)
                               └──────┬───────┬────────┬───────┬────────┬─────┘
                                      │       │        │       │        │
                   ┌──────────────────┘       │        │       │        └──────────────────┐
                   ▼                          ▼        ▼       ▼                           ▼
          ┌──────────────────┐          ┌──────────┐ ┌──────┐ ┌─────────┐          ┌──────────────┐
          │    OpenRouter    │          │   Groq   │ │Gemini│ │ Mistral │          │ Ollama Local │
          │(Claude, Gemini-8B│          │(Llama 3.1│ │(Flash│ │(Small 4)│          │(Gemma, Llama)│
          └────────┬─────────┘          └────┬─────┘ └──┬───┘ └────┬────┘          └──────┬───────┘
                   │                         │          │          │                      │
                   └──────────────────┬──────┴──────────┴──────────┴──────────────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │   Unified LLMResponse     │
                        │ ───────────────────────── │
                        │ • output_text: str        │
                        │ • input_tokens: int       │
                        │ • output_tokens: int      │
                        │ • latency_ms: float       │
                        │ • cost_usd: float         │
                        │ • model_id & provider     │
                        └─────────────┬─────────────┘
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
        [ Return to Caller ]                [ Background Evaluation Engine ]
        (Immediate Response)                (DeepEval Quality & Drift Check)
```

```
RouteMind_AI/
├── .env.example              # Template for API credentials and local endpoint URLs
├── .gitignore                # Git exclusion rules (virtual environments, credentials, caches)
├── LICENSE                   # Project license
├── README.md                 # Project documentation and setup guide
├── requirements.txt          # Pinned runtime dependencies
├── core/                     # Core routing and client modules
│   ├── __init__.py
│   ├── exceptions.py         # ProviderError and exception abstractions
│   └── router_client.py      # Async client dispatcher with tenacity retries
├── models/                   # Data schemas and model registry
│   ├── __init__.py
│   ├── registry.py           # ModelConfig, MODEL_REGISTRY, get_model(), list_models_by_tier()
│   └── response.py           # LLMResponse Pydantic v2 data model
├── scripts/                  # Benchmarking and operational utilities
│   └── baseline_test.py      # 10-prompt cross-provider benchmark & CSV generator
└── venv/                     # Python virtual environment (ignored in version control)
```

## Contributors

- **Soham Deshpande**
- **Riya Gandhi**
- **Avdhut Giri**
