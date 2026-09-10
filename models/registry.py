"""
models.registry — model configuration registry for LLM Cost Autopilot.

Every model available to the routing layer is declared here as a ModelConfig
entry inside MODEL_REGISTRY. Pricing is expressed per *individual* token
(not per million) so that cost calculations in router_client are straightforward
multiplications without any unit conversion.

Pricing notes
-------------
All prices were sourced from official provider documentation as of 2026-09-10.
Each entry carries a ``# TODO: verify pricing`` comment with a direct link to
the relevant pricing page. Re-verify whenever you add a model or before going
to production.

Pricing reference pages:
  - OpenRouter  : https://openrouter.ai/models
  - Groq        : https://console.groq.com/docs/openai
  - Gemini      : https://ai.google.dev/pricing
  - Mistral     : https://mistral.ai/technology/#pricing
  - Ollama      : free (local inference, no API cost)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    """Configuration for a single LLM available in the registry.

    Attributes:
        provider: Short provider identifier used internally for dispatch
                  (``"openrouter"``, ``"groq"``, ``"gemini"``,
                  ``"mistral"``, ``"ollama"``).
        model_id: The exact model identifier string passed to the provider's
                  API endpoint (e.g. ``"anthropic/claude-haiku-4-5"``).
        cost_per_input_token: Provider's listed cost in USD per **single**
                              input token. Use 0.0 for local/free models.
        cost_per_output_token: Provider's listed cost in USD per **single**
                               output token. Use 0.0 for local/free models.
        avg_latency_ms: Approximate median round-trip latency in milliseconds
                        under normal load. Used for routing tie-breaking.
                        Set to 0 if unknown; the baseline test will populate
                        real numbers.
        quality_tier: Coarse quality band used by the routing layer.
                      ``"high"`` → frontier/best model for the task class,
                      ``"medium"`` → capable mid-tier model,
                      ``"low"`` → fast small model for simple tasks.
        endpoint_type: Indicates how the model is called.
                       ``"openai_compat"`` providers share the same request
                       schema (OpenRouter, Groq, Mistral).
                       ``"gemini"`` uses Google's distinct GenerateContent
                       REST schema. ``"ollama"`` uses the Ollama chat API.
        display_name: Human-readable label for logs and the dashboard.
    """

    provider: Literal["openrouter", "groq", "gemini", "mistral", "ollama"]
    model_id: str = Field(..., description="Exact model string passed to the API")
    cost_per_input_token: float = Field(
        ..., ge=0.0, description="USD cost per single input token"
    )
    cost_per_output_token: float = Field(
        ..., ge=0.0, description="USD cost per single output token"
    )
    avg_latency_ms: float = Field(
        default=0.0, ge=0.0, description="Approximate median latency in ms"
    )
    quality_tier: Literal["high", "medium", "low"] = Field(
        ..., description="Coarse quality band for the routing layer"
    )
    endpoint_type: Literal["openai_compat", "gemini", "ollama"] = Field(
        ..., description="API schema family used by the provider"
    )
    display_name: str = Field(
        default="", description="Human-readable name for logs and dashboard"
    )

    model_config = {"frozen": True}  # Pydantic v2 — make instances immutable


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: list[ModelConfig] = [
    # ── OpenRouter ─────────────────────────────────────────────────────────
    # OpenRouter passes through provider token pricing with no per-token markup.
    # A 5.5 % top-up fee applies when purchasing credits, not per-request.
    # Ref: https://openrouter.ai/models
    ModelConfig(
        provider="openrouter",
        model_id="anthropic/claude-haiku-4-5",
        # TODO: verify pricing → https://openrouter.ai/anthropic/claude-haiku-4-5
        cost_per_input_token=1.00 / 1_000_000,   # $1.00 / 1M tokens
        cost_per_output_token=5.00 / 1_000_000,  # $5.00 / 1M tokens
        avg_latency_ms=800.0,
        quality_tier="high",
        endpoint_type="openai_compat",
        display_name="Claude Haiku 4.5 (via OpenRouter)",
    ),
    ModelConfig(
        provider="openrouter",
        model_id="google/gemini-flash-1-5-8b",
        # TODO: verify pricing → https://openrouter.ai/google/gemini-flash-1-5-8b
        cost_per_input_token=0.0375 / 1_000_000,  # $0.0375 / 1M tokens
        cost_per_output_token=0.15 / 1_000_000,   # $0.15 / 1M tokens
        avg_latency_ms=600.0,
        quality_tier="medium",
        endpoint_type="openai_compat",
        display_name="Gemini Flash 1.5 8B (via OpenRouter)",
    ),
    # ── Groq ───────────────────────────────────────────────────────────────
    # Groq LPU inference — extremely low latency, OpenAI-compatible API.
    # Ref: https://console.groq.com/docs/openai
    ModelConfig(
        provider="groq",
        model_id="llama-3.1-8b-instant",
        # TODO: verify pricing → https://console.groq.com/docs/openai#models
        cost_per_input_token=0.05 / 1_000_000,   # $0.05 / 1M tokens
        cost_per_output_token=0.08 / 1_000_000,  # $0.08 / 1M tokens
        avg_latency_ms=250.0,
        quality_tier="medium",
        endpoint_type="openai_compat",
        display_name="Llama 3.1 8B Instant (Groq)",
    ),
    # ── Google Gemini (direct) ─────────────────────────────────────────────
    # Uses Google's GenerateContent REST API — different schema from OpenAI.
    # Introductory pricing for Gemini 2.0 Flash applies through end of 2026.
    # Ref: https://ai.google.dev/pricing
    ModelConfig(
        provider="gemini",
        model_id="gemini-2.0-flash",
        # TODO: verify pricing → https://ai.google.dev/pricing#2_0flash
        cost_per_input_token=0.10 / 1_000_000,   # $0.10 / 1M tokens (≤128k ctx)
        cost_per_output_token=0.40 / 1_000_000,  # $0.40 / 1M tokens
        avg_latency_ms=700.0,
        quality_tier="medium",
        endpoint_type="gemini",
        display_name="Gemini 2.0 Flash (Google AI)",
    ),
    # ── Mistral (direct) ──────────────────────────────────────────────────
    # OpenAI-compatible endpoint. Mistral Small 4 is their cost-efficient tier.
    # Ref: https://mistral.ai/technology/#pricing
    ModelConfig(
        provider="mistral",
        model_id="mistral-small-latest",
        # TODO: verify pricing → https://mistral.ai/technology/#pricing
        cost_per_input_token=0.15 / 1_000_000,  # $0.15 / 1M tokens
        cost_per_output_token=0.60 / 1_000_000, # $0.60 / 1M tokens
        avg_latency_ms=900.0,
        quality_tier="medium",
        endpoint_type="openai_compat",
        display_name="Mistral Small 4 (Mistral AI)",
    ),
    # ── Ollama — local models (zero API cost) ──────────────────────────────
    # All local models have cost_per_*_token = 0.0.
    # Requires Ollama running at OLLAMA_BASE_URL (default: http://localhost:11434).
    # Pull models first: `ollama pull gemma3:4b` etc.
    # Ref: https://ollama.com/library
    ModelConfig(
        provider="ollama",
        model_id="gemma3:4b",
        # TODO: verify pricing — local inference, no provider cost.
        #       Hardware / electricity costs are outside scope of this registry.
        cost_per_input_token=0.0,
        cost_per_output_token=0.0,
        avg_latency_ms=1500.0,
        quality_tier="low",
        endpoint_type="ollama",
        display_name="Gemma 3 4B (Ollama local)",
    ),
    ModelConfig(
        provider="ollama",
        model_id="llama3.1:8b",
        # TODO: verify pricing — local inference, no provider cost.
        cost_per_input_token=0.0,
        cost_per_output_token=0.0,
        avg_latency_ms=2000.0,
        quality_tier="low",
        endpoint_type="ollama",
        display_name="Llama 3.1 8B (Ollama local)",
    ),
    ModelConfig(
        provider="ollama",
        model_id="llama3.2:3b",
        # TODO: verify pricing — local inference, no provider cost.
        #       Using 3B variant which is widely available; swap to 8B if pulled.
        cost_per_input_token=0.0,
        cost_per_output_token=0.0,
        avg_latency_ms=1200.0,
        quality_tier="low",
        endpoint_type="ollama",
        display_name="Llama 3.2 3B (Ollama local)",
    ),
]

# Build a lookup dict at import time for O(1) access.
_REGISTRY_INDEX: dict[str, ModelConfig] = {m.model_id: m for m in MODEL_REGISTRY}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def get_model(model_id: str) -> ModelConfig:
    """Return the ModelConfig for the given ``model_id``.

    Args:
        model_id: The exact model identifier string (e.g.
                  ``"llama-3.1-8b-instant"``). Must match a ``model_id``
                  field in MODEL_REGISTRY.

    Returns:
        The matching ModelConfig instance.

    Raises:
        KeyError: If no model with the given ``model_id`` is registered.
                  The error message lists all available IDs to aid debugging.
    """
    if model_id not in _REGISTRY_INDEX:
        available = ", ".join(sorted(_REGISTRY_INDEX.keys()))
        raise KeyError(
            f"Model '{model_id}' not found in registry. "
            f"Available model IDs: {available}"
        )
    return _REGISTRY_INDEX[model_id]


def list_models_by_tier(quality_tier: Literal["high", "medium", "low"]) -> list[ModelConfig]:
    """Return all ModelConfig entries matching the given quality tier.

    Args:
        quality_tier: One of ``"high"``, ``"medium"``, or ``"low"``.

    Returns:
        Filtered list of ModelConfig instances, sorted by cost_per_input_token
        ascending (cheapest first).
    """
    return sorted(
        [m for m in MODEL_REGISTRY if m.quality_tier == quality_tier],
        key=lambda m: m.cost_per_input_token,
    )
