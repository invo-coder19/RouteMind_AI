"""
models.response — normalized response model for LLM Cost Autopilot.

Every provider returns data in its own schema. The router_client layer parses
each provider's raw response and produces a single LLMResponse so that all
downstream consumers (classifier, quality verifier, logger, dashboard) work
with one consistent type.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    """Normalized response returned by every provider client.

    All provider-specific response formats are parsed and mapped into this
    model before being returned from ``send_request``.

    Attributes:
        output_text: The generated text from the model (assistant message
                     content). Stripped of leading/trailing whitespace.
        input_tokens: Number of tokens in the prompt / input (as reported by
                      the provider). Used for cost and logging.
        output_tokens: Number of tokens in the generated response (as reported
                       by the provider).
        latency_ms: End-to-end wall-clock time from request start to response
                    received, in milliseconds. Measured client-side.
        cost_usd: Estimated cost of this call in USD, calculated as:
                  ``input_tokens * cost_per_input_token
                    + output_tokens * cost_per_output_token``.
                  Always 0.0 for local Ollama models.
        model_id: The model identifier that handled this request (matches the
                  ``model_id`` field from ModelConfig).
        provider: Short provider name (e.g. ``"groq"``, ``"ollama"``).
        raw_response: The full, unmodified JSON response body from the provider
                      as a Python dict. Useful for debugging and for extracting
                      provider-specific metadata not captured in this model.
    """

    output_text: str = Field(
        ..., description="Generated text content from the model"
    )
    input_tokens: int = Field(
        ..., ge=0, description="Number of prompt tokens (provider-reported)"
    )
    output_tokens: int = Field(
        ..., ge=0, description="Number of completion tokens (provider-reported)"
    )
    latency_ms: float = Field(
        ..., ge=0.0, description="Wall-clock request latency in milliseconds"
    )
    cost_usd: float = Field(
        ..., ge=0.0, description="Estimated call cost in USD"
    )
    model_id: str = Field(
        ..., description="Model identifier that served this request"
    )
    provider: str = Field(
        ..., description="Provider name (openrouter / groq / gemini / mistral / ollama)"
    )
    raw_response: dict[str, Any] = Field(
        ..., description="Unmodified JSON response body from the provider"
    )

    model_config = {"frozen": True}  # Pydantic v2 — immutable after creation

    @property
    def total_tokens(self) -> int:
        """Total token count (input + output)."""
        return self.input_tokens + self.output_tokens

    def __str__(self) -> str:
        return (
            f"LLMResponse("
            f"provider={self.provider}, "
            f"model={self.model_id}, "
            f"tokens={self.input_tokens}+{self.output_tokens}, "
            f"cost=${self.cost_usd:.6f}, "
            f"latency={self.latency_ms:.0f}ms)"
        )
