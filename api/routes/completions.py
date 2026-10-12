"""
api/routes/completions.py
=========================
POST /v1/completions — the main routing endpoint.

Flow (per spec):
1. Generate request_id (already in request.state from middleware)
2. classify_and_route(prompt) → tier + ModelConfig
3. send_request() with that model
4. Return response to caller IMMEDIATELY
5. Schedule background: log_request + verification loop + log_verification

If the routed model raises ProviderError, automatically falls back to the
next quality tier (medium → high) and notes it in the response.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api.dependencies import get_app_state, get_request_id, verify_api_key
from core.exceptions import ProviderError
from core.router_client import send_request
from models.registry import get_model, list_models_by_tier

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Completions"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class CompletionRequest(BaseModel):
    """Request body for POST /v1/completions.

    Attributes:
        prompt:       The user's prompt (required, non-empty).
        system_prompt: Optional system-level instruction passed to the model.
        max_tokens:   Maximum tokens to generate (default 1024).
        force_model:  Optional model_id to bypass the classifier entirely.
                      Useful for baseline comparisons and testing.
    """

    prompt: str = Field(..., min_length=1, description="The user's prompt.", examples=["Explain quantum entanglement simply."])
    system_prompt: str | None = Field(None, description="Optional system instruction.")
    max_tokens: int = Field(1024, ge=1, le=8192, description="Max tokens to generate.")
    force_model: str | None = Field(
        None,
        description="Override the router. Pass a model_id from GET /v1/models to skip classification.",
    )


class CompletionResponse(BaseModel):
    """Response body for POST /v1/completions.

    Attributes:
        output_text:     Generated text from the model.
        model_used:      The model_id that actually served the request.
        provider:        Provider name (groq / gemini / ollama / etc.).
        complexity_tier: Classifier output — simple / moderate / complex.
        input_tokens:    Prompt token count.
        output_tokens:   Completion token count.
        cost_usd:        Estimated cost for this call.
        latency_ms:      End-to-end latency in milliseconds.
        request_id:      UUID for this request (also in X-Request-ID header).
        fallback_used:   True if the primary routed model failed and a fallback was used.
    """

    output_text: str
    model_used: str
    provider: str
    complexity_tier: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    request_id: str
    fallback_used: bool = False


# ---------------------------------------------------------------------------
# Background task helpers
# ---------------------------------------------------------------------------


async def _background_logging_and_verification(
    request_id: str,
    prompt: str,
    tier: str,
    response,  # LLMResponse
    request_time: datetime,
) -> None:
    """Fire-and-forget: log request + run Phase 3 verification + log verification.

    All exceptions are caught — this must NEVER propagate and affect the caller.
    """
    # Log request to Postgres
    try:
        from db.logger import log_request
        await log_request(
            response=response,
            tier=tier,
            request_id=request_id,
            prompt_text=prompt,
            timestamp=request_time,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("[bg] log_request failed for %s: %s", request_id, exc)

    # Phase 3: verify quality + log verification
    try:
        from verification.loop import run_verification_loop
        outcome = await run_verification_loop(
            request_id=request_id,
            prompt=prompt,
            tier_assigned=tier,
            cheap_response=response,
            original_request_time=request_time,
        )
        logger.info(
            "[bg] Verification complete for %s: passed=%s escalated=%s",
            request_id,
            outcome.verification_result.passed,
            outcome.escalated,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("[bg] Verification failed for %s: %s", request_id, exc)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/completions",
    response_model=CompletionResponse,
    summary="Route a prompt to the cheapest capable model",
    description=(
        "Classifies the prompt's complexity, routes to the cheapest capable model, "
        "returns the answer immediately, then asynchronously verifies quality in the "
        "background. Use ``force_model`` to bypass the classifier for baseline testing."
    ),
)
async def create_completion(
    body: CompletionRequest,
    background_tasks: BackgroundTasks,
    request_id: Annotated[str, Depends(get_request_id)],
    app_state=Depends(get_app_state),
) -> CompletionResponse:
    """POST /v1/completions — main routing endpoint."""
    request_time = datetime.now(timezone.utc)
    fallback_used = False
    tier = "unknown"

    # ── 1. Resolve model ───────────────────────────────────────────────────
    if body.force_model:
        try:
            model_config = get_model(body.force_model)
            tier = "forced"
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"force_model '{body.force_model}' not found in registry. {exc}",
            )
    else:
        if not app_state.classifier_ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "Classifier model not loaded. "
                    "Run `python classifier/train.py` to generate model.pkl, then restart the API."
                ),
            )
        try:
            from classifier.predict import classify_and_route
            tier, model_config = classify_and_route(body.prompt)
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] Classifier error: %s", request_id, exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Classifier failed to process the prompt.",
            )

    # ── 2. Call model (with automatic fallback) ────────────────────────────
    llm_response = None
    last_exc: ProviderError | None = None

    # Primary attempt
    try:
        llm_response = await send_request(
            prompt=body.prompt,
            model_config=model_config,
            system_prompt=body.system_prompt,
            max_tokens=body.max_tokens,
        )
    except ProviderError as exc:
        last_exc = exc
        logger.warning(
            "[%s] Primary model %s failed (HTTP %s): %s — attempting fallback",
            request_id,
            model_config.model_id,
            exc.status_code,
            exc,
        )

    # Fallback: try next quality tier up
    if llm_response is None:
        fallback_tiers = {"low": "medium", "medium": "high"}
        fallback_quality = fallback_tiers.get(model_config.quality_tier)
        if fallback_quality:
            candidates = list_models_by_tier(fallback_quality)
            for fallback_model in candidates:
                try:
                    llm_response = await send_request(
                        prompt=body.prompt,
                        model_config=fallback_model,
                        system_prompt=body.system_prompt,
                        max_tokens=body.max_tokens,
                    )
                    model_config = fallback_model
                    fallback_used = True
                    logger.info("[%s] Fallback succeeded with %s", request_id, fallback_model.model_id)
                    break
                except ProviderError as exc:  # noqa: BLE001
                    last_exc = exc
                    continue

    if llm_response is None:
        # All attempts failed — return appropriate HTTP error
        status_code = (
            status.HTTP_429_TOO_MANY_REQUESTS
            if last_exc and last_exc.status_code == 429
            else status.HTTP_502_BAD_GATEWAY
        )
        raise HTTPException(
            status_code=status_code,
            detail=f"Upstream provider failed: {last_exc}",
        )

    # ── 3. Schedule background work (non-blocking) ─────────────────────────
    background_tasks.add_task(
        _background_logging_and_verification,
        request_id=request_id,
        prompt=body.prompt,
        tier=tier,
        response=llm_response,
        request_time=request_time,
    )

    # ── 4. Return immediately ──────────────────────────────────────────────
    return CompletionResponse(
        output_text=llm_response.output_text,
        model_used=llm_response.model_id,
        provider=llm_response.provider,
        complexity_tier=tier,
        input_tokens=llm_response.input_tokens,
        output_tokens=llm_response.output_tokens,
        cost_usd=llm_response.cost_usd,
        latency_ms=llm_response.latency_ms,
        request_id=request_id,
        fallback_used=fallback_used,
    )
