"""
verification/comparator.py
==========================
Async reference-model comparator for Phase 3 of LLM Cost Autopilot.

Flow
----
1. Call the pre-configured reference (top-tier) model on the same prompt.
2. Ask a *judge model* to score the cheap response against the reference
   response on a 0.0–1.0 semantic quality scale.
3. Return a ``VerificationResult`` — never block the original caller.

Design notes
------------
- The LLM-as-judge prompt is intentionally semantic, not a simple text diff.
  This catches cases where the cheap model is factually wrong, incomplete, or
  poorly reasoned, even if it produces plausible-looking text.
- The judge uses structured output (we parse a JSON blob from the model
  response) so the score is deterministic rather than parsed from free text.
- If the judge call itself fails, we log the error and return a
  ``VerificationResult`` with quality_score = 0.0 and passed = False, so the
  failure is visible rather than silently swallowed.
"""
from __future__ import annotations

import json
import logging
import sys
import pathlib
from datetime import datetime, timezone
from typing import Final

from pydantic import BaseModel, Field

# Ensure project root is importable from any working directory.
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.exceptions import ProviderError  # noqa: E402
from core.router_client import send_request  # noqa: E402
from models.registry import ModelConfig  # noqa: E402
from models.response import LLMResponse  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configurable constants — change here, not buried in logic
# ---------------------------------------------------------------------------

#: Default quality threshold. A cheap-model response scoring *below* this
#: value is considered a routing failure and will trigger escalation.
#: Range: 0.0 (worst) – 1.0 (perfect). Default 0.7 = "good enough" bar.
DEFAULT_QUALITY_THRESHOLD: Final[float] = 0.7

#: System prompt injected into the judge model call.
#: The judge must respond with a JSON object — see ``_JUDGE_USER_TEMPLATE``.
_JUDGE_SYSTEM_PROMPT: Final[str] = (
    "You are a strict but fair quality evaluator for AI-generated text. "
    "Your job is to assess whether a 'candidate answer' satisfies the user's "
    "request as well as a 'reference answer' does. "
    "You MUST respond with a JSON object and nothing else. "
    'Format: {"score": <float 0.0-1.0>, "justification": "<one sentence>"} '
    "where 1.0 = candidate is equally good or better, "
    "0.0 = candidate is completely wrong or unhelpful."
)

#: Template for the judge's user-facing prompt.
_JUDGE_USER_TEMPLATE: Final[str] = (
    "ORIGINAL QUESTION:\n{prompt}\n\n"
    "REFERENCE ANSWER (high-quality model):\n{reference}\n\n"
    "CANDIDATE ANSWER (cheap model to evaluate):\n{candidate}\n\n"
    "Rate the candidate answer on a 0.0–1.0 scale relative to the reference. "
    "Respond ONLY with a JSON object."
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class VerificationResult(BaseModel):
    """Outcome of comparing a cheap-model response against a reference model.

    Attributes:
        request_id: Unique identifier for the original request.
        prompt: The original user prompt.
        cheap_model_id: model_id of the model that was originally used.
        reference_model_id: model_id of the high-quality reference model.
        cheap_response: The full LLMResponse from the cheap model.
        reference_response: The full LLMResponse from the reference model.
        quality_score: Float 0.0–1.0 assigned by the LLM judge.
        judge_justification: One-sentence explanation from the judge.
        passed: True if quality_score >= threshold.
        threshold_used: The threshold value that was applied.
        timestamp: UTC timestamp when verification completed.
    """

    request_id: str = Field(..., description="Original request ID")
    prompt: str = Field(..., description="Original user prompt")
    cheap_model_id: str = Field(..., description="Cheap model that was used")
    reference_model_id: str = Field(..., description="Reference model used for comparison")
    cheap_response: LLMResponse = Field(..., description="Response from the cheap model")
    reference_response: LLMResponse = Field(..., description="Response from the reference model")
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Judge score 0.0-1.0")
    judge_justification: str = Field(..., description="One-sentence judge explanation")
    passed: bool = Field(..., description="True if quality_score >= threshold_used")
    threshold_used: float = Field(..., description="The quality threshold that was applied")
    timestamp: datetime = Field(..., description="UTC timestamp of verification completion")

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_judge_prompt(
    prompt: str,
    reference_text: str,
    candidate_text: str,
) -> str:
    """Render the judge's user prompt from the template."""
    return _JUDGE_USER_TEMPLATE.format(
        prompt=prompt,
        reference=reference_text,
        candidate=candidate_text,
    )


def _parse_judge_response(raw_text: str) -> tuple[float, str]:
    """Extract (score, justification) from the judge's JSON response.

    Falls back to score=0.0 with an error message if parsing fails,
    so a judge call failure still produces a visible routing failure rather
    than an unhandled exception.
    """
    # Strip markdown code fences if the model wrapped the JSON.
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop first and last fence lines.
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        data = json.loads(text)
        score = float(data["score"])
        score = max(0.0, min(1.0, score))  # clamp to valid range
        justification = str(data.get("justification", "No justification provided."))
        return score, justification
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        logger.warning("Failed to parse judge JSON response: %s | raw=%r", exc, raw_text[:200])
        return 0.0, f"Judge response parsing failed: {exc}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def verify_response(
    request_id: str,
    prompt: str,
    cheap_response: LLMResponse,
    reference_model: ModelConfig,
    judge_model: ModelConfig | None = None,
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
    max_tokens: int = 512,
) -> tuple[VerificationResult, LLMResponse]:
    """Compare a cheap model's response against a reference model.

    This function is designed to be called inside ``asyncio.create_task()``
    so it never blocks the original response to the caller.

    Parameters
    ----------
    request_id:
        Unique ID for the original request (used for logging and storage).
    prompt:
        The original user prompt.
    cheap_response:
        The ``LLMResponse`` already returned to the caller by the cheap model.
    reference_model:
        The high-quality model to use for comparison (typically the highest
        ``quality_tier`` model from the registry).
    judge_model:
        Model to use as the LLM judge. Defaults to ``reference_model`` if not
        provided (re-uses the reference call's model as the judge).
    quality_threshold:
        Score below which the response is considered a routing failure.
        Defaults to ``DEFAULT_QUALITY_THRESHOLD`` (0.7).
    max_tokens:
        Max tokens for the reference model call.

    Returns
    -------
    tuple[VerificationResult, LLMResponse]
        The verification result and the reference model's raw response
        (needed by ``escalation.py`` if escalation is triggered).

    Raises
    ------
    ProviderError
        If the reference model call fails. The caller (background task)
        should catch this and log it without crashing.
    """
    logger.info(
        "[verify] request_id=%s | cheap=%s | reference=%s",
        request_id,
        cheap_response.model_id,
        reference_model.model_id,
    )

    # 1. Call the reference model on the same prompt.
    try:
        ref_response = await send_request(
            prompt=prompt,
            model_config=reference_model,
            max_tokens=max_tokens,
        )
    except ProviderError as exc:
        logger.error("[verify] Reference model call failed: %s", exc)
        raise

    # 2. Score using the LLM-as-judge.
    effective_judge = judge_model or reference_model
    judge_prompt = _build_judge_prompt(
        prompt=prompt,
        reference_text=ref_response.output_text,
        candidate_text=cheap_response.output_text,
    )

    quality_score = 0.0
    justification = "Judge call failed."
    try:
        judge_response = await send_request(
            prompt=judge_prompt,
            model_config=effective_judge,
            system_prompt=_JUDGE_SYSTEM_PROMPT,
            max_tokens=256,
        )
        quality_score, justification = _parse_judge_response(judge_response.output_text)
    except ProviderError as exc:
        logger.error("[verify] Judge model call failed: %s", exc)
        justification = f"Judge call failed: {exc}"

    passed = quality_score >= quality_threshold

    result = VerificationResult(
        request_id=request_id,
        prompt=prompt,
        cheap_model_id=cheap_response.model_id,
        reference_model_id=reference_model.model_id,
        cheap_response=cheap_response,
        reference_response=ref_response,
        quality_score=quality_score,
        judge_justification=justification,
        passed=passed,
        threshold_used=quality_threshold,
        timestamp=datetime.now(timezone.utc),
    )

    log_level = logging.INFO if passed else logging.WARNING
    logger.log(
        log_level,
        "[verify] request_id=%s | score=%.3f | passed=%s | %s",
        request_id,
        quality_score,
        passed,
        justification,
    )

    return result, ref_response
