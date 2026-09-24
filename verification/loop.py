"""
verification/loop.py
====================
Orchestrates the async quality-verification & feedback loop (Phase 3).

This module provides ``run_verification_loop`` — the single function the
router calls *after* returning the cheap-model answer to the caller.

Non-blocking contract
---------------------
``run_verification_loop`` is designed to be called via::

    asyncio.create_task(run_verification_loop(...))

This means it runs concurrently with the rest of the application and NEVER
blocks the response to the end user.

IMPORTANT: ``asyncio.create_task`` only guarantees non-blocking execution if
the calling code is already running inside an async event loop.  If called
from a sync context (e.g. a plain script), use ``asyncio.run(...)`` instead.
The function itself is correct regardless — the non-blocking guarantee depends
on the *caller* using ``create_task``.

Flow inside the task
--------------------
1. Call the reference model + run the LLM judge  (comparator.verify_response)
2. Decide whether to escalate                    (escalation.maybe_escalate)
3. Persist the result                            (feedback_store.save_escalation_outcome)
4. Return EscalationOutcome (caller can inspect if needed)
"""
from __future__ import annotations

import logging
import pathlib
import sys
from datetime import datetime

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.registry import ModelConfig, list_models_by_tier  # noqa: E402
from models.response import LLMResponse  # noqa: E402
from verification.comparator import (  # noqa: E402
    DEFAULT_QUALITY_THRESHOLD,
    VerificationResult,
    verify_response,
)
from verification.escalation import (  # noqa: E402
    ESCALATION_WINDOW_SECONDS,
    EscalationOutcome,
    maybe_escalate,
)
from verification.feedback_store import save_escalation_outcome  # noqa: E402

logger = logging.getLogger(__name__)


async def run_verification_loop(
    request_id: str,
    prompt: str,
    tier_assigned: str,
    cheap_response: LLMResponse,
    original_request_time: datetime,
    reference_model: ModelConfig | None = None,
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
    escalation_window: float = ESCALATION_WINDOW_SECONDS,
) -> EscalationOutcome:
    """Run the full verification and escalation flow for one request.

    Designed to be called with ``asyncio.create_task(run_verification_loop(...))``
    so the caller is not blocked.

    Parameters
    ----------
    request_id:
        Unique identifier for the original request.
    prompt:
        The original user prompt.
    tier_assigned:
        The complexity tier predicted by Phase 2 ("simple"/"moderate"/"complex").
    cheap_response:
        The ``LLMResponse`` already returned to the end user.
    original_request_time:
        UTC datetime when the cheap-model call was dispatched.
    reference_model:
        The high-quality model to compare against.  If None, the function
        will pick the first model with ``quality_tier="high"`` from the
        registry.  Raises ``ValueError`` if none exists.
    quality_threshold:
        Score below which routing is considered a failure.
    escalation_window:
        Seconds after which escalation is no longer possible.

    Returns
    -------
    EscalationOutcome
        Full record of the verification decision.  The caller should check
        ``outcome.should_deliver_reference`` and deliver
        ``outcome.reference_response`` to the end user if True.
    """
    # ------------------------------------------------ resolve reference model
    if reference_model is None:
        high_tier_models = list_models_by_tier("high")
        if not high_tier_models:
            raise ValueError(
                "No model with quality_tier='high' found in the registry. "
                "Add at least one high-tier model or pass reference_model explicitly."
            )
        reference_model = high_tier_models[0]
        logger.debug(
            "[loop] Auto-selected reference model: %s", reference_model.model_id
        )

    # -------------------------------------------- 1. verify (reference + judge)
    try:
        verification_result, ref_response = await verify_response(
            request_id=request_id,
            prompt=prompt,
            cheap_response=cheap_response,
            reference_model=reference_model,
            quality_threshold=quality_threshold,
        )
    except Exception as exc:
        # Verification failure must never crash the application.
        # Log and return a synthetic "passed" outcome so storage/escalation is skipped.
        logger.error(
            "[loop] Verification failed for request_id=%s: %s", request_id, exc
        )
        raise

    # ---------------------------------------------------- 2. escalation decision
    outcome = await maybe_escalate(
        request_id=request_id,
        original_request_time=original_request_time,
        verification_result=verification_result,
        reference_response=ref_response,
        escalation_window=escalation_window,
    )

    # ------------------------------------------------------- 3. persist to DB
    try:
        await save_escalation_outcome(
            outcome=outcome,
            tier_assigned=tier_assigned,
        )
    except Exception as exc:
        # Storage failure must never crash the loop.
        logger.error(
            "[loop] Failed to save escalation outcome for request_id=%s: %s",
            request_id,
            exc,
        )

    return outcome
