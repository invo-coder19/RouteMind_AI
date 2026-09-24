"""
verification/escalation.py
==========================
Escalation logic for Phase 3 of LLM Cost Autopilot.

Escalation rule (concrete, documented)
---------------------------------------
Escalation is only attempted if ALL of the following are true:

1. The verification completed within ``ESCALATION_WINDOW_SECONDS`` of the
   original request being dispatched.
2. The ``VerificationResult.passed`` field is False (cheap model scored below
   threshold).

If the window has elapsed, the reference response is *still logged* (it can be
used for retraining) but it is NOT sent back to the caller — the window for
overwriting the response has passed.

Non-blocking contract
---------------------
``maybe_escalate`` is designed to run inside ``asyncio.create_task()``.
It does not call back into the original HTTP response stream directly; instead
it sets ``EscalationOutcome.should_deliver_reference = True`` and returns the
reference response, trusting the calling layer (e.g. a WebSocket or a response
queue) to handle actual delivery.  If the calling layer doesn't implement
delivery, the result is still logged — nothing crashes.
"""
from __future__ import annotations

import logging
import sys
import pathlib
from datetime import datetime, timezone
from enum import Enum
from typing import Final

from pydantic import BaseModel, Field

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.response import LLMResponse  # noqa: E402
from verification.comparator import VerificationResult  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configurable constants — named, documented, at module top
# ---------------------------------------------------------------------------

#: Maximum seconds between the original request dispatch time and the
#: completion of verification for escalation to be possible.
#: If verification takes longer than this, the window is considered closed
#: and the caller has already received (and likely displayed) the cheap
#: model's answer.
ESCALATION_WINDOW_SECONDS: Final[float] = 5.0


# ---------------------------------------------------------------------------
# Outcome model
# ---------------------------------------------------------------------------


class EscalationReason(str, Enum):
    """Why escalation did or did not occur."""

    ESCALATED = "escalated"
    "Verification failed and window was open — reference response should be delivered."

    WINDOW_CLOSED = "window_closed"
    "Verification completed outside the escalation window — cannot overwrite response."

    PASSED = "passed"
    "Cheap model passed quality check — no escalation needed."


class EscalationOutcome(BaseModel):
    """Full record of an escalation decision.

    Attributes:
        request_id: The original request identifier.
        escalated: True if the reference response should replace the cheap answer.
        reason: Specific reason for the outcome (see ``EscalationReason``).
        should_deliver_reference: True if the calling layer should send the
            reference response to the end user (only possible inside window).
        reference_response: The reference model's answer; present whenever
            escalation was triggered. None if quality passed or not applicable.
        elapsed_seconds: Time in seconds from original request to verification
            completion. Used for observability.
        escalation_window_used: The window constant that was applied.
        verification_result: The full ``VerificationResult`` for storage.
    """

    request_id: str
    escalated: bool
    reason: EscalationReason
    should_deliver_reference: bool
    reference_response: LLMResponse | None
    elapsed_seconds: float
    escalation_window_used: float
    verification_result: VerificationResult

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def maybe_escalate(
    request_id: str,
    original_request_time: datetime,
    verification_result: VerificationResult,
    reference_response: LLMResponse,
    escalation_window: float = ESCALATION_WINDOW_SECONDS,
) -> EscalationOutcome:
    """Decide whether to escalate a failed verification.

    Parameters
    ----------
    request_id:
        Unique identifier for the original request.
    original_request_time:
        UTC ``datetime`` when the original cheap-model call was *dispatched*
        (not when the answer was received).  Used to measure elapsed time.
    verification_result:
        The ``VerificationResult`` produced by ``comparator.verify_response``.
    reference_response:
        The reference model's ``LLMResponse`` (already fetched by the comparator).
    escalation_window:
        Seconds after which escalation is no longer allowed.
        Defaults to ``ESCALATION_WINDOW_SECONDS`` (5 s).

    Returns
    -------
    EscalationOutcome
        A fully typed record of the decision.  The caller is responsible for
        actually delivering ``reference_response`` to the end user when
        ``should_deliver_reference`` is True.

    Notes
    -----
    This function is intentionally non-blocking (no I/O).  Storage is handled
    by ``feedback_store.save_verification`` — call that separately.
    """
    now = datetime.now(timezone.utc)
    elapsed = (now - original_request_time).total_seconds()

    # Case 1: quality passed — no escalation needed.
    if verification_result.passed:
        logger.info(
            "[escalation] request_id=%s | PASSED (score=%.3f) | elapsed=%.2fs",
            request_id,
            verification_result.quality_score,
            elapsed,
        )
        return EscalationOutcome(
            request_id=request_id,
            escalated=False,
            reason=EscalationReason.PASSED,
            should_deliver_reference=False,
            reference_response=None,
            elapsed_seconds=elapsed,
            escalation_window_used=escalation_window,
            verification_result=verification_result,
        )

    # Case 2: quality failed but window closed.
    if elapsed > escalation_window:
        logger.warning(
            "[escalation] request_id=%s | WINDOW CLOSED (score=%.3f, elapsed=%.2fs > %.1fs) "
            "| reference answer logged for retraining only",
            request_id,
            verification_result.quality_score,
            elapsed,
            escalation_window,
        )
        return EscalationOutcome(
            request_id=request_id,
            escalated=False,
            reason=EscalationReason.WINDOW_CLOSED,
            should_deliver_reference=False,
            reference_response=reference_response,  # still carry it for storage
            elapsed_seconds=elapsed,
            escalation_window_used=escalation_window,
            verification_result=verification_result,
        )

    # Case 3: quality failed AND within window — escalate.
    logger.warning(
        "[escalation] request_id=%s | ESCALATED (score=%.3f < %.2f, elapsed=%.2fs) "
        "| delivering reference answer from %s",
        request_id,
        verification_result.quality_score,
        verification_result.threshold_used,
        elapsed,
        reference_response.model_id,
    )
    return EscalationOutcome(
        request_id=request_id,
        escalated=True,
        reason=EscalationReason.ESCALATED,
        should_deliver_reference=True,
        reference_response=reference_response,
        elapsed_seconds=elapsed,
        escalation_window_used=escalation_window,
        verification_result=verification_result,
    )
