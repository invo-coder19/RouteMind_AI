"""
db/logger.py
============
Async request and verification logger for LLM Cost Autopilot (Phase 4).

Provides two fire-and-forget async functions that write to Postgres.

CRITICAL non-blocking contract
--------------------------------
Both functions must be called with ``asyncio.create_task()`` from the
response path so they never delay the answer returned to the end user::

    asyncio.create_task(log_request(response, tier="simple", request_id="req-123"))

If called directly with ``await``, the DB write will be in the critical path
and will add latency to the user-facing response.

Failure behaviour
-----------------
A DB write failure is caught, logged to stderr/console, and silently ignored.
The application continues normally.  A logging failure must NEVER raise an
exception that propagates to the caller.
"""
from __future__ import annotations

import logging
import pathlib
import sys
from datetime import datetime, timezone

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)


async def log_request(
    response: "LLMResponse",  # noqa: F821 — avoid circular import at module level
    tier: str,
    request_id: str,
    prompt_text: str = "",
    timestamp: datetime | None = None,
) -> None:
    """Write one row to ``request_log`` for a completed LLM request.

    Designed to be called with ``asyncio.create_task()`` — never awaited
    directly in the response path.

    Parameters
    ----------
    response:
        The ``LLMResponse`` returned by ``core.router_client.send_request``.
    tier:
        Complexity tier assigned by Phase 2 classifier
        ("simple" / "moderate" / "complex").
    request_id:
        Unique identifier for this request (UUID string).
    prompt_text:
        The original user prompt. Stored for dashboard display and retraining.
    timestamp:
        UTC datetime of the request. Defaults to now if not provided.

    Raises
    ------
    Nothing — all exceptions are caught and logged.
    """
    from db.engine import get_engine
    from db.models import RequestLog
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker

    ts = timestamp or datetime.now(timezone.utc)

    try:
        engine = get_engine()
        factory = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
        async with factory() as session:
            async with session.begin():
                row = RequestLog(
                    request_id=request_id,
                    timestamp=ts,
                    prompt_text=prompt_text,
                    complexity_tier=tier,
                    model_id_used=response.model_id,
                    provider=response.provider,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    cost_usd=response.cost_usd,
                    latency_ms=response.latency_ms,
                )
                session.add(row)
    except Exception as exc:  # noqa: BLE001
        # Must not propagate — log failure is silent from the caller's view.
        logger.error(
            "[db/logger] Failed to log request %s: %s: %s",
            request_id,
            type(exc).__name__,
            exc,
        )


async def log_verification(
    result: "VerificationResult",  # noqa: F821
    escalated: bool,
) -> None:
    """Write one row to ``verification_log`` for a completed quality check.

    Designed to be called with ``asyncio.create_task()`` — never awaited
    directly in the response path.

    Parameters
    ----------
    result:
        The ``VerificationResult`` produced by Phase 3's
        ``verification.comparator.verify_response``.
    escalated:
        Whether escalation was triggered (from ``EscalationOutcome.escalated``).

    Raises
    ------
    Nothing — all exceptions are caught and logged.
    """
    from db.engine import get_engine
    from db.models import VerificationLog
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker

    try:
        engine = get_engine()
        factory = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
        async with factory() as session:
            async with session.begin():
                row = VerificationLog(
                    request_id=result.request_id,
                    timestamp=result.timestamp,
                    reference_model_id=result.reference_model_id,
                    quality_score=result.quality_score,
                    passed=result.passed,
                    escalated=escalated,
                    judge_justification=result.judge_justification,
                )
                session.add(row)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "[db/logger] Failed to log verification %s: %s: %s",
            result.request_id,
            type(exc).__name__,
            exc,
        )
