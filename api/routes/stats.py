"""
api/routes/stats.py
===================
GET /v1/stats — summary metrics from Phase 4's query layer.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Stats"])


class StatsResponse(BaseModel):
    """Summary metrics returned by GET /v1/stats.

    Attributes:
        total_requests:     Total requests logged in the DB.
        total_cost_usd:     Total USD spent across all requests.
        baseline_cost_usd:  Hypothetical cost if all requests used the top model.
        savings_usd:        baseline_cost_usd - total_cost_usd.
        savings_pct:        Savings as a percentage of baseline cost.
        avg_quality_score:  Mean judge score across all verifications (None if none yet).
        escalation_rate_pct: % of verified requests that escalated (None if none yet).
    """

    total_requests: int
    total_cost_usd: float
    baseline_cost_usd: float
    savings_usd: float
    savings_pct: float
    avg_quality_score: float | None
    escalation_rate_pct: float | None


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Aggregate cost and quality metrics",
    description=(
        "Returns headline cost-savings metrics and quality stats from the Postgres "
        "logging database. Returns zeros if the DB is unreachable."
    ),
)
async def get_stats() -> StatsResponse:
    """GET /v1/stats — live metrics from the Phase 4 query layer."""
    try:
        from db.queries import get_baseline_comparison, get_quality_metrics

        df_baseline = get_baseline_comparison()
        df_quality = get_quality_metrics(days=365)  # all-time

        total_actual = float(df_baseline["actual_cost"].sum()) if not df_baseline.empty else 0.0
        total_baseline = float(df_baseline["baseline_cost"].sum()) if not df_baseline.empty else 0.0
        savings = total_baseline - total_actual
        savings_pct = (savings / total_baseline * 100) if total_baseline > 0 else 0.0

        avg_quality: float | None = None
        escalation_rate: float | None = None
        if not df_quality.empty:
            avg_quality = round(float(df_quality["avg_quality"].mean()), 4)
            escalation_rate = round(float(df_quality["escalation_rate"].mean()), 2)

        # Total requests: sum from baseline (has request_count only in cost_time)
        from db.queries import get_cost_over_time
        df_cost = get_cost_over_time(days=365)
        total_requests = int(df_cost["request_count"].sum()) if not df_cost.empty else 0

        return StatsResponse(
            total_requests=total_requests,
            total_cost_usd=round(total_actual, 8),
            baseline_cost_usd=round(total_baseline, 8),
            savings_usd=round(savings, 8),
            savings_pct=round(savings_pct, 2),
            avg_quality_score=avg_quality,
            escalation_rate_pct=escalation_rate,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("[stats] Query failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not fetch stats from database: {exc}",
        )
