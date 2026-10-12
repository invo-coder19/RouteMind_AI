"""
api/routes/health.py
====================
GET /health — liveness + readiness probe for Docker healthchecks.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.dependencies import get_app_state

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    """Health check response.

    Attributes:
        status:             "ok" or "degraded"
        classifier_loaded:  True if model.pkl was successfully loaded at startup.
        db_connected:       True if Postgres was reachable at startup.
    """
    status: str
    classifier_loaded: bool
    db_connected: bool


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness and readiness probe",
    description="Returns 200 when the API is running. Checks classifier and DB status.",
)
async def health_check(app_state=Depends(get_app_state)) -> HealthResponse:
    """GET /health — used by Docker healthchecks and uptime monitors."""
    overall = "ok" if (app_state.classifier_ready and app_state.db_ready) else "degraded"
    return HealthResponse(
        status=overall,
        classifier_loaded=app_state.classifier_ready,
        db_connected=app_state.db_ready,
    )
