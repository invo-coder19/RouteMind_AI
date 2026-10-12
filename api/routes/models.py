"""
api/routes/models.py
====================
GET /v1/models — returns the full model registry.
"""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from models.registry import MODEL_REGISTRY

router = APIRouter(tags=["Models"])


class ModelInfo(BaseModel):
    """Public representation of one registered model.

    Attributes:
        model_id:             Exact ID string passed to the provider API.
        provider:             Provider name.
        display_name:         Human-readable label.
        quality_tier:         "high" / "medium" / "low"
        cost_per_input_token: USD per input token.
        cost_per_output_token: USD per output token.
        avg_latency_ms:       Approximate median latency.
    """

    model_id: str
    provider: str
    display_name: str
    quality_tier: str
    cost_per_input_token: float
    cost_per_output_token: float
    avg_latency_ms: float


class ModelsResponse(BaseModel):
    """Response body for GET /v1/models."""
    models: list[ModelInfo]
    total: int


@router.get(
    "/models",
    response_model=ModelsResponse,
    summary="List all registered models",
    description="Returns every model in the Phase 1 registry with pricing and quality tier info.",
)
async def list_models() -> ModelsResponse:
    """GET /v1/models — read-only view of the model registry."""
    model_list = [
        ModelInfo(
            model_id=m.model_id,
            provider=m.provider,
            display_name=m.display_name or m.model_id,
            quality_tier=m.quality_tier,
            cost_per_input_token=m.cost_per_input_token,
            cost_per_output_token=m.cost_per_output_token,
            avg_latency_ms=m.avg_latency_ms,
        )
        for m in MODEL_REGISTRY
    ]
    return ModelsResponse(models=model_list, total=len(model_list))
