"""
api/routes/routing_config.py
============================
GET  /v1/routing-config  — view current tier-to-model mapping.
PUT  /v1/routing-config  — update and hot-reload the routing map.
                           Requires X-API-Key if API_KEY env var is set.
"""
from __future__ import annotations

import logging
import pathlib
import sys
from typing import Annotated

import yaml
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from api.dependencies import get_app_state, verify_api_key
from models.registry import get_model

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_ROUTING_MAP_YAML = _PROJECT_ROOT / "config" / "routing_map.yaml"

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Routing Config"])

VALID_TIERS = frozenset({"simple", "moderate", "complex"})


class RoutingConfigResponse(BaseModel):
    """Current tier-to-model routing map.

    Attributes:
        routing: Dict mapping tier names to model_id strings.
    """
    routing: dict[str, str]


class RoutingConfigUpdate(BaseModel):
    """Request body for PUT /v1/routing-config.

    Attributes:
        routing: New tier-to-model mapping. All three tiers must be present.
    """
    routing: dict[str, str]


@router.get(
    "/routing-config",
    response_model=RoutingConfigResponse,
    summary="Get current routing configuration",
    description="Returns the current tier → model_id mapping from routing_map.yaml.",
)
async def get_routing_config() -> RoutingConfigResponse:
    """GET /v1/routing-config."""
    if not _ROUTING_MAP_YAML.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Routing map not found at {_ROUTING_MAP_YAML}",
        )
    with _ROUTING_MAP_YAML.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return RoutingConfigResponse(routing=raw.get("routing", {}))


@router.put(
    "/routing-config",
    response_model=RoutingConfigResponse,
    summary="Update and hot-reload the routing configuration",
    description=(
        "Validates every model_id against the registry before writing. "
        "Returns 422 with a clear message if any model_id is unknown. "
        "Requires ``X-API-Key`` header if ``API_KEY`` environment variable is set."
    ),
    dependencies=[Depends(verify_api_key)],
)
async def update_routing_config(
    body: RoutingConfigUpdate,
    app_state=Depends(get_app_state),
) -> RoutingConfigResponse:
    """PUT /v1/routing-config — validate, write, and hot-reload."""
    # ── Validate tiers ─────────────────────────────────────────────────────
    missing = VALID_TIERS - set(body.routing.keys())
    if missing:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Missing required tiers: {sorted(missing)}. All of {sorted(VALID_TIERS)} must be defined.",
        )

    extra = set(body.routing.keys()) - VALID_TIERS
    if extra:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown tiers in request: {sorted(extra)}. Only {sorted(VALID_TIERS)} are valid.",
        )

    # ── Validate every model_id against the registry ───────────────────────
    validation_errors: list[str] = []
    for tier, model_id in body.routing.items():
        try:
            get_model(model_id)
        except KeyError:
            validation_errors.append(
                f"Tier '{tier}': model_id '{model_id}' not found in registry."
            )

    if validation_errors:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "One or more model_ids are not in the registry.",
                "errors": validation_errors,
                "hint": "Call GET /v1/models to see all valid model_ids.",
            },
        )

    # ── Write to disk ──────────────────────────────────────────────────────
    new_config = {"routing": body.routing}
    _ROUTING_MAP_YAML.parent.mkdir(parents=True, exist_ok=True)
    with _ROUTING_MAP_YAML.open("w", encoding="utf-8") as fh:
        yaml.dump(new_config, fh, default_flow_style=False, allow_unicode=True)

    # ── Hot-reload into the running app state ─────────────────────────────
    try:
        from classifier.predict import _load_routing_map
        app_state.routing_map = _load_routing_map()
        logger.info("[routing_config] Hot-reload successful: %s", body.routing)
    except Exception as exc:  # noqa: BLE001
        logger.error("[routing_config] Hot-reload failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Config written to disk but hot-reload failed: {exc}",
        )

    return RoutingConfigResponse(routing=body.routing)
