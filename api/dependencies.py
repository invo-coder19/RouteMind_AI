"""
api/dependencies.py
===================
FastAPI dependency injection helpers shared across all routes.

Provides:
- ``get_app_state``   — typed access to objects loaded at startup
- ``verify_api_key``  — optional API-key auth guard
- ``get_request_id``  — extracts the UUID injected by the request-ID middleware
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, Header, HTTPException, Request, status

if TYPE_CHECKING:
    from api.main import AppState


def get_app_state(request: Request) -> "AppState":
    """Return the ``AppState`` object stored on ``app.state`` at startup.

    Raises
    ------
    RuntimeError
        If the lifespan handler has not run (should never happen in production).
    """
    state = getattr(request.app.state, "app_state", None)
    if state is None:
        raise RuntimeError("AppState not initialised — lifespan handler may have failed.")
    return state


# ---------------------------------------------------------------------------
# Optional API-key auth
# ---------------------------------------------------------------------------

_API_KEY: str | None = os.getenv("API_KEY")


async def verify_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    """FastAPI dependency: enforce ``X-API-Key`` header if ``API_KEY`` env var is set.

    If ``API_KEY`` is not set in the environment, all requests are allowed
    (convenient for local development without keys).

    Raises
    ------
    HTTPException(403)
        If ``API_KEY`` is configured and the header is missing or wrong.
    """
    if _API_KEY is None:
        return  # auth disabled
    if x_api_key != _API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing X-API-Key header.",
        )


# ---------------------------------------------------------------------------
# Request ID
# ---------------------------------------------------------------------------


def get_request_id(request: Request) -> str:
    """Extract the ``request_id`` set by the request-ID middleware."""
    return request.state.request_id  # type: ignore[attr-defined]
