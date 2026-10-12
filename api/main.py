"""
api/main.py
===========
LLM Cost Autopilot — FastAPI application entry-point.

Architecture
------------
- A single async ``lifespan`` handler loads ALL heavy objects (classifier,
  routing config, DB connection pool) exactly once at startup and tears them
  down cleanly at shutdown.  Individual request handlers receive these objects
  via FastAPI's dependency injection system.
- Routes are split across ``api/routes/`` — this file stays thin.
- CORS and request-ID middleware are applied globally.
- Optional API-key authentication is enforced per route via a dependency
  (see ``api/dependencies.py``).

Run with:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
import pathlib
import sys
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from api.routes import completions, models, routing_config, stats, health  # noqa: E402

logger = logging.getLogger("api.main")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s [%(request_id)s]: %(message)s"
    if False  # custom filter not set up — simpler format below
    else "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ---------------------------------------------------------------------------
# Shared application state (set once in lifespan, read via dependency)
# ---------------------------------------------------------------------------


@dataclass
class AppState:
    """Objects that are expensive to load and must be shared across requests.

    Attributes:
        classifier_model:  Loaded sklearn model object from ``model.pkl``.
        routing_map:       Tier → ``ModelConfig`` mapping (reloadable at runtime).
        classifier_ready:  True once model.pkl loaded without error.
        db_ready:          True once the DB engine successfully pinged.
    """

    classifier_model: Any = None
    routing_map: dict = field(default_factory=dict)
    classifier_ready: bool = False
    db_ready: bool = False


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all resources once at startup; clean up at shutdown."""
    state = AppState()
    app.state.app_state = state

    # ── 1. Classifier + routing map ────────────────────────────────────────
    try:
        # Import triggers module-level singleton loading (_MODEL, _ROUTING_MAP)
        import classifier.predict as _predict  # noqa: F401

        state.classifier_model = _predict._MODEL
        state.routing_map = _predict._ROUTING_MAP
        state.classifier_ready = True
        logger.info("[lifespan] Classifier and routing map loaded.")
    except FileNotFoundError as exc:
        logger.warning("[lifespan] Classifier not ready: %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.error("[lifespan] Unexpected error loading classifier: %s", exc)

    # ── 2. DB engine (ping to verify connection) ───────────────────────────
    try:
        from db.engine import get_engine
        from sqlalchemy import text

        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        state.db_ready = True
        logger.info("[lifespan] Database connection verified.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[lifespan] DB not available: %s", exc)

    logger.info(
        "[lifespan] Startup complete — classifier=%s db=%s",
        state.classifier_ready,
        state.db_ready,
    )

    yield  # ← application runs here

    # ── Shutdown ───────────────────────────────────────────────────────────
    try:
        from db.engine import get_engine
        await get_engine().dispose()
        logger.info("[lifespan] DB engine disposed.")
    except Exception:  # noqa: BLE001
        pass

    logger.info("[lifespan] Shutdown complete.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LLM Cost Autopilot",
    description=(
        "Intelligent LLM routing layer that classifies prompt complexity, "
        "selects the cheapest capable model, and asynchronously verifies quality. "
        "Saves money without sacrificing answer quality."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ───────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production via ALLOWED_ORIGINS env var
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request-ID middleware ──────────────────────────────────────────────────


@app.middleware("http")
async def request_id_middleware(request: Request, call_next) -> Response:
    """Generate a UUID per request; attach to request state and response header."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response: Response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ── Routers ────────────────────────────────────────────────────────────────

app.include_router(health.router)
app.include_router(completions.router, prefix="/v1")
app.include_router(models.router,        prefix="/v1")
app.include_router(stats.router,         prefix="/v1")
app.include_router(routing_config.router, prefix="/v1")
