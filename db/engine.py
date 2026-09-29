"""
db/engine.py
============
Async SQLAlchemy engine factory for LLM Cost Autopilot.

Reads ``DATABASE_URL`` from the environment (via python-dotenv).
The engine is created once at module import time (singleton) and shared
across the application lifetime — never create a new engine per request.

Environment variable
--------------------
DATABASE_URL — full Postgres connection URL, e.g.:
    postgresql+asyncpg://user:password@localhost:5432/routemind

Set it in ``.env`` at the project root.
"""
from __future__ import annotations

import logging
import os
import pathlib
import sys

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

_DATABASE_URL: str | None = os.getenv("DATABASE_URL")

if not _DATABASE_URL:
    logger.warning(
        "[db/engine] DATABASE_URL not set — database features will be unavailable. "
        "Add DATABASE_URL=postgresql+asyncpg://... to your .env file."
    )

# ---------------------------------------------------------------------------
# Engine singleton
# ---------------------------------------------------------------------------

_engine: AsyncEngine | None = None


def get_engine() -> AsyncEngine:
    """Return the shared async SQLAlchemy engine (created once).

    Raises
    ------
    RuntimeError
        If ``DATABASE_URL`` is not set in the environment.
    """
    global _engine
    if _engine is None:
        if not _DATABASE_URL:
            raise RuntimeError(
                "DATABASE_URL environment variable is not set. "
                "Add it to your .env file:\n"
                "  DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/routemind"
            )
        _engine = create_async_engine(
            _DATABASE_URL,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,   # detect stale connections before use
            echo=False,           # set True to log all SQL for debugging
        )
        logger.info("[db/engine] Async engine created for %s", _DATABASE_URL.split("@")[-1])
    return _engine


def get_session_factory() -> sessionmaker:
    """Return a factory for async SQLAlchemy sessions.

    Usage::

        async with get_session_factory()() as session:
            # query here
    """
    return sessionmaker(
        bind=get_engine(),
        class_=AsyncSession,
        expire_on_commit=False,
    )
