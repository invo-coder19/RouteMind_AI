"""
db/init_db.py
=============
Idempotent database initialisation script for LLM Cost Autopilot.

Creates all tables defined in ``db/schema.sql`` if they do not already exist.
Safe to run repeatedly — uses ``CREATE TABLE IF NOT EXISTS`` throughout.

Usage
-----
Direct execution:
    python db/init_db.py

Or programmatically:
    import asyncio
    from db.init_db import init_database
    asyncio.run(init_database())

What it does
------------
1. Reads ``DATABASE_URL`` from ``.env``
2. Executes ``db/schema.sql`` verbatim against the Postgres database
3. Prints confirmation or a clear error message

This is the preferred way to set up the schema because it keeps the SQL in
``schema.sql`` as the single source of truth.  SQLAlchemy's
``Base.metadata.create_all`` is available as a fallback if needed.
"""
from __future__ import annotations

import asyncio
import logging
import pathlib
import sys

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("db.init_db")

_SCHEMA_FILE = pathlib.Path(__file__).resolve().parent / "schema.sql"


async def init_database() -> None:
    """Execute schema.sql against the configured Postgres database.

    Raises
    ------
    RuntimeError
        If ``DATABASE_URL`` is not set.
    Exception
        If the SQL execution fails (e.g. connection refused).
    """
    # Import here so a missing DATABASE_URL gives a clear error at runtime.
    from db.engine import get_engine
    from sqlalchemy import text

    logger.info("Reading schema from %s", _SCHEMA_FILE)
    sql = _SCHEMA_FILE.read_text(encoding="utf-8")

    engine = get_engine()
    logger.info("Connecting to database …")

    async with engine.begin() as conn:
        # Split on semicolons and execute each statement individually
        # because some drivers don't support multi-statement execution.
        statements = [s.strip() for s in sql.split(";") if s.strip()]
        for stmt in statements:
            await conn.execute(text(stmt))

    logger.info("Database initialised successfully.")
    logger.info("Tables created (or already existed): request_log, verification_log")


if __name__ == "__main__":
    asyncio.run(init_database())
