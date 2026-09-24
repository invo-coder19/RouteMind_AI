"""
verification/feedback_store.py
==============================
Async SQLite feedback store for Phase 3 of LLM Cost Autopilot.

Schema
------
Table ``verification_results``:

    request_id        TEXT PRIMARY KEY
    prompt            TEXT NOT NULL
    tier_assigned     TEXT NOT NULL          -- Phase 2 classifier output
    cheap_model_id    TEXT NOT NULL
    reference_model_id TEXT NOT NULL
    quality_score     REAL NOT NULL
    passed            INTEGER NOT NULL       -- 0 or 1 (SQLite has no BOOL)
    escalated         INTEGER NOT NULL       -- 0 or 1
    timestamp         TEXT NOT NULL          -- ISO-8601 UTC

Design choices
--------------
- SQLite is used for this phase.  Replacing with Postgres in Phase 4 only
  requires swapping the connection setup — the query strings are compatible.
- ``aiosqlite`` is used for non-blocking I/O so the async event loop is never
  blocked by disk writes.
- ``get_failed_verifications`` returns rows ordered oldest-first so retraining
  always processes the longest-outstanding failures first (FIFO).
- A ``retraining_processed_at`` column is added so the retraining job can mark
  processed rows without deleting them (audit trail preserved).
"""
from __future__ import annotations

import logging
import pathlib
import sys
from datetime import datetime, timezone

import aiosqlite

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from verification.comparator import VerificationResult  # noqa: E402
from verification.escalation import EscalationOutcome  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database path — stored at the project root for portability
# ---------------------------------------------------------------------------

DEFAULT_DB_PATH: pathlib.Path = _PROJECT_ROOT / "feedback.db"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL: str = """
CREATE TABLE IF NOT EXISTS verification_results (
    request_id              TEXT PRIMARY KEY,
    prompt                  TEXT NOT NULL,
    tier_assigned           TEXT NOT NULL,
    cheap_model_id          TEXT NOT NULL,
    reference_model_id      TEXT NOT NULL,
    quality_score           REAL NOT NULL,
    passed                  INTEGER NOT NULL,
    escalated               INTEGER NOT NULL,
    timestamp               TEXT NOT NULL,
    retraining_processed_at TEXT
)
"""


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


async def init_db(db_path: pathlib.Path = DEFAULT_DB_PATH) -> None:
    """Create the feedback database and table if they do not exist.

    Safe to call repeatedly — uses ``CREATE TABLE IF NOT EXISTS``.

    Parameters
    ----------
    db_path:
        Filesystem path to the SQLite database file.
        Defaults to ``feedback.db`` in the project root.
    """
    async with aiosqlite.connect(db_path) as db:
        await db.execute(_CREATE_TABLE_SQL)
        await db.commit()
    logger.info("[feedback_store] Database initialised at %s", db_path)


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


async def save_verification(
    result: VerificationResult,
    escalated: bool,
    tier_assigned: str,
    db_path: pathlib.Path = DEFAULT_DB_PATH,
) -> None:
    """Persist a verification result to the feedback table.

    Parameters
    ----------
    result:
        The ``VerificationResult`` produced by ``comparator.verify_response``.
    escalated:
        Whether escalation was triggered (from ``EscalationOutcome.escalated``).
    tier_assigned:
        The complexity tier predicted by the Phase 2 classifier
        ("simple" / "moderate" / "complex").
    db_path:
        Path to the SQLite database file.

    Raises
    ------
    aiosqlite.IntegrityError
        If ``result.request_id`` already exists (primary key violation).
        The caller should catch this if duplicate saves are possible.
    """
    await init_db(db_path)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            INSERT INTO verification_results
                (request_id, prompt, tier_assigned, cheap_model_id,
                 reference_model_id, quality_score, passed, escalated, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.request_id,
                result.prompt,
                tier_assigned,
                result.cheap_model_id,
                result.reference_model_id,
                result.quality_score,
                int(result.passed),
                int(escalated),
                result.timestamp.isoformat(),
            ),
        )
        await db.commit()

    logger.info(
        "[feedback_store] Saved verification: request_id=%s | passed=%s | escalated=%s",
        result.request_id,
        result.passed,
        escalated,
    )


async def save_escalation_outcome(
    outcome: EscalationOutcome,
    tier_assigned: str,
    db_path: pathlib.Path = DEFAULT_DB_PATH,
) -> None:
    """Convenience wrapper — save from an ``EscalationOutcome`` directly.

    Parameters
    ----------
    outcome:
        The fully resolved ``EscalationOutcome`` from ``escalation.maybe_escalate``.
    tier_assigned:
        Phase 2 classifier tier for this request.
    db_path:
        Path to the SQLite database file.
    """
    await save_verification(
        result=outcome.verification_result,
        escalated=outcome.escalated,
        tier_assigned=tier_assigned,
        db_path=db_path,
    )


# ---------------------------------------------------------------------------
# Read — for retraining
# ---------------------------------------------------------------------------


def get_failed_verifications(
    since: datetime,
    db_path: pathlib.Path = DEFAULT_DB_PATH,
) -> list[dict]:
    """Fetch routing failures since a given timestamp for retraining.

    Uses a synchronous SQLite connection (no async) so this function can be
    called from the retraining script without an event loop.

    Parameters
    ----------
    since:
        Only return failures that occurred after this UTC datetime.
    db_path:
        Path to the SQLite database file.

    Returns
    -------
    list[dict]
        A list of row dicts ordered oldest-first (FIFO for retraining).
        Each dict has the same keys as the ``verification_results`` table.
        Returns an empty list if no failures exist or the DB doesn't exist yet.
    """
    if not db_path.exists():
        logger.info("[feedback_store] No database at %s — returning empty list", db_path)
        return []

    import sqlite3

    since_str = since.astimezone(timezone.utc).isoformat()

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            """
            SELECT *
            FROM   verification_results
            WHERE  passed = 0
              AND  timestamp > ?
              AND  retraining_processed_at IS NULL
            ORDER BY timestamp ASC
            """,
            (since_str,),
        )
        rows = [dict(row) for row in cursor.fetchall()]

    logger.info(
        "[feedback_store] Found %d unprocessed failures since %s",
        len(rows),
        since_str,
    )
    return rows


def mark_retraining_processed(
    request_ids: list[str],
    db_path: pathlib.Path = DEFAULT_DB_PATH,
) -> None:
    """Mark a set of failures as processed by the retraining job.

    Parameters
    ----------
    request_ids:
        List of ``request_id`` values to mark.
    db_path:
        Path to the SQLite database file.
    """
    if not request_ids or not db_path.exists():
        return

    import sqlite3

    now_str = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "UPDATE verification_results SET retraining_processed_at = ? WHERE request_id = ?",
            [(now_str, rid) for rid in request_ids],
        )
        conn.commit()

    logger.info("[feedback_store] Marked %d rows as retraining-processed", len(request_ids))
