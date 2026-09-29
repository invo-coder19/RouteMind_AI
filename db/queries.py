"""
db/queries.py
=============
Dashboard query layer for LLM Cost Autopilot (Phase 4).

All functions return a ``pandas.DataFrame`` and are completely independent of
Streamlit — they can be called and tested without a running dashboard.

Connection
----------
Each function creates its own synchronous Postgres connection via ``psycopg2``
(installed alongside asyncpg) so these can be called from Streamlit's
synchronous rendering context without bridging to an event loop.

If you need async versions for API endpoints, wrap with ``asyncio.run()``.

Database URL
------------
Read from ``DATABASE_URL`` environment variable (set in ``.env``).
"""
from __future__ import annotations

import logging
import os
import pathlib
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
from dotenv import load_dotenv

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

# Convert asyncpg URL to psycopg2-compatible URL for sync queries
_ASYNC_URL: str = os.getenv("DATABASE_URL", "")
_SYNC_URL: str = _ASYNC_URL.replace("postgresql+asyncpg://", "postgresql://", 1)

# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------


def _get_connection():
    """Return a psycopg2 connection.  Caller must close it.

    Raises
    ------
    RuntimeError
        If DATABASE_URL is not configured.
    """
    import psycopg2  # type: ignore[import]

    if not _SYNC_URL or _SYNC_URL == "postgresql://":
        raise RuntimeError(
            "DATABASE_URL is not set. Add it to your .env file:\n"
            "  DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/routemind"
        )
    return psycopg2.connect(_SYNC_URL)


def _query(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Execute a SQL query and return a DataFrame.  Handles connection lifecycle."""
    conn = _get_connection()
    try:
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Dashboard queries
# ---------------------------------------------------------------------------


def get_cost_over_time(days: int = 30) -> pd.DataFrame:
    """Actual cost spent per day over the last N days.

    Parameters
    ----------
    days:
        Number of days to look back (default 30).

    Returns
    -------
    DataFrame with columns:
        - ``date``       : date
        - ``total_cost`` : float — total USD spent that day
        - ``request_count`` : int — number of requests
    """
    sql = """
        SELECT
            DATE(timestamp AT TIME ZONE 'UTC')  AS date,
            CAST(SUM(cost_usd) AS FLOAT)        AS total_cost,
            COUNT(*)                            AS request_count
        FROM request_log
        WHERE timestamp >= NOW() - INTERVAL '%s days'
        GROUP BY 1
        ORDER BY 1 ASC
    """
    df = _query(sql, (days,))
    if df.empty:
        return pd.DataFrame(columns=["date", "total_cost", "request_count"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_baseline_comparison() -> pd.DataFrame:
    """Compare actual spend vs. hypothetical all-top-tier spend.

    The baseline is computed as:
        baseline_cost = input_tokens * ref_cost_per_input
                      + output_tokens * ref_cost_per_output

    The reference model pricing is read from the Phase 1 registry at query
    time (so changes to the registry are reflected immediately).

    Returns
    -------
    DataFrame with columns:
        - ``date``          : date
        - ``actual_cost``   : float — what was actually spent
        - ``baseline_cost`` : float — what would have been spent with top model
        - ``savings``       : float — baseline_cost - actual_cost
        - ``savings_pct``   : float — savings / baseline_cost * 100
    """
    # Load reference model pricing from registry
    from models.registry import list_models_by_tier  # noqa: E402

    high_tier = list_models_by_tier("high")
    if high_tier:
        ref_input_price = float(high_tier[0].cost_per_input_token)
        ref_output_price = float(high_tier[0].cost_per_output_token)
    else:
        # Fallback: use a $0.01/1K token estimate if no high-tier model defined
        ref_input_price = 0.01 / 1000
        ref_output_price = 0.01 / 1000
        logger.warning("[queries] No high-tier model found; using fallback pricing.")

    sql = """
        SELECT
            DATE(timestamp AT TIME ZONE 'UTC')         AS date,
            CAST(SUM(cost_usd) AS FLOAT)               AS actual_cost,
            SUM(input_tokens)                          AS total_input_tokens,
            SUM(output_tokens)                         AS total_output_tokens
        FROM request_log
        GROUP BY 1
        ORDER BY 1 ASC
    """
    df = _query(sql)
    if df.empty:
        return pd.DataFrame(columns=["date", "actual_cost", "baseline_cost", "savings", "savings_pct"])

    df["date"] = pd.to_datetime(df["date"])
    df["baseline_cost"] = (
        df["total_input_tokens"] * ref_input_price
        + df["total_output_tokens"] * ref_output_price
    )
    df["savings"] = df["baseline_cost"] - df["actual_cost"]
    df["savings_pct"] = (df["savings"] / df["baseline_cost"].replace(0, float("nan"))) * 100
    return df[["date", "actual_cost", "baseline_cost", "savings", "savings_pct"]]


def get_routing_distribution() -> pd.DataFrame:
    """Percentage of requests per complexity tier and model.

    Returns
    -------
    DataFrame with columns:
        - ``complexity_tier``  : str
        - ``model_id_used``    : str
        - ``provider``         : str
        - ``request_count``    : int
        - ``pct_of_total``     : float — % of all requests
        - ``total_cost``       : float — cumulative USD for this group
    """
    sql = """
        SELECT
            complexity_tier,
            model_id_used,
            provider,
            COUNT(*)                          AS request_count,
            CAST(SUM(cost_usd) AS FLOAT)      AS total_cost
        FROM request_log
        GROUP BY 1, 2, 3
        ORDER BY request_count DESC
    """
    df = _query(sql)
    if df.empty:
        return pd.DataFrame(columns=[
            "complexity_tier", "model_id_used", "provider",
            "request_count", "pct_of_total", "total_cost",
        ])
    total = df["request_count"].sum()
    df["pct_of_total"] = (df["request_count"] / total * 100).round(2)
    return df


def get_quality_metrics(days: int = 30) -> pd.DataFrame:
    """Daily average quality score and escalation rate.

    Parameters
    ----------
    days:
        Number of days to look back (default 30).

    Returns
    -------
    DataFrame with columns:
        - ``date``             : date
        - ``avg_quality``      : float — average judge score (0.0–1.0)
        - ``pass_rate``        : float — % of verifications that passed
        - ``escalation_rate``  : float — % of verifications that escalated
        - ``verification_count`` : int
    """
    sql = """
        SELECT
            DATE(v.timestamp AT TIME ZONE 'UTC')          AS date,
            CAST(AVG(v.quality_score) AS FLOAT)           AS avg_quality,
            CAST(AVG(CASE WHEN v.passed    THEN 1.0 ELSE 0.0 END) * 100 AS FLOAT) AS pass_rate,
            CAST(AVG(CASE WHEN v.escalated THEN 1.0 ELSE 0.0 END) * 100 AS FLOAT) AS escalation_rate,
            COUNT(*)                                      AS verification_count
        FROM verification_log v
        WHERE v.timestamp >= NOW() - INTERVAL '%s days'
        GROUP BY 1
        ORDER BY 1 ASC
    """
    df = _query(sql, (days,))
    if df.empty:
        return pd.DataFrame(columns=[
            "date", "avg_quality", "pass_rate", "escalation_rate", "verification_count",
        ])
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_recent_escalations(limit: int = 20) -> pd.DataFrame:
    """Most recent escalated requests for spot-checking.

    Parameters
    ----------
    limit:
        Maximum number of rows to return (default 20).

    Returns
    -------
    DataFrame with columns:
        - ``timestamp``           : datetime
        - ``request_id``          : str
        - ``complexity_tier``     : str
        - ``model_id_used``       : str (cheap model)
        - ``reference_model_id``  : str
        - ``quality_score``       : float
        - ``judge_justification`` : str
        - ``prompt_preview``      : str — first 120 chars of the prompt
    """
    sql = """
        SELECT
            r.timestamp,
            r.request_id,
            r.complexity_tier,
            r.model_id_used,
            v.reference_model_id,
            CAST(v.quality_score AS FLOAT) AS quality_score,
            v.judge_justification,
            LEFT(r.prompt_text, 120)       AS prompt_preview
        FROM   verification_log v
        JOIN   request_log      r USING (request_id)
        WHERE  v.escalated = TRUE
        ORDER BY r.timestamp DESC
        LIMIT %s
    """
    df = _query(sql, (limit,))
    if df.empty:
        return pd.DataFrame(columns=[
            "timestamp", "request_id", "complexity_tier", "model_id_used",
            "reference_model_id", "quality_score", "judge_justification", "prompt_preview",
        ])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df
