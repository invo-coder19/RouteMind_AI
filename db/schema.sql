-- =============================================================================
-- db/schema.sql — LLM Cost Autopilot database schema
-- =============================================================================
-- Source of truth for the Postgres schema.
-- Run via: psql $DATABASE_URL -f db/schema.sql
-- Or use db/init_db.py which executes this file programmatically.
--
-- Conventions:
--   - All timestamps stored as TIMESTAMPTZ (UTC-aware).
--   - Cost stored as NUMERIC(18, 10) — avoids float rounding on micro-cents.
--   - UUIDs as TEXT (avoids uuid extension dependency for portability).
--   - All CREATE statements use IF NOT EXISTS — safe to run repeatedly.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 1. request_log
--    One row per LLM request dispatched by the router.
--    Populated by db/logger.py::log_request()
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS request_log (
    request_id      TEXT            PRIMARY KEY,
    timestamp       TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    prompt_text     TEXT            NOT NULL,
    complexity_tier TEXT            NOT NULL,   -- simple | moderate | complex
    model_id_used   TEXT            NOT NULL,
    provider        TEXT            NOT NULL,
    input_tokens    INTEGER         NOT NULL DEFAULT 0,
    output_tokens   INTEGER         NOT NULL DEFAULT 0,
    cost_usd        NUMERIC(18, 10) NOT NULL DEFAULT 0,
    latency_ms      NUMERIC(12, 3)  NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_request_log_timestamp
    ON request_log (timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_request_log_tier
    ON request_log (complexity_tier);

CREATE INDEX IF NOT EXISTS idx_request_log_model
    ON request_log (model_id_used);

-- ---------------------------------------------------------------------------
-- 2. verification_log
--    One row per verification check run by Phase 3.
--    Populated by db/logger.py::log_verification()
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS verification_log (
    request_id          TEXT            PRIMARY KEY
                                        REFERENCES request_log (request_id)
                                        ON DELETE CASCADE,
    timestamp           TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    reference_model_id  TEXT            NOT NULL,
    quality_score       NUMERIC(5, 4)   NOT NULL,   -- 0.0000 – 1.0000
    passed              BOOLEAN         NOT NULL,
    escalated           BOOLEAN         NOT NULL,
    judge_justification TEXT            NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_verification_log_timestamp
    ON verification_log (timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_verification_log_passed
    ON verification_log (passed);

CREATE INDEX IF NOT EXISTS idx_verification_log_escalated
    ON verification_log (escalated);
