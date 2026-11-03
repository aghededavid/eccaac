-- ═══════════════════════════════════════════════════════════
--  ECCAAC — Database Schema
--  TimescaleDB (PostgreSQL extension)
--
--  Run:
--    psql $DSN -f schema.sql
--  Or via Go:
--    store.Migrate(ctx)
--
--  Tables:
--    Relational (plain PostgreSQL):
--      users            — user profiles + phase tracking
--      sessions         — session summaries (written on end)
--      policy_groups    — threshold presets
--      policy_rules     — custom escalation rules
--      devices          — device registry
--
--    Time-series (TimescaleDB hypertables):
--      risk_scores      — raw score per 10s window, one row per event
--      enforcement_log  — every enforcement action taken
--
--    Continuous Aggregates (materialized views, auto-refreshed):
--      risk_scores_1min  — 1-minute bucketed averages
--      risk_scores_1hr   — 1-hour bucketed averages
--      risk_scores_1day  — 1-day bucketed averages
-- ═══════════════════════════════════════════════════════════

-- ── Extension ────────────────────────────────────────────

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ═══════════════════════════════════════════════════════════
--  RELATIONAL TABLES
-- ═══════════════════════════════════════════════════════════

-- ── Policy Groups ─────────────────────────────────────────

CREATE TABLE IF NOT EXISTS policy_groups (
    id              TEXT PRIMARY KEY,
    name            TEXT        NOT NULL,
    description     TEXT        NOT NULL DEFAULT '',
    low_max         FLOAT       NOT NULL DEFAULT 25,
    elevated_max    FLOAT       NOT NULL DEFAULT 50,
    high_max        FLOAT       NOT NULL DEFAULT 75,
    soft_divisor    FLOAT       NOT NULL DEFAULT 2.0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Seed built-in groups (idempotent)
INSERT INTO policy_groups (id, name, description, low_max, elevated_max, high_max, soft_divisor)
VALUES
    ('default', 'Default',                   'Standard thresholds for most users',          25, 50, 75, 2.0),
    ('strict',  'Strict (Privileged/Server)', 'Tighter thresholds for high-value targets',   15, 35, 60, 1.5),
    ('relaxed', 'Relaxed (Low-risk devices)', 'Relaxed thresholds for low-sensitivity hosts', 35, 60, 80, 2.5)
ON CONFLICT (id) DO NOTHING;

-- ── Policy Rules ──────────────────────────────────────────

CREATE TABLE IF NOT EXISTS policy_rules (
    id                  TEXT        PRIMARY KEY,
    name                TEXT        NOT NULL,
    description         TEXT        NOT NULL DEFAULT '',
    priority            INT         NOT NULL DEFAULT 100,
    enabled             BOOLEAN     NOT NULL DEFAULT TRUE,
    scope               TEXT        NOT NULL DEFAULT 'global'  -- global | user | device
        CHECK (scope IN ('global','user','device')),
    target_id           TEXT        NOT NULL DEFAULT '',
    low_max             FLOAT       NOT NULL DEFAULT 25,
    elevated_max        FLOAT       NOT NULL DEFAULT 50,
    high_max            FLOAT       NOT NULL DEFAULT 75,
    escalate_condition  TEXT        NOT NULL DEFAULT '',
    escalate_actions    JSONB       NOT NULL DEFAULT '[]',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_policy_rules_priority ON policy_rules(priority ASC)
    WHERE enabled = TRUE;

-- ── Users ─────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS users (
    user_id          TEXT        PRIMARY KEY,
    username         TEXT        NOT NULL DEFAULT '',
    domain           TEXT        NOT NULL DEFAULT '',
    phase            TEXT        NOT NULL DEFAULT 'OBSERVATION'
        CHECK (phase IN ('OBSERVATION','SOFT','FULL')),
    session_count    INT         NOT NULL DEFAULT 0,
    policy_id        TEXT        NOT NULL DEFAULT 'default'
        REFERENCES policy_groups(id) ON UPDATE CASCADE,
    is_admin         BOOLEAN     NOT NULL DEFAULT FALSE,
    is_suspended     BOOLEAN     NOT NULL DEFAULT FALSE,
    suspended_until  TIMESTAMPTZ,
    notes            TEXT        NOT NULL DEFAULT '',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_last_seen ON users(last_seen_at DESC);
CREATE INDEX IF NOT EXISTS idx_users_suspended ON users(is_suspended) WHERE is_suspended = TRUE;

-- ── Devices ───────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS devices (
    device_id        TEXT        PRIMARY KEY,   -- hostname or fingerprint prefix
    hostname         TEXT        NOT NULL DEFAULT '',
    os_version       TEXT        NOT NULL DEFAULT '',
    domain_joined    BOOLEAN     NOT NULL DEFAULT FALSE,
    policy_id        TEXT        NOT NULL DEFAULT 'default'
        REFERENCES policy_groups(id) ON UPDATE CASCADE,
    is_trusted       BOOLEAN     NOT NULL DEFAULT TRUE,
    notes            TEXT        NOT NULL DEFAULT '',
    first_seen_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Sessions ──────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS sessions (
    session_id       TEXT        PRIMARY KEY,
    user_id          TEXT        NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    device_id        TEXT        NOT NULL,
    phase            TEXT        NOT NULL DEFAULT 'OBSERVATION',
    started_at       TIMESTAMPTZ NOT NULL,
    last_seen_at     TIMESTAMPTZ NOT NULL,
    ended_at         TIMESTAMPTZ,
    active           BOOLEAN     NOT NULL DEFAULT TRUE,

    -- Running stats (updated on session end)
    event_count      INT         NOT NULL DEFAULT 0,
    avg_risk_score   FLOAT       NOT NULL DEFAULT 0,
    max_risk_score   FLOAT       NOT NULL DEFAULT 0,
    critical_count   INT         NOT NULL DEFAULT 0,
    was_flagged      BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_sessions_user        ON sessions(user_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_active      ON sessions(active, started_at DESC)
    WHERE active = TRUE;
CREATE INDEX IF NOT EXISTS idx_sessions_started     ON sessions(started_at DESC);

-- ═══════════════════════════════════════════════════════════
--  TIME-SERIES HYPERTABLES
-- ═══════════════════════════════════════════════════════════

-- ── Risk Scores (raw, one row per 10s scoring window) ────

CREATE TABLE IF NOT EXISTS risk_scores (
    -- Time dimension (partition key — MUST be first in composite indexes)
    time                TIMESTAMPTZ NOT NULL,

    -- Identity
    user_id             TEXT        NOT NULL,
    device_id           TEXT        NOT NULL,
    session_id          TEXT        NOT NULL,

    -- Score breakdown
    risk_score          FLOAT       NOT NULL,  -- 0–100, phase-adjusted
    raw_risk_score      FLOAT       NOT NULL,  -- 0–100, before phase adjustment
    behavioral_dev      FLOAT       NOT NULL,  -- 0–50
    context_dev         FLOAT       NOT NULL,  -- 0–30
    device_risk         FLOAT       NOT NULL,  -- 0–20

    -- Model output
    similarity_score    FLOAT       NOT NULL,  -- 0.0–1.0
    classification      TEXT        NOT NULL,  -- LEGITIMATE | ATTACK | UNKNOWN
    confidence          FLOAT       NOT NULL,  -- 0.0–1.0

    -- Context
    risk_level          TEXT        NOT NULL,  -- LOW | ELEVATED | HIGH | CRITICAL
    phase               TEXT        NOT NULL,  -- OBSERVATION | SOFT | FULL
    sequence_ready      BOOLEAN     NOT NULL DEFAULT FALSE,
    eval_time_ms        FLOAT       NOT NULL DEFAULT 0
);

-- Convert to hypertable, partitioned by time, 1-day chunks
SELECT create_hypertable(
    'risk_scores', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists       => TRUE
);

-- Indexes — always include time first for hypertable efficiency
CREATE INDEX IF NOT EXISTS idx_rs_user_time    ON risk_scores(user_id,    time DESC);
CREATE INDEX IF NOT EXISTS idx_rs_session_time ON risk_scores(session_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_rs_level_time   ON risk_scores(risk_level, time DESC);

-- Data retention: drop chunks older than 90 days automatically
SELECT add_retention_policy(
    'risk_scores',
    INTERVAL '90 days',
    if_not_exists => TRUE
);

-- Compression: compress chunks older than 7 days (LZ4 — fast decompress for queries)
ALTER TABLE risk_scores SET (
    timescaledb.compress,
    timescaledb.compress_orderby    = 'time DESC',
    timescaledb.compress_segmentby  = 'user_id, session_id'
);

SELECT add_compression_policy(
    'risk_scores',
    INTERVAL '7 days',
    if_not_exists => TRUE
);

-- ── Enforcement Log (every action taken by agent) ────────

CREATE TABLE IF NOT EXISTS enforcement_log (
    time            TIMESTAMPTZ NOT NULL,

    -- Identity
    session_id      TEXT        NOT NULL,
    user_id         TEXT        NOT NULL,
    device_id       TEXT        NOT NULL,

    -- Decision
    risk_score      FLOAT       NOT NULL,
    risk_level      TEXT        NOT NULL,
    action          TEXT        NOT NULL,  -- ALLOW | STEP_UP_AUTH | THROTTLE | LOCK_SESSION | ALERT_ADMIN
    policy_id       TEXT        NOT NULL,
    reason          TEXT        NOT NULL DEFAULT '',
    escalated       BOOLEAN     NOT NULL DEFAULT FALSE,
    phase           TEXT        NOT NULL
);

SELECT create_hypertable(
    'enforcement_log', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists       => TRUE
);

CREATE INDEX IF NOT EXISTS idx_el_user_time    ON enforcement_log(user_id,    time DESC);
CREATE INDEX IF NOT EXISTS idx_el_session_time ON enforcement_log(session_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_el_action_time  ON enforcement_log(action,     time DESC);

SELECT add_retention_policy(
    'enforcement_log',
    INTERVAL '180 days',
    if_not_exists => TRUE
);

ALTER TABLE enforcement_log SET (
    timescaledb.compress,
    timescaledb.compress_orderby   = 'time DESC',
    timescaledb.compress_segmentby = 'user_id, action'
);

SELECT add_compression_policy(
    'enforcement_log',
    INTERVAL '7 days',
    if_not_exists => TRUE
);

-- ═══════════════════════════════════════════════════════════
--  CONTINUOUS AGGREGATES
--  Auto-refreshed materialized views for dashboard queries.
--  These answer "average risk score over the last hour" in
--  microseconds instead of scanning millions of raw rows.
-- ═══════════════════════════════════════════════════════════

-- ── 1-minute buckets ──────────────────────────────────────

CREATE MATERIALIZED VIEW IF NOT EXISTS risk_scores_1min
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', time)   AS bucket,
    user_id,
    session_id,
    AVG(risk_score)                 AS avg_risk_score,
    MAX(risk_score)                 AS max_risk_score,
    MIN(risk_score)                 AS min_risk_score,
    AVG(behavioral_dev)             AS avg_behavioral_dev,
    AVG(context_dev)                AS avg_context_dev,
    AVG(device_risk)                AS avg_device_risk,
    AVG(similarity_score)           AS avg_similarity,
    COUNT(*)                        AS event_count,
    -- Fraction of events at each level
    COUNT(*) FILTER (WHERE risk_level = 'CRITICAL') AS critical_count,
    COUNT(*) FILTER (WHERE risk_level = 'HIGH')     AS high_count,
    COUNT(*) FILTER (WHERE risk_level = 'ELEVATED') AS elevated_count,
    COUNT(*) FILTER (WHERE classification = 'ATTACK') AS attack_count
FROM risk_scores
GROUP BY bucket, user_id, session_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'risk_scores_1min',
    start_offset  => INTERVAL '10 minutes',
    end_offset    => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute',
    if_not_exists => TRUE
);

-- ── 1-hour buckets ────────────────────────────────────────

CREATE MATERIALIZED VIEW IF NOT EXISTS risk_scores_1hr
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time)     AS bucket,
    user_id,
    AVG(risk_score)                 AS avg_risk_score,
    MAX(risk_score)                 AS max_risk_score,
    MIN(risk_score)                 AS min_risk_score,
    AVG(behavioral_dev)             AS avg_behavioral_dev,
    AVG(context_dev)                AS avg_context_dev,
    AVG(device_risk)                AS avg_device_risk,
    AVG(similarity_score)           AS avg_similarity,
    COUNT(*)                        AS event_count,
    COUNT(*) FILTER (WHERE risk_level = 'CRITICAL') AS critical_count,
    COUNT(*) FILTER (WHERE risk_level = 'HIGH')     AS high_count,
    COUNT(*) FILTER (WHERE classification = 'ATTACK') AS attack_count
FROM risk_scores
GROUP BY bucket, user_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'risk_scores_1hr',
    start_offset  => INTERVAL '2 hours',
    end_offset    => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- ── 1-day buckets ─────────────────────────────────────────

CREATE MATERIALIZED VIEW IF NOT EXISTS risk_scores_1day
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time)      AS bucket,
    user_id,
    AVG(risk_score)                 AS avg_risk_score,
    MAX(risk_score)                 AS max_risk_score,
    COUNT(*)                        AS event_count,
    COUNT(*) FILTER (WHERE risk_level = 'CRITICAL') AS critical_count,
    COUNT(*) FILTER (WHERE classification = 'ATTACK') AS attack_count,
    COUNT(DISTINCT session_id)      AS session_count
FROM risk_scores
GROUP BY bucket, user_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'risk_scores_1day',
    start_offset  => INTERVAL '2 days',
    end_offset    => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Retention on aggregates (keep longer than raw — they're tiny)
SELECT add_retention_policy('risk_scores_1min',  INTERVAL '30 days',  if_not_exists => TRUE);
SELECT add_retention_policy('risk_scores_1hr',   INTERVAL '365 days', if_not_exists => TRUE);
SELECT add_retention_policy('risk_scores_1day',  INTERVAL '5 years',  if_not_exists => TRUE);

-- ── Enforcement summary (hourly) ──────────────────────────

CREATE MATERIALIZED VIEW IF NOT EXISTS enforcement_summary_1hr
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time)     AS bucket,
    user_id,
    action,
    COUNT(*)                        AS count,
    COUNT(*) FILTER (WHERE escalated = TRUE) AS escalated_count
FROM enforcement_log
GROUP BY bucket, user_id, action
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'enforcement_summary_1hr',
    start_offset  => INTERVAL '2 hours',
    end_offset    => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

SELECT add_retention_policy('enforcement_summary_1hr', INTERVAL '365 days', if_not_exists => TRUE);

-- ═══════════════════════════════════════════════════════════
--  HELPER VIEWS (non-materialized, for convenience)
-- ═══════════════════════════════════════════════════════════

-- Latest score per active session (for dashboard live view)
CREATE OR REPLACE VIEW active_session_scores AS
SELECT DISTINCT ON (rs.session_id)
    rs.session_id,
    rs.user_id,
    rs.device_id,
    rs.time         AS last_scored_at,
    rs.risk_score,
    rs.risk_level,
    rs.phase,
    rs.classification,
    rs.behavioral_dev,
    rs.context_dev,
    rs.device_risk,
    s.started_at,
    u.username
FROM risk_scores rs
JOIN sessions s ON s.session_id = rs.session_id AND s.active = TRUE
JOIN users    u ON u.user_id    = rs.user_id
ORDER BY rs.session_id, rs.time DESC;

-- User risk trend over the last 24 hours
CREATE OR REPLACE VIEW user_risk_24h AS
SELECT
    user_id,
    bucket,
    avg_risk_score,
    max_risk_score,
    event_count,
    critical_count,
    attack_count
FROM risk_scores_1hr
WHERE bucket >= NOW() - INTERVAL '24 hours'
ORDER BY user_id, bucket;

-- Top risky users right now (last 15 minutes)
CREATE OR REPLACE VIEW top_risk_users_15min AS
SELECT
    user_id,
    AVG(risk_score)  AS avg_score,
    MAX(risk_score)  AS max_score,
    COUNT(*)         AS events,
    MAX(time)        AS last_event
FROM risk_scores
WHERE time >= NOW() - INTERVAL '15 minutes'
GROUP BY user_id
ORDER BY avg_score DESC
LIMIT 20;
