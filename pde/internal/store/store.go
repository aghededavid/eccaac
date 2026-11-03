// ECCAAC — Policy Decision Engine
// Database Store — TimescaleDB
//
// Two write paths:
//   Hot path (async, buffered):  risk_scores, enforcement_log
//   Cold path (sync):            users, sessions, policy_rules (low frequency)
//
// Query path:
//   Raw hypertable queries for live data (last N minutes)
//   Continuous aggregate queries for trend/dashboard data
//   Plain SQL for relational data (users, sessions, rules)

package store

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"go.uber.org/zap"

	"github.com/eccaac/pde/internal/types"
)

// ─────────────────────────────────────────────
//  STORE
// ─────────────────────────────────────────────

// Store manages all database operations against TimescaleDB.
type Store struct {
	pool    *pgxpool.Pool
	log     *zap.Logger
	writeCh chan writeOp
}

type writeOp struct {
	kind string
	data interface{}
}

// New connects to TimescaleDB and returns a ready Store.
func New(dsn string, maxConns int32, log *zap.Logger) (*Store, error) {
	cfg, err := pgxpool.ParseConfig(dsn)
	if err != nil {
		return nil, fmt.Errorf("parse dsn: %w", err)
	}
	cfg.MaxConns = maxConns

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("create pool: %w", err)
	}
	if err := pool.Ping(ctx); err != nil {
		return nil, fmt.Errorf("ping: %w", err)
	}

	s := &Store{
		pool:    pool,
		log:     log,
		writeCh: make(chan writeOp, 4096),
	}
	go s.asyncWriter()
	return s, nil
}

// Migrate runs the full schema DDL (idempotent).
func (s *Store) Migrate(ctx context.Context) error {
	_, err := s.pool.Exec(ctx, schemaDDL)
	return err
}

// Close shuts down the store gracefully.
func (s *Store) Close() {
	close(s.writeCh)
	s.pool.Close()
}

// ─────────────────────────────────────────────
//  ASYNC WRITE PATH
// ─────────────────────────────────────────────

func (s *Store) asyncWriter() {
	for op := range s.writeCh {
		var err error
		switch op.kind {
		case "risk_score":
			err = s.writeRiskScore(op.data.(riskScoreRow))
		case "enforcement":
			err = s.writeEnforcement(op.data.(enforcementRow))
		case "session_start":
			err = s.writeSessionStart(op.data.(types.UserSession))
		case "session_end":
			err = s.writeSessionEnd(op.data.(types.UserSession))
		case "user_upsert":
			err = s.writeUserUpsert(op.data.(types.UserProfile))
		}
		if err != nil {
			s.log.Warn("Async DB write failed", zap.String("kind", op.kind), zap.Error(err))
		}
	}
}

func (s *Store) enqueue(kind string, data interface{}) {
	select {
	case s.writeCh <- writeOp{kind, data}:
	default:
		s.log.Warn("DB write queue full — dropping", zap.String("kind", kind))
	}
}

// ─────────────────────────────────────────────
//  ROW TYPES (internal)
// ─────────────────────────────────────────────

type riskScoreRow struct {
	Time, UserID, DeviceID, SessionID string
	RiskScore, RawRiskScore           float64
	BehavioralDev, ContextDev         float64
	DeviceRisk, SimilarityScore       float64
	Classification                    string
	Confidence                        float64
	RiskLevel, Phase                  string
	SequenceReady                     bool
	EvalTimeMS                        float64
	Timestamp                         time.Time
}

type enforcementRow struct {
	Timestamp time.Time
	SessionID, UserID, DeviceID string
	RiskScore                   float64
	RiskLevel, Action, PolicyID string
	Reason                      string
	Escalated                   bool
	Phase                       string
}

// ─────────────────────────────────────────────
//  PUBLIC ASYNC WRITES
// ─────────────────────────────────────────────

// RecordDecision writes one enforcement row per action + one risk_score row.
func (s *Store) RecordDecision(evt types.RiskEvent, d types.Decision) {
	for _, action := range d.Actions {
		s.enqueue("enforcement", enforcementRow{
			Timestamp: d.Timestamp, SessionID: d.SessionID,
			UserID: d.UserID, DeviceID: evt.DeviceID,
			RiskScore: d.RiskScore, RiskLevel: d.RiskLevel,
			Action: action, PolicyID: d.PolicyID,
			Reason: d.Reason, Escalated: d.Escalated, Phase: evt.Phase,
		})
	}
	s.enqueue("risk_score", riskScoreRow{
		Timestamp: evt.Timestamp, UserID: evt.UserID,
		DeviceID: evt.DeviceID, SessionID: evt.SessionID,
		RiskScore: evt.RiskScore, RawRiskScore: evt.RawRiskScore,
		BehavioralDev: evt.BehavioralDeviation, ContextDev: evt.ContextDeviation,
		DeviceRisk: evt.DeviceRisk, SimilarityScore: evt.SimilarityScore,
		Classification: evt.Classification, Confidence: evt.Confidence,
		RiskLevel: d.RiskLevel, Phase: evt.Phase,
		SequenceReady: evt.SequenceReady, EvalTimeMS: evt.EvalTimeMS,
	})
}

// RecordSessionStart asynchronously writes a new session row.
func (s *Store) RecordSessionStart(sess types.UserSession) { s.enqueue("session_start", sess) }

// RecordSessionEnd asynchronously finalises a session row.
func (s *Store) RecordSessionEnd(sess types.UserSession) { s.enqueue("session_end", sess) }

// UpsertUser asynchronously writes/updates a user profile.
func (s *Store) UpsertUser(u types.UserProfile) { s.enqueue("user_upsert", u) }

// ─────────────────────────────────────────────
//  PRIVATE WRITE HELPERS
// ─────────────────────────────────────────────

func (s *Store) writeRiskScore(r riskScoreRow) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, err := s.pool.Exec(ctx, `
		INSERT INTO risk_scores
			(time,user_id,device_id,session_id,
			 risk_score,raw_risk_score,behavioral_dev,context_dev,device_risk,
			 similarity_score,classification,confidence,
			 risk_level,phase,sequence_ready,eval_time_ms)
		VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16)`,
		r.Timestamp, r.UserID, r.DeviceID, r.SessionID,
		r.RiskScore, r.RawRiskScore, r.BehavioralDev, r.ContextDev, r.DeviceRisk,
		r.SimilarityScore, r.Classification, r.Confidence,
		r.RiskLevel, r.Phase, r.SequenceReady, r.EvalTimeMS,
	)
	return err
}

func (s *Store) writeEnforcement(r enforcementRow) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, err := s.pool.Exec(ctx, `
		INSERT INTO enforcement_log
			(time,session_id,user_id,device_id,
			 risk_score,risk_level,action,policy_id,reason,escalated,phase)
		VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)`,
		r.Timestamp, r.SessionID, r.UserID, r.DeviceID,
		r.RiskScore, r.RiskLevel, r.Action, r.PolicyID, r.Reason, r.Escalated, r.Phase,
	)
	return err
}

func (s *Store) writeSessionStart(sess types.UserSession) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, err := s.pool.Exec(ctx, `
		INSERT INTO sessions
			(session_id,user_id,device_id,phase,started_at,last_seen_at,active,
			 event_count,avg_risk_score,max_risk_score,critical_count,was_flagged)
		VALUES ($1,$2,$3,$4,$5,$5,TRUE,0,0,0,0,FALSE)
		ON CONFLICT (session_id) DO NOTHING`,
		sess.SessionID, sess.UserID, sess.DeviceID, sess.Phase, sess.StartedAt,
	)
	return err
}

func (s *Store) writeSessionEnd(sess types.UserSession) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, err := s.pool.Exec(ctx, `
		UPDATE sessions SET
			active=$1, ended_at=$2, last_seen_at=$3, phase=$4,
			event_count=$5, avg_risk_score=$6, max_risk_score=$7,
			critical_count=$8, was_flagged=$9
		WHERE session_id=$10`,
		false, sess.EndedAt, sess.LastSeenAt, sess.Phase,
		sess.EventCount, sess.AvgRiskScore, sess.MaxRiskScore,
		sess.CriticalCount, sess.WasFlagged, sess.SessionID,
	)
	return err
}

func (s *Store) writeUserUpsert(u types.UserProfile) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, err := s.pool.Exec(ctx, `
		INSERT INTO users (user_id,username,phase,session_count,policy_id,is_admin,created_at,last_seen_at)
		VALUES ($1,$2,$3,$4,$5,$6,NOW(),NOW())
		ON CONFLICT (user_id) DO UPDATE SET
			username=$2, phase=$3, session_count=$4,
			policy_id=$5, is_admin=$6, last_seen_at=NOW()`,
		u.UserID, u.Username, u.Phase, u.SessionCount, u.PolicyID, u.IsAdmin,
	)
	return err
}

// ─────────────────────────────────────────────
//  TIME-SERIES QUERY PATH
// ─────────────────────────────────────────────

// RiskScorePoint is one data point in a time-series result.
type RiskScorePoint struct {
	Time      time.Time `json:"time"`
	Score     float64   `json:"score"`
	Level     string    `json:"level"`
}

// GetRiskScoresRaw returns raw score events for a session over the given lookback.
func (s *Store) GetRiskScoresRaw(ctx context.Context, sessionID string, lookback time.Duration) ([]RiskScorePoint, error) {
	rows, err := s.pool.Query(ctx, `
		SELECT time, risk_score, risk_level FROM risk_scores
		WHERE session_id=$1 AND time >= NOW()-$2::interval
		ORDER BY time ASC`,
		sessionID, lookback.String(),
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var pts []RiskScorePoint
	for rows.Next() {
		var p RiskScorePoint
		if err := rows.Scan(&p.Time, &p.Score, &p.Level); err != nil {
			return nil, err
		}
		pts = append(pts, p)
	}
	return pts, rows.Err()
}

// BucketedSeries is one row from a continuous aggregate.
type BucketedSeries struct {
	Bucket        time.Time `json:"bucket"`
	AvgRiskScore  float64   `json:"avg_risk_score"`
	MaxRiskScore  float64   `json:"max_risk_score"`
	EventCount    int64     `json:"event_count"`
	CriticalCount int64     `json:"critical_count"`
	AttackCount   int64     `json:"attack_count"`
}

// GetRiskTrend picks the right continuous aggregate for the time range and returns bucketed data.
// ≤2h → 1-min buckets | ≤7d → 1-hr buckets | >7d → 1-day buckets
func (s *Store) GetRiskTrend(ctx context.Context, userID string, from, to time.Time) ([]BucketedSeries, error) {
	span := to.Sub(from)
	var view, bucket string
	switch {
	case span <= 2*time.Hour:
		view, bucket = "risk_scores_1min", "1 minute"
	case span <= 7*24*time.Hour:
		view, bucket = "risk_scores_1hr", "1 hour"
	default:
		view, bucket = "risk_scores_1day", "1 day"
	}
	q := fmt.Sprintf(`
		SELECT time_bucket('%s',bucket) AS b,
		       AVG(avg_risk_score), MAX(max_risk_score),
		       SUM(event_count), SUM(critical_count), SUM(attack_count)
		FROM %s WHERE user_id=$1 AND bucket>=$2 AND bucket<$3
		GROUP BY 1 ORDER BY 1 ASC`,
		bucket, view,
	)
	rows, err := s.pool.Query(ctx, q, userID, from, to)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var series []BucketedSeries
	for rows.Next() {
		var b BucketedSeries
		if err := rows.Scan(&b.Bucket, &b.AvgRiskScore, &b.MaxRiskScore, &b.EventCount, &b.CriticalCount, &b.AttackCount); err != nil {
			return nil, err
		}
		series = append(series, b)
	}
	return series, rows.Err()
}

// GetEnforcementHistory returns recent enforcement rows, optionally filtered by user.
func (s *Store) GetEnforcementHistory(ctx context.Context, userID string, limit int) ([]types.AuditEntry, error) {
	if limit <= 0 || limit > 1000 {
		limit = 100
	}
	var rows interface{ Close(); Next() bool; Scan(...interface{}) error; Err() error }
	if userID != "" {
		r, err := s.pool.Query(ctx, `
			SELECT time,session_id,user_id,device_id,risk_score,risk_level,action,policy_id,reason,escalated,phase
			FROM enforcement_log WHERE user_id=$1 ORDER BY time DESC LIMIT $2`,
			userID, limit,
		)
		if err != nil {
			return nil, err
		}
		rows = r
	} else {
		r, err := s.pool.Query(ctx, `
			SELECT time,session_id,user_id,device_id,risk_score,risk_level,action,policy_id,reason,escalated,phase
			FROM enforcement_log ORDER BY time DESC LIMIT $1`,
			limit,
		)
		if err != nil {
			return nil, err
		}
		rows = r
	}
	defer rows.Close()
	var entries []types.AuditEntry
	for rows.Next() {
		var e types.AuditEntry
		var action string
		if err := rows.Scan(&e.Timestamp, &e.SessionID, &e.UserID, &e.DeviceID,
			&e.RiskScore, &e.RiskLevel, &action, &e.PolicyID, &e.Reason, &e.Escalated, &e.Phase,
		); err != nil {
			return nil, err
		}
		e.Actions = []string{action}
		entries = append(entries, e)
	}
	return entries, rows.Err()
}

// ─────────────────────────────────────────────
//  RELATIONAL QUERY PATH
// ─────────────────────────────────────────────

func (s *Store) GetUsers(ctx context.Context) ([]types.UserProfile, error) {
	rows, err := s.pool.Query(ctx, `
		SELECT user_id,username,phase,session_count,policy_id,is_admin,
		       is_suspended,suspended_until,notes,created_at,last_seen_at
		FROM users ORDER BY last_seen_at DESC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var users []types.UserProfile
	for rows.Next() {
		var u types.UserProfile
		if err := rows.Scan(&u.UserID, &u.Username, &u.Phase, &u.SessionCount,
			&u.PolicyID, &u.IsAdmin, &u.IsSuspended, &u.SuspendedUntil, &u.Notes,
			&u.CreatedAt, &u.LastSeenAt,
		); err != nil {
			return nil, err
		}
		users = append(users, u)
	}
	return users, rows.Err()
}

func (s *Store) GetUser(ctx context.Context, userID string) (*types.UserProfile, error) {
	var u types.UserProfile
	err := s.pool.QueryRow(ctx, `
		SELECT user_id,username,phase,session_count,policy_id,is_admin,
		       is_suspended,suspended_until,notes,created_at,last_seen_at
		FROM users WHERE user_id=$1`, userID,
	).Scan(&u.UserID, &u.Username, &u.Phase, &u.SessionCount,
		&u.PolicyID, &u.IsAdmin, &u.IsSuspended, &u.SuspendedUntil, &u.Notes,
		&u.CreatedAt, &u.LastSeenAt,
	)
	if err != nil {
		return nil, err
	}
	return &u, nil
}

func (s *Store) GetSessions(ctx context.Context, activeOnly bool, limit int) ([]types.UserSession, error) {
	if limit <= 0 || limit > 500 {
		limit = 100
	}
	q := `SELECT session_id,user_id,device_id,phase,started_at,last_seen_at,ended_at,active,
	             event_count,avg_risk_score,max_risk_score,critical_count,was_flagged
	      FROM sessions`
	if activeOnly {
		q += " WHERE active=TRUE"
	}
	q += " ORDER BY started_at DESC LIMIT $1"
	rows, err := s.pool.Query(ctx, q, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var sessions []types.UserSession
	for rows.Next() {
		var s types.UserSession
		if err := rows.Scan(&s.SessionID, &s.UserID, &s.DeviceID, &s.Phase,
			&s.StartedAt, &s.LastSeenAt, &s.EndedAt, &s.Active,
			&s.EventCount, &s.AvgRiskScore, &s.MaxRiskScore, &s.CriticalCount, &s.WasFlagged,
		); err != nil {
			return nil, err
		}
		sessions = append(sessions, s)
	}
	return sessions, rows.Err()
}

func (s *Store) GetPolicyRules(ctx context.Context) ([]types.PolicyRule, error) {
	rows, err := s.pool.Query(ctx, `
		SELECT id,name,description,priority,enabled,scope,target_id,
		       low_max,elevated_max,high_max,escalate_condition,escalate_actions,created_at,updated_at
		FROM policy_rules ORDER BY priority ASC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var rules []types.PolicyRule
	for rows.Next() {
		var r types.PolicyRule
		var actJSON []byte
		if err := rows.Scan(&r.ID, &r.Name, &r.Description, &r.Priority, &r.Enabled, &r.Scope, &r.TargetID,
			&r.LowMax, &r.ElevatedMax, &r.HighMax, &r.EscalateCondition, &actJSON,
			&r.CreatedAt, &r.UpdatedAt,
		); err != nil {
			return nil, err
		}
		_ = json.Unmarshal(actJSON, &r.EscalateActions)
		rules = append(rules, r)
	}
	return rules, rows.Err()
}

func (s *Store) UpsertPolicyRule(ctx context.Context, rule types.PolicyRule) error {
	actJSON, _ := json.Marshal(rule.EscalateActions)
	_, err := s.pool.Exec(ctx, `
		INSERT INTO policy_rules
			(id,name,description,priority,enabled,scope,target_id,
			 low_max,elevated_max,high_max,escalate_condition,escalate_actions,created_at,updated_at)
		VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,NOW(),NOW())
		ON CONFLICT (id) DO UPDATE SET
			name=$2,description=$3,priority=$4,enabled=$5,
			low_max=$8,elevated_max=$9,high_max=$10,
			escalate_condition=$11,escalate_actions=$12,updated_at=NOW()`,
		rule.ID, rule.Name, rule.Description, rule.Priority, rule.Enabled, rule.Scope, rule.TargetID,
		rule.LowMax, rule.ElevatedMax, rule.HighMax, rule.EscalateCondition, actJSON,
	)
	return err
}

func (s *Store) SuspendUser(ctx context.Context, userID string, until *time.Time) error {
	_, err := s.pool.Exec(ctx, `UPDATE users SET is_suspended=TRUE, suspended_until=$1 WHERE user_id=$2`, until, userID)
	return err
}

func (s *Store) UnsuspendUser(ctx context.Context, userID string) error {
	_, err := s.pool.Exec(ctx, `UPDATE users SET is_suspended=FALSE, suspended_until=NULL WHERE user_id=$1`, userID)
	return err
}

// ─────────────────────────────────────────────
//  SCHEMA DDL
// ─────────────────────────────────────────────

var schemaDDL = strings.TrimSpace(`
CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS policy_groups (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT NULL DEFAULT '',
    low_max FLOAT NOT NULL DEFAULT 25, elevated_max FLOAT NOT NULL DEFAULT 50,
    high_max FLOAT NOT NULL DEFAULT 75, soft_divisor FLOAT NOT NULL DEFAULT 2.0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
INSERT INTO policy_groups (id,name,description,low_max,elevated_max,high_max,soft_divisor) VALUES
    ('default','Default','Standard thresholds',25,50,75,2.0),
    ('strict','Strict (Privileged/Server)','Tight thresholds',15,35,60,1.5),
    ('relaxed','Relaxed (Low-risk devices)','Relaxed thresholds',35,60,80,2.5)
ON CONFLICT (id) DO NOTHING;

CREATE TABLE IF NOT EXISTS policy_rules (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT NULL DEFAULT '',
    priority INT NOT NULL DEFAULT 100, enabled BOOLEAN NOT NULL DEFAULT TRUE,
    scope TEXT NOT NULL DEFAULT 'global', target_id TEXT NOT NULL DEFAULT '',
    low_max FLOAT NOT NULL DEFAULT 25, elevated_max FLOAT NOT NULL DEFAULT 50, high_max FLOAT NOT NULL DEFAULT 75,
    escalate_condition TEXT NOT NULL DEFAULT '', escalate_actions JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_policy_rules_priority ON policy_rules(priority ASC) WHERE enabled=TRUE;

CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY, username TEXT NOT NULL DEFAULT '', domain TEXT NOT NULL DEFAULT '',
    phase TEXT NOT NULL DEFAULT 'OBSERVATION', session_count INT NOT NULL DEFAULT 0,
    policy_id TEXT NOT NULL DEFAULT 'default' REFERENCES policy_groups(id),
    is_admin BOOLEAN NOT NULL DEFAULT FALSE, is_suspended BOOLEAN NOT NULL DEFAULT FALSE,
    suspended_until TIMESTAMPTZ, notes TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_users_last_seen ON users(last_seen_at DESC);
CREATE INDEX IF NOT EXISTS idx_users_suspended ON users(is_suspended) WHERE is_suspended=TRUE;

CREATE TABLE IF NOT EXISTS devices (
    device_id TEXT PRIMARY KEY, hostname TEXT NOT NULL DEFAULT '',
    os_version TEXT NOT NULL DEFAULT '', domain_joined BOOLEAN NOT NULL DEFAULT FALSE,
    policy_id TEXT NOT NULL DEFAULT 'default' REFERENCES policy_groups(id),
    is_trusted BOOLEAN NOT NULL DEFAULT TRUE, notes TEXT NOT NULL DEFAULT '',
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    device_id TEXT NOT NULL, phase TEXT NOT NULL DEFAULT 'OBSERVATION',
    started_at TIMESTAMPTZ NOT NULL, last_seen_at TIMESTAMPTZ NOT NULL,
    ended_at TIMESTAMPTZ, active BOOLEAN NOT NULL DEFAULT TRUE,
    event_count INT NOT NULL DEFAULT 0, avg_risk_score FLOAT NOT NULL DEFAULT 0,
    max_risk_score FLOAT NOT NULL DEFAULT 0, critical_count INT NOT NULL DEFAULT 0,
    was_flagged BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_sessions_user    ON sessions(user_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_active  ON sessions(active, started_at DESC) WHERE active=TRUE;
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC);

CREATE TABLE IF NOT EXISTS risk_scores (
    time TIMESTAMPTZ NOT NULL,
    user_id TEXT NOT NULL, device_id TEXT NOT NULL, session_id TEXT NOT NULL,
    risk_score FLOAT NOT NULL, raw_risk_score FLOAT NOT NULL,
    behavioral_dev FLOAT NOT NULL, context_dev FLOAT NOT NULL, device_risk FLOAT NOT NULL,
    similarity_score FLOAT NOT NULL, classification TEXT NOT NULL, confidence FLOAT NOT NULL,
    risk_level TEXT NOT NULL, phase TEXT NOT NULL,
    sequence_ready BOOLEAN NOT NULL DEFAULT FALSE, eval_time_ms FLOAT NOT NULL DEFAULT 0
);
SELECT create_hypertable('risk_scores','time',chunk_time_interval=>INTERVAL '1 day',if_not_exists=>TRUE);
CREATE INDEX IF NOT EXISTS idx_rs_user_time    ON risk_scores(user_id,    time DESC);
CREATE INDEX IF NOT EXISTS idx_rs_session_time ON risk_scores(session_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_rs_level_time   ON risk_scores(risk_level, time DESC);
SELECT add_retention_policy('risk_scores', INTERVAL '90 days', if_not_exists=>TRUE);
ALTER TABLE risk_scores SET (timescaledb.compress, timescaledb.compress_orderby='time DESC', timescaledb.compress_segmentby='user_id, session_id');
SELECT add_compression_policy('risk_scores', INTERVAL '7 days', if_not_exists=>TRUE);

CREATE TABLE IF NOT EXISTS enforcement_log (
    time TIMESTAMPTZ NOT NULL,
    session_id TEXT NOT NULL, user_id TEXT NOT NULL, device_id TEXT NOT NULL,
    risk_score FLOAT NOT NULL, risk_level TEXT NOT NULL, action TEXT NOT NULL,
    policy_id TEXT NOT NULL, reason TEXT NOT NULL DEFAULT '',
    escalated BOOLEAN NOT NULL DEFAULT FALSE, phase TEXT NOT NULL
);
SELECT create_hypertable('enforcement_log','time',chunk_time_interval=>INTERVAL '1 day',if_not_exists=>TRUE);
CREATE INDEX IF NOT EXISTS idx_el_user_time    ON enforcement_log(user_id,    time DESC);
CREATE INDEX IF NOT EXISTS idx_el_session_time ON enforcement_log(session_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_el_action_time  ON enforcement_log(action,     time DESC);
SELECT add_retention_policy('enforcement_log', INTERVAL '180 days', if_not_exists=>TRUE);
ALTER TABLE enforcement_log SET (timescaledb.compress, timescaledb.compress_orderby='time DESC', timescaledb.compress_segmentby='user_id, action');
SELECT add_compression_policy('enforcement_log', INTERVAL '7 days', if_not_exists=>TRUE);

CREATE MATERIALIZED VIEW IF NOT EXISTS risk_scores_1min WITH (timescaledb.continuous) AS
SELECT time_bucket('1 minute',time) bucket, user_id, session_id,
       AVG(risk_score) avg_risk_score, MAX(risk_score) max_risk_score, MIN(risk_score) min_risk_score,
       AVG(behavioral_dev) avg_behavioral_dev, AVG(context_dev) avg_context_dev,
       AVG(device_risk) avg_device_risk, AVG(similarity_score) avg_similarity, COUNT(*) event_count,
       COUNT(*) FILTER (WHERE risk_level='CRITICAL') critical_count,
       COUNT(*) FILTER (WHERE risk_level='HIGH') high_count,
       COUNT(*) FILTER (WHERE risk_level='ELEVATED') elevated_count,
       COUNT(*) FILTER (WHERE classification='ATTACK') attack_count
FROM risk_scores GROUP BY 1,2,3 WITH NO DATA;
SELECT add_continuous_aggregate_policy('risk_scores_1min',start_offset=>'10 minutes',end_offset=>'1 minute',schedule_interval=>'1 minute',if_not_exists=>TRUE);

CREATE MATERIALIZED VIEW IF NOT EXISTS risk_scores_1hr WITH (timescaledb.continuous) AS
SELECT time_bucket('1 hour',time) bucket, user_id,
       AVG(risk_score) avg_risk_score, MAX(risk_score) max_risk_score, MIN(risk_score) min_risk_score,
       AVG(behavioral_dev) avg_behavioral_dev, AVG(context_dev) avg_context_dev,
       AVG(device_risk) avg_device_risk, AVG(similarity_score) avg_similarity, COUNT(*) event_count,
       COUNT(*) FILTER (WHERE risk_level='CRITICAL') critical_count,
       COUNT(*) FILTER (WHERE risk_level='HIGH') high_count,
       COUNT(*) FILTER (WHERE classification='ATTACK') attack_count
FROM risk_scores GROUP BY 1,2 WITH NO DATA;
SELECT add_continuous_aggregate_policy('risk_scores_1hr',start_offset=>'2 hours',end_offset=>'1 hour',schedule_interval=>'1 hour',if_not_exists=>TRUE);

CREATE MATERIALIZED VIEW IF NOT EXISTS risk_scores_1day WITH (timescaledb.continuous) AS
SELECT time_bucket('1 day',time) bucket, user_id,
       AVG(risk_score) avg_risk_score, MAX(risk_score) max_risk_score,
       COUNT(*) event_count, COUNT(*) FILTER (WHERE risk_level='CRITICAL') critical_count,
       COUNT(*) FILTER (WHERE classification='ATTACK') attack_count,
       COUNT(DISTINCT session_id) session_count
FROM risk_scores GROUP BY 1,2 WITH NO DATA;
SELECT add_continuous_aggregate_policy('risk_scores_1day',start_offset=>'2 days',end_offset=>'1 day',schedule_interval=>'1 day',if_not_exists=>TRUE);

CREATE MATERIALIZED VIEW IF NOT EXISTS enforcement_summary_1hr WITH (timescaledb.continuous) AS
SELECT time_bucket('1 hour',time) bucket, user_id, action,
       COUNT(*) count, COUNT(*) FILTER (WHERE escalated=TRUE) escalated_count
FROM enforcement_log GROUP BY 1,2,3 WITH NO DATA;
SELECT add_continuous_aggregate_policy('enforcement_summary_1hr',start_offset=>'2 hours',end_offset=>'1 hour',schedule_interval=>'1 hour',if_not_exists=>TRUE);

SELECT add_retention_policy('risk_scores_1min',  INTERVAL '30 days',  if_not_exists=>TRUE);
SELECT add_retention_policy('risk_scores_1hr',   INTERVAL '365 days', if_not_exists=>TRUE);
SELECT add_retention_policy('risk_scores_1day',  INTERVAL '5 years',  if_not_exists=>TRUE);
SELECT add_retention_policy('enforcement_summary_1hr', INTERVAL '365 days', if_not_exists=>TRUE);
`)
