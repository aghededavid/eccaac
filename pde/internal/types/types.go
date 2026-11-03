// ECCAAC — Policy Decision Engine
// Core domain types shared across all PDE packages.

package types

import "time"

// ─────────────────────────────────────────────
//  RISK SCORE EVENT
// ─────────────────────────────────────────────

// RiskEvent is a scored evaluation received from an agent.
// This is the primary input to the policy engine.
type RiskEvent struct {
	// Identity
	UserID    string `json:"user_id"`
	DeviceID  string `json:"device_id"`
	SessionID string `json:"session_id"`

	// Scores
	RiskScore           float64 `json:"risk_score"`
	RawRiskScore        float64 `json:"raw_risk_score"`
	BehavioralDeviation float64 `json:"behavioral_deviation"`
	ContextDeviation    float64 `json:"context_deviation"`
	DeviceRisk          float64 `json:"device_risk"`

	// Model output
	SimilarityScore float64 `json:"similarity_score"`
	Classification  string  `json:"classification"` // LEGITIMATE | ATTACK | UNKNOWN
	Confidence      float64 `json:"confidence"`

	// Phase
	Phase         string `json:"phase"` // OBSERVATION | SOFT | FULL
	SequenceReady bool   `json:"sequence_ready"`

	// Timing
	Timestamp  time.Time `json:"timestamp"`
	EvalTimeMS float64   `json:"eval_time_ms"`
}

// ─────────────────────────────────────────────
//  POLICY DECISION
// ─────────────────────────────────────────────

// Decision is the PDE's response to a RiskEvent —
// what enforcement actions should the agent apply.
type Decision struct {
	SessionID   string   `json:"session_id"`
	UserID      string   `json:"user_id"`
	RiskScore   float64  `json:"risk_score"`
	RiskLevel   string   `json:"risk_level"`
	Actions     []string `json:"actions"`        // e.g. ["THROTTLE","ALERT_ADMIN"]
	PolicyID    string   `json:"policy_id"`      // which policy rule fired
	Reason      string   `json:"reason"`         // human-readable explanation
	Timestamp   time.Time `json:"timestamp"`
	Escalated   bool     `json:"escalated"`      // true if escalated beyond base threshold
}

// ─────────────────────────────────────────────
//  USER SESSION
// ─────────────────────────────────────────────

// UserSession tracks the state of one active user session across
// all events received from the agent for that session.
type UserSession struct {
	SessionID  string    `json:"session_id"  db:"session_id"`
	UserID     string    `json:"user_id"     db:"user_id"`
	DeviceID   string    `json:"device_id"   db:"device_id"`
	Phase      string    `json:"phase"       db:"phase"`
	StartedAt  time.Time `json:"started_at"  db:"started_at"`
	LastSeenAt time.Time `json:"last_seen_at" db:"last_seen_at"`
	EndedAt    *time.Time `json:"ended_at"   db:"ended_at"`
	Active     bool      `json:"active"      db:"active"`

	// Running stats for the session
	EventCount    int     `json:"event_count"    db:"event_count"`
	AvgRiskScore  float64 `json:"avg_risk_score" db:"avg_risk_score"`
	MaxRiskScore  float64 `json:"max_risk_score" db:"max_risk_score"`
	CriticalCount int     `json:"critical_count" db:"critical_count"`
	WasFlagged    bool    `json:"was_flagged"    db:"was_flagged"`
}

// ─────────────────────────────────────────────
//  USER PROFILE
// ─────────────────────────────────────────────

// UserProfile holds persistent state for a user across all sessions.
type UserProfile struct {
	UserID         string    `json:"user_id"          db:"user_id"`
	Username       string    `json:"username"         db:"username"`
	Phase          string    `json:"phase"            db:"phase"`
	SessionCount   int       `json:"session_count"    db:"session_count"`
	CreatedAt      time.Time `json:"created_at"       db:"created_at"`
	LastSeenAt     time.Time `json:"last_seen_at"     db:"last_seen_at"`
	PolicyID       string    `json:"policy_id"        db:"policy_id"` // assigned policy group
	IsAdmin        bool      `json:"is_admin"         db:"is_admin"`
	IsSuspended    bool      `json:"is_suspended"     db:"is_suspended"`
	SuspendedUntil *time.Time `json:"suspended_until" db:"suspended_until"`
	Notes          string    `json:"notes"            db:"notes"`
}

// PhaseFor returns the phase a user should be in given their session count.
func PhaseFor(sessionCount int) string {
	switch {
	case sessionCount <= 5:
		return PhaseObservation
	case sessionCount <= 10:
		return PhaseSoft
	default:
		return PhaseFull
	}
}

// ─────────────────────────────────────────────
//  POLICY RULE
// ─────────────────────────────────────────────

// PolicyRule is a named rule that can be assigned to users or devices.
// Rules are evaluated as expressions against a RiskEvent context.
type PolicyRule struct {
	ID          string    `json:"id"           db:"id"`
	Name        string    `json:"name"         db:"name"`
	Description string    `json:"description"  db:"description"`
	Priority    int       `json:"priority"     db:"priority"` // lower = evaluated first
	Enabled     bool      `json:"enabled"      db:"enabled"`
	Scope       string    `json:"scope"`       // "global" | "user" | "device"
	TargetID    string    `json:"target_id"`   // user_id or device_id if scoped

	// Thresholds — override global defaults for this rule
	LowMax      float64 `json:"low_max"      db:"low_max"`
	ElevatedMax float64 `json:"elevated_max" db:"elevated_max"`
	HighMax     float64 `json:"high_max"     db:"high_max"`

	// Escalation condition — expr-lang expression
	// Context variables: score, level, behav, ctx_dev, dev_risk, classification, confidence, phase
	// Example: "score > 40 && classification == 'ATTACK'"
	EscalateCondition string   `json:"escalate_condition" db:"escalate_condition"`
	EscalateActions   []string `json:"escalate_actions"   db:"escalate_actions"`

	CreatedAt time.Time `json:"created_at" db:"created_at"`
	UpdatedAt time.Time `json:"updated_at" db:"updated_at"`
}

// ─────────────────────────────────────────────
//  POLICY GROUP (PRESET)
// ─────────────────────────────────────────────

// PolicyGroup bundles a set of threshold + escalation settings.
// Users are assigned to a group.
type PolicyGroup struct {
	ID          string  `json:"id"          db:"id"`
	Name        string  `json:"name"        db:"name"`
	Description string  `json:"description" db:"description"`
	LowMax      float64 `json:"low_max"     db:"low_max"`
	ElevatedMax float64 `json:"elevated_max" db:"elevated_max"`
	HighMax     float64 `json:"high_max"    db:"high_max"`
	SoftDivisor float64 `json:"soft_divisor" db:"soft_divisor"`
}

// Default policy groups
var (
	PolicyGroupDefault = PolicyGroup{
		ID: "default", Name: "Default",
		LowMax: 25, ElevatedMax: 50, HighMax: 75, SoftDivisor: 2.0,
	}
	PolicyGroupStrict = PolicyGroup{
		ID: "strict", Name: "Strict (Servers / Privileged)",
		LowMax: 15, ElevatedMax: 35, HighMax: 60, SoftDivisor: 1.5,
	}
	PolicyGroupRelaxed = PolicyGroup{
		ID: "relaxed", Name: "Relaxed (Low-risk devices)",
		LowMax: 35, ElevatedMax: 60, HighMax: 80, SoftDivisor: 2.5,
	}
)

// ─────────────────────────────────────────────
//  AUDIT LOG ENTRY
// ─────────────────────────────────────────────

// AuditEntry is a record of one PDE decision for audit and forensics.
type AuditEntry struct {
	ID         int64     `json:"id"          db:"id"`
	SessionID  string    `json:"session_id"  db:"session_id"`
	UserID     string    `json:"user_id"     db:"user_id"`
	DeviceID   string    `json:"device_id"   db:"device_id"`
	RiskScore  float64   `json:"risk_score"  db:"risk_score"`
	RiskLevel  string    `json:"risk_level"  db:"risk_level"`
	Actions    []string  `json:"actions"     db:"actions"`
	PolicyID   string    `json:"policy_id"   db:"policy_id"`
	Reason     string    `json:"reason"      db:"reason"`
	Escalated  bool      `json:"escalated"   db:"escalated"`
	Phase      string    `json:"phase"       db:"phase"`
	Timestamp  time.Time `json:"timestamp"   db:"timestamp"`
}

// ─────────────────────────────────────────────
//  CONSTANTS
// ─────────────────────────────────────────────

const (
	// Risk levels
	LevelLow      = "LOW"
	LevelElevated = "ELEVATED"
	LevelHigh     = "HIGH"
	LevelCritical = "CRITICAL"

	// Enforcement actions
	ActionAllow       = "ALLOW"
	ActionStepUpAuth  = "STEP_UP_AUTH"
	ActionThrottle    = "THROTTLE"
	ActionLockSession = "LOCK_SESSION"
	ActionAlertAdmin  = "ALERT_ADMIN"

	// Phases
	PhaseObservation = "OBSERVATION"
	PhaseSoft        = "SOFT"
	PhaseFull        = "FULL"

	// Classification labels
	ClassLegitimate = "LEGITIMATE"
	ClassAttack     = "ATTACK"
	ClassUnknown    = "UNKNOWN"
)
