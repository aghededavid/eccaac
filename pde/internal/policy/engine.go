// ECCAAC — Policy Decision Engine
// Policy Engine
//
// Evaluates every incoming RiskEvent against a layered set of policy rules
// and produces a Decision (what enforcement actions to apply).
//
// Evaluation order (first match wins for escalation):
//   1. Suspended user check  → immediate LOCK
//   2. Device-scoped rules   → device-specific overrides
//   3. User-scoped rules     → per-user overrides
//   4. Policy group rules    → group thresholds
//   5. Global default        → fallback thresholds
//
// Each layer can:
//   - Override thresholds (low/elevated/high max)
//   - Define custom escalation conditions (expr-lang expressions)
//   - Produce additional actions beyond the base set

package policy

import (
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/eccaac/pde/internal/config"
	"github.com/eccaac/pde/internal/session"
	"github.com/eccaac/pde/internal/types"
	"github.com/expr-lang/expr"
	"github.com/expr-lang/expr/vm"
	"go.uber.org/zap"
)

// ─────────────────────────────────────────────
//  POLICY STORE
// ─────────────────────────────────────────────

// Store holds all policy rules and groups in memory.
// Rules are loaded from the database and refreshed on policy sync.
type Store struct {
	mu      sync.RWMutex
	rules   []types.PolicyRule   // sorted by priority ascending
	groups  map[string]types.PolicyGroup
	users   map[string]types.UserProfile
	log     *zap.Logger

	// Compiled escalation condition programs (cached per rule ID)
	compiled map[string]*vm.Program
}

// NewStore creates a policy store pre-loaded with built-in policy groups.
func NewStore(log *zap.Logger) *Store {
	s := &Store{
		log:      log,
		groups:   make(map[string]types.PolicyGroup),
		users:    make(map[string]types.UserProfile),
		compiled: make(map[string]*vm.Program),
	}
	// Seed built-in groups
	for _, g := range []types.PolicyGroup{
		types.PolicyGroupDefault,
		types.PolicyGroupStrict,
		types.PolicyGroupRelaxed,
	} {
		s.groups[g.ID] = g
	}
	return s
}

// LoadRules replaces all rules atomically (called on DB refresh).
func (s *Store) LoadRules(rules []types.PolicyRule) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.rules = make([]types.PolicyRule, len(rules))
	copy(s.rules, rules)
	// Pre-compile escalation conditions
	for _, r := range s.rules {
		if r.EscalateCondition != "" {
			prog, err := s.compileCondition(r.EscalateCondition)
			if err != nil {
				s.log.Warn("Failed to compile escalation condition",
					zap.String("rule", r.ID),
					zap.Error(err),
				)
			} else {
				s.compiled[r.ID] = prog
			}
		}
	}
}

// UpsertUser updates a user profile in the in-memory cache.
func (s *Store) UpsertUser(u types.UserProfile) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.users[u.UserID] = u
}

// GetUser returns a user profile from cache.
func (s *Store) GetUser(userID string) (types.UserProfile, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	u, ok := s.users[userID]
	return u, ok
}

// UpsertGroup updates a policy group.
func (s *Store) UpsertGroup(g types.PolicyGroup) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.groups[g.ID] = g
}

// compileCondition compiles an expr-lang expression string.
func (s *Store) compileCondition(condition string) (*vm.Program, error) {
	env := escalationEnv{}
	return expr.Compile(condition,
		expr.Env(env),
		expr.AsBool(),
	)
}

// ─────────────────────────────────────────────
//  EVALUATION CONTEXT
// ─────────────────────────────────────────────

// escalationEnv defines the variables available inside
// escalation condition expressions.
type escalationEnv struct {
	Score          float64 `expr:"score"`
	Level          string  `expr:"level"`
	Behav          float64 `expr:"behav"`
	CtxDev         float64 `expr:"ctx_dev"`
	DevRisk        float64 `expr:"dev_risk"`
	Classification string  `expr:"classification"`
	Confidence     float64 `expr:"confidence"`
	Phase          string  `expr:"phase"`
	CriticalCount  int     `expr:"critical_count"`
	EventCount     int     `expr:"event_count"`
	IsEscalating   bool    `expr:"is_escalating"`
}

func envFromEvent(evt types.RiskEvent, sess types.UserSession, escalating bool) escalationEnv {
	return escalationEnv{
		Score:          evt.RiskScore,
		Level:          classify(evt.RiskScore, 25, 50, 75),
		Behav:          evt.BehavioralDeviation,
		CtxDev:         evt.ContextDeviation,
		DevRisk:        evt.DeviceRisk,
		Classification: evt.Classification,
		Confidence:     evt.Confidence,
		Phase:          evt.Phase,
		CriticalCount:  sess.CriticalCount,
		EventCount:     sess.EventCount,
		IsEscalating:   escalating,
	}
}

// ─────────────────────────────────────────────
//  POLICY ENGINE
// ─────────────────────────────────────────────

// Engine evaluates RiskEvents against policies and produces Decisions.
type Engine struct {
	cfg      *config.Config
	store    *Store
	registry *session.Registry
	log      *zap.Logger
}

// NewEngine creates a new policy engine.
func NewEngine(
	cfg *config.Config,
	store *Store,
	registry *session.Registry,
	log *zap.Logger,
) *Engine {
	return &Engine{
		cfg:      cfg,
		store:    store,
		registry: registry,
		log:      log,
	}
}

// Evaluate processes a RiskEvent and returns a Decision.
// This is the hot path — called for every event from every agent.
func (e *Engine) Evaluate(evt types.RiskEvent) types.Decision {
	sess, _ := e.registry.Get(evt.SessionID)
	escalating := e.registry.IsEscalating(evt.SessionID, e.cfg.Policy.EscalationWindow)

	// ── Suspended user — immediate lock ──────────────────
	user, hasUser := e.store.GetUser(evt.UserID)
	if hasUser && user.IsSuspended {
		if user.SuspendedUntil == nil || time.Now().Before(*user.SuspendedUntil) {
			return e.decisionLock(evt, "suspended_user",
				fmt.Sprintf("User %s is suspended", evt.UserID))
		}
	}

	// ── Observation phase — always allow ─────────────────
	if evt.Phase == types.PhaseObservation {
		return e.decisionAllow(evt, "observation_phase", "In OBSERVATION phase — no enforcement")
	}

	// ── Find applicable policy group ─────────────────────
	group := e.resolveGroup(evt.UserID, hasUser, user)

	// ── Phase-adjust the score ────────────────────────────
	score := evt.RiskScore
	if evt.Phase == types.PhaseSoft {
		score = score / group.SoftDivisor
	}

	// ── Base classification ───────────────────────────────
	level   := classify(score, group.LowMax, group.ElevatedMax, group.HighMax)
	actions := e.baseActions(level)
	policyID := group.ID
	reason   := fmt.Sprintf("score=%.1f phase=%s group=%s", score, evt.Phase, group.ID)

	// ── Escalation rules ──────────────────────────────────
	escalated := false
	env := envFromEvent(evt, sess, escalating)

	// Check custom rules (device → user → group order, first match)
	e.store.mu.RLock()
	rules := e.store.rules
	compiled := e.store.compiled
	e.store.mu.RUnlock()

	for _, rule := range rules {
		if !rule.Enabled {
			continue
		}
		if !ruleApplies(rule, evt.UserID, evt.DeviceID) {
			continue
		}

		prog, hasCondition := compiled[rule.ID]
		if !hasCondition {
			continue
		}

		output, err := expr.Run(prog, env)
		if err != nil {
			e.log.Warn("Escalation condition evaluation failed",
				zap.String("rule", rule.ID), zap.Error(err))
			continue
		}
		if fired, ok := output.(bool); ok && fired {
			// Rule fired — merge escalation actions
			actions  = mergeActions(actions, rule.EscalateActions)
			policyID = rule.ID
			reason   = fmt.Sprintf("rule=%s fired: %s", rule.ID, rule.EscalateCondition)
			escalated = true
			e.log.Info("Escalation rule fired",
				zap.String("rule", rule.ID),
				zap.String("session", evt.SessionID),
				zap.Float64("score", evt.RiskScore),
			)
			break // first match wins
		}
	}

	// ── Attack model escalation ───────────────────────────
	// If model is confident about an attack, always step up one level
	if evt.Classification == types.ClassAttack && evt.Confidence >= 0.8 {
		actions = escalateActions(actions)
		if !escalated {
			reason   += " | attack_confidence_escalation"
			escalated = true
		}
	}

	// ── Consecutive escalation ────────────────────────────
	if escalating && !escalated {
		actions = mergeActions(actions, []string{types.ActionAlertAdmin})
		reason  += fmt.Sprintf(" | %d_consecutive_high", e.cfg.Policy.EscalationWindow)
	}

	d := types.Decision{
		SessionID: evt.SessionID,
		UserID:    evt.UserID,
		RiskScore: score,
		RiskLevel: level,
		Actions:   dedupeActions(actions),
		PolicyID:  policyID,
		Reason:    reason,
		Timestamp: time.Now(),
		Escalated: escalated,
	}

	e.log.Info("Decision",
		zap.String("session", evt.SessionID),
		zap.String("user", evt.UserID),
		zap.Float64("score", score),
		zap.String("level", level),
		zap.Strings("actions", d.Actions),
		zap.Bool("escalated", escalated),
	)

	return d
}

// ─────────────────────────────────────────────
//  HELPERS
// ─────────────────────────────────────────────

// resolveGroup finds the best policy group for a user.
// Priority: user's assigned group → default
func (e *Engine) resolveGroup(userID string, hasUser bool, user types.UserProfile) types.PolicyGroup {
	e.store.mu.RLock()
	defer e.store.mu.RUnlock()

	if hasUser && user.PolicyID != "" {
		if g, ok := e.store.groups[user.PolicyID]; ok {
			return g
		}
	}
	return types.PolicyGroupDefault
}

// classify maps a score to a risk level string.
func classify(score, lowMax, elevMax, highMax float64) string {
	switch {
	case score <= lowMax:
		return types.LevelLow
	case score <= elevMax:
		return types.LevelElevated
	case score <= highMax:
		return types.LevelHigh
	default:
		return types.LevelCritical
	}
}

// baseActions returns the default actions for a risk level.
func (e *Engine) baseActions(level string) []string {
	switch level {
	case types.LevelLow:
		return []string{types.ActionAllow}
	case types.LevelElevated:
		return []string{types.ActionStepUpAuth}
	case types.LevelHigh:
		return []string{types.ActionThrottle, types.ActionAlertAdmin}
	case types.LevelCritical:
		return []string{types.ActionLockSession, types.ActionAlertAdmin}
	default:
		return []string{types.ActionAllow}
	}
}

// escalateActions bumps the most severe current action up one level.
func escalateActions(actions []string) []string {
	for _, a := range actions {
		switch a {
		case types.ActionAllow:
			return mergeActions(actions, []string{types.ActionStepUpAuth})
		case types.ActionStepUpAuth:
			return mergeActions(actions, []string{types.ActionThrottle, types.ActionAlertAdmin})
		case types.ActionThrottle:
			return mergeActions(actions, []string{types.ActionLockSession, types.ActionAlertAdmin})
		}
	}
	return actions
}

func mergeActions(base, extra []string) []string {
	set := make(map[string]struct{}, len(base)+len(extra))
	for _, a := range base {
		set[a] = struct{}{}
	}
	for _, a := range extra {
		set[a] = struct{}{}
	}
	// If LOCK_SESSION is present, remove ALLOW and STEP_UP_AUTH (redundant)
	if _, locked := set[types.ActionLockSession]; locked {
		delete(set, types.ActionAllow)
		delete(set, types.ActionStepUpAuth)
	}
	out := make([]string, 0, len(set))
	for a := range set {
		out = append(out, a)
	}
	return out
}

func dedupeActions(actions []string) []string {
	seen := make(map[string]bool, len(actions))
	out  := make([]string, 0, len(actions))
	for _, a := range actions {
		if !seen[a] {
			seen[a] = true
			out = append(out, a)
		}
	}
	return out
}

func ruleApplies(rule types.PolicyRule, userID, deviceID string) bool {
	switch rule.Scope {
	case "global":
		return true
	case "user":
		return rule.TargetID == userID
	case "device":
		return strings.EqualFold(rule.TargetID, deviceID)
	default:
		return false
	}
}

// Pre-built decisions for common fast paths
func (e *Engine) decisionAllow(evt types.RiskEvent, policyID, reason string) types.Decision {
	return types.Decision{
		SessionID: evt.SessionID, UserID: evt.UserID,
		RiskScore: evt.RiskScore, RiskLevel: types.LevelLow,
		Actions: []string{types.ActionAllow},
		PolicyID: policyID, Reason: reason, Timestamp: time.Now(),
	}
}

func (e *Engine) decisionLock(evt types.RiskEvent, policyID, reason string) types.Decision {
	return types.Decision{
		SessionID: evt.SessionID, UserID: evt.UserID,
		RiskScore: evt.RiskScore, RiskLevel: types.LevelCritical,
		Actions:   []string{types.ActionLockSession, types.ActionAlertAdmin},
		PolicyID:  policyID, Reason: reason,
		Timestamp: time.Now(), Escalated: true,
	}
}
