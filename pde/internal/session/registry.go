// ECCAAC — Policy Decision Engine
// Session Registry
//
// Maintains in-memory state for every active user session.
// Sessions are created on first event, updated on each subsequent event,
// and expired by a background reaper after inactivity timeout.

package session

import (
	"sync"
	"time"

	"github.com/eccaac/pde/internal/config"
	"github.com/eccaac/pde/internal/types"
	"go.uber.org/zap"
)

// eventRing is a fixed-size ring buffer of recent RiskEvents for a session.
type eventRing struct {
	events []types.RiskEvent
	head   int
	size   int
	cap    int
}

func newEventRing(capacity int) *eventRing {
	return &eventRing{
		events: make([]types.RiskEvent, capacity),
		cap:    capacity,
	}
}

func (r *eventRing) push(e types.RiskEvent) {
	r.events[r.head%r.cap] = e
	r.head++
	if r.size < r.cap {
		r.size++
	}
}

// recent returns the last n events in chronological order.
func (r *eventRing) recent(n int) []types.RiskEvent {
	if n > r.size {
		n = r.size
	}
	result := make([]types.RiskEvent, n)
	start := r.head - n
	for i := 0; i < n; i++ {
		result[i] = r.events[(start+i)%r.cap]
	}
	return result
}

// ─────────────────────────────────────────────
//  SESSION ENTRY (internal)
// ─────────────────────────────────────────────

type sessionEntry struct {
	session    types.UserSession
	recentEvts *eventRing
	mu         sync.Mutex
}

// update applies a new RiskEvent to the session's running stats.
func (e *sessionEntry) update(evt types.RiskEvent) {
	e.mu.Lock()
	defer e.mu.Unlock()

	s := &e.session
	s.EventCount++
	s.LastSeenAt = evt.Timestamp
	s.Phase = evt.Phase

	// Running average risk score
	s.AvgRiskScore = (s.AvgRiskScore*float64(s.EventCount-1) + evt.RiskScore) / float64(s.EventCount)

	if evt.RiskScore > s.MaxRiskScore {
		s.MaxRiskScore = evt.RiskScore
	}
	if evt.RiskScore >= 76 {
		s.CriticalCount++
		s.WasFlagged = true
	}

	e.recentEvts.push(evt)
}

// ─────────────────────────────────────────────
//  REGISTRY
// ─────────────────────────────────────────────

// Registry is the in-memory session store.
type Registry struct {
	cfg  *config.Config
	log  *zap.Logger
	mu   sync.RWMutex
	data map[string]*sessionEntry // key: session_id

	// Callbacks
	onExpire func(types.UserSession)
}

// NewRegistry creates a new session registry and starts the reaper.
func NewRegistry(cfg *config.Config, log *zap.Logger) *Registry {
	r := &Registry{
		cfg:  cfg,
		log:  log,
		data: make(map[string]*sessionEntry),
	}
	go r.runReaper()
	return r
}

// OnExpire registers a callback invoked when a session expires.
func (r *Registry) OnExpire(fn func(types.UserSession)) {
	r.onExpire = fn
}

// ─────────────────────────────────────────────
//  CORE OPERATIONS
// ─────────────────────────────────────────────

// Upsert creates or updates a session based on an incoming RiskEvent.
// Returns the current session state after the update.
func (r *Registry) Upsert(evt types.RiskEvent) types.UserSession {
	r.mu.Lock()

	entry, exists := r.data[evt.SessionID]
	if !exists {
		// New session
		entry = &sessionEntry{
			session: types.UserSession{
				SessionID:  evt.SessionID,
				UserID:     evt.UserID,
				DeviceID:   evt.DeviceID,
				Phase:      evt.Phase,
				StartedAt:  evt.Timestamp,
				LastSeenAt: evt.Timestamp,
				Active:     true,
			},
			recentEvts: newEventRing(r.cfg.Session.MaxEventsPerSession),
		}
		r.data[evt.SessionID] = entry
		r.log.Info("New session registered",
			zap.String("session", evt.SessionID),
			zap.String("user", evt.UserID),
			zap.String("device", evt.DeviceID),
		)
	}
	r.mu.Unlock()

	entry.update(evt)

	entry.mu.Lock()
	snap := entry.session
	entry.mu.Unlock()

	return snap
}

// Get returns the current state of a session, if it exists.
func (r *Registry) Get(sessionID string) (types.UserSession, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	entry, ok := r.data[sessionID]
	if !ok {
		return types.UserSession{}, false
	}
	entry.mu.Lock()
	snap := entry.session
	entry.mu.Unlock()
	return snap, true
}

// RecentEvents returns the last n events for a session.
func (r *Registry) RecentEvents(sessionID string, n int) []types.RiskEvent {
	r.mu.RLock()
	entry, ok := r.data[sessionID]
	r.mu.RUnlock()
	if !ok {
		return nil
	}
	entry.mu.Lock()
	defer entry.mu.Unlock()
	return entry.recentEvts.recent(n)
}

// End marks a session as ended and removes it from the active registry.
func (r *Registry) End(sessionID string, avgRisk float64, wasFlagged bool) (types.UserSession, bool) {
	r.mu.Lock()
	entry, ok := r.data[sessionID]
	if !ok {
		r.mu.Unlock()
		return types.UserSession{}, false
	}
	delete(r.data, sessionID)
	r.mu.Unlock()

	entry.mu.Lock()
	now := time.Now()
	entry.session.Active = false
	entry.session.EndedAt = &now
	if wasFlagged {
		entry.session.WasFlagged = true
	}
	snap := entry.session
	entry.mu.Unlock()

	r.log.Info("Session ended",
		zap.String("session", sessionID),
		zap.Float64("avg_risk", snap.AvgRiskScore),
		zap.Bool("flagged", snap.WasFlagged),
		zap.Int("events", snap.EventCount),
	)
	return snap, true
}

// All returns a snapshot of all active sessions.
func (r *Registry) All() []types.UserSession {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]types.UserSession, 0, len(r.data))
	for _, entry := range r.data {
		entry.mu.Lock()
		out = append(out, entry.session)
		entry.mu.Unlock()
	}
	return out
}

// ActiveCount returns the number of active sessions.
func (r *Registry) ActiveCount() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.data)
}

// ─────────────────────────────────────────────
//  REAPER
// ─────────────────────────────────────────────

// runReaper periodically expires sessions that have been inactive
// longer than session.inactivity_timeout.
func (r *Registry) runReaper() {
	ticker := time.NewTicker(r.cfg.Session.ReaperInterval)
	defer ticker.Stop()

	for range ticker.C {
		r.reap()
	}
}

func (r *Registry) reap() {
	threshold := time.Now().Add(-r.cfg.Session.InactivityTimeout)

	r.mu.Lock()
	expired := []string{}
	for id, entry := range r.data {
		entry.mu.Lock()
		if entry.session.LastSeenAt.Before(threshold) {
			expired = append(expired, id)
		}
		entry.mu.Unlock()
	}
	snapshots := make([]types.UserSession, 0, len(expired))
	for _, id := range expired {
		entry := r.data[id]
		now := time.Now()
		entry.session.Active = false
		entry.session.EndedAt = &now
		snapshots = append(snapshots, entry.session)
		delete(r.data, id)
	}
	r.mu.Unlock()

	for _, snap := range snapshots {
		r.log.Info("Session expired (inactivity)",
			zap.String("session", snap.SessionID),
			zap.String("user", snap.UserID),
			zap.Duration("inactive", time.Since(snap.LastSeenAt)),
		)
		if r.onExpire != nil {
			r.onExpire(snap)
		}
	}
}

// ─────────────────────────────────────────────
//  ESCALATION DETECTOR
// ─────────────────────────────────────────────

// IsEscalating returns true if the session has seen N consecutive
// HIGH or CRITICAL events — used by the policy engine to auto-escalate.
func (r *Registry) IsEscalating(sessionID string, windowSize int) bool {
	events := r.RecentEvents(sessionID, windowSize)
	if len(events) < windowSize {
		return false
	}
	for _, e := range events {
		if e.RiskScore < 51 { // below HIGH threshold
			return false
		}
	}
	return true
}
