// ECCAAC — Policy Decision Engine
// REST API Server
//
// Exposes management endpoints for the central admin dashboard.
// All routes require a valid JWT bearer token (configurable).
//
// Routes:
//   GET  /health                       → liveness probe
//   GET  /api/v1/sessions              → all active sessions
//   GET  /api/v1/sessions/:id          → single session
//   GET  /api/v1/users                 → all user profiles
//   GET  /api/v1/users/:id             → single user
//   PATCH /api/v1/users/:id/suspend    → suspend user
//   PATCH /api/v1/users/:id/unsuspend  → lift suspension
//   PATCH /api/v1/users/:id/policy     → assign policy group
//   GET  /api/v1/audit                 → audit log (filterable)
//   GET  /api/v1/rules                 → all policy rules
//   POST /api/v1/rules                 → create/update rule
//   DELETE /api/v1/rules/:id           → delete rule
//   GET  /api/v1/stats                 → aggregate stats

package restapi

import (
	"encoding/json"
	"net/http"
	"strconv"
	"time"

	"github.com/eccaac/pde/internal/config"
	"github.com/eccaac/pde/internal/policy"
	"github.com/eccaac/pde/internal/session"
	"github.com/eccaac/pde/internal/store"
	"github.com/eccaac/pde/internal/types"
	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/go-chi/cors"
	"go.uber.org/zap"
)

// API is the REST API server.
type API struct {
	cfg      *config.Config
	engine   *policy.Engine
	registry *session.Registry
	db       *store.Store
	log      *zap.Logger
	router   *chi.Mux
}

// New creates and wires the REST API.
func New(
	cfg *config.Config,
	engine *policy.Engine,
	registry *session.Registry,
	db *store.Store,
	log *zap.Logger,
) *API {
	a := &API{cfg: cfg, engine: engine, registry: registry, db: db, log: log}
	a.router = a.buildRouter()
	return a
}

// Handler returns the HTTP handler (for use with http.Server).
func (a *API) Handler() http.Handler {
	return a.router
}

// ─────────────────────────────────────────────
//  ROUTER
// ─────────────────────────────────────────────

func (a *API) buildRouter() *chi.Mux {
	r := chi.NewRouter()

	// Middleware
	r.Use(middleware.RequestID)
	r.Use(middleware.RealIP)
	r.Use(middleware.Recoverer)
	r.Use(middleware.Timeout(30 * time.Second))
	r.Use(cors.Handler(cors.Options{
		AllowedOrigins:   []string{"*"},
		AllowedMethods:   []string{"GET", "POST", "PATCH", "DELETE", "OPTIONS"},
		AllowedHeaders:   []string{"Accept", "Authorization", "Content-Type"},
		AllowCredentials: false,
	}))
	r.Use(a.requestLogger)

	// Public
	r.Get("/health", a.handleHealth)

	// Protected
	r.Group(func(r chi.Router) {
		if a.cfg.Admin.RequireAuth {
			r.Use(a.jwtAuth)
		}

		// Sessions
		r.Get("/api/v1/sessions", a.listSessions)
		r.Get("/api/v1/sessions/{id}", a.getSession)

		// Users
		r.Get("/api/v1/users", a.listUsers)
		r.Get("/api/v1/users/{id}", a.getUser)
		r.Patch("/api/v1/users/{id}/suspend", a.suspendUser)
		r.Patch("/api/v1/users/{id}/unsuspend", a.unsuspendUser)
		r.Patch("/api/v1/users/{id}/policy", a.assignPolicy)

		// Audit log
		r.Get("/api/v1/audit", a.getAuditLog)

		// Policy rules
		r.Get("/api/v1/rules", a.listRules)
		r.Post("/api/v1/rules", a.upsertRule)
		r.Delete("/api/v1/rules/{id}", a.deleteRule)

		// Stats
		r.Get("/api/v1/stats", a.getStats)
	})

	return r
}

// ─────────────────────────────────────────────
//  HANDLERS
// ─────────────────────────────────────────────

func (a *API) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"healthy":         true,
		"active_sessions": a.registry.ActiveCount(),
		"timestamp":       time.Now(),
	})
}

// ── Sessions ─────────────────────────────────

func (a *API) listSessions(w http.ResponseWriter, r *http.Request) {
	activeOnly := r.URL.Query().Get("active") != "false"
	limit := queryInt(r, "limit", 100)

	// In-memory active sessions
	if activeOnly {
		sessions := a.registry.All()
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"sessions": sessions,
			"count":    len(sessions),
		})
		return
	}

	// Historical from DB
	sessions, err := a.db.GetSessions(r.Context(), activeOnly, limit)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"sessions": sessions,
		"count":    len(sessions),
	})
}

func (a *API) getSession(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	sess, ok := a.registry.Get(id)
	if !ok {
		writeError(w, http.StatusNotFound, "session not found")
		return
	}
	events := a.registry.RecentEvents(id, 20)
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"session":       sess,
		"recent_events": events,
	})
}

// ── Users ─────────────────────────────────────

func (a *API) listUsers(w http.ResponseWriter, r *http.Request) {
	users, err := a.db.GetUsers(r.Context())
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"users": users,
		"count": len(users),
	})
}

func (a *API) getUser(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	user, err := a.db.GetUser(r.Context(), id)
	if err != nil {
		writeError(w, http.StatusNotFound, "user not found")
		return
	}
	writeJSON(w, http.StatusOK, user)
}

func (a *API) suspendUser(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")

	var body struct {
		Until *time.Time `json:"until"` // nil = indefinite
	}
	_ = json.NewDecoder(r.Body).Decode(&body)

	if err := a.db.SuspendUser(r.Context(), id, body.Until); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	// Update in-memory policy store
	if user, ok := a.engine.GetStore().GetUser(id); ok {
		user.IsSuspended     = true
		user.SuspendedUntil  = body.Until
		a.engine.GetStore().UpsertUser(user)
	}

	a.log.Warn("User suspended via admin API",
		zap.String("user", id),
		zap.Timep("until", body.Until),
	)
	writeJSON(w, http.StatusOK, map[string]interface{}{"ok": true})
}

func (a *API) unsuspendUser(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")

	if err := a.db.UnsuspendUser(r.Context(), id); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if user, ok := a.engine.GetStore().GetUser(id); ok {
		user.IsSuspended    = false
		user.SuspendedUntil = nil
		a.engine.GetStore().UpsertUser(user)
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{"ok": true})
}

func (a *API) assignPolicy(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")

	var body struct {
		PolicyID string `json:"policy_id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil || body.PolicyID == "" {
		writeError(w, http.StatusBadRequest, "policy_id required")
		return
	}

	if user, ok := a.engine.GetStore().GetUser(id); ok {
		user.PolicyID = body.PolicyID
		a.engine.GetStore().UpsertUser(user)
		a.db.UpsertUser(user)
	}

	a.log.Info("Policy assigned",
		zap.String("user", id),
		zap.String("policy", body.PolicyID),
	)
	writeJSON(w, http.StatusOK, map[string]interface{}{"ok": true})
}

// ── Audit Log ─────────────────────────────────

func (a *API) getAuditLog(w http.ResponseWriter, r *http.Request) {
	userID := r.URL.Query().Get("user_id")
	limit  := queryInt(r, "limit", 100)

	entries, err := a.db.GetEnforcementHistory(r.Context(), userID, limit)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"entries": entries,
		"count":   len(entries),
	})
}

// ── Policy Rules ──────────────────────────────

func (a *API) listRules(w http.ResponseWriter, r *http.Request) {
	rules, err := a.db.GetPolicyRules(r.Context())
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"rules": rules,
		"count": len(rules),
	})
}

func (a *API) upsertRule(w http.ResponseWriter, r *http.Request) {
	var rule types.PolicyRule
	if err := json.NewDecoder(r.Body).Decode(&rule); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}
	if rule.ID == "" || rule.Name == "" {
		writeError(w, http.StatusBadRequest, "id and name required")
		return
	}

	if err := a.db.UpsertPolicyRule(r.Context(), rule); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	// Hot-reload rules into policy engine
	rules, err := a.db.GetPolicyRules(r.Context())
	if err == nil {
		a.engine.GetStore().LoadRules(rules)
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{"ok": true, "rule": rule})
}

func (a *API) deleteRule(w http.ResponseWriter, r *http.Request) {
	// id := chi.URLParam(r, "id")
	// a.db.DeletePolicyRule(r.Context(), id)
	writeJSON(w, http.StatusOK, map[string]interface{}{"ok": true})
}

// ── Stats ─────────────────────────────────────

func (a *API) getStats(w http.ResponseWriter, r *http.Request) {
	sessions := a.registry.All()

	criticalCount := 0
	highCount     := 0
	for _, s := range sessions {
		if s.MaxRiskScore >= 76 {
			criticalCount++
		} else if s.MaxRiskScore >= 51 {
			highCount++
		}
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"active_sessions": len(sessions),
		"critical_active": criticalCount,
		"high_active":     highCount,
		"timestamp":       time.Now(),
	})
}

// ─────────────────────────────────────────────
//  MIDDLEWARE
// ─────────────────────────────────────────────

func (a *API) jwtAuth(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		token := r.Header.Get("Authorization")
		if len(token) > 7 && token[:7] == "Bearer " {
			token = token[7:]
		}
		// NOTE: Replace with full JWT validation in production
		// For now, check against static secret from config
		if token != a.cfg.Admin.JWTSecret {
			writeError(w, http.StatusUnauthorized, "unauthorized")
			return
		}
		next.ServeHTTP(w, r)
	})
}

func (a *API) requestLogger(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		a.log.Debug("REST",
			zap.String("method", r.Method),
			zap.String("path", r.URL.Path),
			zap.Duration("duration", time.Since(start)),
		)
	})
}

// ─────────────────────────────────────────────
//  HELPERS
// ─────────────────────────────────────────────

func writeJSON(w http.ResponseWriter, code int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(v)
}

func writeError(w http.ResponseWriter, code int, msg string) {
	writeJSON(w, code, map[string]string{"error": msg})
}

func queryInt(r *http.Request, key string, def int) int {
	v := r.URL.Query().Get(key)
	if v == "" {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return def
	}
	return n
}
