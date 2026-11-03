// ECCAAC — Policy Decision Engine
// Main entrypoint
//
// Usage:
//   ./eccaac-pde                          start with defaults
//   ./eccaac-pde --config /etc/eccaac/pde.yaml
//   ./eccaac-pde migrate                  run DB migrations only

package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/eccaac/pde/internal/config"
	"github.com/eccaac/pde/internal/grpcserver"
	"github.com/eccaac/pde/internal/policy"
	"github.com/eccaac/pde/internal/restapi"
	"github.com/eccaac/pde/internal/session"
	"github.com/eccaac/pde/internal/store"
	"github.com/eccaac/pde/internal/types"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

func main() {
	// Config
	cfgPath := ""
	if len(os.Args) > 1 && os.Args[1] != "migrate" {
		cfgPath = os.Args[1]
	}

	cfg, err := config.Load(cfgPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Config error: %v\n", err)
		os.Exit(1)
	}

	// Logger
	log, err := buildLogger(cfg.Log.Level, cfg.Log.Format)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Logger error: %v\n", err)
		os.Exit(1)
	}
	defer log.Sync()

	log.Info("ECCAAC Policy Decision Engine starting",
		zap.String("grpc", cfg.Server.GRPCAddr),
		zap.String("rest", cfg.Server.RESTAddr),
	)

	// Database
	db, err := store.New(
	cfg.DB.DSN,
	int32(cfg.DB.MaxOpenConns),
	log,
)
	if err != nil {
		log.Fatal("Database connection failed", zap.Error(err))
	}
	defer db.Close()

	// Migrate
	ctx := context.Background()
	if err := db.Migrate(ctx); err != nil {
		log.Fatal("DB migration failed", zap.Error(err))
	}
	log.Info("Database migrations applied")

	if len(os.Args) > 1 && os.Args[1] == "migrate" {
		log.Info("Migration-only mode — exiting")
		return
	}

	// Policy store — load rules from DB
	policyStore := policy.NewStore(log)
	if rules, err := db.GetPolicyRules(ctx); err == nil {
		policyStore.LoadRules(rules)
		log.Info("Policy rules loaded", zap.Int("count", len(rules)))
	}

	// Load user profiles into memory
	if users, err := db.GetUsers(ctx); err == nil {
		for _, u := range users {
			policyStore.UpsertUser(u)
		}
		log.Info("User profiles loaded", zap.Int("count", len(users)))
	}

	// Session registry
	registry := session.NewRegistry(cfg, log)
	registry.OnExpire(func(sess types.UserSession) {
	db.RecordSessionEnd(sess)
})

	// Policy engine
	engine := policy.NewEngine(cfg, policyStore, registry, log)

	// gRPC servicer + server
	servicer := grpcserver.NewPDEServicer(cfg, engine, registry, db, log)
	grpcSrv, err := grpcserver.NewServer(cfg, servicer, log)
if err != nil {
	log.Fatal("Failed to create gRPC server", zap.Error(err))
}

	// REST API
	api     := restapi.New(cfg, engine, registry, db, log)
	restSrv := &http.Server{
		Addr:         cfg.Server.RESTAddr,
		Handler:      api.Handler(),
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	// Start REST API in background
	go func() {
		log.Info("PDE REST API listening", zap.String("addr", cfg.Server.RESTAddr))
		if err := restSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Error("REST server error", zap.Error(err))
		}
	}()

	// Start policy sync background task
	go runPolicySync(ctx, cfg, db, policyStore, log)

	// Start gRPC server (blocks until interrupt)
	if err := grpcSrv.RunUntilInterrupt(); err != nil {
		log.Fatal("gRPC server error", zap.Error(err))
	}

	// Graceful shutdown
	log.Info("Shutting down PDE...")
	shutCtx, shutCancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer shutCancel()
	_ = restSrv.Shutdown(shutCtx)
	log.Info("PDE stopped cleanly")
}

// runPolicySync periodically reloads policy rules from the DB
// so that admin changes take effect without a restart.
func runPolicySync(
	ctx context.Context,
	cfg *config.Config,
	db *store.Store,
	policyStore *policy.Store,
	log *zap.Logger,
) {
	ticker := time.NewTicker(cfg.Policy.SyncInterval)
	defer ticker.Stop()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	for {
		select {
		case <-ticker.C:
			rules, err := db.GetPolicyRules(ctx)
			if err != nil {
				log.Warn("Policy sync failed", zap.Error(err))
				continue
			}
			policyStore.LoadRules(rules)
			log.Debug("Policy rules synced", zap.Int("count", len(rules)))

		case <-sigCh:
			return
		}
	}
}

// ─────────────────────────────────────────────
//  LOGGER
// ─────────────────────────────────────────────

func buildLogger(level, format string) (*zap.Logger, error) {
	lvl := zap.NewAtomicLevel()
	if err := lvl.UnmarshalText([]byte(level)); err != nil {
		lvl.SetLevel(zap.InfoLevel)
	}

	var enc zapcore.Encoder
	if format == "console" {
		enc = zapcore.NewConsoleEncoder(zap.NewDevelopmentEncoderConfig())
	} else {
		enc = zapcore.NewJSONEncoder(zap.NewProductionEncoderConfig())
	}

	core := zapcore.NewCore(enc, zapcore.AddSync(os.Stdout), lvl)
	return zap.New(core, zap.AddCaller()), nil
}
