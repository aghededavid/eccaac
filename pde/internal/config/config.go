// ECCAAC — Policy Decision Engine
// Configuration loader.

package config

import (
	"fmt"
	"time"

	"github.com/spf13/viper"
)

// Config is the full PDE configuration.
type Config struct {
	Server   ServerConfig   `mapstructure:"server"`
	TLS      TLSConfig      `mapstructure:"tls"`
	DB       DBConfig       `mapstructure:"db"`
	Policy   PolicyConfig   `mapstructure:"policy"`
	Session  SessionConfig  `mapstructure:"session"`
	Log      LogConfig      `mapstructure:"log"`
	Admin    AdminConfig    `mapstructure:"admin"`
}

// TLSConfig controls mTLS for the PDE gRPC server.
// Agents must present a client certificate signed by CAFile.
type TLSConfig struct {
	Enabled  bool   `mapstructure:"enabled"`
	CertFile string `mapstructure:"cert_file"` // PDE server cert (PEM)
	KeyFile  string `mapstructure:"key_file"`  // PDE server key (PEM)
	CAFile   string `mapstructure:"ca_file"`   // CA cert to verify agent client certs
}

type ServerConfig struct {
	GRPCAddr string `mapstructure:"grpc_addr"` // e.g. "0.0.0.0:50052"
	RESTAddr string `mapstructure:"rest_addr"` // e.g. "0.0.0.0:8080"
}

type DBConfig struct {
	DSN             string        `mapstructure:"dsn"`              // postgres://user:pass@host/db
	MaxOpenConns    int           `mapstructure:"max_open_conns"`
	MaxIdleConns    int           `mapstructure:"max_idle_conns"`
	ConnMaxLifetime time.Duration `mapstructure:"conn_max_lifetime"`
}

type PolicyConfig struct {
	// Default thresholds (overridden per user/device via PolicyGroup)
	DefaultLowMax      float64 `mapstructure:"default_low_max"`
	DefaultElevatedMax float64 `mapstructure:"default_elevated_max"`
	DefaultHighMax     float64 `mapstructure:"default_high_max"`
	DefaultSoftDivisor float64 `mapstructure:"default_soft_divisor"`

	// Escalation: consecutive HIGH events before escalating to CRITICAL
	EscalationWindow int `mapstructure:"escalation_window"`

	// Policy sync interval: how often agents pull updated policy
	SyncInterval time.Duration `mapstructure:"sync_interval"`
}

type SessionConfig struct {
	// How long a session is considered active without an event
	InactivityTimeout time.Duration `mapstructure:"inactivity_timeout"`
	// How often the session reaper runs
	ReaperInterval time.Duration `mapstructure:"reaper_interval"`
	// Max events to keep in memory per session (ring buffer)
	MaxEventsPerSession int `mapstructure:"max_events_per_session"`
}

type LogConfig struct {
	Level  string `mapstructure:"level"`
	Format string `mapstructure:"format"` // json | console
}

type AdminConfig struct {
	// JWT secret for REST API authentication
	JWTSecret string `mapstructure:"jwt_secret"`
	// Whether to require auth on the REST API
	RequireAuth bool `mapstructure:"require_auth"`
}

// Load reads config from file and environment variables.
func Load(cfgPath string) (*Config, error) {
	v := viper.New()
	setDefaults(v)

	if cfgPath != "" {
		v.SetConfigFile(cfgPath)
	} else {
		v.AddConfigPath("/etc/eccaac")
		v.AddConfigPath(".")
		v.SetConfigName("pde")
		v.SetConfigType("yaml")
	}

	v.SetEnvPrefix("ECCAAC_PDE")
	v.AutomaticEnv()

	if err := v.ReadInConfig(); err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
			return nil, fmt.Errorf("reading config: %w", err)
		}
	}

	var cfg Config
	if err := v.Unmarshal(&cfg); err != nil {
		return nil, fmt.Errorf("unmarshaling config: %w", err)
	}
	return &cfg, nil
}

func setDefaults(v *viper.Viper) {
	v.SetDefault("server.grpc_addr", "0.0.0.0:50052")
	v.SetDefault("server.rest_addr", "0.0.0.0:8080")

	v.SetDefault("db.dsn",                "postgres://eccaac:eccaac@localhost:5432/eccaac?sslmode=disable")
	v.SetDefault("db.max_open_conns",     20)
	v.SetDefault("db.max_idle_conns",     5)
	v.SetDefault("db.conn_max_lifetime",  "30m")

	v.SetDefault("policy.default_low_max",       25.0)
	v.SetDefault("policy.default_elevated_max",  50.0)
	v.SetDefault("policy.default_high_max",      75.0)
	v.SetDefault("policy.default_soft_divisor",  2.0)
	v.SetDefault("policy.escalation_window",     3)
	v.SetDefault("policy.sync_interval",         "60s")

	v.SetDefault("session.inactivity_timeout",    "30m")
	v.SetDefault("session.reaper_interval",        "5m")
	v.SetDefault("session.max_events_per_session", 200)

	v.SetDefault("log.level",  "info")
	v.SetDefault("log.format", "json")

	v.SetDefault("admin.require_auth", true)

	v.SetDefault("tls.enabled",   false)
	v.SetDefault("tls.cert_file", "/etc/eccaac/certs/pde.crt")
	v.SetDefault("tls.key_file",  "/etc/eccaac/certs/pde.key")
	v.SetDefault("tls.ca_file",   "/etc/eccaac/certs/ca.crt")
}
