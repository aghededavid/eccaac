// ECCAAC PDE — mTLS credential loader for gRPC server.
//
// When TLS.Enabled = true, presents pde.crt to connecting agents and
// requires agents to present a valid client certificate signed by ca.crt.
// When TLS.Enabled = false, server runs without TLS (dev only).

package tlsutil

import (
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"os"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"

	"github.com/eccaac/pde/internal/config"
)

// ServerOption returns a grpc.ServerOption for mTLS, or nil when TLS is disabled.
func ServerOption(cfg *config.Config) (grpc.ServerOption, error) {
	if !cfg.TLS.Enabled {
		return nil, nil
	}
	creds, err := loadServerCredentials(cfg.TLS)
	if err != nil {
		return nil, fmt.Errorf("mTLS server credentials: %w", err)
	}
	return grpc.Creds(creds), nil
}

func loadServerCredentials(t config.TLSConfig) (credentials.TransportCredentials, error) {
	cert, err := tls.LoadX509KeyPair(t.CertFile, t.KeyFile)
	if err != nil {
		return nil, fmt.Errorf("load server cert/key (%s, %s): %w", t.CertFile, t.KeyFile, err)
	}
	caPool, err := loadCA(t.CAFile)
	if err != nil {
		return nil, err
	}
	return credentials.NewTLS(&tls.Config{
		Certificates: []tls.Certificate{cert},
		ClientCAs:    caPool,
		ClientAuth:   tls.RequireAndVerifyClientCert, // strict mTLS
		MinVersion:   tls.VersionTLS13,
	}), nil
}

func loadCA(caFile string) (*x509.CertPool, error) {
	pem, err := os.ReadFile(caFile)
	if err != nil {
		return nil, fmt.Errorf("read CA cert (%s): %w", caFile, err)
	}
	pool := x509.NewCertPool()
	if !pool.AppendCertsFromPEM(pem) {
		return nil, fmt.Errorf("CA cert (%s): no valid PEM blocks found", caFile)
	}
	return pool, nil
}
