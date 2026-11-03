#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────
#  ECCAAC — Compile Proto Stubs
#  Run from the repo root: bash scripts/compile_proto.sh
#
#  Generates:
#    risk_engine/proto/         Python stubs (for Python Risk Engine)
#    agent/proto/risk/          Go stubs (for Go Agent)
#    pde/proto/risk/            Go stubs (for Go PDE)
#
#  Prerequisites:
#    pip install grpcio-tools
#    go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
#    go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
#    apt install protobuf-compiler  (or brew install protobuf on macOS)
# ─────────────────────────────────────────────────────────

set -euo pipefail
cd "$(dirname "$0")/.."

PROTO_FILE="proto/risk_engine.proto"
PROTO_DIR="proto"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  ECCAAC — Proto Stub Compiler            ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── Validate proto file exists ────────────────────────────
if [ ! -f "$PROTO_FILE" ]; then
  echo "✗ Proto file not found: $PROTO_FILE"
  exit 1
fi

# ── Python stubs (Risk Scoring Engine) ───────────────────
PYTHON_OUT="risk_engine/proto"
mkdir -p "$PYTHON_OUT"

echo "► Compiling Python stubs → $PYTHON_OUT"
python -m grpc_tools.protoc \
  -I "$PROTO_DIR" \
  --python_out="$PYTHON_OUT" \
  --grpc_python_out="$PYTHON_OUT" \
  "$PROTO_FILE"

# grpc_tools generates absolute imports; fix to relative for package use
if [ -f "$PYTHON_OUT/risk_engine_pb2_grpc.py" ]; then
  sed -i 's/^import risk_engine_pb2/from . import risk_engine_pb2/' \
    "$PYTHON_OUT/risk_engine_pb2_grpc.py" 2>/dev/null || \
  # macOS sed requires empty string argument
  sed -i '' 's/^import risk_engine_pb2/from . import risk_engine_pb2/' \
    "$PYTHON_OUT/risk_engine_pb2_grpc.py"
fi

# Ensure __init__.py exists
touch "$PYTHON_OUT/__init__.py"
echo "  ✓ Python stubs generated"

# ── Go stubs (Agent + PDE) ────────────────────────────────
if ! command -v protoc &> /dev/null; then
  echo ""
  echo "  ⚠ protoc not found — skipping Go stubs"
  echo "    Install: https://grpc.io/docs/protoc-installation/"
  echo "    On Ubuntu: apt install -y protobuf-compiler"
  echo "    On macOS:  brew install protobuf"
  echo ""
else
  # Check for Go proto plugins
  MISSING_PLUGINS=0
  for plugin in protoc-gen-go protoc-gen-go-grpc; do
    if ! command -v "$plugin" &> /dev/null; then
      echo "  ⚠ $plugin not found"
      MISSING_PLUGINS=1
    fi
  done
  if [ "$MISSING_PLUGINS" -eq 1 ]; then
    echo "  Install plugins:"
    echo "    go install google.golang.org/protobuf/cmd/protoc-gen-go@latest"
    echo "    go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest"
  else
    # Agent stubs
    AGENT_OUT="agent/proto/risk"
    mkdir -p "$AGENT_OUT"
    echo "► Compiling Go agent stubs → $AGENT_OUT"
    protoc \
      -I "$PROTO_DIR" \
      --go_out="$AGENT_OUT" \
      --go_opt=paths=source_relative \
      --go-grpc_out="$AGENT_OUT" \
      --go-grpc_opt=paths=source_relative \
      "$PROTO_FILE"
    echo "  ✓ Go agent stubs generated"

    # PDE stubs (same proto, different module path)
    PDE_OUT="pde/proto/risk"
    mkdir -p "$PDE_OUT"
    echo "► Compiling Go PDE stubs → $PDE_OUT"
    protoc \
      -I "$PROTO_DIR" \
      --go_out="$PDE_OUT" \
      --go_opt=paths=source_relative \
      --go-grpc_out="$PDE_OUT" \
      --go-grpc_opt=paths=source_relative \
      "$PROTO_FILE"

    # Fix go_package option for PDE (different import path)
    if [ -f "$PDE_OUT/risk_engine.pb.go" ]; then
      sed -i 's|github.com/eccaac/agent/proto/risk|github.com/eccaac/pde/proto/risk|g' \
        "$PDE_OUT/risk_engine.pb.go" \
        "$PDE_OUT/risk_engine_grpc.pb.go" 2>/dev/null || \
      sed -i '' 's|github.com/eccaac/agent/proto/risk|github.com/eccaac/pde/proto/risk|g' \
        "$PDE_OUT/risk_engine.pb.go" \
        "$PDE_OUT/risk_engine_grpc.pb.go"
    fi
    echo "  ✓ Go PDE stubs generated"
  fi
fi

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  Done. Stubs generated:                  ║"
echo "║    risk_engine/proto/  (Python)           ║"
echo "║    agent/proto/risk/   (Go agent)         ║"
echo "║    pde/proto/risk/     (Go PDE)           ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. Run the wiring script:  bash scripts/wire_proto.sh"
echo "  2. Build agent:            cd agent && make build"
echo "  3. Build PDE:              cd pde   && make build"
echo ""
