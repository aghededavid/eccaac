#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────
#  ECCAAC — mTLS Certificate Generator
#
#  Generates:
#    certs/ca/ca.crt             Root CA (distribute to ALL nodes)
#    certs/ca/ca.key             Root CA key  (keep secure — never distribute)
#    certs/server/risk_engine.*  Python Risk Engine server cert
#    certs/server/pde.*          Go PDE server cert
#    certs/agents/<hostname>.*   Per-agent client cert
#
#  Usage:
#    bash scripts/gen_certs.sh --all --agents laptop-01,laptop-02
#    bash scripts/gen_certs.sh --ca --servers
#    bash scripts/gen_certs.sh --agent HOSTNAME
#
#  Deploy to server:
#    sudo cp certs/ca/ca.crt           /etc/eccaac/certs/
#    sudo cp certs/server/pde.*         /etc/eccaac/certs/
#    sudo cp certs/server/risk_engine.* /etc/eccaac/certs/
#
#  Deploy to agent (Windows, PowerShell):
#    Copy-Item .\certs\ca\ca.crt            'C:\ProgramData\ECCAAC\certs\'
#    Copy-Item .\certs\agents\HOST.crt      'C:\ProgramData\ECCAAC\certs\agent.crt'
#    Copy-Item .\certs\agents\HOST.key      'C:\ProgramData\ECCAAC\certs\agent.key'
#
#  Then enable in configs:
#    agent.yaml:  tls.enabled: true
#    pde.yaml:    tls.enabled: true
#    risk engine: tls_enabled: true
#
#  Requirements: openssl
# ─────────────────────────────────────────────────────────
set -euo pipefail

CERTS_DIR="${CERTS_DIR:-./certs}"
COUNTRY="${COUNTRY:-ZA}"
STATE="${STATE:-Gauteng}"
ORG="${ORG:-ECCAAC}"
CA_DAYS=3650   # 10 years
SRV_DAYS=730   # 2 years
AGT_DAYS=365   # 1 year

DO_CA=0; DO_SERVERS=0; AGENT_HOSTS=""

usage() {
  echo "Usage: $0 [--ca] [--servers] [--all] [--agent HOST] [--agents h1,h2] [--out DIR]"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --ca)      DO_CA=1; shift ;;
    --servers) DO_SERVERS=1; shift ;;
    --all)     DO_CA=1; DO_SERVERS=1; shift ;;
    --agent)   AGENT_HOSTS="$2"; shift 2 ;;
    --agents)  AGENT_HOSTS="$2"; shift 2 ;;
    --out)     CERTS_DIR="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown: $1"; usage ;;
  esac
done

[[ $DO_CA -eq 0 && $DO_SERVERS -eq 0 && -z "$AGENT_HOSTS" ]] && usage

mkdir -p "$CERTS_DIR/ca" "$CERTS_DIR/server" "$CERTS_DIR/agents"

CA_CRT="$CERTS_DIR/ca/ca.crt"
CA_KEY="$CERTS_DIR/ca/ca.key"

# sign_cert NAME CN SAN DAYS CA_CRT CA_KEY OUT_DIR IS_CLIENT
sign_cert() {
  local name="$1" cn="$2" san="$3" days="$4"
  local cacrt="$5" cakey="$6" outdir="$7" client="${8:-0}"
  local key="$outdir/$name.key" csr="$outdir/$name.csr"
  local crt="$outdir/$name.crt" ext="$outdir/$name.ext"

  openssl ecparam -name prime256v1 -genkey -noout -out "$key"
  chmod 600 "$key"
  openssl req -new -key "$key" -out "$csr" \
    -subj "/C=$COUNTRY/ST=$STATE/O=$ORG/CN=$cn"

  if [[ "$client" -eq 1 ]]; then
    printf 'basicConstraints=CA:FALSE\nkeyUsage=digitalSignature\nextendedKeyUsage=clientAuth\nsubjectAltName=%s\n' "$san" > "$ext"
  else
    printf 'basicConstraints=CA:FALSE\nkeyUsage=digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\nsubjectAltName=%s\n' "$san" > "$ext"
  fi

  openssl x509 -req -in "$csr" -CA "$cacrt" -CAkey "$cakey" \
    -CAcreateserial -out "$crt" -days "$days" -sha256 -extfile "$ext"
  rm -f "$csr" "$ext"
  echo "  ✓ $crt"
}

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  ECCAAC mTLS Certificate Generator       ║"
echo "╚══════════════════════════════════════════╝"

# ── CA ──────────────────────────────────────────────────
if [[ $DO_CA -eq 1 ]]; then
  if [[ -f "$CA_CRT" ]]; then
    echo "► CA already exists — skipping ($CA_CRT)"
  else
    echo "► Generating Root CA..."
    openssl ecparam -name prime256v1 -genkey -noout -out "$CA_KEY"
    chmod 600 "$CA_KEY"
    openssl req -new -x509 -key "$CA_KEY" -out "$CA_CRT" \
      -days "$CA_DAYS" -sha256 \
      -subj "/C=$COUNTRY/ST=$STATE/O=$ORG/CN=ECCAAC Root CA" \
      -addext "basicConstraints=critical,CA:TRUE,pathlen:0" \
      -addext "keyUsage=critical,keyCertSign,cRLSign"
    echo "  ✓ $CA_CRT  (${CA_DAYS}d / 10yr)"
    echo "  ⚠  Keep $CA_KEY secure — never distribute"
  fi
fi

if [[ $DO_SERVERS -eq 1 || -n "$AGENT_HOSTS" ]]; then
  if [[ ! -f "$CA_CRT" || ! -f "$CA_KEY" ]]; then
    echo "✗ CA not found. Run --ca first."
    exit 1
  fi
fi

# ── Server certs ────────────────────────────────────────
if [[ $DO_SERVERS -eq 1 ]]; then
  echo "► Server certificates..."
  sign_cert "risk_engine" "eccaac-risk-engine" \
    "DNS:risk-engine,DNS:localhost,IP:127.0.0.1" \
    "$SRV_DAYS" "$CA_CRT" "$CA_KEY" "$CERTS_DIR/server" 0
  sign_cert "pde" "eccaac-pde" \
    "DNS:pde,DNS:localhost,IP:127.0.0.1" \
    "$SRV_DAYS" "$CA_CRT" "$CA_KEY" "$CERTS_DIR/server" 0
fi

# ── Agent certs ─────────────────────────────────────────
if [[ -n "$AGENT_HOSTS" ]]; then
  echo "► Agent certificates..."
  IFS=',' read -ra HOSTS <<< "$AGENT_HOSTS"
  for h in "${HOSTS[@]}"; do
    h=$(echo "$h" | tr -d ' ')
    sign_cert "$h" "agent-$h" "DNS:$h" \
      "$AGT_DAYS" "$CA_CRT" "$CA_KEY" "$CERTS_DIR/agents" 1
  done
fi

echo ""
echo "Done. Enable mTLS:"
echo "  agent.yaml:  tls.enabled: true"
echo "  pde.yaml:    tls.enabled: true"
echo "  .env:        TLS_ENABLED=true"
echo ""
