# ECCAAC — Risk Scoring Engine

**Enterprise Continuous Context-Aware Access Control**
Python microservice — LSTM Siamese Network behavioral authentication engine.

---

## Overview

The Risk Scoring Engine is the intelligence core of the ECCAAC system. It continuously evaluates behavioral and contextual signals from every active user session and produces a 0–100 risk score that drives access control decisions.

**Core idea:** Instead of authenticating once at login, the system continuously asks — *"Is this still the legitimate user?"* — using a trained behavioral profile unique to each person.

---

## Architecture

```
Go Agent (device)
      │
      │  gRPC (stream FeatureVector every 10s)
      ▼
┌─────────────────────────────────────────┐
│         gRPC Service                    │
│  ┌─────────────┐  ┌──────────────────┐  │
│  │ Preprocessor│  │  Profile Store   │  │
│  │  Validator  │  │  (anchor seqs)   │  │
│  │  Normalizer │  └──────────────────┘  │
│  │  Seq Buffer │           │            │
│  └──────┬──────┘           │            │
│         │ sequence (6×24)  │ anchor     │
│         ▼                  ▼            │
│  ┌──────────────────────────────────┐   │
│  │    LSTM Siamese Network          │   │
│  │   Branch A ──── Branch B         │   │
│  │   (anchor)      (live)           │   │
│  │         └──distance──┘           │   │
│  │       similarity + classification│   │
│  └──────────────────┬───────────────┘   │
│                     │                   │
│  ┌──────────────────▼───────────────┐   │
│  │         Risk Scorer              │   │
│  │  Behavioral + Context + Device   │   │
│  │         → Risk Score 0–100       │   │
│  │         → Enforcement Action     │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
      │
      │  RiskScoreResponse (score, level, enforcement)
      ▼
Policy Decision Server (Go)
```

---

## Project Structure

```
eccaac-risk-engine/
├── risk_engine/
│   ├── __init__.py           # Public API exports
│   ├── __main__.py           # Service entrypoint (python -m risk_engine)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── siamese_network.py   # LSTM Siamese Network (PyTorch)
│   │   ├── preprocessor.py      # Feature validation, normalization, buffering
│   │   ├── trainer.py           # Training loop + continuous fine-tuning
│   │   └── risk_scorer.py       # Risk score computation + enforcement
│   ├── service/
│   │   ├── __init__.py
│   │   └── grpc_service.py      # gRPC servicer + server lifecycle
│   ├── proto/
│   │   ├── __init__.py
│   │   ├── risk_engine_pb2.py        # Generated — do not edit
│   │   └── risk_engine_pb2_grpc.py   # Generated — do not edit
│   └── utils/
│       ├── __init__.py
│       └── logging.py           # Structured JSON logging setup
├── tests/
│   ├── __init__.py
│   ├── test_siamese_network.py
│   ├── test_preprocessor.py
│   ├── test_trainer.py
│   ├── test_risk_scorer.py
│   └── test_grpc_service.py
├── proto/
│   └── risk_engine.proto        # Source of truth for gRPC contract
├── scripts/
│   └── compile_proto.sh         # Generate Python + Go stubs from .proto
├── checkpoints/                 # Model weights (gitignored)
├── user_stats/                  # Per-user normalization stats (gitignored)
├── profiles/                    # Per-user anchor sequences (gitignored)
├── .env.example                 # Config template — copy to .env
├── .gitignore
├── Makefile
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
# CPU only (on-premise server without GPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Or with GPU (CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your database credentials and paths
```

### 3. Compile proto stubs

```bash
make proto
```

### 4. Train the model (first time)

```python
from risk_engine import SiameseTrainer, TrainingConfig, BehavioralPairDataset

# Load your collected behavioral data
# user_sequences: dict[user_id → list of (6, 24) arrays]
dataset = BehavioralPairDataset(user_sequences, attack_sequences)
trainer = SiameseTrainer(model, cfg=TrainingConfig())
trainer.train(dataset)
```

### 5. Run the service

```bash
# Development
make run

# Production
make run-prod
```

---

## Risk Score Formula

```
Risk Score = Behavioral Deviation + Context Deviation + Device Risk

Component              Source                           Max
──────────────────────────────────────────────────────────
Behavioral Deviation   LSTM Siamese Network             50
Context Deviation      Network / IP / Geo signals       30
Device Risk            Device fingerprint / processes   20
──────────────────────────────────────────────────────────
Total                                                   100
```

### Risk Levels & Enforcement

| Score  | Level     | Action                                        |
|--------|-----------|-----------------------------------------------|
| 0–25   | LOW       | Full access. Passive monitoring.              |
| 26–50  | ELEVATED  | Step-up authentication (MFA prompt).          |
| 51–75  | HIGH      | Throttle access. Block sensitive resources.   |
| 76–100 | CRITICAL  | Lock session immediately. Alert admin.        |

---

## User Onboarding Phases

| Phase       | Sessions | Behavior                                      |
|-------------|----------|-----------------------------------------------|
| OBSERVATION | 1–5      | Data collection only. No risk scoring.        |
| SOFT        | 6–10     | Risk scoring active, thresholds ×2 relaxed.   |
| FULL        | 11+      | Full enforcement. Profile considered mature.  |

---

## Testing

```bash
# Run all tests
make test

# With coverage report
make test-cov

# Run all module smoke tests
make test-smoke
```

---

## Development

```bash
# Lint
make lint

# Format code
make format

# Type checking
make typecheck
```

---

## Tech Stack

| Component       | Technology              |
|-----------------|-------------------------|
| ML Framework    | PyTorch 2.x             |
| Network model   | LSTM Siamese Network    |
| gRPC            | grpcio + protobuf       |
| Preprocessing   | NumPy, scikit-learn     |
| Database        | PostgreSQL + SQLAlchemy |
| Logging         | structlog (JSON)        |
| Config          | pydantic-settings       |
| Testing         | pytest                  |

---

*ECCAAC — Built by David*
