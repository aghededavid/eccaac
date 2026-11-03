# ─────────────────────────────────────────────────────────
#  ECCAAC — Risk Scoring Engine
#  Makefile — common development and operations tasks
# ─────────────────────────────────────────────────────────

.PHONY: install install-dev proto lint format typecheck test test-cov run clean

# ── Setup ─────────────────────────────────────────────────

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

# ── Proto ─────────────────────────────────────────────────

proto:
	bash scripts/compile_proto.sh

# ── Code Quality ──────────────────────────────────────────

lint:
	ruff check risk_engine/ tests/

format:
	black risk_engine/ tests/
	ruff check --fix risk_engine/ tests/

typecheck:
	mypy risk_engine/ --ignore-missing-imports

# ── Testing ───────────────────────────────────────────────

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=risk_engine --cov-report=html --cov-report=term-missing

test-smoke:
	python risk_engine/core/siamese_network.py
	python risk_engine/core/preprocessor.py
	python risk_engine/core/trainer.py
	python risk_engine/core/risk_scorer.py
	python risk_engine/service/grpc_service.py

# ── Run ───────────────────────────────────────────────────

run:
	python -m risk_engine --log-format pretty

run-prod:
	python -m risk_engine \
		--port 50051 \
		--workers 16 \
		--log-format json \
		--log-level INFO

# ── Cleanup ───────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage .mypy_cache
