"""
ECCAAC — Risk Scoring Engine
Service Entrypoint

Usage:
  python -m risk_engine                          # Start with defaults
  python -m risk_engine --port 50051             # Custom port
  python -m risk_engine --config config.env      # Load config from env file
  python -m risk_engine --workers 16             # More gRPC worker threads
"""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from service.grpc_service import RiskScoringServer, ServiceConfig
from utils.logging import configure_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="ECCAAC Risk Scoring Engine — gRPC Service"
    )
    parser.add_argument("--host",       default="0.0.0.0",              help="Bind address")
    parser.add_argument("--port",       type=int, default=50051,         help="gRPC port")
    parser.add_argument("--workers",    type=int, default=10,            help="Thread pool workers")
    parser.add_argument("--model",      default="./checkpoints/best_model.pt", help="Model checkpoint path")
    parser.add_argument("--stats-dir",  default="./user_stats",          help="User stats storage directory")
    parser.add_argument("--config",     default=".env",                  help="Path to .env config file")
    parser.add_argument("--log-level",  default="INFO",                  choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log-format", default="json",                  choices=["json", "pretty"])
    return parser.parse_args()


def main():
    args = parse_args()

    # Load environment config
    env_path = Path(args.config)
    if env_path.exists():
        load_dotenv(env_path)

    # Configure logging
    configure_logging(level=args.log_level, fmt=args.log_format)
    logger = logging.getLogger(__name__)

    logger.info("Starting ECCAAC Risk Scoring Engine", extra={
        "host": args.host,
        "port": args.port,
        "workers": args.workers,
        "model": args.model,
    })

    cfg = ServiceConfig(
        host=args.host,
        port=args.port,
        max_workers=args.workers,
        model_path=args.model,
        user_stats_dir=args.stats_dir,
    )

    server = RiskScoringServer(cfg)
    server.run_until_interrupt()


if __name__ == "__main__":
    main()
