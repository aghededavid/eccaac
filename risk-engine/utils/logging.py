"""
ECCAAC — Risk Scoring Engine
Structured Logging Configuration
"""

import logging
import sys
import structlog


def configure_logging(level: str = "INFO", fmt: str = "json"):
    """
    Configure structured logging for the service.

    Args:
        level:  Log level string (DEBUG / INFO / WARNING / ERROR)
        fmt:    Output format — "json" for production, "pretty" for development
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if fmt == "pretty":
        renderer = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(log_level)

    # Quieten noisy third-party loggers
    logging.getLogger("grpc").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
