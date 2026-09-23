"""
Centralized logging configuration.

Uses Python's standard `logging` module with a structured format.
In production, swap the formatter for a JSON formatter (e.g. python-json-logger)
to feed structured logs into your observability stack (Datadog, Loki, etc.).
"""

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """
    Configure the root logger for the application.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Avoid duplicate handlers on re-configuration (e.g. during tests)
    if not root_logger.handlers:
        root_logger.addHandler(handler)

    # Silence overly verbose third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(
        logging.INFO if level.upper() == "DEBUG" else logging.WARNING
    )
