"""Centralised logging configuration.

Call `setup_logging()` once at startup (in main.py / the API lifespan). Then
in every other module do:

    import logging
    logger = logging.getLogger(__name__)

Using __name__ gives you per-module logger names (app.retrieval.embedder, ...)
which makes logs traceable to where they came from.
"""
from __future__ import annotations

import logging
import sys

from app.config import settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(level: str | None = None) -> None:
    """Configure root logging. Idempotent — safe to call more than once."""
    log_level = (level or settings.log_level).upper()

    root = logging.getLogger()
    root.setLevel(log_level)

    # Avoid duplicate handlers if called twice (e.g. tests + app).
    if root.handlers:
        for h in root.handlers:
            h.setLevel(log_level)
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    handler.setLevel(log_level)
    root.addHandler(handler)

    # Quiet down noisy third-party libraries.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
