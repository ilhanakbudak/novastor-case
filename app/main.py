"""Entrypoint / smoke test.

Run `python -m app.main` to verify the skeleton wires together: config loads,
logging works, and exceptions behave. Replace `run()` with real logic later.
"""
from __future__ import annotations

import logging

from app.config import settings
from app.exceptions import AppError
from app.logging_config import setup_logging

logger = logging.getLogger(__name__)


def run() -> None:
    logger.info("Skeleton is alive.")
    logger.info("LLM provider=%s model=%s", settings.llm_provider, settings.llm_model)
    logger.info(
        "Embeddings provider=%s model=%s | top_k=%s chunk_size=%s",
        settings.embedding_provider,
        settings.embedding_model,
        settings.top_k,
        settings.chunk_size,
    )
    logger.debug("This DEBUG line only shows when LOG_LEVEL=DEBUG.")


def main() -> int:
    setup_logging()
    try:
        run()
    except AppError as exc:  # expected, typed errors
        logger.error("Application error: %s | detail=%s", exc.message, exc.detail)
        return 1
    except Exception:  # unexpected — log full traceback
        logger.exception("Unexpected error")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
