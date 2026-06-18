"""Ingestion entrypoint — build the knowledge-base index.

    python -m app.ingest                 # uses settings.knowledge_base_dir
    python -m app.ingest data/knowledge_base
"""
from __future__ import annotations

import logging
import sys

from app.config import settings
from app.ingestion.chunker import chunk_documents
from app.ingestion.loader import load_documents
from app.logging_config import setup_logging
from app.retrieval.embeddings import get_embedder
from app.retrieval.vector_store import build_index

logger = logging.getLogger(__name__)


def main() -> int:
    setup_logging()
    source = sys.argv[1] if len(sys.argv) > 1 else settings.knowledge_base_dir
    chunks = chunk_documents(load_documents(source))
    build_index(chunks, get_embedder())
    logger.info("Ingestion complete from '%s'.", source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
