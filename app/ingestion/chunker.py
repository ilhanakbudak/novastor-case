"""Chunking.

Splits loaded Documents into smaller, overlapping chunks suitable for embedding
and retrieval, preserving and extending each chunk's metadata.

Why chunk at all?
  * An embedding represents a *bounded* span of meaning well. Embed a whole
    book into one vector and the meaning is averaged into mush.
  * Retrieval should return focused passages, not entire documents.
  * The LLM context window is finite and billed per token.

RecursiveCharacterTextSplitter tries a hierarchy of separators
(paragraph -> line -> sentence -> word) so chunks break at natural boundaries
instead of mid-word. `chunk_overlap` repeats a little text across consecutive
chunks so a sentence split across a boundary isn't orphaned from its context.
"""
from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.exceptions import IngestionError

logger = logging.getLogger(__name__)


def chunk_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split documents into overlapping chunks, adding a stable `chunk_id`."""
    if not documents:
        raise IngestionError("No documents provided to chunk.")

    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    # Give every chunk a stable, human-readable id: "<source>#<n>".
    # Useful for citations, deduping, and debugging which chunk was retrieved.
    counters: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        n = counters.get(source, 0)
        chunk.metadata["chunk_id"] = f"{source}#{n}"
        counters[source] = n + 1

    logger.info(
        "Split %d document(s) into %d chunk(s) (size=%d, overlap=%d)",
        len(documents),
        len(chunks),
        size,
        overlap,
    )
    return chunks
