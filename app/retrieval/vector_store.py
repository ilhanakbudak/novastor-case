"""Vector store (Chroma).

Two responsibilities:
  * build_index(chunks, embedder)  -> embed chunks and persist them to disk
  * load_retriever(embedder)       -> open the persisted store, return a
                                      retriever that fetches the top-k chunks

We use Chroma because it persists to disk, stores text + metadata next to the
vectors, and supports metadata filtering — all with almost no setup. FAISS is
the alternative when raw speed/scale outweighs convenience.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

from app.config import settings
from app.exceptions import VectorStoreError

logger = logging.getLogger(__name__)


def build_index(
    chunks: list[Document],
    embedder: Embeddings,
    persist_dir: str | None = None,
    reset: bool = True,
):
    """Embed chunks and persist them to a Chroma store on disk.

    This is the INGESTION step: run it once when documents change, not on every
    query. Re-running with reset=True wipes the store directory for a clean
    single index (clearing only the Chroma collection via the API leaves
    orphaned segment folders on disk that pile up across runs).

    Call this from a dedicated ingestion process; the query/serving path should
    use load_retriever() to open the existing store instead of rebuilding.
    """
    if not chunks:
        raise VectorStoreError("No chunks provided to index.")
    persist_dir = persist_dir or settings.vector_store_dir

    try:
        from langchain_chroma import Chroma

        if reset and Path(persist_dir).exists():
            shutil.rmtree(persist_dir)  # true clean slate — no orphaned segments
            logger.info("Reset (wiped) vector store at '%s'", persist_dir)

        # Stable, unique id per chunk -> idempotent if you ever re-add.
        ids = [c.metadata.get("chunk_id") or f"auto#{i}" for i, c in enumerate(chunks)]

        store = Chroma.from_documents(
            documents=chunks,
            embedding=embedder,
            persist_directory=persist_dir,
            ids=ids,
        )
        logger.info("Indexed %d chunk(s) into Chroma at '%s'", len(chunks), persist_dir)
        return store
    except Exception as exc:
        raise VectorStoreError("Failed to build the vector index", detail=str(exc)) from exc


def load_retriever(
    embedder: Embeddings,
    persist_dir: str | None = None,
    k: int | None = None,
) -> VectorStoreRetriever:
    """Open the persisted store and return a top-k retriever.

    The embedder passed here MUST be the same kind used to build the index.
    """
    persist_dir = persist_dir or settings.vector_store_dir
    k = k or settings.top_k

    if not Path(persist_dir).exists():
        raise VectorStoreError(
            f"No vector store at '{persist_dir}'",
            detail="Build the index before querying.",
        )

    try:
        from langchain_chroma import Chroma

        store = Chroma(persist_directory=persist_dir, embedding_function=embedder)
        logger.info("Loaded retriever from '%s' (k=%d)", persist_dir, k)
        return store.as_retriever(search_kwargs={"k": k})
    except Exception as exc:
        raise VectorStoreError("Failed to load the retriever", detail=str(exc)) from exc
