"""Document loading / parsing.

Turns raw files on disk into a list of LangChain `Document` objects, each with
`page_content` (the text) and `metadata` (where it came from). Supports .txt,
.md and .pdf. Real-world corpora are messy, so loading is the stage most likely
to throw — hence the typed IngestionError wrapping.
"""
from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

from app.exceptions import IngestionError

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


def load_documents(path: str | Path) -> list[Document]:
    """Load every supported file under `path` (a file or a directory)."""
    path = Path(path)

    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
    elif path.is_file():
        files = [path]
    else:
        raise IngestionError(f"Path does not exist: {path}")

    if not files:
        raise IngestionError(
            f"No supported documents found under {path}",
            detail=f"supported types: {sorted(SUPPORTED_SUFFIXES)}",
        )

    documents: list[Document] = []
    for file in files:
        documents.extend(_load_one(file))

    logger.info("Loaded %d document section(s) from %d file(s)", len(documents), len(files))
    return documents


def _load_one(file: Path) -> list[Document]:
    """Load a single file into one or more Documents (PDFs become one per page)."""
    suffix = file.suffix.lower()
    try:
        if suffix in {".txt", ".md"}:
            text = file.read_text(encoding="utf-8", errors="ignore")
            return [Document(page_content=text, metadata={"source": file.name})]

        if suffix == ".pdf":
            reader = PdfReader(str(file))
            pages: list[Document] = []
            for i, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:  # skip blank/scanned pages with no extractable text
                    pages.append(
                        Document(page_content=text, metadata={"source": file.name, "page": i})
                    )
            if not pages:
                raise IngestionError(
                    f"No extractable text in {file.name}",
                    detail="PDF may be scanned images — OCR would be needed.",
                )
            return pages

        raise IngestionError(f"Unsupported file type: {file.suffix}")

    except IngestionError:
        raise  # already typed — don't double-wrap
    except Exception as exc:  # unexpected parse failure
        raise IngestionError(f"Failed to load {file.name}", detail=str(exc)) from exc
