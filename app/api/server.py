"""FastAPI application — serves the NovaStor customer assistant.

Best practices: resources (embedder, retriever, operations store) load once in
the lifespan; a typed-exception handler maps domain errors to HTTP codes; the
synchronous agent call runs off the event loop; the assistant is built per
request, scoped to the caller's customer_id.
"""
from __future__ import annotations

import asyncio
import logging
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage

from app.agents.graph import build_assistant
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
)
from app.config import settings
from app.data.operations import OperationsStore
from app.exceptions import AppError, ConfigError, VectorStoreError
from app.ingestion.chunker import chunk_documents
from app.ingestion.loader import load_documents
from app.logging_config import setup_logging
from app.retrieval.embeddings import get_embedder
from app.retrieval.vector_store import build_index, load_retriever

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    app.state.embedder = None
    app.state.retriever = None
    app.state.ops = None
    try:
        app.state.ops = OperationsStore()
    except Exception as exc:
        logger.warning("Operations data not loaded: %s", exc)
    try:
        app.state.embedder = get_embedder()
        app.state.retriever = load_retriever(app.state.embedder)
        logger.info("Knowledge-base index ready.")
    except Exception as exc:
        logger.warning("Index not ready (%s). POST /ingest first.", exc)
    yield
    logger.info("Shutting down.")


app = FastAPI(title="NovaStor Customer Assistant", version="1.0.0", lifespan=lifespan)


@app.exception_handler(AppError)
async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
    logger.error("%s: %s", type(exc).__name__, exc.message)
    return JSONResponse(
        status_code=exc.http_status,
        content={"error": exc.message, "detail": exc.detail},
    )


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    ops = request.app.state.ops
    return HealthResponse(
        status="ok",
        index_ready=request.app.state.retriever is not None,
        customers_loaded=len(ops._customers) if ops else 0,
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest, request: Request) -> IngestResponse:
    if request.app.state.embedder is None:
        request.app.state.embedder = get_embedder()
    source = req.source or settings.knowledge_base_dir
    chunks = chunk_documents(load_documents(source))
    await asyncio.to_thread(build_index, chunks, request.app.state.embedder)
    request.app.state.retriever = load_retriever(request.app.state.embedder)
    return IngestResponse(chunks_indexed=len(chunks), source=source)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    if request.app.state.retriever is None:
        raise VectorStoreError("Knowledge base not indexed.", detail="POST /ingest first.")
    if request.app.state.ops is None:
        raise ConfigError("Operational data unavailable.")

    # Build the assistant scoped to THIS customer (identity bound server-side).
    assistant = build_assistant(
        customer_id=req.customer_id,
        retriever=request.app.state.retriever,
        ops=request.app.state.ops,
    )
    out = await asyncio.to_thread(
        assistant.invoke, {"messages": [HumanMessage(req.message)]}
    )
    final = out["messages"][-1].content
    tools_used = [
        c["name"]
        for m in out["messages"]
        if getattr(m, "tool_calls", None)
        for c in m.tool_calls
    ]
    sources = sorted(set(re.findall(r"\[([^\]]+)\]", final)))
    return ChatResponse(answer=final, sources=sources, tools_used=tools_used)
