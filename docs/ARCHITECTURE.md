# Architecture & Design Decisions

## Overview

The assistant is a per-customer LangGraph ReAct agent fronted by a FastAPI `/chat` endpoint.
It unifies two data modalities — unstructured documents (RAG) and structured operational data
(tools) — behind one conversational interface, and routes between them with a system prompt.

## Component responsibilities

| Module | Responsibility |
|---|---|
| `config.py` | Typed settings: providers, retrieval params, data paths. |
| `ingestion/` | Parse txt/md/pdf → chunk with overlap → attach `source`/`page`/`chunk_id`. |
| `retrieval/` | Provider-agnostic embeddings; Chroma `build_index` / `load_retriever`. |
| `data/operations.py` | Load JSON once; account/shipment lookups with per-customer isolation. |
| `agents/tools.py` | Knowledge-search tool + customer-scoped account tools. |
| `agents/graph.py` | The ReAct agent + routing/grounding system prompt. |
| `api/server.py` | Lifespan resource loading, DI, typed-error handler, endpoints. |

## Key decisions and rationale

**Unify two modalities behind an agent.** Knowledge questions need RAG; account questions need
exact structured lookups. Rather than guess intent with brittle keyword rules, the agent's LLM
routes each message to the right tool. This is the natural fit for "agentic AI": the model
reasons about *what kind* of question it is and acts accordingly.

**Grounding for knowledge, tools for facts.** Knowledge answers must come only from retrieved
passages and cite their source, with an explicit refusal when the documents don't cover the
question — this prevents the model from inventing policy or pricing. Account answers come
verbatim from tool output, never from the model's imagination.

**Data isolation as a first-class concern.** The account tools are built per request with the
caller's `customer_id` closed over. The model cannot select whose data to read: account
summary takes no customer argument, and shipment lookups verify ownership and otherwise return
a neutral "not on your account". The principle — *authorization identity must come from the
session, never from the model* — defends against a user (or a prompt injection) trying to read
another company's data.

**Ingestion separated from serving.** Building the index (embedding) is expensive and occasional;
it runs via `python -m app.ingest` or `POST /ingest`. The `/chat` path only loads the retriever.

**Typed exceptions → HTTP.** Each domain error carries `http_status`; a single FastAPI handler
maps them (`NotFoundError→404`, `VectorStoreError→500`, `ConfigError→500`, `DataSourceError→500`).

**Resources loaded once; non-blocking calls.** The embedder, retriever, and `OperationsStore`
load in the FastAPI lifespan. The synchronous agent call runs via `asyncio.to_thread` so a slow
LLM call doesn't block the event loop.

## Request flow (`POST /chat`)

1. Validate `{customer_id, message}` (pydantic).
2. Build a customer-scoped assistant (tools bound to `customer_id`).
3. Agent decides: search knowledge base, call an account tool, or answer directly.
4. Tool results loop back; the LLM produces a grounded, cited final answer.
5. Return `{answer, sources, tools_used}`.

## Trade-offs & extension points

- The assistant is rebuilt per request (clear and secure). For higher throughput, compile the
  graph once and pass `customer_id` via LangGraph `config["configurable"]`, with tools reading it.
- Retrieval is dense-only; add BM25 + a reranker for higher precision.
- Operational data is in-memory JSON; swap `OperationsStore` for a database-backed implementation
  behind the same interface without touching the agent.
