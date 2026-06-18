# NovaStor Customer Assistant

An AI assistant for NovaStor Logistics' corporate customer portal. It answers two kinds of
question through one conversational endpoint: **knowledge questions** about NovaStor's
services, policies, SLAs, pricing, and contract terms (answered with grounded,
**cited** Retrieval-Augmented Generation over the document knowledge base), and
**account questions** about the calling customer's own storage usage, contract, invoice,
and shipments (answered via tools over operational data, strictly scoped to that customer).

An agent decides, per message, which path to take.

---

## Architecture

```
                          POST /chat {customer_id, message}
                                       │
                                       ▼
                         Assistant (LangGraph ReAct agent)
                         scoped to customer_id, decides:
            ┌──────────────────────────┼───────────────────────────┐
            ▼                          ▼                            ▼
   search_knowledge_base      get_account_summary           get_shipment_status
   (RAG over docs)            list_my_shipments             (customer-scoped)
            │                          │                            │
            ▼                          ▼                            ▼
   Chroma vector store        OperationsStore (customers.json / shipments.json)
            │
            ▼
   grounded, cited answer ─────────────────────────────────► {answer, sources, tools_used}
```

- **Knowledge path:** documents are ingested (load → chunk → embed → Chroma) once; at query
  time the agent searches them and answers only from retrieved passages, citing sources.
- **Account path:** the `OperationsStore` serves customer/shipment data; the account tools are
  **bound to the calling customer's id**, so cross-customer access is impossible.

See `docs/ARCHITECTURE.md` for the full rationale.

## Project structure

```
app/
  config.py            # typed settings (providers + data paths)
  logging_config.py    # centralized logging
  exceptions.py        # typed errors (carry http_status)
  llm.py               # provider-agnostic chat LLM factory
  ingest.py            # build the knowledge-base index
  ingestion/           # loader (txt/md/pdf), chunker (+ metadata)
  retrieval/           # embeddings (cloud/local), Chroma vector store
  data/operations.py   # OperationsStore: customer/shipment lookups, isolation
  agents/
    tools.py           # knowledge search + customer-scoped account tools
    graph.py           # the per-customer assistant (routing + grounding prompt)
  api/
    schemas.py         # ChatRequest/ChatResponse + health/ingest
    server.py          # FastAPI app (lifespan, error handler, /chat /ingest /health)
data/                  # provided knowledge base + customers.json + shipments.json
docs/ARCHITECTURE.md
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env       # then fill in keys, or use the offline profile
```

Profiles (in `.env`):
- **Cloud:** `EMBEDDING_PROVIDER=openai`, `LLM_PROVIDER=openai`, `OPENAI_API_KEY=sk-...`
- **Offline (no key):** `EMBEDDING_PROVIDER=sentence-transformers`, `LLM_PROVIDER=ollama`

## Usage

```bash
# 1) Build the knowledge-base index (run once / when docs change)
python -m app.ingest                       # uses data/knowledge_base

# 2) Start the API
uvicorn app.api.server:app --port 8000
# interactive docs: http://localhost:8000/docs
```

Ask a question (note `customer_id` represents the authenticated session):
```bash
curl -X POST localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"customer_id":"CUST-001","message":"How is storage billed?"}'

curl -X POST localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"customer_id":"CUST-001","message":"How much of our quota are we using?"}'
```

## Sample input / output

Request:
```json
{"customer_id": "CUST-001", "message": "What is the status of shipment SH-1042?"}
```
Response:
```json
{
  "answer": "Shipment SH-1042 is In Transit from Hamburg DC to Munich Plant, 320 items, ETA 2026-06-19.",
  "sources": [],
  "tools_used": ["get_shipment_status"]
}
```

Request (knowledge):
```json
{"customer_id": "CUST-001", "message": "How is storage billed?"}
```
Response (abridged):
```json
{
  "answer": "Storage is billed per cubic meter per month against your contracted quota; usage above quota is billed at 1.5x the standard rate [billing_and_pricing.md].",
  "sources": ["billing_and_pricing.md"],
  "tools_used": ["search_knowledge_base"]
}
```

## API

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | — | `{status, index_ready, customers_loaded}` |
| POST | `/ingest` | `{source?}` | `{chunks_indexed, source}` |
| POST | `/chat` | `{customer_id, message}` | `{answer, sources, tools_used}` |

## Models and libraries (and why)

| Component | Choice | Why |
|---|---|---|
| Agent | LangGraph ReAct | stateful tool-using loop; routes between knowledge and account tools |
| LLM | OpenAI `gpt-4o-mini` (default), Anthropic/Ollama optional | cheap, capable, swappable; Ollama enables offline |
| Embeddings | OpenAI `text-embedding-3-small`, local `all-MiniLM-L6-v2` | quality by default, offline fallback |
| Vector store | Chroma | persistent, stores metadata, minimal setup |
| Chunking | RecursiveCharacterTextSplitter | natural-boundary splits |
| API | FastAPI + uvicorn | validation, async, auto docs |
| Config | pydantic-settings | type-safe config from `.env` |

## Data sent to external services

- With `EMBEDDING_PROVIDER=openai`: knowledge-base text (at ingest) and the user's message
  (at query) are sent to OpenAI for embedding.
- With `LLM_PROVIDER=openai`/`anthropic`: the user's message and retrieved passages are sent
  to that provider to generate the answer.
- With the offline profile (`sentence-transformers` + `ollama`): **no data leaves the machine**.

Customer account and shipment data is read locally from JSON and is only sent to an LLM
provider if a tool result is included in the generated answer; consider this when choosing a
provider for production. API keys are read from `.env` and are never committed.

## Security / data isolation

Account tools are constructed per request with the caller's `customer_id` bound server-side.
The model cannot pass an arbitrary customer id: `get_account_summary()` has no customer
argument, and `get_shipment_status(shipment_id)` verifies ownership and returns a neutral
"not on your account" message otherwise. Authorization identity never comes from the model.

## Limitations, assumptions, improvements

**Assumptions:** `customer_id` arrives from an authenticated session (here passed in the
request body for the exercise); documents have extractable text; one knowledge corpus.

**Limitations:** dense retrieval only (no hybrid/rerank); the assistant is rebuilt per request
(simple and secure, but rebuildable via LangGraph config for higher throughput); no auth or
rate limiting on the API; in-memory operational data (no real DB).

**Improvements:** real auth (JWT/session) feeding `customer_id`; pass `customer_id` via
LangGraph `config` to avoid per-request rebuilds; hybrid retrieval + reranker; streaming
responses; conversation memory across turns; an evaluation harness (Ragas) for retrieval and
faithfulness.
