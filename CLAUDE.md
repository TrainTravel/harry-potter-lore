# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Harry Potter lore agent (`lore_agent.ipynb`) that uses Google ADK for agent orchestration and ChromaDB as a vector store for RAG over HP lore content.

## Design decisions

See `docs/adr/` for Architecture Decision Records. Read `docs/adr/README.md`
before proposing changes to cross-cutting concerns (observability, reliability,
schema evolution, agent topology).

## Environment Setup

The virtualenv is at `./list/` (managed by `uv`, Python 3.8):

```bash
# Activate virtualenv
source list/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

## Running the Notebook

```bash
source list/bin/activate
jupyter notebook lore_agent.ipynb
```

## Key Dependencies

- **google-adk** (`0.0.1`) — Google Agent Development Kit; the agent framework driving the lore agent
- **chromadb** — vector database for storing and querying HP lore embeddings
- **huggingface-hub** + **tokenizers** + **onnxruntime** — local embedding model inference
- **fastapi** + **uvicorn** — HTTP server if the agent is exposed as an API
- **python-dotenv** — loads API keys / config from a `.env` file

## Architecture Notes

The intended architecture is a RAG pipeline:
1. HP lore text is embedded and stored in ChromaDB
2. At query time, relevant passages are retrieved from ChromaDB
3. Google ADK orchestrates an agent that uses the retrieved context to answer lore questions

---

## deepTutor Improvements (branch: `claude/deeptutor-context-engineering-gRaiA`)

This project extends the base HP lore agent with a full **context engineering harness**
modelled on the deepTutor open-source tutoring platform, addressing five architectural
weaknesses found in that codebase.

### New modules

#### Python — `context_harness/`

| File | Purpose |
|---|---|
| `context_manager.py` | Token-budget `ContextWindow` with priority eviction and compression hooks |
| `rag_pipeline.py` | ChromaDB RAG with 4 chunking strategies and MMR reranking |
| `context_assembler.py` | 4 assembly strategies: naive, relevance, sandwich, citation |
| `prompt_templates.py` | Slot-based prompt templates: lore expert, Socratic tutor, quiz master |
| `summarizer.py` | **Fix #3** — async post-turn summarizer (see below) |
| `document_registry.py` | **Fix #2+4** — stale document tracker (see below) |
| `cost_tracker.py` | **Fix #5** — token cost aggregation pipeline (see below) |
| `index_version_guard.py` | **Fix #6** — embedding model version validation (see below) |
| `retrieval_cache.py` | **Fix #7** — LRU + TTL retrieval cache (see below) |

#### Scala Typelevel — `scala-harness/`

| File | Purpose |
|---|---|
| `Domain.scala` | Total domain model with typed errors |
| `ContextAlgebra.scala` | Tagless-final context window algebra + `Ref`-based interpreter |
| `ContextPipeline.scala` | fs2 streaming pipeline: scoreFilter → dedup → topK → assemble → inject |
| `VectorStore.scala` | VectorStore algebra + in-memory interpreter (ChromaDB stub ready) |
| `LoreRoutes.scala` | Http4s DSL: `POST /ingest`, `POST /query`, `GET /health`, `GET /cost` |
| `CostTracker.scala` | **Fix #5** — `CostTrackerAlgebra` with per-capability rollups |
| `IndexVersionGuard.scala` | **Fix #6** — manifest validation with typed errors |
| `RetrievalCache.scala` | **Fix #7** — `CachedVectorStore` decorator with hit-rate on `/health` |
| `Main.scala` | `IOApp` entry point wiring all algebras together |

### Weakness fixes applied

#### Fix #3 — Async post-turn summarization (`summarizer.py`)
**Problem:** deepTutor's `ContextBuilder` fires a blocking LLM call to compress history
when the context window overflows, stalling the current user turn.

**Fix:** `SummarizationQueue` is a background `asyncio.Task`. `SummarizingContextWindow`
overrides eviction to submit old entries to the queue non-blocking. Budget split mirrors
deepTutor: 60% recent verbatim / 40% summary slot. Completed summaries are persisted to
SQLite and atomically swapped into the window.

#### Fix #2 + #4 — Stale document tracking (`document_registry.py`)
**Problem:** deepTutor's dedup is insert-only (SHA-256 hash check prevents re-adding
identical files but does nothing if a document *changes*). FAISS has no `delete()` — any
re-index is a full rebuild.

**Fix:** `DocumentRegistry` stores chunk IDs alongside each doc's content hash. On
re-ingest: if hash changed → call ChromaDB `delete(chunk_ids)` then reingest. ChromaDB's
HNSW index supports `O(log N)` deletion; FAISS is abandoned as the primary store.

#### Fix #5 — Token cost aggregation (`cost_tracker.py`, `CostTracker.scala`)
**Problem:** deepTutor logs token counts per call but never aggregates them. No way to
query average cost per capability or total spend.

**Fix:** `CostEvent` records `(model, capability, tokens_in, tokens_out, latency_ms,
cost_usd)` per LLM call. A fire-and-forget asyncio queue writes events to SQLite without
blocking. In-memory rollups answer: cost by capability, cost by model, per-session spend.
`@track_cost` decorator adds tracking with zero boilerplate. Scala: `GET /cost` endpoint.

#### Fix #6 — Index version guard (`index_version_guard.py`, `IndexVersionGuard.scala`)
**Problem:** deepTutor's `info.json` stores embedding dimension but not model name. A
model change silently produces incompatible vectors with no error raised.

**Fix:** `index_manifest.json` stores model name + dimension + schema version. Validated
on every startup; raises a typed `IndexVersionError` (fail-fast). Includes a blue/green
swap helper for zero-downtime model migration.

#### Fix #7 — Retrieval cache (`retrieval_cache.py`, `RetrievalCache.scala`)
**Problem:** deepTutor's LlamaIndex path calls `StorageContext.from_defaults()` +
`load_index_from_storage()` on every query — no in-process cache.

**Fix:** `RetrievalCache` wraps `RAGPipeline.retrieve()` with an LRU + TTL cache
(default 512 entries, 5-minute TTL). Targeted per-collection invalidation is triggered
automatically by `DocumentRegistry.upsert()`. `CachedVectorStore` (Scala) exposes
`hit_rate` on `GET /health`.

### Running the Scala harness

```bash
cd scala-harness
sbt run          # starts Http4s server on :8080
sbt test         # runs munit-cats-effect suites
```

### Key design principles used

- **Tagless-final** — every infrastructure concern is an algebra (trait); the business
  logic depends only on the algebra, not the interpreter. Swap implementations freely.
- **fs2 streaming** — the retrieval pipeline is a composition of typed `Pipe[IO, A, B]`
  stages. Back-pressure and lazy evaluation are built in.
- **Async-first** — nothing that can be deferred (summarization, cost writes) blocks
  the critical user-turn path.
- **Fail-fast on incompatibility** — version mismatches raise typed errors at startup,
  not silent wrong answers at query time.
- **Targeted cache invalidation** — the cache knows which collection changed; it does
  not flush everything on every write.
