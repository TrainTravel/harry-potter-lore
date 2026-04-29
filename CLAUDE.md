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

---

## DSPy mode playbook — Signature is canonical

When adding or modifying a DSPy mode, the Signature's `InputField` names are the
single source of truth. Every other place that names an input must use the
**identical string** — no aliasing, no renaming.

### Rule

1. **Signature** declares the input field names (canonical).
2. **Module.forward()** parameter names match the Signature's input field names
   verbatim. No renaming inside `forward()`.
3. **Trainset `.with_inputs(...)`** uses the identical strings.
4. **Router** (`DSPyAgent.forward`) dispatches by introspecting the Signature's
   first non-`context` input field — it does not hardcode `question=`. See
   `_primary_input_field()` in `context_harness/dspy_agent.py`.

### Why

On 2026-04-17 we hit `TypeError: DebateModule.forward() got an unexpected keyword
argument 'position'` during `optimizer.compile()`. Root cause: three places named
the input (Signature=`position`, trainset=`.with_inputs("position")`,
Module.forward=`question`), and the module renamed internally. That works for
the runtime path but breaks the compile path, because
`BootstrapFewShot.compile()` calls `teacher(**example.inputs())` — it echoes
whatever `.with_inputs(...)` declared. Hiding a mismatch inside the callee only
hides which caller is broken.

### Checklist for adding a new mode

- [ ] Signature defined (names input fields)
- [ ] Module: `forward()` params match Signature input fields verbatim
- [ ] Trainset: `.with_inputs(...)` uses identical strings (5–10 hand-labeled
      gold examples to start; full output fields populated)
- [ ] Metric function added to `context_harness/metrics.py` — write the
      *strictest* check you can defend (the metric is your filter against bad
      teacher demos at compile time)
- [ ] **Sanity-check the teacher BEFORE recompiling** —
      `python -m scripts.sanity_check_teacher --mode <name> --idx 0` runs the
      uncompiled module on one example. Compare the teacher's output to your
      gold `character_response`. If the teacher is meaningfully weaker, skip
      bootstrap: recompile with `--max-demos 0 --max-labeled <N>`. Otherwise
      defaults are fine.
- [ ] `MODE_CONFIG` entry in `evals/slo_check.py`
- [ ] Compile smoke test in `tests/test_compile_smoke.py` (asserts the mode
      survives `BootstrapFewShot.compile(...)` under `DummyLM`)
- [ ] After recompile, inspect demos in `my_profile.agent/<mode>.json` —
      verify they match your gold pattern, not the teacher's rambling

### Bootstrap vs labeled (2026-04-28)

`BootstrapFewShot` populates the compiled prompt with two demo types:
1. **Bootstrapped** — teacher LLM generates a fresh response per trainset
   input; if metric passes, that response becomes the demo
2. **Labeled** — your hand-written `character_response` (or other output
   fields) pasted into the demo verbatim

Use bootstrap when the teacher LLM is meaningfully smarter than your runtime
LLM (e.g., compile with Gemini-Pro / Claude, run on Flash-Lite). The teacher
generates diverse demos beyond what you'd write by hand.

Use labeled-only (`--max-demos 0 --max-labeled <N>`) when the teacher is
weaker than or comparable to your gold demos. This shipped the Sorting Hat
turn-3 commit (2026-04-28): flash-lite teacher couldn't reproduce
"Better be... HUFFLEPUFF!", it generated verbose explanations that mentioned
house names mid-sentence; the lax metric ("contains a house name") let those
through; verbose teacher demos drowned out the one labeled gold demo per
recompile. Labeled-only routes all hand-written gold into the prompt, runtime
imitates the canonical pattern.

### Runtime model decision (2026-04-29)

After comparing flash-lite vs Pro at runtime, **production runs Gemini 2.5 Pro**
(`DSPY_MODEL=gemini/gemini-2.5-pro`). Reasons:

- Flash-lite suffered from "trainset bleed" — when user input was thin
  (e.g., *"any concrete suggestions?"*), it regurgitated trainset demos
  even when chat_history disagreed. Pro reads chat_history correctly.
- The fix is at runtime, not compile-time. Existing flash-lite-compiled
  demos are good enough; runtime discipline is what matters.

**Cost trade-off:** ~$0.02–0.05 per turn at Pro vs ~$0.001 at flash-lite
(12.5–50× more expensive). At small-scale traffic this is pennies; revisit
if usage grows.

**Compile-time still uses flash-lite.** Bootstrap teacher quality matters
less than runtime quality for our case. (See "Bootstrap vs labeled" above
for when teacher choice matters.)

A/B experiment data at `drafts/ab_perspective_shift_results.md` —
DSPy compiled vs simple system-prompt for character modes. Verdict:
roughly tied on quality; DSPy kept for architectural consistency.

### Known gap (as of 2026-04-17)

`slo_check.MODE_CONFIG` wires only `deep_research`, `guided_learning`, `debate`.
Four modes (`satirical_podcast`, `perspective_shift`, `open_analysis`,
`exam_grader`) are not yet gated — add them when their EVALSETs stabilise.
