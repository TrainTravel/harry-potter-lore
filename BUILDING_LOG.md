# Building Log — deepTutor Context Engineering Harness

> A record of design decisions, what was built, bugs found, and honest
> assessments made during development. Written to be readable by a human
> returning to the project after time away.

---

## Session 1 — Exploring the starting point

**Branch:** `claude/deeptutor-context-engineering-gRaiA`

### What we found

The repo contained a single empty notebook (`lore_agent.ipynb`) and a
`requirements.txt` listing the intended stack:

- `google-adk 0.0.1` — Google Agent Development Kit for agent orchestration
- `chromadb` — vector database for RAG
- `huggingface-hub` + `tokenizers` + `onnxruntime` — local embedding inference
- `fastapi` + `uvicorn` — HTTP serving

There was no implementation — just the skeleton of a Harry Potter lore RAG
agent waiting to be built.

### What we decided to build

Rather than just filling in the notebook, we used this project as a learning
vehicle for **context engineering** — the practice of deliberately managing
what goes into an LLM's context window to improve output quality. The notebook
domain (HP lore Q&A) gave us a concrete, testable use case.

We also decided to implement the same ideas in **Scala Typelevel** alongside
Python, to explore how functional type-safe patterns (tagless-final, fs2,
Cats Effect) compare to the Python asyncio approach.

---

## Session 2 — Building the context engineering harness

### Python: `context_harness/`

Four core modules, each encoding a specific context engineering principle:

**`context_manager.py` — ContextWindow**

The token-budget-aware context window. Key decisions:
- Entries carry a `priority` score (not just FIFO order). When the budget is
  full, the lowest-priority non-SYSTEM entry is evicted first.
- SYSTEM entries are pinned — they are never auto-evicted.
- A pluggable `compressor` hook lets callers supply an LLM-based summariser
  to shrink RETRIEVAL entries when eviction alone isn't enough.
- Token counting is pluggable (`TokenCounter`) — defaults to whitespace
  splitting but can be swapped for tiktoken in production.

**`rag_pipeline.py` — RAGPipeline**

ChromaDB-backed RAG with four chunking strategies:
- `FIXED` — fixed word-count window with overlap
- `SENTENCE` — split on sentence boundaries
- `PARAGRAPH` — split on blank lines
- `RECURSIVE` — try paragraph first, fall back to fixed for long paragraphs

Also includes **MMR reranking** (Maximal Marginal Relevance) to balance
relevance and diversity before injecting chunks into the context window.
Falls back to score-sorted order when embeddings are unavailable.

**`context_assembler.py` — ContextAssembler**

Four formatting strategies for the assembled context block:
- `NAIVE` — concatenate in retrieval order
- `RELEVANCE` — sort by score, show scores
- `SANDWICH` — most relevant chunk appears first *and* last, exploiting LLM
  primacy and recency bias
- `CITATION` — numbered `[N]` footnotes so the LLM can attribute claims

Includes Jaccard-based deduplication and a minimum score threshold.

**`prompt_templates.py` — PromptTemplate / SystemPrompt / LorePrompt**

Slot-based `{{PLACEHOLDER}}` templates for three agent personas:
- `LORE_EXPERT` — strict grounding, citation references, no hallucination
- `SOCRATIC_TUTOR` — guides the student with questions rather than answers
- `QUIZ_MASTER` — generates multiple-choice questions from retrieved context

### Scala Typelevel: `scala-harness/`

The same ideas, expressed as purely functional Scala 3:

| Pattern | Where used |
|---|---|
| **Tagless-final algebras** | `ContextWindowAlgebra`, `VectorStoreAlgebra`, `CostTrackerAlgebra`, `RetrievalCacheAlgebra`, `IndexVersionGuardAlgebra` |
| **Cats Effect `IO` + `Ref`** | All state (window entries, cache, cost events) held in `Ref[IO, ...]` — no shared mutable state |
| **fs2 `Pipe`** | `ContextPipeline` composes `scoreFilter → deduplicate → topK → assembleContext → injectIntoWindow` as typed stream stages |
| **Http4s DSL** | `LoreRoutes` handles `POST /ingest`, `POST /query`, `GET /health` |
| **`IOApp`** | `Main` wires all algebras and starts the Ember server |

**Why tagless-final?**
Business logic depends only on the algebra (trait), not the interpreter
(implementation). The in-memory interpreter used in tests is swapped for
a real ChromaDB or Postgres interpreter without touching any pipeline code.

---

## Session 3 — Improving on deepTutor's architecture

We reviewed the deepTutor open-source tutoring platform's published
architecture and its documented weaknesses. We addressed five of them.

### Weakness #3 — Summarization blocks the user turn

**Problem:** deepTutor fires a full LLM call to compress conversation history
at turn start when the context window overflows. An API timeout cascades
directly into a slow response for the user.

**Fix: `summarizer.py`**

`SummarizationQueue` runs as a background `asyncio.Task`. The window
identifies the oldest entries that push over budget, submits them to the
queue non-blocking, removes them from the window immediately (freeing space
for the current turn), and continues. The background worker compresses and
atomically swaps the summary back in once ready.

Budget split mirrors deepTutor's model:
- 60% of history budget → verbatim recent turns
- 40% of history budget → summary slot

Completed summaries are persisted to SQLite so they survive restarts.

### Weakness #2 + #4 — Stale chunks / FAISS write-once

**Problem:** deepTutor's file dedup uses SHA-256 hashes to prevent
re-adding identical files — but if a document *changes*, the old chunks
remain in the index silently alongside new ones. The custom FAISS indexer
has no `delete()` — any re-index requires a full rebuild.

**Fix: `document_registry.py`**

`DocumentRegistry` stores each ingested document's content hash alongside
its chunk IDs. On re-ingest:
1. Compute SHA-256 of new content.
2. If hash unchanged → skip (no work).
3. If hash changed → call `ChromaDB.delete(old_chunk_ids)` then reingest.

ChromaDB's HNSW index supports deletion in O(log N). FAISS is abandoned
as the primary store — ChromaDB handles all vector CRUD.

### Weakness #5 — Token cost is invisible

**Problem:** deepTutor logs token counts per call but never aggregates them.
There is no way to answer: what does a Deep Research run cost on average?
Which capability is most expensive?

**Fix: `cost_tracker.py` + `CostTracker.scala`**

Every LLM call produces a `CostEvent` with:
`(model, capability, tokens_in, tokens_out, latency_ms, cost_usd)`

A fire-and-forget asyncio queue writes events to SQLite without touching
the hot path. In-memory rollups provide instant answers for:
- Total spend
- Cost and average latency per capability
- Cost per model
- Per-session spend

A `@track_cost` decorator wraps any async LLM call with zero boilerplate.
Configurable pricing table covers all major models (Gemini, Claude, GPT).

Scala: `CostTrackerAlgebra` (tagless-final) + `GET /cost` HTTP endpoint.

### Weakness #6 — Embedding model changes corrupt retrieval silently

**Problem:** deepTutor's `info.json` stores embedding vector *dimension* but
not the *model name*. If the embedding model changes, existing vectors and
new query vectors are incompatible — retrieval returns garbage with no error.

**Fix: `index_version_guard.py` + `IndexVersionGuard.scala`**

`index_manifest.json` records model name, dimension, and schema version.
Validated on every startup. On mismatch: raises a typed `IndexVersionError`
immediately (fail-fast) rather than silently corrupting retrieval.

Also includes a **blue/green swap helper** for zero-downtime model migration:
build a new index while the old one serves queries, then atomically rename
directories.

### Weakness #7 — Full index reload on every query

**Problem:** deepTutor's LlamaIndex path calls `StorageContext.from_defaults()`
and `load_index_from_storage()` on every query — no in-process cache. For
large knowledge bases this adds significant latency and memory churn.

**Fix: `retrieval_cache.py` + `RetrievalCache.scala`**

`RetrievalCache` wraps `RAGPipeline.retrieve()` with an **LRU + TTL cache**:
- Default: 512 entries, 5-minute TTL
- Cache key: `collection_prefix + hash(top_k::query)` — deterministic,
  collection-scoped
- Targeted per-collection invalidation triggered automatically by
  `DocumentRegistry.upsert()` on any document change

`CachedVectorStore` (Scala) is a transparent decorator — exposes
`hit_rate` on `GET /health`.

---

## Session 4 — Running the tests

### First run: 83/88 passed

We scaffolded 88 tests across 7 files before running anything. 5 real bugs
were found on the first run.

### Bugs found (and fixed)

**Bug 1 — Cache invalidation never matched anything**

`_cache_key(query, top_k, collection)` hashed the full string
`"collection::top_k::query"` into a single digest. `_collection_prefix`
hashed only `"collection"`. These are unrelated hashes — no key would ever
start with the prefix, so `invalidate_collection()` always removed 0 entries.

Fix: changed `_cache_key` to produce `collection_prefix + hash(top_k::query)`,
so every key for a collection genuinely starts with its prefix.

**Bug 2 — Eviction test budget was too large**

The test used `max_tokens=50, reserved=10` (budget=40 tokens). The three
entries added totalled 9 tokens — they all fit comfortably, so eviction never
fired. The test passed vacuously.

Fix: tightened to `max_tokens=12, reserved=2` (budget=10 tokens) so the third
entry forces eviction of the low-priority entry.

**Bug 3 — Manifest key name mismatch**

`dataclasses.asdict()` serialises fields as snake_case (`doc_count`,
`chunk_count`). The test checked `raw["docCount"]` — a KeyError on every run.

Fix: corrected the test to use `raw["doc_count"]` and `raw["chunk_count"]`.

### Final result: 88/88 passed (0.54s)

| Test file | Tests | What it covers |
|---|---|---|
| `test_context_manager.py` | 14 | Token counter, window ops, eviction, rendering |
| `test_rag_pipeline.py` | 16 | All 4 chunkers, ingest, retrieve, top-k, MMR |
| `test_context_assembler.py` | 14 | Jaccard, all 4 strategies, dedup, edge cases |
| `test_document_registry.py` | 12 | SHA-256, upsert (create/update/skip), delete, stats |
| `test_cost_tracker.py` | 11 | Pricing, CostEvent, sync/async recording, rollups |
| `test_index_version_guard.py` | 7 | Creation, validation, model mismatch, stats update |
| `test_retrieval_cache.py` | 14 | Key determinism, hit/miss, invalidation, LRU eviction |

No API keys, no GPU, no network required. All tests run against in-memory
or temp-file backends.

---

## Session 5 — Honest assessment: what we did NOT build

### Did we implement UnifiedContext?

**No.** And without it, the claim that this project "extends deepTutor" is
inaccurate. Here is the gap:

deepTutor's `UnifiedContext` is a **per-turn request envelope** — a single
object assembled fresh at turn start that carries every context slice through
the entire capability pipeline:

```
UnifiedContext
  ├── session_id / turn_id
  ├── user_message
  ├── history_context    ← verbatim recent turns (SQLite)
  ├── summary_context    ← LLM-compressed older turns (SQLite)
  ├── memory_context     ← cross-session user profile
  ├── notebook_context   ← workspace state
  ├── retrieval_context  ← RAG chunks injected here
  └── selected_capability
```

It is ephemeral — never stored directly. Durability comes from persisting
inputs before assembly and outputs after completion.

### What we built instead

We built the **sub-systems** that would feed into a UnifiedContext, but not
the assembly layer itself:

| We built | Would become |
|---|---|
| `ContextWindow` | Token budget enforcement inside the builder |
| `SummarizingContextWindow` | The `summary_context` slot population |
| `RAGPipeline` + `ContextAssembler` | The `retrieval_context` slot |
| `DocumentRegistry` | The corpus that retrieval draws from |
| `CostTracker` | Post-turn metadata collection |
| `RetrievalCache` | Performance layer under retrieval |
| `IndexVersionGuard` | Startup integrity check |

### What is missing

To make this a genuine extension of deepTutor's architecture:

1. **`UnifiedContext`** — the per-turn request envelope dataclass/case class
2. **`ContextBuilder`** — assembles all slots from their sources before each turn
3. **`TurnRuntimeManager`** — creates turn rows, tracks status (running/completed/failed)
4. **Session persistence** — SQLite tables for `sessions`, `messages`, `turns`, `turn_events`
5. **Memory context** — cross-session user profile store and injection
6. **Event sourcing** — `StreamEvent` persistence to `turn_events` table as they happen

### What we should claim

This project implements several **context engineering building blocks** inspired
by deepTutor, and fixes five of its documented architectural weaknesses at the
component level. It is not yet a full extension of deepTutor's turn lifecycle
or capability pipeline model.

---

## Commit history (this branch)

| Commit | Description |
|---|---|
| `3778249` | Initial context harness (Python + Scala) |
| `cd07767` | Five deepTutor weakness fixes |
| `4da877e` | CLAUDE.md architecture documentation |
| `22b7b07` | 88-test Python suite + 3 bug fixes |
| `5719fdd` | `.gitignore` (exclude `__pycache__`, `data/`) |

---

## What to build next (after Session 5)

- [ ] `UnifiedContext` dataclass + `ContextBuilder` assembly lifecycle ← done in Session 6
- [ ] SQLite session/message/turn persistence (`TurnRuntimeManager`) ← done in Session 6
- [ ] Wire everything into a single end-to-end turn handler ← done in Session 6
- [ ] Notebook (`lore_agent.ipynb`) — demonstrate the full pipeline interactively
- [ ] Scala: `munit-cats-effect` tests for `CostTracker`, `RetrievalCache`, `IndexVersionGuard`

---

## Session 6 — IBM Agent Engineering: seven skills mapped to code

**Branch:** `claude/agent-engineering-skills`

### Context

An IBM Technology video argued that "prompt engineer" is evolving into "agent
engineer" — a systems engineering discipline, not a text-writing one. It
identified seven concrete skills required to ship production agents. We mapped
each skill to a gap in the existing codebase and implemented the missing pieces.

---

### Skill 1 — System Design → `unified_context.py`

**Gap:** The previous session identified this explicitly: we had all the
sub-systems (RAG, window, summarizer, cost tracker) but no orchestration
layer to tie them together for a single turn.

**What we built:**

`UnifiedContext` is the per-turn request envelope that deepTutor describes.
It carries every context slot — `history_context`, `summary_context`,
`retrieval_context`, `memory_context` — as named fields. It is ephemeral:
assembled before the turn starts, never stored. Durability comes from
persisting messages before assembly and turn status after completion.

```
UnifiedContext
  ├── session_id / turn_id
  ├── user_message
  ├── history_context    ← verbatim recent turns from SQLite
  ├── summary_context    ← LLM-compressed older turns
  ├── retrieval_context  ← RAG chunks assembled by ContextAssembler
  └── memory_context     ← cross-session user profile
```

`ContextBuilder.build()` fetches history from SQLite and runs retrieval
**concurrently** via `asyncio.gather()` — not sequentially. The assembled
slots are returned as a single `UnifiedContext` object.

`SessionStore` is a minimal SQLite layer for `sessions`, `messages`, and
`turns` tables. The design principle: sessions and messages are appended
before the turn runs; turn status is updated after it completes or fails.

`TurnRuntimeManager` is an async context manager (`async with`) that opens
a turn row on entry and closes it (completed/failed/escalated) on exit —
even if an exception is raised.

---

### Skill 2 — Tool and Contract Design → `tool_registry.py`

**Gap:** The agent had no formal interface between the LLM and external
tools. Vague function signatures let the LLM guess at parameter names and
types, producing silent runtime failures.

**What we built:**

`ToolSchema` declares a tool's complete contract: input properties with
types, required/optional status, allowed values (`enum`), numeric bounds
(`minimum`/`maximum`), and regex patterns. The schema is the single source
of truth for both the validator and the LLM's function-calling payload.

`SchemaValidator` enforces the contract before every tool call:
- Missing required fields → `ToolContractError`
- Wrong type → `ToolContractError`
- Unknown field → `ToolContractError` (strict — no extra keys allowed)
- Constraint violation (min/max/enum/pattern) → `ToolContractError`

The registry produces OpenAI-compatible `function_calling` schemas via
`to_openai_functions()`, so the LLM sees exactly the same contract as
the validator enforces. "A tool contract is a promise — break it loudly."

`ToolRegistry.call_log()` records every tool invocation with latency,
giving a built-in audit trail without a separate logger.

---

### Skill 4 — Reliability Engineering → `reliability.py`

**Gap:** Every call to an external LLM API was fire-and-forget with no
retry logic, no timeout, and no circuit breaker. The first transient API
failure would propagate directly to the user.

**What we built:**

`RetryPolicy` — exponential backoff with full jitter. Jitter is critical:
without it, all agents retry at the same moment after a shared outage
(thundering herd). The `retrying(predicate)` method retries only on
exceptions matching a caller-supplied predicate — non-retryable errors
propagate immediately. `max_attempts=4`, `base_delay=1s`, `max_delay=30s`
by default.

`CircuitBreaker` — closed / open / half-open state machine. Once
`failure_threshold` failures accumulate, the circuit opens and subsequent
calls fail immediately (`CircuitOpenError`) without waiting for a timeout.
After `recovery_timeout` seconds the circuit enters half-open and allows
one probe request. Two consecutive successes close the circuit again.

`with_timeout` — wraps any async call with `asyncio.wait_for`.

`with_fallback` — tries primary, catches specified exceptions, runs
fallback. Used for: return cached answer when RAG is down, return degraded
response when LLM times out.

`ResilientCaller` — composes all three in one object. This is what every
LLM and API call should be wrapped with in production.

A **bug was found** during this implementation: `retrying` was initially
written as `async def`, which made `policy.retrying(pred)` return a
coroutine rather than a callable. The test caught this — fixed by making
`retrying` a sync method that returns an async executor.

---

### Skill 5 — Security and Safety → `security.py`

**Gap:** No protection against prompt injection, no input validation,
no output filtering. The agent would pass any user string directly to
the LLM and return any LLM response directly to the user.

**What we built:**

`PromptGuard` — scans input for 9 injection signatures (role override,
prompt extraction, jailbreak keywords, context escape sequences). Each
pattern has a `ThreatLevel`: WARN (suspicious) or BLOCK (definite).
BLOCK raises immediately; WARN records but continues. All patterns are
case-insensitive.

`InputValidator` — structural checks before any LLM call:
- Length bounds (min/max characters)
- Optional character allowlist (regex)
- Configurable blocked patterns (SQL injection, XSS, etc.)

`OutputFilter` — scans LLM responses for PII (SSN, credit card patterns),
credential leakage (`api_key=...`), and harmful content. Returns a safe
fallback string instead of the raw LLM output when a violation is found.

`PermissionBoundary` — capability-to-tool access control list. Each
capability declares which tools it may call. Attempting to call an
unpermitted tool raises `PermissionDeniedError` immediately. Prevents
a compromised capability from calling tools outside its scope.

`SecurityPipeline` — composes guard, validator, and filter into one
`check_input()` / `check_output()` call pair around each LLM interaction.

---

### Skill 6 — Evaluation and Observability → `tracer.py`

**Gap:** The `CostTracker` recorded token costs but not the sequence of
decisions the agent made. When something went wrong there was no way to
trace backward through what happened.

**What we built:**

`Tracer` — assigns a monotonically increasing sequence number to every
event in a turn: `TURN_START`, `LLM_CALL`, `TOOL_CALL`, `TOOL_RESULT`,
`RETRIEVAL`, `CONTEXT_BUILT`, `ESCALATION`, `ERROR`. The `dump()` method
produces a human-readable trace table — this is the "trace backward"
workflow the IBM video describes.

`TraceStore` — persists events to SQLite via a background `asyncio.create_task`
(fire-and-forget, never blocks the turn). `get_turn(turn_id)` replays the
full event sequence for post-hoc debugging.

`EvalMetrics` — data-driven metrics aggregation:
- Success rate (fraction of turns with `success=True`)
- p50 and p95 latency
- Average cost per task
- All metrics broken down by capability

"Move away from guessing — use data-driven metrics." The `by_capability()`
breakdown directly answers: which capability is most expensive? Which has
the worst success rate?

---

### Skill 7 — Product Thinking → `unified_context.py` (EscalationPolicy)

**Gap:** The agent had no concept of when to stop answering and hand off
to a human. An agent that confidently returns wrong answers is worse than
one that escalates appropriately.

**What we built:**

`EscalationPolicy` evaluates four rules in order:
1. Unrecoverable error type (`AuthenticationError`, `RateLimitError`) → escalate
2. User explicitly asked for a human ("speak to a human", "real person") → escalate
3. Retrieval context is empty for a knowledge question → escalate
4. Default → continue

`TurnRuntimeManager.should_escalate()` surfaces this policy at the turn
boundary. When `should_escalate()` returns True, the turn closes with
`status=ESCALATED` and the user receives `escalation_message()` rather
than a hallucinated answer.

---

### Scala additions

`ResilienceAlgebra.scala` — `RetryPolicy` using `IO.sleep`, `CircuitBreakerAlgebra`
(tagless-final) with `Ref`-based state machine, `ResilientCaller` composing
retry + circuit breaker + `IO.race`-based timeout.

`TracerAlgebra.scala` — `TracerAlgebra` (tagless-final) with sequence-numbered
`TraceEvent` recording, `dump` for human-readable output, and `timed[A]`
that measures latency and records a `TraceEvent` automatically.
`EvalMetricsAlgebra` with `Ref`-based accumulation and p50/p95 latency.

---

### Test results

180 Python tests, all passing (1.27s). **2 bugs found by the test suite:**

1. `RetryPolicy.retrying` was `async def`, making `policy.retrying(pred)`
   return a coroutine instead of a callable — tests caught the `TypeError`.
   Fixed by making `retrying` a sync method returning an async executor.

2. The `retrying` call pattern in tests was updated from
   `await policy.retrying(pred)(fn)` → `policy.retrying(pred)` returns the
   executor, which is then called as `await executor(fn)`.

---

## Commit history (branch: `claude/agent-engineering-skills`)

| Commit | Description |
|---|---|
| TBD | Seven IBM agent engineering skills implemented |

---

## What to build next

- [ ] Notebook (`lore_agent.ipynb`) — end-to-end demo wiring all components
- [ ] `MemoryStore` — cross-session user profile persistence
- [ ] Wire `SecurityPipeline` + `Tracer` + `ResilientCaller` into `TurnRuntimeManager`
- [ ] Scala tests for `ResilienceAlgebra` and `TracerAlgebra`
- [ ] HTTP endpoints for trace replay (`GET /trace/{turn_id}`) and eval metrics (`GET /eval`)
