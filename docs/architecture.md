# Architecture

## Observability — Events and OpenTelemetry

### The core idea: what is observability?

Observability answers the question: **"what is my system actually doing?"**

Without it you only know two things — the user sent a query and an answer came back.
With it you know every step in between, how long each one took, and exactly where
something went wrong when it does.

### How a single agent turn is recorded

```
  USER
   │
   │  "Who created the Horcruxes?"
   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        Agent Turn                                    │
│                                                                      │
│  step 1 ── Retrieve lore chunks from ChromaDB                       │
│             tracer.event(RETRIEVAL, {query: "...", chunks: 5})       │
│                                                                      │
│  step 2 ── Assemble context window                                   │
│             tracer.event(CONTEXT_BUILT, {tokens: 1420})              │
│                                                                      │
│  step 3 ── Call LLM                                                  │
│             tracer.event(LLM_CALL, {model: "gpt-4o", tokens: 512})  │
│                                                                      │
│  step 4 ── Return answer to user    ◄── user gets answer HERE        │
│             tracer.event(TURN_END,  {latency_ms: 1820})             │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
   │
   │  answer: "Voldemort created seven Horcruxes..."
   ▼
  USER  ◄── receives answer in ~1.8 s, never waits for logging
```

The critical point: **the user gets their answer before any logging happens.**
Observability runs on a background task — it never adds to the user's wait time.

### What happens inside tracer.event()

Every call to `tracer.event()` fans out to two independent destinations at once:

```
                    tracer.event(LLM_CALL, {...})
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
   ┌─────────────────────┐      ┌──────────────────────┐
   │  asyncio.create_task│      │   otel_bridge        │
   │  (fire-and-forget)  │      │   .emit_span()       │
   └─────────────────────┘      └──────────────────────┘
              │                             │
              │ runs AFTER turn returns     │ runs NOW, tiny (μs)
              ▼                             ▼
   ┌─────────────────────┐      ┌──────────────────────┐
   │     SQLite          │      │  BatchSpanProcessor  │
   │   traces.db         │      │  (in-memory buffer)  │
   └─────────────────────┘      └──────────────────────┘
              │                             │
              │                             │ flushes every ~5 s
              │                             ▼
              │                  ┌──────────────────────┐
              │                  │   Jaeger collector   │
              │                  │  (OTLP gRPC :4317)   │
              │                  └──────────────────────┘
              │                             │
              ▼                             ▼
   ┌─────────────────────┐      ┌──────────────────────┐
   │  CLI / HTTP replay  │      │    Jaeger Web UI     │
   │                     │      │   localhost:16686    │
   │  python -m          │      │                      │
   │  context_harness    │      │  flame chart showing │
   │  .trace_view <id>   │      │  every step as a     │
   │                     │      │  coloured bar with   │
   │  GET /trace/<id>    │      │  its exact duration  │
   └─────────────────────┘      └──────────────────────┘
```

### What each destination gives you

**SQLite → CLI / HTTP** — *replay*

You have the turn ID (logged when the user query arrived). Something went wrong.
You run:

```bash
python -m context_harness.trace_view abc-123
```

and see:
```
╭─ Trace  turn_id=abc-123 ──────────────────────────────────────────────╮
│  #   Kind            Latency   Payload                                 │
│  1   turn_start                {}                                      │
│  2   retrieval        42.1ms   {"query": "Horcruxes", "chunks": 5}     │
│  3   context_built     3.2ms   {"tokens": 1420, "strategy": "sandwich"}│
│  4   llm_call        1710.0ms  {"model": "gpt-4o", "tokens_in": 512}  │
│  5   turn_end                  {"answer_len": 340}                     │
╰────────────────────────────────────────────────────────────────────────╯
  5 events   wall 1820ms   sum latency 1755ms
```

You can see immediately that 94% of the turn time was the LLM call — not retrieval,
not context assembly. No guessing.

**Jaeger UI** — *visual flame chart*

Open `http://localhost:16686`, select service `deeptutor`, click Find Traces.
Each turn appears as a horizontal bar. Click it to expand into a flame chart:

```
  turn abc-123  ████████████████████████████████████  1820 ms
    retrieval   ██  42 ms
    context     ░ 3 ms
    llm_call    ████████████████████████████████  1710 ms   ← the slow part
    turn_end    ░ 1 ms
```

This is what observability looks like to a client or non-expert:
> "I can click on any request and see a picture of where the time went."

### What OpenTelemetry actually is

OpenTelemetry (OTel) is a **standard wire format and SDK** for emitting spans.

A **span** = one unit of work with a name, a start time, an end time, and key/value
attributes. Our `tracer.event(LLM_CALL, {...})` becomes one span named `"llm_call"`
with attributes like `turn.id`, `latency_ms`, `payload.model`.

OTel is the standard — Jaeger is just one of many backends that can receive it.
The same spans could go to Datadog, Grafana Tempo, AWS X-Ray, or Honeycomb
by changing one environment variable (`OTEL_EXPORTER_OTLP_ENDPOINT`). That
portability is the reason to use OTel rather than a vendor-specific SDK.

```
  Our code                OTel SDK              Any backend
  ────────                ────────              ───────────
  otel_bridge             BatchSpan             Jaeger
  .emit_span()  ───────►  Processor   ───────►  Datadog
                          (buffers,             Grafana Tempo
                           retries,             AWS X-Ray
                           exports)             Honeycomb
```

### Starting the full observability stack

```bash
# 1. Start Jaeger (accepts OTel spans, serves the UI)
docker compose -f docker-compose.jaeger.yml up -d

# 2. Configure OTel in your Python session
from context_harness.otel_bridge import configure
configure(service_name="deeptutor", endpoint="http://localhost:4317")

# 3. Run agent queries — spans flow to Jaeger automatically
# 4. Open http://localhost:16686 — select service "deeptutor" → Find Traces
```

---

## Retrieval Ranking Pipeline

The pipeline has two stages: ANN retrieval from ChromaDB, then optional MMR reranking.

```
                        ┌─────────────────────────────────────────────────────┐
                        │                   RAGPipeline.retrieve()            │
                        └─────────────────────────────────────────────────────┘

  User query string
        │
        ▼
┌───────────────────┐
│  Embed query      │  DefaultEmbeddingFunction (MiniLM-L6-v2, 384-dim)
│  query_texts=[q]  │  runs ONNX locally, no network call after first download
└───────────────────┘
        │
        │  query vector  [0.12, -0.34, 0.89, ...]
        ▼
┌───────────────────────────────────────────────────────────────────┐
│  ChromaDB HNSW  (cosine space)                                    │
│                                                                   │
│  collection "hp_lore"                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ...    │
│  │ chunk_001│  │ chunk_002│  │ chunk_003│  │ chunk_004│          │
│  │ [emb]    │  │ [emb]    │  │ [emb]    │  │ [emb]    │          │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘          │
│                                                                   │
│  ANN search: find n_results chunks with smallest cosine distance  │
└───────────────────────────────────────────────────────────────────┘
        │
        │  top-k candidates  (text, metadata, cosine_distance)
        │  score = 1.0 − cosine_distance
        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Stage 1 result — sorted by similarity score (pure relevance)                │
│                                                                              │
│  rank 1  score=0.91  doc=horcruxes       "A Horcrux is an object in which…" │
│  rank 2  score=0.87  doc=lord-voldemort  "Voldemort split his soul into…"   │
│  rank 3  score=0.81  doc=harry-potter    "Harry destroyed the final…"       │
│  rank 4  score=0.79  doc=horcruxes       "The diary was the first Horcrux…" │  ← near-duplicate
│  rank 5  score=0.76  doc=deathly-hallows "The Elder Wand was one of…"       │
└──────────────────────────────────────────────────────────────────────────────┘
        │
        │  (optional — call mmr_rerank() explicitly)
        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Stage 2 — MMR reranking  (Maximal Marginal Relevance)                       │
│                                                                              │
│  MMR score = λ · sim(chunk, query)  −  (1−λ) · max sim(chunk, selected)     │
│  default λ = 0.5  →  equal weight on relevance and diversity                 │
│                                                                              │
│  iteration 1: pick rank 1  (nothing selected yet, redundancy = 0)           │
│  iteration 2: rank 4 penalised — too similar to rank 1 (same doc)           │
│               rank 5 promoted — adds new information                        │
│  iteration 3: …                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
        │
        │  reranked top-k  (relevant AND diverse)
        ▼
┌───────────────────────────────────────────────────────────────────┐
│  Stage 2 result — reranked                                        │
│                                                                   │
│  rank 1  score=0.91  doc=horcruxes       "A Horcrux is an…"      │
│  rank 2  score=0.87  doc=lord-voldemort  "Voldemort split…"      │
│  rank 3  score=0.76  doc=deathly-hallows "The Elder Wand…"       │  ← promoted
│  rank 4  score=0.81  doc=harry-potter    "Harry destroyed…"      │
│  rank 5  score=0.79  doc=horcruxes       "The diary was…"        │  ← demoted
└───────────────────────────────────────────────────────────────────┘
        │
        ▼
  List[Chunk]  →  ContextAssembler  →  prompt injection
```

### Key parameters

| Parameter | Default | Effect |
|---|---|---|
| `top_k` | 5 (retrieve), 10 (DeepResearch mode), 3 (GuidedLearning mode) | How many chunks reach the LLM |
| `hnsw:space` | `cosine` | Distance metric. Cosine is query-length-invariant; use `l2` for unit-normalised embeddings |
| `mmr λ` | `0.5` | `1.0` = pure relevance (no diversity penalty); `0.0` = pure diversity (ignores relevance) |
| Chunking strategy | `RECURSIVE` | paragraph → sentence → fixed fallback. Affects what a single chunk contains |

### Where the code lives

| Step | File | Line |
|---|---|---|
| Embed + ANN query | `context_harness/rag_pipeline.py` | `RAGPipeline.retrieve()` L188 |
| Score conversion | `context_harness/rag_pipeline.py` | L207 `score = 1.0 − dist` |
| MMR reranking | `context_harness/rag_pipeline.py` | `RAGPipeline.mmr_rerank()` L234 |
| Chunking strategies | `context_harness/rag_pipeline.py` | L53–92 |
| Collection initialisation | `context_harness/rag_pipeline.py` | `_init_chromadb()` L131 |
| DSPy k=10 / k=3 per mode | `context_harness/dspy_agent.py` | `DeepResearchModule` / `GuidedLearningModule` |

### Multi-universe note

ANN search is purely geometric — it returns the closest vectors regardless of which
universe they originated from. Adding a second universe to the same collection gives
cross-universe semantic search for free. To restrict results to one universe, pass a
metadata filter:

```python
collection.query(query_texts=[q], n_results=k, where={"universe": "hp"})
```

---

## External Reranker (Cohere rerank-v3)

### Motivation

Cosine similarity between embeddings captures semantic nearness, but it's
not trained for *answer relevance*. A dedicated reranker model — trained
on (query, document, relevance-score) triples — typically lifts top-k
hit rate by 10–20% over raw embedding similarity. The canonical example:
Anthropic's Contextual Retrieval paper (Sep 2024) reported ~49%
reduction in retrieval failures with a reranker in the pipeline.

MMR (earlier in this doc) solves a *different* problem — redundancy
across top-k. MMR and Cohere reranker are complementary, not
alternatives. Both are "stage 2" operations over a stage-1 vector-search
candidate set.

### Pipeline shape

```
 User query
     │
     ▼
 Vector search (fetch_k=20)  ←────── over-fetch, 4× the final top_k
     │
     │  20 candidates sorted by cosine
     ▼
 Cohere rerank-v3  ←──────────────── optional, env-gated on COHERE_API_KEY
     │
     │  5 candidates sorted by learned relevance
     ▼
 (optional: MMR reranking for diversity)
     │
     ▼
 Top-k to LLM
```

### Design choices

**1. Protocol + adapter pattern, not subclass.**
`RerankerClient` is a `typing.Protocol` — any object with a `rerank(query,
candidates, top_k)` method is a valid reranker. Three in-tree adapters:
- `CohereReranker` — production
- `IdentityReranker` — no-op; the A/B control arm
- `FakeReranker` — test double that records every call

Matches the same Protocol+adapter style already used for `Summarizer`,
`LLMJudgeClient`, etc. Swap at composition time, no inheritance
entanglement.

**2. Over-fetch at the RAGPipeline layer, not inside the reranker.**
`RAGPipeline.__init__` accepts `rerank_fetch_k=20` (default). `retrieve()`
conditionally over-fetches when a reranker is attached:

```python
fetch_k = self._rerank_fetch_k if self._reranker is not None else k
```

Keeps the reranker abstraction simple — it just gets a list of candidates
and returns a subset. The decision of *how many* candidates to hand it
lives at the pipeline level where there's context on collection size +
cost budget.

**3. Env-gated construction via `build_reranker_from_env()`.**
`ingest_lore.build_pipeline()` calls it; returns a `CohereReranker` if
`COHERE_API_KEY` is set, otherwise `None`. No reranker = pipeline
behaves identically to the pre-reranker version. CI and offline dev
stay unaffected with zero config.

**4. Graceful fallback on reranker failure.**
`RAGPipeline._maybe_rerank()` wraps the reranker call in try/except. If
the reranker raises (rate limit, network outage, bad auth, API-shape
change), retrieval logs + falls back to the un-reranked top-k. Rationale:
reranker is an *optimisation*, not a *correctness* requirement. Losing
the reranker should degrade quality, not availability.

**5. Reranker score replaces vector similarity score.**
`ScoredChunk.score` after reranking is the reranker's `relevance_score`
(0.0–1.0), not the original cosine-distance-derived score. Downstream
consumers (MMR, citation display, traces) see the stronger signal. The
original ranking is preserved in the list order; the score is
overwritten.

### Configuration

```bash
# Enable — set these in .env (local) or Railway dashboard (prod)
COHERE_API_KEY=co-key-...

# Default fetch_k is 20; override via RAGPipeline(rerank_fetch_k=N)
```

### Where the code lives

| Step | File | Symbol |
|---|---|---|
| `RerankerClient` Protocol | `context_harness/reranker.py` | `RerankerClient` |
| Production adapter | `context_harness/reranker.py` | `CohereReranker` |
| No-op baseline | `context_harness/reranker.py` | `IdentityReranker` |
| Test double | `context_harness/reranker.py` | `FakeReranker` |
| Env-gated factory | `context_harness/reranker.py` | `build_reranker_from_env()` |
| Over-fetch + fallback logic | `context_harness/rag_pipeline.py` | `RAGPipeline._maybe_rerank()` |
| Wiring into `build_pipeline` | `context_harness/ingest_lore.py` | `build_pipeline()` |

### Evaluation — measuring the lift

To A/B compare with vs without reranker:

```bash
# Control: no reranker
COHERE_API_KEY= .venv/bin/python -m evals.eval_dspy --runs 3 > baseline.json

# Treatment: Cohere rerank-v3
.venv/bin/python -m evals.eval_dspy --runs 3 > reranked.json

# Compare retrieval hit rate, answer correctness, latency, cost
.venv/bin/python -m evals.compare_dspy baseline.json reranked.json
```

Expected deltas on the 20-question eval set:
- **Retrieval hit rate:** +10–20 percentage points
- **Answer correctness:** +5–10 percentage points (smaller lift because
  retrieval was already decent on this small corpus)
- **Latency:** +100–200 ms per call
- **Cost:** ~$0.001 per call (free tier covers first 1k/month)

### Trade-offs

- **Pro:** biggest single-move quality win for RAG; drop-in.
- **Pro:** composable — can run with MMR after, or instead of.
- **Pro:** env-gated means free tier dev & full-power prod from one binary.
- **Con:** adds a third-party dependency on the critical path (mitigated
  by graceful fallback).
- **Con:** `ScoredChunk.score` semantics change when reranker runs — the
  number means something different. Cite in UI/logs carefully.
- **Con:** DSPy-compiled few-shot demos selected with reranker-off may not
  generalise cleanly to reranker-on (different candidate distribution).
  Consider recompiling after enabling.

or use one collection per universe and route the query before retrieval.
