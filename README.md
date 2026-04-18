# Harry Potter Lore Agent

A RAG-powered agent that answers Harry Potter lore questions with cited sources, built as a pedagogical project to learn agent engineering, context engineering, and prompt optimization end-to-end.

Two implementations: a **Python** stack (ChromaDB + DSPy + Gemini + FastAPI) that runs the full agent loop from question to evaluated answer, and a **Scala Typelevel** stack (Cats Effect + fs2 + Http4s) that implements the same core infrastructure with typed, streaming semantics.

## Quick start

```bash
# 1. Create and activate virtualenv
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
pip install pytest pytest-asyncio

# 3. Set your Gemini API key
echo 'export GOOGLE_API_KEY=your-key-here' > .env
source .env

# 4. Seed the corpus into ChromaDB
python -m context_harness.ingest_lore

# 5. Compile the DSPy agent (bootstraps few-shot demos from trainsets)
python -m context_harness.compile_agent \
    --model gemini/gemini-2.5-flash-lite \
    --out my_profile.agent

# 6. Start the API
pip install fastapi uvicorn
uvicorn api.main:app --reload --port 8000

# 7. Try it
curl -X POST http://localhost:8000/ask \
    -H "Content-Type: application/json" \
    -d '{"question":"Who killed Dumbledore?","mode":"deep_research"}'
```

Interactive Swagger UI at http://localhost:8000/docs.

### Run tests

```bash
# Python (193 tests)
python -m pytest tests/ -v

# Scala (9 tests)
cd scala-harness && sbt "testOnly *Suite" --batch
```

### Run evaluations

```bash
# RAG retrieval eval (no LLM needed)
python -m evals.eval_rag

# Full agent eval with LLM-as-judge (needs GOOGLE_API_KEY)
python -m evals.eval_agent --limit 5    # quick smoke
python -m evals.eval_agent              # full run
```

## Architecture

```
                           ┌─────────────────────────────┐
                           │         User question        │
                           └──────────────┬──────────────┘
                                          │
                                          ▼
                         ┌────────────────────────────────┐
                         │   api/main.py (FastAPI)         │
                         │   POST /ask  GET /trace/{id}    │
                         └────────────────┬───────────────┘
                                          │
                        ┌─────────────────┼─────────────────┐
                        ▼                 ▼                  ▼
               ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
               │  DSPy Agent  │  │  Cost Tracker │  │    Tracer    │
               │  (compiled)  │  │   (per-call)  │  │  (per-turn)  │
               └──────┬───────┘  └──────────────┘  └──────────────┘
                      │
           ┌──────────┼──────────┬──────────┐
           ▼                     ▼          ▼
  ┌─────────────────┐   ┌──────────────┐  ┌──────────────┐
  │  Deep Research   │   │   Guided     │  │    Exam      │
  │  (k=10, CoT)    │   │  Learning    │  │   Grader     │
  │                  │   │ (k=3, Socr.) │  │  (k=5, strict│
  └────────┬────────┘   └──────┬───────┘  └──────┬───────┘
           │                   │                  │
           └──────────┬────────┴──────────────────┘
                      ▼
           ┌─────────────────────┐
           │   RAG Pipeline      │
           │   (retrieve + MMR)  │
           └─────────┬───────────┘
                     ▼
           ┌─────────────────────┐       ┌─────────────────────┐
           │     ChromaDB        │◄──────│  Document Registry   │
           │  (ONNX embeddings)  │       │  (stale-doc guard)   │
           └─────────────────────┘       └─────────────────────┘
                     ▲
                     │
           ┌─────────────────────┐
           │  data/hp_lore.txt   │
           │  (10 hand-written   │
           │   lore documents)   │
           └─────────────────────┘
```

### Three agent modes

| Mode | Retrieval | Output | Metric |
|---|---|---|---|
| **Deep Research** | k=10 chunks, broad retrieval | answer + citations + confidence | citation overlap (>= 50% of expected docs) |
| **Guided Learning** | k=3 chunks, narrow | hint + next_question + explanation | Socratic score (no spoilers, ends with ?) |
| **Exam Grader** | k=5 chunks, strict | score (0-100) + is_passing + critique | score within ±15, passing agrees, critique >20 chars |

### Planned: Creative Storytelling

A fourth mode (`creative_storytelling`) is planned but not yet built. It would weave retrieved facts into a narrative (persona-driven: time-traveler, fantasy narrator) while the DSPy optimizer ensures 100% factual accuracy against the corpus. Harder to evaluate than the other modes — needs an LLM judge that can distinguish "creative and accurate" from "creative and hallucinated."

### Infrastructure layer

| Module | Purpose |
|---|---|
| `context_manager.py` | Token-budget window with priority eviction |
| `context_assembler.py` | 4 assembly strategies (naive, relevance, sandwich, citation) |
| `document_registry.py` | Stale-doc tracking — detects changed docs, deletes old chunks |
| `index_version_guard.py` | Validates embedding model + dimension on startup (fail-fast) |
| `retrieval_cache.py` | LRU + TTL cache with per-collection invalidation |
| `cost_tracker.py` | Per-call cost aggregation with pricing table |
| `summarizer.py` | Async background summarization (non-blocking) |
| `reliability.py` | Retry, circuit breaker, bulkhead patterns |
| `security.py` | Input/output sanitization, tool-use gating |
| `tracer.py` | Per-turn event tracing (SQLite + OTel bridge) |

### Scala harness (`scala-harness/`)

The same core infrastructure re-implemented with Typelevel libraries:
tagless-final algebras, fs2 streaming pipeline, Cats Effect IO, Http4s
server on :8080. See [docs/python-vs-scala.md](docs/python-vs-scala.md)
for a detailed comparison.

```bash
cd scala-harness
sbt run          # starts Http4s server on :8080
sbt test         # runs munit-cats-effect suites
```

## Documentation

| Doc | What it covers |
|---|---|
| [docs/FROM_SCRATCH.md](docs/FROM_SCRATCH.md) | Mentor walkthrough: how to rebuild this project phase-by-phase, with decision rationale and traps to avoid |
| [docs/python-vs-scala.md](docs/python-vs-scala.md) | Surface-area comparison of the two implementations, what each is uniquely valuable for |
| [docs/architecture.md](docs/architecture.md) | Retrieval ranking pipeline diagram |
| [docs/adr/](docs/adr/) | Architecture Decision Records for cross-cutting concerns |
| [docs/notes-asyncio-vs-cats-effect.md](docs/notes-asyncio-vs-cats-effect.md) | Concurrency model comparison |
| [docs/notes-trampoline-and-io.md](docs/notes-trampoline-and-io.md) | Stack safety and trampolining in both paradigms |
| [BUILDING_LOG.md](BUILDING_LOG.md) | Session-by-session development journal |
