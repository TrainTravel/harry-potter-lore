# Lore Whisperer

A RAG-powered agent that answers Harry Potter lore questions with cited sources, built as a pedagogical project to learn agent engineering, context engineering, and prompt optimization end-to-end.

Two implementations: a **Python** stack (ChromaDB + DSPy + Gemini + FastAPI) that runs the full agent loop from question to evaluated answer, and a **Scala Typelevel** stack (Cats Effect + fs2 + Http4s) that implements the same core infrastructure with typed, streaming semantics.

## Try it

**Live demo:** [lore-whisperer-ui.lovable.app](https://lore-whisperer-ui.lovable.app)

Seven agent modes share one architecture. The most distinctive: **perspective_shift** — pick an HP character, ask them about a real-life situation, get a response in their voice grounded in their canon. Example exchange:

> **You:** "I haven't spoken to my brother in three years over a fight neither of us can remember the original cause of, but we both still tell ourselves we were right."
>
> **Dumbledore:** *"I know something of these silences between brothers. My own brother and I endured a silence that lasted much of our lives, born of a terrible argument where being right felt more important than being kind... The question you must ask yourself is not 'who was right?' but 'is this silence worth the price of my brother?'"*

The agent reaches for *Aberforth* without being prompted — the brother Dumbledore was estranged from for most of his life. That's character-specific RAG retrieval at work, not a generic "wizard-themed advice" chatbot.

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
#    Compile-time uses flash-lite (cheap teacher); runtime can be Pro.
python -m context_harness.compile_agent \
    --model gemini/gemini-2.5-flash-lite \
    --out my_profile.agent

# 6. Start the API (set DSPY_MODEL to pick the runtime model)
pip install fastapi uvicorn
DSPY_MODEL=gemini/gemini-2.5-pro uvicorn api.main:app --reload --port 8000

# 7. Try it
curl -X POST http://localhost:8000/ask \
    -H "Content-Type: application/json" \
    -d '{"question":"Who killed Dumbledore?","mode":"deep_research"}'
```

Interactive Swagger UI at http://localhost:8000/docs.

### Run tests

```bash
# Python (364 tests)
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

### Seven agent modes (+ auto-routing)

| Mode | What it does | Example prompt |
|---|---|---|
| `deep_research` | Factual lore Q&A, k=10 retrieval, citations enforced | "Who killed Dumbledore?" |
| `guided_learning` | Socratic tutoring, no spoilers, ends in a question | "Teach me Polyjuice Potion step by step" |
| `exam_grader` | Grades a student's answer vs canon (0-100, passing flag, critique) | submit an answer, get score + critique |
| `open_analysis` | Blends canon facts with broader knowledge (psychology, lit theory) | "Why does the series keep returning to mirrors?" |
| `perspective_shift` | Character advice on a real-life problem (in voice + canon) | "What would Dumbledore say about my career change?" |
| `debate` | Arguments for, arguments against, verdict | "Was Snape a hero or a villain?" |
| `satirical_podcast` | Comedy podcast script with canon citations | "Quidditch as an extreme sport" |
| `auto` | Intent router classifies the user message and dispatches | (no need to specify) |

Each mode is ~30 lines: a DSPy `Signature` (typed prompt template) + `Module` (retrieval + LLM call) + `Metric` (compile-time pass/fail check). The same chassis powers all seven.

Multi-turn modes (`open_analysis`, `guided_learning`, `perspective_shift`) maintain rolling chat history with async background compaction. The intent router has stickiness — once a conversation enters a mode, follow-up turns stay in that mode unless the router emits a high-confidence override.

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
