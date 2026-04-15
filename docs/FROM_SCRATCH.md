# From Scratch: A Mentor's Walkthrough

> Written for someone with a strong software background who is learning
> how to build agent systems from first principles. You've already built
> a working version of this project; this document is what I wish someone
> had written for *future-you* before you started.

The thesis of this guide is one sentence:

> **You cannot improve what you cannot measure, and you cannot measure an
> agent without a corpus, a question set, and a judge — build those first,
> everything else is decoration.**

Every phase below exists to reinforce that thesis.

---

## Phase 0 — Decide what you are building *and what it means to be done*

Before touching code, answer these on one page of notes:

1. **Who is the user?** (In our case: someone asking lore questions.)
2. **What's the input?** (A free-text question, optionally bound to a session.)
3. **What's the output?** (A factual answer grounded in the corpus, with citations.)
4. **What does "correct" mean?** (Answer contains the key fact AND cites
   the right doc. Not "answer looks smart.")
5. **What is the failure mode you *most* want to prevent?** (For a tutor:
   confabulating a fact the corpus doesn't support. For search: missing
   a relevant document.)

If you can't answer #4 concretely in one sentence, stop. Figure that
out first. Every technical decision downstream rests on that sentence.

**Trap to avoid:** skipping this because it feels obvious. "Answer HP
questions well" is not a definition, it's a vibe. Specificity comes from
*writing down bad answers* and asking "why is this one wrong?"

**Output of this phase:** a single markdown file, `docs/what-this-is.md`,
with those five answers. You should be able to show it to a stranger
and they could judge your agent's output.

---

## Phase 1 — Ground truth, not infrastructure

This is the phase I most wish I had nailed first, and it's the one
most people (including me, in this project) skip.

### 1a. Build a small, hand-written corpus

For this project: `data/hp_lore.txt` — 10 documents, one paragraph each,
covering canonical entities. **Hand-written, not scraped.** Why?

- Scraped wikis have noise, HTML, references, irrelevant trivia. You
  will waste days fighting the cleaning pipeline before you've proven
  the agent works at all.
- Hand-written text is the equivalent of "test fixtures" — you know
  every fact in it, you can mentally predict what the retrieval should
  return, and you can write questions whose answers you already know.
- Starting small (10 docs, ~200 words each) lets the *whole* loop —
  ingest → embed → retrieve → answer → judge — fit in 10 seconds on
  your laptop. Fast loops are the single biggest accelerator for
  learning.

You expand the corpus *after* the full loop works end-to-end, not before.

### 1b. Write the eval question set by hand

`evals/questions.py` in our project. Structure each question as:

```python
EvalQuestion(
    id="q01",
    kind="factual" | "multi_hop" | "out_of_corpus",
    question="Who killed Dumbledore?",
    reference_answer="Severus Snape, as part of Dumbledore's own plan.",
    expected_docs=["albus-dumbledore", "severus-snape"],
)
```

Three kinds matter:

| Kind | What it catches |
|---|---|
| `factual` | Retrieval actually working (the single most common bug) |
| `multi_hop` | The agent stitches evidence from >1 document |
| `out_of_corpus` | The agent says "I don't know" instead of confabulating |

Aim for 15–25 hand-written questions. This is enough to discriminate
"works" from "broken" without being so large you dread running it.

### 1c. Write the judge *before* the agent

An LLM-as-judge (`evals/eval_agent.py::judge`) takes (question, agent
answer, reference answer, kind) and returns `{"correct": bool, "reason":
str}`. Writing this first forces you to encode what "correct" means in
code, not in vibes.

Judges are themselves imperfect — the judge is another LLM, it can lie.
Three rules that save you:

- Give the judge the reference answer (don't ask it to judge
  open-ended).
- Ask for a short reason. Skim the reasons when you inspect results;
  the judge often reveals its own failure modes ("I scored this correct
  because the word 'Snape' appeared anywhere in the answer" → bad
  judge).
- Cost-track the judge separately. Judge cost > agent cost is a smell
  that your judge is overwrought.

**Checkpoint before moving on:** you should be able to run
`evals/eval_agent.py --limit 5` against a **stub agent** that returns
`"I don't know"` for everything, and see that all `factual` questions
fail while `out_of_corpus` pass. If that doesn't work, your eval loop
is broken — fix that before building the real agent.

---

## Phase 2 — The dumbest agent that could possibly work

Now build the minimum RAG agent. Resist all abstraction.

```
ingest corpus → ChromaDB
on query:
  retrieve top-5 chunks by cosine similarity
  stuff them into a prompt
  ask the LLM
  return the answer
```

That's it. No chunking strategies, no reranking, no context manager,
no summarizer, no streaming, no caching. One file, maybe 80 lines.
In this repo the closest equivalent is `evals/agent.py`.

### Why you build the dumb version first

1. **It establishes a baseline.** When you later add MMR reranking and
   accuracy jumps 3 points, you know *MMR gave you 3 points*. Without
   a baseline, you have no idea if any of your elaborate infrastructure
   helps at all. (In deepTutor's original codebase: 2,250 lines of
   context harness + 860 lines of tests, none of which measured whether
   the library answered questions better than a 50-line stub. That's
   the failure mode to avoid.)
2. **It reveals the real bottleneck.** You will usually discover the
   bottleneck isn't what you thought. Our session: the first real eval
   didn't reveal a retrieval bug, it revealed two *schema* bugs in our
   `Chunk` class (dual-purpose `score` field, attribute/metadata
   duplication). You cannot predict where the bottleneck is.
3. **Most "improvements" don't help.** Chunking strategies, reranking,
   prompt templates — each of these might move the metric by 0–5
   points, or might break it. You will not know which without a
   baseline.

### What about Google ADK / LangChain / LlamaIndex?

Skip them in Phase 2. They are fine frameworks but each adds a
conceptual tax (agents, tools, chains, callbacks) on top of the core
LLM call. You need to understand the underlying loop — prompt → LLM →
parse → tool? → loop — before delegating it to a framework. Once
you've written the loop yourself, adopting ADK or DSPy is a cost/benefit
decision, not a mystery.

**Checkpoint:** your dumb agent achieves *some* accuracy on the eval
set. If it scores 0%, the problem is plumbing (retrieval returning
nothing, wrong API key, wrong model). If it scores 95%, your questions
are too easy. The sweet spot for starting the interesting work is
40–70%.

---

## Phase 3 — Context engineering, justified by the eval

Now you can add the context harness. Every addition must point at a
specific eval failure mode it addresses. No speculative abstractions.

### The five core primitives (in order of utility)

| Module | What it does | Why it matters |
|---|---|---|
| `rag_pipeline.py` | retrieve + chunk + rerank | controls what evidence the LLM sees |
| `context_manager.py` | token-budget window w/ priority eviction | prevents overflow at long sessions |
| `context_assembler.py` | turns chunks into prompt text | placement affects LLM behaviour |
| `prompt_templates.py` | slot-based prompts per task | separates *what to say* from *what to ask* |
| `summarizer.py` | async compression of old turns | long sessions without losing early context |

Build them in this order. Each one should produce a measurable metric
movement before you add the next.

### Sandwich vs. relevance vs. naive assembly

An underappreciated lesson: the *order* chunks are placed in the prompt
matters more than their text content. LLMs attend disproportionately to
the start and end of the context window (the "U-shaped attention" or
"lost in the middle" phenomenon). Hence:

- **Naive:** `[chunk1, chunk2, chunk3]` — simple, usually fine for short context.
- **Relevance-ranked:** highest-scored chunk first.
- **Sandwich:** highest-scored at start AND end, weaker in middle.
- **Citation-anchored:** each chunk prefixed with `[doc_id]` so the
  LLM can reproduce citations verbatim.

In our project all four live in `context_assembler.py`. The citation
variant turned out to matter most because our eval *requires* citations
— metric-driven design.

### Chunking — the unsexy detail that dominates outcomes

Four strategies in `rag_pipeline.py`:

- **Fixed-size** (N tokens, no overlap): simple, can cut mid-sentence.
- **Sliding window** (N tokens, overlap M): preserves context across boundaries.
- **Sentence**: natural boundaries, variable size.
- **Semantic** (embed sentences, group adjacent with high similarity):
  smartest but expensive to compute.

You will spend more time tuning chunking than almost anything else.
Start with fixed-size at 256 tokens with 32-token overlap. Only add
complexity when a specific eval question fails because of a chunking
boundary.

### Cross-paradigm note: Python asyncio vs Scala fs2

The same pipeline expressed two ways — worth reading both to see the
contrast:

- **Python** (`rag_pipeline.py`): imperative; you write `for chunk in
  chunks: ...` and `await` where needed. Simple to read, easy to get
  subtly wrong around shared state and cancellation.
- **Scala** (`scala-harness/src/main/scala/deeptutor/ContextPipeline.scala`):
  `Pipe[IO, A, B]` stages composed via fs2. Backpressure is free.
  Cancellation is free. State is explicit via `Ref`. Harder to start,
  but once you internalise the types you stop making a whole category
  of race-condition bugs.

Neither is "better" — they are different cost/benefit frontiers.

---

## Phase 4 — Infrastructure, each motivated by a real bug

Only add these after the base pipeline is working and measured.

### 4a. Document registry (`document_registry.py`)

**Real bug this fixes:** you re-ingest a document. Its content changed.
Your index now has both the old chunks (stale) and the new ones
(correct). Retrieval sometimes returns the wrong version.

The fix is: per document, track `(doc_id, content_hash, chunk_ids)`
in SQLite. On re-ingest, if the hash changed, delete the old chunks
from ChromaDB before inserting new. ChromaDB supports `delete(ids=…)`;
FAISS doesn't, which is why FAISS is a poor primary store.

Don't build this before you've actually re-ingested a changed document
and seen a stale result. The code isn't hard; knowing *when* you need
it is the lesson.

### 4b. Index version guard (`index_version_guard.py`)

**Real bug this fixes:** you change embedding models from MiniLM to
BGE-base. The old index is 384-dim; the new embeddings are 768-dim.
Queries silently return garbage because vectors of different dimensions
don't error — they just produce nonsense similarities.

The fix is a manifest: `{model, dim, schema_version, collection}` next
to the index. Validate at startup, raise a typed error on mismatch.
This is the canonical "fail fast at boundaries" pattern.

### 4c. Cost tracker (`cost_tracker.py`)

**Real bug this fixes:** you don't know what each agent call costs,
so you can't tell which capability is the money pit. Token counts
aren't the answer — Opus input tokens cost ~50× Haiku input tokens.

The fix: per LLM call, record `(model, capability, tokens_in,
tokens_out, cost_usd)` via a decorator or wrapper. Fire-and-forget
queue to SQLite. In-memory rollups answer "cost by capability" and
"cost by model". Expose a `/cost` HTTP endpoint.

For a tutoring app, "cost per completed session" is often the single
most important operational metric. A great answer at \$5/session isn't
a product — it's a demo.

### 4d. Retrieval cache (`retrieval_cache.py`)

**Real bug this fixes:** the same question gets asked twice in a
session, you re-embed and re-retrieve, paying latency and (if using a
hosted embedder) money.

The fix: LRU + TTL cache keyed on `(collection, query, top_k)`.
**Crucial:** invalidate per-collection when that collection is written
to. This is where many caches get it wrong — they flush the whole
cache on any write, which makes hit-rate collapse.

Expose `hit_rate` on `/health`. If it's below ~40% in production, the
cache is doing nothing useful and you're paying complexity for no
benefit.

### 4e. Async summarizer (`summarizer.py`)

**Real bug this fixes:** at some point a session exceeds the context
window. The naive fix is a blocking LLM call that compresses old turns
— but that stalls the current user turn by ~2 seconds. In a chat UI
that's a visible hang.

The fix: `asyncio.Task` runs summarization in the background. The
window stores a placeholder. When the summary is ready, atomically
swap it in. The user's turn never waits for it.

Split the budget: 60% recent verbatim / 40% summary slot. If you go
beyond 60/40, you're hiding the most relevant recent context to make
room for compressed old context — usually wrong for short Q&A,
sometimes right for long tutoring sessions.

---

## Phase 5 — Optimization with DSPy

You only reach this phase after Phases 0–4 are proven. Prompt
optimization without evals is fashion, not engineering.

### The mental model

DSPy is *not* fine-tuning. No weights are updated. It's **automatic
prompt engineering**. Concretely: it runs your module with the teacher
LM on a labeled trainset, filters the outputs by your metric, and
saves the passing traces as few-shot demonstrations that get baked
into the prompt at inference time.

The compiled artifact (`my_profile.agent/deep_research.json`) is not
a model — it's a prompt template with exemplars attached.

### The four things you write

1. **A `dspy.Signature`** — the typed input/output contract for each
   task. e.g. `DeepResearchSig(question → answer, citations,
   confidence)`.
2. **A `dspy.Module`** — the wiring. `Predict(DeepResearchSig)`,
   possibly with a `ChainOfThought` wrapper. Pulls context from your
   RAG pipeline before prediction.
3. **A trainset** — `List[dspy.Example]` with hand-labeled inputs and
   **the minimum labels you can get away with**. Our trainset labels
   only citations, not answers — answers are what the teacher LM
   generates during compile.
4. **A metric** — `(example, prediction, trace) → bool | float`. This
   is the single most important piece of code in the DSPy flow. The
   optimizer maximizes this metric; a weak metric produces a weak
   agent.

### The compile command

```bash
python -m context_harness.compile_agent \
    --model gemini/gemini-2.5-flash-lite \
    --out my_profile.agent
```

### Reading compile output

```
Bootstrapped 4 full traces after 4 examples for up to 1 rounds.
```

Translation: the optimizer tried 4 examples, all 4 passed the metric,
it hit `max_bootstrapped_demos=4` and stopped. Clean compile.

```
Bootstrapped 4 full traces after 6 examples.
```

Translation: 2 examples failed the metric. This is normal — Socratic
metrics reject "spoiler" answers, and the teacher LM will sometimes
produce spoilers. Failure rate > 60% is a sign the metric is too
strict OR the teacher LM is too weak for the task.

### How much training data do you need?

**Less than you think, for few-shot bootstrapping.** `max_labeled_demos
= 8` means only 8 examples end up in the prompt. 15–25 trainset
examples is plenty. More labeled data only helps if you later switch
to a tuning optimizer (`MIPROv2`, `BootstrapFewShotWithRandomSearch`).

**More than you think, for evaluation.** You need a held-out eval set
*separate* from the trainset, and it has to be big enough to discriminate
a 5-point accuracy change from noise. 20 questions barely achieves
that; 50 is comfortable.

### When to scrape the wiki

After you've:
1. Compiled the agent on the hand-written corpus.
2. Run evals and categorised failures.
3. Found that >30% of failures are "the corpus doesn't contain this
   fact" rather than "the agent retrieved/answered wrongly".

*Then* scraping is justified, because you know exactly what facts you
need and can target your scrape. Scraping before evals produces
megabytes of irrelevant lore that make retrieval worse, not better.

---

## Phase 6 — Production concerns

### 6a. Tracing (`tracer.py`)

Every turn gets a `turn_id`. Every sub-event (`LlmCall`, `ToolCall`,
`Retrieval`, `ContextBuilt`, `Error`) gets a sequence number and a
latency. Store to SQLite (or Jaeger via OpenTelemetry for
production).

**The diagnostic question tracing answers:** "Why did turn 37 take 8
seconds and produce a wrong answer?" Without a trace you bisect with
print statements; with a trace you look at `GET /trace/37` and see
exactly where time went.

### 6b. Reliability (`reliability.py`)

Three patterns, in order of importance:

- **Retry with exponential backoff** — for 5xx and timeouts. Don't
  retry 4xx; those are your bug.
- **Circuit breaker** — after N failures in W seconds, fail fast for
  cooldown period. Prevents a dead downstream from eating all your
  request budget.
- **Bulkhead** — separate connection pool / rate limit per downstream.
  One downstream slowing down can't starve the others.

Build in the order listed. Don't build all three on day 1.

### 6c. Security (`security.py`)

Three layers:

- **Input**: redact PII, reject prompts that look like injection
  attempts.
- **Egress**: strip PII from outputs, deny URLs not on an allowlist.
- **Tool use**: per-tool allowlist, rate limit, audit log.

Treat the LLM as untrusted code. Anything it produces that touches an
external system (HTTP, DB, filesystem) goes through a gatekeeper.

### 6d. Architecture Decision Records (`docs/adr/`)

For any decision that is (a) cross-cutting, (b) hard to reverse, or
(c) controversial — write an ADR: context, decision, consequences,
one page. `docs/adr/README.md` is the index. Six months later when
someone asks "why did we build X this way?" the ADR answers without
you having to remember.

---

## Phase 7 — The meta-lessons (things that ate time this project)

### Dependency resolution is a tax, budget for it

This project hit four separate resolver conflicts:
- `chromadb` capping `tokenizers<=0.20.3` vs `litellm` requiring
  `tokenizers==0.22.2`.
- `litellm` requiring `pydantic==2.12.5` vs. pin at `2.10.6`.
- Intel Mac can't install `torch>=2.4`, forcing ChromaDB's ONNX
  embedding fallback.
- `pip` is lax, `uv` is strict. CI + local diverge.

Mitigations: loosen pins (`>=` not `==`) in `requirements.txt` unless
you have a specific reason to pin; use `uv` locally to catch conflicts
early; match CI to local as closely as possible.

### CI is a contract, not an afterthought

Every commit should be one the CI can build. If CI is broken for a
week, people stop looking at it, regressions pile up, and you end up
in a tangle like our scala-harness (12 compile errors that accumulated
under a broken installer step). Fix CI the day it breaks, not later.

### Commit discipline

Small commits, each one green, each one with a meaningful message that
explains *why*. The project's git log has examples on both ends of
this — reach for the good ones and imitate.

### The golden rule for learning

> If you can't explain to a newcomer *why* you built this, you haven't
> learned it yet.

Everything above is in service of that.

---

## One-page recap

1. Write what "correct" means. One page.
2. Hand-write a tiny corpus and a tiny eval set. Build the judge first.
3. Build the dumbest agent. Measure.
4. Add context engineering pieces one at a time, each justified by a
   specific eval-failure mode.
5. Add infrastructure (registry, version guard, cost, cache,
   summarizer) when a real bug demands it, not before.
6. Adopt DSPy for prompt optimization only *after* you have evals.
7. Add tracing, reliability, security layers for production.
8. Write ADRs for hard-to-reverse choices.

The whole project is recoverable from this page if you lose everything
else.
