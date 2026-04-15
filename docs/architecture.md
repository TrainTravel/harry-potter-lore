# Architecture

## Retrieval Ranking Pipeline

The pipeline has two stages: ANN retrieval from ChromaDB (with optional metadata
filtering), then optional MMR reranking.

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

### Chunk provenance metadata

Every chunk stored in ChromaDB now carries three provenance fields in its metadata:

| Field | Type | Example values | Purpose |
|---|---|---|---|
| `universe` | `str` | `"hp"`, `"lotr"`, `"real_world"` | Corpus namespace for filtered queries |
| `source_type` | `SourceType` enum | `fictional_canon`, `wiki`, `biography`, `news` | Epistemic classification for LLM framing |
| `as_of` | `str \| None` | `"2024-03"`, `None` | Freshness date for real-world sources |

These fields flow through the full stack:

```
RAGPipeline.ingest(universe=, source_type=, as_of=)
    └─► stored in ChromaDB chunk metadata
DocumentRegistry.upsert(universe=, source_type=, as_of=)
    └─► forwarded to ingest(), also saved in doc_records SQLite table
RAGPipeline.retrieve(universe=, source_type=)
    └─► translated to ChromaDB where= filter
```

### Metadata filter syntax

```python
# Restrict to one universe
pipeline.retrieve("dark lords", universe="hp")
# → where={"universe": {"$eq": "hp"}}

# Restrict to one source type
pipeline.retrieve("Elon Musk", source_type=SourceType.WIKI)
# → where={"source_type": {"$eq": "wiki"}}

# Both filters combined
pipeline.retrieve("powerful leaders", universe="real_world", source_type=SourceType.BIOGRAPHY)
# → where={"$and": [{"universe": {"$eq": "real_world"}}, {"source_type": {"$eq": "biography"}}]}

# Cross-universe (no filter) — semantic search across all corpora
pipeline.retrieve("leaders who consolidated power", use_mmr=True)
```

### MMR is now wired

Pass `use_mmr=True` to `retrieve()` to enable MMR reranking inline.
Previously `mmr_rerank()` existed but was never called from the agent path.

```python
# DeepResearchModule — cross-universe query with diversity reranking
chunks = pipeline.retrieve(question, top_k=10, use_mmr=True, mmr_lambda=0.5)
```

### Chunking strategies

| Strategy | Best for | Behaviour |
|---|---|---|
| `RECURSIVE` | HP-style narrative lore (default) | paragraph → sentence → fixed fallback |
| `WIKI` | Wikipedia / biographical articles | splits on `##` section headers; oversized sections fall back to RECURSIVE |
| `PARAGRAPH` | Short structured documents | blank-line boundaries only |
| `SENTENCE` | Dense academic text | groups N sentences per chunk |
| `FIXED` | Fallback / uniform sizing | sliding window with overlap |

### Multi-universe note

ANN search is purely geometric — it returns the closest vectors regardless of which
universe they originated from. Adding a second universe to the same collection gives
cross-universe semantic search for free. To restrict results to one universe, pass a
`universe=` filter to `retrieve()` or use one collection per universe and route the
query before retrieval.
