# Plan: Re-chunk character_lore to fix retrieval miss on long chunks

**Date:** 2026-05-09
**Branch:** slice-perspective-disambiguation
**Bug evidence:** `albus-dumbledore/relationships-family-aberforth-dumbledore-001` (312 words) ranks #14 for the demo prompt about an estranged brother — embedding signal diluted by chunk length. Shorter Aberforth-002 (81 words) lands at #6. Other albus-dumbledore chunks (Ariana, Grindelwald) outscore Aberforth despite weaker topical alignment.

## Root cause

`scripts/chunk_character_lore.py` config:

```python
TARGET_MIN_WORDS = 200   # line 66
TARGET_MAX_WORDS = 400   # line 67
```

Mid-target emit threshold (line 224) is ~300 words. The `all-MiniLM-L6-v2` embedder's sweet spot is **40–150 words** — chunks 2× too long lose against shorter siblings.

## Architectural decision

**Eval-first, then change.** Build a small Recall@5 harness, baseline current corpus, *then* re-chunk, *then* re-measure. This is `learning_eval_design_best_practices.md`'s rule: don't change retrieval without a measurement.

Target chunk size: **`TARGET_MIN_WORDS=80`, `TARGET_MAX_WORDS=150`** (mid-target emit ≈ 115 words). Aberforth-001's 312 words splits into ~3 pieces, each with concentrated embedding signal.

Re-tagging cost: tagger is Gemini Flash-lite, ~$0.04 for 900 chunks per docs. Re-chunked corpus will be ~1500–2000 chunks → ~$0.07–0.12. Trivial.

## Pipeline being rebuilt

```
data/hp_wiki_raw/<slug>.json                            (untouched)
        │
        ▼  scripts/chunk_character_lore.py  (config change)
data/character_lore.jsonl                                (regenerated)
        │
        ▼  scripts/tag_chunks_with_themes.py  (re-run, ~$0.10)
data/character_lore_tagged.jsonl                         (regenerated)
        │
        ▼  scripts/ingest_character_lore.py
data/chromadb/character_lore                             (rebuilt; gitignored)
```

## Dependency graph

```
T1 (eval set) ─┐
T2 (runner) ───┤
               ├──→ T3 (baseline) ──→ T7 (post-rechunk eval) ──→ T8 (decision)
T4 (config)  ──┤                              ▲
T5 (rechunk) ──┤                              │
T6 (re-tag/ingest) ───────────────────────────┘
```

T1+T2 must complete before T3. T4 → T5 → T6 is a sequential chain. T3 and T6 both feed T7. T7 feeds T8.

---

## Phase 1: Eval harness

### T1 — Write retrieval eval set
**Files:** `evals/eval_retrieval.py` (NEW)
**Scope:** S
**Description:** Hand-curate ~15 (query, character, expected_chunk_pattern) tuples covering:
- The Aberforth bug (the brother prompt + 1 paraphrase)
- 8–10 other clear character→chunk mappings (Snape→Lily, McGonagall→discipline, Luna→nonconformity, Hermione→books-rules, Ron→loyalty, Harry→sacrifice-parents, Dumbledore→Grindelwald, Voldemort→horcruxes, Hagrid→creatures, Draco→father)
- 2 ambiguous cases (sanity check we don't over-fit Aberforth-style direct hits)

Schema:
```python
{
  "query": "I haven't spoken to my brother in three years...",
  "character": "albus-dumbledore",
  "expected_pattern": "aberforth",   # case-insensitive substring vs doc_id
  "rationale": "Demo prompt; should hit Aberforth-001 split children"
}
```

**Acceptance criteria:**
- Defines `EVAL_SET: list[dict]` with 12–18 entries
- Every entry has all four fields
- Patterns are substrings so they survive chunk-id renumbering
- ≥2 Aberforth entries (demo prompt + paraphrase)

**Verification:**
```bash
.venv/bin/python -c "from evals.eval_retrieval import EVAL_SET; print(len(EVAL_SET))"  # 12-18
```

---

### T2 — Write retrieval eval runner
**Files:** `evals/run_retrieval_eval.py` (NEW)
**Scope:** S
**Description:** Script that:
1. Loads `EVAL_SET` from `eval_retrieval.py`
2. Builds `character_lore` pipeline via `build_pipeline(persist=True, collection_name="character_lore")`
3. For each query, runs `pipeline.retrieve(query, top_k=10, where={"character": <slug>})`
4. Computes per-query: `hit@5` (bool — pattern in any top-5 doc_id), `hit@10`, `rank_of_first_match` (or `None`), `mrr_contribution = 1/rank if rank<=10 else 0`
5. Aggregates: `Recall@5`, `Recall@10`, `MRR@10`
6. `--out PATH` flag to dump JSON
7. Prints a per-query table + aggregate

**Acceptance criteria:**
- Runs against existing local ChromaDB without errors
- Output table is readable (no extra deps)
- Aggregates correct (verify by hand on 1 entry)
- `--out` writes parseable JSON

**Verification:**
```bash
.venv/bin/python -m evals.run_retrieval_eval
.venv/bin/python -m evals.run_retrieval_eval --out /tmp/test.json
cat /tmp/test.json | jq '.aggregate'
```

---

### T3 — Baseline measurement
**Files:** `evals/results/baseline_2026-05-09.json` (NEW)
**Scope:** XS
**Description:** Run eval against current (long-chunk) corpus, save the result. The Before snapshot.

**Acceptance criteria:**
- File exists with full per-query + aggregate data
- Aberforth demo-prompt entry shows `hit@5=false, rank_of_first_match=14` (or current rank) — confirms eval reproduces the known bug

**Verification:**
```bash
cat evals/results/baseline_2026-05-09.json | jq '.aggregate'
```

**Dependencies:** T1, T2

---

### Checkpoint A — Eval ready
- [ ] T1, T2, T3 complete
- [ ] Baseline numbers documented
- [ ] Aberforth bug reproduces in eval
- [ ] **Human sign-off** before touching the chunker

---

## Phase 2: Re-chunk

### T4 — Lower chunker word targets
**Files:** `scripts/chunk_character_lore.py` (constants only)
**Scope:** XS
**Description:** Change three constants:
```python
TARGET_MIN_WORDS = 80    # was 200
TARGET_MAX_WORDS = 150   # was 400
# Lower min-section-words guard at line 264 from 60 to 40 only if T5
# reveals tiny sections being dropped. Probably keep at 60.
# Lower tail-emit threshold at line 228 from 80 to ~40 to match new band.
```

Add inline comment explaining WHY (link to this plan + embedder sweet-spot rationale).

**Acceptance criteria:**
- Only constant changes + comment in the diff
- No algorithm changes

**Verification:**
- `git diff scripts/chunk_character_lore.py` shows only constant changes
- One-file smoke:
  ```bash
  .venv/bin/python -c "
  from scripts.chunk_character_lore import _chunks_for_character
  import json
  payload = json.load(open('data/hp_wiki_raw/albus-dumbledore.json'))
  chunks = _chunks_for_character(payload)
  aberforth = [c for c in chunks if 'aberforth' in c.chunk_id.lower()]
  print(f'aberforth chunk count: {len(aberforth)}')
  for c in aberforth: print(c.chunk_id, c.word_count)
  "
  ```
  Expects ≥3 Aberforth chunks (was 2)

---

### T5 — Regenerate `character_lore.jsonl`
**Files:** `data/character_lore.jsonl` (overwrite)
**Scope:** XS
**Description:** Run `.venv/bin/python -m scripts.chunk_character_lore`. Old file overwritten. **Backup first** (`cp data/character_lore.jsonl data/character_lore.jsonl.bak`) since git status didn't show it tracked.

**Acceptance criteria:**
- Backup created
- Total chunk count is 1.5–2.5× previous (~808 → ~1500–2000 expected)
- No script errors
- Every Aberforth chunk in new file ≤180 words

**Verification:**
```bash
wc -l data/character_lore.jsonl
.venv/bin/python -c "
import json
chunks = [json.loads(l) for l in open('data/character_lore.jsonl')]
print('total:', len(chunks))
print('max word_count:', max(c['word_count'] for c in chunks))
print('aberforth chunks:', [c['chunk_id'] for c in chunks if 'aberforth' in c['chunk_id'].lower()])
"
```

**Dependencies:** T4

---

### T6 — Re-tag + re-ingest
**Files:** `data/character_lore_tagged.jsonl` (overwrite), `data/chromadb/character_lore` (rebuild)
**Scope:** S (~$0.10 LLM cost)
**Description:** Two sub-steps:
1. **Re-tag:** `mv data/character_lore_tagged.jsonl data/character_lore_tagged.jsonl.bak` (tagger resumes from existing file — must clear). Run `.venv/bin/python -m scripts.tag_chunks_with_themes`. Spot-check cost first with `--limit 50`.
2. **Re-ingest:** `.venv/bin/python -m scripts.ingest_character_lore`. Auto-deletes old collection (line 92 of ingest script) before re-ingesting.

**Acceptance criteria:**
- `data/character_lore_tagged.jsonl` line count matches `data/character_lore.jsonl` minus chunks dropped for empty themes
- `character_lore` collection has the new chunk count
- Tagger spend ≤$0.20

**Verification:**
```bash
.venv/bin/python -c "
from context_harness.ingest_lore import build_pipeline
p = build_pipeline(persist=True, collection_name='character_lore')
print('total:', p.count())
res = p._collection.get(where={'character':'albus-dumbledore'}, include=['metadatas'])
aberforth = [m['doc_id'] for m in res['metadatas'] if 'aberforth' in m['doc_id'].lower()]
print('aberforth:', aberforth)
"
```
Expects ≥3 Aberforth chunk IDs.

**Dependencies:** T5

---

### Checkpoint B — Corpus rebuilt
- [ ] T4–T6 complete
- [ ] Aberforth chunk count ≥3
- [ ] No chunks exceed ~180 words
- [ ] `.bak` files exist for rollback

---

## Phase 3: Verify

### T7 — Post-rechunk measurement
**Files:** `evals/results/post-rechunk_2026-05-09.json` (NEW)
**Scope:** XS
**Description:** Run `evals/run_retrieval_eval.py` again against rebuilt corpus. Save result.

**Acceptance criteria:**
- File exists with per-query + aggregate data

**Verification:**
```bash
cat evals/results/post-rechunk_2026-05-09.json | jq '.aggregate'
```

**Dependencies:** T2, T6

---

### T8 — Compare baseline vs post-rechunk; decide
**Files:** `tasks/results-rechunk-2026-05-09.md` (NEW)
**Scope:** XS
**Description:** Side-by-side compare:
- Recall@5 baseline vs post-rechunk
- Recall@10 baseline vs post-rechunk
- MRR@10 baseline vs post-rechunk
- Per-query: which improved, regressed, stayed same
- **Specific:** does Aberforth demo prompt now hit top-5?

Three possible verdicts:
- ✅ **Ship**: aggregate Recall@5 improved AND no individual query regressed by >1 rank
- ⚠️ **Mixed**: Aberforth fixed but others regressed → either tighten chunker rule (e.g., split-only-when-section-exceeds-N), or fall back and add reranker
- ❌ **Worse**: rollback chunker config

**Acceptance criteria:**
- Markdown has a clear PASS/MIXED/FAIL verdict at top
- Per-query comparison table
- Recommended next action

**Verification:** Human review. **Decision is the user's, not the agent's.**

**Dependencies:** T3, T7

---

### Checkpoint C — Decision made
- [ ] Verdict written
- [ ] If PASS: commit chunker change + new tagged JSONL + plan note + eval results. ChromaDB dir is gitignored.
- [ ] If MIXED/FAIL: revert chunker, document learnings, propose next experiment (likely cross-encoder reranker)

---

## Files modified/created

| Path | Change | Why |
|---|---|---|
| `evals/eval_retrieval.py` | NEW | 15-query retrieval eval set |
| `evals/run_retrieval_eval.py` | NEW | Recall@5/10 + MRR runner |
| `evals/results/baseline_2026-05-09.json` | NEW | Before snapshot |
| `evals/results/post-rechunk_2026-05-09.json` | NEW | After snapshot |
| `scripts/chunk_character_lore.py` | MODIFY | Lower TARGET word constants |
| `data/character_lore.jsonl` | OVERWRITE (`.bak` first) | Re-chunked output |
| `data/character_lore_tagged.jsonl` | OVERWRITE (`.bak` first) | Re-tagged output |
| `data/chromadb/character_lore` | REBUILD | Gitignored |
| `tasks/results-rechunk-2026-05-09.md` | NEW | Decision write-up |

## What this does NOT do

- **No reranker.** MMR / cross-encoder is the next experiment if re-chunking alone isn't enough.
- **No embedder change.** all-MiniLM-L6-v2 stays.
- **No query rewriting.**
- **No corpus expansion.** Just re-shaping what's there.
- **No `_k=5 → _k=10` change in `PerspectiveShiftModule`.** Test if re-chunking alone is enough.
- **No production deploy.** Local-only verification.

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Tagger costs more than estimated | Low ($0.10 → $1.00 max) | Spot-check with `--limit 50` first |
| Smaller chunks lose narrative coherence (LLM response gets choppy) | Medium | T8 includes side-by-side prompt response check, not just retrieval scores |
| Other queries regress (fix Aberforth, break Snape-Lily) | High | T8 is per-query; mixed verdict triggers fallback |
| Min-section-word guard (60) drops sections we want | Low | T4 inspects chunker output; can lower to 40 if needed |
| New chunk_ids break existing trainset | Medium | `trainset_perspective_shift.py` doesn't reference specific chunk IDs in inputs, but verify in T1 with grep |
| Old `character_lore.jsonl` not in git → no rollback | High | Mitigated: `.bak` copy in T5/T6 |

## Open questions

- Is `data/character_lore.jsonl` tracked in git? (Mitigated by `.bak` regardless.)
- Should `evals/results/*.json` be committed? **Recommendation:** yes — improvement is reviewable in the PR.
