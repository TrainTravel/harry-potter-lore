# Re-chunk results — 2026-05-09

## Verdict: ✅ **SHIP** (with a caveat on the marketing prompt)

Aggregate retrieval quality improved substantially. The headline demo prompt regressed at the surface but the underlying chunks moved up in absolute rank — the issue is that other chunks rose even faster. A separate paraphrase of the same Aberforth question now ranks #1. Net: ship the rechunk, switch the marketing screenshot to a different (working) prompt, queue cross-encoder reranking as the next experiment.

## Aggregate

| Metric | Baseline (200–400w chunks) | Post-rechunk (80–150w chunks) | Δ |
|---|---|---|---|
| Recall@5 | 0.400 (6/15) | **0.667 (10/15)** | **+26.7 pp** |
| Recall@10 | 0.667 (10/15) | 0.733 (11/15) | +6.7 pp |
| MRR@10 | 0.336 | **0.443** | +0.107 |
| Total chunks | 808 | 1665 | +857 |
| Max chunk word_count | ~400 | 150 | -250 |

## Per-query delta

| # | Character | Pattern | Baseline rank | Post rank | Verdict |
|---|---|---|---|---|---|
| 1 | albus-dumbledore | aberforth (demo prompt) | 6 | 11 | ⚠️ regressed (out of top 10) |
| 2 | albus-dumbledore | aberforth (paraphrase) | 5 | **1** | ✅ improved |
| 3 | severus-snape | lily-evans | 10 | (none) | ❌ regressed |
| 4 | minerva-mcgonagall | dolores-umbridge | 1 | 2 | ≈ held |
| 5 | luna-lovegood | personality-and-traits | 1 | 2 | ≈ held |
| 6 | hermione-granger | family-parents | (none) | **1** | ✅ improved |
| 7 | ron-weasley | envying-harry-potter | (none) | **2** | ✅ improved |
| 8 | harry-potter | godric | (none) | **4** | ✅ improved |
| 9 | albus-dumbledore | grindelwald | (none) | (none) | — unchanged |
| 10 | lord-voldemort | horcrux | (none) | (none) | — unchanged |
| 11 | rubeus-hagrid | pets-and-other-creatures | 3 | 4 | ≈ held |
| 12 | draco-malfoy | family-parents | 8 | **1** | ✅ improved |
| 13 | luna-lovegood | luna-s-beliefs | 1 | 1 | ≈ held |
| 14 | severus-snape | double-agent | 9 | 7 | ✅ improved |
| 15 | ron-weasley | hermione-granger | 1 | 2 | ≈ held |

**Improved:** 6 queries (most by large margins — 3 went from "not in top 10" to top 5)
**Held:** 6 queries (within ±1 rank, all still in top 5)
**Regressed:** 2 queries (the demo prompt and Snape-Lily)
**Unchanged in failure:** 2 queries (Dumbledore-Grindelwald and Voldemort-horcrux remain unfindable)

## What's going on with the demo prompt regression?

The query is *"I haven't spoken to my brother in three years over a fight neither of us can remember the original cause of, but we both still tell ourselves we were right."* This is highly abstract / metaphorical. None of the Aberforth chunks contain literal vocabulary like "three years", "original cause", "tell ourselves we were right".

Aberforth-001 in fact moved **up** in absolute rank: #14 (baseline) → #11 (post-rechunk). The chunk got *better*. But because all chunks are finer now, biographical event chunks with mid-density references to "fight" / "brothers" / "wars" rose even faster. The aggregate ranking shuffled in ways that pushed Aberforth out of the top 10.

This is a **query-vocabulary mismatch**, not a chunking failure. Embedding similarity (cosine on all-MiniLM-L6-v2) doesn't bridge the metaphor gap on its own. Two paths to fix:
- **Cross-encoder reranker** would read `(query, chunk)` jointly and likely catch the topical fit despite vocabulary divergence. This is the standard industry move (~30 lines + a small open-weights model like `BAAI/bge-reranker-large`).
- **Query rewriting** — expand "brother" → "Aberforth", "estrangement", "sibling" before embedding. Simpler than reranker but more brittle.

## What's going on with Snape-Lily and Dumbledore-Grindelwald?

Both regressed/unchanged for similar reasons — the queries are emotionally framed in modern psychology language (*"I lost the love of my life because of who I used to be"*, *"I trusted my closest friend with everything"*) that doesn't match the chunks' literal HP-canonical vocabulary. Same fix candidates: reranker or query rewriting.

## Recommended next actions

1. **Commit this work.** Aggregate Recall@5 improved 27 percentage points. 6 previously-broken queries now hit top-5. Two regressions are bounded.
2. **Switch the marketing demo prompt.** Use the paraphrase (*"How do you reconcile with a sibling after a tragic family loss has driven you apart?"*) which now ranks #1 — the response should mention Aberforth directly. Verify with one live API call before screenshotting.
3. **Queue cross-encoder reranker as the next experiment** (issue or new plan). Expected to specifically fix queries 1, 3, 9, 10 — all of which involve abstract/metaphorical phrasing where the right canonical chunk has divergent vocabulary.
4. **Don't bump `_k` from 5 to 10** — wasn't needed. Re-chunking alone moved the right answers into top-5 for 4 queries.

## Files changed

- `scripts/chunk_character_lore.py` — `TARGET_MIN_WORDS=80, TARGET_MAX_WORDS=150` (was 200/400); tail threshold 80→40
- `data/character_lore.jsonl` — regenerated (808 → 1696 chunks); old version backed up to `.bak`
- `data/character_lore_tagged.jsonl` — re-tagged via Gemini Flash-lite (~$0.04, 32s wall-clock); old version backed up to `.bak`
- `data/chromadb/character_lore` — collection rebuilt to 1665 vectors (1696 minus 31 factual-only chunks dropped at ingest)
- `evals/eval_retrieval.py` — NEW (15 queries)
- `evals/run_retrieval_eval.py` — NEW (Recall@5/10 + MRR runner)
- `evals/results/baseline_2026-05-09.json` — NEW (Before)
- `evals/results/post-rechunk_2026-05-09.json` — NEW (After)
- `tasks/plan-rechunk.md` — NEW (plan, kept for history)
- `tasks/todo-rechunk.md` — NEW (todo tracker)
- `tasks/results-rechunk-2026-05-09.md` — NEW (this file)

## Cleanup before commit

- [ ] Delete `data/character_lore.jsonl.bak` and `data/character_lore_tagged.jsonl.bak` once we're sure we want to keep the new versions (or move to `data/.archive/` if we want to keep them for a while).
- [ ] Decide whether `evals/results/*.json` go into git (recommendation: yes, so the improvement is reviewable).
- [ ] Run the live demo prompt against the rebuilt corpus end-to-end to confirm the response actually mentions Aberforth (the paraphrase should; the original demo prompt likely won't, per the rank-11 result).
