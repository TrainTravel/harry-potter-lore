# TODO: Re-chunk character_lore

**Plan:** `tasks/plan-rechunk.md` · **Updated:** 2026-05-09

## Phase 1 — Eval harness (do first; user wants measurement before change)

- [ ] **T1: Write retrieval eval set** → `evals/eval_retrieval.py`
  - 12–18 entries; 4-field schema (query/character/expected_pattern/rationale)
  - ≥2 Aberforth entries
  - Verify: `python -c "from evals.eval_retrieval import EVAL_SET; print(len(EVAL_SET))"`

- [ ] **T2: Write retrieval eval runner** → `evals/run_retrieval_eval.py`
  - Recall@5 + Recall@10 + MRR@10 aggregates
  - `--out PATH` flag → JSON
  - Per-query table to stdout
  - Verify: runs without errors; JSON output parses

- [ ] **T3: Baseline snapshot** → `evals/results/baseline_2026-05-09.json`
  - Run eval against current (long-chunk) corpus
  - Confirm: Aberforth entry shows `hit@5=false, rank=14` (matches the known bug)

### Checkpoint A
- [ ] All Phase 1 done · Baseline reproduces the bug · **HUMAN SIGN-OFF before Phase 2**

## Phase 2 — Re-chunk

- [ ] **T4: Lower chunker word targets** → `scripts/chunk_character_lore.py`
  - `TARGET_MIN_WORDS = 80` (was 200)
  - `TARGET_MAX_WORDS = 150` (was 400)
  - Tail-emit threshold (line 228): 80 → 40 to match new band
  - Add comment with rationale link
  - Verify: one-file smoke produces ≥3 Aberforth chunks

- [ ] **T5: Regenerate character_lore.jsonl**
  - **First:** `cp data/character_lore.jsonl data/character_lore.jsonl.bak`
  - Run: `.venv/bin/python -m scripts.chunk_character_lore`
  - Verify: total 1.5–2.5× prior count; max word_count ≤ 200; ≥3 Aberforth chunks

- [ ] **T6: Re-tag + re-ingest**
  - **First:** `mv data/character_lore_tagged.jsonl data/character_lore_tagged.jsonl.bak`
  - Spot-check tagger cost: `.venv/bin/python -m scripts.tag_chunks_with_themes --limit 50`
  - Full re-tag: `.venv/bin/python -m scripts.tag_chunks_with_themes`
  - Re-ingest: `.venv/bin/python -m scripts.ingest_character_lore`
  - Verify: ≥3 Aberforth chunks queryable in `character_lore` collection

### Checkpoint B
- [ ] All Phase 2 done · `.bak` files exist for rollback

## Phase 3 — Verify

- [ ] **T7: Post-rechunk snapshot** → `evals/results/post-rechunk_2026-05-09.json`
  - Re-run `evals/run_retrieval_eval.py`
  - Verify: file written with full data

- [ ] **T8: Compare + decide** → `tasks/results-rechunk-2026-05-09.md`
  - Side-by-side Recall@5 / Recall@10 / MRR@10
  - Per-query gain/regression table
  - Specific check: Aberforth demo prompt now hits top-5?
  - Verdict: PASS / MIXED / FAIL
  - Recommended next action

### Checkpoint C — Decision
- [ ] If PASS: commit chunker change + new JSONL + eval results + decision note
- [ ] If MIXED: tighten chunker rule OR fall back + queue cross-encoder reranker as next experiment
- [ ] If FAIL: revert (restore from `.bak`), document learnings

## Done criteria

- All checkpoints complete
- Aberforth demo prompt produces a response that mentions Aberforth (not just generic "I have seen such rifts")
- No previously-passing eval queries regressed by more than 1 rank
- Backup `.bak` files removed only after sign-off
