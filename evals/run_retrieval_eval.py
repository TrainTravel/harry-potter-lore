"""
Run the character-lore retrieval eval.

Usage:
    .venv/bin/python -m evals.run_retrieval_eval
    .venv/bin/python -m evals.run_retrieval_eval --out evals/results/baseline_2026-05-09.json

Reports Recall@5, Recall@10, MRR@10. Per-query rows show whether the
expected chunk surfaced and at what rank.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evals.eval_retrieval import EVAL_SET
from context_harness.ingest_lore import build_pipeline


def _eval_one(pipeline, entry: dict, top_k: int) -> dict:
    chunks = pipeline.retrieve(
        entry["query"],
        top_k=top_k,
        where={"character": entry["character"]},
    )
    pattern = entry["expected_pattern"].lower()
    rank = None
    for i, c in enumerate(chunks, start=1):
        if pattern in c.doc_id.lower():
            rank = i
            break
    return {
        "query": entry["query"],
        "character": entry["character"],
        "expected_pattern": entry["expected_pattern"],
        "rank": rank,
        "hit_at_5": rank is not None and rank <= 5,
        "hit_at_10": rank is not None and rank <= 10,
        "mrr_contrib": (1.0 / rank) if (rank is not None and rank <= 10) else 0.0,
        "top_5_doc_ids": [c.doc_id for c in chunks[:5]],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collection", default="character_lore")
    ap.add_argument("--top-k", type=int, default=10,
                    help="Fetch this many chunks; rank is computed within them")
    ap.add_argument("--out", type=Path,
                    help="Optional JSON output path")
    args = ap.parse_args()

    pipeline = build_pipeline(persist=True, collection_name=args.collection)

    per_query = [_eval_one(pipeline, e, args.top_k) for e in EVAL_SET]

    n = len(per_query)
    recall_at_5 = sum(r["hit_at_5"] for r in per_query) / n
    recall_at_10 = sum(r["hit_at_10"] for r in per_query) / n
    mrr_at_10 = sum(r["mrr_contrib"] for r in per_query) / n

    # ---- Per-query table ----
    print(f"\nRetrieval eval — collection={args.collection!r}, n={n}, top_k={args.top_k}")
    print()
    print(f"  {'#':>2}  {'char':<20}  {'pattern':<28}  {'rank':>4}  {'hit5':>5}  {'mrr':>6}")
    print(f"  {'-'*2}  {'-'*20}  {'-'*28}  {'-'*4}  {'-'*5}  {'-'*6}")
    for i, r in enumerate(per_query, 1):
        rank_str = str(r["rank"]) if r["rank"] is not None else "-"
        hit_str = "YES" if r["hit_at_5"] else "no"
        print(f"  {i:>2}  {r['character'][:20]:<20}  "
              f"{r['expected_pattern'][:28]:<28}  {rank_str:>4}  "
              f"{hit_str:>5}  {r['mrr_contrib']:>6.3f}")

    # ---- Aggregate ----
    print()
    print(f"  Recall@5:   {recall_at_5:.3f}  ({sum(r['hit_at_5'] for r in per_query)}/{n})")
    print(f"  Recall@10:  {recall_at_10:.3f}  ({sum(r['hit_at_10'] for r in per_query)}/{n})")
    print(f"  MRR@10:     {mrr_at_10:.3f}")

    # ---- JSON dump ----
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "collection": args.collection,
            "top_k": args.top_k,
            "n_queries": n,
            "aggregate": {
                "recall_at_5": recall_at_5,
                "recall_at_10": recall_at_10,
                "mrr_at_10": mrr_at_10,
            },
            "per_query": per_query,
        }
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"\n  → wrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
