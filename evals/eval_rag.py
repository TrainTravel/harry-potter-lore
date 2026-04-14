"""
RAG-only eval harness
=====================
No LLM call — just measures how well the retrieval pipeline surfaces the
right passages for each eval question.

Metrics:
  - hit@k: does any expected doc appear in the top-k?
  - recall@k: fraction of expected docs that appear in top-k
  - avg tokens retrieved (proxy for context cost)
  - latency ms

Sweeps across ChunkingStrategy so you can see the tradeoffs.

Run:
    list/bin/python -m evals.eval_rag

(Uses the in-memory fallback by default; pass --chromadb to use real embeddings.)
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import List

from context_harness.rag_pipeline import RAGPipeline, ChunkingStrategy
from evals.corpus import all_docs
from evals.questions import QUESTIONS, EvalQuestion


@dataclass
class QuestionResult:
    q: EvalQuestion
    retrieved_doc_ids: List[str]
    hit: bool
    recall: float
    tokens_retrieved: int
    latency_ms: float


def eval_strategy(
    strategy: ChunkingStrategy,
    top_k: int,
    use_chromadb: bool,
) -> List[QuestionResult]:
    pipe = RAGPipeline(
        collection_name=f"eval_{strategy.value}",
        chunking_strategy=strategy,
        top_k=top_k,
        use_chromadb=use_chromadb,
    )
    pipe.ingest_many(all_docs())

    results: List[QuestionResult] = []
    for q in QUESTIONS:
        t0 = time.perf_counter()
        chunks = pipe.retrieve(q.question, top_k=top_k)
        dt = (time.perf_counter() - t0) * 1000

        retrieved_ids = [c.doc_id for c in chunks]
        retrieved_set = set(retrieved_ids)
        tokens = sum(len(c.text.split()) for c in chunks)

        if q.kind == "distractor":
            # For distractors, a "hit" means we DIDN'T surface false-positives
            # with high confidence. We can't fully evaluate without the LLM.
            # Here we just note that expected_docs is empty → recall is undefined.
            hit = True  # retrieval can't fail a distractor by itself
            recall = 1.0
        else:
            hit = bool(retrieved_set & q.expected_docs)
            recall = (
                len(retrieved_set & q.expected_docs) / len(q.expected_docs)
                if q.expected_docs else 1.0
            )

        results.append(QuestionResult(
            q=q, retrieved_doc_ids=retrieved_ids, hit=hit, recall=recall,
            tokens_retrieved=tokens, latency_ms=dt,
        ))
    return results


def summarize(name: str, results: List[QuestionResult]) -> dict:
    non_distractor = [r for r in results if r.q.kind != "distractor"]
    return {
        "strategy": name,
        "hit_rate": sum(r.hit for r in non_distractor) / max(len(non_distractor), 1),
        "mean_recall": statistics.mean(r.recall for r in non_distractor),
        "mean_tokens": statistics.mean(r.tokens_retrieved for r in results),
        "mean_latency_ms": statistics.mean(r.latency_ms for r in results),
        # breakdown by kind
        "easy_hit": sum(r.hit for r in results if r.q.kind == "easy")
                    / max(sum(1 for r in results if r.q.kind == "easy"), 1),
        "multi_hit": sum(r.hit for r in results if r.q.kind == "multi")
                     / max(sum(1 for r in results if r.q.kind == "multi"), 1),
    }


def print_table(rows: List[dict]) -> None:
    cols = ["strategy", "hit_rate", "mean_recall", "easy_hit", "multi_hit",
            "mean_tokens", "mean_latency_ms"]
    widths = {c: max(len(c), max(len(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c])) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(
            (f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c])).ljust(widths[c])
            for c in cols
        ))


def print_miss_report(name: str, results: List[QuestionResult]) -> None:
    misses = [r for r in results if not r.hit and r.q.kind != "distractor"]
    if not misses:
        return
    print(f"\n[{name}] retrieval misses:")
    for r in misses:
        print(f"  {r.q.id} ({r.q.kind}): {r.q.question}")
        print(f"    expected: {sorted(r.q.expected_docs)}")
        print(f"    got:      {r.retrieved_doc_ids}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--chromadb", action="store_true",
                        help="Use real ChromaDB + embeddings (slower, needs model download)")
    args = parser.parse_args()

    strategies = [
        ChunkingStrategy.FIXED,
        ChunkingStrategy.SENTENCE,
        ChunkingStrategy.PARAGRAPH,
        ChunkingStrategy.RECURSIVE,
    ]

    all_rows = []
    all_results = {}
    for s in strategies:
        res = eval_strategy(s, top_k=args.top_k, use_chromadb=args.chromadb)
        all_results[s.value] = res
        all_rows.append(summarize(s.value, res))

    print(f"\n=== RAG eval (top_k={args.top_k}, backend={'chromadb' if args.chromadb else 'keyword-fallback'}) ===\n")
    print_table(all_rows)
    for s, res in all_results.items():
        print_miss_report(s, res)


if __name__ == "__main__":
    main()
