"""
DSPy agent eval — compiled vs uncompiled comparison
====================================================
Runs the DSPyAgent against the same eval questions used by eval_agent.py,
with and without compiled few-shot demos. Reports accuracy, retrieval hit
rate, and cost for each.

Usage:
    # Uncompiled (zero-shot baseline):
    .venv/bin/python -m evals.eval_dspy

    # Compiled (with bootstrapped demos):
    .venv/bin/python -m evals.eval_dspy --agent-dir my_profile.agent

    # Quick smoke test:
    .venv/bin/python -m evals.eval_dspy --limit 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Set

from dotenv import load_dotenv
load_dotenv()

import dspy

from context_harness.dspy_agent import DSPyAgent
from context_harness.ingest_lore import build_pipeline
from evals.questions import QUESTIONS, EvalQuestion
from evals.eval_agent import judge, cost_usd, JUDGE_MODEL


MODEL = os.getenv("DSPY_MODEL", "gemini/gemini-2.5-flash-lite")


@dataclass
class DspyRow:
    id: str
    kind: str
    correct: bool
    retrieval_hit: Optional[bool]
    answer: str
    reason: str
    latency_s: float


def retrieval_hit_from_citations(citations: str, expected_docs: Set[str]) -> Optional[bool]:
    if not expected_docs:
        return None
    cited = {c.strip().lower() for c in citations.split() if c.strip()}
    # Fuzzy match: expected might be "dumbledore", corpus uses "albus-dumbledore"
    for expected in expected_docs:
        for cited_doc in cited:
            if expected in cited_doc or cited_doc in expected:
                return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Eval DSPy agent (compiled vs uncompiled)")
    parser.add_argument("--agent-dir", type=str, default=None,
                        help="path to compiled .agent dir (omit for uncompiled baseline)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    if "GOOGLE_API_KEY" not in os.environ:
        print("ERROR: set GOOGLE_API_KEY", file=sys.stderr)
        sys.exit(2)

    # Set GEMINI_API_KEY for litellm
    key = os.environ["GOOGLE_API_KEY"]
    os.environ.setdefault("GEMINI_API_KEY", key)

    label = "compiled" if args.agent_dir else "uncompiled"
    out_path = args.out or f"evals/results_dspy_{label}.json"

    print(f"=== DSPy eval ({label}) ===")
    print(f"  model:     {MODEL}")
    print(f"  agent_dir: {args.agent_dir or '(none — zero-shot)'}")

    dspy.configure(lm=dspy.LM(MODEL, api_key=key))

    pipeline = build_pipeline(persist=True)
    agent = DSPyAgent(pipeline, export_dir=args.agent_dir)

    questions: List[EvalQuestion] = QUESTIONS[:args.limit] if args.limit else QUESTIONS

    from google import genai
    judge_client = genai.Client()

    rows: List[DspyRow] = []
    for q in questions:
        print(f"\n>>> [{q.id} / {q.kind}] {q.question}")
        t0 = time.perf_counter()

        try:
            pred = agent.forward("deep_research", q.question)
            answer = getattr(pred, "answer", str(pred))
            citations = getattr(pred, "citations", "")
        except Exception as e:
            answer = f"ERROR: {e}"
            citations = ""

        latency = time.perf_counter() - t0

        verdict = judge(
            client=judge_client,
            question=q.question,
            agent_answer=answer,
            reference=q.reference_answer,
            kind=q.kind,
        )

        row = DspyRow(
            id=q.id,
            kind=q.kind,
            correct=bool(verdict.get("correct")),
            retrieval_hit=retrieval_hit_from_citations(citations, q.expected_docs),
            answer=answer[:500],
            reason=str(verdict.get("reason", "")),
            latency_s=latency,
        )
        rows.append(row)
        print(f"    correct={row.correct}  latency={row.latency_s:.1f}s")
        print(f"    answer: {row.answer[:160]}")

        # Rate limit pause for free tier
        time.sleep(2)

    # --- Summary ---
    total = len(rows)
    correct = sum(r.correct for r in rows)
    by_kind = {}
    for r in rows:
        by_kind.setdefault(r.kind, [0, 0])
        by_kind[r.kind][0] += int(r.correct)
        by_kind[r.kind][1] += 1

    hits = [r for r in rows if r.retrieval_hit is not None]
    hit_rate = sum(r.retrieval_hit for r in hits) / max(len(hits), 1)

    print(f"\n========= SUMMARY ({label}) =========")
    print(f"  correct:       {correct}/{total} ({correct/total:.1%})")
    for k, (c, n) in sorted(by_kind.items()):
        print(f"    {k:11s}   {c}/{n}")
    print(f"  retrieval hit: {sum(r.retrieval_hit for r in hits)}/{len(hits)} ({hit_rate:.1%})")
    print(f"  mean latency:  {sum(r.latency_s for r in rows)/total:.1f}s")

    with open(out_path, "w") as f:
        json.dump({"label": label, "accuracy": correct/total, "rows": [asdict(r) for r in rows]}, f, indent=2)
    print(f"\n  wrote {out_path}")


if __name__ == "__main__":
    main()
