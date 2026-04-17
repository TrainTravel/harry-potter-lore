"""
DSPy agent eval — compiled vs uncompiled comparison (v2)
========================================================
Improvements over v1:
  - Runs each eval N times, reports mean ± stddev
  - Drops first query as warmup for latency stats
  - Counts DSPy structured-output retries via log capture
  - Uses all question kinds (easy, multi, inference, distractor)

Usage:
    # Uncompiled baseline:
    .venv/bin/python -m evals.eval_dspy

    # Compiled:
    .venv/bin/python -m evals.eval_dspy --agent-dir my_profile.agent

    # Quick smoke:
    .venv/bin/python -m evals.eval_dspy --limit 5

    # Multiple runs for statistical significance:
    .venv/bin/python -m evals.eval_dspy --runs 3
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
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
    retries: int  # number of structured-output fallbacks


def retrieval_hit_from_citations(citations: str, expected_docs: Set[str]) -> Optional[bool]:
    if not expected_docs:
        return None
    cited = {c.strip().lower() for c in citations.split() if c.strip()}
    for expected in expected_docs:
        for cited_doc in cited:
            if expected in cited_doc or cited_doc in expected:
                return True
    return False


class RetryCounter(logging.Handler):
    """Captures DSPy's 'falling back to JSON mode' warnings to count retries."""

    def __init__(self):
        super().__init__()
        self.count = 0

    def emit(self, record):
        if "falling back" in record.getMessage().lower():
            self.count += 1

    def reset(self):
        self.count = 0


def run_once(agent, questions, judge_client, retry_counter, warmup=True):
    """Run one full eval pass. Returns list of DspyRow."""
    rows: List[DspyRow] = []

    for i, q in enumerate(questions):
        is_warmup = warmup and i == 0
        prefix = "[warmup] " if is_warmup else ""
        print(f"\n{prefix}>>> [{q.id} / {q.kind}] {q.question}")

        retry_counter.reset()
        t0 = time.perf_counter()

        try:
            pred = agent.forward("deep_research", q.question)
            answer = getattr(pred, "answer", str(pred))
            citations = getattr(pred, "citations", "")
        except Exception as e:
            answer = f"ERROR: {e}"
            citations = ""

        latency = time.perf_counter() - t0
        retries = retry_counter.count

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
            retries=retries,
        )
        rows.append(row)

        marker = "✓" if row.correct else "✗"
        print(f"    {marker} correct={row.correct}  latency={row.latency_s:.1f}s  retries={row.retries}")
        print(f"    answer: {row.answer[:160]}")

        # Rate limit pause
        time.sleep(2)

    return rows


def summarize(all_runs, label, warmup=True):
    """Summarize across multiple runs."""
    n_runs = len(all_runs)

    # Per-question accuracy across runs
    q_ids = [r.id for r in all_runs[0]]
    per_q = {}
    for qid in q_ids:
        corrects = []
        latencies = []
        retries_list = []
        for run in all_runs:
            row = next(r for r in run if r.id == qid)
            corrects.append(int(row.correct))
            latencies.append(row.latency_s)
            retries_list.append(row.retries)
        per_q[qid] = {
            "correct_rate": sum(corrects) / len(corrects),
            "kind": next(r for r in all_runs[0] if r.id == qid).kind,
            "mean_latency": sum(latencies) / len(latencies),
            "mean_retries": sum(retries_list) / len(retries_list),
        }

    # Overall accuracy per run
    accs = [sum(r.correct for r in run) / len(run) for run in all_runs]
    mean_acc = sum(accs) / len(accs)
    std_acc = math.sqrt(sum((a - mean_acc) ** 2 for a in accs) / max(len(accs) - 1, 1)) if len(accs) > 1 else 0

    # Latency (drop warmup = first question of each run)
    if warmup:
        lat_rows = [r for run in all_runs for r in run[1:]]  # skip first
    else:
        lat_rows = [r for run in all_runs for r in run]
    mean_lat = sum(r.latency_s for r in lat_rows) / max(len(lat_rows), 1)

    # Total retries
    total_retries = sum(r.retries for run in all_runs for r in run)
    total_questions = sum(len(run) for run in all_runs)

    # By kind
    by_kind = {}
    for qid, info in per_q.items():
        k = info["kind"]
        by_kind.setdefault(k, [])
        by_kind[k].append(info["correct_rate"])

    print(f"\n========= SUMMARY ({label}, {n_runs} run{'s' if n_runs > 1 else ''}) =========")
    if n_runs > 1:
        print(f"  accuracy:         {mean_acc:.1%} ± {std_acc:.1%}")
    else:
        print(f"  accuracy:         {mean_acc:.1%}")

    for k in sorted(by_kind):
        rates = by_kind[k]
        kind_acc = sum(rates) / len(rates)
        print(f"    {k:11s}      {kind_acc:.0%} ({len(rates)} questions)")

    hits = [v for v in per_q.values() if v["kind"] != "distractor"]
    print(f"  mean latency:     {mean_lat:.1f}s (warmup excluded)" if warmup else f"  mean latency:     {mean_lat:.1f}s")
    print(f"  total retries:    {total_retries}/{total_questions} ({total_retries/max(total_questions,1):.0%} of questions)")

    return {
        "label": label,
        "runs": n_runs,
        "accuracy_mean": mean_acc,
        "accuracy_std": std_acc,
        "mean_latency_s": mean_lat,
        "retry_rate": total_retries / max(total_questions, 1),
        "per_question": per_q,
        "raw_runs": [[asdict(r) for r in run] for run in all_runs],
    }


def main():
    parser = argparse.ArgumentParser(description="Eval DSPy agent (v2)")
    parser.add_argument("--agent-dir", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    if "GOOGLE_API_KEY" not in os.environ:
        print("ERROR: set GOOGLE_API_KEY", file=sys.stderr)
        sys.exit(2)

    key = os.environ["GOOGLE_API_KEY"]
    os.environ.setdefault("GEMINI_API_KEY", key)

    label = "compiled" if args.agent_dir else "uncompiled"
    out_path = args.out or f"evals/results_dspy_{label}.json"
    warmup = not args.no_warmup

    print(f"=== DSPy eval ({label}) ===")
    print(f"  model:     {MODEL}")
    print(f"  agent_dir: {args.agent_dir or '(none — zero-shot)'}")
    print(f"  runs:      {args.runs}")
    print(f"  warmup:    {'yes (first question excluded from latency)' if warmup else 'no'}")

    dspy.configure(lm=dspy.LM(MODEL, api_key=key))

    pipeline = build_pipeline(persist=True)
    agent = DSPyAgent(pipeline, export_dir=args.agent_dir)

    questions: List[EvalQuestion] = QUESTIONS[:args.limit] if args.limit else QUESTIONS

    from google import genai
    judge_client = genai.Client()

    # Set up retry counter on DSPy's logger
    retry_counter = RetryCounter()
    dspy_logger = logging.getLogger("dspy")
    dspy_logger.addHandler(retry_counter)

    all_runs = []
    for run_idx in range(args.runs):
        if args.runs > 1:
            print(f"\n{'='*40}")
            print(f"  RUN {run_idx + 1}/{args.runs}")
            print(f"{'='*40}")
        rows = run_once(agent, questions, judge_client, retry_counter, warmup=warmup)
        all_runs.append(rows)

    result = summarize(all_runs, label, warmup=warmup)

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  wrote {out_path}")


if __name__ == "__main__":
    main()
