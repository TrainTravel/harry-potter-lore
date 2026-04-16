"""
DSPy EVALSET SLO check
======================
Runs the programmatic metrics (deep_research_metric, socratic_metric,
debate_metric) against each mode's held-out EVALSET and exits non-zero
if the pass rate falls below the SLO threshold.

Designed to run in CI on every PR (cheap: 5 real LLM calls per mode).

Usage:
    python -m evals.slo_check                        # deep_research, SLO 80%
    python -m evals.slo_check --threshold 0.6        # override threshold
    python -m evals.slo_check --mode debate
    python -m evals.slo_check --all-modes

Requires:
    GEMINI_API_KEY in environment (or GOOGLE_API_KEY)
    ChromaDB populated: python -m context_harness.ingest_lore
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

import dspy

from context_harness.ingest_lore import build_pipeline
from context_harness.dspy_agent import DSPyAgent
from context_harness.metrics import (
    deep_research_metric,
    socratic_metric,
    debate_metric,
)

# ---------------------------------------------------------------------------
# Mode config — which EVALSET + metric to use for each mode
# ---------------------------------------------------------------------------

MODE_CONFIG = {
    "deep_research": {
        "trainset_module": "data.trainset_deep_research",
        "metric": deep_research_metric,
        "input_field": "question",
        "default_slo": 0.80,
    },
    "guided_learning": {
        "trainset_module": "data.trainset_guided_learning",
        "metric": socratic_metric,
        "input_field": "question",
        "default_slo": 0.60,
    },
    "debate": {
        "trainset_module": "data.trainset_debate",
        "metric": debate_metric,
        "input_field": "position",
        "default_slo": 0.60,
    },
}


def _configure_lm() -> bool:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print(
            "ERROR: set GEMINI_API_KEY (or GOOGLE_API_KEY) to run this eval.\n"
            "       Get a free key at https://aistudio.google.com/apikey",
            file=sys.stderr,
        )
        return False
    os.environ.setdefault("GEMINI_API_KEY", api_key)
    dspy.configure(lm=dspy.LM("gemini/gemini-2.5-flash"))
    return True


def run_mode(mode: str, threshold: float, verbose: bool, agent: DSPyAgent) -> dict:
    cfg = MODE_CONFIG[mode]
    mod = importlib.import_module(cfg["trainset_module"])
    evalset = mod.EVALSET
    metric = cfg["metric"]
    input_field = cfg["input_field"]

    passed = 0
    for ex in evalset:
        question = getattr(ex, input_field)
        kwargs = {}
        if mode == "guided_learning":
            kwargs["past_attempts"] = getattr(ex, "past_attempts", "none")

        pred = agent.forward(mode, question, **kwargs)
        ok = bool(metric(ex, pred))
        passed += int(ok)

        if verbose:
            status = "PASS" if ok else "FAIL"
            got = getattr(pred, "citations", "—")
            exp = getattr(ex, "citations", "—")
            print(f"    {status}  {question[:70]}")
            print(f"           citations got={got!r}  expected={exp!r}")

    total = len(evalset)
    rate = passed / total if total else 0.0
    return {"mode": mode, "passed": passed, "total": total, "rate": rate,
            "slo": threshold, "slo_met": rate >= threshold}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DSPy EVALSET SLO check.")
    parser.add_argument("--mode", choices=list(MODE_CONFIG), default="deep_research")
    parser.add_argument("--all-modes", action="store_true")
    parser.add_argument("--threshold", type=float, default=None,
                        help="override default SLO (0.0–1.0)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if not _configure_lm():
        return 2

    pipeline = build_pipeline(persist=True)
    agent = DSPyAgent(pipeline)

    modes_to_run = list(MODE_CONFIG) if args.all_modes else [args.mode]
    results = []

    for mode in modes_to_run:
        cfg = MODE_CONFIG[mode]
        threshold = args.threshold if args.threshold is not None else cfg["default_slo"]
        print(f"\nEvaluating {mode}  (SLO: {threshold:.0%})")
        result = run_mode(mode, threshold, verbose=args.verbose, agent=agent)
        results.append(result)
        icon = "✓" if result["slo_met"] else "✗"
        print(f"  {icon}  {result['passed']}/{result['total']} passed "
              f"({result['rate']:.0%})  — {'OK' if result['slo_met'] else 'BELOW SLO'}")

    print()
    failed = [r for r in results if not r["slo_met"]]
    if failed:
        print(f"FAILED: {len(failed)} mode(s) below SLO:")
        for r in failed:
            print(f"  - {r['mode']}: {r['rate']:.0%} < {r['slo']:.0%}")
        return 1

    print(f"All {len(results)} mode(s) met SLO.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
