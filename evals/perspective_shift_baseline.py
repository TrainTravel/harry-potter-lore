"""
Perspective-shift baseline eval
================================
Runs 10 real-life scenarios against the current ``perspective_shift`` mode
and saves outputs to a JSON file. Used as the BEFORE snapshot — once we
ingest the new 814-chunk character corpus and update the retrieval, we
re-run the same scenarios and diff the outputs.

Scenarios are chosen to span:
  - career / identity (paralysis, impostor syndrome, fame)
  - relationships (unrequited love, betrayal, grief)
  - belonging (class anxiety, nonconformity)
  - power dynamics (toxic authority, family expectations)

Each scenario is labelled with a suggested character (which the current
``PerspectiveShiftModule`` uses as a retrieval filter) and
``expected_principle_keywords`` — the thematic beats a good answer should
touch. No hard scoring yet; outputs are for manual read + future
LLM-as-judge pass.

Usage::

    .venv/bin/python -m evals.perspective_shift_baseline
    .venv/bin/python -m evals.perspective_shift_baseline --out evals/perspective_shift_v2.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import dspy

from context_harness.dspy_agent import DSPyAgent
from context_harness.ingest_lore import build_pipeline


# ---------------------------------------------------------------------------
# Scenario set
# ---------------------------------------------------------------------------

SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "career-paralysis",
        "scenario": (
            "I'm 28, stuck in a stable corporate job that pays well but feels "
            "soul-draining. A riskier creative path has opened up and my partner "
            "is supportive. I can't decide — every morning I tell myself I'll "
            "quit, every evening I talk myself out of it. The indecision is "
            "costing me more than either choice would."
        ),
        "character": "Dumbledore",
        "expected_principle_keywords": ["choice", "right vs easy", "moral weight", "carrying"],
    },
    {
        "id": "impostor-syndrome",
        "scenario": (
            "I got promoted to senior engineer last month. Everyone in the "
            "meetings seems so much more qualified than me. I'm convinced they'll "
            "realize I don't belong here — I keep waiting to be found out."
        ),
        "character": "Neville",
        "expected_principle_keywords": ["late bloomer", "growth", "showing up", "competence develops"],
    },
    {
        "id": "unrequited-love",
        "scenario": (
            "I've been in love with my best friend for eight years. They just "
            "told me they're engaged to someone else. I'm happy for them — "
            "genuinely — and also the floor is gone. How do I stay in their "
            "life without becoming bitter, or lose them by pulling away?"
        ),
        "character": "Snape",
        "expected_principle_keywords": ["sustained", "protect without possessing", "quiet service", "not bitter"],
    },
    {
        "id": "class-anxiety",
        "scenario": (
            "I grew up working-class and just started at a consulting firm where "
            "everyone went to Ivy League schools. Their small talk is about "
            "family trips to Tuscany. I feel like I'm performing a character "
            "all day. My work is solid but I'm exhausted."
        ),
        "character": "Ron",
        "expected_principle_keywords": ["not the money", "belonging", "who shows up for you", "comparison"],
    },
    {
        "id": "grief-hollow",
        "scenario": (
            "My mother died seven months ago. I'm not crying anymore. I'm "
            "supposed to be 'doing better' but I feel hollow and can't connect "
            "with people who haven't lost a parent. The distance is starting "
            "to scare me."
        ),
        "character": "Luna",
        "expected_principle_keywords": ["grief doesn't resolve", "carrying", "unchanged love", "patience"],
    },
    {
        "id": "nonconformity-office",
        "scenario": (
            "I'm the only openly queer person at my office. I've been code-"
            "switching for two years — corporate voice, corporate clothes, "
            "corporate weekend stories. It's costing me real energy and I'm "
            "starting to resent everyone. But the job is genuinely good."
        ),
        "character": "Luna",
        "expected_principle_keywords": ["stay yourself", "the right people find you", "cost of masking"],
    },
    {
        "id": "toxic-advisor",
        "scenario": (
            "My PhD advisor is brilliant but cruel. He publicly humiliates "
            "students in lab meetings. I'm three years in — leaving means "
            "losing all that progress. Staying means another two years of "
            "being torn down. Everyone says 'just finish.'"
        ),
        "character": "McGonagall",
        "expected_principle_keywords": ["no duty requires enduring cruelty", "boundaries", "protection"],
    },
    {
        "id": "family-expectations",
        "scenario": (
            "My parents expect me to take over their small business. I want to "
            "be a public-school teacher. Every time I bring it up, my mother "
            "cries and my father goes silent for days. I've started lying "
            "about interviews to avoid the fight."
        ),
        "character": "Neville",
        "expected_principle_keywords": ["inherited vs chosen path", "quiet courage", "you are not betraying them"],
    },
    {
        "id": "cofounder-betrayal",
        "scenario": (
            "My co-founder has been secretly interviewing at other companies "
            "for six months while I've been pulling 80-hour weeks. I found out "
            "yesterday. I can't look at them the same. I also can't run the "
            "company alone, and we have investors."
        ),
        "character": "Ron",
        "expected_principle_keywords": ["trust broken once", "clear-eyed", "not every breach ends a thing", "what you owe yourself"],
    },
    {
        "id": "viral-fame",
        "scenario": (
            "A video I posted went viral. Now strangers recognize me on the "
            "subway, brands are reaching out, my parents are asking when I'll "
            "'monetize.' I just wanted to make one silly video. I'm exhausted "
            "by attention I never asked for."
        ),
        "character": "Harry",
        "expected_principle_keywords": ["fame is not chosen", "protect your inner life", "no obligation to scale"],
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _configure_lm() -> None:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: set GEMINI_API_KEY or GOOGLE_API_KEY", file=sys.stderr)
        raise SystemExit(2)
    os.environ.setdefault("GEMINI_API_KEY", api_key)
    dspy.configure(lm=dspy.LM("gemini/gemini-2.5-flash-lite"))


# ---------------------------------------------------------------------------
# Percentile helper (linear interpolation — matches DuckDB PERCENTILE_CONT)
# ---------------------------------------------------------------------------

def _percentile(sorted_values: list[float], p: float) -> float:
    """Compute the p-th percentile of a SORTED list via linear interpolation.

    Matches DuckDB's PERCENTILE_CONT semantics and common SRE conventions.
    For n=10 and p=50 returns (items[4] + items[5]) / 2 — i.e. the true
    median, not the slightly-upper items[5] that len//2 gives.

    Args:
        sorted_values: ascending-sorted list of numeric measurements
        p: percentile in [0, 100]
    """
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    # Rank in [0, n-1], fractional allowed
    k = (len(sorted_values) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = k - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def _latency_summary(results: list[dict[str, Any]]) -> dict[str, float]:
    """Extract p50/p95/p99 + min/max/mean from successful scenario results."""
    lats = sorted(r["latency_ms"] for r in results if "latency_ms" in r)
    if not lats:
        return {"n": 0}
    return {
        "n":       len(lats),
        "min_ms":  round(lats[0], 1),
        "max_ms":  round(lats[-1], 1),
        "mean_ms": round(sum(lats) / len(lats), 1),
        "p50_ms":  round(_percentile(lats, 50), 1),
        "p95_ms":  round(_percentile(lats, 95), 1),
        "p99_ms":  round(_percentile(lats, 99), 1),
    }


def run(out_path: Path) -> int:
    _configure_lm()
    pipeline = build_pipeline(persist=True)
    agent = DSPyAgent(pipeline)

    results: list[dict[str, Any]] = []
    t_total = time.time()
    for i, sc in enumerate(SCENARIOS, start=1):
        print(f"[{i}/{len(SCENARIOS)}] {sc['id']}  (character={sc['character']})")
        t0 = time.time()
        try:
            pred = agent.forward(
                "perspective_shift",
                sc["scenario"],
                character=sc["character"],
            )
            latency_ms = (time.time() - t0) * 1000
            result = {
                "id": sc["id"],
                "scenario": sc["scenario"],
                "character": sc["character"],
                "expected_principle_keywords": sc["expected_principle_keywords"],
                "character_principle": getattr(pred, "character_principle", ""),
                "applied_insight":     getattr(pred, "applied_insight",     ""),
                "reasoning":           getattr(pred, "reasoning",           ""),
                "citations":           getattr(pred, "citations",           ""),
                "latency_ms":          round(latency_ms, 1),
            }
            print(f"    principle: {result['character_principle'][:100]}...")
            print(f"    insight:   {result['applied_insight'][:100]}...")
            print(f"    cites:     {result['citations']}")
        except Exception as exc:
            print(f"    FAILED: {exc}", file=sys.stderr)
            result = {
                "id": sc["id"],
                "scenario": sc["scenario"],
                "character": sc["character"],
                "error": str(exc),
            }
        results.append(result)

    latency_stats = _latency_summary(results)
    payload = {
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_latency_s": round(time.time() - t_total, 1),
        "model": "gemini/gemini-2.5-flash-lite",
        "corpus_note": (
            "Baseline against current RAG corpus (10 doc_ids in data/hp_lore.txt). "
            "Re-run after ingesting data/character_lore_tagged.jsonl to measure "
            "corpus-expansion lift."
        ),
        "latency": latency_stats,
        "scenarios": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    print()
    print(f"Saved {len(results)} results to {out_path}")
    print(f"Total wall time: {payload['total_latency_s']}s")
    if latency_stats.get("n"):
        print(
            f"Per-scenario latency (n={latency_stats['n']}): "
            f"p50 {latency_stats['p50_ms']:.0f}ms  "
            f"p95 {latency_stats['p95_ms']:.0f}ms  "
            f"p99 {latency_stats['p99_ms']:.0f}ms  "
            f"(mean {latency_stats['mean_ms']:.0f}ms, "
            f"min {latency_stats['min_ms']:.0f}ms, max {latency_stats['max_ms']:.0f}ms)"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("evals/perspective_shift_baseline.json"),
        help="output JSON path",
    )
    args = ap.parse_args(argv)
    return run(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
