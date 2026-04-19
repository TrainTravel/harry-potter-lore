"""
Compaction-threshold A/B experiment.
=====================================

Compares compaction behaviour at thresholds {6, 10, 12, 16, 20} over a
simulated 25-turn conversation.

No real LLM calls — uses a ``FakeSummarizer`` so results are deterministic
and free. This intentionally trades semantic-quality measurement for
structural measurement: we measure how the history-size curve evolves
and how often compaction fires at each threshold. Semantic-continuity
evaluation (does compaction preserve what matters?) is a separate
experiment that DOES need real LLM calls; not in this script.

Usage::

    .venv/bin/python -m evals.compaction_threshold_experiment
    .venv/bin/python -m evals.compaction_threshold_experiment --turns 30 --keep-recent 5

Outputs:
  - Printed comparison table
  - ``evals/compaction_threshold_results.json`` with per-threshold metrics

What we measure (per threshold, over N turns):
  - compaction_count — how many times compact_if_needed fired a real compaction
  - max_history_chars — peak formatted-history size across all turns
  - median_history_chars — typical size
  - summary_invocations — LLM calls made (= compaction_count)
  - bounded — did history plateau or keep growing?
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from context_harness.conversation import (
    ConversationStore,
    DEFAULT_KEEP_RECENT,
    Summarizer,
)


THRESHOLDS = [6, 10, 12, 16, 20]
DEFAULT_TURNS = 25
OUT_PATH = Path("evals/compaction_threshold_results.json")


# ---------------------------------------------------------------------------
# Synthetic fixtures — realistic-length turns + summaries, no LLM needed
# ---------------------------------------------------------------------------

# 120-ish character user messages + 400-ish character agent responses. The
# specific content doesn't matter for the structural measurement; what
# matters is the realistic size distribution.
_SYNTHETIC_USER_MESSAGES = [
    "What is a Horcrux and how does making one affect the wizard?",
    "So it splits your soul — is the split permanent or reversible?",
    "What happens if the Horcrux object is destroyed while the original wizard is still alive?",
    "Did Voldemort know how many he had made, or did he lose count?",
    "Could a witch or wizard make a Horcrux without committing murder?",
    "Is there any canonical example of a Horcrux being created without murder?",
    "Why is splitting the soul considered the darkest magic, morally?",
    "Does the corpus say anything about the ritual itself — is it documented?",
    "How did Horace Slughorn know about Horcruxes if the ritual is suppressed?",
    "Could Harry have made one? He was technically Voldemort's accidental Horcrux.",
    "What happened to the fragment inside Harry when Voldemort killed him?",
    "Did Dumbledore always know Harry carried a fragment?",
    "When did he tell Snape, and what did Snape think of that plan?",
    "Was there any way to remove the fragment without Harry dying?",
    "Could a person live indefinitely with a Horcrux if they protected it well enough?",
    "What's the longest any character in the corpus kept a Horcrux hidden?",
    "Would the Elder Wand have changed anything if Voldemort had used it to make one?",
    "Is the Resurrection Stone related to Horcruxes in any way?",
    "What's the Peverell connection again — remind me of the lineage.",
    "Okay, completely different angle — why did Dumbledore trust Snape?",
    "How did Dumbledore know Snape wasn't just fooling him?",
    "Did Snape ever know about Harry's Horcrux status?",
    "So Snape protecting Harry was Dumbledore's plan all along?",
    "What was Snape's emotional state during all this, based on the corpus?",
    "And he died knowing Lily's son had to face Voldemort — that's brutal.",
]

_SYNTHETIC_AGENT_RESPONSE = {
    "explanation": (
        "A horcrux is an object in which a wizard has hidden a fragment of "
        "their soul, created through an act of murder that tears the soul "
        "apart. The purpose is to anchor the wizard's life to the physical "
        "world even if their body is destroyed. The canon treats this as "
        "the darkest of dark magic — the violence is not incidental but "
        "structurally required by the ritual. This is a fairly rich "
        "explanation to simulate realistic per-turn agent output size."
    ),
}

_SYNTHETIC_SUMMARY = (
    "The student and tutor have been discussing horcruxes in depth, covering "
    "the soul-splitting mechanism, the moral framing of the murder requirement, "
    "the specific example of Voldemort's seven fragments (including the "
    "unintended Harry fragment), and Dumbledore's long-game plan involving "
    "Snape's protective role. The student is now pivoting toward Dumbledore's "
    "strategic reasoning and Snape's emotional state during these events."
)


class FakeSummarizer:
    def __init__(self):
        self.calls = 0

    def summarize(self, prompt: str) -> str:
        self.calls += 1
        return _SYNTHETIC_SUMMARY


# ---------------------------------------------------------------------------
# Run one experiment condition
# ---------------------------------------------------------------------------

def run_one_threshold(
    threshold: int,
    turns: int,
    keep_recent: int,
) -> dict[str, Any]:
    """Simulate ``turns`` turns with compaction at ``threshold``. Return
    per-turn measurements + aggregates."""
    store = ConversationStore(db_path=":memory:")
    summarizer = FakeSummarizer()

    conv_id = f"exp-thresh-{threshold}"
    per_turn = []

    for i in range(turns):
        user_msg = _SYNTHETIC_USER_MESSAGES[i % len(_SYNTHETIC_USER_MESSAGES)]
        store.save_turn(
            conv_id, user_msg, _SYNTHETIC_AGENT_RESPONSE,
            mode="guided_learning",
            tokens_in=500, tokens_out=200, cost_usd=0.0003,
        )
        # Real deploy uses BackgroundTasks. For the experiment we just call
        # compact synchronously to measure behaviour.
        store.compact_if_needed(
            conv_id, summarizer=summarizer,
            threshold=threshold, keep_recent=keep_recent,
        )
        # Measure formatted-history size as seen by the NEXT turn's LLM
        history = store.load_history(conv_id, max_turns=keep_recent)
        formatted = store.format_for_llm(history, mode="guided_learning")
        per_turn.append({
            "turn": i + 1,
            "history_chars": len(formatted),
            "verbatim_turns_in_history": len(history.turns),
            "has_summary": bool(history.summary),
        })

    char_counts = [p["history_chars"] for p in per_turn]
    return {
        "threshold": threshold,
        "turns_simulated": turns,
        "keep_recent": keep_recent,
        "compaction_count": summarizer.calls,
        "summary_invocations": summarizer.calls,
        "max_history_chars":    max(char_counts),
        "median_history_chars": int(statistics.median(char_counts)),
        "min_history_chars":    min(char_counts),
        "final_history_chars":  char_counts[-1],
        # Did history plateau or keep growing? Compare last quarter vs. first
        # quarter after the first compaction.
        "bounded_approx": _is_bounded(char_counts, threshold),
        "per_turn": per_turn,
    }


def _is_bounded(char_counts: list[int], threshold: int) -> bool:
    """Rough heuristic: a conversation is 'bounded' if, after the first
    compaction, history size never exceeds 1.3x its value at that point."""
    if len(char_counts) <= threshold:
        return True
    after_first = char_counts[threshold:]
    if not after_first:
        return True
    anchor = after_first[0]
    if anchor == 0:
        return True
    return max(after_first) <= anchor * 1.3


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(results: list[dict[str, Any]]) -> None:
    print()
    print("=" * 76)
    print("  Compaction threshold A/B — results")
    print("=" * 76)
    hdr = "threshold | compactions | max chars | median | final | bounded?"
    print(f"  {hdr}")
    print(f"  {'-' * len(hdr)}")
    for r in results:
        print(
            f"  {r['threshold']:>9d} | "
            f"{r['compaction_count']:>11d} | "
            f"{r['max_history_chars']:>9d} | "
            f"{r['median_history_chars']:>6d} | "
            f"{r['final_history_chars']:>5d} | "
            f"{'YES' if r['bounded_approx'] else 'NO':>7s}"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--turns", type=int, default=DEFAULT_TURNS,
                    help=f"turns per experiment condition (default: {DEFAULT_TURNS})")
    ap.add_argument("--keep-recent", type=int, default=DEFAULT_KEEP_RECENT,
                    help=f"turns kept verbatim (default: {DEFAULT_KEEP_RECENT})")
    ap.add_argument("--thresholds", type=str, default=",".join(str(t) for t in THRESHOLDS),
                    help="comma-separated thresholds to compare")
    ap.add_argument("--out", type=Path, default=OUT_PATH,
                    help=f"JSON output path (default: {OUT_PATH})")
    args = ap.parse_args(argv)

    thresholds = [int(t) for t in args.thresholds.split(",")]

    results = []
    for thresh in thresholds:
        print(f"  running threshold={thresh}...")
        r = run_one_threshold(thresh, args.turns, args.keep_recent)
        results.append(r)

    print_table(results)

    # Save full results (including per-turn breakdown) to JSON
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "turns": args.turns,
        "keep_recent": args.keep_recent,
        "results": results,
    }, indent=2))
    print(f"\nFull results: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
