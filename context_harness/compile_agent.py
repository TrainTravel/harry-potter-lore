"""
Compile the DSPy agent from the training sets
==============================================
Runs BootstrapFewShot on both modes using the hand-labelled examples in
data/trainset_*.py and the metric functions in context_harness/metrics.py.
Saves the optimized JSON artifacts + manifest to a .agent directory.

Usage:
    python -m context_harness.compile_agent
    python -m context_harness.compile_agent --out my_profile.agent --model openai/gpt-4o-mini

Prereqs:
    - ChromaDB is populated (run `python -m context_harness.ingest_lore` first)
    - An OpenAI/Anthropic API key in the environment, OR pass --model with
      a configured dspy LM
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import dspy
from dspy.teleprompt import BootstrapFewShot

from .ingest_lore import build_pipeline
from .dspy_agent import DSPyAgent
from .metrics import (
    deep_research_metric,
    socratic_metric,
    exam_grader_metric,
    debate_metric,
    satirical_podcast_metric,
    perspective_shift_metric,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Compile the DSPy agent.")
    ap.add_argument("--out", default="my_profile.agent",
                    help="output directory for compiled JSONs + manifest")
    ap.add_argument("--model", default="openai/gpt-4o-mini",
                    help="dspy.LM model string, e.g. 'openai/gpt-4o-mini'")
    ap.add_argument("--max-demos", type=int, default=4,
                    help="max bootstrapped demonstrations per mode")
    ap.add_argument("--max-labeled", type=int, default=8,
                    help="max labeled demonstrations per mode")
    args = ap.parse_args()

    print(f"Configuring dspy.LM(model={args.model!r})...")
    dspy.configure(lm=dspy.LM(args.model))

    print("Loading training sets...")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.trainset_deep_research import TRAINSET as research_trainset
    from data.trainset_guided_learning import TRAINSET as learning_trainset
    from data.trainset_exam_grader import TRAINSET as grader_trainset
    from data.trainset_debate import TRAINSET as debate_trainset
    from data.trainset_satirical_podcast import TRAINSET as podcast_trainset
    from data.trainset_perspective_shift import TRAINSET as pshift_trainset

    print("Building ChromaDB-backed pipeline...")
    pipeline = build_pipeline(persist=True)

    agent = DSPyAgent(pipeline)

    # ----- Compile deep_research -----
    print(f"\n[1/5] Compiling deep_research ({len(research_trainset)} examples)...")
    research_optimizer = BootstrapFewShot(
        metric=deep_research_metric,
        max_bootstrapped_demos=args.max_demos,
        max_labeled_demos=args.max_labeled,
    )
    agent.compile_deep_research(research_optimizer, trainset=research_trainset)
    print("  ✓ deep_research compiled")

    # ----- Compile guided_learning -----
    print(f"\n[2/5] Compiling guided_learning ({len(learning_trainset)} examples)...")
    learning_optimizer = BootstrapFewShot(
        metric=socratic_metric,
        max_bootstrapped_demos=args.max_demos,
        max_labeled_demos=args.max_labeled,
    )
    agent.compile_guided_learning(learning_optimizer, trainset=learning_trainset)
    print("  ✓ guided_learning compiled")

    # ----- Compile exam_grader -----
    print(f"\n[3/5] Compiling exam_grader ({len(grader_trainset)} examples)...")
    grader_optimizer = BootstrapFewShot(
        metric=exam_grader_metric,
        max_bootstrapped_demos=args.max_demos,
        max_labeled_demos=args.max_labeled,
    )
    agent.compile_exam_grader(grader_optimizer, trainset=grader_trainset)
    print("  ✓ exam_grader compiled")

    # ----- Compile debate -----
    print(f"\n[4/5] Compiling debate ({len(debate_trainset)} examples)...")
    debate_optimizer = BootstrapFewShot(
        metric=debate_metric,
        max_bootstrapped_demos=args.max_demos,
        max_labeled_demos=args.max_labeled,
    )
    agent.compile_debate(debate_optimizer, trainset=debate_trainset)
    print("  ✓ debate compiled")

    # ----- Compile satirical_podcast -----
    print(f"\n[5/6] Compiling satirical_podcast ({len(podcast_trainset)} examples)...")
    podcast_optimizer = BootstrapFewShot(
        metric=satirical_podcast_metric,
        max_bootstrapped_demos=args.max_demos,
        max_labeled_demos=args.max_labeled,
    )
    agent.compile_satirical_podcast(podcast_optimizer, trainset=podcast_trainset)
    print("  ✓ satirical_podcast compiled")

    # ----- Compile perspective_shift -----
    # Trainset includes multi-turn examples (chat_history populated). Metric
    # enforces: principle/insight/reasoning length + character_response
    # substance + citations + NO greeting opener when chat_history is
    # non-empty. Together this produces demos that teach authentic voice
    # + continuation discipline on turn 2+.
    print(f"\n[6/6] Compiling perspective_shift ({len(pshift_trainset)} examples)...")
    pshift_optimizer = BootstrapFewShot(
        metric=perspective_shift_metric,
        max_bootstrapped_demos=args.max_demos,
        max_labeled_demos=args.max_labeled,
    )
    agent.compile_perspective_shift(pshift_optimizer, trainset=pshift_trainset)
    print("  ✓ perspective_shift compiled")

    # ----- Save -----
    print(f"\nSaving compiled agent to {args.out}/ ...")
    agent.save(args.out)
    print("  ✓ deep_research.json")
    print("  ✓ guided_learning.json")
    print("  ✓ exam_grader.json")
    print("  ✓ debate.json")
    print("  ✓ satirical_podcast.json")
    print("  ✓ perspective_shift.json")
    print("  ✓ manifest.json")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
