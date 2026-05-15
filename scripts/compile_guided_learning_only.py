"""
Recompile the guided_learning module only — preserves other compiled modes.

When you've only edited ``data/trainset_guided_learning.py`` or the
``GuidedLearningSignature``, run this instead of the full compile. The
full compile recompiles all seven modes, which can disturb carefully-
tuned demos elsewhere (especially perspective_shift, which uses
labeled-only per CLAUDE.md "Bootstrap vs labeled").

Usage::

    .venv/bin/python -m scripts.compile_guided_learning_only \\
        --model gemini/gemini-2.5-flash-lite \\
        --max-demos 4 --max-labeled 8

Prereqs:
    - ``my_profile.agent/`` exists (from a prior full compile).
    - ``GOOGLE_API_KEY`` (or appropriate provider key) in env.
    - ``--max-demos 0`` enables labeled-only (skip bootstrap teacher).
"""

from __future__ import annotations

import argparse

import dspy
from dspy.teleprompt import BootstrapFewShot

from context_harness.dspy_agent import DSPyAgent
from context_harness.ingest_lore import build_pipeline
from context_harness.metrics import socratic_metric
from data.trainset_guided_learning import TRAINSET as learning_trainset


def main() -> int:
    ap = argparse.ArgumentParser(description="Recompile guided_learning only.")
    ap.add_argument("--profile", default="my_profile.agent",
                    help="profile directory to load + save (default: %(default)s)")
    ap.add_argument("--model", default="gemini/gemini-2.5-flash-lite",
                    help="dspy.LM model string for the teacher")
    ap.add_argument("--max-demos", type=int, default=4,
                    help="max bootstrapped demonstrations (0 = labeled-only)")
    ap.add_argument("--max-labeled", type=int, default=8,
                    help="max labeled demonstrations")
    ap.add_argument("--collection-name", default="hp_lore",
                    help="ChromaDB collection name (matches runtime)")
    args = ap.parse_args()

    print(f"Configuring dspy.LM(model={args.model!r})...")
    dspy.configure(lm=dspy.LM(args.model))

    print(f"Building pipeline + agent for collection {args.collection_name!r}...")
    pipeline = build_pipeline(persist=True, collection_name=args.collection_name)
    agent = DSPyAgent(pipeline, export_dir=args.profile)

    print(f"\nRecompiling guided_learning ({len(learning_trainset)} examples)...")
    optimizer = BootstrapFewShot(
        metric=socratic_metric,
        max_bootstrapped_demos=args.max_demos,
        max_labeled_demos=args.max_labeled,
    )
    agent.compile_guided_learning(optimizer, trainset=learning_trainset)
    print("  ✓ guided_learning compiled")

    print(f"\nSaving agent to {args.profile}/ ...")
    agent.save(args.profile)
    print("  ✓ guided_learning.json")
    print("\nDone. Diff the profile to confirm only guided_learning.json changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
