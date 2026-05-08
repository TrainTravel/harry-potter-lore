"""
Recompile the intent_router only — preserves other compiled modes.

The full ``context_harness.compile_agent`` recompiles all seven modes,
which can disturb the labeled-only ``perspective_shift`` demos (see
CLAUDE.md "Bootstrap vs labeled"). When you've only edited
``data/trainset_intent_router.py``, run this instead — it loads the
existing compiled agent, recompiles just the router, and saves the
agent (other modes' JSONs are unchanged byte-for-byte if no other
state was touched).

Usage::

    .venv/bin/python -m scripts.compile_intent_router_only \\
        --model gemini/gemini-2.5-flash-lite \\
        --max-demos 4 --max-labeled 8

Prereqs:
    - ``my_profile.agent/`` exists (from a prior full compile).
    - ``GOOGLE_API_KEY`` (or appropriate provider key) in env.
    - ``--max-demos 0`` enables labeled-only (skip bootstrap teacher).
"""

from __future__ import annotations

import argparse
import sys

import dspy
from dspy.teleprompt import BootstrapFewShot

from context_harness.dspy_agent import DSPyAgent
from context_harness.ingest_lore import build_pipeline
from context_harness.metrics import intent_router_metric
from data.trainset_intent_router import TRAINSET as router_trainset


def main() -> int:
    ap = argparse.ArgumentParser(description="Recompile intent_router only.")
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
    # Constructing with export_dir auto-loads the existing compiled state.
    agent = DSPyAgent(pipeline, export_dir=args.profile)

    print(f"\nRecompiling intent_router ({len(router_trainset)} examples)...")
    optimizer = BootstrapFewShot(
        metric=intent_router_metric,
        max_bootstrapped_demos=args.max_demos,
        max_labeled_demos=args.max_labeled,
    )
    agent.compile_intent_router(optimizer, trainset=router_trainset)
    print("  ✓ intent_router compiled")

    print(f"\nSaving agent to {args.profile}/ ...")
    agent.save(args.profile)
    print("  ✓ intent_router.json")
    print("\nDone. Diff the profile to confirm only intent_router.json changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
