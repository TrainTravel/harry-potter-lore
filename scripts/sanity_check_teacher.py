"""
Sanity-check the teacher LLM before recompiling
================================================
Runs the *uncompiled* module on a single trainset example and prints the
prediction. The point is to eyeball whether the configured LLM
(``DSPY_MODEL``) can reproduce your gold demos *without* compile.

If the prediction looks weaker than your hand-written ``character_response``
in the trainset, BootstrapFewShot will produce demos at *that* (worse)
quality — bake-in compile time. Switch to labeled-only:

    python -m context_harness.compile_agent --max-demos 0 --max-labeled <N>

If the prediction matches or beats your gold, bootstrap is fine.

Usage
-----
    python -m scripts.sanity_check_teacher --mode perspective_shift --idx 0
    python -m scripts.sanity_check_teacher --mode debate --idx 2

Cost: 1 LLM call (~$0.001 at flash-lite). Compare to ~$0.01–0.30 per
recompile + 3 minutes of waiting.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import dspy

sys.path.insert(0, str(Path(__file__).parent.parent))

from context_harness.ingest_lore import build_pipeline
from context_harness.dspy_agent import (
    DeepResearchModule,
    GuidedLearningModule,
    ExamGraderModule,
    DebateModule,
    SatiricalPodcastModule,
    PerspectiveShiftModule,
)


_MODE_TO_MODULE = {
    "deep_research":     (DeepResearchModule,     "trainset_deep_research"),
    "guided_learning":   (GuidedLearningModule,   "trainset_guided_learning"),
    "exam_grader":       (ExamGraderModule,       "trainset_exam_grader"),
    "debate":            (DebateModule,           "trainset_debate"),
    "satirical_podcast": (SatiricalPodcastModule, "trainset_satirical_podcast"),
    "perspective_shift": (PerspectiveShiftModule, "trainset_perspective_shift"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", required=True, choices=list(_MODE_TO_MODULE),
                    help="DSPy mode to test")
    ap.add_argument("--idx", type=int, default=0,
                    help="Index into the mode's TRAINSET")
    ap.add_argument("--model", default=os.getenv("DSPY_MODEL", "gemini/gemini-2.5-flash-lite"),
                    help="LLM to test as the teacher")
    args = ap.parse_args()

    print(f"Configuring dspy.LM(model={args.model!r})...")
    dspy.configure(lm=dspy.LM(args.model))

    module_cls, trainset_module = _MODE_TO_MODULE[args.mode]
    trainset_pkg = __import__(f"data.{trainset_module}", fromlist=["TRAINSET"])
    trainset = trainset_pkg.TRAINSET

    if args.idx >= len(trainset):
        print(f"Error: --idx {args.idx} out of range (trainset has {len(trainset)})")
        return 1

    example = trainset[args.idx]
    print(f"\nLoaded trainset[{args.idx}] from {trainset_module}.")
    print(f"Trainset size: {len(trainset)}")

    print("\nBuilding pipeline...")
    pipeline = build_pipeline(persist=True)

    # PerspectiveShiftModule needs a character_lore pipeline too — but the
    # module auto-discovers it if available, so plain construction is fine.
    module = module_cls(pipeline)

    inputs = example.inputs()
    inputs_dict = {k: getattr(example, k) for k in inputs.keys()}

    print("\n--- INPUTS ---")
    for k, v in inputs_dict.items():
        v_str = str(v)
        print(f"{k}: {v_str[:200]}{'...' if len(v_str) > 200 else ''}")

    print("\nCalling uncompiled module...")
    pred = module(**inputs_dict)

    print("\n--- TEACHER OUTPUT ---")
    for field in pred.keys():
        v = getattr(pred, field, "")
        v_str = str(v)
        print(f"\n[{field}]")
        print(v_str[:800])
        if len(v_str) > 800:
            print(f"... ({len(v_str) - 800} more chars)")

    # Show gold response if labeled, for direct comparison
    gold_fields = ("character_response", "answer", "transcript", "verdict")
    print("\n--- GOLD (from trainset, if labeled) ---")
    has_gold = False
    for field in gold_fields:
        gold = getattr(example, field, "") or ""
        if gold:
            has_gold = True
            print(f"\n[{field}]")
            print(str(gold)[:800])
    if not has_gold:
        print("(no labeled output fields found — this is an input-only example)")

    print("\nDone. Compare TEACHER OUTPUT vs GOLD: if teacher is weaker,")
    print("recompile with --max-demos 0 --max-labeled <N> to skip bootstrap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
