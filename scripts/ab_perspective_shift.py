"""
A/B test: DSPy-compiled perspective_shift vs simple system-prompt
==================================================================
Runs the same scenarios through two paths:

  A. **DSPy compiled** — existing ``mode=perspective_shift`` via /ask endpoint.
     Uses the trainset-compiled artifact in ``my_profile.agent/perspective_shift.json``.

  B. **Simple system-prompt** — direct litellm call with a 3-line system prompt.
     No DSPy, no demos, no compile, no metric. The pretrained model carries
     the character knowledge.

Both sides use the same runtime LLM (set via ``DSPY_MODEL`` env var). The point
is to test whether the DSPy machinery is earning its keep, *not* whether one
model beats another.

Output: prints both responses side-by-side per scenario, plus saves to
``drafts/ab_perspective_shift_results.md`` for review.

Usage::

    # Start the local server (Pro recommended) in another terminal:
    DSPY_MODEL=gemini/gemini-2.5-pro .venv/bin/python -m uvicorn api.main:app --port 8000

    # Run the A/B:
    .venv/bin/python -m scripts.ab_perspective_shift

Cost: ~$0.10-0.40 total at Pro pricing (8 scenarios × 2 LLM calls each).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import litellm
import requests


SERVER_URL = "http://127.0.0.1:8000/ask"
MODEL = os.getenv("DSPY_MODEL", "gemini/gemini-2.5-flash-lite")

# Strip provider prefix for litellm (e.g. "gemini/gemini-2.5-pro" → litellm
# already accepts that form, but just in case).
LITELLM_MODEL = MODEL


# Single-turn scenarios. Diverse coverage: career, grief, identity, family,
# friendship, sorting, self-doubt, sorting-hat-style.
SCENARIOS: list[dict] = [
    {
        "id": "career_change",
        "character": "Dumbledore",
        "question": "I'm 32, working a stable tech job that pays well but bores me. I keep fantasising about quitting to write full-time. I can't tell if it's a real calling or just escapism.",
    },
    {
        "id": "grief_anticipatory",
        "character": "Harry Potter",
        "question": "My mother has early-onset dementia and already doesn't recognise me some days. I'm mourning someone who's still alive and no one has language for that.",
    },
    {
        "id": "identity_lesbian_love",
        "character": "Dumbledore",
        "question": "I want to know how, as a lesbian, I can navigate the love world with more confidence.",
    },
    {
        "id": "family_pressure",
        "character": "Neville Longbottom",
        "question": "My parents want me to take over the family business. I'd rather go into nursing. Every dinner has become a rehearsal of the same unspoken argument.",
    },
    {
        "id": "friendship_left_behind",
        "character": "Ron Weasley",
        "question": "My closest friend is leaving our shared city for a better job. I'm happy for them and also furious, and the guilt of that fury is eating me.",
    },
    {
        "id": "imposter_syndrome",
        "character": "Hermione Granger",
        "question": "I got into a top-tier graduate program and I've been awake for 48 hours convinced they'll rescind the offer when they realise the mistake.",
    },
    {
        "id": "boundary_setting",
        "character": "Minerva McGonagall",
        "question": "My manager publicly mocks my ideas in meetings and then pitches them upward as his own two weeks later. HR knows. Nothing happens.",
    },
    {
        "id": "sorting_hat_quiz",
        "character": "Sorting Hat",
        "question": "Sort me into a Hogwarts house.",
    },
]


SIMPLE_SYSTEM_PROMPT = (
    "You are {character}, the Harry Potter character. The user has shared a "
    "real-life situation. Respond as {character} would — in their characteristic "
    "voice, drawing on their canonical experiences in the books, and offering "
    "concrete reflection or advice grounded in who they are. 100–250 words. "
    "Specific, not generic. Do not break character or refer to yourself as an AI."
)


def _via_dspy(scenario: dict) -> dict:
    """Hit the local server's existing perspective_shift mode."""
    t0 = time.perf_counter()
    r = requests.post(SERVER_URL, json={
        "question": scenario["question"],
        "mode": "perspective_shift",
        "character": scenario["character"],
    }, timeout=120)
    r.raise_for_status()
    body = r.json()
    return {
        "answer": body.get("answer", ""),
        "latency_s": time.perf_counter() - t0,
        "cost_usd": body.get("cost_usd", 0),
    }


def _via_simple(scenario: dict) -> dict:
    """Direct litellm call with a 3-line system prompt."""
    t0 = time.perf_counter()
    completion = litellm.completion(
        model=LITELLM_MODEL,
        messages=[
            {"role": "system",
             "content": SIMPLE_SYSTEM_PROMPT.format(character=scenario["character"])},
            {"role": "user", "content": scenario["question"]},
        ],
        # Gemini 2.5 Pro reserves ~half of max_tokens for internal reasoning
        # before producing text output. 600 was burned entirely on reasoning
        # → 0-char responses. 4000 gives room for both thinking + ~250 word
        # answers.
        max_tokens=4000,
    )
    answer = completion.choices[0].message.content or ""
    usage = completion.usage if hasattr(completion, "usage") else None
    cost = 0.0
    if usage:
        # Approximate Pro pricing: $1.25/M in, $10/M out. Flash-lite: $0.10/$0.40.
        in_per_m = 1.25 if "pro" in MODEL.lower() else 0.10
        out_per_m = 10.00 if "pro" in MODEL.lower() else 0.40
        cost = (usage.prompt_tokens / 1_000_000) * in_per_m + \
               (usage.completion_tokens / 1_000_000) * out_per_m
    return {
        "answer": answer,
        "latency_s": time.perf_counter() - t0,
        "cost_usd": cost,
    }


def main() -> int:
    print(f"Running A/B against MODEL={MODEL}")
    print(f"Scenarios: {len(SCENARIOS)}\n")

    results: list[dict] = []

    for i, scen in enumerate(SCENARIOS, 1):
        print(f"--- [{i}/{len(SCENARIOS)}] {scen['id']} ({scen['character']}) ---")
        print(f"Q: {scen['question'][:100]}{'...' if len(scen['question']) > 100 else ''}")

        try:
            a = _via_dspy(scen)
        except Exception as e:
            print(f"  A (DSPy)   : FAILED — {e}")
            a = {"answer": f"ERROR: {e}", "latency_s": 0, "cost_usd": 0}

        try:
            b = _via_simple(scen)
        except Exception as e:
            print(f"  B (Simple) : FAILED — {e}")
            b = {"answer": f"ERROR: {e}", "latency_s": 0, "cost_usd": 0}

        print(f"  A (DSPy)   : {a['latency_s']:.1f}s  ${a['cost_usd']:.4f}  "
              f"[{len(a['answer'])} chars]")
        print(f"  B (Simple) : {b['latency_s']:.1f}s  ${b['cost_usd']:.4f}  "
              f"[{len(b['answer'])} chars]\n")

        results.append({
            "scenario": scen,
            "dspy": a,
            "simple": b,
        })

    # Save side-by-side markdown for review
    out_path = Path("drafts/ab_perspective_shift_results.md")
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w") as f:
        f.write(f"# A/B: DSPy compiled vs simple system-prompt\n\n")
        f.write(f"**Model:** `{MODEL}`\n")
        f.write(f"**Run at:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Both paths use the same runtime LLM. A goes through the "
                f"DSPy-compiled `perspective_shift` artifact (with labeled demos "
                f"in the prompt). B is a direct litellm call with a 3-line system "
                f"prompt — no DSPy, no demos.\n\n")
        f.write(f"---\n\n")
        for r in results:
            scen = r["scenario"]
            f.write(f"## {scen['id']} — {scen['character']}\n\n")
            f.write(f"**Q:** {scen['question']}\n\n")
            f.write(f"### A · DSPy compiled  "
                    f"(`{r['dspy']['latency_s']:.1f}s`, "
                    f"`${r['dspy']['cost_usd']:.4f}`, "
                    f"`{len(r['dspy']['answer'])} chars`)\n\n")
            f.write(r["dspy"]["answer"] + "\n\n")
            f.write(f"### B · Simple system-prompt  "
                    f"(`{r['simple']['latency_s']:.1f}s`, "
                    f"`${r['simple']['cost_usd']:.4f}`, "
                    f"`{len(r['simple']['answer'])} chars`)\n\n")
            f.write(r["simple"]["answer"] + "\n\n")
            f.write(f"---\n\n")

    print(f"\nSaved side-by-side markdown to {out_path}")
    total_a_cost = sum(r["dspy"]["cost_usd"] for r in results)
    total_b_cost = sum(r["simple"]["cost_usd"] for r in results)
    total_a_chars = sum(len(r["dspy"]["answer"]) for r in results)
    total_b_chars = sum(len(r["simple"]["answer"]) for r in results)
    print(f"Totals — A (DSPy): ${total_a_cost:.4f}, {total_a_chars} chars")
    print(f"Totals — B (Simple): ${total_b_cost:.4f}, {total_b_chars} chars")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except requests.RequestException as e:
        print(f"\nFAILED — could not reach {SERVER_URL}: {e}")
        print("Is the server running? `uvicorn api.main:app --port 8000`")
        sys.exit(2)
