"""
Judge the A/A'/B outputs from ab_perspective_shift_fast_results.md.

For each scenario, hand the three responses to a judge LLM and ask it
to pick a winner on four axes:

  1. Character voice authenticity
  2. Canon anchor specificity (does the response cite a specific
     canon event, not a generic trait list?)
  3. Actionability (does it give a concrete this-week step?)
  4. Rule-following (no greeting, no character re-introduction, no
     conversation reset on terse input)

Judge model: gemini-2.5-flash (one notch above flash-lite — sharper
discrimination without being slow). One call per scenario; 30 calls
total at ~$0.005.

The judge returns JSON. We aggregate into per-axis vote counts:
  {voice: {A: 12, A': 7, B: 11, tie: 0}, ...}

Usage::

    .venv/bin/python -m scripts.judge_abc_outputs

Reads:  drafts/ab_perspective_shift_fast_results.md
Writes: drafts/abc_judge_verdict.md
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import litellm


JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini/gemini-2.5-flash")
RESULTS_PATH = Path("drafts/ab_perspective_shift_fast_results.md")
VERDICT_PATH = Path("drafts/abc_judge_verdict.md")


JUDGE_PROMPT = """You are evaluating three Harry Potter character responses to the same user scenario. The character should speak in first person, drawing on specific canon events.

SCENARIO ({character}):
{question}

--- RESPONSE A ---
{a}

--- RESPONSE A' ---
{ap}

--- RESPONSE B ---
{b}

Judge each response on these 4 axes. For each axis, pick the BEST response (A, A', or B) — or "tie" only if truly indistinguishable.

1. **voice**: Which response most authentically captures this character's voice (cadence, vocabulary, emotional register)?
2. **canon_anchor**: Which most specifically grounds itself in a canon event from the books (not a generic trait list)?
3. **actionability**: Which gives the user the most concrete, actionable step they could take this week (not vague reassurance)?
4. **rule_following**: Which best follows the rules — NO opening greeting ("hello", "ah, dear"), NO re-introducing the character ("As Luna, I..."), NO summary/closing platitudes? (Reasonable mid-response interjections are fine.)

Return ONLY a single JSON object, no prose:

{{"voice": "A|A'|B|tie", "canon_anchor": "A|A'|B|tie", "actionability": "A|A'|B|tie", "rule_following": "A|A'|B|tie", "one_line_reason": "<your sharpest observation>"}}"""


def parse_scenarios(md: str) -> list[dict]:
    """Extract (character, question, A, A', B) tuples from the markdown."""
    # Section starts after the first `---` divider following the header
    sections = re.split(r"\n## ", md)[1:]  # first chunk is header
    out: list[dict] = []
    for sec in sections:
        # Header line: "scenario_id — Character Name"
        header_match = re.match(r"([^\n]+)", sec)
        if not header_match:
            continue
        header = header_match.group(1)
        char = header.split("—", 1)[1].strip() if "—" in header else "Unknown"

        q_match = re.search(r"\*\*Q:\*\* (.+?)\n", sec)
        if not q_match:
            continue
        question = q_match.group(1).strip()

        # Each variant block: "### X · Label  (`...s`, `... chars`)\n\n<body>\n\n### Y" or "---"
        def extract(label: str) -> str:
            pat = rf"### {re.escape(label)}.*?\n\n(.+?)(?=\n### |\n---)"
            m = re.search(pat, sec, re.S)
            return m.group(1).strip() if m else ""

        a  = extract("A · Current")
        ap = extract("A' · CoT-drop")
        b  = extract("B · Fast")
        if not (a and ap and b):
            continue
        out.append({
            "id": header.split("—", 1)[0].strip(),
            "character": char,
            "question": question,
            "a": a, "ap": ap, "b": b,
        })
    return out


def judge_one(scen: dict) -> dict:
    """Call the judge once. Returns {voice, canon_anchor, actionability,
    rule_following, one_line_reason}."""
    prompt = JUDGE_PROMPT.format(
        character=scen["character"],
        question=scen["question"],
        a=scen["a"], ap=scen["ap"], b=scen["b"],
    )
    resp = litellm.completion(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000,
        temperature=0,
    )
    raw = resp.choices[0].message.content.strip()
    # Strip code fence if present
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.M).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Last-ditch: extract first {...} block
        m = re.search(r"\{[^{}]+\}", raw, re.S)
        if m:
            return json.loads(m.group(0))
        return {"voice": "tie", "canon_anchor": "tie", "actionability": "tie",
                "rule_following": "tie", "one_line_reason": f"PARSE_FAIL: {raw[:120]}"}


def main() -> int:
    md = RESULTS_PATH.read_text()
    scenarios = parse_scenarios(md)
    print(f"Parsed {len(scenarios)} scenarios from {RESULTS_PATH}")
    print(f"Judging with: {JUDGE_MODEL}\n")

    judgments: list[dict] = []
    for i, scen in enumerate(scenarios, 1):
        t0 = time.perf_counter()
        try:
            j = judge_one(scen)
        except Exception as e:
            print(f"  [{i}/{len(scenarios)}] {scen['id']:32s} JUDGE ERROR: {e}")
            j = {"voice": "tie", "canon_anchor": "tie", "actionability": "tie",
                 "rule_following": "tie", "one_line_reason": f"ERROR: {e}"}
        dt = time.perf_counter() - t0
        print(f"  [{i}/{len(scenarios)}] {scen['id']:32s} ({dt:.1f}s) "
              f"v={j.get('voice','?'):4s} c={j.get('canon_anchor','?'):4s} "
              f"a={j.get('actionability','?'):4s} r={j.get('rule_following','?'):4s}")
        judgments.append({"scenario": scen["id"], "character": scen["character"], **j})

    # Aggregate
    axes = ["voice", "canon_anchor", "actionability", "rule_following"]
    tally: dict[str, dict[str, int]] = {ax: {"A": 0, "A'": 0, "B": 0, "tie": 0} for ax in axes}
    for j in judgments:
        for ax in axes:
            tally[ax][j.get(ax, "tie")] = tally[ax].get(j.get(ax, "tie"), 0) + 1

    # Print summary
    print("\n=== AGGREGATE ===")
    print(f"{'axis':<18s} {'A':>5s} {'Ap':>5s} {'B':>5s} {'tie':>5s}")
    for ax in axes:
        t = tally[ax]
        a_n  = t["A"]
        ap_n = t["A'"]
        b_n  = t["B"]
        tie_n = t["tie"]
        print(f"{ax:<18s} {a_n:>5d} {ap_n:>5d} {b_n:>5d} {tie_n:>5d}")

    # Write verdict markdown
    VERDICT_PATH.parent.mkdir(exist_ok=True)
    with VERDICT_PATH.open("w") as f:
        f.write("# A/A'/B Judge Verdict\n\n")
        f.write(f"**Judge:** `{JUDGE_MODEL}`\n")
        f.write(f"**Scenarios:** {len(judgments)}\n")
        f.write(f"**Run at:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Aggregate wins per axis\n\n")
        f.write("| Axis | A (current) | A' (CoT-drop) | B (Fast) | tie |\n")
        f.write("|---|---|---|---|---|\n")
        for ax in axes:
            t = tally[ax]
            a_n, ap_n, b_n, tie_n = t["A"], t["A'"], t["B"], t["tie"]
            f.write(f"| {ax} | {a_n} | {ap_n} | {b_n} | {tie_n} |\n")
        f.write("\n## Per-scenario verdicts\n\n")
        f.write("| Scenario | Character | voice | canon | action | rules | judge note |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for j in judgments:
            note = (j.get("one_line_reason", "") or "").replace("|", "\\|").replace("\n", " ")
            f.write(f"| {j['scenario']} | {j['character']} | "
                    f"{j.get('voice','?')} | {j.get('canon_anchor','?')} | "
                    f"{j.get('actionability','?')} | {j.get('rule_following','?')} | "
                    f"{note[:200]} |\n")

    print(f"\nVerdict written to {VERDICT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
