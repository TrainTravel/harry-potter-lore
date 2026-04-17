"""
Sample tagged chunks for human review.
======================================
Stratified random sample — 2–3 chunks per character — written as a single
Markdown file you can open in your editor.

For each chunk you see:
  - the full text
  - the Gemini-assigned tags
  - a checkbox for your verdict

Review workflow:
  1. Run this script → `data/tag_review.md` is created.
  2. Open it in your editor. For each chunk, tick one checkbox (agree /
     partial / disagree). If disagree, write what you'd tag instead.
  3. Send the list of chunk_ids you disagreed with back to the assistant.
     We compute agreement rate and look for systematic errors.

Usage::

    .venv/bin/python -m scripts.sample_for_tag_review
    .venv/bin/python -m scripts.sample_for_tag_review --per-character 3 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


IN_PATH = Path("data/character_lore_tagged.jsonl")
OUT_PATH = Path("data/tag_review.md")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-path",       type=Path, default=IN_PATH)
    ap.add_argument("--out-path",      type=Path, default=OUT_PATH)
    ap.add_argument("--per-character", type=int, default=3)
    ap.add_argument("--seed",          type=int, default=20260417)
    args = ap.parse_args(argv)

    chunks = [json.loads(line) for line in args.in_path.open(encoding="utf-8")]
    # Skip factual-only chunks from the review sample — we already know
    # those are being filtered downstream.
    non_factual = [c for c in chunks if c["themes"] and c["themes"] != ["factual"]]

    by_character: dict[str, list[dict]] = defaultdict(list)
    for c in non_factual:
        by_character[c["character"]].append(c)

    rng = random.Random(args.seed)
    sample: list[dict] = []
    for character, group in sorted(by_character.items()):
        k = min(args.per_character, len(group))
        sample.extend(rng.sample(group, k))

    # Write markdown
    lines: list[str] = []
    lines.append(f"# Tag review — {len(sample)} chunks")
    lines.append("")
    lines.append(
        "For each chunk: tick one box. If you disagree, write your tags "
        "under **Your tags**."
    )
    lines.append("")
    lines.append(f"Source: `{args.in_path}` (sample seed={args.seed})")
    lines.append("")
    lines.append("---")
    lines.append("")

    for i, chunk in enumerate(sample, start=1):
        section = " > ".join(chunk["section_path"]) or "(top)"
        tags = ", ".join(chunk["themes"])
        lines.append(f"## {i}. `{chunk['chunk_id']}`")
        lines.append("")
        lines.append(f"- **Character:** {chunk['character']}")
        lines.append(f"- **Section:** {section}")
        lines.append(f"- **Gemini tags:** `{tags}`")
        lines.append("")
        lines.append("**Your verdict:**")
        lines.append("")
        lines.append("- [ ] agree")
        lines.append("- [ ] partial (some tags right, some wrong)")
        lines.append("- [ ] disagree (tags miss the point)")
        lines.append("")
        lines.append("**Your tags (if different):** ")
        lines.append("")
        lines.append("**Passage:**")
        lines.append("")
        lines.append("> " + chunk["text"].replace("\n", "\n> "))
        lines.append("")
        lines.append("---")
        lines.append("")

    args.out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(sample)} chunks to {args.out_path}")
    print()
    print("Next steps:")
    print(f"  1. Open {args.out_path} in your editor.")
    print("  2. For each chunk, tick one checkbox. If disagree, write your tags.")
    print("  3. Send the list of chunk_ids you marked disagree/partial back.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
