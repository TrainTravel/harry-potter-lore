"""
Auto-tag character-lore chunks with life-theme tags using Gemini.
================================================================
Reads ``data/character_lore.jsonl`` (from chunk_character_lore.py) and
asks Gemini 2.5 Flash-lite to assign 2–4 life-theme tags per chunk,
picked from a fixed controlled vocabulary.

Writes ``data/character_lore_tagged.jsonl`` — same schema, with
``themes`` populated.

Batches 10 chunks per LLM call for cost efficiency (≈ $0.04 for all ~900
chunks at Gemini 2.5 Flash-lite pricing).

Idempotent: if the output file exists, it resumes from the last fully
written chunk. Safe to interrupt with Ctrl-C.

Usage::

    .venv/bin/python -m scripts.tag_chunks_with_themes
    .venv/bin/python -m scripts.tag_chunks_with_themes --batch-size 10 --limit 50
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types


IN_PATH = Path("data/character_lore.jsonl")
OUT_PATH = Path("data/character_lore_tagged.jsonl")

MODEL = "gemini-2.5-flash-lite"

# Controlled vocabulary — authored by hand to cover the emotional/psychological
# territory our perspective_shift mode needs. Keep ~25; more is overwhelming
# and dilutes the tags' usefulness as retrieval metadata.
VOCABULARY = [
    "grief", "duty", "identity", "ambition", "loyalty",
    "sacrifice", "impostor-syndrome", "leadership", "rebellion",
    "friendship", "redemption", "fame", "isolation", "mentorship",
    "perseverance", "moral-ambiguity", "unrequited-love",
    "fear-of-death", "late-bloomer", "class-anxiety", "prejudice",
    "family", "courage", "discipline", "forgiveness",
    # Added 2026-04-17 after v1 human review — gaps in interpersonal themes:
    "betrayal", "protection", "trust", "manipulation", "trauma",
    "regret", "cruelty", "disillusionment", "jealousy", "double-life",
    # Added 2026-04-17 after v2 subagent review — remaining gaps:
    "legacy", "nonconformity", "humiliation",
]

PROMPT_TEMPLATE = """You are a literary analyst tagging passages about Harry Potter characters.

For each numbered passage, assign 2–4 "life-theme" tags that capture the
emotional, psychological, or ethical territory the passage explores.

Rules:
- Use ONLY tags from the controlled vocabulary below.
- 2–4 tags per passage. Prefer 2–3 unless the passage is rich and layered.
- Pick tags that reflect the **theme**, not the surface plot or skill set.
- Do NOT reach for stereotype tags ("leadership" for anyone who commands
  followers; "mentorship" for anyone who teaches). Look at WHAT the passage
  is really about emotionally/ethically.
- For antagonist passages, pick the dark theme directly — `manipulation`,
  `cruelty`, `betrayal`, `fear-of-death` — rather than neutral words.
- `friendship` is often too generic; if the bond hinges on being believed
  or relied on, prefer `trust`; if familial-bond-like, prefer `family`.
- `duty` is a dumping ground — if the passage has a more specific theme
  (`protection`, `legacy`, `leadership`, `double-life`), prefer that.
- Sustained deception with moral cause (Snape, Regulus, Dumbledore-Grindelwald
  era) → `double-life` + `manipulation`.
- Luna-style gentle refusal to conform → `nonconformity` (NOT `rebellion`,
  which implies political stance).
- Neville-Howler, Ron-mocked-by-Malfoy scenes → `humiliation` fits better
  than `trauma` (too clinical) or `class-anxiety` (too narrow).
- If a passage is purely factual (dates, item lists, dry exposition), return
  an EMPTY array []. We filter empty tags downstream. Do NOT invent tags
  outside the vocabulary.

Controlled vocabulary:
{vocab}

### Examples of good tagging

Passage: "Tom Riddle charmed the professors at Hogwarts while secretly assembling
followers and investigating the school's darkest secrets, hiding his true purposes
behind a model-student facade."
Tags: ["manipulation", "ambition", "double-life"]
(NOT "leadership" — he is recruiting but the theme is deception, not leading.)

Passage: "Hermione spent weeks in the library cross-referencing texts until she
discovered the Basilisk's true identity, then left a single torn page behind so
Harry and Ron could find it before the worst happened."
Tags: ["perseverance", "loyalty", "protection"]
(NOT "research" — research is descriptive. Protecting her friends is the theme.)

Passage: "After Harry lost Sirius, Hermione didn't try to fix anything or offer
easy comfort; she sat with him for hours, asking nothing, simply present."
Tags: ["trust", "grief", "friendship"]
(NOT just "friendship" — the bond is specifically about being a safe presence,
i.e. trust.)

Passage: "Snape reported Voldemort's movements to Dumbledore year after year,
while publicly appearing as a loyal Death Eater to the Dark Lord. Every
conversation he had with either side carried the weight of the other."
Tags: ["double-life", "manipulation", "duty"]
(NOT "loyalty" — loyalty is the surface, the theme is sustained deception
with cause. `duty` stays because the deception is in service of an obligation.)

### Now tag these:

{passages}

Return a single JSON object mapping passage number (as string) to an array of tags.
Example: {{"1": ["grief", "family"], "2": ["impostor-syndrome", "perseverance", "late-bloomer"]}}

Return ONLY the JSON. No prose, no markdown fences."""


def _load_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key:
        return key
    # Fall back to .env
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            m = re.match(r'^\s*(?:export\s+)?(GEMINI_API_KEY|GOOGLE_API_KEY)=(.+)$', line)
            if m:
                return m.group(2).strip().strip('"').strip("'")
    raise RuntimeError(
        "No Gemini API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY, or add "
        "to .env"
    )


def _already_tagged(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    ids = set()
    with out_path.open() as fh:
        for line in fh:
            try:
                ids.add(json.loads(line)["chunk_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return ids


def _format_batch(batch: list[dict[str, Any]]) -> str:
    lines = []
    for i, chunk in enumerate(batch, start=1):
        char = chunk["character"]
        text = chunk["text"]
        # Trim very long chunks to stay under prompt budget
        if len(text) > 1800:
            text = text[:1800] + "…"
        lines.append(f"[{i}] About {char}:\n{text}\n")
    return "\n".join(lines)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_response(raw: str, batch_size: int) -> dict[str, list[str]]:
    """Strip markdown fences / prose, parse JSON, validate shape."""
    match = _JSON_RE.search(raw)
    if not match:
        raise ValueError(f"no JSON object in response: {raw[:200]!r}")
    obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError(f"expected dict, got {type(obj).__name__}")
    missing = [str(i) for i in range(1, batch_size + 1) if str(i) not in obj]
    if missing:
        raise ValueError(f"response missing keys: {missing}")
    return {k: list(v) for k, v in obj.items()}


def _validate_tags(tags: list[str], valid: set[str]) -> list[str]:
    """Keep only tags in the controlled vocabulary, preserve order."""
    seen: set[str] = set()
    out: list[str] = []
    for t in tags:
        t = t.strip().lower().replace("_", "-")
        if t in valid and t not in seen:
            out.append(t)
            seen.add(t)
    return out


def tag_batch(client: genai.Client, batch: list[dict[str, Any]], valid_tags: set[str]) -> list[list[str]]:
    prompt = PROMPT_TEMPLATE.format(
        vocab=", ".join(VOCABULARY) + ", factual",
        passages=_format_batch(batch),
    )
    resp = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
        ),
    )
    parsed = _parse_response(resp.text, batch_size=len(batch))
    return [_validate_tags(parsed[str(i + 1)], valid_tags) for i in range(len(batch))]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-path",     type=Path, default=IN_PATH)
    ap.add_argument("--out-path",    type=Path, default=OUT_PATH)
    ap.add_argument("--batch-size",  type=int, default=10)
    ap.add_argument("--limit",       type=int, default=0, help="0 = no limit")
    ap.add_argument("--dry-run",     action="store_true")
    args = ap.parse_args(argv)

    if not args.in_path.exists():
        print(f"ERROR: input not found: {args.in_path}", file=sys.stderr)
        return 2

    client = genai.Client(api_key=_load_api_key())

    # Note: `factual` is NOT in valid — if Gemini returns it, we drop it.
    # The prompt now tells it to return [] for purely factual passages.
    valid = set(VOCABULARY)
    already = _already_tagged(args.out_path)
    with args.in_path.open(encoding="utf-8") as fh:
        chunks = [json.loads(line) for line in fh]

    todo = [c for c in chunks if c["chunk_id"] not in already]
    if args.limit:
        todo = todo[: args.limit]

    print(f"Loaded {len(chunks)} chunks total.")
    print(f"Already tagged: {len(already)}.")
    print(f"To tag: {len(todo)} chunks in batches of {args.batch_size}.")

    if args.dry_run:
        print("--dry-run set: exiting before LLM calls.")
        return 0

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tagged_count = 0
    factual_count = 0

    with args.out_path.open("a", encoding="utf-8") as out_fh:
        for batch_start in range(0, len(todo), args.batch_size):
            batch = todo[batch_start : batch_start + args.batch_size]
            batch_no = batch_start // args.batch_size + 1
            total_batches = (len(todo) + args.batch_size - 1) // args.batch_size
            try:
                tags_list = tag_batch(client, batch, valid)
            except Exception as exc:
                print(f"  batch {batch_no}/{total_batches} FAILED: {exc}", file=sys.stderr)
                continue

            for chunk, tags in zip(batch, tags_list):
                chunk["themes"] = tags
                if tags == ["factual"] or not tags:
                    factual_count += 1
                out_fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                tagged_count += 1
            out_fh.flush()
            elapsed = time.time() - t0
            rate = tagged_count / elapsed if elapsed > 0 else 0
            eta = (len(todo) - tagged_count) / rate if rate > 0 else 0
            print(f"  batch {batch_no}/{total_batches}  ({tagged_count}/{len(todo)} done, "
                  f"{rate:.1f}/s, ETA {eta:.0f}s)")

    print()
    print(f"Tagged {tagged_count} chunks in {time.time() - t0:.1f}s")
    print(f"Marked as factual (drop downstream): {factual_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
