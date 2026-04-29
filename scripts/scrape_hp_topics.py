"""
Scrape lore-topic pages from the Harry Potter Fandom wiki.
==========================================================
Source: https://harrypotter.fandom.com/
License: CC-BY-SA 3.0 (attribution required on downstream use)

Different from ``scrape_hp_wiki.py`` (which scrapes character pages and dumps
raw wikitext). This script:

  1. Fetches each topic page's wikitext via the MediaWiki ``action=parse`` API.
  2. Cleans wikitext → plain text using ``_render_plain`` from
     ``chunk_character_lore.py``.
  3. Takes the lead section (everything before the first ``==`` heading) — a
     reasonable single-paragraph summary of the topic.
  4. Appends each as ``doc_id: <slug>\\n<text>`` to ``data/hp_lore.txt``,
     matching the format of the existing 11 hand-written docs.
  5. Updates ``data/SOURCES.md`` with attribution per topic.

Usage
-----
    .venv/bin/python -m scripts.scrape_hp_topics
    .venv/bin/python -m scripts.scrape_hp_topics --topics room-of-requirement,patronus-charm
    .venv/bin/python -m scripts.scrape_hp_topics --dry-run   # print, don't write

Rate limit: 1 request/second.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

# Reuse cleaning helpers; we wrap _render_plain with template-stripping below
from scripts.chunk_character_lore import (
    _render_plain,
    _clean_wikitext,
    _strip_bracketed_clerical,
    _IMAGE_RESIDUE_RE,
    _WS_RE,
    _BLANK_RE,
)
import mwparserfromhell


FANDOM_API = "https://harrypotter.fandom.com/api.php"
LORE_FILE = Path("data/hp_lore.txt")
SOURCES_FILE = Path("data/SOURCES.md")
USER_AGENT = "HPLoreAgent-TopicScraper/0.1 (pedagogical research; contact via repo)"


# Slug -> wiki page title. The MediaWiki API follows redirects, so the
# canonical title returned by the API is what ends up in SOURCES.md.
TIER_A_TOPICS: dict[str, str] = {
    "room-of-requirement":   "Room of Requirement",
    "patronus-charm":        "Patronus Charm",
    "wandlore":              "Wandlore",
    "unforgivable-curses":   "Unforgivable Curses",
    "dementors":             "Dementor",
    "azkaban":               "Azkaban",
    "marauders-map":         "Marauder's Map",
    "polyjuice-potion":      "Polyjuice Potion",
    "time-turner":           "Time-Turner",
    "mirror-of-erised":      "Mirror of Erised",
    "triwizard-tournament":  "Triwizard Tournament",
    "dumbledores-army":      "Dumbledore's Army",
    "chamber-of-secrets":    "Chamber of Secrets",
}

TIER_B_CHARACTERS: dict[str, str] = {
    "minerva-mcgonagall":    "Minerva McGonagall",
    "rubeus-hagrid":         "Rubeus Hagrid",
    "sirius-black":          "Sirius Black",
    "remus-lupin":           "Remus Lupin",
    "draco-malfoy":          "Draco Malfoy",
    "luna-lovegood":         "Luna Lovegood",
    "neville-longbottom":    "Neville Longbottom",
    "bellatrix-lestrange":   "Bellatrix Lestrange",
}


@dataclass
class TopicScrapeResult:
    slug: str
    requested_title: str
    canonical_title: str
    canonical_url: str
    lead_text: str        # cleaned plain text of the lead section
    word_count: int
    retrieved_at: str


def fetch_wikitext(title: str, session: requests.Session) -> tuple[str, str, str]:
    """Return ``(canonical_title, canonical_url, raw_wikitext)``."""
    params = {
        "action":         "parse",
        "page":           title,
        "prop":           "wikitext",
        "redirects":      1,
        "format":         "json",
        "formatversion":  2,
    }
    resp = session.get(FANDOM_API, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if "error" in payload:
        raise RuntimeError(f"MediaWiki API error for {title!r}: {payload['error']}")
    parse = payload["parse"]
    canonical_title = parse["title"]
    canonical_url = (
        f"https://harrypotter.fandom.com/wiki/{canonical_title.replace(' ', '_')}"
    )
    return canonical_title, canonical_url, parse["wikitext"]


# Wikitext heading: ==Title==, ===Subtitle===, etc. The lead section is
# everything before the first such heading. Critical: this regex runs against
# the RAW wikitext, before mwparserfromhell strips the `=` markers.
_HEADING_RE = re.compile(r"^={2,}\s*[^=\n].*?\s*={2,}\s*$", re.MULTILINE)


# The first ``'''Name'''`` bold marker is the wiki convention for "here begins
# the actual article prose." Everything before it is infobox + spoiler banner +
# notice templates that we never want in the corpus. Critical because
# mwparserfromhell sometimes fails to parse Fandom's infoboxes as a single
# template (embedded <gallery> tags + HTML comments confuse it), which lets
# infobox key-value text leak into the cleaned output.
_BOLD_MARKER_RE = re.compile(r"'''[^'\n][^'\n]{0,200}?'''")


def extract_lead_wikitext(raw: str) -> str:
    """Slice the raw wikitext to the lead-section prose.

    Two cuts:
      1. Start at the first ``'''Subject Name'''`` bold marker (wiki
         convention for the start of the lead paragraph). This skips any
         infobox / spoiler / notice templates that mwparserfromhell may
         not parse cleanly.
      2. End at the first ``==Heading==`` (start of the next section).

    If the bold marker is missing (rare lore pages without one), fall back
    to slicing after the last top-level ``}}`` that closes the topmost
    templates.
    """
    # Cut at end-of-lead first (start of next section)
    end_match = _HEADING_RE.search(raw)
    body = raw[: end_match.start()] if end_match else raw

    # Cut at start-of-prose (first '''bold''' marker)
    start_match = _BOLD_MARKER_RE.search(body)
    if start_match:
        return body[start_match.start():]

    # Fallback: skip past everything that looks like a leading template block.
    # Find the last ``}}`` in the first 4000 chars (covers even huge infoboxes)
    # and take everything after.
    head = body[:4000]
    last_close = head.rfind("}}")
    return body[last_close + 2:].lstrip() if last_close > 0 else body


def render_plain_no_templates(wikitext: str) -> str:
    """Like ``_render_plain`` but explicitly removes every template before
    stripping markup. Without this, infobox content (`{{Witch|name=...}}`)
    leaks into the cleaned text as garbled key-value fragments — e.g.
    `Professor British Ministry of Magic (b. 4 October) was a Scottish...`
    on the McGonagall page. We drop templates wholesale; navbox / infobox
    text was never useful prose anyway."""
    cleaned = _clean_wikitext(wikitext)
    wikicode = mwparserfromhell.parse(cleaned)

    # Drop every template (infoboxes, navboxes, citation needed, etc.)
    for template in list(wikicode.filter_templates()):
        try:
            wikicode.remove(template)
        except ValueError:
            pass

    # Drop File:/Image:/Category: wikilinks
    for link in list(wikicode.filter_wikilinks()):
        title = str(link.title).strip().lower()
        if title.startswith(("file:", "image:", "category:")):
            try:
                wikicode.remove(link)
            except ValueError:
                pass

    plain = wikicode.strip_code(normalize=True, collapse=True)
    plain = _IMAGE_RESIDUE_RE.sub("", plain)
    plain = _strip_bracketed_clerical(plain)
    plain = _WS_RE.sub(" ", plain)
    plain = _BLANK_RE.sub("\n\n", plain)
    return plain.strip()


# Drop ``Spoiler``-style notice lines and game/film-only canon markers.
# Tier-A and Tier-B docs should stay book-canon-faithful; the wiki sometimes
# folds Hogwarts Mystery / Hogwarts Legacy / Fantastic Beasts material into
# the lead, which gives the agent unexpected vocabulary downstream.
_NOTICE_PREFIXES = (
    "this article is about",
    "you may be looking for",
    "warning:",
    "spoiler:",
)


def collapse_to_paragraph(text: str, max_words: int = 400) -> str:
    """Drop notice lines, fold whitespace, cap to ``max_words``. Keeps the
    output as a single dense paragraph matching the existing 11 hand-written
    docs in ``hp_lore.txt``."""
    out_lines = []
    for line in text.splitlines():
        s = line.strip()
        low = s.lower()
        if any(low.startswith(p) for p in _NOTICE_PREFIXES):
            continue
        if not s:
            continue
        out_lines.append(s)
    body = " ".join(out_lines).strip()
    body = re.sub(r"\s+", " ", body)

    # Cosmetic cleanup of residue from removed templates/refs:
    #   ", ," → ","   " ," → ","   " ." → "."   "( )" → ""   "(,)" → ""
    body = re.sub(r"\s+,", ",", body)
    body = re.sub(r",\s*,+", ",", body)
    body = re.sub(r"\(\s*,?\s*\)", "", body)
    body = re.sub(r"\s+\.", ".", body)
    body = re.sub(r"\s{2,}", " ", body)

    words = body.split()
    if len(words) > max_words:
        body = " ".join(words[:max_words]).rstrip(",.;:") + "…"
    return body


def append_to_lore_file(slug: str, text: str, lore_path: Path) -> None:
    """Append a `doc_id: <slug>\\n<text>` block, with a leading blank line."""
    payload = f"\n\ndoc_id: {slug}\n{text}\n"
    with lore_path.open("a", encoding="utf-8") as f:
        f.write(payload)


def update_sources_md(results: list[TopicScrapeResult], path: Path) -> None:
    """Write/refresh data/SOURCES.md with a table of doc_id → wiki URL."""
    lines = [
        "# Corpus sources",
        "",
        "Topics in `data/hp_lore.txt` derived from the Harry Potter Fandom Wiki "
        "are listed here for attribution. Wiki content is licensed CC-BY-SA 3.0; "
        "downstream use of `hp_lore.txt` must preserve attribution and remain "
        "under the same license.",
        "",
        "| `doc_id` | Wiki page | Retrieved |",
        "|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| `{r.slug}` | [{r.canonical_title}]({r.canonical_url}) | {r.retrieved_at} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--topics",
        help="comma-separated subset of topic slugs; default: all Tier-A + Tier-B",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print scraped lead sections; do not modify data/ files",
    )
    ap.add_argument(
        "--max-words",
        type=int,
        default=500,
        help="cap each lead section to this many words (default: 500)",
    )
    ap.add_argument(
        "--rate-limit-sec",
        type=float,
        default=1.0,
        help="seconds between requests (default: 1.0)",
    )
    args = ap.parse_args(argv)

    all_targets = {**TIER_A_TOPICS, **TIER_B_CHARACTERS}
    if args.topics:
        wanted = [s.strip() for s in args.topics.split(",") if s.strip()]
        unknown = [s for s in wanted if s not in all_targets]
        if unknown:
            print(f"ERROR: unknown topic slug(s): {unknown}", file=sys.stderr)
            print(f"Known slugs: {sorted(all_targets)}", file=sys.stderr)
            return 2
        targets = {s: all_targets[s] for s in wanted}
    else:
        targets = all_targets

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    results: list[TopicScrapeResult] = []
    failures: list[tuple[str, str]] = []

    for i, (slug, title) in enumerate(targets.items(), 1):
        print(f"[{i}/{len(targets)}] {slug}  ({title})")
        try:
            canonical_title, canonical_url, wikitext = fetch_wikitext(title, session)
            # Slice lead from RAW wikitext (before `==` markers get stripped),
            # then drop templates and clean to plain text.
            lead_raw = extract_lead_wikitext(wikitext)
            lead_plain = render_plain_no_templates(lead_raw)
            lead = collapse_to_paragraph(lead_plain, max_words=args.max_words)
            wc = len(lead.split())
            results.append(TopicScrapeResult(
                slug=slug,
                requested_title=title,
                canonical_title=canonical_title,
                canonical_url=canonical_url,
                lead_text=lead,
                word_count=wc,
                retrieved_at=time.strftime("%Y-%m-%d", time.gmtime()),
            ))
            print(f"    -> {wc} words")
            if args.dry_run:
                print(f"    {'-' * 60}")
                print(f"    {lead[:300]}{'...' if len(lead) > 300 else ''}")
                print(f"    {'-' * 60}")
        except Exception as exc:
            print(f"    FAILED: {exc}", file=sys.stderr)
            failures.append((slug, str(exc)))
        time.sleep(args.rate_limit_sec)

    if args.dry_run:
        print(f"\nDry run — {len(results)} topics scraped, no files written.")
        return 0 if not failures else 1

    # Write everything
    for r in results:
        append_to_lore_file(r.slug, r.lead_text, LORE_FILE)
    update_sources_md(results, SOURCES_FILE)
    print()
    print(f"Appended {len(results)} docs to {LORE_FILE}")
    print(f"Wrote attribution table to {SOURCES_FILE}")
    if failures:
        print(f"\nFailures: {len(failures)}")
        for slug, msg in failures:
            print(f"  - {slug}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
