"""
Chunk scraped HP Wiki wikitext into 200–400 word passages.
==========================================================
Reads ``data/hp_wiki_raw/<slug>.json`` (produced by scrape_hp_wiki.py),
cleans the MediaWiki markup into plain text, section-splits, then further
breaks long sections into ~300-word chunks on sentence boundaries.

Output: ``data/character_lore.jsonl`` — one chunk per line.

Chunk schema::

    {
      "chunk_id": "harry-potter/biography-early-life-001",
      "character": "harry-potter",
      "section_path": ["Biography", "Early life"],
      "text": "...",
      "word_count": 312,
      "themes": [],                           # filled in later by theme tagger
      "source_url": "https://harrypotter.fandom.com/wiki/Harry_Potter",
      "source_license": "CC-BY-SA-3.0"
    }

Sections we keep (depth-first match on any heading):
    Biography, Personality and traits, Relationships,
    Magical abilities and skills, Etymology

Sections we skip:
    Appearances, Behind the scenes, Notes and references, See also,
    External links, Gallery, In video games, Non-canon
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator

import mwparserfromhell


RAW_DIR = Path("data/hp_wiki_raw")
OUT_PATH = Path("data/character_lore.jsonl")

KEEP_SECTIONS = {
    "biography",
    "personality and traits",
    "relationships",
    "etymology",
    # NOTE: "magical abilities and skills" intentionally dropped — these
    # sections are stats dumps (spells known, creatures handled) with no
    # life-thematic content, and pollute the retrieval index for the
    # perspective_shift mode. Review 2026-04-17.
}

SKIP_SECTIONS = {
    "appearances", "behind the scenes", "notes and references",
    "references", "see also", "external links", "gallery",
    "in video games", "non-canon", "trivia",
}

# Target band tuned for the all-MiniLM-L6-v2 embedder (sweet spot ~40-150
# words). Was 200/400; 312-word chunks like the Aberforth-Albus relationship
# diluted their embedding signal and lost ranking to shorter siblings even
# when topically more relevant. See tasks/plan-rechunk.md (2026-05-09).
TARGET_MIN_WORDS = 80
TARGET_MAX_WORDS = 150


# ---------------------------------------------------------------------------
# Wikitext -> clean text
# ---------------------------------------------------------------------------

_REF_RE       = re.compile(r"<ref[^>]*>.*?</ref>", re.DOTALL | re.IGNORECASE)
_REF_SELF_RE  = re.compile(r"<ref[^>]*/\s*>",     re.IGNORECASE)
_COMMENT_RE   = re.compile(r"<!--.*?-->", re.DOTALL)
_WS_RE        = re.compile(r"[ \t]+")
_BLANK_RE     = re.compile(r"\n{3,}")


def _clean_wikitext(raw: str) -> str:
    """Strip <ref> blocks, comments, and normalise whitespace BEFORE parsing.

    mwparserfromhell handles most markup but bleeds reference content into the
    text; we nuke refs and HTML comments first to keep the output clean.
    """
    txt = _REF_RE.sub("", raw)
    txt = _REF_SELF_RE.sub("", txt)
    txt = _COMMENT_RE.sub("", txt)
    return txt


def _strip_bracketed_clerical(text: str) -> str:
    """Drop lines that are just template residue or infobox leftovers."""
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            out.append("")
            continue
        # Drop lines that look like pipe-separated infobox residue
        if s.startswith("|") or s.startswith("{{") or s.startswith("}}"):
            continue
        # Drop bare categories
        if s.startswith("Category:"):
            continue
        out.append(line)
    return "\n".join(out)


_IMAGE_RESIDUE_RE = re.compile(
    r"^\s*(?:\d+px\||thumb\||left\||right\||center\||border\|)+[^\n]*$",
    re.MULTILINE,
)


def _render_plain(wikitext: str) -> str:
    """Parse wikitext, drop templates and file links, keep link labels, return plain text."""
    cleaned = _clean_wikitext(wikitext)
    wikicode = mwparserfromhell.parse(cleaned)

    # Remove File:/Image: wikilinks entirely — their captions leak through
    # strip_code and pollute the text. Snapshot first because remove() mutates.
    for link in list(wikicode.filter_wikilinks()):
        title = str(link.title).strip().lower()
        if title.startswith(("file:", "image:", "category:")):
            try:
                wikicode.remove(link)
            except ValueError:
                # link was nested inside a template that's already been stripped
                pass

    # strip_code removes templates by default and turns [[Link|label]] into 'label'
    plain = wikicode.strip_code(normalize=True, collapse=True)
    # Belt-and-braces: kill any residual image-caption prefix lines
    plain = _IMAGE_RESIDUE_RE.sub("", plain)
    plain = _strip_bracketed_clerical(plain)
    plain = _WS_RE.sub(" ", plain)
    plain = _BLANK_RE.sub("\n\n", plain)
    return plain.strip()


# ---------------------------------------------------------------------------
# Section tree walk
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.MULTILINE)


@dataclass
class Section:
    path: list[str]
    text: str


def _parse_sections(raw: str) -> list[Section]:
    """Split raw wikitext into sections keyed by heading path.

    Walks top-down so a nested ``=== Fifth year ===`` inside ``== Biography ==``
    becomes ``path = ["Biography", "Fifth year"]``.
    """
    matches = list(_HEADING_RE.finditer(raw))
    if not matches:
        return [Section(path=[], text=raw)]

    sections: list[Section] = []
    stack: list[tuple[int, str]] = []

    # Leading preamble (before first heading) -> top-level
    if matches[0].start() > 0:
        sections.append(Section(path=[], text=raw[:matches[0].start()]))

    for idx, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()
        # Trim wiki template calls like {{C|1981–1992}} from headings
        title = re.sub(r"\{\{[^}]*\}\}", "", title).strip()
        # Pop deeper-or-equal levels
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw)
        sections.append(Section(path=[t for _, t in stack], text=raw[start:end]))
    return sections


def _is_section_kept(section: Section) -> bool:
    """True if any level of the section's path is in our keep-list AND
    no level is in the skip-list."""
    if not section.path:
        return False
    lowered = [p.lower() for p in section.path]
    if any(p in SKIP_SECTIONS for p in lowered):
        return False
    top = lowered[0]
    return top in KEEP_SECTIONS


# ---------------------------------------------------------------------------
# Word-aware chunking on sentence boundaries
# ---------------------------------------------------------------------------

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def _split_sentences(text: str) -> list[str]:
    # Normalise newlines to spaces so a sentence isn't split across paragraphs
    joined = re.sub(r"\s*\n\s*", " ", text)
    return [s.strip() for s in _SENTENCE_RE.split(joined) if s.strip()]


def _chunk_by_words(text: str) -> Iterator[str]:
    sentences = _split_sentences(text)
    buf: list[str] = []
    buf_wc = 0
    for sent in sentences:
        wc = len(sent.split())
        if buf and buf_wc + wc > TARGET_MAX_WORDS:
            yield " ".join(buf)
            buf, buf_wc = [], 0
        buf.append(sent)
        buf_wc += wc
        if buf_wc >= TARGET_MIN_WORDS + (TARGET_MAX_WORDS - TARGET_MIN_WORDS) // 2:
            # Prefer emitting around mid-target to avoid trailing short chunks
            yield " ".join(buf)
            buf, buf_wc = [], 0
    if buf_wc >= 40:  # don't emit tiny tail fragments (lowered to match new band)
        yield " ".join(buf)


# ---------------------------------------------------------------------------
# Chunk assembly
# ---------------------------------------------------------------------------

def _slugify(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()
    return s or "section"


@dataclass
class Chunk:
    chunk_id:        str
    character:       str
    section_path:    list[str]
    text:            str
    word_count:      int
    themes:          list[str] = field(default_factory=list)
    source_url:      str = ""
    source_license:  str = ""


def _chunks_for_character(raw_payload: dict) -> list[Chunk]:
    slug = raw_payload["slug"]
    url = raw_payload["canonical_url"]
    license_ = raw_payload["license"]
    sections = _parse_sections(raw_payload["wikitext"])

    out: list[Chunk] = []
    for section in sections:
        if not _is_section_kept(section):
            continue
        plain = _render_plain(section.text)
        if len(plain.split()) < 60:
            continue
        section_slug = "-".join(_slugify(p) for p in section.path)
        for i, chunk_text in enumerate(_chunk_by_words(plain), start=1):
            chunk_id = f"{slug}/{section_slug}-{i:03d}"
            out.append(Chunk(
                chunk_id=chunk_id,
                character=slug,
                section_path=section.path,
                text=chunk_text,
                word_count=len(chunk_text.split()),
                source_url=url,
                source_license=license_,
            ))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--out",     type=Path, default=OUT_PATH)
    args = ap.parse_args(argv)

    if not args.raw_dir.exists():
        print(f"ERROR: raw dir does not exist: {args.raw_dir}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    per_char_counts: dict[str, int] = {}
    with args.out.open("w", encoding="utf-8") as fh:
        for raw_file in sorted(args.raw_dir.glob("*.json")):
            payload = json.loads(raw_file.read_text(encoding="utf-8"))
            chunks = _chunks_for_character(payload)
            per_char_counts[payload["slug"]] = len(chunks)
            for c in chunks:
                fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
            total_chunks += len(chunks)
            print(f"  {payload['slug']:22s}  {len(chunks):3d} chunks")

    print()
    print(f"Wrote {total_chunks} chunks to {args.out}")
    if total_chunks < 100:
        print("WARN: fewer than 100 chunks — target was ~150. Consider loosening "
              "KEEP_SECTIONS or lowering TARGET_MIN_WORDS.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
