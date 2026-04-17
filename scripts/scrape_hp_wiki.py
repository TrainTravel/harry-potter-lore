"""
Scrape character pages from the Harry Potter Fandom wiki.
=========================================================
Source: https://harrypotter.fandom.com/
License: CC-BY-SA 3.0 (requires attribution on downstream use)

Uses MediaWiki's ``action=parse`` API to fetch raw wikitext. This is more
robust than HTML scraping — no CSS selectors to break when the skin updates,
and the API follows redirects + gives us section structure out of the box.

Output: ``data/hp_wiki_raw/<character-slug>.json`` per character, containing
the wikitext plus metadata (page title, canonical URL, retrieved-at,
license). Downstream scripts chunk and clean this.

Usage
-----
    .venv/bin/python -m scripts.scrape_hp_wiki
    .venv/bin/python -m scripts.scrape_hp_wiki --characters harry-potter,hermione-granger

Rate limit: 1 request/second (polite default for a 10-page scrape).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests


FANDOM_API = "https://harrypotter.fandom.com/api.php"
OUTPUT_DIR = Path("data/hp_wiki_raw")
LICENSE = "CC-BY-SA-3.0"
USER_AGENT = "HPLoreAgent-CharScraper/0.1 (pedagogical research; contact via repo)"


# Slug -> canonical wiki page title. API follows redirects, so a reasonable
# guess works even when the wiki's canonical title differs (e.g. "Ronald Weasley"
# vs the more common "Ron Weasley").
CHARACTERS: dict[str, str] = {
    "harry-potter":        "Harry Potter",
    "hermione-granger":    "Hermione Granger",
    "ron-weasley":         "Ron Weasley",
    "albus-dumbledore":    "Albus Dumbledore",
    "severus-snape":       "Severus Snape",
    "minerva-mcgonagall":  "Minerva McGonagall",
    "luna-lovegood":       "Luna Lovegood",
    "neville-longbottom":  "Neville Longbottom",
    "lord-voldemort":      "Lord Voldemort",
    "draco-malfoy":        "Draco Malfoy",
    "rubeus-hagrid":       "Rubeus Hagrid",
}


@dataclass
class ScrapeResult:
    slug: str
    requested_title: str
    canonical_title: str
    canonical_url: str
    wikitext: str
    retrieved_at: str
    license: str = LICENSE


def fetch_wikitext(title: str, session: requests.Session) -> ScrapeResult:
    """Hit the MediaWiki API and return the parsed wikitext + metadata."""
    params = {
        "action": "parse",
        "page": title,
        "prop": "wikitext",
        "redirects": 1,
        "format": "json",
        "formatversion": 2,
    }
    resp = session.get(FANDOM_API, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if "error" in payload:
        raise RuntimeError(f"MediaWiki API error for {title!r}: {payload['error']}")
    parse = payload["parse"]
    canonical_title = parse["title"]
    canonical_url = f"https://harrypotter.fandom.com/wiki/{canonical_title.replace(' ', '_')}"
    wikitext = parse["wikitext"]
    return ScrapeResult(
        slug="",  # filled in by caller
        requested_title=title,
        canonical_title=canonical_title,
        canonical_url=canonical_url,
        wikitext=wikitext,
        retrieved_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


def write_result(result: ScrapeResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{result.slug}.json"
    payload = {
        "slug":             result.slug,
        "requested_title":  result.requested_title,
        "canonical_title":  result.canonical_title,
        "canonical_url":    result.canonical_url,
        "retrieved_at":     result.retrieved_at,
        "license":          result.license,
        "wikitext":         result.wikitext,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return out_path


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--characters",
        help="comma-separated subset of character slugs to scrape; default: all",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"output directory (default: {OUTPUT_DIR})",
    )
    ap.add_argument(
        "--rate-limit-sec",
        type=float,
        default=1.0,
        help="seconds to sleep between requests (default: 1.0)",
    )
    args = ap.parse_args(argv)

    if args.characters:
        wanted = [s.strip() for s in args.characters.split(",") if s.strip()]
        unknown = [s for s in wanted if s not in CHARACTERS]
        if unknown:
            print(f"ERROR: unknown character slug(s): {unknown}", file=sys.stderr)
            print(f"Known slugs: {sorted(CHARACTERS)}", file=sys.stderr)
            return 2
        targets = {s: CHARACTERS[s] for s in wanted}
    else:
        targets = CHARACTERS

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    successes, failures = 0, []
    for slug, title in targets.items():
        print(f"[{successes + len(failures) + 1}/{len(targets)}] {slug}  ({title})")
        try:
            result = fetch_wikitext(title, session)
            result.slug = slug
            path = write_result(result, args.out_dir)
            wc = len(result.wikitext.split())
            print(f"    -> {path}  ({wc:,} words of wikitext)")
            successes += 1
        except Exception as exc:
            print(f"    FAILED: {exc}", file=sys.stderr)
            failures.append((slug, str(exc)))
        time.sleep(args.rate_limit_sec)

    print()
    print(f"Scraped: {successes}/{len(targets)} characters.")
    if failures:
        print(f"Failures: {len(failures)}")
        for slug, msg in failures:
            print(f"  - {slug}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
