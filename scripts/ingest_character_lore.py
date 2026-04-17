"""
Ingest the tagged character-lore chunks into ChromaDB.
======================================================
Reads ``data/character_lore_tagged.jsonl`` (808 chunks after skipping the
6 empty/factual ones) and upserts them into a new ``character_lore``
ChromaDB collection, keyed by ``chunk_id``, with metadata suitable for
filtered retrieval.

Why a separate collection:
    The existing ``hp_lore`` collection holds 10 curated docs used by every
    mode. Character lore is per-character and much larger (~800 chunks);
    mixing them would dilute retrieval for the other modes. The
    ``perspective_shift`` mode gets its own retrieval pipeline against this
    collection.

Metadata shape (per chunk)::

    {
      "doc_id":         <chunk_id, e.g. "luna-lovegood/biography-early-life-001">,
      "character":      <slug, e.g. "luna-lovegood">,
      "themes":         "<comma-separated tag list>",
      "section_path":   "<' > '-joined section hierarchy>",
      "word_count":     <int>,
      "source_url":     <HP Wiki URL>,
      "source_license": "CC-BY-SA-3.0",
    }

(ChromaDB's metadata layer does not support list values. `themes` is
stored as a string; character filtering is exact-match on ``character``.
Theme-based filtering/ranking is done post-retrieval in Python.)

Usage::

    .venv/bin/python -m scripts.ingest_character_lore
    .venv/bin/python -m scripts.ingest_character_lore --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from context_harness.ingest_lore import build_pipeline


IN_PATH = Path("data/character_lore_tagged.jsonl")
COLLECTION = "character_lore"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-path",         type=Path, default=IN_PATH)
    ap.add_argument("--collection-name", default=COLLECTION)
    ap.add_argument("--batch-size",      type=int, default=100)
    ap.add_argument("--dry-run",         action="store_true")
    args = ap.parse_args(argv)

    if not args.in_path.exists():
        print(f"ERROR: input not found: {args.in_path}", file=sys.stderr)
        return 2

    chunks = [json.loads(line) for line in args.in_path.open(encoding="utf-8")]
    # Drop empty-theme chunks — the tagger returned [] for purely factual passages
    keep = [c for c in chunks if c.get("themes")]
    dropped = len(chunks) - len(keep)

    print(f"Loaded {len(chunks)} chunks.")
    print(f"Dropping {dropped} empty-theme chunks (purely factual).")
    print(f"Ingesting {len(keep)} into collection {args.collection_name!r}.")

    # Distribution preview — useful sanity check
    char_counts = Counter(c["character"] for c in keep)
    print("\nPer-character counts:")
    for char, n in sorted(char_counts.items()):
        print(f"  {char:22s}  {n:3d}")

    if args.dry_run:
        print("\n--dry-run set: exiting before Chroma writes.")
        return 0

    # Build pipeline pointing at the new collection
    pipeline = build_pipeline(persist=True, collection_name=args.collection_name)
    # Wipe any stale state in this collection so re-running is idempotent
    existing_count = pipeline.count()
    if existing_count:
        print(f"\nCollection already has {existing_count} vectors — clearing before re-ingest.")
        # Chroma: delete by retrieving all IDs then deleting (or use where={} to delete all)
        # Fastest: drop the collection and rebuild
        pipeline._client.delete_collection(name=args.collection_name)
        pipeline = build_pipeline(persist=True, collection_name=args.collection_name)

    # Batch upserts — Chroma handles large batches well but 100-at-a-time
    # keeps progress visible and bounds any single transaction's memory.
    def _metadata(c: dict) -> dict:
        return {
            "doc_id":         c["chunk_id"],
            "character":      c["character"],
            "themes":         ",".join(c["themes"]),
            "section_path":   " > ".join(c["section_path"]),
            "word_count":     int(c["word_count"]),
            "source_url":     c["source_url"],
            "source_license": c["source_license"],
        }

    total = 0
    for start in range(0, len(keep), args.batch_size):
        batch = keep[start : start + args.batch_size]
        pipeline._collection.upsert(
            ids=       [c["chunk_id"]    for c in batch],
            documents= [c["text"]        for c in batch],
            metadatas= [_metadata(c)     for c in batch],
        )
        total += len(batch)
        print(f"  upserted {total}/{len(keep)}")

    final_count = pipeline.count()
    print(f"\nDone. Collection {args.collection_name!r} now has {final_count} vectors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
