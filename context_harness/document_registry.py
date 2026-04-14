"""
Document Registry
=================
Fixes deepTutor Weaknesses #2 and #4:
  #2 "No update/delete path for changed documents — dedup is file-hash based only;
      old chunks remain in the index alongside new ones if a document changes."
  #4 "Custom FAISS indexer is write-once — no deletion support."

Solution:
  - DocumentRegistry tracks every ingested document by SHA-256 content hash.
  - On re-ingest it detects whether the hash changed (same path, different content).
  - If changed: delete old chunks from ChromaDB (which supports deletion natively),
    then ingest fresh chunks. No full rebuild needed.
  - Registry state is persisted in a lightweight SQLite DB so it survives restarts.
  - ChromaDB is used as the exclusive vector store — it has full CRUD, so the
    FAISS write-once limitation is sidestepped entirely.

Why not FAISS?
  IndexFlatIP has no delete(). The only fix would be a full rebuild after every change,
  which is O(N) in corpus size. ChromaDB's HNSW index supports delete() in O(log N).
  We consolidate on ChromaDB to get true incremental updates.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .rag_pipeline import RAGPipeline, Chunk, ChunkingStrategy


# ---------------------------------------------------------------------------
# Registry record
# ---------------------------------------------------------------------------

@dataclass
class DocRecord:
    doc_id: str
    content_hash: str          # SHA-256 of the full document text
    chunk_ids: List[str]       # IDs of all chunks currently in the vector store
    chunk_count: int
    embedding_model: str
    ingested_at: float = field(default_factory=time.time)
    last_modified: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Registry store (SQLite)
# ---------------------------------------------------------------------------

class DocumentRegistry:
    """
    Tracks ingested documents and their chunk IDs so stale chunks can be deleted
    before re-ingesting updated content.

    This makes the RAG pipeline truly incremental:
      - Unchanged document → skip (no work done)
      - Changed document   → delete old chunks, ingest new chunks
      - New document       → ingest

    Usage:
        registry = DocumentRegistry(pipeline, db_path="data/doc_registry.db")
        registry.upsert("hp-book-1", full_text, embedding_model="all-MiniLM-L6-v2")
    """

    def __init__(
        self,
        pipeline: RAGPipeline,
        db_path: str = "data/doc_registry.db",
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self._pipeline = pipeline
        self._embedding_model = embedding_model
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS doc_records (
                doc_id          TEXT PRIMARY KEY,
                content_hash    TEXT NOT NULL,
                chunk_ids       TEXT NOT NULL,
                chunk_count     INTEGER NOT NULL,
                embedding_model TEXT NOT NULL,
                ingested_at     REAL NOT NULL,
                last_modified   REAL NOT NULL
            );
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert(self, doc_id: str, text: str, **metadata) -> UpsertResult:
        """
        Smart upsert: skip unchanged, delete-then-reingest changed, ingest new.
        Returns a UpsertResult describing what happened.
        """
        new_hash = _sha256(text)
        existing = self._get(doc_id)

        if existing is not None:
            if existing.content_hash == new_hash:
                return UpsertResult(doc_id=doc_id, action="skipped", chunk_count=existing.chunk_count)

            # Content changed — delete stale chunks first
            self._delete_chunks(existing.chunk_ids)
            action = "updated"
        else:
            action = "created"

        # Ingest fresh chunks
        chunks = self._pipeline.ingest(text, doc_id, embedding_model=self._embedding_model, **metadata)
        chunk_ids = [c.chunk_id for c in chunks]

        record = DocRecord(
            doc_id=doc_id,
            content_hash=new_hash,
            chunk_ids=chunk_ids,
            chunk_count=len(chunks),
            embedding_model=self._embedding_model,
        )
        self._save(record)
        return UpsertResult(doc_id=doc_id, action=action, chunk_count=len(chunks))

    def delete(self, doc_id: str) -> bool:
        """Delete a document and all its chunks from the vector store."""
        existing = self._get(doc_id)
        if existing is None:
            return False
        self._delete_chunks(existing.chunk_ids)
        self._conn.execute("DELETE FROM doc_records WHERE doc_id=?", (doc_id,))
        self._conn.commit()
        return True

    def get_record(self, doc_id: str) -> Optional[DocRecord]:
        return self._get(doc_id)

    def list_docs(self) -> List[str]:
        rows = self._conn.execute("SELECT doc_id FROM doc_records ORDER BY ingested_at").fetchall()
        return [r[0] for r in rows]

    def stats(self) -> dict:
        row = self._conn.execute(
            "SELECT COUNT(*), SUM(chunk_count) FROM doc_records"
        ).fetchone()
        return {"doc_count": row[0], "total_chunks": row[1] or 0}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, doc_id: str) -> Optional[DocRecord]:
        row = self._conn.execute(
            "SELECT * FROM doc_records WHERE doc_id=?", (doc_id,)
        ).fetchone()
        if row is None:
            return None
        doc_id_, hash_, chunk_ids_json, count, model, ingested, modified = row
        return DocRecord(
            doc_id=doc_id_,
            content_hash=hash_,
            chunk_ids=_deserialise_ids(chunk_ids_json),
            chunk_count=count,
            embedding_model=model,
            ingested_at=ingested,
            last_modified=modified,
        )

    def _save(self, record: DocRecord) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO doc_records
               VALUES (?,?,?,?,?,?,?)""",
            (
                record.doc_id,
                record.content_hash,
                _serialise_ids(record.chunk_ids),
                record.chunk_count,
                record.embedding_model,
                record.ingested_at,
                record.last_modified,
            ),
        )
        self._conn.commit()

    def _delete_chunks(self, chunk_ids: List[str]) -> None:
        """Delete chunk IDs from ChromaDB. No-op if the collection is unavailable."""
        if self._pipeline._collection is not None and chunk_ids:
            try:
                self._pipeline._collection.delete(ids=chunk_ids)
            except Exception as exc:
                print(f"[DocumentRegistry] Warning: failed to delete chunks: {exc}")
        else:
            # In-memory fallback: remove from _chunks list
            self._pipeline._chunks = [
                c for c in self._pipeline._chunks if c.chunk_id not in set(chunk_ids)
            ]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class UpsertResult:
    doc_id: str
    action: str          # 'created' | 'updated' | 'skipped'
    chunk_count: int

    def __repr__(self) -> str:
        return f"UpsertResult(doc={self.doc_id!r}, action={self.action!r}, chunks={self.chunk_count})"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _serialise_ids(ids: List[str]) -> str:
    import json
    return json.dumps(ids)


def _deserialise_ids(s: str) -> List[str]:
    import json
    return json.loads(s)
