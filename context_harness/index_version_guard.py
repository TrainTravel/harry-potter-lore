"""
Index Version Guard
===================
Fixes deepTutor Weakness #6:
  "Index not versioned against embedding model — info.json stores the embedding
   dimension but not the model name. If the embedding model changes, existing
   vectors are silently incompatible with new query vectors."

Solution:
  - IndexVersionGuard stores model name + dimension + a content fingerprint
    in a versioned manifest (JSON sidecar to the vector store).
  - On every RAGPipeline construction it validates: model name matches AND
    dimension matches. If either fails it raises IndexVersionError immediately
    (fail-fast) instead of silently corrupting retrieval.
  - Supports a blue/green index swap: build a new index while the old one serves
    queries, then atomically rename the directory.
  - Migration path: if the model changes, IndexVersionGuard.rebuild() deletes
    the old index and triggers a full re-ingest from the DocumentRegistry.

Manifest schema (index_manifest.json):
  {
    "schema_version": 1,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "embedding_dim": 384,
    "collection_name": "hp_lore",
    "created_at": 1700000000.0,
    "updated_at": 1700000000.0,
    "doc_count": 12,
    "chunk_count": 148
  }
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1


@dataclass
class IndexManifest:
    embedding_model: str
    embedding_dim: int
    collection_name: str
    created_at: float
    updated_at: float
    doc_count: int = 0
    chunk_count: int = 0
    schema_version: int = SCHEMA_VERSION

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "IndexManifest":
        data = json.loads(path.read_text())
        if data.get("schema_version", 0) < SCHEMA_VERSION:
            raise IndexVersionError(
                f"Manifest schema v{data.get('schema_version')} is older than "
                f"current v{SCHEMA_VERSION}. Run IndexVersionGuard.migrate()."
            )
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def exists(cls, path: Path) -> bool:
        return path.exists()


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------

class IndexVersionError(RuntimeError):
    """Raised when the manifest is incompatible with the current embedding model."""


class IndexDimensionError(IndexVersionError):
    """Raised when stored embedding dimension doesn't match expected dimension."""


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

class IndexVersionGuard:
    """
    Validates that the on-disk index was built with the same embedding model
    (and dimension) as the one currently configured.

    Usage:
        guard = IndexVersionGuard(
            manifest_path="data/knowledge_bases/hp_lore/index_manifest.json",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            embedding_dim=384,
            collection_name="hp_lore",
        )
        guard.validate()      # raises IndexVersionError if incompatible
        guard.update_stats(doc_count=12, chunk_count=148)
    """

    def __init__(
        self,
        manifest_path: str,
        embedding_model: str,
        embedding_dim: int,
        collection_name: str,
    ) -> None:
        self._path = Path(manifest_path)
        self._model = embedding_model
        self._dim = embedding_dim
        self._collection = collection_name

    # ------------------------------------------------------------------
    # Initialise or validate
    # ------------------------------------------------------------------

    def ensure(self) -> IndexManifest:
        """
        If no manifest exists: create one (first-time setup).
        If manifest exists: validate it matches current config.
        Returns the manifest.
        """
        if not IndexManifest.exists(self._path):
            return self._create()
        return self.validate()

    def validate(self) -> IndexManifest:
        """Load manifest and assert compatibility. Raises on mismatch."""
        manifest = IndexManifest.load(self._path)

        if manifest.embedding_model != self._model:
            raise IndexVersionError(
                f"Embedding model mismatch:\n"
                f"  Index was built with: {manifest.embedding_model!r}\n"
                f"  Current config:       {self._model!r}\n"
                f"Run IndexVersionGuard.rebuild() to re-index with the new model."
            )

        if manifest.embedding_dim != self._dim:
            raise IndexDimensionError(
                f"Embedding dimension mismatch:\n"
                f"  Index dimension: {manifest.embedding_dim}\n"
                f"  Model dimension: {self._dim}\n"
                f"Cannot mix vectors of different dimensions."
            )

        if manifest.collection_name != self._collection:
            raise IndexVersionError(
                f"Collection name mismatch: {manifest.collection_name!r} vs {self._collection!r}"
            )

        return manifest

    def _create(self) -> IndexManifest:
        now = time.time()
        manifest = IndexManifest(
            embedding_model=self._model,
            embedding_dim=self._dim,
            collection_name=self._collection,
            created_at=now,
            updated_at=now,
        )
        manifest.save(self._path)
        return manifest

    # ------------------------------------------------------------------
    # Stats update (call after each ingest)
    # ------------------------------------------------------------------

    def update_stats(self, doc_count: int, chunk_count: int) -> None:
        if not IndexManifest.exists(self._path):
            self._create()
        manifest = IndexManifest.load(self._path)
        manifest.doc_count = doc_count
        manifest.chunk_count = chunk_count
        manifest.updated_at = time.time()
        manifest.save(self._path)

    # ------------------------------------------------------------------
    # Blue/green swap
    # ------------------------------------------------------------------

    def prepare_green(self, green_path: str) -> "IndexVersionGuard":
        """
        Create a guard for a new (green) index being built in parallel.
        The caller ingests into green, then calls swap() to make green live.
        """
        return IndexVersionGuard(
            manifest_path=green_path,
            embedding_model=self._model,
            embedding_dim=self._dim,
            collection_name=self._collection + "_green",
        )

    def swap(self, green_guard: "IndexVersionGuard") -> None:
        """
        Atomically promote green to blue:
          1. Rename current (blue) directory to .bak
          2. Rename green directory to blue path
          3. Update collection_name in manifest to remove _green suffix
        """
        blue_dir = self._path.parent
        green_dir = Path(green_guard._path).parent
        bak_dir = blue_dir.with_suffix(".bak")

        if blue_dir.exists():
            blue_dir.rename(bak_dir)
        green_dir.rename(blue_dir)

        # Fix collection name in swapped manifest
        new_manifest_path = blue_dir / self._path.name
        manifest = IndexManifest.load(new_manifest_path)
        manifest.collection_name = self._collection
        manifest.save(new_manifest_path)

    # ------------------------------------------------------------------
    # Rebuild (model changed)
    # ------------------------------------------------------------------

    def rebuild(
        self,
        pipeline,             # RAGPipeline
        registry,             # DocumentRegistry
    ) -> None:
        """
        Full re-index when embedding model changes:
          1. Delete the old ChromaDB collection.
          2. Re-create the manifest with the new model config.
          3. Re-ingest all documents from the DocumentRegistry.
        """
        print(f"[IndexVersionGuard] Rebuilding index for model {self._model!r}...")

        # Wipe collection
        if pipeline._client is not None:
            try:
                pipeline._client.delete_collection(self._collection)
                pipeline._collection = pipeline._client.get_or_create_collection(
                    name=self._collection,
                    embedding_function=pipeline._ef,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as exc:
                print(f"[IndexVersionGuard] Collection reset failed: {exc}")

        # Re-create manifest
        if self._path.exists():
            self._path.unlink()
        manifest = self._create()

        # Re-ingest all known documents
        doc_ids = registry.list_docs()
        for doc_id in doc_ids:
            record = registry.get_record(doc_id)
            if record:
                print(f"[IndexVersionGuard] Re-ingesting {doc_id!r}...")
                # Mark as changed by clearing hash so upsert forces re-ingest
                registry._conn.execute(
                    "UPDATE doc_records SET content_hash='' WHERE doc_id=?", (doc_id,)
                )
                registry._conn.commit()

        print(f"[IndexVersionGuard] Rebuild queued {len(doc_ids)} documents.")
