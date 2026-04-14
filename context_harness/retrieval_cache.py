"""
Retrieval Cache
===============
Fixes deepTutor Weakness #7:
  "LlamaIndex loads full index into memory per search — StorageContext.from_defaults()
   + load_index_from_storage() called on every query. For large KBs this is significant
   latency and memory overhead."

Solution:
  - RetrievalCache wraps RAGPipeline.retrieve() with an LRU cache keyed on
    (query, top_k, collection_name).
  - Each cache entry has a configurable TTL (default 5 minutes).
  - Cache size is bounded (default 512 entries) to cap memory use.
  - Thread-safe (asyncio.Lock) for concurrent requests.
  - Cache statistics are exposed for monitoring (hit rate, evictions, etc.).
  - The cache can be invalidated per-collection when documents are updated
    (called from DocumentRegistry.upsert() after a successful write).

Why not just @lru_cache?
  @lru_cache is synchronous and has no TTL. Async retrieval with TTL + targeted
  invalidation requires a hand-rolled cache.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Coroutine, List, Optional

from .rag_pipeline import Chunk, RAGPipeline


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    chunks: List[Chunk]
    created_at: float = field(default_factory=time.monotonic)
    hit_count: int = 0

    def is_expired(self, ttl_seconds: float) -> bool:
        return (time.monotonic() - self.created_at) > ttl_seconds


# ---------------------------------------------------------------------------
# Retrieval Cache
# ---------------------------------------------------------------------------

class RetrievalCache:
    """
    LRU cache with TTL for RAG retrieval results.

    Usage:
        cache = RetrievalCache(pipeline, max_size=512, ttl_seconds=300)
        chunks = await cache.retrieve("Who is Dumbledore?", top_k=5)
        cache.invalidate_collection("hp_lore")   # called after doc update
    """

    def __init__(
        self,
        pipeline: RAGPipeline,
        max_size: int = 512,
        ttl_seconds: float = 300.0,
    ) -> None:
        self._pipeline = pipeline
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()

        # Stats
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    # ------------------------------------------------------------------
    # Retrieve (cached)
    # ------------------------------------------------------------------

    async def retrieve(self, query: str, top_k: int = 5) -> List[Chunk]:
        """
        Return cached chunks if available and fresh; otherwise call the pipeline
        and cache the result.
        """
        key = _cache_key(query, top_k, self._pipeline.collection_name)

        async with self._lock:
            entry = self._store.get(key)
            if entry is not None and not entry.is_expired(self._ttl):
                # Cache hit: move to end (LRU order), bump hit count
                self._store.move_to_end(key)
                entry.hit_count += 1
                self._hits += 1
                return entry.chunks

        # Cache miss: call pipeline (outside lock so other requests can proceed)
        self._misses += 1
        chunks = self._pipeline.retrieve(query, top_k=top_k)

        async with self._lock:
            self._store[key] = CacheEntry(chunks=chunks)
            self._store.move_to_end(key)
            self._evict_if_needed()

        return chunks

    # ------------------------------------------------------------------
    # Invalidation
    # ------------------------------------------------------------------

    def invalidate_collection(self, collection_name: str) -> int:
        """
        Remove all cache entries for a given collection.
        Called when documents in that collection are updated or deleted.
        Returns the number of entries removed.
        """
        prefix = _collection_prefix(collection_name)
        to_remove = [k for k in self._store if k.startswith(prefix)]
        for k in to_remove:
            del self._store[k]
        self._evictions += len(to_remove)
        return len(to_remove)

    def invalidate_query(self, query: str, top_k: int = 5) -> bool:
        """Remove a specific cached query. Returns True if it was present."""
        key = _cache_key(query, top_k, self._pipeline.collection_name)
        if key in self._store:
            del self._store[key]
            self._evictions += 1
            return True
        return False

    def clear(self) -> None:
        self._store.clear()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total else 0.0

    def stats(self) -> dict:
        return {
            "size": self.size,
            "max_size": self._max_size,
            "ttl_seconds": self._ttl,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": round(self.hit_rate, 4),
        }

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def _evict_if_needed(self) -> None:
        """Evict LRU entries and expired entries until within max_size."""
        # First pass: drop expired entries
        expired = [k for k, v in self._store.items() if v.is_expired(self._ttl)]
        for k in expired:
            del self._store[k]
            self._evictions += 1

        # Second pass: LRU eviction if still over limit
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)   # remove LRU (oldest)
            self._evictions += 1


# ---------------------------------------------------------------------------
# Cached RAGPipeline wrapper
# ---------------------------------------------------------------------------

class CachedRAGPipeline:
    """
    Drop-in wrapper around RAGPipeline that caches retrieve() calls and
    auto-invalidates on ingest/delete.

    This is the recommended replacement for direct RAGPipeline use in production.

    Usage:
        pipeline = CachedRAGPipeline(
            RAGPipeline(collection_name="hp_lore"),
            max_size=512, ttl_seconds=300,
        )
        chunks = await pipeline.retrieve("Who is Dumbledore?")
        pipeline.ingest("hp1", long_text)  # invalidates cache for hp_lore
    """

    def __init__(
        self,
        pipeline: RAGPipeline,
        max_size: int = 512,
        ttl_seconds: float = 300.0,
    ) -> None:
        self._pipeline = pipeline
        self._cache = RetrievalCache(pipeline, max_size=max_size, ttl_seconds=ttl_seconds)

    async def retrieve(self, query: str, top_k: int = 5) -> List[Chunk]:
        return await self._cache.retrieve(query, top_k)

    def ingest(self, doc_id: str, text: str, **metadata) -> List[Chunk]:
        chunks = self._pipeline.ingest(doc_id, text, **metadata)
        # Invalidate cache: new chunks may change rankings for existing queries
        self._cache.invalidate_collection(self._pipeline.collection_name)
        return chunks

    def cache_stats(self) -> dict:
        return self._cache.stats()

    def __getattr__(self, name: str):
        """Proxy everything else to the underlying pipeline."""
        return getattr(self._pipeline, name)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collection_prefix(collection_name: str) -> str:
    """8-char prefix shared by every cache key for this collection."""
    return hashlib.sha256(collection_name.encode()).hexdigest()[:8]


def _cache_key(query: str, top_k: int, collection: str) -> str:
    # Key = collection_prefix + hash(topk::query)
    # This ensures key.startswith(collection_prefix) for bulk invalidation.
    prefix = _collection_prefix(collection)
    rest = hashlib.sha256(f"{top_k}::{query.strip().lower()}".encode()).hexdigest()
    return f"{prefix}{rest}"
