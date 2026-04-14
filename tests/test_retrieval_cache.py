"""Tests for context_harness/retrieval_cache.py"""
import asyncio
import pytest
from context_harness.retrieval_cache import RetrievalCache, CachedRAGPipeline, _cache_key
from context_harness.rag_pipeline import RAGPipeline, ChunkingStrategy, Chunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pipeline():
    p = RAGPipeline(use_chromadb=False, chunking_strategy=ChunkingStrategy.PARAGRAPH)
    p.ingest("Dumbledore is headmaster of Hogwarts.", doc_id="hp1")
    p.ingest("Harry Potter defeated Voldemort at the Battle of Hogwarts.", doc_id="hp2")
    return p


@pytest.fixture
def cache(pipeline):
    return RetrievalCache(pipeline, max_size=8, ttl_seconds=60.0)


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

def test_cache_key_deterministic():
    k1 = _cache_key("who is dumbledore", 5, "hp_lore")
    k2 = _cache_key("who is dumbledore", 5, "hp_lore")
    assert k1 == k2


def test_cache_key_different_queries():
    k1 = _cache_key("dumbledore", 5, "hp_lore")
    k2 = _cache_key("voldemort", 5, "hp_lore")
    assert k1 != k2


def test_cache_key_different_top_k():
    k1 = _cache_key("dumbledore", 3, "hp_lore")
    k2 = _cache_key("dumbledore", 5, "hp_lore")
    assert k1 != k2


# ---------------------------------------------------------------------------
# RetrievalCache — miss and hit
# ---------------------------------------------------------------------------

async def test_first_call_is_miss(cache):
    chunks = await cache.retrieve("dumbledore", top_k=2)
    assert isinstance(chunks, list)
    assert cache._misses == 1
    assert cache._hits == 0


async def test_second_call_is_hit(cache):
    await cache.retrieve("dumbledore", top_k=2)
    await cache.retrieve("dumbledore", top_k=2)
    assert cache._hits == 1
    assert cache._misses == 1


async def test_different_query_is_miss(cache):
    await cache.retrieve("dumbledore", top_k=2)
    await cache.retrieve("voldemort", top_k=2)
    assert cache._misses == 2


async def test_hit_rate_after_hits(cache):
    await cache.retrieve("dumbledore", top_k=2)  # miss
    await cache.retrieve("dumbledore", top_k=2)  # hit
    await cache.retrieve("dumbledore", top_k=2)  # hit
    assert cache.hit_rate == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# RetrievalCache — invalidation
# ---------------------------------------------------------------------------

async def test_invalidate_collection_removes_entries(cache, pipeline):
    await cache.retrieve("dumbledore", top_k=2)
    assert cache.size == 1
    removed = cache.invalidate_collection(pipeline.collection_name)
    assert removed == 1
    assert cache.size == 0


async def test_after_invalidation_next_call_is_miss(cache):
    await cache.retrieve("dumbledore", top_k=2)
    cache.invalidate_collection(cache._pipeline.collection_name)
    cache._hits = 0  # reset counters for clean assertion
    cache._misses = 0
    await cache.retrieve("dumbledore", top_k=2)
    assert cache._misses == 1


async def test_invalidate_specific_query(cache):
    await cache.retrieve("dumbledore", top_k=2)
    removed = cache.invalidate_query("dumbledore", top_k=2)
    assert removed is True
    assert cache.size == 0


async def test_invalidate_nonexistent_query(cache):
    removed = cache.invalidate_query("nonexistent query xyz", top_k=5)
    assert removed is False


# ---------------------------------------------------------------------------
# RetrievalCache — eviction and size
# ---------------------------------------------------------------------------

async def test_max_size_respected(pipeline):
    tiny_cache = RetrievalCache(pipeline, max_size=2, ttl_seconds=60.0)
    await tiny_cache.retrieve("dumbledore", top_k=2)
    await tiny_cache.retrieve("voldemort", top_k=2)
    await tiny_cache.retrieve("hogwarts", top_k=2)   # should evict oldest
    assert tiny_cache.size <= 2


async def test_clear_empties_cache(cache):
    await cache.retrieve("dumbledore", top_k=2)
    cache.clear()
    assert cache.size == 0


# ---------------------------------------------------------------------------
# CachedRAGPipeline
# ---------------------------------------------------------------------------

async def test_cached_pipeline_retrieve(pipeline):
    cached = CachedRAGPipeline(pipeline, max_size=16, ttl_seconds=60.0)
    chunks = await cached.retrieve("dumbledore", top_k=2)
    assert isinstance(chunks, list)


async def test_cached_pipeline_ingest_invalidates(pipeline):
    cached = CachedRAGPipeline(pipeline, max_size=16, ttl_seconds=60.0)
    await cached.retrieve("dumbledore", top_k=2)
    size_before = cached._cache.size
    cached.ingest("new-doc", "New lore about the Room of Requirement.")
    assert cached._cache.size < size_before or cached._cache.size == 0


async def test_cached_pipeline_cache_stats(pipeline):
    cached = CachedRAGPipeline(pipeline, max_size=16, ttl_seconds=60.0)
    await cached.retrieve("dumbledore", top_k=2)
    stats = cached.cache_stats()
    assert "hit_rate" in stats
    assert "size" in stats
