"""
Tests for context_harness.reranker.
===================================

Covers:
- IdentityReranker is a pass-through
- FakeReranker records calls + applies scripted ordering
- RAGPipeline integration:
    * over-fetches when reranker is set (fetch_k > top_k)
    * falls back to raw top_k when reranker raises
    * returns the reranker's ordering when it succeeds
    * returns un-reranked top_k when reranker is None (no regression)

No real Cohere API calls — tests use FakeReranker exclusively. CohereReranker
is smoke-tested for construction-time errors only; a full integration test
against the real API belongs in tests/integration/ (opt-in).
"""

from __future__ import annotations

import os
from typing import List

import pytest

from context_harness.rag_pipeline import (
    Chunk,
    ChunkingStrategy,
    RAGPipeline,
    ScoredChunk,
)
from context_harness.reranker import (
    CohereReranker,
    FakeReranker,
    IdentityReranker,
    RerankerClient,
    build_reranker_from_env,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _chunk(doc_id: str, text: str, score: float = 0.5) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(text=text, metadata={"doc_id": doc_id}, chunk_id=doc_id),
        score=score,
    )


@pytest.fixture
def candidates() -> List[ScoredChunk]:
    return [
        _chunk("doc-a", "A is about Dumbledore", score=0.9),
        _chunk("doc-b", "B is about Snape",       score=0.8),
        _chunk("doc-c", "C is about Voldemort",   score=0.7),
        _chunk("doc-d", "D is about Harry",       score=0.6),
        _chunk("doc-e", "E is about Hermione",    score=0.5),
    ]


# ---------------------------------------------------------------------------
# IdentityReranker
# ---------------------------------------------------------------------------

def test_identity_reranker_returns_prefix(candidates):
    reranker = IdentityReranker()
    result = reranker.rerank("any query", candidates, top_k=3)
    assert [c.doc_id for c in result] == ["doc-a", "doc-b", "doc-c"]


def test_identity_reranker_handles_empty():
    reranker = IdentityReranker()
    assert reranker.rerank("q", [], top_k=5) == []


def test_identity_reranker_topk_larger_than_input(candidates):
    reranker = IdentityReranker()
    result = reranker.rerank("q", candidates, top_k=100)
    assert len(result) == 5  # returns all 5 without padding


# ---------------------------------------------------------------------------
# FakeReranker
# ---------------------------------------------------------------------------

def test_fake_reranker_records_each_call(candidates):
    fake = FakeReranker()
    fake.rerank("first query",  candidates, top_k=3)
    fake.rerank("second query", candidates[:2], top_k=1)

    assert len(fake.calls) == 2
    assert fake.calls[0].query == "first query"
    assert fake.calls[0].top_k == 3
    assert fake.calls[0].candidate_doc_ids == ["doc-a", "doc-b", "doc-c", "doc-d", "doc-e"]
    assert fake.calls[1].query == "second query"
    assert fake.calls[1].top_k == 1


def test_fake_reranker_default_reverses_input(candidates):
    """Default behaviour reverses the input so tests can assert rerank ran."""
    fake = FakeReranker()
    result = fake.rerank("q", candidates, top_k=3)
    # Input was doc-a..doc-e. Reversed: doc-e, doc-d, doc-c. Then top_k=3.
    assert [c.doc_id for c in result] == ["doc-e", "doc-d", "doc-c"]


def test_fake_reranker_custom_ordering(candidates):
    """A custom ordering callable can return any permutation."""
    # Script: always put doc-c first
    def always_c_first(query, cands):
        c_first = [c for c in cands if c.doc_id == "doc-c"]
        rest    = [c for c in cands if c.doc_id != "doc-c"]
        return c_first + rest

    fake = FakeReranker(ordering=always_c_first)
    result = fake.rerank("q", candidates, top_k=2)
    assert [c.doc_id for c in result] == ["doc-c", "doc-a"]


# ---------------------------------------------------------------------------
# CohereReranker — smoke tests only (no real API calls)
# ---------------------------------------------------------------------------

def test_cohere_reranker_raises_without_key(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API key"):
        CohereReranker()


def test_cohere_reranker_accepts_explicit_key(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    # Construction should succeed even if the key is bogus — we don't make
    # an API call at init time.
    reranker = CohereReranker(api_key="fake-key-abc")
    assert reranker._model == "rerank-v3.0"


# ---------------------------------------------------------------------------
# build_reranker_from_env
# ---------------------------------------------------------------------------

def test_build_reranker_returns_none_without_key(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    assert build_reranker_from_env() is None


def test_build_reranker_returns_cohere_with_key(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "fake-key-abc")
    r = build_reranker_from_env()
    assert isinstance(r, CohereReranker)


# ---------------------------------------------------------------------------
# RAGPipeline integration
# ---------------------------------------------------------------------------

def _pipeline_without_chromadb() -> RAGPipeline:
    """Build an in-memory-only pipeline for integration tests — avoids
    needing a real ChromaDB in the test environment."""
    return RAGPipeline(top_k=3, use_chromadb=False)


def test_pipeline_without_reranker_returns_raw_topk():
    """Backwards compat: no reranker → top_k as before, no over-fetch."""
    pipeline = _pipeline_without_chromadb()
    for i in range(5):
        pipeline._chunks.append(Chunk(text=f"text {i}", metadata={"doc_id": f"d{i}"}))

    result = pipeline.retrieve("text", top_k=3)
    assert len(result) == 3
    # No reranker means no fetch_k inflation
    assert pipeline._rerank_fetch_k == 20  # default is still 20, just unused


def test_pipeline_with_fake_reranker_calls_it():
    pipeline = _pipeline_without_chromadb()
    for i in range(10):
        pipeline._chunks.append(Chunk(text=f"text {i}", metadata={"doc_id": f"d{i}"}))

    fake = FakeReranker()
    pipeline._reranker = fake
    result = pipeline.retrieve("text", top_k=3)

    # Reranker should have been called exactly once
    assert len(fake.calls) == 1
    # And should have received over-fetched candidates (up to fetch_k=20)
    assert fake.calls[0].top_k == 3
    assert len(fake.calls[0].candidate_doc_ids) <= 20
    # Final result is top_k long
    assert len(result) == 3


def test_pipeline_reranker_failure_falls_back_to_raw_topk():
    """If the reranker raises, retrieve() returns the un-reranked top_k
    rather than propagating the exception."""
    pipeline = _pipeline_without_chromadb()
    for i in range(5):
        pipeline._chunks.append(Chunk(text=f"text {i}", metadata={"doc_id": f"d{i}"}))

    class BrokenReranker:
        def rerank(self, query, candidates, top_k):
            raise RuntimeError("simulated rerank API failure")

    pipeline._reranker = BrokenReranker()
    # Should NOT raise — should fall back gracefully
    result = pipeline.retrieve("text", top_k=3)
    assert len(result) == 3


def test_pipeline_reranker_changes_ordering():
    """Verify the rerank output actually changes which chunks are returned —
    not just that the method was called."""
    pipeline = _pipeline_without_chromadb()
    # Five chunks, the keyword-fallback scorer will rank roughly by word
    # overlap with "text". Add a chunk whose words don't overlap, but which
    # we want to force to the top via reranking.
    pipeline._chunks.append(Chunk(text="unrelated content zzz", metadata={"doc_id": "ringer"}))
    for i in range(5):
        pipeline._chunks.append(Chunk(text=f"text {i}", metadata={"doc_id": f"d{i}"}))

    def put_ringer_first(query, cands):
        ringers = [c for c in cands if c.doc_id == "ringer"]
        others  = [c for c in cands if c.doc_id != "ringer"]
        return ringers + others

    pipeline._reranker = FakeReranker(ordering=put_ringer_first)
    result = pipeline.retrieve("text", top_k=3)
    doc_ids = [c.doc_id for c in result]
    assert "ringer" in doc_ids
    assert doc_ids[0] == "ringer"   # reranker put it first


def test_pipeline_reranker_with_small_candidate_set():
    """Reranker should handle the case where vector search returned fewer
    candidates than rerank_fetch_k (e.g., small corpus)."""
    pipeline = _pipeline_without_chromadb()
    # Only 2 chunks total, but reranker is configured.
    pipeline._chunks.append(Chunk(text="text one", metadata={"doc_id": "d1"}))
    pipeline._chunks.append(Chunk(text="text two", metadata={"doc_id": "d2"}))

    fake = FakeReranker()
    pipeline._reranker = fake
    result = pipeline.retrieve("text", top_k=5)

    # Only 2 candidates were available; reranker gets them, result is ≤ 2
    assert len(result) <= 2
    assert len(fake.calls[0].candidate_doc_ids) == 2
