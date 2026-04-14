"""
Integration tests for ChromaDB — use real embeddings via DefaultEmbeddingFunction.
These tests exercise the full ingest → embed → retrieve path with a live ChromaDB
EphemeralClient (no disk, no HuggingFace downloads required after first run).
"""

import pytest
from context_harness.ingest_lore import build_pipeline, parse_lore_file, LORE_FILE
from context_harness.document_registry import DocumentRegistry
from context_harness.rag_pipeline import ChunkingStrategy


@pytest.fixture(scope="module")
def loaded_pipeline():
    """Ingest all 10 HP lore documents into an ephemeral ChromaDB."""
    pipeline = build_pipeline(persist=False)
    registry = DocumentRegistry(
        pipeline=pipeline,
        db_path=":memory:",
        embedding_model="all-MiniLM-L6-v2",
    )
    docs = parse_lore_file(LORE_FILE)
    for doc_id, text in docs:
        registry.upsert(doc_id, text, source="test")
    return pipeline, registry


def test_chromadb_indexed_all_documents(loaded_pipeline):
    pipeline, registry = loaded_pipeline
    assert pipeline.count() == 10


def test_registry_tracks_all_documents(loaded_pipeline):
    _, registry = loaded_pipeline
    stats = registry.stats()
    assert stats["doc_count"] == 10
    assert stats["total_chunks"] == 10


def test_retrieve_dumbledore_query(loaded_pipeline):
    pipeline, _ = loaded_pipeline
    results = pipeline.retrieve("Who is the headmaster of Hogwarts?", top_k=3)
    assert len(results) > 0
    top_text = results[0].text.lower()
    assert "dumbledore" in top_text or "headmaster" in top_text


def test_retrieve_horcrux_query(loaded_pipeline):
    pipeline, _ = loaded_pipeline
    results = pipeline.retrieve("What are Horcruxes and how are they destroyed?", top_k=3)
    assert len(results) > 0
    combined = " ".join(r.text.lower() for r in results)
    assert "horcrux" in combined


def test_retrieve_voldemort_query(loaded_pipeline):
    pipeline, _ = loaded_pipeline
    results = pipeline.retrieve("dark lord who killed Harry Potter's parents", top_k=3)
    assert len(results) > 0
    combined = " ".join(r.text.lower() for r in results)
    assert "voldemort" in combined or "riddle" in combined


def test_retrieve_returns_scores(loaded_pipeline):
    pipeline, _ = loaded_pipeline
    results = pipeline.retrieve("Deathly Hallows invisibility cloak", top_k=3)
    for chunk in results:
        assert 0.0 <= chunk.score <= 1.0


def test_retrieve_top_k_respected(loaded_pipeline):
    pipeline, _ = loaded_pipeline
    results = pipeline.retrieve("magic school", top_k=2)
    assert len(results) <= 2


def test_upsert_skip_on_unchanged(loaded_pipeline):
    pipeline, registry = loaded_pipeline
    docs = parse_lore_file(LORE_FILE)
    doc_id, text = docs[0]
    result = registry.upsert(doc_id, text)
    assert result.action == "skipped"


def test_upsert_update_on_changed_content(loaded_pipeline):
    pipeline, registry = loaded_pipeline
    result = registry.upsert(
        "albus-dumbledore",
        "Albus Dumbledore was a very powerful wizard and headmaster of Hogwarts — updated text.",
    )
    assert result.action == "updated"
    # Restore original
    docs = {did: txt for did, txt in parse_lore_file(LORE_FILE)}
    registry.upsert("albus-dumbledore", docs["albus-dumbledore"])


def test_mmr_rerank_returns_diverse_results(loaded_pipeline):
    pipeline, _ = loaded_pipeline
    candidates = pipeline.retrieve("wizard magic spell", top_k=5)
    reranked = pipeline.mmr_rerank("wizard magic spell", candidates, lambda_=0.5, k=3)
    assert len(reranked) <= 3
    assert len(reranked) > 0
