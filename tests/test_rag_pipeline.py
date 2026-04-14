"""Tests for context_harness/rag_pipeline.py"""
import pytest
from context_harness.rag_pipeline import (
    RAGPipeline, Chunk, ChunkingStrategy,
    _fixed_chunk, _sentence_chunk, _paragraph_chunk, _recursive_chunk,
)


# ---------------------------------------------------------------------------
# Chunkers (pure functions — no I/O)
# ---------------------------------------------------------------------------

def test_fixed_chunk_single_chunk():
    words = ["word"] * 10
    result = _fixed_chunk(" ".join(words), size=20, overlap=0)
    assert len(result) == 1
    assert result[0] == " ".join(words)


def test_fixed_chunk_overlap_produces_more_chunks():
    words = ["w"] * 20
    no_overlap = _fixed_chunk(" ".join(words), size=10, overlap=0)
    with_overlap = _fixed_chunk(" ".join(words), size=10, overlap=5)
    assert len(with_overlap) > len(no_overlap)


def test_sentence_chunk_splits_on_punctuation():
    text = "Dumbledore is headmaster. Harry is a student. Voldemort is the villain."
    chunks = _sentence_chunk(text, max_sentences=1)
    assert len(chunks) == 3


def test_paragraph_chunk_splits_on_blank_lines():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = _paragraph_chunk(text)
    assert len(chunks) == 3


def test_paragraph_chunk_ignores_empty_paragraphs():
    text = "Para one.\n\n\n\nPara two."
    chunks = _paragraph_chunk(text)
    assert all(c.strip() for c in chunks)


def test_recursive_chunk_handles_short_paragraphs():
    text = "Short para.\n\nAnother short para."
    chunks = _recursive_chunk(text, size=50)
    assert len(chunks) == 2


def test_recursive_chunk_splits_long_paragraphs():
    long_para = " ".join(["word"] * 600)
    chunks = _recursive_chunk(long_para, size=300)
    assert len(chunks) > 1


# ---------------------------------------------------------------------------
# RAGPipeline — in-memory (no ChromaDB)
# ---------------------------------------------------------------------------

@pytest.fixture
def pipeline():
    return RAGPipeline(use_chromadb=False, chunking_strategy=ChunkingStrategy.PARAGRAPH)


def test_pipeline_ingest_returns_chunks(pipeline):
    text = "Dumbledore is headmaster.\n\nHarry defeated Voldemort."
    chunks = pipeline.ingest(text, doc_id="hp1")
    assert len(chunks) >= 1
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.doc_id == "hp1" for c in chunks)


def test_pipeline_count_after_ingest(pipeline):
    pipeline.ingest("Some lore text here.", doc_id="doc1")
    assert pipeline.count() >= 1


def test_pipeline_retrieve_returns_chunks(pipeline):
    pipeline.ingest("Dumbledore is the greatest wizard of his age.", doc_id="hp1")
    pipeline.ingest("Harry Potter survived the killing curse.", doc_id="hp2")
    results = pipeline.retrieve("Dumbledore wizard")
    assert len(results) >= 1
    assert all(isinstance(c, Chunk) for c in results)


def test_pipeline_retrieve_scores_are_nonnegative(pipeline):
    pipeline.ingest("Hogwarts is a school for wizards.", doc_id="hp1")
    results = pipeline.retrieve("hogwarts school", top_k=3)
    assert all(c.score >= 0.0 for c in results)


def test_pipeline_retrieve_top_k_respected(pipeline):
    for i in range(10):
        pipeline.ingest(f"Lore sentence number {i} about magic.", doc_id=f"doc{i}")
    results = pipeline.retrieve("magic lore", top_k=3)
    assert len(results) <= 3


def test_pipeline_ingest_many(pipeline):
    docs = [("Dumbledore leads Hogwarts.", "hp1"), ("Harry fights Voldemort.", "hp2")]
    chunks = pipeline.ingest_many(docs)
    assert len(chunks) >= 2


def test_chunk_ids_are_unique(pipeline):
    chunks = pipeline.ingest("One.\n\nTwo.\n\nThree.", doc_id="hp1")
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_mmr_rerank_without_embeddings(pipeline):
    """MMR falls back to score sort when no embedding function available."""
    pipeline.ingest("Dumbledore is wise.", doc_id="d1")
    pipeline.ingest("Harry is brave.", doc_id="d2")
    candidates = pipeline.retrieve("magic", top_k=5)
    reranked = pipeline.mmr_rerank("magic", candidates, k=2)
    assert len(reranked) <= 2
