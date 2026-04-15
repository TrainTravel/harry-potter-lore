"""Tests for context_harness/context_assembler.py"""
import uuid
import pytest
from context_harness.context_assembler import ContextAssembler, AssemblyStrategy, _jaccard
from context_harness.rag_pipeline import Chunk, ScoredChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chunk(text: str, score: float = 0.5, doc_id: str = "hp1") -> ScoredChunk:
    chunk = Chunk(
        text=text,
        metadata={"doc_id": doc_id},
        chunk_id=str(uuid.uuid4()),
    )
    return ScoredChunk(chunk=chunk, score=score)


# ---------------------------------------------------------------------------
# Jaccard similarity
# ---------------------------------------------------------------------------

def test_jaccard_identical():
    assert _jaccard("hello world", "hello world") == 1.0


def test_jaccard_disjoint():
    assert _jaccard("abc def", "xyz qrs") == 0.0


def test_jaccard_partial():
    score = _jaccard("hello world foo", "hello bar baz")
    assert 0.0 < score < 1.0


def test_jaccard_empty_strings():
    assert _jaccard("", "") == 1.0


# ---------------------------------------------------------------------------
# ContextAssembler — filtering
# ---------------------------------------------------------------------------

def test_min_score_filter():
    assembler = ContextAssembler(min_score=0.5)
    chunks = [make_chunk("relevant", score=0.8), make_chunk("irrelevant", score=0.1)]
    result = assembler.assemble(chunks)
    assert "irrelevant" not in result


def test_max_chunks_respected():
    assembler = ContextAssembler(max_chunks=2, strategy=AssemblyStrategy.NAIVE)
    chunks = [make_chunk(f"chunk {i}", score=float(i)) for i in range(5)]
    result = assembler.assemble(chunks)
    # At most 2 chunks should appear
    count = sum(1 for i in range(5) if f"chunk {i}" in result)
    assert count <= 2


def test_dedup_removes_near_identical_chunks():
    assembler = ContextAssembler(dedup_threshold=0.9, strategy=AssemblyStrategy.NAIVE)
    text = "Dumbledore is headmaster of Hogwarts School of Witchcraft"
    chunks = [make_chunk(text, score=0.9), make_chunk(text, score=0.8)]
    result = assembler.assemble(chunks)
    # Only one copy should appear
    assert result.count(text) == 1


# ---------------------------------------------------------------------------
# ContextAssembler — strategies
# ---------------------------------------------------------------------------

def test_naive_strategy_returns_string():
    assembler = ContextAssembler(strategy=AssemblyStrategy.NAIVE)
    chunks = [make_chunk("Dumbledore is wise.", score=0.9)]
    result = assembler.assemble(chunks)
    assert isinstance(result, str)
    assert "Dumbledore" in result


def test_relevance_strategy_includes_score():
    assembler = ContextAssembler(strategy=AssemblyStrategy.RELEVANCE)
    chunks = [make_chunk("Lore text about magic.", score=0.75)]
    result = assembler.assemble(chunks)
    assert "0.75" in result


def test_sandwich_strategy_repeats_best_chunk():
    assembler = ContextAssembler(strategy=AssemblyStrategy.SANDWICH)
    best = make_chunk("Most relevant passage.", score=0.99)
    other = make_chunk("Less relevant passage.", score=0.5)
    result = assembler.assemble([best, other])
    assert result.count("Most relevant passage.") == 2


def test_citation_strategy_includes_source_numbers():
    assembler = ContextAssembler(strategy=AssemblyStrategy.CITATION)
    chunks = [make_chunk("Passage one.", score=0.9), make_chunk("Passage two.", score=0.8)]
    result = assembler.assemble(chunks)
    assert "[1]" in result
    assert "[2]" in result


def test_empty_chunks_returns_empty_string():
    assembler = ContextAssembler()
    result = assembler.assemble([])
    assert result == ""


def test_assemble_as_messages_empty():
    assembler = ContextAssembler()
    msgs = assembler.assemble_as_messages([])
    assert msgs == []


def test_assemble_as_messages_non_empty():
    assembler = ContextAssembler(strategy=AssemblyStrategy.NAIVE)
    chunks = [make_chunk("Hogwarts is a school.", score=0.8)]
    msgs = assembler.assemble_as_messages(chunks)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "retrieval"
    assert "Hogwarts" in msgs[0]["content"]
