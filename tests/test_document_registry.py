"""Tests for context_harness/document_registry.py"""
import pytest
from context_harness.document_registry import DocumentRegistry, _sha256
from context_harness.rag_pipeline import RAGPipeline, ChunkingStrategy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def registry(tmp_path):
    pipeline = RAGPipeline(use_chromadb=False, chunking_strategy=ChunkingStrategy.PARAGRAPH)
    return DocumentRegistry(pipeline, db_path=str(tmp_path / "registry.db"))


# ---------------------------------------------------------------------------
# SHA-256 helper
# ---------------------------------------------------------------------------

def test_sha256_deterministic():
    assert _sha256("hello") == _sha256("hello")


def test_sha256_different_inputs():
    assert _sha256("hello") != _sha256("world")


def test_sha256_returns_hex_string():
    h = _sha256("test")
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# DocumentRegistry — upsert
# ---------------------------------------------------------------------------

def test_upsert_new_document(registry):
    result = registry.upsert("hp1", "Dumbledore leads Hogwarts School.\n\nHarry defeated Voldemort.")
    assert result.action == "created"
    assert result.doc_id == "hp1"
    assert result.chunk_count >= 1


def test_upsert_unchanged_document_skipped(registry):
    text = "Hermione is the brightest witch of her age."
    registry.upsert("hp2", text)
    result = registry.upsert("hp2", text)
    assert result.action == "skipped"


def test_upsert_changed_document_updated(registry):
    registry.upsert("hp3", "Original content about Hogwarts.")
    result = registry.upsert("hp3", "Completely different content about Diagon Alley.")
    assert result.action == "updated"


def test_upsert_updated_replaces_chunks(registry):
    registry.upsert("hp4", "First version of the document.")
    first_record = registry.get_record("hp4")
    first_ids = set(first_record.chunk_ids)

    registry.upsert("hp4", "Entirely new content about the Forbidden Forest.")
    second_record = registry.get_record("hp4")
    second_ids = set(second_record.chunk_ids)

    # New chunk IDs should be generated
    assert first_ids != second_ids


# ---------------------------------------------------------------------------
# DocumentRegistry — delete
# ---------------------------------------------------------------------------

def test_delete_existing_document(registry):
    registry.upsert("hp5", "Some lore text.")
    removed = registry.delete("hp5")
    assert removed is True
    assert registry.get_record("hp5") is None


def test_delete_nonexistent_document(registry):
    removed = registry.delete("does-not-exist")
    assert removed is False


# ---------------------------------------------------------------------------
# DocumentRegistry — list and stats
# ---------------------------------------------------------------------------

def test_list_docs(registry):
    registry.upsert("doc-a", "Text A.")
    registry.upsert("doc-b", "Text B.")
    docs = registry.list_docs()
    assert "doc-a" in docs
    assert "doc-b" in docs


def test_stats(registry):
    registry.upsert("s1", "Content one.\n\nContent two.")
    stats = registry.stats()
    assert stats["doc_count"] >= 1
    assert stats["total_chunks"] >= 1


def test_get_record_after_upsert(registry):
    registry.upsert("hp6", "Voldemort sought immortality through Horcruxes.")
    record = registry.get_record("hp6")
    assert record is not None
    assert record.doc_id == "hp6"
    assert record.chunk_count >= 1
    assert len(record.content_hash) == 64
