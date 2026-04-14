"""Tests for context_harness/index_version_guard.py"""
import json
import pytest
from context_harness.index_version_guard import (
    IndexVersionGuard, IndexManifest, IndexVersionError, IndexDimensionError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def guard(tmp_path):
    return IndexVersionGuard(
        manifest_path=str(tmp_path / "index_manifest.json"),
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim=384,
        collection_name="hp_lore",
    )


# ---------------------------------------------------------------------------
# First-time creation
# ---------------------------------------------------------------------------

def test_ensure_creates_manifest_if_missing(guard, tmp_path):
    manifest = guard.ensure()
    assert (tmp_path / "index_manifest.json").exists()
    assert manifest.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert manifest.embedding_dim == 384


def test_created_manifest_has_correct_fields(guard):
    manifest = guard.ensure()
    assert manifest.schema_version == 1
    assert manifest.collection_name == "hp_lore"
    assert manifest.doc_count == 0
    assert manifest.chunk_count == 0


# ---------------------------------------------------------------------------
# Validation — happy path
# ---------------------------------------------------------------------------

def test_validate_passes_when_manifest_matches(guard):
    guard.ensure()          # create
    manifest = guard.validate()  # validate same config
    assert manifest.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Validation — model mismatch
# ---------------------------------------------------------------------------

def test_validate_raises_on_model_mismatch(tmp_path):
    guard_a = IndexVersionGuard(
        manifest_path=str(tmp_path / "m.json"),
        embedding_model="model-a", embedding_dim=384, collection_name="hp_lore",
    )
    guard_a.ensure()

    guard_b = IndexVersionGuard(
        manifest_path=str(tmp_path / "m.json"),
        embedding_model="model-b", embedding_dim=384, collection_name="hp_lore",
    )
    with pytest.raises(IndexVersionError):
        guard_b.validate()


# ---------------------------------------------------------------------------
# Validation — dimension mismatch
# ---------------------------------------------------------------------------

def test_validate_raises_on_dimension_mismatch(tmp_path):
    guard_a = IndexVersionGuard(
        manifest_path=str(tmp_path / "m.json"),
        embedding_model="model-x", embedding_dim=384, collection_name="hp_lore",
    )
    guard_a.ensure()

    guard_b = IndexVersionGuard(
        manifest_path=str(tmp_path / "m.json"),
        embedding_model="model-x", embedding_dim=768, collection_name="hp_lore",
    )
    with pytest.raises((IndexVersionError, IndexDimensionError)):
        guard_b.validate()


# ---------------------------------------------------------------------------
# Stats update
# ---------------------------------------------------------------------------

def test_update_stats_persists(guard, tmp_path):
    guard.ensure()
    guard.update_stats(doc_count=5, chunk_count=42)
    raw = json.loads((tmp_path / "index_manifest.json").read_text())
    assert raw["doc_count"] == 5
    assert raw["chunk_count"] == 42


def test_update_stats_bumps_updated_at(guard):
    manifest_before = guard.ensure()
    import time; time.sleep(0.01)
    guard.update_stats(doc_count=1, chunk_count=10)
    manifest_after = guard.validate()
    assert manifest_after.updated_at >= manifest_before.updated_at
