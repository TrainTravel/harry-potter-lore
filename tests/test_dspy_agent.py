"""Tests for context_harness/dspy_agent.py"""

import json
import pytest
import dspy
from dspy.utils import DummyLM

from context_harness.dspy_agent import (
    DeepResearchSignature, GuidedLearningSignature, DebateSignature,
    SatiricalPodcastSignature,
    DeepResearchModule, GuidedLearningModule, DebateModule,
    SatiricalPodcastModule,
    DSPyAgent, BackfillRequired,
    _write_manifest, _validate_manifest, MODES,
)
from context_harness.ingest_lore import build_pipeline, parse_lore_file, LORE_FILE


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline():
    """Ephemeral ChromaDB pipeline with all 10 HP lore docs ingested."""
    from context_harness.document_registry import DocumentRegistry
    p = build_pipeline(persist=False)
    reg = DocumentRegistry(p, db_path=":memory:")
    for doc_id, text in parse_lore_file(LORE_FILE):
        reg.upsert(doc_id, text)
    return p


@pytest.fixture(autouse=True)
def dummy_lm():
    """Configure DummyLM before every test and restore afterwards."""
    lm = DummyLM(answers=[{
        "answer": "Harry Potter defeated Voldemort.",
        "citations": "harry-potter lord-voldemort",
        "confidence": "high",
        "gaps": "none",
        "hint": "Think about what happened at the Battle of Hogwarts.",
        "next_question": "What was the role of the Elder Wand?",
        "explanation": "The concept relates to the protection of love magic.",
        "reasoning": "Step by step reasoning.",
    }])
    dspy.configure(lm=lm)
    yield lm


# ---------------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------------

def test_deep_research_signature_input_fields():
    fields = DeepResearchSignature.input_fields
    assert "question" in fields
    assert "context" in fields


def test_deep_research_signature_output_fields():
    fields = DeepResearchSignature.output_fields
    assert "answer" in fields
    assert "citations" in fields
    assert "confidence" in fields
    assert "gaps" in fields


def test_guided_learning_signature_input_fields():
    fields = GuidedLearningSignature.input_fields
    assert "question" in fields
    assert "context" in fields
    assert "past_attempts" in fields


def test_guided_learning_signature_output_fields():
    fields = GuidedLearningSignature.output_fields
    assert "hint" in fields
    assert "next_question" in fields
    assert "explanation" in fields


def test_debate_signature_input_fields():
    fields = DebateSignature.input_fields
    assert "position" in fields
    assert "context" in fields


def test_debate_signature_output_fields():
    fields = DebateSignature.output_fields
    assert "arguments_for" in fields
    assert "arguments_against" in fields
    assert "verdict" in fields
    assert "citations" in fields


# ---------------------------------------------------------------------------
# DeepResearchModule
# ---------------------------------------------------------------------------

def test_deep_research_module_returns_prediction(pipeline):
    module = DeepResearchModule(pipeline, k=3)
    result = module.forward(question="Who is Dumbledore?")
    assert hasattr(result, "answer")
    assert hasattr(result, "citations")
    assert hasattr(result, "confidence")
    assert hasattr(result, "gaps")


def test_deep_research_module_answer_is_string(pipeline):
    module = DeepResearchModule(pipeline, k=3)
    result = module.forward(question="What are Horcruxes?")
    assert isinstance(result.answer, str)
    assert len(result.answer) > 0


def test_deep_research_module_uses_retrieved_context(pipeline, monkeypatch):
    """Verify retrieve() is actually called."""
    calls = []
    original = pipeline.retrieve
    def spy(query, top_k=None):
        calls.append((query, top_k))
        return original(query, top_k=top_k)
    monkeypatch.setattr(pipeline, "retrieve", spy)

    module = DeepResearchModule(pipeline, k=5)
    module.forward(question="Tell me about Hogwarts")
    assert len(calls) == 1
    assert calls[0][1] == 5


def test_deep_research_module_k_respected(pipeline, monkeypatch):
    retrieved = []
    original = pipeline.retrieve
    def spy(query, top_k=None):
        result = original(query, top_k=top_k)
        retrieved.extend(result)
        return result
    monkeypatch.setattr(pipeline, "retrieve", spy)

    module = DeepResearchModule(pipeline, k=2)
    module.forward(question="Deathly Hallows")
    assert len(retrieved) <= 2


# ---------------------------------------------------------------------------
# GuidedLearningModule
# ---------------------------------------------------------------------------

def test_guided_learning_module_returns_prediction(pipeline):
    module = GuidedLearningModule(pipeline, k=3)
    result = module.forward(question="What is a Horcrux?")
    assert hasattr(result, "hint")
    assert hasattr(result, "next_question")
    assert hasattr(result, "explanation")


def test_guided_learning_module_accepts_past_attempts(pipeline):
    module = GuidedLearningModule(pipeline, k=3)
    result = module.forward(
        question="What is a Horcrux?",
        concept="dark magic",
        past_attempts="I think it's a magical object that stores something.",
    )
    assert isinstance(result.hint, str)


def test_guided_learning_module_accepts_no_past_attempts(pipeline):
    module = GuidedLearningModule(pipeline, k=3)
    result = module.forward(question="Explain the Deathly Hallows")
    assert isinstance(result.hint, str)


def test_guided_learning_module_narrow_k(pipeline, monkeypatch):
    retrieved = []
    original = pipeline.retrieve
    def spy(query, top_k=None):
        result = original(query, top_k=top_k)
        retrieved.extend(result)
        return result
    monkeypatch.setattr(pipeline, "retrieve", spy)

    module = GuidedLearningModule(pipeline, k=3)
    module.forward(question="Who is Snape?")
    assert len(retrieved) <= 3


# ---------------------------------------------------------------------------
# DSPyAgent — routing
# ---------------------------------------------------------------------------

def test_agent_forward_deep_research(pipeline):
    agent = DSPyAgent(pipeline)
    result = agent.forward("deep_research", "Who is Voldemort?")
    assert hasattr(result, "answer")
    assert hasattr(result, "citations")


def test_agent_forward_guided_learning(pipeline):
    agent = DSPyAgent(pipeline)
    result = agent.forward("guided_learning", "What are Horcruxes?")
    assert hasattr(result, "hint")
    assert hasattr(result, "next_question")


def test_agent_raises_on_unknown_mode(pipeline):
    agent = DSPyAgent(pipeline)
    with pytest.raises(ValueError, match="Unknown mode"):
        agent.forward("hallucination_mode", "anything")


def test_satirical_podcast_signature_input_fields():
    fields = SatiricalPodcastSignature.input_fields
    assert "topic" in fields
    assert "modern_angle" in fields
    assert "context" in fields


def test_satirical_podcast_signature_output_fields():
    fields = SatiricalPodcastSignature.output_fields
    assert "transcript" in fields
    assert "comedic_tension" in fields
    assert "citations" in fields


def test_agent_modes_constant():
    assert "deep_research" in MODES
    assert "guided_learning" in MODES
    assert "debate" in MODES
    assert "satirical_podcast" in MODES


# ---------------------------------------------------------------------------
# DSPyAgent — save / load round-trip
# ---------------------------------------------------------------------------

def test_agent_save_creates_json_files(pipeline, tmp_path):
    agent = DSPyAgent(pipeline)
    agent.forward("deep_research", "test question")   # warm up
    agent.save(str(tmp_path))
    assert (tmp_path / "deep_research.json").exists()
    assert (tmp_path / "guided_learning.json").exists()
    assert (tmp_path / "manifest.json").exists()


def test_agent_save_creates_valid_manifest(pipeline, tmp_path):
    agent = DSPyAgent(pipeline)
    agent.save(str(tmp_path))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert "embedding_model" in manifest
    assert "program_hashes" in manifest
    assert "created_at" in manifest


def test_agent_load_from_export(pipeline, tmp_path):
    agent = DSPyAgent(pipeline)
    agent.save(str(tmp_path))
    # Load into a new agent instance
    agent2 = DSPyAgent(pipeline, export_dir=str(tmp_path))
    result = agent2.forward("deep_research", "Who is Harry Potter?")
    assert hasattr(result, "answer")


# ---------------------------------------------------------------------------
# Manifest / backfill guard
# ---------------------------------------------------------------------------

def test_write_manifest_keys(tmp_path):
    _write_manifest(tmp_path, embedding_model="all-MiniLM-L6-v2")
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["embedding_model"] == "all-MiniLM-L6-v2"
    assert manifest["schema_version"] == 1


def test_validate_manifest_passes_matching_model():
    manifest = {"embedding_model": "all-MiniLM-L6-v2", "program_hashes": {}}
    _validate_manifest(manifest, "all-MiniLM-L6-v2")   # should not raise


def test_validate_manifest_raises_on_model_mismatch():
    manifest = {"embedding_model": "text-embedding-3-small", "program_hashes": {}}
    with pytest.raises(BackfillRequired, match="Embedding model changed"):
        _validate_manifest(manifest, "all-MiniLM-L6-v2")


def test_validate_manifest_passes_when_no_model_stored():
    """Empty manifest (first run) should not raise."""
    _validate_manifest({}, "all-MiniLM-L6-v2")


def test_agent_raises_backfill_on_wrong_model(pipeline, tmp_path):
    """Save with one embedding model label, load with a different one."""
    agent = DSPyAgent(pipeline)
    agent.save(str(tmp_path))

    # Tamper with manifest to simulate a model drift
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["embedding_model"] = "different-model-v999"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(BackfillRequired):
        DSPyAgent(pipeline, export_dir=str(tmp_path))
