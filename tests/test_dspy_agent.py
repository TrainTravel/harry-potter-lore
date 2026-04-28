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
        # OpenAnalysis output fields
        "analysis": "Blended analysis drawing from canon and interpretation.",
        "corpus_facts": "Key corpus facts.",
        "own_reasoning": "Extended interpretation beyond the corpus.",
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


# ---------------------------------------------------------------------------
# Character detection — drives open_analysis character-aware retrieval
# ---------------------------------------------------------------------------

def test_detect_character_in_question_finds_common_aliases():
    from context_harness.dspy_agent import _detect_character_in_question
    assert _detect_character_in_question("why did Snape become Snape?") == "severus-snape"
    assert _detect_character_in_question("what does Dumbledore think") == "albus-dumbledore"
    assert _detect_character_in_question("how did Hermione change") == "hermione-granger"


def test_detect_character_prefers_longer_alias():
    """'Albus Dumbledore' should win over 'Albus' alone so we don't get
    'albus-dumbledore' twice for the same phrase."""
    from context_harness.dspy_agent import _detect_character_in_question
    # Both aliases resolve to the same slug here; the key assertion is that
    # no exception is raised and we get the expected slug.
    assert _detect_character_in_question("tell me about Albus Dumbledore") == "albus-dumbledore"


def test_detect_character_respects_word_boundary():
    """'Ron' must not match 'Ronald' (different name) or substrings."""
    from context_harness.dspy_agent import _detect_character_in_question
    # 'Ronaldo the footballer' should not match our Ron slug
    assert _detect_character_in_question("Ronaldo scored a goal") is None


def test_detect_character_returns_none_for_no_character():
    from context_harness.dspy_agent import _detect_character_in_question
    assert _detect_character_in_question("what are the Deathly Hallows?") is None
    assert _detect_character_in_question("") is None


# ---------------------------------------------------------------------------
# Sorting Hat slug resolution
# ---------------------------------------------------------------------------

def test_sorting_hat_slug_resolution():
    from context_harness.dspy_agent import _normalize_character
    assert _normalize_character("Sorting Hat") == "sorting-hat"
    assert _normalize_character("sorting hat") == "sorting-hat"
    assert _normalize_character("The Sorting Hat") == "sorting-hat"
    assert _normalize_character("the sorting hat") == "sorting-hat"


# ---------------------------------------------------------------------------
# OpenAnalysisModule — character-aware retrieval split
# ---------------------------------------------------------------------------

def test_open_analysis_uses_char_pipeline_when_character_detected(pipeline, monkeypatch):
    """When the question names a character AND char_pipeline is available,
    retrieval should split: 5 chunks from char_pipeline (filtered by
    character), 2 chunks from the default pipeline."""
    from context_harness.dspy_agent import OpenAnalysisModule

    # Track every .retrieve call on each pipeline
    char_calls = []
    orig_retrieve = pipeline.retrieve
    def char_retrieve(query, top_k=None, where=None):
        char_calls.append({"query": query, "top_k": top_k, "where": where})
        return orig_retrieve(query, top_k=top_k)

    general_calls = []
    class _GeneralSpy:
        def retrieve(self, query, top_k=None, where=None):
            general_calls.append({"query": query, "top_k": top_k, "where": where})
            return orig_retrieve(query, top_k=top_k)

    # Use the real pipeline as char_pipeline (same embeddings) but wrap to spy
    class _CharSpy:
        def __init__(self, inner):
            self._inner = inner
        def retrieve(self, query, top_k=None, where=None):
            char_calls.append({"query": query, "top_k": top_k, "where": where})
            return self._inner.retrieve(query, top_k=top_k)

    module = OpenAnalysisModule(
        pipeline=_GeneralSpy(), k=7, char_pipeline=_CharSpy(pipeline),
    )
    module.forward(question="why did Snape become Snape?")

    # One call each, with correct top_k split and the character filter
    assert len(char_calls) == 1
    assert char_calls[0]["top_k"] == 5
    assert char_calls[0]["where"] == {"character": "severus-snape"}
    assert len(general_calls) == 1
    assert general_calls[0]["top_k"] == 2


def test_open_analysis_falls_back_to_default_when_no_character(pipeline, monkeypatch):
    """When no character is mentioned, open_analysis retrieves k=7 from
    the default pipeline and never touches char_pipeline."""
    from context_harness.dspy_agent import OpenAnalysisModule

    char_calls = []
    orig_retrieve = pipeline.retrieve

    class _CharSpy:
        def retrieve(self, query, top_k=None, where=None):
            char_calls.append({"query": query, "top_k": top_k})
            return orig_retrieve(query, top_k=top_k)

    module = OpenAnalysisModule(
        pipeline=pipeline, k=7, char_pipeline=_CharSpy(),
    )
    module.forward(question="what are the Deathly Hallows?")

    # No calls to char_pipeline — no character detected
    assert char_calls == []


def test_open_analysis_without_char_pipeline_always_uses_default(pipeline):
    """If no char_pipeline is wired in, open_analysis ignores character
    detection and uses the default pipeline for everything. No crash."""
    from context_harness.dspy_agent import OpenAnalysisModule

    module = OpenAnalysisModule(pipeline=pipeline, k=7, char_pipeline=None)
    result = module.forward(question="why did Snape become Snape?")
    # Just asserting it returns without error — detection happens but
    # falls back to default since char_pipeline is None.
    assert hasattr(result, "analysis")


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
