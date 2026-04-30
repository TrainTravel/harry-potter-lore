"""
Tests for Multi-tool Deep Research

Coverage:
  - _parse_tool_calls          — JSON parsing + fence stripping + fallback
  - _dedup_chunks              — deduplication and cap
  - ToolRegistry.call_sync     — validation, logging, async rejection
  - build_retrieval_registry   — contract enforcement on each tool
  - MultiToolDeepResearchModule — end-to-end with mock pipeline + DummyLM
  - EVALSET multi-hop examples — structure checks
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import dspy
import pytest

from context_harness.dspy_agent import (
    MultiToolDeepResearchModule,
    _dedup_chunks,
    _parse_tool_calls,
)
from context_harness.retrieval_tools import build_retrieval_registry
from context_harness.tool_registry import ToolContractError, ToolNotFoundError, ToolRegistry, ToolSchema, PropertySchema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(doc_id: str, text: str) -> dict:
    return {"doc_id": doc_id, "text": text}


def _mock_pipeline(chunks: list | None = None):
    """Return a mock RAGPipeline whose retrieve() returns ScoredChunk-like objects."""
    chunks = chunks or []
    mock_chunks = []
    for c in chunks:
        mc = MagicMock()
        mc.doc_id = c["doc_id"]
        mc.text = c["text"]
        mock_chunks.append(mc)
    pipeline = MagicMock()
    pipeline.retrieve.return_value = mock_chunks
    return pipeline


# ---------------------------------------------------------------------------
# _parse_tool_calls
# ---------------------------------------------------------------------------

class TestParseToolCalls:
    def test_clean_json_array(self):
        raw = '[{"tool": "vector_search", "args": {"query": "Elder Wand", "k": 5}}]'
        result = _parse_tool_calls(raw)
        assert len(result) == 1
        assert result[0]["tool"] == "vector_search"

    def test_json_in_fenced_block(self):
        raw = '```json\n[{"tool": "topic_search", "args": {"topic": "Horcruxes", "query": "creation", "k": 4}}]\n```'
        result = _parse_tool_calls(raw)
        assert len(result) == 1
        assert result[0]["tool"] == "topic_search"

    def test_fenced_block_without_language(self):
        raw = '```\n[{"tool": "vector_search", "args": {"query": "test", "k": 3}}]\n```'
        result = _parse_tool_calls(raw)
        assert result[0]["tool"] == "vector_search"

    def test_json_embedded_in_prose(self):
        raw = 'I will use these calls: [{"tool": "vector_search", "args": {"query": "Dumbledore", "k": 5}}] to retrieve.'
        result = _parse_tool_calls(raw)
        assert len(result) == 1

    def test_multiple_calls(self):
        calls = [
            {"tool": "topic_search", "args": {"topic": "Deathly Hallows", "query": "Elder Wand masters", "k": 5}},
            {"tool": "character_search", "args": {"character": "albus-dumbledore", "query": "Elder Wand", "k": 4}},
        ]
        result = _parse_tool_calls(json.dumps(calls))
        assert len(result) == 2

    def test_unparseable_returns_empty(self):
        assert _parse_tool_calls("I don't know how to answer this") == []

    def test_empty_string_returns_empty(self):
        assert _parse_tool_calls("") == []

    def test_malformed_json_returns_empty(self):
        assert _parse_tool_calls("[{tool: vector_search}]") == []


# ---------------------------------------------------------------------------
# _dedup_chunks
# ---------------------------------------------------------------------------

class TestDedupChunks:
    def test_removes_exact_duplicates(self):
        c = _make_chunk("horcruxes", "Voldemort created seven Horcruxes.")
        result = _dedup_chunks([[c, c], [c]])
        assert len(result) == 1

    def test_keeps_different_docs(self):
        c1 = _make_chunk("horcruxes", "Voldemort created seven Horcruxes.")
        c2 = _make_chunk("deathly-hallows", "The Elder Wand is one of the Hallows.")
        result = _dedup_chunks([[c1], [c2]])
        assert len(result) == 2

    def test_keeps_different_text_same_doc(self):
        c1 = _make_chunk("horcruxes", "First passage about Horcruxes.")
        c2 = _make_chunk("horcruxes", "Second passage about Horcruxes — different content.")
        result = _dedup_chunks([[c1, c2]])
        assert len(result) == 2

    def test_caps_at_15(self):
        chunks = [_make_chunk(f"doc-{i}", f"unique text number {i} " * 5) for i in range(20)]
        result = _dedup_chunks([chunks])
        assert len(result) == 15

    def test_empty_lists(self):
        assert _dedup_chunks([]) == []
        assert _dedup_chunks([[], []]) == []

    def test_order_preserved_first_seen(self):
        c1 = _make_chunk("doc-a", "first")
        c2 = _make_chunk("doc-b", "second")
        result = _dedup_chunks([[c1, c2]])
        assert result[0]["doc_id"] == "doc-a"


# ---------------------------------------------------------------------------
# ToolRegistry.call_sync
# ---------------------------------------------------------------------------

class TestCallSync:
    def _make_registry(self):
        reg = ToolRegistry()

        @reg.register(schema=ToolSchema(
            name="sync_tool",
            description="A sync test tool",
            properties={"value": PropertySchema(type="string", description="input")},
        ))
        def sync_tool(value: str):
            return f"result:{value}"

        return reg

    def test_call_sync_returns_result(self):
        reg = self._make_registry()
        result = reg.call_sync("sync_tool", {"value": "test"})
        assert result == "result:test"

    def test_call_sync_logs_call(self):
        reg = self._make_registry()
        reg.call_sync("sync_tool", {"value": "logged"})
        log = reg.call_log()
        assert len(log) == 1
        assert log[0]["tool"] == "sync_tool"
        assert log[0]["args"] == {"value": "logged"}

    def test_call_sync_raises_on_unknown_tool(self):
        reg = self._make_registry()
        with pytest.raises(ToolNotFoundError):
            reg.call_sync("no_such_tool", {})

    def test_call_sync_raises_on_schema_violation(self):
        reg = self._make_registry()
        with pytest.raises(ToolContractError):
            reg.call_sync("sync_tool", {"wrong_field": "x"})

    def test_call_sync_raises_on_async_tool(self):
        reg = ToolRegistry()

        @reg.register(schema=ToolSchema(
            name="async_tool",
            description="async",
            properties={"q": PropertySchema(type="string", description="q")},
        ))
        async def async_tool(q: str):
            return q

        with pytest.raises(RuntimeError, match="async"):
            reg.call_sync("async_tool", {"q": "test"})


# ---------------------------------------------------------------------------
# build_retrieval_registry — contract checks
# ---------------------------------------------------------------------------

class TestBuildRetrievalRegistry:
    @pytest.fixture
    def registry(self):
        pipeline = _mock_pipeline([
            {"doc_id": "horcruxes", "text": "Voldemort made Horcruxes."},
        ])
        return build_retrieval_registry(pipeline)

    def test_three_tools_registered(self, registry):
        assert set(registry.list_tools()) == {"vector_search", "character_search", "topic_search"}

    def test_vector_search_executes(self, registry):
        result = registry.call_sync("vector_search", {"query": "Elder Wand", "k": 3})
        assert isinstance(result, list)
        assert result[0]["doc_id"] == "horcruxes"

    def test_vector_search_k_too_large_raises(self, registry):
        with pytest.raises(ToolContractError):
            registry.call_sync("vector_search", {"query": "x", "k": 99})

    def test_vector_search_k_zero_raises(self, registry):
        with pytest.raises(ToolContractError):
            registry.call_sync("vector_search", {"query": "x", "k": 0})

    def test_character_search_requires_character(self, registry):
        with pytest.raises(ToolContractError):
            registry.call_sync("character_search", {"query": "Elder Wand", "k": 3})

    def test_topic_search_requires_topic(self, registry):
        with pytest.raises(ToolContractError):
            registry.call_sync("topic_search", {"query": "wand transfer", "k": 3})

    def test_openai_functions_schema_valid(self, registry):
        schemas = registry.to_openai_functions()
        assert len(schemas) == 3
        names = {s["function"]["name"] for s in schemas}
        assert names == {"vector_search", "character_search", "topic_search"}


# ---------------------------------------------------------------------------
# MultiToolDeepResearchModule — end-to-end with DummyLM
# ---------------------------------------------------------------------------

# Single answer template covering all output fields from both signatures.
# DummyLM cycles through this for every LLM call in the module.
_MULTI_TOOL_ANSWER = {
    # QueryPlanSignature
    "tool_calls": json.dumps([
        {"tool": "topic_search", "args": {"topic": "Deathly Hallows", "query": "Elder Wand masters", "k": 4}},
        {"tool": "character_search", "args": {"character": "albus-dumbledore", "query": "Elder Wand", "k": 3}},
    ]),
    # DeepResearchSignature
    "answer": "The Elder Wand passed from Antioch Peverell to Grindelwald to Dumbledore to Draco to Harry.",
    "citations": "deathly-hallows albus-dumbledore",
    "confidence": "high",
    "gaps": "none",
    # ChainOfThought always emits this
    "reasoning": "Step-by-step reasoning trace.",
}

_FALLBACK_ANSWER = {
    "tool_calls": "I cannot plan this query.",   # unparseable → triggers fallback
    "answer": "Fallback answer about the Elder Wand.",
    "citations": "deathly-hallows",
    "confidence": "medium",
    "gaps": "none",
    "reasoning": "Step-by-step reasoning trace.",
}

_LARGE_K_ANSWER = {
    "tool_calls": json.dumps([
        {"tool": "vector_search", "args": {"query": "Elder Wand", "k": 999}},
    ]),
    "answer": "Answer with clamped k.",
    "citations": "deathly-hallows",
    "confidence": "high",
    "gaps": "none",
    "reasoning": "Step-by-step reasoning trace.",
}


class TestMultiToolDeepResearchModule:
    @pytest.fixture(autouse=True)
    def dummy_lm(self):
        lm = dspy.utils.DummyLM(answers=[_MULTI_TOOL_ANSWER] * 10)
        dspy.configure(lm=lm)
        yield
        dspy.configure(lm=None)

    @pytest.fixture
    def module(self):
        pipeline = _mock_pipeline([
            {"doc_id": "deathly-hallows", "text": "The Elder Wand is the most powerful wand ever made."},
            {"doc_id": "albus-dumbledore", "text": "Dumbledore defeated Grindelwald and won the Elder Wand."},
        ])
        return MultiToolDeepResearchModule(pipeline)

    def test_forward_returns_prediction(self, module):
        pred = module.forward(question="Trace the Elder Wand's ownership.")
        assert hasattr(pred, "answer")
        assert hasattr(pred, "citations")

    def test_answer_is_nonempty(self, module):
        pred = module.forward(question="Trace the Elder Wand's ownership.")
        assert len(pred.answer) > 10

    def test_tool_call_log_populated(self, module):
        module.forward(question="Trace the Elder Wand's ownership.")
        log = module._registry.call_log()
        assert len(log) >= 1

    def test_fallback_on_bad_plan(self):
        """When the LLM returns unparseable tool_calls, falls back to vector_search."""
        lm = dspy.utils.DummyLM(answers=[_FALLBACK_ANSWER] * 10)
        dspy.configure(lm=lm)
        pipeline = _mock_pipeline([
            {"doc_id": "deathly-hallows", "text": "The Elder Wand text."},
        ])
        module = MultiToolDeepResearchModule(pipeline)
        pred = module.forward(question="Who owned the Elder Wand?")
        assert hasattr(pred, "answer")
        dspy.configure(lm=None)

    def test_k_clamped_to_valid_range(self):
        """Tool calls with k > 15 are clamped before hitting ToolRegistry."""
        lm = dspy.utils.DummyLM(answers=[_LARGE_K_ANSWER] * 10)
        dspy.configure(lm=lm)
        pipeline = _mock_pipeline([
            {"doc_id": "deathly-hallows", "text": "Elder Wand passage."},
        ])
        module = MultiToolDeepResearchModule(pipeline)
        pred = module.forward(question="Elder Wand ownership?")
        assert hasattr(pred, "answer")
        dspy.configure(lm=None)


# ---------------------------------------------------------------------------
# EVALSET multi-hop examples
# ---------------------------------------------------------------------------

class TestDeepResearchMultiHopEvalset:
    def test_multi_hop_examples_present(self):
        from data.trainset_deep_research import EVALSET
        multi_hop = [
            ex for ex in EVALSET
            if len(ex.citations.split()) >= 3
        ]
        assert len(multi_hop) >= 3, "Expected at least 3 multi-hop EVALSET examples"

    def test_multi_hop_citations_span_multiple_docs(self):
        from data.trainset_deep_research import EVALSET
        for ex in EVALSET:
            cites = ex.citations.split()
            # All citations should be valid doc_ids
            valid = {
                "albus-dumbledore", "harry-potter", "hermione-granger",
                "ron-weasley", "lord-voldemort", "severus-snape",
                "hogwarts", "horcruxes", "deathly-hallows", "order-of-the-phoenix",
            }
            for c in cites:
                assert c in valid, f"Invalid citation {c!r} in EVALSET"

    def test_evalset_disjoint_from_trainset(self):
        from data.trainset_deep_research import EVALSET, TRAINSET
        train_qs = {ex.question for ex in TRAINSET}
        for ex in EVALSET:
            assert ex.question not in train_qs, f"EVALSET question leaked into TRAINSET: {ex.question!r}"
