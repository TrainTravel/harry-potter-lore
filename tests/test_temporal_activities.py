"""
Unit tests for temporal/activities.py.

Activities are just async functions — tested directly without any Temporal
infrastructure. The global _deps singleton is replaced per-test via a
fixture that calls init_activity_deps() with a mock ActivityDeps.

Coverage:
  - retrieve_chunks     — pipeline.retrieve → chunk dicts
  - route_intent        — agent.route_only via dspy.context
  - plan_tool_calls     — QueryPlanSignature, _parse_tool_calls fallback
  - execute_tool_call   — ToolRegistry.call_sync, k clamping
  - synthesise_research — DeepResearchSignature → dict
  - load/save/compact   — ConversationStore delegation
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from temporal.activities import (
    ActivityDeps,
    compact_conversation,
    execute_tool_call,
    init_activity_deps,
    load_conversation_history,
    plan_tool_calls,
    retrieve_chunks,
    route_intent,
    save_conversation_turn,
    synthesise_research,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class _FakeChunk:
    doc_id: str
    text: str
    score: float = 0.9


def _make_deps(**overrides) -> ActivityDeps:
    agent = MagicMock()
    agent.route_only.return_value = {"mode": "deep_research", "confidence": "high", "kwargs": {}}

    pipeline = MagicMock()
    pipeline.retrieve.return_value = [
        _FakeChunk("doc-a", "The Elder Wand was powerful."),
        _FakeChunk("doc-b", "Dumbledore owned it for decades."),
    ]

    conv_store = MagicMock()
    conv_store.load_history.return_value = MagicMock(turns=[], summary="")
    conv_store.format_for_llm.return_value = ""
    conv_store.save_turn.return_value = 1
    conv_store.compact_if_needed.return_value = False

    deps = ActivityDeps(
        agent=agent,
        pipeline=pipeline,
        conv_store=conv_store,
        model="test-model",
    )
    for k, v in overrides.items():
        setattr(deps, k, v)
    return deps


@pytest.fixture(autouse=True)
def _inject_deps():
    """Install fake deps before each test; restore None after."""
    import temporal.activities as act_mod
    original = act_mod._deps
    init_activity_deps(_make_deps())
    yield
    act_mod._deps = original


# ---------------------------------------------------------------------------
# retrieve_chunks
# ---------------------------------------------------------------------------

class TestRetrieveChunks:
    async def test_returns_serialisable_dicts(self):
        result = await retrieve_chunks("Who owns the Elder Wand?", "hp_lore", 5)
        assert isinstance(result, list)
        assert result[0]["doc_id"] == "doc-a"
        assert isinstance(result[0]["score"], float)

    async def test_passes_k_to_pipeline(self):
        import temporal.activities as act_mod
        await retrieve_chunks("question", "hp_lore", 7)
        act_mod._deps.pipeline.retrieve.assert_called_once_with("question", 7)

    async def test_empty_corpus_returns_empty_list(self):
        import temporal.activities as act_mod
        act_mod._deps.pipeline.retrieve.return_value = []
        result = await retrieve_chunks("anything", "hp_lore", 3)
        assert result == []


# ---------------------------------------------------------------------------
# route_intent
# ---------------------------------------------------------------------------

class TestRouteIntent:
    async def test_returns_router_dict(self):
        with patch("temporal.activities.dspy.context") as mock_ctx, \
             patch("temporal.activities.dspy.LM"):
            mock_ctx.return_value.__enter__ = lambda s: s
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            result = await route_intent("Who destroyed the Horcruxes?")
        assert result["mode"] == "deep_research"
        assert result["confidence"] == "high"

    async def test_off_topic_passthrough(self):
        import temporal.activities as act_mod
        act_mod._deps.agent.route_only.return_value = {
            "mode": "none", "confidence": "high", "kwargs": {},
        }
        with patch("temporal.activities.dspy.context") as mock_ctx, \
             patch("temporal.activities.dspy.LM"):
            mock_ctx.return_value.__enter__ = lambda s: s
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            result = await route_intent("What is the weather?")
        assert result["mode"] == "none"


# ---------------------------------------------------------------------------
# execute_tool_call
# ---------------------------------------------------------------------------

class TestExecuteToolCall:
    async def test_delegates_to_registry(self):
        with patch("context_harness.retrieval_tools.build_retrieval_registry") as mock_reg:
            registry = MagicMock()
            registry.call_sync.return_value = [{"doc_id": "x", "text": "passage"}]
            mock_reg.return_value = registry
            result = await execute_tool_call("vector_search", {"query": "wand", "k": 5}, "hp_lore")
        assert result[0]["doc_id"] == "x"

    async def test_k_clamped_above_max(self):
        with patch("context_harness.retrieval_tools.build_retrieval_registry") as mock_reg:
            registry = MagicMock()
            registry.call_sync.return_value = []
            mock_reg.return_value = registry
            await execute_tool_call("vector_search", {"query": "q", "k": 99}, "hp_lore")
            # call_sync("vector_search", args_dict) — second positional arg is the dict
            passed_args = registry.call_sync.call_args[0][1]
            assert passed_args["k"] == 15

    async def test_k_clamped_below_min(self):
        with patch("context_harness.retrieval_tools.build_retrieval_registry") as mock_reg:
            registry = MagicMock()
            registry.call_sync.return_value = []
            mock_reg.return_value = registry
            await execute_tool_call("vector_search", {"query": "q", "k": 0}, "hp_lore")
            passed_args = registry.call_sync.call_args[0][1]
            assert passed_args["k"] == 1


# ---------------------------------------------------------------------------
# synthesise_research
# ---------------------------------------------------------------------------

class TestSynthesiseResearch:
    async def test_returns_answer_dict(self):
        fake_pred = MagicMock()
        fake_pred.answer = "Harry won."
        fake_pred.citations = "harry-potter"
        fake_pred.confidence = "high"
        fake_pred.gaps = "none"

        with patch("temporal.activities.dspy.ChainOfThought") as mock_cot, \
             patch("temporal.activities.dspy.context") as mock_ctx, \
             patch("temporal.activities.dspy.LM"):
            mock_cot.return_value.return_value = fake_pred
            mock_ctx.return_value.__enter__ = lambda s: s
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            result = await synthesise_research("Who won?", "context text")

        assert result["answer"] == "Harry won."
        assert result["confidence"] == "high"

    async def test_missing_fields_default_to_empty_string(self):
        fake_pred = MagicMock(spec=[])  # no attributes
        with patch("temporal.activities.dspy.ChainOfThought") as mock_cot, \
             patch("temporal.activities.dspy.context") as mock_ctx, \
             patch("temporal.activities.dspy.LM"):
            mock_cot.return_value.return_value = fake_pred
            mock_ctx.return_value.__enter__ = lambda s: s
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            result = await synthesise_research("Q?", "C")
        assert result["answer"] == ""
        assert result["gaps"] == ""


# ---------------------------------------------------------------------------
# Conversation persistence
# ---------------------------------------------------------------------------

class TestConversationActivities:
    async def test_load_history_returns_string(self):
        import temporal.activities as act_mod
        act_mod._deps.conv_store.format_for_llm.return_value = "prior turns text"
        result = await load_conversation_history("conv-1", 5)
        assert result == "prior turns text"

    async def test_save_turn_returns_index(self):
        idx = await save_conversation_turn(
            "conv-1", "user msg", {"answer": "agent reply"}, "deep_research", 10, 20, 0.001,
        )
        assert idx == 1

    async def test_save_turn_delegates_correct_args(self):
        import temporal.activities as act_mod
        await save_conversation_turn("cid", "q", {"a": "r"}, "guided_learning", 5, 8, 0.0)
        call_kwargs = act_mod._deps.conv_store.save_turn.call_args.kwargs
        assert call_kwargs["conversation_id"] == "cid"
        assert call_kwargs["mode"] == "guided_learning"

    async def test_compact_returns_false_when_no_compaction(self):
        result = await compact_conversation("conv-1")
        assert result is False

    async def test_compact_returns_true_when_compacted(self):
        import temporal.activities as act_mod
        act_mod._deps.conv_store.compact_if_needed.return_value = True
        result = await compact_conversation("conv-long")
        assert result is True
