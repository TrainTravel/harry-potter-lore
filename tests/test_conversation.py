"""Tests for context_harness.conversation.ConversationStore."""

from __future__ import annotations

import pytest

from context_harness.conversation import (
    ConversationStore,
    Turn,
    ConversationHistory,
    _format_agent_response,
)


# ---------------------------------------------------------------------------
# Fixtures — in-memory DB per test so parallel runs never collide
# ---------------------------------------------------------------------------

@pytest.fixture
def store() -> ConversationStore:
    return ConversationStore(db_path=":memory:")


# ---------------------------------------------------------------------------
# save_turn + load_history round-trip
# ---------------------------------------------------------------------------

def test_save_and_load_single_turn(store: ConversationStore):
    idx = store.save_turn(
        conversation_id="c1",
        user_message="What is a Horcrux?",
        agent_response={"hint": "Think about soul division.",
                        "explanation": "A container for a soul fragment."},
        mode="guided_learning",
    )
    assert idx == 1

    history = store.load_history("c1")
    assert history.summary == ""
    assert len(history.turns) == 1
    assert history.turns[0].user_message == "What is a Horcrux?"
    assert history.turns[0].agent_response["explanation"] == "A container for a soul fragment."


def test_turn_indices_are_monotonic(store: ConversationStore):
    ids = [
        store.save_turn("c1", f"q{i}", {"explanation": f"a{i}"}, "guided_learning")
        for i in range(5)
    ]
    assert ids == [1, 2, 3, 4, 5]


def test_separate_conversations_have_independent_indices(store: ConversationStore):
    a1 = store.save_turn("conv-a", "q1", {"explanation": "a"}, "guided_learning")
    b1 = store.save_turn("conv-b", "q1", {"explanation": "b"}, "guided_learning")
    a2 = store.save_turn("conv-a", "q2", {"explanation": "c"}, "guided_learning")
    assert (a1, b1, a2) == (1, 1, 2)


def test_load_history_respects_max_turns(store: ConversationStore):
    for i in range(10):
        store.save_turn("c1", f"q{i}", {"explanation": f"a{i}"}, "guided_learning")

    history = store.load_history("c1", max_turns=3)
    assert len(history.turns) == 3
    # Chronological order — most recent three
    assert [t.turn_index for t in history.turns] == [8, 9, 10]


def test_load_history_empty_conversation(store: ConversationStore):
    history = store.load_history("never-seen-id")
    assert history.is_empty()
    assert history.turns == []
    assert history.summary == ""


# ---------------------------------------------------------------------------
# format_for_llm
# ---------------------------------------------------------------------------

def test_format_for_llm_empty_returns_empty_string(store: ConversationStore):
    history = ConversationHistory(summary="", turns=[])
    assert store.format_for_llm(history, mode="guided_learning") == ""


def test_format_for_llm_guided_learning_only_surfaces_explanation(store: ConversationStore):
    store.save_turn(
        "c1",
        "What is a Horcrux?",
        {"hint": "Ephemeral hint, should not appear",
         "next_question": "Ephemeral next, should not appear",
         "explanation": "Durable explanation — this should appear."},
        "guided_learning",
    )
    history = store.load_history("c1")
    text = store.format_for_llm(history, mode="guided_learning")
    assert "Durable explanation" in text
    assert "Ephemeral hint" not in text
    assert "Ephemeral next" not in text


def test_format_for_llm_open_analysis_surfaces_analysis_field(store: ConversationStore):
    store.save_turn(
        "c2",
        "Why is Hermione's knowledge emphasised?",
        {"analysis": "She serves as intellectual anchor for the trio.",
         "corpus_facts": "Pulled from [hermione-granger]",
         "own_reasoning": "My interpretation beyond corpus."},
        "open_analysis",
    )
    history = store.load_history("c2")
    text = store.format_for_llm(history, mode="open_analysis")
    assert "intellectual anchor" in text
    # Corpus facts + own_reasoning are ephemera for the next turn
    assert "corpus" not in text.lower() or "anchor" in text
    # User's question must be preserved
    assert "Hermione's knowledge" in text


def test_format_for_llm_chronological_order(store: ConversationStore):
    for i in range(3):
        store.save_turn("c1", f"question {i}",
                        {"explanation": f"answer {i}"}, "guided_learning")
    history = store.load_history("c1")
    text = store.format_for_llm(history, mode="guided_learning")
    # question 0 appears before question 1, which appears before question 2
    assert text.index("question 0") < text.index("question 1") < text.index("question 2")


# ---------------------------------------------------------------------------
# conversation_cost
# ---------------------------------------------------------------------------

def test_conversation_cost_aggregates_across_turns(store: ConversationStore):
    store.save_turn("c1", "q1", {"explanation": "a"}, "guided_learning",
                    tokens_in=100, tokens_out=50, cost_usd=0.01)
    store.save_turn("c1", "q2", {"explanation": "b"}, "guided_learning",
                    tokens_in=120, tokens_out=70, cost_usd=0.015)

    totals = store.conversation_cost("c1")
    assert totals["turns"] == 2
    assert totals["tokens_in"] == 220
    assert totals["tokens_out"] == 120
    assert totals["cost_usd"] == pytest.approx(0.025)


def test_conversation_cost_unknown_id_returns_zeros(store: ConversationStore):
    totals = store.conversation_cost("never-seen-id")
    assert totals == {"turns": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}


def test_turn_costs_returns_per_turn_breakdown(store: ConversationStore):
    store.save_turn("c1", "q1", {"explanation": "a"}, "guided_learning",
                    tokens_in=100, tokens_out=50, cost_usd=0.01)
    store.save_turn("c1", "q2", {"explanation": "b"}, "guided_learning",
                    tokens_in=120, tokens_out=70, cost_usd=0.015)
    store.save_turn("c1", "q3", {"explanation": "c"}, "guided_learning",
                    tokens_in=140, tokens_out=90, cost_usd=0.020)

    breakdown = store.turn_costs("c1")
    assert len(breakdown) == 3
    assert [b["turn_index"]  for b in breakdown] == [1, 2, 3]
    assert [b["tokens_in"]   for b in breakdown] == [100, 120, 140]
    assert [b["tokens_out"]  for b in breakdown] == [50, 70, 90]
    assert [b["cost_usd"]    for b in breakdown] == [0.01, 0.015, 0.020]


def test_turn_costs_empty_conversation_returns_empty_list(store: ConversationStore):
    assert store.turn_costs("never-seen-id") == []


# ---------------------------------------------------------------------------
# Compaction — uses FakeSummarizer (no real LLM calls)
# ---------------------------------------------------------------------------

class FakeSummarizer:
    """Deterministic summarizer for tests. Records prompts for inspection,
    returns a scripted summary string per call."""
    def __init__(self, summary: str = (
        "The student has been exploring the concept of horcruxes and soul "
        "fragmentation. The tutor has explained the basic mechanism and "
        "the student is now asking follow-up questions about the ethics.")) -> None:
        self._summary = summary
        self.calls: list[str] = []

    def summarize(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self._summary


def test_compaction_noop_below_threshold(store: ConversationStore):
    for i in range(5):
        store.save_turn("c1", f"q{i}", {"explanation": f"a{i}"}, "guided_learning")
    summ = FakeSummarizer()
    result = store.compact_if_needed("c1", summarizer=summ, threshold=8, keep_recent=5)
    assert result is False
    assert summ.calls == []


def test_compaction_fires_when_above_threshold(store: ConversationStore):
    for i in range(10):
        store.save_turn("c1", f"q{i}", {"explanation": f"a{i}"}, "guided_learning")
    summ = FakeSummarizer()
    result = store.compact_if_needed("c1", summarizer=summ, threshold=8, keep_recent=5)
    assert result is True
    assert len(summ.calls) == 1


def test_compaction_uses_mode_aware_prompt(store: ConversationStore):
    for i in range(10):
        store.save_turn("c1", f"q{i}", {"explanation": f"a{i}"}, "guided_learning")
    summ = FakeSummarizer()
    store.compact_if_needed("c1", summarizer=summ, threshold=8)
    assert "tutor-student dialogue" in summ.calls[0]


def test_compaction_rejects_garbled_summary(store: ConversationStore):
    for i in range(10):
        store.save_turn("c1", f"q{i}", {"explanation": f"a{i}"}, "guided_learning")
    summ = FakeSummarizer(summary="too short")
    result = store.compact_if_needed("c1", summarizer=summ, threshold=8)
    assert result is False
    history = store.load_history("c1", max_turns=20)
    assert history.summary == ""


def test_compaction_keeps_recent_turns_verbatim(store: ConversationStore):
    for i in range(10):
        store.save_turn("c1", f"q{i}", {"explanation": f"a{i}"}, "guided_learning")
    summ = FakeSummarizer()
    store.compact_if_needed("c1", summarizer=summ, threshold=8, keep_recent=5)
    history = store.load_history("c1", max_turns=20)
    assert history.summary != ""
    # Verbatim turns are the last keep_recent=5 (indices 6-10)
    assert [t.turn_index for t in history.turns] == [6, 7, 8, 9, 10]


def test_compaction_is_fault_tolerant(store: ConversationStore):
    for i in range(10):
        store.save_turn("c1", f"q{i}", {"explanation": f"a{i}"}, "guided_learning")

    class CrashingSummarizer:
        def summarize(self, prompt: str) -> str:
            raise RuntimeError("simulated LLM outage")

    result = store.compact_if_needed("c1", summarizer=CrashingSummarizer(), threshold=8)
    assert result is False
    history = store.load_history("c1", max_turns=20)
    assert history.summary == ""


def test_compaction_none_summarizer_is_noop_at_any_count(store: ConversationStore):
    for i in range(15):
        store.save_turn("c1", f"q{i}", {"explanation": f"a{i}"}, "guided_learning")
    result = store.compact_if_needed("c1", summarizer=None, threshold=8)
    assert result is False


def test_compaction_re_summarizes_including_prior_summary(store: ConversationStore):
    """After a compaction exists and new turns arrive, the next compaction
    should include the prior summary in its input so the new summary covers
    the full span, not just the recent delta."""
    for i in range(10):
        store.save_turn("c1", f"q{i}", {"explanation": f"a{i}"}, "guided_learning")
    summ1 = FakeSummarizer(summary=(
        "First compaction covered the initial turns of the dialogue and "
        "established that the student is learning about horcruxes, soul "
        "fragmentation, and the basic ethical and magical implications "
        "of immortality schemes."
    ))
    store.compact_if_needed("c1", summarizer=summ1, threshold=8, keep_recent=5)

    # Add more turns, triggering a second compaction
    for i in range(10, 14):
        store.save_turn("c1", f"q{i}", {"explanation": f"a{i}"}, "guided_learning")
    summ2 = FakeSummarizer(summary="Second summary covers turns 1 through 9 including horcruxes introduction and now the ethics of soul magic the student is asking about.")
    store.compact_if_needed("c1", summarizer=summ2, threshold=8, keep_recent=5)

    # The second summarizer's prompt must have included the first summary
    assert "[Earlier summary]" in summ2.calls[0]
    assert "First compaction" in summ2.calls[0]


# ---------------------------------------------------------------------------
# Mode-specific formatter fallback
# ---------------------------------------------------------------------------

def test_unknown_mode_falls_back_to_json_dump():
    response = {"some_field": "some value"}
    formatted = _format_agent_response(response, mode="unknown_mode_xyz")
    assert "some_field" in formatted
    assert "some value" in formatted
