"""
Integration tests — verify trace events fire correctly through /ask.
=====================================================================

These tests spin up the FastAPI app with DummyLM patched in and hit
``POST /ask`` via TestClient. They check that the in-memory trace store
(``api.main._trace_store``) accumulates the expected events in the
correct order for each mode, including the new multi-turn + token
events added recently.

Design:
  - No real LLM calls (DummyLM via dspy.configure)
  - No real ChromaDB writes (in-memory ephemeral client via build_pipeline)
  - No real conversation DB (monkey-patched to ``:memory:`` DuckDB)
  - Tests assert *which events fired + their payload shape*, not the
    content of the agent's answer.
"""

from __future__ import annotations

import pytest
import dspy
from dspy.utils import DummyLM
from fastapi.testclient import TestClient

from api import main as api_main
from context_harness.conversation import ConversationStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patched_lm(monkeypatch):
    """Every test runs with DummyLM — no real API calls.

    Configure DummyLM FIRST, then no-op any subsequent ``dspy.configure``
    calls. FastAPI's lifespan re-configures dspy on TestClient startup
    from a different thread; dspy 3.x forbids that, which crashes the
    integration test. Silencing the second configure lets our DummyLM
    win while the lifespan keeps doing everything else.
    """
    lm = DummyLM(answers=[{
        "answer": "Test answer.",
        "citations": "harry-potter",
        "confidence": "high",
        "gaps": "none",
        "hint": "Test hint.",
        "next_question": "Test next question?",
        "explanation": "Test explanation sufficient for the metric.",
        "reasoning": "Step by step reasoning trace.",
        "analysis": "Test analysis.",
        "corpus_facts": "Test corpus facts.",
        "own_reasoning": "Test own reasoning.",
        "character_principle": "A principle grounded in specific canon events and decisions.",
        "applied_insight": "Applied insight that is specific and actionable within sixty to ninety words, giving the user a concrete stance.",
    }])
    dspy.configure(lm=lm)
    monkeypatch.setattr(dspy, "configure", lambda **kw: None)
    yield lm


@pytest.fixture
def isolated_conv_store(monkeypatch):
    """Replace the module-level conversation store with an in-memory DB so
    each test gets a clean slate."""
    test_store = ConversationStore(db_path=":memory:")
    monkeypatch.setattr(api_main, "_conversation_store", test_store)
    # Also clear the in-memory trace store so tests don't see each other's events
    api_main._trace_store.clear()
    yield test_store


@pytest.fixture
def client(isolated_conv_store):
    """TestClient fires lifespan (agent setup) on creation. Use session-less
    context manager so each test gets a fresh agent."""
    with TestClient(api_main.app) as c:
        yield c


def _events_of_kind(turn_id: str, kind: str) -> list[dict]:
    """All trace events for a turn matching a given name."""
    return [e for e in api_main._trace_store.get(turn_id, []) if e["name"] == kind]


def _all_event_names(turn_id: str) -> list[str]:
    return [e["name"] for e in api_main._trace_store.get(turn_id, [])]


# ---------------------------------------------------------------------------
# One-shot modes — no multi-turn events
# ---------------------------------------------------------------------------

def test_deep_research_fires_expected_events(client):
    resp = client.post("/ask", json={
        "question": "Who killed Dumbledore?",
        "mode": "deep_research",
    })
    assert resp.status_code == 200
    tid = resp.json()["turn_id"]

    names = _all_event_names(tid)
    assert "turn.start" in names
    assert "retrieve.done" in names
    assert "llm.done" in names
    assert "tokens.measured" in names
    assert "turn.end" in names


def test_deep_research_does_not_fire_chat_history_events(client):
    """One-shot modes must NOT trigger chat_history.loaded / saved even
    when conversation_id is sent — the server should silently ignore it."""
    resp = client.post("/ask", json={
        "question": "Who killed Dumbledore?",
        "mode": "deep_research",
        "conversation_id": "should-be-ignored-for-deep-research",
    })
    assert resp.status_code == 200
    tid = resp.json()["turn_id"]

    names = _all_event_names(tid)
    assert "chat_history.loaded" not in names
    assert "chat_history.saved" not in names


# ---------------------------------------------------------------------------
# Chat modes — chat_history events fire when conversation_id is present
# ---------------------------------------------------------------------------

def test_open_analysis_with_conversation_id_fires_chat_history_events(client):
    resp = client.post("/ask", json={
        "question": "Why did Snape become Snape?",
        "mode": "open_analysis",
        "conversation_id": "c-oa-1",
    })
    assert resp.status_code == 200
    tid = resp.json()["turn_id"]

    loaded = _events_of_kind(tid, "chat_history.loaded")
    saved = _events_of_kind(tid, "chat_history.saved")
    assert len(loaded) == 1
    assert len(saved) == 1

    # First turn: no prior turns in history
    assert loaded[0]["attrs"]["prior_turns"] == 0
    assert loaded[0]["attrs"]["has_summary"] is False
    # Saved event carries turn_index
    assert saved[0]["attrs"]["turn_index"] == 1


def test_open_analysis_without_conversation_id_no_chat_history_events(client):
    resp = client.post("/ask", json={
        "question": "Why did Snape become Snape?",
        "mode": "open_analysis",
    })
    assert resp.status_code == 200
    tid = resp.json()["turn_id"]
    names = _all_event_names(tid)
    assert "chat_history.loaded" not in names
    assert "chat_history.saved" not in names


def test_multi_turn_open_analysis_sees_prior_turn(client, isolated_conv_store):
    """Two consecutive calls with the same conversation_id — the second
    turn's chat_history.loaded event should report prior_turns=1."""
    r1 = client.post("/ask", json={
        "question": "Why did Snape become Snape?",
        "mode": "open_analysis",
        "conversation_id": "c-oa-2",
    })
    assert r1.status_code == 200

    r2 = client.post("/ask", json={
        "question": "Did Dumbledore know he was a double agent?",
        "mode": "open_analysis",
        "conversation_id": "c-oa-2",
    })
    assert r2.status_code == 200
    tid2 = r2.json()["turn_id"]

    loaded = _events_of_kind(tid2, "chat_history.loaded")
    assert len(loaded) == 1
    assert loaded[0]["attrs"]["prior_turns"] == 1
    assert loaded[0]["attrs"]["history_chars"] > 0


def test_perspective_shift_is_now_a_chat_mode(client, isolated_conv_store):
    """perspective_shift joined _CHAT_MODES in the latest commit."""
    r1 = client.post("/ask", json={
        "question": "I'm stuck choosing between a safe job and a risky creative path.",
        "mode": "perspective_shift",
        "character": "Dumbledore",
        "conversation_id": "c-ps-1",
    })
    assert r1.status_code == 200
    tid = r1.json()["turn_id"]
    names = _all_event_names(tid)
    assert "chat_history.loaded" in names
    assert "chat_history.saved" in names


# ---------------------------------------------------------------------------
# Token measurement event
# ---------------------------------------------------------------------------

def test_tokens_measured_event_fires_with_attrs(client):
    resp = client.post("/ask", json={
        "question": "Who is Hermione?",
        "mode": "deep_research",
    })
    assert resp.status_code == 200
    tid = resp.json()["turn_id"]
    events = _events_of_kind(tid, "tokens.measured")
    assert len(events) == 1
    attrs = events[0]["attrs"]
    # Keys always present (may be zero under DummyLM — that's fine)
    assert "tokens_in" in attrs
    assert "tokens_out" in attrs
    assert "cost_usd" in attrs


# ---------------------------------------------------------------------------
# Event ordering — start before end, load before save
# ---------------------------------------------------------------------------

def test_event_ordering_within_turn(client, isolated_conv_store):
    """Within a single turn, events should fire in a sensible order:
    turn.start → chat_history.loaded → retrieve.done → llm.done →
    tokens.measured → chat_history.saved → turn.end."""
    resp = client.post("/ask", json={
        "question": "test ordering",
        "mode": "open_analysis",
        "conversation_id": "c-ord-1",
    })
    assert resp.status_code == 200
    tid = resp.json()["turn_id"]
    names = _all_event_names(tid)

    # Spot-check key transitions
    assert names.index("turn.start") < names.index("retrieve.done")
    assert names.index("chat_history.loaded") < names.index("llm.done")
    assert names.index("llm.done") < names.index("tokens.measured")
    assert names.index("chat_history.saved") < names.index("turn.end")


# ---------------------------------------------------------------------------
# Turn persistence across multiple calls in same conversation
# ---------------------------------------------------------------------------

def test_turn_indices_monotonic_per_conversation(client, isolated_conv_store):
    """Three calls with the same conversation_id → chat_history.saved
    attrs should report turn_index 1, 2, 3."""
    ids = []
    for i in range(3):
        r = client.post("/ask", json={
            "question": f"q{i}",
            "mode": "open_analysis",
            "conversation_id": "c-mono-1",
        })
        assert r.status_code == 200
        tid = r.json()["turn_id"]
        saved = _events_of_kind(tid, "chat_history.saved")
        ids.append(saved[0]["attrs"]["turn_index"])
    assert ids == [1, 2, 3]


def test_separate_conversations_have_independent_indices(client, isolated_conv_store):
    ra = client.post("/ask", json={
        "question": "q1-a", "mode": "open_analysis",
        "conversation_id": "c-A",
    })
    rb = client.post("/ask", json={
        "question": "q1-b", "mode": "open_analysis",
        "conversation_id": "c-B",
    })
    ra2 = client.post("/ask", json={
        "question": "q2-a", "mode": "open_analysis",
        "conversation_id": "c-A",
    })
    t_a1 = _events_of_kind(ra.json()["turn_id"], "chat_history.saved")[0]["attrs"]["turn_index"]
    t_b1 = _events_of_kind(rb.json()["turn_id"], "chat_history.saved")[0]["attrs"]["turn_index"]
    t_a2 = _events_of_kind(ra2.json()["turn_id"], "chat_history.saved")[0]["attrs"]["turn_index"]
    assert (t_a1, t_b1, t_a2) == (1, 1, 2)
