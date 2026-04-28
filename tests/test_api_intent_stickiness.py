"""
Integration tests — multi-turn stickiness in ``/ask``.
======================================================

Spins up the FastAPI app with DummyLM and verifies that the stickiness
logic in ``api.main.ask`` correctly orchestrates ``agent.route_only`` +
``_decide_effective_mode`` + dispatch.

Strategy: patch ``agent.route_only`` to return controlled router output
per call, so each test case can simulate the router emitting ``"none"``,
a different mode, etc. without coupling to DummyLM's call ordering.

The decision-helper logic itself is unit-tested in
``test_intent_routing_stickiness.py``. These tests exercise the wiring.
"""

from __future__ import annotations

import contextlib
import pytest
import dspy
from dspy.utils import DummyLM
from fastapi.testclient import TestClient

from api import main as api_main
from context_harness.conversation import ConversationStore


# ---------------------------------------------------------------------------
# Fixtures (mirror test_api_tracing_integration.py — same defusing)
# ---------------------------------------------------------------------------

_ANSWER_TEMPLATE = {
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
    "arguments_for": "Canon-supported arguments in favour, referencing specific events.",
    "arguments_against": "Canon-supported arguments against, referencing specific events.",
    "verdict": "Verdict referencing the canon evidence weighed above.",
    "transcript": "Host A: First line. Host B: Second line.",
    "comedic_tension": "Comedic tension placeholder.",
    "score": 75,
    "is_passing": True,
    "critique": "Test critique.",
    "character_principle": "Principle.",
    "applied_insight": "Insight.",
    "character_response": "Response.",
    # Router-shaped fields too, so DummyLM can serve any caller
    "mode": "open_analysis",
    "kwargs_json": "{}",
}


@pytest.fixture(autouse=True)
def patched_lm(monkeypatch):
    lm = DummyLM(answers=[_ANSWER_TEMPLATE] * 50)
    dspy.configure(lm=lm)
    monkeypatch.setattr(dspy, "configure", lambda **kw: None)

    @contextlib.contextmanager
    def _noop_context(**_kwargs):
        yield
    monkeypatch.setattr(dspy, "context", _noop_context)
    yield lm


@pytest.fixture
def isolated_conv_store(monkeypatch):
    test_store = ConversationStore(db_path=":memory:")
    monkeypatch.setattr(api_main, "_conversation_store", test_store)
    api_main._trace_store.clear()
    yield test_store


@pytest.fixture
def client(isolated_conv_store):
    with TestClient(api_main.app) as c:
        yield c


def _patch_route_only(monkeypatch, *, mode: str, confidence: str = "high"):
    """Replace the cached agent's ``route_only`` with a stub returning
    the given (mode, confidence). kwargs is always empty."""
    agent = api_main._agents_cache["hp_lore"]
    monkeypatch.setattr(
        agent, "route_only",
        lambda text: {"mode": mode, "confidence": confidence, "kwargs": {}},
    )


def _events_of_kind(turn_id: str, name: str) -> list[dict]:
    return [e for e in api_main._trace_store.get(turn_id, []) if e["name"] == name]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_auto_mode_no_conversation_trusts_router(client, monkeypatch):
    """SPEC AC: mode=auto + no conversation_id behaves identically to today —
    router decides, no stickiness consultation."""
    _patch_route_only(monkeypatch, mode="open_analysis", confidence="high")
    r = client.post("/ask", json={
        "question": "What does Hogwarts say about boarding-school culture?",
        "mode": "auto",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["routed_mode"] == "open_analysis"
    assert body["router_confidence"] == "high"

    # Trace: router.decision fired with source="router"
    decision_events = _events_of_kind(body["turn_id"], "router.decision")
    assert len(decision_events) == 1
    assert decision_events[0]["attrs"]["source"] == "router"
    assert decision_events[0]["attrs"]["effective_mode"] == "open_analysis"
    assert decision_events[0]["attrs"]["prior_mode"] is None


def test_auto_mode_continuation_sticks_when_router_says_off_topic(client, monkeypatch):
    """SPEC AC: ambiguous follow-up classified as off-topic must NOT collapse
    the conversation — stickiness keeps the prior mode.

    Turn 1: router says open_analysis → goes to open_analysis.
    Turn 2: router says "none" (off-topic) → stickiness keeps open_analysis.
    """
    cid = "test-stickiness-001"

    # Turn 1
    _patch_route_only(monkeypatch, mode="open_analysis", confidence="high")
    r1 = client.post("/ask", json={
        "question": "What does Hogwarts say about boarding-school culture?",
        "mode": "auto",
        "conversation_id": cid,
    })
    assert r1.status_code == 200
    assert r1.json()["routed_mode"] == "open_analysis"

    # Turn 2 — router classifies the bare "cool" as off-topic
    _patch_route_only(monkeypatch, mode="none", confidence="high")
    r2 = client.post("/ask", json={
        "question": "cool",
        "mode": "auto",
        "conversation_id": cid,
    })
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2["routed_mode"] == "none", "router still emits its raw classification"

    # The router.decision trace event records the actual dispatch decision
    decision = _events_of_kind(body2["turn_id"], "router.decision")[0]["attrs"]
    assert decision["prior_mode"] == "open_analysis"
    assert decision["routed_mode"] == "none"
    assert decision["effective_mode"] == "open_analysis", "stickiness must hold"
    assert decision["source"] == "sticky"


def test_auto_mode_continuation_high_confidence_switch_overrides_stickiness(
    client, monkeypatch,
):
    """SPEC AC: a clear mode-change intent on turn 2+ with high confidence
    is honored even if a different mode is in progress.

    Turn 1: open_analysis.
    Turn 2: router says "debate", confidence "high" → switch.
    """
    cid = "test-stickiness-002"

    # Turn 1
    _patch_route_only(monkeypatch, mode="open_analysis", confidence="high")
    r1 = client.post("/ask", json={
        "question": "What does Hogwarts say about boarding-school culture?",
        "mode": "auto",
        "conversation_id": cid,
    })
    assert r1.status_code == 200

    # Turn 2 — router emits a high-confidence mode change
    _patch_route_only(monkeypatch, mode="debate", confidence="high")
    r2 = client.post("/ask", json={
        "question": "Actually let's debate — was Voldemort redeemable?",
        "mode": "auto",
        "conversation_id": cid,
    })
    assert r2.status_code == 200

    decision = _events_of_kind(r2.json()["turn_id"], "router.decision")[0]["attrs"]
    assert decision["prior_mode"] == "open_analysis"
    assert decision["routed_mode"] == "debate"
    assert decision["effective_mode"] == "debate", "high-confidence override must fire"
    assert decision["source"] == "router-override"


def test_explicit_mode_skips_stickiness(client, monkeypatch):
    """If client sends an explicit non-auto mode, no router/stickiness logic
    runs — caller named the mode, we honor it."""
    # Even if we patch route_only, it must not be called
    agent = api_main._agents_cache["hp_lore"]
    monkeypatch.setattr(
        agent, "route_only",
        lambda text: pytest.fail("route_only must not run for explicit mode"),
    )

    r = client.post("/ask", json={
        "question": "Who killed Dumbledore?",
        "mode": "deep_research",
    })
    assert r.status_code == 200
    body = r.json()
    # No router.decision event when mode is explicit
    assert len(_events_of_kind(body["turn_id"], "router.decision")) == 0
