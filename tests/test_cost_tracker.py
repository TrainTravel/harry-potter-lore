"""Tests for context_harness/cost_tracker.py"""
import asyncio
import pytest
from context_harness.cost_tracker import (
    CostTracker, CostEvent, estimate_cost_usd, DEFAULT_PRICING,
)


# ---------------------------------------------------------------------------
# estimate_cost_usd
# ---------------------------------------------------------------------------

def test_estimate_cost_known_model():
    cost = estimate_cost_usd("gpt-4o", tokens_in=1_000_000, tokens_out=0)
    assert abs(cost - 2.50) < 0.01


def test_estimate_cost_unknown_model_uses_default():
    cost = estimate_cost_usd("unknown-model-xyz", tokens_in=1_000_000, tokens_out=0)
    assert cost == DEFAULT_PRICING["default"]["input"]


def test_estimate_cost_zero_tokens():
    assert estimate_cost_usd("gpt-4o", 0, 0) == 0.0


def test_cost_event_auto_calculates_cost():
    event = CostEvent(
        model="gpt-4o-mini", capability="rag",
        tokens_in=100_000, tokens_out=10_000,
        latency_ms=200.0,
    )
    assert event.cost_usd > 0.0


# ---------------------------------------------------------------------------
# CostTracker — sync recording
# ---------------------------------------------------------------------------

def test_tracker_records_event():
    tracker = CostTracker()
    event = CostEvent(model="gemini-1.5-flash", capability="chat",
                      tokens_in=500, tokens_out=100, latency_ms=50.0)
    tracker.record(event)
    assert tracker.total_cost_usd() > 0.0


def test_tracker_total_cost_accumulates():
    tracker = CostTracker()
    for _ in range(3):
        tracker.record(CostEvent("gemini-1.5-flash", "chat", 1000, 200, 50.0))
    single = estimate_cost_usd("gemini-1.5-flash", 1000, 200)
    assert abs(tracker.total_cost_usd() - single * 3) < 1e-9


def test_tracker_by_capability_groups_correctly():
    tracker = CostTracker()
    tracker.record(CostEvent("gemini-1.5-flash", "rag",  500, 100, 30.0))
    tracker.record(CostEvent("gemini-1.5-flash", "chat", 500, 100, 30.0))
    tracker.record(CostEvent("gemini-1.5-flash", "rag",  500, 100, 30.0))
    by_cap = tracker.by_capability()
    assert "rag" in by_cap
    assert "chat" in by_cap
    assert by_cap["rag"].call_count == 2
    assert by_cap["chat"].call_count == 1


def test_tracker_by_model():
    tracker = CostTracker()
    tracker.record(CostEvent("gpt-4o",      "chat", 100, 50, 20.0))
    tracker.record(CostEvent("gpt-4o-mini", "chat", 100, 50, 20.0))
    by_model = tracker.by_model()
    assert "gpt-4o" in by_model
    assert "gpt-4o-mini" in by_model


def test_tracker_session_cost():
    tracker = CostTracker()
    tracker.record(CostEvent("gpt-4o", "chat", 1000, 200, 50.0, session_id="s1"))
    tracker.record(CostEvent("gpt-4o", "chat", 1000, 200, 50.0, session_id="s2"))
    s1_cost = tracker.session_cost_usd("s1")
    total = tracker.total_cost_usd()
    assert s1_cost < total
    assert s1_cost > 0.0


def test_tracker_summary_keys():
    tracker = CostTracker()
    tracker.record(CostEvent("gpt-4o", "rag", 500, 100, 30.0))
    summary = tracker.summary()
    assert "total_events" in summary
    assert "total_cost_usd" in summary
    assert "by_capability" in summary
    assert "by_model" in summary


# ---------------------------------------------------------------------------
# CostTracker — async (background worker)
# ---------------------------------------------------------------------------

async def test_tracker_async_start_stop():
    tracker = CostTracker()
    await tracker.start()
    tracker.record(CostEvent("gemini-1.5-flash", "rag", 100, 50, 10.0))
    await tracker.stop()
    assert tracker.total_cost_usd() > 0.0
