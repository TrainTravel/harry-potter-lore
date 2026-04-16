"""Tests for context_harness/tracer.py — Skill 6"""
import asyncio
import pytest
from context_harness.tracer import (
    Tracer, TraceEvent, TraceStore, EventKind,
    EvalMetrics, TurnRecord,
)


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------

def test_tracer_events_assigned_sequential_seq():
    tracer = Tracer(turn_id="t1")
    e1 = tracer.event(EventKind.TURN_START)
    e2 = tracer.event(EventKind.RETRIEVAL, {"chunks": "3"})
    e3 = tracer.event(EventKind.LLM_CALL, {"model": "gemini"})
    assert e1.seq == 1
    assert e2.seq == 2
    assert e3.seq == 3


def test_tracer_events_returns_all():
    tracer = Tracer(turn_id="t2")
    tracer.event(EventKind.RETRIEVAL)
    tracer.event(EventKind.LLM_CALL)
    assert len(tracer.events()) == 2


def test_tracer_events_of_kind_filter():
    tracer = Tracer(turn_id="t3")
    tracer.event(EventKind.RETRIEVAL)
    tracer.event(EventKind.LLM_CALL)
    tracer.event(EventKind.RETRIEVAL)
    retrievals = tracer.events_of_kind(EventKind.RETRIEVAL)
    assert len(retrievals) == 2


def test_tracer_has_errors_false_when_clean():
    tracer = Tracer(turn_id="t4")
    tracer.event(EventKind.LLM_CALL)
    assert tracer.has_errors() is False


def test_tracer_has_errors_true_when_error_present():
    tracer = Tracer(turn_id="t5")
    tracer.event(EventKind.ERROR, {"msg": "timeout"})
    assert tracer.has_errors() is True


def test_tracer_dump_is_string():
    tracer = Tracer(turn_id="t6")
    tracer.event(EventKind.TURN_START)
    tracer.event(EventKind.LLM_CALL, {"model": "gemini-flash"})
    dump = tracer.dump()
    assert isinstance(dump, str)
    assert "t6" in dump
    assert "llm_call" in dump.lower()


def test_tracer_dump_contains_all_events():
    tracer = Tracer(turn_id="t7")
    for kind in [EventKind.TURN_START, EventKind.RETRIEVAL, EventKind.LLM_CALL]:
        tracer.event(kind)
    dump = tracer.dump()
    assert "001" in dump
    assert "002" in dump
    assert "003" in dump


def test_tracer_turn_latency_positive():
    tracer = Tracer(turn_id="t8")
    tracer.event(EventKind.TURN_START)
    assert tracer.turn_latency_ms() >= 0


def test_tracer_latency_stored_in_event():
    tracer = Tracer(turn_id="t9")
    evt = tracer.event(EventKind.LLM_CALL, latency_ms=142.5)
    assert evt.latency_ms == 142.5


def test_tracer_payload_stored():
    tracer = Tracer(turn_id="t10")
    evt = tracer.event(EventKind.RETRIEVAL, {"query": "dumbledore", "chunks": "5"})
    assert evt.payload["query"] == "dumbledore"


# ---------------------------------------------------------------------------
# TraceStore
# ---------------------------------------------------------------------------

async def test_trace_store_save_and_retrieve(tmp_path):
    store = TraceStore(db_path=str(tmp_path / "traces.duckdb"))
    tracer = Tracer(turn_id="turn-abc", store=store)
    tracer.event(EventKind.TURN_START)
    tracer.event(EventKind.LLM_CALL, {"model": "gemini"})
    # Give the background tasks time to complete
    await asyncio.sleep(0.05)
    rows = store.get_turn("turn-abc")
    assert len(rows) == 2
    assert rows[0]["kind"] == "turn_start"
    assert rows[1]["kind"] == "llm_call"


# ---------------------------------------------------------------------------
# EvalMetrics
# ---------------------------------------------------------------------------

def test_eval_metrics_empty():
    metrics = EvalMetrics()
    assert metrics.total == 0
    assert metrics.success_rate == 0.0


def test_eval_metrics_success_rate():
    metrics = EvalMetrics()
    metrics.record(TurnRecord("t1", "s", "chat", success=True,  latency_ms=100))
    metrics.record(TurnRecord("t2", "s", "chat", success=True,  latency_ms=200))
    metrics.record(TurnRecord("t3", "s", "chat", success=False, latency_ms=50))
    assert abs(metrics.success_rate - 2/3) < 0.001


def test_eval_metrics_latency_percentiles():
    metrics = EvalMetrics()
    for ms in [100, 200, 300, 400, 500]:
        metrics.record(TurnRecord(f"t{ms}", "s", "chat", success=True, latency_ms=ms))
    assert metrics.latency_percentile(50) == 300
    assert metrics.latency_percentile(95) >= 400


def test_eval_metrics_cost_per_task():
    metrics = EvalMetrics()
    metrics.record(TurnRecord("t1", "s", "rag",  success=True, latency_ms=100, cost_usd=0.01))
    metrics.record(TurnRecord("t2", "s", "chat", success=True, latency_ms=100, cost_usd=0.02))
    rag_cost = metrics.cost_per_task("rag")
    assert abs(rag_cost - 0.01) < 1e-9


def test_eval_metrics_by_capability():
    metrics = EvalMetrics()
    metrics.record(TurnRecord("t1", "s", "chat", success=True,  latency_ms=100))
    metrics.record(TurnRecord("t2", "s", "chat", success=False, latency_ms=200))
    metrics.record(TurnRecord("t3", "s", "quiz", success=True,  latency_ms=300))
    by_cap = metrics.by_capability()
    assert "chat" in by_cap
    assert "quiz" in by_cap
    assert by_cap["chat"]["count"] == 2
    assert by_cap["quiz"]["count"] == 1


def test_eval_metrics_summary_keys():
    metrics = EvalMetrics()
    metrics.record(TurnRecord("t1", "s", "chat", success=True, latency_ms=150))
    summary = metrics.summary()
    assert "total_turns" in summary
    assert "success_rate" in summary
    assert "p50_latency_ms" in summary
    assert "p95_latency_ms" in summary
    assert "avg_cost_usd" in summary
    assert "by_capability" in summary
