"""
FastAPI HTTP interface for the context harness
===============================================
Exposes trace replay, health checks, and (optionally) cost data over HTTP.

Endpoints:
  GET  /health                — liveness probe
  GET  /trace/{turn_id}       — replay a single turn's event sequence as JSON
  GET  /traces                — list recent turn IDs (most-recent-first)

Usage:
    uvicorn context_harness.app:app --reload --port 8000
    # then: curl http://localhost:8000/trace/<turn_id>

OTel (optional):
    Set OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 and the FastAPI
    instrumentation picks up traces automatically via the OTel SDK.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from .tracer import TraceStore

# ---------------------------------------------------------------------------
# App + shared store
# ---------------------------------------------------------------------------

app = FastAPI(
    title="deepTutor Context Harness",
    description="Trace replay and observability endpoints for the HP lore agent.",
    version="1.0.0",
)

_DB_PATH = os.environ.get("TRACE_DB_PATH", "data/traces.db")
_store: Optional[TraceStore] = None


def _get_store() -> TraceStore:
    global _store
    if _store is None:
        _store = TraceStore(db_path=_DB_PATH)
    return _store


# ---------------------------------------------------------------------------
# OTel FastAPI instrumentation (auto-enabled if SDK is configured)
# ---------------------------------------------------------------------------

try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    FastAPIInstrumentor.instrument_app(app)
except Exception:
    pass  # OTel not configured — silently skip


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    """Liveness probe — always returns 200 while the process is alive."""
    db_exists = Path(_DB_PATH).exists()
    return {"status": "ok", "trace_db": str(db_exists)}


@app.get("/traces")
def list_traces(
    limit: int = Query(default=20, ge=1, le=200, description="Max number of turns to return"),
) -> Dict[str, Any]:
    """
    Return the most-recent `limit` turn IDs together with their event counts.

    Example response:
        {
          "turns": [
            {"turn_id": "abc-123", "event_count": 7, "first_ts": 1713200000.0},
            ...
          ]
        }
    """
    db_path = _DB_PATH
    if not Path(db_path).exists():
        return {"turns": []}

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT turn_id, COUNT(*) AS events, MIN(ts) AS first_ts "
        "FROM trace_events GROUP BY turn_id ORDER BY first_ts DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()

    return {
        "turns": [
            {"turn_id": tid, "event_count": cnt, "first_ts": ts}
            for tid, cnt, ts in rows
        ]
    }


@app.get("/trace/{turn_id}")
def get_trace(turn_id: str) -> Dict[str, Any]:
    """
    Replay a single turn's event sequence.

    Returns every TraceEvent recorded for this turn in sequence order, plus
    a computed `wall_latency_ms` (last_ts − first_ts) and an `error_count`.

    Raises 404 if no events exist for the given turn_id.

    Example response:
        {
          "turn_id": "abc-123",
          "event_count": 7,
          "wall_latency_ms": 1420.5,
          "error_count": 0,
          "events": [
            {"seq": 1, "kind": "turn_start",  "payload": {}, "ts": ..., "latency_ms": null},
            {"seq": 2, "kind": "retrieval",   "payload": {"query": "...", "chunks": 5}, ...},
            ...
          ]
        }
    """
    store = _get_store()
    events: List[Dict[str, Any]] = store.get_turn(turn_id)

    if not events:
        raise HTTPException(
            status_code=404,
            detail=f"No trace events found for turn_id={turn_id!r}",
        )

    wall_ms = (events[-1]["ts"] - events[0]["ts"]) * 1000 if len(events) > 1 else 0.0
    error_count = sum(1 for e in events if e["kind"] == "error")

    return {
        "turn_id":        turn_id,
        "event_count":    len(events),
        "wall_latency_ms": round(wall_ms, 2),
        "error_count":    error_count,
        "events":         events,
    }
