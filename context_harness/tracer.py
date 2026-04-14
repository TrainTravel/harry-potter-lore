"""
Tracer and Eval Metrics — Skill 6: Evaluation and Observability
===============================================================
The IBM video's point: "when an agent fails, trace backward through the logs
to find the root cause — don't just tweak the prompt."

This module implements:
  - TraceEvent  — a single structured event (tool call, LLM call, retrieval, etc.)
  - Tracer      — records all events for a turn in sequence-number order
  - TraceStore  — persists traces to SQLite for post-hoc debugging
  - EvalMetrics — aggregates success rate, latency (p50/p95), cost per task

Design principle:
  "Every tool call, every retrieval, every LLM decision gets a sequence number.
   When something goes wrong you replay the trace, not the user."
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class EventKind(str, Enum):
    TURN_START   = "turn_start"
    TURN_END     = "turn_end"
    LLM_CALL     = "llm_call"
    TOOL_CALL    = "tool_call"
    TOOL_RESULT  = "tool_result"
    RETRIEVAL    = "retrieval"
    CONTEXT_BUILT= "context_built"
    ESCALATION   = "escalation"
    ERROR        = "error"
    SECURITY     = "security"


@dataclass
class TraceEvent:
    turn_id: str
    seq: int
    kind: EventKind
    payload: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    latency_ms: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "seq": self.seq,
            "kind": self.kind.value,
            "payload": self.payload,
            "ts": self.ts,
            "latency_ms": self.latency_ms,
        }


# ---------------------------------------------------------------------------
# Tracer (per-turn)
# ---------------------------------------------------------------------------

class Tracer:
    """
    Records every event within a turn in strict sequence order.

    One Tracer instance per turn. The trace is the complete audit log of
    what the agent did — from context assembly through final response.

    Usage:
        tracer = Tracer(turn_id="abc-123")
        tracer.event(EventKind.RETRIEVAL, {"query": "...", "chunks": 5})
        tracer.event(EventKind.LLM_CALL,  {"model": "gemini-1.5-flash", "tokens_in": 512})
        tracer.dump()   # print full trace for debugging
    """

    def __init__(self, turn_id: str, store: Optional["TraceStore"] = None) -> None:
        self._turn_id = turn_id
        self._seq = 0
        self._events: List[TraceEvent] = []
        self._store = store
        self._start = time.time()

    def event(
        self,
        kind: EventKind,
        payload: Dict[str, Any] = None,
        latency_ms: Optional[float] = None,
    ) -> TraceEvent:
        self._seq += 1
        evt = TraceEvent(
            turn_id=self._turn_id,
            seq=self._seq,
            kind=kind,
            payload=payload or {},
            latency_ms=latency_ms,
        )
        self._events.append(evt)
        if self._store:
            # Fire-and-forget — never block the turn
            asyncio.create_task(self._store.save(evt))
        return evt

    def events(self) -> List[TraceEvent]:
        return list(self._events)

    def events_of_kind(self, kind: EventKind) -> List[TraceEvent]:
        return [e for e in self._events if e.kind == kind]

    def turn_latency_ms(self) -> float:
        return (time.time() - self._start) * 1000

    def dump(self) -> str:
        """Human-readable trace for debugging — exactly what the IBM video recommends."""
        lines = [f"=== Trace: turn_id={self._turn_id} ==="]
        for evt in self._events:
            lat = f" [{evt.latency_ms:.1f}ms]" if evt.latency_ms is not None else ""
            payload_str = json.dumps(evt.payload, default=str)[:120]
            lines.append(f"  [{evt.seq:03d}] {evt.kind.value:<18}{lat} {payload_str}")
        lines.append(f"  Total turn latency: {self.turn_latency_ms():.1f}ms")
        return "\n".join(lines)

    def error_events(self) -> List[TraceEvent]:
        return self.events_of_kind(EventKind.ERROR)

    def has_errors(self) -> bool:
        return bool(self.error_events())


# ---------------------------------------------------------------------------
# Trace Store (SQLite)
# ---------------------------------------------------------------------------

class TraceStore:
    """Persists trace events to SQLite for post-hoc debugging."""

    def __init__(self, db_path: str = "data/traces.db") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = asyncio.Lock()
        self._init()

    def _init(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS trace_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                turn_id     TEXT NOT NULL,
                seq         INTEGER NOT NULL,
                kind        TEXT NOT NULL,
                payload     TEXT NOT NULL,
                ts          REAL NOT NULL,
                latency_ms  REAL
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_turn ON trace_events(turn_id)")
        self._conn.commit()

    async def save(self, evt: TraceEvent) -> None:
        async with self._lock:
            self._conn.execute(
                "INSERT INTO trace_events (turn_id,seq,kind,payload,ts,latency_ms) VALUES (?,?,?,?,?,?)",
                (evt.turn_id, evt.seq, evt.kind.value,
                 json.dumps(evt.payload, default=str), evt.ts, evt.latency_ms),
            )
            self._conn.commit()

    def get_turn(self, turn_id: str) -> List[dict]:
        rows = self._conn.execute(
            "SELECT seq,kind,payload,ts,latency_ms FROM trace_events "
            "WHERE turn_id=? ORDER BY seq",
            (turn_id,),
        ).fetchall()
        return [
            {"seq": s, "kind": k, "payload": json.loads(p), "ts": t, "latency_ms": lm}
            for s, k, p, t, lm in rows
        ]


# ---------------------------------------------------------------------------
# Eval Metrics — Skill 6 (data-driven evaluation)
# ---------------------------------------------------------------------------

@dataclass
class TurnRecord:
    turn_id: str
    session_id: str
    capability: str
    success: bool
    latency_ms: float
    cost_usd: float = 0.0
    error: str = ""
    ts: float = field(default_factory=time.time)


class EvalMetrics:
    """
    Aggregates success rate, latency percentiles, and cost per task.

    "Move away from guessing — use data-driven metrics." — IBM video

    Usage:
        metrics = EvalMetrics()
        metrics.record(TurnRecord(turn_id=..., success=True, latency_ms=420, ...))
        print(metrics.summary())
    """

    def __init__(self) -> None:
        self._records: List[TurnRecord] = []

    def record(self, record: TurnRecord) -> None:
        self._records.append(record)

    @property
    def total(self) -> int:
        return len(self._records)

    @property
    def success_rate(self) -> float:
        if not self._records:
            return 0.0
        return sum(1 for r in self._records if r.success) / len(self._records)

    def latency_percentile(self, p: float) -> float:
        """p=50 → median, p=95 → p95, etc."""
        if not self._records:
            return 0.0
        sorted_lat = sorted(r.latency_ms for r in self._records)
        idx = int(len(sorted_lat) * p / 100)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    def cost_per_task(self, capability: Optional[str] = None) -> float:
        records = self._records
        if capability:
            records = [r for r in records if r.capability == capability]
        if not records:
            return 0.0
        return sum(r.cost_usd for r in records) / len(records)

    def by_capability(self) -> Dict[str, dict]:
        caps: Dict[str, List[TurnRecord]] = {}
        for r in self._records:
            caps.setdefault(r.capability, []).append(r)
        return {
            cap: {
                "count": len(rs),
                "success_rate": sum(1 for r in rs if r.success) / len(rs),
                "p50_ms": _percentile([r.latency_ms for r in rs], 50),
                "p95_ms": _percentile([r.latency_ms for r in rs], 95),
                "avg_cost_usd": sum(r.cost_usd for r in rs) / len(rs),
            }
            for cap, rs in caps.items()
        }

    def summary(self) -> dict:
        return {
            "total_turns": self.total,
            "success_rate": round(self.success_rate, 4),
            "p50_latency_ms": round(self.latency_percentile(50), 1),
            "p95_latency_ms": round(self.latency_percentile(95), 1),
            "avg_cost_usd": round(
                sum(r.cost_usd for r in self._records) / max(self.total, 1), 6
            ),
            "by_capability": self.by_capability(),
        }


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(int(len(s) * p / 100), len(s) - 1)]
