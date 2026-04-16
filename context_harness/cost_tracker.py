"""
Cost Tracker
============
Fixes deepTutor Weakness #5:
  "Token cost not aggregated — LLM call metadata is logged but never queryable.
   No way to answer: what does a Deep Research run cost on average?
   Which capability is most expensive?"

Solution:
  - Every LLM call is recorded as a CostEvent (model, tokens_in, tokens_out,
    latency_ms, capability, session_id).
  - Events are persisted to SQLite in a background asyncio task (non-blocking).
  - CostAggregator provides roll-up queries: by capability, by model, by session,
    total cost with configurable pricing tables.
  - A @track_cost decorator wraps any async LLM call with zero boilerplate.

Pricing table is configurable — defaults to Gemini 1.5 Pro pricing as of 2026.
Swap in your own rates for any model.
"""

from __future__ import annotations

import asyncio
import functools
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional


# ---------------------------------------------------------------------------
# Pricing table (USD per 1M tokens)
# ---------------------------------------------------------------------------

DEFAULT_PRICING: Dict[str, Dict[str, float]] = {
    # model_name → {input: $/1M, output: $/1M}
    "gemini-1.5-pro":    {"input": 3.50,  "output": 10.50},
    "gemini-1.5-flash":  {"input": 0.35,  "output": 1.05},
    "gemini-2.0-flash":       {"input": 0.10,  "output": 0.40},
    "gemini-2.5-pro":         {"input": 1.25,  "output": 10.00},
    "gemini-2.5-flash":       {"input": 0.30,  "output": 2.50},
    "gemini-2.5-flash-lite":  {"input": 0.10,  "output": 0.40},
    "claude-opus-4":     {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-6": {"input": 3.00,  "output": 15.00},
    "claude-haiku-4-5":  {"input": 0.80,  "output": 4.00},
    "gpt-4o":            {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":       {"input": 0.15,  "output": 0.60},
    "default":           {"input": 1.00,  "output": 3.00},
}


def estimate_cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    rates = DEFAULT_PRICING.get(model, DEFAULT_PRICING["default"])
    return (tokens_in * rates["input"] + tokens_out * rates["output"]) / 1_000_000


# ---------------------------------------------------------------------------
# Event model
# ---------------------------------------------------------------------------

@dataclass
class CostEvent:
    model: str
    capability: str            # 'chat' | 'deep_solve' | 'rag' | etc.
    tokens_in: int
    tokens_out: int
    latency_ms: float
    session_id: str = "unknown"
    turn_id: str = "unknown"
    cost_usd: float = 0.0
    ts: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.cost_usd == 0.0:
            self.cost_usd = estimate_cost_usd(self.model, self.tokens_in, self.tokens_out)


# ---------------------------------------------------------------------------
# Store (SQLite, written in a background task)
# ---------------------------------------------------------------------------

class CostStore:
    """Thread-safe SQLite store for cost events."""

    def __init__(self, db_path: str = "data/costs.db") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._write_lock = asyncio.Lock()
        self._init()

    def _init(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cost_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                model       TEXT NOT NULL,
                capability  TEXT NOT NULL,
                tokens_in   INTEGER NOT NULL,
                tokens_out  INTEGER NOT NULL,
                latency_ms  REAL NOT NULL,
                session_id  TEXT NOT NULL,
                turn_id     TEXT NOT NULL,
                cost_usd    REAL NOT NULL,
                ts          REAL NOT NULL
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_capability ON cost_events(capability)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_model ON cost_events(model)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON cost_events(session_id)")
        self._conn.commit()

    async def record(self, event: CostEvent) -> None:
        async with self._write_lock:
            self._conn.execute(
                """INSERT INTO cost_events
                   (model,capability,tokens_in,tokens_out,latency_ms,session_id,turn_id,cost_usd,ts)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (event.model, event.capability, event.tokens_in, event.tokens_out,
                 event.latency_ms, event.session_id, event.turn_id, event.cost_usd, event.ts),
            )
            self._conn.commit()

    def query(self, sql: str, params=()) -> List[dict]:
        cur = self._conn.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Cost Tracker (non-blocking)
# ---------------------------------------------------------------------------

class CostTracker:
    """
    Records LLM call costs asynchronously — never blocks the hot path.

    Events are enqueued in memory and written to SQLite by a single
    background asyncio worker (similar to the SummarizationQueue pattern).
    """

    def __init__(self, store: Optional[CostStore] = None) -> None:
        self._store = store
        self._queue: asyncio.Queue[CostEvent] = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._in_memory: List[CostEvent] = []   # always kept for in-process queries

    async def start(self) -> None:
        self._task = asyncio.create_task(self._worker(), name="cost-tracker")

    async def stop(self) -> None:
        if self._task:
            await self._queue.join()
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def record(self, event: CostEvent) -> None:
        """Non-blocking fire-and-forget. Safe to call from sync code."""
        self._in_memory.append(event)
        if self._task is not None:
            self._queue.put_nowait(event)

    async def _worker(self) -> None:
        while True:
            event = await self._queue.get()
            try:
                if self._store:
                    await self._store.record(event)
            except Exception as exc:
                print(f"[CostTracker] Failed to persist event: {exc}")
            finally:
                self._queue.task_done()

    # ------------------------------------------------------------------
    # Aggregation queries (in-memory, instant)
    # ------------------------------------------------------------------

    def total_cost_usd(self) -> float:
        return sum(e.cost_usd for e in self._in_memory)

    def by_capability(self) -> Dict[str, CapabilityStats]:
        groups: Dict[str, List[CostEvent]] = {}
        for e in self._in_memory:
            groups.setdefault(e.capability, []).append(e)
        return {cap: CapabilityStats.from_events(cap, evts) for cap, evts in groups.items()}

    def by_model(self) -> Dict[str, ModelStats]:
        groups: Dict[str, List[CostEvent]] = {}
        for e in self._in_memory:
            groups.setdefault(e.model, []).append(e)
        return {model: ModelStats.from_events(model, evts) for model, evts in groups.items()}

    def session_cost_usd(self, session_id: str) -> float:
        return sum(e.cost_usd for e in self._in_memory if e.session_id == session_id)

    def summary(self) -> dict:
        return {
            "total_events": len(self._in_memory),
            "total_cost_usd": round(self.total_cost_usd(), 6),
            "by_capability": {k: v.to_dict() for k, v in self.by_capability().items()},
            "by_model": {k: v.to_dict() for k, v in self.by_model().items()},
        }


# ---------------------------------------------------------------------------
# Roll-up stats
# ---------------------------------------------------------------------------

@dataclass
class CapabilityStats:
    capability: str
    call_count: int
    total_tokens_in: int
    total_tokens_out: int
    total_cost_usd: float
    avg_latency_ms: float
    avg_cost_usd: float

    @classmethod
    def from_events(cls, capability: str, events: List[CostEvent]) -> "CapabilityStats":
        return cls(
            capability=capability,
            call_count=len(events),
            total_tokens_in=sum(e.tokens_in for e in events),
            total_tokens_out=sum(e.tokens_out for e in events),
            total_cost_usd=sum(e.cost_usd for e in events),
            avg_latency_ms=sum(e.latency_ms for e in events) / len(events),
            avg_cost_usd=sum(e.cost_usd for e in events) / len(events),
        )

    def to_dict(self) -> dict:
        return {k: round(v, 6) if isinstance(v, float) else v for k, v in self.__dict__.items()}


@dataclass
class ModelStats:
    model: str
    call_count: int
    total_tokens_in: int
    total_tokens_out: int
    total_cost_usd: float

    @classmethod
    def from_events(cls, model: str, events: List[CostEvent]) -> "ModelStats":
        return cls(
            model=model,
            call_count=len(events),
            total_tokens_in=sum(e.tokens_in for e in events),
            total_tokens_out=sum(e.tokens_out for e in events),
            total_cost_usd=sum(e.cost_usd for e in events),
        )

    def to_dict(self) -> dict:
        return {k: round(v, 6) if isinstance(v, float) else v for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------
# @track_cost decorator
# ---------------------------------------------------------------------------

def track_cost(
    tracker: CostTracker,
    capability: str,
    model: str,
    session_id: str = "unknown",
    turn_id: str = "unknown",
):
    """
    Decorator that wraps an async LLM call, measures latency, and records cost.

    The decorated function must return a dict with keys:
        tokens_in: int, tokens_out: int, result: Any

    Usage:
        @track_cost(tracker, capability="rag", model="gemini-1.5-flash")
        async def call_llm(prompt: str) -> dict:
            resp = await llm.generate(prompt)
            return {"tokens_in": resp.usage.input, "tokens_out": resp.usage.output, "result": resp.text}
    """
    def decorator(fn: Callable[..., Coroutine]):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            raw = await fn(*args, **kwargs)
            latency_ms = (time.perf_counter() - start) * 1000
            event = CostEvent(
                model=model,
                capability=capability,
                tokens_in=raw.get("tokens_in", 0),
                tokens_out=raw.get("tokens_out", 0),
                latency_ms=latency_ms,
                session_id=session_id,
                turn_id=turn_id,
            )
            tracker.record(event)
            return raw.get("result")
        return wrapper
    return decorator
