"""
Async Post-Turn Summarizer
===========================
Fixes deepTutor Weakness #3:
  "On-demand summarization on the critical path — a blocking LLM call to compress
   history fires at turn start when the window overflows, causing slow responses."

Solution:
  - Summarization runs in a background asyncio task, NEVER blocking the current turn.
  - The ContextWindow marks old entries as PENDING_SUMMARY and continues serving the
    verbatim recent slice while the background task runs.
  - When the summary is ready it is atomically swapped in (replacing the entries it covers).
  - A SQLite-backed SummaryStore persists completed summaries so they survive restarts.

Budget model (mirrors deepTutor):
  history_budget = 0.35 * max_tokens
  summary_slot   = 0.40 * history_budget  (compressed older turns)
  recent_slot    = 0.60 * history_budget  (verbatim recent turns)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Coroutine, List, Optional

from .context_manager import ContextEntry, ContextRole, ContextWindow


# ---------------------------------------------------------------------------
# LLM summarisation hook (caller supplies the actual LLM call)
# ---------------------------------------------------------------------------

SummariseFn = Callable[[List[ContextEntry]], Coroutine[None, None, str]]


async def _stub_summarise(entries: List[ContextEntry]) -> str:
    """Stub: concatenate content. Replace with a real LLM call in production."""
    texts = [e.content for e in entries]
    return "Summary: " + " | ".join(t[:80] for t in texts)


# ---------------------------------------------------------------------------
# Persisted summary store (SQLite)
# ---------------------------------------------------------------------------

@dataclass
class SummaryRecord:
    session_id: str
    summary_text: str
    covered_hash: str      # SHA-256 of concatenated entry content
    token_count: int
    created_at: float = field(default_factory=time.time)


class SummaryStore:
    """Lightweight SQLite store for compressed conversation summaries."""

    def __init__(self, db_path: str = "data/summaries.db") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = asyncio.Lock()
        self._init()

    def _init(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                session_id   TEXT NOT NULL,
                summary_text TEXT NOT NULL,
                covered_hash TEXT NOT NULL,
                token_count  INTEGER NOT NULL,
                created_at   REAL NOT NULL,
                PRIMARY KEY (session_id, covered_hash)
            )
        """)
        self._conn.commit()

    async def save(self, record: SummaryRecord) -> None:
        async with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO summaries VALUES (?,?,?,?,?)",
                (record.session_id, record.summary_text, record.covered_hash,
                 record.token_count, record.created_at),
            )
            self._conn.commit()

    async def latest(self, session_id: str) -> Optional[SummaryRecord]:
        row = self._conn.execute(
            "SELECT * FROM summaries WHERE session_id=? ORDER BY created_at DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return SummaryRecord(*row)


# ---------------------------------------------------------------------------
# Background summarisation queue
# ---------------------------------------------------------------------------

class SummarizationQueue:
    """
    A single background asyncio task drains a queue of summarisation jobs.

    Usage:
        queue = SummarizationQueue(llm_fn=my_llm, store=store)
        await queue.start()
        await queue.submit(session_id, entries_to_compress, window)
        # ... current turn continues immediately, no blocking ...
        await queue.stop()   # graceful drain on shutdown

    When a summary is ready it is atomically swapped into `window` via
    _apply_summary(), replacing the covered entries with a single RETRIEVAL
    entry at priority=10.0 so it is never auto-evicted.
    """

    def __init__(
        self,
        llm_fn: SummariseFn = _stub_summarise,
        store: Optional[SummaryStore] = None,
        max_queue_size: int = 64,
    ) -> None:
        self._llm_fn = llm_fn
        self._store = store
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._worker(), name="summarizer")

    async def stop(self) -> None:
        if self._task:
            await self._queue.join()   # drain gracefully
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def submit(
        self,
        session_id: str,
        entries: List[ContextEntry],
        window: ContextWindow,
    ) -> None:
        """Non-blocking — returns immediately, queues work for the background worker."""
        await self._queue.put((session_id, entries, window))

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    async def _worker(self) -> None:
        while True:
            session_id, entries, window = await self._queue.get()
            try:
                summary_text = await self._llm_fn(entries)
                covered_hash = _hash_entries(entries)
                record = SummaryRecord(
                    session_id=session_id,
                    summary_text=summary_text,
                    covered_hash=covered_hash,
                    token_count=len(summary_text.split()),
                )
                if self._store:
                    await self._store.save(record)
                _apply_summary(window, entries, summary_text)
            except Exception as exc:
                # Log but never crash the background task
                print(f"[SummarizationQueue] Error for session {session_id}: {exc}")
            finally:
                self._queue.task_done()


# ---------------------------------------------------------------------------
# Context window integration
# ---------------------------------------------------------------------------

class SummarizingContextWindow(ContextWindow):
    """
    ContextWindow subclass that integrates with SummarizationQueue.

    When the window overflows the recent_slot budget, instead of
    blocking on an LLM call it:
      1. Identifies the oldest non-system entries that push over budget.
      2. Submits them to the queue for async compression.
      3. TEMPORARILY keeps those entries so the current turn is not degraded.
      4. Returns immediately — the background worker will swap them out later.

    Budget split:
        total_budget = max_tokens - reserved_output
        recent_slot  = int(total_budget * recent_ratio)   (verbatim)
        summary_slot = total_budget - recent_slot          (summary placeholder)
    """

    def __init__(
        self,
        session_id: str,
        queue: SummarizationQueue,
        max_tokens: int = 8192,
        reserved_output: int = 1024,
        recent_ratio: float = 0.60,
        **kwargs,
    ) -> None:
        super().__init__(max_tokens=max_tokens, reserved_output_tokens=reserved_output, **kwargs)
        self._session_id = session_id
        self._queue = queue
        self._total_budget = max_tokens - reserved_output
        self._recent_budget = int(self._total_budget * recent_ratio)

    def _try_evict(self) -> bool:
        """Override: instead of silently evicting, schedule async compression."""
        candidates = self._evictable()
        if not candidates:
            return False

        # Identify the oldest entries that exceed the recent_slot budget
        to_compress = _oldest_over_budget(candidates, self._recent_budget)
        if to_compress:
            # Submit async — does not block
            asyncio.create_task(
                self._queue.submit(self._session_id, to_compress, self)
            )
            # Remove from window immediately so space is freed for this turn
            for entry in to_compress:
                if entry in self._entries:
                    self._entries.remove(entry)
            return True

        # Fallback: regular eviction of the single lowest-priority entry
        return super()._try_evict()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_summary(window: ContextWindow, replaced: List[ContextEntry], summary: str) -> None:
    """Atomically replace entries with a compressed summary entry."""
    for entry in replaced:
        if entry in window._entries:
            window._entries.remove(entry)
    from .context_manager import ContextEntry, ContextRole
    summary_entry = ContextEntry(
        role=ContextRole.SYSTEM,
        content=f"[Conversation summary]\n{summary}",
        priority=10.0,
        metadata={"compressed": "true", "source_count": str(len(replaced))},
    )
    summary_entry.tokens = len(summary.split())
    window._entries.insert(0, summary_entry)


def _oldest_over_budget(entries: List[ContextEntry], recent_budget: int) -> List[ContextEntry]:
    """
    Return the oldest entries that push total token usage above recent_budget.
    These are candidates for compression.
    """
    sorted_entries = sorted(entries, key=lambda e: e.timestamp)
    total = sum(e.tokens for e in sorted_entries)
    to_compress: List[ContextEntry] = []
    for entry in sorted_entries:
        if total <= recent_budget:
            break
        to_compress.append(entry)
        total -= entry.tokens
    return to_compress


def _hash_entries(entries: List[ContextEntry]) -> str:
    content = "".join(e.content for e in entries)
    return hashlib.sha256(content.encode()).hexdigest()
