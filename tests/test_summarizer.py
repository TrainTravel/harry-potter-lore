"""
Tests for context_harness/summarizer.py

Coverage:
  - _oldest_over_budget      — pure helper, budget selection logic
  - _apply_summary           — atomic swap of entries into a window
  - SummaryStore             — SQLite persistence (uses tmp path)
  - SummarizationQueue       — async background worker; submit + drain
  - SummarizingContextWindow — overflow triggers async compression
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from context_harness.context_manager import ContextEntry, ContextRole, ContextWindow
from context_harness.summarizer import (
    SummarizationQueue,
    SummarizingContextWindow,
    SummaryStore,
    _apply_summary,
    _hash_entries,
    _oldest_over_budget,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entry(content: str, role: ContextRole = ContextRole.USER, priority: float = 1.0, ts: float | None = None) -> ContextEntry:
    e = ContextEntry(role=role, content=content, priority=priority)
    if ts is not None:
        e.timestamp = ts
    return e


async def _instant_summarise(entries: list[ContextEntry]) -> str:
    return "SUMMARY:" + "|".join(e.content[:20] for e in entries)


# ---------------------------------------------------------------------------
# _oldest_over_budget
# ---------------------------------------------------------------------------

class TestOldestOverBudget:
    def test_empty_list_returns_empty(self):
        assert _oldest_over_budget([], 100) == []

    def test_within_budget_returns_nothing(self):
        entries = [_entry("hello world", ts=1.0), _entry("foo bar", ts=2.0)]
        # total ≈ 4 tokens, budget=10 → nothing to compress
        assert _oldest_over_budget(entries, 100) == []

    def test_selects_oldest_entries_first(self):
        e1 = _entry("alpha beta gamma delta", ts=1.0)   # 4 tokens
        e2 = _entry("epsilon zeta eta theta", ts=2.0)   # 4 tokens
        e3 = _entry("iota kappa lambda mu",   ts=3.0)   # 4 tokens
        # total=12, budget=6 → need to shed 6 tokens → compress e1 + e2
        result = _oldest_over_budget([e3, e1, e2], recent_budget=6)
        assert result == [e1, e2]

    def test_stops_as_soon_as_budget_met(self):
        e1 = _entry("a b c d e f g h i j", ts=1.0)   # 10 tokens
        e2 = _entry("k l m n o p q r s t", ts=2.0)   # 10 tokens
        # total=20, budget=15 → shed 5 → e1 alone covers it (10 tokens)
        result = _oldest_over_budget([e1, e2], recent_budget=15)
        assert result == [e1]

    def test_all_entries_over_budget(self):
        e1 = _entry("a b c", ts=1.0)
        e2 = _entry("d e f", ts=2.0)
        result = _oldest_over_budget([e1, e2], recent_budget=0)
        assert result == [e1, e2]


# ---------------------------------------------------------------------------
# _apply_summary
# ---------------------------------------------------------------------------

class TestApplySummary:
    def _window_with_entries(self, *contents: str) -> tuple[ContextWindow, list[ContextEntry]]:
        w = ContextWindow(max_tokens=4096)
        entries = []
        for c in contents:
            e = ContextEntry(role=ContextRole.USER, content=c)
            w._entries.append(e)
            entries.append(e)
        return w, entries

    def test_replaced_entries_removed_from_window(self):
        w, entries = self._window_with_entries("turn one", "turn two", "turn three")
        _apply_summary(w, entries[:2], "summary text here")
        contents = [e.content for e in w._entries]
        assert "turn one" not in contents
        assert "turn two" not in contents
        assert "turn three" in contents

    def test_summary_entry_inserted(self):
        w, entries = self._window_with_entries("turn one", "turn two")
        _apply_summary(w, entries, "compressed summary")
        assert any("compressed summary" in e.content for e in w._entries)

    def test_summary_entry_has_high_priority(self):
        w, entries = self._window_with_entries("turn one")
        _apply_summary(w, entries, "summary")
        summary_entry = w._entries[0]
        assert summary_entry.priority == 10.0

    def test_summary_entry_role_is_system(self):
        w, entries = self._window_with_entries("turn one")
        _apply_summary(w, entries, "summary")
        assert w._entries[0].role == ContextRole.SYSTEM

    def test_unreferenced_entries_untouched(self):
        w, entries = self._window_with_entries("keep me", "compress me")
        _apply_summary(w, [entries[1]], "summary")
        assert any("keep me" in e.content for e in w._entries)

    def test_window_shrinks_by_replaced_count(self):
        w, entries = self._window_with_entries("a", "b", "c", "d")
        original_len = len(w._entries)
        _apply_summary(w, entries[:3], "short summary")
        # 4 entries → remove 3, add 1 summary → 2 total
        assert len(w._entries) == original_len - 3 + 1


# ---------------------------------------------------------------------------
# SummaryStore
# ---------------------------------------------------------------------------

class TestSummaryStore:
    @pytest.fixture
    def store(self, tmp_path: Path) -> SummaryStore:
        return SummaryStore(db_path=str(tmp_path / "summaries.db"))

    async def test_save_and_retrieve_latest(self, store: SummaryStore):
        from context_harness.summarizer import SummaryRecord
        rec = SummaryRecord(
            session_id="sess1",
            summary_text="Harry defeated Voldemort.",
            covered_hash="abc123",
            token_count=4,
        )
        await store.save(rec)
        result = await store.latest("sess1")
        assert result is not None
        assert result.summary_text == "Harry defeated Voldemort."
        assert result.covered_hash == "abc123"

    async def test_latest_returns_none_for_unknown_session(self, store: SummaryStore):
        assert await store.latest("no-such-session") is None

    async def test_latest_returns_most_recent(self, store: SummaryStore):
        from context_harness.summarizer import SummaryRecord
        old = SummaryRecord("s", "old summary", "hash1", 2, created_at=1000.0)
        new = SummaryRecord("s", "new summary", "hash2", 2, created_at=2000.0)
        await store.save(old)
        await store.save(new)
        result = await store.latest("s")
        assert result.summary_text == "new summary"

    async def test_upsert_same_key(self, store: SummaryStore):
        from context_harness.summarizer import SummaryRecord
        r1 = SummaryRecord("s", "first", "hashX", 1)
        r2 = SummaryRecord("s", "updated", "hashX", 1)
        await store.save(r1)
        await store.save(r2)
        result = await store.latest("s")
        assert result.summary_text == "updated"

    async def test_isolates_sessions(self, store: SummaryStore):
        from context_harness.summarizer import SummaryRecord
        await store.save(SummaryRecord("sessA", "summary A", "h1", 2))
        await store.save(SummaryRecord("sessB", "summary B", "h2", 2))
        assert (await store.latest("sessA")).summary_text == "summary A"
        assert (await store.latest("sessB")).summary_text == "summary B"


# ---------------------------------------------------------------------------
# SummarizationQueue
# ---------------------------------------------------------------------------

class TestSummarizationQueue:
    async def test_submit_calls_llm_and_applies_summary(self):
        called_with: list[list[ContextEntry]] = []

        async def tracking_llm(entries: list[ContextEntry]) -> str:
            called_with.append(entries)
            return "async summary result"

        queue = SummarizationQueue(llm_fn=tracking_llm)
        await queue.start()

        w = ContextWindow(max_tokens=4096)
        entries = [_entry("turn one"), _entry("turn two")]
        for e in entries:
            w._entries.append(e)

        await queue.submit("sess", entries, w)
        await queue.stop()

        assert len(called_with) == 1
        assert called_with[0] == entries
        assert any("async summary result" in e.content for e in w._entries)

    async def test_summary_persisted_to_store(self, tmp_path: Path):
        store = SummaryStore(db_path=str(tmp_path / "s.db"))
        queue = SummarizationQueue(llm_fn=_instant_summarise, store=store)
        await queue.start()

        w = ContextWindow(max_tokens=4096)
        entries = [_entry("Dumbledore trusted Snape")]
        w._entries.append(entries[0])

        await queue.submit("session-x", entries, w)
        await queue.stop()

        record = await store.latest("session-x")
        assert record is not None
        # _instant_summarise truncates each entry to 20 chars
        assert "Dumbledore" in record.summary_text

    async def test_llm_error_does_not_crash_worker(self):
        error_count = 0

        async def crashing_llm(entries: list[ContextEntry]) -> str:
            nonlocal error_count
            error_count += 1
            raise RuntimeError("LLM unavailable")

        queue = SummarizationQueue(llm_fn=crashing_llm)
        await queue.start()

        w = ContextWindow(max_tokens=4096)
        entries = [_entry("some content")]
        w._entries.append(entries[0])

        await queue.submit("sess", entries, w)
        await queue.stop()

        assert error_count == 1
        # Window entry should still be present (summary was not applied)
        assert any("some content" in e.content for e in w._entries)

    async def test_multiple_jobs_processed_in_order(self):
        results: list[str] = []

        async def recording_llm(entries: list[ContextEntry]) -> str:
            text = entries[0].content
            results.append(text)
            return f"summary of {text}"

        queue = SummarizationQueue(llm_fn=recording_llm)
        await queue.start()

        for i in range(3):
            w = ContextWindow(max_tokens=4096)
            e = _entry(f"turn {i}")
            w._entries.append(e)
            await queue.submit("sess", [e], w)

        await queue.stop()
        assert results == ["turn 0", "turn 1", "turn 2"]

    async def test_stop_drains_queue_before_cancelling(self):
        processed = 0

        async def slow_llm(entries: list[ContextEntry]) -> str:
            nonlocal processed
            await asyncio.sleep(0.01)
            processed += 1
            return "done"

        queue = SummarizationQueue(llm_fn=slow_llm)
        await queue.start()

        for _ in range(5):
            w = ContextWindow(max_tokens=4096)
            e = _entry("content")
            w._entries.append(e)
            await queue.submit("sess", [e], w)

        await queue.stop()
        assert processed == 5


# ---------------------------------------------------------------------------
# SummarizingContextWindow
# ---------------------------------------------------------------------------

class TestSummarizingContextWindow:
    async def test_overflow_triggers_async_compression(self):
        compressed: list[list[ContextEntry]] = []

        async def tracking_llm(entries: list[ContextEntry]) -> str:
            compressed.append(entries)
            return "compressed history"

        queue = SummarizationQueue(llm_fn=tracking_llm)
        await queue.start()

        # Small window: max=50 tokens, reserved=5, budget=45, recent_slot=27 (60%)
        win = SummarizingContextWindow(
            session_id="s1",
            queue=queue,
            max_tokens=50,
            reserved_output=5,
            recent_ratio=0.60,
        )

        # Add entries that collectively overflow the recent_slot
        for i in range(6):
            e = ContextEntry(role=ContextRole.USER, content=f"word{i} " * 5)
            win._entries.append(e)

        # Trigger overflow handling; yield so the create_task inside runs
        win._try_evict()
        await asyncio.sleep(0)
        await queue.stop()

        # Background compression fired at least once
        assert len(compressed) >= 1

    async def test_current_turn_not_blocked(self):
        slow_called = asyncio.Event()

        async def slow_llm(entries: list[ContextEntry]) -> str:
            slow_called.set()
            await asyncio.sleep(0.1)
            return "slow summary"

        queue = SummarizationQueue(llm_fn=slow_llm)
        await queue.start()

        win = SummarizingContextWindow(
            session_id="s2",
            queue=queue,
            max_tokens=30,
            reserved_output=2,
            recent_ratio=0.60,
        )
        for i in range(5):
            e = ContextEntry(role=ContextRole.USER, content=f"tok{i} " * 4)
            win._entries.append(e)

        start = time.monotonic()
        win._try_evict()
        elapsed = time.monotonic() - start

        # _try_evict must return almost instantly — not wait for the LLM
        assert elapsed < 0.05

        await queue.stop()

    async def test_summary_entry_appears_after_drain(self):
        queue = SummarizationQueue(llm_fn=_instant_summarise)
        await queue.start()

        win = SummarizingContextWindow(
            session_id="s3",
            queue=queue,
            max_tokens=40,
            reserved_output=2,
            recent_ratio=0.50,
        )
        entries = []
        for i in range(6):
            e = ContextEntry(role=ContextRole.USER, content=f"word{i} " * 4)
            win._entries.append(e)
            entries.append(e)

        win._try_evict()
        await asyncio.sleep(0)
        await queue.stop()

        contents = [e.content for e in win._entries]
        assert any("SUMMARY:" in c for c in contents)


# ---------------------------------------------------------------------------
# _hash_entries
# ---------------------------------------------------------------------------

class TestHashEntries:
    def test_same_content_same_hash(self):
        e1 = _entry("Snape killed Dumbledore")
        e2 = _entry("Snape killed Dumbledore")
        assert _hash_entries([e1]) == _hash_entries([e2])

    def test_different_content_different_hash(self):
        e1 = _entry("Snape killed Dumbledore")
        e2 = _entry("Harry killed Voldemort")
        assert _hash_entries([e1]) != _hash_entries([e2])

    def test_order_matters(self):
        e1 = _entry("first")
        e2 = _entry("second")
        assert _hash_entries([e1, e2]) != _hash_entries([e2, e1])
