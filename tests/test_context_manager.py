"""Tests for context_harness/context_manager.py"""
import time
import pytest
from context_harness.context_manager import (
    ContextEntry, ContextRole, ContextWindow, whitespace_token_counter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_entry(role=ContextRole.USER, content="hello world", priority=1.0) -> ContextEntry:
    return ContextEntry(role=role, content=content, priority=priority)


def small_window(**kwargs) -> ContextWindow:
    """Window with a tiny budget for eviction tests.
    budget = max_tokens - reserved = 12 - 2 = 10 tokens.
    """
    return ContextWindow(max_tokens=12, reserved_output_tokens=2, **kwargs)


# ---------------------------------------------------------------------------
# Token counter
# ---------------------------------------------------------------------------

def test_whitespace_counter_empty():
    assert whitespace_token_counter("") == 0


def test_whitespace_counter_words():
    assert whitespace_token_counter("one two three") == 3


# ---------------------------------------------------------------------------
# ContextEntry
# ---------------------------------------------------------------------------

def test_entry_auto_counts_tokens():
    entry = ContextEntry(role=ContextRole.USER, content="one two three")
    assert entry.tokens == 3


def test_entry_uses_provided_token_count():
    entry = ContextEntry(role=ContextRole.USER, content="one two three", tokens=99)
    assert entry.tokens == 99


# ---------------------------------------------------------------------------
# ContextWindow — basic operations
# ---------------------------------------------------------------------------

def test_empty_window_zero_usage():
    w = ContextWindow()
    assert w.used_tokens == 0
    assert w.remaining_tokens == w._budget


def test_add_entry_increases_usage():
    w = ContextWindow()
    w.add_text(ContextRole.USER, "who is dumbledore")
    assert w.used_tokens > 0


def test_entries_returns_copy():
    w = ContextWindow()
    w.add_text(ContextRole.USER, "test")
    entries = w.entries
    entries.clear()
    assert len(w.entries) == 1  # original unchanged


def test_reset_clears_all():
    w = ContextWindow()
    w.add_text(ContextRole.USER, "test")
    w.reset()
    assert w.used_tokens == 0
    assert len(w.entries) == 0


def test_clear_retrieval_only_removes_retrieval():
    w = ContextWindow()
    w.add_text(ContextRole.SYSTEM, "system prompt", priority=10.0)
    w.add_text(ContextRole.RETRIEVAL, "some lore chunk", priority=2.0)
    w.add_text(ContextRole.USER, "question", priority=1.0)
    w.clear_retrieval()
    roles = [e.role for e in w.entries]
    assert ContextRole.RETRIEVAL not in roles
    assert ContextRole.SYSTEM in roles
    assert ContextRole.USER in roles


# ---------------------------------------------------------------------------
# ContextWindow — eviction
# ---------------------------------------------------------------------------

def test_evicts_lowest_priority_non_system():
    # budget = 10 tokens
    # "sys" = 1 token  (SYSTEM, pinned)
    # "old low pri"    = 3 tokens (USER, priority=0.1) — fits alongside sys (4 total)
    # "new high pri a b c d e f" = 8 tokens (RETRIEVAL, priority=2.0)
    #   → total would be 12, exceeds budget=10, so the low-priority USER is evicted
    w = small_window()
    w.add_text(ContextRole.SYSTEM, "sys", priority=10.0)
    w.add_text(ContextRole.USER, "old low pri", priority=0.1)
    w.add_text(ContextRole.RETRIEVAL, "new high pri a b c d e f", priority=2.0)

    roles = [e.role for e in w.entries]
    assert ContextRole.SYSTEM in roles, "SYSTEM must not be evicted"
    low_priority_users = [e for e in w.entries if e.role == ContextRole.USER and e.priority == 0.1]
    assert len(low_priority_users) == 0


def test_raises_when_single_entry_exceeds_budget():
    w = small_window()
    giant = "word " * 1000
    with pytest.raises((ValueError, RuntimeError)):
        w.add_text(ContextRole.USER, giant)


# ---------------------------------------------------------------------------
# ContextWindow — rendering
# ---------------------------------------------------------------------------

def test_to_messages_format():
    w = ContextWindow()
    w.add_text(ContextRole.SYSTEM, "you are a tutor")
    w.add_text(ContextRole.USER, "what is a horcrux")
    msgs = w.to_messages()
    assert msgs[0] == {"role": "system", "content": "you are a tutor"}
    assert msgs[1] == {"role": "user", "content": "what is a horcrux"}


def test_snapshot_contains_keys():
    w = ContextWindow()
    w.add_text(ContextRole.USER, "test")
    snap = w.snapshot()
    assert "budget" in snap
    assert "used" in snap
    assert "remaining" in snap
    assert "entries" in snap
