"""
Context Window Manager
======================
Manages the context window for LLM interactions.

Core ideas:
  - Track token budget across turns
  - Evict old entries via LRU or priority-weighted scoring
  - Compress context via summarisation when budget is tight
  - Provide a clean view of the current context for prompt injection
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

class ContextRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    RETRIEVAL = "retrieval"   # injected RAG chunks
    TOOL = "tool"             # tool call / result pairs


@dataclass
class ContextEntry:
    """A single entry in the context window."""

    role: ContextRole
    content: str
    tokens: int = 0                   # estimated token count
    priority: float = 1.0             # higher = more important, kept longer
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tokens == 0:
            # rough whitespace tokeniser — replace with tiktoken in production
            self.tokens = len(self.content.split())


# ---------------------------------------------------------------------------
# Token counter (pluggable)
# ---------------------------------------------------------------------------

TokenCounter = Callable[[str], int]


def whitespace_token_counter(text: str) -> int:
    """Rough token estimate: ~1 token per word."""
    return len(text.split())


# ---------------------------------------------------------------------------
# Context Window
# ---------------------------------------------------------------------------

class ContextWindow:
    """
    A token-budget-aware context window.

    Context Engineering principles applied here:
      1. **Budget awareness** – always know how many tokens remain.
      2. **Priority eviction** – when full, evict lowest-priority entries first.
      3. **Role pinning** – SYSTEM entries are never evicted automatically.
      4. **Compression hook** – caller can supply a compressor that summarises
         a batch of RETRIEVAL entries into a shorter one.
    """

    def __init__(
        self,
        max_tokens: int = 4096,
        reserved_output_tokens: int = 512,
        token_counter: TokenCounter = whitespace_token_counter,
        compressor: Optional[Callable[[List[ContextEntry]], str]] = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.reserved_output_tokens = reserved_output_tokens
        self._budget = max_tokens - reserved_output_tokens
        self._counter = token_counter
        self._compressor = compressor
        self._entries: List[ContextEntry] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def used_tokens(self) -> int:
        return sum(e.tokens for e in self._entries)

    @property
    def remaining_tokens(self) -> int:
        return self._budget - self.used_tokens

    @property
    def entries(self) -> List[ContextEntry]:
        return list(self._entries)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, entry: ContextEntry) -> None:
        """Add an entry, evicting lower-priority content if necessary."""
        # recount tokens with the configured counter
        entry.tokens = self._counter(entry.content)

        if entry.tokens > self._budget:
            raise ValueError(
                f"Entry alone ({entry.tokens} tokens) exceeds budget ({self._budget})."
            )

        while self.remaining_tokens < entry.tokens:
            if not self._try_evict():
                # last resort: compress retrieval entries
                if not self._try_compress():
                    raise RuntimeError("Context window is full and cannot be compacted.")

        self._entries.append(entry)

    def add_text(
        self,
        role: ContextRole,
        content: str,
        priority: float = 1.0,
        **metadata,
    ) -> ContextEntry:
        entry = ContextEntry(role=role, content=content, priority=priority, metadata=metadata)
        self.add(entry)
        return entry

    def clear_retrieval(self) -> None:
        """Remove all RETRIEVAL entries (useful between turns)."""
        self._entries = [e for e in self._entries if e.role != ContextRole.RETRIEVAL]

    def reset(self) -> None:
        """Wipe all entries."""
        self._entries.clear()

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def _evictable(self) -> List[ContextEntry]:
        """Non-SYSTEM entries sorted by ascending priority then ascending timestamp."""
        candidates = [e for e in self._entries if e.role != ContextRole.SYSTEM]
        return sorted(candidates, key=lambda e: (e.priority, e.timestamp))

    def _try_evict(self) -> bool:
        candidates = self._evictable()
        if not candidates:
            return False
        victim = candidates[0]
        self._entries.remove(victim)
        return True

    def _try_compress(self) -> bool:
        """Replace all RETRIEVAL entries with a compressed summary."""
        if self._compressor is None:
            return False
        retrieval = [e for e in self._entries if e.role == ContextRole.RETRIEVAL]
        if not retrieval:
            return False
        summary = self._compressor(retrieval)
        for e in retrieval:
            self._entries.remove(e)
        compressed = ContextEntry(
            role=ContextRole.RETRIEVAL,
            content=summary,
            priority=2.0,  # compressed context is more valuable
            metadata={"compressed": True, "source_count": len(retrieval)},
        )
        compressed.tokens = self._counter(summary)
        self._entries.append(compressed)
        return True

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def to_messages(self) -> List[dict]:
        """Render as a list of {role, content} dicts (OpenAI / ADK format)."""
        return [
            {"role": e.role.value, "content": e.content}
            for e in self._entries
        ]

    def snapshot(self) -> dict:
        """Debug snapshot of current state."""
        return {
            "budget": self._budget,
            "used": self.used_tokens,
            "remaining": self.remaining_tokens,
            "entry_count": len(self._entries),
            "entries": [
                {
                    "role": e.role.value,
                    "tokens": e.tokens,
                    "priority": e.priority,
                    "preview": e.content[:80],
                }
                for e in self._entries
            ],
        }

    def __repr__(self) -> str:
        return (
            f"ContextWindow(budget={self._budget}, "
            f"used={self.used_tokens}, entries={len(self._entries)})"
        )
