"""
Conversation store — per-session turn memory for multi-turn modes.
===================================================================
Phase 0: append-only log of (user_message, agent_response) turns keyed by
a caller-supplied ``conversation_id``. Compaction via LLM summarisation
is scaffolded but intentionally stubbed — it logs a warning when the
threshold is crossed and returns the raw turns. Phase 1 wires in the
summariser.

Design decisions:
  - Co-locates with the existing tracer at ``data/traces.duckdb``.
    Same file means conversations can be JOIN'd with trace events by
    turn_id later (for "which retrieval happened on turn 3?" queries).
  - Two tables, not one: ``conversations`` is append-only (hot path);
    ``conversation_summaries`` is lazily written by compaction. Turns
    are never mutated.
  - Sync I/O. DuckDB has no async driver; FastAPI's sync-route threadpool
    handles the few-ms latency of a single INSERT cleanly.
  - Format-for-LLM is mode-aware: for ``guided_learning`` the prior
    agent responses are reduced to just the ``explanation`` field;
    ``hint`` and ``next_question`` are ephemera that don't belong in
    the next turn's context.

Usage::

    store = ConversationStore()   # defaults to data/traces.duckdb
    # Appending
    turn_idx = store.save_turn(
        conversation_id="abc-123",
        user_message="What is a Horcrux?",
        agent_response={"hint": "...", "explanation": "...", ...},
        mode="guided_learning",
    )
    # Loading for next turn
    history = store.load_history("abc-123", max_turns=5)
    history_text = store.format_for_llm(history, mode="guided_learning")
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import duckdb


log = logging.getLogger(__name__)


DEFAULT_DB_PATH = "data/traces.duckdb"

# Phase 0: log a warning when we blow past this; don't actually compact yet.
DEFAULT_COMPACTION_THRESHOLD = 8


# ---------------------------------------------------------------------------
# Typed value objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Turn:
    """A single exchange. ``agent_response`` is the raw Signature output as
    a dict — downstream formatters decide which fields to surface."""
    conversation_id: str
    turn_index:      int
    user_message:    str
    agent_response:  dict[str, Any]
    mode:            str
    character:       Optional[str] = None
    tokens_in:       int = 0
    tokens_out:      int = 0
    cost_usd:        float = 0.0
    ts:              float = field(default_factory=time.time)


@dataclass(frozen=True)
class ConversationHistory:
    """Everything the LLM needs to know about a conversation so far."""
    summary: str                 # empty string if no compaction yet
    turns:   list[Turn]          # always chronological (oldest first)

    def is_empty(self) -> bool:
        return not self.summary and not self.turns

    def turn_count(self) -> int:
        return len(self.turns) + (1 if self.summary else 0)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class ConversationStore:
    """DuckDB-backed turn log. Sync; methods are fast enough (~ms) that the
    FastAPI sync-route threadpool can handle them without further work."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(db_path)
        self._init_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id VARCHAR NOT NULL,
                turn_index      INTEGER NOT NULL,
                user_message    TEXT    NOT NULL,
                agent_response  JSON    NOT NULL,
                mode            VARCHAR NOT NULL,
                character       VARCHAR,
                tokens_in       INTEGER DEFAULT 0,
                tokens_out      INTEGER DEFAULT 0,
                cost_usd        DOUBLE  DEFAULT 0.0,
                ts              DOUBLE  NOT NULL,
                PRIMARY KEY (conversation_id, turn_index)
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_summaries (
                conversation_id VARCHAR NOT NULL,
                up_to_turn      INTEGER NOT NULL,
                summary         TEXT    NOT NULL,
                created_at      DOUBLE  NOT NULL,
                PRIMARY KEY (conversation_id, up_to_turn)
            )
        """)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save_turn(
        self,
        conversation_id: str,
        user_message: str,
        agent_response: dict[str, Any],
        mode: str,
        character: Optional[str] = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost_usd: float = 0.0,
    ) -> int:
        """Append a turn. Returns the newly-assigned ``turn_index`` (monotonic,
        starts at 1 for the first turn in a conversation)."""
        next_idx = self._next_turn_index(conversation_id)
        self._conn.execute(
            """INSERT INTO conversations
                 (conversation_id, turn_index, user_message, agent_response,
                  mode, character, tokens_in, tokens_out, cost_usd, ts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (conversation_id, next_idx, user_message,
             json.dumps(agent_response, default=str), mode, character,
             tokens_in, tokens_out, cost_usd, time.time()),
        )
        return next_idx

    def _next_turn_index(self, conversation_id: str) -> int:
        row = self._conn.execute(
            "SELECT COALESCE(MAX(turn_index), 0) + 1 FROM conversations "
            "WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        return int(row[0]) if row else 1

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load_history(
        self,
        conversation_id: str,
        max_turns: int = 5,
    ) -> ConversationHistory:
        """Return the rolling summary (if any) plus up to ``max_turns`` most
        recent verbatim turns, chronological order."""
        summary_row = self._conn.execute(
            "SELECT up_to_turn, summary FROM conversation_summaries "
            "WHERE conversation_id = ? ORDER BY up_to_turn DESC LIMIT 1",
            (conversation_id,),
        ).fetchone()
        up_to_turn = int(summary_row[0]) if summary_row else 0
        summary_text = str(summary_row[1]) if summary_row else ""

        rows = self._conn.execute(
            """SELECT turn_index, user_message, agent_response, mode, character,
                      tokens_in, tokens_out, cost_usd, ts
               FROM conversations
               WHERE conversation_id = ? AND turn_index > ?
               ORDER BY turn_index DESC
               LIMIT ?""",
            (conversation_id, up_to_turn, max_turns),
        ).fetchall()
        turns = [
            Turn(
                conversation_id=conversation_id,
                turn_index=int(r[0]),
                user_message=str(r[1]),
                agent_response=json.loads(r[2]) if r[2] else {},
                mode=str(r[3]),
                character=(str(r[4]) if r[4] else None),
                tokens_in=int(r[5] or 0),
                tokens_out=int(r[6] or 0),
                cost_usd=float(r[7] or 0.0),
                ts=float(r[8]),
            )
            for r in rows
        ]
        turns.reverse()   # we fetched DESC; LLM wants chronological
        return ConversationHistory(summary=summary_text, turns=turns)

    # ------------------------------------------------------------------
    # Format for LLM injection
    # ------------------------------------------------------------------

    @staticmethod
    def format_for_llm(history: ConversationHistory, mode: str) -> str:
        """Render a history object into a text block for Signature injection.

        Mode-aware: each mode has ephemeral vs. durable fields in its
        structured output. We surface the durable ones to future turns.
        For ``guided_learning``, only ``explanation`` travels forward
        (hints and next_questions are per-turn artefacts).
        """
        if history.is_empty():
            return ""

        parts: list[str] = []
        if history.summary:
            parts.append(f"Earlier in conversation: {history.summary}")

        for t in history.turns:
            parts.append(f"[{t.turn_index}] User: {t.user_message}")
            agent_text = _format_agent_response(t.agent_response, mode)
            if agent_text:
                parts.append(f"     Tutor: {agent_text}")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Compaction — scaffolded, stubbed for Phase 0
    # ------------------------------------------------------------------

    def compact_if_needed(
        self,
        conversation_id: str,
        threshold: int = DEFAULT_COMPACTION_THRESHOLD,
    ) -> None:
        """Phase 0 stub — logs a warning when a conversation exceeds the
        threshold but does NOT actually summarise. Phase 1 wires in the
        summariser from context_harness.summarizer.

        We expose the method now so call sites can reference it; the
        no-op keeps the plumbing stable when the real implementation lands.
        """
        count = self._conn.execute(
            "SELECT COUNT(*) FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()[0]
        if count > threshold:
            log.warning(
                "Conversation %s has %d turns (threshold %d). Compaction "
                "not yet implemented — history will keep growing.",
                conversation_id, count, threshold,
            )

    # ------------------------------------------------------------------
    # Observability helpers
    # ------------------------------------------------------------------

    def conversation_cost(self, conversation_id: str) -> dict[str, float]:
        """Sum per-turn cost fields. Useful for the /admin endpoint later."""
        row = self._conn.execute(
            """SELECT COUNT(*), COALESCE(SUM(tokens_in),0),
                      COALESCE(SUM(tokens_out),0), COALESCE(SUM(cost_usd),0)
               FROM conversations WHERE conversation_id = ?""",
            (conversation_id,),
        ).fetchone()
        return {
            "turns":      int(row[0]),
            "tokens_in":  int(row[1]),
            "tokens_out": int(row[2]),
            "cost_usd":   float(row[3]),
        }

    def turn_costs(self, conversation_id: str) -> list[dict[str, Any]]:
        """Per-turn cost breakdown. Returns a list of
        ``{turn_index, tokens_in, tokens_out, cost_usd, ts}`` dicts in
        chronological order — useful for inspecting where cost went within
        a single conversation and for spotting runaway context growth."""
        rows = self._conn.execute(
            """SELECT turn_index, tokens_in, tokens_out, cost_usd, ts
               FROM conversations
               WHERE conversation_id = ?
               ORDER BY turn_index ASC""",
            (conversation_id,),
        ).fetchall()
        return [
            {
                "turn_index": int(r[0]),
                "tokens_in":  int(r[1] or 0),
                "tokens_out": int(r[2] or 0),
                "cost_usd":   float(r[3] or 0.0),
                "ts":         float(r[4]),
            }
            for r in rows
        ]


# ---------------------------------------------------------------------------
# Mode-specific agent-response formatters
# ---------------------------------------------------------------------------

def _format_agent_response(response: dict[str, Any], mode: str) -> str:
    """Pick the durable field(s) from a mode's structured output.

    Surfaces only what the next turn will find useful. Hints and
    next_questions are turn-ephemera — a student asking a follow-up
    doesn't need to see the hint they got two turns ago; they need the
    conceptual explanation the tutor gave."""
    if mode == "guided_learning":
        return str(response.get("explanation", "")).strip()
    if mode == "open_analysis":
        return str(response.get("analysis", "")).strip()
    if mode == "perspective_shift":
        principle = str(response.get("character_principle", "")).strip()
        insight = str(response.get("applied_insight", "")).strip()
        return f"{principle} / {insight}" if principle or insight else ""
    # Unknown mode — dump a compact rendering so we don't silently lose context
    return json.dumps(response, default=str)[:400]
