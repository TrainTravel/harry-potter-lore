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
from typing import Any, Optional, Protocol

import duckdb


log = logging.getLogger(__name__)


DEFAULT_DB_PATH = "data/traces.duckdb"

# Compact once a conversation exceeds this many turns. Keep the most recent
# DEFAULT_KEEP_RECENT verbatim; summarise everything older into a single
# rolling summary. Tunable — see evals/compaction_threshold_experiment.py.
DEFAULT_COMPACTION_THRESHOLD = 8
DEFAULT_KEEP_RECENT = 5


# Mode-aware summarization prompts — preserve what the NEXT turn needs to
# know. For guided_learning: the student's evolving understanding. For
# open_analysis: the thesis + established facts. For perspective_shift:
# the character lens + principle already surfaced.
_SUMMARIZER_PROMPTS: dict[str, str] = {
    "guided_learning": (
        "Summarize this tutor-student dialogue in 2-3 sentences. Preserve:\n"
        "  - what concept or problem the student is working on\n"
        "  - their current level of understanding (grasp vs struggle)\n"
        "  - what the tutor has already explained (to avoid repetition)\n"
        "Do NOT include the tutor's exact phrasing or hints — only the "
        "conceptual content. Write in third person.\n\n"
        "Dialogue:\n{dialogue}\n\n"
        "Summary:"
    ),
    "open_analysis": (
        "Summarize this analytical dialogue in 2-3 sentences. Preserve:\n"
        "  - the core thesis or question being explored\n"
        "  - the corpus facts already established\n"
        "  - the angle of the user's follow-up questions\n"
        "Do NOT paraphrase the analysis itself — only capture what ground "
        "has been covered. Write in third person.\n\n"
        "Dialogue:\n{dialogue}\n\n"
        "Summary:"
    ),
    "perspective_shift": (
        "Summarize this character-lens conversation in 2-3 sentences. Preserve:\n"
        "  - which HP character's perspective is being applied\n"
        "  - the user's real-world scenario\n"
        "  - the principle(s) or insight(s) already surfaced\n"
        "Do NOT repeat applied-insight verbatim — capture the theme. "
        "Write in third person.\n\n"
        "Dialogue:\n{dialogue}\n\n"
        "Summary:"
    ),
}
_GENERIC_SUMMARIZER_PROMPT = (
    "Summarize this dialogue in 2-3 sentences, preserving the key points "
    "and the thread of what's been discussed.\n\n"
    "Dialogue:\n{dialogue}\n\n"
    "Summary:"
)


# ---------------------------------------------------------------------------
# Summarizer Protocol — inject a client for testability
# ---------------------------------------------------------------------------

class Summarizer(Protocol):
    """Anything that turns a prompt into a summary string.

    Mirrors the same Protocol-based injection pattern as lore-builder's
    tagger — keeps ConversationStore decoupled from any specific LLM SDK,
    makes testing with a FakeSummarizer trivial.
    """
    def summarize(self, prompt: str) -> str:
        ...


class GeminiSummarizer:
    """Default production summarizer. Uses Gemini 2.5 Flash-lite unless
    overridden via the ``model`` arg. max_output_tokens bounded to 200
    so summaries can't balloon past the expected 80-120 word target."""

    def __init__(self, model: str = "gemini-2.5-flash-lite",
                 api_key: Optional[str] = None,
                 max_output_tokens: int = 200) -> None:
        self._model = model
        self._api_key = api_key
        self._max_output_tokens = max_output_tokens
        self._client = None  # lazy

    def _get_client(self):
        if self._client is None:
            from google import genai
            import os as _os
            key = (self._api_key
                   or _os.environ.get("GEMINI_API_KEY")
                   or _os.environ.get("GOOGLE_API_KEY"))
            self._client = genai.Client(api_key=key)
        return self._client

    def summarize(self, prompt: str) -> str:
        from google.genai import types
        resp = self._get_client().models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=self._max_output_tokens,
            ),
        )
        return (resp.text or "").strip()


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
            # Format each turn under the mode it was SAVED under, not the
            # current query's mode. This makes cross-mode conversations
            # coherent: a perspective_shift turn shows Luna's voice, a
            # subsequent open_analysis turn shows the analytical content,
            # etc. Falls back to `mode` for turns without a saved mode
            # (legacy data or test fixtures).
            turn_mode = t.mode or mode
            agent_text = _format_agent_response(t.agent_response, turn_mode)
            if agent_text:
                parts.append(f"     Tutor: {agent_text}")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Compaction — scaffolded, stubbed for Phase 0
    # ------------------------------------------------------------------

    def compact_if_needed(
        self,
        conversation_id: str,
        summarizer: "Summarizer | None" = None,
        threshold: int = DEFAULT_COMPACTION_THRESHOLD,
        keep_recent: int = DEFAULT_KEEP_RECENT,
    ) -> bool:
        """Run compaction if the conversation has crossed the threshold.

        If ``summarizer`` is None, this remains a no-op (backwards-compatible
        Phase 0 behaviour — useful for call sites that want threshold-aware
        plumbing without the LLM dependency yet).

        Returns True if compaction ran + a summary was written, False otherwise.
        Never raises — failures are logged and the conversation continues with
        its raw history.
        """
        count = self._conn.execute(
            "SELECT COUNT(*) FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()[0]
        if count <= threshold:
            return False

        if summarizer is None:
            log.warning(
                "Conversation %s has %d turns (threshold %d). No summarizer "
                "provided — skipping compaction.",
                conversation_id, count, threshold,
            )
            return False

        try:
            return self._summarize_and_store(conversation_id, summarizer, keep_recent)
        except Exception as exc:
            # Never break the surrounding turn flow on summary failure
            log.error("Compaction failed for %s: %s", conversation_id, exc)
            return False

    def _summarize_and_store(
        self,
        conversation_id: str,
        summarizer: "Summarizer",
        keep_recent: int,
    ) -> bool:
        """Read prior summary (if any) + turns to compact, call summarizer,
        INSERT a new summary row. Caller handles all exceptions."""
        # Find the boundary: turns [1 .. N - keep_recent] get folded into
        # the new summary. Turns (N - keep_recent + 1) .. N stay verbatim.
        row = self._conn.execute(
            "SELECT MAX(turn_index) FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        if not row or row[0] is None:
            return False
        max_turn = int(row[0])
        compact_up_to = max_turn - keep_recent
        if compact_up_to < 1:
            return False

        # Existing summary, if any — we'll regenerate including anything since
        prior_summary_row = self._conn.execute(
            "SELECT up_to_turn, summary FROM conversation_summaries "
            "WHERE conversation_id = ? ORDER BY up_to_turn DESC LIMIT 1",
            (conversation_id,),
        ).fetchone()
        prior_up_to = int(prior_summary_row[0]) if prior_summary_row else 0
        prior_summary = str(prior_summary_row[1]) if prior_summary_row else ""

        # Grab the turns to fold in: the window between the prior summary's
        # up_to_turn and the new compact_up_to
        turns_to_fold = self._conn.execute(
            """SELECT turn_index, user_message, agent_response, mode
               FROM conversations
               WHERE conversation_id = ? AND turn_index > ? AND turn_index <= ?
               ORDER BY turn_index ASC""",
            (conversation_id, prior_up_to, compact_up_to),
        ).fetchall()
        if not turns_to_fold:
            return False

        # Pick a mode for prompt selection — use the most common mode in the
        # window (conversations tend to stay single-mode, but hedge for mixed)
        mode_counts: dict[str, int] = {}
        for r in turns_to_fold:
            mode_counts[str(r[3])] = mode_counts.get(str(r[3]), 0) + 1
        dominant_mode = max(mode_counts, key=mode_counts.get) if mode_counts else ""
        prompt_template = _SUMMARIZER_PROMPTS.get(
            dominant_mode, _GENERIC_SUMMARIZER_PROMPT
        )

        # Format the window + prior summary into a single dialogue string
        dialogue_parts: list[str] = []
        if prior_summary:
            dialogue_parts.append(f"[Earlier summary]: {prior_summary}")
        for r in turns_to_fold:
            idx, user_msg, agent_json, mode_ = int(r[0]), str(r[1]), r[2], str(r[3])
            agent_dict = json.loads(agent_json) if agent_json else {}
            agent_text = _format_agent_response(agent_dict, mode_)
            dialogue_parts.append(f"[{idx}] User: {user_msg}")
            if agent_text:
                dialogue_parts.append(f"     Agent: {agent_text}")
        dialogue = "\n".join(dialogue_parts)

        prompt = prompt_template.format(dialogue=dialogue)
        summary = summarizer.summarize(prompt)

        # Sanity check — discard garbled or empty outputs
        word_count = len(summary.split())
        if not summary or word_count < 20 or word_count > 300:
            log.warning(
                "Compaction for %s produced unusable summary (%d words). Discarding.",
                conversation_id, word_count,
            )
            return False

        self._conn.execute(
            "INSERT OR REPLACE INTO conversation_summaries "
            "(conversation_id, up_to_turn, summary, created_at) "
            "VALUES (?, ?, ?, ?)",
            (conversation_id, compact_up_to, summary, time.time()),
        )
        return True

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

# Ordered durable-content fields, tried when the mode-specific extraction
# produces nothing useful. This catch-all is what makes cross-mode history
# work — if turn-1 was perspective_shift (character_response field) and
# turn-2's mode is open_analysis (looks for analysis field), turn-1 still
# contributes coherent content to the formatted history instead of
# disappearing entirely.
_DURABLE_FIELD_FALLBACK = (
    "character_response",  # perspective_shift synthesis (first-person voice)
    "analysis",            # open_analysis main field
    "answer",              # deep_research main field
    "explanation",         # guided_learning durable field
)


def _format_agent_response(response: dict[str, Any], mode: str) -> str:
    """Pick the durable field(s) from a structured response, mode-aware with
    a cross-mode fallback.

    Design: each mode has its own "durable" field(s) — the content a future
    turn should see. For the current turn's own mode, we extract those
    fields. But prior turns in the conversation might have been in a
    different mode (before 2026-04-21, the frontend silently allowed mode
    switches mid-conversation — a bug being fixed separately but we still
    need to handle the existing data). So: after mode-specific extraction,
    if the result is empty, fall through to a priority list of durable
    fields that exist across modes.

    Without the fallback, a perspective_shift turn's content would vanish
    entirely when the next turn asks for open_analysis formatting —
    `response.get("analysis", "")` returns "" for a pshift response.
    """
    # Mode-specific extraction (preferred path)
    result = ""
    if mode == "guided_learning":
        result = str(response.get("explanation", "")).strip()
    elif mode == "open_analysis":
        result = str(response.get("analysis", "")).strip()
    elif mode == "perspective_shift":
        # Prefer the synthesis field (first-person character voice) — matches
        # the synthesis pattern in PerspectiveShiftSignature.
        char_response = str(response.get("character_response", "")).strip()
        if char_response:
            result = char_response
        else:
            principle = str(response.get("character_principle", "")).strip()
            insight = str(response.get("applied_insight", "")).strip()
            result = f"{principle} / {insight}" if principle or insight else ""

    if result:
        return result

    # Cross-mode fallback — the response was saved under a different mode
    # than the one we're formatting for. Surface any durable field we
    # recognise, in priority order.
    for key in _DURABLE_FIELD_FALLBACK:
        val = str(response.get(key, "")).strip()
        if val:
            return val

    # Last-resort fallback: compact JSON dump so context isn't silently lost
    return json.dumps(response, default=str)[:400]
