"""
UnifiedContext — Skills 1 and 7
================================
Closes the gap identified at the end of the previous PR:
  "UnifiedContext not implemented — we built sub-systems but not the assembly."

Addresses IBM agent engineering skills:
  Skill 1 — System Design: single orchestration envelope for one turn
  Skill 7 — Product Thinking: graceful error handling, human escalation policy

Architecture mirrors deepTutor's request lifecycle:
  1. TurnRuntimeManager opens a turn row (status=running)
  2. ContextBuilder reads session history + retrieval + memory, assembles slots
  3. UnifiedContext flows through the capability pipeline as one envelope
  4. EscalationPolicy decides whether to answer, retry, or hand off to a human
  5. TurnRuntimeManager closes the turn (status=completed|failed|escalated)

UnifiedContext is EPHEMERAL — never persisted directly.
Durability = persist inputs before assembly + outputs after completion.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Turn status
# ---------------------------------------------------------------------------

class TurnStatus(str, Enum):
    RUNNING    = "running"
    COMPLETED  = "completed"
    FAILED     = "failed"
    ESCALATED  = "escalated"   # handed off to a human


# ---------------------------------------------------------------------------
# UnifiedContext — the per-turn envelope  (Skill 1)
# ---------------------------------------------------------------------------

@dataclass
class UnifiedContext:
    """
    Immutable snapshot of everything the agent needs for one turn.
    Assembled fresh at turn start, never stored.

    Slots:
      session_id        — identifies the conversation
      turn_id           — unique ID for this turn (used for tracing/cost)
      user_message      — raw input from the user
      history_context   — verbatim recent turns (within recent_slot budget)
      summary_context   — LLM-compressed older turns
      retrieval_context — RAG chunks assembled by ContextAssembler
      memory_context    — cross-session user profile (preferences, history)
      capability        — which agent pipeline to route to
      metadata          — arbitrary key/value bag for extensions
    """

    session_id: str
    turn_id: str
    user_message: str
    history_context: List[Dict[str, str]] = field(default_factory=list)
    summary_context: str = ""
    retrieval_context: str = ""
    memory_context: str = ""
    capability: str = "chat"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_messages(self) -> List[Dict[str, str]]:
        """
        Flatten all context slots into an ordered message list for the LLM.
        Ordering follows deepTutor's convention:
          system → summary → history → retrieval → user
        """
        messages: List[Dict[str, str]] = []

        if self.summary_context:
            messages.append({"role": "system",
                              "content": f"[Conversation summary]\n{self.summary_context}"})

        messages.extend(self.history_context)

        if self.memory_context:
            messages.append({"role": "system",
                              "content": f"[User profile]\n{self.memory_context}"})

        if self.retrieval_context:
            messages.append({"role": "system",
                              "content": f"[Retrieved lore]\n{self.retrieval_context}"})

        messages.append({"role": "user", "content": self.user_message})
        return messages

    def token_estimate(self) -> int:
        """Rough token count across all slots."""
        all_text = " ".join([
            self.user_message, self.summary_context,
            self.retrieval_context, self.memory_context,
            " ".join(m.get("content", "") for m in self.history_context),
        ])
        return len(all_text.split())


# ---------------------------------------------------------------------------
# Session store (SQLite) — durability layer
# ---------------------------------------------------------------------------

class SessionStore:
    """
    Minimal SQLite store for sessions, messages, and turn status.
    This is the *only* persistent state for conversations.
    """

    def __init__(self, db_path: str = "data/sessions.db") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = asyncio.Lock()
        self._init()

    def _init(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id   TEXT PRIMARY KEY,
                created_at   REAL NOT NULL,
                summary_text TEXT DEFAULT '',
                summary_up_to_msg_id TEXT DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS messages (
                msg_id      TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                created_at  REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS turns (
                turn_id     TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL,
                status      TEXT NOT NULL,
                capability  TEXT NOT NULL,
                created_at  REAL NOT NULL,
                completed_at REAL
            );
        """)
        self._conn.commit()

    async def get_or_create_session(self, session_id: str) -> None:
        async with self._lock:
            exists = self._conn.execute(
                "SELECT 1 FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()
            if not exists:
                self._conn.execute(
                    "INSERT INTO sessions VALUES (?,?,?,?)",
                    (session_id, time.time(), "", ""),
                )
                self._conn.commit()

    async def save_message(self, session_id: str, role: str, content: str) -> str:
        msg_id = str(uuid.uuid4())
        async with self._lock:
            self._conn.execute(
                "INSERT INTO messages VALUES (?,?,?,?,?)",
                (msg_id, session_id, role, content, time.time()),
            )
            self._conn.commit()
        return msg_id

    async def recent_messages(self, session_id: str, limit: int = 20) -> List[Dict[str, str]]:
        rows = self._conn.execute(
            "SELECT role, content FROM messages WHERE session_id=? "
            "ORDER BY created_at DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [{"role": r, "content": c} for r, c in reversed(rows)]

    async def open_turn(self, session_id: str, capability: str) -> str:
        turn_id = str(uuid.uuid4())
        async with self._lock:
            self._conn.execute(
                "INSERT INTO turns VALUES (?,?,?,?,?,NULL)",
                (turn_id, session_id, TurnStatus.RUNNING.value, capability, time.time()),
            )
            self._conn.commit()
        return turn_id

    async def close_turn(self, turn_id: str, status: TurnStatus) -> None:
        async with self._lock:
            self._conn.execute(
                "UPDATE turns SET status=?, completed_at=? WHERE turn_id=?",
                (status.value, time.time(), turn_id),
            )
            self._conn.commit()


# ---------------------------------------------------------------------------
# ContextBuilder — assembles all slots into a UnifiedContext
# ---------------------------------------------------------------------------

class ContextBuilder:
    """
    Assembles a UnifiedContext from its sources.

    This is the object deepTutor was missing a clean abstraction for.
    Each source is optional — pass None to skip that slot.
    """

    def __init__(
        self,
        store: SessionStore,
        rag_pipeline=None,       # RAGPipeline | CachedRAGPipeline
        assembler=None,          # ContextAssembler
        history_limit: int = 20,
    ) -> None:
        self._store = store
        self._rag = rag_pipeline
        self._assembler = assembler
        self._history_limit = history_limit

    async def build(
        self,
        session_id: str,
        turn_id: str,
        user_message: str,
        capability: str = "chat",
        top_k: int = 5,
        memory_context: str = "",
        **metadata,
    ) -> UnifiedContext:
        """Assemble all context slots. Fast path: all I/O runs concurrently."""

        history_task = asyncio.create_task(
            self._store.recent_messages(session_id, self._history_limit)
        )

        retrieval_task = asyncio.create_task(
            self._retrieve(user_message, top_k)
        )

        history, retrieval_context = await asyncio.gather(history_task, retrieval_task)

        return UnifiedContext(
            session_id=session_id,
            turn_id=turn_id,
            user_message=user_message,
            history_context=history,
            retrieval_context=retrieval_context,
            memory_context=memory_context,
            capability=capability,
            metadata=metadata,
        )

    async def _retrieve(self, query: str, top_k: int) -> str:
        if self._rag is None or self._assembler is None:
            return ""
        # Support both sync and async retrieve
        if asyncio.iscoroutinefunction(getattr(self._rag, "retrieve", None)):
            chunks = await self._rag.retrieve(query, top_k=top_k)
        else:
            chunks = self._rag.retrieve(query, top_k=top_k)
        return self._assembler.assemble(chunks, query=query)


# ---------------------------------------------------------------------------
# TurnRuntimeManager — lifecycle wrapper  (Skill 7)
# ---------------------------------------------------------------------------

class TurnRuntimeManager:
    """
    Manages the full lifecycle of one agent turn:
      open → build context → run capability → close

    Usage:
        async with TurnRuntimeManager(store, builder, policy) as turn:
            ctx = await turn.context("session-1", user_msg)
            answer = await my_capability(ctx)
            turn.set_result(answer)

    On unhandled exception: turn is closed with status=FAILED.
    On escalation trigger:  turn is closed with status=ESCALATED.
    """

    def __init__(
        self,
        store: SessionStore,
        builder: ContextBuilder,
        escalation_policy: "EscalationPolicy",
    ) -> None:
        self._store = store
        self._builder = builder
        self._policy = escalation_policy
        self._turn_id: Optional[str] = None
        self._session_id: Optional[str] = None
        self._result: Optional[str] = None

    async def __aenter__(self) -> "TurnRuntimeManager":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            await self._store.close_turn(self._turn_id, TurnStatus.FAILED)
            return False  # re-raise
        status = TurnStatus.COMPLETED if self._result else TurnStatus.FAILED
        await self._store.close_turn(self._turn_id, status)
        return False

    async def context(
        self,
        session_id: str,
        user_message: str,
        capability: str = "chat",
        **kwargs,
    ) -> UnifiedContext:
        self._session_id = session_id
        await self._store.get_or_create_session(session_id)
        await self._store.save_message(session_id, "user", user_message)
        self._turn_id = await self._store.open_turn(session_id, capability)
        return await self._builder.build(
            session_id, self._turn_id, user_message, capability, **kwargs
        )

    def set_result(self, answer: str) -> None:
        self._result = answer

    def should_escalate(self, ctx: UnifiedContext, error: Optional[Exception] = None) -> bool:
        return self._policy.should_escalate(ctx, error)


# ---------------------------------------------------------------------------
# EscalationPolicy — Skill 7: when to hand off to a human
# ---------------------------------------------------------------------------

@dataclass
class EscalationPolicy:
    """
    Decides when the agent should stop trying and hand off to a human.

    Rules are evaluated in order; first match wins.

    Context engineering insight: knowing when NOT to answer is as important
    as knowing how to answer. An agent that confidently gives wrong answers
    is worse than one that escalates appropriately.
    """

    # Escalate if the retrieval context is empty (no relevant lore found)
    escalate_on_empty_retrieval: bool = True

    # Escalate after this many consecutive failed turns in a session
    max_consecutive_failures: int = 3

    # Escalate if the query contains any of these trigger phrases
    escalation_triggers: List[str] = field(default_factory=lambda: [
        "speak to a human",
        "talk to someone",
        "real person",
        "not helpful",
        "wrong answer",
    ])

    # Escalate on these specific exception types (class names as strings)
    error_escalation_types: List[str] = field(default_factory=lambda: [
        "AuthenticationError",
        "RateLimitError",
        "PermissionError",
    ])

    def should_escalate(
        self,
        ctx: UnifiedContext,
        error: Optional[Exception] = None,
    ) -> bool:
        # Rule 1: unrecoverable error type
        if error is not None:
            if type(error).__name__ in self.error_escalation_types:
                return True

        # Rule 2: user explicitly asked for a human
        lower = ctx.user_message.lower()
        if any(trigger in lower for trigger in self.escalation_triggers):
            return True

        # Rule 3: no retrieval context available for a lore question
        if self.escalate_on_empty_retrieval and not ctx.retrieval_context.strip():
            lore_signals = ["who", "what", "where", "when", "why", "how", "tell me"]
            if any(s in lower for s in lore_signals):
                return True

        return False

    def escalation_message(self, ctx: UnifiedContext) -> str:
        return (
            f"I'm not confident I can answer '{ctx.user_message}' accurately. "
            "Let me connect you with a human expert who can help further."
        )
