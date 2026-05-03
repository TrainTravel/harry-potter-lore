"""
Temporal workflows for the HP Lore Agent.

Workflows are pure orchestration — deterministic, no direct I/O.
Every side-effectful step is an activity call so Temporal can:
  - Replay completed steps after a crash (no repeated LLM calls)
  - Retry failed steps with backoff (LLM rate limits)
  - Persist workflow state for days/weeks (long tutoring sessions)

MultiToolResearchWorkflow
    route → plan tool calls → execute in parallel → deduplicate → synthesise → save

GuidedLearningWorkflow
    load history → retrieve → guided_learning LLM → save → compact
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from temporal.activities import (
        compact_conversation,
        execute_tool_call,
        guided_learning_forward,
        load_conversation_history,
        plan_tool_calls,
        retrieve_chunks,
        route_intent,
        save_conversation_turn,
        synthesise_research,
    )


# ---------------------------------------------------------------------------
# Data contracts (must be JSON-serialisable — no DSPy/pydantic objects)
# ---------------------------------------------------------------------------

@dataclass
class ResearchInput:
    question: str
    collection_name: str = "hp_lore"
    conversation_id: Optional[str] = None
    model: str = "gemini/gemini-2.5-flash-lite"


@dataclass
class ResearchOutput:
    answer: str
    citations: list[str]
    confidence: str
    gaps: str
    turn_index: Optional[int] = None


@dataclass
class LearningInput:
    question: str
    conversation_id: str
    collection_name: str = "hp_lore"
    model: str = "gemini/gemini-2.5-flash-lite"


@dataclass
class LearningOutput:
    hint: str
    explanation: str
    next_question: str
    turn_index: Optional[int] = None


# ---------------------------------------------------------------------------
# Retry policies
# ---------------------------------------------------------------------------

_LLM_RETRY = RetryPolicy(
    maximum_attempts=5,
    initial_interval=timedelta(seconds=5),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=60),
    # Bad input is not worth retrying — let the workflow handle it.
    non_retryable_error_types=["ValueError"],
)

_RETRIEVAL_RETRY = RetryPolicy(
    maximum_attempts=3,
    initial_interval=timedelta(seconds=2),
    backoff_coefficient=2.0,
)

_DB_RETRY = RetryPolicy(
    maximum_attempts=5,
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
)


# ---------------------------------------------------------------------------
# Helpers (deterministic — safe to call directly in workflow code)
# ---------------------------------------------------------------------------

def _dedup_and_assemble(tool_results: list[list[dict]], max_chunks: int = 15) -> str:
    """Deduplicate chunk dicts and join into a context string for the LLM."""
    seen: set = set()
    merged: list[dict] = []
    for chunks in tool_results:
        for c in chunks:
            key = (c["doc_id"], c["text"][:60])
            if key not in seen:
                seen.add(key)
                merged.append(c)
    merged = merged[:max_chunks]
    return "\n\n".join(f"[{c['doc_id']}] {c['text']}" for c in merged) or "No context retrieved."


# ---------------------------------------------------------------------------
# MultiToolResearchWorkflow
# ---------------------------------------------------------------------------

@workflow.defn
class MultiToolResearchWorkflow:
    """Durable multi-step deep research.

    Crash after any activity → replay restores completed results from the
    event history → execution continues without repeating LLM calls.
    Rate-limit 429s are handled automatically by _LLM_RETRY backoff.
    """

    @workflow.run
    async def run(self, input: ResearchInput) -> ResearchOutput:
        # Step 1 — route to confirm the question is HP-lore.
        route = await workflow.execute_activity(
            route_intent,
            input.question,
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=_LLM_RETRY,
        )
        if route.get("mode") == "none":
            return ResearchOutput(
                answer="That question is outside HP lore — try something from the books.",
                citations=[],
                confidence="low",
                gaps="",
            )

        # Step 2 — plan 2-3 targeted tool calls.
        tool_calls: list[dict] = await workflow.execute_activity(
            plan_tool_calls,
            input.question,
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=_LLM_RETRY,
        )
        if not tool_calls:
            tool_calls = [{"tool": "vector_search", "args": {"query": input.question, "k": 10}}]

        # Step 3 — execute tool calls in parallel (up to 3).
        tool_results: list[list[dict]] = await asyncio.gather(*[
            workflow.execute_activity(
                execute_tool_call,
                args=[call["tool"], call.get("args", {}), input.collection_name],
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=_RETRIEVAL_RETRY,
            )
            for call in tool_calls[:3]
        ])

        # Dedup + assemble context — deterministic, safe in workflow code.
        context = _dedup_and_assemble(list(tool_results))

        # Step 4 — synthesise the final answer.
        synthesis: dict = await workflow.execute_activity(
            synthesise_research,
            args=[input.question, context],
            start_to_close_timeout=timedelta(seconds=120),
            retry_policy=_LLM_RETRY,
        )

        # Step 5 — persist the turn (skipped if no conversation_id).
        turn_index: Optional[int] = None
        if input.conversation_id:
            turn_index = await workflow.execute_activity(
                save_conversation_turn,
                args=[
                    input.conversation_id,
                    input.question,
                    synthesis,
                    "deep_research",
                    0, 0, 0.0,
                ],
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=_DB_RETRY,
            )

        return ResearchOutput(
            answer=synthesis["answer"],
            citations=synthesis["citations"].split() if synthesis.get("citations") else [],
            confidence=synthesis.get("confidence", ""),
            gaps=synthesis.get("gaps", ""),
            turn_index=turn_index,
        )


# ---------------------------------------------------------------------------
# GuidedLearningWorkflow
# ---------------------------------------------------------------------------

@workflow.defn
class GuidedLearningWorkflow:
    """Long-running Socratic tutoring session.

    Temporal holds the session state durably — a tutoring session can span
    days or weeks with no special recovery code. Crash mid-turn → replay
    restores history and retrieval results → only the LLM call is retried.
    """

    @workflow.run
    async def run(self, input: LearningInput) -> LearningOutput:
        # Step 1 — load prior conversation context.
        chat_history: str = await workflow.execute_activity(
            load_conversation_history,
            args=[input.conversation_id, 10],
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=_DB_RETRY,
        )

        # Step 2 — narrow retrieval (k=3, focused on this turn's question).
        chunks: list[dict] = await workflow.execute_activity(
            retrieve_chunks,
            args=[input.question, input.collection_name, 3],
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=_RETRIEVAL_RETRY,
        )
        context = "\n\n".join(f"[{c['doc_id']}] {c['text']}" for c in chunks)

        # Step 3 — guided learning LLM call (Socratic hint + explanation).
        pred: dict = await workflow.execute_activity(
            guided_learning_forward,
            args=[input.question, context, chat_history],
            start_to_close_timeout=timedelta(seconds=90),
            retry_policy=_LLM_RETRY,
        )

        # Step 4 — persist the turn.
        turn_index: int = await workflow.execute_activity(
            save_conversation_turn,
            args=[
                input.conversation_id,
                input.question,
                pred,
                "guided_learning",
                0, 0, 0.0,
            ],
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=_DB_RETRY,
        )

        # Step 5 — compact old turns (runs last; not on the critical path).
        await workflow.execute_activity(
            compact_conversation,
            input.conversation_id,
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=_DB_RETRY,
        )

        return LearningOutput(
            hint=pred.get("hint", ""),
            explanation=pred.get("explanation", ""),
            next_question=pred.get("next_question", ""),
            turn_index=turn_index,
        )
