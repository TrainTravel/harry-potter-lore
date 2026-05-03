"""
Temporal activities for the HP Lore Agent.

Each @activity.defn wraps one unit of real work — LLM call, retrieval,
DB write. Temporal provides automatic retries, timeouts, and durable state
around these; the activity code itself stays simple.

Shared state (pipeline, agent, conv_store) is initialised once per worker
process via init_activity_deps(). All activities are async; blocking I/O
(dspy, ChromaDB, SQLite) is offloaded via asyncio.to_thread() so the event
loop is never stalled.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

import dspy
from temporalio import activity

from context_harness.conversation import (
    ConversationStore,
    GeminiSummarizer,
    DEFAULT_COMPACTION_THRESHOLD,
    DEFAULT_KEEP_RECENT,
)
from context_harness.dspy_agent import DSPyAgent, QueryPlanSignature, DeepResearchSignature, _parse_tool_calls
from context_harness.rag_pipeline import RAGPipeline
from context_harness.retrieval_tools import build_retrieval_registry


@dataclass
class ActivityDeps:
    agent: DSPyAgent
    pipeline: RAGPipeline
    conv_store: ConversationStore
    model: str = "gemini/gemini-2.5-flash-lite"


_deps: Optional[ActivityDeps] = None


def init_activity_deps(deps: ActivityDeps) -> None:
    global _deps
    _deps = deps


def _get_deps() -> ActivityDeps:
    if _deps is None:
        raise RuntimeError("Activity deps not initialised — call init_activity_deps() in the worker.")
    return _deps


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

@activity.defn
async def retrieve_chunks(question: str, collection: str, k: int) -> list[dict]:
    """Vector-search the corpus. Returns JSON-serialisable chunk dicts."""
    deps = _get_deps()
    chunks = await asyncio.to_thread(deps.pipeline.retrieve, question, k)
    return [{"doc_id": c.doc_id, "text": c.text, "score": float(c.score or 0.0)} for c in chunks]


# ---------------------------------------------------------------------------
# Intent routing
# ---------------------------------------------------------------------------

@activity.defn
async def route_intent(question: str) -> dict:
    """Run the intent router. Returns {mode, confidence, kwargs}."""
    deps = _get_deps()

    def _run() -> dict:
        with dspy.context(lm=dspy.LM(deps.model)):
            return deps.agent.route_only(question)

    return await asyncio.to_thread(_run)


# ---------------------------------------------------------------------------
# Query planning
# ---------------------------------------------------------------------------

@activity.defn
async def plan_tool_calls(question: str) -> list[dict]:
    """Call QueryPlanSignature to get 2-3 targeted tool calls for a question."""
    deps = _get_deps()
    planner = dspy.ChainOfThought(QueryPlanSignature)

    def _run() -> list[dict]:
        with dspy.context(lm=dspy.LM(deps.model)):
            pred = planner(question=question)
        return _parse_tool_calls(pred.tool_calls)

    return await asyncio.to_thread(_run)


@activity.defn
async def execute_tool_call(tool_name: str, args: dict, collection: str) -> list[dict]:
    """Execute one tool call via the ToolRegistry and return chunk dicts."""
    deps = _get_deps()
    registry = build_retrieval_registry(deps.pipeline)
    if "k" in args:
        args["k"] = max(1, min(int(args["k"]), 15))
    result = await asyncio.to_thread(registry.call_sync, tool_name, args)
    return result


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

@activity.defn
async def synthesise_research(question: str, context: str) -> dict:
    """Run DeepResearchSignature over merged context. Returns answer dict."""
    deps = _get_deps()
    synthesiser = dspy.ChainOfThought(DeepResearchSignature)

    def _run() -> dict:
        with dspy.context(lm=dspy.LM(deps.model)):
            pred = synthesiser(question=question, context=context)
        return {
            "answer":     getattr(pred, "answer",     ""),
            "citations":  getattr(pred, "citations",  ""),
            "confidence": getattr(pred, "confidence", ""),
            "gaps":       getattr(pred, "gaps",       ""),
        }

    return await asyncio.to_thread(_run)


@activity.defn
async def guided_learning_forward(question: str, context: str, chat_history: str) -> dict:
    """Run the GuidedLearning DSPy module. Returns {hint, explanation, next_question}."""
    deps = _get_deps()

    def _run() -> dict:
        with dspy.context(lm=dspy.LM(deps.model)):
            pred = deps.agent.forward(
                "guided_learning", question,
                context=context,
                chat_history=chat_history,
            )
        return {
            "hint":          getattr(pred, "hint",          ""),
            "explanation":   getattr(pred, "explanation",   ""),
            "next_question": getattr(pred, "next_question", ""),
        }

    return await asyncio.to_thread(_run)


# ---------------------------------------------------------------------------
# Conversation persistence
# ---------------------------------------------------------------------------

@activity.defn
async def load_conversation_history(conversation_id: str, max_turns: int) -> str:
    """Load prior turns and return as a formatted string for LLM injection."""
    deps = _get_deps()
    history = await asyncio.to_thread(
        deps.conv_store.load_history, conversation_id, max_turns,
    )
    return deps.conv_store.format_for_llm(history, mode="guided_learning")


@activity.defn
async def save_conversation_turn(
    conversation_id: str,
    user_message: str,
    agent_response: dict,
    mode: str,
    tokens_in: int,
    tokens_out: int,
    cost_usd: float,
) -> int:
    """Persist a completed turn to SQLite. Returns the turn index."""
    deps = _get_deps()
    return await asyncio.to_thread(
        deps.conv_store.save_turn,
        conversation_id=conversation_id,
        user_message=user_message,
        agent_response=agent_response,
        mode=mode,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_usd,
    )


@activity.defn
async def compact_conversation(conversation_id: str) -> bool:
    """Run compaction on the conversation history. Returns True if it compacted."""
    deps = _get_deps()
    return await asyncio.to_thread(
        deps.conv_store.compact_if_needed,
        conversation_id,
        summarizer=GeminiSummarizer(),
        threshold=DEFAULT_COMPACTION_THRESHOLD,
        keep_recent=DEFAULT_KEEP_RECENT,
    )
