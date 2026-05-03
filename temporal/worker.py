"""
Temporal worker entrypoint for the HP Lore Agent.

Connects to a Temporal server, builds the RAG pipeline + DSPy agent,
registers all activities and workflows, then polls the "hp-lore" task queue.

Run:
    GOOGLE_API_KEY=... TEMPORAL_HOST=localhost:7233 python -m temporal.worker

Environment variables:
    TEMPORAL_HOST   Temporal server address (default: localhost:7233)
    DSPY_MODEL      LiteLLM model string (default: gemini/gemini-2.5-flash-lite)
    AGENT_DIR       Path to compiled DSPy JSON artefacts (default: my_profile.agent)
"""
from __future__ import annotations

import asyncio
import os

import dspy
from temporalio.client import Client
from temporalio.worker import Worker

from context_harness.conversation import ConversationStore
from context_harness.dspy_agent import DSPyAgent
from context_harness.ingest_lore import build_pipeline

from temporal.activities import (
    ActivityDeps,
    compact_conversation,
    execute_tool_call,
    guided_learning_forward,
    init_activity_deps,
    load_conversation_history,
    plan_tool_calls,
    retrieve_chunks,
    route_intent,
    save_conversation_turn,
    synthesise_research,
)
from temporal.workflows import GuidedLearningWorkflow, MultiToolResearchWorkflow

TASK_QUEUE = "hp-lore"


async def main() -> None:
    model = os.getenv("DSPY_MODEL", "gemini/gemini-2.5-flash-lite")
    temporal_host = os.getenv("TEMPORAL_HOST", "localhost:7233")
    agent_dir = os.getenv("AGENT_DIR", "my_profile.agent")

    # Configure DSPy once for this worker process.
    dspy.configure(lm=dspy.LM(model))

    # Build pipeline + agent — same singleton pattern as the FastAPI app.
    pipeline = build_pipeline(persist=True, collection_name="hp_lore")
    agent = DSPyAgent(pipeline, export_dir=agent_dir)

    init_activity_deps(ActivityDeps(
        agent=agent,
        pipeline=pipeline,
        conv_store=ConversationStore(),
        model=model,
    ))

    client = await Client.connect(temporal_host)

    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MultiToolResearchWorkflow, GuidedLearningWorkflow],
        activities=[
            route_intent,
            retrieve_chunks,
            plan_tool_calls,
            execute_tool_call,
            synthesise_research,
            guided_learning_forward,
            load_conversation_history,
            save_conversation_turn,
            compact_conversation,
        ],
    ):
        print(f"[temporal-worker] Connected to {temporal_host}, polling queue={TASK_QUEUE}")
        await asyncio.Event().wait()  # run until interrupted


if __name__ == "__main__":
    asyncio.run(main())
