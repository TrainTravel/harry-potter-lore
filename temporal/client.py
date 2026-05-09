"""
Temporal client helpers for submitting HP Lore workflows.

Provides thin async wrappers around the Temporal client. Import in FastAPI
endpoints or one-off scripts to submit workflows without managing Temporal
SDK boilerplate.

Usage:
    result = await run_research("Who owned the Elder Wand?")
    wf_id  = await submit_research("...")  # fire and forget
    result = await get_research_result(wf_id)  # poll later
"""
from __future__ import annotations

import os
import uuid
from typing import Optional

from temporalio.client import Client

from temporal.workflows import (
    GuidedLearningWorkflow,
    LearningInput,
    LearningOutput,
    MultiToolResearchWorkflow,
    ResearchInput,
    ResearchOutput,
)

TASK_QUEUE = "hp-lore"


async def get_temporal_client() -> Client:
    host = os.getenv("TEMPORAL_HOST", "localhost:7233")
    return await Client.connect(host)


async def submit_research(
    question: str,
    collection_name: str = "hp_lore",
    conversation_id: Optional[str] = None,
    model: str = "gemini/gemini-2.5-flash-lite",
) -> str:
    """Submit a MultiToolResearchWorkflow. Returns the workflow_id immediately."""
    client = await get_temporal_client()
    wf_id = f"research-{uuid.uuid4()}"
    await client.start_workflow(
        MultiToolResearchWorkflow.run,
        ResearchInput(
            question=question,
            collection_name=collection_name,
            conversation_id=conversation_id,
            model=model,
        ),
        id=wf_id,
        task_queue=TASK_QUEUE,
    )
    return wf_id


async def get_research_result(workflow_id: str) -> ResearchOutput:
    """Wait for a research workflow to complete and return its result."""
    client = await get_temporal_client()
    handle = client.get_workflow_handle(workflow_id)
    return await handle.result()


async def run_research(
    question: str,
    collection_name: str = "hp_lore",
    conversation_id: Optional[str] = None,
) -> ResearchOutput:
    """Submit a research workflow and wait for the result in one call."""
    wf_id = await submit_research(question, collection_name, conversation_id)
    return await get_research_result(wf_id)


async def run_learning(
    question: str,
    conversation_id: str,
    collection_name: str = "hp_lore",
) -> LearningOutput:
    """Submit a GuidedLearningWorkflow and wait for the result."""
    client = await get_temporal_client()
    wf_id = f"learning-{uuid.uuid4()}"
    return await client.execute_workflow(
        GuidedLearningWorkflow.run,
        LearningInput(
            question=question,
            conversation_id=conversation_id,
            collection_name=collection_name,
        ),
        id=wf_id,
        task_queue=TASK_QUEUE,
    )
