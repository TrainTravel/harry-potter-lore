"""
Retrieval Tool Registry
=======================
Registers the three HP-lore retrieval tools with strict ToolSchema contracts.
Used by MultiToolDeepResearchModule to execute validated, logged sub-queries.

Tools
-----
  vector_search(query, k)                    — broad semantic search
  character_search(character, query, k)      — filtered to a character slug
  topic_search(topic, query, k)              — topic-prefixed query

Usage
-----
    from context_harness.retrieval_tools import build_retrieval_registry
    registry = build_retrieval_registry(pipeline, char_pipeline=char_pipeline)
    chunks = registry.call_sync("vector_search", {"query": "Elder Wand", "k": 5})
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .rag_pipeline import RAGPipeline
from .tool_registry import PropertySchema, ToolRegistry, ToolSchema


def build_retrieval_registry(
    pipeline: RAGPipeline,
    char_pipeline: Optional[RAGPipeline] = None,
) -> ToolRegistry:
    """Return a ToolRegistry wired to the given RAGPipeline(s)."""
    registry = ToolRegistry()

    @registry.register(schema=ToolSchema(
        name="vector_search",
        description=(
            "Broad semantic search across the full HP lore corpus. "
            "Use for general questions that span multiple topics or characters."
        ),
        properties={
            "query": PropertySchema(
                type="string",
                description="Natural-language search query",
            ),
            "k": PropertySchema(
                type="integer",
                description="Number of chunks to return (1–15)",
                minimum=1,
                maximum=15,
            ),
        },
        examples=[
            {"query": "Elder Wand ownership history", "k": 5},
            {"query": "Horcrux creation process", "k": 6},
        ],
    ))
    def vector_search(query: str, k: int = 5) -> List[Dict[str, Any]]:
        chunks = pipeline.retrieve(query, top_k=k)
        return [{"doc_id": c.doc_id, "text": c.text} for c in chunks]

    @registry.register(schema=ToolSchema(
        name="character_search",
        description=(
            "Retrieve lore passages scoped to a specific HP character. "
            "Use when the question is primarily about one character's history, "
            "motivations, or actions."
        ),
        properties={
            "character": PropertySchema(
                type="string",
                description=(
                    "Canonical character slug, e.g. 'albus-dumbledore', "
                    "'severus-snape', 'lord-voldemort', 'harry-potter'"
                ),
            ),
            "query": PropertySchema(
                type="string",
                description="The aspect of this character to retrieve",
            ),
            "k": PropertySchema(
                type="integer",
                description="Number of chunks to return (1–10)",
                minimum=1,
                maximum=10,
            ),
        },
        examples=[
            {"character": "albus-dumbledore", "query": "Elder Wand", "k": 4},
            {"character": "severus-snape", "query": "double agent loyalty", "k": 4},
        ],
    ))
    def character_search(character: str, query: str, k: int = 4) -> List[Dict[str, Any]]:
        src = char_pipeline or pipeline
        where = {"character": character} if char_pipeline is not None else None
        chunks = src.retrieve(query, top_k=k, where=where)
        if not chunks:
            # Fallback: prefix character name into the query on generic pipeline
            chunks = pipeline.retrieve(f"{character} {query}", top_k=k)
        return [{"doc_id": c.doc_id, "text": c.text} for c in chunks]

    @registry.register(schema=ToolSchema(
        name="topic_search",
        description=(
            "Retrieve lore passages focused on a specific HP concept or object. "
            "Use for questions about artefacts, spells, institutions, or events "
            "rather than individual characters."
        ),
        properties={
            "topic": PropertySchema(
                type="string",
                description=(
                    "The concept/object to focus on, e.g. 'Horcruxes', "
                    "'Deathly Hallows', 'Order of the Phoenix', 'Hogwarts'"
                ),
            ),
            "query": PropertySchema(
                type="string",
                description="Additional query detail beyond the topic name",
            ),
            "k": PropertySchema(
                type="integer",
                description="Number of chunks to return (1–10)",
                minimum=1,
                maximum=10,
            ),
        },
        examples=[
            {"topic": "Deathly Hallows", "query": "ownership and transfer", "k": 5},
            {"topic": "Horcruxes", "query": "which objects were used", "k": 5},
        ],
    ))
    def topic_search(topic: str, query: str, k: int = 5) -> List[Dict[str, Any]]:
        chunks = pipeline.retrieve(f"{topic} {query}", top_k=k)
        return [{"doc_id": c.doc_id, "text": c.text} for c in chunks]

    return registry
