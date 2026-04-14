"""
deepTutor Context Engineering Harness
======================================
A Python library for context engineering in RAG-based tutoring systems.
Covers: window management, context assembly, RAG pipeline, and prompt templating.
"""

from .context_manager import ContextWindow, ContextEntry, ContextRole
from .rag_pipeline import RAGPipeline, Chunk, ChunkingStrategy
from .context_assembler import ContextAssembler, AssemblyStrategy
from .prompt_templates import PromptTemplate, SystemPrompt, LorePrompt

__all__ = [
    "ContextWindow",
    "ContextEntry",
    "ContextRole",
    "RAGPipeline",
    "Chunk",
    "ChunkingStrategy",
    "ContextAssembler",
    "AssemblyStrategy",
    "PromptTemplate",
    "SystemPrompt",
    "LorePrompt",
]
