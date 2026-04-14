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

# deepTutor weakness fixes
from .summarizer import SummarizationQueue, SummarizingContextWindow, SummaryStore
from .document_registry import DocumentRegistry, UpsertResult
from .cost_tracker import CostTracker, CostEvent, CostStore, track_cost, estimate_cost_usd
from .index_version_guard import IndexVersionGuard, IndexManifest, IndexVersionError
from .retrieval_cache import RetrievalCache, CachedRAGPipeline

__all__ = [
    # Core
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
    # Weakness fixes
    "SummarizationQueue",
    "SummarizingContextWindow",
    "SummaryStore",
    "DocumentRegistry",
    "UpsertResult",
    "CostTracker",
    "CostEvent",
    "CostStore",
    "track_cost",
    "estimate_cost_usd",
    "IndexVersionGuard",
    "IndexManifest",
    "IndexVersionError",
    "RetrievalCache",
    "CachedRAGPipeline",
]
