"""
deepTutor Context Engineering Harness
======================================
A Python library for context engineering in RAG-based tutoring systems.
Covers: window management, context assembly, RAG pipeline, and prompt templating.
"""

from .context_manager import ContextWindow, ContextEntry, ContextRole
from .rag_pipeline import (
    RAGPipeline, Chunk, ScoredChunk, Document, ChunkingStrategy, ChunkType,
)
from .context_assembler import ContextAssembler, AssemblyStrategy
from .prompt_templates import PromptTemplate, SystemPrompt, LorePrompt

# deepTutor weakness fixes
from .summarizer import SummarizationQueue, SummarizingContextWindow, SummaryStore
from .document_registry import DocumentRegistry, UpsertResult
from .cost_tracker import CostTracker, CostEvent, CostStore, track_cost, estimate_cost_usd
from .index_version_guard import IndexVersionGuard, IndexManifest, IndexVersionError
from .retrieval_cache import RetrievalCache, CachedRAGPipeline

# IBM agent engineering skills
from .unified_context import UnifiedContext, ContextBuilder, TurnRuntimeManager, EscalationPolicy, SessionStore, TurnStatus
from .tool_registry import ToolRegistry, ToolSchema, PropertySchema, ToolContractError, ToolNotFoundError
from .reliability import RetryPolicy, CircuitBreaker, CircuitState, ResilientCaller, RetryExhausted, CircuitOpenError, with_timeout, with_fallback
from .security import PromptGuard, InputValidator, OutputFilter, PermissionBoundary, SecurityPipeline, SecurityError, ThreatLevel
from .tracer import Tracer, TraceEvent, TraceStore, EventKind, EvalMetrics, TurnRecord

__all__ = [
    # Core
    "ContextWindow",
    "ContextEntry",
    "ContextRole",
    "RAGPipeline",
    "Chunk",
    "ScoredChunk",
    "Document",
    "ChunkingStrategy",
    "ChunkType",
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
