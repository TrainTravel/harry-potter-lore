"""
Context Assembler
=================
Takes retrieved chunks + conversation history and assembles the final
context block that is injected into the prompt.

Context Engineering principles:
  1. **Ordering matters** – most relevant chunk goes last (recency bias in LLMs).
  2. **Citation formatting** – structured citations help the LLM attribute claims.
  3. **Deduplication** – avoid redundant context that wastes tokens.
  4. **Score thresholding** – drop low-confidence chunks to reduce noise.
  5. **Strategy pattern** – different assembly strategies for different agents.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from .rag_pipeline import ScoredChunk


# ---------------------------------------------------------------------------
# Assembly strategies
# ---------------------------------------------------------------------------

class AssemblyStrategy(str, Enum):
    NAIVE = "naive"               # concatenate in retrieval order
    RELEVANCE = "relevance"       # sort by score descending
    SANDWICH = "sandwich"         # most relevant first and last, rest in middle
    CITATION = "citation"         # numbered footnote-style citations


# ---------------------------------------------------------------------------
# Assembler
# ---------------------------------------------------------------------------

class ContextAssembler:
    """
    Assembles a context block from retrieved chunks ready for prompt injection.

    Usage:
        assembler = ContextAssembler(strategy=AssemblyStrategy.SANDWICH)
        context_block = assembler.assemble(chunks, query="Who is Dumbledore?")
    """

    def __init__(
        self,
        strategy: AssemblyStrategy = AssemblyStrategy.SANDWICH,
        min_score: float = 0.0,
        max_chunks: int = 10,
        dedup_threshold: float = 0.95,   # Jaccard similarity above which we deduplicate
    ) -> None:
        self.strategy = strategy
        self.min_score = min_score
        self.max_chunks = max_chunks
        self.dedup_threshold = dedup_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assemble(
        self,
        chunks: List[ScoredChunk],
        query: Optional[str] = None,
        preamble: str = "Relevant knowledge from the Harry Potter lore:",
    ) -> str:
        """Return a formatted context string ready for prompt injection."""
        filtered = self._filter(chunks)
        deduped = self._dedup(filtered)
        ordered = self._order(deduped)

        if not ordered:
            return ""

        formatter = _FORMATTERS[self.strategy]
        return formatter(ordered, preamble, query)

    def assemble_as_messages(
        self,
        chunks: List[ScoredChunk],
        query: Optional[str] = None,
    ) -> List[dict]:
        """Return a list of message dicts suitable for the context window."""
        block = self.assemble(chunks, query)
        if not block:
            return []
        return [{"role": "retrieval", "content": block}]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _filter(self, chunks: List[ScoredChunk]) -> List[ScoredChunk]:
        return [c for c in chunks if c.score >= self.min_score][: self.max_chunks]

    def _dedup(self, chunks: List[ScoredChunk]) -> List[ScoredChunk]:
        kept: List[ScoredChunk] = []
        for candidate in chunks:
            duplicate = False
            for existing in kept:
                if _jaccard(candidate.text, existing.text) >= self.dedup_threshold:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)
        return kept

    def _order(self, chunks: List[ScoredChunk]) -> List[ScoredChunk]:
        if self.strategy in (AssemblyStrategy.RELEVANCE, AssemblyStrategy.SANDWICH):
            return sorted(chunks, key=lambda c: c.score, reverse=True)
        return chunks   # NAIVE / CITATION keep retrieval order


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _format_naive(chunks: List[ScoredChunk], preamble: str, _query) -> str:
    parts = [preamble, ""]
    for chunk in chunks:
        parts.append(chunk.text)
        parts.append("")
    return "\n".join(parts).strip()


def _format_relevance(chunks: List[ScoredChunk], preamble: str, _query) -> str:
    parts = [preamble, ""]
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[{i}] (score={chunk.score:.2f}) {chunk.text}")
        parts.append("")
    return "\n".join(parts).strip()


def _format_sandwich(chunks: List[ScoredChunk], preamble: str, _query) -> str:
    """
    Sandwich strategy: put the *most* relevant chunk both first and last.
    This exploits LLM primacy + recency bias to emphasise the best evidence.
    """
    if not chunks:
        return ""
    best = chunks[0]
    rest = chunks[1:]

    parts = [preamble, "", f"[Key passage] {best.text}", ""]
    for chunk in rest:
        parts.append(chunk.text)
        parts.append("")
    parts.append(f"[Key passage – repeated for emphasis] {best.text}")
    return "\n".join(parts).strip()


def _format_citation(chunks: List[ScoredChunk], preamble: str, _query) -> str:
    """
    Numbered citation format so the LLM can reference sources explicitly.
    The model is instructed (in the system prompt) to cite [N] inline.
    """
    parts = [preamble, ""]
    refs: List[str] = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.metadata.get("doc_id", "unknown")
        chapter = chunk.metadata.get("chapter", "")
        ref_label = f"{source}" + (f", {chapter}" if chapter else "")
        parts.append(f"[{i}] {chunk.text}")
        parts.append("")
        refs.append(f"[{i}] {ref_label}")
    parts.append("Sources: " + " | ".join(refs))
    return "\n".join(parts).strip()


_FORMATTERS = {
    AssemblyStrategy.NAIVE: _format_naive,
    AssemblyStrategy.RELEVANCE: _format_relevance,
    AssemblyStrategy.SANDWICH: _format_sandwich,
    AssemblyStrategy.CITATION: _format_citation,
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _jaccard(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)
