"""
Reranker Protocol + adapters.
=============================

A reranker scores retrieved candidate chunks against the query with a
stronger relevance model than embedding cosine similarity. Used as a
second stage after vector retrieval:

    vector search (top N=20)  →  reranker  →  top K=5

Why: raw embedding similarity is trained for semantic nearness, not
for answer-relevance ranking. A dedicated reranker model (Cohere, Voyage,
cross-encoders) often gives 10-20% higher hit-rate on the top-k.

Design:
  - ``RerankerClient`` Protocol — structural interface
  - ``CohereReranker`` — production adapter using the Cohere API
  - ``IdentityReranker`` — no-op passthrough, for A/B baselines
  - ``FakeReranker`` — test double with scripted orderings

Integration: ``RAGPipeline.__init__`` accepts an optional ``reranker``;
when set, ``retrieve()`` over-fetches (``fetch_k=20``) and calls the
reranker to cut down to ``top_k``.

Graceful fallback: if the reranker raises, the caller returns the
un-reranked top-k so availability isn't blocked on observability.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Protocol, runtime_checkable

from context_harness.rag_pipeline import ScoredChunk

_log = logging.getLogger(__name__)

# Default Cohere model. The bare "rerank-v3.0" string used in an earlier
# iteration is NOT a valid Cohere model identifier — Cohere publishes
# "rerank-english-v3.0", "rerank-multilingual-v3.0", and "rerank-v3.5".
# Override via env: COHERE_RERANK_MODEL="rerank-v3.5".
_DEFAULT_COHERE_MODEL = os.environ.get("COHERE_RERANK_MODEL", "rerank-english-v3.0")

# Timeout (seconds) applied to the Cohere rerank network call. Bounded so a
# hanging API doesn't stall /ask latency. Override via env.
_DEFAULT_COHERE_TIMEOUT_S = float(os.environ.get("COHERE_RERANK_TIMEOUT_S", "10.0"))


@runtime_checkable
class RerankerClient(Protocol):
    """Anything that can score + reorder candidate chunks by relevance."""

    def rerank(
        self,
        query: str,
        candidates: List[ScoredChunk],
        top_k: int,
    ) -> List[ScoredChunk]:
        ...


# ---------------------------------------------------------------------------
# IdentityReranker — no-op, useful for A/B baselines
# ---------------------------------------------------------------------------

class IdentityReranker:
    """No-op reranker — returns the first ``top_k`` candidates unchanged.

    Used as the control arm in A/B evals: swap a real reranker for
    ``IdentityReranker`` and measure the delta. Also the safe default
    when no Cohere key is configured.
    """

    def rerank(
        self,
        query: str,
        candidates: List[ScoredChunk],
        top_k: int,
    ) -> List[ScoredChunk]:
        return candidates[:top_k]


# ---------------------------------------------------------------------------
# FakeReranker — test double
# ---------------------------------------------------------------------------

@dataclass
class _FakeCall:
    """Test-visible record of a single rerank invocation."""
    query: str
    candidate_doc_ids: List[str]
    top_k: int


class FakeReranker:
    """Test double that records every call and applies a scripted ordering.

    Unlike real rerankers, this one exposes ``calls`` and
    ``last_query`` so tests can assert on what was sent — mirroring the
    FakeLLMClient pattern used elsewhere.

    Args:
        ordering: either a callable ``(query, candidates) -> candidates``
            that returns the scripted new ordering, or ``None`` to
            reverse the input (convenient for tests that want to verify
            "reranker actually ran").
    """

    def __init__(self, ordering=None) -> None:
        self._ordering = ordering
        self.calls: List[_FakeCall] = []

    def rerank(
        self,
        query: str,
        candidates: List[ScoredChunk],
        top_k: int,
    ) -> List[ScoredChunk]:
        self.calls.append(_FakeCall(
            query=query,
            candidate_doc_ids=[c.doc_id for c in candidates],
            top_k=top_k,
        ))
        if self._ordering is not None:
            return self._ordering(query, candidates)[:top_k]
        # default: reverse the input so tests can verify reranker ran
        return list(reversed(candidates))[:top_k]


# ---------------------------------------------------------------------------
# CohereReranker — production adapter
# ---------------------------------------------------------------------------

class CohereReranker:
    """Cohere rerank-v3 adapter.

    Env-driven: construct with ``CohereReranker()`` and it picks up
    ``COHERE_API_KEY`` from the environment. Raising from ``rerank()``
    is OK — the RAGPipeline wrapping this is expected to catch and fall
    back to the un-reranked top-k.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: Optional[float] = None,
    ) -> None:
        key = api_key or os.environ.get("COHERE_API_KEY")
        if not key:
            raise RuntimeError(
                "CohereReranker requires an API key. Either pass api_key=... "
                "or set COHERE_API_KEY in the environment.",
            )
        try:
            import cohere  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "CohereReranker requires the 'cohere' package. "
                "Install with: uv pip install cohere",
            ) from exc
        self._client = cohere.Client(api_key=key)
        self._model = model or _DEFAULT_COHERE_MODEL
        self._timeout_s = timeout_s if timeout_s is not None else _DEFAULT_COHERE_TIMEOUT_S

    def rerank(
        self,
        query: str,
        candidates: List[ScoredChunk],
        top_k: int,
    ) -> List[ScoredChunk]:
        if not candidates:
            return []
        if top_k <= 0:
            return []

        import cohere  # for RequestOptions

        docs = [c.text for c in candidates]
        response = self._client.rerank(
            model=self._model,
            query=query,
            documents=docs,
            top_n=min(top_k, len(docs)),
            request_options=cohere.core.RequestOptions(timeout_in_seconds=self._timeout_s),
        )
        # response.results = list of {index, relevance_score}
        reranked: List[ScoredChunk] = []
        seen_indices: set[int] = set()
        for r in response.results:
            # Defensive bounds check — a buggy API response with index
            # outside [0, len(docs)) shouldn't IndexError the caller.
            if r.index < 0 or r.index >= len(candidates):
                _log.warning(
                    "CohereReranker: response index %d out of range [0, %d); skipping",
                    r.index, len(candidates),
                )
                continue
            # Defensive dedup — shouldn't happen but would produce duplicate
            # chunks in the output if it did.
            if r.index in seen_indices:
                continue
            seen_indices.add(r.index)
            original = candidates[r.index]
            # Replace the vector-similarity score with the reranker's
            # relevance score (0.0-1.0). Note: downstream MMR recomputes
            # its own scores, so this is only load-bearing for direct
            # consumers (citations panel, traces, cost dashboards).
            reranked.append(ScoredChunk(chunk=original.chunk, score=r.relevance_score))
        return reranked[:top_k]


# ---------------------------------------------------------------------------
# Factory — env-gated construction
# ---------------------------------------------------------------------------

def build_reranker_from_env() -> Optional[RerankerClient]:
    """Construct a reranker from environment variables.

    Returns:
        ``CohereReranker`` if ``COHERE_API_KEY`` is set.
        ``None`` otherwise — callers should treat this as "no reranking,
        use raw vector-search top-k".

    Useful default wiring for ``build_pipeline()`` — keeps CI + offline
    dev working unchanged when no API key is configured.
    """
    if os.environ.get("COHERE_API_KEY"):
        try:
            return CohereReranker()
        except Exception as exc:
            _log.warning(
                "CohereReranker init failed (%s); proceeding without reranker.",
                exc,
            )
            return None
    return None
