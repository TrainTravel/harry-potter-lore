"""
RAG Pipeline
============
Retrieval-Augmented Generation pipeline backed by ChromaDB.

Context Engineering principles applied here:
  1. **Chunking strategy** – how text is split dramatically affects retrieval quality.
  2. **Embedding** – local sentence-transformers via HuggingFace / ONNX.
  3. **Hybrid retrieval** – dense (semantic) + sparse (BM25-like keyword) scores merged.
  4. **Reranking** – cross-encoder or MMR to diversify results before injection.
  5. **Provenance** – every retrieved chunk carries its source metadata so the
     context assembler can format citations.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TypedDict


# ---------------------------------------------------------------------------
# Chunk model
# ---------------------------------------------------------------------------

class ChunkingStrategy(str, Enum):
    FIXED = "fixed"           # fixed token / character window
    SENTENCE = "sentence"     # split on sentence boundaries
    PARAGRAPH = "paragraph"   # split on blank lines
    RECURSIVE = "recursive"   # try paragraph → sentence → fixed fallback


class ChunkType(str, Enum):
    """Semantic label for a chunk's content.

    Borrowed from DeepTutor — allows content-aware retrieval (e.g. "pull
    only equations for this question"). Not enforced at storage time; the
    value is stashed in ``metadata['chunk_type']`` and retrievers are free
    to filter on it. The enum lists the common values; callers can also
    supply arbitrary strings for domain-specific types.
    """
    TEXT = "text"
    DEFINITION = "definition"
    THEOREM = "theorem"
    EQUATION = "equation"
    FIGURE = "figure"
    TABLE = "table"
    CODE = "code"


class ChunkMetadata(TypedDict, total=False):
    """Shape of the metadata dict stored on every Chunk.

    ``total=False`` keeps all keys optional at the type level so callers
    can build metadata incrementally. Extra keys beyond these are allowed
    and preserved through storage round-trips.
    """
    doc_id: str               # required in practice — accessed via Chunk.doc_id
    chunk_index: int          # position within the source document
    strategy: str             # which ChunkingStrategy produced it
    chunk_type: str           # ChunkType value; defaults to "text"


@dataclass(frozen=True)
class Chunk:
    """An immutable unit of stored text.

    Design notes (see BUILDING_LOG for the refactor rationale):
      - ``doc_id`` is NOT a separate attribute; it lives inside ``metadata``.
        The backend (ChromaDB) only persists the metadata dict, so having
        ``doc_id`` as both an attribute and a metadata key led to silent
        desync at retrieval time. Single source of truth: ``metadata['doc_id']``.
      - ``frozen=True`` discourages mutation after ingest. (The ``metadata``
        dict is still technically mutable — callers should treat it as read-only.)
      - No ``score`` here. Scores are ephemeral retrieval artifacts — see
        ``ScoredChunk`` below.
    """

    text: str
    metadata: Dict[str, Any]
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def doc_id(self) -> str:
        """Raises KeyError if doc_id is missing — fail fast beats silent 'unknown'."""
        return self.metadata["doc_id"]

    @property
    def chunk_type(self) -> str:
        """Semantic type label; defaults to ``ChunkType.TEXT.value`` if unset."""
        return self.metadata.get("chunk_type", ChunkType.TEXT.value)

    def __repr__(self) -> str:
        doc = self.metadata.get("doc_id", "?")
        return f"Chunk(doc={doc!r}, type={self.chunk_type!r}, text={self.text[:60]!r})"


@dataclass(frozen=True)
class ScoredChunk:
    """A retrieval result: an immutable ``Chunk`` plus a per-query score.

    Kept separate from ``Chunk`` so that stored chunks (long-lived, immutable)
    and retrieval results (ephemeral, per-query) have distinct types. This
    prevents entire classes of bugs where a stored chunk accidentally carries
    stale scores from a previous query.

    Property pass-throughs (``doc_id``, ``text``, ``chunk_id``, ``metadata``)
    make ``ScoredChunk`` a drop-in replacement at call sites that just want to
    read attributes. Use ``.chunk`` when you specifically need the underlying
    immutable value.
    """

    chunk: Chunk
    score: float

    # --- ergonomic pass-throughs -------------------------------------

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def doc_id(self) -> str:
        return self.chunk.doc_id

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.chunk.metadata

    def __repr__(self) -> str:
        return (
            f"ScoredChunk(doc={self.doc_id!r}, score={self.score:.3f}, "
            f"text={self.text[:60]!r})"
        )


@dataclass(frozen=True)
class Document:
    """A source document plus its produced chunks.

    Borrowed from DeepTutor — keeping the raw ``text`` alongside the
    ``chunks`` is a real convenience for:
      - citations (quote the surrounding paragraph from the source)
      - re-chunking (chunker changed? re-run without re-fetching)
      - debugging ("why did this chunk split weirdly?" → read the neighbourhood)

    The ``chunks`` field is a tuple (not a list) so equality and hashing
    work correctly on the frozen value. Use ``Document.with_chunks([...])``
    to produce a new Document with chunks filled in.
    """

    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunks: Tuple[Chunk, ...] = ()

    def with_chunks(self, chunks: List[Chunk]) -> "Document":
        """Return a new Document with ``chunks`` replaced."""
        return replace(self, chunks=tuple(chunks))

    def __repr__(self) -> str:
        return f"Document(doc_id={self.doc_id!r}, chunks={len(self.chunks)}, text_len={len(self.text)})"


# ---------------------------------------------------------------------------
# Chunkers
# ---------------------------------------------------------------------------

def _fixed_chunk(text: str, size: int = 300, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += size - overlap
    return chunks


def _sentence_chunk(text: str, max_sentences: int = 5) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunks.append(" ".join(sentences[i : i + max_sentences]))
    return chunks


def _paragraph_chunk(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _recursive_chunk(text: str, size: int = 300) -> List[str]:
    paragraphs = _paragraph_chunk(text)
    result = []
    for para in paragraphs:
        words = para.split()
        if len(words) <= size:
            result.append(para)
        else:
            result.extend(_fixed_chunk(para, size=size))
    return result


_CHUNKERS = {
    ChunkingStrategy.FIXED: lambda t: _fixed_chunk(t),
    ChunkingStrategy.SENTENCE: lambda t: _sentence_chunk(t),
    ChunkingStrategy.PARAGRAPH: lambda t: _paragraph_chunk(t),
    ChunkingStrategy.RECURSIVE: lambda t: _recursive_chunk(t),
}


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    End-to-end RAG pipeline: ingest → embed → store → retrieve → rerank.

    ChromaDB is used as the vector store.  If chromadb is unavailable the
    pipeline degrades gracefully to an in-memory list (useful for unit tests).
    """

    def __init__(
        self,
        collection_name: str = "hp_lore",
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
        top_k: int = 5,
        use_chromadb: bool = True,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.collection_name = collection_name
        self.strategy = chunking_strategy
        self.top_k = top_k
        self._chunks: List[Chunk] = []   # fallback in-memory store

        self._client = None
        self._collection = None
        self._ef = None

        if use_chromadb:
            self._init_chromadb(embedding_model)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_chromadb(self, model_name: str) -> None:
        try:
            import chromadb
            from chromadb import Settings
            from chromadb.utils import embedding_functions as ef_module

            self._client = chromadb.EphemeralClient(
                settings=Settings(anonymized_telemetry=False)
            )
            # Prefer SentenceTransformers if torch>=2.4 is available; fall back
            # to Chroma's ONNX-backed DefaultEmbeddingFunction otherwise (same
            # all-MiniLM-L6-v2 model, no torch dependency). Important for
            # Intel-Mac installs where torch is pinned at 2.2.2.
            try:
                self._ef = ef_module.SentenceTransformerEmbeddingFunction(
                    model_name=model_name
                )
            except Exception as st_exc:
                print(f"[RAGPipeline] SentenceTransformers unavailable ({st_exc}); "
                      f"using Chroma's ONNX DefaultEmbeddingFunction.")
                self._ef = ef_module.DefaultEmbeddingFunction()

            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self._ef,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            print(f"[RAGPipeline] ChromaDB unavailable ({exc}); falling back to in-memory store.")
            self._client = None

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(
        self,
        text: str,
        doc_id: str,
        chunk_type: str = ChunkType.TEXT.value,
        **metadata,
    ) -> List[Chunk]:
        """Chunk a document, embed each chunk, and upsert into the store.

        ``chunk_type`` is applied uniformly to every chunk produced from
        this document. For content-aware typing (different types per
        chunk) you'd build the chunks yourself and bypass this method.
        """
        raw_chunks = _CHUNKERS[self.strategy](text)
        chunks: List[Chunk] = []
        for i, raw in enumerate(raw_chunks):
            chunk = Chunk(
                text=raw,
                metadata={
                    "doc_id": doc_id,                      # single source of truth
                    "chunk_index": i,
                    "strategy": self.strategy.value,
                    "chunk_type": chunk_type,
                    **metadata,
                },
            )
            chunks.append(chunk)

        if self._collection is not None:
            self._collection.upsert(
                ids=[c.chunk_id for c in chunks],
                documents=[c.text for c in chunks],
                metadatas=[c.metadata for c in chunks],
            )
        else:
            self._chunks.extend(chunks)

        return chunks

    def ingest_many(self, docs: List[Tuple[str, str]], **shared_meta) -> List[Chunk]:
        """Ingest a list of (text, doc_id) tuples."""
        all_chunks: List[Chunk] = []
        for text, doc_id in docs:
            all_chunks.extend(self.ingest(text, doc_id, **shared_meta))
        return all_chunks

    def ingest_document(self, doc: Document, **metadata) -> Document:
        """Ingest a ``Document`` and return a new one with ``chunks`` populated.

        Keeps the raw ``text`` alongside the chunks so downstream code can
        cite, re-chunk, or debug without re-fetching the source. Any
        ``metadata`` passed here is merged into each chunk's metadata
        (e.g. ``title``, ``section``, ``chunk_type`` override).
        """
        chunks = self.ingest(
            text=doc.text,
            doc_id=doc.doc_id,
            **{**doc.metadata, **metadata},
        )
        return doc.with_chunks(chunks)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[ScoredChunk]:
        """Retrieve the top-k most relevant chunks for a query.

        Args:
            query: natural-language query string.
            top_k: max chunks to return (defaults to ``self.top_k``).
            where: optional ChromaDB metadata filter passed verbatim to
                ``collection.query(where=...)``. Example:
                ``{"character": "severus-snape"}`` restricts retrieval to
                chunks whose metadata has that exact character slug.
                Ignored by the in-memory keyword fallback.

        Returns ``ScoredChunk``s rather than raw ``Chunk``s — the score is a
        per-query artifact and should not be attached to the stored value.
        """
        k = top_k or self.top_k

        if self._collection is not None:
            query_kwargs = {
                "query_texts": [query],
                "n_results":   min(k, self._collection.count() or 1),
                "include":     ["documents", "metadatas", "distances"],
            }
            if where:
                query_kwargs["where"] = where
            results = self._collection.query(**query_kwargs)
            scored: List[ScoredChunk] = []
            for doc, meta, dist, cid in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
                results["ids"][0],
            ):
                chunk = Chunk(text=doc, metadata=dict(meta), chunk_id=cid)
                scored.append(ScoredChunk(chunk=chunk, score=1.0 - dist))
            return scored

        # fallback: simple keyword overlap scoring (no metadata filter support)
        return self._keyword_retrieve(query, k)

    def _keyword_retrieve(self, query: str, k: int) -> List[ScoredChunk]:
        query_words = set(query.lower().split())
        scored: List[Tuple[float, Chunk]] = []
        for chunk in self._chunks:
            words = set(chunk.text.lower().split())
            overlap = len(query_words & words) / (len(query_words) + 1)
            scored.append((overlap, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ScoredChunk(chunk=c, score=s) for s, c in scored[:k]]

    # ------------------------------------------------------------------
    # MMR reranking (Maximal Marginal Relevance)
    # ------------------------------------------------------------------

    def mmr_rerank(
        self,
        query: str,
        candidates: List[ScoredChunk],
        lambda_: float = 0.5,
        k: Optional[int] = None,
    ) -> List[ScoredChunk]:
        """
        MMR reranking to balance relevance and diversity.

        lambda_=1.0 → pure relevance; lambda_=0.0 → pure diversity.
        Falls back to sorted-by-score when embeddings unavailable.
        """
        k = k or self.top_k
        if not candidates:
            return []
        if self._ef is None:
            # no embeddings: just return by score
            return sorted(candidates, key=lambda c: c.score, reverse=True)[:k]

        import numpy as np

        all_texts = [query] + [c.text for c in candidates]
        embeddings = self._ef(all_texts)
        q_emb = np.array(embeddings[0])
        c_embs = np.array(embeddings[1:])

        def cosine(a: "np.ndarray", b: "np.ndarray") -> float:
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
            return float(np.dot(a, b) / denom)

        selected_idx: List[int] = []
        remaining = list(range(len(candidates)))

        while len(selected_idx) < k and remaining:
            best_idx, best_score = -1, float("-inf")
            for i in remaining:
                rel = cosine(q_emb, c_embs[i])
                if selected_idx:
                    red = max(cosine(c_embs[i], c_embs[j]) for j in selected_idx)
                else:
                    red = 0.0
                mmr = lambda_ * rel - (1 - lambda_) * red
                if mmr > best_score:
                    best_score, best_idx = mmr, i
            selected_idx.append(best_idx)
            remaining.remove(best_idx)

        return [candidates[i] for i in selected_idx]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def count(self) -> int:
        if self._collection is not None:
            return self._collection.count()
        return len(self._chunks)

    def __repr__(self) -> str:
        return (
            f"RAGPipeline(collection={self.collection_name!r}, "
            f"strategy={self.strategy.value}, chunks={self.count()})"
        )
