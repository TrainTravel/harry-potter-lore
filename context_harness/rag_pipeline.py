"""
RAG Pipeline
============
Retrieval-Augmented Generation pipeline backed by ChromaDB.

Context Engineering principles applied here:
  1. **Chunking strategy** – how text is split dramatically affects retrieval quality.
  2. **Embedding** – local sentence-transformers via HuggingFace / ONNX.
  3. **Hybrid retrieval** – dense (semantic) + sparse (BM25-like keyword) scores merged.
  4. **Reranking** – MMR to diversify results before injection; cross-encoder slot
     reserved for a second-stage precision reranker.
  5. **Provenance** – every retrieved chunk carries universe, source_type, and as_of
     metadata so the context assembler can format citations and caveats correctly.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Source classification
# ---------------------------------------------------------------------------

class SourceType(str, Enum):
    """
    Classifies the epistemic status of a chunk's source.

    Used by the context assembler to:
      - Add "In the HP universe, ..." framing for fictional_canon chunks
      - Add source + date caveats for news chunks
      - Warn when as_of is stale for real_world sources
    """
    FICTIONAL_CANON = "fictional_canon"   # HP, LotR, Witcher, etc.
    BIOGRAPHY       = "biography"         # long-form biographical text
    WIKI            = "wiki"              # Wikipedia-style article
    NEWS            = "news"              # news article (may be time-sensitive)


# ---------------------------------------------------------------------------
# Chunk model
# ---------------------------------------------------------------------------

class ChunkingStrategy(str, Enum):
    FIXED     = "fixed"       # fixed token / character window
    SENTENCE  = "sentence"    # split on sentence boundaries
    PARAGRAPH = "paragraph"   # split on blank lines
    RECURSIVE = "recursive"   # paragraph → sentence → fixed fallback
    WIKI      = "wiki"        # split on markdown ## / ### section headers


@dataclass
class Chunk:
    """A unit of text that will be embedded and stored."""

    text: str
    doc_id: str
    chunk_id: str         = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float          = 0.0

    # --- provenance fields (first-class, also mirrored into metadata) --------
    universe: str                  = "hp"
    source_type: SourceType        = SourceType.FICTIONAL_CANON
    as_of: Optional[str]           = None   # "YYYY-MM" for real-world; None for fiction

    def __repr__(self) -> str:
        return (
            f"Chunk(doc={self.doc_id!r}, universe={self.universe!r}, "
            f"score={self.score:.3f}, text={self.text[:60]!r})"
        )


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


def _wiki_chunk(text: str) -> List[str]:
    """
    Split on markdown section headers (# / ## / ###).

    Respects the editorial structure of Wikipedia-style articles — each
    section becomes its own chunk, keeping the header as the first line so
    the section topic is present in every embedding.

    Example:
        ## Early life
        Born in 1954...     → one chunk: "## Early life\nBorn in 1954..."

        ## Career
        In 1979 he...       → one chunk: "## Career\nIn 1979 he..."
    """
    # Split just before any line that starts with one or more # characters
    raw_sections = re.split(r"(?=\n#{1,3}\s)", text)
    sections = []
    for sec in raw_sections:
        sec = sec.strip()
        if not sec:
            continue
        # If a section is very long, fall back to recursive chunking within it
        if len(sec.split()) > 400:
            sections.extend(_recursive_chunk(sec))
        else:
            sections.append(sec)
    return sections if sections else _paragraph_chunk(text)


_CHUNKERS = {
    ChunkingStrategy.FIXED:     lambda t: _fixed_chunk(t),
    ChunkingStrategy.SENTENCE:  lambda t: _sentence_chunk(t),
    ChunkingStrategy.PARAGRAPH: lambda t: _paragraph_chunk(t),
    ChunkingStrategy.RECURSIVE: lambda t: _recursive_chunk(t),
    ChunkingStrategy.WIKI:      lambda t: _wiki_chunk(t),
}


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    End-to-end RAG pipeline: ingest → embed → store → retrieve → rerank.

    ChromaDB is used as the vector store.  If chromadb is unavailable the
    pipeline degrades gracefully to an in-memory list (useful for unit tests).

    Every chunk carries three provenance fields stored in ChromaDB metadata:
      universe    — which fictional/real-world corpus the chunk came from
      source_type — epistemic classification (fictional_canon, wiki, news, …)
      as_of       — freshness date for real-world sources (None for fiction)

    These fields enable metadata-filtered queries so a user asking purely about
    HP lore never gets Tolkien or Wikipedia chunks back, and vice-versa.
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
        self.embedding_model_name = embedding_model

        if use_chromadb:
            self._init_chromadb(embedding_model)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_chromadb(self, model_name: str) -> None:
        try:
            import chromadb
            from chromadb import Settings
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

            self._client = chromadb.EphemeralClient(
                settings=Settings(anonymized_telemetry=False)
            )
            self._ef = DefaultEmbeddingFunction()
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
        universe: str = "hp",
        source_type: SourceType = SourceType.FICTIONAL_CANON,
        as_of: Optional[str] = None,
        **metadata,
    ) -> List[Chunk]:
        """
        Chunk a document, embed each chunk, and upsert into the store.

        Args:
            text:        Full document text.
            doc_id:      Stable identifier, e.g. "hp:horcruxes" or "real:elon-musk".
            universe:    Corpus namespace, e.g. "hp", "lotr", "real_world".
            source_type: Epistemic classification of the source.
            as_of:       Freshness date "YYYY-MM" for real-world docs; None for fiction.
            **metadata:  Additional key/value pairs stored alongside the chunk.
        """
        raw_chunks = _CHUNKERS[self.strategy](text)
        chunks: List[Chunk] = []

        for i, raw in enumerate(raw_chunks):
            chunk = Chunk(
                text=raw,
                doc_id=doc_id,
                universe=universe,
                source_type=source_type,
                as_of=as_of,
                metadata={
                    "doc_id":      doc_id,
                    "chunk_index": i,
                    "strategy":    self.strategy.value,
                    "universe":    universe,
                    "source_type": source_type.value,
                    "as_of":       as_of or "",
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

    def ingest_many(
        self,
        docs: List[Tuple[str, str]],
        universe: str = "hp",
        source_type: SourceType = SourceType.FICTIONAL_CANON,
        as_of: Optional[str] = None,
        **shared_meta,
    ) -> List[Chunk]:
        """Ingest a list of (text, doc_id) tuples with shared provenance metadata."""
        all_chunks: List[Chunk] = []
        for text, doc_id in docs:
            all_chunks.extend(
                self.ingest(text, doc_id, universe=universe,
                            source_type=source_type, as_of=as_of, **shared_meta)
            )
        return all_chunks

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        universe: Optional[str] = None,
        source_type: Optional[SourceType] = None,
        use_mmr: bool = False,
        mmr_lambda: float = 0.5,
    ) -> List[Chunk]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Args:
            query:       Natural-language query string.
            top_k:       Number of chunks to return (overrides instance default).
            universe:    If set, restrict results to this universe namespace.
            source_type: If set, restrict results to this source type.
            use_mmr:     If True, apply MMR reranking after ANN retrieval to
                         balance relevance and diversity.
            mmr_lambda:  MMR trade-off (1.0 = pure relevance, 0.0 = pure diversity).
        """
        k = top_k or self.top_k

        if self._collection is not None:
            where = _build_where(universe=universe, source_type=source_type)
            results = self._collection.query(
                query_texts=[query],
                n_results=min(k, self._collection.count() or 1),
                include=["documents", "metadatas", "distances"],
                **({"where": where} if where else {}),
            )
            chunks = []
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                chunks.append(Chunk(
                    text=doc,
                    doc_id=meta.get("doc_id", "unknown"),
                    universe=meta.get("universe", "unknown"),
                    source_type=SourceType(meta["source_type"])
                        if "source_type" in meta else SourceType.FICTIONAL_CANON,
                    as_of=meta.get("as_of") or None,
                    metadata=meta,
                    score=1.0 - dist,   # cosine distance → similarity
                ))
        else:
            chunks = self._keyword_retrieve(query, k, universe=universe,
                                            source_type=source_type)

        if use_mmr and len(chunks) > 1:
            chunks = self.mmr_rerank(query, chunks, lambda_=mmr_lambda, k=k)

        return chunks

    def _keyword_retrieve(
        self,
        query: str,
        k: int,
        universe: Optional[str] = None,
        source_type: Optional[SourceType] = None,
    ) -> List[Chunk]:
        query_words = set(query.lower().split())
        candidates = [
            c for c in self._chunks
            if (universe is None or c.universe == universe)
            and (source_type is None or c.source_type == source_type)
        ]
        scored = []
        for chunk in candidates:
            words = set(chunk.text.lower().split())
            overlap = len(query_words & words) / (len(query_words) + 1)
            scored.append((overlap, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, chunk in scored[:k]:
            chunk.score = score
            results.append(chunk)
        return results

    # ------------------------------------------------------------------
    # MMR reranking (Maximal Marginal Relevance)
    # ------------------------------------------------------------------

    def mmr_rerank(
        self,
        query: str,
        candidates: List[Chunk],
        lambda_: float = 0.5,
        k: Optional[int] = None,
    ) -> List[Chunk]:
        """
        MMR reranking to balance relevance and diversity.

        lambda_=1.0 → pure relevance; lambda_=0.0 → pure diversity.
        Falls back to sorted-by-score when embeddings unavailable.
        """
        k = k or self.top_k
        if not candidates:
            return []
        if self._ef is None:
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
                red = max((cosine(c_embs[i], c_embs[j]) for j in selected_idx), default=0.0)
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_where(
    universe: Optional[str] = None,
    source_type: Optional[SourceType] = None,
) -> Optional[Dict[str, Any]]:
    """
    Build a ChromaDB metadata filter dict.

    Single condition  → {"universe": {"$eq": "hp"}}
    Two conditions    → {"$and": [{"universe": ...}, {"source_type": ...}]}
    No conditions     → None  (no filter applied)
    """
    clauses = []
    if universe is not None:
        clauses.append({"universe": {"$eq": universe}})
    if source_type is not None:
        clauses.append({"source_type": {"$eq": source_type.value}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}
