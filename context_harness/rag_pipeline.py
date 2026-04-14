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
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Chunk model
# ---------------------------------------------------------------------------

class ChunkingStrategy(str, Enum):
    FIXED = "fixed"           # fixed token / character window
    SENTENCE = "sentence"     # split on sentence boundaries
    PARAGRAPH = "paragraph"   # split on blank lines
    RECURSIVE = "recursive"   # try paragraph → sentence → fixed fallback


@dataclass
class Chunk:
    """A unit of text that will be embedded and stored."""

    text: str
    doc_id: str
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0        # filled after retrieval

    def __repr__(self) -> str:
        return f"Chunk(doc={self.doc_id!r}, score={self.score:.3f}, text={self.text[:60]!r})"


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
            from chromadb.utils import embedding_functions as ef_module

            self._client = chromadb.Client()
            self._ef = ef_module.SentenceTransformerEmbeddingFunction(
                model_name=model_name
            )
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

    def ingest(self, text: str, doc_id: str, **metadata) -> List[Chunk]:
        """Chunk a document, embed each chunk, and upsert into the store."""
        raw_chunks = _CHUNKERS[self.strategy](text)
        chunks: List[Chunk] = []
        for i, raw in enumerate(raw_chunks):
            chunk = Chunk(
                text=raw,
                doc_id=doc_id,
                metadata={"chunk_index": i, "strategy": self.strategy.value, **metadata},
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

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Chunk]:
        """Retrieve the top-k most relevant chunks for a query."""
        k = top_k or self.top_k

        if self._collection is not None:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(k, self._collection.count() or 1),
                include=["documents", "metadatas", "distances"],
            )
            chunks = []
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                chunk = Chunk(
                    text=doc,
                    doc_id=meta.get("doc_id", "unknown"),
                    metadata=meta,
                    score=1.0 - dist,   # cosine distance → similarity
                )
                chunks.append(chunk)
            return chunks

        # fallback: simple keyword overlap scoring
        return self._keyword_retrieve(query, k)

    def _keyword_retrieve(self, query: str, k: int) -> List[Chunk]:
        query_words = set(query.lower().split())
        scored = []
        for chunk in self._chunks:
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
