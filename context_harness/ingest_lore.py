"""
HP Lore Ingestion
=================
Parses data/hp_lore.txt, ingests each document into ChromaDB via
DocumentRegistry, and reports a summary.

Run:
    python -m context_harness.ingest_lore
"""

from __future__ import annotations

import re
from pathlib import Path

from .rag_pipeline import RAGPipeline, ChunkingStrategy, SourceType
from .document_registry import DocumentRegistry


LORE_FILE = Path(__file__).parent.parent / "data" / "hp_lore.txt"
DB_PATH   = Path(__file__).parent.parent / "data" / "doc_registry.db"
CHROMA_PATH = str(Path(__file__).parent.parent / "data" / "chromadb")


def parse_lore_file(path: Path) -> list[tuple[str, str]]:
    """
    Parse a lore file where each document is preceded by a 'doc_id: <id>' line.
    Returns a list of (doc_id, text) tuples.
    """
    docs: list[tuple[str, str]] = []
    current_id: str | None = None
    current_lines: list[str] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^doc_id:\s*(.+)$", line.strip())
        if m:
            if current_id and current_lines:
                docs.append((current_id, "\n".join(current_lines).strip()))
            current_id = m.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_id and current_lines:
        docs.append((current_id, "\n".join(current_lines).strip()))

    return docs


def build_pipeline(persist: bool = True) -> RAGPipeline:
    """Build a RAGPipeline backed by a real ChromaDB store."""
    import chromadb
    from chromadb import Settings
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    settings = Settings(anonymized_telemetry=False)
    client = (
        chromadb.PersistentClient(path=CHROMA_PATH, settings=settings)
        if persist
        else chromadb.EphemeralClient(settings=settings)
    )
    ef = DefaultEmbeddingFunction()

    # Build with use_chromadb=False so __init__ doesn't try to connect;
    # we wire in our own persistent client directly.
    pipeline = RAGPipeline(
        collection_name="hp_lore",
        chunking_strategy=ChunkingStrategy.PARAGRAPH,
        top_k=5,
        use_chromadb=False,
    )
    pipeline._client = client
    pipeline._ef = ef
    pipeline._collection = client.get_or_create_collection(
        name="hp_lore",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return pipeline


def ingest(verbose: bool = True, persist: bool = True) -> DocumentRegistry:
    docs = parse_lore_file(LORE_FILE)
    pipeline = build_pipeline(persist=persist)
    registry = DocumentRegistry(
        pipeline=pipeline,
        db_path=str(DB_PATH) if persist else ":memory:",
        embedding_model="all-MiniLM-L6-v2",
    )

    for doc_id, text in docs:
        result = registry.upsert(
            doc_id, text,
            universe="hp",
            source_type=SourceType.FICTIONAL_CANON,
            source="hp_lore.txt",
        )
        if verbose:
            print(f"  [{result.action:8s}] {doc_id} — {result.chunk_count} chunks")

    stats = registry.stats()
    if verbose:
        print(f"\nRegistry : {stats['doc_count']} docs, {stats['total_chunks']} total chunks")
        print(f"ChromaDB : {pipeline.count()} vectors indexed")

    return registry


if __name__ == "__main__":
    print("Ingesting HP lore into ChromaDB...\n")
    ingest()
    print("\nDone.")
