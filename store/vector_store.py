"""
VectorStore — persists and queries text embeddings using ChromaDB.

Why ChromaDB?
  Fully local, no server needed, persists to disk, and has a simple
  Python API. Perfect for a local RAG pipeline.

How it works:
  - Data is stored in a local directory (./chroma_db by default)
  - Each "collection" is a named bucket of embeddings + documents + metadata
  - add() stores chunks with their embeddings
  - query() finds the most similar chunks to a given query vector

Usage:
    store = VectorStore()
    store.add(chunks, embeddings)

    results = store.query(query_embedding, n_results=3)
    for r in results:
        print(r.text, r.score)
"""

import chromadb
from dataclasses import dataclass
from embeddings.chunker import Chunk

# Local directory where ChromaDB persists its data.
# Relative to wherever the script is run from.
DEFAULT_PERSIST_DIR = "./chroma_db"

# Default collection name — all scraped content goes here unless overridden
DEFAULT_COLLECTION = "rag_knowledge_base"


@dataclass
class SearchResult:
    """
    A single result from a vector similarity query.

    text     — the chunk text retrieved
    metadata — source URL, title, chunk index
    score    — cosine distance (lower = more similar, range 0–2)
    """
    text: str
    metadata: dict
    score: float

    @property
    def url(self) -> str:
        return self.metadata.get("url", "")

    @property
    def title(self) -> str:
        return self.metadata.get("title", "")

    def __repr__(self) -> str:
        return f"SearchResult(score={self.score:.3f}, title={self.title!r}, chars={len(self.text)})"


class VectorStore:
    """
    ChromaDB-backed vector store for RAG retrieval.

    Persists to disk so the knowledge base survives restarts.
    Supports upsert — re-adding the same chunk ID overwrites the old entry,
    so re-scraping a page updates the store without creating duplicates.

    Args:
        persist_dir:     Directory to store ChromaDB data
        collection_name: Name of the ChromaDB collection to use

    Usage:
        store = VectorStore()
        store.add(chunks, embeddings)
        results = store.query(query_vector, n_results=5)
    """

    def __init__(
        self,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION,
    ):
        # PersistentClient saves data to disk between runs
        self._client = chromadb.PersistentClient(path=persist_dir)
        # get_or_create — safe to call multiple times, never overwrites existing data
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            # cosine distance is standard for semantic similarity
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """
        Add chunks and their embeddings to the store.

        Uses upsert semantics — if a chunk_id already exists, it is updated.
        This means re-scraping a page refreshes its embeddings without duplicates.

        Args:
            chunks:     List of Chunk objects (text + metadata + chunk_id)
            embeddings: Corresponding embedding vectors (same length as chunks)
        """
        if not chunks:
            return

        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[c.metadata for c in chunks],
        )

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 5,
    ) -> list[SearchResult]:
        """
        Find the most similar chunks to a query vector.

        Args:
            query_embedding: The embedded query (from Embedder.embed_one())
            n_results:       How many results to return (default: 5)

        Returns:
            List of SearchResult objects sorted by similarity (best first).
            Score is cosine distance — lower means more similar.
        """
        # Cap n_results to the actual collection size — ChromaDB errors if you ask for more
        n = min(n_results, self.count())
        if n == 0:
            return []

        raw = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        results = []
        for text, meta, score in zip(
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        ):
            results.append(SearchResult(text=text, metadata=meta, score=score))

        return results

    def count(self) -> int:
        """Return the number of chunks currently stored."""
        return self._collection.count()

    def clear(self) -> None:
        """Delete all documents from the collection (used in tests)."""
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection.name,
            metadata={"hnsw:space": "cosine"},
        )
