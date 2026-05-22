"""
VectorSearch — semantic similarity search using ChromaDB embeddings.

This wraps VectorStore.query() with a consistent SearchResult interface
so it can be combined with keyword search in the hybrid retriever.

Why semantic search?
  Finds conceptually similar content even when exact words don't match.
  "automobile" matches "car", "distributed consensus" matches "leader election".
  Great for natural language questions but misses exact terms (names, IDs, codes).

Usage:
    searcher = VectorSearch()
    results = searcher.search("What is eventual consistency?", top_k=5)
    for r in results:
        print(r.text, r.score)
"""

from dataclasses import dataclass
from embeddings.embedder import Embedder
from store.vector_store import VectorStore, SearchResult


class VectorSearch:
    """
    Semantic search over the ChromaDB vector store.

    Embeds the query and finds the most similar stored chunks
    using cosine similarity.

    Args:
        embedder: Embedder instance (created fresh if not provided)
        store:    VectorStore instance (created fresh if not provided)

    Usage:
        searcher = VectorSearch()
        results = searcher.search("distributed consensus algorithms", top_k=5)
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
    ):
        self.embedder = embedder or Embedder()
        self.store = store or VectorStore()

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """
        Embed the query and retrieve the top-k semantically similar chunks.

        Args:
            query:  The search query string
            top_k:  Number of results to return

        Returns:
            List of SearchResult sorted by similarity (best = lowest distance first)
        """
        query_embedding = self.embedder.embed_one(query)
        return self.store.query(query_embedding, n_results=top_k)
