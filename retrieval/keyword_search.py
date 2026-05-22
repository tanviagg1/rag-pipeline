"""
KeywordSearch — BM25 keyword search over stored chunks.

Why keyword search?
  Vector search is great for semantic similarity but misses exact matches.
  If you search for "GPT-4" or "Barack Obama" or an error code like "ECONNREFUSED",
  a vector model might return semantically similar but textually different results.
  BM25 is the gold standard for exact keyword retrieval — it's what search engines
  like Elasticsearch use under the hood.

BM25 (Best Match 25):
  A ranking function that scores documents by term frequency (how often the
  search term appears) and inverse document frequency (how rare the term is
  across all documents). Terms that appear often in one doc but rarely elsewhere
  get a high score.

How this works:
  - On first search, loads all chunks from ChromaDB and builds a BM25 index
  - Index is cached in memory for subsequent searches
  - Call reset_index() to rebuild after new chunks are added

Usage:
    searcher = KeywordSearch()
    results = searcher.search("Barack Obama", top_k=5)
    for r in results:
        print(r.text, r.score)
"""

import math
from dataclasses import dataclass

from rank_bm25 import BM25Okapi
from store.vector_store import VectorStore, SearchResult


@dataclass
class KeywordResult:
    """
    A single result from BM25 keyword search.

    Mirrors SearchResult from vector search so hybrid.py can treat them uniformly.
    score here is a BM25 relevance score (higher = more relevant, unlike vector distance).
    """
    text: str
    metadata: dict
    score: float
    doc_id: str = ""

    @property
    def url(self) -> str:
        return self.metadata.get("url", "")

    @property
    def title(self) -> str:
        return self.metadata.get("title", "")


class KeywordSearch:
    """
    BM25 keyword search over all chunks in the vector store.

    Builds a BM25 index from all stored chunk texts. The index is lazy-loaded
    on first search and cached. Call reset_index() when new chunks are added.

    Args:
        store: VectorStore instance (created fresh if not provided)

    Usage:
        searcher = KeywordSearch()
        results = searcher.search("consensus algorithm", top_k=5)
    """

    def __init__(self, store: VectorStore | None = None):
        self.store = store or VectorStore()
        self._bm25: BM25Okapi | None = None
        self._docs: list[dict] | None = None  # raw docs from ChromaDB

    def search(self, query: str, top_k: int = 5) -> list[KeywordResult]:
        """
        Search for chunks containing keywords from the query.

        Args:
            query:  The search query string
            top_k:  Maximum number of results to return

        Returns:
            List of KeywordResult sorted by BM25 score (highest first).
            Returns empty list if the store is empty.
        """
        if self.store.count() == 0:
            return []

        # Build index on first call (lazy)
        if self._bm25 is None:
            self._build_index()

        # Tokenize query — lowercase, split on whitespace
        query_tokens = query.lower().split()

        # BM25 scores — one per document in the index
        scores = self._bm25.get_scores(query_tokens)

        # Pair scores with their documents, sort by score descending
        scored = sorted(
            zip(scores, self._docs),
            key=lambda x: x[0],
            reverse=True,
        )

        results = []
        for score, doc in scored[:top_k]:
            if score <= 0:
                continue  # skip irrelevant docs (BM25 score of 0)
            results.append(KeywordResult(
                text=doc["text"],
                metadata=doc["metadata"],
                score=float(score),
                doc_id=doc["id"],
            ))

        return results

    def reset_index(self) -> None:
        """
        Clear the cached BM25 index so it rebuilds on the next search.

        Call this after adding new chunks to the vector store.
        """
        self._bm25 = None
        self._docs = None

    def _build_index(self) -> None:
        """
        Load all chunks from ChromaDB and build the BM25 index.

        ChromaDB's get() returns all stored documents with their metadata.
        We tokenize each chunk text (lowercase, split on whitespace) to
        build the BM25 corpus.
        """
        # Pull all documents from ChromaDB
        raw = self.store._collection.get(include=["documents", "metadatas"])

        self._docs = [
            {"id": doc_id, "text": text, "metadata": meta}
            for doc_id, text, meta in zip(
                raw["ids"], raw["documents"], raw["metadatas"]
            )
        ]

        # BM25Okapi expects a list of token lists
        tokenized = [doc["text"].lower().split() for doc in self._docs]
        self._bm25 = BM25Okapi(tokenized)
