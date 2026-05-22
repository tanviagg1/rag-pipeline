"""
HybridSearch — combines vector search and BM25 keyword search using RRF.

Why hybrid?
  - Vector search alone misses exact keyword matches ("ECONNREFUSED", "GPT-4")
  - Keyword search alone misses semantic matches ("car" vs "automobile")
  - Hybrid combines both, outperforming either method alone on most queries

Reciprocal Rank Fusion (RRF):
  The standard algorithm for merging ranked lists. For each document, its
  RRF score is the sum of 1/(k + rank) across all lists it appears in.
  k=60 is the standard constant — it dampens the influence of very top-ranked
  items and gives more weight to documents that appear in multiple lists.

  Example with k=60:
    Rank 1 → 1/(60+1) = 0.0164
    Rank 5 → 1/(60+5) = 0.0154
    A doc at rank 3 in both lists → 2 × 1/(60+3) = 0.0317

  A document ranking in the top 5 of both vector AND keyword search
  gets a higher combined score than one that ranks #1 in only one list.

Usage:
    searcher = HybridSearch()
    results = searcher.search("consensus algorithm leader election", top_k=5)
    for r in results:
        print(r.text)
        print(f"  RRF score: {r.rrf_score:.4f} | in_vector: {r.in_vector} | in_keyword: {r.in_keyword}")
"""

from dataclasses import dataclass, field
from retrieval.vector_search import VectorSearch
from retrieval.keyword_search import KeywordSearch

# RRF constant — dampens top-rank dominance. k=60 is the standard default.
RRF_K = 60


@dataclass
class HybridResult:
    """
    A single result from hybrid search.

    Contains the chunk text, metadata, and fusion scores that show
    exactly where the result came from.
    """
    text: str
    metadata: dict
    rrf_score: float         # combined Reciprocal Rank Fusion score (higher = better)
    in_vector: bool = False  # appeared in vector search results
    in_keyword: bool = False # appeared in keyword search results
    vector_rank: int | None = None
    keyword_rank: int | None = None

    @property
    def url(self) -> str:
        return self.metadata.get("url", "")

    @property
    def title(self) -> str:
        return self.metadata.get("title", "")

    def source_label(self) -> str:
        """Human-readable label showing which retrieval methods found this chunk."""
        sources = []
        if self.in_vector:
            sources.append(f"vector(#{self.vector_rank})")
        if self.in_keyword:
            sources.append(f"keyword(#{self.keyword_rank})")
        return " + ".join(sources) if sources else "unknown"


class HybridSearch:
    """
    Hybrid retrieval combining vector similarity and BM25 keyword search via RRF.

    Runs both searches in parallel, then merges results using Reciprocal Rank
    Fusion. Documents appearing in both ranked lists get a higher combined score.

    Args:
        vector_search:  VectorSearch instance (created fresh if not provided)
        keyword_search: KeywordSearch instance (created fresh if not provided)
        rrf_k:          RRF constant (default: 60)

    Usage:
        searcher = HybridSearch()
        results = searcher.search("distributed consensus", top_k=5)
    """

    def __init__(
        self,
        vector_search: VectorSearch | None = None,
        keyword_search: KeywordSearch | None = None,
        rrf_k: int = RRF_K,
    ):
        self.vector_search = vector_search or VectorSearch()
        self.keyword_search = keyword_search or KeywordSearch()
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 5) -> list[HybridResult]:
        """
        Run hybrid search: vector + keyword → RRF merge → top-k results.

        Args:
            query:  The search query string
            top_k:  Number of results to return after fusion

        Returns:
            List of HybridResult sorted by RRF score (highest first)
        """
        # Fetch more candidates than top_k from each source so the
        # fusion has enough material to work with
        fetch_k = max(top_k * 3, 20)

        vector_results = self.vector_search.search(query, top_k=fetch_k)
        keyword_results = self.keyword_search.search(query, top_k=fetch_k)

        return self._fuse(vector_results, keyword_results, top_k)

    def _fuse(
        self,
        vector_results: list,
        keyword_results: list,
        top_k: int,
    ) -> list[HybridResult]:
        """
        Merge two ranked lists using Reciprocal Rank Fusion.

        Uses chunk text as the deduplication key — a chunk appearing in both
        lists gets contributions from both, boosting its combined score.

        Args:
            vector_results:  Results from VectorSearch (ordered best-first)
            keyword_results: Results from KeywordSearch (ordered best-first)
            top_k:           How many final results to return

        Returns:
            Top-k HybridResult objects sorted by RRF score descending
        """
        # rrf_scores: text → running RRF score
        # doc_map: text → HybridResult (for building the final list)
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, HybridResult] = {}

        # Process vector results (rank 1 = best = lowest distance)
        for rank, result in enumerate(vector_results, start=1):
            key = result.text
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank)
            if key not in doc_map:
                doc_map[key] = HybridResult(
                    text=result.text,
                    metadata=result.metadata,
                    rrf_score=0.0,
                    in_vector=True,
                    vector_rank=rank,
                )
            else:
                doc_map[key].in_vector = True
                doc_map[key].vector_rank = rank

        # Process keyword results (rank 1 = best = highest BM25 score)
        for rank, result in enumerate(keyword_results, start=1):
            key = result.text
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank)
            if key not in doc_map:
                doc_map[key] = HybridResult(
                    text=result.text,
                    metadata=result.metadata,
                    rrf_score=0.0,
                    in_keyword=True,
                    keyword_rank=rank,
                )
            else:
                doc_map[key].in_keyword = True
                doc_map[key].keyword_rank = rank

        # Assign final RRF scores and sort
        for key, score in rrf_scores.items():
            doc_map[key].rrf_score = score

        sorted_results = sorted(
            doc_map.values(),
            key=lambda r: r.rrf_score,
            reverse=True,
        )

        return sorted_results[:top_k]
