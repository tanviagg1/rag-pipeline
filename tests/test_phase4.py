"""
Tests for Phase 4 — VectorSearch, KeywordSearch, HybridSearch (RRF).

All external dependencies mocked — no real ChromaDB or embeddings needed.

Run:
    pytest tests/test_phase4.py -v -m "not integration"
"""

import pytest
from unittest.mock import MagicMock, patch

from store.vector_store import SearchResult
from retrieval.vector_search import VectorSearch
from retrieval.keyword_search import KeywordSearch, KeywordResult
from retrieval.hybrid import HybridSearch, HybridResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_search_result(text: str, score: float = 0.1, url: str = "https://x.com") -> SearchResult:
    return SearchResult(
        text=text,
        metadata={"url": url, "title": "Test", "chunk_index": 0},
        score=score,
    )

def make_keyword_result(text: str, score: float = 1.0, url: str = "https://x.com") -> KeywordResult:
    return KeywordResult(
        text=text,
        metadata={"url": url, "title": "Test", "chunk_index": 0},
        score=score,
    )


# ---------------------------------------------------------------------------
# VectorSearch tests
# ---------------------------------------------------------------------------

class TestVectorSearch:

    def test_search_returns_results(self):
        searcher = VectorSearch()
        searcher.embedder = MagicMock()
        searcher.embedder.embed_one.return_value = [0.1] * 384
        searcher.store = MagicMock()
        searcher.store.query.return_value = [
            make_search_result("The CAP theorem text", score=0.1)
        ]

        results = searcher.search("CAP theorem", top_k=3)

        assert len(results) == 1
        assert results[0].text == "The CAP theorem text"

    def test_search_passes_top_k_to_store(self):
        searcher = VectorSearch()
        searcher.embedder = MagicMock()
        searcher.embedder.embed_one.return_value = [0.1] * 384
        searcher.store = MagicMock()
        searcher.store.query.return_value = []

        searcher.search("query", top_k=7)

        searcher.store.query.assert_called_once_with([0.1] * 384, n_results=7)

    def test_search_embeds_query(self):
        searcher = VectorSearch()
        searcher.embedder = MagicMock()
        searcher.embedder.embed_one.return_value = [0.5] * 384
        searcher.store = MagicMock()
        searcher.store.query.return_value = []

        searcher.search("my query")

        searcher.embedder.embed_one.assert_called_once_with("my query")


# ---------------------------------------------------------------------------
# KeywordSearch tests
# ---------------------------------------------------------------------------

class TestKeywordSearch:

    def _make_searcher_with_docs(self, texts: list[str]) -> KeywordSearch:
        """Build a KeywordSearch with mocked ChromaDB returning given texts."""
        searcher = KeywordSearch()
        searcher.store = MagicMock()
        searcher.store.count.return_value = len(texts)
        searcher.store._collection = MagicMock()
        searcher.store._collection.get.return_value = {
            "ids": [f"chunk#{i}" for i in range(len(texts))],
            "documents": texts,
            "metadatas": [{"url": f"https://x.com/{i}", "title": f"Doc {i}"} for i in range(len(texts))],
        }
        return searcher

    def test_search_returns_relevant_results(self):
        searcher = self._make_searcher_with_docs([
            "The CAP theorem is about distributed systems",
            "Python is a programming language",
            "Consistency and availability in distributed databases",
        ])
        results = searcher.search("CAP theorem distributed")

        assert len(results) > 0
        # The CAP theorem doc should score higher than Python
        assert results[0].text != "Python is a programming language"

    def test_search_returns_empty_for_empty_store(self):
        searcher = KeywordSearch()
        searcher.store = MagicMock()
        searcher.store.count.return_value = 0

        results = searcher.search("anything")
        assert results == []

    def test_search_skips_zero_score_docs(self):
        searcher = self._make_searcher_with_docs([
            "The CAP theorem",
            "completely unrelated topic about cooking recipes",
        ])
        results = searcher.search("CAP theorem")
        # Only docs with score > 0 should be returned
        assert all(r.score > 0 for r in results)

    def test_reset_index_clears_cache(self):
        searcher = self._make_searcher_with_docs(["some text"])
        searcher.search("text")  # builds index
        assert searcher._bm25 is not None

        searcher.reset_index()
        assert searcher._bm25 is None

    def test_search_respects_top_k(self):
        texts = [f"document number {i} about topic" for i in range(20)]
        searcher = self._make_searcher_with_docs(texts)
        results = searcher.search("document topic", top_k=3)
        assert len(results) <= 3


# ---------------------------------------------------------------------------
# HybridSearch (RRF) tests
# ---------------------------------------------------------------------------

class TestHybridSearch:

    def _make_hybrid(
        self,
        vector_results: list,
        keyword_results: list,
    ) -> HybridSearch:
        """Build a HybridSearch with mocked sub-searchers."""
        vector_search = MagicMock()
        vector_search.search.return_value = vector_results
        keyword_search = MagicMock()
        keyword_search.search.return_value = keyword_results

        return HybridSearch(
            vector_search=vector_search,
            keyword_search=keyword_search,
        )

    def test_search_returns_hybrid_results(self):
        hybrid = self._make_hybrid(
            vector_results=[make_search_result("doc A"), make_search_result("doc B")],
            keyword_results=[make_keyword_result("doc B"), make_keyword_result("doc C")],
        )
        results = hybrid.search("query", top_k=5)

        assert len(results) > 0
        assert all(isinstance(r, HybridResult) for r in results)

    def test_doc_in_both_lists_gets_higher_score(self):
        """A doc appearing in both vector and keyword results should rank higher."""
        shared_text = "shared document about consensus"
        vector_only = "only in vector results"
        keyword_only = "only in keyword results"

        hybrid = self._make_hybrid(
            vector_results=[
                make_search_result(shared_text, score=0.1),
                make_search_result(vector_only, score=0.2),
            ],
            keyword_results=[
                make_keyword_result(shared_text, score=5.0),
                make_keyword_result(keyword_only, score=4.0),
            ],
        )
        results = hybrid.search("consensus", top_k=5)

        # The shared doc should be ranked first
        assert results[0].text == shared_text
        assert results[0].in_vector is True
        assert results[0].in_keyword is True

    def test_results_sorted_by_rrf_score_descending(self):
        hybrid = self._make_hybrid(
            vector_results=[make_search_result(f"doc {i}") for i in range(5)],
            keyword_results=[make_keyword_result(f"doc {i}") for i in range(3, 8)],
        )
        results = hybrid.search("query", top_k=10)

        scores = [r.rrf_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limits_results(self):
        hybrid = self._make_hybrid(
            vector_results=[make_search_result(f"doc {i}") for i in range(10)],
            keyword_results=[make_keyword_result(f"doc {i}") for i in range(10)],
        )
        results = hybrid.search("query", top_k=3)
        assert len(results) <= 3

    def test_source_label_shows_both_sources(self):
        shared = "shared content"
        hybrid = self._make_hybrid(
            vector_results=[make_search_result(shared)],
            keyword_results=[make_keyword_result(shared)],
        )
        results = hybrid.search("query", top_k=5)

        shared_result = next(r for r in results if r.text == shared)
        label = shared_result.source_label()
        assert "vector" in label
        assert "keyword" in label

    def test_rrf_scores_are_positive(self):
        hybrid = self._make_hybrid(
            vector_results=[make_search_result("doc A")],
            keyword_results=[make_keyword_result("doc B")],
        )
        results = hybrid.search("query", top_k=5)
        assert all(r.rrf_score > 0 for r in results)

    def test_empty_results_from_both_returns_empty(self):
        hybrid = self._make_hybrid(vector_results=[], keyword_results=[])
        results = hybrid.search("query")
        assert results == []
