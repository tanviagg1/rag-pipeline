"""
Tests for Phase 1 — scraper, chunker, embedder, vector store, RAG query.

Unit tests mock external dependencies (requests, ChromaDB, Ollama, sentence-transformers)
so no real network or model calls are needed.

Run:
    pytest tests/test_phase1.py -v -m "not integration"
"""

import pytest
from unittest.mock import MagicMock, patch

from scraper.scraper import WebScraper, ScraperError, ScrapedPage
from embeddings.chunker import TextChunker, Chunk
from embeddings.embedder import Embedder
from store.vector_store import VectorStore, SearchResult
from llm.rag import RAGQuery, RAGResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_HTML = """
<html>
<head><title>CAP Theorem</title></head>
<body>
<nav>Navigation stuff</nav>
<p>The CAP theorem states that a distributed system cannot simultaneously guarantee
consistency, availability, and partition tolerance.</p>
<p>Consistency means every read receives the most recent write.
Availability means every request receives a response.
Partition tolerance means the system continues operating despite network failures.</p>
<p>According to CAP, a system can only guarantee two of the three properties at once.</p>
</body>
</html>
"""

SAMPLE_TEXT = (
    "The CAP theorem states that a distributed system cannot simultaneously guarantee "
    "consistency, availability, and partition tolerance. "
    "Consistency means every read receives the most recent write. "
    "Availability means every request receives a response. "
    "Partition tolerance means the system continues operating despite network failures. "
    "According to CAP, a system can only guarantee two of the three properties at once."
)


# ---------------------------------------------------------------------------
# WebScraper tests
# ---------------------------------------------------------------------------

class TestWebScraper:

    def test_scrapes_title_from_title_tag(self):
        scraper = WebScraper()
        with patch("scraper.scraper.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200, text=SAMPLE_HTML)
            mock_get.return_value.raise_for_status = MagicMock()
            page = scraper.scrape("https://example.com/cap")
        assert page.title == "CAP Theorem"

    def test_scrapes_paragraph_text(self):
        scraper = WebScraper()
        with patch("scraper.scraper.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200, text=SAMPLE_HTML)
            mock_get.return_value.raise_for_status = MagicMock()
            page = scraper.scrape("https://example.com/cap")
        assert "CAP theorem" in page.text
        assert "consistency" in page.text

    def test_removes_nav_boilerplate(self):
        scraper = WebScraper()
        with patch("scraper.scraper.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200, text=SAMPLE_HTML)
            mock_get.return_value.raise_for_status = MagicMock()
            page = scraper.scrape("https://example.com/cap")
        assert "Navigation stuff" not in page.text

    def test_stores_url_in_result(self):
        scraper = WebScraper()
        with patch("scraper.scraper.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200, text=SAMPLE_HTML)
            mock_get.return_value.raise_for_status = MagicMock()
            page = scraper.scrape("https://example.com/cap")
        assert page.url == "https://example.com/cap"

    def test_raises_scraper_error_on_http_error(self):
        scraper = WebScraper()
        with patch("scraper.scraper.requests.get") as mock_get:
            import requests as req
            mock_resp = MagicMock()
            mock_resp.status_code = 404
            mock_get.return_value.raise_for_status.side_effect = req.HTTPError(
                response=mock_resp
            )
            with pytest.raises(ScraperError):
                scraper.scrape("https://example.com/notfound")


# ---------------------------------------------------------------------------
# TextChunker tests
# ---------------------------------------------------------------------------

class TestTextChunker:

    def test_returns_chunks_for_long_text(self):
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(SAMPLE_TEXT, url="https://x.com", title="Test")
        assert len(chunks) > 1

    def test_short_text_returns_single_chunk(self):
        chunker = TextChunker(chunk_size=1000)
        chunks = chunker.chunk("Short text.", url="https://x.com", title="Test")
        assert len(chunks) == 1

    def test_empty_text_returns_no_chunks(self):
        chunker = TextChunker()
        chunks = chunker.chunk("", url="https://x.com", title="Test")
        assert chunks == []

    def test_chunks_have_unique_ids(self):
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(SAMPLE_TEXT, url="https://x.com", title="Test")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))  # all unique

    def test_chunk_metadata_contains_url(self):
        chunker = TextChunker()
        chunks = chunker.chunk(SAMPLE_TEXT, url="https://x.com", title="Test")
        assert all(c.metadata["url"] == "https://x.com" for c in chunks)

    def test_chunk_id_contains_url(self):
        chunker = TextChunker()
        chunks = chunker.chunk(SAMPLE_TEXT, url="https://x.com", title="Test")
        assert all("https://x.com" in c.chunk_id for c in chunks)

    def test_no_empty_chunks(self):
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(SAMPLE_TEXT, url="https://x.com", title="Test")
        assert all(len(c.text) > 0 for c in chunks)


# ---------------------------------------------------------------------------
# Embedder tests (mocked — no real model loaded)
# ---------------------------------------------------------------------------

class TestEmbedder:

    def test_embed_returns_list_of_vectors(self):
        embedder = Embedder()
        # Mock the underlying model
        embedder._model = MagicMock()
        import numpy as np
        embedder._model.encode.return_value = np.array([[0.1] * 384, [0.2] * 384])

        result = embedder.embed(["text one", "text two"])
        assert len(result) == 2
        assert len(result[0]) == 384

    def test_embed_one_returns_single_vector(self):
        embedder = Embedder()
        embedder._model = MagicMock()
        import numpy as np
        embedder._model.encode.return_value = np.array([[0.5] * 384])

        result = embedder.embed_one("single text")
        assert len(result) == 384

    def test_embed_empty_list_returns_empty(self):
        embedder = Embedder()
        result = embedder.embed([])
        assert result == []


# ---------------------------------------------------------------------------
# VectorStore tests (mocked ChromaDB)
# ---------------------------------------------------------------------------

class TestVectorStore:

    def test_add_calls_upsert(self):
        store = VectorStore.__new__(VectorStore)
        store._collection = MagicMock()
        store._client = MagicMock()

        chunks = [Chunk(text="hello", metadata={"url": "x", "title": "t", "chunk_index": 0}, chunk_id="x#0")]
        embeddings = [[0.1] * 384]
        store.add(chunks, embeddings)

        store._collection.upsert.assert_called_once()

    def test_query_returns_search_results(self):
        store = VectorStore.__new__(VectorStore)
        store._collection = MagicMock()
        store._collection.count.return_value = 3
        store._collection.query.return_value = {
            "documents": [["chunk text"]],
            "metadatas": [[{"url": "https://x.com", "title": "Test", "chunk_index": 0}]],
            "distances": [[0.15]],
        }

        results = store.query([0.1] * 384, n_results=1)
        assert len(results) == 1
        assert results[0].text == "chunk text"
        assert results[0].score == 0.15

    def test_query_returns_empty_when_store_empty(self):
        store = VectorStore.__new__(VectorStore)
        store._collection = MagicMock()
        store._collection.count.return_value = 0

        results = store.query([0.1] * 384)
        assert results == []

    def test_add_empty_chunks_is_noop(self):
        store = VectorStore.__new__(VectorStore)
        store._collection = MagicMock()
        store.add([], [])
        store._collection.upsert.assert_not_called()


# ---------------------------------------------------------------------------
# RAGQuery tests (mocked embedder + store + ollama)
# ---------------------------------------------------------------------------

class TestRAGQuery:

    def _make_rag(self) -> RAGQuery:
        """Build a RAGQuery with all dependencies mocked."""
        embedder = MagicMock()
        embedder.embed_one.return_value = [0.1] * 384

        store = MagicMock()
        store.query.return_value = [
            SearchResult(
                text="The CAP theorem states distributed systems trade-offs.",
                metadata={"url": "https://en.wikipedia.org/wiki/CAP_theorem", "title": "CAP theorem"},
                score=0.12,
            )
        ]

        rag = RAGQuery(embedder=embedder, store=store)
        return rag

    def test_query_returns_rag_result(self):
        rag = self._make_rag()
        with patch("llm.rag.ollama.chat") as mock_chat:
            mock_chat.return_value = MagicMock(message=MagicMock(content="CAP theorem is about trade-offs."))
            result = rag.query("What is the CAP theorem?")
        assert isinstance(result, RAGResult)
        assert "CAP" in result.answer

    def test_query_returns_sources(self):
        rag = self._make_rag()
        with patch("llm.rag.ollama.chat") as mock_chat:
            mock_chat.return_value = MagicMock(message=MagicMock(content="Answer."))
            result = rag.query("What is CAP?")
        assert len(result.sources) == 1
        assert "CAP_theorem" in result.sources[0].url

    def test_empty_store_returns_no_kb_message(self):
        embedder = MagicMock()
        embedder.embed_one.return_value = [0.1] * 384
        store = MagicMock()
        store.query.return_value = []  # empty store

        rag = RAGQuery(embedder=embedder, store=store)
        result = rag.query("What is CAP?")
        assert "knowledge base" in result.answer.lower()

    def test_raises_on_empty_question(self):
        rag = self._make_rag()
        with pytest.raises(ValueError):
            rag.query("   ")

    def test_result_str_contains_question(self):
        rag = self._make_rag()
        with patch("llm.rag.ollama.chat") as mock_chat:
            mock_chat.return_value = MagicMock(message=MagicMock(content="Answer."))
            result = rag.query("What is the CAP theorem?")
        assert "What is the CAP theorem?" in str(result)


# ---------------------------------------------------------------------------
# Integration tests — require network + Ollama + sentence-transformers
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestPhase1Integration:
    """
    Full end-to-end test. Requires:
    - Internet access (to scrape Wikipedia)
    - Ollama running with llama3.1:8b
    - sentence-transformers installed

    Run: pytest tests/test_phase1.py -v -m integration
    """

    def test_scrape_wikipedia_page(self):
        scraper = WebScraper()
        page = scraper.scrape("https://en.wikipedia.org/wiki/CAP_theorem")
        assert page.title != ""
        assert len(page.text) > 500

    def test_full_rag_pipeline(self, tmp_path):
        from embeddings.chunker import TextChunker

        scraper = WebScraper()
        chunker = TextChunker()
        embedder = Embedder()
        store = VectorStore(persist_dir=str(tmp_path / "chroma"))
        rag = RAGQuery(embedder=embedder, store=store)

        # Scrape
        page = scraper.scrape("https://en.wikipedia.org/wiki/CAP_theorem")
        chunks = chunker.chunk(page.text, url=page.url, title=page.title)
        embeddings = embedder.embed([c.text for c in chunks])
        store.add(chunks, embeddings)

        # Query
        result = rag.query("What does CAP stand for?")
        assert len(result.answer) > 10
        assert len(result.sources) > 0
