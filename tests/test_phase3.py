"""
Tests for Phase 3 — PlaywrightScraper, ScrapingPipeline, Scheduler.

Unit tests mock all external calls (Playwright, network, ChromaDB, SQLite).

Run:
    pytest tests/test_phase3.py -v -m "not integration"
"""

import pytest
from unittest.mock import MagicMock, patch

from scraper.scraper import ScrapedPage, ScraperError
from scraper.pipeline import ScrapingPipeline, IngestResult
from scraper.scheduler import Scheduler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_page():
    return ScrapedPage(
        url="https://en.wikipedia.org/wiki/Raft_(algorithm)",
        title="Raft algorithm",
        text="Raft is a consensus algorithm. " * 30,  # enough text to produce chunks
    )


def make_mock_pipeline(sample_page: ScrapedPage) -> ScrapingPipeline:
    """Build a ScrapingPipeline with all external dependencies mocked."""
    pipeline = ScrapingPipeline.__new__(ScrapingPipeline)
    pipeline.scraper = MagicMock()
    pipeline.scraper.scrape.return_value = sample_page
    pipeline.chunker = MagicMock()
    pipeline.chunker.chunk.return_value = [
        MagicMock(text="chunk 1", chunk_id="url#0", metadata={}),
        MagicMock(text="chunk 2", chunk_id="url#1", metadata={}),
    ]
    pipeline.embedder = MagicMock()
    pipeline.embedder.embed.return_value = [[0.1] * 384, [0.2] * 384]
    pipeline.vector_store = MagicMock()
    pipeline.vector_store.add = MagicMock()
    pipeline.sql_store = MagicMock()
    pipeline.sql_store.execute.return_value = []  # not already stored
    pipeline.sql_store.upsert = MagicMock()
    return pipeline


# ---------------------------------------------------------------------------
# PlaywrightScraper tests (mocked Playwright)
# ---------------------------------------------------------------------------

class TestPlaywrightScraper:

    def test_scrape_returns_scraped_page(self):
        from scraper.playwright_scraper import PlaywrightScraper

        scraper = PlaywrightScraper()
        mock_html = """
        <html><head><title>Test Page</title></head>
        <body><p>This is a long enough paragraph to pass the length filter for testing purposes here.</p></body>
        </html>"""

        with patch("scraper.playwright_scraper.sync_playwright") as mock_pw:
            mock_browser = MagicMock()
            mock_page = MagicMock()
            mock_page.content.return_value = mock_html
            mock_browser.new_page.return_value = mock_page
            mock_pw.return_value.__enter__.return_value.chromium.launch.return_value = mock_browser

            result = scraper.scrape("https://example.com")

        assert result.title == "Test Page"
        assert "paragraph" in result.text
        assert result.url == "https://example.com"

    def test_scrape_raises_on_playwright_error(self):
        from scraper.playwright_scraper import PlaywrightScraper

        scraper = PlaywrightScraper()
        with patch("scraper.playwright_scraper.sync_playwright") as mock_pw:
            mock_pw.return_value.__enter__.side_effect = Exception("Browser crash")
            with pytest.raises(ScraperError):
                scraper.scrape("https://example.com")


# ---------------------------------------------------------------------------
# ScrapingPipeline tests
# ---------------------------------------------------------------------------

class TestScrapingPipeline:

    def test_ingest_new_url_succeeds(self, sample_page):
        pipeline = make_mock_pipeline(sample_page)
        result = pipeline.ingest("https://en.wikipedia.org/wiki/Raft_(algorithm)")

        assert result.success
        assert not result.skipped
        assert result.chunks == 2

    def test_ingest_stores_in_both_stores(self, sample_page):
        pipeline = make_mock_pipeline(sample_page)
        pipeline.ingest("https://example.com")

        pipeline.vector_store.add.assert_called_once()
        pipeline.sql_store.upsert.assert_called_once()

    def test_ingest_skips_duplicate_url(self, sample_page):
        pipeline = make_mock_pipeline(sample_page)
        # Simulate URL already in SQL store
        pipeline.sql_store.execute.return_value = [{"id": "abc123"}]

        result = pipeline.ingest("https://example.com")

        assert result.skipped
        pipeline.scraper.scrape.assert_not_called()  # should not scrape

    def test_ingest_force_overrides_dedup(self, sample_page):
        pipeline = make_mock_pipeline(sample_page)
        # Simulate URL already stored
        pipeline.sql_store.execute.return_value = [{"id": "abc123"}]

        result = pipeline.ingest("https://example.com", force=True)

        # force=True should bypass dedup and scrape anyway
        assert not result.skipped
        pipeline.scraper.scrape.assert_called_once()

    def test_ingest_handles_scraper_error(self, sample_page):
        pipeline = make_mock_pipeline(sample_page)
        pipeline.scraper.scrape.side_effect = ScraperError("404 Not Found")

        result = pipeline.ingest("https://example.com/missing")

        assert not result.success
        assert "404" in result.error

    def test_ingest_many_returns_result_per_url(self, sample_page):
        pipeline = make_mock_pipeline(sample_page)
        urls = ["https://example.com/1", "https://example.com/2"]
        results = pipeline.ingest_many(urls)

        assert len(results) == 2

    def test_ingest_many_continues_after_error(self, sample_page):
        pipeline = make_mock_pipeline(sample_page)
        # First URL fails, second succeeds
        pipeline.scraper.scrape.side_effect = [
            ScraperError("failed"),
            sample_page,
        ]
        results = pipeline.ingest_many(["https://fail.com", "https://ok.com"])

        assert results[0].error != ""
        assert results[1].success

    def test_ingest_result_str_ingested(self, sample_page):
        pipeline = make_mock_pipeline(sample_page)
        result = pipeline.ingest("https://example.com")
        assert "INGESTED" in str(result)

    def test_ingest_result_str_skipped(self, sample_page):
        pipeline = make_mock_pipeline(sample_page)
        pipeline.sql_store.execute.return_value = [{"id": "abc"}]
        result = pipeline.ingest("https://example.com")
        assert "SKIPPED" in str(result)

    def test_ingest_result_str_failed(self, sample_page):
        pipeline = make_mock_pipeline(sample_page)
        pipeline.scraper.scrape.side_effect = ScraperError("network error")
        result = pipeline.ingest("https://example.com")
        assert "FAILED" in str(result)


# ---------------------------------------------------------------------------
# Scheduler tests
# ---------------------------------------------------------------------------

class TestScheduler:

    def test_run_once_calls_ingest_many(self, sample_page):
        pipeline = make_mock_pipeline(sample_page)
        pipeline.ingest_many = MagicMock(return_value=[
            IngestResult(url="https://example.com", chunks=3)
        ])

        scheduler = Scheduler(
            urls=["https://example.com"],
            pipeline=pipeline,
        )
        results = scheduler.run_once()

        pipeline.ingest_many.assert_called_once_with(
            ["https://example.com"], force=False
        )
        assert len(results) == 1

    def test_run_once_passes_force_flag(self, sample_page):
        pipeline = make_mock_pipeline(sample_page)
        pipeline.ingest_many = MagicMock(return_value=[])

        scheduler = Scheduler(
            urls=["https://example.com"],
            pipeline=pipeline,
            force_refresh=True,
        )
        scheduler.run_once()

        pipeline.ingest_many.assert_called_once_with(
            ["https://example.com"], force=True
        )
