"""
ScrapingPipeline — scrape → clean → deduplicate → upsert into both stores.

This is the central ingestion pipeline. It ties together:
  - Scraper (static or Playwright)
  - Chunker + Embedder
  - VectorStore (ChromaDB)
  - SQLStore (SQLite)
  - Deduplication by URL hash

Why deduplication?
  Without it, running the scheduler multiple times would re-embed the same
  pages, bloating the vector store with duplicate chunks and slowing down search.
  We use a SHA256 hash of the URL as a stable identifier — if the URL is already
  in the SQL store, we skip re-ingesting unless force=True.

Usage:
    pipeline = ScrapingPipeline()
    result = pipeline.ingest("https://en.wikipedia.org/wiki/CAP_theorem")
    print(result)  # ScrapedPage or None (if skipped as duplicate)

    # Force re-ingest even if already stored
    result = pipeline.ingest(url, force=True)
"""

import hashlib
from dataclasses import dataclass

from scraper.scraper import WebScraper, ScraperError
from scraper.playwright_scraper import PlaywrightScraper
from embeddings.chunker import TextChunker
from embeddings.embedder import Embedder
from store.vector_store import VectorStore
from store.sql_store import SQLStore


@dataclass
class IngestResult:
    """
    Result of a single pipeline ingestion.

    url      — the URL that was processed
    skipped  — True if the URL was already in the store (duplicate)
    chunks   — number of chunks added to the vector store (0 if skipped)
    error    — set if ingestion failed
    """
    url: str
    skipped: bool = False
    chunks: int = 0
    error: str = ""

    @property
    def success(self) -> bool:
        return not self.error

    def __str__(self) -> str:
        if self.error:
            return f"FAILED  {self.url}: {self.error}"
        if self.skipped:
            return f"SKIPPED {self.url} (already stored)"
        return f"INGESTED {self.url} ({self.chunks} chunks)"


class ScrapingPipeline:
    """
    Full ingestion pipeline: scrape → chunk → embed → store.

    Handles both static and JS-rendered pages. Deduplicates by URL
    so re-running never creates duplicate entries.

    Args:
        vector_store:  ChromaDB store (created fresh if not provided)
        sql_store:     SQLite store (created fresh if not provided)
        embedder:      Embedder instance (created fresh if not provided)
        use_playwright: If True, use Playwright instead of BeautifulSoup
                        for JS-heavy pages. Default: False (faster).

    Usage:
        pipeline = ScrapingPipeline()
        result = pipeline.ingest("https://en.wikipedia.org/wiki/Raft_(algorithm)")
        print(result)
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        sql_store: SQLStore | None = None,
        embedder: Embedder | None = None,
        use_playwright: bool = False,
    ):
        self.vector_store = vector_store or VectorStore()
        self.sql_store = sql_store or SQLStore()
        self.embedder = embedder or Embedder()
        self.chunker = TextChunker()
        self.scraper = PlaywrightScraper() if use_playwright else WebScraper()

    def ingest(self, url: str, force: bool = False) -> IngestResult:
        """
        Scrape, embed, and store a single URL.

        If the URL already exists in the SQL store, it is skipped unless
        force=True. This prevents duplicate ingestion during scheduled runs.

        Args:
            url:   The URL to scrape and ingest
            force: If True, re-ingest even if already stored (refreshes content)

        Returns:
            IngestResult describing what happened
        """
        # Deduplication check — look up by URL in SQL store
        if not force and self._already_stored(url):
            return IngestResult(url=url, skipped=True)

        # Scrape the page
        try:
            page = self.scraper.scrape(url)
        except ScraperError as e:
            return IngestResult(url=url, error=str(e))

        # Chunk the text
        chunks = self.chunker.chunk(page.text, url=page.url, title=page.title)
        if not chunks:
            return IngestResult(url=url, error="No chunks produced — page may be empty")

        # Embed and store in ChromaDB
        embeddings = self.embedder.embed([c.text for c in chunks])
        self.vector_store.add(chunks, embeddings)

        # Store full article in SQLite
        self.sql_store.upsert(page)

        return IngestResult(url=url, chunks=len(chunks))

    def ingest_many(self, urls: list[str], force: bool = False) -> list[IngestResult]:
        """
        Ingest multiple URLs, returning a result for each.

        Continues on error — a single failed URL doesn't stop the batch.

        Args:
            urls:  List of URLs to ingest
            force: If True, re-ingest all even if already stored

        Returns:
            List of IngestResult, one per URL
        """
        results = []
        for url in urls:
            result = self.ingest(url, force=force)
            print(str(result))  # log progress
            results.append(result)
        return results

    def _already_stored(self, url: str) -> bool:
        """
        Check if a URL is already in the SQL store.

        Uses SQLStore.execute() with the URL directly — fast index lookup.
        """
        rows = self.sql_store.execute(
            "SELECT id FROM articles WHERE url = ?", (url,)
        )
        return len(rows) > 0
