"""
SQLStore — stores scraped articles in a SQLite database.

Why SQLite alongside ChromaDB?
  ChromaDB is great for semantic search ("find chunks similar to this query")
  but terrible for structured queries ("how many articles were scraped today?",
  "show all articles from wikipedia.org"). SQLite handles these naturally.

  Together they give us two retrieval modes:
    - Semantic  → ChromaDB (Phase 1, Phase 4)
    - Structured → SQLite  (Phase 2)

Schema:
    articles(id, url, title, content, source_domain, scraped_at)

  url is the primary key — re-scraping the same URL updates the existing row.

Usage:
    db = SQLStore()
    db.upsert(page)                          # store a scraped page
    articles = db.search("machine learning") # full-text keyword search
    stats = db.stats()                       # row counts and domain breakdown
"""

import sqlite3
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass

from scraper.scraper import ScrapedPage

# Default path for the SQLite database file
DEFAULT_DB_PATH = "./rag_store.db"


@dataclass
class Article:
    """A row from the articles table."""
    id: str           # MD5 hash of the URL — stable unique identifier
    url: str
    title: str
    content: str
    source_domain: str
    scraped_at: str   # ISO 8601 timestamp


class SQLStore:
    """
    SQLite store for scraped articles.

    Stores the full text of each scraped page as a structured row.
    Used by the SQL Agent (Phase 2) to answer structured questions
    that don't require semantic search.

    Args:
        db_path: Path to the SQLite database file (created if it doesn't exist)

    Usage:
        store = SQLStore()
        store.upsert(scraped_page)
        results = store.search("machine learning")
        print(store.stats())
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row  # rows act like dicts
        self._create_tables()

    def _create_tables(self) -> None:
        """Create the articles table if it doesn't exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id            TEXT PRIMARY KEY,
                url           TEXT UNIQUE NOT NULL,
                title         TEXT,
                content       TEXT,
                source_domain TEXT,
                scraped_at    TEXT
            )
        """)
        self._conn.commit()

    def upsert(self, page: ScrapedPage) -> None:
        """
        Insert or update an article in the database.

        Uses INSERT OR REPLACE so re-scraping the same URL refreshes
        the content without creating duplicate rows.

        Args:
            page: ScrapedPage from the web scraper
        """
        article_id = hashlib.md5(page.url.encode()).hexdigest()
        domain = self._extract_domain(page.url)
        scraped_at = datetime.now(timezone.utc).isoformat()

        self._conn.execute("""
            INSERT OR REPLACE INTO articles (id, url, title, content, source_domain, scraped_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (article_id, page.url, page.title, page.text, domain, scraped_at))
        self._conn.commit()

    def execute(self, sql: str, params: tuple = ()) -> list[dict]:
        """
        Execute an arbitrary SQL query and return results as a list of dicts.

        This is the method the SQL Agent calls with LLM-generated queries.
        Only SELECT statements are allowed — write operations are blocked
        to prevent the LLM from accidentally modifying data.

        Args:
            sql:    SQL query string (SELECT only)
            params: Optional parameter tuple for parameterised queries

        Returns:
            List of row dicts

        Raises:
            ValueError: If the query is not a SELECT statement
            sqlite3.Error: If the SQL is invalid
        """
        # Safety check — only allow SELECT queries from LLM-generated SQL
        if not sql.strip().upper().startswith("SELECT"):
            raise ValueError("Only SELECT queries are allowed.")

        cursor = self._conn.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]

    def search(self, keyword: str, limit: int = 10) -> list[dict]:
        """
        Simple keyword search across title and content.

        Uses SQLite LIKE — fast enough for small datasets, no FTS index needed.
        For large datasets, consider enabling SQLite FTS5.

        Args:
            keyword: Search term
            limit:   Max results to return

        Returns:
            List of matching article dicts (id, url, title, scraped_at)
        """
        like = f"%{keyword}%"
        cursor = self._conn.execute("""
            SELECT id, url, title, source_domain, scraped_at
            FROM articles
            WHERE title LIKE ? OR content LIKE ?
            LIMIT ?
        """, (like, like, limit))
        return [dict(row) for row in cursor.fetchall()]

    def stats(self) -> dict:
        """
        Return summary statistics about the stored articles.

        Used by --stats CLI command and the SQL Agent to describe the DB.
        """
        total = self._conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        domains = self._conn.execute("""
            SELECT source_domain, COUNT(*) as count
            FROM articles
            GROUP BY source_domain
            ORDER BY count DESC
        """).fetchall()

        return {
            "total_articles": total,
            "domains": [dict(row) for row in domains],
        }

    def get_schema(self) -> str:
        """
        Return the table schema as a string — given to the LLM as context
        so it can generate correct SQL queries.
        """
        return """
Table: articles
Columns:
  id            TEXT  — MD5 hash of the URL (primary key)
  url           TEXT  — full URL of the scraped page
  title         TEXT  — page title
  content       TEXT  — full body text of the page
  source_domain TEXT  — e.g. "en.wikipedia.org"
  scraped_at    TEXT  — ISO 8601 timestamp of when the page was scraped
""".strip()

    def count(self) -> int:
        """Return total number of stored articles."""
        return self._conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def _extract_domain(self, url: str) -> str:
        """Extract the domain from a URL. e.g. https://en.wikipedia.org/wiki/X → en.wikipedia.org"""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except Exception:
            return ""
