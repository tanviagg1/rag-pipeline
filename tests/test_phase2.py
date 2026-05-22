"""
Tests for Phase 2 — SQLStore and SQLAgent.

Unit tests use a temp SQLite file and mock Ollama calls.
No real database or LLM needed.

Run:
    pytest tests/test_phase2.py -v -m "not integration"
"""

import pytest
import tempfile
import os
from unittest.mock import MagicMock, patch

from scraper.scraper import ScrapedPage
from store.sql_store import SQLStore
from agent.sql_agent import SQLAgent, SQLResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_store(tmp_path):
    """SQLStore backed by a temp file — deleted after each test."""
    store = SQLStore(db_path=str(tmp_path / "test.db"))
    yield store
    store.close()


@pytest.fixture
def sample_page():
    return ScrapedPage(
        url="https://en.wikipedia.org/wiki/CAP_theorem",
        title="CAP theorem",
        text="The CAP theorem states that a distributed system cannot simultaneously provide all three of consistency, availability, and partition tolerance.",
    )


@pytest.fixture
def sample_page_2():
    return ScrapedPage(
        url="https://en.wikipedia.org/wiki/Raft_(algorithm)",
        title="Raft algorithm",
        text="Raft is a consensus algorithm designed as an alternative to Paxos.",
    )


# ---------------------------------------------------------------------------
# SQLStore tests
# ---------------------------------------------------------------------------

class TestSQLStore:

    def test_upsert_stores_article(self, tmp_store, sample_page):
        tmp_store.upsert(sample_page)
        assert tmp_store.count() == 1

    def test_upsert_same_url_does_not_duplicate(self, tmp_store, sample_page):
        tmp_store.upsert(sample_page)
        tmp_store.upsert(sample_page)  # same URL
        assert tmp_store.count() == 1

    def test_upsert_different_urls_creates_two_rows(self, tmp_store, sample_page, sample_page_2):
        tmp_store.upsert(sample_page)
        tmp_store.upsert(sample_page_2)
        assert tmp_store.count() == 2

    def test_execute_select_returns_rows(self, tmp_store, sample_page):
        tmp_store.upsert(sample_page)
        rows = tmp_store.execute("SELECT url, title FROM articles")
        assert len(rows) == 1
        assert rows[0]["url"] == sample_page.url
        assert rows[0]["title"] == sample_page.title

    def test_execute_blocks_non_select(self, tmp_store):
        with pytest.raises(ValueError, match="Only SELECT"):
            tmp_store.execute("DELETE FROM articles")

    def test_execute_blocks_insert(self, tmp_store):
        with pytest.raises(ValueError, match="Only SELECT"):
            tmp_store.execute("INSERT INTO articles VALUES (1,2,3,4,5,6)")

    def test_search_finds_by_title(self, tmp_store, sample_page):
        tmp_store.upsert(sample_page)
        results = tmp_store.search("CAP theorem")
        assert len(results) == 1
        assert results[0]["title"] == "CAP theorem"

    def test_search_finds_by_content(self, tmp_store, sample_page):
        tmp_store.upsert(sample_page)
        results = tmp_store.search("distributed system")
        assert len(results) == 1

    def test_search_returns_empty_for_no_match(self, tmp_store, sample_page):
        tmp_store.upsert(sample_page)
        results = tmp_store.search("quantum computing")
        assert results == []

    def test_stats_returns_total_count(self, tmp_store, sample_page, sample_page_2):
        tmp_store.upsert(sample_page)
        tmp_store.upsert(sample_page_2)
        stats = tmp_store.stats()
        assert stats["total_articles"] == 2

    def test_stats_groups_by_domain(self, tmp_store, sample_page, sample_page_2):
        tmp_store.upsert(sample_page)
        tmp_store.upsert(sample_page_2)
        stats = tmp_store.stats()
        domains = {d["source_domain"]: d["count"] for d in stats["domains"]}
        assert domains.get("en.wikipedia.org") == 2

    def test_get_schema_contains_table_name(self, tmp_store):
        schema = tmp_store.get_schema()
        assert "articles" in schema
        assert "url" in schema
        assert "title" in schema

    def test_count_empty_store_is_zero(self, tmp_store):
        assert tmp_store.count() == 0


# ---------------------------------------------------------------------------
# SQLAgent tests (mocked Ollama)
# ---------------------------------------------------------------------------

class TestSQLAgent:

    def _make_agent(self, tmp_store: SQLStore) -> SQLAgent:
        """Build a SQLAgent with mocked Ollama."""
        return SQLAgent(db_store=tmp_store)

    def test_ask_returns_sql_result(self, tmp_store, sample_page):
        tmp_store.upsert(sample_page)
        agent = self._make_agent(tmp_store)

        with patch("agent.sql_agent.ollama.chat") as mock_chat:
            # First call: SQL generation
            # Second call: explanation
            mock_chat.side_effect = [
                MagicMock(message=MagicMock(content="SELECT COUNT(*) as total FROM articles;")),
                MagicMock(message=MagicMock(content="There is 1 article stored.")),
            ]
            result = agent.ask("How many articles are stored?")

        assert isinstance(result, SQLResult)
        assert result.success
        assert "SELECT" in result.sql.upper()

    def test_ask_returns_explanation(self, tmp_store, sample_page):
        tmp_store.upsert(sample_page)
        agent = self._make_agent(tmp_store)

        with patch("agent.sql_agent.ollama.chat") as mock_chat:
            mock_chat.side_effect = [
                MagicMock(message=MagicMock(content="SELECT COUNT(*) FROM articles;")),
                MagicMock(message=MagicMock(content="There is 1 article in the database.")),
            ]
            result = agent.ask("How many articles?")

        assert result.explanation == "There is 1 article in the database."

    def test_ask_handles_invalid_sql_from_llm(self, tmp_store):
        agent = self._make_agent(tmp_store)

        with patch("agent.sql_agent.ollama.chat") as mock_chat:
            # LLM returns garbage SQL
            mock_chat.return_value = MagicMock(
                message=MagicMock(content="THIS IS NOT SQL AT ALL")
            )
            result = agent.ask("How many articles?")

        # Should return an error result, not raise
        assert not result.success
        assert result.error != ""

    def test_ask_blocks_delete_sql(self, tmp_store):
        agent = self._make_agent(tmp_store)

        with patch("agent.sql_agent.ollama.chat") as mock_chat:
            mock_chat.return_value = MagicMock(
                message=MagicMock(content="DELETE FROM articles;")
            )
            result = agent.ask("Delete everything")

        assert not result.success
        assert "blocked" in result.error.lower()

    def test_clean_sql_strips_markdown_fences(self, tmp_store):
        agent = self._make_agent(tmp_store)
        raw = "```sql\nSELECT * FROM articles;\n```"
        cleaned = agent._clean_sql(raw)
        assert "```" not in cleaned
        assert "SELECT" in cleaned

    def test_result_str_contains_question(self, tmp_store, sample_page):
        tmp_store.upsert(sample_page)
        agent = self._make_agent(tmp_store)

        with patch("agent.sql_agent.ollama.chat") as mock_chat:
            mock_chat.side_effect = [
                MagicMock(message=MagicMock(content="SELECT COUNT(*) FROM articles;")),
                MagicMock(message=MagicMock(content="One article.")),
            ]
            result = agent.ask("How many articles?")

        assert "How many articles?" in str(result)

    def test_raises_on_empty_question(self, tmp_store):
        agent = self._make_agent(tmp_store)
        with pytest.raises(ValueError):
            agent.ask("  ")


# ---------------------------------------------------------------------------
# Integration tests — require Ollama running
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestPhase2Integration:
    """
    Requires Ollama with llama3.1:8b.
    Run: pytest tests/test_phase2.py -v -m integration
    """

    def test_sql_agent_counts_articles(self, tmp_path):
        from scraper.scraper import ScrapedPage
        store = SQLStore(db_path=str(tmp_path / "test.db"))
        store.upsert(ScrapedPage(
            url="https://example.com/1",
            title="Article One",
            text="Content about machine learning.",
        ))
        store.upsert(ScrapedPage(
            url="https://example.com/2",
            title="Article Two",
            text="Content about deep learning.",
        ))

        agent = SQLAgent(db_store=store)
        result = agent.ask("How many articles are stored?")

        assert result.success
        assert len(result.rows) > 0
        store.close()
