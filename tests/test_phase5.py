"""
Tests for Phase 5 — AgentTools, ReActAgent, FastAPI endpoints.

All external dependencies mocked — no Ollama, ChromaDB, or SQLite needed.

Run:
    pytest tests/test_phase5.py -v -m "not integration"
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# AgentTools tests
# ---------------------------------------------------------------------------

class TestAgentTools:

    def _make_tools(self):
        from agent.tools import AgentTools
        tools = AgentTools.__new__(AgentTools)
        tools.vector_store = MagicMock()
        tools.sql_store = MagicMock()
        tools.embedder = MagicMock()
        tools.hybrid_search = MagicMock()
        return tools

    def test_search_returns_tool_result(self):
        from agent.tools import AgentTools
        from retrieval.hybrid import HybridResult
        tools = self._make_tools()
        tools.hybrid_search.search.return_value = [
            HybridResult(
                text="Raft is a consensus algorithm.",
                metadata={"url": "https://x.com", "title": "Raft"},
                rrf_score=0.03,
                in_vector=True,
            )
        ]

        result = tools.search_knowledge_base("Raft consensus")
        assert result.success
        assert "Raft" in result.output

    def test_search_returns_no_content_message_when_empty(self):
        from agent.tools import AgentTools
        tools = self._make_tools()
        tools.hybrid_search.search.return_value = []

        result = tools.search_knowledge_base("anything")
        assert result.success
        assert "No relevant" in result.output

    def test_get_kb_stats_returns_counts(self):
        from agent.tools import AgentTools
        tools = self._make_tools()
        tools.vector_store.count.return_value = 42
        tools.sql_store.stats.return_value = {"total_articles": 5, "domains": []}

        result = tools.get_kb_stats()
        assert result.success
        assert "42" in result.output
        assert "5" in result.output

    def test_scrape_and_ingest_success(self):
        from agent.tools import AgentTools
        from scraper.pipeline import IngestResult
        tools = self._make_tools()

        with patch("agent.tools.ScrapingPipeline") as MockPipeline:
            MockPipeline.return_value.ingest.return_value = IngestResult(
                url="https://x.com", chunks=10
            )
            result = tools.scrape_and_ingest("https://x.com")

        assert result.success
        assert "10 chunks" in result.output

    def test_scrape_and_ingest_already_stored(self):
        from agent.tools import AgentTools
        from scraper.pipeline import IngestResult
        tools = self._make_tools()

        with patch("agent.tools.ScrapingPipeline") as MockPipeline:
            MockPipeline.return_value.ingest.return_value = IngestResult(
                url="https://x.com", skipped=True
            )
            result = tools.scrape_and_ingest("https://x.com")

        assert result.success
        assert "already" in result.output.lower()


# ---------------------------------------------------------------------------
# ReActAgent tests
# ---------------------------------------------------------------------------

class TestReActAgent:

    def _make_agent(self, llm_responses: list[str]):
        from agent.agent import ReActAgent
        from agent.tools import AgentTools

        tools = MagicMock(spec=AgentTools)
        tools.search_knowledge_base.return_value = MagicMock(
            success=True, output="Raft is a consensus algorithm for log replication."
        )
        tools.get_kb_stats.return_value = MagicMock(success=True, output="10 chunks, 2 articles")

        agent = ReActAgent.__new__(ReActAgent)
        agent.model = "llama3.1:8b"
        agent.tools = tools
        agent.max_steps = 5

        # Mock the LLM call sequence
        call_count = [0]
        def fake_call_llm(messages):
            resp = llm_responses[min(call_count[0], len(llm_responses) - 1)]
            call_count[0] += 1
            return resp

        agent._call_llm = fake_call_llm
        return agent

    def test_run_returns_agent_result(self):
        from agent.agent import AgentResult
        agent = self._make_agent([
            "Thought: I should search the KB.\nAction: answer(Raft is a consensus algorithm.)"
        ])
        result = agent.run("What is Raft?")
        assert isinstance(result, AgentResult)
        assert result.answer == "Raft is a consensus algorithm."

    def test_run_executes_tool_before_answering(self):
        from agent.agent import AgentResult
        agent = self._make_agent([
            "Thought: Search first.\nAction: search_knowledge_base(Raft consensus)",
            "Thought: I have context.\nAction: answer(Raft uses leader election.)",
        ])
        result = agent.run("What is Raft?")

        agent.tools.search_knowledge_base.assert_called_once_with("Raft consensus")
        assert len(result.steps) == 1  # one tool call before answer

    def test_run_stops_at_max_steps(self):
        # LLM never calls answer() — should hit max_steps and force answer
        from agent.agent import ReActAgent
        agent = self._make_agent(
            ["Thought: thinking.\nAction: search_knowledge_base(query)"] * 10
        )
        agent.max_steps = 2

        result = agent.run("What is Raft?")
        assert result.stopped_early is True

    def test_run_raises_on_empty_question(self):
        agent = self._make_agent(["Thought: answer.\nAction: answer(x)"])
        with pytest.raises(ValueError):
            agent.run("  ")

    def test_parse_response_extracts_action(self):
        from agent.agent import ReActAgent
        agent = ReActAgent.__new__(ReActAgent)
        thought, action, action_input = agent._parse_response(
            "Thought: I should search.\nAction: search_knowledge_base(Raft consensus)"
        )
        assert thought == "I should search."
        assert action == "search_knowledge_base"
        assert action_input == "Raft consensus"

    def test_parse_response_fallback_on_bad_format(self):
        from agent.agent import ReActAgent
        agent = ReActAgent.__new__(ReActAgent)
        thought, action, action_input = agent._parse_response("just some text with no format")
        # Should fall back to answer action
        assert action == "answer"

    def test_execute_unknown_tool_returns_error_string(self):
        from agent.agent import ReActAgent
        agent = ReActAgent.__new__(ReActAgent)
        agent.tools = MagicMock()
        result = agent._execute_tool("nonexistent_tool", "input")
        assert "Unknown tool" in result


# ---------------------------------------------------------------------------
# FastAPI endpoint tests
# ---------------------------------------------------------------------------

class TestAPI:

    @pytest.fixture
    def client(self):
        from api.main import app
        return TestClient(app)

    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_ask_returns_200_with_mocked_agent(self, client):
        from agent.agent import AgentResult
        with patch("api.main.agent") as mock_agent:
            mock_agent.run.return_value = AgentResult(
                question="What is Raft?",
                answer="Raft is a consensus algorithm.",
                steps=[],
            )
            r = client.post("/ask", json={"question": "What is Raft?"})
        assert r.status_code == 200
        assert r.json()["answer"] == "Raft is a consensus algorithm."

    def test_ask_returns_422_on_empty_question(self, client):
        r = client.post("/ask", json={"question": ""})
        assert r.status_code == 422

    def test_scrape_returns_200(self, client):
        from scraper.pipeline import IngestResult
        with patch("api.main.ScrapingPipeline") as MockPipeline:
            MockPipeline.return_value.ingest.return_value = IngestResult(
                url="https://example.com", chunks=5
            )
            r = client.post("/scrape", json={"url": "https://example.com"})
        assert r.status_code == 200
        assert r.json()["chunks"] == 5

    def test_stats_returns_200(self, client):
        with patch("api.main.tools") as mock_tools:
            mock_tools.get_kb_stats.return_value = MagicMock(output="10 chunks")
            r = client.get("/stats")
        assert r.status_code == 200

    def test_search_returns_results(self, client):
        from retrieval.hybrid import HybridResult
        with patch("api.main.tools") as mock_tools:
            mock_tools.hybrid_search.search.return_value = [
                HybridResult(
                    text="Raft is a consensus algorithm.",
                    metadata={"url": "https://x.com", "title": "Raft"},
                    rrf_score=0.03,
                    in_vector=True,
                )
            ]
            r = client.get("/search", params={"q": "Raft", "top_k": 3})
        assert r.status_code == 200
        assert len(r.json()) == 1
