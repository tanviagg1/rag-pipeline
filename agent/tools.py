"""
Agent tools — functions the ReAct agent can call to answer questions.

Each tool is a plain Python function with:
  - A clear name
  - A docstring describing what it does and when to use it
  - Structured input/output

The agent (agent.py) decides which tool to call based on the question.
Tools are the "hands" of the agent — they do the actual work.

Available tools:
  search_knowledge_base  — hybrid search (vector + BM25) over stored chunks
  query_database         — SQL query for structured/statistical questions
  scrape_and_ingest      — scrape a new URL and add it to the knowledge base
  answer_from_context    — generate an LLM answer from provided context

Usage:
    tools = AgentTools()
    result = tools.search_knowledge_base("What is eventual consistency?")
    result = tools.query_database("How many articles from Wikipedia?")
    result = tools.scrape_and_ingest("https://en.wikipedia.org/wiki/Raft_(algorithm)")
"""

from dataclasses import dataclass, field

from retrieval.hybrid import HybridSearch
from store.sql_store import SQLStore
from store.vector_store import VectorStore
from embeddings.embedder import Embedder
from scraper.pipeline import ScrapingPipeline
from retrieval.vector_search import VectorSearch
from retrieval.keyword_search import KeywordSearch


@dataclass
class ToolResult:
    """
    Output from any tool call.

    tool    — name of the tool that ran
    output  — the result (text, list, dict — depends on tool)
    error   — set if the tool failed
    """
    tool: str
    output: str
    error: str = ""

    @property
    def success(self) -> bool:
        return not self.error

    def __str__(self) -> str:
        if self.error:
            return f"[{self.tool}] ERROR: {self.error}"
        return f"[{self.tool}] {self.output}"


class AgentTools:
    """
    Collection of tools available to the ReAct agent.

    All tools return ToolResult so the agent has a consistent interface
    regardless of which tool it called.

    Args:
        vector_store:  Shared VectorStore instance
        sql_store:     Shared SQLStore instance
        embedder:      Shared Embedder instance
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        sql_store: SQLStore | None = None,
        embedder: Embedder | None = None,
    ):
        self.vector_store = vector_store or VectorStore()
        self.sql_store = sql_store or SQLStore()
        self.embedder = embedder or Embedder()

        # Initialise searchers with shared instances
        self.hybrid_search = HybridSearch(
            vector_search=VectorSearch(embedder=self.embedder, store=self.vector_store),
            keyword_search=KeywordSearch(store=self.vector_store),
        )

    def search_knowledge_base(self, query: str, top_k: int = 4) -> ToolResult:
        """
        Search the knowledge base for relevant content using hybrid search.

        Use this tool when the question is about concepts, explanations,
        or anything that requires reading scraped content.

        Args:
            query:  The search query
            top_k:  Number of results to return

        Returns:
            ToolResult with formatted context chunks
        """
        try:
            results = self.hybrid_search.search(query, top_k=top_k)
            if not results:
                return ToolResult(
                    tool="search_knowledge_base",
                    output="No relevant content found in the knowledge base.",
                )

            # Format results as a numbered context block
            lines = []
            for i, r in enumerate(results, 1):
                lines.append(
                    f"[{i}] {r.title} ({r.source_label()})\n{r.text}"
                )
            output = "\n\n".join(lines)
            return ToolResult(tool="search_knowledge_base", output=output)

        except Exception as e:
            return ToolResult(tool="search_knowledge_base", output="", error=str(e))

    def query_database(self, question: str) -> ToolResult:
        """
        Answer a structured question about stored articles using the SQL agent.

        Use this tool for counting, filtering, grouping, or any question
        about the metadata of stored articles (not their content).
        Examples: "How many articles?", "Which domains?", "Articles from today?"

        Args:
            question: Natural language question about the articles database

        Returns:
            ToolResult with SQL query + plain English explanation
        """
        try:
            from agent.sql_agent import SQLAgent
            agent = SQLAgent(db_store=self.sql_store)
            result = agent.ask(question)
            if result.success:
                output = f"SQL: {result.sql}\n\nResult: {result.explanation}"
            else:
                output = f"Query failed: {result.error}"
            return ToolResult(tool="query_database", output=output)
        except Exception as e:
            return ToolResult(tool="query_database", output="", error=str(e))

    def scrape_and_ingest(self, url: str) -> ToolResult:
        """
        Scrape a URL and add its content to the knowledge base.

        Use this tool when the user asks about a topic and the knowledge
        base doesn't have relevant content — scrape a page about it first.

        Args:
            url: The URL to scrape

        Returns:
            ToolResult describing how many chunks were added
        """
        try:
            pipeline = ScrapingPipeline(
                vector_store=self.vector_store,
                sql_store=self.sql_store,
                embedder=self.embedder,
            )
            result = pipeline.ingest(url)
            if result.success and not result.skipped:
                output = f"Scraped and stored {result.chunks} chunks from {url}"
            elif result.skipped:
                output = f"URL already in knowledge base: {url}"
            else:
                return ToolResult(tool="scrape_and_ingest", output="", error=result.error)
            return ToolResult(tool="scrape_and_ingest", output=output)
        except Exception as e:
            return ToolResult(tool="scrape_and_ingest", output="", error=str(e))

    def get_kb_stats(self) -> ToolResult:
        """
        Return statistics about the knowledge base.

        Use this when the user asks what's in the knowledge base,
        how many articles are stored, or which domains are covered.

        Returns:
            ToolResult with chunk count and article stats
        """
        try:
            chunks = self.vector_store.count()
            stats = self.sql_store.stats()
            lines = [
                f"Vector store: {chunks} chunks",
                f"SQL store: {stats['total_articles']} articles",
            ]
            if stats["domains"]:
                lines.append("Domains: " + ", ".join(
                    f"{d['source_domain']} ({d['count']})" for d in stats["domains"]
                ))
            return ToolResult(tool="get_kb_stats", output="\n".join(lines))
        except Exception as e:
            return ToolResult(tool="get_kb_stats", output="", error=str(e))
