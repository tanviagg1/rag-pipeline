"""
RAG Pipeline — FastAPI application.

Endpoints:
  POST /ask     — run the ReAct agent on a question, return answer + reasoning trace
  POST /scrape  — scrape a URL and add it to the knowledge base
  GET  /search  — hybrid search (vector + BM25) over stored chunks
  GET  /stats   — knowledge base stats
  GET  /health  — liveness check

Run:
    uvicorn api.main:app --reload
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from agent.agent import ReActAgent, AgentResult
from agent.tools import AgentTools
from store.vector_store import VectorStore
from store.sql_store import SQLStore
from embeddings.embedder import Embedder
from scraper.pipeline import ScrapingPipeline

app = FastAPI(
    title="RAG Pipeline",
    description="Agentic RAG — scrape, store, and answer questions with a local LLM.",
    version="5.0.0",
)

# Module-level singletons — shared across all requests
vector_store = VectorStore()
sql_store = SQLStore()
embedder = Embedder()
tools = AgentTools(vector_store=vector_store, sql_store=sql_store, embedder=embedder)
agent = ReActAgent(tools=tools)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question for the agent")


class AskResponse(BaseModel):
    question: str
    answer: str
    steps: list[dict]       # [{thought, action, action_input, observation}]
    stopped_early: bool


class ScrapeRequest(BaseModel):
    url: str = Field(..., description="URL to scrape and ingest")
    force: bool = Field(False, description="Re-ingest even if already stored")


class ScrapeResponse(BaseModel):
    url: str
    chunks: int
    skipped: bool
    message: str


class SearchResult(BaseModel):
    text: str
    title: str
    url: str
    rrf_score: float
    sources: str            # "vector(#1) + keyword(#2)"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """
    Run the ReAct agent on a question.

    The agent reasons about which tools to use, calls them, and returns
    a final answer plus the full reasoning trace.
    """
    try:
        result: AgentResult = agent.run(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent failed: {e}")

    return AskResponse(
        question=result.question,
        answer=result.answer,
        steps=[
            {
                "thought": s.thought,
                "action": s.action,
                "action_input": s.action_input,
                "observation": s.observation,
            }
            for s in result.steps
        ],
        stopped_early=result.stopped_early,
    )


@app.post("/scrape", response_model=ScrapeResponse)
def scrape(request: ScrapeRequest) -> ScrapeResponse:
    """Scrape a URL and add its content to the knowledge base."""
    pipeline = ScrapingPipeline(
        vector_store=vector_store,
        sql_store=sql_store,
        embedder=embedder,
    )
    result = pipeline.ingest(request.url, force=request.force)

    if not result.success:
        raise HTTPException(status_code=422, detail=result.error)

    return ScrapeResponse(
        url=result.url,
        chunks=result.chunks,
        skipped=result.skipped,
        message=str(result),
    )


@app.get("/search", response_model=list[SearchResult])
def search(q: str = Query(..., description="Search query"), top_k: int = 5) -> list[SearchResult]:
    """Run hybrid search (vector + BM25) and return ranked results."""
    results = tools.hybrid_search.search(q, top_k=top_k)
    return [
        SearchResult(
            text=r.text,
            title=r.title,
            url=r.url,
            rrf_score=r.rrf_score,
            sources=r.source_label(),
        )
        for r in results
    ]


@app.get("/stats")
def stats() -> dict:
    """Return knowledge base stats."""
    result = tools.get_kb_stats()
    return {"stats": result.output}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
