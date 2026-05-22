# RAG Pipeline

## Purpose
Step-by-step project to learn RAG (Retrieval Augmented Generation) — scraping real data,
storing it in a vector + SQL database, and letting an LLM answer questions grounded in
that knowledge. Built entirely with open source tools, runs fully local via Ollama.

## Stack
- LLM: Ollama (llama3.1:8b)
- Embeddings: sentence-transformers (all-MiniLM-L6-v2)
- Vector DB: ChromaDB (local)
- SQL DB: SQLite
- Scraping: BeautifulSoup4 + requests, Playwright (JS-heavy sites)
- Scheduling: APScheduler
- Keyword search: rank-bm25
- API: FastAPI
- Dashboard: Streamlit

## Folder Structure
```
rag-pipeline/
├── scraper/          # Web scraping + scheduling pipeline
│   ├── scraper.py    # BeautifulSoup scraper
│   ├── playwright_scraper.py  # JS-heavy sites (Phase 3)
│   ├── pipeline.py   # scrape → clean → deduplicate → store
│   └── scheduler.py  # APScheduler cron jobs (Phase 3)
├── embeddings/       # Text chunking + embedding
│   ├── chunker.py    # Split text into overlapping chunks
│   └── embedder.py   # Embed chunks via sentence-transformers
├── store/            # Storage layer
│   ├── vector_store.py  # ChromaDB wrapper
│   └── sql_store.py     # SQLite wrapper (Phase 2)
├── retrieval/        # Search strategies (Phase 4)
│   ├── vector_search.py
│   ├── keyword_search.py
│   └── hybrid.py
├── llm/              # LLM querying
│   ├── rag.py        # Retrieve + generate
│   └── sql_prompt.py # SQL generation prompts (Phase 2)
├── agent/            # Agentic RAG (Phase 5)
│   ├── tools.py
│   └── agent.py
├── api/              # FastAPI (Phase 5)
│   └── main.py
├── dashboard/        # Streamlit UI (Phase 5)
│   └── app.py
├── tests/
├── main.py           # CLI entry point
├── CLAUDE.md         # This file
├── PHASES.md         # Build roadmap
└── requirements.txt
```

## How to Run

### Prerequisites
```bash
ollama pull llama3.1:8b
pip install -r requirements.txt
```

### CLI (Phase 1+)
```bash
python main.py --ask "What is the CAP theorem?"
python main.py --scrape "https://en.wikipedia.org/wiki/CAP_theorem"
```

### API (Phase 5)
```bash
uvicorn api.main:app --reload
```

### Dashboard (Phase 5)
```bash
streamlit run dashboard/app.py
```

### Tests
```bash
pytest tests/ -v
pytest tests/ -m "not integration"
```

## Branch Strategy
- `main` — stable
- `feature/phase-1-rag-pipeline` — scrape → embed → vector DB → RAG
- `feature/phase-2-sql-agent` — SQL DB + LLM SQL generation
- `feature/phase-3-scheduler` — scheduled scraping pipeline
- `feature/phase-4-hybrid-search` — BM25 + vector hybrid search
- `feature/phase-5-agentic-rag` — ReAct agent with tools

## Conventions
- All LLM calls go through Ollama — no external API keys
- Integration tests (real Ollama/ChromaDB calls) marked `@pytest.mark.integration`
- Embeddings model: all-MiniLM-L6-v2 (fast, local, good quality)
- Chunk size: 500 tokens, overlap: 50 tokens (default, tunable)
