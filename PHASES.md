# RAG Pipeline — Build Phases

All open source. No API keys. Runs fully local via Ollama.

---

## Phase 1: RAG Pipeline with Real Web Scraping
**Branch:** `feature/phase-1-rag-pipeline`

### What gets built
- `scraper/scraper.py` — scrape a real website (e.g. Wikipedia or a docs site) using BeautifulSoup4
- `embeddings/chunker.py` — split scraped text into overlapping chunks
- `embeddings/embedder.py` — embed chunks using `sentence-transformers` (local, no API key)
- `store/vector_store.py` — store and query embeddings in ChromaDB
- `llm/rag.py` — retrieve relevant chunks, build a prompt, query Ollama (llama3.1:8b), return answer
- CLI — ask a question, get an answer grounded in scraped content

### Open source tools
| Tool | Purpose |
|------|---------|
| BeautifulSoup4 | HTML parsing and text extraction |
| sentence-transformers | Local text embeddings (all-MiniLM-L6-v2) |
| ChromaDB | Local vector database |
| Ollama (llama3.1:8b) | LLM for answer generation |

### What you learn
- How RAG works end to end: scrape → chunk → embed → store → retrieve → generate
- Why chunking strategy matters (size, overlap)
- How vector similarity search finds relevant context

### Definition of Done
- `python main.py --ask "What is the CAP theorem?"` returns an answer grounded in scraped docs
- Answer includes which source chunks were used

---

## Phase 2: SQL Agent
**Branch:** `feature/phase-2-sql-agent`

### What gets built
- `store/sql_store.py` — SQLite DB, stores scraped articles as structured rows (title, url, content, date)
- `agent/sql_agent.py` — LLM generates a SQL query from a natural language question, runs it, returns results
- `llm/sql_prompt.py` — prompt template that gives the LLM the DB schema and question
- CLI — ask a question, get SQL + result + LLM explanation

### Open source tools
| Tool | Purpose |
|------|---------|
| SQLite (built-in) | Structured storage |
| Ollama (llama3.1:8b) | SQL generation + explanation |

### What you learn
- LLM-as-SQL-writer pattern
- How to give an LLM a schema so it generates correct queries
- Difference between SQL agent (structured) vs RAG (semantic)

### Definition of Done
- "How many articles were scraped about machine learning?" → correct SQL → correct count
- LLM explains the result in plain English

---

## Phase 3: Scheduled Scraping Pipeline
**Branch:** `feature/phase-3-scheduler`

### What gets built
- `scraper/playwright_scraper.py` — Playwright scraper for JS-heavy sites
- `scraper/pipeline.py` — scrape → clean → deduplicate → upsert into both vector DB and SQL DB
- `scraper/scheduler.py` — APScheduler job that runs the pipeline on a cron schedule
- Deduplication by URL hash — never re-embed the same page twice

### Open source tools
| Tool | Purpose |
|------|---------|
| Playwright | Scraping JS-rendered pages |
| APScheduler | Cron-style job scheduling |
| hashlib (built-in) | URL deduplication |

### What you learn
- Keeping a knowledge base fresh automatically
- Deduplication strategies for incremental ingestion
- Difference between static (BeautifulSoup) and dynamic (Playwright) scraping

### Definition of Done
- Scheduler runs every hour, scrapes new pages, skips already-seen URLs
- Vector DB and SQL DB stay in sync

---

## Phase 4: Hybrid Search
**Branch:** `feature/phase-4-hybrid-search`

### What gets built
- `retrieval/vector_search.py` — semantic search via ChromaDB embeddings
- `retrieval/keyword_search.py` — BM25 keyword search over scraped text (rank-bm25)
- `retrieval/hybrid.py` — merge and re-rank results from both, pass top-k to LLM
- Comparison mode — show what vector-only vs keyword-only vs hybrid returns

### Open source tools
| Tool | Purpose |
|------|---------|
| rank-bm25 | BM25 keyword search |
| ChromaDB | Vector similarity search |

### What you learn
- Why pure vector search misses exact matches (names, codes, IDs)
- Why pure keyword search misses semantic matches ("car" vs "automobile")
- Reciprocal Rank Fusion (RRF) — the standard way to merge ranked lists

### Definition of Done
- Hybrid search outperforms either method alone on a test query set
- Side-by-side comparison visible in output

---

## Phase 5: Agentic RAG
**Branch:** `feature/phase-5-agentic-rag`

### What gets built
- `agent/tools.py` — LLM tools: `search_vector_db`, `query_sql`, `scrape_url`, `answer_from_memory`
- `agent/agent.py` — ReAct-style agent: LLM reasons about which tool to call, calls it, uses result
- `api/main.py` — FastAPI wrapping the agent
- `dashboard/app.py` — Streamlit UI showing the agent's reasoning steps + final answer

### Open source tools
| Tool | Purpose |
|------|---------|
| Ollama (llama3.1:8b) | Agent reasoning (ReAct loop) |
| ChromaDB | Vector search tool |
| SQLite | SQL query tool |
| Playwright | Live scraping tool |
| FastAPI | API layer |
| Streamlit | Dashboard |

### What you learn
- ReAct (Reason + Act) pattern — the foundation of most LLM agents
- How agents decide which tool to use based on the question
- How to chain tool results into a final grounded answer

### Definition of Done
- Agent correctly decides whether to search vector DB, run SQL, or scrape fresh data
- Reasoning steps visible in the dashboard ("I will search the vector DB because...")

---

## Tech Stack Summary

| Layer | Tool | Why |
|-------|------|-----|
| LLM | Ollama llama3.1:8b | Local, free, capable |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 | Fast, local, good quality |
| Vector DB | ChromaDB | Simple, local, no server needed |
| SQL DB | SQLite | Built-in, zero setup |
| Scraping (static) | BeautifulSoup4 + requests | Fast, lightweight |
| Scraping (dynamic) | Playwright | Handles JS-rendered pages |
| Scheduling | APScheduler | Simple cron-style jobs |
| Keyword search | rank-bm25 | Standard BM25 implementation |
| API | FastAPI | Already familiar from smart-router |
| Dashboard | Streamlit | Already familiar from smart-router |
