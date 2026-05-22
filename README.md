# rag-pipeline

A step-by-step project to learn RAG (Retrieval Augmented Generation) —
scraping real data, storing it in a vector + SQL database, and letting an
LLM answer questions using that knowledge.

Built entirely with open source tools. No API keys needed.

## Phases
See [PHASES.md](PHASES.md) for the full build plan.

## Stack
- LLM: Ollama (llama3.1:8b, nomic-embed-text)
- Vector DB: ChromaDB
- SQL DB: SQLite
- Scraping: BeautifulSoup4 + Playwright
- Embeddings: sentence-transformers
- Scheduling: APScheduler
- API: FastAPI
- Dashboard: Streamlit
