"""
main.py — CLI entry point for the RAG pipeline.

Usage:
    # Scrape a page and add it to the knowledge base
    python main.py --scrape "https://en.wikipedia.org/wiki/CAP_theorem"

    # Ask a question grounded in the scraped content (RAG)
    python main.py --ask "What is the CAP theorem?"

    # Ask a structured question via SQL agent
    python main.py --sql "How many articles are stored?"

    # Show stats about the knowledge base
    python main.py --stats
"""

import argparse
import sys

from scraper.scraper import WebScraper, ScraperError
from embeddings.chunker import TextChunker
from embeddings.embedder import Embedder
from store.vector_store import VectorStore
from store.sql_store import SQLStore
from llm.rag import RAGQuery
from agent.sql_agent import SQLAgent


def parse_args():
    parser = argparse.ArgumentParser(
        description="RAG Pipeline — scrape pages and ask questions grounded in that content.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --scrape "https://en.wikipedia.org/wiki/CAP_theorem"
  python main.py --ask "What is the CAP theorem?"
  python main.py --scrape "https://en.wikipedia.org/wiki/Raft_(algorithm)" --ask "How does Raft achieve consensus?"
  python main.py --stats
        """,
    )
    parser.add_argument("--scrape", metavar="URL", help="Scrape a URL and add to knowledge base")
    parser.add_argument("--ask", metavar="QUESTION", help="Ask a question via RAG (semantic search)")
    parser.add_argument("--sql", metavar="QUESTION", help="Ask a structured question via SQL agent")
    parser.add_argument("--stats", action="store_true", help="Show knowledge base stats")
    return parser.parse_args()


def cmd_scrape(url: str, vector_store: VectorStore, sql_store: SQLStore, embedder: Embedder) -> None:
    """Scrape a URL, chunk the text, embed chunks, and store in both vector DB and SQL DB."""
    scraper = WebScraper()
    chunker = TextChunker()

    print(f"\nScraping: {url}")
    print("-" * 60)

    try:
        page = scraper.scrape(url)
    except ScraperError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Title:  {page.title}")
    print(f"Length: {len(page.text):,} characters")

    # Chunk the text
    chunks = chunker.chunk(page.text, url=page.url, title=page.title)
    print(f"Chunks: {len(chunks)}")

    # Embed and store in ChromaDB (for RAG)
    print("Embedding chunks...")
    texts = [c.text for c in chunks]
    embeddings = embedder.embed(texts)
    vector_store.add(chunks, embeddings)

    # Store full article in SQLite (for SQL agent)
    sql_store.upsert(page)

    print(f"Stored {len(chunks)} chunks in vector store.")
    print(f"Stored article in SQL store.")
    print(f"Total chunks: {vector_store.count()} | Total articles: {sql_store.count()}")


def cmd_ask(question: str, store: VectorStore, embedder: Embedder) -> None:
    """Ask a question and answer it using RAG."""
    rag = RAGQuery(store=store, embedder=embedder)

    print(f"\nQuestion: {question}")
    print("-" * 60)

    result = rag.query(question)

    print(f"\nAnswer:\n{result.answer}")

    if result.sources:
        print(f"\nSources used:")
        for i, s in enumerate(result.sources, 1):
            print(f"  [{i}] {s.title}")
            print(f"       {s.url}")
            print(f"       similarity score: {s.score:.3f}")


def cmd_sql(question: str, sql_store: SQLStore) -> None:
    """Answer a structured question using the SQL agent."""
    agent = SQLAgent(db_store=sql_store)

    print(f"\nSQL Agent")
    print(f"Question: {question}")
    print("-" * 60)

    result = agent.ask(question)
    print(str(result))


def cmd_stats(vector_store: VectorStore, sql_store: SQLStore) -> None:
    """Print knowledge base stats."""
    print(f"\nKnowledge Base Stats")
    print("-" * 60)
    print(f"Vector store chunks: {vector_store.count()}")
    stats = sql_store.stats()
    print(f"SQL store articles:  {stats['total_articles']}")
    if stats["domains"]:
        print("Domains:")
        for d in stats["domains"]:
            print(f"  {d['source_domain']}: {d['count']} articles")


def main():
    args = parse_args()

    if not any([args.scrape, args.ask, args.sql, args.stats]):
        print("Error: provide --scrape, --ask, --sql, or --stats. Use --help for usage.")
        sys.exit(1)

    # Shared instances — created once and reused across commands
    embedder = Embedder()
    vector_store = VectorStore()
    sql_store = SQLStore()

    if args.scrape:
        cmd_scrape(args.scrape, vector_store, sql_store, embedder)

    if args.ask:
        cmd_ask(args.ask, vector_store, embedder)

    if args.sql:
        cmd_sql(args.sql, sql_store)

    if args.stats:
        cmd_stats(vector_store, sql_store)


if __name__ == "__main__":
    main()
