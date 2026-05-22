"""
main.py — CLI entry point for the RAG pipeline.

Usage:
    # Scrape a page and add it to the knowledge base
    python main.py --scrape "https://en.wikipedia.org/wiki/CAP_theorem"

    # Ask a question grounded in the scraped content
    python main.py --ask "What is the CAP theorem?"

    # Scrape and immediately ask a question
    python main.py --scrape "https://en.wikipedia.org/wiki/CAP_theorem" --ask "What is eventual consistency?"

    # Show stats about the knowledge base
    python main.py --stats
"""

import argparse
import sys

from scraper.scraper import WebScraper, ScraperError
from embeddings.chunker import TextChunker
from embeddings.embedder import Embedder
from store.vector_store import VectorStore
from llm.rag import RAGQuery


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
    parser.add_argument("--ask", metavar="QUESTION", help="Ask a question using the knowledge base")
    parser.add_argument("--stats", action="store_true", help="Show knowledge base stats")
    return parser.parse_args()


def cmd_scrape(url: str, store: VectorStore, embedder: Embedder) -> None:
    """Scrape a URL, chunk the text, embed chunks, and store in vector DB."""
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

    # Embed all chunks
    print("Embedding chunks...")
    texts = [c.text for c in chunks]
    embeddings = embedder.embed(texts)

    # Store in ChromaDB
    store.add(chunks, embeddings)
    print(f"Stored {len(chunks)} chunks in knowledge base.")
    print(f"Total chunks in KB: {store.count()}")


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


def cmd_stats(store: VectorStore) -> None:
    """Print knowledge base stats."""
    print(f"\nKnowledge Base Stats")
    print("-" * 60)
    print(f"Total chunks: {store.count()}")


def main():
    args = parse_args()

    if not any([args.scrape, args.ask, args.stats]):
        print("Error: provide --scrape, --ask, or --stats. Use --help for usage.")
        sys.exit(1)

    # Shared instances — embedder loads model once, store opens DB once
    embedder = Embedder()
    store = VectorStore()

    if args.scrape:
        cmd_scrape(args.scrape, store, embedder)

    if args.ask:
        cmd_ask(args.ask, store, embedder)

    if args.stats:
        cmd_stats(store)


if __name__ == "__main__":
    main()
