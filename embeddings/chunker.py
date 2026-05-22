"""
Chunker — splits long text into overlapping chunks for embedding.

Why chunk?
  LLMs and embedding models have a context limit. A scraped Wikipedia
  article might be 10,000 words — we can't embed it as one piece and
  expect meaningful similarity search. Breaking it into smaller,
  overlapping chunks means each chunk is focused, and overlap ensures
  sentences at chunk boundaries aren't split in a way that loses context.

Chunking strategy:
  - Split on sentence boundaries (period + space) where possible
  - Target chunk size: CHUNK_SIZE characters (default 800)
  - Overlap: CHUNK_OVERLAP characters (default 100)
  - Each chunk carries metadata: source URL, title, chunk index

Usage:
    chunker = TextChunker()
    chunks = chunker.chunk(page.text, url=page.url, title=page.title)
    # → list of Chunk objects, each with .text and .metadata
"""

from dataclasses import dataclass, field

# Target character length for each chunk.
# 800 chars ≈ 150–200 tokens — fits well within embedding model limits
# and gives enough context for meaningful retrieval.
CHUNK_SIZE = 800

# Number of characters to overlap between consecutive chunks.
# Overlap ensures sentences at chunk boundaries aren't cut off,
# preserving context for retrieval.
CHUNK_OVERLAP = 100


@dataclass
class Chunk:
    """
    A single chunk of text ready for embedding.

    text     — the chunk content
    metadata — source URL, title, chunk index (stored alongside embedding in ChromaDB)
    chunk_id — unique string ID used as the ChromaDB document ID
    """
    text: str
    metadata: dict = field(default_factory=dict)
    chunk_id: str = ""

    def __repr__(self) -> str:
        return f"Chunk(id={self.chunk_id!r}, chars={len(self.text)}, source={self.metadata.get('url', '?')!r})"


class TextChunker:
    """
    Splits text into overlapping chunks for embedding.

    Args:
        chunk_size:    Target character length per chunk (default: 800)
        chunk_overlap: Overlap between consecutive chunks (default: 100)

    Usage:
        chunker = TextChunker()
        chunks = chunker.chunk(text, url="https://...", title="My Page")
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, url: str = "", title: str = "") -> list[Chunk]:
        """
        Split text into overlapping chunks and attach metadata.

        The strategy:
        1. Split the full text into sentences (on ". ")
        2. Accumulate sentences until we hit chunk_size
        3. When a chunk is full, record it and backtrack by chunk_overlap chars
           to start the next chunk — this creates the overlap

        Args:
            text:  The full text to chunk (e.g. a scraped article)
            url:   Source URL — stored in chunk metadata for citation
            title: Page title — stored in chunk metadata

        Returns:
            List of Chunk objects, each with text + metadata + unique ID
        """
        if not text.strip():
            return []

        # Split into sentences as the base unit — avoids cutting mid-sentence
        sentences = self._split_sentences(text)

        chunks = []
        current = ""
        chunk_index = 0

        for sentence in sentences:
            # If adding this sentence would exceed chunk_size, save current chunk
            if current and len(current) + len(sentence) > self.chunk_size:
                chunk = self._make_chunk(current, url, title, chunk_index)
                chunks.append(chunk)
                chunk_index += 1

                # Start next chunk from the overlap tail of the current chunk
                # This preserves context across chunk boundaries
                current = current[-self.chunk_overlap:] + " " + sentence
            else:
                current = (current + " " + sentence).strip()

        # Don't forget the final chunk
        if current.strip():
            chunks.append(self._make_chunk(current, url, title, chunk_index))

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """
        Split text into sentence-ish units on ". " boundaries.

        Not a full NLP sentence tokeniser — fast and good enough for chunking.
        Filters out empty strings from splitting.
        """
        raw = text.replace("\n\n", ". ").replace("\n", " ")
        parts = raw.split(". ")
        return [p.strip() for p in parts if p.strip()]

    def _make_chunk(self, text: str, url: str, title: str, index: int) -> Chunk:
        """
        Build a Chunk with a deterministic ID and metadata.

        The chunk ID is {url}#{index} — unique per source page + position.
        ChromaDB uses this as the document ID, so re-ingesting the same
        page overwrites existing chunks (upsert behaviour).
        """
        chunk_id = f"{url}#{index}" if url else f"chunk#{index}"
        return Chunk(
            text=text.strip(),
            metadata={"url": url, "title": title, "chunk_index": index},
            chunk_id=chunk_id,
        )
