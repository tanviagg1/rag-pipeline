"""
Embedder — converts text chunks into vector embeddings using sentence-transformers.

Why sentence-transformers?
  Fully local, no API key, fast, and all-MiniLM-L6-v2 produces 384-dimension
  embeddings that work well for semantic similarity search. It's the standard
  open source choice for RAG pipelines.

How it works:
  - Loads the model once on first use (cached by sentence-transformers)
  - Encodes a list of strings → numpy array of shape (n, 384)
  - ChromaDB accepts these as Python lists

Usage:
    embedder = Embedder()
    vectors = embedder.embed(["What is RAG?", "RAG stands for..."])
    # → list of 384-dimension float lists
"""

from sentence_transformers import SentenceTransformer

# all-MiniLM-L6-v2:
#   - 384 dimensions (small, fast)
#   - Good semantic similarity quality for English text
#   - Downloads ~90MB on first use, then cached locally
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class Embedder:
    """
    Wraps sentence-transformers to produce text embeddings.

    The model is loaded lazily on first call to embed() — this avoids
    loading a large model at import time if the embedder isn't used.

    Args:
        model_name: sentence-transformers model to use (default: all-MiniLM-L6-v2)

    Usage:
        embedder = Embedder()
        vectors = embedder.embed(["Hello world", "Goodbye world"])
        # vectors[0] is a list of 384 floats
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the model on first access."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of strings into vectors.

        Args:
            texts: List of strings to embed (chunks, queries, etc.)

        Returns:
            List of embedding vectors — each is a list of 384 floats.
            Index matches input: embeddings[i] corresponds to texts[i].

        Note:
            Batching is handled internally by sentence-transformers.
            For large lists (1000+), it will batch automatically.
        """
        if not texts:
            return []

        # encode() returns a numpy array — convert to Python lists for ChromaDB
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_one(self, text: str) -> list[float]:
        """
        Embed a single string — convenience wrapper for query embedding.

        Args:
            text: The string to embed (e.g. a user query)

        Returns:
            A single embedding vector (list of 384 floats)
        """
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        """Return the embedding dimension (384 for all-MiniLM-L6-v2)."""
        return self.model.get_sentence_embedding_dimension()
