"""
RAG — Retrieval Augmented Generation query engine.

This is the core of Phase 1. Given a user question:
  1. Embed the question using the same model used to embed chunks
  2. Retrieve the top-k most similar chunks from the vector store
  3. Build a prompt that includes the retrieved context
  4. Send to Ollama (llama3.1:8b) to generate a grounded answer
  5. Return the answer + the source chunks used

Why RAG?
  Without retrieval, the LLM can only answer from its training data
  (which may be outdated or incomplete). RAG grounds the answer in
  real, up-to-date content that we scraped ourselves.

Usage:
    rag = RAGQuery()
    result = rag.query("What is the CAP theorem?")
    print(result.answer)
    for source in result.sources:
        print(source.url, source.score)
"""

import ollama
from dataclasses import dataclass, field

from embeddings.embedder import Embedder
from store.vector_store import VectorStore, SearchResult

# Ollama model for answer generation.
# llama3.1:8b is a capable open source model that handles reasoning well.
LLM_MODEL = "llama3.1:8b"

# Number of chunks to retrieve and pass as context to the LLM.
# More chunks = more context but larger prompt. 4 is a good default.
TOP_K = 4

# System prompt — instructs the LLM to stay grounded in the provided context
SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

Rules:
- Answer using ONLY the information in the context below
- If the context does not contain enough information to answer, say "I don't have enough information in my knowledge base to answer this."
- Keep your answer concise and factual
- Do not make up information not present in the context"""


@dataclass
class RAGResult:
    """
    The full output of a RAG query.

    answer  — the LLM's answer grounded in retrieved context
    sources — the chunks retrieved and used as context
    query   — the original question
    """
    query: str
    answer: str
    sources: list[SearchResult] = field(default_factory=list)

    def __str__(self) -> str:
        source_lines = "\n".join(
            f"  [{i+1}] {s.title} ({s.url}) — similarity score: {s.score:.3f}"
            for i, s in enumerate(self.sources)
        )
        return (
            f"Q: {self.query}\n\n"
            f"A: {self.answer}\n\n"
            f"Sources:\n{source_lines}"
        )


class RAGQuery:
    """
    Retrieval Augmented Generation engine.

    Wires together the Embedder, VectorStore, and Ollama to answer
    questions grounded in scraped content.

    Args:
        model:      Ollama model for generation (default: llama3.1:8b)
        top_k:      Number of chunks to retrieve (default: 4)
        embedder:   Embedder instance (created fresh if not provided)
        store:      VectorStore instance (created fresh if not provided)

    Usage:
        rag = RAGQuery()
        result = rag.query("What is eventual consistency?")
        print(result.answer)
    """

    def __init__(
        self,
        model: str = LLM_MODEL,
        top_k: int = TOP_K,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
    ):
        self.model = model
        self.top_k = top_k
        self.embedder = embedder or Embedder()
        self.store = store or VectorStore()

    def query(self, question: str) -> RAGResult:
        """
        Answer a question using retrieved context from the vector store.

        Steps:
          1. Embed the question
          2. Retrieve top-k similar chunks
          3. Build a prompt with context
          4. Call Ollama for the answer
          5. Return RAGResult with answer + sources

        Args:
            question: The user's natural language question

        Returns:
            RAGResult with answer and source chunks
        """
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        # Step 1 — embed the question using the same model as the chunks
        query_embedding = self.embedder.embed_one(question)

        # Step 2 — retrieve the most relevant chunks
        sources = self.store.query(query_embedding, n_results=self.top_k)

        if not sources:
            return RAGResult(
                query=question,
                answer="No content in the knowledge base yet. Run --scrape first.",
                sources=[],
            )

        # Step 3 — build the context block from retrieved chunks
        context = self._build_context(sources)

        # Step 4 — generate the answer
        answer = self._generate(question, context)

        return RAGResult(query=question, answer=answer, sources=sources)

    def _build_context(self, sources: list[SearchResult]) -> str:
        """
        Format retrieved chunks into a context block for the LLM prompt.

        Each chunk is labelled with its source title and URL so the LLM
        can reference where information came from.
        """
        parts = []
        for i, source in enumerate(sources, 1):
            parts.append(
                f"[Source {i}: {source.title}]\n{source.text}"
            )
        return "\n\n---\n\n".join(parts)

    def _generate(self, question: str, context: str) -> str:
        """
        Call Ollama to generate an answer given a question and context.

        The system prompt instructs the LLM to stay grounded in context.
        Temperature=0 for consistent, factual answers.
        """
        prompt = f"Context:\n{context}\n\nQuestion: {question}"

        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.1},
        )
        return response.message.content.strip()
