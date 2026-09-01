"""
TrainingDataset — in-memory dataset for teacher-student distillation.

Holds source text, chunks it, indexes into a vector store, and provides
both the teacher (context retrieval) and student (next-token prediction)
with live access to the data.

Both models read from the same TrainingDataset instance during training.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger("slo.training.dataset")


class TrainingDataset:
    """In-memory dataset shared by teacher and student during distillation.

    On construction, the source text is chunked and indexed into an
    InMemoryVectorStore. The teacher queries the store for context;
    the student trains on the raw text via next-token prediction.

    Attributes:
        source_text: The raw source text.
        chunks: List of overlapping text chunks.
        store: InMemoryVectorStore for context retrieval.
        chunk_size: Characters per chunk.
    """

    def __init__(
        self,
        source_text: str,
        chunk_size: int = 400,
        chunk_overlap: int = 50,
    ):
        self.source_text = source_text
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: List[str] = []
        self.store: Any = None

        self._build()

    def _build(self) -> None:
        """Chunk source text and index into vector store."""
        from domains.inference.vector_store import (
            InMemoryVectorStore,
            VectorEntry,
            simple_embed,
        )

        # Chunk
        text = self.source_text
        if len(text) <= self.chunk_size:
            self.chunks = [text.strip()] if text.strip() and len(text.strip()) > 20 else []
        else:
            self.chunks = []
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk = text[start:end].strip()
                if chunk and len(chunk) > 20:
                    self.chunks.append(chunk)
                start += self.chunk_size - self.chunk_overlap

        if not self.chunks:
            logger.warning("No chunks from source text (len=%d)", len(text))
            self.store = InMemoryVectorStore(dimension=384)
            return

        # Index into vector store
        self.store = InMemoryVectorStore(dimension=384)
        entries = []
        for i, chunk in enumerate(self.chunks):
            vec = simple_embed(chunk, dimension=384)
            entries.append(
                VectorEntry(
                    id=f"chunk_{i}",
                    vector=vec,
                    text=chunk,
                    metadata={"index": i, "length": len(chunk)},
                )
            )
        self.store.upsert_sync(entries)

        logger.info(
            "TrainingDataset built: %d chunks, %d store entries (source_len=%d)",
            len(self.chunks),
            self.store.count_sync(),
            len(self.source_text),
        )

    def get_teacher_context(self, query: str, top_k: int = 3, min_score: float = 0.15) -> str:
        """Retrieve relevant context for the teacher from the vector store.

        Args:
            query: The prompt/query to search for.
            top_k: Number of results to retrieve.
            min_score: Minimum cosine similarity to include.

        Returns:
            Concatenated relevant passages, or empty string.
        """
        from domains.inference.vector_store import simple_embed

        if self.store is None or self.store.count_sync() == 0:
            return ""

        query_vec = simple_embed(query, dimension=384)
        results = self.store.query_sync(query_vec, top_k=top_k)

        relevant = [r.text for r in results if r.score >= min_score]
        if not relevant and results:
            relevant = [results[0].text]

        return "\n\n".join(relevant)

    def get_student_text(self) -> str:
        """Return the full source text for student next-token training.

        Returns:
            The raw source text as a single string.
        """
        return self.source_text

    def get_student_pairs(self) -> List[Dict[str, str]]:
        """Return chunks formatted as user/assistant pairs for chat training.

        Each chunk becomes a pair where user_msg is the first sentence
        and assistant_msg is the rest.

        Returns:
            List of {"user_msg", "assistant_msg"} dicts.
        """
        pairs = []
        for chunk in self.chunks:
            sentences = chunk.split(".")
            if len(sentences) > 1:
                user_msg = sentences[0].strip() + "."
                assistant_msg = ".".join(sentences[1:]).strip()
            else:
                user_msg = "Tell me about this topic."
                assistant_msg = chunk

            if assistant_msg and len(assistant_msg) > 10:
                pairs.append({"user_msg": user_msg, "assistant_msg": assistant_msg})

        return pairs

    @classmethod
    def from_file(cls, path: str, chunk_size: int = 400, chunk_overlap: int = 50) -> "TrainingDataset":
        """Load a TrainingDataset from a text file.

        Args:
            path: Path to a UTF-8 text file.
            chunk_size: Characters per chunk.
            chunk_overlap: Overlap between chunks.

        Returns:
            TrainingDataset instance.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file is empty or too short.
        """
        text = Path(path).read_text(encoding="utf-8")
        if len(text.strip()) < 50:
            raise ValueError(f"Source text too short for training (need >50 chars, got {len(text.strip())})")
        return cls(source_text=text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    @classmethod
    def from_text(cls, text: str, chunk_size: int = 400, chunk_overlap: int = 50) -> "TrainingDataset":
        """Create a TrainingDataset from raw text.

        Args:
            text: Raw source text.
            chunk_size: Characters per chunk.
            chunk_overlap: Overlap between chunks.

        Returns:
            TrainingDataset instance.

        Raises:
            ValueError: If text is too short.
        """
        if len(text.strip()) < 50:
            raise ValueError(f"Source text too short for training (need >50 chars, got {len(text.strip())})")
        return cls(source_text=text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    @property
    def source_length(self) -> int:
        return len(self.source_text)

    def __repr__(self) -> str:
        return f"TrainingDataset(chunks={self.num_chunks}, source_len={self.source_length})"
