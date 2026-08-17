"""
RAG Service — Production RAG with real embeddings and document persistence.

Bridges the production-grade ``ProductionRAG`` from ``cognitive/rag.py`` into
the chat pipeline by:

1. Replacing simulated random embeddings with the project's n-gram TF-IDF
   embedder (zero downloads, works on CPU).
2. Persisting ingested documents to disk so they survive server restarts.
3. Providing a singleton accessor for the chat router.
"""

import json
import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from domains.cognitive.rag import (
    ProductionRAG,
    HybridRetriever,
    TextChunk,
)
from domains.inference.vector_store import simple_embed

logger = logging.getLogger("slo.rag_service")

_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "rag_store"
_DOCUMENTS_FILE = _DATA_DIR / "documents.jsonl"


class ProductionRAGWithRealEmbeddings(ProductionRAG):
    """ProductionRAG subclass that uses the project's real embedder."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.retriever._get_embedding = self._real_embed

    @staticmethod
    def _real_embed(text: str) -> np.ndarray:
        """Get a real 384-dim embedding via the project's n-gram TF-IDF embedder."""
        vec = simple_embed(text, dimension=384)
        return np.asarray(vec, dtype=np.float32)


class RAGService:
    """High-level RAG service with document persistence and query."""

    def __init__(self):
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.rag = ProductionRAGWithRealEmbeddings()
        self._documents: List[Dict[str, Any]] = []
        self._load_documents()

    def _load_documents(self):
        """Load persisted documents from disk."""
        if not _DOCUMENTS_FILE.exists():
            return
        try:
            with open(_DOCUMENTS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    doc = json.loads(line)
                    self._documents.append(doc)
                    self.rag.add_document(
                        content=doc["content"],
                        metadata=doc.get("metadata", {}),
                        chunk_size=doc.get("chunk_size", 512),
                        overlap=doc.get("overlap", 50),
                    )
            logger.info("Loaded %d documents from RAG store", len(self._documents))
        except Exception as e:
            logger.warning("Failed to load RAG documents: %s", e)

    def _save_document(self, doc: Dict[str, Any]):
        """Append a single document to the JSONL file."""
        try:
            with open(_DOCUMENTS_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("Failed to persist RAG document: %s", e)

    def add_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_size: int = 512,
        overlap: int = 50,
    ) -> List[str]:
        """Ingest a document into the RAG index and persist it.

        Args:
            content: Raw text to ingest.
            metadata: Optional metadata dict (source, topic, etc.).
            chunk_size: Max tokens per chunk.
            overlap: Overlap between adjacent chunks.

        Returns:
            List of chunk IDs created.
        """
        metadata = metadata or {"source": "user"}
        chunk_ids = self.rag.add_document(
            content=content,
            metadata=metadata,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        self._documents.append({
            "content": content,
            "metadata": metadata,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "chunk_ids": chunk_ids,
            "added_at": time.time(),
        })
        self._save_document(self._documents[-1])
        logger.info(
            "Ingested document (%d chars → %d chunks) into RAG index",
            len(content), len(chunk_ids),
        )
        return chunk_ids

    def query(
        self,
        question: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Query the RAG index for relevant context.

        Args:
            question: User question to search for.
            top_k: Maximum number of results.

        Returns:
            Dict with 'context' (concatenated text), 'results' (ranked list), 'num_results'.
        """
        return self.rag.query(question, top_k=top_k, return_context=True)

    def verify_and_ground(
        self,
        generated_text: str,
        question: str,
    ) -> Dict[str, Any]:
        """Verify generated text against the RAG index and add citations.

        Args:
            generated_text: The model's response to verify.
            question: The original user question.

        Returns:
            Dict with 'verification', 'citations', 'confidence', 'is_verified'.
        """
        return self.rag.verify_and_ground(generated_text, question)

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all ingested documents (metadata only, no content)."""
        return [
            {
                "metadata": doc.get("metadata", {}),
                "chunk_size": doc.get("chunk_size", 512),
                "num_chunks": len(doc.get("chunk_ids", [])),
                "added_at": doc.get("added_at", 0),
            }
            for doc in self._documents
        ]

    def clear(self) -> int:
        """Clear the entire RAG index and persisted documents.

        Returns:
            Number of documents removed.
        """
        count = len(self._documents)
        self._documents.clear()
        self.rag = ProductionRAGWithRealEmbeddings()
        try:
            if _DOCUMENTS_FILE.exists():
                _DOCUMENTS_FILE.unlink()
        except Exception:
            pass
        logger.info("Cleared RAG index (%d documents removed)", count)
        return count

    def stats(self) -> Dict[str, Any]:
        """Return RAG index statistics."""
        return {
            "total_documents": len(self._documents),
            "total_chunks": len(self.rag.retriever.chunks),
            "index_size": len(self.rag.retriever.bm25.inverted_index),
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """Get or create the singleton RAGService."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
