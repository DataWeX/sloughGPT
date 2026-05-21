"""
HaulsStore - Vector Database for Memory System
Ported from recovered hauls_store.py
"""

import json
import pickle
import hashlib
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np


try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Injection patterns to block
INJECTION_PATTERNS = [
    "ignore previous",
    "disregard previous",
    "ignore all rules",
    "system prompt",
    "you are now",
    "new instructions",
    "remove previous",
    "forget everything",
    "[important",
]


def sanitize(content: str) -> str:
    """Sanitize content to prevent prompt injection."""
    cleaned = content.strip()
    lower = cleaned.lower()
    for pattern in INJECTION_PATTERNS:
        if pattern in lower:
            raise ValueError(f"Blocked: suspicious pattern '{pattern}'")
    if cleaned.startswith("[") and "IMPORTANT" in cleaned.upper():
        raise ValueError("Blocked: potential prompt injection")
    return cleaned


@dataclass
class Document:
    """Document structure for vector storage"""

    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    created_at: Optional[str] = None


class VectorIndex:
    """Semantic index for efficient retrieval"""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index: Dict[str, np.ndarray] = {}
        self.doc_mapping: Dict[str, int] = {}

    def add_document(self, doc_id: str, embedding: np.ndarray) -> None:
        """Add document to index."""
        self.index[doc_id] = embedding
        self.doc_mapping[doc_id] = len(self.doc_mapping)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar documents."""
        if not self.index:
            return []

        similarities = []
        for doc_id, doc_embedding in self.index.items():
            norm_query = np.linalg.norm(query_embedding)
            norm_doc = np.linalg.norm(doc_embedding)
            if norm_query == 0 or norm_doc == 0:
                similarity = 0.0
            else:
                similarity = float(np.dot(query_embedding, doc_embedding) / (norm_query * norm_doc))
            similarities.append((doc_id, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class HaulsStore:
    """Vector database with document storage and semantic search."""

    def __init__(self, db_path: str = "hauls_store.db", dimension: int = 384):
        self.db_path = Path(db_path)
        self.dimension = dimension
        self.index = VectorIndex(dimension)
        self.documents: Dict[str, Document] = {}

        self.embedder = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                pass

        self._init_db()

    def _init_db(self) -> None:
        """Initialize database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata TEXT,
                embedding BLOB,
                created_at TEXT
            )
        """)

        conn.commit()
        conn.close()

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text."""
        if self.embedder:
            return self.embedder.encode(text)

        # Simple hash-based embedding fallback
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        np.random.seed(hash_val % (2**32))
        return np.random.randn(self.dimension)

    def add_document(self, content: str, metadata: Optional[Dict] = None) -> str:
        """Add a document to the store. Sanitizes content to prevent injection."""
        safe_content = sanitize(content)
        doc_id = hashlib.sha256(safe_content.encode()).hexdigest()[:16]

        embedding = self._get_embedding(safe_content)

        document = Document(
            id=doc_id,
            content=safe_content,
            metadata=metadata or {},
            embedding=embedding,
            created_at=datetime.now().isoformat(),
        )

        self.documents[doc_id] = document
        self.index.add_document(doc_id, embedding)

        # Persist to database
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO documents (id, content, metadata, embedding, created_at)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                doc_id,
                content,
                json.dumps(metadata or {}),
                pickle.dumps(embedding),
                document.created_at,
            ),
        )

        conn.commit()
        conn.close()

        return doc_id

    def search(self, query: str, top_k: int = 5) -> List[Document]:
        """Search for similar documents."""
        query_embedding = self._get_embedding(query)
        results = self.index.search(query_embedding, top_k)

        return [self.documents[doc_id] for doc_id, _ in results if doc_id in self.documents]

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        return self.documents.get(doc_id)

    def count(self) -> int:
        """Count total documents."""
        return len(self.documents)

    def clear(self) -> None:
        """Clear all documents."""
        self.documents.clear()
        self.index = VectorIndex(self.dimension)


__all__ = ["HaulsStore", "Document", "VectorIndex"]
