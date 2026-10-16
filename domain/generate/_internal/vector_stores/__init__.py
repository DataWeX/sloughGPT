"""
Provider implementations for ``VectorStore`` ABC.

Each module implements the interface defined in ``domain.generate._internal.vector_store.VectorStore``.

Re-exports for backward compatibility::
    from domain.generate._internal.vector_stores import PineconeVectorStore
    from domain.generate._internal.vector_stores import ChromaDBVectorStore
"""

from __future__ import annotations

from .chromadb_store import ChromaDBVectorStore as ChromaDBVectorStore
from .pinecone_store import PineconeVectorStore as PineconeVectorStore
