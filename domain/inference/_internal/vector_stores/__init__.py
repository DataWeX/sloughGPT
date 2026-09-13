from __future__ import annotations

"""
Provider implementations for ``VectorStore`` ABC.

Each module implements the interface defined in ``domains.inference.vector_store.VectorStore``.

Re-exports for backward compatibility::
    from domain.inference._internal.vector_stores import PineconeVectorStore
    from domain.inference._internal.vector_stores import ChromaDBVectorStore
"""

from .pinecone_store import PineconeVectorStore as PineconeVectorStore
from .chromadb_store import ChromaDBVectorStore as ChromaDBVectorStore
