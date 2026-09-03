from __future__ import annotations

"""
Provider implementations for ``VectorStore`` ABC.

Each module implements the interface defined in ``domains.inference.vector_store.VectorStore``.

Re-exports for backward compatibility::
    from domains.inference.vector_stores import PineconeVectorStore
    from domains.inference.vector_stores import ChromaDBVectorStore
"""

from .pinecone_store import PineconeVectorStore
from .chromadb_store import ChromaDBVectorStore
