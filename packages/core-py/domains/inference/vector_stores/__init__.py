"""Backward-compatibility shim — imports from the new ``domain.inference._internal.vector_stores`` package."""

from domain.inference._internal.vector_stores import PineconeVectorStore, ChromaDBVectorStore

__all__ = ["PineconeVectorStore", "ChromaDBVectorStore"]
