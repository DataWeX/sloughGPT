"""Backward-compatibility shim — imports from the new ``knowledge`` package."""

from knowledge import (
    KnowledgeFact,
    KnowledgeMemory,
    KnowledgeIngestor,
    get_knowledge_memory,
    get_knowledge_ingestor,
    DataFilter,
    get_data_filter,
)

__all__ = [
    "KnowledgeFact",
    "KnowledgeMemory",
    "KnowledgeIngestor",
    "get_knowledge_memory",
    "get_knowledge_ingestor",
    "DataFilter",
    "get_data_filter",
]
