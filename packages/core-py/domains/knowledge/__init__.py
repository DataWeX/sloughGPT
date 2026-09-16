"""Backward-compatibility shim — imports from the new ``domain.knowledge`` package."""

from domain.knowledge import (
    DataFilter,
    KnowledgeFact,
    KnowledgeIngestor,
    KnowledgeMemory,
    get_data_filter,
    get_knowledge_ingestor,
    get_knowledge_memory,
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
