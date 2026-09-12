"""knowledge — Structured fact storage, ingestion pipeline, and knowledge graph.

Public API:
    KnowledgeFact       — data class for a single fact
    KnowledgeMemory     — vector-store backed fact storage (add/query/search)
    KnowledgeIngestor   — multi-source ingestion (RSS, web search, URLs)
    DataFilter          — quality/relevance gate
    get_knowledge_memory()  — singleton accessor
    get_knowledge_ingestor() — singleton accessor
"""

from knowledge._internal.data_filter import DataFilter, get_data_filter
from knowledge._internal.knowledge import (
    KnowledgeFact,
    KnowledgeIngestor,
    KnowledgeMemory,
    get_knowledge_ingestor,
    get_knowledge_memory,
)

__all__ = [
    "KnowledgeFact",
    "KnowledgeMemory",
    "KnowledgeIngestor",
    "DataFilter",
    "get_knowledge_memory",
    "get_knowledge_ingestor",
    "get_data_filter",
]
