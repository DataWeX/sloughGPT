"""Backward-compatibility shim — imports from the canonical ``packages/core-py/domains/knowledge`` package."""

from domain.knowledge._internal.data_filter import DataFilter, get_data_filter
from domain.knowledge._internal.knowledge import (
    KnowledgeFact,
    KnowledgeIngestor,
    KnowledgeMemory,
    get_knowledge_ingestor,
    get_knowledge_memory,
)
from domain.knowledge._internal.knowledge_ops import (
    AutoCategorizer,
    BulkProcessor,
    DuplicateDetector,
    FileIndex,
    KnowledgeGapDetector,
)
from domain.knowledge.engine import KnowledgeEngine, get_knowledge_engine

__all__ = [
    "KnowledgeEngine",
    "get_knowledge_engine",
    "KnowledgeFact",
    "KnowledgeMemory",
    "KnowledgeIngestor",
    "get_knowledge_memory",
    "get_knowledge_ingestor",
    "DataFilter",
    "get_data_filter",
    "AutoCategorizer",
    "BulkProcessor",
    "DuplicateDetector",
    "FileIndex",
    "KnowledgeGapDetector",
]
