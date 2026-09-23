"""knowledge — high-level knowledge API.

The concrete ``KnowledgeMemory`` store and ``KnowledgeIngestor`` loader are
canonical in ``domain.memory`` (single source of truth). This package is the
thin ``KnowledgeEngine`` facade plus the knowledge-specific ops that keep
their meaning here (data filtering, search indexes, dedup, gap detection).
"""

from domain.knowledge._internal.data_filter import DataFilter, get_data_filter
from domain.knowledge._internal.knowledge_ops import (
    AutoCategorizer,
    BulkProcessor,
    DuplicateDetector,
    FileIndex,
    KnowledgeGapDetector,
)
from domain.knowledge.engine import KnowledgeEngine, get_knowledge_engine
from domain.memory import (
    KnowledgeFact,
    KnowledgeIngestor,
    KnowledgeMemory,
    get_knowledge_ingestor,
    get_knowledge_memory,
)
from domain.memory._internal.knowledge_store import (
    _extract_facts_from_text as _extract_facts_from_text,
)
from domain.memory._internal.knowledge_store import _extract_topics as _extract_topics

__all__ = [
    "KnowledgeEngine",
    "get_knowledge_engine",
    "KnowledgeFact",
    "KnowledgeMemory",
    "KnowledgeIngestor",
    "get_knowledge_memory",
    "get_knowledge_ingestor",
    "_extract_facts_from_text",
    "_extract_topics",
    "DataFilter",
    "get_data_filter",
    "AutoCategorizer",
    "BulkProcessor",
    "DuplicateDetector",
    "FileIndex",
    "KnowledgeGapDetector",
]
