"""Backward-compatibility shim — knowledge store/ingestor now live in ``domain.memory``.

The concrete ``KnowledgeMemory`` store and ``KnowledgeIngestor`` loader were
consolidated into ``domain.memory._internal.knowledge_store`` so that
``domain.memory`` is the single source of truth for memory persistence.
This module re-exports the same symbols so existing ``_internal`` importers
keep working; new code should import from ``domain.knowledge`` (facade) or
``domain.memory`` (canonical owner).
"""

from domain.memory._internal.knowledge_store import FeedSubscription as FeedSubscription
from domain.memory._internal.knowledge_store import KnowledgeFact as KnowledgeFact
from domain.memory._internal.knowledge_store import KnowledgeIngestor as KnowledgeIngestor
from domain.memory._internal.knowledge_store import KnowledgeMemory as KnowledgeMemory
from domain.memory._internal.knowledge_store import (
    _extract_facts_from_text as _extract_facts_from_text,
)
from domain.memory._internal.knowledge_store import _extract_topics as _extract_topics
from domain.memory._internal.knowledge_store import _find_repo_root as _find_repo_root
from domain.memory._internal.knowledge_store import _scrape_article as _scrape_article
from domain.memory._internal.knowledge_store import _search_ddg as _search_ddg
from domain.memory._internal.knowledge_store import _topic_slug as _topic_slug
from domain.memory._internal.knowledge_store import chunk_by_fixed_size as chunk_by_fixed_size
from domain.memory._internal.knowledge_store import chunk_by_heading as chunk_by_heading
from domain.memory._internal.knowledge_store import chunk_by_paragraph as chunk_by_paragraph
from domain.memory._internal.knowledge_store import chunk_by_semantic as chunk_by_semantic
from domain.memory._internal.knowledge_store import chunk_text as chunk_text
from domain.memory._internal.knowledge_store import get_knowledge_ingestor as get_knowledge_ingestor
from domain.memory._internal.knowledge_store import get_knowledge_memory as get_knowledge_memory

__all__ = [
    "KnowledgeFact",
    "FeedSubscription",
    "KnowledgeMemory",
    "KnowledgeIngestor",
    "get_knowledge_memory",
    "get_knowledge_ingestor",
]
