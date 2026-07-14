"""
Cognitive Architecture Domain

Components for cognitive processing, memory management, reasoning,
learning, and creativity.
"""

from .base import CognitiveDomain
from .processor import CognitiveProcessor
from .core import (
    CognitiveCore,
    ThinkingMode,
    ReasoningType,
    ThoughtProcess,
    CreativeIdea,
    ReasoningChain,
)
from .knowledge_graph import KnowledgeGraph, KnowledgeNode, KnowledgeEdge, RelationType, Confidence

# RAG exports (canonical location)
from .rag import (
    ProductionRAG,
    BM25Indexer,
    TextChunk,
    RetrievalResult,
    HybridRetriever,
    HallucinationDetector,
)

__all__ = [
    "CognitiveDomain",
    "CognitiveProcessor",
    "CognitiveCore",
    "ThinkingMode",
    "ReasoningType",
    "ThoughtProcess",
    "CreativeIdea",
    "ReasoningChain",
    "KnowledgeGraph",
    "KnowledgeNode",
    "KnowledgeEdge",
    "RelationType",
    "Confidence",
    # RAG
    "ProductionRAG",
    "BM25Indexer",
    "TextChunk",
    "RetrievalResult",
    "HybridRetriever",
    "HallucinationDetector",
]
