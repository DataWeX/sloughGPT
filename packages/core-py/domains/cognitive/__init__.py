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
from .spaced_repetition import SpacedRepetitionScheduler, LearningItem, Difficulty, MemoryStrength
from .knowledge_graph import KnowledgeGraph, KnowledgeNode, KnowledgeEdge, RelationType, Confidence
from .metacognition_impl import Metacognition, SelfAssessment, Contradiction, ContradictionType

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
    "SpacedRepetitionScheduler",
    "LearningItem",
    "Difficulty",
    "MemoryStrength",
    "KnowledgeGraph",
    "KnowledgeNode",
    "KnowledgeEdge",
    "RelationType",
    "Confidence",
    "Metacognition",
    "SelfAssessment",
    "Contradiction",
    "ContradictionType",
    # RAG
    "ProductionRAG",
    "BM25Indexer",
    "TextChunk",
    "RetrievalResult",
    "HybridRetriever",
    "HallucinationDetector",
]
