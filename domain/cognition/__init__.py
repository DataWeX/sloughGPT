"""cognition — Reasoning, thinking, creativity, RAG.

The model's reasoning engine. Self-awareness and personality live
in domain.metacognition (separate).

Public API:
    CognitiveDomain, CognitiveException
    CognitiveCore, ThinkingMode, ReasoningType, ThoughtProcess
    CreativeIdea, ReasoningChain, CognitiveProcessor
    RAGService, KGTrainingPipeline, get_rag_service, is_rag_service_ready
"""

from domain.cognition._internal.base import (
    CognitiveDomain,
    CognitiveException,
)
from domain.cognition._internal.core import (
    CognitiveCore,
    CreativeIdea,
    ReasoningChain,
    ReasoningType,
    ThinkingMode,
    ThoughtProcess,
)
from domain.cognition._internal.processor import (
    CognitiveProcessor,
)
from domain.cognition._internal.rag_service import (
    KGTrainingPipeline,
    RAGService,
    get_rag_service,
    is_rag_service_ready,
)

__all__ = [
    "CognitiveDomain",
    "CognitiveException",
    "CognitiveCore",
    "ThinkingMode",
    "ReasoningType",
    "ThoughtProcess",
    "CreativeIdea",
    "ReasoningChain",
    "CognitiveProcessor",
    "RAGService",
    "KGTrainingPipeline",
    "get_rag_service",
    "is_rag_service_ready",
]
