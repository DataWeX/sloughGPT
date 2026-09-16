"""Backward-compatibility shim — imports from the new ``domain.cognitive`` package."""

from domain.cognitive import (
    CognitiveCore,
    CognitiveDomain,
    CognitiveException,
    CognitiveProcessor,
    CreativeIdea,
    ReasoningChain,
    ReasoningType,
    ThinkingMode,
    ThoughtProcess,
)
from domain.cognitive._internal import rag_service  # noqa: F401

__all__ = [
    "CognitiveDomain",
    "CognitiveException",
    "CognitiveProcessor",
    "CognitiveCore",
    "ThinkingMode",
    "ReasoningType",
    "ThoughtProcess",
    "CreativeIdea",
    "ReasoningChain",
]
