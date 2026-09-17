"""Backward-compatibility shim — imports from the new ``domain.cognition`` package."""

from domain.cognition import (
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
from domain.cognition._internal import rag_service  # noqa: F401

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
