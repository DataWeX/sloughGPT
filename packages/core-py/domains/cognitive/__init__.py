"""Backward-compatibility shim — imports from the new ``domain.cognitive`` package."""

from domain.cognitive import (
    CognitiveDomain,
    CognitiveException,
    CognitiveCore,
    ThinkingMode,
    ReasoningType,
    ThoughtProcess,
    CreativeIdea,
    ReasoningChain,
    CognitiveProcessor,
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
