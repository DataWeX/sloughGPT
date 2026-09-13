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
