"""Backward-compatibility shim — re-exports from the local cognitive submodules."""

from domains.cognitive.base import CognitiveDomain, CognitiveException
from domains.cognitive.core import (
    CognitiveCore,
    ThinkingMode,
    ReasoningType,
    ThoughtProcess,
    CreativeIdea,
    ReasoningChain,
)
from domains.cognitive.processor import CognitiveProcessor

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
