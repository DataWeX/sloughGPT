"""cognitive — Cognitive architecture domain.

Public API:
    CognitiveDomain, CognitiveException
    CognitiveCore, ThinkingMode, ReasoningType, ThoughtProcess
    CreativeIdea, ReasoningChain, CognitiveProcessor
"""

from domain.cognitive._internal.base import (
    CognitiveDomain,
    CognitiveException,
)
from domain.cognitive._internal.core import (
    CognitiveCore,
    ThinkingMode,
    ReasoningType,
    ThoughtProcess,
    CreativeIdea,
    ReasoningChain,
)
from domain.cognitive._internal.processor import (
    CognitiveProcessor,
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
]
