"""
Cognitive Architecture Domain

Components for cognitive processing, memory management, reasoning,
learning, and creativity.
"""

from __future__ import annotations

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

__all__ = [
    "CognitiveDomain",
    "CognitiveProcessor",
    "CognitiveCore",
    "ThinkingMode",
    "ReasoningType",
    "ThoughtProcess",
    "CreativeIdea",
    "ReasoningChain",
]
