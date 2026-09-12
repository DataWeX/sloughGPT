"""Backward-compatibility shim — imports from the new ``soul`` package."""

from soul import (
    SentimentAnalyzer,
    HDMemoryStore,
    QuantumCognitiveEngine,
)

__all__ = [
    "SentimentAnalyzer",
    "HDMemoryStore",
    "QuantumCognitiveEngine",
]
