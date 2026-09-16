"""Backward-compatibility shim — imports from the new ``domain.soul`` package."""

from domain.soul import (
    HDMemoryStore,
    QuantumCognitiveEngine,
    SentimentAnalyzer,
)

__all__ = [
    "SentimentAnalyzer",
    "HDMemoryStore",
    "QuantumCognitiveEngine",
]
