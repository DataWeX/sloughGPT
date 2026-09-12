"""Backward-compatibility shim — imports from the new ``domain.soul`` package."""

from domain.soul import (
    SentimentAnalyzer,
    HDMemoryStore,
    QuantumCognitiveEngine,
)

__all__ = [
    "SentimentAnalyzer",
    "HDMemoryStore",
    "QuantumCognitiveEngine",
]
