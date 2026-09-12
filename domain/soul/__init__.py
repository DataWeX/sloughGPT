"""soul — Cognitive engine, HD memory, quantum processing.

Public API:
    SentimentAnalyzer, HDMemoryStore, QuantumCognitiveEngine
"""

from domain.soul._internal.cognitive import SentimentAnalyzer
from domain.soul._internal.hd_memory import HDMemoryStore
from domain.soul._internal.quantum import QuantumCognitiveEngine

__all__ = [
    "SentimentAnalyzer",
    "HDMemoryStore",
    "QuantumCognitiveEngine",
]
