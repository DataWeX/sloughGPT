"""soul — Cognitive engine, HD memory, quantum processing.

Public API:
    SentimentAnalyzer, HDMemoryStore, QuantumCognitiveEngine
"""

from soul._internal.cognitive import SentimentAnalyzer
from soul._internal.hd_memory import HDMemoryStore
from soul._internal.quantum import QuantumCognitiveEngine

__all__ = [
    "SentimentAnalyzer",
    "HDMemoryStore",
    "QuantumCognitiveEngine",
]
