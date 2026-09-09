"""
SLO Slo Module - The Evolving Core Intelligence

Modules:
- cognitive: Sentiment analysis (standalone)
- hd_memory: Hyperdimensional memory store
- quantum: Quantum cognitive engine
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cognitive import SentimentAnalyzer
    from .hd_memory import HDMemoryStore
    from .quantum import QuantumCognitiveEngine

__all__ = [
    "SentimentAnalyzer",
    "HDMemoryStore",
    "QuantumCognitiveEngine",
]

_lazy_imports = {
    "SentimentAnalyzer": ".cognitive",
    "HDMemoryStore": ".hd_memory",
    "QuantumCognitiveEngine": ".quantum",
}


def __getattr__(name: str):
    if name in _lazy_imports:
        import importlib
        mod = importlib.import_module(_lazy_imports[name], __package__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
