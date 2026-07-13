"""
SLO Slo Module - The Evolving Core Intelligence

Modules:
- cognitive: Sentiment analysis (standalone)
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cognitive import SentimentAnalyzer

__all__ = [
    "SentimentAnalyzer",
]

_lazy_imports = {
    "SentimentAnalyzer": ".cognitive",
}


def __getattr__(name: str):
    if name in _lazy_imports:
        import importlib
        mod = importlib.import_module(_lazy_imports[name], __package__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
