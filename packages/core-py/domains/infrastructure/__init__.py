"""
Infrastructure package exports.

Provides:
  - SpacedRepetitionScheduler — memory scheduling
  - IpcChannel, IpcConfig, is_rust_available — inter-process communication

RAGEngine is at domains.cognitive.rag (deprecated here).
Experiment tracking and model versioning are in domains/benchmark/domain.py
and domains/training/status.py respectively.
"""
from .spaced_repetition_engine import SpacedRepetitionScheduler
from .ipc import IpcChannel, IpcConfig, is_rust_available

# Lazy import for deprecated rag module — avoids heavy import chain on startup
def __getattr__(name):
    if name in ("RAGEngine", "SLOKnowledgeGraph"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from .rag import RAGEngine, SLOKnowledgeGraph
        return RAGEngine if name == "RAGEngine" else SLOKnowledgeGraph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "SpacedRepetitionScheduler",
    "IpcChannel", "IpcConfig", "is_rust_available",
    "RAGEngine", "SLOKnowledgeGraph",
]
