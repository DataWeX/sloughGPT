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
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from .rag import RAGEngine, SLOKnowledgeGraph

__all__ = [
    "SpacedRepetitionScheduler",
    "IpcChannel", "IpcConfig", "is_rust_available",
    "RAGEngine", "SLOKnowledgeGraph",
]
