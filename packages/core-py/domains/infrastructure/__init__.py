"""
Infrastructure package exports.

Provides:
  - SpacedRepetitionScheduler — memory scheduling
  - IpcChannel, IpcConfig, is_rust_available — inter-process communication
  - ConfigManager, AppConfig, get_config — typed config with env override
  - EventBus — typed async pub/sub
  - TaskQueue, InProcessTaskQueue — async priority queue
  - LifecycleManager — ordered startup/shutdown with health gates

RAGEngine is at domains.cognitive.rag (deprecated here).
"""
from .spaced_repetition_engine import SpacedRepetitionScheduler
from .ipc import IpcChannel, IpcConfig, is_rust_available
from .lifecycle import (
    LifecycleManager,
    LifecyclePhase,
    StartupHook,
    ShutdownHook,
    get_lifecycle_manager,
)

# Lazy import for deprecated rag module
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
