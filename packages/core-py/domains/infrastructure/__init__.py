"""
Infrastructure package exports.

Provides:
  - MeaningTags — fixed semantic reference vectors for embedding space
  - EmbeddingService — foundational base layer for vector operations
  - TruthLabeler — rule-based first-glance text classification
  - TruthMaintainer — self-retrain on misclassified texts
  - SpacedRepetitionScheduler — memory scheduling
  - IpcChannel, IpcConfig, is_rust_available — inter-process communication
  - ConfigManager, AppConfig, get_config — typed config with env override
  - EventBus — typed async pub/sub
  - TaskQueue, InProcessTaskQueue — async priority queue
  - LifecycleManager — ordered startup/shutdown with health gates
  - DataRepository — domain-specific CRUD repos with pluggable backends
    (SessionRepository, ConversationRepository, FeedbackRepository,
     KnowledgeRepository, DatasetRepository, AgentRepository, TrainingJobRepository)
  - DataModels — typed dataclasses for all persisted entities

RAGEngine is at domains.cognitive.rag (deprecated here).
"""
from .anchor_store import MeaningTags, get_default_meaning_tags
from .embedding_service import EmbeddingService, get_embedding_service
from .truth_labeler import TruthLabeler, get_truth_labeler
from .truth_maintainer import TruthMaintainer, get_truth_maintainer
from .spaced_repetition_engine import SpacedRepetitionScheduler
from .ipc import IpcChannel, IpcConfig, is_rust_available
from .lifecycle import (
    LifecycleManager,
    LifecyclePhase,
    StartupHook,
    ShutdownHook,
    get_lifecycle_manager,
)
from .data_models import (
    AgentData,
    ConversationData,
    DatasetData,
    FeedbackData,
    FeedbackStatsData,
    KnowledgeFactData,
    KnowledgeEntryData,
    SessionContextData,
    SessionData,
    TrainingJobData,
)
from .data_repository import (
    AgentRepository,
    ConversationRepository,
    DatasetRepository,
    FeedbackRepository,
    KnowledgeRepository,
    RepositoryFactory,
    SessionContextRepository,
    SessionRepository,
    SyncSQLiteRepository,
    TrainingJobRepository,
    get_repository_factory,
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
    "MeaningTags", "get_default_meaning_tags",
    "EmbeddingService", "get_embedding_service",
    "TruthLabeler", "get_truth_labeler",
    "TruthMaintainer", "get_truth_maintainer",
    "SpacedRepetitionScheduler",
    "IpcChannel", "IpcConfig", "is_rust_available",
    "RAGEngine", "SLOKnowledgeGraph",

    # Data Models
    "SessionData", "ConversationData", "FeedbackData", "FeedbackStatsData",
    "KnowledgeFactData", "KnowledgeEntryData", "DatasetData", "AgentData",
    "TrainingJobData", "SessionContextData",

    # Data Repositories
    "SessionRepository", "ConversationRepository", "FeedbackRepository",
    "KnowledgeRepository", "DatasetRepository", "AgentRepository",
    "TrainingJobRepository", "SessionContextRepository",

    # Repository infrastructure
    "RepositoryFactory", "SyncSQLiteRepository", "get_repository_factory",
]
