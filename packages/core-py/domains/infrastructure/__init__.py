"""
Infrastructure package exports.

Provides:
  - MeaningTags — fixed semantic reference vectors for embedding space
  - EmbeddingService — foundational base layer for vector operations
  - TruthLabeler — rule-based first-glance text classification
  - TruthMaintainer — self-retrain on misclassified texts
  - SpacedRepetitionScheduler — memory scheduling
  - ConfigManager, AppConfig, get_config — typed config with env override
  - EventBus — typed async pub/sub
  - TaskQueue, InProcessTaskQueue — async priority queue
  - LifecycleManager — ordered startup/shutdown with health gates
"""
from .anchor_store import MeaningTags, get_default_meaning_tags
from .embedding_service import EmbeddingService, get_embedding_service
from .truth_labeler import TruthLabeler, get_truth_labeler
from .truth_maintainer import TruthMaintainer, get_truth_maintainer
from .spaced_repetition_engine import SpacedRepetitionScheduler
from .lifecycle import (
    LifecycleManager,
    LifecyclePhase,
    StartupHook,
    ShutdownHook,
    get_lifecycle_manager,
)

__all__ = [
    "MeaningTags", "get_default_meaning_tags",
    "EmbeddingService", "get_embedding_service",
    "TruthLabeler", "get_truth_labeler",
    "TruthMaintainer", "get_truth_maintainer",
    "SpacedRepetitionScheduler",
]
