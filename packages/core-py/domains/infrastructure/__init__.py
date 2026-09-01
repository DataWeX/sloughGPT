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

from __future__ import annotations

from .anchor_store import MeaningTags, get_default_meaning_tags
from .embedding_service import EmbeddingService, get_embedding_service, reset_embedding_service
from .truth_labeler import TruthLabeler, get_truth_labeler
from .truth_maintainer import TruthMaintainer, get_truth_maintainer
from .spaced_repetition_engine import SpacedRepetitionScheduler
from .lifecycle import (
    LifecycleManager,
    LifecyclePhase,
    StartupHook,
    ShutdownHook,
    get_lifecycle_manager,
    reset_lifecycle_manager,
)
from .cpu_topology import CpuTopology, detect_topology
from .resource_manager import ResourceManager, ResourceAllocation, get_resource_manager, compute_allocation

__all__ = [
    "MeaningTags", "get_default_meaning_tags",
    "EmbeddingService", "get_embedding_service", "reset_embedding_service",
    "TruthLabeler", "get_truth_labeler",
    "TruthMaintainer", "get_truth_maintainer",
    "SpacedRepetitionScheduler",
    "LifecycleManager", "LifecyclePhase", "StartupHook", "ShutdownHook", "get_lifecycle_manager", "reset_lifecycle_manager",
    "CpuTopology", "detect_topology",
    "ResourceManager", "ResourceAllocation", "get_resource_manager", "compute_allocation",
]
