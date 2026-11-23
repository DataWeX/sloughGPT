"""learner — Continual learner (ingests data, fine-tunes incrementally).

Public API:
    ContinualLearner, get_learner, extract_and_store, get_knowledge_memory
"""

from domain.knowledge import get_knowledge_memory
from domain.learner._internal.continual import (
    ContinualLearner,
    get_learner,
)
from domain.learner._internal.entity_extractor import extract_and_store

__all__ = [
    "ContinualLearner",
    "get_learner",
    "extract_and_store",
    "get_knowledge_memory",
]
