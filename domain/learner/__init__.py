"""learner — Continual learner (ingests data, fine-tunes incrementally).

Public API:
    ContinualLearner, get_learner, extract_and_store, enrich_with_knowledge
"""

from domain.learner._internal.continual import (
    ContinualLearner,
    get_learner,
)
from domain.learner._internal.entity_extractor import extract_and_store
from domain.learner._internal.knowledge_augmenter import enrich_with_knowledge

__all__ = [
    "ContinualLearner",
    "enrich_with_knowledge",
    "extract_and_store",
    "get_learner",
]
