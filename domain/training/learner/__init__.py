"""learner — Continual learner (ingests data, fine-tunes incrementally).

Public API:
    ContinualLearner, get_learner, extract_and_store
"""

from __future__ import annotations

from domain.training.learner._internal.continual import (
    ContinualLearner,
    get_learner,
)
from domain.training.learner._internal.entity_extractor import extract_and_store

__all__ = [
    "ContinualLearner",
    "get_learner",
    "extract_and_store",
]
