"""learner — Continual learner (ingests data, fine-tunes incrementally).

Public API:
    ContinualLearner, get_learner
"""

from domain.learner._internal.continual import (
    ContinualLearner,
    get_learner,
)

__all__ = [
    "ContinualLearner",
    "get_learner",
]
