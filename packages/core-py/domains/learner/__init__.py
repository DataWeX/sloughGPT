"""Backward-compatibility shim — imports from the new ``domain.learner`` package."""

from domain.learner import (
    ContinualLearner,
    get_learner,
)

__all__ = [
    "ContinualLearner",
    "get_learner",
]
