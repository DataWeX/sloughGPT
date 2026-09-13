"""Backward-compatibility shim — imports from the new ``domain.learner`` package."""

from domain.learner import (
    ContinualLearner,
    get_learner,
)
from domain.learner._internal import continual, knowledge  # noqa: F401

__all__ = [
    "ContinualLearner",
    "get_learner",
    "continual",
]
