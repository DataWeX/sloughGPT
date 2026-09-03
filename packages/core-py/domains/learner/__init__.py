"""Continual learner — ingests web + conversation data, fine-tunes SloNet incrementally."""

from __future__ import annotations

from .continual import ContinualLearner, get_learner

__all__ = ["ContinualLearner", "get_learner"]
