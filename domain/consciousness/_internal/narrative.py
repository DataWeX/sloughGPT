"""Backward-compatibility shim — see ``domain.cognition._internal.consciousness.narrative``."""

from domain.cognition._internal.consciousness.narrative import *  # noqa: F401,F403
from domain.cognition._internal.consciousness.narrative import NarrativeGenerator

__all__ = ["NarrativeGenerator"]
