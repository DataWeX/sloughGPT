"""Backward-compatibility shim — see ``domain.cognition._internal.consciousness.meta_cognition``."""

from domain.cognition._internal.consciousness.meta_cognition import *  # noqa: F401,F403
from domain.cognition._internal.consciousness.meta_cognition import (
    MetaCognition,
    MetaCognitiveReport,
)

__all__ = ["MetaCognition", "MetaCognitiveReport"]
