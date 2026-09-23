"""Backward-compatibility shim — see ``domain.cognition._internal.consciousness.personality``."""

from domain.cognition._internal.consciousness.personality import *  # noqa: F401,F403
from domain.cognition._internal.consciousness.personality import (
    PersonalityManager,
    PersonalityProfile,
)

__all__ = ["PersonalityManager", "PersonalityProfile"]
