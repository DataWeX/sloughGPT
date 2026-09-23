"""Backward-compatibility shim — see ``domain.cognition._internal.consciousness.qualia``."""

from domain.cognition._internal.consciousness.qualia import *  # noqa: F401,F403
from domain.cognition._internal.consciousness.qualia import QualiaEngine, QualiaState

__all__ = ["QualiaEngine", "QualiaState"]
