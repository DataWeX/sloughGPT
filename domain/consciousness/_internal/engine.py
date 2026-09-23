"""Backward-compatibility shim — see ``domain.cognition._internal.consciousness.engine``."""

from domain.cognition._internal.consciousness.engine import *  # noqa: F401,F403
from domain.cognition._internal.consciousness.engine import (
    ConsciousnessEngine,
    get_consciousness,
    reset_consciousness,
)

__all__ = ["ConsciousnessEngine", "get_consciousness", "reset_consciousness"]
