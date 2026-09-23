"""Backward-compatibility shim — see ``domain.cognition._internal.consciousness.config``."""

from domain.cognition._internal.consciousness.config import *  # noqa: F401,F403
from domain.cognition._internal.consciousness.config import ConsciousnessConfig

__all__ = ["ConsciousnessConfig"]
