"""Backward-compatibility shim — see ``domain.cognition._internal.consciousness.self_model``."""

from domain.cognition._internal.consciousness.self_model import *  # noqa: F401,F403
from domain.cognition._internal.consciousness.self_model import (
    Reflection,
    SelfEpisode,
    SelfIdentity,
    SelfModel,
)

__all__ = ["Reflection", "SelfEpisode", "SelfIdentity", "SelfModel"]
