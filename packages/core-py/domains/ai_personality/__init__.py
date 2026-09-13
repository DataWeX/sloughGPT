"""Backward-compatibility shim — imports from the new ``domain.ai_personality`` package."""

from domain.ai_personality import (
    PersonalityType,
    Personality,
    PersonalityManager,
    get_personality_manager,
    list_personalities,
    PERSONALITIES,
)

_default_manager = get_personality_manager()

__all__ = [
    "PersonalityType",
    "Personality",
    "PersonalityManager",
    "get_personality_manager",
    "list_personalities",
    "PERSONALITIES",
    "_default_manager",
]
