"""ai_personality — Personality system for model outputs.

Public API:
    PersonalityType, Personality, PersonalityManager
    get_personality_manager, list_personalities, PERSONALITIES
"""

from domain.ai_personality._internal.personality import (
    PersonalityType,
    Personality,
    PersonalityManager,
    get_personality_manager,
    list_personalities,
    PERSONALITIES,
)

__all__ = [
    "PersonalityType",
    "Personality",
    "PersonalityManager",
    "get_personality_manager",
    "list_personalities",
    "PERSONALITIES",
]
