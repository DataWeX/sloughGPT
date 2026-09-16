"""Backward-compatibility shim — imports from the new ``domain.companion`` package."""

from domain.companion import (
    CompanionSystem,
    CompanionTraits,
    ConversationContext,
    ResponseStyle,
    create_companion,
    get_companion,
)

__all__ = [
    "ResponseStyle",
    "CompanionTraits",
    "ConversationContext",
    "CompanionSystem",
    "get_companion",
    "create_companion",
]
