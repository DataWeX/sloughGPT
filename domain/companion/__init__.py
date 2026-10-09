"""companion — AI companion system for natural conversation.

Public API:
    ResponseStyle, CompanionTraits, ConversationContext
    CompanionSystem, get_companion, create_companion
"""

from domain.companion._internal.companion import (
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
