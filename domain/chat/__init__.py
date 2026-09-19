"""chat — Chat domain for message handling.

CCGT mapping: Core layer. Gateway calls in, Cognitive feeds in (processors,
memory), Training improves it (SloChatTrainer adapters).

Public API:
    ChatRequest, ChatResponse, ChatDomain, get_chat_domain
    ChatManager, get_chat_manager, reset_chat_manager
"""

from domain.chat._internal.domain import (
    ChatDomain,
    ChatRequest,
    ChatResponse,
    get_chat_domain,
)
from domain.chat._internal.manager import (
    ChatManager,
    get_chat_manager,
    reset_chat_manager,
)

__all__ = [
    "ChatRequest",
    "ChatResponse",
    "ChatDomain",
    "get_chat_domain",
    "ChatManager",
    "get_chat_manager",
    "reset_chat_manager",
]
