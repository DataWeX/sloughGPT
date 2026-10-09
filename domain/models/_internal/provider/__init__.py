"""Provider package — well-structured, backward compatible.

Re-exports the public API of the legacy provider.py so
`from domain.models._internal.provider import X` keeps working.
"""

from __future__ import annotations

from .processors.knowledge import KnowledgeProcessor
from .processors.personality import PersonalityProcessor
from .processors.style import StyleProcessor
from .processors.tool_use import ToolDef, ToolUseProcessor
from .processors.vision import VisionProcessor
from .protocols import ChatMessage, MessageProcessor, ModelCapabilities, ModelProvider
from .registry import (
    _processors,
    _providers,
    apply_processors,
    attach_process_guard_to_provider,
    clear_providers,
    get_processor,
    get_provider,
    list_processors,
    list_providers,
    register_processor,
    register_provider,
)
from .router import ProviderRouter
from .setup import _server_from_provider, setup_providers, update_personality_traits
from .slo_transformer import SloTransformerProvider

__all__ = [
    "ChatMessage",
    "ModelCapabilities",
    "ModelProvider",
    "MessageProcessor",
    "register_provider",
    "get_provider",
    "list_providers",
    "clear_providers",
    "attach_process_guard_to_provider",
    "register_processor",
    "get_processor",
    "list_processors",
    "apply_processors",
    "VisionProcessor",
    "KnowledgeProcessor",
    "ToolUseProcessor",
    "ToolDef",
    "PersonalityProcessor",
    "StyleProcessor",
    "ProviderRouter",
    "SloTransformerProvider",
    "setup_providers",
    "update_personality_traits",
]
