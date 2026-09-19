"""Backward-compatibility shim — imports from the new ``domain.chat`` package."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from domain.chat import (
        ChatDomain,
        ChatManager,
        ChatRequest,
        ChatResponse,
        get_chat_domain,
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

_lazy_imports = {
    "ChatRequest": "domain.chat",
    "ChatResponse": "domain.chat",
    "ChatDomain": "domain.chat",
    "get_chat_domain": "domain.chat",
    "ChatManager": "domain.chat",
    "get_chat_manager": "domain.chat",
    "reset_chat_manager": "domain.chat",
    "domain": "domain.chat._internal",
}


def __getattr__(name: str):
    if name in _lazy_imports:
        import importlib

        mod = importlib.import_module(_lazy_imports[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
