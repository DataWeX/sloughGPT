"""Backward-compatibility shim — imports from the new ``domain.chat`` package."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from domain.chat import (
        ChatRequest,
        ChatResponse,
        ChatDomain,
        get_chat_domain,
    )

__all__ = [
    "ChatRequest",
    "ChatResponse",
    "ChatDomain",
    "get_chat_domain",
]

_lazy_imports = {
    "ChatRequest": "domain.chat",
    "ChatResponse": "domain.chat",
    "ChatDomain": "domain.chat",
    "get_chat_domain": "domain.chat",
}


def __getattr__(name: str):
    if name in _lazy_imports:
        import importlib
        mod = importlib.import_module(_lazy_imports[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
