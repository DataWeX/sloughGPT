"""Provider + processor registries — module-level, no per-call imports."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("slo.models.provider")

_providers: dict[str, Any] = {}
_processors: dict[str, Any] = {}


def register_provider(name: str, provider) -> None:
    _providers[name] = provider


def get_provider(name: str):
    return _providers.get(name)


def list_providers() -> list[str]:
    return list(_providers.keys())


def clear_providers() -> None:
    _providers.clear()


def attach_process_guard_to_provider(process_guard: Any) -> bool:
    from .registry import get_provider as _get

    provider = _get("slonet-native")
    if provider is None:
        return False
    server = getattr(provider, "get_server", lambda: None)()
    if server is None:
        return False
    setter = getattr(server, "set_process_guard", None)
    if setter is None:
        return False
    setter(process_guard)
    return True


def register_processor(name: str, processor) -> None:
    _processors[name] = processor


def get_processor(name: str):
    return _processors.get(name)


def list_processors() -> list[str]:
    return list(_processors.keys())


async def apply_processors(messages: list, processors: list) -> list:
    for proc in processors:
        try:
            messages = await proc.process(messages)
        except Exception as e:
            logger.warning("Processor %s failed: %s", type(proc).__name__, e, extra={"tag": "MODEL"})
    return messages
