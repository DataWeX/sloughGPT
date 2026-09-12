"""Tools Domain — declarative everyday tools (writing, translate, rewrite, ...).

The tools are prompt-layer skills that run on top of the inference engine.
Each tool is defined in ``profiles.py`` and exposed through the API router
(``apps/api/server/routers/tools.py``) so the frontend never has to build
prompts itself.

Usage:
    from domains.tools import get_tools_engine

    engine = get_tools_engine()
    profile = engine.get_profile("writing")
    prompt = profile.render_prompt({"action": "write", "tone": "friendly", "text": "..."})
"""

from __future__ import annotations

import threading
from typing import Any

from .profiles import (
    TOOL_PROFILES,
    ToolOption,
    ToolParam,
    ToolProfile,
    get_tool_profile,
)

__all__ = [
    "ToolsEngine",
    "ToolOption",
    "ToolParam",
    "ToolProfile",
    "get_tool_profile",
    "get_tools_engine",
]


class ToolsEngine:
    """Registry + prompt rendering for the everyday tools."""

    def __init__(self) -> None:
        self._profiles = dict(TOOL_PROFILES)

    def list_tools(self) -> list[dict[str, Any]]:
        """Return public metadata for every tool (used by GET /tools)."""
        result = []
        for profile in self._profiles.values():
            result.append(self._public_profile(profile))
        return result

    def get_profile(self, tool_id: str) -> ToolProfile | None:
        """Look up a tool profile by id."""
        return self._profiles.get(tool_id)

    def render_prompt(self, tool_id: str, payload: dict[str, Any]) -> str:
        """Render the user prompt for a tool, merging in default options."""
        profile = self._profiles.get(tool_id)
        if profile is None:
            raise KeyError(f"Unknown tool: {tool_id}")
        merged = self._merge_defaults(profile, payload)
        return profile.render_prompt(merged)

    @staticmethod
    def _merge_defaults(profile: ToolProfile, payload: dict[str, Any]) -> dict[str, Any]:
        merged = dict(payload)
        for key, default in profile.default_options.items():
            if key not in merged or not merged.get(key):
                merged[key] = default
        return merged

    @staticmethod
    def _public_profile(profile: ToolProfile) -> dict[str, Any]:
        return {
            "id": profile.id,
            "name": profile.name,
            "description": profile.description,
            "icon": profile.icon,
            "params": [
                {
                    "id": p.id,
                    "label": p.label,
                    "placeholder": p.placeholder,
                    "multiline": p.multiline,
                    "optional": p.optional,
                }
                for p in profile.params
            ],
            "options": {
                group: [
                    {"id": o.id, "label": o.label, "description": o.description}
                    for o in options
                ]
                for group, options in profile.options.items()
            },
            "default_options": dict(profile.default_options),
            "system_prompt": profile.system_prompt,
            "max_tokens": profile.max_tokens,
        }


_tools_engine: ToolsEngine | None = None
_tools_engine_lock = threading.Lock()


def get_tools_engine() -> ToolsEngine:
    """Get the global tools engine singleton."""
    global _tools_engine
    if _tools_engine is None:
        with _tools_engine_lock:
            if _tools_engine is None:
                _tools_engine = ToolsEngine()
    return _tools_engine