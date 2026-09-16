"""Backward-compatibility shim — imports from the new ``domain.tools`` package."""

from domain.tools import (
    ToolOption,
    ToolParam,
    ToolProfile,
    ToolsEngine,
    get_tool_profile,
    get_tools_engine,
)

__all__ = [
    "ToolsEngine",
    "ToolOption",
    "ToolParam",
    "ToolProfile",
    "get_tool_profile",
    "get_tools_engine",
]
