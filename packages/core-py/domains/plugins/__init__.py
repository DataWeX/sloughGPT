"""Backward-compatibility shim — imports from the new ``domain.plugins`` package."""

from domain.plugins import (
    Plugin,
    PluginHook,
    PluginManager,
    PluginMetadata,
    get_plugin_manager,
)

__all__ = [
    "Plugin",
    "PluginHook",
    "PluginManager",
    "PluginMetadata",
    "get_plugin_manager",
]
