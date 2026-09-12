"""plugins — Plugin framework for loading and managing external plugins.

Public API:
    Plugin, PluginHook, PluginManager, PluginMetadata, get_plugin_manager
"""

from domain.plugins._internal.plugins import (
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
