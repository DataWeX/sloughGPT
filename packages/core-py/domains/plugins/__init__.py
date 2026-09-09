"""
Plugin Framework — load and manage external plugins.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("slo.plugins")


@dataclass
class PluginMetadata:
    """Metadata for a loaded plugin."""
    name: str
    version: str
    description: str
    author: str
    hooks: List[str] = field(default_factory=list)
    enabled: bool = True


class PluginHook:
    """Represents a hook that plugins can register against."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._handlers: List[Callable] = []

    def register(self, handler: Callable) -> None:
        """Register a handler for this hook."""
        self._handlers.append(handler)
        logger.debug("Registered handler for hook '%s': %s", self.name, handler.__name__)

    def unregister(self, handler: Callable) -> None:
        """Unregister a handler."""
        self._handlers = [h for h in self._handlers if h is not handler]

    def run(self, *args: Any, **kwargs: Any) -> List[Any]:
        """Run all handlers for this hook."""
        results = []
        for handler in self._handlers:
            try:
                result = handler(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.warning("Hook '%s' handler '%s' failed: %s",
                              self.name, handler.__name__, e)
        return results

    async def run_async(self, *args: Any, **kwargs: Any) -> List[Any]:
        """Run all handlers for this hook (async version)."""
        results = []
        for handler in self._handlers:
            try:
                import asyncio
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(*args, **kwargs)
                else:
                    result = handler(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.warning("Hook '%s' handler '%s' failed: %s",
                              self.name, handler.__name__, e)
        return results


class Plugin:
    """Base class for plugins."""

    def __init__(self):
        self.metadata: Optional[PluginMetadata] = None

    def setup(self, context: Dict[str, Any]) -> None:
        """Called when the plugin is loaded."""
        pass

    def teardown(self) -> None:
        """Called when the plugin is unloaded."""
        pass


class PluginManager:
    """Manages loading, unloading, and lifecycle of plugins."""

    def __init__(self, plugin_dirs: Optional[List[Path]] = None):
        self._plugin_dirs = plugin_dirs or []
        self._plugins: Dict[str, Plugin] = {}
        self._hooks: Dict[str, PluginHook] = {}
        self._metadata: Dict[str, PluginMetadata] = {}

    def register_hook(self, name: str, description: str = "") -> PluginHook:
        """Register a new hook point."""
        if name not in self._hooks:
            self._hooks[name] = PluginHook(name, description)
        return self._hooks[name]

    def get_hook(self, name: str) -> Optional[PluginHook]:
        """Get a hook by name."""
        return self._hooks.get(name)

    def load_plugin(self, plugin_class: type, context: Optional[Dict[str, Any]] = None) -> bool:
        """Load a plugin from a class."""
        try:
            plugin = plugin_class()
            name = plugin_class.__name__

            metadata = PluginMetadata(
                name=name,
                version=getattr(plugin_class, '__version__', '0.0.1'),
                description=getattr(plugin_class, '__doc__', '') or '',
                author=getattr(plugin_class, '__author__', 'Unknown'),
            )
            plugin.metadata = metadata

            plugin.setup(context or {})
            self._plugins[name] = plugin
            self._metadata[name] = metadata
            logger.info("Loaded plugin: %s v%s", name, metadata.version)
            return True
        except Exception as e:
            logger.error("Failed to load plugin %s: %s", plugin_class.__name__, e)
            return False

    def unload_plugin(self, name: str) -> bool:
        """Unload a plugin by name."""
        plugin = self._plugins.get(name)
        if plugin is None:
            return False
        try:
            plugin.teardown()
            del self._plugins[name]
            del self._metadata[name]
            logger.info("Unloaded plugin: %s", name)
            return True
        except Exception as e:
            logger.error("Failed to unload plugin %s: %s", name, e)
            return False

    def get_plugin(self, name: str) -> Optional[Plugin]:
        """Get a loaded plugin by name."""
        return self._plugins.get(name)

    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all loaded plugins with metadata."""
        result = []
        for name, meta in self._metadata.items():
            result.append({
                "name": meta.name,
                "version": meta.version,
                "description": meta.description,
                "author": meta.author,
                "hooks": meta.hooks,
                "enabled": meta.enabled,
            })
        return result

    def load_from_directory(self, directory: Path) -> int:
        """Load all plugins from a directory."""
        if not directory.exists():
            return 0

        loaded = 0
        for py_file in directory.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            try:
                module_name = py_file.stem
                spec = importlib.util.spec_from_file_location(
                    f"plugins.{module_name}", py_file
                )
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    if hasattr(mod, "plugin_class"):
                        if self.load_plugin(mod.plugin_class):
                            loaded += 1
            except Exception as e:
                logger.warning("Failed to load plugin from %s: %s", py_file, e)
        return loaded


# Singleton
_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager


__all__ = [
    "Plugin",
    "PluginHook",
    "PluginManager",
    "PluginMetadata",
    "get_plugin_manager",
]
