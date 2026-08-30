"""
Kernel Module Loader — dynamic addon loading and management.

Provides:
  - ModuleLoader: loads Python modules as kernel addons at runtime
  - ModuleRegistry: tracks loaded modules, their state, and dependencies
  - Hot-reload support for development
"""

from __future__ import annotations

import sys
import importlib
import importlib.util
import logging
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable

from .base import Addon

logger = logging.getLogger("slo.shell.addon_loader")


# ── Module State ──────────────────────────────────────────────────────────────

@dataclass
class ModuleInfo:
    """Metadata for a loaded kernel module."""
    name: str
    path: str
    version: str = "0.0.0"
    description: str = ""
    author: str = ""
    loaded_at: float = 0.0
    load_time_ms: float = 0.0
    state: str = "unloaded"  # unloaded, loading, loaded, error, disabled
    error: str | None = None
    instance: Addon | None = None
    dependencies: list[str] = field(default_factory=list)


# ── Module Loader ─────────────────────────────────────────────────────────────

class ModuleLoader:
    """
    Dynamic kernel module loader.

    Loads Python files as kernel addons, manages their lifecycle,
    and supports hot-reload for development.
    """

    def __init__(self, addon_dirs: list[str | Path] | None = None):
        self._addon_dirs = [Path(d) for d in (addon_dirs or [])]
        self._modules: dict[str, ModuleInfo] = {}
        self._kernel: Any = None
        self._hooks: dict[str, list[Callable]] = {
            "pre_load": [],
            "post_load": [],
            "pre_unload": [],
            "post_unload": [],
        }

    def set_kernel(self, kernel):
        """Set the kernel instance for addon initialization."""
        self._kernel = kernel

    def add_addon_dir(self, path: str | Path):
        """Add a directory to search for addons."""
        self._addon_dirs.append(Path(path))

    # ── Discovery ─────────────────────────────────────────────────────────

    def discover(self) -> list[str]:
        """Discover available modules in addon directories."""
        found = []
        for addon_dir in self._addon_dirs:
            if not addon_dir.exists():
                continue
            for py_file in addon_dir.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                module_name = py_file.stem
                if module_name not in self._modules:
                    self._modules[module_name] = ModuleInfo(
                        name=module_name,
                        path=str(py_file),
                    )
                    found.append(module_name)
        return found

    def list_modules(self) -> list[ModuleInfo]:
        """List all known modules."""
        return list(self._modules.values())

    def get_module(self, name: str) -> ModuleInfo | None:
        """Get module info by name."""
        return self._modules.get(name)

    # ── Loading ───────────────────────────────────────────────────────────

    def load(self, name: str, hot_reload: bool = False) -> Addon:
        """
        Load a kernel module by name.

        Args:
            name: Module name (without .py)
            hot_reload: If True, reload even if already loaded

        Returns:
            The addon instance

        Raises:
            ImportError: If module not found
            RuntimeError: If module fails to load
        """
        if name in self._modules:
            info = self._modules[name]
            if info.state == "loaded" and not hot_reload:
                return info.instance
            if info.state == "loaded" and hot_reload:
                self.unload(name)
        else:
            # Try to find it
            self.discover()
            if name not in self._modules:
                raise ImportError(f"Module not found: {name}")

        info = self._modules[name]
        info.state = "loading"
        info.error = None
        start_time = time.monotonic()

        try:
            # Fire pre_load hooks
            for hook in self._hooks["pre_load"]:
                hook(name)

            # Load dependencies first
            for dep in info.dependencies:
                if dep not in self._modules or self._modules[dep].state != "loaded":
                    self.load(dep)

            # Import the module
            module_path = Path(info.path)
            module_key = f"slo_addon_{name}"

            # Invalidate bytecode cache for hot reload
            if hot_reload:
                importlib.invalidate_caches()
                # Remove any cached .pyc files
                pyc_dir = module_path.parent / "__pycache__"
                if pyc_dir.exists():
                    for pyc in pyc_dir.glob(f"{module_path.stem}*.pyc"):
                        pyc.unlink(missing_ok=True)

            spec = importlib.util.spec_from_file_location(
                module_key, str(module_path)
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load module spec: {name}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_key] = module
            spec.loader.exec_module(module)

            # Find the setup function or Addon class
            addon_instance = None
            if hasattr(module, "setup"):
                # Legacy: module with setup(kernel) function
                if self._kernel:
                    module.setup(self._kernel)
                addon_instance = module
            elif hasattr(module, "Addon"):
                # New: module with Addon class
                addon_class = module.Addon
                addon_instance = addon_class()
                if self._kernel and isinstance(addon_instance, Addon):
                    addon_instance.setup(self._kernel)
            else:
                # Try to find any class that implements Addon protocol
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and issubclass(attr, Addon) and attr is not Addon:
                        addon_instance = attr()
                        if self._kernel:
                            addon_instance.setup(self._kernel)
                        break

            # Update state
            elapsed = (time.monotonic() - start_time) * 1000
            info.loaded_at = time.time()
            info.load_time_ms = elapsed
            info.state = "loaded"
            info.instance = addon_instance

            # Extract metadata from module
            if hasattr(module, "__version__"):
                info.version = module.__version__
            if hasattr(module, "__description__"):
                info.description = module.__description__
            if hasattr(module, "__author__"):
                info.author = module.__author__

            logger.info(f"Loaded addon: {name} ({elapsed:.1f}ms)")

            # Fire post_load hooks
            for hook in self._hooks["post_load"]:
                hook(name)

            return addon_instance

        except Exception as e:
            info.state = "error"
            info.error = str(e)
            logger.error(f"Failed to load addon {name}: {e}")
            raise RuntimeError(f"Failed to load addon {name}: {e}") from e

    def unload(self, name: str) -> bool:
        """
        Unload a kernel module.

        Returns True if unloaded, False if not loaded.
        """
        if name not in self._modules:
            return False

        info = self._modules[name]
        if info.state != "loaded":
            return False

        # Fire pre_unload hooks
        for hook in self._hooks["pre_unload"]:
            hook(name)

        # Call cleanup if available
        if info.instance and hasattr(info.instance, "cleanup"):
            try:
                info.instance.cleanup()
            except Exception as e:
                logger.warning(f"Addon {name} cleanup failed: {e}")

        # Remove from sys.modules
        module_key = f"slo_addon_{name}"
        if module_key in sys.modules:
            del sys.modules[module_key]

        info.state = "unloaded"
        info.instance = None
        logger.info(f"Unloaded addon: {name}")

        # Fire post_unload hooks
        for hook in self._hooks["post_unload"]:
            hook(name)

        return True

    def reload(self, name: str) -> Addon:
        """Hot-reload a module."""
        return self.load(name, hot_reload=True)

    # ── Hooks ─────────────────────────────────────────────────────────────

    def on(self, event: str, callback: Callable):
        """Register a lifecycle hook."""
        if event in self._hooks:
            self._hooks[event].append(callback)

    # ── Query ─────────────────────────────────────────────────────────────

    def loaded(self) -> list[str]:
        """Return names of loaded modules."""
        return [name for name, info in self._modules.items() if info.state == "loaded"]

    def errors(self) -> list[ModuleInfo]:
        """Return modules in error state."""
        return [info for info in self._modules.values() if info.state == "error"]

    def summary(self) -> dict:
        """Return a summary of all modules."""
        states = {}
        for info in self._modules.values():
            states[info.state] = states.get(info.state, 0) + 1
        return {
            "total": len(self._modules),
            "by_state": states,
            "loaded": self.loaded(),
        }
