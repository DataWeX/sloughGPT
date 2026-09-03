"""Tests for domains.shell.addons.module_loader — ModuleLoader, ModuleInfo, hot-reload."""

from __future__ import annotations

import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from domains.shell.addons.module_loader import ModuleLoader, ModuleInfo


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_addon_dir(tmp_path):
    """Create a temporary addon directory with a test addon."""
    addon_dir = tmp_path / "addons"
    addon_dir.mkdir()
    return addon_dir


@pytest.fixture
def loader(tmp_addon_dir):
    return ModuleLoader(addon_dirs=[str(tmp_addon_dir)])


def _write_addon(addon_dir, name, content=None):
    """Write a minimal addon .py file."""
    if content is None:
        content = f'''
from domains.shell.addons.base import Addon

class Addon:
    def setup(self, kernel):
        self.kernel = kernel
        self.loaded = True

    def cleanup(self):
        self.cleaned = True
'''
    p = addon_dir / f"{name}.py"
    p.write_text(content)
    return p


def _write_setup_addon(addon_dir, name):
    """Write a legacy addon using setup() function."""
    content = '''
def setup(kernel):
    kernel._test_loaded = True
'''
    p = addon_dir / f"{name}.py"
    p.write_text(content)
    return p


def _write_broken_addon(addon_dir, name):
    """Write an addon that raises on import."""
    content = 'raise RuntimeError("intentional import failure")'
    p = addon_dir / f"{name}.py"
    p.write_text(content)
    return p


def _write_private_addon(addon_dir, name):
    """Write a private addon (starts with _)."""
    content = '# private addon\nx = 1\n'
    p = addon_dir / f"_{name}.py"
    p.write_text(content)
    return p


# ── ModuleInfo ────────────────────────────────────────────────────────────────

class TestModuleInfo:
    def test_defaults(self):
        info = ModuleInfo(name="test", path="/tmp/test.py")
        assert info.name == "test"
        assert info.version == "0.0.0"
        assert info.state == "unloaded"
        assert info.error is None
        assert info.instance is None
        assert info.dependencies == []

    def test_custom_fields(self):
        info = ModuleInfo(
            name="my_mod", path="/x.py",
            version="1.2.3", description="desc", author="me",
            dependencies=["dep1"],
        )
        assert info.version == "1.2.3"
        assert info.dependencies == ["dep1"]


# ── ModuleLoader init ─────────────────────────────────────────────────────────

class TestModuleLoaderInit:
    def test_empty_loader(self):
        loader = ModuleLoader()
        assert loader.list_modules() == []

    def test_with_dirs(self, tmp_addon_dir):
        loader = ModuleLoader(addon_dirs=[str(tmp_addon_dir)])
        assert len(loader._addon_dirs) == 1

    def test_add_addon_dir(self, tmp_path):
        loader = ModuleLoader()
        loader.add_addon_dir(tmp_path / "extra")
        assert len(loader._addon_dirs) == 1

    def test_set_kernel(self, loader):
        kernel = MagicMock()
        loader.set_kernel(kernel)
        assert loader._kernel is kernel


# ── Discovery ─────────────────────────────────────────────────────────────────

class TestDiscovery:
    def test_discover_empty_dir(self, loader):
        found = loader.discover()
        assert found == []

    def test_discover_addons(self, loader, tmp_addon_dir):
        _write_addon(tmp_addon_dir, "alpha")
        _write_addon(tmp_addon_dir, "beta")
        found = loader.discover()
        assert "alpha" in found
        assert "beta" in found

    def test_discover_skips_private(self, loader, tmp_addon_dir):
        _write_addon(tmp_addon_dir, "public")
        _write_private_addon(tmp_addon_dir, "secret")
        found = loader.discover()
        assert "public" in found
        assert "secret" not in found

    def test_discover_nonexistent_dir(self):
        loader = ModuleLoader(addon_dirs=["/nonexistent/path"])
        found = loader.discover()
        assert found == []

    def test_discover_idempotent(self, loader, tmp_addon_dir):
        _write_addon(tmp_addon_dir, "mod1")
        found1 = loader.discover()
        found2 = loader.discover()
        # First call discovers mod1, second call finds nothing new (already known)
        assert found1 == ["mod1"]
        assert found2 == []
        assert len(loader.list_modules()) == 1


# ── Loading ───────────────────────────────────────────────────────────────────

class TestLoading:
    def test_load_addon_class(self, loader, tmp_addon_dir):
        _write_addon(tmp_addon_dir, "my_addon")
        loader.set_kernel(MagicMock())
        loader.discover()
        instance = loader.load("my_addon")
        assert instance is not None
        assert instance.loaded is True

    def test_load_setup_function(self, loader, tmp_addon_dir):
        _write_setup_addon(tmp_addon_dir, "legacy_addon")
        kernel = MagicMock()
        loader.set_kernel(kernel)
        loader.discover()
        instance = loader.load("legacy_addon")
        assert kernel._test_loaded is True

    def test_load_not_found(self, loader):
        with pytest.raises(ImportError, match="Module not found"):
            loader.load("nonexistent")

    def test_load_broken_addon(self, loader, tmp_addon_dir):
        _write_broken_addon(tmp_addon_dir, "broken")
        loader.discover()
        with pytest.raises(RuntimeError, match="Failed to load"):
            loader.load("broken")
        info = loader.get_module("broken")
        assert info.state == "error"
        assert "intentional" in info.error

    def test_load_returns_cached(self, loader, tmp_addon_dir):
        _write_addon(tmp_addon_dir, "cached")
        loader.discover()
        i1 = loader.load("cached")
        i2 = loader.load("cached")
        assert i1 is i2

    def test_load_sets_state(self, loader, tmp_addon_dir):
        _write_addon(tmp_addon_dir, "stateful")
        loader.discover()
        loader.load("stateful")
        info = loader.get_module("stateful")
        assert info.state == "loaded"
        assert info.load_time_ms > 0
        assert info.loaded_at > 0

    def test_load_extracts_metadata(self, loader, tmp_addon_dir):
        content = '''
__version__ = "2.0.0"
__description__ = "A test addon"
__author__ = "Tester"

from domains.shell.addons.base import Addon

class Addon:
    def setup(self, kernel):
        pass
'''
        _write_addon(tmp_addon_dir, "meta_addon", content)
        loader.discover()
        loader.load("meta_addon")
        info = loader.get_module("meta_addon")
        assert info.version == "2.0.0"
        assert info.description == "A test addon"
        assert info.author == "Tester"


# ── Hot Reload ────────────────────────────────────────────────────────────────

class TestHotReload:
    def test_reload(self, loader, tmp_addon_dir):
        _write_addon(tmp_addon_dir, "reloadable")
        loader.discover()
        i1 = loader.load("reloadable")
        i2 = loader.load("reloadable", hot_reload=True)
        assert i1 is not i2

    def test_reload_method(self, loader, tmp_addon_dir):
        _write_addon(tmp_addon_dir, "reload_mod")
        loader.discover()
        i1 = loader.reload("reload_mod")
        i2 = loader.reload("reload_mod")
        assert i1 is not i2


# ── Unloading ─────────────────────────────────────────────────────────────────

class TestUnloading:
    def test_unload(self, loader, tmp_addon_dir):
        _write_addon(tmp_addon_dir, "unloadable")
        loader.discover()
        loader.load("unloadable")
        assert loader.unload("unloadable") is True
        info = loader.get_module("unloadable")
        assert info.state == "unloaded"
        assert info.instance is None

    def test_unload_unknown(self, loader):
        assert loader.unload("nonexistent") is False

    def test_unload_not_loaded(self, loader, tmp_addon_dir):
        _write_addon(tmp_addon_dir, "not_loaded")
        loader.discover()
        assert loader.unload("not_loaded") is False

    def test_unload_calls_cleanup(self, loader, tmp_addon_dir):
        _write_addon(tmp_addon_dir, "with_cleanup")
        loader.discover()
        instance = loader.load("with_cleanup")
        loader.unload("with_cleanup")
        assert instance.cleaned is True

    def test_unload_cleans_sys_modules(self, loader, tmp_addon_dir):
        _write_addon(tmp_addon_dir, "sys_mod")
        loader.discover()
        loader.load("sys_mod")
        assert "slo_addon_sys_mod" in sys.modules
        loader.unload("sys_mod")
        assert "slo_addon_sys_mod" not in sys.modules


# ── Hooks ─────────────────────────────────────────────────────────────────────

class TestHooks:
    def test_pre_load_hook(self, loader, tmp_addon_dir):
        calls = []
        loader.on("pre_load", lambda name: calls.append(("pre", name)))
        _write_addon(tmp_addon_dir, "hooked")
        loader.discover()
        loader.load("hooked")
        assert ("pre", "hooked") in calls

    def test_post_load_hook(self, loader, tmp_addon_dir):
        calls = []
        loader.on("post_load", lambda name: calls.append(("post", name)))
        _write_addon(tmp_addon_dir, "hooked2")
        loader.discover()
        loader.load("hooked2")
        assert ("post", "hooked2") in calls

    def test_pre_unload_hook(self, loader, tmp_addon_dir):
        calls = []
        loader.on("pre_unload", lambda name: calls.append(("pre_unload", name)))
        _write_addon(tmp_addon_dir, "unhook")
        loader.discover()
        loader.load("unhook")
        loader.unload("unhook")
        assert ("pre_unload", "unhook") in calls

    def test_post_unload_hook(self, loader, tmp_addon_dir):
        calls = []
        loader.on("post_unload", lambda name: calls.append(("post_unload", name)))
        _write_addon(tmp_addon_dir, "unhook2")
        loader.discover()
        loader.load("unhook2")
        loader.unload("unhook2")
        assert ("post_unload", "unhook2") in calls

    def test_invalid_hook_event(self, loader):
        loader.on("invalid_event", lambda name: None)
        assert loader._hooks.get("invalid_event") is None


# ── Query ─────────────────────────────────────────────────────────────────────

class TestQuery:
    def test_loaded(self, loader, tmp_addon_dir):
        _write_addon(tmp_addon_dir, "a")
        _write_addon(tmp_addon_dir, "b")
        loader.discover()
        loader.load("a")
        assert loader.loaded() == ["a"]

    def test_errors(self, loader, tmp_addon_dir):
        _write_broken_addon(tmp_addon_dir, "bad")
        loader.discover()
        try:
            loader.load("bad")
        except RuntimeError:
            pass
        errors = loader.errors()
        assert len(errors) == 1
        assert errors[0].name == "bad"

    def test_summary(self, loader, tmp_addon_dir):
        _write_addon(tmp_addon_dir, "x")
        _write_broken_addon(tmp_addon_dir, "y")
        loader.discover()
        loader.load("x")
        try:
            loader.load("y")
        except RuntimeError:
            pass
        s = loader.summary()
        assert s["total"] == 2
        assert s["by_state"]["loaded"] == 1
        assert s["by_state"]["error"] == 1
        assert "x" in s["loaded"]

    def test_get_module(self, loader, tmp_addon_dir):
        _write_addon(tmp_addon_dir, "findme")
        loader.discover()
        info = loader.get_module("findme")
        assert info.name == "findme"

    def test_get_module_not_found(self, loader):
        assert loader.get_module("nope") is None


# ── Dependencies ──────────────────────────────────────────────────────────────

class TestDependencies:
    def test_load_with_dependencies(self, loader, tmp_addon_dir):
        dep_content = '''
from domains.shell.addons.base import Addon

class Addon:
    def setup(self, kernel):
        self.dep_loaded = True
'''
        main_content = '''
from domains.shell.addons.base import Addon

class Addon:
    def setup(self, kernel):
        self.main_loaded = True
'''
        _write_addon(tmp_addon_dir, "dep", dep_content)
        _write_addon(tmp_addon_dir, "main", main_content)

        loader.discover()
        info = loader.get_module("main")
        info.dependencies = ["dep"]
        loader.load("main")
        dep_info = loader.get_module("dep")
        assert dep_info.state == "loaded"
