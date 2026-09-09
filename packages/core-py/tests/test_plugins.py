"""Tests for the plugin framework."""

import pytest
from domains.plugins import (
    Plugin,
    PluginHook,
    PluginManager,
    PluginMetadata,
    get_plugin_manager,
)


class SamplePlugin(Plugin):
    """A sample plugin for testing."""

    __version__ = "1.0.0"
    __author__ = "Test Author"

    def setup(self, context):
        self.setup_called = True
        self.context = context

    def teardown(self):
        self.teardown_called = True


class FailingPlugin(Plugin):
    """A plugin that fails on setup."""

    def setup(self, context):
        raise RuntimeError("Setup failed")


class TestPluginHook:
    def test_register_and_run(self):
        hook = PluginHook("test_hook")
        results = []
        hook.register(lambda: results.append(1))
        hook.register(lambda: results.append(2))
        hook.run()
        assert results == [1, 2]

    def test_unregister(self):
        hook = PluginHook("test_hook")
        handler = lambda: None
        hook.register(handler)
        hook.unregister(handler)
        assert len(hook._handlers) == 0

    def test_run_with_args(self):
        hook = PluginHook("test_hook")
        hook.register(lambda x, y: x + y)
        results = hook.run(1, 2)
        assert results == [3]

    def test_run_handler_error(self):
        hook = PluginHook("test_hook")
        hook.register(lambda: 1)
        hook.register(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        results = hook.run()
        assert results == [1]


class TestPluginManager:
    def test_load_plugin(self):
        pm = PluginManager()
        assert pm.load_plugin(SamplePlugin)
        assert "SamplePlugin" in pm._plugins

    def test_load_plugin_metadata(self):
        pm = PluginManager()
        pm.load_plugin(SamplePlugin)
        meta = pm._metadata["SamplePlugin"]
        assert meta.version == "1.0.0"
        assert meta.author == "Test Author"

    def test_unload_plugin(self):
        pm = PluginManager()
        pm.load_plugin(SamplePlugin)
        assert pm.unload_plugin("SamplePlugin")
        assert "SamplePlugin" not in pm._plugins

    def test_unload_nonexistent(self):
        pm = PluginManager()
        assert not pm.unload_plugin("Nonexistent")

    def test_list_plugins(self):
        pm = PluginManager()
        pm.load_plugin(SamplePlugin)
        plugins = pm.list_plugins()
        assert len(plugins) == 1
        assert plugins[0]["name"] == "SamplePlugin"

    def test_get_plugin(self):
        pm = PluginManager()
        pm.load_plugin(SamplePlugin)
        plugin = pm.get_plugin("SamplePlugin")
        assert plugin is not None
        assert isinstance(plugin, SamplePlugin)

    def test_get_nonexistent(self):
        pm = PluginManager()
        assert pm.get_plugin("Nonexistent") is None

    def test_register_hook(self):
        pm = PluginManager()
        hook = pm.register_hook("test", "Test hook")
        assert isinstance(hook, PluginHook)
        assert pm.get_hook("test") is hook


class TestPluginMetadata:
    def test_defaults(self):
        meta = PluginMetadata(name="test", version="1.0", description="Test", author="Author")
        assert meta.enabled is True
        assert meta.hooks == []


class TestGetPluginManager:
    def test_singleton(self):
        pm1 = get_plugin_manager()
        pm2 = get_plugin_manager()
        assert pm1 is pm2
