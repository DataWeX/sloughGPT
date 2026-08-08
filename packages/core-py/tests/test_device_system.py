"""Tests for domains.shell.device_system — central device registry."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from domains.shell.device_system import (
    DeviceSystem,
    get_device_system,
    reset_device_system,
)
from domains.shell.vm import Device, DeviceFault


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset global singleton before each test."""
    reset_device_system()
    yield
    reset_device_system()


class StubDevice(Device):
    """Minimal Device stub for testing."""

    def __init__(self, info_dict=None):
        self._info = info_dict or {"name": "stub", "version": "1.0"}

    def call(self, method, *args):
        if method == "echo":
            return args[0] if args else None
        if method == "fail":
            raise DeviceFault("intentional fault")
        raise DeviceFault(f"unknown method: {method}")

    def info(self):
        return self._info


class TestRegister:
    def test_register_adds_device(self):
        ds = DeviceSystem()
        dev = StubDevice()
        ds.register("stub", dev)
        assert ds.get("stub") is dev

    def test_register_sets_metadata(self):
        ds = DeviceSystem()
        dev = StubDevice()
        ds.register("stub", dev, registered_by="test", version="2.0")
        meta = ds.metadata("stub")
        assert meta["registered_by"] == "test"
        assert meta["version"] == "2.0"

    def test_register_default_registered_by(self):
        ds = DeviceSystem()
        ds.register("dev", StubDevice())
        assert ds.metadata("dev")["registered_by"] == ""

    def test_register_adds_to_bus(self):
        ds = DeviceSystem()
        dev = StubDevice()
        ds.register("stub", dev)
        assert ds.bus.open("stub") is dev

    def test_register_overwrites_same_name(self):
        ds = DeviceSystem()
        d1 = StubDevice({"name": "v1"})
        d2 = StubDevice({"name": "v2"})
        ds.register("x", d1)
        ds.register("x", d2)
        assert ds.get("x") is d2


class TestUnregister:
    def test_unregister_removes_device(self):
        ds = DeviceSystem()
        ds.register("stub", StubDevice())
        ds.unregister("stub")
        assert ds.get("stub") is None

    def test_unregister_removes_metadata(self):
        ds = DeviceSystem()
        ds.register("stub", StubDevice())
        ds.unregister("stub")
        assert ds.metadata("stub") == {}

    def test_unregister_nonexistent_is_noop(self):
        ds = DeviceSystem()
        ds.unregister("nonexistent")  # should not raise


class TestCall:
    def test_call_delegates_to_device(self):
        ds = DeviceSystem()
        ds.register("stub", StubDevice())
        assert ds.call("stub", "echo", "hello") == "hello"

    def test_call_no_args(self):
        ds = DeviceSystem()
        ds.register("stub", StubDevice())
        assert ds.call("stub", "echo") is None

    def test_call_raises_on_missing_device(self):
        ds = DeviceSystem()
        with pytest.raises(DeviceFault, match="no such device"):
            ds.call("missing", "echo")

    def test_call_raises_on_method_fault(self):
        ds = DeviceSystem()
        ds.register("stub", StubDevice())
        with pytest.raises(DeviceFault, match="intentional fault"):
            ds.call("stub", "fail")

    def test_call_raises_on_unknown_method(self):
        ds = DeviceSystem()
        ds.register("stub", StubDevice())
        with pytest.raises(DeviceFault, match="unknown method"):
            ds.call("stub", "nonexistent")


class TestListDevices:
    def test_empty(self):
        assert DeviceSystem().list_devices() == []

    def test_sorted(self):
        ds = DeviceSystem()
        ds.register("zebra", StubDevice())
        ds.register("alpha", StubDevice())
        ds.register("mid", StubDevice())
        assert ds.list_devices() == ["alpha", "mid", "zebra"]


class TestInfo:
    def test_info_returns_device_info(self):
        ds = DeviceSystem()
        ds.register("stub", StubDevice({"name": "test"}))
        assert ds.info("stub") == {"name": "test"}

    def test_info_empty_for_missing(self):
        assert DeviceSystem().info("missing") == {}


class TestMetadata:
    def test_metadata_empty_for_missing(self):
        assert DeviceSystem().metadata("missing") == {}

    def test_metadata_custom_fields(self):
        ds = DeviceSystem()
        ds.register("dev", StubDevice(), foo="bar", baz=42)
        meta = ds.metadata("dev")
        assert meta["foo"] == "bar"
        assert meta["baz"] == 42


class TestBus:
    def test_bus_property(self):
        ds = DeviceSystem()
        assert ds.bus is not None
        assert hasattr(ds.bus, "register")

    def test_bus_has_registered_devices(self):
        ds = DeviceSystem()
        dev = StubDevice()
        ds.register("stub", dev)
        assert ds.bus.open("stub") is dev


class TestLenContains:
    def test_len(self):
        ds = DeviceSystem()
        assert len(ds) == 0
        ds.register("a", StubDevice())
        assert len(ds) == 1
        ds.register("b", StubDevice())
        assert len(ds) == 2

    def test_contains(self):
        ds = DeviceSystem()
        ds.register("stub", StubDevice())
        assert "stub" in ds
        assert "missing" not in ds


class TestSingleton:
    def test_get_returns_same(self):
        a = get_device_system()
        b = get_device_system()
        assert a is b

    def test_reset_creates_new(self):
        a = get_device_system()
        reset_device_system()
        b = get_device_system()
        assert a is not b
