"""Tests for DeviceSystem — central device registry."""

import pytest
from domains.shell.vm import Device, DeviceBus, DeviceFault
from domains.shell.device_system import DeviceSystem, get_device_system, reset_device_system


class MockDevice(Device):
    """Test device with configurable ops."""

    def __init__(self, ops=None):
        self._ops = ops or {}
        self._call_log = []

    def call(self, method, *args):
        self._call_log.append((method, args))
        if method in self._ops:
            return self._ops[method](*args)
        raise DeviceFault(f"unknown op: {method}")

    def info(self):
        return {"type": "mock", "ops": list(self._ops.keys())}


class TestDeviceSystem:
    def setup_method(self):
        reset_device_system()

    def teardown_method(self):
        reset_device_system()

    def test_register_and_get(self):
        ds = DeviceSystem()
        dev = MockDevice({"ping": lambda: "pong"})
        ds.register("test", dev)
        assert ds.get("test") is dev

    def test_get_nonexistent_returns_none(self):
        ds = DeviceSystem()
        assert ds.get("nope") is None

    def test_unregister(self):
        ds = DeviceSystem()
        ds.register("test", MockDevice())
        ds.unregister("test")
        assert ds.get("test") is None

    def test_unregister_nonexistent_no_error(self):
        ds = DeviceSystem()
        ds.unregister("nope")  # should not raise

    def test_call(self):
        ds = DeviceSystem()
        dev = MockDevice({"add": lambda a, b: a + b})
        ds.register("calc", dev)
        assert ds.call("calc", "add", 2, 3) == 5

    def test_call_nonexistent_raises(self):
        ds = DeviceSystem()
        with pytest.raises(DeviceFault, match="no such device"):
            ds.call("nope", "method")

    def test_call_unknown_method_raises(self):
        ds = DeviceSystem()
        ds.register("test", MockDevice())
        with pytest.raises(DeviceFault, match="unknown op"):
            ds.call("test", "nonexistent")

    def test_list_devices_sorted(self):
        ds = DeviceSystem()
        ds.register("zebra", MockDevice())
        ds.register("alpha", MockDevice())
        ds.register("middle", MockDevice())
        assert ds.list_devices() == ["alpha", "middle", "zebra"]

    def test_list_devices_empty(self):
        ds = DeviceSystem()
        assert ds.list_devices() == []

    def test_info(self):
        ds = DeviceSystem()
        dev = MockDevice({"ping": lambda: "pong"})
        ds.register("test", dev)
        info = ds.info("test")
        assert info["type"] == "mock"
        assert "ping" in info["ops"]

    def test_info_nonexistent_returns_empty(self):
        ds = DeviceSystem()
        assert ds.info("nope") == {}

    def test_metadata(self):
        ds = DeviceSystem()
        ds.register("test", MockDevice(), registered_by="shell", version="1.0")
        meta = ds.metadata("test")
        assert meta["registered_by"] == "shell"
        assert meta["version"] == "1.0"

    def test_metadata_nonexistent_returns_empty(self):
        ds = DeviceSystem()
        assert ds.metadata("nope") == {}

    def test_bus_exposure(self):
        ds = DeviceSystem()
        assert isinstance(ds.bus, DeviceBus)

    def test_bus_has_registered_devices(self):
        ds = DeviceSystem()
        dev = MockDevice()
        ds.register("test", dev)
        # Bus should have the device
        opened = ds.bus.open("test")
        assert opened is dev

    def test_len(self):
        ds = DeviceSystem()
        assert len(ds) == 0
        ds.register("a", MockDevice())
        ds.register("b", MockDevice())
        assert len(ds) == 2

    def test_contains(self):
        ds = DeviceSystem()
        ds.register("test", MockDevice())
        assert "test" in ds
        assert "nope" not in ds

    def test_singleton(self):
        ds1 = get_device_system()
        ds2 = get_device_system()
        assert ds1 is ds2

    def test_reset_singleton(self):
        ds1 = get_device_system()
        reset_device_system()
        ds2 = get_device_system()
        assert ds1 is not ds2

    def test_register_overwrites_same_name(self):
        ds = DeviceSystem()
        dev1 = MockDevice({"ping": lambda: "v1"})
        dev2 = MockDevice({"ping": lambda: "v2"})
        ds.register("test", dev1)
        ds.register("test", dev2)
        assert ds.call("test", "ping") == "v2"

    def test_metadata_registered_by_default(self):
        ds = DeviceSystem()
        ds.register("test", MockDevice())
        meta = ds.metadata("test")
        assert meta["registered_by"] == ""
