"""Coverage tests for DeviceSystem (domains.shell.device_system)."""

import pytest

from domains.shell.device_system import (
    DeviceSystem,
    get_device_system,
    reset_device_system,
)
from domains.shell.vm import Device, DeviceFault


class _FakeDevice(Device):
    description = "a test device"

    def call(self, method, *args):
        if method == "ping":
            return "pong"
        raise DeviceFault(f"device does not support: {method}")

    def info(self):
        return {"type": "fake", "methods": ["ping"]}


class TestDeviceSystem:
    def test_register_and_get(self):
        ds = DeviceSystem()
        dev = _FakeDevice()
        ds.register("alpha", dev, registered_by="tester", version=2)
        assert ds.get("alpha") is dev
        assert ds.get("missing") is None
        assert "alpha" in ds
        assert "missing" not in ds
        assert len(ds) == 1

    def test_register_metadata(self):
        ds = DeviceSystem()
        ds.register("alpha", _FakeDevice(), registered_by="tester", version=2)
        meta = ds.metadata("alpha")
        assert meta["description"] == "a test device"
        assert meta["registered_by"] == "tester"
        assert meta["version"] == 2
        assert ds.metadata("missing") == {}

    def test_register_without_registered_by(self):
        ds = DeviceSystem()
        ds.register("alpha", _FakeDevice())
        assert ds.metadata("alpha")["registered_by"] == ""

    def test_unregister(self):
        ds = DeviceSystem()
        ds.register("alpha", _FakeDevice())
        ds.unregister("alpha")
        assert ds.get("alpha") is None
        assert "alpha" not in ds
        assert ds.metadata("alpha") == {}
        assert len(ds) == 0

    def test_unregister_missing_is_noop(self):
        ds = DeviceSystem()
        ds.unregister("ghost")
        assert len(ds) == 0

    def test_call_success(self):
        ds = DeviceSystem()
        ds.register("alpha", _FakeDevice())
        assert ds.call("alpha", "ping") == "pong"

    def test_call_missing_device_raises(self):
        ds = DeviceSystem()
        with pytest.raises(DeviceFault):
            ds.call("ghost", "ping")

    def test_call_unknown_method_raises(self):
        ds = DeviceSystem()
        ds.register("alpha", _FakeDevice())
        with pytest.raises(DeviceFault):
            ds.call("alpha", "bogus")

    def test_list_devices_sorted(self):
        ds = DeviceSystem()
        ds.register("zeta", _FakeDevice())
        ds.register("alpha", _FakeDevice())
        assert ds.list_devices() == ["alpha", "zeta"]

    def test_info_present_and_missing(self):
        ds = DeviceSystem()
        ds.register("alpha", _FakeDevice())
        assert ds.info("alpha") == {"type": "fake", "methods": ["ping"]}
        assert ds.info("missing") == {}

    def test_bus_property(self):
        ds = DeviceSystem()
        assert ds.bus is ds._bus
        ds.register("alpha", _FakeDevice())
        assert ds.bus.list_devices() == ["alpha"]


class TestDeviceSystemSingleton:
    def test_singleton_identity(self):
        reset_device_system()
        a = get_device_system()
        b = get_device_system()
        assert a is b
        reset_device_system()

    def test_reset_creates_fresh(self):
        reset_device_system()
        a = get_device_system()
        reset_device_system()
        b = get_device_system()
        assert a is not b
        reset_device_system()
