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


class _EchoDevice(Device):
    description = "echoes arguments"

    def call(self, method, *args):
        if method == "echo":
            return args
        raise DeviceFault(f"device does not support: {method}")

    def info(self):
        return {"type": "echo", "methods": ["echo"]}


class _NoDescriptionDevice(Device):
    def call(self, method, *args):
        return "ok"

    def info(self):
        return {}


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

    def test_register_multiple_devices(self):
        ds = DeviceSystem()
        ds.register("a", _FakeDevice())
        ds.register("b", _EchoDevice())
        assert len(ds) == 2
        assert ds.get("a") is not ds.get("b")

    def test_register_overwrites_same_name(self):
        ds = DeviceSystem()
        dev1 = _FakeDevice()
        dev2 = _EchoDevice()
        ds.register("alpha", dev1)
        ds.register("alpha", dev2)
        assert ds.get("alpha") is dev2
        assert len(ds) == 1

    def test_call_echo_device(self):
        ds = DeviceSystem()
        ds.register("echo", _EchoDevice())
        result = ds.call("echo", "echo", "hello", "world")
        assert result == ("hello", "world")

    def test_call_with_args(self):
        ds = DeviceSystem()
        ds.register("echo", _EchoDevice())
        result = ds.call("echo", "echo", 42)
        assert result == (42,)

    def test_unregister_then_reregister(self):
        ds = DeviceSystem()
        dev1 = _FakeDevice()
        dev2 = _EchoDevice()
        ds.register("x", dev1)
        ds.unregister("x")
        ds.register("x", dev2)
        assert ds.get("x") is dev2
        assert len(ds) == 1

    def test_list_devices_empty(self):
        ds = DeviceSystem()
        assert ds.list_devices() == []

    def test_list_devices_single(self):
        ds = DeviceSystem()
        ds.register("only", _FakeDevice())
        assert ds.list_devices() == ["only"]

    def test_metadata_extra_kwargs(self):
        ds = DeviceSystem()
        ds.register("dev", _FakeDevice(), registered_by="test", version=3, author="alice")
        meta = ds.metadata("dev")
        assert meta["version"] == 3
        assert meta["author"] == "alice"

    def test_bus_has_registered_device(self):
        ds = DeviceSystem()
        ds.register("dev", _FakeDevice())
        bus = ds.bus
        assert "dev" in bus.list_devices()

    def test_len_increases_on_register(self):
        ds = DeviceSystem()
        assert len(ds) == 0
        ds.register("a", _FakeDevice())
        assert len(ds) == 1
        ds.register("b", _FakeDevice())
        assert len(ds) == 2

    def test_len_decreases_on_unregister(self):
        ds = DeviceSystem()
        ds.register("a", _FakeDevice())
        ds.register("b", _FakeDevice())
        ds.unregister("a")
        assert len(ds) == 1

    def test_contains_after_unregister(self):
        ds = DeviceSystem()
        ds.register("a", _FakeDevice())
        assert "a" in ds
        ds.unregister("a")
        assert "a" not in ds

    def test_get_returns_none_for_missing(self):
        ds = DeviceSystem()
        assert ds.get("nonexistent") is None

    def test_no_description_device(self):
        ds = DeviceSystem()
        ds.register("nodevice", _NoDescriptionDevice())
        meta = ds.metadata("nodevice")
        assert meta["description"] == ""


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

    def test_singleton_preserves_state(self):
        reset_device_system()
        ds = get_device_system()
        ds.register("test", _FakeDevice())
        ds2 = get_device_system()
        assert "test" in ds2
        reset_device_system()

    def test_multiple_resets(self):
        reset_device_system()
        reset_device_system()
        reset_device_system()
        ds = get_device_system()
        assert len(ds) == 0
        reset_device_system()

    def test_singleton_is_device_system(self):
        reset_device_system()
        ds = get_device_system()
        assert isinstance(ds, DeviceSystem)
        reset_device_system()


class TestDeviceSystemEdgeCases:
    def test_register_many_devices(self):
        ds = DeviceSystem()
        for i in range(100):
            ds.register(f"dev{i}", _FakeDevice())
        assert len(ds) == 100
        assert ds.list_devices() == sorted([f"dev{i}" for i in range(100)])

    def test_unregister_all_devices(self):
        ds = DeviceSystem()
        ds.register("a", _FakeDevice())
        ds.register("b", _FakeDevice())
        ds.unregister("a")
        ds.unregister("b")
        assert len(ds) == 0
        assert ds.list_devices() == []

    def test_call_returns_different_results(self):
        ds = DeviceSystem()
        ds.register("a", _FakeDevice())
        ds.register("b", _EchoDevice())
        assert ds.call("a", "ping") == "pong"
        assert ds.call("b", "echo", "test") == ("test",)

    def test_bus_list_devices_matches_system(self):
        ds = DeviceSystem()
        ds.register("x", _FakeDevice())
        ds.register("y", _EchoDevice())
        assert sorted(ds.bus.list_devices()) == ds.list_devices()

    def test_metadata_returns_same_dict(self):
        ds = DeviceSystem()
        ds.register("a", _FakeDevice(), registered_by="user1")
        meta1 = ds.metadata("a")
        meta2 = ds.metadata("a")
        assert meta1 == meta2
        assert meta1 is meta2

    def test_register_device_with_no_info(self):
        ds = DeviceSystem()
        dev = _NoDescriptionDevice()
        ds.register("nodevice", dev)
        assert ds.info("nodevice") == {}
        assert ds.metadata("nodevice")["description"] == ""


class TestDeviceSystemExpanded:
    def test_base_device_call_raises(self):
        dev = Device()
        with pytest.raises(DeviceFault):
            dev.call("anything")

    def test_base_device_info_returns_default(self):
        dev = Device()
        result = dev.info()
        assert result == {"type": "base", "methods": []}

    def test_register_empty_name(self):
        ds = DeviceSystem()
        ds.register("", _FakeDevice())
        assert "" in ds
        assert ds.get("") is not None

    def test_same_device_object_under_different_names(self):
        ds = DeviceSystem()
        dev = _FakeDevice()
        ds.register("alias1", dev)
        ds.register("alias2", dev)
        assert ds.get("alias1") is ds.get("alias2")
        assert len(ds) == 2

    def test_unregister_one_of_two_names_sharing_device(self):
        ds = DeviceSystem()
        dev = _FakeDevice()
        ds.register("alias1", dev)
        ds.register("alias2", dev)
        ds.unregister("alias1")
        assert "alias1" not in ds
        assert "alias2" in ds
        assert len(ds) == 1

    def test_bus_reflects_all_registered_devices(self):
        ds = DeviceSystem()
        ds.register("a", _FakeDevice())
        ds.register("b", _EchoDevice())
        bus_devices = ds.bus.list_devices()
        assert "a" in bus_devices
        assert "b" in bus_devices

    def test_bus_keeps_stale_reference_after_unregister(self):
        ds = DeviceSystem()
        ds.register("alpha", _FakeDevice())
        ds.unregister("alpha")
        bus_devices = ds.bus.list_devices()
        assert "alpha" in bus_devices

    def test_metadata_preserves_all_extra_kwargs(self):
        ds = DeviceSystem()
        ds.register("dev", _FakeDevice(), registered_by="u", version=5, author="a", tag="t")
        meta = ds.metadata("dev")
        assert meta["version"] == 5
        assert meta["author"] == "a"
        assert meta["tag"] == "t"

    def test_call_with_zero_args(self):
        ds = DeviceSystem()
        ds.register("alpha", _FakeDevice())
        assert ds.call("alpha", "ping") == "pong"

    def test_call_result_type_preserved(self):
        ds = DeviceSystem()
        ds.register("echo", _EchoDevice())
        result = ds.call("echo", "echo", 42)
        assert isinstance(result, tuple)
        assert result == (42,)

    def test_list_devices_with_numeric_names(self):
        ds = DeviceSystem()
        ds.register("3", _FakeDevice())
        ds.register("1", _FakeDevice())
        ds.register("2", _FakeDevice())
        assert ds.list_devices() == ["1", "2", "3"]

    def test_contains_after_overwrite(self):
        ds = DeviceSystem()
        ds.register("x", _FakeDevice())
        assert "x" in ds
        ds.register("x", _EchoDevice())
        assert "x" in ds
        assert ds.get("x") is not None

    def test_unregister_re_register_cycle(self):
        ds = DeviceSystem()
        ds.register("dev", _FakeDevice())
        ds.unregister("dev")
        ds.register("dev", _EchoDevice())
        assert "dev" in ds
        assert ds.call("dev", "echo", "hi") == ("hi",)

    def test_info_matches_device_info_method(self):
        ds = DeviceSystem()
        dev = _FakeDevice()
        ds.register("alpha", dev)
        assert ds.info("alpha") == dev.info()

    def test_call_with_none_arg(self):
        ds = DeviceSystem()
        ds.register("echo", _EchoDevice())
        result = ds.call("echo", "echo", None)
        assert result == (None,)

    def test_register_with_version_zero(self):
        ds = DeviceSystem()
        ds.register("dev", _FakeDevice(), version=0)
        meta = ds.metadata("dev")
        assert meta["version"] == 0

    def test_get_after_unregister_returns_none(self):
        ds = DeviceSystem()
        ds.register("a", _FakeDevice())
        ds.unregister("a")
        assert ds.get("a") is None

    def test_multiple_devices_different_types(self):
        ds = DeviceSystem()
        ds.register("fake", _FakeDevice())
        ds.register("echo", _EchoDevice())
        ds.register("no_desc", _NoDescriptionDevice())
        assert len(ds) == 3
        assert ds.call("fake", "ping") == "pong"
        assert ds.call("echo", "echo", "x") == ("x",)
        assert ds.call("no_desc", "anything") == "ok"

    def test_metadata_empty_for_nonexistent(self):
        ds = DeviceSystem()
        assert ds.metadata("ghost") == {}
        assert ds.metadata("") == {}

    def test_singleton_get_returns_device_system_type(self):
        reset_device_system()
        ds = get_device_system()
        assert type(ds).__name__ == "DeviceSystem"
        reset_device_system()

    def test_singleton_get_then_register(self):
        reset_device_system()
        ds = get_device_system()
        ds.register("test_dev", _FakeDevice())
        ds2 = get_device_system()
        assert ds2.get("test_dev") is not None
        reset_device_system()

    def test_singleton_reset_clears_devices(self):
        reset_device_system()
        ds = get_device_system()
        ds.register("temp", _FakeDevice())
        reset_device_system()
        ds2 = get_device_system()
        assert "temp" not in ds2
        reset_device_system()

    def test_bus_property_always_returns_same_bus(self):
        ds = DeviceSystem()
        bus1 = ds.bus
        bus2 = ds.bus
        assert bus1 is bus2

    def test_register_overwrite_preserves_len(self):
        ds = DeviceSystem()
        ds.register("a", _FakeDevice())
        assert len(ds) == 1
        ds.register("a", _EchoDevice())
        assert len(ds) == 1

    def test_call_multiple_different_devices(self):
        ds = DeviceSystem()
        ds.register("a", _FakeDevice())
        ds.register("b", _EchoDevice())
        ds.register("c", _NoDescriptionDevice())
        assert ds.call("a", "ping") == "pong"
        assert ds.call("b", "echo", "data") == ("data",)
        assert ds.call("c", "method") == "ok"
