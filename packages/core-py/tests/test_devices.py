"""
Tests for Shell Device Nodes — AIDevice, DeviceManager, default devices.
"""

import pytest
from domains.shell.devices import (
    AIDevice, NullDevice, RandomDevice, LLMDevice, EmbeddingDevice,
    KnowledgeDevice, VisionDevice, ProcDevice, DeviceManager,
    create_default_devices,
)


class FakeProcess:
    def __init__(self, pid, name, state="running", uptime=10.0):
        self.pid = pid
        self.name = name
        self.state = state
        self.uptime = uptime


class FakeKernel:
    def __init__(self):
        self._procs = [FakeProcess(1, "kernel"), FakeProcess(42, "agent")]
        self._uptime = 123.45

    @property
    def uptime(self):
        return self._uptime

    def list_processes(self):
        return self._procs

    def get_process(self, pid):
        for p in self._procs:
            if p.pid == pid:
                return p
        return None


class TestAIDevice:
    def test_read_raises(self):
        with pytest.raises(NotImplementedError):
            AIDevice().read()

    def test_write_raises(self):
        with pytest.raises(NotImplementedError):
            AIDevice().write("data")

    def test_default_name(self):
        assert AIDevice().name == ""


class TestNullDevice:
    def test_name(self):
        assert NullDevice().name == "null"

    def test_read(self):
        assert NullDevice().read() == ""

    def test_write(self):
        assert NullDevice().write("data") == ""


class TestRandomDevice:
    def test_read_default_length(self):
        assert len(RandomDevice().read()) == 64

    def test_read_custom_length(self):
        assert len(RandomDevice().read("16")) == 16

    def test_read_clamps_to_4096(self):
        assert len(RandomDevice().read("5000")) == 4096

    def test_read_minimum_1(self):
        assert len(RandomDevice().read("0")) == 1

    def test_write(self):
        r = RandomDevice().write("test")
        assert "wrote" in r and "4 bytes" in r


class TestLLMDevice:
    def test_read_with_fn(self):
        dev = LLMDevice(generate_fn=lambda p: f"AI: {p}")
        assert dev.read("hello") == "AI: hello"

    def test_write_with_fn(self):
        dev = LLMDevice(generate_fn=lambda p: f"Resp: {p}")
        assert dev.write("hi") == "Resp: hi"

    def test_write_empty(self):
        dev = LLMDevice()
        r = dev.write("")
        assert "Usage" in r


class TestEmbeddingDevice:
    def test_read_before_write(self):
        dev = EmbeddingDevice()
        assert "No embedding" in dev.read()

    def test_write_then_read(self):
        dev = EmbeddingDevice(embed_fn=lambda t: [0.1, 0.2, 0.3])
        dev.write("hello")
        r = dev.read()
        assert "0.1" in r and "3 dims" in r

    def test_write_empty(self):
        dev = EmbeddingDevice()
        assert "Usage" in dev.write("")


class TestKnowledgeDevice:
    def test_read(self):
        dev = KnowledgeDevice(api_base="http://localhost:1")
        r = dev.read()
        assert r is not None and len(r) > 0

    def test_write(self):
        dev = KnowledgeDevice(api_base="http://localhost:1")
        r = dev.write("test")
        assert r is not None and len(r) > 0


class TestVisionDevice:
    def test_read(self):
        assert "image path" in VisionDevice().read()
    def test_write_bad_path(self):
        assert "not found" in VisionDevice().write("/x/y.jpg")


class TestProcDevice:
    @pytest.fixture
    def dev(self):
        return ProcDevice(lambda: FakeKernel())

    def test_uptime(self, dev):
        assert "123.45" in dev.read("uptime")
    def test_loadavg(self, dev):
        assert dev.read("loadavg").count("/") == 1
    def test_stat(self, dev):
        r = dev.read("stat")
        assert "kernel" in r and "agent" in r
    def test_pid_status(self, dev):
        r = dev.read("42/status")
        assert "agent" in r and "42" in r
    def test_nonexistent_pid(self, dev):
        assert "No such process" in dev.read("999/status")
    def test_nonexistent_path(self, dev):
        assert "No such file" in dev.read("xyz")
    def test_write_ro(self, dev):
        assert "read-only" in dev.write("data")


class TestDeviceManager:
    def test_register_get(self):
        mgr = DeviceManager()
        mgr.register(NullDevice())
        assert mgr.get("null") is not None

    def test_get_missing(self):
        assert DeviceManager().get("x") is None

    def test_names(self):
        mgr = DeviceManager()
        mgr.register(NullDevice())
        mgr.register(RandomDevice())
        assert "null" in mgr.names and "random" in mgr.names

    def test_list_devices(self):
        mgr = DeviceManager()
        mgr.register(NullDevice())
        assert "/dev/null" in mgr.list_devices()

    def test_read_unknown(self):
        assert "No such device" in DeviceManager().read("/dev/x")

    def test_write_unknown(self):
        assert "No such device" in DeviceManager().write("/dev/x", "d")

    def test_is_device_path(self):
        assert DeviceManager.is_device_path("/dev/llm") is True
        assert DeviceManager.is_device_path("dev/llm") is True
        assert DeviceManager.is_device_path("/etc") is False

    def test_proc_subpath(self):
        mgr = DeviceManager()
        mgr.register(ProcDevice(lambda: FakeKernel()))
        assert "123.45" in mgr.read("/dev/proc/uptime")

    def test_default_devices(self):
        mgr = create_default_devices()
        assert mgr.get("null") is not None
        assert mgr.get("random") is not None
        assert mgr.get("llm") is not None
        assert mgr.get("proc") is not None
        assert len(mgr.names) >= 7
