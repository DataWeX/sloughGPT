"""Coverage tests for AI device nodes (domains.shell.devices)."""

import builtins
from unittest.mock import patch

import pytest

from domains.shell.devices import (
    AIDevice,
    DeviceManager,
    EmbeddingDevice,
    KnowledgeDevice,
    LLMDevice,
    NullDevice,
    ProcDevice,
    RandomDevice,
    VisionDevice,
    create_default_devices,
)


class _Resp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


class TestAIDeviceBase:
    def test_read_raises(self):
        with pytest.raises(NotImplementedError):
            AIDevice().read("x")

    def test_write_raises(self):
        with pytest.raises(NotImplementedError):
            AIDevice().write("x")


class TestNullAndRandom:
    def test_null(self):
        dev = NullDevice()
        assert dev.read() == ""
        assert dev.write("anything") == ""

    def test_random_default_length(self):
        dev = RandomDevice()
        assert len(dev.read()) == 64

    def test_random_custom_length(self):
        dev = RandomDevice()
        out = dev.read("10")
        assert len(out) == 10

    def test_random_invalid_length(self):
        dev = RandomDevice()
        assert len(dev.read("abc")) == 64

    def test_random_clamped(self):
        dev = RandomDevice()
        assert len(dev.read("10000")) == 4096
        assert len(dev.read("-5")) == 1

    def test_random_write(self):
        dev = RandomDevice()
        out = dev.write("hello")
        assert "wrote 5 bytes" in out


class TestLLMDevice:
    def test_read_default_prompt_with_generate_fn(self):
        dev = LLMDevice(generate_fn=lambda p: f"echo:{p}")
        out = dev.read("hello world")
        assert out == "echo:hello world"

    def test_read_defaults_prompt(self):
        dev = LLMDevice(generate_fn=lambda p: f"echo:{p}")
        assert dev.read("") == "echo:continue this thought"

    def test_write_empty_shows_usage(self):
        dev = LLMDevice()
        assert "Usage" in dev.write("  ")

    def test_write_calls_llm(self):
        dev = LLMDevice(generate_fn=lambda p: f"answer:{p}")
        assert dev.write("hi") == "answer:hi"

    def test_call_api_success(self, monkeypatch):
        monkeypatch.setattr("requests.post", lambda *a, **k: _Resp(200, {"text": "ok"}))
        assert LLMDevice()._call_llm("p") == "ok"

    def test_call_api_error(self, monkeypatch):
        monkeypatch.setattr("requests.post", lambda *a, **k: _Resp(500))
        assert "API error 500" in LLMDevice()._call_llm("p")

    def test_call_no_requests(self, monkeypatch):
        real_import = builtins.__import__

        def _block(name, *a, **k):
            if name == "requests":
                raise ImportError("no requests")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", _block)
        assert "requests not available" in LLMDevice()._call_llm("p")

    def test_call_exception(self, monkeypatch):
        def _boom(*a, **k):
            raise ConnectionError("refused")

        monkeypatch.setattr("requests.post", _boom)
        out = LLMDevice()._call_llm("p")
        assert "refused" in out


class TestEmbeddingDevice:
    def test_read_before_write(self):
        assert "No embedding" in EmbeddingDevice().read()

    def test_read_after_write_fallback(self):
        dev = EmbeddingDevice()
        assert dev.write("hello") == "  embedding: 384 dims"
        out = dev.read()
        assert "(384 dims" in out

    def test_write_empty_shows_usage(self):
        assert "Usage" in EmbeddingDevice().write("  ")

    def test_compute_uses_embed_fn(self):
        dev = EmbeddingDevice(embed_fn=lambda t: [1.0, 2.0])
        assert dev.write("x") == "  embedding: 2 dims"
        assert dev.read().startswith("[1.0000, 2.0000")

    def test_compute_fallback_is_deterministic(self):
        dev = EmbeddingDevice()
        with patch("domains.inference.vector_store.simple_embed", side_effect=ImportError):
            a = dev._compute_embedding("same text")
            b = dev._compute_embedding("same text")
        assert a == b
        assert len(a) == 32


class TestKnowledgeDevice:
    def _dev(self, api_base="http://test"):
        return KnowledgeDevice(api_base=api_base)

    def test_read_facts_present(self, monkeypatch):
        monkeypatch.setattr(
            "requests.get",
            lambda *a, **k: _Resp(200, [{"content": "fact one", "topic": "math"}]),
        )
        assert self._dev().read() == "[math] fact one"

    def test_read_fact_without_topic(self, monkeypatch):
        monkeypatch.setattr(
            "requests.get",
            lambda *a, **k: _Resp(200, [{"text": "plain fact"}]),
        )
        assert self._dev().read() == "[general] plain fact"

    def test_read_empty_store(self, monkeypatch):
        monkeypatch.setattr("requests.get", lambda *a, **k: _Resp(200, []))
        assert "empty" in self._dev().read()

    def test_read_api_error(self, monkeypatch):
        monkeypatch.setattr("requests.get", lambda *a, **k: _Resp(500))
        assert "API error 500" in self._dev().read()

    def test_read_no_requests(self, monkeypatch):
        real_import = builtins.__import__

        def _block(name, *a, **k):
            if name == "requests":
                raise ImportError("no requests")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", _block)
        assert "requests not available" in self._dev().read()

    def test_read_exception(self, monkeypatch):
        def _boom(*a, **k):
            raise TimeoutError("timed out")

        monkeypatch.setattr("requests.get", _boom)
        assert "timed out" in self._dev().read()

    def test_write_empty_shows_usage(self):
        assert "Usage" in self._dev().write("  ")

    def test_write_success(self, monkeypatch):
        monkeypatch.setattr("requests.post", lambda *a, **k: _Resp(201))
        out = self._dev().write("some fact here")
        assert out == "  Stored: some fact here..."

    def test_write_api_error(self, monkeypatch):
        monkeypatch.setattr("requests.post", lambda *a, **k: _Resp(400))
        assert "API error 400" in self._dev().write("x")

    def test_write_no_requests(self, monkeypatch):
        real_import = builtins.__import__

        def _block(name, *a, **k):
            if name == "requests":
                raise ImportError("no requests")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", _block)
        assert "requests not available" in self._dev().write("x")

    def test_write_exception(self, monkeypatch):
        def _boom(*a, **k):
            raise ConnectionError("down")

        monkeypatch.setattr("requests.post", _boom)
        assert "down" in self._dev().write("x")


class TestVisionDevice:
    def test_read(self):
        assert "Write an image path" in VisionDevice().read()

    def test_write_empty_path(self):
        assert "File not found" in VisionDevice().write("  ")

    def test_write_missing_file(self):
        assert "File not found: /no/such/img.png" in VisionDevice().write("/no/such/img.png")

    def test_write_file_without_vision(self, tmp_path, monkeypatch):
        p = tmp_path / "img.png"
        p.write_bytes(b"fakeimage")
        real_import = builtins.__import__

        def _block(name, *a, **k):
            if name == "domains.multimodal.vision":
                raise ImportError("no vision")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", _block)
        out = VisionDevice().write(str(p))
        assert "VisionCNN not available" in out
        assert "file exists" in out

    def test_write_file_with_vision(self, tmp_path):
        from PIL import Image

        p = tmp_path / "img.png"
        Image.new("RGB", (8, 8), (10, 20, 30)).save(p)
        out = VisionDevice().write(str(p))
        assert out.startswith("  Vision: ")
        assert len(out) > len("  Vision: ")


class _Proc:
    pid = 1
    name = "init"
    state = "RUNNING"
    uptime = 12.5


class _Kernel:
    uptime = 5.0

    def list_processes(self):
        return [_Proc()]

    def get_process(self, pid):
        return _Proc() if pid == 1 else None


class TestProcDevice:
    def _dev(self, kernel):
        return ProcDevice(kernel)

    def test_uptime_with_kernel(self):
        assert self._dev(_Kernel()).read("uptime") == "5.00"

    def test_empty_path_is_uptime(self):
        assert self._dev(_Kernel()).read("") == "5.00"

    def test_uptime_no_kernel(self):
        assert self._dev(None).read("uptime") == "0.00"

    def test_loadavg_with_kernel(self):
        assert self._dev(_Kernel()).read("loadavg") == "0.00 0.00 0.00 1/1"

    def test_loadavg_no_kernel(self):
        assert self._dev(None).read("loadavg") == "0.00 0.00 0.00 1/0"

    def test_stat(self):
        out = self._dev(_Kernel()).read("stat")
        assert "processes 1" in out
        assert "pid 1  init  RUNNING  12.5s" in out

    def test_stat_no_kernel(self):
        assert self._dev(None).read("stat") == "kernel not available"

    def test_pid_status_found(self):
        out = self._dev(_Kernel()).read("1/status")
        assert "Name:\tinit" in out
        assert "Pid:\t1" in out

    def test_pid_status_missing(self):
        assert "No such process: 999" in self._dev(_Kernel()).read("999/status")

    def test_pid_status_no_kernel(self):
        assert "No such file or directory" in self._dev(None).read("1/status")

    def test_unknown_path(self):
        assert "No such file or directory" in self._dev(_Kernel()).read("bogus")

    def test_kernel_as_object_not_callable(self):
        kernel = _Kernel()
        dev = ProcDevice(kernel)
        assert dev.read("uptime") == "5.00"

    def test_write_read_only(self):
        assert "/proc is read-only" in self._dev(_Kernel()).write("x")


class TestDeviceManager:
    def _mgr(self):
        mgr = DeviceManager()
        mgr.register(NullDevice())
        mgr.register(RandomDevice())
        mgr.register(LLMDevice(generate_fn=lambda p: f"reply:{p}"))
        return mgr

    def test_names_sorted(self):
        assert self._mgr().names == ["llm", "null", "random"]

    def test_list_devices(self):
        out = self._mgr().list_devices()
        assert "/dev/llm" in out
        assert "/dev/null" in out
        assert "Discards all written data" in out

    def test_get(self):
        mgr = self._mgr()
        assert isinstance(mgr.get("null"), NullDevice)
        assert mgr.get("ghost") is None

    def test_read_slash_dev(self):
        mgr = self._mgr()
        assert mgr.read("/dev/null", "x") == ""
        assert mgr.read("/dev/llm", "hi") == "reply:hi"

    def test_read_without_leading_slash(self):
        mgr = self._mgr()
        assert mgr.read("dev/null", "x") == ""

    def test_read_subpath_passes_args(self):
        mgr = DeviceManager()
        mgr.register(ProcDevice(_Kernel()))
        assert mgr.read("/dev/proc/uptime") == "5.00"

    def test_read_missing_device(self):
        assert "No such device" in self._mgr().read("/dev/ghost", "x")

    def test_write_slash_dev(self):
        mgr = self._mgr()
        out = mgr.write("/dev/random", "payload")
        assert "wrote 7 bytes" in out

    def test_write_dev_without_slash(self):
        mgr = self._mgr()
        assert mgr.write("dev/null", "x") == ""

    def test_write_missing_device(self):
        assert "No such device" in self._mgr().write("/dev/ghost", "x")

    def test_is_device_path(self):
        assert DeviceManager.is_device_path("/dev/llm") is True
        assert DeviceManager.is_device_path("dev/llm") is True
        assert DeviceManager.is_device_path("llm") is False

    def test_create_default_devices(self):
        mgr = create_default_devices(_Kernel)
        assert set(mgr.names) == {
            "null", "random", "llm", "embedding", "knowledge", "vision", "proc",
        }
