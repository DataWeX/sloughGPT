"""Coverage tests for AI device nodes (domain.shell._internal.devices)."""

import builtins
from unittest.mock import patch

import pytest

from domain.shell._internal.devices import (
    AIDevice,
    AIDeviceDriver,
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
        with patch("domain.inference._internal.vector_store.simple_embed", side_effect=ImportError):
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
            if name == "domain.multimodal._internal.vision":
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
        # AI family: the unified node, its two aliases, and the per-capability
        # nodes that stay separate (knowledge/vision/null/random/proc).
        ai_family = {"ai", "llm", "embedding", "null", "random", "knowledge", "vision", "proc"}
        # Hardware family (unchanged) shares the same registry.
        hardware_family = {"tensor", "npu", "storage", "network", "display", "input"}
        assert set(mgr.names) == ai_family | hardware_family
        # /dev/ai, /dev/llm and /dev/embedding are one driver, three names.
        assert mgr.get("llm") is mgr.get("ai") is mgr.get("embedding")
        assert mgr.get("null") is not mgr.get("ai")


# ── /dev/ai — the unified AI node ──────────────────────────────────────────


class TestAIDeviceDriver:
    """The one node that replaces /dev/llm + /dev/embedding."""

    @staticmethod
    def _driver(**kwargs) -> AIDeviceDriver:
        return AIDeviceDriver(
            "ai",
            generate_fn=lambda p: f"AI: {p}",
            embed_fn=lambda t: [0.1, 0.2, 0.3],
            **kwargs,
        )

    # ── advertised contract ──

    def test_ops_advertised(self):
        driver = self._driver()
        assert driver.OPS == ("generate", "embed", "health", "info")
        assert driver.list_commands() == ["generate", "embed", "health", "info"]

    def test_capabilities_via_manager(self):
        assert create_default_devices().capabilities("ai") == [
            "generate",
            "embed",
            "health",
            "info",
        ]

    def test_info_dict_carries_description_ops_aliases(self):
        info = create_default_devices().get("ai").info()
        assert info["name"] == "ai"
        assert "Unified AI device" in info["description"]
        assert info["ops"] == ["generate", "embed", "health", "info"]
        assert set(info["aliases"]) == {"llm", "embedding"}

    def test_base_ai_device_info_shape(self):
        info = NullDevice().info()
        assert info["name"] == "null"
        assert info["type"] == "ai"
        assert info["state"] == "READY"

    # ── operations ──

    def test_info_card_is_offline(self):
        result = self._driver().ioctl("info")
        assert result.success is True
        assert "/dev/ai" in result.value
        assert "ops: generate, embed, health, info" in result.value

    def test_card_states_the_write_default_of_that_node(self):
        driver = create_default_devices().get("ai")
        assert "# generate" in driver.ioctl("info", "ai").value
        assert "# generate" in driver.ioctl("info", "llm").value
        card = driver.ioctl("info", "embedding").value
        assert "# embed" in card and 'echo "<text>"' in card

    def test_generate_uses_injected_fn(self):
        assert self._driver().ioctl("generate", "hi").value == "AI: hi"

    def test_generate_without_prompt_prints_usage(self):
        assert "no prompt" in self._driver().ioctl("generate", "   ").value

    def test_embed_reports_dims_and_preview(self):
        out = self._driver().ioctl("embed", "hi").value
        assert "3 dims" in out
        assert "[0.1000" in out

    def test_embed_without_payload_previews_last(self):
        assert "No embedding computed yet" in self._driver().ioctl("embed").value

    def test_unknown_operation_fails(self):
        result = self._driver().ioctl("frobnicate")
        assert result.success is False
        assert "unknown operation" in result.error
        assert "ops: generate, embed, health, info" in result.error

    def test_health_reports_unreachable_api(self):
        probe = {"available": False, "error": "boom"}
        with patch("domain.shell._internal.runtime._probe_api", return_value=probe):
            assert "API unreachable — boom" in self._driver().ioctl("health").value

    def test_health_reports_loaded_model(self):
        probe = {"available": True, "status": "ok", "model_id": "slo-1", "model_loaded": True}
        with patch("domain.shell._internal.runtime._probe_api", return_value=probe):
            out = self._driver().ioctl("health").value
        assert "API ok" in out and "model slo-1 (loaded)" in out

    # ── write payload dispatch ──

    def test_write_dispatches_on_op_prefix(self):
        driver = self._driver()
        assert driver.write("embed: hi").startswith("embedding: 3 dims")
        assert driver.write("plain prompt") == "AI: plain prompt"

    def test_write_leaves_non_op_colon_in_prompt(self):
        assert self._driver().write("note: hi") == "AI: note: hi"

    def test_write_tolerates_echoed_quotes(self):
        """``echo "embed: hi" > /dev/ai`` — echo keeps the quotes verbatim."""
        driver = self._driver()
        assert driver.write('"embed: hi"').startswith("embedding: 3 dims")
        assert driver.write("'embed: hi'").startswith("embedding: 3 dims")
        # outer quotes stripped from a plain prompt too (they aren't the prompt)
        assert driver.write('"plain prompt"') == "AI: plain prompt"

    def test_write_keeps_quotes_when_inner_quotes_present(self):
        """A same-kind quote inside means the pair is not a wrapper."""
        driver = self._driver()
        assert driver.write('"say \\"hi\\""') == 'AI: "say \\"hi\\""'
        # the apostrophe does not block stripping of a double-quote wrapper
        assert driver.write('"it\'s fine"') == "AI: it's fine"

    def test_write_empty_prints_usage(self):
        assert "no prompt" in self._driver().write("")

    def test_auto_op_accepts_a_default_suffix(self):
        """``auto:<op>`` — the node's fallback when the payload has no prefix."""
        driver = self._driver()
        assert driver.ioctl("auto", "hi").value == "AI: hi"
        assert driver.ioctl("auto:embed", "hi").value.startswith("embedding: 3 dims")
        # prefix still wins over the fallback
        assert driver.ioctl("auto:embed", "generate: hi").value == "AI: hi"
        assert driver.ioctl("auto:embed", '"embed: hi"').value.startswith("embedding: 3 dims")
        bad = driver.ioctl("auto:bogus", "hi")
        assert bad.success is False and "bad default op" in bad.error

    def test_read_is_card_then_generation(self):
        driver = self._driver()
        assert driver.read("").startswith("  /dev/ai")
        assert driver.read("hello") == "AI: hello"

    # ── node identity ──

    def test_info_payload_is_only_a_node_when_it_is_one(self):
        assert self._driver().ioctl("info", "hello").value.startswith("  /dev/ai")

    def test_info_names_the_alias_that_was_opened(self):
        out = create_default_devices().get("ai").ioctl("info", "llm").value
        assert out.startswith("  /dev/llm")
        assert "canonical node: /dev/ai" in out

    def test_vfs_ops_per_node(self):
        driver = create_default_devices().get("ai")
        # All three names read as the card; the write default differs by intent.
        assert driver.vfs_ops("ai") == ("info", "auto")
        assert driver.vfs_ops("llm") == ("info", "auto")
        assert driver.vfs_ops("embedding") == ("info", "auto:embed")
        # Not ours — the VFS must fall back to an inert entry.
        assert driver.vfs_ops("tensor") is None
        assert driver.vfs_ops("null") is None

    def test_aliases_open_as_file_descriptors(self):
        mgr = create_default_devices()
        fd = mgr.open("llm")
        assert fd >= 0
        try:
            assert mgr.ioctl(fd, "info").success is True
        finally:
            assert mgr.close(fd) is True

    def test_aliases_are_one_instance_not_copies(self):
        mgr = create_default_devices()
        assert mgr.get("llm") is mgr.get("ai") is mgr.get("embedding")
        assert mgr.list_devices().count(mgr.get("ai").info()) == 1
