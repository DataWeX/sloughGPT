"""Tests for shell.kernel_npu — NPUDevice model management."""

from __future__ import annotations

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from domains.shell.kernel_npu import NPUDevice
from domains.shell.kernel_devices import DeviceState
from domains.shell.kernel_syscall import SyscallResult


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_provider(name="test"):
    p = MagicMock()
    p.tokenize.return_value = [1, 2, 3]
    p.embed.return_value = np.zeros(16)
    p.generate_numpy.return_value = np.array([[1, 2, 3, 4]])
    p.detokenize.return_value = "hello"
    p.forward_numpy.return_value = np.zeros((1, 4, 32))
    p._model = MagicMock()
    p._model._params = {
        "w": np.random.randn(16, 16),
        "b": np.random.randn(16),
    }
    return p


# ── NPUDevice basics ───────────────────────────────────────────────────────


class TestNPUBasics:

    def test_init(self):
        npu = NPUDevice()
        assert npu.name == "npu"
        assert npu._models == {}

    def test_init_custom_name(self):
        npu = NPUDevice(name="custom_npu")
        assert npu.name == "custom_npu"

    def test_open_close(self):
        npu = NPUDevice()
        assert npu.open() is True
        assert npu._state == DeviceState.OPEN
        assert npu.close() is True
        assert npu._state == DeviceState.CLOSED

    def test_open_count(self):
        npu = NPUDevice()
        npu.open()
        npu.open()
        assert npu._open_count == 2

    def test_info_empty(self):
        npu = NPUDevice()
        info = npu.info()
        assert info["device"] == "npu"
        assert info["models"] == 0
        assert info["default"] == ""


# ── Model loading ──────────────────────────────────────────────────────────


class TestModelLoading:

    def test_load_numpy(self):
        npu = NPUDevice()
        with patch("domains.shell.kernel_npu.NPUDevice._load_numpy") as mock_load:
            mock_load.return_value = _make_provider()
            provider = npu.load("/tmp/test.npy", "m1")
            mock_load.assert_called_once_with("/tmp/test.npy", "m1")
            assert npu._models["m1"] is provider
            assert npu._default_model == "m1"

    def test_load_slnc(self):
        npu = NPUDevice()
        with patch("domains.shell.kernel_npu.NPUDevice._load_slnc") as mock_load:
            mock_load.return_value = _make_provider()
            npu.load("/tmp/model.slnc", "slnc1")
            mock_load.assert_called_once()
            assert "slnc1" in npu._models

    def test_load_python(self):
        npu = NPUDevice()
        with patch("domains.shell.kernel_npu.NPUDevice._load_python") as mock_load:
            mock_load.return_value = {"type": "python"}
            npu.load("/tmp/model.py", "py1")
            assert "py1" in npu._models

    def test_load_dataset_csv(self):
        npu = NPUDevice()
        with patch("domains.shell.kernel_npu.NPUDevice._load_dataset") as mock_load:
            mock_load.return_value = {"type": "dataset"}
            npu.load("/tmp/data.csv", "ds1")
            assert "ds1" in npu._models

    def test_load_no_path_raises(self):
        npu = NPUDevice()
        with pytest.raises(ValueError, match="no path"):
            npu.load("")

    def test_load_unsupported_format(self):
        npu = NPUDevice()
        with pytest.raises(ValueError, match="unsupported"):
            npu.load("/tmp/model.bin", "bad")

    def test_load_default_name_from_path(self):
        npu = NPUDevice()
        with patch("domains.shell.kernel_npu.NPUDevice._load_numpy") as mock_load:
            mock_load.return_value = _make_provider()
            npu.load("/tmp/data.npy")
            assert "data.npy" in npu._models

    def test_unload(self):
        npu = NPUDevice()
        npu._models["m1"] = _make_provider()
        npu._default_model = "m1"
        assert npu.unload("m1") is True
        assert "m1" not in npu._models
        assert npu._default_model == ""

    def test_unload_missing(self):
        npu = NPUDevice()
        assert npu.unload("nonexistent") is False

    def test_unload_default_changes(self):
        npu = NPUDevice()
        npu._models["m1"] = _make_provider("m1")
        npu._models["m2"] = _make_provider("m2")
        npu._default_model = "m1"
        npu.unload("m1")
        assert npu._default_model == "m2"


# ── Execution ───────────────────────────────────────────────────────────────


class TestExecution:

    def test_call_text(self):
        npu = NPUDevice()
        provider = _make_provider()
        npu._models["m1"] = provider
        npu._default_model = "m1"
        result = npu("m1", "hello world")
        assert "text" in result

    def test_call_text_tokenize_mode(self):
        npu = NPUDevice()
        provider = _make_provider()
        npu._models["m1"] = provider
        result = npu("m1", "hello", mode="tokenize")
        assert "tokens" in result
        assert result["count"] == 3

    def test_call_text_embed_mode(self):
        npu = NPUDevice()
        provider = _make_provider()
        npu._models["m1"] = provider
        result = npu("m1", "hello", mode="embed")
        assert "embedding" in result

    def test_call_tokens(self):
        npu = NPUDevice()
        provider = _make_provider()
        npu._models["m1"] = provider
        result = npu("m1", [1, 2, 3])
        assert "logits" in result

    def test_call_numpy_input(self):
        npu = NPUDevice()
        provider = _make_provider()
        npu._models["m1"] = provider
        result = npu("m1", np.array([1, 2, 3]))
        assert "logits" in result

    def test_call_default_model(self):
        npu = NPUDevice()
        provider = _make_provider()
        npu._models["m1"] = provider
        npu._default_model = "m1"
        result = npu("", "hello")
        assert "text" in result

    def test_call_missing_model(self):
        npu = NPUDevice()
        with pytest.raises(ValueError, match="not loaded"):
            npu("missing", "hello")

    def test_call_passthrough(self):
        npu = NPUDevice()
        npu._models["m1"] = _make_provider()
        result = npu("m1", 42)
        assert result == {"data": 42}


# ── Advanced operations ────────────────────────────────────────────────────


class TestAdvancedOps:

    def test_batch(self):
        npu = NPUDevice()
        provider = _make_provider()
        npu._models["m1"] = provider
        result = npu.batch("m1", ["hello", "world"])
        assert "results" in result
        assert result["count"] == 2

    def test_batch_missing_model(self):
        npu = NPUDevice()
        with pytest.raises(ValueError, match="not loaded"):
            npu.batch("missing", ["hello"])

    def test_pipeline(self):
        npu = NPUDevice()
        p1 = _make_provider("p1")
        p2 = _make_provider("p2")
        npu._models["m1"] = p1
        npu._models["m2"] = p2
        result = npu.pipeline(["m1", "m2"], "hello")
        assert "output" in result
        assert "trace" in result
        assert len(result["trace"]) == 2

    def test_pipeline_missing_model(self):
        npu = NPUDevice()
        npu._models["m1"] = _make_provider()
        with pytest.raises(ValueError, match="not loaded"):
            npu.pipeline(["m1", "missing"], "hello")

    def test_profile(self):
        npu = NPUDevice()
        provider = _make_provider()
        npu._models["m1"] = provider
        result = npu.profile("m1", seq_len=32, batch_sizes=[1, 2])
        assert "profiles" in result
        assert len(result["profiles"]) == 2

    def test_profile_missing_model(self):
        npu = NPUDevice()
        with pytest.raises(ValueError, match="not loaded"):
            npu.profile("missing")


# ── Quantization ────────────────────────────────────────────────────────────


class TestQuantization:

    def test_quantize(self):
        npu = NPUDevice()
        provider = _make_provider()
        npu._models["m1"] = provider
        result = npu.quantize("m1", bits=8)
        assert result["bits"] == 8
        assert result["params_quantized"] >= 1

    def test_quantize_no_model_params(self):
        npu = NPUDevice()
        provider = MagicMock()
        provider._model = MagicMock(spec=[])
        npu._models["m1"] = provider
        result = npu.quantize("m1")
        assert "error" in result


# ── Checkpoints ─────────────────────────────────────────────────────────────


class TestCheckpoints:

    def test_checkpoint_save(self):
        npu = NPUDevice()
        provider = _make_provider()
        npu._models["m1"] = provider
        result = npu.checkpoint_save("m1", "/tmp/ckpt")
        assert result["saved"] == "m1"
        assert "m1" in npu._checkpoints

    def test_checkpoint_load(self):
        npu = NPUDevice()
        npu._checkpoints["m1"] = {"name": "m1", "type": "Mock"}
        result = npu.checkpoint_load("m1", "/tmp/ckpt")
        assert result["loaded"] == "m1"

    def test_checkpoint_load_missing(self):
        npu = NPUDevice()
        result = npu.checkpoint_load("missing", "/tmp/ckpt")
        assert "error" in result


# ── Memory ──────────────────────────────────────────────────────────────────


class TestMemory:

    def test_memory_empty(self):
        npu = NPUDevice()
        with patch("psutil.Process") as MockProc:
            MockProc.return_value.memory_info.return_value.rss = 1024 * 1024
            result = npu.memory()
            assert result["rss_mb"] == 1.0
            assert result["total_model_mb"] == 0.0


# ── Ioctl dispatch ─────────────────────────────────────────────────────────


class TestIoctl:

    def test_ioctl_info(self):
        npu = NPUDevice()
        result = npu.ioctl("INFO")
        assert isinstance(result, SyscallResult)
        assert result.success is True

    def test_ioctl_load(self):
        npu = NPUDevice()
        with patch("domains.shell.kernel_npu.NPUDevice._load_numpy") as mock_load:
            mock_load.return_value = _make_provider()
            result = npu.ioctl("LOAD", "/tmp/test.npy", "m1")
            assert result.success is True

    def test_ioctl_unload(self):
        npu = NPUDevice()
        npu._models["m1"] = _make_provider()
        result = npu.ioctl("UNLOAD", "m1")
        assert result.success is True

    def test_ioctl_call(self):
        npu = NPUDevice()
        npu._models["m1"] = _make_provider()
        result = npu.ioctl("CALL", "m1", "hello")
        assert result.success is True

    def test_ioctl_batch(self):
        npu = NPUDevice()
        npu._models["m1"] = _make_provider()
        result = npu.ioctl("BATCH", "m1", ["hello", "world"])
        assert result.success is True

    def test_ioctl_pipe(self):
        npu = NPUDevice()
        npu._models["m1"] = _make_provider()
        npu._models["m2"] = _make_provider()
        result = npu.ioctl("PIPE", ["m1", "m2"], "hello")
        assert result.success is True

    def test_ioctl_profile(self):
        npu = NPUDevice()
        npu._models["m1"] = _make_provider()
        result = npu.ioctl("PROFILE", "m1", 32)
        assert result.success is True

    def test_ioctl_quantize(self):
        npu = NPUDevice()
        npu._models["m1"] = _make_provider()
        result = npu.ioctl("QUANTIZE", "m1", 8)
        assert result.success is True

    def test_ioctl_checkpoint_save(self):
        npu = NPUDevice()
        npu._models["m1"] = _make_provider()
        result = npu.ioctl("CHECKPOINT_SAVE", "m1", "/tmp/ckpt")
        assert result.success is True

    def test_ioctl_checkpoint_load(self):
        npu = NPUDevice()
        npu._checkpoints["m1"] = {"name": "m1"}
        result = npu.ioctl("CHECKPOINT_LOAD", "m1", "/tmp/ckpt")
        assert result.success is True

    def test_ioctl_memory(self):
        npu = NPUDevice()
        with patch("domains.shell.kernel_npu.NPUDevice.memory") as mock_mem:
            mock_mem.return_value = {"rss_mb": 1.0}
            result = npu.ioctl("MEMORY")
            assert result.success is True

    def test_ioctl_unknown(self):
        npu = NPUDevice()
        result = npu.ioctl("UNKNOWN")
        assert result.success is False
        assert "unknown ioctl" in result.error

    def test_ioctl_exception(self):
        npu = NPUDevice()
        with patch.object(npu, "load", side_effect=RuntimeError("boom")):
            result = npu.ioctl("LOAD", "/tmp/test.npy")
            assert result.success is False
