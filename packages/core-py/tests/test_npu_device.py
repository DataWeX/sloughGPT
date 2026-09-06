"""Tests for shell.npu_device — NPUDevice."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

from domains.shell.npu_device import NPUDevice
from domains.shell.kernel_syscall import SyscallResult


def _make_provider():
    prov = MagicMock()
    prov.tokenize.return_value = [1, 2, 3]
    prov.generate_numpy.return_value = np.array([[1, 2, 3, 4]])
    prov.detokenize.return_value = "generated"
    prov.embed.return_value = np.zeros(128)
    prov.forward_numpy.return_value = np.random.randn(1, 4, 100)
    prov._model = MagicMock()
    prov._model._params = {
        "w1": np.random.randn(10, 10).astype(np.float32),
        "w2": np.random.randn(10, 10).astype(np.float32),
    }
    return prov


# ── NPUDevice ──────────────────────────────────────────────────────────────


class TestNPUDevice:

    def test_init(self):
        npu = NPUDevice()
        assert npu.name == "npu"
        assert npu._models == {}

    def test_init_custom_name(self):
        npu = NPUDevice(name="custom")
        assert npu.name == "custom"

    def test_info(self):
        npu = NPUDevice()
        info = npu.info()
        assert info["name"] == "npu"
        assert info["type"] == "npu"
        assert "compute_ops" in info

    def test_list_commands(self):
        npu = NPUDevice()
        cmds = npu.list_commands()
        assert "INFO" in cmds
        assert "LOAD" in cmds
        assert "COMPUTE" in cmds

    def test_call_success(self):
        npu = NPUDevice()
        result = npu.call("INFO")
        assert result["name"] == "npu"

    def test_call_failure(self):
        npu = NPUDevice()
        with pytest.raises(Exception):
            npu.call("NONEXISTENT")

    def test_ioctl_info(self):
        npu = NPUDevice()
        result = npu.ioctl("INFO")
        assert result.success is True

    def test_ioctl_unknown_command(self):
        npu = NPUDevice()
        result = npu.ioctl("NONEXISTENT")
        assert result.success is False
        assert "unknown command" in result.error

    def test_ioctl_load(self):
        npu = NPUDevice()
        with patch.object(npu, "_load_numpy", return_value={"type": "model"}):
            result = npu.ioctl("LOAD", "/tmp/model.npy", "mymodel")
        assert result.success is True
        assert "mymodel" in npu._models

    def test_ioctl_load_no_path(self):
        npu = NPUDevice()
        result = npu.ioctl("LOAD")
        assert result.success is False
        assert "requires path" in result.error

    def test_ioctl_unload(self):
        npu = NPUDevice()
        npu._models["test"] = {"type": "model"}
        result = npu.ioctl("UNLOAD", "test")
        assert result.success is True
        assert "test" not in npu._models

    def test_ioctl_unload_no_name(self):
        npu = NPUDevice()
        result = npu.ioctl("UNLOAD")
        assert result.success is False
        assert "requires name" in result.error

    def test_ioctl_call(self):
        npu = NPUDevice()
        prov = _make_provider()
        npu._models["test"] = prov
        result = npu.ioctl("CALL", "test", "hello")
        assert result.success is True

    def test_ioctl_call_no_args(self):
        npu = NPUDevice()
        result = npu.ioctl("CALL")
        assert result.success is False

    def test_ioctl_batch(self):
        npu = NPUDevice()
        prov = _make_provider()
        npu._models["test"] = prov
        result = npu.ioctl("BATCH", "test", ["hello", "world"])
        assert result.success is True

    def test_ioctl_batch_no_args(self):
        npu = NPUDevice()
        result = npu.ioctl("BATCH")
        assert result.success is False

    def test_ioctl_pipeline(self):
        npu = NPUDevice()
        prov = _make_provider()
        npu._models["a"] = prov
        npu._models["b"] = prov
        result = npu.ioctl("PIPELINE", ["a", "b"], "hello")
        assert result.success is True

    def test_ioctl_pipeline_no_args(self):
        npu = NPUDevice()
        result = npu.ioctl("PIPELINE")
        assert result.success is False

    def test_ioctl_profile(self):
        npu = NPUDevice()
        prov = _make_provider()
        npu._models["test"] = prov
        result = npu.ioctl("PROFILE", "test", 64)
        assert result.success is True

    def test_ioctl_profile_no_name(self):
        npu = NPUDevice()
        result = npu.ioctl("PROFILE")
        assert result.success is False

    def test_ioctl_quantize(self):
        npu = NPUDevice()
        prov = _make_provider()
        npu._models["test"] = prov
        result = npu.ioctl("QUANTIZE", "test", 8)
        assert result.success is True

    def test_ioctl_quantize_no_name(self):
        npu = NPUDevice()
        result = npu.ioctl("QUANTIZE")
        assert result.success is False

    def test_ioctl_checkpoint_save(self):
        npu = NPUDevice()
        npu._models["test"] = {"type": "model"}
        result = npu.ioctl("CHECKPOINT_SAVE", "test", "/tmp/ckpt")
        assert result.success is True

    def test_ioctl_checkpoint_save_no_args(self):
        npu = NPUDevice()
        result = npu.ioctl("CHECKPOINT_SAVE")
        assert result.success is False

    def test_ioctl_checkpoint_load(self):
        npu = NPUDevice()
        npu._checkpoints["test"] = {"state": "saved"}
        result = npu.ioctl("CHECKPOINT_LOAD", "test", "/tmp/ckpt")
        assert result.success is True

    def test_ioctl_checkpoint_load_no_checkpoint(self):
        npu = NPUDevice()
        result = npu.ioctl("CHECKPOINT_LOAD", "nonexistent", "/tmp/ckpt")
        assert result.success is True
        assert "error" in result.value

    def test_ioctl_checkpoint_load_no_args(self):
        npu = NPUDevice()
        result = npu.ioctl("CHECKPOINT_LOAD")
        assert result.success is False

    def test_ioctl_memory(self):
        npu = NPUDevice()
        result = npu.ioctl("MEMORY")
        assert result.success is True
        assert "rss_mb" in result.value

    def test_ioctl_compute(self):
        npu = NPUDevice()
        mock_result = SyscallResult.ok({"result": 3})
        with patch.object(npu._compute, "ioctl", return_value=mock_result):
            result = npu.ioctl("COMPUTE", "add", np.array([1]), np.array([2]))
        assert result.success is True

    def test_ioctl_compute_no_args(self):
        npu = NPUDevice()
        result = npu.ioctl("COMPUTE")
        assert result.success is False

    def test_ioctl_unsupported_command(self):
        npu = NPUDevice()
        result = npu.ioctl("UNKNOWN_STRING")
        assert result.success is False
        assert "unknown command" in result.error

    def test_load_model(self):
        npu = NPUDevice()
        with patch.object(npu, "_load_numpy", return_value={"type": "model"}):
            npu.load("/tmp/model.npy", "test")
        assert "test" in npu._models
        assert npu._default_model == "test"

    def test_load_no_path(self):
        npu = NPUDevice()
        with pytest.raises(ValueError, match="no path"):
            npu.load("")

    def test_load_unsupported(self):
        npu = NPUDevice()
        with pytest.raises(ValueError, match="unsupported"):
            npu.load("/tmp/model.xyz")

    def test_unload_model(self):
        npu = NPUDevice()
        npu._models["a"] = {"type": "model"}
        assert npu.unload("a") is True
        assert "a" not in npu._models

    def test_unload_nonexistent(self):
        npu = NPUDevice()
        assert npu.unload("nope") is False

    def test_execute_text(self):
        npu = NPUDevice()
        prov = _make_provider()
        npu._models["test"] = prov
        result = npu.execute("test", "hello")
        assert "text" in result

    def test_execute_tokenize_mode(self):
        npu = NPUDevice()
        prov = _make_provider()
        npu._models["test"] = prov
        result = npu.execute("test", "hello", mode="tokenize")
        assert "tokens" in result

    def test_execute_embed_mode(self):
        npu = NPUDevice()
        prov = _make_provider()
        npu._models["test"] = prov
        result = npu.execute("test", "hello", mode="embed")
        assert "embedding" in result

    def test_execute_tokens(self):
        npu = NPUDevice()
        prov = _make_provider()
        npu._models["test"] = prov
        result = npu.execute("test", [1, 2, 3])
        assert "logits" in result

    def test_execute_other_input(self):
        npu = NPUDevice()
        prov = _make_provider()
        npu._models["test"] = prov
        result = npu.execute("test", 42)
        assert result == {"data": 42}

    def test_execute_no_model(self):
        npu = NPUDevice()
        with pytest.raises(ValueError, match="not loaded"):
            npu.execute("nonexistent", "hello")

    def test_batch(self):
        npu = NPUDevice()
        prov = _make_provider()
        npu._models["test"] = prov
        result = npu.batch("test", ["hello", "world"])
        assert result["count"] == 2

    def test_batch_no_model(self):
        npu = NPUDevice()
        with pytest.raises(ValueError):
            npu.batch("nope", ["hello"])

    def test_pipeline(self):
        npu = NPUDevice()
        prov = _make_provider()
        npu._models["a"] = prov
        npu._models["b"] = prov
        result = npu.pipeline(["a", "b"], "hello")
        assert "trace" in result
        assert len(result["trace"]) == 2

    def test_pipeline_no_model(self):
        npu = NPUDevice()
        with pytest.raises(ValueError):
            npu.pipeline(["nope"], "hello")

    def test_profile(self):
        npu = NPUDevice()
        prov = _make_provider()
        npu._models["test"] = prov
        result = npu.profile("test", seq_len=32, batch_sizes=[1, 2])
        assert len(result["profiles"]) == 2

    def test_profile_no_model(self):
        npu = NPUDevice()
        with pytest.raises(ValueError):
            npu.profile("nope")

    def test_quantize(self):
        npu = NPUDevice()
        prov = _make_provider()
        npu._models["test"] = prov
        result = npu.quantize("test", bits=8)
        assert result["bits"] == 8
        assert result["params_quantized"] > 0

    def test_quantize_no_model(self):
        npu = NPUDevice()
        with pytest.raises(ValueError):
            npu.quantize("nope")

    def test_quantize_no_params(self):
        npu = NPUDevice()
        prov = MagicMock()
        prov._model = MagicMock(spec=[])  # no _params
        npu._models["test"] = prov
        result = npu.quantize("test")
        assert "error" in result

    def test_checkpoint_save_load(self):
        npu = NPUDevice()
        npu._models["test"] = {"type": "model"}
        npu.checkpoint_save("test", "/tmp/ckpt")
        assert "test" in npu._checkpoints
        result = npu.checkpoint_load("test", "/tmp/ckpt")
        assert "loaded" in result

    def test_checkpoint_load_no_checkpoint(self):
        npu = NPUDevice()
        result = npu.checkpoint_load("nope", "/tmp/ckpt")
        assert "error" in result

    def test_memory(self):
        npu = NPUDevice()
        result = npu.memory()
        assert "rss_mb" in result
        assert "models" in result

    def test_unload_default_model_updates(self):
        npu = NPUDevice()
        npu._models["a"] = {"type": "model"}
        npu._models["b"] = {"type": "model"}
        npu._default_model = "a"
        npu.unload("a")
        assert npu._default_model == "b"
