"""
Comprehensive tests for standalone VM devices and DeviceTable.

Tests TensorDevice, NPUDevice, StorageDevice, NetworkDevice, DisplayDevice,
InputDevice, and DeviceTable (kernel_devices.py). Covers ioctl() and call()
interfaces, bit-based fd management, device registration/routing, and error
handling.
"""

from __future__ import annotations

import os
import io
import sys
import socket
import tempfile
import threading
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from domains.shell.tensor_device import TensorDevice
from domains.shell.npu_device import NPUDevice
from domains.shell.storage_device import StorageDevice
from domains.shell.network_device import NetworkDevice
from domains.shell.display_device import DisplayDevice
from domains.shell.input_device import InputDevice
from domains.shell.kernel_devices import (
    DeviceType,
    DeviceState,
    DeviceHandle,
    DeviceDriver,
    DeviceTable,
    DeviceManager,
    NullDevice,
)
from domains.shell.ioctl import IoctlCommand
from domains.shell.kernel_syscall import SyscallResult


# =============================================================================
# TensorDevice
# =============================================================================


class TestTensorDevice:
    """Tests for TensorDevice — standalone compute hardware."""

    @pytest.fixture
    def dev(self):
        return TensorDevice(name="test_tensor")

    # -- info / list --

    def test_name(self, dev):
        assert dev.name == "test_tensor"

    def test_info(self, dev):
        info = dev.info()
        assert info["name"] == "test_tensor"
        assert info["type"] == "tensor"
        assert isinstance(info["commands"], int)
        assert info["commands"] > 0

    def test_list_commands(self, dev):
        cmds = dev.list_commands()
        assert isinstance(cmds, list)
        assert "MATMUL" in cmds
        assert "RELU" in cmds
        assert "SOFTMAX" in cmds
        assert cmds == sorted(cmds)  # sorted

    # -- ioctl interface --

    def test_ioctl_matmul(self, dev):
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        result = dev.ioctl(IoctlCommand.MATMUL, a, b)
        assert result.success
        expected = np.array([[19, 22], [43, 50]])
        assert np.allclose(result.value, expected)

    def test_ioctl_string_command(self, dev):
        result = dev.ioctl("RELU", np.array([-1, 2, -3, 4]))
        assert result.success
        assert np.array_equal(result.value, [0, 2, 0, 4])

    def test_ioctl_unknown_string_command(self, dev):
        result = dev.ioctl("NONEXISTENT")
        assert not result.success
        assert "unknown command" in result.error

    def test_ioctl_unimplemented_command(self, dev):
        # Use an IoctlCommand that exists but isn't in _ops
        result = dev.ioctl(IoctlCommand.INFO)
        assert not result.success
        assert "command not implemented" in result.error

    def test_ioctl_error_returns_failure(self, dev):
        # MATMUL with incompatible shapes
        result = dev.ioctl(IoctlCommand.MATMUL, np.array([1, 2, 3]), np.array([1, 2]))
        assert not result.success
        assert "ioctl error" in result.error

    # -- call interface --

    def test_call_matmul(self, dev):
        result = dev.call("MATMUL", np.array([[1, 2]]), np.array([[3], [4]]))
        assert result == pytest.approx(11.0)

    def test_call_relu(self, dev):
        result = dev.call("RELU", np.array([-1, 2, 0]))
        assert np.array_equal(result, [0, 2, 0])

    def test_call_raises_on_error(self, dev):
        with pytest.raises(Exception, match="unknown command"):
            dev.call("NONEXISTENT")

    # -- linear algebra --

    def test_dot(self, dev):
        result = dev.ioctl(IoctlCommand.DOT, np.array([1, 2, 3]), np.array([4, 5, 6]))
        assert result.success
        assert result.value == pytest.approx(32.0)

    def test_inv(self, dev):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = dev.ioctl(IoctlCommand.INV, a)
        assert result.success
        product = a @ result.value
        assert np.allclose(product, np.eye(2), atol=1e-6)

    def test_svd(self, dev):
        result = dev.ioctl(IoctlCommand.SVD, np.array([[1, 2], [3, 4]]))
        assert result.success
        u, s, vh = result.value
        assert len(s) == 2

    def test_eig(self, dev):
        a = np.array([[2.0, 0.0], [0.0, 3.0]])
        result = dev.ioctl(IoctlCommand.EIG, a)
        assert result.success
        eigenvalues = result.values[0] if hasattr(result, "values") else result.value[0]
        assert len(eigenvalues) == 2

    # -- activation functions --

    def test_leaky_relu(self, dev):
        result = dev.ioctl(IoctlCommand.LEAKY_RELU, np.array([-2, 0, 3]))
        assert result.success
        assert result.value[0] == pytest.approx(-0.02)
        assert result.value[1] == pytest.approx(0.0)
        assert result.value[2] == pytest.approx(3.0)

    def test_sigmoid(self, dev):
        result = dev.ioctl(IoctlCommand.SIGMOID, 0.0)
        assert result.success
        assert result.value == pytest.approx(0.5)

    def test_tanh(self, dev):
        result = dev.ioctl(IoctlCommand.TANH, 0.0)
        assert result.success
        assert result.value == pytest.approx(0.0)

    def test_softmax(self, dev):
        result = dev.ioctl(IoctlCommand.SOFTMAX, np.array([1.0, 2.0, 3.0]))
        assert result.success
        assert result.value.sum() == pytest.approx(1.0)

    def test_log_softmax(self, dev):
        result = dev.ioctl(IoctlCommand.LOG_SOFTMAX, np.array([1.0, 2.0, 3.0]))
        assert result.success
        assert np.all(np.isfinite(result.value))

    def test_gelu(self, dev):
        result = dev.ioctl(IoctlCommand.GELU, np.array([0.0, 1.0, -1.0]))
        assert result.success
        assert result.value[0] == pytest.approx(0.0, abs=0.01)

    def test_silu(self, dev):
        result = dev.ioctl(IoctlCommand.SILU, np.array([0.0, 1.0]))
        assert result.success
        assert result.value[0] == pytest.approx(0.0, abs=0.01)

    def test_elu(self, dev):
        result = dev.ioctl(IoctlCommand.ELU, np.array([-1.0, 0.0, 1.0]))
        assert result.success
        assert result.value[2] == pytest.approx(1.0)

    def test_selu(self, dev):
        result = dev.ioctl(IoctlCommand.SELU, np.array([-1.0, 0.0, 1.0]))
        assert result.success
        assert np.all(np.isfinite(result.value))

    # -- arithmetic --

    def test_add(self, dev):
        result = dev.ioctl(IoctlCommand.ADD, np.array([1, 2]), np.array([3, 4]))
        assert result.success
        assert np.array_equal(result.value, [4, 6])

    def test_sub(self, dev):
        result = dev.ioctl(IoctlCommand.SUB, np.array([10, 5]), np.array([3, 2]))
        assert result.success
        assert np.array_equal(result.value, [7, 3])

    def test_mul(self, dev):
        result = dev.ioctl(IoctlCommand.MUL, np.array([2, 3]), np.array([4, 5]))
        assert result.success
        assert np.array_equal(result.value, [8, 15])

    def test_div(self, dev):
        result = dev.ioctl(IoctlCommand.DIV, np.array([10.0, 9.0]), np.array([2.0, 3.0]))
        assert result.success
        assert np.allclose(result.value, [5.0, 3.0])

    def test_neg(self, dev):
        result = dev.ioctl(IoctlCommand.NEG, np.array([1, -2, 3]))
        assert result.success
        assert np.array_equal(result.value, [-1, 2, -3])

    def test_abs(self, dev):
        result = dev.ioctl(IoctlCommand.ABS, np.array([-5, 3, -1]))
        assert result.success
        assert np.array_equal(result.value, [5, 3, 1])

    def test_pow(self, dev):
        result = dev.ioctl(IoctlCommand.POW, np.array([2.0, 3.0]), 2)
        assert result.success
        assert np.allclose(result.value, [4.0, 9.0])

    def test_sqrt(self, dev):
        result = dev.ioctl(IoctlCommand.SQRT, np.array([4.0, 9.0, 16.0]))
        assert result.success
        assert np.allclose(result.value, [2.0, 3.0, 4.0])

    def test_exp(self, dev):
        result = dev.ioctl(IoctlCommand.EXP, np.array([0.0, 1.0]))
        assert result.success
        assert np.allclose(result.value, [1.0, np.e])

    def test_log(self, dev):
        result = dev.ioctl(IoctlCommand.LOG, np.array([1.0, np.e]))
        assert result.success
        assert np.allclose(result.value, [0.0, 1.0], atol=1e-6)

    # -- reduction --

    def test_sum(self, dev):
        result = dev.ioctl(IoctlCommand.SUM, np.array([1, 2, 3, 4]))
        assert result.success
        assert result.value == pytest.approx(10.0)

    def test_mean(self, dev):
        result = dev.ioctl(IoctlCommand.MEAN, np.array([1, 2, 3, 4]))
        assert result.success
        assert result.value == pytest.approx(2.5)

    def test_std(self, dev):
        result = dev.ioctl(IoctlCommand.STD, np.array([1, 2, 3, 4]))
        assert result.success
        assert result.value > 0

    def test_var(self, dev):
        result = dev.ioctl(IoctlCommand.VAR, np.array([1, 2, 3, 4]))
        assert result.success
        assert result.value > 0

    def test_max(self, dev):
        result = dev.ioctl(IoctlCommand.MAX, np.array([1, 5, 3]))
        assert result.success
        assert result.value == pytest.approx(5.0)

    def test_min(self, dev):
        result = dev.ioctl(IoctlCommand.MIN, np.array([1, 5, 3]))
        assert result.success
        assert result.value == pytest.approx(1.0)

    def test_argmax(self, dev):
        result = dev.ioctl(IoctlCommand.ARGMAX, np.array([1, 5, 3]))
        assert result.success
        assert result.value == 1

    def test_argmin(self, dev):
        result = dev.ioctl(IoctlCommand.ARGMIN, np.array([1, 5, 3]))
        assert result.success
        assert result.value == 0

    # -- shape --

    def test_reshape(self, dev):
        result = dev.ioctl(IoctlCommand.RESHAPE, np.array([1, 2, 3, 4]), (2, 2))
        assert result.success
        assert result.value.shape == (2, 2)

    def test_transpose(self, dev):
        result = dev.ioctl(IoctlCommand.TRANSPOSE, np.array([[1, 2], [3, 4]]))
        assert result.success
        assert result.value.shape == (2, 2)
        assert result.value[0, 1] == 3

    def test_flatten(self, dev):
        result = dev.ioctl(IoctlCommand.FLATTEN, np.array([[1, 2], [3, 4]]))
        assert result.success
        assert result.value.shape == (4,)

    def test_squeeze(self, dev):
        result = dev.ioctl(IoctlCommand.SQUEEZE, np.array([[[1, 2], [3, 4]]]))
        assert result.success
        assert result.value.shape == (2, 2)

    def test_unsqueeze(self, dev):
        result = dev.ioctl(IoctlCommand.UNSQUEEZE, np.array([1, 2, 3]), 0)
        assert result.success
        assert result.value.shape == (1, 3)

    def test_cat(self, dev):
        result = dev.ioctl(IoctlCommand.CAT, [np.array([1, 2]), np.array([3, 4])], 0)
        assert result.success
        assert np.array_equal(result.value, [1, 2, 3, 4])

    def test_stack(self, dev):
        result = dev.ioctl(IoctlCommand.STACK, [np.array([1, 2]), np.array([3, 4])], 0)
        assert result.success
        assert result.value.shape == (2, 2)

    # -- loss functions --

    def test_mse(self, dev):
        result = dev.ioctl(IoctlCommand.MSE, np.array([1.0, 2.0]), np.array([1.5, 2.5]))
        assert result.success
        assert result.value == pytest.approx(0.25)

    def test_mae(self, dev):
        result = dev.ioctl(IoctlCommand.MAE, np.array([1.0, 2.0]), np.array([1.5, 2.5]))
        assert result.success
        assert result.value == pytest.approx(0.5)

    def test_cross_entropy(self, dev):
        logits = np.array([[2.0, 1.0, 0.1]])
        target = np.array([0])
        result = dev.ioctl(IoctlCommand.CROSS_ENTROPY, logits, target)
        assert result.success
        assert result.value > 0

    # -- optimizer --

    def test_sgd_step(self, dev):
        params = {"w": np.array([1.0, 2.0])}
        grads = {"w": np.array([0.1, 0.2])}
        result = dev.ioctl(IoctlCommand.SGD_STEP, params, grads, 0.1)
        assert result.success
        assert "w" in result.value
        assert np.allclose(result.value["w"], [0.99, 1.98])

    def test_adam_step(self, dev):
        params = {"w": np.array([1.0, 2.0])}
        grads = {"w": np.array([0.1, 0.2])}
        state = {}
        result = dev.ioctl(IoctlCommand.ADAM_STEP, params, grads, 0.001, (0.9, 0.999), 1e-8, state)
        assert result.success
        assert "w" in result.value
        assert state  # state should be populated

    # -- utility --

    def test_clip_grad_norm(self, dev):
        grads = {"w": np.array([10.0, 20.0])}
        result = dev.ioctl(IoctlCommand.CLIP_GRAD_NORM, grads, 1.0)
        assert result.success
        assert "w" in result.value
        norm = np.sqrt(np.sum(result.value["w"] ** 2))
        assert norm <= 1.0 + 1e-6

    def test_dropout_training(self, dev):
        result = dev.ioctl(IoctlCommand.DROPOUT, np.ones(1000), 0.5, True)
        assert result.success
        # Some elements should be zeroed
        assert np.any(result.value == 0)

    def test_dropout_eval(self, dev):
        x = np.ones(100)
        result = dev.ioctl(IoctlCommand.DROPOUT, x, 0.5, False)
        assert result.success
        assert np.array_equal(result.value, x)

    def test_embedding(self, dev):
        weight = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        indices = np.array([0, 2])
        result = dev.ioctl(IoctlCommand.EMBEDDING, indices, weight)
        assert result.success
        assert result.value.shape == (2, 3)

    def test_linear(self, dev):
        x = np.array([[1.0, 2.0]])
        w = np.array([[3.0, 4.0], [5.0, 6.0]])
        b = np.array([0.5, 0.5])
        result = dev.ioctl(IoctlCommand.LINEAR, x, w, b)
        assert result.success
        expected = x @ w + b
        assert np.allclose(result.value, expected)

    def test_linear_no_bias(self, dev):
        x = np.array([[1.0, 2.0]])
        w = np.array([[3.0, 4.0], [5.0, 6.0]])
        result = dev.ioctl(IoctlCommand.LINEAR, x, w)
        assert result.success
        expected = x @ w
        assert np.allclose(result.value, expected)

    # -- normalization --

    def test_batch_norm(self, dev):
        x = np.array([[[1.0, 2.0, 3.0]]])
        w = np.array([1.0])
        b = np.array([0.0])
        m = np.array([2.0])
        v = np.array([1.0])
        result = dev.ioctl(IoctlCommand.BATCH_NORM, x, w, b, m, v)
        assert result.success
        assert np.all(np.isfinite(result.value))

    def test_layer_norm(self, dev):
        x = np.array([[1.0, 2.0, 3.0]])
        w = np.ones(3)
        b = np.zeros(3)
        result = dev.ioctl(IoctlCommand.LAYER_NORM, x, w, b)
        assert result.success
        assert result.value.mean() == pytest.approx(0.0, abs=1e-5)

    def test_rms_norm(self, dev):
        x = np.array([[1.0, 2.0, 3.0]])
        w = np.ones(3)
        result = dev.ioctl(IoctlCommand.RMS_NORM, x, w)
        assert result.success
        assert np.all(np.isfinite(result.value))

    def test_attention(self, dev):
        q = np.random.randn(1, 4, 8)
        k = np.random.randn(1, 4, 8)
        v = np.random.randn(1, 4, 8)
        result = dev.ioctl(IoctlCommand.ATTENTION, q, k, v)
        assert result.success
        assert result.value.shape == (1, 4, 8)

    # -- direct function calls --

    def test_direct_matmul(self, dev):
        result = dev.matmul(np.array([[1, 2]]), np.array([[3], [4]]))
        assert result == pytest.approx(11.0)

    def test_direct_relu(self, dev):
        result = dev.relu(np.array([-1, 2, 0]))
        assert np.array_equal(result, [0, 2, 0])

    def test_direct_softmax(self, dev):
        result = dev.softmax(np.array([1.0, 2.0, 3.0]))
        assert result.sum() == pytest.approx(1.0)

    def test_direct_add(self, dev):
        result = dev.add(np.array([1, 2]), np.array([3, 4]))
        assert np.array_equal(result, [4, 6])

    def test_direct_mul(self, dev):
        result = dev.mul(np.array([2, 3]), np.array([4, 5]))
        assert np.array_equal(result, [8, 15])

    def test_direct_linear(self, dev):
        x = np.array([[1.0, 2.0]])
        w = np.array([[3.0, 4.0], [5.0, 6.0]])
        result = dev.linear(x, w)
        assert np.allclose(result, x @ w)

    def test_direct_embedding(self, dev):
        weight = np.array([[0, 1], [2, 3], [4, 5]])
        result = dev.embedding(np.array([0, 2]), weight)
        assert result.shape == (2, 2)

    def test_direct_attention(self, dev):
        q = k = v = np.random.randn(1, 4, 8)
        result = dev.attention(q, k, v)
        assert result.shape == (1, 4, 8)

    def test_direct_cross_entropy(self, dev):
        logits = np.array([[2.0, 1.0, 0.1]])
        target = np.array([0])
        result = dev.cross_entropy(logits, target)
        assert result > 0

    def test_direct_dropout_eval(self, dev):
        x = np.ones(100)
        result = dev.dropout(x, 0.5, training=False)
        assert np.array_equal(result, x)


# =============================================================================
# NPUDevice
# =============================================================================


class TestNPUDevice:
    """Tests for NPUDevice — neural processing hardware."""

    @pytest.fixture
    def dev(self):
        return NPUDevice(name="test_npu")

    @pytest.fixture
    def dev_with_model(self, tmp_path):
        """NPUDevice with a fake model loaded."""
        npu = NPUDevice(name="test_npu_models")
        # Manually inject a fake model provider
        class FakeProvider:
            def __init__(self):
                self._model = MagicMock()
                self._model._params = {
                    "w": np.array([1.0, 2.0, 3.0]),
                    "b": np.array([0.1]),
                }

            def tokenize(self, text):
                return list(range(len(text)))

            def detokenize(self, ids):
                return "".join(chr(i + 65) for i in ids)

            def generate_numpy(self, ids, max_new_tokens=10, **kw):
                return np.array([[1, 2, 3]])

            def forward_numpy(self, ids):
                return np.ones((1, 5, 10))

            def embed(self, text):
                return np.ones(64)

        npu._models["fake_model"] = FakeProvider()
        npu._default_model = "fake_model"
        return npu

    # -- info --

    def test_name(self, dev):
        assert dev.name == "test_npu"

    def test_info(self, dev):
        info = dev.info()
        assert info["name"] == "test_npu"
        assert info["type"] == "npu"
        assert isinstance(info["compute_ops"], list)
        assert info["models"] == 0

    def test_list_commands(self, dev):
        cmds = dev.list_commands()
        assert "LOAD" in cmds
        assert "UNLOAD" in cmds
        assert "CALL" in cmds
        assert "COMPUTE" in cmds
        assert "INFO" in cmds

    # -- ioctl interface --

    def test_ioctl_info(self, dev):
        result = dev.ioctl("INFO")
        assert result.success
        assert result.value["type"] == "npu"

    def test_ioctl_list_commands(self, dev):
        result = dev.ioctl("LIST_COMMANDS")
        assert result.success
        assert isinstance(result.value, list)

    def test_ioctl_unknown_command(self, dev):
        result = dev.ioctl("NONEXISTENT")
        assert not result.success
        assert "unknown command" in result.error

    def test_ioctl_load_requires_path(self, dev):
        result = dev.ioctl("LOAD")
        assert not result.success
        assert "requires path" in result.error

    def test_ioctl_unload_requires_name(self, dev):
        result = dev.ioctl("UNLOAD")
        assert not result.success
        assert "requires name" in result.error

    def test_ioctl_call_requires_args(self, dev):
        result = dev.ioctl("CALL")
        assert not result.success
        assert "requires name and input" in result.error

    def test_ioctl_batch_requires_args(self, dev):
        result = dev.ioctl("BATCH")
        assert not result.success
        assert "requires name and inputs" in result.error

    def test_ioctl_pipeline_requires_args(self, dev):
        result = dev.ioctl("PIPELINE")
        assert not result.success
        assert "requires names and input" in result.error

    def test_ioctl_profile_requires_name(self, dev):
        result = dev.ioctl("PROFILE")
        assert not result.success
        assert "requires name" in result.error

    def test_ioctl_quantize_requires_name(self, dev):
        result = dev.ioctl("QUANTIZE")
        assert not result.success
        assert "requires name" in result.error

    def test_ioctl_checkpoint_save_requires_args(self, dev):
        result = dev.ioctl("CHECKPOINT_SAVE")
        assert not result.success
        assert "requires name and path" in result.error

    def test_ioctl_checkpoint_load_requires_args(self, dev):
        result = dev.ioctl("CHECKPOINT_LOAD")
        assert not result.success
        assert "requires name and path" in result.error

    def test_ioctl_compute_requires_op(self, dev):
        result = dev.ioctl("COMPUTE")
        assert not result.success
        assert "requires op" in result.error

    # -- model management --

    def test_unload_nonexistent(self, dev):
        result = dev.ioctl("UNLOAD", "nope")
        assert result.success
        assert result.value["unloaded"] is False

    def test_call_nonexistent_model(self, dev):
        with pytest.raises(Exception, match="not loaded"):
            dev.call("CALL", "nope", "input")

    def test_execute_model(self, dev_with_model):
        result = dev_with_model.execute("fake_model", "hello")
        assert "tokens" in result or "text" in result

    def test_execute_nonexistent_model(self, dev_with_model):
        with pytest.raises(ValueError, match="not loaded"):
            dev_with_model.execute("missing", "input")

    # -- checkpoint --

    def test_checkpoint_save_and_load(self, dev_with_model):
        result = dev_with_model.checkpoint_save("fake_model", "/tmp/ckpt")
        assert result["saved"] == "fake_model"
        result = dev_with_model.checkpoint_load("fake_model", "/tmp/ckpt")
        assert result["loaded"] == "fake_model"

    def test_checkpoint_load_nonexistent(self, dev):
        result = dev.checkpoint_load("nope", "/tmp/ckpt")
        assert "error" in result

    # -- compute passthrough --

    def test_compute_relu_via_npu(self, dev):
        result = dev.ioctl("COMPUTE", "RELU", np.array([-1, 2, 0]))
        assert result.success
        assert np.array_equal(result.value, [0, 2, 0])


# =============================================================================
# StorageDevice
# =============================================================================


class TestStorageDevice:
    """Tests for StorageDevice — file system operations."""

    @pytest.fixture
    def dev(self, tmp_path):
        return StorageDevice(name="test_storage", base_path=str(tmp_path))

    # -- info --

    def test_name(self, dev):
        assert dev.name == "test_storage"

    def test_info(self, dev):
        info = dev.info()
        assert info["name"] == "test_storage"
        assert info["type"] == "storage"
        assert info["open_files"] == 0

    def test_list_commands(self, dev):
        cmds = dev.list_commands()
        assert "READ" in cmds
        assert "WRITE" in cmds
        assert "OPEN" in cmds
        assert "CLOSE" in cmds
        assert "STAT" in cmds
        assert "LIST" in cmds

    # -- ioctl interface --

    def test_ioctl_unknown_command(self, dev):
        result = dev.ioctl("NONEXISTENT")
        assert not result.success
        assert "unknown command" in result.error

    def test_ioctl_info(self, dev):
        result = dev.ioctl("INFO")
        assert result.success
        assert result.value["type"] == "storage"

    # -- file operations --

    def test_open_close(self, dev, tmp_path):
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        fd = dev.open_file("test.txt", "r")
        assert fd >= 0
        assert dev.close_file(fd) is True

    def test_read_write(self, dev, tmp_path):
        test_file = tmp_path / "rw.txt"
        test_file.write_bytes(b"")

        fd = dev.open_file("rw.txt", "r+b")
        dev.write(fd, b"hello")
        dev.seek(fd, 0)
        data = dev.read(fd)
        assert data == b"hello"
        dev.close_file(fd)

    def test_read_with_size(self, dev, tmp_path):
        test_file = tmp_path / "partial.txt"
        test_file.write_bytes(b"abcdef")

        fd = dev.open_file("partial.txt", "rb")
        data = dev.read(fd, 3)
        assert data == b"abc"
        dev.close_file(fd)

    def test_read_bad_fd(self, dev):
        with pytest.raises(ValueError, match="bad fd"):
            dev.read(999)

    def test_write_bad_fd(self, dev):
        with pytest.raises(ValueError, match="bad fd"):
            dev.write(999, b"data")

    def test_close_bad_fd(self, dev):
        assert dev.close_file(999) is False

    def test_seek(self, dev, tmp_path):
        test_file = tmp_path / "seek.txt"
        test_file.write_bytes(b"abcdef")

        fd = dev.open_file("seek.txt", "rb")
        dev.seek(fd, 3)
        assert dev.tell(fd) == 3
        data = dev.read(fd, 2)
        assert data == b"de"
        dev.close_file(fd)

    def test_seek_bad_fd(self, dev):
        with pytest.raises(ValueError, match="bad fd"):
            dev.seek(999, 0)

    def test_tell_bad_fd(self, dev):
        with pytest.raises(ValueError, match="bad fd"):
            dev.tell(999)

    # -- directory operations --

    def test_stat(self, dev, tmp_path):
        test_file = tmp_path / "stat.txt"
        test_file.write_text("hello")

        result = dev.stat("stat.txt")
        assert result["size"] == 5

    def test_list_dir(self, dev, tmp_path):
        (tmp_path / "a.txt").write_text("")
        (tmp_path / "b.txt").write_text("")

        files = dev.list_dir(".")
        assert "a.txt" in files
        assert "b.txt" in files

    def test_mkdir(self, dev, tmp_path):
        assert dev.mkdir("subdir") is True
        assert (tmp_path / "subdir").is_dir()

    def test_remove(self, dev, tmp_path):
        (tmp_path / "remove_me.txt").write_text("")
        assert dev.remove("remove_me.txt") is True
        assert not (tmp_path / "remove_me.txt").exists()

    def test_rename(self, dev, tmp_path):
        (tmp_path / "old.txt").write_text("")
        assert dev.rename("old.txt", "new.txt") is True
        assert (tmp_path / "new.txt").exists()
        assert not (tmp_path / "old.txt").exists()

    def test_exists(self, dev, tmp_path):
        (tmp_path / "exists.txt").write_text("")
        assert dev.exists("exists.txt") is True
        assert dev.exists("nope.txt") is False

    # -- ioctl dispatch --

    def test_ioctl_read(self, dev, tmp_path):
        test_file = tmp_path / "ioctl_read.txt"
        test_file.write_bytes(b"test data")

        fd = dev.ioctl("OPEN", "ioctl_read.txt", "rb")
        assert fd.success

        data = dev.ioctl("READ", fd.value)
        assert data.success
        assert data.value == b"test data"

        dev.ioctl("CLOSE", fd.value)

    def test_ioctl_write(self, dev, tmp_path):
        test_file = tmp_path / "ioctl_write.txt"
        test_file.write_bytes(b"")

        fd = dev.ioctl("OPEN", "ioctl_write.txt", "r+b")
        assert fd.success

        result = dev.ioctl("WRITE", fd.value, b"hello")
        assert result.success
        assert result.value == 5

        dev.ioctl("CLOSE", fd.value)

    def test_ioctl_stat(self, dev, tmp_path):
        (tmp_path / "stat_me.txt").write_text("data")
        result = dev.ioctl("STAT", "stat_me.txt")
        assert result.success
        assert result.value["size"] == 4

    def test_ioctl_list(self, dev, tmp_path):
        (tmp_path / "x.txt").write_text("")
        result = dev.ioctl("LIST", ".")
        assert result.success
        assert "x.txt" in result.value

    def test_ioctl_mkdir(self, dev):
        result = dev.ioctl("MKDIR", "new_dir")
        assert result.success
        assert result.value is True

    def test_ioctl_remove(self, dev, tmp_path):
        (tmp_path / "rm.txt").write_text("")
        result = dev.ioctl("REMOVE", "rm.txt")
        assert result.success

    def test_ioctl_rename(self, dev, tmp_path):
        (tmp_path / "a.txt").write_text("")
        result = dev.ioctl("RENAME", "a.txt", "b.txt")
        assert result.success

    def test_ioctl_exists(self, dev, tmp_path):
        (tmp_path / "exists.txt").write_text("")
        result = dev.ioctl("EXISTS", "exists.txt")
        assert result.success
        assert result.value is True

    # -- call interface --

    def test_call(self, dev, tmp_path):
        (tmp_path / "call.txt").write_text("data")
        result = dev.call("STAT", "call.txt")
        assert result["size"] == 4


# =============================================================================
# NetworkDevice
# =============================================================================


class TestNetworkDevice:
    """Tests for NetworkDevice — socket operations."""

    @pytest.fixture
    def dev(self):
        return NetworkDevice(name="test_network")

    # -- info --

    def test_name(self, dev):
        assert dev.name == "test_network"

    def test_info(self, dev):
        info = dev.info()
        assert info["name"] == "test_network"
        assert info["type"] == "network"
        assert info["open_sockets"] == 0

    def test_list_commands(self, dev):
        cmds = dev.list_commands()
        assert "TCP_CONNECT" in cmds
        assert "TCP_LISTEN" in cmds
        assert "DNS_RESOLVE" in cmds
        assert "HTTP_GET" in cmds

    # -- ioctl interface --

    def test_ioctl_unknown_command(self, dev):
        result = dev.ioctl("NONEXISTENT")
        assert not result.success
        assert "unknown command" in result.error

    def test_ioctl_info(self, dev):
        result = dev.ioctl("INFO")
        assert result.success
        assert result.value["type"] == "network"

    # -- TCP operations --

    def test_tcp_listen_and_close(self, dev):
        fd = dev.tcp_listen("127.0.0.1", 0)  # port 0 = OS assigns
        assert fd >= 0
        assert dev.tcp_close(fd) is True
        assert fd not in dev._sockets

    def test_tcp_close_nonexistent(self, dev):
        assert dev.tcp_close(999) is False

    def test_tcp_send_bad_fd(self, dev):
        with pytest.raises(ValueError, match="bad fd"):
            dev.tcp_send(999, b"data")

    def test_tcp_recv_bad_fd(self, dev):
        with pytest.raises(ValueError, match="bad fd"):
            dev.tcp_recv(999)

    def test_tcp_accept_bad_fd(self, dev):
        with pytest.raises(ValueError, match="bad fd"):
            dev.tcp_accept(999)

    def test_tcp_connect_and_recv(self, dev):
        """Start a server, connect, send data, receive it."""
        # Start a simple echo server
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("127.0.0.1", 0))
        server_sock.listen(1)
        port = server_sock.getsockname()[1]

        def echo_server():
            conn, _ = server_sock.accept()
            data = conn.recv(1024)
            conn.sendall(data)
            conn.close()
            server_sock.close()

        t = threading.Thread(target=echo_server, daemon=True)
        t.start()

        # Connect via NetworkDevice
        fd = dev.tcp_connect("127.0.0.1", port)
        assert fd >= 0

        dev.tcp_send(fd, b"hello")
        data = dev.tcp_recv(fd, 1024)
        assert data == b"hello"

        dev.tcp_close(fd)
        t.join(timeout=2)

    def test_tcp_ioctl_dispatch(self, dev):
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("127.0.0.1", 0))
        server_sock.listen(1)
        port = server_sock.getsockname()[1]

        t = threading.Thread(target=lambda: server_sock.close(), daemon=True)
        t.start()

        fd_result = dev.ioctl("TCP_LISTEN", "127.0.0.1", port)
        assert fd_result.success
        fd = fd_result.value

        close_result = dev.ioctl("TCP_CLOSE", fd)
        assert close_result.success

        t.join(timeout=1)

    # -- UDP operations --

    def test_udp_send_recv(self, dev):
        """UDP send/receive via ioctl (fire-and-forget for send)."""
        # Just test that the methods don't crash on valid inputs
        # UDP is connectionless, so we test send then recv
        # This is tricky to test without blocking, so we just verify the interface
        with patch("socket.socket") as mock_sock_cls:
            mock_sock = MagicMock()
            mock_sock.sendto.return_value = 5
            mock_sock_cls.return_value = mock_sock

            result = dev.udp_send("127.0.0.1", 12345, b"test")
            assert result == 5

    # -- DNS --

    def test_dns_resolve_local(self, dev):
        with patch("socket.gethostbyname") as mock_resolve:
            mock_resolve.return_value = "127.0.0.1"
            result = dev.dns_resolve("localhost")
            assert result == "127.0.0.1"

    def test_dns_resolve_ioctl(self, dev):
        with patch("socket.gethostbyname") as mock_resolve:
            mock_resolve.return_value = "1.2.3.4"
            result = dev.ioctl("DNS_RESOLVE", "example.com")
            assert result.success
            assert result.value == "1.2.3.4"

    # -- HTTP --

    def test_http_get(self, dev):
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = b"<html>ok</html>"
            mock_response.__enter__ = lambda s: s
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            result = dev.http_get("http://example.com")
            assert result == b"<html>ok</html>"

    def test_http_post(self, dev):
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = b"ok"
            mock_response.__enter__ = lambda s: s
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            result = dev.http_post("http://example.com", b"data")
            assert result == b"ok"

    # -- call interface --

    def test_call_info(self, dev):
        result = dev.call("INFO")
        assert result["type"] == "network"

    def test_call_raises_on_error(self, dev):
        with pytest.raises(Exception, match="unknown command"):
            dev.call("NONEXISTENT")


# =============================================================================
# DisplayDevice
# =============================================================================


class TestDisplayDevice:
    """Tests for DisplayDevice — output operations."""

    @pytest.fixture
    def dev(self):
        return DisplayDevice(name="test_display")

    # -- info --

    def test_name(self, dev):
        assert dev.name == "test_display"

    def test_info(self, dev):
        info = dev.info()
        assert info["name"] == "test_display"
        assert info["type"] == "display"
        assert info["buffer_size"] == 0
        assert info["cursor"] == (0, 0)

    def test_list_commands(self, dev):
        cmds = dev.list_commands()
        assert "PRINT" in cmds
        assert "PRINTLN" in cmds
        assert "CLEAR" in cmds
        assert "MOVE" in cmds
        assert "COLOR" in cmds
        assert "STYLE" in cmds
        assert "FLUSH" in cmds
        assert "WRITE" in cmds

    # -- ioctl interface --

    def test_ioctl_unknown_command(self, dev):
        result = dev.ioctl("NONEXISTENT")
        assert not result.success
        assert "unknown command" in result.error

    def test_ioctl_print(self, dev, capsys):
        result = dev.ioctl("PRINT", "hello")
        assert result.success
        assert result.value == 5
        captured = capsys.readouterr()
        assert "hello" in captured.out

    def test_ioctl_println(self, dev, capsys):
        result = dev.ioctl("PRINTLN", "world")
        assert result.success
        assert result.value == 6
        captured = capsys.readouterr()
        assert "world\n" in captured.out

    def test_ioctl_println_empty(self, dev, capsys):
        result = dev.ioctl("PRINTLN")
        assert result.success
        assert result.value == 1

    def test_ioctl_clear(self, dev, capsys):
        result = dev.ioctl("CLEAR")
        assert result.success
        assert result.value is True
        captured = capsys.readouterr()
        assert "\033[2J\033[H" in captured.out

    def test_ioctl_move(self, dev, capsys):
        result = dev.ioctl("MOVE", 5, 10)
        assert result.success
        assert dev._cursor_x == 5
        assert dev._cursor_y == 10

    def test_ioctl_color(self, dev, capsys):
        result = dev.ioctl("COLOR", "red", "blue")
        assert result.success
        assert result.value is True

    def test_ioctl_style(self, dev, capsys):
        result = dev.ioctl("STYLE", True, False)
        assert result.success
        captured = capsys.readouterr()
        assert "\033[1m" in captured.out

    def test_ioctl_flush(self, dev):
        result = dev.ioctl("FLUSH")
        assert result.success

    def test_ioctl_write(self, dev, capsys):
        result = dev.ioctl("WRITE", "raw data")
        assert result.success
        assert result.value == 8

    def test_ioctl_info(self, dev):
        result = dev.ioctl("INFO")
        assert result.success
        assert result.value["type"] == "display"

    # -- direct function calls --

    def test_print_text(self, dev, capsys):
        n = dev.print_text("hello")
        assert n == 5
        captured = capsys.readouterr()
        assert "hello" in captured.out

    def test_println(self, dev, capsys):
        n = dev.println("line")
        assert n == 5
        captured = capsys.readouterr()
        assert "line\n" in captured.out

    def test_clear(self, dev, capsys):
        dev.clear()
        assert dev._cursor_x == 0
        assert dev._cursor_y == 0

    def test_move(self, dev):
        dev.move(3, 7)
        assert dev._cursor_x == 3
        assert dev._cursor_y == 7

    def test_flush(self, dev):
        assert dev.flush() is True

    def test_write(self, dev, capsys):
        n = dev.write("test")
        assert n == 4

    # -- call interface --

    def test_call(self, dev, capsys):
        result = dev.call("PRINT", "hi")
        assert result == 2


# =============================================================================
# InputDevice
# =============================================================================


class TestInputDevice:
    """Tests for InputDevice — user input operations."""

    @pytest.fixture
    def dev(self):
        return InputDevice(name="test_input")

    # -- info --

    def test_name(self, dev):
        assert dev.name == "test_input"

    def test_info(self, dev):
        info = dev.info()
        assert info["name"] == "test_input"
        assert info["type"] == "input"

    def test_list_commands(self, dev):
        cmds = dev.list_commands()
        assert "READLINE" in cmds
        assert "READCHAR" in cmds
        assert "READKEY" in cmds
        assert "POLL" in cmds
        assert "FLUSH" in cmds

    # -- ioctl interface --

    def test_ioctl_unknown_command(self, dev):
        result = dev.ioctl("NONEXISTENT")
        assert not result.success
        assert "unknown command" in result.error

    def test_ioctl_readline(self, dev):
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.readline.return_value = "hello\n"
            result = dev.ioctl("READLINE")
            assert result.success
            assert result.value == "hello"

    def test_ioctl_readchar(self, dev):
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.read.return_value = "a"
            result = dev.ioctl("READCHAR")
            assert result.success
            assert result.value == "a"

    def test_ioctl_poll(self, dev):
        with patch("select.select") as mock_select:
            mock_select.return_value = ([True], [], [])
            result = dev.ioctl("POLL", 0.0)
            assert result.success
            assert result.value is True

    def test_ioctl_poll_no_data(self, dev):
        with patch("select.select") as mock_select:
            mock_select.return_value = ([], [], [])
            result = dev.ioctl("POLL", 0.0)
            assert result.success
            assert result.value is False

    def test_ioctl_flush(self, dev):
        with patch("termios.tcflush") as mock_flush:
            result = dev.ioctl("FLUSH")
            assert result.success
            assert result.value is True

    def test_ioctl_info(self, dev):
        result = dev.ioctl("INFO")
        assert result.success
        assert result.value["type"] == "input"

    # -- direct function calls --

    def test_readline(self, dev):
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.readline.return_value = "test\n"
            assert dev.readline() == "test"

    def test_readchar(self, dev):
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.read.return_value = "x"
            assert dev.readchar() == "x"

    # -- call interface --

    def test_call(self, dev):
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.readline.return_value = "input\n"
            result = dev.call("READLINE")
            assert result == "input"

    def test_call_raises_on_error(self, dev):
        with pytest.raises(Exception, match="unknown command"):
            dev.call("NONEXISTENT")


# =============================================================================
# DeviceTable (kernel_devices.py)
# =============================================================================


class TestDeviceTable:
    """Tests for DeviceTable — bit-based fd management and ioctl dispatch."""

    @pytest.fixture
    def table(self):
        return DeviceTable(max_fds=64)

    @pytest.fixture
    def sample_device(self):
        dev = DeviceDriver("sample", DeviceType.INFERENCE)
        dev.ioctl = lambda cmd, *args: SyscallResult.ok(f"handled:{cmd}")
        return dev

    # -- FD bitmap operations --

    def test_alloc_fd(self, table):
        fd = table._alloc_fd()
        assert fd == 0
        assert table._fd_is_open(0)

    def test_alloc_multiple_fds(self, table):
        fds = [table._alloc_fd() for _ in range(5)]
        assert fds == [0, 1, 2, 3, 4]
        for fd in fds:
            assert table._fd_is_open(fd)

    def test_free_fd(self, table):
        fd = table._alloc_fd()
        assert fd == 0
        table._free_fd(fd)
        assert not table._fd_is_open(fd)

    def test_alloc_after_free(self, table):
        fd0 = table._alloc_fd()
        fd1 = table._alloc_fd()
        table._free_fd(fd0)
        fd_new = table._alloc_fd()
        assert fd_new == fd0  # reuses freed fd

    def test_alloc_exhausted(self, table):
        small_table = DeviceTable(max_fds=3)
        fds = [small_table._alloc_fd() for _ in range(3)]
        assert fds == [0, 1, 2]
        assert small_table._alloc_fd() == -1

    def test_fd_bitmap_accumulates(self, table):
        fd0 = table._alloc_fd()
        fd1 = table._alloc_fd()
        # bitmap should have bits 0 and 1 set
        assert table._fd_bitmap & (1 << fd0) != 0
        assert table._fd_bitmap & (1 << fd1) != 0
        table._free_fd(fd0)
        assert table._fd_bitmap & (1 << fd0) == 0
        assert table._fd_bitmap & (1 << fd1) != 0

    def test_fd_is_open(self, table):
        assert not table._fd_is_open(0)
        fd = table._alloc_fd()
        assert table._fd_is_open(fd)
        table._free_fd(fd)
        assert not table._fd_is_open(fd)

    # -- Device registration --

    def test_register_device(self, table, sample_device):
        result = table.register(sample_device, DeviceType.INFERENCE)
        assert result is True
        assert table.get("sample") is sample_device

    def test_register_duplicate_name(self, table, sample_device):
        table.register(sample_device)
        result = table.register(sample_device)
        assert result is False

    def test_unregister_device(self, table, sample_device):
        table.register(sample_device)
        result = table.unregister("sample")
        assert result is True
        assert table.get("sample") is None

    def test_get_nonexistent(self, table):
        assert table.get("nope") is None

    # -- Open / Close --

    def test_open_device(self, table, sample_device):
        table.register(sample_device, DeviceType.INFERENCE)
        fd = table.open("sample")
        assert fd >= 0
        assert table._fd_is_open(fd)

    def test_open_nonexistent_device(self, table):
        fd = table.open("nope")
        assert fd == -1

    def test_open_when_exhausted(self, sample_device):
        small_table = DeviceTable(max_fds=1)
        small_table.register(sample_device)
        fd1 = small_table.open("sample")
        assert fd1 >= 0
        fd2 = small_table.open("sample")
        assert fd2 == -1

    def test_close_fd(self, table, sample_device):
        table.register(sample_device)
        fd = table.open("sample")
        result = table.close(fd)
        assert result is True
        assert not table._fd_is_open(fd)

    def test_close_bad_fd_negative(self, table):
        assert table.close(-1) is False

    def test_close_bad_fd_too_large(self, table):
        assert table.close(999) is False

    def test_close_already_closed(self, table):
        assert table.close(0) is False

    # -- ioctl dispatch --

    def test_ioctl_dispatch(self, table, sample_device):
        table.register(sample_device)
        fd = table.open("sample")
        result = table.ioctl(fd, "TEST_CMD")
        assert result.success
        assert result.value == "handled:TEST_CMD"

    def test_ioctl_bad_fd_negative(self, table):
        result = table.ioctl(-1, "CMD")
        assert not result.success
        assert "bad fd" in result.error

    def test_ioctl_bad_fd_too_large(self, table):
        result = table.ioctl(999, "CMD")
        assert not result.success
        assert "bad fd" in result.error

    def test_ioctl_fd_not_open(self, table):
        result = table.ioctl(0, "CMD")
        assert not result.success
        assert "fd not open" in result.error

    def test_ioctl_device_not_found(self, table):
        # Manually set fd bitmap without device
        table._fd_bitmap |= (1 << 0)
        result = table.ioctl(0, "CMD")
        assert not result.success
        assert "no device" in result.error

    # -- Stats --

    def test_stats_empty(self, table):
        stats = table.stats()
        assert stats["total_devices"] == 0
        assert stats["open_fds"] == 0

    def test_stats_with_device(self, table, sample_device):
        table.register(sample_device)
        fd = table.open("sample")
        stats = table.stats()
        assert stats["total_devices"] == 1
        assert stats["open_fds"] == 1

    def test_list_devices(self, table, sample_device):
        table.register(sample_device)
        devices = table.list_devices()
        assert len(devices) == 1
        assert devices[0]["name"] == "sample"


# =============================================================================
# DeviceManager (backward compat wrapper)
# =============================================================================


class TestDeviceManager:
    """Tests for DeviceManager — high-level device manager."""

    @pytest.fixture
    def mgr(self):
        return DeviceManager()

    @pytest.fixture
    def sample_driver(self):
        dev = DeviceDriver("sample_dev", DeviceType.STORAGE)
        dev.ioctl = lambda cmd, *args: SyscallResult.ok(f"ok:{cmd}")
        return dev

    def test_register(self, mgr, sample_driver):
        assert mgr.register(sample_driver) is True
        assert mgr.get("sample_dev") is sample_driver

    def test_unregister(self, mgr, sample_driver):
        mgr.register(sample_driver)
        assert mgr.unregister("sample_dev") is True
        assert mgr.get("sample_dev") is None

    def test_names(self, mgr, sample_driver):
        mgr.register(sample_driver)
        assert "sample_dev" in mgr.names

    def test_open_close_ioctl(self, mgr, sample_driver):
        mgr.register(sample_driver)
        fd = mgr.open("sample_dev")
        assert fd >= 0
        result = mgr.ioctl(fd, "TEST")
        assert result.success
        assert result.value == "ok:TEST"
        assert mgr.close(fd) is True

    def test_list_devices(self, mgr, sample_driver):
        mgr.register(sample_driver)
        devices = mgr.list_devices()
        assert len(devices) == 1

    def test_stats(self, mgr, sample_driver):
        mgr.register(sample_driver)
        stats = mgr.stats()
        assert stats["total_devices"] == 1


# =============================================================================
# NullDevice
# =============================================================================


class TestNullDevice:
    """Tests for NullDevice — /dev/null equivalent."""

    def test_info(self):
        dev = NullDevice()
        info = dev.info()
        assert info["name"] == "null"
        assert info["type"] == DeviceType.CUSTOM

    def test_ioctl(self):
        dev = NullDevice()
        result = dev.ioctl("ANYTHING")
        assert result.success
        assert result.value == b""

    def test_ioctl_with_args(self):
        dev = NullDevice()
        result = dev.ioctl("WRITE", "data")
        assert result.success


# =============================================================================
# DeviceDriver base class
# =============================================================================


class TestDeviceDriver:
    """Tests for DeviceDriver base class."""

    def test_default_state(self):
        dev = DeviceDriver("test")
        assert dev.name == "test"
        assert dev.device_type == DeviceType.CUSTOM
        assert dev.state == DeviceState.CLOSED

    def test_custom_type(self):
        dev = DeviceDriver("test", DeviceType.DISPLAY)
        assert dev.device_type == DeviceType.DISPLAY

    def test_ioctl_not_implemented(self):
        dev = DeviceDriver("test")
        with pytest.raises(NotImplementedError, match="not implemented"):
            dev.ioctl("CMD")

    def test_info(self):
        dev = DeviceDriver("mydev", DeviceType.NETWORK)
        info = dev.info()
        assert info["name"] == "mydev"
        assert info["type"] == DeviceType.NETWORK
        assert info["state"] == "CLOSED"


# =============================================================================
# Thread safety
# =============================================================================


class TestDeviceTableThreadSafety:
    """Tests for DeviceTable thread safety."""

    def test_concurrent_open_close(self):
        table = DeviceTable(max_fds=32)
        dev = DeviceDriver("concurrent_test")
        dev.ioctl = lambda cmd, *args: SyscallResult.ok()
        table.register(dev)

        errors = []

        def worker():
            try:
                fd = table.open("concurrent_test")
                if fd >= 0:
                    table.ioctl(fd, "PING")
                    table.close(fd)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(errors) == 0
        assert table.stats()["open_fds"] == 0


# =============================================================================
# Integration: DeviceTable + standalone devices
# =============================================================================


class TestDeviceTableIntegration:
    """Integration tests: register standalone devices in DeviceTable."""

    def test_tensor_device_in_table(self):
        table = DeviceTable()
        tensor = TensorDevice("tensor0")
        # Wrap TensorDevice as a DeviceDriver
        class TensorDriver(DeviceDriver):
            def __init__(self):
                super().__init__("tensor0", DeviceType.INFERENCE)
                self._tensor = tensor

            def ioctl(self, command, *args):
                return self._tensor.ioctl(command, *args)

        driver = TensorDriver()
        table.register(driver)
        fd = table.open("tensor0")
        assert fd >= 0

        result = table.ioctl(fd, "RELU", np.array([-1, 2]))
        assert result.success
        assert np.array_equal(result.value, [0, 2])

        table.close(fd)

    def test_storage_device_in_table(self):
        table = DeviceTable()
        storage = StorageDevice("storage0", base_path="/tmp")

        class StorageDriver(DeviceDriver):
            def __init__(self):
                super().__init__("storage0", DeviceType.STORAGE)
                self._storage = storage

            def ioctl(self, command, *args):
                return self._storage.ioctl(command, *args)

        driver = StorageDriver()
        table.register(driver)
        fd = table.open("storage0")
        assert fd >= 0

        result = table.ioctl(fd, "INFO")
        assert result.success
        assert result.value["type"] == "storage"

        table.close(fd)

    def test_display_device_in_table(self):
        table = DeviceTable()
        display = DisplayDevice("display0")

        class DisplayDriver(DeviceDriver):
            def __init__(self):
                super().__init__("display0", DeviceType.DISPLAY)
                self._display = display

            def ioctl(self, command, *args):
                return self._display.ioctl(command, *args)

        driver = DisplayDriver()
        table.register(driver)
        fd = table.open("display0")
        assert fd >= 0

        result = table.ioctl(fd, "INFO")
        assert result.success
        assert result.value["type"] == "display"

        table.close(fd)

    def test_multiple_devices_routing(self):
        table = DeviceTable()

        class FakeDriver(DeviceDriver):
            def __init__(self, name, dtype, tag):
                super().__init__(name, dtype)
                self._tag = tag

            def ioctl(self, command, *args):
                return SyscallResult.ok(self._tag)

        table.register(FakeDriver("dev_a", DeviceType.INFERENCE, "A"))
        table.register(FakeDriver("dev_b", DeviceType.STORAGE, "B"))

        fd_a = table.open("dev_a")
        fd_b = table.open("dev_b")

        assert table.ioctl(fd_a, "PING").value == "A"
        assert table.ioctl(fd_b, "PING").value == "B"

        table.close(fd_a)
        table.close(fd_b)
