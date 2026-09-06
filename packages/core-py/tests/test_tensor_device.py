"""Tests for shell.tensor_device — TensorDevice compute operations."""

from __future__ import annotations

import pytest
import numpy as np

from domains.shell.tensor_device import TensorDevice
from domains.shell.kernel_syscall import SyscallResult


# ── TensorDevice ───────────────────────────────────────────────────────────


class TestTensorDevice:

    def test_init(self):
        td = TensorDevice()
        assert td.name == "tensor"

    def test_init_custom_name(self):
        td = TensorDevice(name="custom")
        assert td.name == "custom"

    def test_info(self):
        td = TensorDevice()
        info = td.info()
        assert info["type"] == "tensor"
        assert info["commands"] > 0

    def test_list_commands(self):
        td = TensorDevice()
        cmds = td.list_commands()
        assert "ADD" in cmds
        assert "RELU" in cmds
        assert "MATMUL" in cmds

    def test_call_success(self):
        td = TensorDevice()
        result = td.call("ADD", np.array([1]), np.array([2]))
        assert result == np.array([3])

    def test_call_failure(self):
        td = TensorDevice()
        with pytest.raises(Exception):
            td.call("NONEXISTENT")

    def test_ioctl_success(self):
        td = TensorDevice()
        result = td.ioctl("ADD", np.array([1]), np.array([2]))
        assert result.success is True
        assert result.value == np.array([3])

    def test_ioctl_unknown_command(self):
        td = TensorDevice()
        result = td.ioctl("NONEXISTENT")
        assert result.success is False
        assert "unknown command" in result.error

    def test_ioctl_exception(self):
        td = TensorDevice()
        result = td.ioctl("MATMUL", "bad", "input")
        assert result.success is False


# ── Linear Algebra ─────────────────────────────────────────────────────────


class TestLinearAlgebra:

    def test_matmul(self):
        td = TensorDevice()
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        result = td.matmul(a, b)
        expected = np.array([[19, 22], [43, 50]])
        assert np.allclose(result, expected)

    def test_dot(self):
        td = TensorDevice()
        result = td.ioctl("DOT", np.array([1, 2, 3]), np.array([4, 5, 6]))
        assert result.success is True
        assert result.value == 32

    def test_inv(self):
        td = TensorDevice()
        a = np.array([[1, 2], [3, 4]])
        result = td.ioctl("INV", a)
        assert result.success is True
        assert np.allclose(result.value @ a, np.eye(2))

    def test_svd(self):
        td = TensorDevice()
        result = td.ioctl("SVD", np.array([[1, 2], [3, 4]]))
        assert result.success is True
        assert len(result.value) == 3

    def test_eig(self):
        td = TensorDevice()
        result = td.ioctl("EIG", np.array([[1, 2], [3, 4]]))
        assert result.success is True


# ── Activation Functions ───────────────────────────────────────────────────


class TestActivations:

    def test_relu(self):
        td = TensorDevice()
        result = td.relu(np.array([-1, 0, 1, 2]))
        assert np.allclose(result, [0, 0, 1, 2])

    def test_leaky_relu(self):
        td = TensorDevice()
        result = td.ioctl("LEAKY_RELU", np.array([-1, 1]), 0.01)
        assert result.success is True
        assert result.value[1] == 1.0

    def test_sigmoid(self):
        td = TensorDevice()
        result = td.ioctl("SIGMOID", np.array([0]))
        assert result.success is True
        assert np.allclose(result.value, 0.5, atol=1e-6)

    def test_tanh(self):
        td = TensorDevice()
        result = td.ioctl("TANH", np.array([0]))
        assert result.success is True
        assert np.allclose(result.value, 0.0)

    def test_softmax(self):
        td = TensorDevice()
        result = td.softmax(np.array([1, 2, 3]))
        assert np.allclose(np.sum(result), 1.0)

    def test_log_softmax(self):
        td = TensorDevice()
        result = td.ioctl("LOG_SOFTMAX", np.array([1, 2, 3]))
        assert result.success is True

    def test_gelu(self):
        td = TensorDevice()
        result = td.ioctl("GELU", np.array([0]))
        assert result.success is True

    def test_silu(self):
        td = TensorDevice()
        result = td.ioctl("SILU", np.array([1]))
        assert result.success is True

    def test_elu(self):
        td = TensorDevice()
        result = td.ioctl("ELU", np.array([-1, 1]))
        assert result.success is True

    def test_selu(self):
        td = TensorDevice()
        result = td.ioctl("SELU", np.array([-1, 1]))
        assert result.success is True


# ── Arithmetic ──────────────────────────────────────────────────────────────


class TestArithmetic:

    def test_add(self):
        td = TensorDevice()
        assert np.allclose(td.add(np.array([1, 2]), np.array([3, 4])), [4, 6])

    def test_sub(self):
        td = TensorDevice()
        result = td.ioctl("SUB", np.array([5, 6]), np.array([1, 2]))
        assert np.allclose(result.value, [4, 4])

    def test_mul(self):
        td = TensorDevice()
        assert np.allclose(td.mul(np.array([2, 3]), np.array([4, 5])), [8, 15])

    def test_div(self):
        td = TensorDevice()
        result = td.ioctl("DIV", np.array([10, 20]), np.array([2, 5]))
        assert np.allclose(result.value, [5, 4])

    def test_neg(self):
        td = TensorDevice()
        result = td.ioctl("NEG", np.array([1, -2]))
        assert np.allclose(result.value, [-1, 2])

    def test_abs(self):
        td = TensorDevice()
        result = td.ioctl("ABS", np.array([-1, 2]))
        assert np.allclose(result.value, [1, 2])

    def test_pow(self):
        td = TensorDevice()
        result = td.ioctl("POW", np.array([2, 3]), 2)
        assert np.allclose(result.value, [4, 9])

    def test_sqrt(self):
        td = TensorDevice()
        result = td.ioctl("SQRT", np.array([4, 9]))
        assert np.allclose(result.value, [2, 3])

    def test_exp(self):
        td = TensorDevice()
        result = td.ioctl("EXP", np.array([0, 1]))
        assert np.allclose(result.value, [1, np.e], atol=1e-6)

    def test_log(self):
        td = TensorDevice()
        result = td.ioctl("LOG", np.array([1, np.e]))
        assert np.allclose(result.value, [0, 1], atol=1e-4)


# ── Reduction ───────────────────────────────────────────────────────────────


class TestReduction:

    def test_sum(self):
        td = TensorDevice()
        result = td.ioctl("SUM", np.array([1, 2, 3]))
        assert result.value == 6

    def test_mean(self):
        td = TensorDevice()
        result = td.ioctl("MEAN", np.array([1, 2, 3]))
        assert result.value == 2.0

    def test_std(self):
        td = TensorDevice()
        result = td.ioctl("STD", np.array([1, 2, 3]))
        assert result.success is True

    def test_var(self):
        td = TensorDevice()
        result = td.ioctl("VAR", np.array([1, 2, 3]))
        assert result.success is True

    def test_max(self):
        td = TensorDevice()
        result = td.ioctl("MAX", np.array([1, 3, 2]))
        assert result.value == 3

    def test_min(self):
        td = TensorDevice()
        result = td.ioctl("MIN", np.array([1, 3, 2]))
        assert result.value == 1

    def test_argmax(self):
        td = TensorDevice()
        result = td.ioctl("ARGMAX", np.array([1, 3, 2]))
        assert result.value == 1

    def test_argmin(self):
        td = TensorDevice()
        result = td.ioctl("ARGMIN", np.array([1, 3, 2]))
        assert result.value == 0


# ── Shape Operations ───────────────────────────────────────────────────────


class TestShape:

    def test_reshape(self):
        td = TensorDevice()
        result = td.ioctl("RESHAPE", np.array([1, 2, 3, 4]), (2, 2))
        assert result.value.shape == (2, 2)

    def test_transpose(self):
        td = TensorDevice()
        result = td.ioctl("TRANSPOSE", np.array([[1, 2], [3, 4]]))
        assert result.value.shape == (2, 2)

    def test_flatten(self):
        td = TensorDevice()
        result = td.ioctl("FLATTEN", np.array([[1, 2], [3, 4]]))
        assert result.value.shape == (4,)

    def test_squeeze(self):
        td = TensorDevice()
        result = td.ioctl("SQUEEZE", np.array([[[1], [2]]]))
        assert result.value.shape == (2,)

    def test_unsqueeze(self):
        td = TensorDevice()
        result = td.ioctl("UNSQUEEZE", np.array([1, 2, 3]), 0)
        assert result.value.shape == (1, 3)

    def test_cat(self):
        td = TensorDevice()
        result = td.ioctl("CAT", [np.array([1, 2]), np.array([3, 4])])
        assert np.allclose(result.value, [1, 2, 3, 4])

    def test_stack(self):
        td = TensorDevice()
        result = td.ioctl("STACK", [np.array([1, 2]), np.array([3, 4])])
        assert result.value.shape == (2, 2)


# ── Loss Functions ─────────────────────────────────────────────────────────


class TestLoss:

    def test_cross_entropy(self):
        td = TensorDevice()
        logits = np.array([[1.0, 2.0, 3.0]])
        target = np.array([2])
        result = td.cross_entropy(logits, target)
        assert result > 0

    def test_mse(self):
        td = TensorDevice()
        result = td.ioctl("MSE", np.array([1, 2, 3]), np.array([1, 2, 4]))
        assert result.success is True
        assert np.isclose(result.value, 1 / 3)

    def test_mae(self):
        td = TensorDevice()
        result = td.ioctl("MAE", np.array([1, 2, 3]), np.array([1, 2, 4]))
        assert result.success is True
        assert np.isclose(result.value, 1 / 3)


# ── Optimizers ──────────────────────────────────────────────────────────────


class TestOptimizers:

    def test_sgd_step(self):
        td = TensorDevice()
        params = {"w": np.array([1.0, 2.0])}
        grads = {"w": np.array([0.1, 0.2])}
        result = td._sgd_step(params, grads, 0.1)
        assert "w" in result
        assert result["w"][0] < 1.0

    def test_sgd_step_with_momentum(self):
        td = TensorDevice()
        params = {"w": np.array([1.0])}
        grads = {"w": np.array([0.1])}
        state = {}
        result = td._sgd_step(params, grads, 0.1, 0.9, state)
        assert "w" in state

    def test_adam_step(self):
        td = TensorDevice()
        params = {"w": np.array([1.0, 2.0])}
        grads = {"w": np.array([0.1, 0.2])}
        state = {}
        result = td._adam_step(params, grads, 0.001, (0.9, 0.999), 1e-8, state)
        assert "w" in result
        assert "w_m" in state
        assert "w_v" in state


# ── Utility ─────────────────────────────────────────────────────────────────


class TestUtility:

    def test_clip_grad_norm(self):
        td = TensorDevice()
        grads = {"w": np.array([1.0, 2.0, 3.0])}
        result = td._clip_grad_norm(grads, 1.0)
        assert "w" in result
        total_norm = np.sqrt(np.sum(result["w"] ** 2))
        assert total_norm <= 1.0 + 1e-6

    def test_clip_grad_norm_no_clip(self):
        td = TensorDevice()
        grads = {"w": np.array([0.01, 0.02])}
        result = td._clip_grad_norm(grads, 100.0)
        assert np.allclose(result["w"], grads["w"])

    def test_dropout_training(self):
        td = TensorDevice()
        result = td.dropout(np.ones(1000), p=0.5, training=True)
        assert np.mean(result) > 0.3
        assert np.mean(result) < 2.0

    def test_dropout_eval(self):
        td = TensorDevice()
        x = np.ones(100)
        result = td.dropout(x, p=0.5, training=False)
        assert np.allclose(result, x)

    def test_embedding(self):
        td = TensorDevice()
        weight = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        indices = np.array([0, 2])
        result = td.embedding(indices, weight)
        assert result.shape == (2, 3)
        assert np.allclose(result[0], [0, 1, 2])

    def test_linear_no_bias(self):
        td = TensorDevice()
        input = np.array([[1, 2]])
        weight = np.array([[3, 4], [5, 6]])
        result = td.linear(input, weight)
        assert result.shape == (1, 2)

    def test_linear_with_bias(self):
        td = TensorDevice()
        input = np.array([[1, 2]])
        weight = np.array([[3, 4], [5, 6]])
        bias = np.array([10, 20])
        result = td.linear(input, weight, bias)
        assert np.allclose(result, [[23, 36]])

    def test_attention_2d(self):
        td = TensorDevice()
        q = np.random.randn(2, 4)
        k = np.random.randn(2, 4)
        v = np.random.randn(2, 4)
        result = td.attention(q, k, v)
        assert result.shape == (2, 4)

    def test_attention_3d(self):
        td = TensorDevice()
        q = np.random.randn(1, 3, 8)
        k = np.random.randn(1, 3, 8)
        v = np.random.randn(1, 3, 8)
        result = td.attention(q, k, v)
        assert result.shape == (1, 3, 8)


# ── Normalization ──────────────────────────────────────────────────────────


class TestNormalization:

    def test_batch_norm(self):
        td = TensorDevice()
        x = np.random.randn(2, 3, 4)
        w = np.ones(4)
        b = np.zeros(4)
        mean = np.zeros(4)
        var = np.ones(4)
        result = td.ioctl("BATCH_NORM", x, w, b, mean, var)
        assert result.success is True

    def test_layer_norm(self):
        td = TensorDevice()
        x = np.random.randn(2, 4)
        w = np.ones(4)
        b = np.zeros(4)
        result = td.ioctl("LAYER_NORM", x, w, b)
        assert result.success is True

    def test_rms_norm(self):
        td = TensorDevice()
        x = np.random.randn(2, 4)
        w = np.ones(4)
        result = td.ioctl("RMS_NORM", x, w)
        assert result.success is True


# ── Pooling ─────────────────────────────────────────────────────────────────


class TestPooling:

    def test_max_pool1d(self):
        td = TensorDevice()
        x = np.random.randn(1, 1, 8)
        result = td.ioctl("MAX_POOL1D", x, 2)
        assert result.success is True
        assert result.value.shape == (1, 1, 4)

    def test_max_pool2d(self):
        td = TensorDevice()
        x = np.random.randn(1, 1, 4, 4)
        result = td.ioctl("MAX_POOL2D", x, 2)
        assert result.success is True
        assert result.value.shape == (1, 1, 2, 2)

    def test_avg_pool1d(self):
        td = TensorDevice()
        x = np.random.randn(1, 1, 8)
        result = td.ioctl("AVG_POOL1D", x, 2)
        assert result.success is True
        assert result.value.shape == (1, 1, 4)

    def test_avg_pool2d(self):
        td = TensorDevice()
        x = np.random.randn(1, 1, 4, 4)
        result = td.ioctl("AVG_POOL2D", x, 2)
        assert result.success is True
        assert result.value.shape == (1, 1, 2, 2)


# ── Convolution ─────────────────────────────────────────────────────────────


class TestConvolution:

    def test_conv1d(self):
        td = TensorDevice()
        x = np.random.randn(1, 1, 8)
        w = np.random.randn(1, 1, 3)
        result = td.ioctl("CONV1D", x, w)
        assert result.success is True
        assert result.value.shape[0] == 1

    def test_conv2d(self):
        td = TensorDevice()
        x = np.random.randn(1, 1, 4, 4)
        w = np.random.randn(1, 1, 3, 3)
        result = td.ioctl("CONV2D", x, w)
        assert result.success is True
        assert result.value.shape == (1, 1, 2, 2)


# ── Type Conversion ────────────────────────────────────────────────────────


class TestTypeConversion:

    def test_to_arr_list(self):
        td = TensorDevice()
        result = td._to_arr([1, 2, 3])
        assert isinstance(result, np.ndarray)

    def test_to_arr_tuple(self):
        td = TensorDevice()
        result = td._to_arr((1, 2, 3))
        assert isinstance(result, np.ndarray)

    def test_to_arr_scalar(self):
        td = TensorDevice()
        result = td._to_arr(5)
        assert isinstance(result, np.floating)
