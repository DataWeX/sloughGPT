"""Tests for domains.shell.vm_devices — TensorDevice and PythonExecDevice."""

import numpy as np
import pytest
from domains.shell.vm_devices import TensorDevice, PythonExecDevice
from domains.shell.vm import DeviceFault


class TestTensorDevice:
    def setup_method(self):
        self.dev = TensorDevice()

    def test_info(self):
        info = self.dev.info()
        assert info["type"] == "tensor"
        assert "matmul" in info["ops"]
        assert "forward" in info["ops"]

    def test_call_unknown_op(self):
        with pytest.raises(DeviceFault, match="unknown op"):
            self.dev.call("nonexistent")

    def test_matmul(self):
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        result = self.dev.call("matmul", a, b)
        assert np.allclose(result, a @ b)

    def test_relu(self):
        result = self.dev.call("relu", np.array([-1, 0, 2]))
        assert np.allclose(result, [0, 0, 2])

    def test_softmax(self):
        result = self.dev.call("softmax", np.array([1.0, 2.0, 3.0]))
        assert abs(sum(result) - 1.0) < 1e-6

    def test_sigmoid(self):
        result = self.dev.call("sigmoid", np.array([0.0]))
        assert result[0] == pytest.approx(0.5)

    def test_tanh(self):
        result = self.dev.call("tanh", np.array([0.0]))
        assert result[0] == pytest.approx(0.0)

    def test_add(self):
        result = self.dev.call("add", np.array([1, 2]), np.array([3, 4]))
        assert np.allclose(result, [4, 6])

    def test_mul(self):
        result = self.dev.call("mul", np.array([2, 3]), np.array([4, 5]))
        assert np.allclose(result, [8, 15])

    def test_sub(self):
        result = self.dev.call("sub", np.array([5, 6]), np.array([1, 2]))
        assert np.allclose(result, [4, 4])

    def test_neg(self):
        result = self.dev.call("neg", np.array([1, -2]))
        assert np.allclose(result, [-1, 2])

    def test_abs(self):
        result = self.dev.call("abs", np.array([-3, 4]))
        assert np.allclose(result, [3, 4])

    def test_sum(self):
        result = self.dev.call("sum", np.array([1, 2, 3]))
        assert result == 6.0

    def test_mean(self):
        result = self.dev.call("mean", np.array([1, 2, 3]))
        assert result == 2.0

    def test_max(self):
        result = self.dev.call("max", np.array([1, 5, 3]))
        assert result == 5.0

    def test_argmax(self):
        result = self.dev.call("argmax", np.array([1, 5, 3]))
        assert result == 1

    def test_norm(self):
        result = self.dev.call("norm", np.array([3, 4]))
        assert result == 5.0

    def test_shape(self):
        result = self.dev.call("shape", np.zeros((3, 4)))
        assert result == [3, 4]

    def test_zeros(self):
        result = self.dev.call("zeros", 3, 4)
        assert result.shape == (3, 4)
        assert np.allclose(result, 0)

    def test_randn(self):
        result = self.dev.call("randn", 2, 3)
        assert result.shape == (2, 3)

    def test_load_weight(self):
        dev = TensorDevice({"w": np.array([1, 2])})
        result = dev.call("load", "w")
        assert np.allclose(result, [1, 2])

    def test_load_weight_missing(self):
        with pytest.raises(DeviceFault, match="no weight"):
            self.dev.call("load", "nonexistent")

    def test_store_weight(self):
        self.dev.call("store", "w", np.array([3, 4]))
        assert np.allclose(self.dev._weights["w"], [3, 4])

    def test_forward(self):
        dev = TensorDevice({
            "w1": np.random.randn(4, 3),
            "b1": np.zeros(4),
            "w2": np.random.randn(3, 4),
            "b2": np.zeros(3),
        })
        result = dev.call("forward", np.array([1.0, 2.0, 3.0]))
        assert abs(sum(result) - 1.0) < 1e-6

    def test_forward_missing_weights(self):
        with pytest.raises(DeviceFault, match="requires w1, w2"):
            self.dev.call("forward", np.array([1.0]))

    def test_to_arr_from_list(self):
        result = self.dev._to_arr([1, 2, 3])
        assert isinstance(result, np.ndarray)

    def test_to_arr_from_json_string(self):
        result = self.dev._to_arr("[1, 2, 3]")
        assert np.allclose(result, [1, 2, 3])

    def test_to_arr_from_int(self):
        result = self.dev._to_arr(42)
        assert result == 42.0


class TestPythonExecDevice:
    def setup_method(self):
        self.dev = PythonExecDevice()

    def test_info(self):
        info = self.dev.info()
        assert info["type"] == "python_exec"
        assert "eval" in info["ops"]

    def test_call_unknown_op(self):
        with pytest.raises(DeviceFault, match="unknown op"):
            self.dev.call("nonexistent")

    def test_eval_basic(self):
        result = self.dev.call("eval", "2 + 3")
        assert result == 5

    def test_eval_with_numpy(self):
        result = self.dev.call("eval", "np.sum([1,2,3])")
        assert result == 6

    def test_call_len(self):
        result = self.dev.call("call", "len", [1, 2, 3])
        assert result == 3

    def test_call_min(self):
        result = self.dev.call("call", "min", [3, 1, 2])
        assert result == 1

    def test_call_unknown_func(self):
        with pytest.raises(DeviceFault, match="unknown callable"):
            self.dev.call("call", "nonexistent", 1)

    def test_import(self):
        result = self.dev.call("import", "math")
        assert result is not None

    def test_set_and_get(self):
        self.dev.call("set", "x", 42)
        result = self.dev.call("get", "x")
        assert result == 42

    def test_scope(self):
        self.dev.call("set", "x", 10)
        scope = self.dev.call("scope")
        assert scope["x"] == 10
        assert "np" in scope

    def test_exec_side_effect(self):
        self.dev.call("exec", "x = 100")
        result = self.dev.call("get", "x")
        assert result == 100

    def test_eval_builtins_restricted(self):
        # open is not in _SAFE_BUILTINS
        with pytest.raises((DeviceFault, NameError)):
            self.dev.call("eval", "__import__('os')")
