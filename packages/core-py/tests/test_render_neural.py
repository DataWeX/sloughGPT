"""Tests for domains.shell.render_neural — RenderNeuralDevice neural processor."""

import numpy as np
import pytest
from domains.shell.render_neural import RenderNeuralDevice, _softmax
from domains.shell.cycles_device import CyclesDevice
from domains.shell.vm import DeviceFault


class TestSoftmax:
    def test_sums_to_one(self):
        x = np.array([1.0, 2.0, 3.0])
        result = _softmax(x)
        assert abs(sum(result) - 1.0) < 1e-6

    def test_2d(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _softmax(x, axis=-1)
        for row in result:
            assert abs(sum(row) - 1.0) < 1e-6

    def test_large_values_stable(self):
        x = np.array([1000.0, 1001.0, 1002.0])
        result = _softmax(x)
        assert abs(sum(result) - 1.0) < 1e-6


class TestRenderNeuralDevice:
    def setup_method(self):
        self.dev = RenderNeuralDevice(embed_dim=32, num_classes=5)

    def test_info(self):
        info = self.dev.info()
        assert info["type"] == "render_neural"
        assert info["embed_dim"] == 32
        assert info["num_classes"] == 5
        assert info["has_source"] is False
        assert info["input_channels"] == 6

    def test_call_unknown_op(self):
        with pytest.raises(DeviceFault, match="unknown op"):
            self.dev.call("nonexistent")

    def test_no_source_raises(self):
        with pytest.raises(DeviceFault, match="no CyclesDevice"):
            self.dev.call("process")

    def test_with_source(self):
        cycles = CyclesDevice(width=16, height=16, samples=1)
        cycles._add_sphere(radius=0.3)
        cycles._add_light(y=2.0)
        dev = RenderNeuralDevice(cycles_device=cycles, embed_dim=32, num_classes=5)
        assert dev.info()["has_source"] is True

    def test_set_source(self):
        dev = RenderNeuralDevice()
        cycles = CyclesDevice(width=16, height=16, samples=1)
        dev.call("set_source", cycles)
        assert dev._cycles is cycles

    def test_stack_channels(self):
        tensors = {
            "image": np.random.rand(8, 8, 3).astype(np.float32),
            "depth": np.random.rand(8, 8).astype(np.float32),
            "normal": np.random.rand(8, 8, 3).astype(np.float32),
            "albedo": np.random.rand(8, 8, 3).astype(np.float32),
        }
        result = self.dev._stack_channels(tensors)
        assert result.shape == (1, 6, 8, 8)

    def test_stack_channels_missing(self):
        tensors = {"image": np.random.rand(8, 8, 3).astype(np.float32)}
        result = self.dev._stack_channels(tensors)
        assert result.shape == (1, 6, 8, 8)

    def test_stack_channels_empty_raises(self):
        with pytest.raises(DeviceFault, match="no valid"):
            self.dev._stack_channels({})

    def test_conv2d_relu(self):
        x = np.random.rand(1, 6, 8, 8).astype(np.float32)
        w = np.random.rand(16, 6, 3, 3).astype(np.float32)
        b = np.zeros(16, dtype=np.float32)
        result = self.dev._conv2d_relu(x, w, b)
        assert result.shape == (1, 16, 8, 8)
        assert np.all(result >= 0)  # ReLU

    def test_adaptive_avg_pool(self):
        x = np.random.rand(1, 32, 4, 4).astype(np.float32)
        result = self.dev._adaptive_avg_pool(x)
        assert result.shape == (1, 32)

    def test_forward(self):
        x = np.random.rand(1, 6, 8, 8).astype(np.float32)
        out = self.dev._forward(x)
        assert "embedding" in out
        assert "logits" in out
        assert "probabilities" in out
        assert "features" in out
        assert out["embedding"].shape == (32,)
        assert out["logits"].shape == (5,)
        assert out["probabilities"].shape == (5,)
        assert abs(sum(out["probabilities"]) - 1.0) < 1e-5

    def test_process(self):
        cycles = CyclesDevice(width=16, height=16, samples=1)
        cycles._add_sphere(radius=0.3)
        cycles._add_light(y=2.0)
        dev = RenderNeuralDevice(cycles_device=cycles, embed_dim=32, num_classes=5)
        result = dev.call("process")
        assert "embedding" in result

    def test_embed(self):
        cycles = CyclesDevice(width=16, height=16, samples=1)
        cycles._add_sphere(radius=0.3)
        cycles._add_light(y=2.0)
        dev = RenderNeuralDevice(cycles_device=cycles, embed_dim=32, num_classes=5)
        result = dev.call("embed")
        assert result.shape == (32,)

    def test_classify(self):
        cycles = CyclesDevice(width=16, height=16, samples=1)
        cycles._add_sphere(radius=0.3)
        cycles._add_light(y=2.0)
        dev = RenderNeuralDevice(cycles_device=cycles, embed_dim=32, num_classes=5)
        result = dev.call("classify")
        assert "labels" in result
        assert "probabilities" in result

    def test_descriptor(self):
        cycles = CyclesDevice(width=16, height=16, samples=1)
        cycles._add_sphere(radius=0.3)
        cycles._add_light(y=2.0)
        dev = RenderNeuralDevice(cycles_device=cycles, embed_dim=32, num_classes=5)
        result = dev.call("descriptor")
        assert "neural_embedding_norm" in result
        assert "neural_entropy" in result
        assert "dominant_class" in result

    def test_forward_raw_with_dict(self):
        cycles = CyclesDevice(width=16, height=16, samples=1)
        cycles._add_sphere(radius=0.3)
        cycles._add_light(y=2.0)
        dev = RenderNeuralDevice(cycles_device=cycles, embed_dim=32, num_classes=5)
        state = cycles.call("state_tensors")
        result = dev.call("forward", {"state_tensors": state})
        assert "embedding" in result

    def test_forward_raw_no_inputs_raises(self):
        dev = RenderNeuralDevice()
        with pytest.raises(DeviceFault, match="no inputs"):
            dev.call("forward")
