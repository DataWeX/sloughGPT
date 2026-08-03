"""
Edge-case tests for RenderNeuralDevice channel stacking and raw forward pass.

Run: PYTHONPATH=packages/core-py python -m pytest tests/test_render_neural_edges.py -x -q
"""

import numpy as np
import pytest
from domains.shell.render_neural import RenderNeuralDevice


def _tensors(keys=("image", "depth", "normal", "albedo")):
    t = {}
    if "image" in keys:
        t["image"] = np.random.rand(8, 6, 3).astype(np.float32)
    if "depth" in keys:
        t["depth"] = np.random.rand(8, 6).astype(np.float32)
    if "normal" in keys:
        t["normal"] = np.random.rand(8, 6, 3).astype(np.float32)
    if "albedo" in keys:
        t["albedo"] = np.random.rand(8, 6, 3).astype(np.float32)
    return t


class _FakeCycles:
    def __init__(self, tensors):
        self._tensors = tensors

    def call(self, method):
        assert method == "state_tensors"
        return self._tensors


class TestStackChannelsEdgeCases:
    def test_missing_tensor_key_is_skipped(self):
        dev = RenderNeuralDevice()
        tensors = _tensors(keys=("image",))
        stacked = dev._stack_channels(tensors)
        assert stacked.shape == (1, 6, 8, 6)
        assert np.allclose(stacked[0, 3], 0)  # depth zero-filled
        assert np.allclose(stacked[0, 4], 0)  # normal zero-filled
        assert np.allclose(stacked[0, 5], 0)  # albedo zero-filled

    def test_no_valid_tensors_raises(self):
        dev = RenderNeuralDevice()
        with pytest.raises(Exception, match="no valid state tensors"):
            dev._stack_channels({})

    def test_2d_depth_used_directly(self):
        dev = RenderNeuralDevice()
        tensors = _tensors(keys=("image", "depth"))
        stacked = dev._stack_channels(tensors)
        assert np.allclose(stacked[0, 3], tensors["depth"])

    def test_3d_depth_uses_first_channel(self):
        dev = RenderNeuralDevice()
        tensors = _tensors(keys=("image", "depth"))
        tensors["depth"] = np.random.rand(8, 6, 3).astype(np.float32)
        stacked = dev._stack_channels(tensors)
        assert np.allclose(stacked[0, 3], tensors["depth"][..., 0])

    def test_missing_normal_uses_zeros(self):
        dev = RenderNeuralDevice()
        tensors = _tensors(keys=("image", "depth", "albedo"))
        stacked = dev._stack_channels(tensors)
        assert np.allclose(stacked[0, 4], 0)

    def test_2d_normal_uses_zeros(self):
        dev = RenderNeuralDevice()
        tensors = _tensors(keys=("image", "depth"))
        tensors["normal"] = np.random.rand(8, 6).astype(np.float32)
        stacked = dev._stack_channels(tensors)
        assert np.allclose(stacked[0, 4], 0)

    def test_3d_normal_uses_y_component(self):
        dev = RenderNeuralDevice()
        tensors = _tensors(keys=("image", "depth"))
        tensors["normal"] = np.random.rand(8, 6, 3).astype(np.float32)
        stacked = dev._stack_channels(tensors)
        assert np.allclose(stacked[0, 4], tensors["normal"][..., 1])

    def test_missing_albedo_uses_zeros(self):
        dev = RenderNeuralDevice()
        tensors = _tensors(keys=("image", "depth", "normal"))
        stacked = dev._stack_channels(tensors)
        assert np.allclose(stacked[0, 5], 0)

    def test_2d_albedo_uses_zeros(self):
        dev = RenderNeuralDevice()
        tensors = _tensors(keys=("image", "depth", "normal"))
        tensors["albedo"] = np.random.rand(8, 6).astype(np.float32)
        stacked = dev._stack_channels(tensors)
        assert np.allclose(stacked[0, 5], 0)

    def test_3d_albedo_uses_r_component(self):
        dev = RenderNeuralDevice()
        tensors = _tensors(keys=("image", "depth", "normal"))
        tensors["albedo"] = np.random.rand(8, 6, 3).astype(np.float32)
        stacked = dev._stack_channels(tensors)
        assert np.allclose(stacked[0, 5], tensors["albedo"][..., 0])

    def test_missing_image_uses_zero_rgb(self):
        dev = RenderNeuralDevice()
        tensors = _tensors(keys=("depth",))
        tensors["depth"] = np.random.rand(8, 6, 3).astype(np.float32)
        stacked = dev._stack_channels(tensors)
        assert np.allclose(stacked[0, 0], 0)
        assert np.allclose(stacked[0, 1], 0)
        assert np.allclose(stacked[0, 2], 0)

    def test_rgb_image_uses_all_channels(self):
        dev = RenderNeuralDevice()
        tensors = _tensors(keys=("image",))
        stacked = dev._stack_channels(tensors)
        assert np.allclose(stacked[0, 0], tensors["image"][..., 0])
        assert np.allclose(stacked[0, 1], tensors["image"][..., 1])
        assert np.allclose(stacked[0, 2], tensors["image"][..., 2])


class TestForwardRaw:
    def test_forward_with_state_tensors_dict(self):
        dev = RenderNeuralDevice()
        tensors = _tensors()
        out = dev.call("forward", {"state_tensors": tensors})
        assert out["embedding"].shape == (64,)

    def test_forward_with_extra_dict_uses_state_tensors(self):
        dev = RenderNeuralDevice()
        tensors = _tensors()
        out = dev.call("forward", {"state_tensors": tensors, "extra": 1})
        assert out["embedding"].shape == (64,)

    def test_forward_uses_last_inputs(self):
        dev = RenderNeuralDevice(cycles_device=_FakeCycles(_tensors()))
        dev.call("process")
        out = dev.call("forward", {"other": None})
        assert out["embedding"].shape == (64,)

    def test_forward_uses_last_inputs_with_no_args(self):
        dev = RenderNeuralDevice(cycles_device=_FakeCycles(_tensors()))
        dev.call("process")
        out = dev.call("forward")
        assert out["embedding"].shape == (64,)

    def test_forward_no_inputs_raises(self):
        dev = RenderNeuralDevice()
        with pytest.raises(Exception, match="no inputs available"):
            dev.call("forward")
