"""Coverage tests for RenderNeuralDevice (domains.shell.render_neural)."""

import numpy as np
import pytest

from domains.shell.render_neural import RenderNeuralDevice
from domains.shell.cycles_device import CyclesDevice
from domains.shell.vm import DeviceFault


def _tensor_dict(H=6, W=8):
    return {
        "image": np.random.rand(H, W, 3).astype(np.float32),
        "depth": np.random.rand(H, W).astype(np.float32),
        "normal": np.random.rand(H, W, 3).astype(np.float32),
        "albedo": np.random.rand(H, W, 3).astype(np.float32),
    }


def _render_source():
    cycles = CyclesDevice(width=8, height=6, samples=1)
    cycles.call("add_light")
    cycles.call("set_camera")
    return cycles


class TestRenderNeuralBasics:
    def test_init_and_info(self):
        d = RenderNeuralDevice()
        info = d.call("info")
        assert info["type"] == "render_neural"
        assert info["embed_dim"] == 64
        assert info["num_classes"] == 8
        assert info["has_source"] is False
        assert info["input_channels"] == 6
        for op in ("process", "embed", "classify", "descriptor", "set_source", "forward"):
            assert op in info["ops"]

    def test_init_custom_dims(self):
        d = RenderNeuralDevice(embed_dim=32, num_classes=5)
        assert d._embed_dim == 32
        assert d._num_classes == 5
        assert d._proj_w.shape == (32, 32)
        assert d._classify_w.shape == (32, 5)

    def test_call_unknown_raises(self):
        d = RenderNeuralDevice()
        with pytest.raises(DeviceFault):
            d.call("bogus")

    def test_info_has_source(self):
        d = RenderNeuralDevice(_render_source())
        assert d.call("info")["has_source"] is True


class TestNoSource:
    def test_process_no_source_raises(self):
        with pytest.raises(DeviceFault):
            RenderNeuralDevice().call("process")

    def test_embed_no_source_raises(self):
        with pytest.raises(DeviceFault):
            RenderNeuralDevice().call("embed")

    def test_classify_no_source_raises(self):
        with pytest.raises(DeviceFault):
            RenderNeuralDevice().call("classify")

    def test_descriptor_no_source_raises(self):
        with pytest.raises(DeviceFault):
            RenderNeuralDevice().call("descriptor")

    def test_forward_raw_no_inputs_raises(self):
        with pytest.raises(DeviceFault):
            RenderNeuralDevice().call("forward")


class TestPipeline:
    def test_process_end_to_end(self):
        d = RenderNeuralDevice(_render_source())
        out = d.call("process")
        assert set(out.keys()) == {"embedding", "logits", "probabilities", "features"}
        assert out["embedding"].shape == (64,)
        assert out["logits"].shape == (8,)
        assert out["probabilities"].shape == (8,)
        assert out["features"].shape == (32,)
        assert np.isclose(out["probabilities"].sum(), 1.0, atol=1e-4)
        assert d._last_inputs is not None
        assert d._last_outputs is not None

    def test_embed(self):
        d = RenderNeuralDevice(_render_source())
        emb = d.call("embed")
        assert emb.shape == (64,)
        assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-4)

    def test_classify(self):
        d = RenderNeuralDevice(_render_source())
        out = d.call("classify")
        assert out["labels"].shape == (6, 8)
        assert out["probabilities"].shape == (6, 8, 8)

    def test_descriptor(self):
        d = RenderNeuralDevice(_render_source())
        desc = d.call("descriptor")
        for key in ("image", "depth", "normal", "albedo"):
            stat = desc[key]
            assert set(stat.keys()) == {"mean", "std", "min", "max", "shape"}
            assert stat["shape"] == ([6, 8, 3] if key != "depth" else [6, 8])
        assert "neural_embedding_norm" in desc
        assert desc["neural_entropy"] >= 0
        assert 0 <= desc["dominant_class"] < 8


class TestStackChannels:
    def test_stack_full(self):
        d = RenderNeuralDevice()
        t = _tensor_dict()
        stacked = d._stack_channels(t)
        assert stacked.shape == (1, 6, 6, 8)
        assert stacked.dtype == np.float32

    def test_stack_depth_3d(self):
        d = RenderNeuralDevice()
        t = _tensor_dict()
        t["depth"] = np.random.rand(6, 8, 1).astype(np.float32)
        stacked = d._stack_channels(t)
        assert stacked.shape == (1, 6, 6, 8)

    def test_stack_missing_image_uses_zeros(self):
        d = RenderNeuralDevice()
        t = _tensor_dict()
        del t["image"]
        stacked = d._stack_channels(t)
        assert stacked.shape == (1, 6, 6, 8)
        assert np.all(stacked[0, 0] == 0)

    def test_stack_2d_normal_falls_back(self):
        d = RenderNeuralDevice()
        t = _tensor_dict()
        t["normal"] = np.random.rand(6, 8).astype(np.float32)
        stacked = d._stack_channels(t)
        assert stacked.shape == (1, 6, 6, 8)
        assert np.all(stacked[0, 4] == 0)

    def test_stack_2d_albedo_falls_back(self):
        d = RenderNeuralDevice()
        t = _tensor_dict()
        t["albedo"] = np.random.rand(6, 8).astype(np.float32)
        stacked = d._stack_channels(t)
        assert stacked.shape == (1, 6, 6, 8)
        assert np.all(stacked[0, 5] == 0)

    def test_stack_missing_depth_falls_back(self):
        d = RenderNeuralDevice()
        t = _tensor_dict()
        del t["depth"]
        stacked = d._stack_channels(t)
        assert np.all(stacked[0, 3] == 0)

    def test_stack_missing_albedo_falls_back(self):
        d = RenderNeuralDevice()
        t = _tensor_dict()
        del t["albedo"]
        stacked = d._stack_channels(t)
        assert np.all(stacked[0, 5] == 0)

    def test_stack_no_valid_tensor_raises(self):
        d = RenderNeuralDevice()
        with pytest.raises(DeviceFault):
            d._stack_channels({"image": np.zeros((0, 0, 3))})


class TestConv:
    def test_conv2d_relu_shape_and_cache(self):
        d = RenderNeuralDevice()
        x = np.random.rand(1, 6, 8, 8).astype(np.float32)
        out = d._conv2d_relu(x, d._conv1_w, d._conv1_b)
        assert out.shape == (1, 16, 8, 8)
        assert out.min() >= 0
        assert hasattr(d, "_col_indices")
        out2 = d._conv2d_relu(x, d._conv1_w, d._conv1_b)
        np.testing.assert_array_equal(out, out2)

    def test_conv2d_relu_rebuilds_cache_for_new_shape(self):
        d = RenderNeuralDevice()
        x = np.random.rand(1, 6, 8, 8).astype(np.float32)
        d._conv2d_relu(x, d._conv1_w, d._conv1_b)
        x2 = np.random.rand(1, 6, 4, 4).astype(np.float32)
        out = d._conv2d_relu(x2, d._conv1_w, d._conv1_b)
        assert out.shape == (1, 16, 4, 4)
        assert d._col_shape == (6, 3, 4, 4)

    def test_adaptive_avg_pool(self):
        d = RenderNeuralDevice()
        x = np.random.rand(2, 8, 4, 4).astype(np.float32)
        pooled = d._adaptive_avg_pool(x)
        assert pooled.shape == (2, 8)


class TestForwardRaw:
    def test_forward_raw_with_dict(self):
        d = RenderNeuralDevice()
        out = d.call("forward", {"state_tensors": _tensor_dict()})
        assert set(out.keys()) == {"embedding", "logits", "probabilities", "features"}
        assert out["embedding"].shape == (64,)

    def test_forward_raw_dict_without_tensors_uses_last_inputs(self):
        d = RenderNeuralDevice(_render_source())
        d.call("process")
        out = d.call("forward", {"other": 1})
        assert out["embedding"].shape == (64,)

    def test_forward_raw_uses_last_inputs(self):
        d = RenderNeuralDevice(_render_source())
        d.call("process")
        out = d.call("forward")
        assert out["embedding"].shape == (64,)


class TestSetSource:
    def test_set_source_enables_processing(self):
        cycles = _render_source()
        d = RenderNeuralDevice()
        assert d.call("info")["has_source"] is False
        d.call("set_source", cycles)
        assert d.call("info")["has_source"] is True
        out = d.call("process")
        assert out["embedding"].shape == (64,)

    def test_set_source_resets_caches(self):
        d = RenderNeuralDevice(_render_source())
        d.call("process")
        assert d._last_inputs is not None
        d.call("set_source", None)
        assert d._last_inputs is None
        assert d._last_outputs is None
        with pytest.raises(DeviceFault):
            d.call("process")
