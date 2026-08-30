"""
Edge-case tests for RenderNeuralDevice channel stacking and raw forward pass.

Run: PYTHONPATH=packages/core-py python -m pytest tests/test_render_neural_edges.py -x -q
"""

import numpy as np
import pytest
from domains.shell.render_neural import RenderNeuralDevice, _softmax
from domains.shell.vm import DeviceFault


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


# =============================================================================
# StackChannels
# =============================================================================

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

    def test_all_tensors_produce_6_channels(self):
        dev = RenderNeuralDevice()
        tensors = _tensors()
        stacked = dev._stack_channels(tensors)
        assert stacked.shape == (1, 6, 8, 6)

    def test_batch_dim_is_one(self):
        dev = RenderNeuralDevice()
        tensors = _tensors()
        stacked = dev._stack_channels(tensors)
        assert stacked.shape[0] == 1

    def test_depth_4d_uses_first_channel(self):
        """4D depth tensor: reshape to (H, W) via first channel."""
        dev = RenderNeuralDevice()
        tensors = _tensors(keys=("image",))
        tensors["depth"] = np.random.rand(8, 6, 3, 1).astype(np.float32)
        stacked = dev._stack_channels(tensors)
        # 4D depth is reshaped to (H, W) for the channel
        assert stacked[0, 3].shape == (8, 6)

    def test_normal_only_2_channels_uses_first(self):
        """Normal with only 2 channels: shape[-1] >= 2 is True, uses Y component."""
        dev = RenderNeuralDevice()
        tensors = _tensors(keys=("image",))
        tensors["normal"] = np.random.rand(8, 6, 2).astype(np.float32)
        stacked = dev._stack_channels(tensors)
        assert np.allclose(stacked[0, 4], tensors["normal"][..., 1])

    def test_single_channel_image(self):
        """Single-channel image adds 1 channel, not 3. Total channels = 4."""
        dev = RenderNeuralDevice()
        t = np.random.rand(8, 6, 1).astype(np.float32)
        stacked = dev._stack_channels({"image": t})
        # 1 (image) + 1 (depth zero) + 1 (normal zero) + 1 (albedo zero) = 4
        assert stacked.shape == (1, 4, 8, 6)
        assert np.allclose(stacked[0, 0], t[..., 0])

    def test_depth_only_no_image(self):
        """Depth without image but with 3D depth to determine H,W."""
        dev = RenderNeuralDevice()
        d = np.random.rand(8, 6, 1).astype(np.float32)
        stacked = dev._stack_channels({"depth": d})
        assert np.allclose(stacked[0, 3], d[..., 0])
        assert np.allclose(stacked[0, 0], 0)
        assert np.allclose(stacked[0, 1], 0)
        assert np.allclose(stacked[0, 2], 0)

    def test_albedo_only_no_image(self):
        """Albedo without image: image channels are zero-filled."""
        dev = RenderNeuralDevice()
        a = np.random.rand(8, 6, 3).astype(np.float32)
        stacked = dev._stack_channels({"albedo": a})
        assert np.allclose(stacked[0, 5], a[..., 0])
        assert stacked.shape == (1, 6, 8, 6)

    def test_normal_only_no_image(self):
        """Normal without image: image channels are zero-filled."""
        dev = RenderNeuralDevice()
        n = np.random.rand(8, 6, 3).astype(np.float32)
        stacked = dev._stack_channels({"normal": n})
        assert np.allclose(stacked[0, 4], n[..., 1])

    def test_output_is_float32(self):
        dev = RenderNeuralDevice()
        tensors = _tensors()
        stacked = dev._stack_channels(tensors)
        assert stacked.dtype == np.float32

    def test_large_spatial_dims(self):
        dev = RenderNeuralDevice()
        t = {"image": np.random.rand(64, 64, 3).astype(np.float32)}
        stacked = dev._stack_channels(t)
        assert stacked.shape == (1, 6, 64, 64)

    def test_small_spatial_dims(self):
        dev = RenderNeuralDevice()
        t = {"image": np.random.rand(1, 1, 3).astype(np.float32)}
        stacked = dev._stack_channels(t)
        assert stacked.shape == (1, 6, 1, 1)

    def test_depth_5d_uses_first_channel(self):
        """5D depth with image for H,W: depth[..., 0, 0] gives 2D."""
        dev = RenderNeuralDevice()
        tensors = _tensors(keys=("image",))
        tensors["depth"] = np.random.rand(8, 6, 2, 2, 1).astype(np.float32)
        stacked = dev._stack_channels(tensors)
        assert stacked.shape == (1, 6, 8, 6)

    def test_only_unknown_keys_ignored(self):
        """Extra keys not in (image, depth, normal, albedo) are ignored."""
        dev = RenderNeuralDevice()
        t = {
            "image": np.random.rand(8, 6, 3).astype(np.float32),
            "emission": np.random.rand(8, 6, 3).astype(np.float32),
            "mask": np.random.rand(8, 6).astype(np.float32),
        }
        stacked = dev._stack_channels(t)
        assert stacked.shape == (1, 6, 8, 6)


# =============================================================================
# TestForwardRaw
# =============================================================================

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

    def test_forward_raw_with_empty_dict(self):
        """Empty dict without state_tensors key and no last_inputs raises."""
        dev = RenderNeuralDevice()
        with pytest.raises(DeviceFault, match="no inputs available"):
            dev.call("forward", {})

    def test_forward_raw_dict_without_state_tensors_key(self):
        """Dict without state_tensors key and no last_inputs raises."""
        dev = RenderNeuralDevice()
        with pytest.raises(DeviceFault, match="no inputs available"):
            dev.call("forward", {"not_state_tensors": None})


# =============================================================================
# TestDeviceInfo
# =============================================================================

class TestDeviceInfo:
    def test_info_returns_dict(self):
        dev = RenderNeuralDevice()
        info = dev.info()
        assert info["type"] == "render_neural"
        assert info["embed_dim"] == 64
        assert info["num_classes"] == 8
        assert info["input_channels"] == 6
        assert info["has_source"] is False

    def test_info_with_source(self):
        dev = RenderNeuralDevice(cycles_device=_FakeCycles(_tensors()))
        info = dev.info()
        assert info["has_source"] is True

    def test_info_ops_list(self):
        dev = RenderNeuralDevice()
        info = dev.info()
        assert "process" in info["ops"]
        assert "embed" in info["ops"]
        assert "classify" in info["ops"]
        assert "descriptor" in info["ops"]
        assert "forward" in info["ops"]
        assert "set_source" in info["ops"]

    def test_custom_embed_dim(self):
        dev = RenderNeuralDevice(embed_dim=128)
        assert dev._embed_dim == 128
        assert dev.info()["embed_dim"] == 128

    def test_custom_num_classes(self):
        dev = RenderNeuralDevice(num_classes=16)
        assert dev._num_classes == 16
        assert dev.info()["num_classes"] == 16

    def test_info_type_always_render_neural(self):
        dev = RenderNeuralDevice()
        assert dev.info()["type"] == "render_neural"

    def test_info_input_channels_always_6(self):
        dev = RenderNeuralDevice()
        assert dev.info()["input_channels"] == 6


# =============================================================================
# TestCallUnknownOp
# =============================================================================

class TestCallUnknownOp:
    def test_unknown_op_raises(self):
        dev = RenderNeuralDevice()
        with pytest.raises(DeviceFault, match="unknown op"):
            dev.call("nonexistent_op")

    def test_empty_string_op_raises(self):
        dev = RenderNeuralDevice()
        with pytest.raises(DeviceFault, match="unknown op"):
            dev.call("")

    def test_none_op_raises(self):
        dev = RenderNeuralDevice()
        with pytest.raises(DeviceFault, match="unknown op"):
            dev.call(None)

    def test_numeric_op_raises(self):
        dev = RenderNeuralDevice()
        with pytest.raises(DeviceFault, match="unknown op"):
            dev.call(123)


# =============================================================================
# TestSetSource
# =============================================================================

class TestSetSource:
    def test_set_source_clears_cache(self):
        dev = RenderNeuralDevice(cycles_device=_FakeCycles(_tensors()))
        dev.call("process")
        assert dev._last_inputs is not None
        assert dev._last_outputs is not None
        dev.call("set_source", _FakeCycles(_tensors()))
        assert dev._last_inputs is None
        assert dev._last_outputs is None

    def test_set_source_changes_cycles(self):
        dev = RenderNeuralDevice()
        fake = _FakeCycles(_tensors())
        dev.call("set_source", fake)
        assert dev._cycles is fake

    def test_set_source_none(self):
        dev = RenderNeuralDevice(cycles_device=_FakeCycles(_tensors()))
        dev.call("set_source", None)
        assert dev._cycles is None

    def test_set_source_then_process_works(self):
        dev = RenderNeuralDevice()
        dev.call("set_source", _FakeCycles(_tensors()))
        out = dev.call("process")
        assert "embedding" in out

    def test_set_source_then_process_forward_works(self):
        dev = RenderNeuralDevice()
        dev.call("set_source", _FakeCycles(_tensors()))
        dev.call("process")  # populates _last_inputs/_last_outputs
        out = dev.call("forward")
        assert "embedding" in out

    def test_set_source_then_forward_with_tensors(self):
        dev = RenderNeuralDevice()
        dev.call("set_source", _FakeCycles(_tensors()))
        tensors = _tensors()
        out = dev.call("forward", {"state_tensors": tensors})
        assert "embedding" in out


# =============================================================================
# TestEnsureSource
# =============================================================================

class TestEnsureSource:
    def test_no_source_raises(self):
        dev = RenderNeuralDevice()
        with pytest.raises(DeviceFault, match="no CyclesDevice source"):
            dev._ensure_source()

    def test_with_source_returns_tensors(self):
        t = _tensors()
        dev = RenderNeuralDevice(cycles_device=_FakeCycles(t))
        result = dev._ensure_source()
        assert "image" in result


# =============================================================================
# TestConv2dRelu
# =============================================================================

class TestConv2dRelu:
    def test_output_shape(self):
        dev = RenderNeuralDevice()
        x = np.random.rand(1, 6, 8, 6).astype(np.float32)
        out = dev._conv2d_relu(x, dev._conv1_w, dev._conv1_b)
        assert out.shape[0] == 1
        assert out.shape[1] == 16  # C_out of conv1

    def test_relu_applied(self):
        dev = RenderNeuralDevice()
        x = np.random.rand(1, 6, 8, 6).astype(np.float32)
        out = dev._conv2d_relu(x, dev._conv1_w, dev._conv1_b)
        assert np.all(out >= 0)

    def test_conv2_second_layer(self):
        dev = RenderNeuralDevice()
        x = np.random.rand(1, 6, 8, 6).astype(np.float32)
        h1 = dev._conv2d_relu(x, dev._conv1_w, dev._conv1_b)
        h2 = dev._conv2d_relu(h1, dev._conv2_w, dev._conv2_b)
        assert h2.shape[1] == 32

    def test_col_indices_cached(self):
        dev = RenderNeuralDevice()
        x = np.random.rand(1, 6, 8, 6).astype(np.float32)
        dev._conv2d_relu(x, dev._conv1_w, dev._conv1_b)
        assert hasattr(dev, '_col_indices')
        first = dev._col_indices
        dev._conv2d_relu(x, dev._conv1_w, dev._conv1_b)
        assert dev._col_indices is first

    def test_col_indices_recomputed_on_shape_change(self):
        dev = RenderNeuralDevice()
        x1 = np.random.rand(1, 6, 8, 6).astype(np.float32)
        dev._conv2d_relu(x1, dev._conv1_w, dev._conv1_b)
        shape1 = dev._col_shape
        x2 = np.random.rand(1, 6, 10, 8).astype(np.float32)
        dev._conv2d_relu(x2, dev._conv1_w, dev._conv1_b)
        assert dev._col_shape != shape1

    def test_preserves_spatial_dims(self):
        dev = RenderNeuralDevice()
        x = np.random.rand(1, 6, 8, 6).astype(np.float32)
        out = dev._conv2d_relu(x, dev._conv1_w, dev._conv1_b)
        assert out.shape[2] == 8
        assert out.shape[3] == 6

    def test_negative_input_relu_zeroed(self):
        dev = RenderNeuralDevice()
        x = np.full((1, 6, 4, 4), -10.0, dtype=np.float32)
        out = dev._conv2d_relu(x, dev._conv1_w, dev._conv1_b)
        assert np.all(out >= 0)

    def test_batch_size_preserved(self):
        dev = RenderNeuralDevice()
        x = np.random.rand(2, 6, 8, 6).astype(np.float32)
        out = dev._conv2d_relu(x, dev._conv1_w, dev._conv1_b)
        assert out.shape[0] == 2

    def test_conv2_output_channels(self):
        dev = RenderNeuralDevice()
        x = np.random.rand(1, 6, 8, 6).astype(np.float32)
        h1 = dev._conv2d_relu(x, dev._conv1_w, dev._conv1_b)
        h2 = dev._conv2d_relu(h1, dev._conv2_w, dev._conv2_b)
        assert h2.shape[1] == 32


# =============================================================================
# TestAdaptiveAvgPool
# =============================================================================

class TestAdaptiveAvgPool:
    def test_reduces_to_1d(self):
        dev = RenderNeuralDevice()
        x = np.random.rand(1, 32, 8, 6).astype(np.float32)
        out = dev._adaptive_avg_pool(x)
        assert out.shape == (1, 32)

    def test_preserves_batch(self):
        dev = RenderNeuralDevice()
        x = np.random.rand(2, 32, 4, 4).astype(np.float32)
        out = dev._adaptive_avg_pool(x)
        assert out.shape == (2, 32)

    def test_values_are_means(self):
        dev = RenderNeuralDevice()
        x = np.ones((1, 4, 3, 3), dtype=np.float32)
        out = dev._adaptive_avg_pool(x)
        assert np.allclose(out, 1.0)

    def test_pool_reduces_spatial_dims(self):
        dev = RenderNeuralDevice()
        x = np.random.rand(1, 16, 10, 10).astype(np.float32)
        out = dev._adaptive_avg_pool(x)
        assert out.shape == (1, 16)

    def test_pool_with_zeros(self):
        dev = RenderNeuralDevice()
        x = np.zeros((1, 8, 4, 4), dtype=np.float32)
        out = dev._adaptive_avg_pool(x)
        assert np.allclose(out, 0.0)

    def test_pool_with_random_values(self):
        dev = RenderNeuralDevice()
        x = np.random.rand(1, 8, 5, 5).astype(np.float32)
        out = dev._adaptive_avg_pool(x)
        assert out.shape == (1, 8)
        # Mean should be between min and max
        assert out[0, 0] >= x.min()
        assert out[0, 0] <= x.max()


# =============================================================================
# TestForward
# =============================================================================

class TestForward:
    def test_embedding_shape(self):
        dev = RenderNeuralDevice(embed_dim=64)
        x = np.random.rand(1, 6, 8, 6).astype(np.float32)
        out = dev._forward(x)
        assert out["embedding"].shape == (64,)

    def test_logits_shape(self):
        dev = RenderNeuralDevice(num_classes=8)
        x = np.random.rand(1, 6, 8, 6).astype(np.float32)
        out = dev._forward(x)
        assert out["logits"].shape == (8,)

    def test_probabilities_sum_to_one(self):
        dev = RenderNeuralDevice()
        x = np.random.rand(1, 6, 8, 6).astype(np.float32)
        out = dev._forward(x)
        assert abs(out["probabilities"].sum() - 1.0) < 1e-4

    def test_features_shape(self):
        dev = RenderNeuralDevice()
        x = np.random.rand(1, 6, 8, 6).astype(np.float32)
        out = dev._forward(x)
        assert out["features"].shape == (32,)

    def test_embedding_normalized(self):
        dev = RenderNeuralDevice()
        x = np.random.rand(1, 6, 8, 6).astype(np.float32)
        out = dev._forward(x)
        norm = np.linalg.norm(out["embedding"])
        assert abs(norm - 1.0) < 1e-4

    def test_all_keys_present(self):
        dev = RenderNeuralDevice()
        x = np.random.rand(1, 6, 8, 6).astype(np.float32)
        out = dev._forward(x)
        assert set(out.keys()) == {"embedding", "logits", "probabilities", "features"}

    def test_custom_embed_dim(self):
        dev = RenderNeuralDevice(embed_dim=128)
        x = np.random.rand(1, 6, 8, 6).astype(np.float32)
        out = dev._forward(x)
        assert out["embedding"].shape == (128,)

    def test_custom_num_classes(self):
        dev = RenderNeuralDevice(num_classes=4)
        x = np.random.rand(1, 6, 8, 6).astype(np.float32)
        out = dev._forward(x)
        assert out["logits"].shape == (4,)

    def test_probabilities_all_positive(self):
        dev = RenderNeuralDevice()
        x = np.random.rand(1, 6, 8, 6).astype(np.float32)
        out = dev._forward(x)
        assert np.all(out["probabilities"] >= 0)

    def test_logits_range(self):
        dev = RenderNeuralDevice()
        x = np.random.rand(1, 6, 8, 6).astype(np.float32)
        out = dev._forward(x)
        # logits can be any real number
        assert out["logits"].shape == (8,)

    def test_features_dtype(self):
        dev = RenderNeuralDevice()
        x = np.random.rand(1, 6, 8, 6).astype(np.float32)
        out = dev._forward(x)
        assert np.issubdtype(out["features"].dtype, np.floating)

    def test_embedding_dtype(self):
        dev = RenderNeuralDevice()
        x = np.random.rand(1, 6, 8, 6).astype(np.float32)
        out = dev._forward(x)
        assert np.issubdtype(out["embedding"].dtype, np.floating)


# =============================================================================
# TestProcessPipeline
# =============================================================================

class TestProcessPipeline:
    def test_process_with_source(self):
        t = _tensors()
        dev = RenderNeuralDevice(cycles_device=_FakeCycles(t))
        out = dev.call("process")
        assert "embedding" in out
        assert out["embedding"].shape == (64,)

    def test_process_caches_inputs_outputs(self):
        t = _tensors()
        dev = RenderNeuralDevice(cycles_device=_FakeCycles(t))
        dev.call("process")
        assert dev._last_inputs is not None
        assert dev._last_outputs is not None

    def test_embed_with_source(self):
        t = _tensors()
        dev = RenderNeuralDevice(cycles_device=_FakeCycles(t))
        emb = dev.call("embed")
        assert emb.shape == (64,)

    def test_classify_with_source(self):
        t = _tensors()
        dev = RenderNeuralDevice(cycles_device=_FakeCycles(t))
        result = dev.call("classify")
        assert "labels" in result
        assert "probabilities" in result
        assert result["labels"].shape == (8, 6)

    def test_descriptor_with_source(self):
        t = _tensors()
        dev = RenderNeuralDevice(cycles_device=_FakeCycles(t))
        desc = dev.call("descriptor")
        assert "image" in desc
        assert "depth" in desc
        assert "neural_embedding_norm" in desc
        assert "neural_entropy" in desc
        assert "dominant_class" in desc

    def test_descriptor_has_stats(self):
        t = _tensors()
        dev = RenderNeuralDevice(cycles_device=_FakeCycles(t))
        desc = dev.call("descriptor")
        for key in ("mean", "std", "min", "max", "shape"):
            assert key in desc["image"]

    def test_descriptor_dominant_class_range(self):
        t = _tensors()
        dev = RenderNeuralDevice(cycles_device=_FakeCycles(t))
        desc = dev.call("descriptor")
        assert 0 <= desc["dominant_class"] < 8

    def test_process_embedding_normalized(self):
        t = _tensors()
        dev = RenderNeuralDevice(cycles_device=_FakeCycles(t))
        out = dev.call("process")
        norm = np.linalg.norm(out["embedding"])
        assert abs(norm - 1.0) < 1e-4

    def test_classify_labels_range(self):
        t = _tensors()
        dev = RenderNeuralDevice(cycles_device=_FakeCycles(t))
        result = dev.call("classify")
        assert np.all(result["labels"] >= 0)
        assert np.all(result["labels"] < 8)

    def test_descriptor_neural_entropy_non_negative(self):
        t = _tensors()
        dev = RenderNeuralDevice(cycles_device=_FakeCycles(t))
        desc = dev.call("descriptor")
        assert desc["neural_entropy"] >= 0

    def test_descriptor_embedding_norm_positive(self):
        t = _tensors()
        dev = RenderNeuralDevice(cycles_device=_FakeCycles(t))
        desc = dev.call("descriptor")
        assert desc["neural_embedding_norm"] > 0

    def test_process_without_source_raises(self):
        dev = RenderNeuralDevice()
        with pytest.raises(DeviceFault, match="no CyclesDevice source"):
            dev.call("process")

    def test_embed_without_source_raises(self):
        dev = RenderNeuralDevice()
        with pytest.raises(DeviceFault, match="no CyclesDevice source"):
            dev.call("embed")

    def test_classify_without_source_raises(self):
        dev = RenderNeuralDevice()
        with pytest.raises(DeviceFault, match="no CyclesDevice source"):
            dev.call("classify")

    def test_descriptor_without_source_raises(self):
        dev = RenderNeuralDevice()
        with pytest.raises(DeviceFault, match="no CyclesDevice source"):
            dev.call("descriptor")


# =============================================================================
# TestSoftmax
# =============================================================================

class TestSoftmax:
    def test_basic(self):
        x = np.array([1.0, 2.0, 3.0])
        result = _softmax(x)
        assert abs(result.sum() - 1.0) < 1e-6

    def test_uniform(self):
        x = np.array([1.0, 1.0, 1.0])
        result = _softmax(x)
        assert all(abs(v - 1.0 / 3.0) < 1e-6 for v in result)

    def test_large_values(self):
        x = np.array([1000.0, 1001.0, 1002.0])
        result = _softmax(x)
        assert abs(result.sum() - 1.0) < 1e-4

    def test_2d_axis(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _softmax(x, axis=-1)
        assert result.shape == (2, 2)
        assert all(abs(row.sum() - 1.0) < 1e-6 for row in result)

    def test_preserves_order(self):
        x = np.array([1.0, 5.0, 2.0])
        result = _softmax(x)
        assert result[1] > result[2] > result[0]

    def test_single_element(self):
        x = np.array([42.0])
        result = _softmax(x)
        assert abs(result[0] - 1.0) < 1e-6

    def test_negative_values(self):
        x = np.array([-3.0, -1.0, -2.0])
        result = _softmax(x)
        assert abs(result.sum() - 1.0) < 1e-6
        assert result[1] > result[2] > result[0]

    def test_all_zero(self):
        x = np.array([0.0, 0.0, 0.0])
        result = _softmax(x)
        assert all(abs(v - 1.0 / 3.0) < 1e-6 for v in result)

    def test_axis_0(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _softmax(x, axis=0)
        assert result.shape == (2, 2)
        assert all(abs(col.sum() - 1.0) < 1e-6 for col in result.T)


# =============================================================================
# TestXavierInit
# =============================================================================

class TestXavierInit:
    def test_conv1_weight_shape(self):
        dev = RenderNeuralDevice()
        assert dev._conv1_w.shape == (16, 6, 3, 3)

    def test_conv2_weight_shape(self):
        dev = RenderNeuralDevice()
        assert dev._conv2_w.shape == (32, 16, 3, 3)

    def test_proj_weight_shape(self):
        dev = RenderNeuralDevice()
        assert dev._proj_w.shape == (32, 64)

    def test_classify_weight_shape(self):
        dev = RenderNeuralDevice()
        assert dev._classify_w.shape == (32, 8)

    def test_biases_zero(self):
        dev = RenderNeuralDevice()
        assert np.allclose(dev._conv1_b, 0)
        assert np.allclose(dev._conv2_b, 0)
        assert np.allclose(dev._proj_b, 0)
        assert np.allclose(dev._classify_b, 0)

    def test_weights_not_all_zero(self):
        dev = RenderNeuralDevice()
        assert not np.allclose(dev._conv1_w, 0)
        assert not np.allclose(dev._conv2_w, 0)

    def test_weights_finite(self):
        dev = RenderNeuralDevice()
        assert np.all(np.isfinite(dev._conv1_w))
        assert np.all(np.isfinite(dev._conv2_w))
        assert np.all(np.isfinite(dev._proj_w))
        assert np.all(np.isfinite(dev._classify_w))

    def test_custom_embed_dim_proj_shape(self):
        dev = RenderNeuralDevice(embed_dim=32)
        assert dev._proj_w.shape == (32, 32)

    def test_custom_num_classes_shape(self):
        dev = RenderNeuralDevice(num_classes=4)
        assert dev._classify_w.shape == (32, 4)
        assert dev._classify_b.shape == (4,)
