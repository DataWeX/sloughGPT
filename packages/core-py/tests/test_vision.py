"""
Unit tests for domains.multimodal.vision module.
Covers ImageCaption, VisualObject dataclasses, VisionCNN model operations,
and get_vision_model factory.
"""

import sys
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, "packages/core-py")

from domains.multimodal.vision import (
    ImageCaption,
    VisualObject,
    VisionCNN,
    get_vision_model,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rgb_image(size=(32, 32)):
    """Return a small RGB PIL image filled with random pixels."""
    arr = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def _make_rgba_image(size=(32, 32)):
    """Return a small RGBA PIL image (tests RGB conversion)."""
    arr = np.random.randint(0, 256, (*size, 4), dtype=np.uint8)
    return Image.fromarray(arr, "RGBA")


def _make_batch(num=4):
    """Return (N, C, H, W) float32 batch and matching targets."""
    x = np.random.randn(num, 3, 32, 32).astype(np.float32)
    y = np.random.randn(num, 128).astype(np.float32)
    return x, y


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------

class TestImageCaption:
    def test_construction_all_fields(self):
        c = ImageCaption(text="hello", confidence=0.9, tags=["a", "b"], accuracy=0.8)
        assert c.text == "hello"
        assert c.confidence == 0.9
        assert c.tags == ["a", "b"]
        assert c.accuracy == 0.8

    def test_construction_default_accuracy(self):
        c = ImageCaption(text="x", confidence=0.5, tags=[])
        assert c.accuracy == 0.0

    def test_tags_are_mutable(self):
        c = ImageCaption(text="t", confidence=1.0, tags=["old"])
        c.tags.append("new")
        assert "new" in c.tags

    def test_confidence_can_be_zero(self):
        c = ImageCaption(text="", confidence=0.0, tags=[])
        assert c.confidence == 0.0


class TestVisualObject:
    def test_construction(self):
        obj = VisualObject(label="cat", bbox=[10.0, 20.0, 50.0, 60.0], confidence=0.75)
        assert obj.label == "cat"
        assert obj.bbox == [10.0, 20.0, 50.0, 60.0]
        assert obj.confidence == 0.75

    def test_bbox_is_mutable_list(self):
        obj = VisualObject(label="x", bbox=[0, 0, 0, 0], confidence=0.0)
        obj.bbox[0] = 99.0
        assert obj.bbox[0] == 99.0


# ---------------------------------------------------------------------------
# VisionCNN — construction and build_model
# ---------------------------------------------------------------------------

class TestVisionCNNConstruction:
    def test_default_state(self):
        v = VisionCNN()
        assert v._model is None
        assert v._optimizer is None
        assert v._learned is False


@pytest.mark.slow
class TestVisionCNNBuildModel:
    def test_build_model_sets_model(self):
        v = VisionCNN()
        v.build_model(embed_dim=64)
        assert v._model is not None
        assert v._optimizer is not None

    def test_build_model_default_embed_dim(self):
        v = VisionCNN()
        v.build_model()
        assert v._embed_dim == 128

    def test_build_model_custom_embed_dim(self):
        v = VisionCNN()
        v.build_model(embed_dim=256)
        assert v._embed_dim == 256

    def test_build_model_sets_learned_false(self):
        v = VisionCNN()
        v._learned = True
        v.build_model()
        assert v._learned is False


# ---------------------------------------------------------------------------
# _preprocess
# ---------------------------------------------------------------------------

class TestPreprocess:
    def test_rgb_image_shape(self):
        v = VisionCNN()
        img = _make_rgb_image()
        out = v._preprocess(img)
        assert out.shape == (1, 3, 32, 32)

    def test_rgba_image_converts_to_rgb(self):
        v = VisionCNN()
        img = _make_rgba_image()
        out = v._preprocess(img)
        assert out.shape == (1, 3, 32, 32)

    def test_output_dtype_is_float32(self):
        v = VisionCNN()
        out = v._preprocess(_make_rgb_image())
        assert out.dtype == np.float32

    def test_values_in_0_1_range(self):
        v = VisionCNN()
        out = v._preprocess(_make_rgb_image())
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_large_image_is_resized(self):
        v = VisionCNN()
        img = _make_rgb_image(size=(256, 256))
        out = v._preprocess(img)
        assert out.shape == (1, 3, 32, 32)

    def test_grayscale_image_converts(self):
        v = VisionCNN()
        arr = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        img = Image.fromarray(arr, "L")
        out = v._preprocess(img)
        assert out.shape == (1, 3, 32, 32)


# ---------------------------------------------------------------------------
# get_embedding
# ---------------------------------------------------------------------------

class TestGetEmbedding:
    def test_returns_1d_array(self):
        v = VisionCNN()
        v.build_model(embed_dim=64)
        emb = v.get_embedding(_make_rgb_image())
        assert isinstance(emb, np.ndarray)
        assert emb.ndim == 1

    def test_embedding_shape_matches_embed_dim(self):
        v = VisionCNN()
        v.build_model(embed_dim=64)
        emb = v.get_embedding(_make_rgb_image())
        assert emb.shape == (64,)

    def test_auto_builds_model_if_none(self):
        v = VisionCNN()
        assert v._model is None
        emb = v.get_embedding(_make_rgb_image())
        assert v._model is not None
        assert emb.shape[0] == 128  # default embed_dim


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestForward:
    def test_forward_returns_tensor(self):
        from domains.training.slonet import Tensor
        v = VisionCNN()
        v.build_model(embed_dim=64)
        x_np = np.random.randn(1, 3, 32, 32).astype(np.float32)
        from domains.training.slonet import tensor as _tensor
        x = _tensor(x_np, requires_grad=False)
        out = v.forward(x)
        assert isinstance(out, Tensor)

    def test_forward_output_shape(self):
        v = VisionCNN()
        v.build_model(embed_dim=64)
        x_np = np.random.randn(1, 3, 32, 32).astype(np.float32)
        from domains.training.slonet import tensor as _tensor
        x = _tensor(x_np, requires_grad=False)
        out = v.forward(x)
        assert out.data.shape == (1, 64)


# ---------------------------------------------------------------------------
# caption
# ---------------------------------------------------------------------------

class TestCaption:
    def test_untrained_returns_placeholder(self):
        v = VisionCNN()
        cap = v.caption(_make_rgb_image())
        assert isinstance(cap, ImageCaption)
        assert cap.confidence == 0.0
        assert cap.tags == []
        assert "untrained" in cap.text.lower() or "vision model" in cap.text.lower()

    def test_caption_auto_builds_model(self):
        v = VisionCNN()
        assert v._model is None
        cap = v.caption(_make_rgb_image())
        assert v._model is not None
        assert isinstance(cap, ImageCaption)

    def test_trained_returns_learned_text(self):
        v = VisionCNN()
        v.build_model(embed_dim=64)
        v._learned = True
        cap = v.caption(_make_rgb_image())
        assert isinstance(cap, ImageCaption)
        assert isinstance(cap.text, str)
        assert isinstance(cap.tags, list)


# ---------------------------------------------------------------------------
# detect
# ---------------------------------------------------------------------------

class TestDetect:
    def test_returns_list_of_visual_objects(self):
        v = VisionCNN()
        result = v.detect(_make_rgb_image())
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], VisualObject)

    def test_detect_bbox_is_zero(self):
        v = VisionCNN()
        result = v.detect(_make_rgb_image())
        assert result[0].bbox == [0, 0, 0, 0]

    def test_detect_label_matches_caption(self):
        v = VisionCNN()
        cap = v.caption(_make_rgb_image())
        result = v.detect(_make_rgb_image())
        assert result[0].label == cap.text


# ---------------------------------------------------------------------------
# train_on_batch
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestTrainOnBatch:
    def test_returns_float_loss(self):
        v = VisionCNN()
        x, y = _make_batch(2)
        loss = v.train_on_batch(x, y)
        assert isinstance(loss, float)

    def test_loss_is_finite(self):
        v = VisionCNN()
        x, y = _make_batch(2)
        loss = v.train_on_batch(x, y)
        assert np.isfinite(loss)

    def test_sets_learned_flag(self):
        v = VisionCNN()
        assert v._learned is False
        v.train_on_batch(*_make_batch(2))
        assert v._learned is True

    def test_multiple_steps_run_without_error(self):
        v = VisionCNN()
        x, y = _make_batch(2)
        losses = [v.train_on_batch(x, y) for _ in range(5)]
        assert len(losses) == 5
        assert all(isinstance(l, float) for l in losses)
        assert all(np.isfinite(l) for l in losses)

    def test_auto_builds_model(self):
        v = VisionCNN()
        assert v._model is None
        v.train_on_batch(*_make_batch(2))
        assert v._model is not None


# ---------------------------------------------------------------------------
# get_vision_model factory
# ---------------------------------------------------------------------------

class TestGetVisionModel:
    def test_returns_vision_cnn(self):
        model = get_vision_model()
        assert isinstance(model, VisionCNN)

    def test_returns_vision_cnn_with_default_name(self):
        model = get_vision_model(model_name="slonet")
        assert isinstance(model, VisionCNN)

    def test_returns_fresh_instance(self):
        m1 = get_vision_model()
        m2 = get_vision_model()
        assert m1 is not m2
