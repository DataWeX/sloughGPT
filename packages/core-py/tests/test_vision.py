"""Tests for domains.multimodal.vision — ImageCaption, VisualObject, VisionCNN.

Covers: dataclass creation, model building, get_vision_model factory,
preprocessing, embedding, captioning, detection, training, forward pass.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_core_dir = str(Path(__file__).resolve().parents[2])
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

from domains.multimodal.vision import (
    ImageCaption,
    VisualObject,
    VisionCNN,
    get_vision_model,
)


class TestImageCaption:
    def test_creation(self):
        c = ImageCaption(text="a cat", confidence=0.9, tags=["animal", "cat"])
        assert c.text == "a cat"
        assert c.confidence == 0.9
        assert c.tags == ["animal", "cat"]
        assert c.accuracy == 0.0

    def test_custom_accuracy(self):
        c = ImageCaption(text="dog", confidence=0.8, tags=["dog"], accuracy=0.85)
        assert c.accuracy == 0.85

    def test_empty_tags(self):
        c = ImageCaption(text="x", confidence=0.5, tags=[])
        assert c.tags == []

    def test_zero_confidence(self):
        c = ImageCaption(text="none", confidence=0.0, tags=[])
        assert c.confidence == 0.0

    def test_full_confidence(self):
        c = ImageCaption(text="perfect", confidence=1.0, tags=["a"])
        assert c.confidence == 1.0

    def test_long_text(self):
        long_text = "word " * 100
        c = ImageCaption(text=long_text, confidence=0.5, tags=[])
        assert c.text == long_text

    def test_many_tags(self):
        tags = [f"tag{i}" for i in range(50)]
        c = ImageCaption(text="multi", confidence=0.5, tags=tags)
        assert len(c.tags) == 50

    def test_negative_confidence(self):
        c = ImageCaption(text="neg", confidence=-0.1, tags=[])
        assert c.confidence == -0.1

    def test_equality(self):
        c1 = ImageCaption(text="a", confidence=0.5, tags=["x"])
        c2 = ImageCaption(text="a", confidence=0.5, tags=["x"])
        assert c1.text == c2.text
        assert c1.confidence == c2.confidence

    def test_default_accuracy(self):
        c = ImageCaption(text="test", confidence=0.5, tags=[])
        assert c.accuracy == 0.0


class TestVisualObject:
    def test_creation(self):
        o = VisualObject(label="cat", bbox=[10, 20, 50, 60], confidence=0.95)
        assert o.label == "cat"
        assert o.bbox == [10, 20, 50, 60]
        assert o.confidence == 0.95

    def test_empty_bbox(self):
        o = VisualObject(label="x", bbox=[], confidence=0.5)
        assert o.bbox == []

    def test_zero_bbox(self):
        o = VisualObject(label="x", bbox=[0, 0, 0, 0], confidence=0.0)
        assert o.bbox == [0, 0, 0, 0]

    def test_large_bbox(self):
        o = VisualObject(label="big", bbox=[0, 0, 10000, 10000], confidence=1.0)
        assert o.bbox[2] == 10000

    def test_negative_coordinates(self):
        o = VisualObject(label="neg", bbox=[-10, -20, -5, -6], confidence=0.3)
        assert o.bbox[0] == -10

    def test_float_bbox(self):
        o = VisualObject(label="f", bbox=[1.5, 2.5, 3.5, 4.5], confidence=0.7)
        assert o.bbox[0] == 1.5

    def test_empty_label(self):
        o = VisualObject(label="", bbox=[0, 0, 1, 1], confidence=0.5)
        assert o.label == ""

    def test_equality(self):
        o1 = VisualObject(label="a", bbox=[1, 2, 3, 4], confidence=0.5)
        o2 = VisualObject(label="a", bbox=[1, 2, 3, 4], confidence=0.5)
        assert o1.label == o2.label
        assert o1.bbox == o2.bbox


class TestVisionCNN:
    def test_build_model(self):
        cnn = VisionCNN()
        cnn.build_model(embed_dim=64)
        assert cnn._model is not None

    def test_build_model_default_embed(self):
        cnn = VisionCNN()
        cnn.build_model()
        assert cnn._model is not None
        assert cnn._embed_dim == 128

    def test_initial_state(self):
        cnn = VisionCNN()
        assert cnn._model is None
        assert cnn._optimizer is None
        assert cnn._learned is False

    def test_caption_untrained(self):
        cnn = VisionCNN()
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        from PIL import Image
        pil_img = Image.fromarray(img)
        cap = cnn.caption(pil_img)
        assert "untrained" in cap.text.lower()
        assert cap.confidence == 0.0

    def test_detect(self):
        cnn = VisionCNN()
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        from PIL import Image
        pil_img = Image.fromarray(img)
        objs = cnn.detect(pil_img)
        assert len(objs) == 1
        assert objs[0].bbox == [0, 0, 0, 0]

    def test_train_on_batch(self):
        cnn = VisionCNN()
        cnn.build_model(embed_dim=64)
        images = np.random.randn(2, 3, 32, 32).astype(np.float32)
        targets = np.random.randn(2, 64).astype(np.float32)
        loss = cnn.train_on_batch(images, targets)
        assert isinstance(loss, float)
        assert cnn._learned is True

    def test_train_on_batch_loss_positive(self):
        cnn = VisionCNN()
        cnn.build_model(embed_dim=64)
        images = np.random.randn(2, 3, 32, 32).astype(np.float32)
        targets = np.random.randn(2, 64).astype(np.float32)
        loss = cnn.train_on_batch(images, targets)
        assert loss >= 0.0

    def test_multiple_train_steps(self):
        cnn = VisionCNN()
        cnn.build_model(embed_dim=64)
        images = np.random.randn(4, 3, 32, 32).astype(np.float32)
        targets = np.random.randn(4, 64).astype(np.float32)
        losses = []
        for _ in range(3):
            loss = cnn.train_on_batch(images, targets)
            losses.append(loss)
        assert cnn._learned is True
        assert all(isinstance(l, float) for l in losses)

    def test_caption_after_training(self):
        cnn = VisionCNN()
        cnn.build_model(embed_dim=64)
        images = np.random.randn(2, 3, 32, 32).astype(np.float32)
        targets = np.random.randn(2, 64).astype(np.float32)
        cnn.train_on_batch(images, targets)
        from PIL import Image
        img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
        cap = cnn.caption(img)
        assert "learned_feat" in cap.text or "untrained" in cap.text.lower()

    def test_preprocess_rgb(self):
        cnn = VisionCNN()
        from PIL import Image
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        result = cnn._preprocess(img)
        assert result.shape == (1, 3, 32, 32)
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_preprocess_grayscale_to_rgb(self):
        cnn = VisionCNN()
        from PIL import Image
        img = Image.fromarray(np.random.randint(0, 255, (32, 32), dtype=np.uint8), mode="L")
        img = img.convert("RGB")
        result = cnn._preprocess(img)
        assert result.shape == (1, 3, 32, 32)

    def test_preprocess_resizes(self):
        cnn = VisionCNN()
        from PIL import Image
        img = Image.fromarray(np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8))
        result = cnn._preprocess(img)
        assert result.shape == (1, 3, 32, 32)

    def test_get_embedding_builds_model_if_needed(self):
        cnn = VisionCNN()
        assert cnn._model is None
        from PIL import Image
        img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
        embed = cnn.get_embedding(img)
        assert cnn._model is not None
        assert isinstance(embed, np.ndarray)

    def test_get_embedding_shape(self):
        cnn = VisionCNN()
        cnn.build_model(embed_dim=64)
        from PIL import Image
        img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
        embed = cnn.get_embedding(img)
        assert embed.ndim == 1

    def test_forward_pass(self):
        from domains.training.slonet import tensor as _tensor
        cnn = VisionCNN()
        cnn.build_model(embed_dim=64)
        x = _tensor(np.random.randn(1, 3, 32, 32).astype(np.float32), requires_grad=False)
        out = cnn.forward(x)
        assert out.data.shape[0] == 1

    def test_detect_after_training(self):
        cnn = VisionCNN()
        cnn.build_model(embed_dim=64)
        images = np.random.randn(2, 3, 32, 32).astype(np.float32)
        targets = np.random.randn(2, 64).astype(np.float32)
        cnn.train_on_batch(images, targets)
        from PIL import Image
        img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
        objs = cnn.detect(img)
        assert len(objs) == 1
        assert isinstance(objs[0], VisualObject)

    def test_build_model_different_dims(self):
        cnn = VisionCNN()
        cnn.build_model(embed_dim=32)
        assert cnn._embed_dim == 32
        from PIL import Image
        img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
        embed = cnn.get_embedding(img)
        assert embed.ndim == 1


class TestGetVisionModel:
    def test_returns_cnn(self):
        model = get_vision_model()
        assert isinstance(model, VisionCNN)

    def test_returns_new_instance(self):
        m1 = get_vision_model()
        m2 = get_vision_model()
        assert m1 is not m2

    def test_default_model_not_built(self):
        model = get_vision_model()
        assert model._model is None

    def test_with_model_name(self):
        model = get_vision_model(model_name="slonet")
        assert isinstance(model, VisionCNN)

    def test_cnn_initial_learned_false(self):
        model = get_vision_model()
        assert model._learned is False

    def test_cnn_initial_optimizer_none(self):
        model = get_vision_model()
        assert model._optimizer is None


class TestVisionCNNTraining:
    def test_train_loss_finite(self):
        cnn = VisionCNN()
        cnn.build_model(embed_dim=64)
        images = np.random.randn(2, 3, 32, 32).astype(np.float32)
        targets = np.random.randn(2, 64).astype(np.float32)
        loss = cnn.train_on_batch(images, targets)
        assert np.isfinite(loss)

    def test_train_single_sample(self):
        cnn = VisionCNN()
        cnn.build_model(embed_dim=64)
        images = np.random.randn(1, 3, 32, 32).astype(np.float32)
        targets = np.random.randn(1, 64).astype(np.float32)
        loss = cnn.train_on_batch(images, targets)
        assert isinstance(loss, float)

    def test_train_large_batch(self):
        cnn = VisionCNN()
        cnn.build_model(embed_dim=64)
        images = np.random.randn(16, 3, 32, 32).astype(np.float32)
        targets = np.random.randn(16, 64).astype(np.float32)
        loss = cnn.train_on_batch(images, targets)
        assert isinstance(loss, float)
        assert cnn._learned is True

    def test_train_clears_grads(self):
        cnn = VisionCNN()
        cnn.build_model(embed_dim=64)
        images = np.random.randn(2, 3, 32, 32).astype(np.float32)
        targets = np.random.randn(2, 64).astype(np.float32)
        cnn.train_on_batch(images, targets)
        for p in cnn._model.parameters():
            assert p.grad is None

    def test_caption_empty_image(self):
        cnn = VisionCNN()
        from PIL import Image
        img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
        cap = cnn.caption(img)
        assert isinstance(cap, ImageCaption)

    def test_caption_white_image(self):
        cnn = VisionCNN()
        from PIL import Image
        img = Image.fromarray(np.full((32, 32, 3), 255, dtype=np.uint8))
        cap = cnn.caption(img)
        assert isinstance(cap, ImageCaption)

    def test_detect_empty_image(self):
        cnn = VisionCNN()
        from PIL import Image
        img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
        objs = cnn.detect(img)
        assert len(objs) == 1
        assert isinstance(objs[0], VisualObject)

    def test_embedding_consistency(self):
        cnn = VisionCNN()
        cnn.build_model(embed_dim=64)
        from PIL import Image
        img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
        e1 = cnn.get_embedding(img)
        e2 = cnn.get_embedding(img)
        np.testing.assert_array_equal(e1, e2)

    def test_embedding_different_images(self):
        cnn = VisionCNN()
        cnn.build_model(embed_dim=64)
        from PIL import Image
        img1 = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
        img2 = Image.fromarray(np.full((32, 32, 3), 255, dtype=np.uint8))
        e1 = cnn.get_embedding(img1)
        e2 = cnn.get_embedding(img2)
        assert not np.array_equal(e1, e2)

    def test_build_model_different_sizes(self):
        for embed_dim in [32, 64, 128]:
            cnn = VisionCNN()
            cnn.build_model(embed_dim=embed_dim)
            from PIL import Image
            img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
            embed = cnn.get_embedding(img)
            assert embed.ndim == 1

    def test_forward_output_shape(self):
        from domains.training.slonet import tensor as _tensor
        cnn = VisionCNN()
        cnn.build_model(embed_dim=64)
        x = _tensor(np.random.randn(2, 3, 32, 32).astype(np.float32), requires_grad=False)
        out = cnn.forward(x)
        assert out.data.shape[0] == 2

    def test_detect_returns_label_from_caption(self):
        cnn = VisionCNN()
        cnn.build_model(embed_dim=64)
        from PIL import Image
        img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
        objs = cnn.detect(img)
        cap = cnn.caption(img)
        assert objs[0].label == cap.text


class TestImageCaptionEdgeCases:
    def test_unicode_text(self):
        c = ImageCaption(text="UTF-8: cat", confidence=0.5, tags=["a"])
        assert "UTF-8" in c.text

    def test_special_characters(self):
        c = ImageCaption(text="a]b[c{d}e", confidence=0.5, tags=[])
        assert c.text == "a]b[c{d}e"

    def test_whitespace_only(self):
        c = ImageCaption(text="   ", confidence=0.5, tags=[])
        assert c.text == "   "

    def test_newlines_in_text(self):
        c = ImageCaption(text="line1\nline2", confidence=0.5, tags=[])
        assert "\n" in c.text

    def test_confidence_boundary_0(self):
        c = ImageCaption(text="x", confidence=0.0, tags=[])
        assert c.confidence == 0.0

    def test_confidence_boundary_1(self):
        c = ImageCaption(text="x", confidence=1.0, tags=[])
        assert c.confidence == 1.0

    def test_tags_with_duplicates(self):
        c = ImageCaption(text="x", confidence=0.5, tags=["a", "a", "b"])
        assert len(c.tags) == 3

    def test_tags_with_empty_strings(self):
        c = ImageCaption(text="x", confidence=0.5, tags=["", "", ""])
        assert len(c.tags) == 3

