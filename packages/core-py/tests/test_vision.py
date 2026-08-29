"""Tests for domains.multimodal.vision — ImageCaption, VisualObject, VisionCNN.

Covers: dataclass creation, model building, get_vision_model factory.
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


class TestVisualObject:
    def test_creation(self):
        o = VisualObject(label="cat", bbox=[10, 20, 50, 60], confidence=0.95)
        assert o.label == "cat"
        assert o.bbox == [10, 20, 50, 60]
        assert o.confidence == 0.95


class TestVisionCNN:
    def test_build_model(self):
        cnn = VisionCNN()
        cnn.build_model(embed_dim=64)
        assert cnn._model is not None

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


class TestGetVisionModel:
    def test_returns_cnn(self):
        model = get_vision_model()
        assert isinstance(model, VisionCNN)
