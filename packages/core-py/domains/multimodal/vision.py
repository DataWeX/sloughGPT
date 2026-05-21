"""
Image Understanding Module using SloNet.

Custom CNN for image captioning and visual understanding.
No external model downloads — uses the own-trained SloNet.
"""

from typing import List
from dataclasses import dataclass
import logging

logger = logging.getLogger("sloughgpt.vision")

from domains.training.slonet import (
    Tensor, SloNet, SloConv2D, SloBatchNorm2D, SloMaxPool2D,
    SloLinear, SloAdam, softmax as _softmax, relu as _relu,
    flatten as _flatten, tensor as _tensor,
)
import numpy as np


@dataclass
class ImageCaption:
    text: str
    confidence: float
    tags: List[str]


@dataclass
class VisualObject:
    label: str
    bbox: List[float]
    confidence: float


class VisionCNN:
    """Custom CNN for image understanding using SloNet.

    Learns freely from training data — no predefined categories.
    Outputs learned feature embeddings that can describe images.
    """

    def __init__(self):
        self._model = None
        self._optimizer = None
        self._learned = False

    def build_model(self, embed_dim=128):
        """Build a CNN that learns free representations."""
        self._embed_dim = embed_dim
        self._model = SloNet(layers=[
            SloConv2D(3, 32, kernel_size=3, padding=1),
            SloMaxPool2D(kernel_size=2, stride=2),
            _relu,

            SloConv2D(32, 64, kernel_size=3, padding=1),
            SloMaxPool2D(kernel_size=2, stride=2),
            _relu,

            SloConv2D(64, 128, kernel_size=3, padding=1),
            SloMaxPool2D(kernel_size=2, stride=2),
            _relu,

            _flatten,

            SloLinear(128 * 4 * 4, embed_dim),
        ])
        self._optimizer = SloAdam(lr=0.01)
        self._learned = False

    def _preprocess(self, img):
        """Convert PIL Image to normalized (N, C, H, W) tensor."""
        from PIL import Image
        if isinstance(img, str):
            img = Image.open(img)
        img = img.convert("RGB")
        img = img.resize((32, 32))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)
        return arr.reshape(1, 3, 32, 32)

    def get_embedding(self, image) -> np.ndarray:
        """Get learned feature embedding for an image."""
        if self._model is None:
            self.build_model()

        x_np = self._preprocess(image)
        x = _tensor(x_np, requires_grad=False)
        for layer in self._model.layers:
            if callable(layer):
                x = layer(x)
            else:
                x = layer.forward(x)
        return x.data.flatten()

    def caption(self, image) -> ImageCaption:
        """Generate a learned description from the feature embedding."""
        if self._model is None:
            self.build_model()

        try:
            embed = self.get_embedding(image)
            if not self._learned:
                return ImageCaption(
                    text="[vision model untrained — train on images to unlock free description]",
                    confidence=0.0,
                    tags=[],
                )
            vals = embed[:8]
            high = [i for i, v in enumerate(vals) if v > np.mean(vals)]
            return ImageCaption(
                text=f"learned_feat_[{'|'.join(str(i) for i in high)}]",
                confidence=float(np.mean(np.abs(embed))),
                tags=[f"f{i}" for i in high],
            )
        except Exception as e:
            logger.error(f"VisionCNN caption error: {e}")
            return ImageCaption(text="[vision model error]", confidence=0.0, tags=[])

    def detect(self, image) -> List[VisualObject]:
        caption = self.caption(image)
        return [VisualObject(label=caption.text, bbox=[0, 0, 0, 0], confidence=caption.confidence)]

    def train_on_batch(self, images_np: np.ndarray, targets_np: np.ndarray) -> float:
        """Train freely on batch — no predefined vocabulary."""
        if self._model is None:
            self.build_model()

        x = _tensor(images_np, requires_grad=True)
        y = _tensor(targets_np, requires_grad=False)

        embed = self.forward(x)
        loss = mse_loss(embed, y)
        loss.backward()
        self._optimizer.step(self._model.parameters())
        for p in self._model.parameters():
            p.grad = None

        self._learned = True
        return float(loss.data)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self._model.layers:
            if callable(layer):
                x = layer(x)
            else:
                x = layer.forward(x)
        return x


def get_vision_model(model_name: str = "slonet") -> object:
    """Get SloNet-based vision model."""
    return VisionCNN()


__all__ = [
    "ImageCaption",
    "VisualObject",
    "VisionCNN",
    "get_vision_model",
]
