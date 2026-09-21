"""Visual detection — screenshot hashing, pixel diff, OCR, templates.

Heavy work (OCR, template matching, image decode) needs optional
dependencies (pytesseract, opencv-python, Pillow). When they are missing
every method degrades gracefully: hashes and byte comparison always work,
detection methods return [] instead of crashing.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from enum import Enum
from typing import Any


class MatchMethod(Enum):
    """How a visual match was found."""

    TEXT_REGION = "text_region"
    TEMPLATE = "template"
    PIXEL_DIFF = "pixel_diff"
    HASH = "hash"


@dataclass
class VisualMatch:
    """A region found in a screenshot."""

    bbox: Any  # BoundingBox from arken.interact.primitives
    confidence: float = 1.0
    method: MatchMethod = MatchMethod.HASH
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": self.bbox.to_dict(),
            "confidence": round(self.confidence, 4),
            "method": self.method.value,
            "label": self.label,
        }


def _pil():
    try:
        from PIL import Image  # type: ignore[import]

        return Image
    except ImportError:
        return None


def _pixels(img):
    """Pixel values, Pillow-version agnostic."""
    get = getattr(img, "get_flattened_data", None) or img.getdata
    return get()


class ImageAnalyzer:
    """Analyzes screenshot bytes.

    Usage::

        a = ImageAnalyzer()
        h = a.compute_hash(png_bytes)
        diff = a.pixel_diff_bytes(before, after)  # 0.0 identical → 1.0
        for text, bbox in a.ocr_text(png_bytes):
            ...
    """

    def compute_hash(self, data: bytes) -> str:
        """Stable short hash of image bytes (change detection)."""
        return hashlib.sha256(data).hexdigest()[:16]

    def pixel_diff_bytes(self, a: bytes, b: bytes) -> float:
        """Fraction of pixels that differ, 0.0–1.0.

        Identical bytes → 0.0. Without Pillow, any difference → 1.0.
        With Pillow, decodes both images and compares grayscale pixels.
        """
        if a == b:
            return 0.0
        Image = _pil()
        if Image is None:
            return 1.0
        try:
            ia = Image.open(io.BytesIO(a)).convert("L")
            ib = Image.open(io.BytesIO(b)).convert("L")
        except Exception:
            return 1.0
        size = (64, 64)
        pa = list(_pixels(ia.resize(size)))
        pb = list(_pixels(ib.resize(size)))
        diff = sum(1 for x, y in zip(pa, pb) if abs(x - y) > 8)
        return diff / (size[0] * size[1])

    def ocr_text(self, data: bytes) -> list[tuple[str, Any]]:
        """OCR a screenshot → [(text, bbox)]. [] when tesseract missing."""
        try:
            import pytesseract  # type: ignore[import]
        except ImportError:
            return []
        Image = _pil()
        if Image is None:
            return []
        try:
            from arken.interact.primitives import BoundingBox

            img = Image.open(io.BytesIO(data))
            words = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            out = []
            n = len(words.get("text", []))
            for i in range(n):
                text = (words["text"][i] or "").strip()
                if not text:
                    continue
                out.append(
                    (
                        text,
                        BoundingBox(
                            x=float(words["left"][i]),
                            y=float(words["top"][i]),
                            width=float(words["width"][i]),
                            height=float(words["height"][i]),
                        ),
                    )
                )
            return out
        except Exception:
            return []

    def find_template(
        self, screenshot: bytes, template: bytes, threshold: float = 0.8
    ) -> list[VisualMatch]:
        """Template-match → [VisualMatch]. [] when opencv missing."""
        try:
            import cv2  # type: ignore[import]
            import numpy as np  # type: ignore[import]
        except ImportError:
            return []
        Image = _pil()
        if Image is None:
            return []
        try:
            from arken.interact.primitives import BoundingBox

            img = np.array(Image.open(io.BytesIO(screenshot)).convert("L"))
            tpl = np.array(Image.open(io.BytesIO(template)).convert("L"))
            res = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
            matches = []
            ys, xs = (res >= threshold).nonzero()
            h, w = tpl.shape
            for x, y in zip(xs.tolist(), ys.tolist()):
                matches.append(
                    VisualMatch(
                        bbox=BoundingBox(x=float(x), y=float(y), width=float(w), height=float(h)),
                        confidence=float(res[y, x]),
                        method=MatchMethod.TEMPLATE,
                    )
                )
            return matches
        except Exception:
            return []
