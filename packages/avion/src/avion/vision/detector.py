"""Visual detection — screenshot hashing, pixel diff, OCR, templates.

Heavy work (OCR, template matching, image decode) needs optional
dependencies (pytesseract, opencv-python, Pillow). When they are missing
every method degrades gracefully: hashes and byte comparison always work,
detection methods return [] instead of crashing.

Because ``[]`` cannot distinguish "nothing matched" from "the tool that
would have matched is not installed", ask first::

    caps = ImageAnalyzer().capabilities()
    if not caps.ocr:
        report(caps.reason)      # "ocr: missing pytesseract"
    else:
        hits = analyzer.ocr_text(png)
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import shutil
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any


class MatchMethod(Enum):
    """How a visual match was found."""

    TEXT_REGION = "text_region"
    TEMPLATE = "template"
    PIXEL_DIFF = "pixel_diff"
    HASH = "hash"


@dataclass(frozen=True)
class Capability:
    """Whether one detection method can run here, and why not if it can't."""

    name: str
    available: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.available

    def to_dict(self) -> dict[str, Any]:
        return {"available": self.available, "reason": self.reason}


def _missing_modules(*modules: str) -> list[str]:
    """Modules that are not importable here (never imports them)."""
    missing = []
    for name in modules:
        if name in sys.modules and sys.modules[name] is None:
            missing.append(name)  # explicitly stubbed out
            continue
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError, ModuleNotFoundError):
            found = False
        if not found:
            missing.append(name)
    return missing


def _capability(name: str, *modules: str) -> Capability:
    missing = _missing_modules(*modules)
    if missing:
        return Capability(name, False, f"missing {', '.join(missing)}")
    return Capability(name, True)


@dataclass(frozen=True)
class Capabilities:
    """Typed availability for every method that degrades."""

    ocr: Capability
    template: Capability
    pixel: Capability

    @property
    def all(self) -> tuple[Capability, ...]:
        return (self.ocr, self.template, self.pixel)

    @property
    def available(self) -> list[str]:
        """Names of the methods that will actually work."""
        return [c.name for c in self.all if c.available]

    @property
    def reason(self) -> str:
        """Why anything is unavailable ("" when everything works)."""
        return "; ".join(
            f"{c.name}: {c.reason}" for c in self.all if not c.available
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ocr": self.ocr.available,
            "template": self.template.available,
            "pixel": self.pixel.available,
            "reason": self.reason,
        }


@dataclass
class VisualMatch:
    """A region found in a screenshot."""

    bbox: Any  # BoundingBox from avion.interact.primitives
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
        caps = a.capabilities()          # what works in this environment
        h = a.compute_hash(png_bytes)
        diff = a.pixel_diff_bytes(before, after)  # 0.0 identical → 1.0
        for text, bbox in a.ocr_text(png_bytes):
            ...

    Hashing always works; the optional-dependency methods return [] when
    their library is missing — ``capabilities()`` says which and why.
    """

    def capabilities(self) -> Capabilities:
        """Typed availability of the degrading methods.

        Detects (without importing) pytesseract/Pillow for OCR, opencv +
        numpy + Pillow for template matching, Pillow for real pixel diffs.
        OCR additionally needs the ``tesseract`` binary on PATH, which
        pytesseract shells out to.
        """
        ocr = _capability("ocr", "pytesseract", "PIL")
        if ocr.available and shutil.which("tesseract") is None:
            ocr = Capability("ocr", False, "tesseract binary not on PATH")
        return Capabilities(
            ocr=ocr,
            template=_capability("template", "cv2", "numpy", "PIL"),
            pixel=_capability("pixel", "PIL"),
        )

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
        """OCR a screenshot → [(text, bbox)]. [] when OCR is unavailable.

        Check ``capabilities().ocr`` (and ``capabilities().reason``) to
        tell "no text found" apart from "tesseract not installed".
        """
        try:
            import pytesseract  # type: ignore[import]
        except ImportError:
            return []
        Image = _pil()
        if Image is None:
            return []
        try:
            from avion.interact.primitives import BoundingBox

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
        """Template-match → [VisualMatch]. [] when matching is unavailable.

        Check ``capabilities().template`` to tell "no match" apart from
        "opencv not installed".
        """
        try:
            import cv2  # type: ignore[import]
            import numpy as np  # type: ignore[import]
        except ImportError:
            return []
        Image = _pil()
        if Image is None:
            return []
        try:
            from avion.interact.primitives import BoundingBox

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
