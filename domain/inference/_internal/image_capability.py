"""image_capability — the inference-owned image analysis contract.

Engine-first / dependency-inversion seam (card 20261031_081502): the
cognitive layer (``domain.agents``) must not reach into another domain's
private internals for a capability it consumes. ``analyze_image()`` is the
public contract; the provider is swappable — multimodal's ``VisionCNN``
backs it today, a VLM-backed provider lands when ``PDFVLMProcessor._get_vlm()``
stops returning ``None``.

The contract returns **raw data plus an error taxonomy**
(``error_kind``: ``not_found`` | ``unavailable`` | ``failed``) and never
formats: each caller (structured tool dict vs. model-facing text lines)
keeps its own presentation shim, so existing output shapes stay identical.

``VisionCNN`` is imported lazily inside the call, never at module import:
``domain.inference`` must not eagerly depend on ``domain.multimodal``, and
the honesty tests stub ``sys.modules["domain.multimodal._internal.vision"]``
before the call (a module-level import would bypass the stub).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# error_kind values — the complete taxonomy of non-success outcomes.
NOT_FOUND = "not_found"  # path does not exist (caller maps to its message)
UNAVAILABLE = "unavailable"  # vision provider import failed (deps missing)
FAILED = "failed"  # provider raised while analyzing


@dataclass
class ImageAnalysis:
    """Result of :func:`analyze_image` — data only, no presentation."""

    success: bool
    error_kind: str = ""
    error: str = ""  # NOT_FOUND: full message; FAILED: exception text
    format: str = ""  # lowercase extension incl. dot, e.g. ".png"
    size_bytes: int = 0
    caption: str = ""
    confidence: float = 0.0  # raw provider value — callers round
    tags: list[str] = field(default_factory=list)
    objects: list[dict[str, Any]] = field(default_factory=list)  # raw confidences
    embedding: Any = None  # np.ndarray | None — callers derive stats


def analyze_image(image_path: str) -> ImageAnalysis:
    """Analyze an image: caption, objects, embedding.

    Provider-backed (multimodal ``VisionCNN`` today). Never raises for the
    taxonomy above; ``OSError`` from stat-ing the path propagates so callers
    keep their existing outer error handling.
    """
    if not os.path.exists(image_path):
        return ImageAnalysis(
            success=False,
            error_kind=NOT_FOUND,
            error=f"Image not found: {image_path}",
        )

    size = os.path.getsize(image_path)
    ext = os.path.splitext(image_path)[1].lower()

    try:
        from domain.multimodal._internal.vision import VisionCNN

        vision = VisionCNN()
        caption_result = vision.caption(image_path)
        objects = vision.detect(image_path)
        embedding = vision.get_embedding(image_path)
    except ImportError:
        # Honest failure: the loop must see the tool did not run.
        return ImageAnalysis(
            success=False,
            error_kind=UNAVAILABLE,
            error="Vision model not available",
            format=ext,
            size_bytes=size,
        )
    except Exception as ve:
        logger.warning("VisionCNN analysis failed: %s", ve)
        return ImageAnalysis(
            success=False,
            error_kind=FAILED,
            error=str(ve),
            format=ext,
            size_bytes=size,
        )

    return ImageAnalysis(
        success=True,
        format=ext,
        size_bytes=size,
        caption=caption_result.text,
        confidence=caption_result.confidence,
        tags=caption_result.tags,
        objects=[{"label": obj.label, "confidence": obj.confidence} for obj in objects],
        embedding=embedding,
    )
