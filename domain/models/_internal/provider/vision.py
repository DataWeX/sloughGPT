"""vision — resolve the configured vision provider (card 20261116_144837).

Single resolution point for "which vision model runs": ``config.vision.model``
(env ``SLO_VISION__MODEL``) names a registered provider; consumers
(``analyze_image``, ``VisionProcessor``, the router's ``describe_image`` tool)
call :func:`get_vision_provider` instead of hardcoding ``VisionCNN`` or the
``"multimodal"`` name. The text LLM is configured independently (``model.name``
/ external-provider settings) — the two seams never touch each other.

Honesty contract: an unknown configured name resolves to ``None`` — callers
report unavailable rather than silently substituting the built-in backend.
Only the built-in ``"multimodal"`` backend gets the lazy-init path (it is the
only provider this package can construct itself).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("slo.models.provider")

MULTIMODAL = "multimodal"


def get_vision_model_name() -> str:
    """Configured vision provider name; defaults to the built-in backend."""
    try:
        from domain.infrastructure._internal.config import get_config

        return get_config().vision.model or MULTIMODAL
    except Exception as e:
        logger.debug("vision config unavailable, defaulting to %s: %s", MULTIMODAL, e)
        return MULTIMODAL


def get_vision_provider(*, name: str | None = None, init: bool = False) -> Any | None:
    """Resolve a vision provider from the registry.

    Registry lookup only by default — never has side effects. ``init=True``
    lazily initializes the built-in multimodal backend when the resolved name
    is ``"multimodal"`` and it is not registered yet (the chat cold path).
    An unregistered non-multimodal name resolves to ``None``.
    """
    from .registry import get_provider

    resolved = name or get_vision_model_name()
    provider = get_provider(resolved)
    if provider is not None:
        return provider
    if init and resolved == MULTIMODAL:
        try:
            from domain.multimodal._internal.manager import initialize_multimodal

            initialize_multimodal()
        except Exception as e:
            logger.warning("Multimodal vision init failed: %s", e, extra={"tag": "MODEL"})
            return None
        provider = get_provider(resolved)
    return provider
