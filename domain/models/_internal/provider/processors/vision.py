"""VisionProcessor — module-level imports, __slots__."""

from __future__ import annotations

import base64
import io
import logging
import re

# module-level imports (was inside _caption)
try:
    from PIL import Image

    _has_pil = True
except ImportError:
    Image = None  # type: ignore
    _has_pil = False

logger = logging.getLogger("slo.models.provider")


class VisionProcessor:
    """Extracts images, captions via vision provider, injects as text."""

    __slots__ = ("_provider_name",)

    def __init__(self, provider_name: str = "multimodal"):
        self._provider_name = provider_name

    def _extract_images(self, messages: list) -> list:
        images = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        images.append(url)
            if isinstance(content, str):
                for m in re.finditer(r'data:image/\w+;base64,([^"]+)', content):
                    images.append(m.group(0))
        return images

    def _ensure_provider(self) -> bool:
        from ..registry import get_provider

        if get_provider(self._provider_name) is not None:
            return True
        try:
            from domain.multimodal._internal.manager import (
                get_multimodal_manager,
                initialize_multimodal,
            )

            initialize_multimodal()
            mgr = get_multimodal_manager()
            if hasattr(mgr, "initialize"):
                mgr.initialize(vision_model="slonet")
            return get_provider(self._provider_name) is not None
        except Exception as e:
            logger.warning("Failed to init multimodal: %s", e, extra={"tag": "MODEL"})
            return False

    async def _caption(self, img_data: str) -> str:
        from ..registry import get_provider

        vision = get_provider(self._provider_name)
        if vision is not None:
            try:
                msg = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": img_data}}]}]
                text = ""
                async for token in vision.chat_stream(msg, max_tokens=30, temperature=0.7):
                    text += token
                if text.strip():
                    return text.strip()
            except Exception as e:
                logger.debug("Vision chat_stream caption failed: %s", e)
        try:
            clean = img_data.split(",")[1] if "," in img_data else img_data
            if not _has_pil:
                return "[image]"
            img = Image.open(io.BytesIO(base64.b64decode(clean))).convert("RGB")
            from domain.multimodal._internal.manager import get_multimodal_manager

            mgr = get_multimodal_manager()
            return mgr.caption_image(img).text
        except Exception as e:
            logger.warning("Caption failed: %s", e, extra={"tag": "MODEL"})
            return "[image]"

    async def process(self, messages: list) -> list:
        images = self._extract_images(messages)
        if not images:
            return messages
        if not self._ensure_provider():
            logger.warning("Vision provider unavailable, stripping images", extra={"tag": "MODEL"})
            result = []
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, list):
                    text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                    content = "\n".join(text_parts)
                    if text_parts and msg.get("role") == "user":
                        content = f"{content}\n[Image attached but model does not support vision]"
                result.append({"role": msg["role"], "content": content})
            return result
        captions = []
        for i, img in enumerate(images):
            cap = await self._caption(img)
            captions.append(cap)
            logger.info("Image %s captioned: %s", i + 1, cap[:60], extra={"tag": "MODEL"})
        caption_block = "\n".join(f"[Image: {c}]" for c in captions)
        result = []
        injected = False
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                content = "\n".join(text_parts)
            if not injected and msg.get("role") == "user":
                content = f"{content}\n{caption_block}"
                injected = True
            result.append({"role": msg["role"], "content": content})
        return result
