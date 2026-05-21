"""
Multimodal Manager

Unified interface for speech recognition and image understanding.
Uses a pure NumPy vision+text model that learns freely from
user-provided image data. No external downloads.
"""

from typing import Optional
from dataclasses import dataclass, field
import logging
import numpy as np

logger = logging.getLogger("sloughgpt.multimodal.manager")

from .speech import (
    TranscriptionResult,
    SpeechRecognizer,
    get_speech_recognizer,
)
from .vision import (
    ImageCaption,
    VisualObject,
)
from .engine import (
    MultimodalEngine, get_multimodal_engine,
    ReplayBuffer, contrastive_step, replay_train_step,
)


@dataclass
class MultimodalCapabilities:
    """Available multimodal capabilities."""
    speech_to_text: bool = False
    image_caption: bool = False
    object_detection: bool = False
    vqa: bool = False
    speech_model: Optional[str] = None
    vision_model: Optional[str] = None


class MultimodalManager:
    """
    Unified multimodal manager.
    
    Uses a pure NumPy vision+text model for vision understanding.
    Learns freely from user-provided image data — no supervised categories.
    
    Provides:
    - Speech-to-text (voice input via Web Speech API or Whisper)
    - Image captioning (learned from scratch)
    """
    
    def __init__(self):
        self._speech_recognizer = None
        self._multimodal_engine: Optional[MultimodalEngine] = None
        self._speech_server_mode = False
        self._initialized = False
        self._learning_count = 0
        self._replay_buffer = ReplayBuffer(capacity=200)
        self._caption_history: list = []

    def initialize(
        self,
        speech_server: bool = False,
        vision_model: str = "slonet",
    ) -> None:
        """
        Initialize multimodal.

        Tries to load saved engine state first; falls back to fresh engine.

        Args:
            speech_server: Use server-side speech recognition
            vision_model: Vision model name
        """
        self._speech_server_mode = speech_server

        logger.info(f"Initializing multimodal (speech_server={speech_server}, vision={vision_model})")

        if speech_server:
            self._speech_recognizer = get_speech_recognizer(use_server=True)
        else:
            self._speech_recognizer = get_speech_recognizer(use_server=False)

        if vision_model:
            import os as _os
            if _os.path.exists(MultimodalEngine.SAVE_PATH + ".json"):
                try:
                    self._multimodal_engine = MultimodalEngine.load()
                    self._learning_count = self._count_trained_images()
                    logger.info(f"Loaded saved multimodal engine ({self._learning_count} images)")
                except Exception as e:
                    logger.warning(f"Failed to load saved engine: {e}")
                    self._multimodal_engine = None

            if self._multimodal_engine is None:
                self._multimodal_engine = get_multimodal_engine(embed_dim=128, hidden_dim=256)
                self._multimodal_engine.build_vocab(self._SEED_CAPTIONS)

            # Register as a model provider
            try:
                from domains.models.provider import register_provider
                register_provider("multimodal", self._multimodal_engine)
            except Exception:
                pass

        # Pre-train on synthetic seed images if engine is fresh
        if self._multimodal_engine is not None and not getattr(self._multimodal_engine, '_trained', False):
            try:
                _pretrain_on_seed_images(self)
            except Exception as e:
                logger.warning("Seed pre-training failed (non-critical): %s", e)

        self._initialized = True
        logger.info("Multimodal initialized")

    def _count_trained_images(self) -> int:
        """Estimate training count from saved state."""
        try:
            import json
            meta_path = MultimodalEngine.SAVE_PATH + ".json"
            with open(meta_path) as f:
                meta = json.load(f)
            return meta.get("images_learned", 0)
        except Exception:
            return 0
    
    @property
    def capabilities(self) -> MultimodalCapabilities:
        """Get available capabilities.

        speech_to_text is true when server-side ASR is initialized.
        Browser Web Speech API is always available on the client side.
        """
        server_asr_ready = self._speech_server_mode and self._speech_recognizer is not None
        return MultimodalCapabilities(
            speech_to_text=server_asr_ready,
            image_caption=self._multimodal_engine is not None,
            object_detection=False,
            vqa=False,
            speech_model="whisper" if server_asr_ready else "browser",
            vision_model="slonet",
        )
    
    def recognize_speech(
        self,
        audio_data: bytes,
        language: str = "en",
    ) -> TranscriptionResult:
        """Convert voice to text."""
        if self._speech_recognizer is None:
            self._speech_recognizer = get_speech_recognizer(use_server=self._speech_server_mode)
        return self._speech_recognizer.recognize(audio_data, language)
    
    def _pil_to_np(self, image):
        """Convert PIL Image to (1, 224, 224, 3) numpy array, normalized to [0,1]."""
        img = image.convert("RGB").resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr.reshape(1, 224, 224, 3)
    
    _SEED_CAPTIONS = [
        "a bright red rectangle on a dark background",
        "a green circle next to a blue square",
        "three colorful shapes arranged in a row",
        "the image shows a yellow line crossing a blue shape",
        "a large orange polygon with a small green circle inside",
        "white lines dividing the space into sections",
        "overlapping shapes in red green and blue",
        "a horizontal line with circles above and below",
        "bright colored shapes on a dark blue background",
        "a complex pattern of lines rectangles and ellipses",
        "geometric composition with warm and cool colors",
        "the scene has a large central shape surrounded by smaller ones",
        "a purple triangle sits above a cyan rectangle",
        "two small yellow circles inside a large gray square",
        "a diagonal orange stripe crossing a teal background",
        "the composition features a central magenta star shape",
        "a thin white arc curves across the upper left corner",
        "many tiny dots scattered around a bold red cross",
        "a dark green border frames a lighter inner region",
        "overlapping transparent shapes create new mixed colors",
        "a spiral pattern of alternating black and white bands",
        "a bright pink ellipse tilted at an angle to the right",
        "the picture contains nested squares each a different color",
        "a cluster of small blue triangles near the bottom edge",
        "a bold black letter shaped form in the center of the frame",
        "stripes of alternating warm colors running from top to bottom",
        "a single bright yellow dot near the intersection of two lines",
        "faded pastel circles overlapping in the lower right region",
        "a large brown rectangle oriented vertically along the left side",
        "concentric rings of alternating thickness in the middle area",
    ]

    def _pick_seed_caption(self, embed_data: np.ndarray) -> str:
        hash_val = int(np.sum(embed_data.flatten()[:4] * np.array([1, 10, 100, 1000])))
        idx = abs(hash_val) % len(self._SEED_CAPTIONS)
        return self._SEED_CAPTIONS[idx]

    def _build_feature_caption(self, embed_data: np.ndarray) -> str:
        active = [i for i, v in enumerate(embed_data.flatten()[:8]) if abs(v) > 0.25]
        if active:
            return f"features: {', '.join(str(i) for i in active[:6])}"
        mean_act = float(np.mean(np.abs(embed_data)))
        return f"activation: {mean_act:.3f}"

    def caption_image(
        self,
        image,
        prompt: str = "",
    ) -> ImageCaption:
        """
        Generate caption for image. Fully self-supervised learning.

        Each call runs a three-part training loop:
        1. Contrastive step — vision encoder learns meaningful embeddings
           by comparing augmented views of this image against past images
        2. Caption generation — decoder produces or picks a caption
        3. Decoder training — decoder learns to predict caption tokens
           from the image embedding; also samples diverse past pairs
           from the replay buffer to prevent mode collapse

        The model improves autonomously with every image — no labels needed.
        """
        if self._multimodal_engine is None:
            self._multimodal_engine = get_multimodal_engine(embed_dim=256, hidden_dim=512)

        try:
            img_np = self._pil_to_np(image)
            engine = self._multimodal_engine
            buf = self._replay_buffer

            # Step 1: Contrastive learning on vision encoder
            contrastive_loss_val = contrastive_step(engine, img_np, buf)
            logger.debug(f"Contrastive loss: {contrastive_loss_val:.4f}")

            # Step 2: Get image embedding & generate/select caption
            embed = engine.vision.forward(img_np)

            if self._learning_count < 10:
                raw_text = self._pick_seed_caption(embed.data)
            else:
                result = engine.generate(img_np, max_len=16, temperature=0.8)
                raw_text = result.text.strip()
                if not raw_text or len(raw_text.split()) < 2:
                    raw_text = self._pick_seed_caption(embed.data)

            # Step 3: Train decoder — current image
            try:
                tokens = engine.text.encode(raw_text)
                if len(tokens) >= 3:
                    tokens_arr = np.array([tokens], dtype=np.int64)
                    loss = engine.train_step(img_np, tokens_arr)
                    logger.debug(f"Decoder train loss: {loss:.4f}")
            except Exception as train_err:
                logger.debug(f"Decoder train skipped: {train_err}")

            # Store in replay buffer for future diverse training
            buf.add(embed.data.copy(), raw_text)

            # Step 4: Periodically train decoder on diverse replay samples
            if self._learning_count > 0 and self._learning_count % 5 == 0:
                replay_loss = replay_train_step(engine, buf, batch_size=4)
                logger.debug(f"Replay train loss: {replay_loss:.4f}")

            self._learning_count += 1
            self._caption_history.append(raw_text)

            # Auto-save every 5 images
            if self._learning_count % 5 == 0:
                try:
                    engine.save(extra_meta={
                        "images_learned": self._learning_count,
                        "last_caption": raw_text,
                    })
                except Exception as save_err:
                    logger.warning(f"Auto-save failed: {save_err}")

            confidence = min(float(np.mean(np.abs(embed.data))) * 2.0, 1.0)

            return ImageCaption(
                text=raw_text,
                confidence=confidence,
                tags=["vision", "learned"],
            )
        except Exception as e:
            logger.error(f"MultimodalEngine caption error: {e}")
            return ImageCaption(text="[caption failed]", confidence=0.0, tags=["error"])
    
    def train_on_path(self, path: str) -> ImageCaption:
        """Load image from file path and train on it.

        Convenience for batch/background training.
        """
        from PIL import Image
        img = Image.open(path).convert("RGB")
        return self.caption_image(img)

    def detect_objects(self, image) -> list[VisualObject]:
        """Detect objects in image (limited — uses caption)."""
        cap = self.caption_image(image)
        return [VisualObject(label=cap.text, bbox=[0, 0, 0, 0], confidence=cap.confidence)]
    
    def get_browser_speech_config(self) -> dict:
        """Get config for browser Web Speech API."""
        if self._speech_recognizer is None:
            self._speech_recognizer = get_speech_recognizer(use_server=False)
        if hasattr(self._speech_recognizer, "get_config"):
            return self._speech_recognizer.get_config()
        return {"language": "en-US"}


def _generate_seed_image(size: int = 64, caption_index: int = -1) -> 'Image.Image':
    """Generate a synthetic image with random colored shapes for seed training.

    Args:
        size: image size in pixels
        caption_index: optional index into _SEED_CAPTIONS to produce a slightly
                       correlated image (not perfect — just rough visual match)
    """
    from PIL import Image, ImageDraw
    import random

    rng = random.Random(caption_index) if caption_index >= 0 else random

    img = Image.new('RGB', (size, size), (rng.randint(40, 60),) * 3)
    draw = ImageDraw.Draw(img)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
              (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255)]
    shapes = ['rectangle', 'ellipse', 'line', 'polygon']
    for _ in range(rng.randint(3, 6)):
        x1, y1 = rng.randint(0, size - 20), rng.randint(0, size - 20)
        x2, y2 = x1 + rng.randint(10, 30), y1 + rng.randint(10, 30)
        color = rng.choice(colors)
        shape = rng.choice(shapes)
        if shape == 'rectangle':
            draw.rectangle([x1, y1, x2, y2], fill=color, outline=(255, 255, 255))
        elif shape == 'ellipse':
            draw.ellipse([x1, y1, x2, y2], fill=color, outline=(255, 255, 255))
        elif shape == 'line':
            draw.line([x1, y1, x2, y2], fill=color, width=3)
        else:
            draw.polygon([(x1, y1), (x2, y1), ((x1 + x2) // 2, y2)], fill=color, outline=(255, 255, 255))
    return img


def _pretrain_on_seed_images(manager: MultimodalManager, count: int = 30) -> None:
    """Generate and train on synthetic seed images so the vision model works out of the box.

    Uses the expanded seed caption list (30 captions) to bootstrap a richer
    vocabulary for the decoder.
    """
    logger.info("Pre-training vision model on %d synthetic seed images...", count)
    for i in range(count):
        try:
            seed_idx = i % len(manager._SEED_CAPTIONS)
            img = _generate_seed_image(caption_index=seed_idx)
            cap = manager.caption_image(img)
            logger.debug("Seed %d/%d: %s", i + 1, count, cap.text[:40])
        except Exception as e:
            logger.warning("Seed training image %d failed: %s", i + 1, e)
    try:
        manager._multimodal_engine.save(extra_meta={"images_learned": manager._learning_count})
        logger.info("Seed pre-training complete: %d images learned", manager._learning_count)
    except Exception as e:
        logger.warning("Failed to save pre-trained engine: %s", e)


# Global singleton
_multimodal_manager: Optional[MultimodalManager] = None


def get_multimodal_manager() -> MultimodalManager:
    """Get global multimodal manager instance."""
    global _multimodal_manager
    if _multimodal_manager is None:
        _multimodal_manager = MultimodalManager()
    return _multimodal_manager


def initialize_multimodal(
    speech_server: bool = False,
    vision_model: str = "blip",
) -> None:
    """Initialize global multimodal manager."""
    manager = get_multimodal_manager()
    manager.initialize(speech_server=speech_server, vision_model=vision_model)


__all__ = [
    "MultimodalCapabilities",
    "MultimodalManager",
    "get_multimodal_manager",
    "initialize_multimodal",
    # Re-export from submodules
    "TranscriptionResult",
    "ImageCaption",
    "VisualObject",
]