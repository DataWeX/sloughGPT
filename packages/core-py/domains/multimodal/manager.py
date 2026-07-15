"""
Multimodal Manager

Unified interface for speech recognition and image understanding.
Uses a pure NumPy vision+text model that learns freely from
user-provided image data. No external downloads.
"""

from typing import Optional
from dataclasses import dataclass
import logging
import numpy as np

logger = logging.getLogger("man.multimodal.manager")

from .speech import (
    TranscriptionResult,
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
        self._accuracy_history: list = []

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

        logger.info(f"Initializing multimodal (speech_server={speech_server}, vision={vision_model})", extra={"tag": "MODEL"})

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
                    logger.info(f"Loaded saved multimodal engine ({self._learning_count} images)", extra={"tag": "MODEL"})
                except Exception as e:
                    logger.warning(f"Failed to load saved engine: {e}", extra={"tag": "MODEL"})
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

        # Pre-train on synthetic seed images in background (non-blocking)
        if self._multimodal_engine is not None and not getattr(self._multimodal_engine, '_trained', False):
            import threading
            t = threading.Thread(target=self._pretrain_engine, daemon=True, kwargs={"epochs": 10, "samples": 216})
            t.start()

        self._initialized = True
        logger.info("Multimodal initialized", extra={"tag": "MODEL"})

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

    def _gen_synthetic_data(self, count: int) -> tuple:
        """Generate synthetic image-caption pairs for seed training.

        Uses 6 colors × 3 shapes × 4 backgrounds × 3 templates = 216 combinations.
        """
        from PIL import Image, ImageDraw
        rng = np.random.RandomState(42)

        colors = {"red": (255,50,50), "green": (50,180,50), "blue": (50,50,255),
                  "yellow": (255,255,50), "purple": (180,50,180), "orange": (255,150,50)}
        shapes = ["circle", "square", "triangle"]
        backgrounds = {"black": (30,30,30), "gray": (100,100,100), "beige": (210,200,170), "white": (230,230,230)}
        templates = [
            "{color} {shape} on {bg} background",
            "{color} {shape} centered on {bg} background",
            "{color} {shape} over {bg} background",
        ]

        images, captions = [], []
        color_names = list(colors.keys())
        bg_names = list(backgrounds.keys())

        for i in range(count):
            c = color_names[i % len(color_names)]
            s = shapes[i % len(shapes)]
            b = bg_names[(i // 3) % len(bg_names)]
            t = templates[(i // 12) % len(templates)]
            cap = t.format(color=c, shape=s, bg=b)

            img = Image.new("RGB", (224, 224), backgrounds[b])
            draw = ImageDraw.Draw(img)
            cx, cy = 112, 112
            color_rgb = colors[c]
            r = 40
            if s == "circle":
                draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=color_rgb)
            elif s == "square":
                draw.rectangle([cx-r, cy-r, cx+r, cy+r], fill=color_rgb)
            elif s == "triangle":
                draw.polygon([(cx, cy-r), (cx-r, cy+r), (cx+r, cy+r)], fill=color_rgb)

            arr = np.array(img, dtype=np.float32) / 255.0
            images.append(arr)
            captions.append(cap)

        return np.stack(images), captions

    def _pretrain_engine(self, epochs: int = 10, samples: int = 216,
                         batch_size: int = 8, lr: float = 5e-4) -> float:
        """Run multi-epoch batched synthetic training to initialize the engine.

        Generates shape-caption pairs and trains both vision encoder and
        transformer decoder for the given number of epochs.

        Returns final loss.
        """
        engine = self._multimodal_engine
        if engine is None:
            return float("inf")

        images, captions = self._gen_synthetic_data(samples)
        n = len(images)
        logger.info("Pretraining on %d synthetic image-caption pairs (%d epochs, batch_size=%d)",
                     n, epochs, batch_size, extra={"tag": "MODEL"})

        # Build vocab from our captions
        engine.text.build_vocab(captions)

        final_loss = float("inf")
        for ep in range(epochs):
            idx = np.random.permutation(n)
            epoch_loss = 0.0
            steps = 0
            for start in range(0, n, batch_size):
                batch_idx = idx[start:start + batch_size]
                batch_imgs = images[batch_idx]
                batch_caps = [captions[i] for i in batch_idx]

                # Tokenize
                token_ids = []
                max_len = 0
                for c in batch_caps:
                    ids = engine.text.encode(c)
                    token_ids.append(ids)
                    max_len = max(max_len, len(ids))

                # Pad to max_len
                batch_tokens = np.zeros((len(batch_caps), max_len), dtype=np.int64)
                for i, ids in enumerate(token_ids):
                    batch_tokens[i, :len(ids)] = ids

                # Train step — vision encoder + transformer decoder
                loss = engine.train_step(batch_imgs, batch_tokens, lr=lr)
                epoch_loss += loss
                steps += 1

            avg_loss = epoch_loss / max(steps, 1)
            final_loss = avg_loss
            if (ep + 1) % 5 == 0 or ep == 0:
                logger.info("  Pretrain epoch %d/%d — loss: %.4f", ep + 1, epochs, avg_loss, extra={"tag": "MODEL"})

                sample_input = images[:1]
                result = engine.generate(sample_input, max_len=16, temperature=0.5)
                logger.info("    Sample: %s → %s", captions[0][:30], result.text.strip()[:40], extra={"tag": "MODEL"})

        # Fill replay buffer with training data
        for i in range(min(samples, len(images))):
            self._replay_buffer.add(images[i:i+1], captions[i])

        engine._trained = True
        engine.save(extra_meta={"images_learned": self._learning_count})
        logger.info("Pretrain complete — final loss: %.4f, engine saved", final_loss, extra={"tag": "MODEL"})
        return final_loss

    def caption_image(
        self,
        image,
        prompt: str = "",
        ground_truth: Optional[str] = None,
    ) -> ImageCaption:
        """
        Generate caption for image. Supports self-supervised and supervised learning.

        Each call runs a three-part training loop:
        1. Contrastive step — vision encoder learns meaningful embeddings
           by comparing augmented views of this image against past images
        2. Caption generation — decoder produces or picks a caption
        3. Decoder training — decoder learns to predict caption tokens
           from the image embedding; also samples diverse past pairs
           from the replay buffer to prevent mode collapse

        When ground_truth is provided, uses it as the training target
        instead of self-generated text, and computes BLEU accuracy.
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

            # Supervised mode: use ground truth as target
            if ground_truth and ground_truth.strip():
                raw_text = ground_truth.strip()
                # Still generate to compute accuracy
                result = engine.generate(img_np, max_len=16, temperature=0.8)
                generated_text = result.text.strip()
                # Compute BLEU accuracy
                from domains.feedback.lora_eval import BLEUScorer
                accuracy = BLEUScorer.score(generated_text, raw_text)
                self._accuracy_history.append(accuracy)
                logger.debug(f"Supervised training: BLEU={accuracy:.2f}")
            else:
                # Self-supervised mode
                if self._learning_count < 10:
                    raw_text = self._pick_seed_caption(embed.data)
                else:
                    result = engine.generate(img_np, max_len=16, temperature=0.8)
                    raw_text = result.text.strip()
                    if not raw_text or len(raw_text.split()) < 2:
                        raw_text = self._pick_seed_caption(embed.data)
                accuracy = 0.0

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
            buf.add(img_np.copy(), raw_text)

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
                    logger.warning(f"Auto-save failed: {save_err}", extra={"tag": "MODEL"})

            confidence = min(float(np.mean(np.abs(embed.data))) * 2.0, 1.0)

            return ImageCaption(
                text=raw_text,
                confidence=confidence,
                tags=["vision", "learned"] if not ground_truth else ["vision", "supervised"],
                accuracy=accuracy,
            )
        except Exception as e:
            logger.error(f"MultimodalEngine caption error: {e}", extra={"tag": "MODEL"})
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
