"""
Multimodal Engine — Vision + Text understanding.

No external downloads. Everything learned from scratch.
"""

import os
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging
import numpy as np

logger = logging.getLogger("man.multimodal.engine")

from domains.training.slonet import (
    Tensor, SloNet, SloConv2D, SloMaxPool2D, SloLinear,
    SloLSTM, SloEmbedding, SloLayerNorm, SloTransformerBlock, SloCrossAttention,
    SloMultiHeadAttention, SloFeedForward, SloRMSNorm, SloDropout, SloLayer,
    SloAdam, softmax as _softmax, relu as _relu,
    flatten as _flatten, tensor as _tensor,
    cross_entropy as _cross_entropy, sigmoid as _sigmoid,
    zeros, ones,
)

from .char_tokenizer import CharTokenizer


@dataclass
class MultimodalOutput:
    text: str
    confidence: float


class TextDecoder:
    """Learns a character-level vocabulary and generates text from image embeddings.

    Uses CharTokenizer — simple, deterministic, produces consistent token
    lengths (one token per character). Ideal for small-vocabulary debugging.
    """

    def __init__(self, embed_dim=256, hidden_dim=512):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.char = CharTokenizer()

    def build_vocab(self, texts: List[str]):
        """Build character vocabulary from training texts."""
        self.char.build_vocab(texts)

    def encode(self, text: str) -> List[int]:
        """Encode text using char tokenizer."""
        return self.char.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text using char tokenizer."""
        return self.char.decode(token_ids)

    @property
    def vocab_size(self) -> int:
        return self.char.vocab_size


class MultimodalEngine:
    """Unified vision + text engine — learns freely from images.

    Implements the ModelProvider protocol (duck typing):
    - chat_stream / chat for image captioning
    - embed for image → vector
    """

    SAVE_PATH = "data/multimodal/multimodal_engine.npz"
    _model_id = "multimodal-v1"

    def __init__(self, embed_dim=256, hidden_dim=512, n_vit_layers=3, n_heads=4,
                 n_decoder_layers=3, n_audio_layers=2):
        self.vision = VisionEncoder(embed_dim, n_heads, n_vit_layers)
        self.audio = AudioEncoder(embed_dim, n_heads, n_audio_layers)
        self.text = TextDecoder(embed_dim, hidden_dim)
        self.decoder = SloTransformerDecoder(
            vocab_size=0,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_decoder_layers,
        )
        self._trained = False

    def _pil_to_np(self, img) -> np.ndarray:
        """Convert PIL Image to (1, 224, 224, 3) numpy array, normalized to [0,1]."""
        img = img.convert("RGB").resize((VisionEncoder.IMAGE_SIZE, VisionEncoder.IMAGE_SIZE))
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr.reshape(1, VisionEncoder.IMAGE_SIZE, VisionEncoder.IMAGE_SIZE, 3)

    # ── ModelProvider protocol implementation ──

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def capabilities(self):
        from domains.models.provider import ModelCapabilities
        return ModelCapabilities(chat=True, streaming=False, embedding=True, vision=True)

    def _extract_images(self, messages: list) -> list:
        """Pull base64 images from messages."""
        images = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        images.append(url)
            if isinstance(content, str):
                import re
                for m in re.finditer(r'data:image/\w+;base64,([^"]+)', content):
                    images.append(m.group(0))
        return images

    async def chat_stream(self, messages: list, max_tokens: int = 512, temperature: float = 0.8, **kwargs):
        """Stream caption for the first image found in messages."""
        images = self._extract_images(messages)
        if not images:
            yield "no image found in message"
            return
        try:
            import base64
            from PIL import Image
            import io
            img_data = images[0]
            if "," in img_data:
                img_data = img_data.split(",")[1]
            img_bytes = base64.b64decode(img_data)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            from .manager import get_multimodal_manager
            mgr = get_multimodal_manager()
            caption = mgr.caption_image(img)
            yield caption.text
        except Exception as e:
            yield f"[error: {e}]"

    async def chat(self, messages: list, max_tokens: int = 512, temperature: float = 0.8, **kwargs) -> str:
        chunks = []
        async for chunk in self.chat_stream(messages, max_tokens, temperature, **kwargs):
            chunks.append(chunk)
        return "".join(chunks)

    def embed(self, text: str):
        """Image caption → embedding (identity, since captions ARE embeddings here)."""
        if not self._trained or not self.text.char._built:
            return [0.0] * 128
        tokens = self.text.char.encode(text)
        if not tokens:
            return [0.0] * 128
        vec = [0.0] * 128
        vec[tokens[0] % 128] = 1.0
        return vec

    @property
    def metadata(self):
        return {
            "vocab_size": self.text.vocab_size,
            "trained": self._trained,
            "embed_dim": self.vision.embed_dim,
        }

    def build_vocab(self, texts: List[str]):
        self.text.build_vocab(texts)
        char_vocab_size = self.text.vocab_size
        self.decoder = SloTransformerDecoder(
            vocab_size=char_vocab_size,
            embed_dim=self.text.embed_dim,
            hidden_dim=self.decoder.hidden_dim,
            n_heads=self.decoder.n_heads,
            n_layers=self.decoder.n_layers,
        )

    def save(self, path: str = "", extra_meta: dict = None) -> str:
        """Save engine state (vision + decoder weights + vocab) to .npz + JSON."""
        if not path:
            path = self.SAVE_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)

        weights = {}
        weights["vision_cls_token"] = self.vision.cls_token.data
        weights["vision_pos_embed"] = self.vision.pos_embed.data
        weights["vision_patch_proj_w"] = self.vision.patch_proj.weight.data
        weights["vision_patch_proj_b"] = self.vision.patch_proj.bias.data
        weights["vision_norm_w"] = self.vision.norm.weight.data
        weights["vision_norm_b"] = self.vision.norm.bias.data
        for i, block in enumerate(self.vision.blocks):
            for j, p in enumerate(block.parameters()):
                weights[f"vision_block{i}_{j}"] = p.data
        weights["audio_cls_token"] = self.audio.cls_token.data
        weights["audio_pos_embed"] = self.audio.pos_embed.data
        weights["audio_patch_proj_w"] = self.audio.patch_proj.weight.data
        weights["audio_patch_proj_b"] = self.audio.patch_proj.bias.data
        weights["audio_norm_w"] = self.audio.norm.weight.data
        weights["audio_norm_b"] = self.audio.norm.bias.data
        for i, block in enumerate(self.audio.blocks):
            for j, p in enumerate(block.parameters()):
                weights[f"audio_block{i}_{j}"] = p.data
        for i, p in enumerate(self.decoder.parameters()):
            weights[f"decoder_{i}"] = p.data
        np.savez_compressed(path, **weights)

        meta = {
            "char_vocab": [c for c in self.text.char.vocab if c not in set(CharTokenizer.SPECIAL_TOKENS)],
            "embed_dim": self.vision.embed_dim,
            "hidden_dim": self.decoder.hidden_dim,
            "n_vit_layers": len(self.vision.blocks),
            "n_audio_layers": len(self.audio.blocks),
            "n_heads": self.vision.n_heads,
            "n_decoder_layers": self.decoder.n_layers,
            "trained": self._trained,
        }
        if extra_meta:
            meta.update(extra_meta)
        meta_path = path + ".json"
        import json
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Multimodal engine saved to {path}")
        return path

    @classmethod
    def load(cls, path: str = "") -> "MultimodalEngine":
        """Load engine state from .npz + JSON."""
        if not path:
            path = cls.SAVE_PATH
        meta_path = path + ".json"

        import json
        with open(meta_path) as f:
            meta = json.load(f)

        n_vit_layers = meta.get("n_vit_layers", 3)
        n_heads = meta.get("n_heads", 4)
        n_decoder_layers = meta.get("n_decoder_layers", 3)
        n_audio_layers = meta.get("n_audio_layers", 2)
        embed_dim = meta["embed_dim"]
        hidden_dim = meta["hidden_dim"]
        engine = cls(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_vit_layers=n_vit_layers,
            n_heads=n_heads,
            n_decoder_layers=n_decoder_layers,
            n_audio_layers=n_audio_layers,
        )
        # Restore char tokenizer
        chars = meta.get("char_vocab", [])
        engine.text.char.build_vocab(chars or ["a", "b", "c"])
        char_vocab_size = engine.text.vocab_size
        engine.decoder = SloTransformerDecoder(
            vocab_size=char_vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_decoder_layers,
        )
        engine._trained = meta.get("trained", False)

        data = np.load(path)
        # Load vision transformer weights
        engine.vision.cls_token.data = data["vision_cls_token"]
        engine.vision.pos_embed.data = data["vision_pos_embed"]
        engine.vision.patch_proj.weight.data = data["vision_patch_proj_w"]
        engine.vision.patch_proj.bias.data = data["vision_patch_proj_b"]
        engine.vision.norm.weight.data = data["vision_norm_w"]
        engine.vision.norm.bias.data = data["vision_norm_b"]
        for i, block in enumerate(engine.vision.blocks):
            j = 0
            while f"vision_block{i}_{j}" in data:
                params = block.parameters()
                if j < len(params):
                    params[j].data = data[f"vision_block{i}_{j}"]
                j += 1
        # Load audio encoder weights
        if "audio_cls_token" in data:
            engine.audio.cls_token.data = data["audio_cls_token"]
            engine.audio.pos_embed.data = data["audio_pos_embed"]
            engine.audio.patch_proj.weight.data = data["audio_patch_proj_w"]
            engine.audio.patch_proj.bias.data = data["audio_patch_proj_b"]
            engine.audio.norm.weight.data = data["audio_norm_w"]
            engine.audio.norm.bias.data = data["audio_norm_b"]
            for i, block in enumerate(engine.audio.blocks):
                j = 0
                while f"audio_block{i}_{j}" in data:
                    params = block.parameters()
                    if j < len(params):
                        params[j].data = data[f"audio_block{i}_{j}"]
                    j += 1
        # Load decoder weights
        d_idx = 0
        for p in engine.decoder.parameters():
            key = f"decoder_{d_idx}"
            if key in data:
                p.data = data[key]
            d_idx += 1

        logger.info(f"Multimodal engine loaded from {path}")
        return engine

    @property
    def embed_dim(self):
        return self.vision.embed_dim

    def _concat_modalities(self, images_np: Optional[np.ndarray] = None, audio_np: Optional[np.ndarray] = None,
                           audio_patches: Optional[np.ndarray] = None) -> Tuple[Tensor, Tensor, List]:
        """Produce (embed, patches, optimizers) from optional image and audio inputs.

        Returns:
            embed: (B, 1, embed_dim) — cls token from lead modality (image > audio)
            patches: (B, total_patches, embed_dim) — concatenated patch embeddings
            optimizers: list of optimizers to step
        """
        patches_list = []
        optimizers = [self.decoder.optimizer]

        if images_np is not None:
            img_embed = self.vision.forward(images_np)
            img_patches = self.vision.get_patch_embeddings(images_np)
            patches_list.append(img_patches)
            optimizers.append(self.vision.optimizer)
        else:
            img_embed = None

        aud_embed = None
        if audio_patches is not None:
            aud_patches = self.audio._embed_patches(audio_patches)
            patches_list.append(aud_patches)
            optimizers.append(self.audio.optimizer)
            aud_embed = aud_patches[:, 0:1, :]
        elif audio_np is not None:
            aud_patches = self.audio.get_patch_embeddings(audio_np)
            patches_list.append(aud_patches)
            optimizers.append(self.audio.optimizer)
            # CLS token from patch embeddings (avoids 2nd transformer pass)
            aud_embed = aud_patches[:, 0:1, :]

        if len(patches_list) == 0:
            raise ValueError("At least one of images_np or audio_np must be provided")

        embed = img_embed if img_embed is not None else aud_embed

        # Concatenate all patch embeddings along sequence dimension
        all_patches = Tensor(
            np.concatenate([p.data for p in patches_list], axis=1),
            requires_grad=True,
            _children=tuple(patches_list),
        )
        return embed, all_patches, optimizers

    def forward(self, images_np: np.ndarray, token_ids: np.ndarray,
                audio_np: Optional[np.ndarray] = None) -> Tuple[Tensor, Tensor]:
        img_embed = self.vision.forward(images_np)
        patches = None
        if audio_np is not None:
            aud_patches = self.audio.get_patch_embeddings(audio_np)
            img_patches = self.vision.get_patch_embeddings(images_np)
            patches = Tensor(
                np.concatenate([img_patches.data, aud_patches.data], axis=1),
                requires_grad=True,
                _children=(img_patches, aud_patches),
            )
        logits, _ = self.decoder.forward(img_embed, token_ids, patches)
        return logits, img_embed

    def precompute_audio_patches(self, audio_np: np.ndarray) -> np.ndarray:
        """Precompute audio raw patches (B, N, input_dim) without STFT per epoch."""
        return self.audio.extract_patches(audio_np)

    def _maybe_set_lr(self, lr: Optional[float], optimizers: list):
        """Temporarily override optimiser LR if provided."""
        if lr is not None:
            for opt in optimizers:
                opt.lr = lr

    def _maybe_restore_lr(self, lr: Optional[float], optimizers: list, old_lrs: dict):
        if lr is not None:
            for opt in optimizers:
                opt.lr = old_lrs.get(id(opt), opt.lr)

    def _clip_gradients(self, params: list, max_norm: float = 1.0):
        """Clip gradients by global norm."""
        total_norm = 0.0
        for p in params:
            if p.grad is not None:
                g_data = p.grad.data if hasattr(p.grad, 'data') else p.grad
                total_norm += float(np.sum(np.asarray(g_data, dtype=np.float64).ravel() ** 2))
        total_norm = np.sqrt(total_norm)
        if total_norm > max_norm and total_norm > 0:
            scale = float(max_norm / total_norm)
            for p in params:
                if p.grad is not None:
                    g_data = p.grad.data if hasattr(p.grad, 'data') else p.grad
                    g_data *= scale

    def _zero_grad(self, optimizers: list):
        for opt in optimizers:
            for p in self._params_for_optimizer(opt, None, None):
                p.grad = None

    def _sum_grads(self, params: list, scale: float):
        """Scale all gradients by a factor (in-place)."""
        for p in params:
            if p.grad is not None:
                g_data = p.grad.data if hasattr(p.grad, 'data') else p.grad
                g_data *= scale

    def train_step(
        self,
        images_np: Optional[np.ndarray] = None,
        text_tokens: Optional[np.ndarray] = None,
        lr: Optional[float] = None,
        audio_np: Optional[np.ndarray] = None,
        audio_patches: Optional[np.ndarray] = None,
        temperature: float = 1.0,
    ) -> float:
        if text_tokens is None:
            raise ValueError("text_tokens is required")
        embed, patches, optimizers = self._concat_modalities(images_np, audio_np, audio_patches)
        logits, _ = self.decoder.forward(embed, text_tokens[:, :-1], patches)
        targets = _tensor(text_tokens[:, 1:].reshape(-1), requires_grad=False)
        if temperature != 1.0:
            logits = logits / temperature
        loss = _cross_entropy(logits, targets)
        loss.backward()
        old_lrs = {}
        if lr is not None:
            old_lrs = {id(opt): opt.lr for opt in optimizers}
            self._maybe_set_lr(lr, optimizers)
        # Gradient clipping
        for opt in optimizers:
            self._clip_gradients(self._params_for_optimizer(opt, None, None))
        for opt in optimizers:
            opt.step(self._params_for_optimizer(opt, embed, patches))
        self._maybe_restore_lr(lr, optimizers, old_lrs)
        self._zero_grad(optimizers)
        self._trained = True
        return float(loss.data)

    def train_batch(
        self,
        samples: list,
        lr: Optional[float] = None,
        temperature: float = 1.0,
    ) -> float:
        """Accumulate gradients over multiple samples then step once.

        Each sample is ``(images_np, text_tokens, audio_np, audio_patches)``
        where ``audio_np`` and ``audio_patches`` are optional (None).  Gradients
        are summed (not averaged) across the batch, then clipped and applied.

        Args:
            samples: list of (images_np, text_tokens, audio_np, audio_patches) tuples
            lr: optional learning rate override
            temperature: softmax temperature for logit scaling (1.0 = standard CE).
                         Values >1 create softer targets for better exploration.

        Returns:
            mean loss across samples
        """
        if not samples:
            return 0.0

        # Determine which optimisers are needed from the first sample
        first_img, first_tok, first_aud_np, first_aud_pt = samples[0]
        _, _, all_opts = self._concat_modalities(first_img, first_aud_np, first_aud_pt)
        # Zero all relevant param gradients before accumulation
        self._zero_grad(all_opts)

        total_loss = 0.0
        n = 0
        old_lrs = {} if lr is None else {id(o): o.lr for o in all_opts}
        if lr is not None:
            self._maybe_set_lr(lr, all_opts)

        for images_np, text_tokens, audio_np, audio_patches in samples:
            if text_tokens is None:
                continue
            embed, patches, _ = self._concat_modalities(images_np, audio_np, audio_patches)
            logits, _ = self.decoder.forward(embed, text_tokens[:, :-1], patches)
            targets = _tensor(text_tokens[:, 1:].reshape(-1), requires_grad=False)
            if temperature != 1.0:
                logits = logits / temperature
            loss = _cross_entropy(logits, targets)
            # Scale loss by 1/N so the sum ≈ mean (even gradient contribution)
            loss = loss * (1.0 / len(samples))
            loss.backward()
            total_loss += float(loss.data) * len(samples)
            n += 1

        if n == 0:
            return 0.0

        # Gradient clipping
        for opt in all_opts:
            self._clip_gradients(self._params_for_optimizer(opt, None, None))
        # Single step
        for opt in all_opts:
            opt.step(self._params_for_optimizer(opt, None, None))
        self._maybe_restore_lr(lr, all_opts, old_lrs)
        self._zero_grad(all_opts)
        self._trained = True
        return total_loss / n

    def _params_for_optimizer(self, opt, embed, patches):
        if opt is self.decoder.optimizer:
            return self.decoder.parameters()
        if opt is self.vision.optimizer:
            return self.vision.parameters()
        if opt is self.audio.optimizer:
            return self.audio.parameters()
        return []

    def generate(self, image_np: Optional[np.ndarray] = None, max_len: int = 20,
                 temperature: float = 1.0, audio_np: Optional[np.ndarray] = None,
                 audio_patches: Optional[np.ndarray] = None) -> MultimodalOutput:
        embed, patches, _ = self._concat_modalities(image_np, audio_np, audio_patches)
        bos = 0
        eos = 1
        tokens = [bos]

        for _ in range(max_len):
            inp = _tensor(np.array([tokens]), requires_grad=False)
            logits, _ = self.decoder.forward(embed, inp, patches)
            logits_2d = logits.data.reshape(-1, logits.data.shape[-1])  # (seq_len, vocab_size)
            last_pos = logits_2d[-1]  # (vocab_size,)
            if temperature > 0 and self._trained:
                probs = _softmax(_tensor(last_pos[np.newaxis, :], requires_grad=False) / temperature)
                probs_np = probs.data.flatten()
                probs_np = np.maximum(probs_np, 1e-8)
                for t in tokens[1:]:
                    if 0 <= t < len(probs_np):
                        probs_np[t] *= 0.4
                probs_np /= probs_np.sum()
                next_tok = int(np.random.choice(len(probs_np), p=probs_np))
            else:
                scores = last_pos.copy()
                for t in tokens[1:]:
                    if 0 <= t < len(scores):
                        scores[t] -= 5.0
                next_tok = int(np.argmax(scores))
            if next_tok == eos:
                break
            tokens.append(next_tok)

        text = self.text.decode(tokens)
        conf = float(np.mean(np.abs(embed.data)))
        return MultimodalOutput(text=text, confidence=conf)


class VisionEncoder:
    """ViT-style image encoder with patch positional embeddings.
    
    Processes 224x224 RGB images by splitting into 32x32 patches (49 patches),
    projecting each patch to embed_dim, and adding positional embeddings.
    Output: (B, num_patches+1, embed_dim) with class token.
    """
    PATCH_SIZE = 32
    IMAGE_SIZE = 224

    def __init__(self, embed_dim=256, n_heads=8, n_layers=3):
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.patch_dim = 3 * self.PATCH_SIZE * self.PATCH_SIZE  # 3 * 32 * 32 = 3072
        self.num_patches = (self.IMAGE_SIZE // self.PATCH_SIZE) ** 2  # 7*7 = 49
        self.cls_token = Tensor(np.random.randn(1, 1, embed_dim).astype(np.float32) * 0.02, requires_grad=True)
        self.pos_embed = Tensor(np.random.randn(1, self.num_patches + 1, embed_dim).astype(np.float32) * 0.02, requires_grad=True)
        self.patch_proj = SloLinear(self.patch_dim, embed_dim)
        self.norm = SloLayerNorm(embed_dim)
        # Transformer blocks for vision
        self.blocks = [
            SloTransformerBlock(embed_dim, n_heads, use_rope=True, dropout=0.1, name=f"vit_block_{i}")
            for i in range(n_layers)
        ]
        self.optimizer = SloAdam(lr=3e-4)

    def extract_patches(self, images_np: np.ndarray) -> np.ndarray:
        """Split (B, H, W, C) images into (B, num_patches, patch_dim) patches."""
        B, H, W, C = images_np.shape
        p = self.PATCH_SIZE
        assert H == self.IMAGE_SIZE and W == self.IMAGE_SIZE, f"Expected {self.IMAGE_SIZE}x{self.IMAGE_SIZE}, got {H}x{W}"
        # Reshape to (B, H/p, p, W/p, p, C) -> (B, (H/p)*(W/p), p*p*C)
        patches = images_np.reshape(B, H // p, p, W // p, p, C)
        patches = patches.transpose(0, 1, 3, 2, 4, 5)  # (B, H/p, W/p, p, p, C)
        patches = patches.reshape(B, -1, p * p * C)
        return patches

    def forward(self, images_np: np.ndarray) -> Tensor:
        """Forward pass: patches -> embeddings -> transformer -> cls token."""
        patches = self.extract_patches(images_np)
        B = patches.shape[0]
        x = self.patch_proj.forward(_tensor(patches, requires_grad=False))
        # Prepend cls token - preserve gradient by using Tensor repeat
        cls_tokens_data = self.cls_token.data.repeat(B, axis=0)
        cls_tokens = Tensor(cls_tokens_data, requires_grad=True, _children=(self.cls_token,))
        x_data = np.concatenate([cls_tokens.data, x.data], axis=1)
        # Add positional embeddings
        x_data = x_data + self.pos_embed.data
        x = Tensor(x_data, requires_grad=True, _children=(x, self.pos_embed, cls_tokens))
        # Pass through transformer blocks
        for block in self.blocks:
            x, _ = block.forward(x)
        # Return normalized cls token (first position) as image embedding
        cls_out = x[:, 0:1, :]  # (B, 1, embed_dim)
        cls_out = self.norm.forward(cls_out)
        return cls_out

    def get_patch_embeddings(self, images_np: np.ndarray) -> Tensor:
        """Return all patch embeddings (B, num_patches+1, embed_dim) for cross-attention."""
        patches = self.extract_patches(images_np)
        B = patches.shape[0]
        x = self.patch_proj.forward(_tensor(patches, requires_grad=False))
        cls_tokens_data = self.cls_token.data.repeat(B, axis=0)
        cls_tokens = Tensor(cls_tokens_data, requires_grad=True, _children=(self.cls_token,))
        x_data = np.concatenate([cls_tokens.data, x.data], axis=1)
        x_data = x_data + self.pos_embed.data
        x = Tensor(x_data, requires_grad=True, _children=(x, self.pos_embed, cls_tokens))
        for block in self.blocks:
            x, _ = block.forward(x)
        return self.norm.forward(x)

    def parameters(self):
        """Return all trainable parameters."""
        params = [self.cls_token, self.pos_embed]
        params += self.patch_proj.parameters()
        params += self.norm.parameters()
        for block in self.blocks:
            params += block.parameters()
        return [p for p in params if p.requires_grad]


class AudioEncoder:
    """Spectrogram-based audio encoder feeding into the transformer decoder.

    Takes raw audio waveform -> mel spectrogram -> patches -> embeddings -> CLS token.
    Output: (B, num_patches+1, embed_dim) — same shape as VisionEncoder for cross-attention.
    """

    SAMPLE_RATE = 16000
    N_MELS = 80
    N_FFT = 512
    HOP_LENGTH = 160  # 10ms at 16kHz
    PATCH_SECONDS = 0.5  # seconds of audio per patch
    MAX_SECONDS = 30     # max audio duration

    def __init__(self, embed_dim=256, n_heads=4, n_layers=2):
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        patches_per_sec = self.SAMPLE_RATE / self.HOP_LENGTH  # 100 frames/sec
        self.frames_per_patch = int(patches_per_sec * self.PATCH_SECONDS)  # 500 frames
        self.max_patches = int(self.MAX_SECONDS / self.PATCH_SECONDS)  # 6
        self.input_dim = self.N_MELS * self.frames_per_patch  # 80 * 500 = 40000
        self.cls_token = Tensor(np.random.randn(1, 1, embed_dim).astype(np.float32) * 0.02, requires_grad=True)
        self.pos_embed = Tensor(np.random.randn(1, self.max_patches + 1, embed_dim).astype(np.float32) * 0.02, requires_grad=True)
        self.patch_proj = SloLinear(self.input_dim, embed_dim)
        self.norm = SloLayerNorm(embed_dim)
        self.blocks = [
            SloTransformerBlock(embed_dim, n_heads, use_rope=True, dropout=0.1, name=f"aud_block_{i}")
            for i in range(n_layers)
        ]
        self.optimizer = SloAdam(lr=3e-4)

    def _mel_spectrogram(self, waveform: np.ndarray) -> np.ndarray:
        """Compute mel spectrogram from raw waveform. Returns (N_MELS, T).
        Vectorized STFT via strided frames + batch FFT."""
        n_fft = self.N_FFT
        hop = self.HOP_LENGTH
        window = np.hanning(n_fft)

        num_frames = (len(waveform) - n_fft) // hop + 1
        if num_frames <= 0:
            return np.zeros((self.N_MELS, 1), dtype=np.float32)

        # Vectorized frame extraction via strided view
        shape = (num_frames, n_fft)
        strides = (waveform.strides[0] * hop, waveform.strides[0])
        frames = np.lib.stride_tricks.as_strided(waveform, shape=shape, strides=strides)
        frames = frames * window  # (num_frames, n_fft)
        spec = np.abs(np.fft.rfft(frames, axis=1)).T  # (n_fft//2+1, num_frames)

        sr = self.SAMPLE_RATE
        n_mels = self.N_MELS
        # Vectorized mel filterbank
        f_max = sr / 2.0
        mel_pts = np.linspace(0, 2595.0 * np.log10(1 + f_max / 700.0), n_mels + 2)
        hz_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
        fft_bins = ((n_fft + 1) * hz_pts / sr).astype(int)
        mel_basis = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
        for i in range(n_mels):
            l, c, r = fft_bins[i], fft_bins[i+1], fft_bins[i+2]
            denom_l = max(c - l, 1)
            denom_r = max(r - c, 1)
            if r > l:
                idx = np.arange(l, r)
                vals = np.where(idx < c, (idx - l) / denom_l, (r - idx) / denom_r)
                mel_basis[i, l:r] = vals

        mel_spec = mel_basis @ spec
        mel_spec = np.log(np.maximum(mel_spec, 1e-8))
        return mel_spec

    def extract_patches(self, waveform_np: np.ndarray) -> np.ndarray:
        """Convert (B, T) audio to (B, num_patches, input_dim) patches."""
        if waveform_np.ndim == 1:
            waveform_np = waveform_np.reshape(1, -1)
        B = waveform_np.shape[0]
        patches_list = []
        for b in range(B):
            mel = self._mel_spectrogram(waveform_np[b])
            T = mel.shape[1]
            fp = self.frames_per_patch
            n = min(T // fp, self.max_patches)
            if n == 0:
                pad = fp - T
                mel = np.pad(mel, ((0, 0), (0, pad)), mode='constant')
                n = 1
            batch_patches = []
            for i in range(n):
                seg = mel[:, i*fp:(i+1)*fp].reshape(-1)
                batch_patches.append(seg)
            patches_list.append(np.stack(batch_patches))
        max_n = max(p.shape[0] for p in patches_list)
        out = np.zeros((B, max_n, self.input_dim), dtype=np.float32)
        for b, p in enumerate(patches_list):
            out[b, :p.shape[0]] = p
        return out

    def forward(self, waveform_np: np.ndarray) -> Tensor:
        """Audio -> patches -> transformer -> CLS token embedding. Returns (B, 1, embed_dim)."""
        patches = self.extract_patches(waveform_np)
        B, N = patches.shape[0], patches.shape[1]
        x = self.patch_proj.forward(_tensor(patches, requires_grad=False))
        cls_tokens_data = self.cls_token.data.repeat(B, axis=0)
        cls_tokens = Tensor(cls_tokens_data, requires_grad=True, _children=(self.cls_token,))
        x_data = np.concatenate([cls_tokens.data, x.data], axis=1)
        if N < self.max_patches:
            x_data = np.pad(x_data, ((0,0), (0, self.max_patches - N), (0,0)), mode='constant')
        x_data = x_data + self.pos_embed.data[:, :x_data.shape[1], :]
        x = Tensor(x_data, requires_grad=True, _children=(x, self.pos_embed, cls_tokens))
        for block in self.blocks:
            x, _ = block.forward(x)
        cls_out = x[:, 0:1, :]
        cls_out = self.norm.forward(cls_out)
        return cls_out

    def get_patch_embeddings(self, waveform_np: np.ndarray) -> Tensor:
        """Return all patch embeddings (B, num_patches+1, embed_dim) for cross-attention."""
        patches = self.extract_patches(waveform_np)
        return self._embed_patches(patches)

    def _embed_patches(self, patches: np.ndarray) -> Tensor:
        """Embed pre-extracted patches (B, N, input_dim) → (B, N+1, embed_dim).
        Can be called with precomputed patches to avoid recomputing STFT."""
        B, N = patches.shape[0], patches.shape[1]
        x = self.patch_proj.forward(_tensor(patches, requires_grad=False))
        cls_tokens_data = self.cls_token.data.repeat(B, axis=0)
        cls_tokens = Tensor(cls_tokens_data, requires_grad=True, _children=(self.cls_token,))
        x_data = np.concatenate([cls_tokens.data, x.data], axis=1)
        if N < self.max_patches:
            x_data = np.pad(x_data, ((0,0), (0, self.max_patches - N), (0,0)), mode='constant')
        x_data = x_data + self.pos_embed.data[:, :x_data.shape[1], :]
        x = Tensor(x_data, requires_grad=True, _children=(x, self.pos_embed, cls_tokens))
        for block in self.blocks:
            x, _ = block.forward(x)
        return self.norm.forward(x)

    def parameters(self):
        params = [self.cls_token, self.pos_embed]
        params += self.patch_proj.parameters()
        params += self.norm.parameters()
        for block in self.blocks:
            params += block.parameters()
        return [p for p in params if p.requires_grad]


class SloTransformerDecoderBlock(SloLayer):
    """Transformer decoder block with causal self-attention + cross-attention + FFN.

    Architecture:
        x → self_attn_norm → masked self-attention → + residual
          → cross_attn_norm → cross-attention (to image patches) → + residual
          → ff_norm → FFN → + residual

    All sub-layers use pre-norm (norm before each sub-layer).
    """

    def __init__(self, d_model: int, n_heads: int, dim_ff: int = None,
                 use_rope: bool = False, max_seq_len: int = 2048,
                 rope_base: float = 10000.0, dropout: float = 0.1, name=""):
        super().__init__(name or f"TransformerDecoder{d_model}")
        dim_ff = dim_ff or d_model * 4
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.self_attn_norm = SloRMSNorm(d_model, name=name + "_self_attn_norm")
        self.self_attn = SloMultiHeadAttention(
            d_model, n_heads, use_rope=use_rope,
            max_seq_len=max_seq_len, rope_base=rope_base,
            name=name + "_self_attn",
        )
        self.cross_attn_norm = SloRMSNorm(d_model, name=name + "_cross_attn_norm")
        self.cross_attn = SloCrossAttention(d_model, n_heads, name=name + "_cross_attn")
        self.ff_norm = SloRMSNorm(d_model, name=name + "_ff_norm")
        self.ff = SloFeedForward(d_model, dim_ff, name=name + "_ff")
        self.drop = SloDropout(dropout) if dropout > 0 else None

    def train(self, mode: bool = True):
        self.self_attn_norm.train(mode)
        self.self_attn.train(mode)
        self.cross_attn_norm.train(mode)
        self.cross_attn.train(mode)
        self.ff_norm.train(mode)
        self.ff.train(mode)
        if self.drop:
            self.drop.train(mode)

    def forward(self, x: Tensor, context: Optional[Tensor] = None,
                mask: Optional[Tensor] = None,
                kv_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                start_pos: int = 0) -> Tensor:
        """
        Args:
            x: (B, seq_len, d_model) input from previous layer
            context: (B, img_tokens, d_model) image patch embeddings for cross-attention
            mask: causal attention mask (B, 1, seq_len, seq_len) or None
            kv_cache: optional (K, V) cache tuple for self-attention
            start_pos: starting position in sequence (for incremental decoding)
        Returns:
            output: (B, seq_len, d_model)
        """
        h = self.self_attn_norm.forward(x)
        h, _ = self.self_attn.forward(h, h, h, mask, kv_cache=kv_cache, start_pos=start_pos)
        if self.drop:
            h = self.drop.forward(h)
        x = x + h

        if context is not None:
            h = self.cross_attn_norm.forward(x)
            h = self.cross_attn.forward(h, context)
            if self.drop:
                h = self.drop.forward(h)
            x = x + h

        h = self.ff_norm.forward(x)
        h = self.ff.forward(h)
        if self.drop:
            h = self.drop.forward(h)
        x = x + h
        return x

    def parameters(self) -> List[Tensor]:
        ps = self.self_attn_norm.parameters() + self.self_attn.parameters()
        ps += self.cross_attn_norm.parameters() + self.cross_attn.parameters()
        ps += self.ff_norm.parameters() + self.ff.parameters()
        if self.drop:
            ps += self.drop.parameters()
        return ps


def _causal_mask(seq_len: int, dtype=np.float32) -> Tensor:
    """Create a causal attention mask: upper triangular filled with -inf.

    Shape: (1, 1, seq_len, seq_len). Token i can only attend to j <= i.
    Broadcasts over batch and head dimensions.
    """
    mask = np.triu(np.full((seq_len, seq_len), -1e9, dtype=dtype), k=1)
    return _tensor(mask.reshape(1, 1, seq_len, seq_len), requires_grad=False)


class SloTransformerDecoder(SloLayer):
    """Transformer text decoder with cross-attention to image features.

    Architecture:
        token_ids → embedding → RoPE
          → N× SloTransformerDecoderBlock (self-attn → cross-attn → FFN)
          → output projection → logits

    Supports parallel training (teacher forcing with causal mask)
    and autoregressive generation (one token at a time with KV cache).
    """

    def __init__(self, vocab_size: int, embed_dim: int = 256, hidden_dim: int = 512,
                 n_heads: int = 8, n_layers: int = 4, max_seq_len: int = 512,
                 dropout: float = 0.1, name=""):
        super().__init__(name or f"TransformerDecoder{hidden_dim}")
        self.vocab_size = max(1, vocab_size)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.embedding = SloEmbedding(self.vocab_size, embed_dim)
        self.input_proj = SloLinear(embed_dim, hidden_dim, name=name + "_input_proj")
        self.blocks = [
            SloTransformerDecoderBlock(
                hidden_dim, n_heads,
                use_rope=True, max_seq_len=max_seq_len,
                dropout=dropout, name=name + f"_block_{i}",
            )
            for i in range(n_layers)
        ]
        self.output_norm = SloRMSNorm(hidden_dim, name=name + "_output_norm")
        self.fc_out = SloLinear(hidden_dim, self.vocab_size, name=name + "_fc_out")
        self.img_proj = SloLinear(embed_dim, hidden_dim, name=name + "_img_proj")
        self.optimizer = SloAdam(lr=3e-4)

    def parameters(self) -> List[Tensor]:
        ps = self.embedding.parameters()
        ps += self.input_proj.parameters()
        for block in self.blocks:
            ps += block.parameters()
        ps += self.output_norm.parameters()
        ps += self.fc_out.parameters()
        ps += self.img_proj.parameters()
        return [p for p in ps if p.requires_grad]

    def forward(self, img_embed: Tensor, token_ids: Tensor,
                img_patches: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Args:
            img_embed: (B, 1, embed_dim) cls token (used for KV cache init, not directly as state)
            token_ids: (B, seq_len) token IDs (teacher forcing in training)
            img_patches: (B, num_patches+1, embed_dim) full patch embeddings for cross-attention

        Returns:
            logits: (seq_len, vocab_size) output logits
            last_out: (B, hidden_dim) last hidden state
        """
        if token_ids.data.ndim == 1:
            token_ids_data = token_ids.data.reshape(1, -1)
        else:
            token_ids_data = token_ids.data

        B, seq_len = token_ids_data.shape

        # Embed tokens and project to hidden_dim
        tok_clipped = np.clip(token_ids_data, 0, self.vocab_size - 1).astype(np.int64)
        emb = self.embedding.forward(_tensor(tok_clipped, requires_grad=False))
        x = self.input_proj.forward(emb)

        # Project image patches from embed_dim to hidden_dim
        context = None
        if img_patches is not None:
            context = self.img_proj.forward(img_patches)

        # Create causal mask for parallel training
        mask = _causal_mask(seq_len)

        # Pass through decoder blocks
        for block in self.blocks:
            x = block.forward(x, context, mask)

        # Output projection — keep graph by NOT calling .data
        x = self.output_norm.forward(x)
        logits = self.fc_out.forward(x)  # (B, seq_len, vocab_size)
        last_out = x[:, -1:, :]  # (B, 1, hidden_dim), keep graph
        return logits, last_out


# =============================================================================
# Self-Supervised Components
# =============================================================================

class ReplayBuffer:
    """Stores past (image, caption) pairs for diverse multimodal training.

    Stores raw images (1, H, W, C) alongside captions so the full vision
    encoder + decoder pipeline can be trained on replay. When full, oldest
    entries are evicted. Sampling prioritizes captions that appear less
    frequently (diverse sampling).
    """

    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.images: list = []
        self.captions: list = []
        self._counts: dict = {}

    def add(self, image: np.ndarray, caption: str):
        if len(self.images) >= self.capacity:
            self.images.pop(0)
            removed_cap = self.captions.pop(0)
            old = self._counts.get(removed_cap, 1)
            if old > 1:
                self._counts[removed_cap] = old - 1
            else:
                self._counts.pop(removed_cap, None)
        self.images.append(image.copy())
        self.captions.append(caption)
        self._counts[caption] = self._counts.get(caption, 0) + 1

    def sample(self, n: int = 8) -> Tuple[List[np.ndarray], List[str]]:
        if len(self.images) < n:
            return self.images.copy(), self.captions.copy()
        total = len(self.captions)
        weights = np.array([1.0 / (self._counts.get(c, 1) + 1) for c in self.captions])
        weights /= weights.sum()
        idx = np.random.choice(total, size=n, p=weights, replace=False)
        return [self.images[i] for i in idx], [self.captions[i] for i in idx]

    @property
    def size(self):
        return len(self.images)


def augment_image(img_np: np.ndarray) -> np.ndarray:
    """Apply random augmentation to a single image (B, H, W, C).

    Accepts and returns VisionEncoder-compatible format (batch, height,
    width, channels). Augmentations: horizontal flip, random crop, color
    jitter. All pure NumPy — no external dependencies.
    """
    img = img_np.copy()
    _, h, w, _ = img.shape

    # Random horizontal flip (50% chance) — flip width axis
    if np.random.rand() < 0.5:
        img = img[:, :, ::-1, :]

    # Random crop with reflection padding
    if np.random.rand() < 0.5:
        pad = 4
        padded = np.pad(img, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='reflect')
        top = np.random.randint(0, 2 * pad + 1)
        left = np.random.randint(0, 2 * pad + 1)
        img = padded[:, top:top + h, left:left + w, :]

    # Color jitter
    if np.random.rand() < 0.8:
        brightness = 0.4 * (np.random.rand() - 0.5) * 2
        contrast = 0.4 * (np.random.rand() - 0.5) * 2 + 1.0
        img = img * contrast + brightness
        img = np.clip(img, 0.0, 1.0)

    return img


def contrastive_loss(z1: Tensor, z2: Tensor, negatives: List[Tensor], temperature: float = 0.5) -> Tensor:
    """NT-Xent-style contrastive loss between two views and negatives.

    Args:
        z1: embedding of first augmented view (1, D)
        z2: embedding of second augmented view (1, D)
        negatives: list of negative sample embeddings (1, D) each
        temperature: softmax temperature

    Returns:
        loss Tensor (scalar)
    """
    eps = 1e-8

    def _l2_norm(t: Tensor) -> Tensor:
        n = _tensor(np.sqrt((t.data ** 2).sum() + eps), requires_grad=False)
        return t / n

    z1_n = _l2_norm(z1)
    z2_n = _l2_norm(z2)

    # Positive similarity
    sim_pos = (z1_n * z2_n).sum() / temperature

    # Similarity with negatives  
    neg_sims = []
    for neg in negatives:
        neg_n = _l2_norm(neg)
        neg_sims.append((z1_n * neg_n).sum() / temperature)

    # InfoNCE: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
    all_sims = [sim_pos] + neg_sims
    max_val = max(s.data for s in all_sims)
    shifted = [s - max_val for s in all_sims]
    exp_sum = sum(np.exp(s.data) for s in shifted)
    loss_val = -shifted[0].data + np.log(exp_sum)
    loss = _tensor(loss_val, requires_grad=True)

    def bk(g):
        probs = np.array([np.exp(s.data - loss_val) for s in shifted])
        probs[0] -= 1.0
        grad_factor = g / temperature
        z1_n.grad = _tensor(probs[0] * z2_n.data * grad_factor * z1_n.data, requires_grad=False)
        z2_n.grad = _tensor(probs[0] * z1_n.data * grad_factor * z2_n.data, requires_grad=False)
        for i, neg in enumerate(negatives):
            if neg.requires_grad:
                neg.grad = _tensor(probs[i + 1] * z1_n.data * grad_factor * neg.data, requires_grad=False)

    loss._backward_fn = bk
    return loss


def contrastive_step(engine: MultimodalEngine, img_np: np.ndarray, buffer: ReplayBuffer) -> float:
    """Run one contrastive learning step on the vision encoder.

    Creates two augmented views, computes NT-Xent loss against
    negatives from the replay buffer, and updates vision encoder weights.

    Returns:
        loss value
    """
    if buffer.size < 2:
        return 0.0

    v1 = augment_image(img_np)
    v2 = augment_image(img_np)

    embed1 = engine.vision.forward(v1)
    embed2 = engine.vision.forward(v2)

    neg_imgs, _ = buffer.sample(min(buffer.size, 16))
    negatives = [engine.vision.forward(img) for img in neg_imgs]

    loss = contrastive_loss(embed1, embed2, negatives, temperature=0.5)
    loss.backward()
    engine._clip_gradients(engine.vision.parameters(), max_norm=1.0)
    engine.vision.optimizer.step(engine.vision.parameters())
    for p in engine.vision.parameters():
        p.grad = None

    return float(loss.data)


def replay_train_step(engine: MultimodalEngine, buffer: ReplayBuffer, batch_size: int = 4) -> float:
    """Train decoder + vision on a diverse sample from the replay buffer.

    Uses ``engine.train_step()`` which passes image patches for
    cross-attention — the transformer decoder's cross-attention layers
    learn to attend to image regions.

    Returns:
        average loss
    """
    if buffer.size < 2:
        return 0.0

    images, caps = buffer.sample(batch_size)
    total_loss = 0.0
    count = 0

    for img, cap in zip(images, caps):
        try:
            tokens = engine.text.encode(cap)
            if len(tokens) < 3:
                continue
            tokens_arr = np.array([tokens], dtype=np.int64)
            loss_val = engine.train_step(img, tokens_arr)
            total_loss += loss_val
            count += 1
        except Exception:
            continue

    if count > 0:
        return total_loss / count
    return 0.0


def get_multimodal_engine(embed_dim=256, hidden_dim=512, n_vit_layers=4, n_heads=8,
                          n_decoder_layers=4, n_audio_layers=2) -> MultimodalEngine:
    """Get a new multimodal engine."""
    return MultimodalEngine(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        n_vit_layers=n_vit_layers,
        n_heads=n_heads,
        n_decoder_layers=n_decoder_layers,
        n_audio_layers=n_audio_layers,
    )
