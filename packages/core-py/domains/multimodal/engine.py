"""
Multimodal Engine — Vision + Text understanding.

No external downloads. Everything learned from scratch.
"""

import os
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging
import numpy as np

logger = logging.getLogger("sloughgpt.multimodal.engine")

from domains.training.slonet import (
    Tensor, SloNet, SloConv2D, SloMaxPool2D, SloLinear,
    SloLSTM, SloEmbedding, SloLayerNorm, SloTransformerBlock, SloCrossAttention,
    SloAdam, softmax as _softmax, relu as _relu,
    flatten as _flatten, tensor as _tensor,
    cross_entropy as _cross_entropy, sigmoid as _sigmoid,
    zeros, ones,
)

from .bpe_tokenizer import BPETokenizer


@dataclass
class MultimodalOutput:
    text: str
    confidence: float


class TextDecoder:
    """Learns its own vocabulary and generates text from image embeddings."""

    def __init__(self, embed_dim=256, hidden_dim=512, vocab_size=4096):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.bpe = BPETokenizer(vocab_size=vocab_size)
        self.vocab: List[str] = []
        self.stoi: dict = {}
        self.itos: dict = {}

    def build_vocab(self, texts: List[str]):
        """Build vocabulary from training texts using BPE."""
        self.bpe.train(texts)
        # Also maintain word-level fallback for compatibility
        vocab_set = set()
        for t in texts:
            for w in t.lower().split():
                vocab_set.add(w)
        vocab_set.update(["<BOS>", "<EOS>", "<PAD>"])
        self.vocab = sorted(vocab_set)
        self.stoi = {w: i for i, w in enumerate(self.vocab)}
        self.itos = {i: w for w, i in self.stoi.items()}

    def encode(self, text: str) -> List[int]:
        """Encode text using BPE tokenizer."""
        if self.bpe._built:
            return self.bpe.encode(text)
        # Fallback to word-level encoding
        tokens = [self.stoi.get("<BOS>", 0)]
        for w in text.lower().split():
            tokens.append(self.stoi.get(w, self.stoi.get("<PAD>", 0)))
        tokens.append(self.stoi.get("<EOS>", 0))
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        if self.bpe._built:
            return self.bpe.decode(token_ids)
        # Fallback to word-level decoding
        words = []
        for tid in token_ids:
            w = self.itos.get(tid, "")
            if w in ("<BOS>", "<EOS>", "<PAD>", ""):
                continue
            words.append(w)
        return " ".join(words)


class MultimodalEngine:
    """Unified vision + text engine — learns freely from images.

    Implements the ModelProvider protocol (duck typing):
    - chat_stream / chat for image captioning
    - embed for image → vector
    """

    SAVE_PATH = "data/multimodal/multimodal_engine.npz"
    _model_id = "multimodal-v1"

    def __init__(self, embed_dim=256, hidden_dim=512, n_vit_layers=4, n_heads=8):
        self.vision = VisionEncoder(embed_dim, n_heads, n_vit_layers)
        self.text = TextDecoder(embed_dim, hidden_dim)
        self.decoder = DecoderLSTM(
            vocab_size=0,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_heads=max(1, n_heads // 2),
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
        if not self._trained:
            return [0.0] * 128
        for word in self.text.vocab:
            if word in text.lower():
                idx = self.text.stoi.get(word, 0)
                vec = [0.0] * 128
                vec[idx % 128] = 1.0
                return vec
        return [0.0] * 128

    @property
    def metadata(self):
        return {
            "vocab_size": len(self.text.vocab) if hasattr(self.text, "vocab") else 0,
            "trained": self._trained,
            "embed_dim": self.vision.embed_dim,
        }

    def build_vocab(self, texts: List[str]):
        self.text.build_vocab(texts)
        self.decoder = DecoderLSTM(
            vocab_size=len(self.text.vocab),
            embed_dim=self.text.embed_dim,
            hidden_dim=self.text.hidden_dim,
            n_heads=max(1, self.vision.n_heads // 2),
        )

    def save(self, path: str = "", extra_meta: dict = None) -> str:
        """Save engine state (vision + decoder weights + vocab) to .npz + JSON."""
        if not path:
            path = self.SAVE_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)

        weights = {}
        # Save vision transformer weights
        weights["vision_cls_token"] = self.vision.cls_token.data
        weights["vision_pos_embed"] = self.vision.pos_embed.data
        weights["vision_patch_proj_w"] = self.vision.patch_proj.weight.data
        weights["vision_patch_proj_b"] = self.vision.patch_proj.bias.data
        weights["vision_norm_w"] = self.vision.norm.weight.data
        weights["vision_norm_b"] = self.vision.norm.bias.data
        for i, block in enumerate(self.vision.blocks):
            for j, p in enumerate(block.parameters()):
                weights[f"vision_block{i}_{j}"] = p.data
        # Save decoder weights
        for i, p in enumerate(self.decoder.parameters()):
            weights[f"decoder_{i}"] = p.data
        np.savez_compressed(path, **weights)

        meta = {
            "vocab": self.text.vocab,
            "embed_dim": self.vision.embed_dim,
            "hidden_dim": self.decoder.hidden_dim,
            "n_vit_layers": len(self.vision.blocks),
            "n_heads": self.vision.n_heads,
            "trained": self._trained,
            "stoi": self.text.stoi,
            "itos": {str(k): v for k, v in self.text.itos.items()},
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

        n_vit_layers = meta.get("n_vit_layers", 4)
        n_heads = meta.get("n_heads", 8)
        engine = cls(
            embed_dim=meta["embed_dim"],
            hidden_dim=meta["hidden_dim"],
            n_vit_layers=n_vit_layers,
            n_heads=n_heads,
        )
        engine.text.vocab = list(meta["vocab"])
        engine.text.stoi = meta["stoi"]
        engine.text.itos = {int(k): v for k, v in meta["itos"].items()}
        engine.decoder = DecoderLSTM(
            vocab_size=len(engine.text.vocab),
            embed_dim=meta["embed_dim"],
            hidden_dim=meta["hidden_dim"],
            n_heads=max(1, n_heads // 2),
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

    def forward(self, images_np: np.ndarray, token_ids: np.ndarray) -> Tuple[Tensor, Tensor]:
        img_embed = self.vision.forward(images_np)
        logits, _ = self.decoder.forward(img_embed, token_ids)
        return logits, img_embed

    def train_step(
        self,
        images_np: np.ndarray,
        text_tokens: np.ndarray,
    ) -> float:
        # Get both cls token and full patch embeddings
        img_embed = self.vision.forward(images_np)  # (B, 1, embed_dim)
        img_patches = self.vision.get_patch_embeddings(images_np)  # (B, num_patches+1, embed_dim)
        logits, _ = self.decoder.forward(img_embed, text_tokens[:, :-1], img_patches)
        targets = _tensor(text_tokens[:, 1:].reshape(-1), requires_grad=False)
        loss = _cross_entropy(logits, targets)
        loss.backward()
        self.decoder.optimizer.step(self.decoder.parameters())
        self.decoder.optimizer.step(self.vision.parameters())
        for p in self.decoder.parameters() + self.vision.parameters():
            p.grad = None
        self._trained = True
        return float(loss.data)

    def generate(self, image_np: np.ndarray, max_len: int = 20, temperature: float = 1.0) -> MultimodalOutput:
        img_embed = self.vision.forward(image_np)  # (B, 1, embed_dim)
        img_patches = self.vision.get_patch_embeddings(image_np)  # (B, num_patches+1, embed_dim)
        bos = self.text.stoi.get("<BOS>", 0)
        tokens = [bos]

        for _ in range(max_len):
            inp = _tensor(np.array([tokens]), requires_grad=False)
            logits, _ = self.decoder.forward(img_embed, inp, img_patches)
            if temperature > 0 and self._trained:
                probs = _softmax(logits / temperature)
                probs_np = probs.data.flatten()
                probs_np = np.maximum(probs_np, 1e-8)
                # Repetition penalty: down-weight already generated tokens
                for t in tokens[1:]:
                    if 0 <= t < len(probs_np):
                        probs_np[t] *= 0.4
                probs_np /= probs_np.sum()
                next_tok = int(np.random.choice(len(probs_np), p=probs_np))
            else:
                scores = logits.data[-1].copy()
                for t in tokens[1:]:
                    if 0 <= t < len(scores):
                        scores[t] -= 5.0
                next_tok = int(np.argmax(scores))
            if next_tok == self.text.stoi.get("<EOS>", 0):
                break
            tokens.append(next_tok)

        text = self.text.decode(tokens)
        conf = float(np.mean(np.abs(img_embed.data)))
        return MultimodalOutput(text=text, confidence=conf)


class VisionEncoder:
    """ViT-style image encoder with patch positional embeddings.
    
    Processes 224x224 RGB images by splitting into 16x16 patches,
    projecting each patch to embed_dim, and adding positional embeddings.
    Output: (B, num_patches+1, embed_dim) with class token.
    """
    PATCH_SIZE = 16
    IMAGE_SIZE = 224

    def __init__(self, embed_dim=256, n_heads=8, n_layers=4):
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.patch_dim = 3 * self.PATCH_SIZE * self.PATCH_SIZE  # 3 * 16 * 16 = 768
        self.num_patches = (self.IMAGE_SIZE // self.PATCH_SIZE) ** 2  # 14*14 = 196
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


class DecoderLSTM:
    """LSTM text decoder with cross-attention to image features.
    
    Image patch embeddings are used as context for cross-attention at each timestep.
    The cls token initializes h0/c0 as before, but cross-attention allows the decoder
    to attend to specific image regions while generating each token.
    """

    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, n_heads=4):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.embedding = SloEmbedding(max(1, vocab_size), embed_dim)
        self.proj_h = SloLinear(embed_dim, hidden_dim)
        self.proj_c = SloLinear(embed_dim, hidden_dim)
        self.W_ih = SloLinear(embed_dim, 4 * hidden_dim)
        self.W_hh = SloLinear(hidden_dim, 4 * hidden_dim)
        # Cross-attention: query from hidden state, key/value from image patches
        # Need projection to match image embed_dim to hidden_dim
        self.img_proj = SloLinear(embed_dim, hidden_dim)
        self.cross_attn = SloCrossAttention(hidden_dim, n_heads, name="decoder_cross_attn")
        self.fc_out = SloLinear(hidden_dim, max(1, vocab_size))
        self.optimizer = SloAdam(lr=3e-4)
        self._trained = False

    def parameters(self):
        ps = self.embedding.parameters()
        ps += self.proj_h.parameters() + self.proj_c.parameters()
        ps += self.W_ih.parameters() + self.W_hh.parameters()
        ps += self.img_proj.parameters()
        ps += self.cross_attn.parameters()
        ps += self.fc_out.parameters()
        return [p for p in ps if p.requires_grad]

    def forward(self, img_embed: Tensor, token_ids: Tensor, img_patches: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Args:
            img_embed: (B, 1, embed_dim) cls token for h0/c0 init
            token_ids: (B, seq_len) token IDs
            img_patches: (B, num_patches+1, embed_dim) full patch embeddings for cross-attention
        """
        if token_ids.data.ndim == 1:
            token_ids_data = token_ids.data.reshape(1, -1)
        else:
            token_ids_data = token_ids.data

        # Project img_embed to hidden dim and reshape to (B, hidden_dim)
        h = self.proj_h.forward(img_embed)
        c = self.proj_c.forward(img_embed)
        if h.data.ndim == 3:
            h = Tensor(h.data.reshape(h.data.shape[0], -1), requires_grad=True, _children=(h,))
        if c.data.ndim == 3:
            c = Tensor(c.data.reshape(c.data.shape[0], -1), requires_grad=True, _children=(c,))
        all_logits = []

        for t in range(token_ids_data.shape[1]):
            tok_t = int(np.clip(token_ids_data[0, t], 0, max(1, self.vocab_size - 1)))
            idx = _tensor(np.array([[tok_t]]), requires_grad=False)
            emb_t = self.embedding.forward(idx)
            # Reshape embedding to (B, embed_dim)
            if emb_t.data.ndim == 3:
                emb_t = Tensor(emb_t.data.reshape(emb_t.data.shape[0], -1), requires_grad=False)

            gates_ih = self.W_ih.forward(emb_t)
            gates_hh = self.W_hh.forward(h)
            gates_data = gates_ih.data + gates_hh.data

            hd = self.hidden_dim
            gi = _sigmoid(_tensor(gates_data[:, :hd], requires_grad=False))
            gf = _sigmoid(_tensor(gates_data[:, hd:2*hd], requires_grad=False))
            gg = _tensor(np.tanh(gates_data[:, 2*hd:3*hd]), requires_grad=False)
            go = _sigmoid(_tensor(gates_data[:, 3*hd:], requires_grad=False))

            c_new = gf * c + gi * gg
            h = go * Tensor(np.tanh(c_new.data), requires_grad=False)
            c = c_new

            # Cross-attention if image patches provided
            if img_patches is not None:
                # Project image patches from embed_dim to hidden_dim
                img_ctx = self.img_proj.forward(img_patches)
                # Reshape h to (B, 1, hidden_dim) for cross-attention
                h_3d = Tensor(h.data.reshape(h.data.shape[0], 1, -1), requires_grad=True, _children=(h,))
                h_3d = self.cross_attn.forward(h_3d, img_ctx)
                # Back to (B, hidden_dim)
                h = Tensor(h_3d.data.reshape(h_3d.data.shape[0], -1), requires_grad=True, _children=(h_3d,))

            # Reshape h to (1, hidden_dim) for fc_out
            h_for_fc = h if h.data.ndim == 2 else Tensor(h.data.reshape(h.data.shape[0], -1), requires_grad=True, _children=(h,))
            all_logits.append(self.fc_out.forward(h_for_fc))

        logits_data = np.concatenate([l.data for l in all_logits], axis=0)
        logits = _tensor(logits_data, requires_grad=True)
        return logits, h


# =============================================================================
# Self-Supervised Components
# =============================================================================

class ReplayBuffer:
    """Stores past (embedding, caption) pairs for diverse decoder training.

    When full, oldest entries are evicted. Sampling prioritizes captions
    that appear less frequently (diverse sampling).
    """

    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.embeddings: list = []
        self.captions: list = []
        self._counts: dict = {}

    def add(self, embedding: np.ndarray, caption: str):
        if len(self.embeddings) >= self.capacity:
            removed_emb = self.embeddings.pop(0)
            removed_cap = self.captions.pop(0)
            old = self._counts.get(removed_cap, 1)
            if old > 1:
                self._counts[removed_cap] = old - 1
            else:
                self._counts.pop(removed_cap, None)
        self.embeddings.append(embedding.copy())
        self.captions.append(caption)
        self._counts[caption] = self._counts.get(caption, 0) + 1

    def sample(self, n: int = 8) -> Tuple[List[np.ndarray], List[str]]:
        if len(self.embeddings) < n:
            return self.embeddings.copy(), self.captions.copy()
        # Diversity weighting: rarer captions get higher weight
        total = len(self.captions)
        weights = np.array([1.0 / (self._counts.get(c, 1) + 1) for c in self.captions])
        weights /= weights.sum()
        idx = np.random.choice(total, size=n, p=weights, replace=False)
        return [self.embeddings[i] for i in idx], [self.captions[i] for i in idx]

    @property
    def size(self):
        return len(self.embeddings)


def augment_image(img_np: np.ndarray) -> np.ndarray:
    """Apply random augmentation to a single image (1, C, H, W).

    Augmentations: horizontal flip, random crop, color jitter.
    All pure NumPy — no external dependencies.
    """
    img = img_np.copy()
    _, _, h, w = img.shape

    # Random horizontal flip (50% chance)
    if np.random.rand() < 0.5:
        img = img[:, :, :, ::-1]

    # Random crop with reflection padding
    if np.random.rand() < 0.5:
        pad = 4
        padded = np.pad(img, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='reflect')
        top = np.random.randint(0, 2 * pad + 1)
        left = np.random.randint(0, 2 * pad + 1)
        img = padded[:, :, top:top + h, left:left + w]

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

    neg_embs, _ = buffer.sample(min(buffer.size, 16))
    negatives = [_tensor(e, requires_grad=False) for e in neg_embs]

    loss = contrastive_loss(embed1, embed2, negatives, temperature=0.5)
    loss.backward()
    engine.vision.optimizer.step(engine.vision.parameters())
    for p in engine.vision.parameters():
        p.grad = None

    return float(loss.data)


def replay_train_step(engine: MultimodalEngine, buffer: ReplayBuffer, batch_size: int = 4) -> float:
    """Train decoder on a diverse sample from the replay buffer.

    Returns:
        average loss
    """
    if buffer.size < 2:
        return 0.0

    embs, caps = buffer.sample(batch_size)
    total_loss = 0.0
    count = 0

    for emb, cap in zip(embs, caps):
        try:
            tokens = engine.text.encode(cap)
            if len(tokens) < 3:
                continue
            tokens_arr = np.array([tokens], dtype=np.int64)
            logits, _ = engine.forward(emb.reshape(1, 3, 32, 32), tokens_arr[:, :-1])
            targets = _tensor(tokens_arr[:, 1:].reshape(-1), requires_grad=False)
            loss = _cross_entropy(logits, targets)
            loss.backward()
            engine.decoder.optimizer.step(engine.decoder.parameters())
            for p in engine.decoder.parameters():
                p.grad = None
            total_loss += float(loss.data)
            count += 1
        except Exception:
            continue

    if count > 0:
        engine._trained = True
        return total_loss / count
    return 0.0


def get_multimodal_engine(embed_dim=256, hidden_dim=512, n_vit_layers=4, n_heads=8) -> MultimodalEngine:
    """Get a new multimodal engine."""
    return MultimodalEngine(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        n_vit_layers=n_vit_layers,
        n_heads=n_heads,
    )
