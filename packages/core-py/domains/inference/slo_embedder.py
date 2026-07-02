"""
SloNet-based text embedder — train on your own corpus, no downloads.

Uses the existing SloNet primitives (SloEmbedding, SloTransformerBlock,
SloLayerNorm, SloLinear) with contrastive learning to produce 384-dim
text embeddings.  Trains on knowledge files + chat history and saves
to a ``.sou`` checkpoint that the vector store loads automatically.

Architecture::

    input tokens
        ↓
    SloEmbedding (vocab_size → embed_dim)
        ↓
    + positional embedding (learned)
        ↓
    N × SloTransformerBlock (self-attention + FFN)
        ↓
    SloLayerNorm
        ↓
    mean-pool over sequence → single vector
        ↓
    SloLinear (embed_dim → embed_dim)  # projection head
        ↓
    L2-normalize → 384-dim unit vector
"""

from __future__ import annotations

import json
import logging
import math
import os
import struct
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_VOCAB_SIZE = 4096
DEFAULT_EMBED_DIM = 384
DEFAULT_MAX_SEQ_LEN = 128
DEFAULT_N_HEADS = 6
DEFAULT_N_LAYERS = 4
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 3e-4
DEFAULT_EPOCHS = 20
CONTRASTIVE_TEMPERATURE = 0.07

# Where the trained embedder lives
_EMBEDDER_DIR = Path(__file__).resolve().parents[4] / "data" / "models"
_EMBEDDER_PATH = _EMBEDDER_DIR / "text-embedder.sou"
_TOKENIZER_PATH = _EMBEDDER_DIR / "text-embedder-tokenizer.json"


# ---------------------------------------------------------------------------
# Simple whitespace tokenizer (fast, no BPE dependency for training pairs)
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "out", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "because", "and", "but",
    "or", "if", "while", "about", "up", "it", "its", "this", "that",
    "these", "those", "i", "me", "my", "we", "our", "you", "your", "he",
    "she", "they", "them", "their", "what", "which", "who", "whom",
})


def _tokenize_simple(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    import re
    return [t for t in re.findall(r"[a-z0-9']+", text.lower()) if t not in _STOPWORDS]


def _build_vocab(texts: List[str], vocab_size: int = DEFAULT_VOCAB_SIZE) -> Tuple[dict, dict]:
    """Build character-level vocab from texts."""
    from collections import Counter
    counts = Counter()
    for t in texts:
        counts.update(_tokenize_simple(t))

    # Reserve 0=PAD, 1=BOS, 2=EOS, 3=UNK
    vocab = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
    for word, _ in counts.most_common(vocab_size - len(vocab)):
        if word not in vocab:
            vocab[word] = len(vocab)
        if len(vocab) >= vocab_size:
            break

    itos = {i: w for w, i in vocab.items()}
    return vocab, itos


def _encode_tokens(text: str, vocab: dict, max_len: int) -> np.ndarray:
    """Encode text to padded token ID array."""
    tokens = _tokenize_simple(text)
    ids = [vocab.get(t, 3) for t in tokens]  # 3 = UNK
    ids = ids[:max_len]
    # Pad
    padded = np.zeros(max_len, dtype=np.int64)
    padded[: len(ids)] = ids
    return padded


# ---------------------------------------------------------------------------
# SloNet text encoder (lightweight — no external deps)
# ---------------------------------------------------------------------------

def _build_encoder(
    vocab_size: int,
    embed_dim: int,
    max_seq_len: int,
    n_heads: int,
    n_layers: int,
):
    """Build a SloNet text encoder from existing primitives."""
    from domains.training.slonet import (
        SloEmbedding,
        SloTransformerBlock,
        SloLayerNorm,
        SloLinear,
        Tensor,
    )

    class _Encoder:
        """Minimal encoder that mirrors TextEncoder interface."""

        def __init__(self):
            self.tok_emb = SloEmbedding(vocab_size, embed_dim, "tok_emb")
            self.pos_emb = Tensor(
                np.random.randn(1, max_seq_len, embed_dim).astype(np.float32) * 0.02,
                requires_grad=True,
            )
            self.blocks = [
                SloTransformerBlock(
                    embed_dim, n_heads, use_rope=True, dropout=0.1,
                    name=f"emb_block_{i}",
                )
                for i in range(n_layers)
            ]
            self.norm = SloLayerNorm(embed_dim)
            self.proj = SloLinear(embed_dim, embed_dim)

        def forward(self, token_ids: np.ndarray):
            """token_ids: (B, seq_len) → (B, embed_dim)"""
            from domains.training.slonet import Tensor as _T, tensor as _tensor

            B, S = token_ids.shape
            tok = self.tok_emb.forward(_tensor(token_ids, requires_grad=False))
            pos = self.pos_emb.data[:, :S, :]
            x_data = tok.data + pos
            x = _T(x_data, requires_grad=True, _children=(tok, self.pos_emb))

            for block in self.blocks:
                x, _ = block.forward(x)

            x = self.norm.forward(x)
            # Mean-pool: (B, S, D) → (B, D)
            pooled = x.data.mean(axis=1)
            # Project
            out = self.proj.forward(_T(pooled, requires_grad=True))
            return out

        def parameters(self):
            from domains.training.slonet import SloLayer
            ps = []
            for attr in [self.tok_emb, self.pos_emb, self.norm, self.proj]:
                if hasattr(attr, "parameters"):
                    ps.extend(attr.parameters())
                elif hasattr(attr, "requires_grad") and attr.requires_grad:
                    ps.append(attr)
            for block in self.blocks:
                if hasattr(block, "parameters"):
                    ps.extend(block.parameters())
            return ps

    return _Encoder()


# ---------------------------------------------------------------------------
# Contrastive training
# ---------------------------------------------------------------------------

def _augment_text(text: str, rng: np.random.RandomState) -> str:
    """Simple text augmentation for contrastive pairs."""
    tokens = _tokenize_simple(text)
    if not tokens:
        return text

    method = rng.choice(["drop", "shuffle", "crop"])

    if method == "drop" and len(tokens) > 3:
        # Random token dropout
        keep = [i for i in range(len(tokens)) if rng.random() > 0.15]
        if keep:
            tokens = [tokens[i] for i in keep]

    elif method == "shuffle" and len(tokens) > 3:
        # Slightly shuffle word order
        idx = list(range(len(tokens)))
        rng.shuffle(idx[: max(3, len(idx) // 2)])
        tokens = [tokens[i] for i in idx]

    elif method == "crop" and len(tokens) > 4:
        # Random crop
        start = rng.randint(0, max(1, len(tokens) // 3))
        end = rng.randint(start + 2, len(tokens))
        tokens = tokens[start:end]

    return " ".join(tokens)


def _contrastive_loss(
    z_i: np.ndarray,
    z_j: np.ndarray,
    temperature: float = CONTRASTIVE_TEMPERATURE,
) -> float:
    """InfoNCE loss for a batch of positive pairs.

    z_i, z_j: (B, D) — L2-normalized embeddings
    """
    B = z_i.shape[0]
    # Cosine similarity matrix (already L2-normalized → dot product)
    sim = z_i @ z_j.T  # (B, B)
    sim = sim / temperature

    # Labels: diagonal is positive pair
    labels = np.arange(B)

    # Numerator: positive pair similarity
    logits_max = sim.max(axis=1, keepdims=True)
    logits = sim - logits_max  # numerical stability
    exp_logits = np.exp(logits)
    log_sum_exp = np.log(exp_logits.sum(axis=1) + 1e-10)
    log_probs = logits - log_sum_exp[:, None]
    loss = -log_probs[labels, labels].mean()
    return float(loss)


def train_embedder(
    texts: List[str],
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    embed_dim: int = DEFAULT_EMBED_DIM,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    n_heads: int = DEFAULT_N_HEADS,
    n_layers: int = DEFAULT_N_LAYERS,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    save_path: Optional[str] = None,
    progress_callback=None,
) -> dict:
    """Train a text embedder on a corpus using contrastive learning.

    Args:
        texts: list of text strings to train on
        vocab_size: maximum vocabulary size
        embed_dim: embedding dimension (384 for cosine-sim compat with InMemoryVectorStore)
        max_seq_len: maximum token sequence length
        n_heads: attention heads
        n_layers: transformer layers
        epochs: training epochs
        lr: learning rate
        batch_size: mini-batch size
        save_path: where to save the .sou checkpoint
        progress_callback: optional callable(epoch, loss, total_epochs)

    Returns:
        dict with training stats
    """
    from domains.training.slonet import SloAdam, Tensor, tensor as _tensor

    if len(texts) < 2:
        raise ValueError("Need at least 2 text samples for contrastive training")

    rng = np.random.RandomState(42)

    # 1. Build vocab
    logger.info("Building vocab from %d texts", len(texts))
    vocab, itos = _build_vocab(texts, vocab_size)
    actual_vocab = len(vocab)
    logger.info("Vocab size: %d", actual_vocab)

    # 2. Build encoder
    encoder = _build_encoder(actual_vocab, embed_dim, max_seq_len, n_heads, n_layers)
    optimizer = SloAdam(lr=lr)
    params = encoder.parameters()
    logger.info("Encoder params: %d tensors", len(params))

    # 3. Training loop
    logger.info("Training embedder: %d texts, %d epochs, batch_size=%d", len(texts), epochs, batch_size)
    rng_train = np.random.RandomState(123)
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        # Shuffle indices
        indices = rng_train.permutation(len(texts))

        for start in range(0, len(texts), batch_size):
            batch_idx = indices[start: start + batch_size]
            if len(batch_idx) < 2:
                continue

            # Build positive pairs: original + augmented
            orig_texts = [texts[i] for i in batch_idx]
            aug_texts = [_augment_text(t, rng_train) for t in orig_texts]

            # Encode both views
            orig_ids = np.stack([_encode_tokens(t, vocab, max_seq_len) for t in orig_texts])
            aug_ids = np.stack([_encode_tokens(t, vocab, max_seq_len) for t in aug_texts])

            orig_emb = encoder.forward(orig_ids)
            aug_emb = encoder.forward(aug_ids)

            # L2-normalize
            orig_norm = orig_emb.data / (np.linalg.norm(orig_emb.data, axis=1, keepdims=True) + 1e-10)
            aug_norm = aug_emb.data / (np.linalg.norm(aug_emb.data, axis=1, keepdims=True) + 1e-10)

            # Compute contrastive loss
            loss_val = _contrastive_loss(orig_norm, aug_norm)

            # Backward pass (manual gradient computation for contrastive loss)
            B = len(batch_idx)
            D = embed_dim
            temperature = CONTRASTIVE_TEMPERATURE

            # Gradient of InfoNCE w.r.t. orig_norm
            sim = orig_norm @ aug_norm.T / temperature
            sim_max = sim.max(axis=1, keepdims=True)
            sim_shifted = sim - sim_max
            exp_sim = np.exp(sim_shifted)
            sum_exp = exp_sim.sum(axis=1, keepdims=True)
            probs = exp_sim / sum_exp  # (B, B)

            # d(loss)/d(orig_norm) = (1/B) * sum_j [ (probs[j,j] - 1_{i=j}) * aug_norm[j] / temp ]
            labels = np.arange(B)
            grad_orig = np.zeros_like(orig_norm)
            for i in range(B):
                for j in range(B):
                    coeff = probs[i, j] - (1.0 if i == j else 0.0)
                    grad_orig[i] += coeff * aug_norm[j]
                grad_orig[i] /= (B * temperature)

            # Backprop through encoder (simplified: update last projection + norm)
            # For full backprop we'd need the full autograd graph; here we
            # approximate by updating the projection head directly
            proj = encoder.proj
            if hasattr(proj, "weight") and hasattr(proj.weight, "grad"):
                # Approximate gradient for projection layer
                grad_proj = grad_orig.T @ orig_emb.data  # (D, D)
                if proj.weight.grad is None:
                    proj.weight.grad = Tensor(grad_proj)
                else:
                    proj.weight.grad.data += grad_proj

            epoch_loss += loss_val
            n_batches += 1

            # Step optimizer (SloAdam.step sets p.grad=None internally)
            for p in params:
                if p.grad is not None:
                    p.grad.data *= 0.01  # scale down to prevent explosion
            optimizer.step(params)

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if progress_callback:
            progress_callback(epoch + 1, avg_loss, epochs)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info("Epoch %d/%d — loss: %.4f", epoch + 1, epochs, avg_loss)

    # 4. Save checkpoint
    out_path = save_path or str(_EMBEDDER_PATH)
    _save_checkpoint(out_path, encoder, vocab, itos, embed_dim, max_seq_len, n_heads, n_layers)
    logger.info("Saved embedder to %s", out_path)

    return {
        "epochs": epochs,
        "final_loss": losses[-1] if losses else 0.0,
        "vocab_size": actual_vocab,
        "embed_dim": embed_dim,
        "n_params": sum(p.data.size for p in params),
        "save_path": out_path,
    }


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def _save_checkpoint(
    path: str,
    encoder,
    vocab: dict,
    itos: dict,
    embed_dim: int,
    max_seq_len: int,
    n_heads: int,
    n_layers: int,
):
    """Save embedder as .sou checkpoint with vocab sidecar."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Save vocab as JSON sidecar
    vocab_path = path.replace(".sou", "-vocab.json")
    with open(vocab_path, "w") as f:
        json.dump({"vocab": vocab, "itos": {str(k): v for k, v in itos.items()}}, f)

    # Save as .sou (binary format compatible with import_from_sou)
    from domains.training.slonet import export_to_sou, SloNet

    net = SloNet(
        soul_name="text-embedder",
        soul_traits={"warmth": 0.5, "creativity": 0.5, "curiosity": 0.5, "confidence": 0.5},
        system_prompt=f"embed_dim={embed_dim} max_seq_len={max_seq_len} n_heads={n_heads} n_layers={n_layers}",
        lineage="slonet-embedder",
    )
    net.layers = encoder.blocks  # so export_to_sou can find parameters
    net._parameters_cache = encoder.parameters()

    # Manual save: just dump all parameter arrays
    import tempfile
    meta = {
        "version": 3,
        "soul_name": "text-embedder",
        "lineage": "slonet-embedder",
        "system_prompt": f"embed_dim={embed_dim} max_seq_len={max_seq_len} n_heads={n_heads} n_layers={n_layers}",
        "metadata": {"embed_dim": embed_dim, "max_seq_len": max_seq_len},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    json_bytes = json.dumps(meta, allow_nan=False).encode()

    from domains.training.slonet import SOU_MAGIC
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(path) or ".", suffix=".tmp",
    )
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(SOU_MAGIC)
            f.write(struct.pack("<I", 3))
            f.write(struct.pack("<I", len(json_bytes)))
            f.write(json_bytes)

            params = [(f"p{i}", np.asarray(p.data, dtype=np.float32))
                      for i, p in enumerate(encoder.parameters())]
            f.write(struct.pack("<I", len(params)))
            for key, arr in params:
                name_bytes = key.encode()
                f.write(struct.pack("<I", len(name_bytes)))
                f.write(name_bytes)
                f.write(struct.pack("<I", arr.ndim))
                for dim in arr.shape:
                    f.write(struct.pack("<I", dim))
                f.write(arr.tobytes())

        os.rename(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


class SloTextEmbedder:
    """Trained SloNet text embedder for vector store integration.

    Usage::

        embedder = SloTextEmbedder.load()
        vec = embedder.embed("neural network training")
        # → List[float] of length 384
    """

    def __init__(
        self,
        encoder,
        vocab: dict,
        embed_dim: int = DEFAULT_EMBED_DIM,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    ):
        self.encoder = encoder
        self.vocab = vocab
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

    @classmethod
    def load(cls, path: Optional[str] = None) -> Optional["SloTextEmbedder"]:
        """Load a trained embedder from disk. Returns None if not found."""
        path = path or str(_EMBEDDER_PATH)
        if not os.path.exists(path):
            return None

        try:
            from domains.training.slonet import import_from_sou

            # Load metadata to get architecture params
            with open(path, "rb") as f:
                raw = f.read()

            from domains.training.slonet import SOU_MAGIC
            if raw[:4] != SOU_MAGIC:
                return None

            version = struct.unpack("<I", raw[4:8])[0]
            json_len = struct.unpack("<I", raw[8:12])[0]
            meta_bytes = raw[12:12 + json_len].rstrip(b"\x00")
            meta = json.loads(meta_bytes.decode())

            system_prompt = meta.get("system_prompt", "")
            params = {}
            for part in system_prompt.split():
                if "=" in part:
                    k, v = part.split("=", 1)
                    params[k] = int(v)

            embed_dim = params.get("embed_dim", DEFAULT_EMBED_DIM)
            max_seq_len = params.get("max_seq_len", DEFAULT_MAX_SEQ_LEN)
            n_heads = params.get("n_heads", DEFAULT_N_HEADS)
            n_layers = params.get("n_layers", DEFAULT_N_LAYERS)

            # Load vocab
            vocab_path = path.replace(".sou", "-vocab.json")
            if os.path.exists(vocab_path):
                with open(vocab_path) as f:
                    vdata = json.load(f)
                vocab = vdata["vocab"]
            else:
                vocab = {}

            actual_vocab = max(vocab.values()) + 1 if vocab else DEFAULT_VOCAB_SIZE

            # Build encoder
            encoder = _build_encoder(actual_vocab, embed_dim, max_seq_len, n_heads, n_layers)

            # Load weights
            weight_offset = 12 + json_len
            rem = raw[weight_offset:]
            if len(rem) >= 4:
                num_params = struct.unpack("<I", rem[:4])[0]
                pos = 4
                param_idx = 0
                for _ in range(num_params):
                    name_len = struct.unpack("<I", rem[pos:pos + 4])[0]
                    pos += 4
                    pos += name_len  # skip name
                    ndim = struct.unpack("<I", rem[pos:pos + 4])[0]
                    pos += 4
                    shape = tuple(
                        struct.unpack("<I", rem[pos + 4 * i:pos + 4 * i + 4])[0]
                        for i in range(ndim)
                    )
                    pos += 4 * ndim
                    count = int(np.prod(shape))
                    arr = np.frombuffer(rem[pos:pos + count * 4], dtype=np.float32).copy().reshape(shape)
                    pos += count * 4

                    # Load into encoder parameters
                    enc_params = encoder.parameters()
                    if param_idx < len(enc_params):
                        if enc_params[param_idx].data.shape == arr.shape:
                            enc_params[param_idx].data = arr
                    param_idx += 1

            logger.info("Loaded SloTextEmbedder from %s (embed_dim=%d)", path, embed_dim)
            return cls(encoder, vocab, embed_dim, max_seq_len)

        except Exception as e:
            logger.warning("Failed to load SloTextEmbedder: %s", e)
            return None

    def embed(self, text: str) -> List[float]:
        """Encode text to a 384-dim L2-normalized vector.

        Args:
            text: input text string

        Returns:
            list of floats (L2-normalized)
        """
        ids = _encode_tokens(text, self.vocab, self.max_seq_len)
        ids = ids[np.newaxis, :]  # (1, max_seq_len)

        emb = self.encoder.forward(ids)  # (1, embed_dim)
        vec = emb.data.squeeze(0)

        # L2-normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        # Pad/truncate to 384 for InMemoryVectorStore compatibility
        if len(vec) < DEFAULT_EMBED_DIM:
            vec = np.pad(vec, (0, DEFAULT_EMBED_DIM - len(vec)))
        elif len(vec) > DEFAULT_EMBED_DIM:
            vec = vec[:DEFAULT_EMBED_DIM]
            n = np.linalg.norm(vec)
            if n > 0:
                vec = vec / n

        return vec.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode a batch of texts."""
        return [self.embed(t) for t in texts]
