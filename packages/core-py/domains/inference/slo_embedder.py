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

DEFAULT_VOCAB_SIZE = 2048
DEFAULT_EMBED_DIM = 128
DEFAULT_MAX_SEQ_LEN = 64
DEFAULT_N_HEADS = 4
DEFAULT_N_LAYERS = 2
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-4
DEFAULT_EPOCHS = 15
CONTRASTIVE_TEMPERATURE = 0.07
LSE_THRESHOLD = 15.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from contextlib import contextmanager

@contextmanager
def _no_accel():
    """Context manager to temporarily disable Metal/GPU accelerator."""
    import domains.training.slonet as _slonet
    prev = getattr(_slonet, "_ACCELERATOR", None)
    try:
        _slonet._ACCELERATOR = "none"
        yield
    finally:
        _slonet._ACCELERATOR = prev

# Where the trained embedder lives
_EMBEDDER_DIR = Path(__file__).resolve().parents[4] / "data" / "models"
_EMBEDDER_PATH = _EMBEDDER_DIR / "text-embedder.sou"
_TOKENIZER_PATH = _EMBEDDER_DIR / "text-embedder-tokenizer.json"


# ---------------------------------------------------------------------------
# Binary log-sum-exp tree — no flat sum of exponentials
# ---------------------------------------------------------------------------

def _lse_pair(a, b, coeff=1.0, threshold=LSE_THRESHOLD):
    """Binary log-sum-exp pair: log(coeff*exp(a) + exp(b)).

    coeff=1 → standard LSE (expands)
    coeff=0 → returns max(a, b) - |diff| = min(a, b) (contracts)

    The threshold prevents overflow: when |a - b| > threshold,
    the smaller term is negligible and we return max directly.
    """
    mx = np.maximum(a, b)
    diff = np.abs(a - b)
    mask = diff <= threshold
    adj = np.where(mask, np.log(coeff + np.exp(-diff)), 0.0)
    return mx + adj


def _lse_tree(x, axis=-1, threshold=LSE_THRESHOLD):
    """Log-sum-exp via binary pairing tree with threshold overflow prevention.

    Every node uses coeff=1 (standard LSE). The threshold prevents
    overflow: when |a - b| > threshold, the smaller term is negligible
    and we return max directly.

    This is mathematically equivalent to flat LSE for values within
    threshold, and prevents spillover for extreme values.
    """
    N = x.shape[axis]
    if N == 1:
        return x.squeeze(axis=axis)
    if N == 2:
        return _lse_pair(x[..., 0], x[..., 1], coeff=1.0, threshold=threshold)

    pairs = []
    for i in range(0, N, 2):
        a = np.take(x, i, axis=axis)
        if i + 1 < N:
            b = np.take(x, i + 1, axis=axis)
            pairs.append(_lse_pair(a, b, coeff=1.0, threshold=threshold))
        else:
            pairs.append(a)
    return _lse_tree(np.stack(pairs, axis=axis), axis=axis, threshold=threshold)


# ---------------------------------------------------------------------------
# Tokenizer — BPE with whitespace fallback
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


def _build_bpe_tokenizer(texts: List[str], vocab_size: int = 2048):
    """Train a BPE tokenizer on the corpus and return (bpe, encode_fn)."""
    try:
        from domains.multimodal.bpe_tokenizer import BPETokenizer
        bpe = BPETokenizer(vocab_size=vocab_size)
        bpe.train(texts)
        def encode_fn(text: str, max_len: int) -> np.ndarray:
            ids = bpe.encode(text)
            ids = ids[:max_len]
            padded = np.zeros(max_len, dtype=np.int64)
            padded[:len(ids)] = ids
            return padded
        return bpe, encode_fn
    except Exception as e:
        logger.warning("BPE tokenizer failed (%s), falling back to whitespace", e, extra={"tag": "INFRA"})
        return None, None


def _build_vocab(texts: List[str], vocab_size: int = DEFAULT_VOCAB_SIZE) -> Tuple[dict, dict]:
    """Build vocabulary from texts (used by whitespace fallback)."""
    from collections import Counter
    counts = Counter()
    for t in texts:
        counts.update(_tokenize_simple(t))

    vocab = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
    for word, _ in counts.most_common(vocab_size - len(vocab)):
        if word not in vocab:
            vocab[word] = len(vocab)
        if len(vocab) >= vocab_size:
            break

    itos = {i: w for w, i in vocab.items()}
    return vocab, itos


def _encode_tokens(text: str, vocab: dict, max_len: int) -> np.ndarray:
    """Encode text to padded token ID array (whitespace fallback)."""
    tokens = _tokenize_simple(text)
    ids = [vocab.get(t, 3) for t in tokens]  # 3 = UNK
    ids = ids[:max_len]
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
    point_labels: Optional[List[str]] = None,
    meaning_tags=None,
    constraint_weight: float = 0.5,
) -> Tuple[float, float]:
    """InfoNCE loss with meaning tag constraints.

    z_i, z_j: (B, D) — L2-normalized embeddings
    point_labels: optional list of meaning tag labels per text (e.g., ["factual", "interrogative"])
    meaning_tags: MeaningTags instance with fixed reference vectors
    constraint_weight: how much to weight the meaning tag constraint (0.0 = ignore)

    Returns (total_loss, constraint_loss) — constraint_loss is 0 if no labels provided.
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
    log_sum_exp = _lse_tree(logits, axis=1)
    log_probs = logits - log_sum_exp[:, None]
    info_nce_loss = -log_probs[labels, labels].mean()

    # Hard constraint: penalize distance from known meaning point
    constraint_loss = 0.0
    if point_labels and meaning_tags is not None:
        penalties = []
        for i, label in enumerate(point_labels):
            if label and meaning_tags.get(label) is not None:
                tag_vec = meaning_tags.get(label)
                # Cosine similarity to meaning tag — should be high
                sim_to_tag = float(np.dot(z_i[i], tag_vec))
                # Penalty: 1 - sim (zero when perfectly aligned)
                penalties.append(1.0 - sim_to_tag)
        if penalties:
            constraint_loss = float(np.mean(penalties))

    total = info_nce_loss + constraint_weight * constraint_loss
    return float(total), constraint_loss


def _label_by_meaning(text: str, points_store=None) -> Optional[str]:
    """Label text by nearest meaning point in embedding space.

    Uses n-gram TF-IDF embed (always available, no download) to position
    text relative to fixed semantic meaning points. The label is the meaning
    point whose region the text falls into.

    This bootstraps without requiring a trained embedder. As training
    progresses the embedding space aligns such that texts semantically
    near each other land near the same meaning point.
    """
    from domains.inference.vector_store import simple_embed
    if points_store is None:
        return None
    vec = simple_embed(text, dimension=points_store.dimension)
    if not vec or all(v == 0.0 for v in vec):
        return None
    return points_store.classify(vec)


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

    # 1. Build tokenizer — prefer BPE, fall back to whitespace
    logger.info("Building tokenizer from %d texts", len(texts), extra={"tag": "INFRA"})
    bpe, encode_fn = _build_bpe_tokenizer(texts, vocab_size)
    if bpe is not None:
        logger.info("Using BPE tokenizer (vocab=%d)", len(bpe.vocab), extra={"tag": "INFRA"})
        actual_vocab = len(bpe.vocab)
        vocab = bpe.vocab
        itos = bpe.itos
    else:
        vocab, itos = _build_vocab(texts, vocab_size)
        actual_vocab = len(vocab)
        encode_fn = None
        logger.info("Using whitespace tokenizer (vocab=%d)", actual_vocab, extra={"tag": "INFRA"})

    # 2. Build encoder
    encoder = _build_encoder(actual_vocab, embed_dim, max_seq_len, n_heads, n_layers)
    optimizer = SloAdam(lr=lr)
    params = encoder.parameters()
    logger.info("Encoder params: %d tensors", len(params), extra={"tag": "INFRA"})

    # 2b. Load meaning tags (the stars — fixed semantic reference points)
    from domains.infrastructure.anchor_store import MeaningTags, get_default_meaning_tags
    meaning_tags = get_default_meaning_tags(dimension=embed_dim)
    logger.info("Loaded %d meaning tags: %s", len(meaning_tags.names()), meaning_tags.names(), extra={"tag": "INFRA"})

    # 3. Training loop
    logger.info("Training embedder: %d texts, %d epochs, batch_size=%d", len(texts), epochs, batch_size, extra={"tag": "INFRA"})
    rng_train = np.random.RandomState(123)
    losses = []
    refine_stats = []
    maintain_stats = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_constraint = 0.0
        n_batches = 0

        # Shuffle indices
        indices = rng_train.permutation(len(texts))

        for start in range(0, len(texts), batch_size):
            batch_idx = indices[start: start + batch_size]
            if len(batch_idx) < 2:
                continue

            # Zero gradients
            for p in params:
                p.grad = None

            # Build positive pairs: original + augmented
            orig_texts = [texts[i] for i in batch_idx]
            aug_texts = [_augment_text(t, rng_train) for t in orig_texts]

            # Encode both views — use BPE if available, else whitespace
            if encode_fn is not None:
                orig_ids = np.stack([encode_fn(t, max_seq_len) for t in orig_texts])
                aug_ids = np.stack([encode_fn(t, max_seq_len) for t in aug_texts])
            else:
                orig_ids = np.stack([_encode_tokens(t, vocab, max_seq_len) for t in orig_texts])
                aug_ids = np.stack([_encode_tokens(t, vocab, max_seq_len) for t in aug_texts])

            orig_emb = encoder.forward(orig_ids)
            aug_emb = encoder.forward(aug_ids)

            # L2-normalize
            orig_norm = orig_emb.data / (np.linalg.norm(orig_emb.data, axis=1, keepdims=True) + 1e-10)
            aug_norm = aug_emb.data / (np.linalg.norm(aug_emb.data, axis=1, keepdims=True) + 1e-10)

            # Label by meaning (nearest meaning tag in embedding space)
            batch_labels = [_label_by_meaning(t, meaning_tags) for t in orig_texts]

            # Compute contrastive loss with meaning tag constraints
            loss_val, constraint_loss = _contrastive_loss(
                orig_norm, aug_norm,
                point_labels=batch_labels,
                meaning_tags=meaning_tags,
                constraint_weight=0.5,
            )

            # Vectorized InfoNCE gradient w.r.t. orig_norm
            B = len(batch_idx)
            temperature = CONTRASTIVE_TEMPERATURE
            logits = orig_norm @ aug_norm.T / temperature
            log_sum_exp_tree = _lse_tree(logits, axis=1)
            probs = np.exp(logits - log_sum_exp_tree[:, np.newaxis])  # (B, B)

            # d(loss)/d(orig_norm) — fully vectorized, no Python loops
            eye = np.eye(B)
            coeff = (probs - eye) / (B * temperature)  # (B, B)
            grad_orig_norm = coeff @ aug_norm  # (B, D)

            # Backprop through L2 norm: d(L2)/d(orig_emb)
            norms = np.linalg.norm(orig_emb.data, axis=1, keepdims=True) + 1e-10
            grad_orig_emb = (grad_orig_norm / norms) - (orig_emb.data * (grad_orig_norm * orig_emb.data).sum(axis=1, keepdims=True) / (norms ** 3))

            # Set gradient on encoder output and backward through SloNet autograd
            orig_emb.grad = Tensor(grad_orig_emb)
            orig_emb.backward()

            epoch_loss += loss_val
            epoch_constraint += constraint_loss
            n_batches += 1

            # Clip gradients to prevent explosion
            for p in params:
                if p.grad is not None:
                    g = p.grad.data
                    norm = np.linalg.norm(g)
                    if norm > 1.0:
                        p.grad.data = g / norm

            # Step optimizer (SloAdam.step sets p.grad=None internally)
            optimizer.step(params)

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_constraint = epoch_constraint / max(n_batches, 1)
        losses.append(avg_loss)

        # --- Post-epoch self-correction ---

        # Step A: Refine meaning tags toward centroids
        with _no_accel():
            all_ids = np.stack([_encode_tokens(t, vocab, max_seq_len) for t in texts])
            all_emb = encoder.forward(all_ids)
            all_norm = all_emb.data / (np.linalg.norm(all_emb.data, axis=1, keepdims=True) + 1e-10)

        refined = meaning_tags.refine(texts, all_norm, lr=0.1, min_samples=max(3, len(texts) // 10))
        if refined:
            refine_stats.append({"epoch": epoch + 1, "refined": refined})
            logger.info("Epoch %d refine: %s", epoch + 1, {k: v for k, v in refined.items()}, extra={"tag": "INFRA"})

        # Step B: Correct misclassified texts via TruthMaintainer
        from domains.infrastructure.truth_maintainer import get_truth_maintainer
        maintainer = get_truth_maintainer()
        misclassified = maintainer.find_misclassified(texts, all_norm, meaning_tags)
        if misclassified and len(misclassified) >= 3:
            queries, positives, negatives = maintainer.generate_corrective_pairs(
                misclassified, texts, all_norm, meaning_tags, max_pairs=min(30, len(misclassified))
            )
            if queries:
                corr_loss = maintainer.apply_correction(
                    encoder, queries, positives, negatives,
                    meaning_tags, vocab, encode_fn=encode_fn,
                    max_seq_len=max_seq_len, lr=lr * 0.5,
                )
                maintain_stats.append({"epoch": epoch + 1, "misclassified": len(misclassified), "corrected": len(queries), "loss": corr_loss})
                logger.info("Epoch %d maintain: %d misclassified → %d corrected (loss=%.4f)",
                           epoch + 1, len(misclassified), len(queries), corr_loss, extra={"tag": "INFRA"})

        if progress_callback:
            progress_callback(epoch + 1, avg_loss, epochs)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info("Epoch %d/%d — loss: %.4f (constraint: %.4f)", epoch + 1, epochs, avg_loss, avg_constraint, extra={"tag": "INFRA"})

    # 4. Save checkpoint
    out_path = save_path or str(_EMBEDDER_PATH)
    _save_checkpoint(out_path, encoder, vocab, itos, embed_dim, max_seq_len, n_heads, n_layers, bpe=bpe)
    logger.info("Saved embedder to %s", out_path, extra={"tag": "INFRA"})

    return {
        "epochs": epochs,
        "final_loss": losses[-1] if losses else 0.0,
        "vocab_size": actual_vocab,
        "embed_dim": embed_dim,
        "n_params": sum(p.data.size for p in params),
        "save_path": out_path,
        "refine_epochs": len(refine_stats),
        "maintain_epochs": len(maintain_stats),
        "total_corrected": sum(s["corrected"] for s in maintain_stats),
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
    bpe=None,
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

    # Save BPE tokenizer alongside the checkpoint
    if bpe is not None:
        bpe_save_path = str(path).replace(".sou", "-bpe.json")
        try:
            bpe.save(bpe_save_path)
            logger.info("Saved BPE tokenizer to %s", bpe_save_path, extra={"tag": "INFRA"})
        except Exception as e:
            logger.warning("Failed to save BPE tokenizer: %s", e, extra={"tag": "INFRA"})


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
        encode_fn=None,
    ):
        self.encoder = encoder
        self.vocab = vocab
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.encode_fn = encode_fn  # BPE encode function if available

    def eval(self):
        """Set encoder to eval mode (disables dropout for deterministic output)."""
        for block in self.encoder.blocks:
            if hasattr(block, "train"):
                block.train(False)

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

            # Try loading BPE tokenizer
            bpe_tokenizer = None
            try:
                from domains.multimodal.bpe_tokenizer import BPETokenizer
                bpe_path = path.replace(".sou", "-bpe.json")
                if os.path.exists(bpe_path):
                    bpe_tokenizer = BPETokenizer.load(bpe_path)
                    logger.info("Loaded BPE tokenizer from %s", bpe_path, extra={"tag": "INFRA"})
            except Exception:
                pass

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

            logger.info("Loaded SloTextEmbedder from %s (embed_dim=%d)", path, embed_dim, extra={"tag": "INFRA"})
            encode_fn = None
            if bpe_tokenizer is not None:
                def encode_fn(text: str, max_len: int) -> np.ndarray:
                    ids = bpe_tokenizer.encode(text)
                    ids = ids[:max_len]
                    padded = np.zeros(max_len, dtype=np.int64)
                    padded[:len(ids)] = ids
                    return padded
            embedder = cls(encoder, vocab, embed_dim, max_seq_len, encode_fn=encode_fn)
            embedder.eval()
            return embedder

        except Exception as e:
            logger.warning("Failed to load SloTextEmbedder: %s", e, extra={"tag": "INFRA"})
            return None

    def embed(self, text: str) -> List[float]:
        """Encode text to an L2-normalized vector.

        Args:
            text: input text string

        Returns:
            list of floats (L2-normalized)
        """
        import domains.training.slonet as _slonet
        _prev_accel = getattr(_slonet, "_ACCELERATOR", None)
        try:
            _slonet._ACCELERATOR = "none"
            if self.encode_fn is not None:
                ids = self.encode_fn(text, self.max_seq_len)
            else:
                ids = _encode_tokens(text, self.vocab, self.max_seq_len)
            ids = ids[np.newaxis, :]  # (1, max_seq_len)

            emb = self.encoder.forward(ids)  # (1, embed_dim)
            vec = emb.data.squeeze(0)
        finally:
            _slonet._ACCELERATOR = _prev_accel

        # L2-normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        # Pad/truncate to self.embed_dim
        if len(vec) < self.embed_dim:
            vec = np.pad(vec, (0, self.embed_dim - len(vec)))
        elif len(vec) > self.embed_dim:
            vec = vec[:self.embed_dim]
            n = np.linalg.norm(vec)
            if n > 0:
                vec = vec / n

        return vec.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode a batch of texts."""
        return [self.embed(t) for t in texts]
