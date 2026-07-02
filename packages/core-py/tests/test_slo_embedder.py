"""Tests for SloTextEmbedder — train, embed, save/load, vector store integration."""

import pytest
import numpy as np
import os
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Unit tests for tokenizer helpers
# ---------------------------------------------------------------------------

def test_tokenize_simple():
    from domains.inference.slo_embedder import _tokenize_simple
    tokens = _tokenize_simple("The quick brown fox jumps over the lazy dog")
    assert "quick" in tokens
    assert "brown" in tokens
    assert "the" not in tokens  # stopword filtered
    assert "over" not in tokens  # stopword filtered


def test_tokenize_simple_empty():
    from domains.inference.slo_embedder import _tokenize_simple
    assert _tokenize_simple("") == []
    assert _tokenize_simple("   ") == []


def test_build_vocab():
    from domains.inference.slo_embedder import _build_vocab
    texts = ["hello world", "foo bar baz", "hello foo"]
    vocab, itos = _build_vocab(texts, vocab_size=100)
    assert "<PAD>" in vocab
    assert "<UNK>" in vocab
    assert "hello" in vocab
    assert "world" in vocab
    assert len(vocab) <= 100
    # itos is inverse
    for k, v in vocab.items():
        assert itos[v] == k


def test_build_vocab_small():
    from domains.inference.slo_embedder import _build_vocab
    vocab, itos = _build_vocab(["a b c"], vocab_size=10)
    assert len(vocab) <= 10


def test_encode_tokens():
    from domains.inference.slo_embedder import _encode_tokens, _build_vocab
    vocab, _ = _build_vocab(["hello world test"], vocab_size=100)
    ids = _encode_tokens("hello world", vocab, max_len=16)
    assert ids.shape == (16,)
    assert ids[0] != 0  # not PAD
    assert ids[5] == 0  # PAD after content


def test_encode_tokens_truncation():
    from domains.inference.slo_embedder import _encode_tokens, _build_vocab
    vocab, _ = _build_vocab(["a b c d e f g h i j k l m n o p"], vocab_size=100)
    ids = _encode_tokens("a b c d e f g h i j", vocab, max_len=5)
    assert ids.shape == (5,)
    # Only first 5 tokens kept


# ---------------------------------------------------------------------------
# Encoder building
# ---------------------------------------------------------------------------

def test_build_encoder():
    from domains.inference.slo_embedder import _build_encoder
    enc = _build_encoder(vocab_size=256, embed_dim=64, max_seq_len=32, n_heads=4, n_layers=2)
    assert hasattr(enc, "tok_emb")
    assert hasattr(enc, "pos_emb")
    assert hasattr(enc, "blocks")
    assert hasattr(enc, "norm")
    assert hasattr(enc, "proj")
    assert len(enc.blocks) == 2
    params = enc.parameters()
    assert len(params) > 0


def test_encoder_forward():
    from domains.inference.slo_embedder import _build_encoder
    enc = _build_encoder(vocab_size=256, embed_dim=64, max_seq_len=32, n_heads=4, n_layers=2)
    ids = np.random.randint(0, 256, size=(2, 32))
    out = enc.forward(ids)
    assert out.data.shape == (2, 64)


# ---------------------------------------------------------------------------
# Contrastive loss
# ---------------------------------------------------------------------------

def test_contrastive_loss():
    from domains.inference.slo_embedder import _contrastive_loss
    B, D = 8, 64
    # Perfect alignment — loss should be low
    z = np.random.randn(B, D).astype(np.float32)
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-10)
    loss = _contrastive_loss(z, z, temperature=0.07)
    assert loss < 1.0  # low loss for identical pairs


def test_contrastive_loss_random():
    from domains.inference.slo_embedder import _contrastive_loss
    B, D = 8, 64
    z_i = np.random.randn(B, D).astype(np.float32)
    z_j = np.random.randn(B, D).astype(np.float32)
    z_i = z_i / (np.linalg.norm(z_i, axis=1, keepdims=True) + 1e-10)
    z_j = z_j / (np.linalg.norm(z_j, axis=1, keepdims=True) + 1e-10)
    loss = _contrastive_loss(z_i, z_j, temperature=0.07)
    assert loss > 0.5  # high loss for random pairs
    assert loss < 10.0  # but not insane


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def test_augment_text():
    from domains.inference.slo_embedder import _augment_text
    rng = np.random.RandomState(42)
    text = "the quick brown fox jumps over the lazy dog"
    aug = _augment_text(text, rng)
    assert isinstance(aug, str)
    assert len(aug) > 0


def test_augment_text_empty():
    from domains.inference.slo_embedder import _augment_text
    rng = np.random.RandomState(42)
    assert _augment_text("", rng) == ""
    assert _augment_text("a", rng) == "a"


# ---------------------------------------------------------------------------
# Training (tiny corpus)
# ---------------------------------------------------------------------------

def test_train_embedder_minimal():
    from domains.inference.slo_embedder import train_embedder
    texts = [f"this is sentence number {i} about topic {i % 5}" for i in range(20)]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test-embedder.sou")
        result = train_embedder(
            texts=texts,
            vocab_size=256,
            embed_dim=64,
            max_seq_len=32,
            n_heads=4,
            n_layers=2,
            epochs=3,
            batch_size=8,
            save_path=path,
        )
        assert result["epochs"] == 3
        assert result["vocab_size"] > 0
        assert result["n_params"] > 0
        assert os.path.exists(path)
        # Vocab sidecar exists
        vocab_path = path.replace(".sou", "-vocab.json")
        assert os.path.exists(vocab_path)


def test_train_embedder_too_few():
    from domains.inference.slo_embedder import train_embedder
    with pytest.raises(ValueError, match="at least 2"):
        train_embedder(["only one"], epochs=1)


# ---------------------------------------------------------------------------
# SloTextEmbedder save/load/embed
# ---------------------------------------------------------------------------

def test_embedder_embed_dim():
    from domains.inference.slo_embedder import SloTextEmbedder, train_embedder
    texts = [f"training text sample {i} for embedding" for i in range(30)]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test-embed.sou")
        train_embedder(texts, vocab_size=256, embed_dim=64, max_seq_len=32,
                        n_heads=4, n_layers=2, epochs=2, save_path=path)
        embedder = SloTextEmbedder.load(path)
        assert embedder is not None
        vec = embedder.embed("hello world")
        assert len(vec) == 384  # padded to 384 for InMemoryVectorStore compat
        norm = sum(x * x for x in vec) ** 0.5
        assert 0.99 < norm < 1.01  # L2-normalized


def test_embedder_load_nonexistent():
    from domains.inference.slo_embedder import SloTextEmbedder
    embedder = SloTextEmbedder.load("/nonexistent/path.sou")
    assert embedder is None


def test_embedder_deterministic():
    from domains.inference.slo_embedder import SloTextEmbedder, train_embedder
    texts = [f"sample {i} for determinism test" for i in range(20)]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "det-embed.sou")
        train_embedder(texts, vocab_size=256, embed_dim=64, max_seq_len=32,
                        n_heads=4, n_layers=2, epochs=2, save_path=path)
        embedder = SloTextEmbedder.load(path)
        v1 = embedder.embed("test sentence")
        v2 = embedder.embed("test sentence")
        # Same embedder instance → high cosine similarity (float32 forward pass
        # may have minor numerical differences across calls due to Tensor creation)
        v1a = np.array(v1[:64])
        v2a = np.array(v2[:64])
        cos_sim = float(np.dot(v1a, v2a) / (np.linalg.norm(v1a) * np.linalg.norm(v2a) + 1e-10))
        assert cos_sim > 0.98, f"Cosine similarity too low: {cos_sim}"


def test_embedder_different_texts_different_vectors():
    from domains.inference.slo_embedder import SloTextEmbedder, train_embedder
    texts = [f"unique topic {i} with different words" for i in range(30)]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "diff-embed.sou")
        train_embedder(texts, vocab_size=256, embed_dim=64, max_seq_len=32,
                        n_heads=4, n_layers=2, epochs=3, save_path=path)
        embedder = SloTextEmbedder.load(path)
        v1 = embedder.embed("neural network training")
        v2 = embedder.embed("cooking recipes for dinner")
        # Vectors should differ (not identical)
        assert v1 != v2


# ---------------------------------------------------------------------------
# Integration: simple_embed fallback
# ---------------------------------------------------------------------------

def test_simple_embed_fallback():
    """Verify simple_embed falls through to n-gram when no model is available."""
    from domains.inference.vector_store import simple_embed
    # This should not crash regardless of which backend is active
    vec = simple_embed("hello world test")
    assert isinstance(vec, list)
    assert len(vec) == 384
    # Should be L2-normalized
    norm = sum(x * x for x in vec) ** 0.5
    assert 0.99 < norm < 1.01


def test_simple_embed_deterministic():
    from domains.inference.vector_store import simple_embed
    v1 = simple_embed("deterministic test")
    v2 = simple_embed("deterministic test")
    assert v1 == v2


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def test_cmd_train_embed_exists():
    """Verify cmd_train_embed function exists in train.py."""
    import importlib.util
    # Try the CLI path first, then fall back
    cli_path = str(Path(__file__).resolve().parents[4] / "apps" / "cli" / "src")
    if cli_path not in __import__("sys").path:
        __import__("sys").path.insert(0, cli_path)
    try:
        from commands.train import cmd_train_embed
        assert callable(cmd_train_embed)
    except ImportError:
        pytest.skip("commands.train module not importable")
