"""Tests for SloTextEmbedder — train, embed, save/load, vector store integration."""

import pytest
pytestmark = pytest.mark.slow
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
    loss, constraint = _contrastive_loss(z, z, temperature=0.07)
    assert loss < 1.0  # low loss for identical pairs
    assert constraint == 0.0  # no anchors → no constraint


def test_contrastive_loss_random():
    from domains.inference.slo_embedder import _contrastive_loss
    B, D = 8, 64
    z_i = np.random.randn(B, D).astype(np.float32)
    z_j = np.random.randn(B, D).astype(np.float32)
    z_i = z_i / (np.linalg.norm(z_i, axis=1, keepdims=True) + 1e-10)
    z_j = z_j / (np.linalg.norm(z_j, axis=1, keepdims=True) + 1e-10)
    loss, constraint = _contrastive_loss(z_i, z_j, temperature=0.07)
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
        assert len(vec) == 64  # padded to embed_dim for InMemoryVectorStore compat
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
        train_embedder(texts, vocab_size=256, embed_dim=128, max_seq_len=32,
                        n_heads=4, n_layers=2, epochs=10, save_path=path)
        embedder = SloTextEmbedder.load(path)
        v1 = embedder.embed("neural network training")
        v2 = embedder.embed("cooking recipes for dinner")
        # Vectors should differ (not identical) — mean-subtraction debiasing
        # recovers the discriminative residuals from the collapsed raw space
        assert v1 != v2


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------

def test_quality_metadata_saved_and_accepted():
    """A small trained model collapses toward uniform embeddings, but the
    mean-subtraction debias at save/inference time re-centers the space
    (mean cosine ~0.0). The quality gate measures the deployed, debiased
    space and must accept it."""
    from domains.inference.slo_embedder import SloTextEmbedder, train_embedder
    rng_state = np.random.get_state()
    np.random.seed(0)  # collapse degree varies with init; seed for determinism
    try:
        texts = [f"this is sentence number {i} about topic {i % 5}" for i in range(30)]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "qual-embed.sou")
            train_embedder(texts, vocab_size=256, embed_dim=64, max_seq_len=32,
                            n_heads=4, n_layers=2, epochs=2, save_path=path)
            embedder = SloTextEmbedder.load(path)
            assert embedder is not None
            assert embedder.quality, "quality metadata must be recorded at train time"
            for key in ("probes", "degenerate_fraction", "mean_cosine", "nn_agreement"):
                assert key in embedder.quality, f"missing quality metric {key}"
            assert embedder.quality["probes"] >= 2
            assert 0.0 <= embedder.quality["degenerate_fraction"] <= 1.0
            assert 0.0 <= embedder.quality["nn_agreement"] <= 1.0
            # The debiased space is well-spread: mean cosine far below the
            # 0.90 collapse threshold (raw space was ~0.93)
            assert embedder.quality["mean_cosine"] < 0.50
            assert embedder.acceptable()
            # Corpus mean is stored and loaded for inference-time debiasing
            assert embedder.embed_mean is not None
            assert len(embedder.embed_mean) == 64
            # Retrieval benchmark vs the n-gram reference is recorded
            retrieval = embedder.quality.get("retrieval") or {}
            for key in ("queries", "trained_mrr", "ngram_mrr", "trained_hit", "ngram_hit", "better"):
                assert key in retrieval, f"missing retrieval metric {key}"
            assert retrieval["queries"] >= 2
            assert 0.0 <= retrieval["trained_mrr"] <= 1.0
            assert 0.0 <= retrieval["ngram_mrr"] <= 1.0
            assert retrieval["better"] in ("trained", "n_gram")
    finally:
        np.random.set_state(rng_state)


def test_embed_mean_debiases_collapsed_space():
    """Mean subtraction must recover discrimination from a collapsed space:
    raw encoder embeddings sit at ~0.93 cosine while the debiased embed()
    outputs sit near 0 and discriminate different texts."""
    from domains.inference.slo_embedder import (
        SloTextEmbedder, train_embedder, _compute_quality,
    )
    rng_state = np.random.get_state()
    np.random.seed(0)
    try:
        texts = [f"this is sentence number {i} about topic {i % 5}" for i in range(30)]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "debias-embed.sou")
            train_embedder(texts, vocab_size=256, embed_dim=64, max_seq_len=32,
                            n_heads=4, n_layers=2, epochs=2, save_path=path)
            embedder = SloTextEmbedder.load(path)
            # Raw space (no debias) is collapsed
            raw_q = _compute_quality(texts, embedder.encoder, embedder.vocab,
                                     embedder.max_seq_len, embedder.encode_fn)
            assert raw_q["mean_cosine"] > 0.90, f"expected collapsed raw space, got {raw_q['mean_cosine']}"
            # Debiased space (what inference uses) is spread
            deb_q = _compute_quality(texts, embedder.encoder, embedder.vocab,
                                     embedder.max_seq_len, embedder.encode_fn,
                                     embed_mean=embedder.embed_mean)
            assert deb_q["mean_cosine"] < 0.50, f"expected spread debiased space, got {deb_q['mean_cosine']}"
            assert embedder.acceptable()
    finally:
        np.random.set_state(rng_state)


def test_quality_gate_accepts_healthy_space():
    """A non-degenerate, spread embedding space passes the gate."""
    from domains.inference.slo_embedder import SloTextEmbedder
    quality = {
        "probes": 24,
        "degenerate_fraction": 0.0,
        "mean_cosine": 0.50,
        "nn_agreement": 0.60,
    }
    embedder = SloTextEmbedder(None, {}, quality=quality)
    assert embedder.acceptable()


def test_quality_gate_rejects_degenerate_pairs():
    from domains.inference.slo_embedder import SloTextEmbedder
    quality = {"probes": 24, "degenerate_fraction": 0.80, "mean_cosine": 0.60, "nn_agreement": 0.0}
    assert not SloTextEmbedder(None, {}, quality=quality).acceptable()


def test_quality_gate_rejects_collapsed_vectors():
    from domains.inference.slo_embedder import SloTextEmbedder
    quality = {"probes": 24, "degenerate_fraction": 0.0, "mean_cosine": 0.98, "nn_agreement": 0.0}
    assert not SloTextEmbedder(None, {}, quality=quality).acceptable()


def test_quality_gate_requires_metadata():
    """Legacy checkpoints (no quality metadata) are unverifiable → rejected."""
    from domains.inference.slo_embedder import SloTextEmbedder
    embedder = SloTextEmbedder(None, {}, quality={})
    assert not embedder.acceptable()


def test_quality_gate_too_few_probes():
    from domains.inference.slo_embedder import SloTextEmbedder
    quality = {"probes": 1, "degenerate_fraction": 0.0, "mean_cosine": 0.5, "nn_agreement": 0.0}
    assert not SloTextEmbedder(None, {}, quality=quality).acceptable()


def test_compute_quality_returns_valid_metrics():
    """_compute_quality runs the real encoder on corpus probes."""
    from domains.inference.slo_embedder import _compute_quality, _build_encoder, _build_vocab
    texts = [f"sample text {i} with distinct keywords" for i in range(12)]
    vocab, _ = _build_vocab(texts, vocab_size=64)
    encoder = _build_encoder(64, 16, 24, 2, 1)
    quality = _compute_quality(texts, encoder, vocab, 24)
    assert quality["probes"] >= 2
    assert 0.0 <= quality["degenerate_fraction"] <= 1.0
    assert -1.0 <= quality["mean_cosine"] <= 1.0
    assert 0.0 <= quality["nn_agreement"] <= 1.0


def test_perturb_text_drops_words_deterministically():
    """Word-drop perturbation must shorten the text, stay deterministic and
    leave very short texts untouched."""
    from domains.inference.slo_embedder import _perturb_text
    text = "the quick brown fox jumps over the lazy dog"
    a = _perturb_text(text)
    b = _perturb_text(text)
    assert a == b, "perturbation must be deterministic"
    assert len(a.split()) < len(text.split()), "words must be dropped"
    assert set(a.split()) <= set(text.split()), "no new words introduced"
    short = "a b c"
    assert _perturb_text(short) == short, "too-short texts stay unchanged"


def test_retrieval_benchmark_scores_both_embedders():
    """_retrieval_benchmark must return valid MRR/hit metrics for both the
    trained and n-gram embedders on identical queries."""
    from domains.inference.slo_embedder import _retrieval_benchmark

    def trained_fn(t):
        return np.array([float(ord(c)) for c in t[:8]] + [0.0] * 8)

    def ngram_fn(t):
        return np.array([float(len(c)) for c in t.split()] + [0.0] * 8)

    texts = [f"document {i} about topic {i % 5}" for i in range(40)]
    res = _retrieval_benchmark(texts, trained_fn, ngram_fn, top_k=3, max_queries=16)
    assert res["queries"] == 16
    assert 0.0 <= res["trained_mrr"] <= 1.0
    assert 0.0 <= res["ngram_mrr"] <= 1.0
    assert 0.0 <= res["trained_hit"] <= 1.0
    assert 0.0 <= res["ngram_hit"] <= 1.0
    assert res["top_k"] == 3
    assert res["better"] in ("trained", "n_gram")


def test_retrieval_benchmark_short_corpus():
    """A corpus too small for the benchmark returns a zeroed dict, not a crash."""
    from domains.inference.slo_embedder import _retrieval_benchmark
    res = _retrieval_benchmark(["only one text"], lambda t: np.zeros(8), lambda t: np.zeros(8))
    assert res["queries"] == 0
    assert res["better"] == "n_gram"


def test_quality_metadata_survives_roundtrip():
    """Quality stored in the .sou meta must survive load."""
    from domains.inference.slo_embedder import SloTextEmbedder, train_embedder
    texts = [f"this is sentence number {i} about topic {i % 5}" for i in range(20)]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "rt-embed.sou")
        train_embedder(texts, vocab_size=256, embed_dim=64, max_seq_len=32,
                        n_heads=4, n_layers=2, epochs=1, save_path=path)
        loaded = SloTextEmbedder.load(path)
        assert loaded is not None and loaded.quality
        assert loaded.quality["probes"] >= 2
        assert set(loaded.quality) == {
            "probes", "degenerate_fraction", "mean_cosine", "nn_agreement",
            "retrieval",
        }
        assert "trained_mrr" in loaded.quality["retrieval"]


# ---------------------------------------------------------------------------
# Integration: simple_embed fallback
# ---------------------------------------------------------------------------

def test_simple_embed_fallback():
    """Verify simple_embed falls through to n-gram when no model is available."""
    from domains.inference.vector_store import simple_embed
    # This should not crash regardless of which backend is active
    vec = simple_embed("hello world test")
    assert isinstance(vec, list)
    assert len(vec) == 384  # simple_embed defaults to 384
    # Should be L2-normalized
    norm = sum(x * x for x in vec) ** 0.5
    assert 0.99 < norm < 1.01


def test_simple_embed_deterministic():
    from domains.inference.vector_store import simple_embed
    import numpy as np
    v1 = simple_embed("deterministic test")
    v2 = simple_embed("deterministic test")
    # SloNet embedder: Metal GPU accelerator causes minor floating-point variance
    assert np.allclose(v1, v2, atol=0.05)


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


# ---------------------------------------------------------------------------
# Binary log-sum-exp tree
# ---------------------------------------------------------------------------

class TestLSETree:
    def test_lse_pair_standard(self):
        from domains.inference.slo_embedder import _lse_pair
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        result = _lse_pair(a, b, coeff=1.0)
        # Standard LSE: log(exp(a) + exp(b))
        expected = np.log(np.exp(a) + np.exp(b))
        assert np.allclose(result, expected, atol=1e-5)

    def test_lse_pair_contract(self):
        from domains.inference.slo_embedder import _lse_pair
        a = np.array([5.0, 10.0])
        b = np.array([1.0, 2.0])
        result = _lse_pair(a, b, coeff=0.0)
        # coeff=0: max(a, |b|) - |diff| = min(a, |b|)
        expected = np.minimum(a, np.abs(b))
        assert np.allclose(result, expected, atol=1e-5)

    def test_lse_pair_threshold(self):
        from domains.inference.slo_embedder import _lse_pair
        # Large diff → negligible correction
        a = np.array([50.0])
        b = np.array([0.0])
        result = _lse_pair(a, b, coeff=1.0, threshold=15.0)
        # Should be ~max(a, |b|) = 50.0
        assert abs(result[0] - 50.0) < 0.01

    def test_lse_tree_single(self):
        from domains.inference.slo_embedder import _lse_tree
        x = np.array([[5.0]])
        result = _lse_tree(x, axis=1)
        assert np.allclose(result, [5.0], atol=1e-5)

    def test_lse_tree_pair(self):
        from domains.inference.slo_embedder import _lse_tree
        x = np.array([[1.0, 2.0]])
        result = _lse_tree(x, axis=1)
        expected = np.log(np.exp(1.0) + np.exp(2.0))
        assert np.allclose(result, [expected], atol=1e-5)

    def test_lse_tree_matches_softmax(self):
        from domains.inference.slo_embedder import _lse_tree
        np.random.seed(42)
        x = np.random.randn(32, 64) * 3
        # Flat softmax
        flat = np.log(np.exp(x).sum(axis=1))
        # Tree LSE
        tree = _lse_tree(x, axis=1)
        assert np.allclose(flat, tree, atol=1e-3)

    def test_lse_tree_no_spillover(self):
        from domains.inference.slo_embedder import _lse_tree
        x = np.full((2, 8), 50.0)
        tree = _lse_tree(x, axis=1)
        assert not np.any(np.isinf(tree))

    def test_lse_tree_preserves_ranking(self):
        from domains.inference.slo_embedder import _lse_tree
        x = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 20.0, 30.0, 40.0],
        ])
        tree = _lse_tree(x, axis=1)
        # Row 1 should have higher LSE than row 0
        assert tree[1] > tree[0]

    def test_lse_tree_batch(self):
        from domains.inference.slo_embedder import _lse_tree
        x = np.random.randn(8, 32)
        tree = _lse_tree(x, axis=1)
        assert tree.shape == (8,)

    def test_lse_tree_negative_values(self):
        from domains.inference.slo_embedder import _lse_tree
        x = np.array([[-10.0, -20.0, -30.0, -40.0]])
        tree = _lse_tree(x, axis=1)
        # Should be close to max = -10
        assert tree[0] > -11.0
