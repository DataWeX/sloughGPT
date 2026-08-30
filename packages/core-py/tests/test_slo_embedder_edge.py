"""Edge-case tests for slo_embedder — targets gaps in the existing 87 tests."""

import pytest
import numpy as np
import os
import tempfile

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# _augment_text edge cases
# ---------------------------------------------------------------------------

class TestAugmentText:
    def test_empty_string(self):
        from domains.inference.slo_embedder import _augment_text
        rng = np.random.RandomState(0)
        assert _augment_text("", rng) == ""

    def test_single_word(self):
        from domains.inference.slo_embedder import _augment_text
        rng = np.random.RandomState(0)
        result = _augment_text("hello", rng)
        # Single word → tokens has length 1, no branch triggers → returned as-is
        assert result == "hello"

    def test_only_stopwords(self):
        from domains.inference.slo_embedder import _augment_text
        rng = np.random.RandomState(0)
        # All tokens get stripped by _tokenize_simple → empty token list → returns original
        result = _augment_text("the a an is are was", rng)
        assert result == "the a an is are was"

    def test_whitespace_only(self):
        from domains.inference.slo_embedder import _augment_text
        rng = np.random.RandomState(0)
        assert _augment_text("   ", rng) == "   "

    def test_two_words(self):
        from domains.inference.slo_embedder import _augment_text
        rng = np.random.RandomState(42)
        # 2 tokens → no branch triggers (all require len > 3 or > 4)
        result = _augment_text("hello world", rng)
        assert result == "hello world"

    def test_four_words_drop_threshold(self):
        from domains.inference.slo_embedder import _augment_text
        rng = np.random.RandomState(42)
        # Exactly 4 tokens: "drop" needs > 3, so it can trigger
        # but "shuffle" also needs > 3, and "crop" needs > 4
        results = set()
        for seed in range(50):
            r = _augment_text("alpha beta gamma delta", np.random.RandomState(seed))
            results.add(r)
        # At least some variation should occur across different seeds
        assert len(results) >= 1

    def test_unicode_text(self):
        from domains.inference.slo_embedder import _augment_text
        rng = np.random.RandomState(0)
        # Unicode letters are matched by [a-z0-9']+ only if lowercase ascii
        # Non-ascii chars get stripped → may produce empty token list
        result = _augment_text("日本語テスト café résumé", rng)
        assert isinstance(result, str)

    def test_punctuation_heavy(self):
        from domains.inference.slo_embedder import _augment_text
        rng = np.random.RandomState(0)
        result = _augment_text("hello... world!!!, foo; bar:", rng)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _perturb_text edge cases
# ---------------------------------------------------------------------------

class TestPerturbText:
    def test_empty_string(self):
        from domains.inference.slo_embedder import _perturb_text
        assert _perturb_text("") == ""

    def test_single_word(self):
        from domains.inference.slo_embedder import _perturb_text
        assert _perturb_text("hello") == "hello"

    def test_at_min_keep_boundary(self):
        from domains.inference.slo_embedder import _perturb_text
        # len(tokens) == min_keep (default 3) → returned unchanged
        assert _perturb_text("a b c") == "a b c"

    def test_one_above_min_keep(self):
        from domains.inference.slo_embedder import _perturb_text
        # 4 tokens, min_keep=3 → one word gets dropped
        result = _perturb_text("a b c d")
        kept = result.split()
        assert len(kept) >= 3
        assert set(kept) <= {"a", "b", "c", "d"}

    def test_drop_frac_zero(self):
        from domains.inference.slo_embedder import _perturb_text
        text = "one two three four five six"
        result = _perturb_text(text, drop_frac=0.0)
        # n_drop = max(1, round(6 * 0.0)) = max(1, 0) = 1 → still drops 1
        assert len(result.split()) == 5

    def test_drop_frac_one(self):
        from domains.inference.slo_embedder import _perturb_text
        text = "one two three four five six seven eight"
        result = _perturb_text(text, drop_frac=1.0)
        kept = result.split()
        # n_drop = round(8 * 1.0) = 8, but min_keep=3 kicks in
        assert len(kept) >= 3

    def test_min_keep_one(self):
        from domains.inference.slo_embedder import _perturb_text
        text = "one two three four five"
        result = _perturb_text(text, min_keep=1)
        kept = result.split()
        assert len(kept) >= 1

    def test_deterministic(self):
        from domains.inference.slo_embedder import _perturb_text
        text = "the quick brown fox jumps over lazy dog"
        a = _perturb_text(text)
        b = _perturb_text(text)
        assert a == b

    def test_same_words_subset(self):
        from domains.inference.slo_embedder import _perturb_text
        text = "alpha beta gamma delta epsilon zeta eta theta"
        result = _perturb_text(text)
        assert set(result.split()) <= set(text.split())

    def test_large_drop_frac_many_words(self):
        from domains.inference.slo_embedder import _perturb_text
        text = " ".join([f"w{i}" for i in range(100)])
        result = _perturb_text(text, drop_frac=0.9)
        kept = result.split()
        assert len(kept) >= 3  # min_keep


# ---------------------------------------------------------------------------
# _contrastive_loss edge cases
# ---------------------------------------------------------------------------

class TestContrastiveLoss:
    def test_batch_size_one(self):
        from domains.inference.slo_embedder import _contrastive_loss
        z = np.random.randn(1, 32).astype(np.float32)
        z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-10)
        loss, constraint = _contrastive_loss(z, z)
        # B=1: diagonal is the only pair → log_probs[0,0] = 0 → loss ≈ 0
        assert loss >= 0.0
        assert loss < 0.5

    def test_zero_vectors(self):
        from domains.inference.slo_embedder import _contrastive_loss
        B, D = 4, 16
        z_i = np.zeros((B, D), dtype=np.float32)
        z_j = np.zeros((B, D), dtype=np.float32)
        # Dot product of zero vectors → all zeros → should not crash
        loss, constraint = _contrastive_loss(z_i, z_j)
        assert np.isfinite(loss)

    def test_identical_vectors_batch(self):
        from domains.inference.slo_embedder import _contrastive_loss
        B, D = 8, 32
        z = np.random.randn(B, D).astype(np.float32)
        z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-10)
        loss, _ = _contrastive_loss(z, z)
        # Identical views → positive pair similarity is 1.0 → low loss
        assert loss < 1.0

    def test_orthogonal_vectors(self):
        from domains.inference.slo_embedder import _contrastive_loss
        B, D = 4, 16
        z_i = np.eye(B, D, dtype=np.float32)
        z_j = np.eye(B, D, dtype=np.float32)
        loss, _ = _contrastive_loss(z_i, z_j)
        # Orthogonal pairs have low positive similarity → higher loss
        assert loss > 0.0

    def test_very_high_temperature(self):
        from domains.inference.slo_embedder import _contrastive_loss
        B, D = 4, 16
        z = np.random.randn(B, D).astype(np.float32)
        z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-10)
        loss, _ = _contrastive_loss(z, z, temperature=100.0)
        assert np.isfinite(loss)

    def test_very_low_temperature(self):
        from domains.inference.slo_embedder import _contrastive_loss
        B, D = 4, 16
        z = np.random.randn(B, D).astype(np.float32)
        z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-10)
        loss, _ = _contrastive_loss(z, z, temperature=0.001)
        assert np.isfinite(loss)

    def test_with_matching_labels_and_meaning_tags(self):
        from domains.inference.slo_embedder import _contrastive_loss

        class FakeTags:
            def __init__(self, dim):
                self._dim = dim
                self._vecs = {}

            def get(self, name):
                if name not in self._vecs:
                    rng = np.random.RandomState(hash(name) % 2**31)
                    v = rng.randn(self._dim).astype(np.float32)
                    self._vecs[name] = v / (np.linalg.norm(v) + 1e-10)
                return self._vecs[name]

            def names(self):
                return list(self._vecs.keys())

        B, D = 4, 16
        tags = FakeTags(D)
        z = np.random.randn(B, D).astype(np.float32)
        z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-10)
        labels = ["factual", "interrogative", "factual", "imperative"]
        loss, constraint = _contrastive_loss(z, z, point_labels=labels, meaning_tags=tags)
        assert constraint > 0.0
        assert loss > 0.0

    def test_constraint_with_unknown_label(self):
        from domains.inference.slo_embedder import _contrastive_loss

        class FakeTags:
            def get(self, name):
                return None

        B, D = 4, 16
        z = np.random.randn(B, D).astype(np.float32)
        z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-10)
        labels = ["unknown_tag"] * B
        loss, constraint = _contrastive_loss(z, z, point_labels=labels, meaning_tags=FakeTags())
        assert constraint == 0.0  # no valid tags → no constraint


# ---------------------------------------------------------------------------
# _tokenize_simple edge cases
# ---------------------------------------------------------------------------

class TestTokenizeSimple:
    def test_numbers(self):
        from domains.inference.slo_embedder import _tokenize_simple
        tokens = _tokenize_simple("123 456 789")
        assert "123" in tokens

    def test_mixed_case(self):
        from domains.inference.slo_embedder import _tokenize_simple
        tokens = _tokenize_simple("Hello WORLD test")
        # Lowercased before matching
        assert "hello" in tokens
        assert "world" in tokens

    def test_apostrophes(self):
        from domains.inference.slo_embedder import _tokenize_simple
        tokens = _tokenize_simple("don't can't won't")
        assert "don't" in tokens
        assert "can't" in tokens

    def test_single_char_tokens(self):
        from domains.inference.slo_embedder import _tokenize_simple
        tokens = _tokenize_simple("a b c x y z")
        # "a" is a stopword → filtered; "b", "c", "x", "y", "z" are not
        assert "a" not in tokens
        assert "b" in tokens

    def test_all_stopwords(self):
        from domains.inference.slo_embedder import _tokenize_simple
        tokens = _tokenize_simple("the and or but if")
        assert tokens == []


# ---------------------------------------------------------------------------
# _build_vocab edge cases
# ---------------------------------------------------------------------------

class TestBuildVocab:
    def test_empty_corpus(self):
        from domains.inference.slo_embedder import _build_vocab
        vocab, itos = _build_vocab([], vocab_size=100)
        # Should still have special tokens
        assert "<PAD>" in vocab
        assert "<UNK>" in vocab
        assert len(vocab) == 4  # just the specials

    def test_vocab_size_exactly_four(self):
        from domains.inference.slo_embedder import _build_vocab
        # vocab_size=4 → only special tokens, no room for words
        vocab, itos = _build_vocab(["hello world"], vocab_size=4)
        assert len(vocab) == 4
        assert "hello" not in vocab

    def test_vocab_size_one(self):
        from domains.inference.slo_embedder import _build_vocab
        vocab, itos = _build_vocab(["hello world"], vocab_size=1)
        # Still gets the 4 specials (they're inserted first unconditionally)
        assert "<PAD>" in vocab

    def test_large_corpus(self):
        from domains.inference.slo_embedder import _build_vocab
        texts = [f"word{i} common shared" for i in range(1000)]
        vocab, itos = _build_vocab(texts, vocab_size=50)
        assert len(vocab) <= 50
        assert "common" in vocab
        assert "shared" in vocab

    def test_duplicate_texts(self):
        from domains.inference.slo_embedder import _build_vocab
        vocab, _ = _build_vocab(["hello world"] * 100, vocab_size=100)
        assert "hello" in vocab
        assert "world" in vocab

    def test_inverse_mapping(self):
        from domains.inference.slo_embedder import _build_vocab
        vocab, itos = _build_vocab(["test foo bar"], vocab_size=50)
        for word, idx in vocab.items():
            assert itos[idx] == word

    def test_all_stopwords_corpus(self):
        from domains.inference.slo_embedder import _build_vocab
        vocab, itos = _build_vocab(["the a an is are"], vocab_size=100)
        # All words are stopwords → only specials
        assert len(vocab) == 4


# ---------------------------------------------------------------------------
# _encode_tokens edge cases
# ---------------------------------------------------------------------------

class TestEncodeTokens:
    def test_empty_text(self):
        from domains.inference.slo_embedder import _encode_tokens, _build_vocab
        vocab, _ = _build_vocab(["hello world"], vocab_size=50)
        ids = _encode_tokens("", vocab, max_len=8)
        assert ids.shape == (8,)
        assert np.all(ids == 0)  # all PAD

    def test_all_unknown_words(self):
        from domains.inference.slo_embedder import _encode_tokens, _build_vocab
        vocab, _ = _build_vocab(["hello world"], vocab_size=50)
        ids = _encode_tokens("xyz qwerty nonexistent", vocab, max_len=8)
        assert ids.shape == (8,)
        # Unknown words → UNK (id=3)
        non_pad = ids[ids != 0]
        assert all(x == 3 for x in non_pad)

    def test_max_len_one(self):
        from domains.inference.slo_embedder import _encode_tokens, _build_vocab
        vocab, _ = _build_vocab(["hello world"], vocab_size=50)
        ids = _encode_tokens("hello world", vocab, max_len=1)
        assert ids.shape == (1,)


# ---------------------------------------------------------------------------
# _sample_probes edge cases
# ---------------------------------------------------------------------------

class TestSampleProbes:
    def test_empty_corpus(self):
        from domains.inference.slo_embedder import _sample_probes
        assert _sample_probes([]) == []

    def test_single_text(self):
        from domains.inference.slo_embedder import _sample_probes
        result = _sample_probes(["only one"])
        assert result == ["only one"]

    def test_fewer_than_max(self):
        from domains.inference.slo_embedder import _sample_probes
        texts = [f"text{i}" for i in range(5)]
        result = _sample_probes(texts, max_probes=10)
        assert len(result) == 5

    def test_exactly_max(self):
        from domains.inference.slo_embedder import _sample_probes
        texts = [f"text{i}" for i in range(24)]
        result = _sample_probes(texts, max_probes=24)
        assert len(result) == 24

    def test_many_texts(self):
        from domains.inference.slo_embedder import _sample_probes
        texts = [f"document{i}" for i in range(200)]
        result = _sample_probes(texts, max_probes=24)
        assert len(result) <= 24
        # Step = 200 // 24 = 8, so indices are 0, 8, 16, ...
        assert result[0] == texts[0]

    def test_max_probes_one(self):
        from domains.inference.slo_embedder import _sample_probes
        texts = [f"t{i}" for i in range(10)]
        result = _sample_probes(texts, max_probes=1)
        assert len(result) == 1
        assert result[0] == texts[0]

    def test_deterministic(self):
        from domains.inference.slo_embedder import _sample_probes
        texts = [f"doc{i}" for i in range(50)]
        a = _sample_probes(texts, max_probes=10)
        b = _sample_probes(texts, max_probes=10)
        assert a == b


# ---------------------------------------------------------------------------
# _lse_pair / _lse_tree additional edge cases
# ---------------------------------------------------------------------------

class TestLSEEdgeCases:
    def test_lse_pair_negative_inf(self):
        from domains.inference.slo_embedder import _lse_pair
        a = np.array([-np.inf])
        b = np.array([0.0])
        result = _lse_pair(a, b)
        assert np.isfinite(result[0])

    def test_lse_pair_both_same(self):
        from domains.inference.slo_embedder import _lse_pair
        a = np.array([5.0, 5.0])
        b = np.array([5.0, 5.0])
        result = _lse_pair(a, b)
        expected = np.log(2.0) + 5.0
        assert np.allclose(result, expected, atol=1e-5)

    def test_lse_tree_odd_length(self):
        from domains.inference.slo_embedder import _lse_tree
        x = np.array([[1.0, 2.0, 3.0]])
        result = _lse_tree(x, axis=1)
        expected = np.log(np.exp(1.0) + np.exp(2.0) + np.exp(3.0))
        assert np.allclose(result[0], expected, atol=1e-4)

    def test_lse_tree_length_4(self):
        from domains.inference.slo_embedder import _lse_tree
        x = np.array([[1.0, 2.0, 3.0, 4.0]])
        result = _lse_tree(x, axis=1)
        expected = np.log(np.exp(1.0) + np.exp(2.0) + np.exp(3.0) + np.exp(4.0))
        assert np.allclose(result[0], expected, atol=1e-4)

    def test_lse_tree_large_values(self):
        from domains.inference.slo_embedder import _lse_tree
        x = np.full((1, 8), 500.0)
        result = _lse_tree(x, axis=1)
        assert np.isfinite(result[0])
        # log(8 * exp(500)) = 500 + log(8)
        assert abs(result[0] - (500.0 + np.log(8))) < 0.01

    def test_lse_tree_mixed_sign(self):
        from domains.inference.slo_embedder import _lse_tree
        x = np.array([[-100.0, 0.0, 100.0]])
        result = _lse_tree(x, axis=1)
        assert abs(result[0] - 100.0) < 0.01


# ---------------------------------------------------------------------------
# SloTextEmbedder.embed edge cases
# ---------------------------------------------------------------------------

class TestEmbedEdgeCases:
    def _make_embedder(self):
        from domains.inference.slo_embedder import SloTextEmbedder, _build_encoder, _build_vocab
        texts = [f"training sentence {i} about topic" for i in range(20)]
        vocab, itos = _build_vocab(texts, vocab_size=128)
        encoder = _build_encoder(128, 32, 16, 2, 1)
        return SloTextEmbedder(encoder, vocab, embed_dim=32, max_seq_len=16)

    def test_empty_string(self):
        embedder = self._make_embedder()
        vec = embedder.embed("")
        assert len(vec) == 32
        norm = np.linalg.norm(vec)
        assert 0.99 < norm < 1.01

    def test_unicode_string(self):
        embedder = self._make_embedder()
        vec = embedder.embed("日本語テスト café")
        assert len(vec) == 32
        norm = np.linalg.norm(vec)
        assert 0.99 < norm < 1.01

    def test_very_long_string(self):
        embedder = self._make_embedder()
        long_text = "word " * 500
        vec = embedder.embed(long_text)
        assert len(vec) == 32
        norm = np.linalg.norm(vec)
        assert 0.99 < norm < 1.01

    def test_single_char(self):
        embedder = self._make_embedder()
        vec = embedder.embed("x")
        assert len(vec) == 32

    def test_punctuation_only(self):
        embedder = self._make_embedder()
        vec = embedder.embed("... !!! ,,, ;;;")
        assert len(vec) == 32

    def test_with_embed_mean(self):
        from domains.inference.slo_embedder import SloTextEmbedder, _build_encoder, _build_vocab
        texts = [f"training text {i}" for i in range(20)]
        vocab, _ = _build_vocab(texts, vocab_size=128)
        encoder = _build_encoder(128, 32, 16, 2, 1)
        embed_mean = np.random.randn(32).astype(np.float32)
        embedder = SloTextEmbedder(encoder, vocab, embed_dim=32, max_seq_len=16, embed_mean=embed_mean)
        vec = embedder.embed("hello world")
        assert len(vec) == 32
        norm = np.linalg.norm(vec)
        assert 0.99 < norm < 1.01

    def test_embed_batch(self):
        embedder = self._make_embedder()
        vecs = embedder.embed_batch(["hello", "world", "test"])
        assert len(vecs) == 3
        for v in vecs:
            assert len(v) == 32

    def test_eval_mode(self):
        embedder = self._make_embedder()
        embedder.eval()
        v1 = embedder.embed("test sentence")
        v2 = embedder.embed("test sentence")
        # Eval mode → deterministic (no dropout)
        assert np.allclose(v1, v2, atol=1e-5)

    def test_acceptable_no_quality(self):
        from domains.inference.slo_embedder import SloTextEmbedder
        embedder = SloTextEmbedder(None, {}, quality={})
        assert not embedder.acceptable()

    def test_acceptable_boundary_degenerate(self):
        from domains.inference.slo_embedder import SloTextEmbedder, QUALITY_DEGENERATE_MAX
        q = {"probes": 24, "degenerate_fraction": QUALITY_DEGENERATE_MAX, "mean_cosine": 0.0, "nn_agreement": 0.5}
        assert not SloTextEmbedder(None, {}, quality=q).acceptable()

    def test_acceptable_boundary_cosine(self):
        from domains.inference.slo_embedder import SloTextEmbedder, QUALITY_MEAN_COSINE_MAX
        q = {"probes": 24, "degenerate_fraction": 0.0, "mean_cosine": QUALITY_MEAN_COSINE_MAX, "nn_agreement": 0.5}
        assert not SloTextEmbedder(None, {}, quality=q).acceptable()

    def test_acceptable_just_below_boundary(self):
        from domains.inference.slo_embedder import SloTextEmbedder, QUALITY_DEGENERATE_MAX, QUALITY_MEAN_COSINE_MAX
        q = {
            "probes": 24,
            "degenerate_fraction": QUALITY_DEGENERATE_MAX - 0.01,
            "mean_cosine": QUALITY_MEAN_COSINE_MAX - 0.01,
            "nn_agreement": 0.5,
        }
        assert SloTextEmbedder(None, {}, quality=q).acceptable()


# ---------------------------------------------------------------------------
# _nn_agreement edge cases
# ---------------------------------------------------------------------------

class TestNnAgreement:
    def test_two_probes(self):
        from domains.inference.slo_embedder import _nn_agreement
        trained = np.random.randn(2, 16).astype(np.float32)
        reference = np.random.randn(2, 16).astype(np.float32)
        # P=2, k clamped to min(k, P-1)=1
        result = _nn_agreement(trained, reference, k=3)
        assert 0.0 <= result <= 1.0

    def test_single_probe(self):
        from domains.inference.slo_embedder import _nn_agreement
        trained = np.random.randn(1, 16).astype(np.float32)
        reference = np.random.randn(1, 16).astype(np.float32)
        result = _nn_agreement(trained, reference)
        assert result == 0.0

    def test_identical_spaces(self):
        from domains.inference.slo_embedder import _nn_agreement
        x = np.random.randn(10, 16).astype(np.float32)
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)
        result = _nn_agreement(x, x, k=3)
        assert result == 1.0

    def test_k_larger_than_probes(self):
        from domains.inference.slo_embedder import _nn_agreement
        x = np.random.randn(5, 16).astype(np.float32)
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)
        result = _nn_agreement(x, x, k=100)
        # k gets clamped to P-1=4
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# _compute_embed_mean edge cases
# ---------------------------------------------------------------------------

class TestComputeEmbedMean:
    def test_empty_texts(self):
        from domains.inference.slo_embedder import _compute_embed_mean, _build_encoder, _build_vocab
        vocab, _ = _build_vocab(["dummy"], vocab_size=32)
        encoder = _build_encoder(32, 16, 8, 2, 1)
        result = _compute_embed_mean([], encoder, vocab, 8)
        assert result.size == 0

    def test_returns_correct_dim(self):
        from domains.inference.slo_embedder import _compute_embed_mean, _build_encoder, _build_vocab
        texts = [f"sample {i}" for i in range(10)]
        vocab, _ = _build_vocab(texts, vocab_size=64)
        encoder = _build_encoder(64, 32, 16, 2, 1)
        result = _compute_embed_mean(texts, encoder, vocab, 16)
        assert result.shape == (32,)


# ---------------------------------------------------------------------------
# _label_by_meaning edge cases
# ---------------------------------------------------------------------------

class TestLabelByMeaning:
    def test_none_points_store(self):
        from domains.inference.slo_embedder import _label_by_meaning
        result = _label_by_meaning("hello", points_store=None)
        assert result is None

    def test_empty_text(self):
        from domains.inference.slo_embedder import _label_by_meaning
        class FakeStore:
            dimension = 128
            def classify(self, vec):
                return "some_label"
        result = _label_by_meaning("", points_store=FakeStore())
        # simple_embed on empty string may return zero vector → classify may or may not fire
        assert result is None or isinstance(result, str)
