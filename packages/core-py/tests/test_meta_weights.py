"""Tests for domains.feedback.meta_weights — MetaWeights dataclass and MetaWeightManager logic."""

import numpy as np
import pytest
from dataclasses import fields
from unittest.mock import patch, MagicMock

from domains.feedback.meta_weights import MetaWeights, MetaWeightManager


# ── MetaWeights dataclass ──────────────────────────────────────────────


class TestMetaWeights:
    def test_default_values(self):
        w = MetaWeights()
        assert w.temperature == 0.7
        assert w.repetition_penalty == 1.15
        assert w.top_p == 0.85
        assert w.top_k == 40
        assert w.length_penalty == 1.0
        assert w.style_bias == 0.0
        assert w.confidence_boost == 0.0

    def test_custom_values(self):
        w = MetaWeights(temperature=1.2, top_k=100, style_bias=0.5)
        assert w.temperature == 1.2
        assert w.top_k == 100
        assert w.style_bias == 0.5
        assert w.repetition_penalty == 1.15  # default unchanged

    def test_is_dataclass(self):
        assert fields(MetaWeights) is not None
        names = [f.name for f in fields(MetaWeights)]
        assert "temperature" in names
        assert "repetition_penalty" in names
        assert "top_p" in names
        assert "top_k" in names
        assert "length_penalty" in names
        assert "style_bias" in names
        assert "confidence_boost" in names

    def test_mutable_fields(self):
        w = MetaWeights()
        w.temperature = 0.5
        w.top_k = 20
        assert w.temperature == 0.5
        assert w.top_k == 20


# ── Simple embedding ───────────────────────────────────────────────────


class TestSimpleEmbed:
    def test_deterministic(self):
        mgr = MetaWeightManager.__new__(MetaWeightManager)
        mgr.embedding_dim = 64
        a = mgr._simple_embed("hello world")
        b = mgr._simple_embed("hello world")
        np.testing.assert_array_equal(a, b)

    def test_unit_norm(self):
        mgr = MetaWeightManager.__new__(MetaWeightManager)
        mgr.embedding_dim = 128
        v = mgr._simple_embed("test text here")
        norm = np.linalg.norm(v)
        assert abs(norm - 1.0) < 1e-5

    def test_empty_text(self):
        mgr = MetaWeightManager.__new__(MetaWeightManager)
        mgr.embedding_dim = 32
        v = mgr._simple_embed("")
        assert v.shape == (32,)
        # all zeros because no words
        assert np.all(v == 0.0)

    def test_dimension(self):
        mgr = MetaWeightManager.__new__(MetaWeightManager)
        mgr.embedding_dim = 256
        v = mgr._simple_embed("one two three")
        assert v.shape == (256,)

    def test_different_texts_different_vectors(self):
        mgr = MetaWeightManager.__new__(MetaWeightManager)
        mgr.embedding_dim = 64
        a = mgr._simple_embed("cats are great")
        b = mgr._simple_embed("dogs are okay")
        assert not np.allclose(a, b)


# ── Aggregate patterns ─────────────────────────────────────────────────


class TestAggregatePatterns:
    def _make_pattern(self, rating, similarity):
        from domains.feedback.database import SimilarPattern
        return SimilarPattern(content="x", rating=rating, similarity=similarity, pattern_type="msg")

    def test_empty_patterns(self):
        mgr = MetaWeightManager.__new__(MetaWeightManager)
        result = mgr._aggregate_patterns([])
        assert result == {}

    def test_single_thumbs_up(self):
        mgr = MetaWeightManager.__new__(MetaWeightManager)
        patterns = [self._make_pattern("thumbs_up", 1.0)]
        adj = mgr._aggregate_patterns(patterns)
        assert adj["temperature_boost"] == pytest.approx(0.05)
        assert adj["repetition_boost"] == pytest.approx(-0.05)
        assert adj["top_p_boost"] == pytest.approx(0.03)
        assert adj["top_k_boost"] == pytest.approx(-2.0)
        assert adj["style_bias"] == pytest.approx(0.05)
        assert adj["confidence_boost"] == pytest.approx(0.03)

    def test_single_thumbs_down(self):
        mgr = MetaWeightManager.__new__(MetaWeightManager)
        patterns = [self._make_pattern("thumbs_down", 1.0)]
        adj = mgr._aggregate_patterns(patterns)
        assert adj["temperature_boost"] == pytest.approx(-0.05)
        assert adj["repetition_boost"] == pytest.approx(0.05)
        assert adj["top_p_boost"] == pytest.approx(-0.03)
        assert adj["top_k_boost"] == pytest.approx(2.0)
        assert adj["style_bias"] == pytest.approx(-0.05)
        assert adj["confidence_boost"] == pytest.approx(-0.03)

    def test_weighted_average_equal_weights(self):
        mgr = MetaWeightManager.__new__(MetaWeightManager)
        patterns = [
            self._make_pattern("thumbs_up", 0.5),
            self._make_pattern("thumbs_down", 0.5),
        ]
        adj = mgr._aggregate_patterns(patterns)
        for key in adj:
            assert adj[key] == pytest.approx(0.0), f"{key} should cancel out"

    def test_weighted_average_unequal_weights(self):
        mgr = MetaWeightManager.__new__(MetaWeightManager)
        patterns = [
            self._make_pattern("thumbs_up", 0.9),
            self._make_pattern("thumbs_down", 0.1),
        ]
        adj = mgr._aggregate_patterns(patterns)
        # temperature: (0.9*0.05 + 0.1*-0.05) / 1.0 = 0.04
        assert adj["temperature_boost"] == pytest.approx(0.04)
        # top_k: (0.9*-2 + 0.1*2) / 1.0 = -1.6
        assert adj["top_k_boost"] == pytest.approx(-1.6)

    def test_all_same_rating_amplifies(self):
        mgr = MetaWeightManager.__new__(MetaWeightManager)
        patterns = [
            self._make_pattern("thumbs_up", 0.5),
            self._make_pattern("thumbs_up", 0.5),
            self._make_pattern("thumbs_up", 0.5),
        ]
        adj = mgr._aggregate_patterns(patterns)
        # Same as single thumbs_up since weights normalize
        assert adj["temperature_boost"] == pytest.approx(0.05)
        assert adj["confidence_boost"] == pytest.approx(0.03)


# ── Weight clamping ────────────────────────────────────────────────────


class TestWeightClamping:
    def _make_manager(self):
        from unittest.mock import MagicMock
        mgr = MetaWeightManager.__new__(MetaWeightManager)
        mgr.db = MagicMock()
        mgr.embedding_dim = 64
        mgr.use_simple_search = True
        mgr._weight_history = []
        mgr._default_weights = MetaWeights()
        mgr.decay_factor = 0.9
        mgr._embed_model = "simple"
        mgr._embedder = None
        return mgr

    def test_temperature_clamped_low(self):
        mgr = self._make_manager()
        mgr.db.get_user_meta_weights.return_value = {"temperature_boost": -2.0}
        mgr.db.find_similar_messages.return_value = []
        w = mgr.get_adjustment("test", rating="thumbs_up")
        assert w.temperature >= 0.1

    def test_temperature_clamped_high(self):
        mgr = self._make_manager()
        mgr.db.get_user_meta_weights.return_value = {"temperature_boost": 5.0}
        mgr.db.find_similar_messages.return_value = []
        w = mgr.get_adjustment("test", rating="thumbs_up")
        assert w.temperature <= 1.5

    def test_repetition_penalty_clamped(self):
        mgr = self._make_manager()
        mgr.db.get_user_meta_weights.return_value = {"repetition_boost": -5.0}
        mgr.db.find_similar_messages.return_value = []
        w = mgr.get_adjustment("test", rating="thumbs_up")
        assert w.repetition_penalty >= 0.8
        assert w.repetition_penalty <= 1.3

    def test_top_p_clamped(self):
        mgr = self._make_manager()
        mgr.db.get_user_meta_weights.return_value = {"top_p_boost": -5.0}
        mgr.db.find_similar_messages.return_value = []
        w = mgr.get_adjustment("test", rating="thumbs_up")
        assert w.top_p >= 0.1
        assert w.top_p <= 1.0

    def test_top_k_clamped(self):
        mgr = self._make_manager()
        mgr.db.get_user_meta_weights.return_value = {"top_k_boost": 500}
        mgr.db.find_similar_messages.return_value = []
        w = mgr.get_adjustment("test", rating="thumbs_up")
        assert w.top_k >= 5
        assert w.top_k <= 200

    def test_style_bias_clamped(self):
        mgr = self._make_manager()
        mgr.db.get_user_meta_weights.return_value = {"style_bias": 5.0}
        mgr.db.find_similar_messages.return_value = []
        w = mgr.get_adjustment("test", rating="thumbs_up")
        assert w.style_bias <= 1.0
        assert w.style_bias >= -1.0

    def test_confidence_boost_clamped(self):
        mgr = self._make_manager()
        mgr.db.get_user_meta_weights.return_value = {"confidence_boost": -5.0}
        mgr.db.find_similar_messages.return_value = []
        w = mgr.get_adjustment("test", rating="thumbs_up")
        assert w.confidence_boost >= -1.0
        assert w.confidence_boost <= 1.0


# ── get_adjustment basics ──────────────────────────────────────────────


class TestGetAdjustment:
    def _make_manager(self):
        mgr = MetaWeightManager.__new__(MetaWeightManager)
        mgr.db = MagicMock()
        mgr.embedding_dim = 64
        mgr.use_simple_search = True
        mgr._weight_history = []
        mgr._default_weights = MetaWeights()
        mgr.decay_factor = 0.9
        mgr._embed_model = "simple"
        mgr._embedder = None
        return mgr

    def test_returns_metaweights(self):
        mgr = self._make_manager()
        mgr.db.get_user_meta_weights.return_value = None
        mgr.db.find_similar_messages.return_value = []
        w = mgr.get_adjustment("hello")
        assert isinstance(w, MetaWeights)

    def test_default_when_no_data(self):
        mgr = self._make_manager()
        mgr.db.get_user_meta_weights.return_value = None
        mgr.db.find_similar_messages.return_value = []
        w = mgr.get_adjustment("hello")
        assert w.temperature == 0.7
        assert w.repetition_penalty == 1.15
        assert w.top_p == 0.85
        assert w.top_k == 40

    def test_user_weights_applied(self):
        mgr = self._make_manager()
        mgr.db.get_user_meta_weights.return_value = {
            "temperature_boost": 0.1,
            "repetition_boost": -0.05,
            "top_p_boost": 0.05,
            "top_k_boost": 3.0,
            "style_bias": 0.2,
            "confidence_boost": 0.1,
        }
        mgr.db.find_similar_messages.return_value = []
        w = mgr.get_adjustment("hello")
        assert w.temperature == pytest.approx(0.8)
        assert w.repetition_penalty == pytest.approx(1.10)
        assert w.top_p == pytest.approx(0.90)
        assert w.top_k == 43
        assert w.style_bias == pytest.approx(0.2)
        assert w.confidence_boost == pytest.approx(0.1)

    def test_weight_history_recorded(self):
        mgr = self._make_manager()
        mgr.db.get_user_meta_weights.return_value = None
        mgr.db.find_similar_messages.return_value = []
        mgr.get_adjustment("hello")
        assert len(mgr._weight_history) == 1
        assert "timestamp" in mgr._weight_history[0]

    def test_weight_history_trimming(self):
        mgr = self._make_manager()
        mgr.db.get_user_meta_weights.return_value = None
        mgr.db.find_similar_messages.return_value = []
        mgr._weight_history = [{"x": i} for i in range(101)]
        mgr.get_adjustment("hello")
        # 101 existing + 1 new = 102, trimmed to last 50
        assert len(mgr._weight_history) == 50

    def test_exception_returns_defaults(self):
        mgr = self._make_manager()
        mgr.db.get_user_meta_weights.side_effect = RuntimeError("db down")
        w = mgr.get_adjustment("hello")
        assert isinstance(w, MetaWeights)
        assert w.temperature == 0.7


# ── get_quality_trend ──────────────────────────────────────────────────


class TestGetQualityTrend:
    def _make_manager(self):
        mgr = MetaWeightManager.__new__(MetaWeightManager)
        mgr.db = MagicMock()
        mgr.embedding_dim = 64
        mgr.use_simple_search = True
        mgr._weight_history = []
        mgr._default_weights = MetaWeights()
        mgr.decay_factor = 0.9
        mgr._embed_model = "simple"
        mgr._embedder = None
        return mgr

    def test_empty_feedback(self):
        mgr = self._make_manager()
        mgr.db.get_all_feedback.return_value = []
        trend = mgr.get_quality_trend()
        assert trend["trend"] == 0.0
        assert trend["thumbs_up_ratio"] == 0.0

    def test_all_positive(self):
        mgr = self._make_manager()
        mgr.db.get_all_feedback.return_value = [
            {"rating": "thumbs_up"},
            {"rating": "thumbs_up"},
        ]
        trend = mgr.get_quality_trend()
        assert trend["trend"] == 1.0
        assert trend["sample_count"] == 2

    def test_mixed_ratings(self):
        mgr = self._make_manager()
        mgr.db.get_all_feedback.return_value = [
            {"rating": "thumbs_up"},
            {"rating": "thumbs_down"},
            {"rating": "thumbs_up"},
        ]
        trend = mgr.get_quality_trend()
        assert trend["trend"] == pytest.approx(2 / 3)

    def test_all_negative(self):
        mgr = self._make_manager()
        mgr.db.get_all_feedback.return_value = [
            {"rating": "thumbs_down"},
        ]
        trend = mgr.get_quality_trend()
        assert trend["trend"] == 0.0
        assert trend["thumbs_up_ratio"] == 0.0


# ── get_stats ──────────────────────────────────────────────────────────


class TestGetStats:
    def _make_manager(self):
        mgr = MetaWeightManager.__new__(MetaWeightManager)
        mgr.db = MagicMock()
        mgr.embedding_dim = 64
        mgr.use_simple_search = True
        mgr._weight_history = []
        mgr._default_weights = MetaWeights()
        mgr.decay_factor = 0.9
        mgr._embed_model = "simple"
        mgr._embedder = None
        return mgr

    def test_empty_history_uses_defaults(self):
        mgr = self._make_manager()
        mgr.db.get_stats.return_value = {"conversations": 0}
        mgr.db.get_all_feedback.return_value = []
        stats = mgr.get_stats()
        assert stats["current_weights"]["temperature"] == 0.7
        assert stats["current_weights"]["repetition_penalty"] == 1.15
        assert stats["current_weights"]["top_k"] == 40
        assert stats["history_length"] == 0

    def test_with_history(self):
        mgr = self._make_manager()
        mgr.db.get_stats.return_value = {}
        mgr.db.get_all_feedback.return_value = []
        mgr._weight_history = [
            {"temperature": 0.8, "repetition_penalty": 1.1, "top_p": 0.9,
             "top_k": 30, "style_bias": 0.1, "confidence_boost": 0.05},
            {"temperature": 0.6, "repetition_penalty": 1.2, "top_p": 0.8,
             "top_k": 50, "style_bias": -0.1, "confidence_boost": -0.05},
        ]
        stats = mgr.get_stats()
        assert stats["current_weights"]["temperature"] == pytest.approx(0.7)
        assert stats["current_weights"]["top_k"] == 40
        assert stats["history_length"] == 2


# ── Global singleton ──────────────────────────────────────────────────


class TestGlobalSingleton:
    def test_returns_manager(self):
        import domains.feedback.meta_weights as mod
        old = mod._meta_weight_manager
        mod._meta_weight_manager = None
        try:
            mgr = mod.get_meta_weight_manager()
            assert isinstance(mgr, MetaWeightManager)
            # Second call returns same instance
            assert mod.get_meta_weight_manager() is mgr
        finally:
            mod._meta_weight_manager = old
