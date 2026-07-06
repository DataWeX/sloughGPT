"""Tests for meta-weight manager — feedback-based generation adjustment."""

import pytest
pytestmark = pytest.mark.slow
import numpy as np
from unittest.mock import patch, MagicMock
from domains.feedback.meta_weights import (
    MetaWeights, MetaWeightManager, get_meta_weight_manager,
)
from domains.feedback.database import SimilarPattern


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def mock_db():
    db = MagicMock()
    db.get_user_meta_weights.return_value = {}
    db.find_similar_messages.return_value = []
    db.find_similar_by_text.return_value = []
    db.get_all_feedback.return_value = []
    db.get_stats.return_value = {"total": 0}
    return db


@pytest.fixture
def manager(mock_db):
    with patch('domains.feedback.meta_weights.get_feedback_db', return_value=mock_db):
        mgr = MetaWeightManager(db_path=":memory:")
        mgr.db = mock_db
    return mgr


# ── MetaWeights Dataclass ─────────────────────────────────────────────────

class TestMetaWeights:

    def test_defaults(self):
        w = MetaWeights()
        assert w.temperature == 0.8
        assert w.repetition_penalty == 1.0
        assert w.top_p == 0.9
        assert w.top_k == 50
        assert w.length_penalty == 1.0
        assert w.style_bias == 0.0
        assert w.confidence_boost == 0.0

    def test_custom_values(self):
        w = MetaWeights(temperature=1.2, top_p=0.8, top_k=40)
        assert w.temperature == 1.2
        assert w.top_p == 0.8
        assert w.top_k == 40


# ── MetaWeightManager ──────────────────────────────────────────────────────

class TestMetaWeightManager:

    def test_init(self, manager, mock_db):
        assert manager.db is mock_db
        assert manager._weight_history == []
        assert manager._default_weights.temperature == 0.8
        assert manager.decay_factor == 0.9

    def test_simple_embed(self, manager):
        vec = manager._simple_embed("hello world")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (384,)
        assert vec.dtype == np.float64 or vec.dtype == np.float32

    def test_simple_embed_normalized(self, manager):
        vec = manager._simple_embed("test text here")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-6

    def test_simple_embed_empty_string(self, manager):
        vec = manager._simple_embed("")
        assert vec.shape == (384,)
        assert np.allclose(vec, 0.0)

    def test_simple_embed_deterministic(self, manager):
        v1 = manager._simple_embed("same input")
        v2 = manager._simple_embed("same input")
        np.testing.assert_array_equal(v1, v2)

    def test_aggregate_patterns_empty(self, manager):
        result = manager._aggregate_patterns([])
        assert result == {}

    def test_aggregate_patterns_thumbs_up(self, manager):
        patterns = [SimilarPattern(
            content="good response", rating="thumbs_up",
            similarity=0.9, pattern_type="text",
        )]
        result = manager._aggregate_patterns(patterns)
        assert "temperature_boost" in result
        assert "repetition_boost" in result
        assert result["temperature_boost"] > 0
        assert result["repetition_boost"] < 0

    def test_aggregate_patterns_thumbs_down(self, manager):
        patterns = [SimilarPattern(
            content="bad response", rating="thumbs_down",
            similarity=0.8, pattern_type="text",
        )]
        result = manager._aggregate_patterns(patterns)
        assert result["temperature_boost"] < 0
        assert result["repetition_boost"] > 0

    def test_aggregate_patterns_mixed(self, manager):
        patterns = [
            SimilarPattern(content="a", rating="thumbs_up", similarity=0.9, pattern_type="text"),
            SimilarPattern(content="b", rating="thumbs_down", similarity=0.9, pattern_type="text"),
        ]
        result = manager._aggregate_patterns(patterns)
        # Equal weight up/down → boosts should cancel
        assert abs(result["temperature_boost"]) < 0.01

    def test_get_adjustment_empty_db(self, manager, mock_db):
        weights = manager.get_adjustment("hello")
        assert isinstance(weights, MetaWeights)
        assert 0.1 <= weights.temperature <= 2.0
        assert 0.8 <= weights.repetition_penalty <= 1.5

    def test_get_adjustment_with_user_weights(self, manager, mock_db):
        mock_db.get_user_meta_weights.return_value = {
            "temperature_boost": 0.1,
            "repetition_boost": -0.05,
        }
        weights = manager.get_adjustment("hello", user_id="user1")
        assert weights.temperature == pytest.approx(0.9, abs=0.1)
        assert weights.repetition_penalty == pytest.approx(0.95, abs=0.1)

    def test_get_adjustment_clamps_temperature(self, manager, mock_db):
        mock_db.get_user_meta_weights.return_value = {
            "temperature_boost": 5.0,
        }
        weights = manager.get_adjustment("hello")
        assert weights.temperature == 2.0

    def test_get_adjustment_clamps_repetition(self, manager, mock_db):
        mock_db.get_user_meta_weights.return_value = {
            "repetition_boost": -10.0,
        }
        weights = manager.get_adjustment("hello")
        assert weights.repetition_penalty == 0.8

    def test_get_adjustment_stores_history(self, manager, mock_db):
        manager.get_adjustment("test message")
        assert len(manager._weight_history) == 1
        assert "timestamp" in manager._weight_history[0]
        assert "temperature" in manager._weight_history[0]

    def test_get_adjustment_history_trimming(self, manager, mock_db):
        manager._weight_history = [{"t": i} for i in range(101)]
        manager.get_adjustment("test")
        assert len(manager._weight_history) == 50

    def test_get_quality_trend_empty(self, manager, mock_db):
        mock_db.get_all_feedback.return_value = []
        result = manager.get_quality_trend()
        assert result["trend"] == 0.0
        assert result["thumbs_up_ratio"] == 0.0

    def test_get_quality_trend_with_data(self, manager, mock_db):
        mock_db.get_all_feedback.return_value = [
            {"rating": "thumbs_up"},
            {"rating": "thumbs_up"},
            {"rating": "thumbs_down"},
        ]
        result = manager.get_quality_trend()
        assert result["trend"] == pytest.approx(2.0 / 3.0)
        assert result["thumbs_up_ratio"] == pytest.approx(2.0 / 3.0)
        assert result["sample_count"] == 3

    def test_get_quality_trend_all_positive(self, manager, mock_db):
        mock_db.get_all_feedback.return_value = [
            {"rating": "thumbs_up"} for _ in range(5)
        ]
        result = manager.get_quality_trend()
        assert result["trend"] == 1.0

    def test_get_stats(self, manager, mock_db):
        mock_db.get_stats.return_value = {"total": 42}
        stats = manager.get_stats()
        assert "db_stats" in stats
        assert stats["db_stats"]["total"] == 42
        assert "quality_trend" in stats
        assert "current_weights" in stats
        assert "history_length" in stats

    def test_record_feedback(self, manager, mock_db):
        mock_db.create_conversation.return_value = "conv1"
        mock_db.add_message.return_value = "msg1"
        mock_db.add_feedback.return_value = "fb1"
        result = manager.record_feedback("hi", "hello", "thumbs_up")
        assert result == "fb1"
        mock_db.add_feedback.assert_called_once()

    def test_record_feedback_with_conversation(self, manager, mock_db):
        mock_db.add_message.return_value = "msg1"
        mock_db.add_feedback.return_value = "fb2"
        result = manager.record_feedback("q", "a", "thumbs_down", conversation_id="existing")
        mock_db.create_conversation.assert_not_called()
        assert result == "fb2"


# ── Global singleton ──────────────────────────────────────────────────────

class TestGetMetaWeightManager:

    def test_returns_singleton(self):
        import domains.feedback.meta_weights as mod
        original = mod._meta_weight_manager
        mod._meta_weight_manager = None
        try:
            m1 = get_meta_weight_manager()
            m2 = get_meta_weight_manager()
            assert m1 is m2
        finally:
            mod._meta_weight_manager = original
