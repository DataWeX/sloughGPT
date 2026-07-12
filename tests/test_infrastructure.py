"""Tests for infrastructure safety: model_loader integrity checks, meta_weights manager."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import numpy as np
import torch
import pytest


# =============================================================================
# model_loader Tests
# =============================================================================


class TestModelLoaderPlatformDetection:
    def test_mps_available_returns_bool(self):
        from domains.infrastructure.ml_types import _mps_available
        assert isinstance(_mps_available(), bool)

    def test_cuda_available_returns_bool(self):
        from domains.infrastructure.ml_types import _cuda_available
        assert isinstance(_cuda_available(), bool)

    def test_mps_available_handles_exception(self):
        from domains.infrastructure.ml_types import _mps_available
        with patch("torch.backends.mps.is_available", side_effect=AttributeError("no mps")):
            assert _mps_available() is False

    def test_cuda_available_handles_exception(self):
        from domains.infrastructure.ml_types import _cuda_available
        with patch("torch.cuda.is_available", side_effect=AttributeError("no cuda")):
            assert _cuda_available() is False


# =============================================================================
# MetaWeightManager Tests
# =============================================================================


@pytest.fixture(autouse=True)
def _reset_feedback_db_global():
    """Reset FeedbackDB singleton before each test to avoid path pollution."""
    import domains.feedback.database as db_mod
    db_mod._feedback_db = None
    yield
    db_mod._feedback_db = None


class TestMetaWeights:
    def test_defaults(self):
        from domains.feedback.meta_weights import MetaWeights
        mw = MetaWeights()
        assert mw.temperature == 0.8
        assert mw.repetition_penalty == 1.0
        assert mw.top_p == 0.9
        assert mw.top_k == 50
        assert mw.style_bias == 0.0

    def test_custom(self):
        from domains.feedback.meta_weights import MetaWeights
        mw = MetaWeights(temperature=1.2, style_bias=0.5)
        assert mw.temperature == 1.2
        assert mw.style_bias == 0.5


class TestMetaWeightManagerSimple:
    def test_init(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "feedback.db")
            from domains.feedback.meta_weights import MetaWeightManager
            mwm = MetaWeightManager(db_path=db_path)
            assert mwm.embedding_dim == 384
            assert mwm.use_simple_search is True

    def test_simple_embed_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            from domains.feedback.meta_weights import MetaWeightManager
            mwm = MetaWeightManager(db_path=str(Path(tmp) / "feedback.db"))
            emb = mwm._simple_embed("hello world")
            assert emb.shape == (384,)

    def test_simple_embed_normalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            from domains.feedback.meta_weights import MetaWeightManager
            mwm = MetaWeightManager(db_path=str(Path(tmp) / "feedback.db"))
            emb = mwm._simple_embed("this is a test message")
            norm = np.linalg.norm(emb)
            assert abs(norm - 1.0) < 0.01

    def test_simple_embed_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            from domains.feedback.meta_weights import MetaWeightManager
            mwm = MetaWeightManager(db_path=str(Path(tmp) / "feedback.db"))
            emb = mwm._simple_embed("")
            assert np.linalg.norm(emb) == 0.0

    def test_simple_embed_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            from domains.feedback.meta_weights import MetaWeightManager
            mwm = MetaWeightManager(db_path=str(Path(tmp) / "feedback.db"))
            emb1 = mwm._simple_embed("hello world")
            emb2 = mwm._simple_embed("hello world")
            assert np.allclose(emb1, emb2)

    def test_aggregate_patterns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            from domains.feedback.meta_weights import MetaWeightManager
            mwm = MetaWeightManager(db_path=str(Path(tmp) / "feedback.db"))
            assert mwm._aggregate_patterns([]) == {}

    def test_aggregate_patterns_thumbs_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            from domains.feedback.meta_weights import MetaWeightManager
            from domains.feedback.database import SimilarPattern
            mwm = MetaWeightManager(db_path=str(Path(tmp) / "feedback.db"))
            patterns = [SimilarPattern(content="good", rating="thumbs_up", similarity=0.8, pattern_type="msg")]
            result = mwm._aggregate_patterns(patterns)
            assert result["temperature_boost"] > 0

    def test_aggregate_patterns_thumbs_down(self):
        with tempfile.TemporaryDirectory() as tmp:
            from domains.feedback.meta_weights import MetaWeightManager
            from domains.feedback.database import SimilarPattern
            mwm = MetaWeightManager(db_path=str(Path(tmp) / "feedback.db"))
            patterns = [SimilarPattern(content="bad", rating="thumbs_down", similarity=0.9, pattern_type="msg")]
            result = mwm._aggregate_patterns(patterns)
            assert result["temperature_boost"] < 0

    def test_get_adjustment_no_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            from domains.feedback.meta_weights import MetaWeightManager
            mwm = MetaWeightManager(db_path=str(Path(tmp) / "feedback.db"))
            weights = mwm.get_adjustment("hello", k=5)
            assert weights.temperature == 0.8
            assert weights.repetition_penalty == 1.0

    def test_get_adjustment_clamps_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            from domains.feedback.meta_weights import MetaWeightManager
            from domains.feedback.database import SimilarPattern
            mwm = MetaWeightManager(db_path=str(Path(tmp) / "feedback.db"))
            patterns = [SimilarPattern(content="good", rating="thumbs_up", similarity=1.0, pattern_type="msg")]
            # Turn off text fallback so only vector search is used
            mwm.use_simple_search = False
            with patch.object(mwm.db, "find_similar_messages", return_value=patterns):
                weights = mwm.get_adjustment("hello" * 100, k=5)
                assert 0.1 <= weights.temperature <= 2.0
                assert 0.8 <= weights.repetition_penalty <= 1.5

    def test_record_feedback_creates_feedback(self):
        with tempfile.TemporaryDirectory() as tmp:
            from domains.feedback.meta_weights import MetaWeightManager
            mwm = MetaWeightManager(db_path=str(Path(tmp) / "feedback.db"))
            fb_id = mwm.record_feedback("hello", "hi there", "thumbs_up", user_id="user1")
            assert fb_id is not None
            all_fb = mwm.db.get_all_feedback()
            assert len(all_fb) >= 1
            assert all_fb[0]["rating"] == "thumbs_up"

    def test_record_feedback_creates_conversation(self):
        with tempfile.TemporaryDirectory() as tmp:
            from domains.feedback.meta_weights import MetaWeightManager
            mwm = MetaWeightManager(db_path=str(Path(tmp) / "feedback.db"))
            mwm.record_feedback("hello", "world", "thumbs_up")
            assert len(mwm.db.list_conversations(limit=5)) >= 1

    def test_record_feedback_updates_user_meta_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            from domains.feedback.meta_weights import MetaWeightManager
            mwm = MetaWeightManager(db_path=str(Path(tmp) / "feedback.db"))
            mwm.record_feedback("hi", "hello", "thumbs_up", user_id="user1")
            uw = mwm.db.get_user_meta_weights("user1")
            assert uw is not None
            assert uw["thumbs_up_count"] == 1

    def test_get_quality_trend_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            from domains.feedback.meta_weights import MetaWeightManager
            mwm = MetaWeightManager(db_path=str(Path(tmp) / "feedback.db"))
            trend = mwm.get_quality_trend(window=10)
            assert trend["trend"] == 0.0
            assert trend["thumbs_up_ratio"] == 0.0

    def test_get_quality_trend_with_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            from domains.feedback.meta_weights import MetaWeightManager
            mwm = MetaWeightManager(db_path=str(Path(tmp) / "feedback.db"))
            for i in range(3):
                mwm.record_feedback(f"msg{i}", f"resp{i}", "thumbs_up")
            trend = mwm.get_quality_trend(window=10)
            assert trend["thumbs_up_ratio"] == 1.0
            assert trend["sample_count"] == 3

    def test_get_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            from domains.feedback.meta_weights import MetaWeightManager
            mwm = MetaWeightManager(db_path=str(Path(tmp) / "feedback.db"))
            mwm.record_feedback("hi", "hello", "thumbs_up")
            mwm.get_adjustment("test")
            stats = mwm.get_stats()
            assert "db_stats" in stats
            assert "quality_trend" in stats
            assert "current_weights" in stats
            assert stats["history_length"] >= 1

    def test_export_training_data_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            from domains.feedback.meta_weights import MetaWeightManager
            mwm = MetaWeightManager(db_path=str(Path(tmp) / "feedback.db"))
            mwm.record_feedback("hi", "hello", "thumbs_up")
            export_path = str(Path(tmp) / "export.jsonl")
            mwm.export_training_data(export_path, format="jsonl")
            assert Path(export_path).exists()

    def test_get_adjustment_with_user_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            from domains.feedback.meta_weights import MetaWeightManager
            mwm = MetaWeightManager(db_path=str(Path(tmp) / "feedback.db"))
            mwm.record_feedback("hi", "hello", "thumbs_up", user_id="user1")
            weights = mwm.get_adjustment("test", user_id="user1")
            assert weights.temperature >= 0.8

    def test_get_adjustment_fallback_to_simple_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            from domains.feedback.meta_weights import MetaWeightManager
            mwm = MetaWeightManager(db_path=str(Path(tmp) / "feedback.db"))
            conv_id = mwm.db.create_conversation()
            mid = mwm.db.add_message(conv_id, "user", "hello world python")
            mwm.db.add_feedback(mid, "thumbs_up")
            with patch.object(mwm.db, "find_similar_messages", return_value=[]):
                weights = mwm.get_adjustment("python", k=5)
                assert isinstance(weights.temperature, float)

    def test_get_adjustment_history_trimming(self):
        with tempfile.TemporaryDirectory() as tmp:
            from domains.feedback.meta_weights import MetaWeightManager
            mwm = MetaWeightManager(db_path=str(Path(tmp) / "feedback.db"))
            for i in range(150):
                mwm._weight_history.append({"temperature": 0.8, "repetition_penalty": 1.0, "pattern_count": 0})
            mwm.get_adjustment("hello")
            assert len(mwm._weight_history) <= 100


class TestMetaWeightManagerEmbed:
    def test_embed_uses_simple_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            from domains.feedback.meta_weights import MetaWeightManager
            mwm = MetaWeightManager(db_path=str(Path(tmp) / "feedback.db"))
            mwm._embed_model = None
            emb = mwm._embed("test message")
            assert emb.shape == (384,)
