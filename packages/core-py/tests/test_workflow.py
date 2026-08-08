"""Tests for domains.feedback.workflow — automated feedback workflow manager."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from domains.feedback.workflow import (
    FeedbackWorkflowManager,
    WorkflowConfig,
    get_feedback_workflow,
)


class TestWorkflowConfig:
    def test_defaults(self):
        cfg = WorkflowConfig()
        assert cfg.aggregate_interval_minutes == 60
        assert cfg.prune_interval_minutes == 120
        assert cfg.auto_dpo_interval_minutes == 120
        assert cfg.export_interval_hours == 24
        assert cfg.health_check_interval_seconds == 30
        assert cfg.auto_aggregate_threshold == 50
        assert cfg.auto_prune_threshold == 100
        assert cfg.min_feedback_for_aggregation == 3
        assert cfg.export_format == "dpo"
        assert cfg.export_path == "data/training_exports"

    def test_custom(self):
        cfg = WorkflowConfig(aggregate_interval_minutes=30, export_format="jsonl")
        assert cfg.aggregate_interval_minutes == 30
        assert cfg.export_format == "jsonl"


class TestFeedbackWorkflowManager:
    @pytest.fixture()
    def manager(self):
        mock_db = MagicMock()
        mock_db.get_stats.return_value = {}
        mock_meta = MagicMock()
        mock_meta.record_feedback.return_value = "fb_123"
        mock_meta.get_stats.return_value = {}
        mock_lora = MagicMock()
        mock_lora.get_stats.return_value = {}
        mock_updater = MagicMock()
        mock_updater.get_stats.return_value = {}
        return FeedbackWorkflowManager(
            config=WorkflowConfig(),
            feedback_db=mock_db,
            meta_manager=mock_meta,
            lora_store=mock_lora,
            lora_updater=mock_updater,
        )

    def test_init(self, manager):
        assert manager.config is not None
        assert manager._running is False
        assert manager._stats["feedback_recorded"] == 0

    def test_init_default_config(self):
        mf = MagicMock()
        mf.get_stats.return_value = {}
        mm = MagicMock()
        mm.get_stats.return_value = {}
        ml = MagicMock()
        ml.get_stats.return_value = {}
        mu = MagicMock()
        mu.get_stats.return_value = {}
        mgr = FeedbackWorkflowManager(
            feedback_db=mf, meta_manager=mm, lora_store=ml, lora_updater=mu
        )
        assert mgr.config.aggregate_interval_minutes == 60

    def test_get_status(self, manager):
        status = manager.get_status()
        assert status["running"] is False
        assert "stats" in status
        assert "config" in status
        assert "last_runs" in status
        assert "systems" in status

    def test_get_status_running(self, manager):
        manager._running = True
        status = manager.get_status()
        assert status["running"] is True

    def test_record_feedback_calls_meta_manager(self, manager):
        feedback_id = manager.record_feedback(
            user_message="Hello",
            assistant_response="Hi there",
            rating="thumbs_up",
            user_id="user1",
        )
        assert feedback_id == "fb_123"
        manager.meta_manager.record_feedback.assert_called_once()

    def test_record_feedback_calls_lora_updater(self, manager):
        manager.record_feedback("Hello", "Hi", "thumbs_up", user_id="u1")
        manager.lora_updater.add_feedback.assert_called_once_with(
            prompt="Hello",
            response="Hi",
            rating="thumbs_up",
            quality_score=None,
        )

    def test_record_feedback_calls_lora_store(self, manager):
        manager.record_feedback("Hello", "Hi", "thumbs_up", user_id="u1")
        manager.lora_store.update_adapter.assert_called_once_with(
            user_id="u1", feedback_signal=1.0
        )

    def test_record_feedback_thumbs_down_negative_signal(self, manager):
        manager.record_feedback("Hello", "Hi", "thumbs_down", user_id="u1")
        manager.lora_store.update_adapter.assert_called_once_with(
            user_id="u1", feedback_signal=-1.0
        )

    def test_record_feedback_increments_stats(self, manager):
        manager.record_feedback("a", "b", "thumbs_up")
        assert manager._stats["feedback_recorded"] == 1
        manager.record_feedback("c", "d", "thumbs_down")
        assert manager._stats["feedback_recorded"] == 2

    def test_record_feedback_thumbs_up_increments_counter(self, manager):
        manager.record_feedback("a", "b", "thumbs_up")
        assert manager._new_thumbs_up == 1

    def test_record_feedback_thumbs_down_does_not_increment_up(self, manager):
        manager.record_feedback("a", "b", "thumbs_down")
        assert manager._new_thumbs_up == 0

    def test_trigger_aggregate(self, manager):
        result = manager.trigger_aggregate()
        assert result["status"] == "aggregated"
        assert "timestamp" in result

    def test_trigger_prune(self, manager):
        result = manager.trigger_prune()
        assert result["status"] == "pruned"

    def test_trigger_export(self, manager):
        result = manager.trigger_export()
        assert result["status"] == "exported"


class TestFeedbackWorkflowManagerConcurrency:
    def test_concurrent_record_feedback(self):
        mock_db = MagicMock()
        mock_db.get_stats.return_value = {}
        mock_meta = MagicMock()
        mock_meta.record_feedback.return_value = "fb"
        mock_meta.get_stats.return_value = {}
        mock_lora = MagicMock()
        mock_lora.get_stats.return_value = {}
        mock_updater = MagicMock()
        mock_updater.get_stats.return_value = {}
        mgr = FeedbackWorkflowManager(
            feedback_db=mock_db, meta_manager=mock_meta,
            lora_store=mock_lora, lora_updater=mock_updater,
        )
        errors = []

        def writer(n):
            try:
                for i in range(20):
                    mgr.record_feedback(f"msg_{n}_{i}", "resp", "thumbs_up")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(n,)) for n in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert mgr._stats["feedback_recorded"] == 60


class TestSingleton:
    def test_same_instance(self):
        a = get_feedback_workflow()
        b = get_feedback_workflow()
        assert a is b

    def test_singleton_type(self):
        mgr = get_feedback_workflow()
        assert isinstance(mgr, FeedbackWorkflowManager)
