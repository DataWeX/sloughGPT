"""Tests for domains.feedback.workflow — WorkflowConfig, FeedbackWorkflowManager."""

import copy
import time
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import fields

from domains.feedback.workflow import (
    WorkflowConfig,
    FeedbackWorkflowManager,
    get_feedback_workflow,
)


# ---------------------------------------------------------------------------
# WorkflowConfig
# ---------------------------------------------------------------------------

class TestWorkflowConfig:
    def test_defaults(self):
        cfg = WorkflowConfig()
        assert cfg.aggregate_interval_minutes == 60
        assert cfg.prune_interval_minutes == 120
        assert cfg.auto_dpo_interval_minutes == 120
        assert cfg.export_interval_hours == 24
        assert cfg.health_check_interval_seconds == 30
        assert cfg.background_training_interval_seconds == 300
        assert cfg.background_training_enabled is True
        assert cfg.auto_aggregate_threshold == 50
        assert cfg.auto_prune_threshold == 100
        assert cfg.min_feedback_for_aggregation == 3
        assert cfg.export_format == "dpo"
        assert cfg.export_path == "data/training_exports"

    def test_custom_values(self):
        cfg = WorkflowConfig(
            aggregate_interval_minutes=10,
            prune_interval_minutes=20,
            auto_dpo_interval_minutes=30,
            export_interval_hours=6,
            health_check_interval_seconds=10,
            background_training_interval_seconds=60,
            background_training_enabled=False,
            auto_aggregate_threshold=10,
            auto_prune_threshold=20,
            min_feedback_for_aggregation=2,
            export_format="sft",
            export_path="/tmp/exports",
        )
        assert cfg.aggregate_interval_minutes == 10
        assert cfg.prune_interval_minutes == 20
        assert cfg.auto_dpo_interval_minutes == 30
        assert cfg.export_interval_hours == 6
        assert cfg.health_check_interval_seconds == 10
        assert cfg.background_training_interval_seconds == 60
        assert cfg.background_training_enabled is False
        assert cfg.auto_aggregate_threshold == 10
        assert cfg.auto_prune_threshold == 20
        assert cfg.min_feedback_for_aggregation == 2
        assert cfg.export_format == "sft"
        assert cfg.export_path == "/tmp/exports"

    def test_all_fields_present(self):
        field_names = {f.name for f in fields(WorkflowConfig)}
        expected = {
            "aggregate_interval_minutes",
            "prune_interval_minutes",
            "auto_dpo_interval_minutes",
            "export_interval_hours",
            "health_check_interval_seconds",
            "background_training_interval_seconds",
            "background_training_enabled",
            "auto_aggregate_threshold",
            "auto_prune_threshold",
            "min_feedback_for_aggregation",
            "export_format",
            "export_path",
        }
        assert field_names == expected

    def test_is_dataclass(self):
        import dataclasses
        assert dataclasses.is_dataclass(WorkflowConfig)


# ---------------------------------------------------------------------------
# Helper to build a FeedbackWorkflowManager with mock dependencies
# ---------------------------------------------------------------------------

def _make_manager(**overrides):
    """Build a FeedbackWorkflowManager with lightweight mock dependencies."""
    db = MagicMock()
    db.get_stats.return_value = {}
    db.export_feedback_jsonl.return_value = None

    meta = MagicMock()
    meta.record_feedback.return_value = "fb-001"
    meta.get_stats.return_value = {}

    lora_store = MagicMock()
    lora_store.get_stats.return_value = {"total_users": 0}
    lora_store.aggregate_best_adapters.return_value = {
        "output_path": "/tmp/agg.npz",
        "sou_checkpoint": None,
        "user_count": 2,
        "total_feedback": 10,
        "weights": {},
        "eval": {"delta": {"verdict": "ok"}},
    }
    lora_store.prune_low_quality.return_value = []

    lora_updater = MagicMock()
    lora_updater.get_stats.return_value = {}

    cfg = overrides.pop("config", WorkflowConfig())
    mgr = FeedbackWorkflowManager(
        config=cfg,
        feedback_db=db,
        meta_manager=meta,
        lora_store=lora_store,
        lora_updater=lora_updater,
    )
    return mgr


# ---------------------------------------------------------------------------
# FeedbackWorkflowManager — initialisation
# ---------------------------------------------------------------------------

class TestManagerInit:
    def test_default_config(self):
        mgr = _make_manager()
        assert isinstance(mgr.config, WorkflowConfig)
        assert mgr._running is False
        assert mgr._new_thumbs_up == 0
        assert mgr._auto_train_threshold == 3

    def test_custom_config_applied(self):
        cfg = WorkflowConfig(aggregate_interval_minutes=5)
        mgr = _make_manager(config=cfg)
        assert mgr.config.aggregate_interval_minutes == 5

    def test_injected_dependencies(self):
        mgr = _make_manager()
        assert mgr.db is not None
        assert mgr.meta_manager is not None
        assert mgr.lora_store is not None
        assert mgr.lora_updater is not None

    def test_stats_initialised(self):
        mgr = _make_manager()
        assert mgr._stats["workflow_runs"] == 0
        assert mgr._stats["aggregations_performed"] == 0
        assert mgr._stats["prunes_performed"] == 0
        assert mgr._stats["exports_performed"] == 0
        assert mgr._stats["feedback_recorded"] == 0
        assert mgr._stats["auto_train_steps"] == 0
        assert mgr._stats["dpo_train_steps"] == 0
        assert mgr._stats["dpo_train_rejected"] == 0
        assert mgr._stats["user_adapter_trained"] == 0
        assert mgr._stats["user_adapter_rejected"] == 0
        assert mgr._stats["start_time"] is None

    def test_timestamps_initialised_to_zero(self):
        mgr = _make_manager()
        assert mgr._last_aggregate_time == 0
        assert mgr._last_prune_time == 0
        assert mgr._last_export_time == 0
        assert mgr._last_dpo_time == 0
        assert mgr._last_rollback_time == 0.0
        assert mgr._last_health_check == 0


# ---------------------------------------------------------------------------
# set_model
# ---------------------------------------------------------------------------

class TestSetModel:
    def test_sets_model_and_tokenizer(self):
        mgr = _make_manager()
        model = MagicMock()
        tokenizer = MagicMock()
        mgr.set_model(model, tokenizer)
        assert mgr._model is model
        assert mgr._tokenizer is tokenizer

    def test_overwrites_previous(self):
        mgr = _make_manager()
        m1, t1 = MagicMock(), MagicMock()
        m2, t2 = MagicMock(), MagicMock()
        mgr.set_model(m1, t1)
        mgr.set_model(m2, t2)
        assert mgr._model is m2
        assert mgr._tokenizer is t2


# ---------------------------------------------------------------------------
# record_feedback — pure logic path
# ---------------------------------------------------------------------------

class TestRecordFeedback:
    def test_calls_meta_manager_record_feedback(self):
        mgr = _make_manager()
        fb_id = mgr.record_feedback(
            user_message="hi",
            assistant_response="hello",
            rating="thumbs_up",
            conversation_id="conv-1",
            quality_score=0.9,
            user_id="u1",
        )
        mgr.meta_manager.record_feedback.assert_called_once_with(
            user_message="hi",
            assistant_response="hello",
            rating="thumbs_up",
            conversation_id="conv-1",
            quality_score=0.9,
            user_id="u1",
        )
        assert fb_id == "fb-001"

    def test_calls_lora_updater_add_feedback(self):
        mgr = _make_manager()
        mgr.record_feedback("q", "a", "thumbs_down")
        mgr.lora_updater.add_feedback.assert_called_once_with(
            prompt="q",
            response="a",
            rating="thumbs_down",
            quality_score=None,
        )

    def test_lora_store_update_adapter_thumbs_up(self):
        mgr = _make_manager()
        mgr.record_feedback("q", "a", "thumbs_up", user_id="u1")
        mgr.lora_store.update_adapter.assert_called_once_with(
            user_id="u1",
            feedback_signal=1.0,
        )

    def test_lora_store_update_adapter_thumbs_down(self):
        mgr = _make_manager()
        mgr.record_feedback("q", "a", "thumbs_down", user_id="u1")
        mgr.lora_store.update_adapter.assert_called_once_with(
            user_id="u1",
            feedback_signal=-1.0,
        )

    def test_stats_feedback_recorded_increments(self):
        mgr = _make_manager()
        assert mgr._stats["feedback_recorded"] == 0
        mgr.record_feedback("q", "a", "thumbs_up")
        assert mgr._stats["feedback_recorded"] == 1
        mgr.record_feedback("q", "a", "thumbs_down")
        assert mgr._stats["feedback_recorded"] == 2

    def test_thumbs_up_increments_pending_counter(self):
        mgr = _make_manager()
        mgr.record_feedback("q", "a", "thumbs_up")
        assert mgr._new_thumbs_up == 1

    def test_thumbs_down_does_not_increment_pending_counter(self):
        mgr = _make_manager()
        mgr.record_feedback("q", "a", "thumbs_down")
        assert mgr._new_thumbs_up == 0

    def test_returns_feedback_id(self):
        mgr = _make_manager()
        result = mgr.record_feedback("q", "a", "thumbs_up")
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# _snapshot_weights / _restore_weights
# ---------------------------------------------------------------------------

class TestSnapshotRestore:
    def _make_layer(self, data):
        layer = MagicMock()
        layer.weight = MagicMock()
        layer.weight.data = data
        return layer

    def test_snapshot_captures_all_weight_layers(self):
        mgr = _make_manager()
        w1 = np.array([1.0, 2.0])
        w2 = np.array([3.0, 4.0])
        net = MagicMock()
        net.layers = [self._make_layer(w1), self._make_layer(w2)]
        snap = mgr._snapshot_weights(net)
        assert "0" in snap
        assert "1" in snap
        np.testing.assert_array_equal(snap["0"], w1)
        np.testing.assert_array_equal(snap["1"], w2)

    def test_snapshot_skips_layers_without_weight(self):
        mgr = _make_manager()
        layer_no_weight = MagicMock(spec=[])  # no 'weight' attribute
        net = MagicMock()
        net.layers = [layer_no_weight]
        snap = mgr._snapshot_weights(net)
        assert snap == {}

    def test_snapshot_is_deep_copy(self):
        mgr = _make_manager()
        w = np.array([1.0, 2.0])
        layer = self._make_layer(w)
        net = MagicMock()
        net.layers = [layer]
        snap = mgr._snapshot_weights(net)
        w[0] = 999.0
        assert snap["0"][0] == 1.0  # unchanged

    def test_restore_weights(self):
        mgr = _make_manager()
        original = np.array([1.0, 2.0])
        new = np.array([10.0, 20.0])
        layer = self._make_layer(original)
        net = MagicMock()
        net.layers = [layer]
        snap = {"0": new.copy()}
        mgr._restore_weights(net, snap)
        np.testing.assert_array_equal(layer.weight.data, new)

    def test_restore_skips_missing_keys(self):
        mgr = _make_manager()
        w = np.array([1.0])
        layer = self._make_layer(w)
        net = MagicMock()
        net.layers = [layer]
        mgr._restore_weights(net, {})
        np.testing.assert_array_equal(layer.weight.data, w)

    def test_restore_skips_layers_without_weight(self):
        mgr = _make_manager()
        layer = MagicMock(spec=[])  # no weight
        net = MagicMock()
        net.layers = [layer]
        mgr._restore_weights(net, {"0": np.array([1.0])})
        # no error


# ---------------------------------------------------------------------------
# run_scheduled_tasks — time-based state machine
# ---------------------------------------------------------------------------

class TestRunScheduledTasks:
    def test_aggregation_runs_when_interval_elapsed(self):
        mgr = _make_manager(config=WorkflowConfig(aggregate_interval_minutes=1))
        mgr._last_aggregate_time = time.time() - 200  # well past 60s
        mgr.run_scheduled_tasks()
        mgr.lora_store.aggregate_best_adapters.assert_called_once()

    def test_aggregation_skipped_when_interval_not_elapsed(self):
        mgr = _make_manager(config=WorkflowConfig(aggregate_interval_minutes=60))
        mgr._last_aggregate_time = time.time()  # just set
        mgr.run_scheduled_tasks()
        mgr.lora_store.aggregate_best_adapters.assert_not_called()

    def test_pruning_runs_when_interval_elapsed(self):
        mgr = _make_manager(config=WorkflowConfig(prune_interval_minutes=1))
        mgr._last_prune_time = time.time() - 200
        mgr.run_scheduled_tasks()
        mgr.lora_store.prune_low_quality.assert_called_once()

    def test_pruning_skipped_when_interval_not_elapsed(self):
        mgr = _make_manager(config=WorkflowConfig(prune_interval_minutes=120))
        mgr._last_prune_time = time.time()
        mgr.run_scheduled_tasks()
        mgr.lora_store.prune_low_quality.assert_not_called()

    def test_export_runs_when_interval_elapsed(self):
        mgr = _make_manager(config=WorkflowConfig(export_interval_hours=0))
        mgr._last_export_time = 0
        mgr.run_scheduled_tasks()
        mgr.db.export_feedback_jsonl.assert_called_once()

    def test_export_skipped_when_interval_not_elapsed(self):
        mgr = _make_manager(config=WorkflowConfig(export_interval_hours=24))
        mgr._last_export_time = time.time()
        mgr.run_scheduled_tasks()
        mgr.db.export_feedback_jsonl.assert_not_called()

    def test_dpo_runs_when_interval_elapsed(self):
        mgr = _make_manager(config=WorkflowConfig(auto_dpo_interval_minutes=0))
        mgr._last_dpo_time = 0
        # _maybe_dpo_train returns early when no model, so no error
        mgr.run_scheduled_tasks()

    def test_stats_updated_after_aggregation(self):
        mgr = _make_manager(config=WorkflowConfig(aggregate_interval_minutes=0))
        mgr._last_aggregate_time = 0
        mgr.run_scheduled_tasks()
        assert mgr._stats["aggregations_performed"] == 1

    def test_multiple_intervals_independent(self):
        mgr = _make_manager(config=WorkflowConfig(
            aggregate_interval_minutes=0,
            prune_interval_minutes=999,
            export_interval_hours=999,
            auto_dpo_interval_minutes=999,
        ))
        mgr._last_aggregate_time = 0
        mgr._last_prune_time = time.time()
        mgr._last_export_time = time.time()
        mgr._last_dpo_time = time.time()
        mgr.run_scheduled_tasks()
        mgr.lora_store.aggregate_best_adapters.assert_called_once()
        mgr.lora_store.prune_low_quality.assert_not_called()
        mgr.db.export_feedback_jsonl.assert_not_called()


# ---------------------------------------------------------------------------
# _health_check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_increments_workflow_runs(self):
        mgr = _make_manager()
        mgr._health_check()
        assert mgr._stats["workflow_runs"] == 1

    def test_updates_last_health_check(self):
        mgr = _make_manager()
        before = time.time()
        mgr._health_check()
        after = time.time()
        assert before <= mgr._last_health_check <= after

    def test_calls_run_scheduled_tasks(self):
        mgr = _make_manager()
        with patch.object(mgr, "run_scheduled_tasks") as mock:
            mgr._health_check()
            mock.assert_called_once()


# ---------------------------------------------------------------------------
# _do_aggregate
# ---------------------------------------------------------------------------

class TestDoAggregate:
    def test_calls_lora_store_aggregate_best_adapters(self):
        mgr = _make_manager()
        mgr._do_aggregate()
        mgr.lora_store.aggregate_best_adapters.assert_called_once()

    def test_increments_aggregations_performed(self):
        mgr = _make_manager()
        mgr._do_aggregate()
        assert mgr._stats["aggregations_performed"] == 1

    def test_exception_does_not_propagate(self):
        mgr = _make_manager()
        mgr.lora_store.aggregate_best_adapters.side_effect = RuntimeError("fail")
        mgr._do_aggregate()  # should not raise
        assert mgr._stats["aggregations_performed"] == 0


# ---------------------------------------------------------------------------
# _do_prune
# ---------------------------------------------------------------------------

class TestDoPrune:
    def test_calls_lora_store_prune_low_quality(self):
        mgr = _make_manager()
        mgr._do_prune()
        mgr.lora_store.prune_low_quality.assert_called_once_with(
            min_feedback_count=1,
            max_age_days=7,
        )

    def test_increments_prunes_when_deleted(self):
        mgr = _make_manager()
        mgr.lora_store.prune_low_quality.return_value = ["user1"]
        mgr._do_prune()
        assert mgr._stats["prunes_performed"] == 1

    def test_no_increment_when_no_deletions(self):
        mgr = _make_manager()
        mgr.lora_store.prune_low_quality.return_value = []
        mgr._do_prune()
        assert mgr._stats["prunes_performed"] == 0

    def test_exception_does_not_propagate(self):
        mgr = _make_manager()
        mgr.lora_store.prune_low_quality.side_effect = RuntimeError("fail")
        mgr._do_prune()  # should not raise


# ---------------------------------------------------------------------------
# _do_export
# ---------------------------------------------------------------------------

class TestDoExport:
    def test_calls_db_export_feedback_jsonl(self):
        mgr = _make_manager()
        mgr._do_export()
        mgr.db.export_feedback_jsonl.assert_called_once()
        args = mgr.db.export_feedback_jsonl.call_args
        assert "feedback_export_" in args[0][0]

    def test_increments_exports_performed(self):
        mgr = _make_manager()
        mgr._do_export()
        assert mgr._stats["exports_performed"] == 1

    def test_exception_does_not_propagate(self):
        mgr = _make_manager()
        mgr.db.export_feedback_jsonl.side_effect = RuntimeError("fail")
        mgr._do_export()  # should not raise


# ---------------------------------------------------------------------------
# trigger_* manual methods
# ---------------------------------------------------------------------------

class TestTriggerMethods:
    def test_trigger_aggregate(self):
        mgr = _make_manager()
        result = mgr.trigger_aggregate()
        assert result["status"] == "aggregated"
        assert "timestamp" in result
        mgr.lora_store.aggregate_best_adapters.assert_called_once()

    def test_trigger_prune(self):
        mgr = _make_manager()
        result = mgr.trigger_prune()
        assert result["status"] == "pruned"
        assert "timestamp" in result
        mgr.lora_store.prune_low_quality.assert_called_once()

    def test_trigger_export(self):
        mgr = _make_manager()
        result = mgr.trigger_export()
        assert result["status"] == "exported"
        assert "timestamp" in result
        mgr.db.export_feedback_jsonl.assert_called_once()


# ---------------------------------------------------------------------------
# get_status
# ---------------------------------------------------------------------------

class TestGetStatus:
    def test_running_key(self):
        mgr = _make_manager()
        status = mgr.get_status()
        assert status["running"] is False

    def test_stats_key(self):
        mgr = _make_manager()
        status = mgr.get_status()
        assert isinstance(status["stats"], dict)
        assert "feedback_recorded" in status["stats"]

    def test_stats_is_copy(self):
        mgr = _make_manager()
        status = mgr.get_status()
        status["stats"]["feedback_recorded"] = 999
        assert mgr._stats["feedback_recorded"] == 0

    def test_pending_thumbs_up(self):
        mgr = _make_manager()
        mgr._new_thumbs_up = 2
        status = mgr.get_status()
        assert status["pending_thumbs_up"] == 2

    def test_auto_train_threshold(self):
        mgr = _make_manager()
        status = mgr.get_status()
        assert status["auto_train_threshold"] == 3

    def test_config_section(self):
        mgr = _make_manager()
        status = mgr.get_status()
        cfg = status["config"]
        assert cfg["aggregate_interval_minutes"] == 60
        assert cfg["prune_interval_minutes"] == 120
        assert cfg["export_interval_hours"] == 24
        assert cfg["auto_dpo_interval_minutes"] == 120
        assert cfg["health_check_interval_seconds"] == 30
        assert cfg["background_training_interval_seconds"] == 300
        assert cfg["background_training_enabled"] is True

    def test_last_runs_section(self):
        mgr = _make_manager()
        status = mgr.get_status()
        lr = status["last_runs"]
        assert "aggregate" in lr
        assert "prune" in lr
        assert "export" in lr
        assert "dpo" in lr
        assert "health_check" in lr
        assert "last_rollback" in lr
        assert "background_training" in lr

    def test_systems_section(self):
        mgr = _make_manager()
        status = mgr.get_status()
        sys = status["systems"]
        assert "feedback_db" in sys
        assert "meta_weights" in sys
        assert "lora_store" in sys
        assert "lora_updater" in sys


# ---------------------------------------------------------------------------
# start / stop
# ---------------------------------------------------------------------------

class TestStartStop:
    def test_start_sets_running(self):
        mgr = _make_manager(config=WorkflowConfig(
            health_check_interval_seconds=999,
            background_training_enabled=False,
        ))
        mgr.start()
        assert mgr._running is True
        mgr.stop()

    def test_start_sets_start_time(self):
        mgr = _make_manager(config=WorkflowConfig(
            health_check_interval_seconds=999,
            background_training_enabled=False,
        ))
        before = time.time()
        mgr.start()
        after = time.time()
        assert mgr._stats["start_time"] is not None
        assert before <= mgr._stats["start_time"] <= after
        mgr.stop()

    def test_stop_sets_running_false(self):
        mgr = _make_manager(config=WorkflowConfig(
            health_check_interval_seconds=999,
            background_training_enabled=False,
        ))
        mgr.start()
        mgr.stop()
        assert mgr._running is False

    def test_start_idempotent(self):
        mgr = _make_manager(config=WorkflowConfig(
            health_check_interval_seconds=999,
            background_training_enabled=False,
        ))
        mgr.start()
        mgr.start()  # second call should be no-op
        assert mgr._running is True
        mgr.stop()

    def test_background_training_thread_created_when_enabled(self):
        mgr = _make_manager(config=WorkflowConfig(
            health_check_interval_seconds=999,
            background_training_enabled=True,
            background_training_interval_seconds=999,
        ))
        mgr.start()
        assert hasattr(mgr, "_training_thread")
        assert mgr._training_thread.is_alive()
        mgr.stop()

    def test_background_training_not_created_when_disabled(self):
        mgr = _make_manager(config=WorkflowConfig(
            health_check_interval_seconds=999,
            background_training_enabled=False,
        ))
        mgr.start()
        assert not hasattr(mgr, "_training_thread") or not mgr._training_thread.is_alive()
        mgr.stop()


# ---------------------------------------------------------------------------
# get_feedback_workflow singleton
# ---------------------------------------------------------------------------

class TestGetFeedbackWorkflow:
    def test_returns_manager(self):
        import domains.feedback.workflow as mod
        original = mod._workflow_manager
        try:
            mod._workflow_manager = None
            mgr = get_feedback_workflow()
            assert isinstance(mgr, FeedbackWorkflowManager)
        finally:
            mod._workflow_manager = original

    def test_returns_same_instance(self):
        import domains.feedback.workflow as mod
        original = mod._workflow_manager
        try:
            mod._workflow_manager = None
            mgr1 = get_feedback_workflow()
            mgr2 = get_feedback_workflow()
            assert mgr1 is mgr2
        finally:
            mod._workflow_manager = original

    def test_accepts_config(self):
        import domains.feedback.workflow as mod
        original = mod._workflow_manager
        try:
            mod._workflow_manager = None
            cfg = WorkflowConfig(aggregate_interval_minutes=7)
            mgr = get_feedback_workflow(config=cfg)
            assert mgr.config.aggregate_interval_minutes == 7
        finally:
            mod._workflow_manager = original


# ---------------------------------------------------------------------------
# record_feedback — thumbs_up auto-train threshold
# ---------------------------------------------------------------------------

class TestAutoTrainThreshold:
    def test_thumbs_up_resets_counter_at_threshold(self):
        mgr = _make_manager()
        mgr._auto_train_threshold = 3
        mgr._new_thumbs_up = 0
        mgr.record_feedback("q", "a", "thumbs_up")
        assert mgr._new_thumbs_up == 1
        mgr.record_feedback("q", "a", "thumbs_up")
        assert mgr._new_thumbs_up == 2
        mgr.record_feedback("q", "a", "thumbs_up")
        # After hitting threshold, counter resets
        assert mgr._new_thumbs_up == 0

    def test_thumbs_down_does_not_affect_thumbs_up_counter(self):
        mgr = _make_manager()
        mgr.record_feedback("q", "a", "thumbs_up")
        mgr.record_feedback("q", "a", "thumbs_up")
        mgr.record_feedback("q", "a", "thumbs_down")
        assert mgr._new_thumbs_up == 2


# ---------------------------------------------------------------------------
# run_scheduled_tasks — exception safety
# ---------------------------------------------------------------------------

class TestScheduledTasksExceptionSafety:
    def test_aggregate_exception_does_not_block_prune(self):
        mgr = _make_manager(config=WorkflowConfig(
            aggregate_interval_minutes=0,
            prune_interval_minutes=0,
            export_interval_hours=999,
            auto_dpo_interval_minutes=999,
        ))
        mgr._last_aggregate_time = 0
        mgr._last_prune_time = 0
        mgr.lora_store.aggregate_best_adapters.side_effect = RuntimeError("boom")
        mgr.run_scheduled_tasks()
        mgr.lora_store.prune_low_quality.assert_called_once()

    def test_prune_exception_does_not_block_export(self):
        mgr = _make_manager(config=WorkflowConfig(
            aggregate_interval_minutes=999,
            prune_interval_minutes=0,
            export_interval_hours=0,
            auto_dpo_interval_minutes=999,
        ))
        mgr._last_prune_time = 0
        mgr._last_export_time = 0
        mgr.lora_store.prune_low_quality.side_effect = RuntimeError("boom")
        mgr.run_scheduled_tasks()
        mgr.db.export_feedback_jsonl.assert_called_once()
