"""Tests for training.status — enums, dataclasses, TrainingStatusTracker."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from domains.training.status import (
    TrainingStage, CompletionStatus, StageStatus,
    TrainingCompletionReport, TrainingStatusTracker,
    _tensors_to_numpy,
)
import numpy as np


# ── Enums ───────────────────────────────────────────────────────────────────


class TestEnums:

    def test_training_stage_values(self):
        assert TrainingStage.NOT_STARTED.value == "not_started"
        assert TrainingStage.PRETRAINING.value == "pretraining"
        assert TrainingStage.COMPLETE.value == "complete"

    def test_completion_status_values(self):
        assert CompletionStatus.IN_PROGRESS.value == "in_progress"
        assert CompletionStatus.COMPLETED.value == "completed"
        assert CompletionStatus.FAILED.value == "failed"


# ── StageStatus ─────────────────────────────────────────────────────────────


class TestStageStatus:

    def test_defaults(self):
        s = StageStatus(name="test")
        assert s.started_at is None
        assert s.completed_at is None
        assert s.epochs_completed == 0
        assert s.status == CompletionStatus.NOT_STARTED

    def test_custom(self):
        s = StageStatus(name="stage", total_epochs=10, best_loss=0.3)
        assert s.total_epochs == 10
        assert s.best_loss == 0.3


# ── TrainingCompletionReport ───────────────────────────────────────────────


class TestTrainingCompletionReport:

    def test_defaults(self):
        r = TrainingCompletionReport(model_name="test", created_at="2024-01-01")
        assert r.is_complete() is False
        assert r.can_resume() is False
        assert "not started" in r.get_progress_summary().lower()

    def test_is_complete(self):
        r = TrainingCompletionReport(
            model_name="test", created_at="2024",
            completion_status=CompletionStatus.COMPLETED,
        )
        assert r.is_complete() is True

    def test_can_resume_in_progress(self):
        r = TrainingCompletionReport(
            model_name="test", created_at="2024",
            completion_status=CompletionStatus.IN_PROGRESS,
            checkpoint_path="/path/ckpt",
        )
        assert r.can_resume() is True

    def test_can_resume_no_checkpoint(self):
        r = TrainingCompletionReport(
            model_name="test", created_at="2024",
            completion_status=CompletionStatus.IN_PROGRESS,
        )
        assert r.can_resume() is False

    def test_can_resume_interrupted(self):
        r = TrainingCompletionReport(
            model_name="test", created_at="2024",
            completion_status=CompletionStatus.INTERRUPTED,
            checkpoint_path="/path/ckpt",
        )
        assert r.can_resume() is True

    def test_progress_summary_completed(self):
        r = TrainingCompletionReport(
            model_name="test", created_at="2024",
            completion_status=CompletionStatus.COMPLETED,
            final_loss=0.1234,
        )
        assert "0.1234" in r.get_progress_summary()

    def test_progress_summary_in_progress(self):
        r = TrainingCompletionReport(
            model_name="test", created_at="2024",
            completion_status=CompletionStatus.IN_PROGRESS,
            completion_percentage=50.0,
        )
        assert "50.0%" in r.get_progress_summary()

    def test_progress_summary_interrupted(self):
        r = TrainingCompletionReport(
            model_name="test", created_at="2024",
            completion_status=CompletionStatus.INTERRUPTED,
            completion_percentage=75.0,
        )
        assert "75.0%" in r.get_progress_summary()


# ── TrainingStatusTracker ──────────────────────────────────────────────────


class TestTrainingStatusTracker:

    def test_init(self):
        tracker = TrainingStatusTracker("mymodel")
        assert tracker.model_name == "mymodel"
        assert tracker.report.completion_status == CompletionStatus.NOT_STARTED

    def test_start_training(self):
        tracker = TrainingStatusTracker()
        tracker.start_training(
            dataset="test.jsonl", batch_size=8, learning_rate=0.001,
            pretrain_epochs=5, federated_rounds=3, rlhf_epochs=2,
        )
        assert tracker.report.completion_status == CompletionStatus.IN_PROGRESS
        assert tracker.report.dataset == "test.jsonl"
        assert tracker.report.batch_size == 8
        assert tracker.report.pretraining is not None
        assert tracker.report.federated is not None
        assert tracker.report.rlhf is not None

    def test_start_training_no_stages(self):
        tracker = TrainingStatusTracker()
        tracker.start_training()
        assert tracker.report.pretraining is None
        assert tracker.report.federated is None
        assert tracker.report.rlhf is None

    def test_start_stage(self):
        tracker = TrainingStatusTracker()
        tracker.start_training(pretrain_epochs=5)
        tracker.start_stage(TrainingStage.PRETRAINING)
        assert tracker.report.pretraining.started_at is not None
        assert tracker.report.pretraining.status == CompletionStatus.IN_PROGRESS

    def test_update_stage(self):
        tracker = TrainingStatusTracker()
        tracker.start_training(pretrain_epochs=5)
        tracker.start_stage(TrainingStage.PRETRAINING)
        tracker.update_stage(TrainingStage.PRETRAINING, epoch=0, loss=0.5, val_loss=0.6)
        assert tracker.report.pretraining.epochs_completed == 1
        assert tracker.report.pretraining.final_loss == 0.5
        assert tracker.report.pretraining.best_loss == 0.6

    def test_update_stage_no_val_loss(self):
        tracker = TrainingStatusTracker()
        tracker.start_training(pretrain_epochs=5)
        tracker.start_stage(TrainingStage.PRETRAINING)
        tracker.update_stage(TrainingStage.PRETRAINING, epoch=0, loss=0.5)
        assert tracker.report.pretraining.best_loss == 0

    def test_complete_stage(self):
        tracker = TrainingStatusTracker()
        tracker.start_training(pretrain_epochs=5)
        tracker.start_stage(TrainingStage.PRETRAINING)
        tracker.update_stage(TrainingStage.PRETRAINING, epoch=4, loss=0.1, val_loss=0.15)
        tracker.complete_stage(TrainingStage.PRETRAINING)
        assert tracker.report.pretraining.status == CompletionStatus.COMPLETED
        assert tracker.report.pretraining.completed_at is not None
        assert tracker.report.final_loss == 0.1

    def test_fail_stage(self):
        tracker = TrainingStatusTracker()
        tracker.start_training(pretrain_epochs=5)
        tracker.start_stage(TrainingStage.PRETRAINING)
        tracker.fail_stage(TrainingStage.PRETRAINING, "OOM")
        assert tracker.report.pretraining.status == CompletionStatus.FAILED
        assert tracker.report.pretraining.error == "OOM"
        assert tracker.report.completion_status == CompletionStatus.FAILED
        assert any("OOM" in e for e in tracker.report.errors)

    def test_record_checkpoint(self):
        tracker = TrainingStatusTracker()
        tracker.record_checkpoint("/path/ckpt.soul", step=100, loss=0.3)
        assert tracker.report.checkpoint_path == "/path/ckpt.soul"
        assert tracker.report.last_checkpoint_step == 100
        assert tracker.report.checkpoint_count == 1
        assert len(tracker.checkpoints) == 1

    def test_mark_complete(self):
        tracker = TrainingStatusTracker()
        tracker.start_training()
        tracker.mark_complete()
        assert tracker.report.completion_status == CompletionStatus.COMPLETED
        assert tracker.report.completion_percentage == 100.0
        assert tracker.report.trained_at is not None

    def test_save_and_load_report(self, tmp_path):
        tracker = TrainingStatusTracker("save_test")
        tracker.start_training(dataset="test.jsonl", pretrain_epochs=3)
        tracker.start_stage(TrainingStage.PRETRAINING)
        tracker.update_stage(TrainingStage.PRETRAINING, epoch=1, loss=0.5)
        tracker.record_checkpoint("/path/ckpt", step=100, loss=0.5)

        path = str(tmp_path / "report.json")
        tracker.save_report(path)

        loaded = TrainingStatusTracker.load_report(path)
        assert loaded.report.model_name == "save_test"
        assert loaded.report.dataset == "test.jsonl"
        assert loaded.report.completion_status == CompletionStatus.IN_PROGRESS

    def test_get_report(self):
        tracker = TrainingStatusTracker()
        report = tracker.get_report()
        assert isinstance(report, TrainingCompletionReport)

    def test_auto_completion(self):
        tracker = TrainingStatusTracker()
        tracker.start_training(pretrain_epochs=2, federated_rounds=2)
        # Complete all stages
        tracker.start_stage(TrainingStage.PRETRAINING)
        tracker.update_stage(TrainingStage.PRETRAINING, epoch=0, loss=0.5)
        tracker.update_stage(TrainingStage.PRETRAINING, epoch=1, loss=0.3)
        tracker.complete_stage(TrainingStage.PRETRAINING)

        tracker.start_stage(TrainingStage.FEDERATED)
        tracker.update_stage(TrainingStage.FEDERATED, epoch=0, loss=0.2)
        tracker.update_stage(TrainingStage.FEDERATED, epoch=1, loss=0.1)
        tracker.complete_stage(TrainingStage.FEDERATED)

        assert tracker.report.completion_status == CompletionStatus.COMPLETED
        assert tracker.report.total_epochs == 4


# ── _tensors_to_numpy ──────────────────────────────────────────────────────


class TestTensorsToNumpy:

    def test_numpy_passthrough(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = _tensors_to_numpy({"a": arr})
        assert np.array_equal(result["a"], arr)

    def test_list_conversion(self):
        result = _tensors_to_numpy({"a": [1, 2, 3]})
        assert isinstance(result["a"], np.ndarray)

    def test_dict_conversion(self):
        result = _tensors_to_numpy({"a": {"b": np.array([1.0])}})
        assert isinstance(result["a"], dict)
        assert np.array_equal(result["a"]["b"], np.array([1.0]))
