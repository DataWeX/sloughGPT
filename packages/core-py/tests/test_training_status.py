"""Tests for domains.training.status — TrainingStatusTracker, enums, report."""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from domains.training.status import (
    TrainingStage,
    CompletionStatus,
    StageStatus,
    TrainingCompletionReport,
    TrainingStatusTracker,
)


# ── Enums ─────────────────────────────────────────────────────────────────────

class TestEnums:
    def test_training_stage_values(self):
        assert TrainingStage.NOT_STARTED.value == "not_started"
        assert TrainingStage.PRETRAINING.value == "pretraining"
        assert TrainingStage.COMPLETE.value == "complete"

    def test_completion_status_values(self):
        assert CompletionStatus.IN_PROGRESS.value == "in_progress"
        assert CompletionStatus.COMPLETED.value == "completed"
        assert CompletionStatus.FAILED.value == "failed"


# ── TrainingCompletionReport ──────────────────────────────────────────────────

class TestTrainingCompletionReport:
    def test_is_complete(self):
        report = TrainingCompletionReport(
            model_name="test",
            created_at="2024-01-01T00:00:00Z",
            completion_status=CompletionStatus.COMPLETED,
        )
        assert report.is_complete() is True

    def test_is_not_complete(self):
        report = TrainingCompletionReport(
            model_name="test",
            created_at="2024-01-01T00:00:00Z",
        )
        assert report.is_complete() is False

    def test_can_resume_in_progress(self):
        report = TrainingCompletionReport(
            model_name="test",
            created_at="2024-01-01T00:00:00Z",
            completion_status=CompletionStatus.IN_PROGRESS,
            checkpoint_path="/tmp/ckpt.npz",
        )
        assert report.can_resume() is True

    def test_can_resume_no_checkpoint(self):
        report = TrainingCompletionReport(
            model_name="test",
            created_at="2024-01-01T00:00:00Z",
            completion_status=CompletionStatus.IN_PROGRESS,
        )
        assert report.can_resume() is False

    def test_can_resume_completed(self):
        report = TrainingCompletionReport(
            model_name="test",
            created_at="2024-01-01T00:00:00Z",
            completion_status=CompletionStatus.COMPLETED,
            checkpoint_path="/tmp/ckpt.npz",
        )
        assert report.can_resume() is False

    def test_get_progress_summary_complete(self):
        report = TrainingCompletionReport(
            model_name="test",
            created_at="2024-01-01T00:00:00Z",
            completion_status=CompletionStatus.COMPLETED,
            final_loss=0.5,
        )
        assert "complete" in report.get_progress_summary().lower()

    def test_get_progress_summary_in_progress(self):
        report = TrainingCompletionReport(
            model_name="test",
            created_at="2024-01-01T00:00:00Z",
            completion_status=CompletionStatus.IN_PROGRESS,
            completion_percentage=50.0,
        )
        assert "50.0%" in report.get_progress_summary()

    def test_get_progress_summary_not_started(self):
        report = TrainingCompletionReport(
            model_name="test",
            created_at="2024-01-01T00:00:00Z",
        )
        assert "not started" in report.get_progress_summary().lower()


# ── TrainingStatusTracker ────────────────────────────────────────────────────

class TestTrainingStatusTracker:
    def test_init(self):
        tracker = TrainingStatusTracker("mymodel")
        assert tracker.model_name == "mymodel"
        assert tracker.report.completion_status == CompletionStatus.NOT_STARTED

    def test_start_training(self):
        tracker = TrainingStatusTracker()
        tracker.start_training(
            dataset="data.txt",
            batch_size=32,
            learning_rate=0.001,
            pretrain_epochs=10,
        )
        assert tracker.report.completion_status == CompletionStatus.IN_PROGRESS
        assert tracker.report.dataset == "data.txt"
        assert tracker.report.batch_size == 32
        assert tracker.report.pretraining is not None
        assert tracker.report.pretraining.total_epochs == 10

    def test_start_stage(self):
        tracker = TrainingStatusTracker()
        tracker.start_training(pretrain_epochs=5)
        tracker.start_stage(TrainingStage.PRETRAINING)
        assert tracker.report.pretraining.status == CompletionStatus.IN_PROGRESS
        assert tracker.report.pretraining.started_at is not None

    def test_update_stage(self):
        tracker = TrainingStatusTracker()
        tracker.start_training(pretrain_epochs=5)
        tracker.start_stage(TrainingStage.PRETRAINING)
        tracker.update_stage(TrainingStage.PRETRAINING, epoch=0, loss=1.0, val_loss=0.9)
        assert tracker.report.pretraining.epochs_completed == 1
        assert tracker.report.pretraining.final_loss == 1.0
        assert tracker.report.pretraining.best_loss == 0.9

    def test_complete_stage(self):
        tracker = TrainingStatusTracker()
        tracker.start_training(pretrain_epochs=5)
        tracker.start_stage(TrainingStage.PRETRAINING)
        tracker.update_stage(TrainingStage.PRETRAINING, epoch=4, loss=0.5)
        tracker.complete_stage(TrainingStage.PRETRAINING)
        assert tracker.report.pretraining.status == CompletionStatus.COMPLETED
        assert tracker.report.pretraining.completed_at is not None
        assert tracker.report.total_epochs == 5

    def test_fail_stage(self):
        tracker = TrainingStatusTracker()
        tracker.start_training(pretrain_epochs=5)
        tracker.start_stage(TrainingStage.PRETRAINING)
        tracker.fail_stage(TrainingStage.PRETRAINING, "OOM error")
        assert tracker.report.pretraining.status == CompletionStatus.FAILED
        assert tracker.report.pretraining.error == "OOM error"
        assert tracker.report.completion_status == CompletionStatus.FAILED
        assert any("OOM error" in e for e in tracker.report.errors)

    def test_record_checkpoint(self):
        tracker = TrainingStatusTracker()
        tracker.record_checkpoint("/tmp/ckpt.npz", step=100, loss=0.5)
        assert tracker.report.checkpoint_path == "/tmp/ckpt.npz"
        assert tracker.report.last_checkpoint_step == 100
        assert tracker.report.checkpoint_count == 1
        assert len(tracker.checkpoints) == 1

    def test_mark_complete(self):
        tracker = TrainingStatusTracker()
        tracker.start_training(pretrain_epochs=5)
        tracker.mark_complete()
        assert tracker.report.completion_status == CompletionStatus.COMPLETED
        assert tracker.report.completion_percentage == 100.0
        assert tracker.report.trained_at is not None

    def test_get_report(self):
        tracker = TrainingStatusTracker()
        report = tracker.get_report()
        assert report.model_name == "sloughgpt"

    def test_save_and_load_report(self, tmp_path):
        tracker = TrainingStatusTracker("testmodel")
        tracker.start_training(dataset="data.txt", pretrain_epochs=5)
        tracker.start_stage(TrainingStage.PRETRAINING)
        tracker.update_stage(TrainingStage.PRETRAINING, epoch=2, loss=0.8)

        path = str(tmp_path / "report.json")
        tracker.save_report(path)

        loaded = TrainingStatusTracker.load_report(path)
        assert loaded.model_name == "testmodel"
        assert loaded.report.dataset == "data.txt"
        assert loaded.report.pretraining.epochs_completed == 3

    def test_overall_progress_calculation(self):
        tracker = TrainingStatusTracker()
        tracker.start_training(pretrain_epochs=10, federated_rounds=5)
        tracker.start_stage(TrainingStage.PRETRAINING)
        tracker.update_stage(TrainingStage.PRETRAINING, epoch=9, loss=0.5)
        # 10/15 = 66.7%
        assert tracker.report.completion_percentage == pytest.approx(66.67, abs=0.1)

    def test_all_stages_complete_auto_completes(self):
        tracker = TrainingStatusTracker()
        tracker.start_training(pretrain_epochs=2, federated_rounds=2)

        tracker.start_stage(TrainingStage.PRETRAINING)
        tracker.update_stage(TrainingStage.PRETRAINING, epoch=1, loss=0.5)
        tracker.complete_stage(TrainingStage.PRETRAINING)

        tracker.start_stage(TrainingStage.FEDERATED)
        tracker.update_stage(TrainingStage.FEDERATED, epoch=1, loss=0.3)
        tracker.complete_stage(TrainingStage.FEDERATED)

        assert tracker.report.completion_status == CompletionStatus.COMPLETED
        assert tracker.report.completion_percentage == 100.0
