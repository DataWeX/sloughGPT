"""Tests for domains/training/status.py — training status tracker + enums."""

import json
import tempfile
import pytest
from domains.training.status import (
    TrainingStage,
    CompletionStatus,
    StageStatus,
    TrainingCompletionReport,
    TrainingStatusTracker,
)


class TestEnums:
    def test_training_stage_values(self):
        assert TrainingStage.PRETRAINING.value == "pretraining"
        assert TrainingStage.FEDERATED.value == "federated"
        assert TrainingStage.RLHF.value == "rlhf"
        assert TrainingStage.COMPLETE.value == "complete"
        assert TrainingStage.FAILED.value == "failed"

    def test_completion_status_values(self):
        assert CompletionStatus.IN_PROGRESS.value == "in_progress"
        assert CompletionStatus.COMPLETED.value == "completed"
        assert CompletionStatus.FAILED.value == "failed"
        assert CompletionStatus.INTERRUPTED.value == "interrupted"
        assert CompletionStatus.NOT_STARTED.value == "not_started"


class TestStageStatus:
    def test_defaults(self):
        s = StageStatus(name="Pretraining")
        assert s.started_at is None
        assert s.completed_at is None
        assert s.epochs_completed == 0
        assert s.total_epochs == 0
        assert s.best_loss == 0.0
        assert s.final_loss == 0.0
        assert s.status == CompletionStatus.NOT_STARTED
        assert s.error is None


class TestTrainingCompletionReport:
    def test_is_complete(self):
        r = TrainingCompletionReport(model_name="test", created_at="2024-01-01")
        assert r.is_complete() is False
        r.completion_status = CompletionStatus.COMPLETED
        assert r.is_complete() is True

    def test_can_resume_in_progress(self):
        r = TrainingCompletionReport(
            model_name="test", created_at="2024-01-01",
            completion_status=CompletionStatus.IN_PROGRESS,
            checkpoint_path="/tmp/ckpt",
        )
        assert r.can_resume() is True

    def test_can_resume_interrupted(self):
        r = TrainingCompletionReport(
            model_name="test", created_at="2024-01-01",
            completion_status=CompletionStatus.INTERRUPTED,
            checkpoint_path="/tmp/ckpt",
        )
        assert r.can_resume() is True

    def test_cannot_resume_without_checkpoint(self):
        r = TrainingCompletionReport(
            model_name="test", created_at="2024-01-01",
            completion_status=CompletionStatus.INTERRUPTED,
            checkpoint_path=None,
        )
        assert r.can_resume() is False

    def test_cannot_resume_completed(self):
        r = TrainingCompletionReport(
            model_name="test", created_at="2024-01-01",
            completion_status=CompletionStatus.COMPLETED,
        )
        assert r.can_resume() is False

    def test_progress_summary_completed(self):
        r = TrainingCompletionReport(
            model_name="test", created_at="2024-01-01",
            completion_status=CompletionStatus.COMPLETED,
            final_loss=0.5,
        )
        assert "complete" in r.get_progress_summary().lower()
        assert "0.5" in r.get_progress_summary()

    def test_progress_summary_in_progress(self):
        r = TrainingCompletionReport(
            model_name="test", created_at="2024-01-01",
            completion_status=CompletionStatus.IN_PROGRESS,
            completion_percentage=45.0,
        )
        assert "45" in r.get_progress_summary()

    def test_progress_summary_interrupted(self):
        r = TrainingCompletionReport(
            model_name="test", created_at="2024-01-01",
            completion_status=CompletionStatus.INTERRUPTED,
            completion_percentage=30.0,
        )
        assert "interrupted" in r.get_progress_summary().lower()

    def test_progress_summary_not_started(self):
        r = TrainingCompletionReport(
            model_name="test", created_at="2024-01-01",
            completion_status=CompletionStatus.NOT_STARTED,
        )
        assert "not started" in r.get_progress_summary().lower()


class TestTrainingStatusTracker:
    def test_init(self):
        tracker = TrainingStatusTracker("my_model")
        assert tracker.model_name == "my_model"
        assert tracker.report.completion_status == CompletionStatus.NOT_STARTED

    def test_start_training(self):
        tracker = TrainingStatusTracker()
        tracker.start_training(
            dataset="shakespeare",
            batch_size=32,
            learning_rate=0.001,
            pretrain_epochs=5,
            federated_rounds=3,
            rlhf_epochs=2,
        )
        assert tracker.report.completion_status == CompletionStatus.IN_PROGRESS
        assert tracker.report.dataset == "shakespeare"
        assert tracker.report.pretraining is not None
        assert tracker.report.pretraining.total_epochs == 5
        assert tracker.report.federated is not None
        assert tracker.report.rlhf is not None

    def test_update_stage(self):
        tracker = TrainingStatusTracker()
        tracker.start_training(pretrain_epochs=10)
        tracker.update_stage(TrainingStage.PRETRAINING, epoch=0, loss=2.0)
        assert tracker.report.pretraining.epochs_completed == 1
        assert tracker.report.pretraining.final_loss == 2.0

    def test_update_stage_with_val_loss(self):
        tracker = TrainingStatusTracker()
        tracker.start_training(pretrain_epochs=10)
        tracker.update_stage(TrainingStage.PRETRAINING, epoch=0, loss=2.0, val_loss=1.8)
        assert tracker.report.pretraining.best_loss == 1.8

    def test_complete_stage(self):
        tracker = TrainingStatusTracker()
        tracker.start_training(pretrain_epochs=5)
        tracker.start_stage(TrainingStage.PRETRAINING)
        tracker.complete_stage(TrainingStage.PRETRAINING)
        assert tracker.report.pretraining.status == CompletionStatus.COMPLETED
        assert tracker.report.pretraining.completed_at is not None

    def test_fail_stage(self):
        tracker = TrainingStatusTracker()
        tracker.start_training(pretrain_epochs=5)
        tracker.fail_stage(TrainingStage.PRETRAINING, "OOM error")
        assert tracker.report.pretraining.status == CompletionStatus.FAILED
        assert tracker.report.pretraining.error == "OOM error"
        assert tracker.report.completion_status == CompletionStatus.FAILED
        assert len(tracker.report.errors) == 1

    def test_record_checkpoint(self):
        tracker = TrainingStatusTracker()
        tracker.record_checkpoint("/tmp/ckpt.pt", step=100, loss=0.5)
        assert tracker.report.checkpoint_path == "/tmp/ckpt.pt"
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

    def test_auto_complete_all_stages(self):
        tracker = TrainingStatusTracker()
        tracker.start_training(pretrain_epochs=2, federated_rounds=2)
        tracker.start_stage(TrainingStage.PRETRAINING)
        tracker.update_stage(TrainingStage.PRETRAINING, epoch=0, loss=1.0)
        tracker.update_stage(TrainingStage.PRETRAINING, epoch=1, loss=0.8)
        tracker.complete_stage(TrainingStage.PRETRAINING)
        tracker.start_stage(TrainingStage.FEDERATED)
        tracker.update_stage(TrainingStage.FEDERATED, epoch=0, loss=0.7)
        tracker.update_stage(TrainingStage.FEDERATED, epoch=1, loss=0.5)
        tracker.complete_stage(TrainingStage.FEDERATED)
        assert tracker.report.completion_status == CompletionStatus.COMPLETED

    def test_save_and_load_report(self):
        tracker = TrainingStatusTracker("test_model")
        tracker.start_training(dataset="test", pretrain_epochs=3)
        tracker.mark_complete()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        tracker.save_report(path)
        loaded = TrainingStatusTracker.load_report(path)
        assert loaded.report.model_name == "test_model"
        assert loaded.report.dataset == "test"
        # load_report doesn't deserialize enums back from strings
        assert loaded.report.completion_status == "completed"
