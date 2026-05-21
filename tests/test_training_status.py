"""
Tests for Training Status & Checkpoint Management
"""

import pytest
from domains.training.status import (
    TrainingStage,
    CompletionStatus,
    StageStatus,
    TrainingCompletionReport,
    TrainingStatusTracker,
    CheckpointManager,
)


class TestTrainingStatusTracker:
    """Tests for training status tracker."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = TrainingStatusTracker("test_model")

        assert tracker.model_name == "test_model"
        assert tracker.report.completion_status == CompletionStatus.NOT_STARTED
        assert tracker.report.completion_percentage == 0.0

    def test_start_training(self):
        """Test starting training."""
        tracker = TrainingStatusTracker("test_model")

        tracker.start_training(
            dataset="test_data",
            batch_size=32,
            learning_rate=1e-4,
            pretrain_epochs=10,
            federated_rounds=5,
            rlhf_epochs=4,
        )

        assert tracker.report.completion_status == CompletionStatus.IN_PROGRESS
        assert tracker.report.pretraining is not None
        assert tracker.report.federated is not None
        assert tracker.report.rlhf is not None

    def test_update_stage(self):
        """Test updating stage progress."""
        tracker = TrainingStatusTracker("test_model")

        tracker.start_training(pretrain_epochs=10)
        tracker.start_stage(TrainingStage.PRETRAINING)

        tracker.update_stage(TrainingStage.PRETRAINING, epoch=5, loss=0.5, val_loss=0.6)

        assert tracker.report.pretraining.epochs_completed == 6
        assert tracker.report.pretraining.final_loss == 0.5
        assert tracker.report.pretraining.best_loss == 0.6

    def test_complete_stage(self):
        """Test completing a stage."""
        tracker = TrainingStatusTracker("test_model")

        tracker.start_training(pretrain_epochs=10)
        tracker.start_stage(TrainingStage.PRETRAINING)

        for i in range(10):
            tracker.update_stage(TrainingStage.PRETRAINING, i, 0.5 - i * 0.05)

        tracker.complete_stage(TrainingStage.PRETRAINING)

        assert tracker.report.pretraining.status == CompletionStatus.COMPLETED
        assert tracker.report.total_epochs == 10

    def test_record_checkpoint(self):
        """Test recording checkpoints."""
        tracker = TrainingStatusTracker("test_model")

        tracker.record_checkpoint("checkpoint_step100.pt", 100, 0.5)

        assert tracker.report.checkpoint_path == "checkpoint_step100.pt"
        assert tracker.report.last_checkpoint_step == 100
        assert len(tracker.checkpoints) == 1

    def test_is_complete(self):
        """Test completion check."""
        tracker = TrainingStatusTracker("test_model")

        assert tracker.report.is_complete() == False

        tracker.report.completion_status = CompletionStatus.COMPLETED

        assert tracker.report.is_complete() == True

    def test_can_resume(self):
        """Test resume capability check."""
        tracker = TrainingStatusTracker("test_model")

        assert tracker.report.can_resume() == False

        tracker.report.completion_status = CompletionStatus.IN_PROGRESS
        tracker.report.checkpoint_path = "checkpoint.pt"

        assert tracker.report.can_resume() == True

    def test_progress_summary(self):
        """Test progress summary."""
        tracker = TrainingStatusTracker("test_model")

        tracker.start_training(pretrain_epochs=10)
        tracker.update_stage(TrainingStage.PRETRAINING, 5, 0.5)

        summary = tracker.report.get_progress_summary()
        assert "50.0%" in summary or "in progress" in summary.lower()


class TestTrainingCompletionReport:
    """Tests for completion report."""

    def test_initialization(self):
        """Test report initialization."""
        report = TrainingCompletionReport(
            model_name="test_model",
            created_at="2024-01-01T00:00:00Z",
        )

        assert report.model_name == "test_model"
        assert report.completion_status == CompletionStatus.NOT_STARTED


class TestStageStatus:
    """Tests for stage status."""

    def test_initialization(self):
        """Test stage status initialization."""
        stage = StageStatus(name="Pretraining", total_epochs=10)

        assert stage.name == "Pretraining"
        assert stage.total_epochs == 10
        assert stage.epochs_completed == 0
        assert stage.status == CompletionStatus.NOT_STARTED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
