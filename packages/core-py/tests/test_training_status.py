"""Tests for domains/training/status.py — training status tracker + enums + checkpoints."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
from domains.training.status import (
    TrainingStage,
    CompletionStatus,
    StageStatus,
    TrainingCompletionReport,
    TrainingStatusTracker,
    CheckpointManager,
    save_checkpoint_npz,
    load_checkpoint_npz,
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


# =============================================================================
# NPZ CHECKPOINT TESTS
# =============================================================================


class _StubModel:
    """Minimal model stub with state_dict / load_state_dict for checkpoint tests."""

    def __init__(self, params: Dict[str, np.ndarray]):
        self._params = {k: v.copy() for k, v in params.items()}

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self._params.items()}

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        for k in state_dict:
            if hasattr(state_dict[k], "cpu"):
                self._params[k] = state_dict[k].cpu().numpy()
            elif isinstance(state_dict[k], np.ndarray):
                self._params[k] = state_dict[k].copy()
            else:
                self._params[k] = np.asarray(state_dict[k])


class TestSaveLoadCheckpointNpz:
    """Standalone save_checkpoint_npz / load_checkpoint_npz helpers."""

    def test_roundtrip_simple_weights(self):
        state = {"weight": np.array([[1.0, 2.0], [3.0, 4.0]]), "bias": np.array([0.1, 0.2])}
        meta = {"loss": 0.42, "epoch": 5, "stage": "pretraining"}

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            saved = save_checkpoint_npz(path, state, meta)
            assert saved == path
            loaded = load_checkpoint_npz(path)
            assert loaded["loss"] == 0.42
            assert loaded["epoch"] == 5
            assert loaded["stage"] == "pretraining"
            assert np.allclose(loaded["model_state_dict"]["weight"], [[1., 2.], [3., 4.]])
            assert np.allclose(loaded["model_state_dict"]["bias"], [0.1, 0.2])
        finally:
            os.unlink(path)

    def test_auto_appends_npz_extension(self):
        state = {"w": np.array([1.0])}
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name  # no suffix
        try:
            saved = save_checkpoint_npz(path, state)
            assert saved.endswith(".npz")
            assert Path(saved).exists()
            loaded = load_checkpoint_npz(saved)
            assert "model_state_dict" in loaded
        finally:
            os.unlink(saved)

    def test_empty_state_dict(self):
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            save_checkpoint_npz(path, {}, {"note": "empty"})
            loaded = load_checkpoint_npz(path)
            assert loaded["note"] == "empty"
            assert loaded["model_state_dict"] == {}
        finally:
            os.unlink(path)

    def test_binary_data_in_meta(self):
        """Metadata values that aren't JSON-serializable get str()-ified via default=str."""
        state = {"w": np.array([1.0])}
        meta = {"bytes_val": b"hello\xff"}
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            saved = save_checkpoint_npz(path, state, meta)
            loaded = load_checkpoint_npz(saved)
            # bytes → str via default=str
            assert isinstance(loaded["bytes_val"], str)
        finally:
            os.unlink(path)


class TestCheckpointManagerNpz:
    """CheckpointManager writes torch-free .npz checkpoints by default."""

    def test_save_and_load_npz(self, tmp_path):
        mgr = CheckpointManager(str(tmp_path))
        params = {
            "encoder.weight": np.array([[1.0, 0.0], [0.0, 1.0]]),
            "decoder.bias": np.array([0.0, 0.0]),
        }
        model = _StubModel(params)

        path = mgr.save_checkpoint(
            model, optimizer=None, step=10, epoch=2, loss=0.5, val_loss=0.45,
            metadata={"note": "test"},
        )
        assert path.endswith(".npz")
        assert Path(path).exists()

        # Load into fresh model
        fresh = _StubModel({"encoder.weight": np.eye(2), "decoder.bias": np.zeros(2)})
        info = mgr.load_checkpoint(path, fresh)
        assert info["step"] == 10
        assert info["epoch"] == 2
        assert info["loss"] == 0.5
        assert info["val_loss"] == 0.45
        assert np.allclose(fresh._params["encoder.weight"], [[1., 0.], [0., 1.]])
        assert np.allclose(fresh._params["decoder.bias"], [0., 0.])

    def test_list_returns_npz_checkpoints(self, tmp_path):
        mgr = CheckpointManager(str(tmp_path))
        model = _StubModel({"w": np.array([1.0])})

        mgr.save_checkpoint(model, None, step=1, epoch=0, loss=0.8)
        mgr.save_checkpoint(model, None, step=2, epoch=1, loss=0.5)

        ckpts = mgr.list_checkpoints()
        assert len(ckpts) == 2
        assert ckpts[0]["step"] == 2  # sorted desc
        assert ckpts[1]["step"] == 1

    def test_ignores_foreign_checkpoint_files(self, tmp_path):
        """list_checkpoints skips non-.npz files in the checkpoint dir."""
        mgr = CheckpointManager(str(tmp_path))
        model = _StubModel({"w": np.array([1.0])})

        mgr.save_checkpoint(model, None, step=1, epoch=0, loss=0.8)
        # A .pt file (or any non-npz) must be ignored by list_checkpoints
        pt_path = tmp_path / "checkpoint_step2.pt"
        pt_path.write_text("not a valid checkpoint")

        ckpts = mgr.list_checkpoints()
        assert len(ckpts) == 1
        assert ckpts[0]["step"] == 1
        assert ckpts[0]["path"].endswith(".npz")

    def test_get_best_with_npz(self, tmp_path):
        mgr = CheckpointManager(str(tmp_path))
        model = _StubModel({"w": np.array([1.0])})

        mgr.save_checkpoint(model, None, step=1, epoch=0, loss=0.9)
        mgr.save_checkpoint(model, None, step=2, epoch=1, loss=0.3)

        best = mgr.get_best_checkpoint()
        assert best is not None
        loaded = mgr.load_checkpoint(best, _StubModel({"w": np.array([1.0])}))
        assert loaded["loss"] == 0.3

    def test_get_latest_with_npz(self, tmp_path):
        mgr = CheckpointManager(str(tmp_path))
        model = _StubModel({"w": np.array([1.0])})

        mgr.save_checkpoint(model, None, step=1, epoch=0, loss=0.9)
        import time
        time.sleep(0.01)
        mgr.save_checkpoint(model, None, step=2, epoch=1, loss=0.3)

        latest = mgr.get_latest_checkpoint()
        assert latest is not None
        loaded = mgr.load_checkpoint(latest, _StubModel({"w": np.array([1.0])}))
        assert loaded["step"] == 2

    def test_npz_updates_tracker(self, tmp_path):
        mgr = CheckpointManager(str(tmp_path))
        model = _StubModel({"w": np.array([1.0])})

        mgr.save_checkpoint(model, None, step=5, epoch=2, loss=0.5)
        assert mgr.tracker.report.last_checkpoint_step == 5
        assert mgr.tracker.report.checkpoint_count == 1
        assert mgr.tracker.report.checkpoint_path is not None
        assert mgr.tracker.report.checkpoint_path.endswith(".npz")
