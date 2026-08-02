"""Tests for domains/training/status.py."""

import json

import numpy as np
import pytest

from domains.training.status import (
    CheckpointManager,
    CompletionStatus,
    StageStatus,
    TrainingCompletionReport,
    TrainingStage,
    TrainingStatusTracker,
    load_checkpoint_npz,
    save_checkpoint_npz,
)


class TestEnums:
    def test_training_stage_values(self):
        assert TrainingStage.NOT_STARTED.value == "not_started"
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
        stage = StageStatus(name="Pretraining")
        assert stage.total_epochs == 0
        assert stage.status == CompletionStatus.NOT_STARTED
        assert stage.error is None


class TestTrainingCompletionReport:
    def test_is_complete(self):
        report = TrainingCompletionReport(model_name="m", created_at="now")
        assert not report.is_complete()
        report.completion_status = CompletionStatus.COMPLETED
        assert report.is_complete()

    def test_can_resume(self):
        report = TrainingCompletionReport(model_name="m", created_at="now")
        assert not report.can_resume()
        report.completion_status = CompletionStatus.INTERRUPTED
        assert not report.can_resume()  # no checkpoint
        report.checkpoint_path = "/tmp/ckpt.npz"
        assert report.can_resume()

    def test_get_progress_summary(self):
        report = TrainingCompletionReport(model_name="m", created_at="now")
        assert "not started" in report.get_progress_summary()
        report.completion_status = CompletionStatus.IN_PROGRESS
        report.completion_percentage = 42.5
        assert "42.5%" in report.get_progress_summary()
        report.completion_status = CompletionStatus.INTERRUPTED
        assert "Can resume" in report.get_progress_summary()
        report.completion_status = CompletionStatus.COMPLETED
        report.final_loss = 0.1234
        assert "0.1234" in report.get_progress_summary()


class TestTrainingStatusTracker:
    def test_init(self):
        tracker = TrainingStatusTracker("mymodel")
        assert tracker.model_name == "mymodel"
        assert tracker.report.completion_status == CompletionStatus.NOT_STARTED
        assert tracker.checkpoints == []

    def test_start_training_initializes_stages(self):
        tracker = TrainingStatusTracker()
        tracker.start_training(
            dataset="d", batch_size=4, learning_rate=1e-3,
            pretrain_epochs=3, federated_rounds=2, rlhf_epochs=1,
        )
        assert tracker.report.completion_status == CompletionStatus.IN_PROGRESS
        assert tracker.report.pretraining is not None
        assert tracker.report.pretraining.total_epochs == 3
        assert tracker.report.federated.total_epochs == 2
        assert tracker.report.rlhf.total_epochs == 1
        assert tracker.report.dataset == "d"
        assert tracker.report.batch_size == 4
        assert tracker.report.learning_rate == 1e-3

    def test_start_training_without_stages(self):
        tracker = TrainingStatusTracker()
        tracker.start_training()
        assert tracker.report.pretraining is None
        assert tracker.report.federated is None
        assert tracker.report.rlhf is None

    def test_start_stage(self):
        tracker = TrainingStatusTracker()
        tracker.start_training(pretrain_epochs=3)
        tracker.start_stage(TrainingStage.PRETRAINING)
        assert tracker.report.pretraining.status == CompletionStatus.IN_PROGRESS
        assert tracker.report.pretraining.started_at is not None

    def test_start_unknown_stage_is_noop(self):
        tracker = TrainingStatusTracker()
        tracker.start_training(pretrain_epochs=3)
        tracker.start_stage(TrainingStage.COMPLETE)
        assert tracker.report.pretraining.status == CompletionStatus.NOT_STARTED

    def test_update_stage(self):
        tracker = TrainingStatusTracker()
        tracker.start_training(pretrain_epochs=10)
        tracker.update_stage(TrainingStage.PRETRAINING, epoch=1, loss=0.5, val_loss=0.4)
        assert tracker.report.pretraining.epochs_completed == 2
        assert tracker.report.pretraining.final_loss == 0.5
        assert tracker.report.pretraining.best_loss == 0.4

    def test_update_stage_best_loss_only_decreases(self):
        tracker = TrainingStatusTracker()
        tracker.start_training(pretrain_epochs=10)
        tracker.update_stage(TrainingStage.PRETRAINING, epoch=0, loss=1.0, val_loss=0.5)
        tracker.update_stage(TrainingStage.PRETRAINING, epoch=1, loss=0.9, val_loss=0.7)
        assert tracker.report.pretraining.best_loss == 0.5

    def test_complete_stage_updates_report(self):
        tracker = TrainingStatusTracker()
        tracker.start_training(pretrain_epochs=5)
        tracker.update_stage(TrainingStage.PRETRAINING, epoch=4, loss=0.3, val_loss=0.2)
        tracker.complete_stage(TrainingStage.PRETRAINING)
        assert tracker.report.pretraining.status == CompletionStatus.COMPLETED
        assert tracker.report.best_loss == 0.2
        assert tracker.report.final_loss == 0.3
        assert tracker.report.total_epochs == 5

    def test_all_stages_complete_sets_completed(self):
        tracker = TrainingStatusTracker()
        tracker.start_training(pretrain_epochs=2, federated_rounds=2, rlhf_epochs=2)
        for stage in [TrainingStage.PRETRAINING, TrainingStage.FEDERATED, TrainingStage.RLHF]:
            tracker.update_stage(stage, epoch=1, loss=0.5)
            tracker.complete_stage(stage)
        assert tracker.report.completion_status == CompletionStatus.COMPLETED
        assert tracker.report.completion_percentage == 100.0

    def test_fail_stage(self):
        tracker = TrainingStatusTracker()
        tracker.start_training(pretrain_epochs=2)
        tracker.fail_stage(TrainingStage.PRETRAINING, "OOM")
        assert tracker.report.pretraining.status == CompletionStatus.FAILED
        assert tracker.report.pretraining.error == "OOM"
        assert tracker.report.completion_status == CompletionStatus.FAILED
        assert any("pretraining" in e for e in tracker.report.errors)

    def test_record_checkpoint(self):
        tracker = TrainingStatusTracker()
        tracker.record_checkpoint("/tmp/ckpt.npz", step=10, loss=0.4)
        assert tracker.report.checkpoint_path == "/tmp/ckpt.npz"
        assert tracker.report.last_checkpoint_step == 10
        assert tracker.report.checkpoint_count == 1
        assert tracker.checkpoints[0]["step"] == 10

    def test_mark_complete(self):
        tracker = TrainingStatusTracker()
        tracker.mark_complete()
        assert tracker.report.completion_status == CompletionStatus.COMPLETED
        assert tracker.report.completion_percentage == 100.0
        assert tracker.report.trained_at is not None

    def test_save_and_load_report(self, tmp_path):
        tracker = TrainingStatusTracker("save-me")
        tracker.start_training(pretrain_epochs=2)
        tracker.update_stage(TrainingStage.PRETRAINING, epoch=1, loss=0.3)
        path = str(tmp_path / "report.json")
        tracker.save_report(path)
        loaded = TrainingStatusTracker.load_report(path)
        assert loaded.model_name == "save-me"
        assert loaded.report.completion_status == CompletionStatus.IN_PROGRESS
        assert loaded.report.pretraining.epochs_completed == 2

    def test_save_report_serializes_enums(self, tmp_path):
        tracker = TrainingStatusTracker()
        tracker.start_training(pretrain_epochs=2)
        path = str(tmp_path / "report.json")
        tracker.save_report(path)
        data = json.loads(open(path).read())
        assert data["completion_status"] == "in_progress"
        assert data["pretraining"]["status"] == "not_started"


class TestCheckpointManager:
    def test_init_creates_dir(self, tmp_path):
        mgr = CheckpointManager(checkpoint_dir=str(tmp_path / "ckpts"))
        assert (tmp_path / "ckpts").exists()

    def test_save_checkpoint(self, tmp_path):
        class FakeModel:
            def state_dict(self):
                return {"w": np.array([1.0, 2.0])}

        mgr = CheckpointManager(checkpoint_dir=str(tmp_path / "ckpts"))
        path = mgr.save_checkpoint(FakeModel(), None, step=1, epoch=1, loss=0.5)
        assert path.endswith("checkpoint_step1.npz")
        assert (tmp_path / "ckpts" / "checkpoint_step1.npz").exists()
        assert mgr.tracker.report.checkpoint_count == 1

    def test_load_checkpoint_roundtrip(self, tmp_path):
        class FakeModel:
            def __init__(self):
                self.data = None

            def state_dict(self):
                return {"w": np.array([1.0, 2.0])}

            def load_state_dict(self, state_dict):
                self.data = state_dict

        mgr = CheckpointManager(checkpoint_dir=str(tmp_path / "ckpts"))
        model = FakeModel()
        path = mgr.save_checkpoint(model, None, step=5, epoch=2, loss=0.3)
        loaded = mgr.load_checkpoint(path, FakeModel())
        assert loaded["step"] == 5
        assert loaded["epoch"] == 2
        assert loaded["loss"] == 0.3
        assert loaded["stage"] == TrainingStage.PRETRAINING

    def test_get_latest_checkpoint(self, tmp_path):
        mgr = CheckpointManager(checkpoint_dir=str(tmp_path / "ckpts"))
        model = type("M", (), {"state_dict": lambda self: {}})()
        mgr.save_checkpoint(model, None, step=1, epoch=1, loss=0.5)
        mgr.save_checkpoint(model, None, step=2, epoch=2, loss=0.4)
        latest = mgr.get_latest_checkpoint()
        assert latest is not None
        assert latest.endswith("checkpoint_step2.npz")

    def test_get_latest_checkpoint_empty(self, tmp_path):
        mgr = CheckpointManager(checkpoint_dir=str(tmp_path / "ckpts"))
        assert mgr.get_latest_checkpoint() is None

    def test_get_best_checkpoint(self, tmp_path):
        mgr = CheckpointManager(checkpoint_dir=str(tmp_path / "ckpts"))
        model = type("M", (), {"state_dict": lambda self: {}})()
        mgr.save_checkpoint(model, None, step=1, epoch=1, loss=0.9)
        mgr.save_checkpoint(model, None, step=2, epoch=2, loss=0.2)
        best = mgr.get_best_checkpoint()
        assert best is not None
        assert best.endswith("checkpoint_step2.npz")

    def test_list_checkpoints_sorted_by_step_desc(self, tmp_path):
        mgr = CheckpointManager(checkpoint_dir=str(tmp_path / "ckpts"))
        model = type("M", (), {"state_dict": lambda self: {}})()
        mgr.save_checkpoint(model, None, step=1, epoch=1, loss=0.5)
        mgr.save_checkpoint(model, None, step=2, epoch=2, loss=0.4)
        ckpts = mgr.list_checkpoints()
        assert len(ckpts) == 2
        assert ckpts[0]["step"] == 2
        assert ckpts[1]["step"] == 1

    def test_tensors_to_numpy(self):
        class FakeTensor:
            def __init__(self, arr):
                self._arr = arr

            def cpu(self):
                return self

            def numpy(self):
                return self._arr

        result = CheckpointManager._tensors_to_numpy(
            {"a": FakeTensor(np.array([1.0])), "b": np.array([2.0]), "c": 3}
        )
        assert isinstance(result["a"], np.ndarray)
        assert isinstance(result["b"], np.ndarray)
        assert isinstance(result["c"], np.ndarray)


class TestStandaloneNpzHelpers:
    def test_save_load_roundtrip(self, tmp_path):
        path = str(tmp_path / "ckpt.npz")
        save_checkpoint_npz(path, {"w": np.array([1.0, 2.0])}, {"step": 3})
        loaded = load_checkpoint_npz(path)
        assert loaded["step"] == 3
        np.testing.assert_array_equal(loaded["model_state_dict"]["w"], [1.0, 2.0])

    def test_standalone_tensors_to_numpy(self):
        from domains.training.status import _tensors_to_numpy

        result = _tensors_to_numpy({"w": np.array([1.0])})
        assert isinstance(result["w"], np.ndarray)


def test_get_report_returns_report():
    tracker = TrainingStatusTracker()
    assert tracker.get_report() is tracker.report


def test_save_report_raises_on_unserializable(tmp_path):
    tracker = TrainingStatusTracker()
    tracker.report.warnings.append({1, 2, 3})
    with pytest.raises(TypeError):
        tracker.save_report(str(tmp_path / "bad.json"))


def test_print_summary_full_report():
    tracker = TrainingStatusTracker("model-x")
    tracker.start_training(pretrain_epochs=3, federated_rounds=2, rlhf_epochs=1)
    tracker.start_stage(TrainingStage.PRETRAINING)
    tracker.update_stage(TrainingStage.PRETRAINING, epoch=2, loss=0.4, val_loss=0.3)
    tracker.complete_stage(TrainingStage.PRETRAINING)
    tracker.report.pretraining.error = "recovered"
    tracker.print_summary()


def test_get_best_checkpoint_empty(tmp_path):
    mgr = CheckpointManager(checkpoint_dir=str(tmp_path / "ckpts"))
    assert mgr.get_best_checkpoint() is None


def test_get_best_checkpoint_skips_corrupt(tmp_path):
    mgr = CheckpointManager(checkpoint_dir=str(tmp_path / "ckpts"))
    model = type("M", (), {"state_dict": lambda self: {}})()
    good = mgr.save_checkpoint(model, None, step=1, epoch=1, loss=0.9)
    (tmp_path / "ckpts" / "checkpoint_bad.npz").write_bytes(b"garbage not npz")
    assert mgr.get_best_checkpoint() == good


def test_list_checkpoints_skips_corrupt(tmp_path):
    mgr = CheckpointManager(checkpoint_dir=str(tmp_path / "ckpts"))
    model = type("M", (), {"state_dict": lambda self: {}})()
    mgr.save_checkpoint(model, None, step=1, epoch=1, loss=0.5)
    (tmp_path / "ckpts" / "checkpoint_bad.npz").write_bytes(b"garbage")
    ckpts = mgr.list_checkpoints()
    assert len(ckpts) == 1
    assert ckpts[0]["step"] == 1


def test_tensors_to_numpy_nested_dict():
    result = CheckpointManager._tensors_to_numpy(
        {"nested": {"w": np.array([1.0, 2.0])}}
    )
    assert isinstance(result["nested"]["w"], np.ndarray)
