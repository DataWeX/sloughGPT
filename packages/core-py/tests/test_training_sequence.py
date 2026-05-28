"""Tests for TrainingSequence state machine and CheckpointFormat."""
import os
import tempfile
import time
from domains.training.sequence import (
    TrainingSequence, TrainingSequenceState, StageResult,
    TrainingRunConfig, CheckpointFormat,
)


class TestTrainingSequence:
    def test_enum_values(self):
        assert TrainingSequence.IDLE.value == "idle"
        assert TrainingSequence.GENERATE_DATA.value == "generate_data"
        assert TrainingSequence.DISTILL.value == "distill"
        assert TrainingSequence.TRAIN.value == "train"
        assert TrainingSequence.EVALUATE.value == "evaluate"
        assert TrainingSequence.DEPLOY.value == "deploy"
        assert TrainingSequence.COMPLETE.value == "complete"
        assert TrainingSequence.FAILED.value == "failed"
        assert TrainingSequence.EARLY_STOP.value == "early_stop"

    def test_default_state_is_idle(self):
        s = TrainingSequenceState()
        assert s.current_stage == TrainingSequence.IDLE
        assert s.stages_completed == []
        assert s.total_steps == 0

    def test_start_stage_creates_result(self):
        s = TrainingSequenceState()
        r = s.start_stage(TrainingSequence.TRAIN)
        assert isinstance(r, StageResult)
        assert r.stage == TrainingSequence.TRAIN
        assert s.current_stage == TrainingSequence.TRAIN
        assert r.started_at > 0
        assert r.completed_at is None

    def test_complete_stage_records_success(self):
        s = TrainingSequenceState()
        r = s.start_stage(TrainingSequence.GENERATE_DATA)
        s.complete_stage(TrainingSequence.GENERATE_DATA, r)
        assert r.success is True
        assert r.completed_at is not None
        assert TrainingSequence.GENERATE_DATA in s.stages_completed

    def test_complete_stage_no_dup_in_stages_completed(self):
        s = TrainingSequenceState()
        r = s.start_stage(TrainingSequence.TRAIN)
        s.complete_stage(TrainingSequence.TRAIN, r)
        s.complete_stage(TrainingSequence.TRAIN, r)
        assert s.stages_completed.count(TrainingSequence.TRAIN) == 1

    def test_fail_stage_sets_failed_and_error(self):
        s = TrainingSequenceState()
        r = s.start_stage(TrainingSequence.DISTILL)
        s.fail_stage(TrainingSequence.DISTILL, "CUDA OOM", r)
        assert r.success is False
        assert r.error == "CUDA OOM"
        assert s.current_stage == TrainingSequence.FAILED

    def test_fail_stage_without_preexisting_result(self):
        s = TrainingSequenceState()
        s.fail_stage(TrainingSequence.EVALUATE, "model not found")
        assert s.current_stage == TrainingSequence.FAILED
        r = s.stage_results[TrainingSequence.EVALUATE]
        assert r.success is False
        assert r.error == "model not found"

    def test_update_progress_tracks_best_loss(self):
        s = TrainingSequenceState(total_epochs=10)
        s.update_progress(step=50, loss=2.5, epoch=1)
        assert s.total_steps == 50
        assert s.current_loss == 2.5
        assert s.best_loss == 2.5
        s.update_progress(step=100, loss=1.8, epoch=2)
        assert s.best_loss == 1.8
        s.update_progress(step=150, loss=2.0, epoch=3)
        assert s.best_loss == 1.8

    def test_update_progress_pct_clamped(self):
        s = TrainingSequenceState(total_epochs=5)
        s.update_progress(step=0, loss=0, epoch=10)
        assert s.progress_pct == 100.0

    def test_update_progress_zero_epochs_no_division_error(self):
        s = TrainingSequenceState(total_epochs=0)
        s.update_progress(step=10, loss=1.0, epoch=0)
        assert s.progress_pct == 0.0

    def test_to_dict_includes_stages(self):
        s = TrainingSequenceState(total_epochs=3)
        s.start_stage(TrainingSequence.TRAIN)
        d = s.to_dict()
        assert d["stage"] == "train"
        assert d["total_epochs"] == 3
        assert "stage_results" in d

    def test_to_dict_after_complete(self):
        s = TrainingSequenceState()
        r = s.start_stage(TrainingSequence.GENERATE_DATA)
        r.data_generated = 500
        r.metrics = {"pairs": 25}
        s.complete_stage(TrainingSequence.GENERATE_DATA, r)
        d = s.to_dict()
        assert "generate_data" in d["stage_results"]
        sd = d["stage_results"]["generate_data"]
        assert sd["success"] is True
        assert sd["data_generated"] == 500


class TestStageResult:
    def test_duration_increases(self):
        r = StageResult(stage=TrainingSequence.TRAIN, started_at=time.time() - 2)
        assert r.duration() >= 1.9

    def test_duration_with_completed_at(self):
        r = StageResult(stage=TrainingSequence.TRAIN)
        r.completed_at = r.started_at + 5
        assert abs(r.duration() - 5.0) < 0.01

    def test_to_dict_roundtrip(self):
        r = StageResult(stage=TrainingSequence.EVALUATE, data_generated=100, metrics={"acc": 0.95})
        r.completed_at = r.started_at + 3
        d = r.to_dict()
        assert d["stage"] == "evaluate"
        assert d["data_generated"] == 100
        assert d["duration_s"] >= 2.9


class TestTrainingRunConfig:
    def test_default_stages(self):
        c = TrainingRunConfig()
        assert len(c.stages) == 5
        assert c.stages[0] == TrainingSequence.GENERATE_DATA

    def test_skip_generate_available(self):
        c = TrainingRunConfig(skip_generate=True)
        assert c.skip_generate is True

    def test_soul_name_default(self):
        c = TrainingRunConfig()
        assert c.soul_name == "assistant"

    def test_checkpoint_dir_default(self):
        c = TrainingRunConfig()
        assert c.checkpoint_dir == "models/auto-training"


class TestCheckpointFormat:
    def test_create_includes_version(self):
        ckpt = CheckpointFormat.create(
            model_state={"w": [1, 2, 3]},
            stoi={"a": 0, "b": 1},
            itos={0: "a", 1: "b"},
            soul_name="test",
            system_prompt="hello",
            train_loss=0.5,
            steps=100,
            epochs=5,
        )
        assert ckpt["version"] == "1.0"
        assert ckpt["vocab_size"] == 2
        assert ckpt["soul_name"] == "test"
        assert ckpt["steps"] == 100

    def test_create_default_traits(self):
        ckpt = CheckpointFormat.create(
            model_state={}, stoi={"x": 0}, itos={0: "x"},
            soul_name="t", system_prompt="p", train_loss=0, steps=0, epochs=0,
        )
        t = ckpt["personality_traits"]
        assert t["warmth"] == 0.5
        assert t["creativity"] == 0.5
        assert t["curiosity"] == 0.5
        assert t["confidence"] == 0.5

    def test_save_and_load_roundtrip(self):
        ckpt = CheckpointFormat.create(
            model_state={"w": [42]},
            stoi={"a": 0}, itos={0: "a"},
            soul_name="roundtrip", system_prompt="test",
            train_loss=0.1, steps=10, epochs=1,
        )
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            CheckpointFormat.save(ckpt, path)
            loaded = CheckpointFormat.load(path)
            assert loaded["soul_name"] == "roundtrip"
            assert loaded["train_loss"] == 0.1
            assert loaded["vocab_size"] == 1
        finally:
            os.unlink(path)

    def test_load_returns_dict(self):
        ckpt = CheckpointFormat.create(
            model_state={}, stoi={"x": 0}, itos={0: "x"},
            soul_name="ld", system_prompt="", train_loss=0, steps=0, epochs=0,
        )
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            CheckpointFormat.save(ckpt, path)
            loaded = CheckpointFormat.load(path)
            assert isinstance(loaded, dict)
        finally:
            os.unlink(path)
