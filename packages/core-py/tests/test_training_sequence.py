"""Tests for domains.training.sequence — TrainingSequence, PhaseResult, TrainingSequenceState, TrainingRunConfig, CheckpointFormat."""

import pytest
from domains.training.sequence import (
    TrainingSequence, PhaseResult, TrainingSequenceState,
    TrainingRunConfig, CheckpointFormat, DataGenerator, StudentModel,
)


class TestTrainingSequence:
    def test_ordered_phases(self):
        phases = TrainingSequence.ordered_phases()
        assert phases[0] == TrainingSequence.IDLE
        assert phases[1] == TrainingSequence.GENERATE_DATA
        assert phases[-3] == TrainingSequence.COMPLETE

    def test_all_values(self):
        values = [p.value for p in TrainingSequence]
        assert "idle" in values
        assert "complete" in values
        assert "failed" in values
        assert "early_stop" in values


class TestPhaseResult:
    def test_defaults(self):
        pr = PhaseResult(phase=TrainingSequence.TRAIN)
        assert pr.status == "working"
        assert pr.message == ""
        assert pr.metrics == {}

    def test_to_dict(self):
        pr = PhaseResult(phase=TrainingSequence.TRAIN, status="success", message="done", metrics={"loss": 0.5})
        d = pr.to_dict()
        assert d["phase"] == "train"
        assert d["status"] == "success"
        assert d["message"] == "done"
        assert d["metrics"] == {"loss": 0.5}


class TestTrainingSequenceState:
    def test_initial_state(self):
        s = TrainingSequenceState()
        assert s.current_phase == TrainingSequence.IDLE
        assert s.is_running is False
        assert s.is_done is False

    def test_start_phase(self):
        s = TrainingSequenceState()
        s.start_phase(TrainingSequence.TRAIN)
        assert s.current_phase == TrainingSequence.TRAIN
        assert s.is_running is True
        assert s.is_done is False

    def test_complete_phase(self):
        s = TrainingSequenceState()
        s.start_phase(TrainingSequence.TRAIN)
        s.complete_phase(TrainingSequence.TRAIN, metrics={"loss": 0.3})
        assert s.phase_results[-1].status == "success"
        assert s.phase_results[-1].metrics["loss"] == 0.3

    def test_fail_phase(self):
        s = TrainingSequenceState()
        s.start_phase(TrainingSequence.TRAIN)
        s.fail_phase(TrainingSequence.TRAIN, message="OOM")
        assert s.current_phase == TrainingSequence.FAILED
        assert s.is_done is True
        assert s.error == "OOM"

    def test_skip_phase(self):
        s = TrainingSequenceState()
        s.skip_phase(TrainingSequence.DISTILL, reason="not needed")
        assert len(s.phase_results) == 1
        assert s.phase_results[0].status == "skipped"

    def test_to_dict(self):
        s = TrainingSequenceState()
        s.start_phase(TrainingSequence.TRAIN)
        d = s.to_dict()
        assert d["current_phase"] == "train"
        assert d["is_running"] is True
        assert d["is_done"] is False

    def test_to_sse_event(self):
        s = TrainingSequenceState()
        s.start_phase(TrainingSequence.TRAIN)
        event = s.to_sse_event()
        assert event["stream"] == "auto-train"
        assert event["phase"] == "train"
        assert event["status"] == "working"

    def test_to_sse_event_complete(self):
        s = TrainingSequenceState()
        s.start_phase(TrainingSequence.DEPLOY)
        s.complete_phase(TrainingSequence.DEPLOY)
        s.current_phase = TrainingSequence.COMPLETE
        event = s.to_sse_event()
        assert event["status"] == "complete"

    def test_to_sse_event_error(self):
        s = TrainingSequenceState()
        s.start_phase(TrainingSequence.TRAIN)
        s.fail_phase(TrainingSequence.TRAIN, message="crash")
        # is_done is True (FAILED is terminal), so SSE status is "complete"
        event = s.to_sse_event()
        assert event["status"] == "complete"
        assert event["meta"]["error"] == "crash"

    def test_early_stop(self):
        s = TrainingSequenceState()
        s.start_phase(TrainingSequence.TRAIN)
        s.early_stop_reason = "patience exceeded"
        assert s.is_done is False  # EARLY_STOP is a done phase but current_phase is still TRAIN
        s.current_phase = TrainingSequence.EARLY_STOP
        assert s.is_done is True


class TestTrainingRunConfig:
    def test_defaults(self):
        cfg = TrainingRunConfig.defaults()
        assert cfg.skip_generate is False
        assert cfg.skip_distill is False
        assert cfg.max_epochs == 10
        assert cfg.early_stop_patience == 3

    def test_effective_phases_all(self):
        cfg = TrainingRunConfig()
        phases = cfg.effective_phases()
        assert TrainingSequence.GENERATE_DATA in phases
        assert TrainingSequence.DISTILL in phases
        assert TrainingSequence.TRAIN in phases
        assert TrainingSequence.EVALUATE in phases
        assert TrainingSequence.DEPLOY in phases

    def test_skip_generate(self):
        cfg = TrainingRunConfig(skip_generate=True)
        phases = cfg.effective_phases()
        assert TrainingSequence.GENERATE_DATA not in phases
        assert TrainingSequence.TRAIN in phases

    def test_skip_all(self):
        cfg = TrainingRunConfig(
            skip_generate=True, skip_distill=True, skip_train=True,
            skip_evaluate=True, skip_deploy=True,
        )
        phases = cfg.effective_phases()
        assert phases == []


class TestCheckpointFormat:
    def test_basic(self):
        cp = CheckpointFormat(name="ckpt_1", step=100, loss=0.5)
        assert cp.name == "ckpt_1"
        assert cp.step == 100
        assert cp.loss == 0.5
        assert cp.epoch == 0

    def test_to_dict(self):
        cp = CheckpointFormat(
            name="ckpt_1", step=100, loss=0.5, val_loss=0.6, epoch=2,
            personality_traits={"warmth": 0.8},
        )
        d = cp.to_dict()
        assert d["name"] == "ckpt_1"
        assert d["step"] == 100
        assert d["loss"] == 0.5
        assert d["val_loss"] == 0.6
        assert d["epoch"] == 2
        assert d["personality_traits"]["warmth"] == 0.8


class TestProtocols:
    def test_data_generator_protocol(self):
        class GoodGen:
            def generate(self, prompt, num_samples, max_length):
                return ["text"]

        assert isinstance(GoodGen(), DataGenerator)

    def test_student_model_protocol(self):
        class GoodStudent:
            def train_step(self, inputs, labels):
                return 0.5

            def evaluate(self, inputs, labels):
                return {"loss": 0.5}

        assert isinstance(GoodStudent(), StudentModel)

    def test_incomplete_protocol_rejected(self):
        class BadStudent:
            def train_step(self, inputs, labels):
                return 0.5

        assert not isinstance(BadStudent(), StudentModel)
