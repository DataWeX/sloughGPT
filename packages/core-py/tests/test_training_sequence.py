"""Tests for TrainingSequence — phase protocol, state tracking, config."""

import pytest
from domains.training.sequence import (
    TrainingSequence,
    PhaseResult,
    TrainingSequenceState,
    TrainingRunConfig,
    CheckpointFormat,
)


class TestTrainingSequence:
    def test_ordered_phases_count(self):
        phases = TrainingSequence.ordered_phases()
        assert len(phases) == 9

    def test_ordered_phases_starts_with_idle(self):
        phases = TrainingSequence.ordered_phases()
        assert phases[0] == TrainingSequence.IDLE

    def test_ordered_phases_ends_with_early_stop(self):
        phases = TrainingSequence.ordered_phases()
        assert phases[-1] == TrainingSequence.EARLY_STOP

    def test_all_phases_have_values(self):
        for phase in TrainingSequence:
            assert isinstance(phase.value, str)
            assert len(phase.value) > 0

    def test_phase_values_are_unique(self):
        values = [p.value for p in TrainingSequence]
        assert len(values) == len(set(values))


class TestPhaseResult:
    def test_default_status(self):
        pr = PhaseResult(phase=TrainingSequence.TRAIN)
        assert pr.status == "working"
        assert pr.message == ""
        assert pr.metrics == {}

    def test_to_dict(self):
        pr = PhaseResult(
            phase=TrainingSequence.DISTILL,
            status="success",
            message="done",
            metrics={"loss": 0.5},
        )
        d = pr.to_dict()
        assert d["phase"] == "distill"
        assert d["status"] == "success"
        assert d["message"] == "done"
        assert d["metrics"]["loss"] == 0.5


class TestTrainingSequenceState:
    def test_initial_state(self):
        s = TrainingSequenceState()
        assert s.current_phase == TrainingSequence.IDLE
        assert s.is_running is False
        assert s.is_done is False
        assert s.error is None

    def test_start_phase(self):
        s = TrainingSequenceState()
        s.start_phase(TrainingSequence.TRAIN)
        assert s.current_phase == TrainingSequence.TRAIN
        assert s.is_running is True

    def test_complete_phase(self):
        s = TrainingSequenceState()
        s.start_phase(TrainingSequence.DISTILL)
        s.complete_phase(TrainingSequence.DISTILL, metrics={"loss": 0.3})
        pr = s.phase_results[0]
        assert pr.status == "success"
        assert pr.metrics["loss"] == 0.3

    def test_fail_phase(self):
        s = TrainingSequenceState()
        s.start_phase(TrainingSequence.EVALUATE)
        s.fail_phase(TrainingSequence.EVALUATE, "OOM")
        assert s.current_phase == TrainingSequence.FAILED
        assert s.is_done is True
        assert s.error == "OOM"

    def test_skip_phase(self):
        s = TrainingSequenceState()
        s.skip_phase(TrainingSequence.DEPLOY, "not needed")
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
        s.start_phase(TrainingSequence.GENERATE_DATA)
        ev = s.to_sse_event("auto-train")
        assert ev["stream"] == "auto-train"
        assert ev["phase"] == "generate_data"
        assert ev["status"] == "working"

    def test_to_sse_event_complete(self):
        s = TrainingSequenceState()
        s.current_phase = TrainingSequence.COMPLETE
        ev = s.to_sse_event()
        assert ev["status"] == "complete"

    def test_to_sse_event_error(self):
        s = TrainingSequenceState()
        s.current_phase = TrainingSequence.FAILED
        s.error = "crash"
        ev = s.to_sse_event()
        # FAILED is in is_done, so status is "complete" per to_sse_event logic
        assert ev["status"] == "complete"
        assert ev["meta"]["error"] == "crash"

    def test_is_running_phases(self):
        s = TrainingSequenceState()
        for phase in [TrainingSequence.GENERATE_DATA, TrainingSequence.DISTILL,
                      TrainingSequence.TRAIN, TrainingSequence.EVALUATE,
                      TrainingSequence.DEPLOY]:
            s.current_phase = phase
            assert s.is_running is True, f"{phase} should be running"

    def test_is_done_phases(self):
        s = TrainingSequenceState()
        for phase in [TrainingSequence.COMPLETE, TrainingSequence.FAILED,
                      TrainingSequence.EARLY_STOP]:
            s.current_phase = phase
            assert s.is_done is True, f"{phase} should be done"

    def test_multiple_phases(self):
        s = TrainingSequenceState()
        s.start_phase(TrainingSequence.GENERATE_DATA)
        s.complete_phase(TrainingSequence.GENERATE_DATA)
        s.start_phase(TrainingSequence.DISTILL)
        s.complete_phase(TrainingSequence.DISTILL)
        s.start_phase(TrainingSequence.TRAIN)
        assert len(s.phase_results) == 3
        assert s.phase_results[0].status == "success"
        assert s.phase_results[1].status == "success"
        assert s.phase_results[2].status == "working"


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
        assert len(phases) == 5

    def test_effective_phases_skip_generate(self):
        cfg = TrainingRunConfig(skip_generate=True)
        phases = cfg.effective_phases()
        assert TrainingSequence.GENERATE_DATA not in phases
        assert len(phases) == 4

    def test_effective_phases_skip_all(self):
        cfg = TrainingRunConfig(
            skip_generate=True, skip_distill=True, skip_train=True,
            skip_evaluate=True, skip_deploy=True,
        )
        phases = cfg.effective_phases()
        assert len(phases) == 0

    def test_effective_phases_skip_train(self):
        cfg = TrainingRunConfig(skip_train=True)
        phases = cfg.effective_phases()
        assert TrainingSequence.TRAIN not in phases


class TestCheckpointFormat:
    def test_to_dict(self):
        cf = CheckpointFormat(name="cp1", step=100, loss=0.5, epoch=2)
        d = cf.to_dict()
        assert d["name"] == "cp1"
        assert d["step"] == 100
        assert d["loss"] == 0.5
        assert d["epoch"] == 2

    def test_to_dict_with_traits(self):
        cf = CheckpointFormat(
            name="cp2", step=200, loss=0.3,
            personality_traits={"warmth": 0.8},
        )
        d = cf.to_dict()
        assert d["personality_traits"]["warmth"] == 0.8

    def test_to_dict_with_metadata(self):
        cf = CheckpointFormat(
            name="cp3", step=300, loss=0.1,
            metadata={"dataset": "shakespeare"},
        )
        d = cf.to_dict()
        assert d["dataset"] == "shakespeare"

    def test_defaults(self):
        cf = CheckpointFormat(name="x", step=0, loss=0.0)
        assert cf.val_loss is None
        assert cf.epoch == 0
        assert cf.stoi is None
        assert cf.itos is None
