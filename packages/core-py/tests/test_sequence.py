"""Tests for training sequence — phases, state tracking, config, and checkpoint format."""

from domains.training.sequence import (
    TrainingSequence,
    PhaseResult,
    TrainingSequenceState,
    TrainingRunConfig,
    CheckpointFormat,
)


class TestTrainingSequence:
    def test_ordered_phases(self):
        phases = TrainingSequence.ordered_phases()
        assert phases[0] == TrainingSequence.IDLE
        assert phases[-1] == TrainingSequence.EARLY_STOP
        assert len(phases) == 9

    def test_all_phases_have_values(self):
        for p in TrainingSequence:
            assert isinstance(p.value, str)
            assert len(p.value) > 0

    def test_failed_is_terminal(self):
        assert TrainingSequence.FAILED.value == "failed"


class TestPhaseResult:
    def test_creation(self):
        pr = PhaseResult(phase=TrainingSequence.TRAIN)
        assert pr.status == "working"
        assert pr.message == ""
        assert pr.metrics == {}

    def test_to_dict(self):
        pr = PhaseResult(phase=TrainingSequence.EVALUATE, status="success", metrics={"loss": 0.5})
        d = pr.to_dict()
        assert d["phase"] == "evaluate"
        assert d["status"] == "success"
        assert d["metrics"]["loss"] == 0.5


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
        assert len(s.phase_results) == 1

    def test_complete_phase(self):
        s = TrainingSequenceState()
        s.start_phase(TrainingSequence.TRAIN)
        s.complete_phase(TrainingSequence.TRAIN, metrics={"loss": 0.3})
        assert s.phase_results[0].status == "success"
        assert s.phase_results[0].metrics["loss"] == 0.3

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

    def test_full_lifecycle(self):
        s = TrainingSequenceState()
        s.start_phase(TrainingSequence.GENERATE_DATA)
        s.complete_phase(TrainingSequence.GENERATE_DATA, {"samples": 100})
        s.start_phase(TrainingSequence.TRAIN)
        s.complete_phase(TrainingSequence.TRAIN, {"loss": 0.5})
        s.current_phase = TrainingSequence.COMPLETE
        assert s.is_done is True
        assert len(s.phase_results) == 2

    def test_to_dict(self):
        s = TrainingSequenceState()
        s.start_phase(TrainingSequence.TRAIN)
        d = s.to_dict()
        assert "current_phase" in d
        assert "phase_results" in d
        assert "is_running" in d

    def test_to_sse_event(self):
        s = TrainingSequenceState()
        s.start_phase(TrainingSequence.TRAIN)
        event = s.to_sse_event()
        assert event["stream"] == "auto-train"
        assert event["phase"] == "train"
        assert event["status"] == "working"


class TestTrainingRunConfig:
    def test_defaults(self):
        c = TrainingRunConfig.defaults()
        assert c.max_epochs == 10
        assert c.skip_train is False

    def test_effective_phases_all(self):
        c = TrainingRunConfig()
        phases = c.effective_phases()
        assert len(phases) == 5
        assert TrainingSequence.TRAIN in phases

    def test_effective_phases_skip_train(self):
        c = TrainingRunConfig(skip_train=True)
        phases = c.effective_phases()
        assert TrainingSequence.TRAIN not in phases
        assert len(phases) == 4

    def test_effective_phases_skip_all(self):
        c = TrainingRunConfig(
            skip_generate=True, skip_distill=True,
            skip_train=True, skip_evaluate=True, skip_deploy=True,
        )
        phases = c.effective_phases()
        assert len(phases) == 0


class TestCheckpointFormat:
    def test_creation(self):
        cp = CheckpointFormat(name="cp1", step=100, loss=0.5)
        assert cp.name == "cp1"
        assert cp.step == 100
        assert cp.loss == 0.5
        assert cp.val_loss is None

    def test_to_dict(self):
        cp = CheckpointFormat(name="cp2", step=200, loss=0.3, val_loss=0.4, epoch=2)
        d = cp.to_dict()
        assert d["name"] == "cp2"
        assert d["val_loss"] == 0.4
        assert d["epoch"] == 2
