"""Tests for domains.training.sequence — TrainingSequence, PhaseResult, TrainingRunConfig, CheckpointFormat, TrainingSequenceState."""

from domains.training.sequence import (
    TrainingSequence, PhaseResult, TrainingRunConfig, CheckpointFormat, TrainingSequenceState,
)


class TestTrainingSequence:
    def test_all_members(self):
        assert len(TrainingSequence) == 9

    def test_values(self):
        assert TrainingSequence.IDLE.value == "idle"
        assert TrainingSequence.GENERATE_DATA.value == "generate_data"
        assert TrainingSequence.COMPLETE.value == "complete"
        assert TrainingSequence.FAILED.value == "failed"

    def test_ordered_phases(self):
        phases = TrainingSequence.ordered_phases()
        assert len(phases) == 9
        assert phases[0] == TrainingSequence.IDLE
        assert phases[-1] == TrainingSequence.EARLY_STOP


class TestPhaseResult:
    def test_defaults(self):
        pr = PhaseResult(phase=TrainingSequence.TRAIN)
        assert pr.phase == TrainingSequence.TRAIN
        assert pr.status == "working"
        assert pr.message == ""
        assert pr.metrics == {}

    def test_to_dict(self):
        pr = PhaseResult(phase=TrainingSequence.EVALUATE, status="success", metrics={"loss": 0.5})
        d = pr.to_dict()
        assert d["phase"] == "evaluate"
        assert d["status"] == "success"
        assert d["metrics"]["loss"] == 0.5


class TestTrainingRunConfig:
    def test_defaults(self):
        trc = TrainingRunConfig()
        assert trc.skip_generate is False
        assert trc.max_epochs == 10
        assert trc.early_stop_patience == 3
        assert trc.deploy_on_complete is True

    def test_defaults_classmethod(self):
        trc = TrainingRunConfig.defaults()
        assert isinstance(trc, TrainingRunConfig)

    def test_effective_phases(self):
        trc = TrainingRunConfig()
        phases = trc.effective_phases()
        assert len(phases) == 5

    def test_skip_phases(self):
        trc = TrainingRunConfig(skip_generate=True, skip_deploy=True)
        phases = trc.effective_phases()
        assert len(phases) == 3
        assert TrainingSequence.GENERATE_DATA not in phases
        assert TrainingSequence.DEPLOY not in phases


class TestCheckpointFormat:
    def test_fields(self):
        cf = CheckpointFormat(name="cp1", step=100, loss=0.5)
        assert cf.name == "cp1"
        assert cf.step == 100
        assert cf.loss == 0.5
        assert cf.val_loss is None

    def test_to_dict(self):
        cf = CheckpointFormat(name="cp1", step=100, loss=0.5, val_loss=0.6, epoch=5)
        d = cf.to_dict()
        assert d["name"] == "cp1"
        assert d["loss"] == 0.5
        assert d["val_loss"] == 0.6


class TestTrainingSequenceState:
    def test_init(self):
        tss = TrainingSequenceState()
        assert tss.current_phase == TrainingSequence.IDLE
        assert len(tss.phase_results) == 0

    def test_start_phase(self):
        tss = TrainingSequenceState()
        tss.start_phase(TrainingSequence.TRAIN)
        assert tss.current_phase == TrainingSequence.TRAIN

    def test_complete_phase(self):
        tss = TrainingSequenceState()
        tss.start_phase(TrainingSequence.TRAIN)
        tss.complete_phase(TrainingSequence.TRAIN, metrics={"loss": 0.5})
        assert len(tss.phase_results) == 1
        assert tss.phase_results[0].status == "success"
        assert tss.phase_results[0].metrics["loss"] == 0.5
