"""Tests for training sequence — phases, state, config, checkpoint format."""

import pytest
from domains.training.sequence import (
    TrainingSequence, PhaseResult, TrainingSequenceState,
    TrainingRunConfig, CheckpointFormat, DataGenerator, StudentModel,
)


# ── TrainingSequence Enum ──────────────────────────────────────────────────

class TestTrainingSequence:

    def test_ordered_phases_returns_correct_order(self):
        phases = TrainingSequence.ordered_phases()
        expected = [
            TrainingSequence.IDLE,
            TrainingSequence.GENERATE_DATA,
            TrainingSequence.DISTILL,
            TrainingSequence.TRAIN,
            TrainingSequence.EVALUATE,
            TrainingSequence.DEPLOY,
            TrainingSequence.COMPLETE,
            TrainingSequence.FAILED,
            TrainingSequence.EARLY_STOP,
        ]
        assert phases == expected

    def test_enum_values_are_strings(self):
        assert TrainingSequence.IDLE.value == "idle"
        assert TrainingSequence.GENERATE_DATA.value == "generate_data"
        assert TrainingSequence.COMPLETE.value == "complete"
        assert TrainingSequence.FAILED.value == "failed"
        assert TrainingSequence.EARLY_STOP.value == "early_stop"

    def test_all_phases_present(self):
        assert len(TrainingSequence.ordered_phases()) == 9


# ── PhaseResult ────────────────────────────────────────────────────────────

class TestPhaseResult:

    def test_defaults(self):
        pr = PhaseResult(phase=TrainingSequence.TRAIN)
        assert pr.phase == TrainingSequence.TRAIN
        assert pr.status == "working"
        assert pr.message == ""
        assert pr.metrics == {}

    def test_to_dict_all_fields(self):
        pr = PhaseResult(
            phase=TrainingSequence.EVALUATE,
            status="success",
            message="done",
            metrics={"loss": 0.5, "ppl": 12.3},
        )
        d = pr.to_dict()
        assert d["phase"] == "evaluate"
        assert d["status"] == "success"
        assert d["message"] == "done"
        assert d["metrics"]["loss"] == 0.5
        assert d["metrics"]["ppl"] == 12.3

    def test_to_dict_defaults(self):
        pr = PhaseResult(phase=TrainingSequence.DISTILL)
        d = pr.to_dict()
        assert d["status"] == "working"
        assert d["metrics"] == {}


# ── TrainingSequenceState ──────────────────────────────────────────────────

class TestTrainingSequenceState:

    def test_initial_state(self):
        s = TrainingSequenceState()
        assert s.current_phase == TrainingSequence.IDLE
        assert s.is_running is False
        assert s.is_done is False
        assert s.phase_results == []
        assert s.error is None

    def test_start_phase(self):
        s = TrainingSequenceState()
        s.start_phase(TrainingSequence.GENERATE_DATA)
        assert s.current_phase == TrainingSequence.GENERATE_DATA
        assert len(s.phase_results) == 1
        assert s.phase_results[0].phase == TrainingSequence.GENERATE_DATA
        assert s.phase_results[0].status == "working"
        assert s.is_running is True

    def test_complete_phase(self):
        s = TrainingSequenceState()
        s.start_phase(TrainingSequence.DISTILL)
        s.complete_phase(TrainingSequence.DISTILL, metrics={"samples": 100})
        pr = s.phase_results[0]
        assert pr.status == "success"
        assert pr.metrics["samples"] == 100

    def test_complete_phase_no_metrics(self):
        s = TrainingSequenceState()
        s.start_phase(TrainingSequence.TRAIN)
        s.complete_phase(TrainingSequence.TRAIN)
        assert s.phase_results[0].status == "success"
        assert s.phase_results[0].metrics == {}

    def test_fail_phase(self):
        s = TrainingSequenceState()
        s.start_phase(TrainingSequence.EVALUATE)
        s.fail_phase(TrainingSequence.EVALUATE, message="OOM")
        assert s.current_phase == TrainingSequence.FAILED
        assert s.error == "OOM"
        assert s.phase_results[0].status == "error"
        assert s.phase_results[0].message == "OOM"
        assert s.is_done is True

    def test_fail_phase_default_message(self):
        s = TrainingSequenceState()
        s.start_phase(TrainingSequence.TRAIN)
        s.fail_phase(TrainingSequence.TRAIN)
        assert s.error == "Phase train failed"

    def test_skip_phase(self):
        s = TrainingSequenceState()
        s.skip_phase(TrainingSequence.DEPLOY, reason="not needed")
        assert len(s.phase_results) == 1
        assert s.phase_results[0].status == "skipped"
        assert s.phase_results[0].message == "not needed"

    def test_to_dict(self):
        s = TrainingSequenceState()
        s.start_phase(TrainingSequence.TRAIN)
        d = s.to_dict()
        assert d["current_phase"] == "train"
        assert d["is_running"] is True
        assert d["is_done"] is False
        assert d["error"] is None
        assert len(d["phase_results"]) == 1

    def test_to_sse_event(self):
        s = TrainingSequenceState()
        s.start_phase(TrainingSequence.DISTILL)
        event = s.to_sse_event(stream_name="test-stream")
        assert event["stream"] == "test-stream"
        assert event["phase"] == "distill"
        assert event["status"] == "working"
        assert "phase_results" in event["data"]
        assert "meta" in event

    def test_to_sse_event_done(self):
        s = TrainingSequenceState()
        s.current_phase = TrainingSequence.COMPLETE
        event = s.to_sse_event()
        assert event["status"] == "complete"

    def test_is_running_TRAIN(self):
        s = TrainingSequenceState()
        s.current_phase = TrainingSequence.TRAIN
        assert s.is_running is True

    def test_is_running_EVALUATE(self):
        s = TrainingSequenceState()
        s.current_phase = TrainingSequence.EVALUATE
        assert s.is_running is True

    def test_is_done_COMPLETE(self):
        s = TrainingSequenceState()
        s.current_phase = TrainingSequence.COMPLETE
        assert s.is_done is True

    def test_is_done_FAILED(self):
        s = TrainingSequenceState()
        s.current_phase = TrainingSequence.FAILED
        assert s.is_done is True

    def test_is_done_EARLY_STOP(self):
        s = TrainingSequenceState()
        s.current_phase = TrainingSequence.EARLY_STOP
        assert s.is_done is True

    def test_is_not_done_IDLE(self):
        s = TrainingSequenceState()
        s.current_phase = TrainingSequence.IDLE
        assert s.is_done is False

    def test_multiple_phases(self):
        s = TrainingSequenceState()
        s.start_phase(TrainingSequence.GENERATE_DATA)
        s.complete_phase(TrainingSequence.GENERATE_DATA)
        s.start_phase(TrainingSequence.DISTILL)
        s.complete_phase(TrainingSequence.DISTILL)
        s.start_phase(TrainingSequence.TRAIN)
        assert s.current_phase == TrainingSequence.TRAIN
        assert len(s.phase_results) == 3
        assert s.is_running is True

    def test_start_after_complete_sets_running(self):
        s = TrainingSequenceState()
        s.current_phase = TrainingSequence.COMPLETE
        s.start_phase(TrainingSequence.DEPLOY)
        assert s.is_running is True
        assert s.is_done is False


# ── TrainingRunConfig ──────────────────────────────────────────────────────

class TestTrainingRunConfig:

    def test_defaults(self):
        cfg = TrainingRunConfig.defaults()
        assert cfg.skip_generate is False
        assert cfg.skip_distill is False
        assert cfg.skip_train is False
        assert cfg.skip_evaluate is False
        assert cfg.skip_deploy is False
        assert cfg.max_epochs == 10
        assert cfg.early_stop_patience == 3
        assert cfg.early_stop_min_delta == 0.01

    def test_effective_phases_all(self):
        cfg = TrainingRunConfig()
        phases = cfg.effective_phases()
        assert phases == [
            TrainingSequence.GENERATE_DATA,
            TrainingSequence.DISTILL,
            TrainingSequence.TRAIN,
            TrainingSequence.EVALUATE,
            TrainingSequence.DEPLOY,
        ]

    def test_effective_phases_skip_generate(self):
        cfg = TrainingRunConfig(skip_generate=True)
        phases = cfg.effective_phases()
        assert TrainingSequence.GENERATE_DATA not in phases
        assert len(phases) == 4

    def test_effective_phases_skip_distill(self):
        cfg = TrainingRunConfig(skip_distill=True)
        phases = cfg.effective_phases()
        assert TrainingSequence.DISTILL not in phases
        assert len(phases) == 4

    def test_effective_phases_skip_train(self):
        cfg = TrainingRunConfig(skip_train=True)
        phases = cfg.effective_phases()
        assert TrainingSequence.TRAIN not in phases
        assert len(phases) == 4

    def test_effective_phases_skip_evaluate(self):
        cfg = TrainingRunConfig(skip_evaluate=True)
        phases = cfg.effective_phases()
        assert TrainingSequence.EVALUATE not in phases
        assert len(phases) == 4

    def test_effective_phases_skip_deploy(self):
        cfg = TrainingRunConfig(skip_deploy=True)
        phases = cfg.effective_phases()
        assert TrainingSequence.DEPLOY not in phases
        assert len(phases) == 4

    def test_effective_phases_all_skip(self):
        cfg = TrainingRunConfig(
            skip_generate=True, skip_distill=True, skip_train=True,
            skip_evaluate=True, skip_deploy=True,
        )
        assert cfg.effective_phases() == []

    def test_effective_phases_custom(self):
        cfg = TrainingRunConfig(skip_generate=True, skip_deploy=True)
        phases = cfg.effective_phases()
        assert phases == [
            TrainingSequence.DISTILL,
            TrainingSequence.TRAIN,
            TrainingSequence.EVALUATE,
        ]


# ── CheckpointFormat ───────────────────────────────────────────────────────

class TestCheckpointFormat:

    def test_to_dict_required_fields(self):
        cp = CheckpointFormat(name="cp-1", step=100, loss=0.5)
        d = cp.to_dict()
        assert d["name"] == "cp-1"
        assert d["step"] == 100
        assert d["loss"] == 0.5
        assert d["val_loss"] is None
        assert d["epoch"] == 0
        assert d["personality_traits"] is None

    def test_to_dict_with_optional_fields(self):
        cp = CheckpointFormat(
            name="cp-2", step=200, loss=0.3,
            val_loss=0.35, epoch=5,
            personality_traits={"warmth": 0.8, "creativity": 0.6},
            metadata={"bleu": 0.42, "source": "auto"},
        )
        d = cp.to_dict()
        assert d["val_loss"] == 0.35
        assert d["epoch"] == 5
        assert d["personality_traits"]["warmth"] == 0.8
        assert d["bleu"] == 0.42
        assert d["source"] == "auto"

    def test_to_dict_metadata_merged(self):
        cp = CheckpointFormat(
            name="cp-3", step=50, loss=1.0,
            metadata={"extra": True},
        )
        d = cp.to_dict()
        assert d["extra"] is True


# ── Protocols (runtime_checkable) ─────────────────────────────────────────

class TestDataGeneratorProtocol:

    def test_compliant_class_matches(self):
        class Good:
            def generate(self, prompt, num_samples, max_length):
                return []

        assert isinstance(Good(), DataGenerator)

    def test_no_generate_method_fails(self):
        class Bad:
            pass

        assert not isinstance(Bad(), DataGenerator)


class TestStudentModelProtocol:

    def test_compliant_class_matches(self):
        class Good:
            def train_step(self, inputs, labels):
                return 0.0
            def evaluate(self, inputs, labels):
                return {"loss": 0.0}

        assert isinstance(Good(), StudentModel)

    def test_non_compliant_class_no_match(self):
        class Bad:
            def train_step(self, inputs, labels):
                return 0.0

        assert not isinstance(Bad(), StudentModel)
