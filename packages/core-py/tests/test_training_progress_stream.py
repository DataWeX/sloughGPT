"""
Training Progress Streaming Tests — verifies SSE-based training progress.

Tests the training progress streaming pipeline: trainer → progress callbacks → SSE events.
Validates that training progress is properly streamed to connected clients.

Usage:
    .venv/bin/python -m pytest tests/test_training_progress_stream.py -x -v
"""
import tempfile

FAST_CONFIG = {
    "method": "sft",
    "data_quality_threshold": 0.0,
    "epochs": 1,
    "batch_size": 8,
    "block_size": 32,
    "max_steps": 2,
    "n_embed": 32,
    "n_layer": 2,
    "n_head": 2,
}

DATA_TEXT = "The quick brown fox jumps over the lazy dog. " * 50


class TestTrainingProgressCallbacks:
    """Tests the training progress callback system."""

    def test_progress_callback_receives_events(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainer = ComprehensiveTrainer()
            events = []
            trainer.on_progress(lambda d: events.append(d))
            trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)
            assert len(events) >= 4
            assert events[0]["phase"] == "validating"
            assert events[-1]["phase"] == "complete"

    def test_progress_callback_has_run_id(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainer = ComprehensiveTrainer()
            events = []
            trainer.on_progress(lambda d: events.append(d))
            result = trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)
            assert all("run_id" in e for e in events)
            assert events[0]["run_id"] == result.run_id

    def test_progress_callback_has_progress_percentage(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainer = ComprehensiveTrainer()
            events = []
            trainer.on_progress(lambda d: events.append(d))
            trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)
            assert all("progress" in e for e in events)
            assert events[0]["progress"] < events[-1]["progress"]
            assert events[-1]["progress"] == 1.0

    def test_progress_events_are_monotonic(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainer = ComprehensiveTrainer()
            events = []
            trainer.on_progress(lambda d: events.append(d))
            trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)
            progress_values = [e["progress"] for e in events]
            assert progress_values == sorted(progress_values)

    def test_multiple_progress_callbacks(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainer = ComprehensiveTrainer()
            events1 = []
            events2 = []
            trainer.on_progress(lambda d: events1.append(d))
            trainer.on_progress(lambda d: events2.append(d))
            trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)
            assert len(events1) == len(events2)

    def test_progress_callback_error_does_not_crash(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainer = ComprehensiveTrainer()
            bad_called = [False]
            def bad_callback(d):
                bad_called[0] = True
                raise RuntimeError("callback error")
            good_called = [False]
            def good_callback(d):
                good_called[0] = True
            trainer.on_progress(bad_callback)
            trainer.on_progress(good_callback)
            result = trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)
            assert result.success is True
            assert good_called[0]


class TestSSEEnvelopeIntegration:
    """Tests SSE envelope format for training progress events."""

    def test_sse_event_creation(self):
        from domains.api.sse_envelope import sse_event

        event = sse_event(
            stream="training",
            phase="TRAINING",
            status="working",
            data={"progress": 0.5, "loss": 1.23},
        )
        assert "data:" in event
        assert "training" in event
        assert "TRAINING" in event

    def test_sse_envelope_creation(self):
        from domains.api.sse_envelope import SSEEnvelope, StreamPhase, StreamStatus

        env = SSEEnvelope(
            stream="training",
            phase=StreamPhase.TRAIN.value,
            status=StreamStatus.SUCCESS.value,
            data={"progress": 0.5},
        )
        assert env.stream == "training"
        assert env.phase == StreamPhase.TRAIN.value

    def test_sse_training_event_format(self):
        from domains.api.sse_envelope import sse_event

        event = sse_event(
            stream="training",
            phase="TRAINING",
            status="working",
            data={
                "global_step": 10,
                "epoch": 1,
                "train_loss": 1.5,
                "progress_percent": 50,
            },
        )
        assert "global_step" in event
        assert "train_loss" in event


class TestTrainingQueueSSE:
    """Tests the training queue SSE integration."""

    def test_training_queue_event_format(self):
        from domains.infrastructure.training_queue import _json_safe_payload

        data = {
            "progress": 0.5,
            "loss": float("inf"),
            "step": 10,
            "nested": {"a": float("nan")},
        }
        safe = _json_safe_payload(data)
        assert safe["progress"] == 0.5
        assert safe["loss"] is None
        assert safe["nested"]["a"] is None

    def test_training_queue_event_buffer(self):
        from domains.infrastructure.event_buffer import get_event_buffer

        buffer = get_event_buffer()
        buffer.record("TRAIN", "test_progress_event")
        recent = buffer.recent(n=100)
        assert len(recent) > 0


class TestProgressStreamingEndToEnd:
    """End-to-end tests for progress streaming through the full pipeline."""

    def test_trainer_emits_all_phase_events(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainer = ComprehensiveTrainer()
            phases_seen = []
            trainer.on_progress(lambda d: phases_seen.append(d.get("phase")))
            trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)
            expected_phases = ["validating", "configuring", "preprocessing", "training", "complete"]
            for phase in expected_phases:
                assert phase in phases_seen, f"Missing phase: {phase}"

    def test_trainer_progress_includes_phase_durations(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)
            assert "phase_durations" in result.performance
            assert "training" in result.performance["phase_durations"]
            assert result.performance["phase_durations"]["training"] > 0

    def test_trainer_result_has_all_phases(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)
            phase_names = [p.phase.value for p in result.phases]
            assert "validating" in phase_names
            assert "configuring" in phase_names
            assert "preprocessing" in phase_names
            assert "training" in phase_names
            assert "recording" in phase_names
