"""
Tests for TrainerProtocol and TrainResult.
"""

from dataclasses import dataclass
from domains.training.trainer_protocol import TrainerProtocol, TrainResult


def test_train_result_defaults():
    """Default TrainResult should be successful."""
    r = TrainResult()
    assert r.success is True
    assert r.status == "completed"
    assert r.final_loss is None
    assert r.model_path is None
    assert r.error is None


def test_train_result_all_fields():
    """TrainResult accepts all fields via constructor."""
    r = TrainResult(
        success=False,
        status="failed",
        final_loss=2.5,
        best_eval_loss=2.3,
        global_step=100,
        total_steps=200,
        epochs_completed=5,
        model_path="/tmp/test",
        checkpoint_name="ckpt-100",
        method="hf",
        metrics={"perplexity": 10.0, "bleu": 0.85},
        error="OOM",
    )
    assert r.success is False
    assert r.status == "failed"
    assert r.final_loss == 2.5
    assert r.best_eval_loss == 2.3
    assert r.global_step == 100
    assert r.total_steps == 200
    assert r.epochs_completed == 5
    assert r.model_path == "/tmp/test"
    assert r.checkpoint_name == "ckpt-100"
    assert r.method == "hf"
    assert r.metrics == {"perplexity": 10.0, "bleu": 0.85}
    assert r.error == "OOM"


def test_train_result_get():
    """TrainResult.get() provides dict-like access."""
    r = TrainResult(final_loss=1.0)
    assert r.get("final_loss") == 1.0
    assert r.get("success") is True
    assert r.get("nonexistent", "fallback") == "fallback"


def test_train_result_getitem():
    """TrainResult supports dict-like key access via __getitem__."""
    r = TrainResult(status="no_data")
    assert r["status"] == "no_data"
    assert r["success"] is True


def test_train_result_to_dict():
    """TrainResult.to_dict() returns flat dict with all fields plus metrics."""
    r = TrainResult(
        success=True,
        status="completed",
        final_loss=0.5,
        model_path="/tmp/model",
        method="hf",
        metrics={"perplexity": 8.0},
    )
    d = r.to_dict()
    assert d["success"] is True
    assert d["status"] == "completed"
    assert d["final_loss"] == 0.5
    assert d["model_path"] == "/tmp/model"
    assert d["method"] == "hf"
    assert d["perplexity"] == 8.0
    assert d["error"] is None


def test_trainer_protocol_runtime_checkable():
    """A class with train(), is_training, stop() should satisfy TrainerProtocol."""

    @dataclass
    class DummyTrainer:
        done: bool = False

        def train(self, **kwargs) -> TrainResult:
            return TrainResult(success=True, method="dummy")

        @property
        def is_training(self) -> bool:
            return not self.done

        def stop(self) -> None:
            self.done = True

    t = DummyTrainer()
    assert isinstance(t, TrainerProtocol)
    result = t.train()
    assert isinstance(result, TrainResult)
    assert result.success is True
    assert result.method == "dummy"

    assert t.is_training is True
    t.stop()
    assert t.is_training is False


def test_trainer_protocol_structural_missing_method():
    """A class missing required methods should NOT satisfy TrainerProtocol."""

    class IncompleteTrainer:
        def train(self, **kwargs) -> TrainResult:
            return TrainResult()

    t = IncompleteTrainer()
    assert not isinstance(t, TrainerProtocol)
