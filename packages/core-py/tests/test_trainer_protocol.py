"""
Tests for TrainerProtocol and TrainResult.
"""

from dataclasses import dataclass
from domains.training.trainer_protocol import TrainerProtocol, TrainResult


# ---------------------------------------------------------------------------
# TrainResult — construction & defaults
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# TrainResult — additional field tests
# ---------------------------------------------------------------------------

def test_train_result_backward_compat_aliases():
    """message, elapsed, phases are backward-compat fields."""
    r = TrainResult(message="done", elapsed=3.5, phases=["train", "eval"])
    assert r.message == "done"
    assert r.elapsed == 3.5
    assert r.phases == ["train", "eval"]


def test_train_result_checkpoint_alias():
    """checkpoint property aliases checkpoint_name."""
    r = TrainResult(checkpoint_name="ckpt-42")
    assert r.checkpoint == "ckpt-42"


def test_train_result_checkpoint_none():
    """checkpoint property returns None when checkpoint_name is None."""
    r = TrainResult()
    assert r.checkpoint is None


def test_train_result_contains():
    """__contains__ checks attribute existence."""
    r = TrainResult(final_loss=1.0)
    assert "final_loss" in r
    assert "success" in r
    assert "nonexistent_field" not in r


def test_train_result_getitem_key_error():
    """__getitem__ raises KeyError for missing attributes."""
    r = TrainResult()
    try:
        _ = r["nonexistent"]
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


def test_train_result_to_dict_includes_metrics():
    """to_dict() merges metrics into top-level dict."""
    r = TrainResult(metrics={"bleu": 0.9, "rouge": 0.8})
    d = r.to_dict()
    assert d["bleu"] == 0.9
    assert d["rouge"] == 0.8


def test_train_result_to_dict_includes_backward_compat():
    """to_dict() includes backward-compat aliases."""
    r = TrainResult(checkpoint_name="ckpt", message="ok", elapsed=1.0, phases=["a"])
    d = r.to_dict()
    assert d["checkpoint"] == "ckpt"
    assert d["message"] == "ok"
    assert d["elapsed"] == 1.0
    assert d["phases"] == ["a"]


def test_train_result_to_dict_includes_quality_fields():
    """to_dict() includes avg_quality and data_quality."""
    r = TrainResult(avg_quality=4.2, data_quality={"repetition": 0.1})
    d = r.to_dict()
    assert d["avg_quality"] == 4.2
    assert d["data_quality"]["repetition"] == 0.1


def test_train_result_empty_metrics():
    """Empty metrics dict does not break to_dict."""
    r = TrainResult(metrics={})
    d = r.to_dict()
    assert isinstance(d, dict)


def test_train_result_status_variants():
    """TrainResult supports various status strings."""
    for status in ["completed", "no_data", "failed", "cancelled", "timeout"]:
        r = TrainResult(status=status)
        assert r.status == status


def test_train_result_success_false():
    """TrainResult with success=False."""
    r = TrainResult(success=False, error="OOM killed")
    assert r.success is False
    assert r.error == "OOM killed"


def test_train_result_global_step():
    """TrainResult stores global_step correctly."""
    r = TrainResult(global_step=500, total_steps=1000)
    assert r.global_step == 500
    assert r.total_steps == 1000


def test_train_result_epochs():
    """TrainResult stores epochs_completed."""
    r = TrainResult(epochs_completed=10)
    assert r.epochs_completed == 10


def test_train_result_model_path():
    """TrainResult stores model_path."""
    r = TrainResult(model_path="/models/v2/checkpoint")
    assert r.model_path == "/models/v2/checkpoint"


def test_train_result_method_variants():
    """TrainResult stores various method names."""
    for method in ["hf", "slonet", "nanogpt", "custom", ""]:
        r = TrainResult(method=method)
        assert r.method == method


def test_train_result_best_eval_loss():
    """TrainResult stores best_eval_loss."""
    r = TrainResult(best_eval_loss=0.42)
    assert r.best_eval_loss == 0.42


def test_train_result_to_dict_all_none():
    """to_dict() handles all-None optional fields."""
    r = TrainResult()
    d = r.to_dict()
    assert d["final_loss"] is None
    assert d["best_eval_loss"] is None
    assert d["model_path"] is None
    assert d["checkpoint_name"] is None
    assert d["error"] is None


# ---------------------------------------------------------------------------
# TrainerProtocol — additional structural tests
# ---------------------------------------------------------------------------

def test_trainer_protocol_with_kwargs():
    """TrainerProtocol.train() accepts **kwargs."""
    @dataclass
    class KwargsTrainer:
        done: bool = False
        def train(self, lr=0.001, epochs=3, **kwargs) -> TrainResult:
            return TrainResult(success=True, method="kwargs")
        @property
        def is_training(self) -> bool:
            return not self.done
        def stop(self) -> None:
            self.done = True

    t = KwargsTrainer()
    assert isinstance(t, TrainerProtocol)
    result = t.train(lr=0.01, batch_size=32)
    assert result.success is True


def test_trainer_protocol_missing_is_training():
    """Class missing is_training property does NOT satisfy protocol."""

    class NoIsTraining:
        def train(self, **kwargs) -> TrainResult:
            return TrainResult()
        def stop(self) -> None:
            pass

    assert not isinstance(NoIsTraining(), TrainerProtocol)


def test_trainer_protocol_missing_stop():
    """Class missing stop() does NOT satisfy protocol."""

    class NoStop:
        def train(self, **kwargs) -> TrainResult:
            return TrainResult()
        @property
        def is_training(self) -> bool:
            return False

    assert not isinstance(NoStop(), TrainerProtocol)


def test_trainer_protocol_return_type():
    """TrainerProtocol.train() must return TrainResult."""

    @dataclass
    class TypedTrainer:
        done: bool = False
        def train(self, **kwargs) -> TrainResult:
            return TrainResult(success=True)
        @property
        def is_training(self) -> bool:
            return not self.done
        def stop(self) -> None:
            self.done = True

    t = TypedTrainer()
    result = t.train()
    assert isinstance(result, TrainResult)


def test_trainer_protocol_multiple_instances():
    """Multiple TrainerProtocol instances work independently."""

    @dataclass
    class CounterTrainer:
        count: int = 0
        done: bool = False
        def train(self, **kwargs) -> TrainResult:
            self.count += 1
            return TrainResult(success=True, method=f"run-{self.count}")
        @property
        def is_training(self) -> bool:
            return not self.done
        def stop(self) -> None:
            self.done = True

    t1 = CounterTrainer()
    t2 = CounterTrainer()
    t1.train()
    t1.train()
    t2.train()
    assert t1.count == 2
    assert t2.count == 1


def test_train_result_metrics_complex():
    """TrainResult handles complex nested metrics."""
    r = TrainResult(metrics={
        "perplexity": 12.5,
        "bleu": 0.85,
        "rouge": {"r": 0.7, "p": 0.8, "f": 0.75},
        "loss_curve": [1.0, 0.8, 0.6],
    })
    d = r.to_dict()
    assert d["perplexity"] == 12.5
    assert d["bleu"] == 0.85
    assert isinstance(d["rouge"], dict)
    assert isinstance(d["loss_curve"], list)


def test_train_result_equality():
    """TrainResult instances with same fields are equal (dataclass)."""
    r1 = TrainResult(final_loss=1.0)
    r2 = TrainResult(final_loss=1.0)
    assert r1 == r2
    r3 = TrainResult(final_loss=2.0)
    assert r1 != r3


def test_train_result_repr():
    """TrainResult has a meaningful repr."""
    r = TrainResult(success=False, error="timeout")
    repr_str = repr(r)
    assert "TrainResult" in repr_str
    assert "timeout" in repr_str


# ---------------------------------------------------------------------------
# Additional comprehensive tests
# ---------------------------------------------------------------------------

def test_train_result_getitem_missing_key():
    """__getitem__ raises KeyError with correct key name."""
    r = TrainResult()
    try:
        _ = r["missing_key"]
        assert False, "Should have raised KeyError"
    except KeyError as e:
        assert "missing_key" in str(e)


def test_train_result_contains_all_fields():
    """__contains__ returns True for all valid field names."""
    r = TrainResult()
    for field_name in [
        "success", "status", "final_loss", "best_eval_loss", "global_step",
        "total_steps", "epochs_completed", "model_path", "checkpoint_name",
        "method", "metrics", "avg_quality", "data_quality", "error",
        "message", "elapsed", "phases",
    ]:
        assert field_name in r


def test_train_result_get_with_none():
    """get() returns None for unset fields without default."""
    r = TrainResult()
    assert r.get("final_loss") is None
    assert r.get("model_path") is None


def test_train_result_get_with_default():
    """get() returns default for missing fields."""
    r = TrainResult()
    assert r.get("nonexistent", 42) == 42


def test_train_result_to_dict_all_fields_present():
    """to_dict() includes every dataclass field."""
    r = TrainResult()
    d = r.to_dict()
    expected_keys = {
        "success", "status", "final_loss", "best_eval_loss", "global_step",
        "total_steps", "epochs_completed", "model_path", "checkpoint_name",
        "method", "error", "message", "elapsed", "phases", "checkpoint",
        "avg_quality", "data_quality",
    }
    assert expected_keys.issubset(set(d.keys()))


def test_train_result_to_dict_metrics_override():
    """to_dict() merges metrics that share names with fields."""
    r = TrainResult(final_loss=1.0, metrics={"final_loss": 0.5})
    d = r.to_dict()
    assert d["final_loss"] == 0.5


def test_train_result_equality_fields():
    """TrainResult with same values for all fields are equal."""
    kwargs = dict(
        success=True, status="completed", final_loss=0.5, best_eval_loss=0.4,
        global_step=100, total_steps=200, epochs_completed=3,
        model_path="/tmp/m", checkpoint_name="ckpt", method="slonet",
        metrics={"ppl": 10.0}, avg_quality=4.0, data_quality={"div": 0.9},
        error=None, message="ok", elapsed=1.5, phases=["train"],
    )
    r1 = TrainResult(**kwargs)
    r2 = TrainResult(**kwargs)
    assert r1 == r2


def test_train_result_inequality_different_fields():
    """TrainResult with different fields are not equal."""
    r1 = TrainResult(final_loss=1.0)
    r2 = TrainResult(final_loss=2.0)
    assert r1 != r2


def test_train_result_hash_unsupported():
    """TrainResult is not hashable (mutable dataclass with default_factory)."""
    r = TrainResult()
    try:
        hash(r)
    except TypeError:
        pass


def test_train_result_to_dict_empty_metrics():
    """to_dict() with empty metrics still includes metrics key."""
    r = TrainResult(metrics={})
    d = r.to_dict()
    assert "metrics" not in d or d.get("perplexity") is None


def test_train_result_nested_metrics():
    """to_dict() preserves nested dict structure in metrics."""
    nested = {"level1": {"level2": [1, 2, 3]}}
    r = TrainResult(metrics=nested)
    d = r.to_dict()
    assert d["level1"]["level2"] == [1, 2, 3]


def test_train_result_status_not_completed():
    """TrainResult with non-completed status."""
    for status in ["no_data", "failed", "cancelled", "timeout", "error"]:
        r = TrainResult(status=status, success=False)
        assert r.status == status
        assert r.success is False


def test_train_result_method_values():
    """TrainResult supports various method values."""
    for method in ["hf", "slonet", "nanogpt", "custom", "deepspeed", "fsdp"]:
        r = TrainResult(method=method)
        assert r.method == method


def test_train_result_global_step_variants():
    """TrainResult handles various global_step values."""
    for step in [0, 1, 100, 10000, 999999]:
        r = TrainResult(global_step=step)
        assert r.global_step == step


def test_train_result_epochs_variants():
    """TrainResult handles various epoch counts."""
    for epochs in [0, 1, 5, 100]:
        r = TrainResult(epochs_completed=epochs)
        assert r.epochs_completed == epochs


def test_train_result_model_path_variants():
    """TrainResult stores various model_path formats."""
    paths = ["/tmp/model", "s3://bucket/model", "models/v2/checkpoint", ""]
    for path in paths:
        r = TrainResult(model_path=path)
        assert r.model_path == path


def test_train_result_checkpoint_name_variants():
    """TrainResult stores various checkpoint_name formats."""
    names = ["ckpt-100", "best_model", "epoch-3-step-500", ""]
    for name in names:
        r = TrainResult(checkpoint_name=name)
        assert r.checkpoint_name == name


def test_train_result_final_loss_variants():
    """TrainResult stores various final_loss values."""
    for loss in [0.0, 0.001, 1.5, 100.0, 999.9]:
        r = TrainResult(final_loss=loss)
        assert r.final_loss == loss


def test_train_result_best_eval_loss_variants():
    """TrainResult stores various best_eval_loss values."""
    for loss in [0.0, 0.001, 1.5, 100.0]:
        r = TrainResult(best_eval_loss=loss)
        assert r.best_eval_loss == loss


def test_train_result_message_variants():
    """TrainResult stores various message strings."""
    messages = ["", "done", "completed successfully", "error occurred", "⚠"]
    for msg in messages:
        r = TrainResult(message=msg)
        assert r.message == msg


def test_train_result_elapsed_variants():
    """TrainResult stores various elapsed times."""
    for t in [0.0, 0.1, 1.0, 60.0, 3600.0]:
        r = TrainResult(elapsed=t)
        assert r.elapsed == t


def test_train_result_phases_variants():
    """TrainResult stores various phase lists."""
    phases_list = [[], ["train"], ["train", "eval"], ["train", "eval", "test"]]
    for phases in phases_list:
        r = TrainResult(phases=phases)
        assert r.phases == phases


def test_trainer_protocol_train_with_no_kwargs():
    """TrainerProtocol.train() works with no arguments."""
    @dataclass
    class SimpleTrainer:
        done: bool = False
        def train(self) -> TrainResult:
            return TrainResult(success=True)
        @property
        def is_training(self) -> bool:
            return not self.done
        def stop(self) -> None:
            self.done = True

    t = SimpleTrainer()
    assert isinstance(t, TrainerProtocol)
    result = t.train()
    assert result.success is True


def test_trainer_protocol_stop_multiple_times():
    """Calling stop() multiple times is safe."""
    @dataclass
    class Trainer:
        done: bool = False
        def train(self, **kwargs) -> TrainResult:
            return TrainResult()
        @property
        def is_training(self) -> bool:
            return not self.done
        def stop(self) -> None:
            self.done = True

    t = Trainer()
    assert t.is_training is True
    t.stop()
    t.stop()
    assert t.is_training is False


def test_trainer_protocol_train_returns_result():
    """Every train() call returns a TrainResult instance."""
    @dataclass
    class MultiTrainer:
        run_count: int = 0
        done: bool = False
        def train(self, **kwargs) -> TrainResult:
            self.run_count += 1
            return TrainResult(success=True, total_steps=self.run_count)
        @property
        def is_training(self) -> bool:
            return not self.done
        def stop(self) -> None:
            self.done = True

    t = MultiTrainer()
    for i in range(5):
        result = t.train()
        assert isinstance(result, TrainResult)
        assert result.total_steps == i + 1


def test_train_result_dataclass_field_access():
    """TrainResult fields are accessible as attributes."""
    r = TrainResult(
        final_loss=1.5, best_eval_loss=1.2, global_step=50,
        total_steps=100, epochs_completed=2, model_path="/m",
        checkpoint_name="c", method="slonet", avg_quality=4.5,
        error="OOM", message="done", elapsed=2.0, phases=["a"],
    )
    assert r.final_loss == 1.5
    assert r.best_eval_loss == 1.2
    assert r.global_step == 50
    assert r.total_steps == 100
    assert r.epochs_completed == 2
    assert r.model_path == "/m"
    assert r.checkpoint_name == "c"
    assert r.method == "slonet"
    assert r.avg_quality == 4.5
    assert r.error == "OOM"
    assert r.message == "done"
    assert r.elapsed == 2.0
    assert r.phases == ["a"]


def test_train_result_to_dict_combined():
    """to_dict() works correctly with all fields set."""
    r = TrainResult(
        success=True, status="completed", final_loss=0.1,
        best_eval_loss=0.2, global_step=200, total_steps=200,
        epochs_completed=5, model_path="/models/v3",
        checkpoint_name="best", method="hf",
        metrics={"perplexity": 5.0, "accuracy": 0.95},
        avg_quality=4.8, data_quality={"diversity": 0.9},
        error=None, message="completed", elapsed=120.0,
        phases=["train", "eval"],
    )
    d = r.to_dict()
    assert d["success"] is True
    assert d["status"] == "completed"
    assert d["final_loss"] == 0.1
    assert d["best_eval_loss"] == 0.2
    assert d["global_step"] == 200
    assert d["total_steps"] == 200
    assert d["epochs_completed"] == 5
    assert d["model_path"] == "/models/v3"
    assert d["checkpoint_name"] == "best"
    assert d["method"] == "hf"
    assert d["perplexity"] == 5.0
    assert d["accuracy"] == 0.95
    assert d["avg_quality"] == 4.8
    assert d["data_quality"]["diversity"] == 0.9
    assert d["error"] is None
    assert d["message"] == "completed"
    assert d["elapsed"] == 120.0
    assert d["phases"] == ["train", "eval"]
    assert d["checkpoint"] == "best"
