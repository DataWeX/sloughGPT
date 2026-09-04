"""Tests for TrainResult and TrainerProtocol."""
from __future__ import annotations

from domains.training.trainer_protocol import TrainResult, TrainerProtocol


class TestTrainResult:
    def test_defaults(self):
        r = TrainResult()
        assert r.success is True
        assert r.status == "completed"
        assert r.final_loss is None
        assert r.global_step == 0

    def test_get_method(self):
        r = TrainResult(final_loss=0.5)
        assert r.get("final_loss") == 0.5
        assert r.get("missing", "default") == "default"

    def test_getitem(self):
        r = TrainResult(final_loss=0.3)
        assert r["final_loss"] == 0.3
        try:
            _ = r["nonexistent"]
            assert False, "should raise"
        except KeyError:
            pass

    def test_contains(self):
        r = TrainResult()
        assert "success" in r
        assert "nonexistent" not in r

    def test_checkpoint_alias(self):
        r = TrainResult(checkpoint_name="cp1")
        assert r.checkpoint == "cp1"

    def test_to_dict(self):
        r = TrainResult(final_loss=1.0, method="hf")
        d = r.to_dict()
        assert d["final_loss"] == 1.0
        assert d["method"] == "hf"
        assert d["success"] is True

    def test_backward_compat_aliases(self):
        r = TrainResult(message="done", elapsed=2.5, phases=["train", "eval"])
        assert r.message == "done"
        assert r.elapsed == 2.5
        assert r.phases == ["train", "eval"]

    def test_dict_merge_metrics(self):
        r = TrainResult(metrics={"perplexity": 10.0})
        d = r.to_dict()
        assert d["perplexity"] == 10.0


class TestTrainerProtocol:
    def test_satisfies_protocol(self):
        class GoodTrainer:
            def train(self, **kwargs):
                return TrainResult()
            @property
            def is_training(self):
                return False
            def stop(self):
                pass

        assert isinstance(GoodTrainer(), TrainerProtocol)

    def test_rejects_incomplete(self):
        class BadTrainer:
            pass

        assert not isinstance(BadTrainer(), TrainerProtocol)
