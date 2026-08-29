"""Tests for domains.training.trainer_protocol — TrainResult; domains.inference.embeddings — EmbeddingProvider, EmbeddingResult."""

from domains.training.trainer_protocol import TrainResult
from domains.inference.embeddings import EmbeddingProvider, EmbeddingResult


class TestTrainResult:
    def test_defaults(self):
        r = TrainResult()
        assert r.success is True
        assert r.status == "completed"
        assert r.final_loss is None
        assert r.metrics == {}

    def test_get(self):
        r = TrainResult(final_loss=0.5)
        assert r.get("final_loss") == 0.5
        assert r.get("nonexistent", "default") == "default"

    def test_getitem(self):
        r = TrainResult(final_loss=0.5)
        assert r["final_loss"] == 0.5
        try:
            _ = r["nonexistent"]
            assert False, "Should have raised KeyError"
        except KeyError:
            pass

    def test_contains(self):
        r = TrainResult(final_loss=0.5)
        assert "final_loss" in r
        assert "nonexistent" not in r

    def test_to_dict(self):
        r = TrainResult(final_loss=0.5, method="slonet", error=None)
        d = r.to_dict()
        assert isinstance(d, dict)
        assert d["final_loss"] == 0.5
        assert d["method"] == "slonet"
        assert "checkpoint" in d

    def test_checkpoint_alias(self):
        r = TrainResult(checkpoint_name="cp1")
        assert r.checkpoint == "cp1"

    def test_failure(self):
        r = TrainResult(success=False, error="OOM", status="failed")
        assert r.success is False
        assert r.error == "OOM"


class TestEmbeddingProvider:
    def test_all_members(self):
        assert len(EmbeddingProvider) == 4

    def test_values(self):
        assert EmbeddingProvider.OPENAI.value == "openai"
        assert EmbeddingProvider.IN_MEMORY.value == "in_memory"


class TestEmbeddingResult:
    def test_fields(self):
        er = EmbeddingResult(embedding=[0.1, 0.2], model="test", dimension=2, token_count=5)
        assert er.embedding == [0.1, 0.2]
        assert er.dimension == 2
        assert er.token_count == 5

    def test_defaults(self):
        er = EmbeddingResult(embedding=[0.1], model="m", dimension=1)
        assert er.token_count is None
