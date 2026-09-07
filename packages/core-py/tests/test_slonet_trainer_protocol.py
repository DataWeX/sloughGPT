"""Tests for training.trainer_protocol — TrainResult."""

from __future__ import annotations

import pytest
from domains.training.trainer_protocol import TrainResult


# ── TrainResult ─────────────────────────────────────────────────────────────


class TestTrainResult:

    def test_default(self):
        result = TrainResult()
        assert result.success is True
        assert result.status == "completed"
        assert result.final_loss is None

    def test_custom(self):
        result = TrainResult(success=False, status="failed", error="test error")
        assert result.success is False
        assert result.error == "test error"

    def test_get(self):
        result = TrainResult(final_loss=0.5)
        assert result.get("final_loss") == 0.5
        assert result.get("missing", "default") == "default"

    def test_getitem(self):
        result = TrainResult(final_loss=0.5)
        assert result["final_loss"] == 0.5

    def test_getitem_missing(self):
        result = TrainResult()
        with pytest.raises(KeyError):
            result["missing"]

    def test_checkpoint_alias(self):
        result = TrainResult(checkpoint_name="test.ckpt")
        assert result.checkpoint == "test.ckpt"

    def test_metrics(self):
        result = TrainResult(metrics={"perplexity": 10.0})
        assert result.metrics["perplexity"] == 10.0
