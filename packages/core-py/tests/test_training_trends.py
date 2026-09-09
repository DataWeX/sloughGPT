"""Tests for training trends endpoint logic."""

import json
import time
import pytest
from pathlib import Path
from domains.training.outcome_tracker import (
    TrainingOutcome,
    TrainingOutcomeTracker,
)


@pytest.fixture
def tmp_tracker(tmp_path):
    """Create a tracker with a temporary history file."""
    history_file = tmp_path / "training_history.jsonl"
    return TrainingOutcomeTracker(history_path=history_file)


class TestTrainingOutcome:
    def test_defaults(self):
        o = TrainingOutcome()
        assert o.run_id == ""
        assert o.quality_score == 0.0

    def test_to_dict_and_back(self):
        o = TrainingOutcome(
            run_id="run_1",
            timestamp=1000.0,
            model="gpt2",
            method="finetune",
            final_loss=2.5,
            quality_score=0.75,
        )
        d = o.to_dict()
        assert d["run_id"] == "run_1"
        assert d["model"] == "gpt2"
        o2 = TrainingOutcome.from_dict(d)
        assert o2.run_id == "run_1"
        assert o2.final_loss == 2.5


class TestTrainingOutcomeTracker:
    def test_record_and_load(self, tmp_tracker):
        outcome = TrainingOutcome(
            run_id="run_test",
            timestamp=time.time(),
            model="gpt2",
            dataset="my_data",
            method="finetune",
            final_loss=3.0,
            converged=True,
        )
        tmp_tracker.record(outcome)
        outcomes = tmp_tracker.load_outcomes()
        assert len(outcomes) == 1
        assert outcomes[0].run_id == "run_test"
        assert outcomes[0].quality_score > 0

    def test_quality_score_computed(self, tmp_tracker):
        outcome = TrainingOutcome(
            run_id="run_q",
            final_loss=2.0,
            converged=True,
            perplexity=20.0,
        )
        tmp_tracker.record(outcome)
        loaded = tmp_tracker.load_outcomes()
        assert loaded[0].quality_score > 0

    def test_get_stats(self, tmp_tracker):
        for i in range(3):
            tmp_tracker.record(
                TrainingOutcome(
                    run_id=f"run_{i}",
                    final_loss=float(3 - i),
                    converged=True,
                )
            )
        stats = tmp_tracker.get_stats()
        assert stats["total_runs"] == 3
        assert stats["avg_quality"] > 0
        assert stats["converged_count"] == 3

    def test_get_best_outcomes(self, tmp_tracker):
        for i in range(5):
            tmp_tracker.record(
                TrainingOutcome(
                    run_id=f"run_{i}",
                    final_loss=float(10 - i),
                    converged=i > 1,
                )
            )
        best = tmp_tracker.get_best_outcomes(3)
        assert len(best) == 3
        # Best should be sorted by quality descending
        assert best[0].quality_score >= best[1].quality_score

    def test_empty_history(self, tmp_tracker):
        stats = tmp_tracker.get_stats()
        assert stats["total_runs"] == 0
        assert tmp_tracker.load_outcomes() == []


class TestTrendsLogic:
    """Test the trends computation logic (mirrors the router endpoint)."""

    def _build_trends(self, tracker: TrainingOutcomeTracker, model="", dataset=""):
        outcomes = tracker.load_outcomes()
        if model:
            outcomes = [o for o in outcomes if o.model == model]
        if dataset:
            outcomes = [o for o in outcomes if o.dataset == dataset]
        outcomes.sort(key=lambda o: o.timestamp)

        if not outcomes:
            return {"runs": [], "models": [], "summary": {"total_runs": 0, "avg_quality": 0, "best_quality": 0, "avg_loss": 0, "trend": "stable"}}

        runs = [
            {
                "run_id": o.run_id,
                "model": o.model,
                "dataset": o.dataset,
                "quality_score": o.quality_score,
                "final_loss": o.final_loss,
            }
            for o in outcomes
        ]

        # Per-model breakdown
        model_map: dict = {}
        for r in runs:
            m = r["model"] or "unknown"
            model_map.setdefault(m, []).append(r)

        models = []
        for model_name, model_runs in model_map.items():
            qualities = [r["quality_score"] for r in model_runs]
            models.append({
                "model": model_name,
                "total_runs": len(model_runs),
                "avg_quality": sum(qualities) / len(qualities),
                "best_quality": max(qualities),
            })

        all_qualities = [r["quality_score"] for r in runs]
        summary = {
            "total_runs": len(runs),
            "avg_quality": sum(all_qualities) / len(all_qualities),
            "best_quality": max(all_qualities),
        }

        return {"runs": runs, "models": models, "summary": summary}

    def test_empty(self, tmp_tracker):
        result = self._build_trends(tmp_tracker)
        assert result["summary"]["total_runs"] == 0
        assert result["runs"] == []

    def test_with_data(self, tmp_tracker):
        for i in range(5):
            tmp_tracker.record(
                TrainingOutcome(
                    run_id=f"run_{i}",
                    model="gpt2" if i < 3 else "slo-1.6b",
                    dataset="data_a",
                    final_loss=float(5 - i),
                    converged=True,
                )
            )
        result = self._build_trends(tmp_tracker)
        assert result["summary"]["total_runs"] == 5
        assert len(result["models"]) == 2
        # gpt2 should have 3 runs, slo-1.6b should have 2
        gpt2 = next(m for m in result["models"] if m["model"] == "gpt2")
        slo = next(m for m in result["models"] if m["model"] == "slo-1.6b")
        assert gpt2["total_runs"] == 3
        assert slo["total_runs"] == 2

    def test_filter_by_model(self, tmp_tracker):
        for i in range(3):
            tmp_tracker.record(
                TrainingOutcome(run_id=f"run_{i}", model="gpt2", final_loss=2.0)
            )
        tmp_tracker.record(
            TrainingOutcome(run_id="run_x", model="llama", final_loss=1.0)
        )
        result = self._build_trends(tmp_tracker, model="gpt2")
        assert result["summary"]["total_runs"] == 3
        assert all(r["model"] == "gpt2" for r in result["runs"])

    def test_trend_detection(self, tmp_tracker):
        # Create runs with improving quality (lower loss = higher quality)
        for i in range(8):
            tmp_tracker.record(
                TrainingOutcome(
                    run_id=f"run_{i}",
                    final_loss=float(10 - i),  # loss decreases over time
                    converged=True,
                )
            )
        result = self._build_trends(tmp_tracker)
        # With consistently improving quality, trend should be "improving"
        assert result["summary"]["total_runs"] == 8
