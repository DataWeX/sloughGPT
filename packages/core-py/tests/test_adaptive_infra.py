"""Tests for adaptive training infrastructure."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from domains.training.outcome_tracker import TrainingOutcome, TrainingOutcomeTracker
from domains.training.adaptive_config import AdaptiveConfigEngine, AdaptiveRecommendation
from domain.settings._internal.persistent import (
    AppSettings,
    GenerationSettings,
    TrainingSettings,
    AdaptiveSettings,
    VoiceSettings,
    PersistentSettings,
)


# ── Outcome Tracker ──────────────────────────────────────────────────────────


class TestTrainingOutcome:

    def test_to_dict(self):
        o = TrainingOutcome(
            run_id="run_1", dataset="data", model="gpt2", final_loss=2.5
        )
        d = o.to_dict()
        assert d["run_id"] == "run_1"
        assert d["final_loss"] == 2.5

    def test_from_dict(self):
        d = {"run_id": "run_2", "model": "gpt2", "final_loss": 1.8, "converged": True}
        o = TrainingOutcome.from_dict(d)
        assert o.run_id == "run_2"
        assert o.final_loss == 1.8
        assert o.converged is True

    def test_from_dict_extra_keys_ignored(self):
        d = {"run_id": "run_3", "unknown_field": 42}
        o = TrainingOutcome.from_dict(d)
        assert o.run_id == "run_3"


class TestTrainingOutcomeTracker:

    def test_record_and_load(self, tmp_path):
        tracker = TrainingOutcomeTracker(tmp_path / "history.jsonl")
        o = TrainingOutcome(
            run_id="run_1", dataset="data", model="gpt2",
            method="finetune", final_loss=2.5, converged=True,
        )
        tracker.record(o)

        outcomes = tracker.load_outcomes()
        assert len(outcomes) == 1
        assert outcomes[0].run_id == "run_1"
        assert outcomes[0].quality_score > 0

    def test_multiple_records(self, tmp_path):
        tracker = TrainingOutcomeTracker(tmp_path / "h.jsonl")
        for i in range(5):
            tracker.record(TrainingOutcome(
                run_id=f"run_{i}", dataset="data", final_loss=5.0 - i,
            ))
        assert len(tracker.load_outcomes()) == 5

    def test_get_best_outcomes(self, tmp_path):
        tracker = TrainingOutcomeTracker(tmp_path / "h.jsonl")
        for i in range(5):
            tracker.record(TrainingOutcome(
                run_id=f"run_{i}", final_loss=5.0 - i, converged=i < 3,
            ))
        best = tracker.get_best_outcomes(2)
        assert len(best) == 2
        # Best should have lowest loss and convergence
        assert best[0].quality_score >= best[1].quality_score

    def test_get_outcomes_for_dataset(self, tmp_path):
        tracker = TrainingOutcomeTracker(tmp_path / "h.jsonl")
        tracker.record(TrainingOutcome(run_id="r1", dataset="a"))
        tracker.record(TrainingOutcome(run_id="r2", dataset="b"))
        tracker.record(TrainingOutcome(run_id="r3", dataset="a"))
        a_outcomes = tracker.get_outcomes_for_dataset("a")
        assert len(a_outcomes) == 2

    def test_get_stats(self, tmp_path):
        tracker = TrainingOutcomeTracker(tmp_path / "h.jsonl")
        tracker.record(TrainingOutcome(
            run_id="r1", final_loss=2.0, converged=True,
        ))
        tracker.record(TrainingOutcome(
            run_id="r2", final_loss=3.0, converged=False, early_stopped=True,
        ))
        stats = tracker.get_stats()
        assert stats["total_runs"] == 2
        assert stats["converged_count"] == 1
        assert stats["early_stopped_count"] == 1

    def test_clear(self, tmp_path):
        tracker = TrainingOutcomeTracker(tmp_path / "h.jsonl")
        tracker.record(TrainingOutcome(run_id="r1"))
        tracker.record(TrainingOutcome(run_id="r2"))
        removed = tracker.clear()
        assert removed == 2
        assert len(tracker.load_outcomes()) == 0

    def test_empty_history(self, tmp_path):
        tracker = TrainingOutcomeTracker(tmp_path / "h.jsonl")
        assert tracker.load_outcomes() == []
        assert tracker.get_stats() == {"total_runs": 0}

    def test_quality_score_converged_beats_not(self, tmp_path):
        tracker = TrainingOutcomeTracker(tmp_path / "h.jsonl")
        tracker.record(TrainingOutcome(
            run_id="good", final_loss=2.0, converged=True,
        ))
        tracker.record(TrainingOutcome(
            run_id="bad", final_loss=4.0, converged=False,
        ))
        outcomes = tracker.load_outcomes()
        good = next(o for o in outcomes if o.run_id == "good")
        bad = next(o for o in outcomes if o.run_id == "bad")
        assert good.quality_score > bad.quality_score


# ── Adaptive Config Engine ───────────────────────────────────────────────────


class TestAdaptiveConfigEngine:

    def _make_tracker(self, tmp_path, n=5):
        tracker = TrainingOutcomeTracker(tmp_path / "h.jsonl")
        for i in range(n):
            tracker.record(TrainingOutcome(
                run_id=f"run_{i}",
                dataset="test",
                dataset_size=5000,
                model="gpt2",
                method="finetune",
                epochs=5,
                batch_size=8,
                learning_rate=2e-4,
                lora_rank=8,
                lora_alpha=16,
                final_loss=3.0 - i * 0.3,
                converged=True,
            ))
        return tracker

    def test_recommend_with_history(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        engine = AdaptiveConfigEngine(tracker)
        rec = engine.recommend(5000, "gpt2", "finetune")
        assert rec.based_on_runs == 5
        assert rec.confidence > 0.5
        assert rec.learning_rate > 0
        assert rec.batch_size >= 2

    def test_recommend_no_history_fallback(self, tmp_path):
        tracker = TrainingOutcomeTracker(tmp_path / "h.jsonl")
        engine = AdaptiveConfigEngine(tracker)
        rec = engine.recommend(5000, "gpt2", "finetune")
        assert rec.based_on_runs == 0
        assert rec.confidence == 0.3
        assert "heuristic" in rec.reason.lower()

    def test_recommend_small_dataset(self, tmp_path):
        tracker = TrainingOutcomeTracker(tmp_path / "h.jsonl")
        engine = AdaptiveConfigEngine(tracker)
        rec = engine.recommend(500, "gpt2", "finetune")
        assert rec.based_on_runs == 0
        assert rec.batch_size <= 8

    def test_suggest_variation(self, tmp_path):
        tracker = TrainingOutcomeTracker(tmp_path / "h.jsonl")
        engine = AdaptiveConfigEngine(tracker)
        base = AdaptiveRecommendation(learning_rate=2e-4, batch_size=8, confidence=0.4)
        varied = engine.suggest_variation(base, iteration=0)
        # Should differ from base
        assert varied.learning_rate != base.learning_rate or varied.batch_size != base.batch_size

    def test_suggest_variation_high_confidence_no_change(self, tmp_path):
        tracker = TrainingOutcomeTracker(tmp_path / "h.jsonl")
        engine = AdaptiveConfigEngine(tracker)
        base = AdaptiveRecommendation(learning_rate=2e-4, confidence=0.9)
        varied = engine.suggest_variation(base, iteration=0)
        assert varied.learning_rate == base.learning_rate

    def test_update_after_run(self, tmp_path):
        tracker = TrainingOutcomeTracker(tmp_path / "h.jsonl")
        engine = AdaptiveConfigEngine(tracker)
        outcome = TrainingOutcome(
            run_id="r1", dataset="data", model="gpt2",
            method="finetune", final_loss=2.5, converged=True,
        )
        engine.update_after_run(outcome)
        assert len(tracker.load_outcomes()) == 1

    def test_get_insights(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        engine = AdaptiveConfigEngine(tracker)
        insights = engine.get_insights(model="gpt2")
        assert insights["total_runs"] == 5
        assert "best_config" in insights
        assert "trend" in insights

    def test_get_insights_empty(self, tmp_path):
        tracker = TrainingOutcomeTracker(tmp_path / "h.jsonl")
        engine = AdaptiveConfigEngine(tracker)
        insights = engine.get_insights()
        assert "message" in insights


# ── Persistent Settings ──────────────────────────────────────────────────────


class TestAppSettings:

    def test_defaults(self):
        s = AppSettings()
        assert s.generation.temperature == 0.8
        assert s.training.preferred_model == ""
        assert s.voice.noise_gate_db == -40.0
        assert s.version == 1

    def test_to_dict(self):
        s = AppSettings()
        d = s.to_dict()
        assert "generation" in d
        assert "training" in d
        assert "voice" in d

    def test_from_dict(self):
        d = {
            "generation": {"temperature": 0.5},
            "voice": {"noise_gate_db": -35.0},
        }
        s = AppSettings.from_dict(d)
        assert s.generation.temperature == 0.5
        assert s.voice.noise_gate_db == -35.0
        # Defaults preserved
        assert s.training.preferred_model == ""


class TestPersistentSettings:

    def test_update_and_load(self, tmp_path):
        ps = PersistentSettings(tmp_path / "settings.json")
        ps.update("generation", temperature=0.5)
        s = ps.settings
        assert s.generation.temperature == 0.5

    def test_persists_across_instances(self, tmp_path):
        path = tmp_path / "settings.json"
        ps1 = PersistentSettings(path)
        ps1.update("generation", temperature=0.3)
        ps1.update("voice", noise_gate_db=-30.0)

        ps2 = PersistentSettings(path)
        s = ps2.settings
        assert s.generation.temperature == 0.3
        assert s.voice.noise_gate_db == -30.0

    def test_reset(self, tmp_path):
        ps = PersistentSettings(tmp_path / "settings.json")
        ps.update("generation", temperature=0.1)
        ps.reset()
        assert ps.settings.generation.temperature == 0.8

    def test_get(self, tmp_path):
        ps = PersistentSettings(tmp_path / "settings.json")
        ps.update("generation", temperature=0.6)
        assert ps.get("generation", "temperature") == 0.6

    def test_on_change_listener(self, tmp_path):
        ps = PersistentSettings(tmp_path / "settings.json")
        called = []
        ps.on_change(lambda s: called.append(s.generation.temperature))
        ps.update("generation", temperature=0.7)
        assert called == [0.7]

    def test_corrupt_file_uses_defaults(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text("not json!!!")
        ps = PersistentSettings(path)
        s = ps.settings
        assert s.generation.temperature == 0.8

    def test_unknown_section(self, tmp_path):
        ps = PersistentSettings(tmp_path / "settings.json")
        with pytest.raises(ValueError):
            ps.update("nonexistent", foo=1)

    def test_unknown_key_warns(self, tmp_path):
        ps = PersistentSettings(tmp_path / "settings.json")
        ps.update("generation", nonexistent_key=42)
        # Should not raise, just warn


# ── Integration: auto_configure with adaptive ────────────────────────────────


class TestAutoConfigureAdaptive:

    def test_auto_configure_uses_adaptive_when_history_exists(self, tmp_path):
        from domains.training.auto_config import auto_configure

        # Create a dataset file
        data_file = tmp_path / "data.txt"
        data_file.write_text("\n".join(["hello world"] * 100))

        # Seed history with good outcomes
        tracker = TrainingOutcomeTracker(tmp_path / "history.jsonl")
        for i in range(5):
            tracker.record(TrainingOutcome(
                run_id=f"seed_{i}",
                dataset="test",
                dataset_size=100,
                model="gpt2",
                method="finetune",
                epochs=10,
                batch_size=4,
                learning_rate=1e-4,
                lora_rank=4,
                final_loss=2.0,
                converged=True,
            ))

        with patch("domains.training.adaptive_config.AdaptiveConfigEngine") as MockEngine:
            mock_engine = MagicMock()
            mock_engine.recommend.return_value = AdaptiveRecommendation(
                learning_rate=5e-5,
                batch_size=2,
                epochs=8,
                warmup_steps=10,
                lora_rank=4,
                lora_alpha=8,
                confidence=0.9,
                based_on_runs=5,
                reason="Learned from 5 runs",
            )
            MockEngine.return_value = mock_engine

            config = auto_configure(
                dataset="test",
                dataset_path=str(data_file),
            )

            # Adaptive should have overridden these
            assert config.learning_rate == 5e-5
            assert config.batch_size == 2
            assert config.epochs == 8
            assert "Adaptive" in config.explanation
