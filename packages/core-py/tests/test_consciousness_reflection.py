"""Tests for the structured reflection loop in the consciousness domain."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from domain.cognition._internal.consciousness.self_model import Reflection, SelfModel


@pytest.fixture
def tmp_store(tmp_path: Path) -> str:
    """Provide a temporary store path for SelfModel."""
    return str(tmp_path / "consciousness")


@pytest.fixture
def model(tmp_store: str) -> SelfModel:
    """Fresh SelfModel with empty store."""
    return SelfModel(store_path=tmp_store)


class TestReflectionDataclass:
    """Tests for the Reflection dataclass."""

    def test_reflection_fields(self):
        """Reflection has all required fields."""
        r = Reflection(
            narrative="test narrative",
            belief_deltas={"competence": 0.01},
            strategy_notes=["note A"],
            avg_growth=0.05,
            trajectory="growth",
            episode_count=10,
        )
        assert r.narrative == "test narrative"
        assert r.belief_deltas == {"competence": 0.01}
        assert r.strategy_notes == ["note A"]
        assert r.avg_growth == 0.05
        assert r.trajectory == "growth"
        assert r.episode_count == 10
        assert r.created_at > 0

    def test_to_dict_roundtrip(self):
        """to_dict produces JSON-serializable output with all keys."""
        r = Reflection(
            narrative="n",
            belief_deltas={"a": 0.1},
            strategy_notes=["s"],
            avg_growth=0.0,
            trajectory="stagnation",
            episode_count=1,
        )
        d = r.to_dict()
        assert set(d.keys()) == {
            "narrative",
            "belief_deltas",
            "strategy_notes",
            "avg_growth",
            "trajectory",
            "episode_count",
            "created_at",
        }
        # Verify JSON-serializable
        json.dumps(d)


class TestReflectEmpty:
    """Tests for reflect() with no episodes."""

    def test_empty_returns_stagnation(self, model: SelfModel):
        """No episodes -> stagnation trajectory, empty deltas."""
        r = model.reflect()
        assert isinstance(r, Reflection)
        assert r.trajectory == "stagnation"
        assert r.belief_deltas == {}
        assert r.episode_count == 0
        assert "no experiences" in r.narrative

    def test_empty_has_no_strategy_notes(self, model: SelfModel):
        """No episodes -> no strategy notes."""
        r = model.reflect()
        assert r.strategy_notes == []


class TestReflectWithEpisodes:
    """Tests for reflect() after observing episodes."""

    def _add_episodes(self, model: SelfModel, count: int, **overrides) -> None:
        for i in range(count):
            exp = {
                "input_text": overrides.get("input_text", f"question {i}?"),
                "response": overrides.get("response", f"answer {i}" * (i + 1)),
                "qualia": overrides.get(
                    "qualia", {"novelty": 0.5, "valence": 0.0, "coherence": 0.5}
                ),
            }
            if "feedback_rating" in overrides:
                exp["feedback_rating"] = overrides["feedback_rating"]
            model.observe(exp)

    def test_reflect_returns_reflection(self, model: SelfModel):
        """reflect() returns Reflection after episodes."""
        self._add_episodes(model, 3)
        r = model.reflect()
        assert isinstance(r, Reflection)
        assert r.episode_count == 3

    def test_trajectory_growth(self, model: SelfModel):
        """Positive feedback -> growth trajectory."""
        self._add_episodes(model, 6, feedback_rating=5)
        r = model.reflect()
        assert r.trajectory == "growth"
        assert r.avg_growth > 0

    def test_trajectory_decline(self, model: SelfModel):
        """Negative feedback -> decline trajectory."""
        self._add_episodes(model, 6, feedback_rating=1, response="x")
        r = model.reflect()
        assert r.trajectory == "decline"
        assert r.avg_growth < 0

    def test_trajectory_stagnation(self, model: SelfModel):
        """Neutral feedback -> stagnation trajectory."""
        self._add_episodes(model, 6, feedback_rating=3)
        r = model.reflect()
        assert r.trajectory == "stagnation"

    def test_belief_deltas_populated(self, model: SelfModel):
        """Growth episodes produce belief deltas."""
        self._add_episodes(model, 6, feedback_rating=5)
        r = model.reflect()
        assert len(r.belief_deltas) > 0
        assert "competence" in r.belief_deltas

    def test_belief_deltas_empty_on_stagnation(self, model: SelfModel):
        """Stagnation may produce few or no deltas."""
        self._add_episodes(model, 3, feedback_rating=3)
        r = model.reflect()
        # Stagnation can still produce novelty/accuracy deltas but no growth deltas
        assert "competence" not in r.belief_deltas
        assert "helpfulness" not in r.belief_deltas

    def test_narrative_contains_beliefs(self, model: SelfModel):
        """Narrative includes current belief values."""
        self._add_episodes(model, 3)
        r = model.reflect()
        assert "My current beliefs:" in r.narrative

    def test_narrative_contains_growth_rate(self, model: SelfModel):
        """Narrative includes average growth rate."""
        self._add_episodes(model, 3)
        r = model.reflect()
        assert "Average growth rate:" in r.narrative

    def test_strategy_notes_on_decline(self, model: SelfModel):
        """Decline trajectory produces strategy notes."""
        self._add_episodes(model, 6, feedback_rating=1, response="x")
        r = model.reflect()
        assert r.trajectory == "decline"
        assert len(r.strategy_notes) > 0

    def test_to_dict(self, model: SelfModel):
        """to_dict returns serializable dict."""
        self._add_episodes(model, 3)
        r = model.reflect()
        d = r.to_dict()
        assert isinstance(d, dict)
        json.dumps(d)  # should not raise


class TestApplyBeliefs:
    """Tests for apply_beliefs() persistence."""

    def test_apply_updates_values(self, model: SelfModel):
        """apply_beliefs updates belief values."""
        old = model.get_belief("competence")
        model.apply_beliefs({"competence": 0.05})
        assert model.get_belief("competence") == pytest.approx(old + 0.05, abs=1e-6)

    def test_apply_clamps_to_01(self, model: SelfModel):
        """apply_beliefs clamps values to [0, 1]."""
        model.apply_beliefs({"competence": 1.0})
        assert model.get_belief("competence") == 1.0
        model.apply_beliefs({"competence": 1.0})
        assert model.get_belief("competence") == 1.0

        model.apply_beliefs({"empathy": -1.0})
        assert model.get_belief("empathy") == 0.0

    def test_apply_persists(self, model: SelfModel):
        """apply_beliefs persists to disk."""
        model.apply_beliefs({"competence": 0.05})
        # Reload and verify
        model2 = SelfModel(store_path=model._store_path)
        model2.load()
        assert model2.get_belief("competence") == pytest.approx(
            model.get_belief("competence"), abs=1e-6
        )

    def test_apply_returns_updated_beliefs(self, model: SelfModel):
        """apply_beliefs returns the updated beliefs dict."""
        result = model.apply_beliefs({"creativity": 0.1})
        assert isinstance(result, dict)
        assert "creativity" in result


class TestReflectionPersistence:
    """Tests for reflection round-trip through save/load."""

    def test_reflection_reflect_after_reload(self, tmp_store: str):
        """Model with episodes persists and reflects after reload."""
        m1 = SelfModel(store_path=tmp_store)
        for i in range(5):
            m1.observe({"input_text": f"q{i}?", "response": f"a{i}" * 10})
        m1.save()

        m2 = SelfModel(store_path=tmp_store)
        m2.load()
        assert len(m2.episodes) == 5
        r = m2.reflect()
        assert isinstance(r, Reflection)
        assert r.episode_count == 5
