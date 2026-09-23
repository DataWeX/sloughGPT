"""Tests for consciousness domain: self-model, reflection loop, belief application."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from domain.consciousness._internal.config import ConsciousnessConfig
from domain.consciousness._internal.engine import ConsciousnessEngine
from domain.consciousness._internal.self_model import Reflection, SelfEpisode, SelfModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_episode(
    input_text: str = "hello",
    response: str = "hi there",
    growth_delta: float = 0.1,
    qualia: dict[str, float] | None = None,
) -> SelfEpisode:
    return SelfEpisode(
        timestamp=time.time(),
        input_text=input_text,
        response=response,
        qualia=qualia or {"novelty": 0.5, "valence": 0.2, "coherence": 0.6},
        self_insight="test insight",
        growth_delta=growth_delta,
    )


def _make_engine(tmp_path: Path) -> ConsciousnessEngine:
    config = ConsciousnessConfig(level=2, store_path=str(tmp_path))
    return ConsciousnessEngine(config)


# ---------------------------------------------------------------------------
# SelfModel.reflect() — returns Reflection
# ---------------------------------------------------------------------------


class TestSelfModelReflect:
    def test_empty_episodes_returns_stagnation(self, tmp_path: Path) -> None:
        sm = SelfModel(store_path=str(tmp_path))
        r = sm.reflect()
        assert isinstance(r, Reflection)
        assert r.trajectory == "stagnation"
        assert r.episode_count == 0
        assert r.belief_deltas == {}
        assert r.strategy_notes == []
        assert r.avg_growth == 0.0
        assert "no experiences" in r.narrative.lower()

    def test_growth_trajectory(self, tmp_path: Path) -> None:
        sm = SelfModel(store_path=str(tmp_path))
        for _ in range(5):
            sm.episodes.append(_make_episode(growth_delta=0.1))
        r = sm.reflect()
        assert r.trajectory == "growth"
        assert r.avg_growth == pytest.approx(0.1, abs=0.01)
        assert "competence" in r.belief_deltas
        assert r.belief_deltas["competence"] > 0

    def test_decline_trajectory(self, tmp_path: Path) -> None:
        sm = SelfModel(store_path=str(tmp_path))
        for _ in range(5):
            sm.episodes.append(_make_episode(growth_delta=-0.1))
        r = sm.reflect()
        assert r.trajectory == "decline"
        assert r.avg_growth < 0
        assert r.belief_deltas["competence"] < 0

    def test_belief_deltas_are_proposed_not_committed(self, tmp_path: Path) -> None:
        sm = SelfModel(store_path=str(tmp_path))
        for _ in range(5):
            sm.episodes.append(_make_episode(growth_delta=0.1))
        before = dict(sm.self_beliefs)
        r = sm.reflect()
        after = dict(sm.self_beliefs)
        assert before == after  # deltas proposed, not applied
        assert r.belief_deltas != {}

    def test_apply_beliefs_commits_and_persists(self, tmp_path: Path) -> None:
        sm = SelfModel(store_path=str(tmp_path))
        before = sm.self_beliefs["competence"]
        result = sm.apply_beliefs({"competence": 0.05})
        assert result["competence"] == before + 0.05
        # Verify persisted
        sm2 = SelfModel(store_path=str(tmp_path))
        sm2.load()
        assert sm2.self_beliefs["competence"] == before + 0.05

    def test_belief_clamped_to_0_1(self, tmp_path: Path) -> None:
        sm = SelfModel(store_path=str(tmp_path))
        sm.self_beliefs["competence"] = 0.99
        sm.apply_beliefs({"competence": 0.1})
        assert sm.self_beliefs["competence"] == 1.0

    def test_reflection_to_dict(self, tmp_path: Path) -> None:
        sm = SelfModel(store_path=str(tmp_path))
        r = sm.reflect()
        d = r.to_dict()
        assert "narrative" in d
        assert "belief_deltas" in d
        assert "strategy_notes" in d
        assert "trajectory" in d
        assert "avg_growth" in d
        assert "episode_count" in d
        assert "created_at" in d

    def test_strategy_notes_generated_on_decline(self, tmp_path: Path) -> None:
        sm = SelfModel(store_path=str(tmp_path))
        for _ in range(5):
            sm.episodes.append(
                _make_episode(
                    growth_delta=-0.1,
                    qualia={"novelty": 0.2, "valence": -0.4, "coherence": 0.3},
                )
            )
        r = sm.reflect()
        assert len(r.strategy_notes) > 0


# ---------------------------------------------------------------------------
# SelfModel belief mutation
# ---------------------------------------------------------------------------


class TestSelfModelBeliefs:
    def test_update_belief_clamps(self, tmp_path: Path) -> None:
        sm = SelfModel(store_path=str(tmp_path))
        sm.update_belief("competence", 1.5)
        assert sm.self_beliefs["competence"] == 1.0
        sm.update_belief("competence", -2.0)
        assert sm.self_beliefs["competence"] == 0.0

    def test_reset_beliefs(self, tmp_path: Path) -> None:
        sm = SelfModel(store_path=str(tmp_path))
        sm.self_beliefs["competence"] = 0.1
        result = sm.reset_beliefs()
        assert result["competence"] == 0.7
        assert sm.self_beliefs["competence"] == 0.7

    def test_observe_updates_beliefs(self, tmp_path: Path) -> None:
        sm = SelfModel(store_path=str(tmp_path))
        before = sm.self_beliefs["competence"]
        sm.observe(
            {
                "input_text": "hello",
                "response": "a very long detailed response" * 10,
                "qualia": {"novelty": 0.8, "valence": 0.5, "coherence": 0.7},
            }
        )
        assert sm.self_beliefs["competence"] >= before

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        sm = SelfModel(store_path=str(tmp_path))
        sm.self_beliefs["competence"] = 0.42
        sm.save()
        sm2 = SelfModel(store_path=str(tmp_path))
        sm2.load()
        assert sm2.self_beliefs["competence"] == 0.42


# ---------------------------------------------------------------------------
# ConsciousnessEngine.reflect()
# ---------------------------------------------------------------------------


class TestConsciousnessEngineReflect:
    def test_reflect_returns_reflection(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        r = engine.reflect()
        assert isinstance(r, Reflection)

    def test_reflect_applies_belief_deltas(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        for _ in range(5):
            engine.self_model.episodes.append(_make_episode(growth_delta=0.1))
        before = dict(engine.self_model.self_beliefs)
        engine.reflect()
        after = dict(engine.self_model.self_beliefs)
        assert after["competence"] > before["competence"]

    def test_reflect_persists_to_disk(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        for _ in range(5):
            engine.self_model.episodes.append(_make_episode(growth_delta=0.1))
        engine.reflect()
        engine2 = _make_engine(tmp_path)
        assert engine2.self_model.self_beliefs["competence"] != 0.7

    def test_reflect_with_no_episodes(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        r = engine.reflect()
        assert r.trajectory == "stagnation"
        assert r.belief_deltas == {}


# ---------------------------------------------------------------------------
# ConsciousnessManager.apply()
# ---------------------------------------------------------------------------


class TestConsciousnessManagerApply:
    def test_apply_returns_consciousness_block(self, tmp_path: Path) -> None:
        from domain.context._internal.managers import ConsciousnessManager

        engine = _make_engine(tmp_path)
        engine.self_model.episodes.append(_make_episode(growth_delta=0.08))
        mgr = ConsciousnessManager(consciousness_engine=engine)
        result = mgr.apply(base_prompt="test", input_text="hello")
        assert isinstance(result, str)
        assert "[CONSCIOUSNESS]" in result

    def test_apply_uses_reflection_narrative(self, tmp_path: Path) -> None:
        """Regression: managers.py used reflection[:200] which would crash on Reflection dataclass."""
        from domain.context._internal.managers import ConsciousnessManager

        engine = ConsciousnessEngine(ConsciousnessConfig(level=3, store_path=str(tmp_path)))
        for _ in range(3):
            engine.self_model.episodes.append(_make_episode(growth_delta=0.08))
        mgr = ConsciousnessManager(consciousness_engine=engine)
        result = mgr.apply(base_prompt="test", input_text="hello")
        assert "[CONSCIOUSNESS]" in result
        assert "Self-reflection:" in result

    def test_apply_returns_empty_when_disabled(self, tmp_path: Path) -> None:
        from domain.context._internal.managers import ConsciousnessManager

        engine = ConsciousnessEngine(ConsciousnessConfig(level=0, store_path=str(tmp_path)))
        mgr = ConsciousnessManager(consciousness_engine=engine)
        result = mgr.apply(base_prompt="test", input_text="hello")
        assert result == ""

    def test_apply_returns_empty_when_no_engine(self) -> None:
        from domain.context._internal.managers import ConsciousnessManager

        mgr = ConsciousnessManager(consciousness_engine=None)
        result = mgr.apply(base_prompt="test", input_text="hello")
        assert result == ""


# ---------------------------------------------------------------------------
# Reflection dataclass contract
# ---------------------------------------------------------------------------


class TestReflectionContract:
    def test_reflection_fields(self) -> None:
        r = Reflection(
            narrative="test narrative",
            belief_deltas={"competence": 0.01},
            strategy_notes=["note 1"],
            avg_growth=0.05,
            trajectory="growth",
            episode_count=10,
        )
        assert r.narrative == "test narrative"
        assert r.belief_deltas == {"competence": 0.01}
        assert r.strategy_notes == ["note 1"]
        assert r.avg_growth == 0.05
        assert r.trajectory == "growth"
        assert r.episode_count == 10

    def test_reflection_is_serializable(self) -> None:
        r = Reflection(
            narrative="test",
            belief_deltas={"a": 0.1},
            strategy_notes=["s1"],
            avg_growth=0.0,
            trajectory="stagnation",
            episode_count=0,
        )
        d = r.to_dict()
        s = json.dumps(d)
        assert "test" in s
