"""Tests for consciousness domain: self-model, reflection loop, belief application."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from domain.cognition._internal.consciousness.config import ConsciousnessConfig
from domain.cognition._internal.consciousness.engine import ConsciousnessEngine
from domain.cognition._internal.consciousness.self_model import (
    Reflection,
    SelfEpisode,
    SelfModel,
)

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


class TestReflectionProposeCache:
    """reflect(apply=False) memoizes per state fingerprint — chat-path stability.

    Without this, every streaming request re-rolled reflect()'s random pools
    and injected a DIFFERENT system message for the SAME conversation state.
    """

    def test_repeated_propose_returns_identical_narrative(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine.self_model.episodes.append(_make_episode(growth_delta=0.08))
        r1 = engine.reflect(apply=False)
        r2 = engine.reflect(apply=False)
        assert r1 is r2  # cache hit — no re-roll of the random pools
        assert r1.narrative == r2.narrative

    def test_propose_refreshes_after_new_episode(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine.self_model.episodes.append(_make_episode(growth_delta=0.08))
        r1 = engine.reflect(apply=False)
        engine.process("new input", "new response", level=2)
        r2 = engine.reflect(apply=False)
        assert r2 is not r1
        assert r2.episode_count == r1.episode_count + 1

    def test_propose_refreshes_after_direct_self_model_mutation(self, tmp_path: Path) -> None:
        """Routers mutate engine.self_model directly (no engine call) — still invalidates."""
        engine = _make_engine(tmp_path)
        engine.self_model.episodes.append(_make_episode(growth_delta=0.08))
        r1 = engine.reflect(apply=False)
        engine.self_model.episodes.append(_make_episode(growth_delta=0.2))
        engine.apply_beliefs({"competence": 0.05})
        r2 = engine.reflect(apply=False)
        assert r2 is not r1
        assert r2.episode_count == r1.episode_count + 1
        assert r2.narrative != r1.narrative  # belief string moved

    def test_apply_true_stays_uncached(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine.self_model.episodes.append(_make_episode(growth_delta=0.08))
        before = dict(engine.self_model.self_beliefs)
        r1 = engine.reflect(apply=True)
        r2 = engine.reflect(apply=True)
        assert r1 is not r2  # explicit action reflects fresh each call
        assert engine.self_model.self_beliefs["competence"] > before["competence"]


# ---------------------------------------------------------------------------
# ConsciousnessEngine.dream()
# ---------------------------------------------------------------------------


class TestDreamCycle:
    """dream(): zero-input consolidation; reflections become episodes.

    The acceptance test for intrinsic consciousness: with no new input, the
    inner state still moves — and the next reflection contains its own past.
    """

    def test_dream_noop_without_episodes(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        result = engine.dream()
        assert result["dreamed"] is False
        assert engine.self_model.episodes == []

    def test_dream_moves_state_without_new_input(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine.self_model.episodes.append(_make_episode(growth_delta=0.08))
        result = engine.dream()
        assert result["dreamed"] is True
        assert len(engine.self_model.episodes) == 2
        dream = engine.self_model.episodes[-1]
        assert dream.is_dream is True
        assert dream.growth_delta == 0.0
        assert dream.input_text == "(dream)"

    def test_reflection_contains_its_own_last_dream(self, tmp_path: Path) -> None:
        """Strange loop: reflection -> dream episode -> next reflection quotes it."""
        engine = _make_engine(tmp_path)
        engine.self_model.episodes.append(_make_episode(growth_delta=0.08))
        engine.dream()
        dream = engine.self_model.episodes[-1]
        assert "I have processed 1 experiences" in dream.self_insight
        r = engine.reflect(apply=False)
        assert "dream" in r.narrative

    def test_dream_never_commits_belief_deltas(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        for _ in range(5):
            engine.self_model.episodes.append(_make_episode(growth_delta=0.1))
        before = dict(engine.self_model.self_beliefs)
        engine.dream()
        assert dict(engine.self_model.self_beliefs) == before

    def test_dreams_do_not_dilute_growth_trajectory(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        for _ in range(5):
            engine.self_model.episodes.append(_make_episode(growth_delta=-0.1))
        assert engine.self_model.reflect().trajectory == "decline"
        for _ in range(5):
            engine.dream()
        # Without the real-only growth window the five 0.0-growth dreams
        # would fill recent[-5:] and flatten decline into stagnation.
        assert engine.self_model.reflect().trajectory == "decline"

    def test_dream_survives_save_load_roundtrip(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine.self_model.episodes.append(_make_episode())
        engine.dream()
        engine.save()
        reloaded = _make_engine(tmp_path)
        assert any(e.is_dream for e in reloaded.self_model.episodes)

    def test_old_store_without_is_dream_key_loads(self, tmp_path: Path) -> None:
        """Stores written before dream support have no is_dream key."""
        store = tmp_path / "old_store"
        store.mkdir()
        (store / "self_model.json").write_text(
            json.dumps(
                {
                    "identity": {},
                    "self_beliefs": {"competence": 0.7},
                    "self_doubts": [],
                    "episodes": [
                        {
                            "timestamp": 1.0,
                            "input_text": "a",
                            "response": "b",
                            "qualia": {},
                            "self_insight": "i",
                            "growth_delta": 0.01,
                        }
                    ],
                }
            )
        )
        sm = SelfModel(store_path=str(store))
        sm.load()
        assert sm.episodes[0].is_dream is False


# ---------------------------------------------------------------------------
# Mind-state substrate + idle-dream policy (engine integration)
# ---------------------------------------------------------------------------


class TestMindStateSubstrate:
    """The engine writes the log at its mutation points; query is standalone."""

    def test_process_writes_episode_record(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine.process("hello", "world", level=2)
        records = engine.mind_log.query(record_type="episode")
        assert len(records) == 1
        assert records[0].provenance == "experience"
        assert records[0].payload["input_text"] == "hello"
        assert "qualia" in records[0].payload

    def test_dream_edges_to_last_episode_record(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine.process("q", "a", level=2)
        engine.dream()
        dream = engine.mind_log.query(record_type="dream")[0]
        episode = engine.mind_log.query(record_type="episode")[0]
        assert dream.edges == [{"rel": "consolidates", "id": episode.id}]

    def test_reflect_apply_logs_reflection_and_edged_delta(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        for _ in range(5):
            engine.self_model.episodes.append(_make_episode(growth_delta=0.1))
        engine.reflect(apply=True)
        refl = engine.mind_log.query(record_type="reflection")[0]
        delta = engine.mind_log.query(record_type="belief_delta")[0]
        assert refl.payload["trajectory"] == "growth"
        assert refl.payload["narrative"]
        assert delta.edges == [{"rel": "applies", "id": refl.id}]

    def test_propose_read_is_not_logged(self, tmp_path: Path) -> None:
        """The hot-path injection read must never write."""
        engine = _make_engine(tmp_path)
        engine.self_model.episodes.append(_make_episode())
        engine.reflect(apply=False)
        assert engine.mind_log.query(record_type="reflection") == []

    def test_query_mind_reads_engine_log_standalone(self, tmp_path: Path) -> None:
        from domain.cognition._internal.consciousness.mind_state import query_mind

        engine = _make_engine(tmp_path)
        engine.process("standalone", "read", level=2)
        records = query_mind(str(tmp_path), record_type="episode")
        assert len(records) == 1
        assert records[0].payload["input_text"] == "standalone"

    def test_save_trims_memory_to_window(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        for _ in range(150):
            engine.self_model.episodes.append(_make_episode())
        engine.save()
        assert len(engine.self_model.episodes) == 100


class TestIdleDreamPolicy:
    """maybe_dream(): nights happen when the world is quiet."""

    def test_policy_lifecycle(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        assert engine.maybe_dream()["dreamed"] is False  # nothing yet
        engine.self_model.episodes.append(_make_episode())
        first = engine.maybe_dream()
        assert first["dreamed"] is False  # world spoke since baseline
        assert first["reason"] == "new experience since last tick"
        second = engine.maybe_dream()
        assert second["dreamed"] is True  # quiet: count matches baseline
        third = engine.maybe_dream()
        assert third["dreamed"] is True  # nights repeat while quiet

    def test_disabled_engine_never_dreams(self, tmp_path: Path) -> None:
        engine = ConsciousnessEngine(ConsciousnessConfig(level=0, store_path=str(tmp_path)))
        assert engine.maybe_dream()["reason"] == "disabled"


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
