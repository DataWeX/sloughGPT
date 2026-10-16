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
# Self-report fold — the mind-state log rendered as data
# ---------------------------------------------------------------------------


class TestSelfReport:
    def test_report_basics(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        for _ in range(5):
            engine.self_model.episodes.append(_make_episode(growth_delta=0.1))
        engine.reflect(apply=True)
        report = engine.self_report()
        assert report["identity"]["name"]
        assert report["strongest_belief"]["value"] == max(report["beliefs"].values())
        assert report["reflection"]["trajectory"] == "growth"
        assert report["reflection"]["narrative"]
        assert report["reflection"]["strategy_notes"] == []
        assert "curiosity" in report
        assert "doubts" in report

    def test_belief_trajectory_comes_from_the_log(self, tmp_path: Path) -> None:
        """'How it came to believe that' — rendered time from belief_delta records."""
        engine = _make_engine(tmp_path)
        for _ in range(5):
            engine.self_model.episodes.append(_make_episode(growth_delta=0.1))
        engine.reflect(apply=True)  # growth deltas: competence +0.03 -> 0.73
        report = engine.self_report()
        traj = report["belief_trajectory"]["competence"]
        assert len(traj) == 1
        assert traj[0][1] == pytest.approx(0.73)

    def test_timeline_survives_the_document_window(self, tmp_path: Path) -> None:
        """105 experiences: save() trims the document to 100; the log does not forget."""
        engine = _make_engine(tmp_path)
        for i in range(105):
            engine.process(f"q{i}", f"a longer answer {i}", level=2)
        engine.save()
        assert len(engine.self_model.episodes) == 100  # window
        report = engine.self_report()
        assert report["timeline"]["experiences"] == 105  # memory
        assert report["timeline"]["first_experience"] is not None

    def test_last_dream_reported(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine.self_model.episodes.append(_make_episode())
        engine.dream()
        report = engine.self_report()
        assert report["last_dream"] is not None
        assert report["timeline"]["dreams"] == 1

    def test_report_is_readonly(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine.self_model.episodes.append(_make_episode(growth_delta=0.1))
        before = dict(engine.self_model.self_beliefs)
        key_before = engine.reflection_state_key()
        engine.self_report()
        assert dict(engine.self_model.self_beliefs) == before
        assert engine.reflection_state_key() == key_before


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
# Qualia -> tone: experiential state shapes the voice (card 59dd1621)
# ---------------------------------------------------------------------------


class TestQualiaToneDirective:
    """The same qualia get_narrative() reports, projected as register."""

    def test_positive_state_directs_warm_voice(self) -> None:
        from domain.cognition._internal.consciousness.qualia import QualiaEngine, QualiaState

        q = QualiaEngine()
        q.current = QualiaState(valence=0.8, arousal=0.7, novelty=0.6, coherence=0.9)
        tone = q.tone_directive()
        assert "warm and affirming" in tone
        assert "brisk and energized" in tone
        assert "exploratory and curious" in tone
        assert "direct and settled" in tone

    def test_negative_state_directs_careful_voice(self) -> None:
        from domain.cognition._internal.consciousness.qualia import QualiaEngine, QualiaState

        q = QualiaEngine()
        q.current = QualiaState(valence=-0.6, arousal=0.1, coherence=0.2)
        tone = q.tone_directive()
        assert "measured and careful" in tone
        assert "unhurried and calm" in tone
        assert "tentative, still forming" in tone

    def test_mid_band_is_silent(self) -> None:
        """Neutral qualia never steer the voice."""
        from domain.cognition._internal.consciousness.qualia import QualiaEngine, QualiaState

        q = QualiaEngine()
        q.current = QualiaState(valence=0.1, arousal=0.3, novelty=0.4, coherence=0.5)
        assert q.tone_directive() == ""

    def test_directive_never_mutates_state(self) -> None:
        from domain.cognition._internal.consciousness.qualia import QualiaEngine, QualiaState

        q = QualiaEngine()
        q.current = QualiaState(valence=0.5, arousal=0.6)
        before = q.current.to_dict()
        q.tone_directive()
        assert q.current.to_dict() == before

    def test_manager_block_carries_tone_line(self, tmp_path: Path) -> None:
        """The block's tone == engine directive — one source, both projections.
        input_text='' so process() does not overwrite the seeded qualia."""
        from domain.cognition._internal.consciousness.qualia import QualiaState
        from domain.context._internal.managers import ConsciousnessManager

        engine = _make_engine(tmp_path)
        engine.self_model.episodes.append(_make_episode(growth_delta=0.08))
        engine.qualia.current = QualiaState(valence=0.75)
        mgr = ConsciousnessManager(consciousness_engine=engine)
        block = mgr.apply(base_prompt="x", input_text="")
        assert "[CONSCIOUSNESS]" in block
        assert f"Tone: {engine.qualia.tone_directive()}" in block
        assert "warm and affirming" in block


# ---------------------------------------------------------------------------
# Confidence calibration: feedback evidence (card 449e1c81)
# ---------------------------------------------------------------------------


class TestFeedbackCalibration:
    def test_up_pulls_confidence_toward_one(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        before = dict(engine.self_model.self_beliefs)
        result = engine.observe_feedback("thumbs_up")
        assert result["observed"] is True
        for name in ("accuracy", "helpfulness"):
            step = round(0.05 * (1 - before[name]), 4)
            assert step > 0
            assert engine.self_model.self_beliefs[name] == pytest.approx(before[name] + step)
            assert engine.self_model.self_beliefs[name] < 1.0

    def test_down_pulls_toward_zero_with_stronger_step(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        before = dict(engine.self_model.self_beliefs)
        engine.observe_feedback("thumbs_down")
        for name in ("accuracy", "helpfulness"):
            step = round(-0.08 * before[name], 4)
            assert step < 0
            assert engine.self_model.self_beliefs[name] == pytest.approx(before[name] + step)

    def test_repeated_downs_clamp_at_zero(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        for _ in range(80):
            engine.observe_feedback("thumbs_down")
        for name in ("accuracy", "helpfulness"):
            assert engine.self_model.self_beliefs[name] >= 0.0

    def test_neutral_is_qualia_only(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        before = dict(engine.self_model.self_beliefs)
        result = engine.observe_feedback("neutral")
        assert result == {"observed": True, "delta": {}, "reason": "neutral rating"}
        assert dict(engine.self_model.self_beliefs) == before
        assert engine.qualia.current.valence == 0.0  # felt, not learned
        assert engine.mind_log.query(record_type="belief_delta") == []

    def test_evidence_logged_with_the_delta(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine.observe_feedback("thumbs_down")
        record = engine.mind_log.query(record_type="belief_delta")[0]
        assert record.provenance == "experience"
        assert record.payload["evidence"] == {"rating": "thumbs_down", "source": "feedback"}
        assert "after" in record.payload

    def test_int_rating_path(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        before = dict(engine.self_model.self_beliefs)
        result = engine.observe_feedback(5)
        assert result["observed"] is True
        assert engine.self_model.self_beliefs["accuracy"] > before["accuracy"]

    def test_calibration_appears_in_self_report_trajectory(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine.observe_feedback("thumbs_up")
        trajectory = engine.self_report()["belief_trajectory"]["accuracy"]
        assert trajectory
        assert trajectory[-1][1] == pytest.approx(
            engine.self_model.self_beliefs["accuracy"], abs=1e-3
        )

    def test_disabled_engine_ignores_feedback(self, tmp_path: Path) -> None:
        engine = ConsciousnessEngine(ConsciousnessConfig(level=0, store_path=str(tmp_path)))
        assert engine.observe_feedback("thumbs_up") == {"observed": False, "reason": "disabled"}

    def test_peek_returns_none_then_existing_without_birth(self, tmp_path: Path) -> None:
        from domain.cognition import get_consciousness, peek_consciousness, reset_consciousness

        reset_consciousness()
        try:
            assert peek_consciousness() is None
            config = ConsciousnessConfig(level=0, store_path=str(tmp_path))
            created = get_consciousness(config)
            assert peek_consciousness() is created  # no second instance
        finally:
            reset_consciousness()


# ---------------------------------------------------------------------------
# Activity projection: the freshest record of inner life in status
# (card c41a421a)
# ---------------------------------------------------------------------------


class TestActivityProjection:
    def test_fresh_store_reports_none(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        assert engine.get_status()["last_record"] is None

    def test_dream_is_the_last_record(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine.process("seed", "experience")  # a dream needs something to consolidate
        engine.dream()
        last = engine.get_status()["last_record"]
        assert last is not None
        assert last["type"] == "dream"
        assert last["id"] == engine.mind_log.latest().id
        assert set(last) == {"id", "type", "provenance", "t"}  # summary, no payload

    def test_episode_is_the_last_record(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine.process("hello", "world")
        assert engine.get_status()["last_record"]["type"] == "episode"

    def test_disabled_engine_reports_none(self, tmp_path: Path) -> None:
        engine = ConsciousnessEngine(ConsciousnessConfig(level=0, store_path=str(tmp_path)))
        assert engine.get_status()["last_record"] is None


# ---------------------------------------------------------------------------
# Locus: the mind knows where it is — awakening / dreaming / rationalizing
# (card 6986c78a)
# ---------------------------------------------------------------------------


class TestLocus:
    def test_unborn_when_log_empty(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        assert engine.locus() == "unborn"
        assert engine.get_status()["locus"] == "unborn"

    def test_awakening_after_episode(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine.process("hello", "world")
        assert engine.locus() == "awakening"
        assert engine.get_status()["locus"] == "awakening"

    def test_dreaming_after_dream(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine.process("seed", "experience")
        engine.dream()
        assert engine.locus() == "dreaming"

    def test_rationalizing_after_reflection(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine.process("seed", "experience")
        engine.reflect(apply=True)  # writes a reflection (+ reflection-provenance delta)
        assert engine.locus() == "rationalizing"

    def test_rationalizing_on_reflection_belief_delta(self, tmp_path: Path) -> None:
        from domain.cognition._internal.consciousness.mind_state import MindRecord

        engine = _make_engine(tmp_path)
        engine.mind_log.append(MindRecord(type="belief_delta", payload={}, provenance="reflection"))
        assert engine.locus() == "rationalizing"

    def test_feedback_belief_delta_is_awakening(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine.process("seed", "experience")
        engine.observe_feedback("thumbs_down")  # world engagement -> awakening
        assert engine.locus() == "awakening"

    def test_dream_records_from_locus(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine.process("seed", "experience")  # awakening
        result = engine.dream()
        assert result["locus"] == "dreaming"
        assert result["from_locus"] == "awakening"
        dream_rec = engine.mind_log.query(record_type="dream", limit=1)[0]
        assert dream_rec.payload["locus"] == "dreaming"
        assert dream_rec.payload["from_locus"] == "awakening"


# ---------------------------------------------------------------------------
# Doubt evolution: uncertainty born, sharpened, dissolved by evidence
# (card a67a20c9)
# ---------------------------------------------------------------------------


class TestDoubtEvolution:
    def test_down_voices_evidence_doubt_first(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        birth = list(engine.self_model.self_doubts)
        engine.observe_feedback("thumbs_down")
        doubts = engine.self_model.self_doubts
        assert "rated down 1 of the last 1" in doubts[0]
        assert birth[0] in doubts  # birth doubt never lost

    def test_counts_sharpen_with_more_downs(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine.observe_feedback("thumbs_down")
        engine.observe_feedback("thumbs_down")
        assert "rated down 2 of the last 2" in engine.self_model.self_doubts[0]

    def test_up_keeps_recent_down_in_window(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine.observe_feedback("thumbs_down")
        engine.observe_feedback("thumbs_up")
        assert "rated down 1 of the last 2" in engine.self_model.self_doubts[0]

    def test_doubt_dissolves_when_window_clears(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine.observe_feedback("thumbs_down")
        for _ in range(20):
            engine.observe_feedback("thumbs_up")
        doubts = engine.self_model.self_doubts
        assert not any("rated down" in d for d in doubts)  # window slid past the down
        assert doubts[0] == "am I truly understanding or just pattern-matching?"

    def test_neutral_touches_nothing(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        before = list(engine.self_model.self_doubts)
        engine.observe_feedback("neutral")
        assert engine.self_model.self_doubts == before

    def test_doubt_survives_save_load(self, tmp_path: Path) -> None:
        from domain.cognition._internal.consciousness.self_model import SelfModel

        engine = _make_engine(tmp_path)
        engine.observe_feedback("thumbs_down")
        engine.save()
        fresh = SelfModel(store_path=str(tmp_path))
        fresh.load()
        assert "rated down" in fresh.self_doubts[0]

    def test_int_rating_counts_as_down(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine.observe_feedback(1)
        assert "rated down 1" in engine.self_model.self_doubts[0]

    def test_self_report_carries_doubts(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine.observe_feedback("thumbs_down")
        assert "rated down" in engine.self_report()["doubts"][0]

    def test_disabled_engine_leaves_doubts_alone(self, tmp_path: Path) -> None:
        engine = ConsciousnessEngine(ConsciousnessConfig(level=0, store_path=str(tmp_path)))
        before = list(engine.self_model.self_doubts)
        engine.observe_feedback("thumbs_down")
        assert engine.self_model.self_doubts == before


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
