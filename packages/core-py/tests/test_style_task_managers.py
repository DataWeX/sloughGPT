"""Tests for domains.context.managers — StyleManager, TaskManager, TraitWeightsConfig."""

import pytest
from domains.context.managers import (
    StyleManager, TaskManager, TraitWeightsConfig,
    PersonalityManager, MemoryManager,
    TRAIT_SCHEMA, ALL_TRAITS,
    get_trait_config, reset_trait_config,
    _describe_trait, _if_above,
)


def _make_cfg(overrides, tmp_path=None):
    path = str(tmp_path / "traits.json") if tmp_path else None
    cfg = TraitWeightsConfig(path=path) if path else TraitWeightsConfig()
    for k, v in overrides.items():
        cfg.set(k, v)
    return cfg


# ── TraitWeightsConfig ──────────────────────────────────────────────

class TestTraitWeightsConfig:
    def test_get_default(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        assert cfg.get("warmth") == 0.5

    def test_set_and_get(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set("warmth", 0.8)
        assert cfg.get("warmth") == pytest.approx(0.8, abs=1e-4)

    def test_set_clamps_above(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set("warmth", 1.5)
        assert cfg.get("warmth") == 1.0

    def test_set_clamps_below(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set("warmth", -0.5)
        assert cfg.get("warmth") == 0.0

    def test_update_deltas(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set("warmth", 0.5)
        cfg.update({"warmth": 0.1})
        assert cfg.get("warmth") == pytest.approx(0.6, abs=1e-4)

    def test_update_clamps(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set("warmth", 0.95)
        cfg.update({"warmth": 0.2})
        assert cfg.get("warmth") == 1.0

    def test_set_many(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set_many({"warmth": 0.3, "creativity": 0.7})
        assert cfg.get("warmth") == pytest.approx(0.3, abs=1e-4)
        assert cfg.get("creativity") == pytest.approx(0.7, abs=1e-4)

    def test_set_many_ignores_non_traits(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set_many({"not_a_trait": 0.9})
        assert cfg.get("not_a_trait") == 0.5

    def test_reset(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set("warmth", 0.9)
        cfg.reset()
        assert cfg.get("warmth") == 0.5

    def test_all_groups(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        all_w = cfg.all()
        assert "personality" in all_w
        assert "cognition" in all_w
        assert "emotion" in all_w
        for group_traits in all_w.values():
            for v in group_traits.values():
                assert 0.0 <= v <= 1.0

    def test_persistence_roundtrip(self, tmp_path):
        path = tmp_path / "traits.json"
        cfg = TraitWeightsConfig(path=str(path))
        cfg.set("warmth", 0.73)
        cfg2 = TraitWeightsConfig(path=str(path))
        assert cfg2.get("warmth") == pytest.approx(0.73, abs=0.01)

    def test_snapshot_save_and_load(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set("warmth", 0.88)
        cfg.save_snapshot("test_snap")
        cfg.set("warmth", 0.1)
        loaded = cfg.load_snapshot("test_snap")
        assert loaded > 0
        assert cfg.get("warmth") == pytest.approx(0.88, abs=0.01)

    def test_snapshot_list(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set("warmth", 0.5)
        cfg.save_snapshot("snap_a")
        snaps = cfg.list_snapshots()
        names = [s["name"] for s in snaps]
        assert "snap_a" in names

    def test_snapshot_delete(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.save_snapshot("to_delete")
        result = cfg.delete_snapshot("to_delete")
        assert result is True

    def test_snapshot_delete_nonexistent(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        result = cfg.delete_snapshot("nonexistent")
        assert result is False

    def test_load_snapshot_nonexistent(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        count = cfg.load_snapshot("nonexistent")
        assert count == 0

    def test_update_from_feedback_thumbs_up(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        count = cfg.update_from_feedback("thumbs_up")
        assert count > 0
        assert cfg.get("warmth") >= 0.5

    def test_update_from_feedback_thumbs_down(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        count = cfg.update_from_feedback("thumbs_down")
        assert count > 0

    def test_update_from_feedback_with_user_message(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.update_from_feedback("thumbs_up", user_message="make it funny and creative")
        assert cfg.get("humor") > 0.5

    def test_update_from_feedback_with_negation(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.update_from_feedback("thumbs_down", user_message="not formal please")
        assert cfg.get("formality") <= 0.5

    def test_update_from_feedback_short_response(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.update_from_feedback("thumbs_up", response="ok")
        assert cfg.get("directness") > 0.5

    def test_update_from_feedback_long_response(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        long_resp = " ".join(["word"] * 100)
        cfg.update_from_feedback("thumbs_up", response=long_resp)
        assert cfg.get("patience") > 0.5

    def test_update_from_feedback_code_response(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.update_from_feedback("thumbs_up", response="```python\nprint(1)\n```")
        assert cfg.get("factual_precision") > 0.5

    def test_trait_schema_completeness(self):
        schema_traits = set()
        for traits in TRAIT_SCHEMA.values():
            schema_traits.update(traits)
        assert set(ALL_TRAITS) == schema_traits


# ── Helper functions ────────────────────────────────────────────────

class TestHelpers:
    def test_describe_trait_high(self):
        assert "warm" in _describe_trait(0.9, "warm", "cold")

    def test_describe_trait_low(self):
        assert "cold" in _describe_trait(0.2, "warm", "cold")

    def test_describe_trait_mid(self):
        result = _describe_trait(0.6, "high", "low", "mid")
        assert result == "mid"

    def test_if_above_above(self):
        assert _if_above(0.8, 0.5, "yes") == "yes"

    def test_if_above_below(self):
        assert _if_above(0.3, 0.5, "yes") == ""


# ── PersonalityManager ──────────────────────────────────────────────

class TestPersonalityManager:
    def test_apply_default(self, tmp_path):
        pm = PersonalityManager(TraitWeightsConfig(path=str(tmp_path / "t.json")))
        out = pm.apply()
        assert "PERSONALITY INSTRUCTIONS" in out

    def test_apply_warm(self, tmp_path):
        pm = PersonalityManager(_make_cfg({"warmth": 0.9, "empathy": 0.9}, tmp_path))
        out = pm.apply()
        assert "warm" in out.lower() or "empath" in out.lower()

    def test_apply_humor(self, tmp_path):
        pm = PersonalityManager(_make_cfg({"humor": 0.9}, tmp_path))
        out = pm.apply()
        assert "humor" in out.lower() or "wit" in out.lower()

    def test_apply_confidence(self, tmp_path):
        pm = PersonalityManager(_make_cfg({"confidence": 0.9}, tmp_path))
        out = pm.apply()
        assert "authority" in out.lower() or "conviction" in out.lower()

    def test_apply_formal(self, tmp_path):
        pm = PersonalityManager(_make_cfg({"formality": 0.9}, tmp_path))
        out = pm.apply()
        assert "formal" in out.lower() or "polished" in out.lower()

    def test_apply_curiosity(self, tmp_path):
        pm = PersonalityManager(_make_cfg({"curiosity": 0.9}, tmp_path))
        out = pm.apply()
        assert "curious" in out.lower() or "explore" in out.lower()

    def test_apply_patience(self, tmp_path):
        pm = PersonalityManager(_make_cfg({"patience": 0.9}, tmp_path))
        out = pm.apply()
        assert "explain" in out.lower() or "thorough" in out.lower()

    def test_get_mode(self, tmp_path):
        pm = PersonalityManager(_make_cfg({"warmth": 0.9, "empathy": 0.9}, tmp_path))
        mode = pm.get_mode()
        assert "label" in mode
        assert "confidence" in mode
        assert mode["label"] == "Warm"

    def test_get_mode_playful(self, tmp_path):
        pm = PersonalityManager(_make_cfg({"humor": 0.9, "creativity": 0.9}, tmp_path))
        mode = pm.get_mode()
        assert mode["label"] == "Playful"

    def test_get_mode_analytical(self, tmp_path):
        pm = PersonalityManager(_make_cfg({"formality": 0.9, "directness": 0.9}, tmp_path))
        mode = pm.get_mode()
        assert mode["label"] == "Analytical"

    def test_get_weights_snapshot(self, tmp_path):
        pm = PersonalityManager(_make_cfg({"warmth": 0.7}, tmp_path))
        snap = pm.get_weights_snapshot()
        assert "warmth" in snap
        assert snap["warmth"] == pytest.approx(0.7, abs=0.01)


# ── StyleManager ────────────────────────────────────────────────────

class TestStyleManager:
    def test_apply_formal(self, tmp_path):
        sm = StyleManager(_make_cfg({"formality": 0.9}, tmp_path))
        out = sm.apply()
        assert "formal" in out.lower()

    def test_apply_casual(self, tmp_path):
        sm = StyleManager(_make_cfg({"formality": 0.1}, tmp_path))
        out = sm.apply()
        assert "casual" in out.lower()

    def test_apply_direct(self, tmp_path):
        sm = StyleManager(_make_cfg({"directness": 0.9, "formality": 0.5}, tmp_path))
        out = sm.apply()
        assert "direct" in out.lower()

    def test_apply_neutral(self, tmp_path):
        sm = StyleManager(_make_cfg({"formality": 0.5}, tmp_path))
        out = sm.apply()
        assert "neutral" in out.lower() or "professional" in out.lower()

    def test_apply_precise(self, tmp_path):
        sm = StyleManager(_make_cfg({"factual_precision": 0.9}, tmp_path))
        out = sm.apply()
        assert "accuracy" in out.lower() or "factual" in out.lower()

    def test_apply_diplomatic(self, tmp_path):
        sm = StyleManager(_make_cfg({"directness": 0.1}, tmp_path))
        out = sm.apply()
        assert "diplomatic" in out.lower() or "tactful" in out.lower()

    def test_apply_fluent(self, tmp_path):
        sm = StyleManager(_make_cfg({"factual_precision": 0.1}, tmp_path))
        out = sm.apply()
        assert "fluency" in out.lower() or "engagement" in out.lower()

    def test_apply_tone_flex(self, tmp_path):
        sm = StyleManager(_make_cfg({"tone_flexibility": 0.9}, tmp_path))
        out = sm.apply()
        assert "adapt" in out.lower() or "tone" in out.lower()

    def test_apply_style_header(self, tmp_path):
        sm = StyleManager(TraitWeightsConfig(path=str(tmp_path / "t.json")))
        out = sm.apply()
        assert "STYLE INSTRUCTIONS" in out

    def test_get_mode_returns_label(self, tmp_path):
        sm = StyleManager(TraitWeightsConfig(path=str(tmp_path / "t.json")))
        mode = sm.get_mode()
        assert "label" in mode
        assert "confidence" in mode
        assert "scores" in mode
        assert isinstance(mode["scores"], dict)

    def test_get_mode_formal(self, tmp_path):
        sm = StyleManager(_make_cfg({"formality": 0.9, "factual_precision": 0.9}, tmp_path))
        mode = sm.get_mode()
        assert mode["label"] == "Formal"

    def test_get_mode_casual(self, tmp_path):
        sm = StyleManager(_make_cfg({"formality": 0.1, "directness": 0.8}, tmp_path))
        mode = sm.get_mode()
        assert mode["label"] == "Casual"

    def test_get_mode_direct(self, tmp_path):
        sm = StyleManager(_make_cfg({"directness": 0.9, "formality": 0.1}, tmp_path))
        mode = sm.get_mode()
        assert mode["label"] == "Direct"

    def test_get_mode_diplomatic(self, tmp_path):
        sm = StyleManager(_make_cfg({"directness": 0.1, "formality": 0.8}, tmp_path))
        mode = sm.get_mode()
        assert mode["label"] == "Diplomatic"

    def test_get_mode_precise(self, tmp_path):
        sm = StyleManager(_make_cfg({"factual_precision": 0.9, "patience": 0.9}, tmp_path))
        mode = sm.get_mode()
        assert mode["label"] == "Precise"

    def test_get_mode_flexible(self, tmp_path):
        sm = StyleManager(_make_cfg({"tone_flexibility": 0.9}, tmp_path))
        mode = sm.get_mode()
        assert mode["label"] == "Flexible"


# ── TaskManager ─────────────────────────────────────────────────────

class TestTaskManager:
    def test_apply_analytical(self, tmp_path):
        tm = TaskManager(_make_cfg({"abstract_reasoning": 0.9}, tmp_path))
        out = tm.apply()
        assert "analogies" in out.lower() or "abstract" in out.lower()

    def test_apply_creative(self, tmp_path):
        tm = TaskManager(_make_cfg({"creative_divergence": 0.9}, tmp_path))
        out = tm.apply()
        assert "perspectives" in out.lower() or "unconventional" in out.lower()

    def test_apply_methodical(self, tmp_path):
        tm = TaskManager(_make_cfg({"systematic_planning": 0.9}, tmp_path))
        out = tm.apply()
        assert "methodically" in out.lower() or "break down" in out.lower()

    def test_apply_metacognitive(self, tmp_path):
        tm = TaskManager(_make_cfg({"metacognitive_awareness": 0.9}, tmp_path))
        out = tm.apply()
        assert "reflect" in out.lower() or "thinking" in out.lower()

    def test_apply_concrete(self, tmp_path):
        tm = TaskManager(_make_cfg({"abstract_reasoning": 0.1}, tmp_path))
        out = tm.apply()
        assert "concrete" in out.lower() or "step-by-step" in out.lower()

    def test_apply_conventional(self, tmp_path):
        tm = TaskManager(_make_cfg({"creative_divergence": 0.1}, tmp_path))
        out = tm.apply()
        assert "conventional" in out.lower() or "well-established" in out.lower()

    def test_apply_fluid(self, tmp_path):
        tm = TaskManager(_make_cfg({"systematic_planning": 0.1}, tmp_path))
        out = tm.apply()
        assert "fluid" in out.lower() or "without heavy" in out.lower()

    def test_apply_task_header(self, tmp_path):
        tm = TaskManager(TraitWeightsConfig(path=str(tmp_path / "t.json")))
        out = tm.apply()
        assert "TASK APPROACH" in out

    def test_get_mode_returns_label(self, tmp_path):
        tm = TaskManager(TraitWeightsConfig(path=str(tmp_path / "t.json")))
        mode = tm.get_mode()
        assert "label" in mode
        assert "confidence" in mode
        assert isinstance(mode["scores"], dict)

    def test_get_mode_analytical(self, tmp_path):
        tm = TaskManager(_make_cfg({
            "abstract_reasoning": 0.9, "metacognitive_awareness": 0.9, "systematic_planning": 0.9
        }, tmp_path))
        mode = tm.get_mode()
        assert mode["label"] == "Analytical"

    def test_get_mode_creative(self, tmp_path):
        tm = TaskManager(_make_cfg({
            "creative_divergence": 0.9, "systematic_planning": 0.1
        }, tmp_path))
        mode = tm.get_mode()
        assert mode["label"] == "Creative"

    def test_get_mode_methodical(self, tmp_path):
        tm = TaskManager(_make_cfg({
            "systematic_planning": 0.9, "patience": 0.9
        }, tmp_path))
        mode = tm.get_mode()
        assert mode["label"] == "Methodical"

    def test_get_mode_exploratory(self, tmp_path):
        tm = TaskManager(_make_cfg({
            "creative_divergence": 0.8, "systematic_planning": 0.1
        }, tmp_path))
        mode = tm.get_mode()
        assert mode["label"] == "Exploratory"

    def test_get_mode_structured(self, tmp_path):
        tm = TaskManager(_make_cfg({
            "systematic_planning": 0.9, "abstract_reasoning": 0.9
        }, tmp_path))
        mode = tm.get_mode()
        assert mode["label"] in ("Structured", "Analytical")

    def test_get_mode_reflective(self, tmp_path):
        tm = TaskManager(_make_cfg({
            "metacognitive_awareness": 0.9, "patience": 0.9
        }, tmp_path))
        mode = tm.get_mode()
        assert mode["label"] in ("Reflective", "Analytical")


# ── MemoryManager ───────────────────────────────────────────────────

class TestMemoryManager:
    def test_working_capacity_default(self, tmp_path):
        mm = MemoryManager(TraitWeightsConfig(path=str(tmp_path / "t.json")))
        assert 5 <= mm.working_capacity <= 11

    def test_working_capacity_high_ctx(self, tmp_path):
        mm = MemoryManager(_make_cfg({"long_context_handling": 0.9}, tmp_path))
        assert mm.working_capacity >= 5

    def test_working_capacity_low_ctx(self, tmp_path):
        mm = MemoryManager(_make_cfg({"long_context_handling": 0.1}, tmp_path))
        assert mm.working_capacity >= 5

    def test_memory_importance_threshold(self, tmp_path):
        mm = MemoryManager(TraitWeightsConfig(path=str(tmp_path / "t.json")))
        assert 0.1 <= mm.memory_importance_threshold <= 0.5

    def test_retention_decay(self, tmp_path):
        mm = MemoryManager(TraitWeightsConfig(path=str(tmp_path / "t.json")))
        assert 0.01 <= mm.retention_decay <= 0.1

    def test_should_consolidate_above(self, tmp_path):
        mm = MemoryManager(TraitWeightsConfig(path=str(tmp_path / "t.json")))
        assert mm.should_consolidate(0.9) is True

    def test_should_consolidate_below(self, tmp_path):
        mm = MemoryManager(_make_cfg({"learning_adaptability": 0.9}, tmp_path))
        threshold = mm.memory_importance_threshold
        assert mm.should_consolidate(threshold - 0.01) is True

    def test_apply_memory_context_empty(self, tmp_path):
        mm = MemoryManager(TraitWeightsConfig(path=str(tmp_path / "t.json")))
        assert mm.apply_memory_context([]) == []

    def test_apply_memory_context_filters(self, tmp_path):
        mm = MemoryManager(_make_cfg({"learning_adaptability": 0.9}, tmp_path))
        episodes = [
            {"importance": 0.1, "content": "low"},
            {"importance": 0.9, "content": "high"},
        ]
        result = mm.apply_memory_context(episodes)
        assert len(result) >= 1

    def test_get_mode(self, tmp_path):
        mm = MemoryManager(TraitWeightsConfig(path=str(tmp_path / "t.json")))
        mode = mm.get_mode()
        assert "label" in mode
        assert "confidence" in mode
        assert "capacity" in mode
        assert isinstance(mode["scores"], dict)
