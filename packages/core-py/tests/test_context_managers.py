"""Comprehensive tests for domains.context.managers — TraitWeightsConfig lifecycle,
PersonalityManager, MemoryManager, StyleManager, TaskManager, helpers."""

import json
import pytest
from pathlib import Path
from domains.context.managers import (
    TraitWeightsConfig,
    PersonalityManager,
    MemoryManager,
    StyleManager,
    TaskManager,
    TRAIT_SCHEMA,
    ALL_TRAITS,
    _describe_trait,
    _if_above,
    reset_trait_config,
)


# ── Helpers ──────────────────────────────────────────────────────────────

class TestDescribeTrait:
    def test_high_value(self):
        assert _describe_trait(0.9, "high", "low") == "high"

    def test_mid_value_custom(self):
        assert _describe_trait(0.6, "high", "low", "custom mid") == "custom mid"

    def test_mid_value_default(self):
        assert _describe_trait(0.5, "high", "low") == "moderately low"

    def test_low_value(self):
        assert _describe_trait(0.2, "high", "low") == "low"

    def test_boundary_high(self):
        assert _describe_trait(0.75, "high", "low") == "high"

    def test_boundary_mid(self):
        assert _describe_trait(0.45, "high", "low") == "moderately low"

    def test_boundary_zero(self):
        assert _describe_trait(0.0, "high", "low") == "low"

    def test_boundary_one(self):
        assert _describe_trait(1.0, "high", "low") == "high"


class TestIfAbove:
    def test_above(self):
        assert _if_above(0.8, 0.5, "yes") == "yes"

    def test_below(self):
        assert _if_above(0.3, 0.5, "yes") == ""

    def test_at_threshold(self):
        assert _if_above(0.5, 0.5, "yes") == "yes"


# ── TraitWeightsConfig ──────────────────────────────────────────────────

class TestTraitWeightsConfig:
    def test_init_creates_directory(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
        assert cfg._path.parent.exists()

    def test_get_default(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
        cfg._weights = {}
        assert cfg.get("warmth") == 0.5

    def test_get_custom_default(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
        assert cfg.get("missing", 0.3) == 0.3

    def test_set_and_get(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
        cfg.set("warmth", 0.8)
        assert cfg.get("warmth") == 0.8

    def test_set_clamps_high(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
        cfg.set("warmth", 1.5)
        assert cfg.get("warmth") == 1.0

    def test_set_clamps_low(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
        cfg.set("warmth", -0.5)
        assert cfg.get("warmth") == 0.0

    def test_set_persists(self, tmp_path):
        path = tmp_path / "traits.json"
        cfg = TraitWeightsConfig(path=str(path))
        cfg.set("warmth", 0.9)
        cfg2 = TraitWeightsConfig(path=str(path))
        assert cfg2.get("warmth") == 0.9

    def test_all_returns_all_traits(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
        result = cfg.all()
        for group, traits in TRAIT_SCHEMA.items():
            assert group in result
            for t in traits:
                assert t in result[group]
                assert result[group][t] == 0.5

    def test_all_respects_set(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
        cfg.set("warmth", 0.9)
        assert cfg.all()["personality"]["warmth"] == 0.9

    def test_update_adds_delta(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
        cfg.update({"warmth": 0.1})
        assert cfg.get("warmth") == pytest.approx(0.6)
        cfg.update({"warmth": -0.2})
        assert cfg.get("warmth") == pytest.approx(0.4)

    def test_update_clamps(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
        cfg.update({"warmth": 2.0})
        assert cfg.get("warmth") == 1.0
        cfg.update({"warmth": -5.0})
        assert cfg.get("warmth") == 0.0

    def test_set_many(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
        cfg.set_many({"warmth": 0.9, "humor": 0.1})
        assert cfg.get("warmth") == 0.9
        assert cfg.get("humor") == 0.1

    def test_set_many_ignores_unknown(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
        cfg.set_many({"not_a_trait": 0.9, "warmth": 0.8})
        assert cfg.get("warmth") == 0.8
        assert cfg.get("not_a_trait", None) is None

    def test_reset(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
        cfg.set("warmth", 0.9)
        cfg.reset()
        assert cfg.get("warmth") == 0.5

    def test_invalid_json_on_disk(self, tmp_path):
        path = tmp_path / "traits.json"
        path.write_text("NOT JSON{{{")
        cfg = TraitWeightsConfig(path=str(path))
        assert cfg.get("warmth") == 0.5

    def test_non_trait_keys_filtered(self, tmp_path):
        path = tmp_path / "traits.json"
        path.write_text(json.dumps({"warmth": 0.9, "unknown": 0.3}))
        cfg = TraitWeightsConfig(path=str(path))
        assert cfg.get("warmth") == 0.9
        assert cfg.get("unknown", None) is None

    def test_all_traits_list_is_flat(self):
        assert len(ALL_TRAITS) == 23
        for traits in TRAIT_SCHEMA.values():
            for t in traits:
                assert t in ALL_TRAITS


# ── TraitWeightsConfig snapshots ────────────────────────────────────────

class TestTraitSnapshots:
    def test_save_and_list(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
        cfg.set("warmth", 0.9)
        cfg.save_snapshot("baseline")
        snapshots = cfg.list_snapshots()
        assert len(snapshots) >= 1
        names = [s["name"] for s in snapshots]
        assert "baseline" in names

    def test_load_snapshot(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
        cfg.set("warmth", 0.9)
        cfg.save_snapshot("v1")
        cfg.set("warmth", 0.1)
        loaded = cfg.load_snapshot("v1")
        assert loaded >= 1
        assert cfg.get("warmth") == 0.9

    def test_load_nonexistent(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
        assert cfg.load_snapshot("nope") == 0

    def test_delete_snapshot(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
        cfg.save_snapshot("to_delete")
        assert cfg.delete_snapshot("to_delete") is True
        assert cfg.delete_snapshot("to_delete") is False

    def test_save_snapshot_safe_name(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
        cfg.save_snapshot("my snapshot/v1")
        snapshots = cfg.list_snapshots()
        names = [s["name"] for s in snapshots]
        assert "my_snapshot_v1" in names

    def test_snapshot_contains_meta(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
        cfg.save_snapshot("meta_test")
        snapshots = cfg.list_snapshots()
        meta = [s for s in snapshots if s["name"] == "meta_test"][0]
        assert "saved_at" in meta

    def test_load_snapshot_overwrites(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
        cfg.set_many({"warmth": 0.9, "humor": 0.1})
        cfg.save_snapshot("s1")
        cfg.set_many({"warmth": 0.2, "humor": 0.8})
        cfg.save_snapshot("s2")
        cfg.load_snapshot("s1")
        assert cfg.get("warmth") == 0.9
        assert cfg.get("humor") == 0.1


# ── Feedback-driven update ──────────────────────────────────────────────

class TestUpdateFromFeedback:
    def test_thumbs_up_increases(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
        count = cfg.update_from_feedback("thumbs_up")
        assert count == len(ALL_TRAITS)
        for trait in ALL_TRAITS:
            assert cfg.get(trait) >= 0.5

    def test_thumbs_down_decreases(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
        cfg.update_from_feedback("thumbs_down")
        for trait in ALL_TRAITS:
            assert cfg.get(trait) <= 0.5

    def test_humor_boost(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
        cfg.update_from_feedback("thumbs_up", user_message="that was a funny joke")
        assert cfg.get("humor") > cfg.get("warmth")

    def test_negation_flips(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
        cfg.update_from_feedback("thumbs_up", user_message="not formal at all")
        assert cfg.get("formality") < 0.5

    def test_short_response_boosts_directness(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
        cfg.update_from_feedback("thumbs_up", response="Short answer.")
        assert cfg.get("directness") > 0.5

    def test_long_response_boosts_patience(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
        long_resp = " ".join(["word"] * 100)
        cfg.update_from_feedback("thumbs_up", response=long_resp)
        assert cfg.get("patience") > 0.5

    def test_code_response_boosts_precision(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
        cfg.update_from_feedback("thumbs_up", response="Here is `code`:\n```python\npass\n```")
        assert cfg.get("factual_precision") > 0.5

    def test_paragraph_response_boosts_planning(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
        cfg.update_from_feedback("thumbs_up", response="Part one.\n\nPart two.")
        assert cfg.get("systematic_planning") > 0.5

    def test_confidence_optimistic_up(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
        cfg.update_from_feedback("thumbs_up")
        assert cfg.get("confidence") > 0.5
        assert cfg.get("optimism") > 0.5


# ── PersonalityManager ──────────────────────────────────────────────────

class TestPersonalityManager:
    def test_apply_returns_block(self, tmp_path):
        pm = PersonalityManager(config=TraitWeightsConfig(path=str(tmp_path / "t.json")))
        block = pm.apply()
        assert "[PERSONALITY INSTRUCTIONS]" in block
        assert "Personality:" in block

    def test_apply_warm(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set_many({"warmth": 0.9, "empathy": 0.9})
        block = PersonalityManager(config=cfg).apply()
        assert "warm" in block.lower()

    def test_apply_cool(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set("warmth", 0.1)
        block = PersonalityManager(config=cfg).apply()
        assert "reserved" in block.lower() or "distant" in block.lower()

    def test_get_weights_snapshot(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set("warmth", 0.8)
        w = PersonalityManager(config=cfg).get_weights_snapshot()
        assert w["warmth"] == 0.8

    def test_get_mode_analytical(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set_many({"formality": 0.9, "directness": 0.8, "patience": 0.7, "curiosity": 0.7})
        mode = PersonalityManager(config=cfg).get_mode()
        assert mode["label"] == "Analytical"

    def test_get_mode_warm(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set_many({"warmth": 0.9, "empathy": 0.9, "optimism": 0.8})
        mode = PersonalityManager(config=cfg).get_mode()
        assert mode["label"] == "Warm"

    def test_get_mode_playful(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set_many({"humor": 0.9, "creativity": 0.8, "optimism": 0.7, "formality": 0.2})
        mode = PersonalityManager(config=cfg).get_mode()
        assert mode["label"] == "Playful"

    def test_get_mode_confident(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set_many({"confidence": 0.9, "directness": 0.8, "optimism": 0.7})
        mode = PersonalityManager(config=cfg).get_mode()
        assert mode["label"] == "Confident"

    def test_get_mode_reserved(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set_many({"warmth": 0.1, "humor": 0.1, "confidence": 0.1, "optimism": 0.1})
        mode = PersonalityManager(config=cfg).get_mode()
        assert mode["label"] == "Reserved"

    def test_get_mode_creative(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set_many({"creativity": 0.9, "curiosity": 0.8, "humor": 0.6, "formality": 0.2})
        mode = PersonalityManager(config=cfg).get_mode()
        assert mode["label"] == "Creative"

    def test_mode_has_all_fields(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        mode = PersonalityManager(config=cfg).get_mode()
        assert "label" in mode
        assert "confidence" in mode
        assert "scores" in mode
        assert len(mode["scores"]) == 6

    def test_apply_humor_above_threshold(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set("humor", 0.8)
        block = PersonalityManager(config=cfg).apply()
        assert "humor" in block.lower() or "wit" in block.lower()

    def test_apply_patience_above_threshold(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set("patience", 0.8)
        block = PersonalityManager(config=cfg).apply()
        assert "explain" in block.lower() or "thoroughly" in block.lower()

    def test_apply_curiosity_above_threshold(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set("curiosity", 0.8)
        block = PersonalityManager(config=cfg).apply()
        assert "curious" in block.lower() or "tangent" in block.lower()

    def test_apply_confidence_high(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set("confidence", 0.8)
        block = PersonalityManager(config=cfg).apply()
        assert "authority" in block.lower() or "conviction" in block.lower()


# ── MemoryManager ──────────────────────────────────────────────────────

class TestMemoryManager:
    def test_working_capacity_default(self, tmp_path):
        mm = MemoryManager(config=TraitWeightsConfig(path=str(tmp_path / "t.json")))
        assert mm.working_capacity == 8

    def test_working_capacity_low(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set("long_context_handling", 0.1)
        mm = MemoryManager(config=cfg)
        assert mm.working_capacity == 5

    def test_working_capacity_high(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set("long_context_handling", 1.0)
        mm = MemoryManager(config=cfg)
        assert mm.working_capacity == 11

    def test_importance_threshold_default(self, tmp_path):
        mm = MemoryManager(config=TraitWeightsConfig(path=str(tmp_path / "t.json")))
        assert mm.memory_importance_threshold == pytest.approx(0.35)

    def test_importance_threshold_high_adapt(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set("learning_adaptability", 1.0)
        mm = MemoryManager(config=cfg)
        assert mm.memory_importance_threshold == pytest.approx(0.2)

    def test_importance_threshold_low_adapt(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set("learning_adaptability", 0.0)
        mm = MemoryManager(config=cfg)
        assert mm.memory_importance_threshold == pytest.approx(0.5)

    def test_retention_decay_default(self, tmp_path):
        mm = MemoryManager(config=TraitWeightsConfig(path=str(tmp_path / "t.json")))
        assert mm.retention_decay == pytest.approx(0.06)

    def test_retention_decay_high_pattern(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set("pattern_recognition", 1.0)
        mm = MemoryManager(config=cfg)
        assert mm.retention_decay == pytest.approx(0.02)

    def test_retention_decay_low_pattern(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set("pattern_recognition", 0.0)
        mm = MemoryManager(config=cfg)
        assert mm.retention_decay == pytest.approx(0.1)

    def test_should_consolidate_above(self, tmp_path):
        mm = MemoryManager(config=TraitWeightsConfig(path=str(tmp_path / "t.json")))
        assert mm.should_consolidate(0.5) is True

    def test_should_consolidate_below(self, tmp_path):
        mm = MemoryManager(config=TraitWeightsConfig(path=str(tmp_path / "t.json")))
        assert mm.should_consolidate(0.1) is False

    def test_should_consolidate_at_boundary(self, tmp_path):
        mm = MemoryManager(config=TraitWeightsConfig(path=str(tmp_path / "t.json")))
        assert mm.should_consolidate(0.35) is True

    def test_apply_memory_context_filters(self, tmp_path):
        mm = MemoryManager(config=TraitWeightsConfig(path=str(tmp_path / "t.json")))
        episodes = [
            {"importance": 0.8, "content": "keep"},
            {"importance": 0.1, "content": "drop"},
        ]
        result = mm.apply_memory_context(episodes)
        assert len(result) == 1
        assert result[0]["content"] == "keep"

    def test_apply_memory_context_empty(self, tmp_path):
        mm = MemoryManager(config=TraitWeightsConfig(path=str(tmp_path / "t.json")))
        assert mm.apply_memory_context([]) == []

    def test_apply_memory_context_default_importance(self, tmp_path):
        mm = MemoryManager(config=TraitWeightsConfig(path=str(tmp_path / "t.json")))
        episodes = [{"content": "no importance key"}]
        result = mm.apply_memory_context(episodes)
        assert len(result) == 1

    def test_get_mode_deep_context(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set_many({"long_context_handling": 0.9, "pattern_recognition": 0.2})
        mm = MemoryManager(config=cfg)
        mode = mm.get_mode()
        assert mode["label"] == "Deep Context"
        assert mode["capacity"] == mm.working_capacity

    def test_get_mode_focused(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set_many({"long_context_handling": 0.1, "pattern_recognition": 0.1})
        mm = MemoryManager(config=cfg)
        mode = mm.get_mode()
        assert mode["label"] == "Focused"

    def test_get_mode_expansive(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set_many({"long_context_handling": 0.1, "pattern_recognition": 0.1, "learning_adaptability": 0.9})
        mm = MemoryManager(config=cfg)
        mode = mm.get_mode()
        assert mode["label"] == "Expansive"

    def test_get_mode_has_capacity_and_scores(self, tmp_path):
        mm = MemoryManager(config=TraitWeightsConfig(path=str(tmp_path / "t.json")))
        mode = mm.get_mode()
        assert "capacity" in mode
        assert "scores" in mode
        assert "confidence" in mode
        assert len(mode["scores"]) == 5


# ── StyleManager ───────────────────────────────────────────────────────

class TestStyleManager:
    def test_apply_returns_block(self, tmp_path):
        sm = StyleManager(config=TraitWeightsConfig(path=str(tmp_path / "t.json")))
        block = sm.apply()
        assert "[STYLE INSTRUCTIONS]" in block

    def test_apply_formal(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set("formality", 0.9)
        block = StyleManager(config=cfg).apply()
        assert "formal" in block.lower()

    def test_apply_casual(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set("formality", 0.1)
        block = StyleManager(config=cfg).apply()
        assert "casual" in block.lower()

    def test_apply_neutral(self, tmp_path):
        sm = StyleManager(config=TraitWeightsConfig(path=str(tmp_path / "t.json")))
        block = sm.apply()
        assert "neutral" in block.lower()

    def test_apply_direct(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set("directness", 0.9)
        block = StyleManager(config=cfg).apply()
        assert "direct" in block.lower()

    def test_apply_diplomatic(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set("directness", 0.1)
        block = StyleManager(config=cfg).apply()
        assert "diplomatic" in block.lower()

    def test_apply_precise(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set("factual_precision", 0.9)
        block = StyleManager(config=cfg).apply()
        assert "accuracy" in block.lower() or "precise" in block.lower()

    def test_apply_fluency(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set("factual_precision", 0.1)
        block = StyleManager(config=cfg).apply()
        assert "fluency" in block.lower()

    def test_apply_tone_flex(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set("tone_flexibility", 0.9)
        block = StyleManager(config=cfg).apply()
        assert "adapt" in block.lower()

    def test_get_mode(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set("formality", 0.9)
        mode = StyleManager(config=cfg).get_mode()
        assert "label" in mode
        assert "scores" in mode
        assert len(mode["scores"]) == 6


# ── TaskManager ────────────────────────────────────────────────────────

class TestTaskManager:
    def test_apply_returns_block(self, tmp_path):
        tm = TaskManager(config=TraitWeightsConfig(path=str(tmp_path / "t.json")))
        block = tm.apply()
        assert "[TASK APPROACH]" in block

    def test_apply_abstract_high(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set("abstract_reasoning", 0.9)
        block = TaskManager(config=cfg).apply()
        assert "analogy" in block.lower() or "concept" in block.lower()

    def test_apply_abstract_low(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set("abstract_reasoning", 0.1)
        block = TaskManager(config=cfg).apply()
        assert "concrete" in block.lower() or "step" in block.lower()

    def test_apply_creative_high(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set("creative_divergence", 0.9)
        block = TaskManager(config=cfg).apply()
        assert "perspective" in block.lower() or "unconventional" in block.lower()

    def test_apply_creative_low(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set("creative_divergence", 0.1)
        block = TaskManager(config=cfg).apply()
        assert "conventional" in block.lower() or "established" in block.lower()

    def test_apply_planning_high(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set("systematic_planning", 0.9)
        block = TaskManager(config=cfg).apply()
        assert "methodically" in block.lower() or "step" in block.lower()

    def test_apply_planning_low(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set("systematic_planning", 0.1)
        block = TaskManager(config=cfg).apply()
        assert "fluidly" in block.lower() or "without heavy" in block.lower()

    def test_apply_metacog_high(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set("metacognitive_awareness", 0.9)
        block = TaskManager(config=cfg).apply()
        assert "reflect" in block.lower() or "thinking" in block.lower()

    def test_get_mode_analytical(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set_many({"abstract_reasoning": 0.9, "metacognitive_awareness": 0.8, "systematic_planning": 0.7})
        mode = TaskManager(config=cfg).get_mode()
        assert mode["label"] == "Analytical"

    def test_get_mode_creative(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set_many({"creative_divergence": 0.9, "systematic_planning": 0.1, "curiosity": 0.8})
        mode = TaskManager(config=cfg).get_mode()
        assert mode["label"] == "Creative"

    def test_get_mode_methodical(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set_many({"systematic_planning": 0.9, "abstract_reasoning": 0.7, "patience": 0.9})
        mode = TaskManager(config=cfg).get_mode()
        assert mode["label"] == "Methodical"

    def test_get_mode_exploratory(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set_many({"curiosity": 0.9, "creative_divergence": 0.8, "systematic_planning": 0.1})
        mode = TaskManager(config=cfg).get_mode()
        assert mode["label"] == "Exploratory"

    def test_get_mode_structured(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set_many({"systematic_planning": 0.9, "abstract_reasoning": 0.8, "metacognitive_awareness": 0.7})
        mode = TaskManager(config=cfg).get_mode()
        assert mode["label"] == "Structured"

    def test_get_mode_reflective(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        cfg.set_many({"metacognitive_awareness": 0.9, "patience": 0.8, "abstract_reasoning": 0.7})
        mode = TaskManager(config=cfg).get_mode()
        assert mode["label"] == "Reflective"

    def test_get_mode_has_all_fields(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "t.json"))
        mode = TaskManager(config=cfg).get_mode()
        assert "label" in mode
        assert "confidence" in mode
        assert "scores" in mode
        assert len(mode["scores"]) == 6


# ── reset_trait_config ─────────────────────────────────────────────────

class TestResetTraitConfig:
    def test_reset_clears_singleton(self):
        reset_trait_config()
        from domains.context.managers import _trait_config
        assert _trait_config is None
