"""Meaningful tests for TraitWeightsConfig.update_from_feedback, TaskManager mode derivation."""

import pytest
import os
from domains.context.managers import (
    TraitWeightsConfig, PersonalityManager, MemoryManager,
    StyleManager, TaskManager,
)


def _make_config(tmp_path, personality=None, cognition=None):
    cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
    updates = {}
    if personality:
        updates.update(personality)
    if cognition:
        updates.update(cognition)
    if updates:
        cfg.set_many(updates)
    return cfg


# ── update_from_feedback ───────────────────────────────────────────────

class TestUpdateFromFeedback:
    def test_thumbs_up_increases_all(self, tmp_path):
        cfg = _make_config(tmp_path)
        before = {k: cfg.get(k) for k in ["warmth", "humor", "confidence", "optimism"]}
        cfg.update_from_feedback("thumbs_up")
        after = {k: cfg.get(k) for k in before}
        for k in before:
            assert after[k] > before[k], f"{k} should increase on thumbs_up"

    def test_thumbs_down_decreases_all(self, tmp_path):
        cfg = _make_config(tmp_path, personality={"warmth": 0.8, "humor": 0.8})
        cfg.update_from_feedback("thumbs_down")
        assert cfg.get("warmth") < 0.8
        assert cfg.get("humor") < 0.8

    def test_returns_modified_count(self, tmp_path):
        cfg = _make_config(tmp_path)
        count = cfg.update_from_feedback("thumbs_up")
        assert count > 0

    def test_content_aware_boost(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.update_from_feedback("thumbs_up", user_message="that's funny and幽默")
        assert cfg.get("humor") > cfg.get("warmth")

    def test_negation_flips_traits(self, tmp_path):
        cfg = _make_config(tmp_path)
        before_formality = cfg.get("formality")
        # thumbs_down base = -0.03; negation flips to +0.03 for formality/confidence/directness
        cfg.update_from_feedback("thumbs_down", user_message="not formal at all")
        after_formality = cfg.get("formality")
        assert after_formality > before_formality  # negation flipped the negative delta

    def test_response_short_boosts_directness(self, tmp_path):
        cfg = _make_config(tmp_path)
        before = cfg.get("directness")
        cfg.update_from_feedback("thumbs_up", response="short reply")
        assert cfg.get("directness") > before

    def test_response_long_boosts_patience(self, tmp_path):
        cfg = _make_config(tmp_path)
        before = cfg.get("patience")
        long_response = " ".join(["word"] * 100)
        cfg.update_from_feedback("thumbs_up", response=long_response)
        assert cfg.get("patience") > before

    def test_response_code_boosts_precision(self, tmp_path):
        cfg = _make_config(tmp_path)
        before = cfg.get("factual_precision")
        cfg.update_from_feedback("thumbs_up", response="use `code` here")
        assert cfg.get("factual_precision") > before

    def test_response_paragraphs_boost_planning(self, tmp_path):
        cfg = _make_config(tmp_path)
        before = cfg.get("systematic_planning")
        cfg.update_from_feedback("thumbs_up", response="first part\n\nsecond part")
        assert cfg.get("systematic_planning") > before

    def test_thumbs_up_boosts_confidence_and_optimism(self, tmp_path):
        cfg = _make_config(tmp_path)
        before_conf = cfg.get("confidence")
        before_opt = cfg.get("optimism")
        cfg.update_from_feedback("thumbs_up")
        assert cfg.get("confidence") > before_conf
        assert cfg.get("optimism") > before_opt

    def test_weights_clamped_0_1(self, tmp_path):
        cfg = _make_config(tmp_path, personality={"warmth": 0.99})
        for _ in range(20):
            cfg.update_from_feedback("thumbs_up")
        assert 0.0 <= cfg.get("warmth") <= 1.0

    def test_content_aware_with_no_match(self, tmp_path):
        cfg = _make_config(tmp_path)
        before = cfg.get("humor")
        cfg.update_from_feedback("thumbs_up", user_message="random unrelated text")
        assert cfg.get("humor") >= before


# ── TaskManager ────────────────────────────────────────────────────────

class TestTaskManager:
    def test_apply_returns_block(self, tmp_path):
        tm = TaskManager(config=_make_config(tmp_path))
        block = tm.apply()
        assert "[TASK APPROACH]" in block

    def test_apply_abstract_reasoning(self, tmp_path):
        tm = TaskManager(config=_make_config(tmp_path, cognition={"abstract_reasoning": 0.9}))
        block = tm.apply()
        assert "analogies" in block.lower() or "high-level" in block.lower()

    def test_apply_concrete(self, tmp_path):
        tm = TaskManager(config=_make_config(tmp_path, cognition={"abstract_reasoning": 0.1}))
        block = tm.apply()
        assert "concrete" in block.lower() or "step-by-step" in block.lower()

    def test_apply_creative(self, tmp_path):
        tm = TaskManager(config=_make_config(tmp_path, cognition={"creative_divergence": 0.9}))
        block = tm.apply()
        assert "multiple perspectives" in block.lower() or "unconventional" in block.lower()

    def test_apply_systematic(self, tmp_path):
        tm = TaskManager(config=_make_config(tmp_path, cognition={"systematic_planning": 0.9}))
        block = tm.apply()
        assert "methodically" in block.lower() or "steps" in block.lower()

    def test_apply_metacognitive(self, tmp_path):
        tm = TaskManager(config=_make_config(tmp_path, cognition={"metacognitive_awareness": 0.9}))
        block = tm.apply()
        assert "reflect" in block.lower() or "thinking" in block.lower()

    def test_get_mode_analytical(self, tmp_path):
        tm = TaskManager(config=_make_config(tmp_path, cognition={
            "abstract_reasoning": 0.9, "metacognitive_awareness": 0.8,
            "systematic_planning": 0.7
        }))
        mode = tm.get_mode()
        assert mode["label"] == "Analytical"

    def test_get_mode_creative(self, tmp_path):
        tm = TaskManager(config=_make_config(tmp_path, cognition={
            "creative_divergence": 0.9, "systematic_planning": 0.1
        }, personality={"curiosity": 0.8}))
        mode = tm.get_mode()
        assert mode["label"] == "Creative"

    def test_get_mode_methodical(self, tmp_path):
        tm = TaskManager(config=_make_config(tmp_path, cognition={
            "systematic_planning": 0.9, "abstract_reasoning": 0.7
        }, personality={"patience": 0.8}))
        mode = tm.get_mode()
        assert mode["label"] == "Methodical"

    def test_get_mode_exploratory(self, tmp_path):
        tm = TaskManager(config=_make_config(tmp_path, cognition={
            "creative_divergence": 0.8, "systematic_planning": 0.1
        }, personality={"curiosity": 0.9}))
        mode = tm.get_mode()
        assert mode["label"] == "Exploratory"

    def test_get_mode_structured(self, tmp_path):
        tm = TaskManager(config=_make_config(tmp_path, cognition={
            "systematic_planning": 0.9, "abstract_reasoning": 0.8,
            "metacognitive_awareness": 0.7
        }))
        mode = tm.get_mode()
        assert mode["label"] == "Structured"

    def test_get_mode_reflective(self, tmp_path):
        tm = TaskManager(config=_make_config(tmp_path, cognition={
            "metacognitive_awareness": 0.9, "abstract_reasoning": 0.7
        }, personality={"patience": 0.9}))
        mode = tm.get_mode()
        assert mode["label"] == "Reflective"

    def test_get_mode_has_scores(self, tmp_path):
        tm = TaskManager(config=_make_config(tmp_path))
        mode = tm.get_mode()
        assert "scores" in mode
        assert "confidence" in mode
        assert len(mode["scores"]) == 6
