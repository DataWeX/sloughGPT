"""Meaningful tests for PersonalityManager, MemoryManager, StyleManager — mode derivation, trait weights, prompt generation."""

import pytest
from pathlib import Path
from domains.context.managers import PersonalityManager, MemoryManager, StyleManager, TraitWeightsConfig


def _make_config(tmp_path, personality=None, cognition=None):
    """Create a TraitWeightsConfig backed by a temp file, with custom values."""
    cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
    updates = {}
    if personality:
        updates.update(personality)
    if cognition:
        updates.update(cognition)
    if updates:
        cfg.set_many(updates)
    return cfg


# ── PersonalityManager ─────────────────────────────────────────────────

class TestPersonalityManager:
    def test_apply_returns_block(self, tmp_path):
        pm = PersonalityManager(config=_make_config(tmp_path))
        block = pm.apply()
        assert "[PERSONALITY INSTRUCTIONS]" in block
        assert "Personality:" in block

    def test_apply_warm(self, tmp_path):
        pm = PersonalityManager(config=_make_config(tmp_path, personality={"warmth": 0.9, "empathy": 0.9}))
        block = pm.apply()
        assert "warm" in block.lower()

    def test_apply_cool(self, tmp_path):
        pm = PersonalityManager(config=_make_config(tmp_path, personality={"warmth": 0.1}))
        block = pm.apply()
        assert "reserved" in block.lower() or "distant" in block.lower()

    def test_get_weights_snapshot(self, tmp_path):
        pm = PersonalityManager(config=_make_config(tmp_path, personality={"warmth": 0.8}))
        w = pm.get_weights_snapshot()
        assert w["warmth"] == 0.8

    def test_get_mode_analytical(self, tmp_path):
        pm = PersonalityManager(config=_make_config(tmp_path, personality={
            "formality": 0.9, "directness": 0.8, "patience": 0.7, "curiosity": 0.7
        }))
        mode = pm.get_mode()
        assert mode["label"] == "Analytical"

    def test_get_mode_warm(self, tmp_path):
        pm = PersonalityManager(config=_make_config(tmp_path, personality={
            "warmth": 0.9, "empathy": 0.9, "optimism": 0.8
        }))
        mode = pm.get_mode()
        assert mode["label"] == "Warm"

    def test_get_mode_playful(self, tmp_path):
        pm = PersonalityManager(config=_make_config(tmp_path, personality={
            "humor": 0.9, "creativity": 0.8, "optimism": 0.7, "formality": 0.2
        }))
        mode = pm.get_mode()
        assert mode["label"] == "Playful"

    def test_get_mode_confident(self, tmp_path):
        pm = PersonalityManager(config=_make_config(tmp_path, personality={
            "confidence": 0.9, "directness": 0.8, "optimism": 0.7
        }))
        mode = pm.get_mode()
        assert mode["label"] == "Confident"

    def test_get_mode_reserved(self, tmp_path):
        pm = PersonalityManager(config=_make_config(tmp_path, personality={
            "warmth": 0.1, "humor": 0.1, "confidence": 0.1, "optimism": 0.1
        }))
        mode = pm.get_mode()
        assert mode["label"] == "Reserved"

    def test_get_mode_creative(self, tmp_path):
        pm = PersonalityManager(config=_make_config(tmp_path, personality={
            "creativity": 0.9, "curiosity": 0.8, "humor": 0.6, "formality": 0.2
        }))
        mode = pm.get_mode()
        assert mode["label"] == "Creative"

    def test_get_mode_has_scores(self, tmp_path):
        pm = PersonalityManager(config=_make_config(tmp_path))
        mode = pm.get_mode()
        assert "scores" in mode
        assert "confidence" in mode
        assert len(mode["scores"]) == 6


# ── MemoryManager ──────────────────────────────────────────────────────

class TestMemoryManager:
    def test_working_capacity_default(self, tmp_path):
        mm = MemoryManager(config=_make_config(tmp_path))
        assert mm.working_capacity == 8  # 5 + (0.5 * 6) = 8

    def test_working_capacity_low_ctx(self, tmp_path):
        mm = MemoryManager(config=_make_config(tmp_path, cognition={"long_context_handling": 0.1}))
        assert mm.working_capacity == 5  # 5 + (0.1 * 6) = 5.6 → 5

    def test_working_capacity_high_ctx(self, tmp_path):
        mm = MemoryManager(config=_make_config(tmp_path, cognition={"long_context_handling": 1.0}))
        assert mm.working_capacity == 11  # 5 + (1.0 * 6) = 11

    def test_importance_threshold_default(self, tmp_path):
        mm = MemoryManager(config=_make_config(tmp_path))
        assert mm.memory_importance_threshold == 0.35  # max(0.1, 0.5 - 0.5*0.3)

    def test_retention_decay_default(self, tmp_path):
        mm = MemoryManager(config=_make_config(tmp_path))
        assert mm.retention_decay == pytest.approx(0.06, abs=1e-10)

    def test_should_consolidate_above(self, tmp_path):
        mm = MemoryManager(config=_make_config(tmp_path))
        assert mm.should_consolidate(0.5) is True

    def test_should_consolidate_below(self, tmp_path):
        mm = MemoryManager(config=_make_config(tmp_path))
        assert mm.should_consolidate(0.1) is False

    def test_apply_memory_context_filters(self, tmp_path):
        mm = MemoryManager(config=_make_config(tmp_path))
        episodes = [
            {"importance": 0.8, "content": "keep"},
            {"importance": 0.1, "content": "drop"},
        ]
        result = mm.apply_memory_context(episodes)
        assert len(result) == 1
        assert result[0]["content"] == "keep"

    def test_apply_memory_context_empty(self, tmp_path):
        mm = MemoryManager(config=_make_config(tmp_path))
        assert mm.apply_memory_context([]) == []

    def test_get_mode_deep_context(self, tmp_path):
        mm = MemoryManager(config=_make_config(tmp_path, cognition={
            "long_context_handling": 0.9, "pattern_recognition": 0.2,
            "learning_adaptability": 0.5
        }))
        mode = mm.get_mode()
        assert mode["label"] == "Deep Context"
        assert mode["capacity"] == mm.working_capacity

    def test_get_mode_focused(self, tmp_path):
        mm = MemoryManager(config=_make_config(tmp_path, cognition={
            "long_context_handling": 0.1, "pattern_recognition": 0.1,
            "learning_adaptability": 0.5
        }))
        mode = mm.get_mode()
        assert mode["label"] == "Focused"

    def test_get_mode_expansive(self, tmp_path):
        mm = MemoryManager(config=_make_config(tmp_path, cognition={
            "long_context_handling": 0.1, "pattern_recognition": 0.1,
            "learning_adaptability": 0.9
        }))
        mode = mm.get_mode()
        assert mode["label"] == "Expansive"

    def test_get_mode_has_capacity(self, tmp_path):
        mm = MemoryManager(config=_make_config(tmp_path))
        mode = mm.get_mode()
        assert "capacity" in mode
        assert "scores" in mode


# ── StyleManager ───────────────────────────────────────────────────────

class TestStyleManager:
    def test_apply_returns_block(self, tmp_path):
        sm = StyleManager(config=_make_config(tmp_path))
        block = sm.apply()
        assert "[STYLE INSTRUCTIONS]" in block

    def test_apply_formal(self, tmp_path):
        sm = StyleManager(config=_make_config(tmp_path, personality={"formality": 0.9}))
        block = sm.apply()
        assert "formal" in block.lower()

    def test_apply_casual(self, tmp_path):
        sm = StyleManager(config=_make_config(tmp_path, personality={"formality": 0.1}))
        block = sm.apply()
        assert "casual" in block.lower()

    def test_apply_direct(self, tmp_path):
        sm = StyleManager(config=_make_config(tmp_path, personality={"directness": 0.9}))
        block = sm.apply()
        assert "direct" in block.lower()

    def test_apply_diplomatic(self, tmp_path):
        sm = StyleManager(config=_make_config(tmp_path, personality={"directness": 0.1}))
        block = sm.apply()
        assert "diplomatic" in block.lower()

    def test_apply_precise(self, tmp_path):
        sm = StyleManager(config=_make_config(tmp_path, cognition={"factual_precision": 0.9}))
        block = sm.apply()
        assert "accuracy" in block.lower() or "precise" in block.lower()

    def test_get_mode(self, tmp_path):
        sm = StyleManager(config=_make_config(tmp_path, personality={"formality": 0.9}))
        mode = sm.get_mode()
        assert "label" in mode
        assert "scores" in mode
