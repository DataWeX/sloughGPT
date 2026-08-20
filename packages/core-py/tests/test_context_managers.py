"""Tests for domains.context.managers — TraitWeightsConfig, PersonalityManager,
MemoryManager, StyleManager, TaskManager.

Covers: trait CRUD, clamping, snapshots, feedback updates, mode derivation,
memory consolidation, style/task instruction generation. Uses temp dir for persistence.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

_core_dir = str(Path(__file__).resolve().parents[2])
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

from domains.context.managers import (
    TRAIT_SCHEMA,
    ALL_TRAITS,
    TraitWeightsConfig,
    PersonalityManager,
    MemoryManager,
    StyleManager,
    TaskManager,
    _describe_trait,
    _if_above,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _tmp_config(tmp_path: Path) -> TraitWeightsConfig:
    return TraitWeightsConfig(path=str(tmp_path / "traits.json"))


class TestDescribeTrait:
    def test_high(self):
        assert _describe_trait(0.8, "high", "low") == "high"

    def test_mid(self):
        assert _describe_trait(0.5, "high", "low") == "moderately low"

    def test_mid_custom(self):
        assert _describe_trait(0.6, "high", "low", "custom mid") == "custom mid"

    def test_low(self):
        assert _describe_trait(0.2, "high", "low") == "low"

    def test_boundary(self):
        assert _describe_trait(0.75, "high", "low") == "high"
        assert _describe_trait(0.74, "high", "low") == "moderately low"


class TestIfAbove:
    def test_above(self):
        assert _if_above(0.8, 0.7, "yes") == "yes"

    def test_below(self):
        assert _if_above(0.3, 0.7, "yes") == ""


# ── TraitWeightsConfig ───────────────────────────────────────────────

class TestTraitWeightsConfig:
    def test_defaults(self, tmp_path):
        cfg = _tmp_config(tmp_path)
        assert cfg.get("warmth") == 0.5
        assert cfg.get("nonexistent") == 0.5

    def test_set_and_get(self, tmp_path):
        cfg = _tmp_config(tmp_path)
        cfg.set("warmth", 0.8)
        assert cfg.get("warmth") == 0.8

    def test_clamping(self, tmp_path):
        cfg = _tmp_config(tmp_path)
        cfg.set("warmth", 1.5)
        assert cfg.get("warmth") == 1.0
        cfg.set("warmth", -0.5)
        assert cfg.get("warmth") == 0.0

    def test_all_groups(self, tmp_path):
        cfg = _tmp_config(tmp_path)
        result = cfg.all()
        assert "personality" in result
        assert "cognition" in result
        assert "emotion" in result
        assert len(result["personality"]) == 10

    def test_update_deltas(self, tmp_path):
        cfg = _tmp_config(tmp_path)
        cfg.update({"warmth": 0.1, "humor": -0.2})
        assert cfg.get("warmth") == 0.6
        assert cfg.get("humor") == 0.3

    def test_set_many(self, tmp_path):
        cfg = _tmp_config(tmp_path)
        cfg.set_many({"warmth": 0.9, "humor": 0.1})
        assert cfg.get("warmth") == 0.9
        assert cfg.get("humor") == 0.1
        # Unknown traits unchanged
        assert cfg.get("nonexistent") == 0.5

    def test_reset(self, tmp_path):
        cfg = _tmp_config(tmp_path)
        cfg.set("warmth", 0.9)
        cfg.reset()
        assert cfg.get("warmth") == 0.5

    def test_persistence(self, tmp_path):
        path = tmp_path / "traits.json"
        cfg1 = TraitWeightsConfig(path=str(path))
        cfg1.set("warmth", 0.8)
        cfg2 = TraitWeightsConfig(path=str(path))
        assert cfg2.get("warmth") == 0.8

    def test_update_from_feedback_thumbs_up(self, tmp_path):
        cfg = _tmp_config(tmp_path)
        count = cfg.update_from_feedback("thumbs_up", "that was funny", "short reply")
        assert count > 0
        assert cfg.get("humor") > 0.5

    def test_update_from_feedback_thumbs_down(self, tmp_path):
        cfg = _tmp_config(tmp_path)
        cfg.update_from_feedback("thumbs_down")
        assert cfg.get("warmth") < 0.5

    def test_update_from_feedback_negation(self, tmp_path):
        cfg = _tmp_config(tmp_path)
        cfg.update_from_feedback("thumbs_up", "don't be formal")
        # Negation flips formality
        assert cfg.get("formality") < 0.5

    def test_snapshots(self, tmp_path):
        cfg = _tmp_config(tmp_path)
        cfg.set("warmth", 0.9)
        cfg.save_snapshot("test1")
        cfg.set("warmth", 0.3)
        assert cfg.get("warmth") == 0.3
        count = cfg.load_snapshot("test1")
        assert count > 0
        assert cfg.get("warmth") == 0.9

    def test_delete_snapshot(self, tmp_path):
        cfg = _tmp_config(tmp_path)
        cfg.save_snapshot("to_delete")
        assert cfg.delete_snapshot("to_delete") is True
        assert cfg.delete_snapshot("nonexistent") is False

    def test_list_snapshots(self, tmp_path):
        cfg = _tmp_config(tmp_path)
        cfg.save_snapshot("snap_a")
        cfg.save_snapshot("snap_b")
        snaps = cfg.list_snapshots()
        assert len(snaps) >= 2


# ── PersonalityManager ───────────────────────────────────────────────

class TestPersonalityManager:
    def test_apply(self, tmp_path):
        cfg = _tmp_config(tmp_path)
        mgr = PersonalityManager(config=cfg)
        text = mgr.apply()
        assert "[PERSONALITY INSTRUCTIONS]" in text
        assert "Personality:" in text

    def test_weights_snapshot(self, tmp_path):
        cfg = _tmp_config(tmp_path)
        mgr = PersonalityManager(config=cfg)
        snap = mgr.get_weights_snapshot()
        assert "warmth" in snap

    def test_mode(self, tmp_path):
        cfg = _tmp_config(tmp_path)
        mgr = PersonalityManager(config=cfg)
        mode = mgr.get_mode()
        assert "label" in mode
        assert "confidence" in mode
        assert mode["label"] in ["Analytical", "Warm", "Playful", "Confident", "Reserved", "Creative"]


# ── MemoryManager ────────────────────────────────────────────────────

class TestMemoryManager:
    def test_working_capacity(self, tmp_path):
        cfg = _tmp_config(tmp_path)
        mgr = MemoryManager(config=cfg)
        assert 5 <= mgr.working_capacity <= 11

    def test_retention_decay(self, tmp_path):
        cfg = _tmp_config(tmp_path)
        mgr = MemoryManager(config=cfg)
        assert 0.01 <= mgr.retention_decay <= 0.1

    def test_should_consolidate(self, tmp_path):
        cfg = _tmp_config(tmp_path)
        mgr = MemoryManager(config=cfg)
        assert mgr.should_consolidate(1.0) is True
        assert mgr.should_consolidate(0.0) is False

    def test_apply_memory_context(self, tmp_path):
        cfg = _tmp_config(tmp_path)
        mgr = MemoryManager(config=cfg)
        episodes = [
            {"importance": 0.9, "content": "high"},
            {"importance": 0.1, "content": "low"},
        ]
        result = mgr.apply_memory_context(episodes)
        assert len(result) >= 1

    def test_mode(self, tmp_path):
        cfg = _tmp_config(tmp_path)
        mgr = MemoryManager(config=cfg)
        mode = mgr.get_mode()
        assert "label" in mode
        assert mode["label"] in ["Deep Context", "Focused", "Adaptive", "Stable", "Expansive"]


# ── StyleManager ─────────────────────────────────────────────────────

class TestStyleManager:
    def test_apply(self, tmp_path):
        cfg = _tmp_config(tmp_path)
        mgr = StyleManager(config=cfg)
        text = mgr.apply()
        assert "[STYLE INSTRUCTIONS]" in text

    def test_mode(self, tmp_path):
        cfg = _tmp_config(tmp_path)
        mgr = StyleManager(config=cfg)
        mode = mgr.get_mode()
        assert "label" in mode
        assert mode["label"] in ["Formal", "Casual", "Direct", "Diplomatic", "Precise", "Flexible"]

    def test_formal_mode(self, tmp_path):
        cfg = _tmp_config(tmp_path)
        cfg.set("formality", 0.9)
        cfg.set("factual_precision", 0.9)
        mgr = StyleManager(config=cfg)
        mode = mgr.get_mode()
        assert mode["label"] == "Formal"


# ── TaskManager ──────────────────────────────────────────────────────

class TestTaskManager:
    def test_apply(self, tmp_path):
        cfg = _tmp_config(tmp_path)
        mgr = TaskManager(config=cfg)
        text = mgr.apply()
        assert "[TASK APPROACH]" in text

    def test_mode(self, tmp_path):
        cfg = _tmp_config(tmp_path)
        mgr = TaskManager(config=cfg)
        mode = mgr.get_mode()
        assert "label" in mode
        assert mode["label"] in ["Analytical", "Creative", "Methodical", "Exploratory", "Structured", "Reflective"]

    def test_creative_mode(self, tmp_path):
        cfg = _tmp_config(tmp_path)
        cfg.set("creative_divergence", 0.9)
        cfg.set("systematic_planning", 0.2)
        mgr = TaskManager(config=cfg)
        mode = mgr.get_mode()
        assert mode["label"] == "Creative"
