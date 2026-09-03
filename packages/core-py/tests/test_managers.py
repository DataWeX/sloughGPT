"""Tests for domains.context.managers — TraitWeightsConfig, PersonalityManager."""

from __future__ import annotations

import pytest
from pathlib import Path

from domains.context.managers import (
    TRAIT_SCHEMA,
    ALL_TRAITS,
    TraitWeightsConfig,
    PersonalityManager,
    _describe_trait,
    _if_above,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

class TestHelpers:
    def test_describe_trait_high(self):
        assert "warm and nurturing" in _describe_trait(0.9, "warm and nurturing", "cold", "neutral")

    def test_describe_trait_low(self):
        assert "cold" in _describe_trait(0.1, "warm and nurturing", "cold", "neutral")

    def test_describe_trait_mid(self):
        assert "neutral" in _describe_trait(0.5, "warm and nurturing", "cold", "neutral")

    def test_if_above_true(self):
        assert _if_above(0.8, 0.5, "yes") == "yes"

    def test_if_above_false(self):
        assert _if_above(0.3, 0.5, "yes") == ""


# ── TraitWeightsConfig ────────────────────────────────────────────────────────

class TestTraitWeightsConfig:
    def test_set_and_get(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
        cfg.set("warmth", 0.8)
        assert cfg.get("warmth") == 0.8

    def test_clamping(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
        cfg.set("warmth", 2.0)
        assert cfg.get("warmth") == 1.0
        cfg.set("warmth", -1.0)
        assert cfg.get("warmth") == 0.0

    def test_update_deltas(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
        original = cfg.get("warmth")
        cfg.update({"warmth": 0.2})
        assert cfg.get("warmth") == pytest.approx(original + 0.2)

    def test_set_many(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
        cfg.set_many({"warmth": 0.9, "humor": 0.1})
        assert cfg.get("warmth") == 0.9
        assert cfg.get("humor") == 0.1

    def test_set_many_ignores_unknown(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
        cfg.set_many({"warmth": 0.9, "nonexistent_trait": 0.1})
        assert cfg.get("warmth") == 0.9

    def test_reset(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
        cfg.set("warmth", 0.9)
        assert cfg.get("warmth") == 0.9
        cfg.reset()
        assert cfg.get("warmth") == 0.5

    def test_all_groups(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
        all_weights = cfg.all()
        assert set(all_weights.keys()) == set(TRAIT_SCHEMA.keys())
        for group, traits in TRAIT_SCHEMA.items():
            assert set(all_weights[group].keys()) == set(traits)

    def test_get_default(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
        assert cfg.get("nonexistent", 0.7) == 0.7


# ── PersonalityManager ────────────────────────────────────────────────────────

class TestPersonalityManager:
    def test_apply_generates_block(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
        pm = PersonalityManager(config=cfg)
        block = pm.apply()
        assert "[PERSONALITY INSTRUCTIONS]" in block
        assert "Personality:" in block

    def test_get_weights_snapshot(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
        cfg.set("warmth", 0.9)
        pm = PersonalityManager(config=cfg)
        snap = pm.get_weights_snapshot()
        assert snap["warmth"] == 0.9
        assert set(snap.keys()) == set(TRAIT_SCHEMA["personality"])

    def test_get_mode_default(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
        pm = PersonalityManager(config=cfg)
        mode = pm.get_mode()
        assert "label" in mode
        assert "confidence" in mode
        assert "scores" in mode
        assert mode["label"] in ["Analytical", "Warm", "Playful", "Confident", "Reserved", "Creative"]

    def test_get_mode_warm(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
        cfg.set_many({"warmth": 1.0, "empathy": 1.0, "optimism": 1.0})
        pm = PersonalityManager(config=cfg)
        mode = pm.get_mode()
        assert mode["label"] == "Warm"

    def test_get_mode_confident(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
        cfg.set_many({"confidence": 1.0, "directness": 1.0, "optimism": 1.0})
        pm = PersonalityManager(config=cfg)
        mode = pm.get_mode()
        assert mode["label"] == "Confident"

    def test_apply_high_warmth(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "traits.json"))
        cfg.set("warmth", 1.0)
        pm = PersonalityManager(config=cfg)
        block = pm.apply()
        assert "warm" in block.lower()
