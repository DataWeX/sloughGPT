"""Tests for domains.context.managers — TraitWeightsConfig, PersonalityManager, MemoryManager."""

import os
import pytest
from domains.context.managers import (
    TraitWeightsConfig, PersonalityManager, MemoryManager,
)


class TestTraitWeightsConfig:
    def test_get_default(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "w.json"))
        assert cfg.get("warmth") == 0.5

    def test_set_and_get(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "w.json"))
        cfg.set("warmth", 0.8)
        assert cfg.get("warmth") == 0.8

    def test_set_clamps(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "w.json"))
        cfg.set("warmth", 2.0)
        assert cfg.get("warmth") == 1.0
        cfg.set("warmth", -1.0)
        assert cfg.get("warmth") == 0.0

    def test_update_deltas(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "w.json"))
        cfg.set("warmth", 0.5)
        cfg.update({"warmth": 0.2})
        assert cfg.get("warmth") == 0.7

    def test_all_returns_groups(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "w.json"))
        result = cfg.all()
        assert "personality" in result
        assert "warmth" in result["personality"]

    def test_persistence(self, tmp_path):
        path = str(tmp_path / "w.json")
        cfg1 = TraitWeightsConfig(path=path)
        cfg1.set("warmth", 0.9)
        cfg2 = TraitWeightsConfig(path=path)
        assert cfg2.get("warmth") == 0.9

    def test_reset(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "w.json"))
        cfg.set("warmth", 0.9)
        cfg.reset()
        assert cfg.get("warmth") == 0.5


class TestPersonalityManager:
    def test_apply_returns_string(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "w.json"))
        pm = PersonalityManager(config=cfg)
        result = pm.apply("base prompt")
        assert isinstance(result, str)

    def test_get_weights_snapshot(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "w.json"))
        cfg.set("warmth", 0.9)
        pm = PersonalityManager(config=cfg)
        snap = pm.get_weights_snapshot()
        assert "warmth" in snap
        assert snap["warmth"] == 0.9

    def test_get_mode(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "w.json"))
        pm = PersonalityManager(config=cfg)
        mode = pm.get_mode()
        assert "label" in mode


class TestMemoryManager:
    def test_working_capacity(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "w.json"))
        mm = MemoryManager(config=cfg)
        cap = mm.working_capacity
        assert isinstance(cap, int)
        assert cap > 0

    def test_importance_threshold(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "w.json"))
        mm = MemoryManager(config=cfg)
        thresh = mm.memory_importance_threshold
        assert isinstance(thresh, float)
        assert 0.0 <= thresh <= 1.0

    def test_should_consolidate(self, tmp_path):
        cfg = TraitWeightsConfig(path=str(tmp_path / "w.json"))
        mm = MemoryManager(config=cfg)
        assert isinstance(mm.should_consolidate(0.8), bool)
