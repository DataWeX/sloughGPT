"""Tests for SloManager — hot-swappable personality system."""

import os
import struct
import json
import shutil
import pytest
from pathlib import Path
from unittest.mock import patch
from domains.inference.slo_manager import SloInfo, SloManager, get_slo_manager


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_souls_dir(tmp_path):
    """Create a temporary souls directory."""
    souls_dir = tmp_path / "models"
    souls_dir.mkdir()
    return souls_dir


def _write_mock_soul(path, name="test_soul", description="A test soul", personality=None):
    """Write a minimal binary .soul file (SOUL magic + config JSON)."""
    config = {
        "name": name,
        "description": description,
        "personality": personality or {"warmth": 0.8, "creativity": 0.6},
    }
    config_bytes = json.dumps(config).encode("utf-8")
    with open(path, "wb") as f:
        f.write(b"SOUL")
        f.write(struct.pack("<I", 1))  # version
        f.write(struct.pack("<I", len(config_bytes)))
        f.write(config_bytes)

@pytest.fixture
def manager(tmp_souls_dir):
    """SloManager backed by a temp dir (no saved preference)."""
    return SloManager(souls_dir=str(tmp_souls_dir))


# ── SloInfo ────────────────────────────────────────────────────────────────


class TestSloInfo:

    def test_defaults(self):
        info = SloInfo(name="s", path="/p")
        assert info.name == "s"
        assert info.path == "/p"
        assert info.description == ""
        assert info.personality == {}
        assert info.traits == []
        assert info.loaded_at is None

    def test_custom_fields(self):
        info = SloInfo(
            name="custom", path="/c",
            description="desc", personality={"warmth": 0.9},
            traits=["warm", "creative"],
        )
        assert info.personality["warmth"] == 0.9
        assert len(info.traits) == 2


class TestSloManager:
    """SloManager tests that avoid scanning the real models/souls/ directory."""

    @pytest.fixture(autouse=True)
    def _patch_souls_walk(self, monkeypatch):
        """Prevent _scan_souls from finding real models/souls/ dir via parent walk."""
        original = SloManager._scan_souls
        def patched_scan(self):
            self._souls_cache.clear()
            if not self.slos_dir.exists():
                return
            for ext in ("*.slo", "*.soul"):
                import glob
                for sou_path in glob.glob(str(self.slos_dir / ext)):
                    info = self._parse_soul_info(sou_path)
                    if info:
                        self._souls_cache[info.name] = info
        monkeypatch.setattr(SloManager, "_scan_souls", patched_scan)

    def test_init_empty_dir(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        m = SloManager(souls_dir=str(empty_dir))
        assert m.list_souls() == []
        assert m.get_current_soul() is None

    def test_init_nonexistent_dir(self, tmp_path):
        m = SloManager(souls_dir=str(tmp_path / "nonexistent"))
        assert m.list_souls() == []

    def test_scan_finds_soul_files(self, tmp_souls_dir):
        _write_mock_soul(tmp_souls_dir / "my_soul.soul", name="my_soul")
        m = SloManager(souls_dir=str(tmp_souls_dir))
        souls = m.list_souls()
        assert len(souls) == 1
        assert souls[0].name == "my_soul"

    def test_get_soul(self, tmp_souls_dir):
        _write_mock_soul(tmp_souls_dir / "alpha.soul", name="alpha")
        m = SloManager(souls_dir=str(tmp_souls_dir))
        soul = m.get_soul("alpha")
        assert soul is not None
        assert soul.name == "alpha"

    def test_get_soul_missing(self, tmp_souls_dir):
        m = SloManager(souls_dir=str(tmp_souls_dir))
        assert m.get_soul("nonexistent") is None

    def test_switch_soul_success(self, tmp_souls_dir):
        _write_mock_soul(tmp_souls_dir / "helper.soul", name="helper")
        m = SloManager(souls_dir=str(tmp_souls_dir))
        result = m.switch_soul("helper")
        assert result["success"] is True
        assert result["name"] == "helper"
        assert m.get_current_soul() is not None
        assert m.get_current_soul().name == "helper"

    def test_switch_soul_not_found(self, tmp_souls_dir):
        m = SloManager(souls_dir=str(tmp_souls_dir))
        result = m.switch_soul("ghost")
        assert result["success"] is False
        assert "not found" in result["error"]
        assert "available" in result

    def test_switch_soul_sets_loaded_at(self, tmp_souls_dir):
        _write_mock_soul(tmp_souls_dir / "timed.soul", name="timed")
        m = SloManager(souls_dir=str(tmp_souls_dir))
        m.switch_soul("timed")
        soul = m.get_current_soul()
        assert soul is not None
        assert soul.loaded_at is not None

    def test_switch_soul_persists_preference(self, tmp_souls_dir):
        _write_mock_soul(tmp_souls_dir / "persist.soul", name="persist")
        m = SloManager(souls_dir=str(tmp_souls_dir))
        with patch.object(m, "_preference_file", tmp_souls_dir / ".pref"):
            m.switch_soul("persist")
            assert m._preference_file.exists()
            assert m._preference_file.read_text().strip() == "persist"

    def test_register_soul(self, tmp_souls_dir):
        soul_path = str(tmp_souls_dir / "new.soul")
        _write_mock_soul(soul_path, name="registered")
        m = SloManager(souls_dir=str(tmp_souls_dir))
        info = m.register_soul(soul_path, name="custom_name")
        assert info.name == "custom_name"
        assert m.get_soul("custom_name") is not None

    def test_register_soul_invalid_path(self, tmp_souls_dir):
        m = SloManager(souls_dir=str(tmp_souls_dir))
        with pytest.raises(ValueError, match="Failed to parse"):
            m.register_soul("/nonexistent/file.soul")

    def test_create_default_souls(self, tmp_souls_dir):
        m = SloManager(souls_dir=str(tmp_souls_dir))
        m.create_default_souls()
        stats = m.get_stats()
        names = stats["available_souls"]
        assert "assistant" in names
        assert "creative" in names
        assert "analyst" in names

    def test_create_default_souls_no_overwrite(self, tmp_souls_dir):
        _write_mock_soul(tmp_souls_dir / "assistant.soul", name="assistant")
        m = SloManager(souls_dir=str(tmp_souls_dir))
        m._scan_souls()
        assert "assistant" in m._souls_cache
        m.create_default_souls()
        assert "assistant" in m._souls_cache
        assert m._souls_cache["assistant"].path != ""

    def test_get_stats(self, tmp_souls_dir):
        _write_mock_soul(tmp_souls_dir / "a.soul", name="a")
        m = SloManager(souls_dir=str(tmp_souls_dir))
        stats = m.get_stats()
        assert stats["total_souls"] == 1
        assert stats["current_soul"] is None
        assert "a" in stats["available_souls"]

    def test_get_stats_with_current(self, tmp_souls_dir):
        _write_mock_soul(tmp_souls_dir / "x.soul", name="x")
        m = SloManager(souls_dir=str(tmp_souls_dir))
        m.switch_soul("x")
        stats = m.get_stats()
        assert stats["current_soul"] == "x"

    def test_parse_binary_soul(self, tmp_souls_dir):
        path = tmp_souls_dir / "binary.soul"
        _write_mock_soul(path, name="bin", personality={"warmth": 0.9})
        m = SloManager(souls_dir=str(tmp_souls_dir))
        info = m.get_soul("bin")
        assert info is not None
        assert info.personality["warmth"] == 0.9

    def test_parse_multiple_souls(self, tmp_souls_dir):
        _write_mock_soul(tmp_souls_dir / "one.soul", name="one")
        _write_mock_soul(tmp_souls_dir / "two.soul", name="two")
        m = SloManager(souls_dir=str(tmp_souls_dir))
        assert len(m.list_souls()) == 2

    def test_soul_traits_from_personality(self, tmp_souls_dir):
        path = tmp_souls_dir / "traited.soul"
        _write_mock_soul(path, name="traited",
                         personality={"warmth": 0.8, "creativity": 0.9, "confidence": 0.3})
        m = SloManager(souls_dir=str(tmp_souls_dir))
        info = m.get_soul("traited")
        assert "warmth" in info.traits
        assert "creativity" in info.traits
        assert "confidence" not in info.traits

    def test_switch_soul_returns_description(self, tmp_souls_dir):
        _write_mock_soul(tmp_souls_dir / "desc.soul", name="desc", description="My description")
        m = SloManager(souls_dir=str(tmp_souls_dir))
        result = m.switch_soul("desc")
        assert result["description"] == "My description"

    def test_switch_soul_returns_personality(self, tmp_souls_dir):
        _write_mock_soul(tmp_souls_dir / "p.soul", name="p", personality={"warmth": 0.5})
        m = SloManager(souls_dir=str(tmp_souls_dir))
        result = m.switch_soul("p")
        assert result["personality"]["warmth"] == 0.5


# ── Global singleton ──────────────────────────────────────────────────────

class TestGetSloManager:

    def test_returns_singleton(self):
        import domains.inference.slo_manager as mod
        original = mod._slo_manager
        mod._slo_manager = None
        try:
            m1 = get_slo_manager()
            m2 = get_slo_manager()
            assert m1 is m2
        finally:
            mod._slo_manager = original


# ── Real _scan_souls (unpatched) ──────────────────────────────────────────

_TXT_SLO = """# test text profile
SOUL txtsoul
DESCRIPTION A text personality profile
PERSONALITY
    warmth 0.9
    creativity 0.7
    confidence 0.3
    END
BEHAVIOR
    reasoning_approach analytical
    END
"""


def _write_binary_soul(path, config):
    """Write a binary .soul file with an arbitrary config JSON."""
    config_bytes = json.dumps(config).encode("utf-8")
    with open(path, "wb") as f:
        f.write(b"SOUL")
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<I", len(config_bytes)))
        f.write(config_bytes)


class _DefaultConfig:
    """TraitWeightsConfig stand-in returning pure 0.5 defaults."""

    def all(self):
        from domains.context.managers import TRAIT_SCHEMA
        return {g: {t: 0.5 for t in ts} for g, ts in TRAIT_SCHEMA.items()}


def _raise_config():
    """Stand-in for a failing get_trait_config()."""
    raise RuntimeError("config down")


class TestRealScan:

    def test_top_level_binary_and_text(self, tmp_souls_dir):
        _write_mock_soul(tmp_souls_dir / "bin.soul", name="bin")
        (tmp_souls_dir / "txt.slo").write_text(_TXT_SLO)
        m = SloManager(souls_dir=str(tmp_souls_dir))
        souls = m.list_souls()
        names = {s.name for s in souls}
        assert "bin" in names
        assert "txtsoul" in names
        txt = m.get_soul("txtsoul")
        assert txt.description == "A text personality profile"
        assert txt.personality["warmth"] == 0.9
        assert "analytical" in txt.traits
        assert "warmth" in txt.traits
        assert "confidence" not in txt.traits

    def test_souls_subdirectory(self, tmp_souls_dir):
        sub = tmp_souls_dir / "souls"
        sub.mkdir()
        _write_mock_soul(sub / "sub.soul", name="sub")
        m = SloManager(souls_dir=str(tmp_souls_dir))
        assert m.get_soul("sub") is not None

    def test_repo_models_souls_resolution(self, tmp_souls_dir):
        import domains.inference.slo_manager as mod
        models_dir = Path(mod.__file__).resolve().parents[2] / "models"
        created = not models_dir.exists()
        souls_dir = models_dir / "souls"
        souls_dir.mkdir(parents=True, exist_ok=True)
        _write_mock_soul(souls_dir / "repo.soul", name="repo")
        try:
            m = SloManager(souls_dir=str(tmp_souls_dir))
            assert m.get_soul("repo") is not None
        finally:
            if created:
                shutil.rmtree(models_dir)

    def test_subdir_duplicate_name_not_overwritten(self, tmp_souls_dir):
        _write_mock_soul(tmp_souls_dir / "dup.soul", name="dup")
        sub = tmp_souls_dir / "souls"
        sub.mkdir()
        _write_mock_soul(sub / "dup.soul", name="dup")
        m = SloManager(souls_dir=str(tmp_souls_dir))
        assert m.get_soul("dup").path.startswith(str(tmp_souls_dir))


# ── Preference load/save edge cases ───────────────────────────────────────

class TestPreferenceEdgeCases:

    def test_load_restores_saved_soul(self, tmp_souls_dir):
        _write_mock_soul(tmp_souls_dir / "pref.soul", name="pref")
        m = SloManager(souls_dir=str(tmp_souls_dir))
        pref = tmp_souls_dir / ".pref"
        pref.write_text("pref")
        m._preference_file = pref
        m._load_preference()
        assert m._current_soul == "pref"

    def test_load_read_error_is_swallowed(self, tmp_souls_dir):
        m = SloManager(souls_dir=str(tmp_souls_dir))
        bad = tmp_souls_dir / "prefdir"
        bad.mkdir()
        m._preference_file = bad
        m._load_preference()
        assert m._current_soul is None

    def test_save_write_error_is_swallowed(self, tmp_souls_dir):
        blocker = tmp_souls_dir / "afile"
        blocker.write_text("x")
        m = SloManager(souls_dir=str(tmp_souls_dir))
        m._current_soul = "x"
        m._preference_file = blocker / "pref"
        m._save_preference()


# ── get_trait_weights ─────────────────────────────────────────────────────

class TestGetTraitWeights:

    def _manager_with_current(self, tmp_souls_dir, name):
        m = SloManager(souls_dir=str(tmp_souls_dir))
        m._preference_file = tmp_souls_dir / ".pref"
        m.switch_soul(name)
        return m

    def test_no_soul_returns_full_schema(self, tmp_souls_dir, monkeypatch):
        monkeypatch.setattr(
            "domains.context.managers.get_trait_config", lambda: _DefaultConfig()
        )
        m = SloManager(souls_dir=str(tmp_souls_dir))
        result = m.get_trait_weights()
        assert set(result.keys()) == {"personality", "cognition", "emotion"}
        assert result["personality"]["warmth"] == 0.5
        assert len(result["cognition"]) == 8
        assert len(result["emotion"]) == 5

    def test_soul_personality_overrides_defaults(self, tmp_souls_dir, monkeypatch):
        monkeypatch.setattr("domains.context.managers.get_trait_config", _raise_config)
        _write_mock_soul(tmp_souls_dir / "p.soul", name="p",
                         personality={"warmth": 0.9, "creativity": 0.6})
        m = self._manager_with_current(tmp_souls_dir, "p")
        result = m.get_trait_weights()
        assert result["personality"]["warmth"] == 0.9
        assert result["personality"]["creativity"] == 0.6

    def test_soul_metadata_cognition_emotion_overlay(self, tmp_souls_dir, tmp_path, monkeypatch):
        monkeypatch.setattr("domains.context.managers.get_trait_config", _raise_config)
        meta = {
            "personality": {"warmth": 0.8},
            "cognition": {"abstract_reasoning": 0.8, "systematic_planning": 0.7},
            "emotion": {"empathy_depth": 0.7},
        }
        meta_bytes = json.dumps(meta).encode("utf-8")
        path = tmp_path / "meta.soul"
        path.write_bytes(b"SOUL" + struct.pack("<I", len(meta_bytes)) + meta_bytes)

        m = SloManager(souls_dir=str(tmp_souls_dir))
        m._preference_file = tmp_souls_dir / ".pref"
        m._souls_cache["meta"] = SloInfo(
            name="meta", path=str(path), personality={"warmth": 0.8}
        )
        m.switch_soul("meta")
        result = m.get_trait_weights()
        assert result["cognition"]["abstract_reasoning"] == 0.8
        assert result["cognition"]["systematic_planning"] == 0.7
        assert result["emotion"]["empathy_depth"] == 0.7
        assert result["personality"]["warmth"] == 0.8

    def test_text_soul_skips_binary_metadata(self, tmp_souls_dir, monkeypatch):
        monkeypatch.setattr("domains.context.managers.get_trait_config", _raise_config)
        (tmp_souls_dir / "txt.slo").write_text(_TXT_SLO)
        m = self._manager_with_current(tmp_souls_dir, "txtsoul")
        result = m.get_trait_weights()
        assert result["personality"]["warmth"] == 0.9
        assert result["personality"]["creativity"] == 0.7

    def test_metadata_read_error_is_swallowed(self, tmp_souls_dir, monkeypatch):
        monkeypatch.setattr("domains.context.managers.get_trait_config", _raise_config)
        _write_mock_soul(tmp_souls_dir / "s.soul", name="s")
        m = self._manager_with_current(tmp_souls_dir, "s")
        m.get_soul("s").path = str(tmp_souls_dir / "missing.soul")
        result = m.get_trait_weights()
        assert result["personality"]["warmth"] == 0.8

    def test_empty_personality_skips_overlay(self, tmp_souls_dir, monkeypatch):
        monkeypatch.setattr("domains.context.managers.get_trait_config", _raise_config)
        _write_binary_soul(tmp_souls_dir / "e.soul", {
            "name": "e",
            "description": "no personality",
            "personality": {},
        })
        m = self._manager_with_current(tmp_souls_dir, "e")
        result = m.get_trait_weights()
        assert result["personality"]["warmth"] == 0.5

    def test_live_config_overrides_soul(self, tmp_souls_dir, monkeypatch):
        _write_mock_soul(tmp_souls_dir / "s.soul", name="s",
                         personality={"warmth": 0.8, "creativity": 0.6})
        m = self._manager_with_current(tmp_souls_dir, "s")

        class _Live:
            def all(self):
                return {"personality": {"warmth": 0.95}}

        monkeypatch.setattr("domains.context.managers.get_trait_config", lambda: _Live())
        result = m.get_trait_weights()
        assert result["personality"]["warmth"] == 0.95
        assert result["personality"]["creativity"] == 0.6

    def test_config_exception_falls_back_to_defaults(self, tmp_souls_dir, monkeypatch):
        _write_mock_soul(tmp_souls_dir / "s.soul", name="s",
                         personality={"warmth": 0.8, "creativity": 0.6})
        m = self._manager_with_current(tmp_souls_dir, "s")

        def _boom():
            raise RuntimeError("config down")

        monkeypatch.setattr("domains.context.managers.get_trait_config", _boom)
        result = m.get_trait_weights()
        assert result["personality"]["warmth"] == 0.8


# ── Module-level convenience functions ────────────────────────────────────

class TestModuleLevelFunctions:

    def test_module_switch_and_list(self):
        import domains.inference.slo_manager as mod
        original = mod._slo_manager
        mod._slo_manager = None
        try:
            result = mod.switch_soul("does_not_exist")
            assert result["success"] is False
            souls = mod.list_souls()
            assert isinstance(souls, list)
        finally:
            mod._slo_manager = original
