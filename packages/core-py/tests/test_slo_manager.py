"""Tests for SloManager — hot-swappable personality system."""

import os
import struct
import json
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


# ── SloManager ─────────────────────────────────────────────────────────────

class TestSloManager:

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
