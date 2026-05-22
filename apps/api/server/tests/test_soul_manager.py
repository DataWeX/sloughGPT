"""
Tests for SloManager - soul/personality discovery, parsing, and switching.
"""
import os
import json
import struct
import tempfile
from pathlib import Path
from typing import Optional, List
from domains.inference.slo_manager import SloManager, SloInfo


def _make_binary_sou(tmpdir: str, name: str, traits: Optional[List[str]] = None) -> str:
    """Create a minimal binary .soul file for testing."""
    config = {
        "name": name,
        "description": f"{name} test soul",
        "personality": {"warmth": 0.7, "creativity": 0.5},
        "personality_traits": traits or ["curious"],
        "behavior": {"reasoning_approach": "balanced"},
    }
    config_bytes = json.dumps(config).encode("utf-8")
    path = os.path.join(tmpdir, f"{name}.soul")
    with open(path, "wb") as f:
        f.write(b"SOUL")
        f.write(struct.pack("<I", 2))  # version
        f.write(struct.pack("<I", len(config_bytes)))
        f.write(config_bytes)
    return path


_TEXT_SOUL_TPL = """SOUL {name}
VERSION 1.0.0
DESCRIPTION {desc}
PERSONALITY
    warmth {warmth}
    creativity {creativity}
    curiosity 0.6
    confidence 0.5
    END
BEHAVIOR
    speaking_style {style}
    reasoning_approach {approach}
    END
SYSTEM You are {name}.
TAG {name},test
"""


def _make_text_soul(tmpdir: str, name: str, warmth: float = 0.5,
                    approach: str = "balanced") -> str:
    """Create a plain-text .soul profile file for testing."""
    path = os.path.join(tmpdir, f"{name}.soul")
    content = _TEXT_SOUL_TPL.format(
        name=name,
        desc=f"{name} test personality",
        warmth=warmth,
        creativity=0.6,
        style="conversational",
        approach=approach,
    )
    with open(path, "w") as f:
        f.write(content)
    return path


class TestSoulManagerBinary:
    """Tests for SloManager with binary .soul files."""

    def test_list_souls_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_binary_sou(tmp, "alice", ["curious"])
            _make_binary_sou(tmp, "bob", ["analytical"])
            sm = SloManager(souls_dir=tmp)
            souls = sm.list_souls()
            names = {s.name for s in souls}
            # May include souls from repo's models/souls/ fallback
            assert "alice" in names, f"Expected alice, got {names}"
            assert "bob" in names, f"Expected bob, got {names}"

    def test_parse_binary_soul_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_binary_sou(tmp, "testy", ["brave", "curious"])
            sm = SloManager(souls_dir=tmp)
            info = sm._parse_soul_info(path)
            assert info is not None
            assert info.name == "testy"
            assert "brave" in info.traits
            assert "curious" in info.traits
            assert info.personality.get("warmth") == 0.7

    def test_switch_soul(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_binary_sou(tmp, "alpha")
            _make_binary_sou(tmp, "beta")
            sm = SloManager(souls_dir=tmp)
            result = sm.switch_soul("alpha")
            assert result["success"] is True
            assert sm.get_current_soul().name == "alpha"
            # Switch to another
            sm.switch_soul("beta")
            assert sm.get_current_soul().name == "beta"

    def test_switch_soul_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            sm = SloManager(souls_dir=tmp)
            result = sm.switch_soul("nonexistent")
            assert result["success"] is False
            assert "error" in result

    def test_preference_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_binary_sou(tmp, "persist")
            # Create manager, switch, then re-create to test persistence
            sm1 = SloManager(souls_dir=tmp)
            sm1.switch_soul("persist")
            del sm1
            sm2 = SloManager(souls_dir=tmp)
            current = sm2.get_current_soul()
            assert current is not None
            assert current.name == "persist"


class TestSoulManagerTextProfiles:
    """Tests for SloManager with plain-text .soul profile files."""

    def test_list_souls_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            souls_dir = os.path.join(tmp, "souls")
            os.makedirs(souls_dir)
            _make_text_soul(souls_dir, "zen")
            _make_text_soul(souls_dir, "fuego")
            # SloManager looks for .soul in souls/ subdir of souls_dir
            sm = SloManager(souls_dir=tmp)
            souls = sm.list_souls()
            names = {s.name for s in souls}
            assert "zen" in names, f"Expected zen, got {names}"
            assert "fuego" in names, f"Expected fuego, got {names}"

    def test_parse_text_soul_traits(self):
        with tempfile.TemporaryDirectory() as tmp:
            souls_dir = os.path.join(tmp, "souls")
            os.makedirs(souls_dir)
            _make_text_soul(souls_dir, "warmy", warmth=0.9)
            sm = SloManager(souls_dir=tmp)
            info = sm.get_soul("warmy")
            assert info is not None
            # warmth=0.9 > 0.6 → should appear in traits
            assert "warmth" in info.traits

    def test_parse_text_soul_personality_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            souls_dir = os.path.join(tmp, "souls")
            os.makedirs(souls_dir)
            _make_text_soul(souls_dir, "scorey")
            sm = SloManager(souls_dir=tmp)
            info = sm.get_soul("scorey")
            assert info is not None
            assert "warmth" in info.personality
            assert "creativity" in info.personality
            assert info.personality["creativity"] == 0.6

    def test_hybrid_binary_and_text(self):
        """SloManager should discover both .soul and .soul files."""
        with tempfile.TemporaryDirectory() as tmp:
            _make_binary_sou(tmp, "bin1")
            souls_dir = os.path.join(tmp, "souls")
            os.makedirs(souls_dir)
            _make_text_soul(souls_dir, "txt1")
            sm = SloManager(souls_dir=tmp)
            souls = sm.list_souls()
            names = {s.name for s in souls}
            assert "bin1" in names, f"Expected bin1, got {names}"
            assert "txt1" in names, f"Expected txt1, got {names}"


class TestSoulManagerAPI:
    """Tests for SloManager through the FastAPI endpoint."""

    def test_souls_endpoint_returns_list(self):
        from apps.api.server.main import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        resp = client.get("/souls")
        assert resp.status_code == 200
        data = resp.json()
        assert "souls" in data
        assert isinstance(data["souls"], list)
        # current_soul can be None or a string
        assert "current_soul" in data

    def test_souls_endpoint_soul_has_personality(self):
        from apps.api.server.main import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        resp = client.get("/souls")
        data = resp.json()
        for soul in data["souls"]:
            assert "name" in soul
            assert "personality" in soul, f"Slo {soul.get('name')} missing personality"
            assert isinstance(soul["personality"], dict)

    def test_souls_endpoint_soul_has_traits(self):
        from apps.api.server.main import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        resp = client.get("/souls")
        data = resp.json()
        for soul in data["souls"]:
            assert "traits" in soul
            assert isinstance(soul["traits"], list)
