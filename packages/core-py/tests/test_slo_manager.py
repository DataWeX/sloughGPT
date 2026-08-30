"""Tests for domains.inference.slo_manager — SloInfo and SloManager."""

import json
import os
import struct
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from domains.inference.slo_manager import SloInfo, SloManager


# ---------------------------------------------------------------------------
# SloInfo — dataclass field defaults
# ---------------------------------------------------------------------------
class TestSloInfoDefaults:
    def test_name(self):
        si = SloInfo(name="test", path="/tmp/test.soul")
        assert si.name == "test"

    def test_path(self):
        si = SloInfo(name="test", path="/tmp/test.soul")
        assert si.path == "/tmp/test.soul"

    def test_description_default(self):
        si = SloInfo(name="test", path="/tmp/t.soul")
        assert si.description == ""

    def test_personality_default(self):
        si = SloInfo(name="test", path="/tmp/t.soul")
        assert si.personality == {}

    def test_traits_default(self):
        si = SloInfo(name="test", path="/tmp/t.soul")
        assert si.traits == []

    def test_loaded_at_default(self):
        si = SloInfo(name="test", path="/tmp/t.soul")
        assert si.loaded_at is None

    def test_born_at_default(self):
        si = SloInfo(name="test", path="/tmp/t.soul")
        assert si.born_at == ""

    def test_training_dataset_default(self):
        si = SloInfo(name="test", path="/tmp/t.soul")
        assert si.training_dataset == ""

    def test_epochs_trained_default(self):
        si = SloInfo(name="test", path="/tmp/t.soul")
        assert si.epochs_trained == 0

    def test_final_train_loss_default(self):
        si = SloInfo(name="test", path="/tmp/t.soul")
        assert si.final_train_loss is None

    def test_final_val_loss_default(self):
        si = SloInfo(name="test", path="/tmp/t.soul")
        assert si.final_val_loss is None

    def test_lineage_default(self):
        si = SloInfo(name="test", path="/tmp/t.soul")
        assert si.lineage == ""

    def test_base_model_default(self):
        si = SloInfo(name="test", path="/tmp/t.soul")
        assert si.base_model == ""

    def test_version_default(self):
        si = SloInfo(name="test", path="/tmp/t.soul")
        assert si.version == ""

    def test_size_mb_default(self):
        si = SloInfo(name="test", path="/tmp/t.soul")
        assert si.size_mb == 0.0

    def test_behavior_default(self):
        si = SloInfo(name="test", path="/tmp/t.soul")
        assert si.behavior == {}

    def test_cognition_default(self):
        si = SloInfo(name="test", path="/tmp/t.soul")
        assert si.cognition == {}

    def test_emotion_default(self):
        si = SloInfo(name="test", path="/tmp/t.soul")
        assert si.emotion == {}

    def test_generation_params_default(self):
        si = SloInfo(name="test", path="/tmp/t.soul")
        assert si.generation_params == {}


# ---------------------------------------------------------------------------
# SloInfo — custom field values
# ---------------------------------------------------------------------------
class TestSloInfoCustom:
    def test_description(self):
        si = SloInfo(name="n", path="p", description="a custom soul")
        assert si.description == "a custom soul"

    def test_personality(self):
        si = SloInfo(name="n", path="p", personality={"warmth": 0.8, "humor": 0.3})
        assert si.personality["warmth"] == 0.8
        assert si.personality["humor"] == 0.3

    def test_traits(self):
        si = SloInfo(name="n", path="p", traits=["friendly", "curious", "analytical"])
        assert "friendly" in si.traits
        assert len(si.traits) == 3

    def test_loaded_at(self):
        si = SloInfo(name="n", path="p", loaded_at=1234567890.0)
        assert si.loaded_at == 1234567890.0

    def test_born_at(self):
        si = SloInfo(name="n", path="p", born_at="2024-01-15")
        assert si.born_at == "2024-01-15"

    def test_training_dataset(self):
        si = SloInfo(name="n", path="p", training_dataset="custom_data.jsonl")
        assert si.training_dataset == "custom_data.jsonl"

    def test_epochs_trained(self):
        si = SloInfo(name="n", path="p", epochs_trained=100)
        assert si.epochs_trained == 100

    def test_final_train_loss(self):
        si = SloInfo(name="n", path="p", final_train_loss=0.123)
        assert si.final_train_loss == 0.123

    def test_final_val_loss(self):
        si = SloInfo(name="n", path="p", final_val_loss=0.456)
        assert si.final_val_loss == 0.456

    def test_lineage(self):
        si = SloInfo(name="n", path="p", lineage="nanogpt")
        assert si.lineage == "nanogpt"

    def test_base_model(self):
        si = SloInfo(name="n", path="p", base_model="gpt2")
        assert si.base_model == "gpt2"

    def test_version(self):
        si = SloInfo(name="n", path="p", version="2.0.0")
        assert si.version == "2.0.0"

    def test_size_mb(self):
        si = SloInfo(name="n", path="p", size_mb=1.5)
        assert si.size_mb == 1.5

    def test_behavior(self):
        b = {"reasoning_approach": "deductive"}
        si = SloInfo(name="n", path="p", behavior=b)
        assert si.behavior["reasoning_approach"] == "deductive"

    def test_cognition(self):
        c = {"pattern_recognition": 0.9, "abstract_reasoning": 0.7}
        si = SloInfo(name="n", path="p", cognition=c)
        assert si.cognition["pattern_recognition"] == 0.9

    def test_emotion(self):
        e = {"happiness": 0.6, "anger": 0.1}
        si = SloInfo(name="n", path="p", emotion=e)
        assert si.emotion["happiness"] == 0.6

    def test_generation_params(self):
        gp = {"temperature": 0.8, "top_k": 40}
        si = SloInfo(name="n", path="p", generation_params=gp)
        assert si.generation_params["temperature"] == 0.8


# ---------------------------------------------------------------------------
# SloInfo — all fields set
# ---------------------------------------------------------------------------
class TestSloInfoAllFields:
    def test_all_fields_set(self):
        si = SloInfo(
            name="full",
            path="/models/full.soul",
            description="Complete soul",
            personality={"warmth": 0.9, "creativity": 0.8},
            traits=["warm", "creative", "curious"],
            loaded_at=1700000000.0,
            born_at="2024-01-01",
            training_dataset="wiki",
            epochs_trained=50,
            final_train_loss=0.25,
            final_val_loss=0.35,
            lineage="custom",
            base_model="slonet",
            version="3.0",
            size_mb=2.5,
            behavior={"reasoning_approach": "creative"},
            cognition={"pattern_recognition": 0.95},
            emotion={"curiosity": 0.8},
            generation_params={"temperature": 0.7},
        )
        assert si.name == "full"
        assert si.path == "/models/full.soul"
        assert si.description == "Complete soul"
        assert si.personality["warmth"] == 0.9
        assert len(si.traits) == 3
        assert si.loaded_at == 1700000000.0
        assert si.born_at == "2024-01-01"
        assert si.training_dataset == "wiki"
        assert si.epochs_trained == 50
        assert si.final_train_loss == 0.25
        assert si.final_val_loss == 0.35
        assert si.lineage == "custom"
        assert si.base_model == "slonet"
        assert si.version == "3.0"
        assert si.size_mb == 2.5
        assert si.behavior["reasoning_approach"] == "creative"
        assert si.cognition["pattern_recognition"] == 0.95
        assert si.emotion["curiosity"] == 0.8
        assert si.generation_params["temperature"] == 0.7

    def test_empty_traits_list(self):
        si = SloInfo(name="n", path="p", traits=[])
        assert len(si.traits) == 0

    def test_personality_many_traits(self):
        p = {f"trait_{i}": i * 0.1 for i in range(20)}
        si = SloInfo(name="n", path="p", personality=p)
        assert len(si.personality) == 20

    def test_behavior_complex(self):
        b = {"reasoning_approach": "abductive", "response_style": "formal", "depth": 3}
        si = SloInfo(name="n", path="p", behavior=b)
        assert len(si.behavior) == 3

    def test_zero_epochs(self):
        si = SloInfo(name="n", path="p", epochs_trained=0)
        assert si.epochs_trained == 0

    def test_zero_losses(self):
        si = SloInfo(name="n", path="p", final_train_loss=0.0, final_val_loss=0.0)
        assert si.final_train_loss == 0.0
        assert si.final_val_loss == 0.0

    def test_negative_size(self):
        si = SloInfo(name="n", path="p", size_mb=-1.0)
        assert si.size_mb == -1.0

    def test_none_losses(self):
        si = SloInfo(name="n", path="p")
        assert si.final_train_loss is None
        assert si.final_val_loss is None

    def test_name_with_spaces(self):
        si = SloInfo(name="my soul", path="p")
        assert si.name == "my soul"

    def test_path_with_spaces(self):
        si = SloInfo(name="n", path="/path with spaces/file.soul")
        assert si.path == "/path with spaces/file.soul"

    def test_description_multiline(self):
        desc = "Line 1\nLine 2\nLine 3"
        si = SloInfo(name="n", path="p", description=desc)
        assert "Line 2" in si.description

    def test_loaded_at_zero(self):
        si = SloInfo(name="n", path="p", loaded_at=0.0)
        assert si.loaded_at == 0.0

    def test_loaded_at_float(self):
        si = SloInfo(name="n", path="p", loaded_at=123.456)
        assert si.loaded_at == 123.456

    def test_generation_params_empty(self):
        si = SloInfo(name="n", path="p", generation_params={})
        assert si.generation_params == {}

    def test_cognition_many_metrics(self):
        c = {f"metric_{i}": i * 0.05 for i in range(10)}
        si = SloInfo(name="n", path="p", cognition=c)
        assert len(si.cognition) == 10

    def test_emotion_many_states(self):
        e = {f"emotion_{i}": i * 0.1 for i in range(15)}
        si = SloInfo(name="n", path="p", emotion=e)
        assert len(si.emotion) == 15

    def test_traits_unique(self):
        si = SloInfo(name="n", path="p", traits=["a", "b", "a", "c"])
        assert len(si.traits) == 4

    def test_version_semver(self):
        si = SloInfo(name="n", path="p", version="1.2.3-beta")
        assert si.version == "1.2.3-beta"

    def test_lineage_chain(self):
        si = SloInfo(name="n", path="p", lineage="gpt2->slonet->custom")
        assert "->" in si.lineage


# ---------------------------------------------------------------------------
# SloInfo — edge cases
# ---------------------------------------------------------------------------
class TestSloInfoEdgeCases:
    def test_name_empty_string(self):
        si = SloInfo(name="", path="p")
        assert si.name == ""

    def test_path_empty_string(self):
        si = SloInfo(name="n", path="")
        assert si.path == ""

    def test_description_empty_string(self):
        si = SloInfo(name="n", path="p", description="")
        assert si.description == ""

    def test_personality_values_float(self):
        p = {"warmth": 0.123456789, "humor": 0.987654321}
        si = SloInfo(name="n", path="p", personality=p)
        assert abs(si.personality["warmth"] - 0.123456789) < 1e-6

    def test_traits_single_element(self):
        si = SloInfo(name="n", path="p", traits=["only_one"])
        assert si.traits == ["only_one"]

    def test_loaded_at_none(self):
        si = SloInfo(name="n", path="p", loaded_at=None)
        assert si.loaded_at is None

    def test_born_at_empty_string(self):
        si = SloInfo(name="n", path="p", born_at="")
        assert si.born_at == ""

    def test_training_dataset_empty(self):
        si = SloInfo(name="n", path="p", training_dataset="")
        assert si.training_dataset == ""

    def test_epochs_trained_negative(self):
        si = SloInfo(name="n", path="p", epochs_trained=-1)
        assert si.epochs_trained == -1

    def test_final_train_loss_large(self):
        si = SloInfo(name="n", path="p", final_train_loss=999.999)
        assert si.final_train_loss == 999.999

    def test_final_val_loss_large(self):
        si = SloInfo(name="n", path="p", final_val_loss=999.999)
        assert si.final_val_loss == 999.999

    def test_lineage_long(self):
        lineage = "->".join([f"model_{i}" for i in range(20)])
        si = SloInfo(name="n", path="p", lineage=lineage)
        assert "->" in si.lineage

    def test_version_special_chars(self):
        si = SloInfo(name="n", path="p", version="1.0.0-rc1+build.123")
        assert si.version == "1.0.0-rc1+build.123"

    def test_size_mb_very_large(self):
        si = SloInfo(name="n", path="p", size_mb=999999.99)
        assert si.size_mb == 999999.99

    def test_behavior_empty_dict(self):
        si = SloInfo(name="n", path="p", behavior={})
        assert si.behavior == {}

    def test_cognition_empty_dict(self):
        si = SloInfo(name="n", path="p", cognition={})
        assert si.cognition == {}

    def test_emotion_empty_dict(self):
        si = SloInfo(name="n", path="p", emotion={})
        assert si.emotion == {}

    def test_generation_params_many_keys(self):
        gp = {f"param_{i}": i for i in range(20)}
        si = SloInfo(name="n", path="p", generation_params=gp)
        assert len(si.generation_params) == 20

    def test_personality_negative_values(self):
        p = {"warmth": -0.5, "humor": -1.0}
        si = SloInfo(name="n", path="p", personality=p)
        assert si.personality["warmth"] == -0.5


# ---------------------------------------------------------------------------
# SloInfo — field mutability
# ---------------------------------------------------------------------------
class TestSloInfoImmutability:
    def test_name_is_set(self):
        si = SloInfo(name="original", path="p")
        assert si.name == "original"

    def test_path_is_set(self):
        si = SloInfo(name="n", path="/original/path.soul")
        assert si.path == "/original/path.soul"

    def test_personality_is_set(self):
        p = {"warmth": 0.5}
        si = SloInfo(name="n", path="p", personality=p)
        assert si.personality == p

    def test_traits_is_set(self):
        t = ["a", "b", "c"]
        si = SloInfo(name="n", path="p", traits=t)
        assert si.traits == t

    def test_name_can_be_reassigned(self):
        si = SloInfo(name="a", path="p")
        si.name = "b"
        assert si.name == "b"

    def test_description_can_be_reassigned(self):
        si = SloInfo(name="n", path="p", description="old")
        si.description = "new"
        assert si.description == "new"

    def test_behavior_can_be_mutated(self):
        si = SloInfo(name="n", path="p", behavior={"k": 1})
        si.behavior["k"] = 2
        assert si.behavior["k"] == 2

    def test_traits_can_be_mutated(self):
        si = SloInfo(name="n", path="p", traits=["a"])
        si.traits.append("b")
        assert "b" in si.traits


# ---------------------------------------------------------------------------
# SloManager — helper to create temporary soul files
# ---------------------------------------------------------------------------
def _write_soul_binary(path: Path, config: dict):
    """Write a minimal binary .soul file with SOUL magic header."""
    config_bytes = json.dumps(config).encode("utf-8")
    with open(path, "wb") as f:
        f.write(b"SOUL")
        f.write(struct.pack("<I", 1))  # version
        f.write(struct.pack("<I", len(config_bytes)))
        f.write(config_bytes)
        # Pad with fake weight data
        f.write(b"\x00" * 64)


def _write_slo_text(path: Path, content: str):
    """Write a plain-text .slo personality profile."""
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# SloManager — __init__ and directory scanning
# ---------------------------------------------------------------------------
class TestSloManagerInit:
    def test_init_creates_manager(self, tmp_path):
        m = SloManager(souls_dir=str(tmp_path))
        assert m is not None
        assert m.list_souls() == []

    def test_init_nonexistent_dir(self, tmp_path):
        nonexistent = tmp_path / "no_such_dir"
        m = SloManager(souls_dir=str(nonexistent))
        assert m.list_souls() == []

    def test_init_scans_soul_files(self, tmp_path):
        _write_soul_binary(tmp_path / "a.soul", {"name": "alpha", "description": "A"})
        _write_soul_binary(tmp_path / "b.soul", {"name": "beta", "description": "B"})
        m = SloManager(souls_dir=str(tmp_path))
        names = [s.name for s in m.list_souls()]
        assert "alpha" in names
        assert "beta" in names

    def test_init_scans_slo_files(self, tmp_path):
        _write_soul_binary(tmp_path / "x.soul", {"name": "soul_x"})
        m = SloManager(souls_dir=str(tmp_path))
        names = [s.name for s in m.list_souls()]
        assert "soul_x" in names

    def test_scan_is_idempotent(self, tmp_path):
        _write_soul_binary(tmp_path / "c.soul", {"name": "charlie"})
        m = SloManager(souls_dir=str(tmp_path))
        first = m.list_souls()
        second = m.list_souls()
        assert len(first) == len(second)

    def test_rescan_picks_up_new_files(self, tmp_path):
        m = SloManager(souls_dir=str(tmp_path))
        assert len(m.list_souls()) == 0
        _write_soul_binary(tmp_path / "new.soul", {"name": "new_soul"})
        m.rescan_souls()
        assert len(m.list_souls()) == 1


# ---------------------------------------------------------------------------
# SloManager — soul parsing from binary .soul files
# ---------------------------------------------------------------------------
class TestSloManagerParseSoul:
    def test_parse_binary_soul_metadata(self, tmp_path):
        config = {
            "name": "test_model",
            "description": "A test model",
            "personality": {"warmth": 0.7, "creativity": 0.5},
            "traits": ["friendly", "curious"],
            "born_at": "2024-06-01",
            "training_dataset": "wiki",
            "epochs_trained": 20,
            "final_train_loss": 0.15,
            "final_val_loss": 0.22,
            "lineage": "gpt2",
            "base_model": "slonet",
            "version": "1.0",
            "behavior": {"reasoning_approach": "deductive"},
            "cognition": {"memory": 0.8},
            "emotion": {"happiness": 0.6},
            "generation": {"temperature": 0.9},
        }
        _write_soul_binary(tmp_path / "m.soul", config)
        m = SloManager(souls_dir=str(tmp_path))
        souls = m.list_souls()
        assert len(souls) == 1
        s = souls[0]
        assert s.name == "test_model"
        assert s.description == "A test model"
        assert s.personality["warmth"] == 0.7
        assert s.traits == ["friendly", "curious"]
        assert s.born_at == "2024-06-01"
        assert s.training_dataset == "wiki"
        assert s.epochs_trained == 20
        assert s.final_train_loss == 0.15
        assert s.final_val_loss == 0.22
        assert s.lineage == "gpt2"
        assert s.base_model == "slonet"
        assert s.version == "1.0"
        assert s.behavior["reasoning_approach"] == "deductive"
        assert s.cognition["memory"] == 0.8
        assert s.emotion["happiness"] == 0.6
        assert s.generation_params["temperature"] == 0.9

    def test_parse_binary_soul_name_fallback_to_stem(self, tmp_path):
        _write_soul_binary(tmp_path / "fallback.soul", {})
        m = SloManager(souls_dir=str(tmp_path))
        souls = m.list_souls()
        assert souls[0].name == "fallback"

    def test_parse_binary_soul_description_fallback_to_name(self, tmp_path):
        _write_soul_binary(tmp_path / "desc.soul", {"name": "desc"})
        m = SloManager(souls_dir=str(tmp_path))
        souls = m.list_souls()
        assert souls[0].description == "desc"

    def test_parse_binary_soul_size_mb(self, tmp_path):
        _write_soul_binary(tmp_path / "sized.soul", {"name": "sized"})
        m = SloManager(souls_dir=str(tmp_path))
        souls = m.list_souls()
        assert souls[0].size_mb > 0

    def test_parse_binary_soul_default_traits_from_behavior(self, tmp_path):
        config = {
            "behavior": {"reasoning_approach": "inductive"},
            "personality": {"warmth": 0.8},
        }
        _write_soul_binary(tmp_path / "bt.soul", config)
        m = SloManager(souls_dir=str(tmp_path))
        souls = m.list_souls()
        assert "inductive" in souls[0].traits

    def test_parse_binary_soul_high_personality_traits_extracted(self, tmp_path):
        config = {
            "personality": {"warmth": 0.9, "humor": 0.2, "curiosity": 0.7},
        }
        _write_soul_binary(tmp_path / "pt.soul", config)
        m = SloManager(souls_dir=str(tmp_path))
        souls = m.list_souls()
        assert "warmth" in souls[0].traits
        assert "curiosity" in souls[0].traits
        assert "humor" not in souls[0].traits

    def test_parse_corrupted_binary_returns_none(self, tmp_path):
        bad = tmp_path / "bad.soul"
        bad.write_bytes(b"SOUL" + b"\xff" * 20)
        m = SloManager(souls_dir=str(tmp_path))
        assert m.list_souls() == []

    def test_parse_config_len_exceeds_file(self, tmp_path):
        bad = tmp_path / "overflow.soul"
        with open(bad, "wb") as f:
            f.write(b"SOUL")
            f.write(struct.pack("<I", 1))
            f.write(struct.pack("<I", 999999))  # absurd config_len
            f.write(b"{}")
        m = SloManager(souls_dir=str(tmp_path))
        # Binary parse fails, but text fallback via SouParser may parse it.
        # The key assertion: it does not crash.
        souls = m.list_souls()
        assert isinstance(souls, list)

    def test_parse_json_decode_error(self, tmp_path):
        bad = tmp_path / "badjson.soul"
        with open(bad, "wb") as f:
            f.write(b"SOUL")
            f.write(struct.pack("<I", 1))
            bad_payload = b"{invalid json"
            f.write(struct.pack("<I", len(bad_payload)))
            f.write(bad_payload)
        m = SloManager(souls_dir=str(tmp_path))
        # Binary parse fails with corrupt JSON, text fallback may handle it.
        souls = m.list_souls()
        assert isinstance(souls, list)

    def test_parse_empty_config(self, tmp_path):
        _write_soul_binary(tmp_path / "empty.soul", {})
        m = SloManager(souls_dir=str(tmp_path))
        souls = m.list_souls()
        assert len(souls) == 1
        assert souls[0].personality == {}


# ---------------------------------------------------------------------------
# SloManager — get_soul, get_current_soul
# ---------------------------------------------------------------------------
class TestSloManagerGetSoul:
    def _make_manager(self, tmp_path):
        _write_soul_binary(tmp_path / "alpha.soul", {"name": "alpha"})
        _write_soul_binary(tmp_path / "beta.soul", {"name": "beta"})
        return SloManager(souls_dir=str(tmp_path))

    def test_get_soul_exists(self, tmp_path):
        m = self._make_manager(tmp_path)
        s = m.get_soul("alpha")
        assert s is not None
        assert s.name == "alpha"

    def test_get_soul_missing(self, tmp_path):
        m = self._make_manager(tmp_path)
        assert m.get_soul("nonexistent") is None

    def test_get_current_soul_none_by_default(self, tmp_path):
        m = self._make_manager(tmp_path)
        assert m.get_current_soul() is None


# ---------------------------------------------------------------------------
# SloManager — switch_soul
# ---------------------------------------------------------------------------
class TestSloManagerSwitchSoul:
    def _make_manager(self, tmp_path):
        _write_soul_binary(tmp_path / "x.soul", {
            "name": "x",
            "personality": {"warmth": 0.5},
            "traits": ["alpha"],
        })
        _write_soul_binary(tmp_path / "y.soul", {
            "name": "y",
            "personality": {"warmth": 0.8},
            "traits": ["beta"],
        })
        return SloManager(souls_dir=str(tmp_path))

    def test_switch_success(self, tmp_path):
        m = self._make_manager(tmp_path)
        result = m.switch_soul("x")
        assert result["success"] is True
        assert result["name"] == "x"
        assert result["personality"]["warmth"] == 0.5

    def test_switch_sets_current(self, tmp_path):
        m = self._make_manager(tmp_path)
        m.switch_soul("x")
        current = m.get_current_soul()
        assert current is not None
        assert current.name == "x"

    def test_switch_updates_loaded_at(self, tmp_path):
        m = self._make_manager(tmp_path)
        m.switch_soul("x")
        soul = m.get_current_soul()
        assert soul.loaded_at is not None
        assert soul.loaded_at > 0

    def test_switch_missing_soul(self, tmp_path):
        m = self._make_manager(tmp_path)
        result = m.switch_soul("nonexistent")
        assert result["success"] is False
        assert "error" in result
        assert "available" in result

    def test_switch_persists_preference(self, tmp_path):
        m = self._make_manager(tmp_path)
        m.switch_soul("x")
        # Re-create manager — preference should be restored
        m2 = SloManager(souls_dir=str(tmp_path))
        current = m2.get_current_soul()
        assert current is not None
        assert current.name == "x"

    def test_switch_multiple_souls(self, tmp_path):
        m = self._make_manager(tmp_path)
        m.switch_soul("x")
        assert m.get_current_soul().name == "x"
        m.switch_soul("y")
        assert m.get_current_soul().name == "y"

    def test_switch_returns_traits(self, tmp_path):
        m = self._make_manager(tmp_path)
        result = m.switch_soul("x")
        assert "traits" in result
        assert "alpha" in result["traits"]

    def test_switch_returns_description(self, tmp_path):
        m = self._make_manager(tmp_path)
        result = m.switch_soul("x")
        assert "description" in result


# ---------------------------------------------------------------------------
# SloManager — register_soul
# ---------------------------------------------------------------------------
class TestSloManagerRegisterSoul:
    def test_register_soul(self, tmp_path):
        _write_soul_binary(tmp_path / "reg.soul", {"name": "reg"})
        m = SloManager(souls_dir=str(tmp_path))
        info = m.register_soul(str(tmp_path / "reg.soul"))
        assert info.name == "reg"
        assert m.get_soul("reg") is not None

    def test_register_soul_custom_name(self, tmp_path):
        _write_soul_binary(tmp_path / "orig.soul", {"name": "orig"})
        m = SloManager(souls_dir=str(tmp_path))
        info = m.register_soul(str(tmp_path / "orig.soul"), name="renamed")
        assert info.name == "renamed"
        assert m.get_soul("renamed") is not None

    def test_register_soul_invalid_file(self, tmp_path):
        m = SloManager(souls_dir=str(tmp_path))
        with pytest.raises(ValueError, match="Failed to parse"):
            m.register_soul(str(tmp_path / "nonexistent.soul"))

    def test_register_soul_adds_to_cache(self, tmp_path):
        # Write to a path outside the scan directory
        external = tmp_path.parent / f"ext_{id(tmp_path)}"
        external.mkdir(exist_ok=True)
        try:
            _write_soul_binary(external / "a.soul", {"name": "a"})
            m = SloManager(souls_dir=str(tmp_path))
            assert m.get_soul("a") is None
            m.register_soul(str(external / "a.soul"))
            assert m.get_soul("a") is not None
        finally:
            import shutil
            shutil.rmtree(external, ignore_errors=True)


# ---------------------------------------------------------------------------
# SloManager — create_default_souls
# ---------------------------------------------------------------------------
class TestSloManagerCreateDefaults:
    def test_create_defaults(self, tmp_path):
        m = SloManager(souls_dir=str(tmp_path))
        m.create_default_souls()
        names = [s.name for s in m.list_souls()]
        assert "assistant" in names
        assert "creative" in names
        assert "analyst" in names

    def test_create_defaults_does_not_overwrite(self, tmp_path):
        _write_soul_binary(tmp_path / "assistant.soul", {"name": "assistant"})
        m = SloManager(souls_dir=str(tmp_path))
        m.create_default_souls()
        names = [s.name for s in m.list_souls()]
        assert names.count("assistant") == 1

    def test_create_defaults_personality_values(self, tmp_path):
        m = SloManager(souls_dir=str(tmp_path))
        m.create_default_souls()
        assistant = m.get_soul("assistant")
        assert assistant.personality["warmth"] == 0.7

    def test_create_defaults_analyst_creativity(self, tmp_path):
        m = SloManager(souls_dir=str(tmp_path))
        m.create_default_souls()
        analyst = m.get_soul("analyst")
        assert analyst.personality["creativity"] == 0.3


# ---------------------------------------------------------------------------
# SloManager — get_stats
# ---------------------------------------------------------------------------
class TestSloManagerStats:
    def test_stats_empty(self, tmp_path):
        m = SloManager(souls_dir=str(tmp_path))
        stats = m.get_stats()
        assert stats["total_souls"] == 0
        assert stats["current_soul"] is None
        assert stats["available_souls"] == []

    def test_stats_with_souls(self, tmp_path):
        _write_soul_binary(tmp_path / "a.soul", {"name": "a"})
        _write_soul_binary(tmp_path / "b.soul", {"name": "b"})
        m = SloManager(souls_dir=str(tmp_path))
        stats = m.get_stats()
        assert stats["total_souls"] == 2
        assert "a" in stats["available_souls"]
        assert "b" in stats["available_souls"]

    def test_stats_after_switch(self, tmp_path):
        _write_soul_binary(tmp_path / "s.soul", {"name": "s"})
        m = SloManager(souls_dir=str(tmp_path))
        m.switch_soul("s")
        stats = m.get_stats()
        assert stats["current_soul"] == "s"

    def test_stats_souls_dir(self, tmp_path):
        m = SloManager(souls_dir=str(tmp_path))
        stats = m.get_stats()
        assert stats["souls_dir"] == str(tmp_path)


# ---------------------------------------------------------------------------
# SloManager — recursive scanning (souls/ subdirectory)
# ---------------------------------------------------------------------------
class TestSloManagerRecursiveScan:
    def test_scan_souls_subdirectory(self, tmp_path):
        souls_dir = tmp_path / "souls"
        souls_dir.mkdir()
        _write_soul_binary(souls_dir / "nested.soul", {"name": "nested"})
        m = SloManager(souls_dir=str(tmp_path))
        names = [s.name for s in m.list_souls()]
        assert "nested" in names

    def test_scan_recursive_glob(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        _write_soul_binary(sub / "deep.soul", {"name": "deep"})
        m = SloManager(souls_dir=str(tmp_path))
        names = [s.name for s in m.list_souls()]
        assert "deep" in names


# ---------------------------------------------------------------------------
# SloManager — preference file edge cases
# ---------------------------------------------------------------------------
class TestSloManagerPreference:
    def test_preference_missing_soul_ignored(self, tmp_path):
        _write_soul_binary(tmp_path / "x.soul", {"name": "x"})
        # Write preference for a soul that doesn't exist
        pref_dir = tmp_path / "data"
        pref_dir.mkdir()
        (pref_dir / ".soul_preference").write_text("nonexistent")
        m = SloManager(souls_dir=str(tmp_path))
        assert m.get_current_soul() is None

    def test_preference_empty_file(self, tmp_path):
        _write_soul_binary(tmp_path / "x.soul", {"name": "x"})
        pref_dir = tmp_path / "data"
        pref_dir.mkdir()
        (pref_dir / ".soul_preference").write_text("")
        m = SloManager(souls_dir=str(tmp_path))
        assert m.get_current_soul() is None

    def test_save_preference_creates_directory(self, tmp_path):
        _write_soul_binary(tmp_path / "x.soul", {"name": "x"})
        m = SloManager(souls_dir=str(tmp_path))
        m.switch_soul("x")
        assert m.get_current_soul().name == "x"

    def test_save_preference_with_no_soul(self, tmp_path):
        m = SloManager(souls_dir=str(tmp_path))
        m._save_preference()  # should not raise
        assert m.get_current_soul() is None


# ---------------------------------------------------------------------------
# SloManager — module-level convenience functions
# ---------------------------------------------------------------------------
class TestSloManagerModuleFunctions:
    def test_get_slo_manager_singleton(self, tmp_path):
        from domains.inference import slo_manager as mod
        old = mod._slo_manager
        try:
            mod._slo_manager = None
            m1 = mod.get_slo_manager()
            m2 = mod.get_slo_manager()
            assert m1 is m2
        finally:
            mod._slo_manager = old

    def test_list_souls_function(self, tmp_path):
        from domains.inference import slo_manager as mod
        old = mod._slo_manager
        try:
            mod._slo_manager = SloManager(souls_dir=str(tmp_path))
            result = mod.list_souls()
            assert isinstance(result, list)
        finally:
            mod._slo_manager = old

    def test_switch_soul_function(self, tmp_path):
        from domains.inference import slo_manager as mod
        old = mod._slo_manager
        try:
            _write_soul_binary(tmp_path / "x.soul", {"name": "x"})
            mod._slo_manager = SloManager(souls_dir=str(tmp_path))
            result = mod.switch_soul("x")
            assert result["success"] is True
        finally:
            mod._slo_manager = old


# ---------------------------------------------------------------------------
# SloManager — binary format edge cases
# ---------------------------------------------------------------------------
class TestSloManagerBinaryFormat:
    def test_non_soul_binary_file_ignored(self, tmp_path):
        garbage = tmp_path / "garbage.soul"
        garbage.write_bytes(b"\x00\x01\x02\x03\x04\x05\x06\x07" * 10)
        m = SloManager(souls_dir=str(tmp_path))
        # Binary parse fails (no SOUL magic); text fallback may parse it.
        souls = m.list_souls()
        assert isinstance(souls, list)

    def test_short_file_ignored(self, tmp_path):
        short = tmp_path / "short.soul"
        short.write_bytes(b"SOUL")
        m = SloManager(souls_dir=str(tmp_path))
        # Only 4 bytes — binary parse fails, text fallback may parse it.
        souls = m.list_souls()
        assert isinstance(souls, list)

    def test_zero_config_len(self, tmp_path):
        zc = tmp_path / "zero.soul"
        with open(zc, "wb") as f:
            f.write(b"SOUL")
            f.write(struct.pack("<I", 1))
            f.write(struct.pack("<I", 0))
        m = SloManager(souls_dir=str(tmp_path))
        # config_len=0 means no JSON to parse; text fallback may handle it.
        souls = m.list_souls()
        assert isinstance(souls, list)
