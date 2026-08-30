"""Tests for auto_train.py helper functions (pure functions, no mocking needed).

Covers: _parse_subtitle_text, _resolve_dataset_path, _build_soul_prompt,
_get_soul_name, _get_soul_traits, _describe_checkpoint.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from routers.auto_train import (
    _parse_subtitle_text,
    _resolve_dataset_path,
    _build_soul_prompt,
    _get_soul_name,
    _get_soul_traits,
    _describe_checkpoint,
)  # noqa: E402


# ---------------------------------------------------------------------------
# _parse_subtitle_text
# ---------------------------------------------------------------------------

class TestParseSubtitleText:
    def test_plain_text(self):
        result = _parse_subtitle_text("hello world\nfoo bar")
        assert result == ["hello world", "foo bar"]

    def test_empty_lines_filtered(self):
        result = _parse_subtitle_text("line1\n\n\nline2")
        assert result == ["line1", "line2"]

    def test_short_lines_filtered(self):
        result = _parse_subtitle_text("ok\na\nlong enough line")
        # "ok" and "a" are too short, filtered out
        assert "long enough line" in result
        assert len(result) >= 1

    def test_srt_format(self):
        srt = """1
00:00:01,000 --> 00:00:03,000
Hello world

2
00:00:04,000 --> 00:00:06,000
Goodbye world
"""
        result = _parse_subtitle_text(srt)
        assert "Hello world" in result
        assert "Goodbye world" in result
        # Timestamps and numbers filtered out
        assert not any("-->" in line for line in result)

    def test_vtt_format(self):
        vtt = """WEBVTT

00:00.000 --> 00:02.000
First line

00:03.000 --> 00:05.000
Second line
"""
        result = _parse_subtitle_text(vtt)
        assert "First line" in result
        assert "Second line" in result

    def test_bracket_lines_not_filtered(self):
        result = _parse_subtitle_text("[music]\nActual text\n[silence]")
        # Bracket lines are NOT filtered by the parser
        assert "Actual text" in result
        assert len(result) == 3


# ---------------------------------------------------------------------------
# _resolve_dataset_path
# ---------------------------------------------------------------------------

class TestResolveDatasetPath:
    def test_nonexistent_dataset(self):
        result = _resolve_dataset_path("nonexistent_dataset_xyz")
        assert result == ""

    def test_with_corpus_jsonl(self, tmp_path):
        ds_dir = tmp_path / "datasets" / "test_ds"
        ds_dir.mkdir(parents=True)
        (ds_dir / "corpus.jsonl").write_text('{"text":"hello"}')
        with pytest.MonkeyPatch.context() as m:
            m.setattr("domains.training.service.REPO_ROOT", tmp_path)
            result = _resolve_dataset_path("test_ds")
        assert result.endswith("corpus.jsonl")

    def test_with_input_txt(self, tmp_path):
        ds_dir = tmp_path / "datasets" / "test_ds2"
        ds_dir.mkdir(parents=True)
        (ds_dir / "input.txt").write_text("hello world")
        with pytest.MonkeyPatch.context() as m:
            m.setattr("domains.training.service.REPO_ROOT", tmp_path)
            result = _resolve_dataset_path("test_ds2")
        assert result.endswith("input.txt")

    def test_with_any_txt_fallback(self, tmp_path):
        ds_dir = tmp_path / "datasets" / "test_ds3"
        ds_dir.mkdir(parents=True)
        (ds_dir / "custom.txt").write_text("data")
        with pytest.MonkeyPatch.context() as m:
            m.setattr("domains.training.service.REPO_ROOT", tmp_path)
            result = _resolve_dataset_path("test_ds3")
        assert result.endswith("custom.txt")

    def test_empty_dir(self, tmp_path):
        ds_dir = tmp_path / "datasets" / "empty_ds"
        ds_dir.mkdir(parents=True)
        with pytest.MonkeyPatch.context() as m:
            m.setattr("domains.training.service.REPO_ROOT", tmp_path)
            result = _resolve_dataset_path("empty_ds")
        assert result == ""


# ---------------------------------------------------------------------------
# _build_soul_prompt
# ---------------------------------------------------------------------------

class TestBuildSoulPrompt:
    def test_known_souls(self):
        assert "assistant" in _build_soul_prompt("assistant").lower()
        assert "creative" in _build_soul_prompt("creative").lower()
        assert "analyst" in _build_soul_prompt("analyst").lower()
        assert "coder" in _build_soul_prompt("coder").lower()
        assert "teacher" in _build_soul_prompt("teacher").lower()

    def test_unknown_soul_defaults_to_assistant(self):
        result = _build_soul_prompt("unknown_soul")
        assert "assistant" in result.lower() or "helpful" in result.lower()


# ---------------------------------------------------------------------------
# _get_soul_name / _get_soul_traits
# ---------------------------------------------------------------------------

class TestGetSoulName:
    def test_name_attr(self):
        soul = SimpleNamespace(name="Alice")
        assert _get_soul_name(soul) == "Alice"

    def test_soul_name_attr(self):
        soul = SimpleNamespace(soul_name="Bob")
        assert _get_soul_name(soul) == "Bob"

    def test_empty_name_fallback(self):
        soul = SimpleNamespace(name="", soul_name="fallback")
        assert _get_soul_name(soul) == "fallback"

    def test_no_attrs(self):
        soul = SimpleNamespace()
        assert _get_soul_name(soul) == "unknown"


class TestGetSoulTraits:
    def test_soul_traits_attr(self):
        soul = SimpleNamespace(soul_traits={"warmth": 0.8})
        assert _get_soul_traits(soul) == {"warmth": 0.8}

    def test_personality_dict(self):
        soul = SimpleNamespace(personality={"warmth": 0.7})
        assert _get_soul_traits(soul) == {"warmth": 0.7}

    def test_personality_to_dict(self):
        soul = SimpleNamespace(personality=SimpleNamespace(to_dict=lambda: {"warmth": 0.9}))
        assert _get_soul_traits(soul) == {"warmth": 0.9}

    def test_personality_dict_attr(self):
        # personality is an object with __dict__ containing traits
        class FakePersonality:
            def __init__(self):
                self.warmth = 0.5
        soul = SimpleNamespace(personality=FakePersonality())
        assert _get_soul_traits(soul) == {"warmth": 0.5}

    def test_no_traits(self):
        soul = SimpleNamespace()
        assert _get_soul_traits(soul) == {}


# ---------------------------------------------------------------------------
# _describe_checkpoint
# ---------------------------------------------------------------------------

class TestDescribeCheckpoint:
    def test_basic_checkpoint(self):
        desc = _describe_checkpoint({"soul": "assistant", "loss": 1.2, "epochs": 5})
        assert "assistant" in desc.lower() or "Soul" in desc
        assert "5 epoch" in desc
        assert "1.20" in desc

    def test_low_loss(self):
        desc = _describe_checkpoint({"loss": 0.8})
        assert "learned well" in desc

    def test_medium_loss(self):
        desc = _describe_checkpoint({"loss": 2.5})
        assert "moderate" in desc

    def test_high_loss(self):
        desc = _describe_checkpoint({"loss": 4.0})
        assert "needs more training" in desc

    def test_with_traits(self):
        desc = _describe_checkpoint({"traits": {"warmth": 0.8, "humor": 0.5}})
        assert "Personality" in desc

    def test_with_dataset(self):
        desc = _describe_checkpoint({"training_dataset": "shakespeare"})
        assert "shakespeare" in desc.lower()

    def test_with_model_type(self):
        desc = _describe_checkpoint({"model_type": "transformer"})
        assert "transformer" in desc.lower()

    def test_unknown_model_type_not_shown(self):
        desc = _describe_checkpoint({"model_type": "slonet"})
        assert "slonet" not in desc.lower()

    def test_steps_fallback(self):
        desc = _describe_checkpoint({"steps": 100})
        assert "100 steps" in desc

    def test_minimal_checkpoint(self):
        desc = _describe_checkpoint({})
        assert "A trained model" in desc
