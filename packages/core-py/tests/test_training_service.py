"""Tests for training/service.py — pure business logic, no HTTP dependencies."""

from __future__ import annotations

import json
import math
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

from domains.training.service import (
    _finite_payload,
    parse_subtitle_text,
    resolve_dataset_path,
    build_soul_prompt,
    get_soul_name,
    get_soul_traits,
    read_slo_json_header,
    describe_checkpoint,
    find_checkpoint,
    log_experiment_metric,
    log_experiment_param,
    CHECKPOINTS_DIR,
    TURBO_DIR,
    SOU_MAGIC,
    VALID_CKPT_NAME,
    _VALID_DATASET_ID,
)


# ── _finite_payload ───────────────────────────────────────────────────────────


class TestFinitePayload:
    def test_normal_dict(self):
        assert _finite_payload({"a": 1, "b": 2.5}) == {"a": 1, "b": 2.5}

    def test_nan_becomes_none(self):
        assert _finite_payload({"a": float("nan")}) == {"a": None}

    def test_inf_becomes_none(self):
        assert _finite_payload({"a": float("inf")}) == {"a": None}

    def test_neg_inf_becomes_none(self):
        assert _finite_payload({"a": float("-inf")}) == {"a": None}

    def test_nested_dict(self):
        result = _finite_payload({"a": {"b": float("nan"), "c": 1.0}})
        assert result == {"a": {"b": None, "c": 1.0}}

    def test_list_with_nan(self):
        result = _finite_payload([1.0, float("nan"), 3.0])
        assert result == [1.0, None, 3.0]

    def test_tuple_becomes_list(self):
        result = _finite_payload((1.0, float("nan")))
        assert result == [1.0, None]

    def test_dataclass(self):
        @dataclass
        class Point:
            x: float
            y: float

        result = _finite_payload(Point(x=1.0, y=float("nan")))
        assert result == {"x": 1.0, "y": None}

    def test_non_float_passthrough(self):
        assert _finite_payload({"a": "hello", "b": 42, "c": None}) == {
            "a": "hello", "b": 42, "c": None
        }


# ── parse_subtitle_text ───────────────────────────────────────────────────────


class TestParseSubtitleText:
    def test_plain_text(self):
        text = "Line one\nLine two\nLine three"
        assert parse_subtitle_text(text) == ["Line one", "Line two", "Line three"]

    def test_plain_text_filters_short_lines(self):
        text = "Long line\nHi\nAnother long line"
        assert parse_subtitle_text(text) == ["Long line", "Another long line"]

    def test_srt_format(self):
        text = (
            "1\n"
            "00:00:01,000 --> 00:00:04,000\n"
            "Hello world\n"
            "\n"
            "2\n"
            "00:00:05,000 --> 00:00:08,000\n"
            "Second subtitle\n"
        )
        result = parse_subtitle_text(text)
        assert result == ["Hello world", "Second subtitle"]

    def test_srt_skips_numbers_and_timestamps(self):
        text = (
            "42\n"
            "00:01:00,000 --> 00:01:05,000\n"
            "Actual text\n"
        )
        result = parse_subtitle_text(text)
        assert result == ["Actual text"]

    def test_vtt_format(self):
        text = (
            "WEBVTT\n"
            "\n"
            "00:00.000 --> 00:03.000\n"
            "VTT subtitle\n"
        )
        result = parse_subtitle_text(text)
        assert result == ["VTT subtitle"]

    def test_empty_text(self):
        assert parse_subtitle_text("") == []

    def test_only_short_lines(self):
        text = "A\nBB\nCCC"
        # "A" is 1 char (<=2), "BB" is 2 chars (<=2), "CCC" is 3 chars (>2)
        assert parse_subtitle_text(text) == ["CCC"]

    def test_srt_skips_bracket_lines(self):
        text = (
            "1\n"
            "00:00:01,000 --> 00:00:04,000\n"
            "[music]\n"
            "Actual dialogue\n"
        )
        result = parse_subtitle_text(text)
        assert result == ["Actual dialogue"]


# ── resolve_dataset_path ──────────────────────────────────────────────────────


class TestResolveDatasetPath:
    def test_invalid_dataset_id_rejects_slash(self):
        with pytest.raises(ValueError, match="Invalid dataset ID"):
            resolve_dataset_path("../etc/passwd")

    def test_invalid_dataset_id_rejects_spaces(self):
        with pytest.raises(ValueError, match="Invalid dataset ID"):
            resolve_dataset_path("my dataset")

    def test_valid_dataset_id_format(self):
        assert _VALID_DATASET_ID.match("shakespeare")
        assert _VALID_DATASET_ID.match("my-dataset_v2")
        assert not _VALID_DATASET_ID.match("bad name")

    def test_nonexistent_dataset_returns_empty(self):
        assert resolve_dataset_path("nonexistent-dataset-xyz") == ""

    def test_finds_corpus_jsonl(self, tmp_path):
        ds_dir = tmp_path / "datasets" / "test-ds"
        ds_dir.mkdir(parents=True)
        (ds_dir / "corpus.jsonl").write_text('{"text": "hello"}\n')

        with patch("domains.training.helpers.REPO_ROOT", tmp_path):
            result = resolve_dataset_path("test-ds")
        assert result.endswith("corpus.jsonl")

    def test_finds_input_txt(self, tmp_path):
        ds_dir = tmp_path / "datasets" / "test-ds"
        ds_dir.mkdir(parents=True)
        (ds_dir / "input.txt").write_text("hello world")

        with patch("domains.training.helpers.REPO_ROOT", tmp_path):
            result = resolve_dataset_path("test-ds")
        assert result.endswith("input.txt")

    def test_finds_any_txt(self, tmp_path):
        ds_dir = tmp_path / "datasets" / "test-ds"
        ds_dir.mkdir(parents=True)
        (ds_dir / "custom.txt").write_text("data")

        with patch("domains.training.helpers.REPO_ROOT", tmp_path):
            result = resolve_dataset_path("test-ds")
        assert result.endswith("custom.txt")


# ── build_soul_prompt ────────────────────────────────────────────────────────


class TestBuildSoulPrompt:
    def test_known_souls(self):
        assert "helpful" in build_soul_prompt("assistant")
        assert "creative" in build_soul_prompt("creative")
        assert "analyst" in build_soul_prompt("analyst")
        assert "coder" in build_soul_prompt("coder")
        assert "teacher" in build_soul_prompt("teacher")

    def test_unknown_soul_defaults_to_assistant(self):
        result = build_soul_prompt("nonexistent")
        assert "helpful" in result

    def test_empty_string_defaults_to_assistant(self):
        result = build_soul_prompt("")
        assert "helpful" in result


# ── get_soul_name ────────────────────────────────────────────────────────────


class TestGetSoulName:
    def test_from_name_attr(self):
        class Soul:
            name = "test-soul"
        assert get_soul_name(Soul()) == "test-soul"

    def test_from_soul_name_attr(self):
        class Soul:
            soul_name = "fallback-soul"
        assert get_soul_name(Soul()) == "fallback-soul"

    def test_empty_name_falls_back(self):
        class Soul:
            name = ""
            soul_name = "fallback"
        assert get_soul_name(Soul()) == "fallback"

    def test_no_attrs_returns_unknown(self):
        assert get_soul_name(object()) == "unknown"


# ── get_soul_traits ──────────────────────────────────────────────────────────


class TestGetSoulTraits:
    def test_from_soul_traits(self):
        class Soul:
            soul_traits = {"friendly": 0.8}
        assert get_soul_traits(Soul()) == {"friendly": 0.8}

    def test_from_personality_dict(self):
        class Soul:
            personality = {"creative": 0.9}
        assert get_soul_traits(Soul()) == {"creative": 0.9}

    def test_from_personality_object_with_to_dict(self):
        class Personality:
            def to_dict(self):
                return {"bold": 0.7}
        class Soul:
            personality = Personality()
        assert get_soul_traits(Soul()) == {"bold": 0.7}

    def test_from_personality_object_with_dict(self):
        class Personality:
            __dict__ = {"shy": 0.3}
        class Soul:
            personality = Personality()
        result = get_soul_traits(Soul())
        assert "shy" in result

    def test_empty_returns_dict(self):
        assert get_soul_traits(object()) == {}


# ── read_slo_json_header ─────────────────────────────────────────────────────


class TestReadSloJsonHeader:
    def _make_soul_file(self, tmp_path, meta: dict) -> Path:
        fp = tmp_path / "test.soul"
        json_bytes = json.dumps(meta).encode()
        with open(fp, "wb") as f:
            f.write(SOU_MAGIC)
            f.write(b"\x00\x00\x00\x00")  # 4 bytes padding
            f.write(struct.pack("<I", len(json_bytes)))
            f.write(json_bytes)
        return fp

    def test_reads_valid_header(self, tmp_path):
        meta = {"soul_name": "test", "version": 1}
        fp = self._make_soul_file(tmp_path, meta)
        result = read_slo_json_header(fp)
        assert result["soul_name"] == "test"
        assert result["version"] == 1

    def test_bad_magic_returns_empty(self, tmp_path):
        fp = tmp_path / "bad.soul"
        fp.write_bytes(b"BADFILE")
        assert read_slo_json_header(fp) == {}

    def test_nonexistent_file_returns_empty(self, tmp_path):
        assert read_slo_json_header(tmp_path / "nope.soul") == {}

    def test_corrupted_json_returns_empty(self, tmp_path):
        fp = tmp_path / "corrupt.soul"
        with open(fp, "wb") as f:
            f.write(SOU_MAGIC)
            f.write(b"\x00\x00\x00\x00")
            f.write(struct.pack("<I", 10))
            f.write(b"not valid json")
        assert read_slo_json_header(fp) == {}


# ── describe_checkpoint ──────────────────────────────────────────────────────


class TestDescribeCheckpoint:
    def test_with_dataset(self):
        ckpt = {"training_dataset": "shakespeare", "loss": 1.2}
        desc = describe_checkpoint(ckpt)
        assert "shakespeare" in desc
        assert "1.20" in desc
        assert "learned well" in desc

    def test_with_soul(self):
        ckpt = {"soul": "creative", "epochs": 3}
        desc = describe_checkpoint(ckpt)
        assert "creative" in desc
        assert "3 epochs" in desc

    def test_with_steps(self):
        ckpt = {"steps": 500}
        desc = describe_checkpoint(ckpt)
        assert "500 steps" in desc

    def test_loss_moderate(self):
        ckpt = {"loss": 2.5}
        desc = describe_checkpoint(ckpt)
        assert "moderate" in desc

    def test_loss_needs_training(self):
        ckpt = {"loss": 4.0}
        desc = describe_checkpoint(ckpt)
        assert "needs more training" in desc

    def test_with_traits(self):
        ckpt = {"traits": {"friendly": 0.9, "creative": 0.8}}
        desc = describe_checkpoint(ckpt)
        assert "Personality:" in desc

    def test_with_model_type(self):
        ckpt = {"model_type": "llama"}
        desc = describe_checkpoint(ckpt)
        assert "[llama]" in desc

    def test_slonet_type_not_shown(self):
        ckpt = {"model_type": "slonet"}
        desc = describe_checkpoint(ckpt)
        assert "[slonet]" not in desc

    def test_empty_checkpoint(self):
        desc = describe_checkpoint({})
        assert "A trained model" in desc

    def test_single_epoch(self):
        ckpt = {"epochs": 1}
        desc = describe_checkpoint(ckpt)
        assert "1 epoch" in desc
        assert "1 epochs" not in desc


# ── find_checkpoint ──────────────────────────────────────────────────────────


class TestFindCheckpoint:
    def test_finds_soul_file(self, tmp_path):
        ckpt_dir = tmp_path / "auto-training"
        ckpt_dir.mkdir()
        soul_file = ckpt_dir / "my-model.soul"
        soul_file.write_bytes(b"\x00" * 5000)

        with patch("domains.training.service.CHECKPOINTS_DIR", ckpt_dir):
            result = find_checkpoint("my-model")
        assert result is not None
        assert result.name == "my-model.soul"

    def test_finds_slo_file(self, tmp_path):
        ckpt_dir = tmp_path / "auto-training"
        ckpt_dir.mkdir()
        slo_file = ckpt_dir / "model.slo"
        slo_file.write_text("test")

        with patch("domains.training.service.CHECKPOINTS_DIR", ckpt_dir):
            result = find_checkpoint("model")
        assert result is not None
        assert result.name == "model.slo"

    def test_finds_with_full_extension(self, tmp_path):
        ckpt_dir = tmp_path / "auto-training"
        ckpt_dir.mkdir()
        soul_file = ckpt_dir / "test.soul"
        soul_file.write_bytes(b"\x00" * 5000)

        with patch("domains.training.service.CHECKPOINTS_DIR", ckpt_dir):
            result = find_checkpoint("test.soul")
        assert result is not None

    def test_not_found(self, tmp_path):
        ckpt_dir = tmp_path / "auto-training"
        ckpt_dir.mkdir()

        with patch("domains.training.service.CHECKPOINTS_DIR", ckpt_dir):
            result = find_checkpoint("nonexistent")
        assert result is None

    def test_turbo_dir_searched(self, tmp_path):
        turbo_dir = tmp_path / "turbo"
        turbo_dir.mkdir()
        soul_file = turbo_dir / "turbo-model.soul"
        soul_file.write_bytes(b"\x00" * 5000)

        with patch("domains.training.service.CHECKPOINTS_DIR", tmp_path / "empty"):
            with patch("domains.training.service.TURBO_DIR", turbo_dir):
                result = find_checkpoint("turbo-model")
        assert result is not None


# ── log_experiment_metric ────────────────────────────────────────────────────


class TestLogExperimentMetric:
    def test_creates_metric_file(self, tmp_path):
        with patch("domains.training.helpers.REPO_ROOT", tmp_path):
            log_experiment_metric("exp-1", "loss", 0.5, step=10)

        metrics_file = tmp_path / "data" / "experiments" / "exp-1_metrics.jsonl"
        assert metrics_file.exists()
        line = json.loads(metrics_file.read_text().strip())
        assert line["metric"] == "loss"
        assert line["value"] == 0.5
        assert line["step"] == 10

    def test_appends_multiple_entries(self, tmp_path):
        with patch("domains.training.helpers.REPO_ROOT", tmp_path):
            log_experiment_metric("exp-1", "loss", 0.5)
            log_experiment_metric("exp-1", "accuracy", 0.9)

        metrics_file = tmp_path / "data" / "experiments" / "exp-1_metrics.jsonl"
        lines = metrics_file.read_text().strip().split("\n")
        assert len(lines) == 2


# ── log_experiment_param ────────────────────────────────────────────────────


class TestLogExperimentParam:
    def test_creates_param_file(self, tmp_path):
        with patch("domains.training.helpers.REPO_ROOT", tmp_path):
            log_experiment_param("exp-1", "learning_rate", 0.001)

        params_file = tmp_path / "data" / "experiments" / "exp-1_params.jsonl"
        assert params_file.exists()
        line = json.loads(params_file.read_text().strip())
        assert line["param"] == "learning_rate"
        assert line["value"] == 0.001


# ── Constants ────────────────────────────────────────────────────────────────


class TestConstants:
    def test_valid_ckpt_name_pattern(self):
        assert VALID_CKPT_NAME.match("my-model")
        assert VALID_CKPT_NAME.match("model_v2.soul")
        assert not VALID_CKPT_NAME.match("bad name")
        assert not VALID_CKPT_NAME.match("path/traversal")

    def test_soul_magic(self):
        assert SOU_MAGIC == b"SOUL"
        assert len(SOU_MAGIC) == 4
