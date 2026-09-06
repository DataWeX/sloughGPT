"""Tests for training.helpers — pure functions for training operations."""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from domains.training.helpers import (
    _finite_payload,
    log_experiment_metric,
    log_experiment_param,
    parse_subtitle_text,
    resolve_dataset_path,
    build_soul_prompt,
    get_soul_name,
    get_soul_traits,
    read_slo_json_header,
    describe_checkpoint,
    cross_entropy_loss,
)
from domains.training.state import SOU_MAGIC


# ── _finite_payload ───────────────────────────────────────────────────────


class TestFinitePayload:

    def test_float_normal(self):
        assert _finite_payload(1.5) == 1.5

    def test_float_nan(self):
        assert _finite_payload(float("nan")) is None

    def test_float_inf(self):
        assert _finite_payload(float("inf")) is None

    def test_float_neg_inf(self):
        assert _finite_payload(float("-inf")) is None

    def test_dict(self):
        result = _finite_payload({"a": 1.0, "b": float("nan")})
        assert result == {"a": 1.0, "b": None}

    def test_list(self):
        result = _finite_payload([1.0, float("inf"), 2.0])
        assert result == [1.0, None, 2.0]

    def test_nested(self):
        result = _finite_payload({"a": [1.0, float("nan")]})
        assert result == {"a": [1.0, None]}

    def test_int_passthrough(self):
        assert _finite_payload(42) == 42

    def test_string_passthrough(self):
        assert _finite_payload("hello") == "hello"

    def test_dataclass(self):
        @dataclass
        class Point:
            x: float
            y: float
        p = Point(1.0, float("nan"))
        result = _finite_payload(p)
        assert result == {"x": 1.0, "y": None}


# ── log_experiment_metric ─────────────────────────────────────────────────


class TestLogExperimentMetric:

    def test_writes_metric_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("domains.training.helpers.REPO_ROOT", tmp_path)
        log_experiment_metric("exp1", "loss", 0.5, step=10)
        metrics_file = tmp_path / "data" / "experiments" / "exp1_metrics.jsonl"
        assert metrics_file.exists()
        line = json.loads(metrics_file.read_text().strip())
        assert line["experiment_id"] == "exp1"
        assert line["metric"] == "loss"
        assert line["value"] == 0.5
        assert line["step"] == 10


# ── log_experiment_param ──────────────────────────────────────────────────


class TestLogExperimentParam:

    def test_writes_param_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("domains.training.helpers.REPO_ROOT", tmp_path)
        log_experiment_param("exp1", "lr", 0.001)
        params_file = tmp_path / "data" / "experiments" / "exp1_params.jsonl"
        assert params_file.exists()
        line = json.loads(params_file.read_text().strip())
        assert line["param"] == "lr"
        assert line["value"] == 0.001


# ── parse_subtitle_text ──────────────────────────────────────────────────


class TestParseSubtitleText:

    def test_plain_text(self):
        text = "Hello world\nThis is a test\n"
        result = parse_subtitle_text(text)
        assert result == ["Hello world", "This is a test"]

    def test_plain_text_filters_short(self):
        text = "Hi\nHello world\nOk\n"
        result = parse_subtitle_text(text)
        assert "Hi" not in result
        assert "Hello world" in result

    def test_srt_format(self):
        text = "00:00:01,000 --> 00:00:02,000\nHello world\n\n00:00:03,000 --> 00:00:04,000\nGoodbye\n"
        result = parse_subtitle_text(text)
        assert "Hello world" in result
        assert "Goodbye" in result

    def test_vtt_format(self):
        text = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nHello\n"
        result = parse_subtitle_text(text)
        assert "Hello" in result

    def test_srt_skips_indices(self):
        text = "00:00:01,000 --> 00:00:02,000\nFirst\n\n1\n\n00:00:03,000 --> 00:00:04,000\nSecond\n"
        result = parse_subtitle_text(text)
        assert "1" not in result

    def test_empty(self):
        assert parse_subtitle_text("") == []


# ── resolve_dataset_path ─────────────────────────────────────────────────


class TestResolveDatasetPath:

    def test_invalid_id(self):
        with pytest.raises(ValueError, match="Invalid dataset ID"):
            resolve_dataset_path("../etc/passwd")

    def test_nonexistent(self):
        assert resolve_dataset_path("nonexistent_dataset_xyz") == ""


# ── build_soul_prompt ────────────────────────────────────────────────────


class TestBuildSoulPrompt:

    def test_known_souls(self):
        assert "helpful" in build_soul_prompt("assistant")
        assert "creative" in build_soul_prompt("creative")
        assert "analyst" in build_soul_prompt("analyst")
        assert "coder" in build_soul_prompt("coder")
        assert "teacher" in build_soul_prompt("teacher")

    def test_unknown_defaults_to_assistant(self):
        prompt = build_soul_prompt("unknown_soul")
        assert "helpful" in prompt


# ── get_soul_name ─────────────────────────────────────────────────────────


class TestGetSoulName:

    def test_from_name_attr(self):
        class Soul:
            name = "alice"
        assert get_soul_name(Soul()) == "alice"

    def test_from_soul_name_attr(self):
        class Soul:
            soul_name = "bob"
        assert get_soul_name(Soul()) == "bob"

    def test_unknown(self):
        class Soul:
            pass
        assert get_soul_name(Soul()) == "unknown"


# ── get_soul_traits ───────────────────────────────────────────────────────


class TestGetSoulTraits:

    def test_from_soul_traits(self):
        class Soul:
            soul_traits = {"friendly": 0.8}
        assert get_soul_traits(Soul()) == {"friendly": 0.8}

    def test_from_personality_dict(self):
        class Soul:
            personality = {"curious": 0.9}
        assert get_soul_traits(Soul()) == {"curious": 0.9}

    def test_from_personality_object(self):
        class Personality:
            def to_dict(self):
                return {"brave": 0.7}
        class Soul:
            personality = Personality()
        assert get_soul_traits(Soul()) == {"brave": 0.7}

    def test_empty(self):
        class Soul:
            pass
        assert get_soul_traits(Soul()) == {}


# ── read_slo_json_header ─────────────────────────────────────────────────


class TestReadSloJsonHeader:

    def test_valid_header(self, tmp_path):
        header = json.dumps({"model": "test"}).encode()
        padding = b"\x00" * 4
        content = b"some model data"
        path = tmp_path / "model.soul"
        with open(path, "wb") as f:
            f.write(SOU_MAGIC)
            f.write(padding)
            f.write(struct.pack("<I", len(header)))
            f.write(header)
            f.write(content)
        result = read_slo_json_header(path)
        assert result["model"] == "test"

    def test_bad_magic(self, tmp_path):
        path = tmp_path / "bad.bin"
        path.write_bytes(b"XXXX")
        result = read_slo_json_header(path)
        assert result == {}

    def test_nonexistent(self):
        result = read_slo_json_header(Path("/nonexistent"))
        assert result == {}


# ── describe_checkpoint ──────────────────────────────────────────────────


class TestDescribeCheckpoint:

    def test_with_dataset(self):
        desc = describe_checkpoint({"training_dataset": "mydata"})
        assert "mydata" in desc

    def test_with_soul(self):
        desc = describe_checkpoint({"soul": "alice"})
        assert "alice" in desc

    def test_with_epochs(self):
        desc = describe_checkpoint({"epochs": 5})
        assert "5 epochs" in desc

    def test_with_steps(self):
        desc = describe_checkpoint({"steps": 100})
        assert "100 steps" in desc

    def test_loss_low(self):
        desc = describe_checkpoint({"loss": 1.0})
        assert "learned well" in desc

    def test_loss_medium(self):
        desc = describe_checkpoint({"loss": 2.0})
        assert "moderate" in desc

    def test_loss_high(self):
        desc = describe_checkpoint({"loss": 4.0})
        assert "needs more training" in desc

    def test_with_traits(self):
        desc = describe_checkpoint({"traits": {"friendly": 0.8, "curious": 0.9}})
        assert "Personality" in desc

    def test_with_model_type(self):
        desc = describe_checkpoint({"model_type": "transformer"})
        assert "transformer" in desc

    def test_unknown_model_type_hidden(self):
        desc = describe_checkpoint({"model_type": "slonet"})
        assert "slonet" not in desc

    def test_empty(self):
        desc = describe_checkpoint({})
        assert "A trained model" in desc


# ── cross_entropy_loss ───────────────────────────────────────────────────


class TestCrossEntropyLoss:

    def test_perfect_prediction(self):
        logits = np.array([[10.0, 0.0, 0.0]])
        targets = np.array([0])
        loss = cross_entropy_loss(logits, targets)
        assert loss < 0.1

    def test_worse_prediction(self):
        logits = np.array([[0.0, 0.0, 10.0]])
        targets = np.array([0])
        loss = cross_entropy_loss(logits, targets)
        assert loss > 5.0

    def test_batch(self):
        logits = np.array([[10.0, 0.0], [0.0, 10.0]])
        targets = np.array([0, 1])
        loss = cross_entropy_loss(logits, targets)
        assert loss < 0.1
