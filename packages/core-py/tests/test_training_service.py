"""Tests for domains.training.service — pure logic, no heavy mocking.

Covers: _finite_payload, log_experiment_metric, log_experiment_param,
parse_subtitle_text, resolve_dataset_path, build_soul_prompt, get_soul_name,
get_soul_traits, describe_checkpoint, find_checkpoint, TrainingState,
get_state, get_turbo_state, get_turbo_lock, get_turbo_status,
start_from_sessions_training, cleanup_stream_state.
"""
from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from domains.training.service import (
    _finite_payload,
    build_soul_prompt,
    cleanup_stream_state,
    describe_checkpoint,
    find_checkpoint,
    get_soul_name,
    get_soul_traits,
    get_state,
    get_turbo_lock,
    get_turbo_state,
    get_turbo_status,
    log_experiment_metric,
    log_experiment_param,
    parse_subtitle_text,
    resolve_dataset_path,
    start_from_sessions_training,
    TrainingState,
)


# ---------------------------------------------------------------------------
# _finite_payload
# ---------------------------------------------------------------------------

class TestFinitePayload:
    def test_integer_passthrough(self):
        assert _finite_payload(42) == 42

    def test_string_passthrough(self):
        assert _finite_payload("hello") == "hello"

    def test_none_passthrough(self):
        assert _finite_payload(None) is None

    def test_finite_float_passthrough(self):
        assert _finite_payload(3.14) == 3.14

    def test_inf_replaced_with_none(self):
        assert _finite_payload(float("inf")) is None

    def test_neg_inf_replaced_with_none(self):
        assert _finite_payload(float("-inf")) is None

    def test_nan_replaced_with_none(self):
        assert _finite_payload(float("nan")) is None

    def test_dict_with_inf(self):
        result = _finite_payload({"loss": float("inf"), "lr": 0.01})
        assert result == {"loss": None, "lr": 0.01}

    def test_dict_with_nan(self):
        result = _finite_payload({"val": float("nan")})
        assert result == {"val": None}

    def test_nested_dict(self):
        result = _finite_payload({"a": {"b": float("inf"), "c": 1}})
        assert result == {"a": {"b": None, "c": 1}}

    def test_list_with_inf(self):
        result = _finite_payload([1.0, float("inf"), 2.0])
        assert result == [1.0, None, 2.0]

    def test_tuple_with_inf(self):
        result = _finite_payload((1.0, float("nan")))
        assert result == [1.0, None]

    def test_mixed_nested_structures(self):
        result = _finite_payload({"data": [1, float("nan"), {"x": float("inf")}]})
        assert result == {"data": [1, None, {"x": None}]}

    def test_bool_passthrough(self):
        assert _finite_payload(True) is True
        assert _finite_payload(False) is False

    def test_zero_float_finite(self):
        assert _finite_payload(0.0) == 0.0

    def test_negative_float_finite(self):
        assert _finite_payload(-1.5) == -1.5

    def test_dataclass_converted(self):
        @dataclass
        class Params:
            lr: float = 0.01
            epochs: int = 5
            bad_val: float = float("nan")

        result = _finite_payload(Params())
        assert result == {"lr": 0.01, "epochs": 5, "bad_val": None}

    def test_nested_dataclass(self):
        from dataclasses import field

        @dataclass
        class Inner:
            val: float = float("inf")

        @dataclass
        class Outer:
            inner: Inner = field(default_factory=lambda: Inner())
            name: str = "test"

        result = _finite_payload(Outer())
        assert result == {"inner": {"val": None}, "name": "test"}

    def test_mixed_list_inf_and_valid(self):
        result = _finite_payload([float("inf"), 1, None, "x", float("nan")])
        assert result == [None, 1, None, "x", None]

    def test_boolean_in_dict(self):
        result = _finite_payload({"flag": True, "score": float("nan")})
        assert result == {"flag": True, "score": None}


# ---------------------------------------------------------------------------
# log_experiment_metric
# ---------------------------------------------------------------------------

class TestLogExperimentMetric:
    def _patch_root(self, tmp_path):
        import domains.training.service as svc
        return patch.object(svc, "REPO_ROOT", tmp_path)

    def test_writes_metric_file(self, tmp_path):
        with self._patch_root(tmp_path):
            log_experiment_metric("exp1", "loss", 0.5, step=10)
        metric_file = tmp_path / "data" / "experiments" / "exp1_metrics.jsonl"
        assert metric_file.exists()
        line = metric_file.read_text().strip()
        entry = json.loads(line)
        assert entry["experiment_id"] == "exp1"
        assert entry["metric"] == "loss"
        assert entry["value"] == 0.5
        assert entry["step"] == 10
        assert "timestamp" in entry

    def test_appends_multiple_entries(self, tmp_path):
        with self._patch_root(tmp_path):
            log_experiment_metric("exp1", "loss", 1.0, step=1)
            log_experiment_metric("exp1", "loss", 0.8, step=2)
            log_experiment_metric("exp1", "accuracy", 0.9, step=2)
        metric_file = tmp_path / "data" / "experiments" / "exp1_metrics.jsonl"
        lines = metric_file.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_default_step_zero(self, tmp_path):
        with self._patch_root(tmp_path):
            log_experiment_metric("exp1", "val_loss", 2.0)
        metric_file = tmp_path / "data" / "experiments" / "exp1_metrics.jsonl"
        entry = json.loads(metric_file.read_text().strip())
        assert entry["step"] == 0

    def test_creates_experiment_dir(self, tmp_path):
        with self._patch_root(tmp_path):
            log_experiment_metric("new_exp", "loss", 0.1)
        assert (tmp_path / "data" / "experiments").is_dir()

    def test_different_experiments_separate_files(self, tmp_path):
        with self._patch_root(tmp_path):
            log_experiment_metric("exp_a", "loss", 1.0)
            log_experiment_metric("exp_b", "loss", 2.0)
        assert (tmp_path / "data" / "experiments" / "exp_a_metrics.jsonl").exists()
        assert (tmp_path / "data" / "experiments" / "exp_b_metrics.jsonl").exists()

    def test_numeric_string_value(self, tmp_path):
        with self._patch_root(tmp_path):
            log_experiment_metric("exp1", "score", "0.95")
        metric_file = tmp_path / "data" / "experiments" / "exp1_metrics.jsonl"
        entry = json.loads(metric_file.read_text().strip())
        assert entry["value"] == "0.95"


# ---------------------------------------------------------------------------
# log_experiment_param
# ---------------------------------------------------------------------------

class TestLogExperimentParam:
    def _patch_root(self, tmp_path):
        import domains.training.service as svc
        return patch.object(svc, "REPO_ROOT", tmp_path)

    def test_writes_param_file(self, tmp_path):
        with self._patch_root(tmp_path):
            log_experiment_param("exp1", "lr", 0.001)
        param_file = tmp_path / "data" / "experiments" / "exp1_params.jsonl"
        assert param_file.exists()
        entry = json.loads(param_file.read_text().strip())
        assert entry["experiment_id"] == "exp1"
        assert entry["param"] == "lr"
        assert entry["value"] == 0.001
        assert "timestamp" in entry

    def test_appends_multiple_params(self, tmp_path):
        with self._patch_root(tmp_path):
            log_experiment_param("exp1", "lr", 0.001)
            log_experiment_param("exp1", "epochs", 10)
        param_file = tmp_path / "data" / "experiments" / "exp1_params.jsonl"
        lines = param_file.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_dict_value(self, tmp_path):
        with self._patch_root(tmp_path):
            log_experiment_param("exp1", "model_config", {"layers": 4})
        param_file = tmp_path / "data" / "experiments" / "exp1_params.jsonl"
        entry = json.loads(param_file.read_text().strip())
        assert entry["value"] == {"layers": 4}

    def test_creates_experiment_dir(self, tmp_path):
        with self._patch_root(tmp_path):
            log_experiment_param("exp_new", "lr", 0.01)
        assert (tmp_path / "data" / "experiments").is_dir()


# ---------------------------------------------------------------------------
# parse_subtitle_text
# ---------------------------------------------------------------------------

class TestParseSubtitleText:
    def test_plain_text_single_line(self):
        assert parse_subtitle_text("hello world") == ["hello world"]

    def test_plain_text_multiple_lines(self):
        result = parse_subtitle_text("line one\nline two\nline three")
        assert result == ["line one", "line two", "line three"]

    def test_empty_lines_filtered(self):
        result = parse_subtitle_text("line1\n\n\nline2")
        assert result == ["line1", "line2"]

    def test_short_lines_filtered(self):
        result = parse_subtitle_text("ok\na\nlong enough line")
        assert "ok" not in result
        assert "a" not in result
        assert "long enough line" in result

    def test_exactly_3_chars_kept(self):
        result = parse_subtitle_text("abc\nde")
        assert "abc" in result

    def test_srt_format(self):
        srt = (
            "1\n"
            "00:00:01,000 --> 00:00:03,000\n"
            "Hello world\n"
            "\n"
            "2\n"
            "00:00:04,000 --> 00:00:06,000\n"
            "Goodbye world\n"
        )
        result = parse_subtitle_text(srt)
        assert "Hello world" in result
        assert "Goodbye world" in result
        assert not any("-->" in line for line in result)

    def test_srt_index_numbers_filtered(self):
        srt = "1\n00:00:00,000 --> 00:00:01,000\nText\n"
        result = parse_subtitle_text(srt)
        assert "1" not in result
        assert "Text" in result

    def test_vtt_format(self):
        vtt = (
            "WEBVTT\n"
            "\n"
            "00:00.000 --> 00:02.000\n"
            "First line\n"
            "\n"
            "00:03.000 --> 00:05.000\n"
            "Second line\n"
        )
        result = parse_subtitle_text(vtt)
        assert "First line" in result
        assert "Second line" in result

    def test_webvtt_header_filtered(self):
        vtt = "WEBVTT\n\n00:00.000 --> 00:01.000\nText\n"
        result = parse_subtitle_text(vtt)
        assert "WEBVTT" not in result

    def test_bracket_lines_not_filtered(self):
        result = parse_subtitle_text("[music]\nActual text\n[silence]")
        assert "[music]" in result
        assert "Actual text" in result
        assert "[silence]" in result

    def test_empty_string(self):
        assert parse_subtitle_text("") == []

    def test_whitespace_only_lines_filtered(self):
        result = parse_subtitle_text("  \n  \n  ")
        assert result == []

    def test_srt_with_multiple_voice_tags(self):
        srt = (
            "1\n"
            "00:00:00,000 --> 00:00:01,000\n"
            "Hello\n"
            "2\n"
            "00:00:01,000 --> 00:00:02,000\n"
            "World\n"
        )
        result = parse_subtitle_text(srt)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# resolve_dataset_path
# ---------------------------------------------------------------------------

class TestResolveDatasetPath:
    def _patch_root(self, tmp_path):
        import domains.training.service as svc
        return patch.object(svc, "REPO_ROOT", tmp_path)

    def test_nonexistent_dataset(self, tmp_path):
        with self._patch_root(tmp_path):
            result = resolve_dataset_path("nonexistent_ds")
        assert result == ""

    def test_invalid_id_special_chars(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid dataset ID"):
            resolve_dataset_path("bad id!")

    def test_invalid_id_dotdot(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid dataset ID"):
            resolve_dataset_path("../../../etc")

    def test_with_corpus_jsonl(self, tmp_path):
        ds_dir = tmp_path / "datasets" / "test_ds"
        ds_dir.mkdir(parents=True)
        (ds_dir / "corpus.jsonl").write_text('{"text":"hello"}')
        with self._patch_root(tmp_path):
            result = resolve_dataset_path("test_ds")
        assert result.endswith("corpus.jsonl")

    def test_with_input_txt(self, tmp_path):
        ds_dir = tmp_path / "datasets" / "test_ds2"
        ds_dir.mkdir(parents=True)
        (ds_dir / "input.txt").write_text("hello")
        with self._patch_root(tmp_path):
            result = resolve_dataset_path("test_ds2")
        assert result.endswith("input.txt")

    def test_with_train_txt(self, tmp_path):
        ds_dir = tmp_path / "datasets" / "test_ds3"
        ds_dir.mkdir(parents=True)
        (ds_dir / "train.txt").write_text("data")
        with self._patch_root(tmp_path):
            result = resolve_dataset_path("test_ds3")
        assert result.endswith("train.txt")

    def test_with_text_txt(self, tmp_path):
        ds_dir = tmp_path / "datasets" / "test_ds4"
        ds_dir.mkdir(parents=True)
        (ds_dir / "text.txt").write_text("content")
        with self._patch_root(tmp_path):
            result = resolve_dataset_path("test_ds4")
        assert result.endswith("text.txt")

    def test_priority_order(self, tmp_path):
        ds_dir = tmp_path / "datasets" / "test_ds5"
        ds_dir.mkdir(parents=True)
        (ds_dir / "corpus.jsonl").write_text("a")
        (ds_dir / "input.txt").write_text("b")
        with self._patch_root(tmp_path):
            result = resolve_dataset_path("test_ds5")
        assert result.endswith("corpus.jsonl")

    def test_any_txt_fallback(self, tmp_path):
        ds_dir = tmp_path / "datasets" / "test_ds6"
        ds_dir.mkdir(parents=True)
        (ds_dir / "custom.txt").write_text("data")
        with self._patch_root(tmp_path):
            result = resolve_dataset_path("test_ds6")
        assert result.endswith("custom.txt")

    def test_empty_dir(self, tmp_path):
        ds_dir = tmp_path / "datasets" / "empty_ds"
        ds_dir.mkdir(parents=True)
        with self._patch_root(tmp_path):
            result = resolve_dataset_path("empty_ds")
        assert result == ""

    def test_invalid_id_underscore(self, tmp_path):
        result = resolve_dataset_path("valid_id-123")
        assert result == ""

    def test_invalid_id_slash(self, tmp_path):
        with pytest.raises(ValueError):
            resolve_dataset_path("bad/path")


# ---------------------------------------------------------------------------
# build_soul_prompt
# ---------------------------------------------------------------------------

class TestBuildSoulPrompt:
    def test_assistant(self):
        prompt = build_soul_prompt("assistant")
        assert "helpful" in prompt.lower()

    def test_creative(self):
        prompt = build_soul_prompt("creative")
        assert "creative" in prompt.lower()

    def test_analyst(self):
        prompt = build_soul_prompt("analyst")
        assert "analyst" in prompt.lower() or "precise" in prompt.lower()

    def test_coder(self):
        prompt = build_soul_prompt("coder")
        assert "coder" in prompt.lower() or "code" in prompt.lower()

    def test_teacher(self):
        prompt = build_soul_prompt("teacher")
        assert "teacher" in prompt.lower() or "explain" in prompt.lower()

    def test_unknown_defaults_to_assistant(self):
        prompt = build_soul_prompt("nonexistent_soul")
        assert "helpful" in prompt.lower()

    def test_empty_string_defaults_to_assistant(self):
        prompt = build_soul_prompt("")
        assert "helpful" in prompt.lower()

    def test_returns_string(self):
        for name in ("assistant", "creative", "analyst", "coder", "teacher", "nope"):
            assert isinstance(build_soul_prompt(name), str)


# ---------------------------------------------------------------------------
# get_soul_name
# ---------------------------------------------------------------------------

class TestGetSoulName:
    def test_name_attr(self):
        soul = SimpleNamespace(name="Alice")
        assert get_soul_name(soul) == "Alice"

    def test_soul_name_attr(self):
        soul = SimpleNamespace(soul_name="Bob")
        assert get_soul_name(soul) == "Bob"

    def test_name_takes_priority(self):
        soul = SimpleNamespace(name="Alice", soul_name="Bob")
        assert get_soul_name(soul) == "Alice"

    def test_empty_name_fallback_to_soul_name(self):
        soul = SimpleNamespace(name="", soul_name="fallback")
        assert get_soul_name(soul) == "fallback"

    def test_no_attrs_returns_unknown(self):
        soul = SimpleNamespace()
        assert get_soul_name(soul) == "unknown"

    def test_name_none_fallback(self):
        soul = SimpleNamespace(name=None, soul_name="fallback")
        assert get_soul_name(soul) == "fallback"

    def test_dict_object(self):
        class DictSoul:
            def __init__(self):
                self.name = "DictSoul"
        soul = DictSoul()
        assert get_soul_name(soul) == "DictSoul"


# ---------------------------------------------------------------------------
# get_soul_traits
# ---------------------------------------------------------------------------

class TestGetSoulTraits:
    def test_soul_traits_attr(self):
        soul = SimpleNamespace(soul_traits={"warmth": 0.8})
        assert get_soul_traits(soul) == {"warmth": 0.8}

    def test_personality_dict(self):
        soul = SimpleNamespace(personality={"warmth": 0.7})
        assert get_soul_traits(soul) == {"warmth": 0.7}

    def test_personality_to_dict(self):
        soul = SimpleNamespace(personality=SimpleNamespace(to_dict=lambda: {"warmth": 0.9}))
        assert get_soul_traits(soul) == {"warmth": 0.9}

    def test_personality_has_dict(self):
        class FakePersonality:
            def __init__(self):
                self.warmth = 0.5
                self.humor = 0.3
        soul = SimpleNamespace(personality=FakePersonality())
        traits = get_soul_traits(soul)
        assert traits["warmth"] == 0.5
        assert traits["humor"] == 0.3

    def test_personality_is_iterable(self):
        soul = SimpleNamespace(personality=[("warmth", 0.8)])
        assert get_soul_traits(soul) == {"warmth": 0.8}

    def test_no_traits_returns_empty_dict(self):
        soul = SimpleNamespace()
        assert get_soul_traits(soul) == {}

    def test_soul_traits_takes_priority_over_personality(self):
        soul = SimpleNamespace(
            soul_traits={"primary": 1.0},
            personality={"secondary": 0.5},
        )
        assert get_soul_traits(soul) == {"primary": 1.0}

    def test_empty_soul_traits_falls_to_personality(self):
        soul = SimpleNamespace(soul_traits=None, personality={"a": 1})
        assert get_soul_traits(soul) == {"a": 1}


# ---------------------------------------------------------------------------
# describe_checkpoint
# ---------------------------------------------------------------------------

class TestDescribeCheckpoint:
    def test_minimal_checkpoint(self):
        desc = describe_checkpoint({})
        assert "A trained model" in desc

    def test_with_dataset(self):
        desc = describe_checkpoint({"training_dataset": "shakespeare"})
        assert "shakespeare" in desc.lower()

    def test_with_soul(self):
        desc = describe_checkpoint({"soul": "assistant"})
        assert "Soul: assistant" in desc

    def test_unknown_soul_fallback(self):
        desc = describe_checkpoint({"soul": "unknown"})
        assert "A trained model" in desc

    def test_with_epochs(self):
        desc = describe_checkpoint({"epochs": 5})
        assert "5 epochs" in desc

    def test_single_epoch(self):
        desc = describe_checkpoint({"epochs": 1})
        assert "1 epoch" in desc
        assert "epochs" not in desc

    def test_epochs_trained_fallback(self):
        desc = describe_checkpoint({"epochs_trained": 3})
        assert "3 epochs" in desc

    def test_with_steps(self):
        desc = describe_checkpoint({"steps": 100})
        assert "100 steps" in desc

    def test_low_loss(self):
        desc = describe_checkpoint({"loss": 0.8})
        assert "learned well" in desc

    def test_medium_loss(self):
        desc = describe_checkpoint({"loss": 2.5})
        assert "moderate" in desc

    def test_high_loss(self):
        desc = describe_checkpoint({"loss": 4.0})
        assert "needs more training" in desc

    def test_loss_exactly_1_5(self):
        desc = describe_checkpoint({"loss": 1.5})
        assert "moderate" in desc

    def test_loss_exactly_3_0(self):
        desc = describe_checkpoint({"loss": 3.0})
        assert "needs more training" in desc

    def test_with_traits(self):
        desc = describe_checkpoint({"traits": {"warmth": 0.8, "humor": 0.5}})
        assert "Personality" in desc
        assert "warmth" in desc

    def test_traits_limited_to_3(self):
        desc = describe_checkpoint({
            "traits": {"aaa": 1, "bbb": 2, "ccc": 3, "ddd": 4}
        })
        assert "aaa" in desc
        assert "Personality: aaa, bbb, ccc" in desc

    def test_with_model_type(self):
        desc = describe_checkpoint({"model_type": "transformer"})
        assert "transformer" in desc

    def test_slonet_model_type_not_shown(self):
        desc = describe_checkpoint({"model_type": "slonet"})
        assert "slonet" not in desc

    def test_unknown_model_type_not_shown(self):
        desc = describe_checkpoint({"model_type": "unknown"})
        assert "unknown" not in desc

    def test_full_checkpoint(self):
        desc = describe_checkpoint({
            "soul": "coder",
            "training_dataset": "code",
            "loss": 1.0,
            "epochs": 10,
            "traits": {"skill": 0.9},
            "model_type": "gpt2",
        })
        assert "code" in desc.lower()
        assert "10 epochs" in desc
        assert "1.00" in desc
        assert "learned well" in desc
        assert "gpt2" in desc

    def test_loses_priority_over_dataset(self):
        desc = describe_checkpoint({
            "soul": "assistant",
            "training_dataset": "wiki",
        })
        assert "wiki" in desc.lower()

    def test_epochs_priority_over_steps(self):
        desc = describe_checkpoint({"epochs": 3, "steps": 100})
        assert "3 epochs" in desc
        assert "100 steps" not in desc

    def test_no_loss_no_steps(self):
        desc = describe_checkpoint({"soul": "x"})
        assert "Soul: x" in desc


# ---------------------------------------------------------------------------
# find_checkpoint
# ---------------------------------------------------------------------------

class TestFindCheckpoint:
    def _patch_dirs(self, tmp_path):
        import domains.training.service as svc
        ckpt_dir = tmp_path / "models" / "auto-training"
        turbo_dir = tmp_path / "models" / "turbo-trained"
        ckpt_dir.mkdir(parents=True)
        turbo_dir.mkdir(parents=True)
        return patch.object(svc, "CHECKPOINTS_DIR", ckpt_dir), \
               patch.object(svc, "TURBO_DIR", turbo_dir)

    def test_finds_soul_in_ckpt_dir(self, tmp_path):
        p1, p2 = self._patch_dirs(tmp_path)
        ckpt_dir = tmp_path / "models" / "auto-training"
        (ckpt_dir / "my_model.soul").write_text("data")
        with p1, p2:
            result = find_checkpoint("my_model")
        assert result is not None
        assert result.name == "my_model.soul"

    def test_finds_slo_in_ckpt_dir(self, tmp_path):
        p1, p2 = self._patch_dirs(tmp_path)
        ckpt_dir = tmp_path / "models" / "auto-training"
        (ckpt_dir / "model.slo").write_text("data")
        with p1, p2:
            result = find_checkpoint("model")
        assert result is not None
        assert result.name == "model.slo"

    def test_finds_in_turbo_dir(self, tmp_path):
        p1, p2 = self._patch_dirs(tmp_path)
        turbo_dir = tmp_path / "models" / "turbo-trained"
        (turbo_dir / "turbo_ckpt.soul").write_text("data")
        with p1, p2:
            result = find_checkpoint("turbo_ckpt")
        assert result is not None

    def test_not_found(self, tmp_path):
        p1, p2 = self._patch_dirs(tmp_path)
        with p1, p2:
            result = find_checkpoint("nonexistent")
        assert result is None

    def test_direct_soul_filename(self, tmp_path):
        p1, p2 = self._patch_dirs(tmp_path)
        ckpt_dir = tmp_path / "models" / "auto-training"
        (ckpt_dir / "test.soul").write_text("data")
        with p1, p2:
            result = find_checkpoint("test.soul")
        assert result is not None

    def test_direct_slo_filename(self, tmp_path):
        p1, p2 = self._patch_dirs(tmp_path)
        turbo_dir = tmp_path / "models" / "turbo-trained"
        (turbo_dir / "fast.slo").write_text("data")
        with p1, p2:
            result = find_checkpoint("fast.slo")
        assert result is not None

    def test_prefers_ckpt_dir_over_turbo(self, tmp_path):
        p1, p2 = self._patch_dirs(tmp_path)
        ckpt_dir = tmp_path / "models" / "auto-training"
        turbo_dir = tmp_path / "models" / "turbo-trained"
        (ckpt_dir / "dup.soul").write_text("ckpt")
        (turbo_dir / "dup.soul").write_text("turbo")
        with p1, p2:
            result = find_checkpoint("dup")
        assert result is not None
        assert "auto-training" in str(result)


# ---------------------------------------------------------------------------
# TrainingState dataclass
# ---------------------------------------------------------------------------

class TestTrainingState:
    def test_defaults(self):
        s = TrainingState()
        assert s.running is False
        assert s.config == {}
        assert s.student_net is None
        assert s.student_tokenizer is None
        assert s.complete_enqueued is False

    def test_post_init_sets_config_to_dict(self):
        s = TrainingState(config=None)
        assert s.config == {}

    def test_post_init_preserves_config(self):
        cfg = {"method": "test"}
        s = TrainingState(config=cfg)
        assert s.config is cfg

    def test_running_can_be_set(self):
        s = TrainingState(running=True)
        assert s.running is True

    def test_custom_config(self):
        s = TrainingState(config={"a": 1, "b": 2})
        assert s.config["a"] == 1
        assert s.config["b"] == 2


# ---------------------------------------------------------------------------
# get_state / get_turbo_state / get_turbo_lock
# ---------------------------------------------------------------------------

class TestStateAccessors:
    def test_get_state_returns_training_state(self):
        state = get_state()
        assert isinstance(state, TrainingState)

    def test_get_state_is_singleton(self):
        assert get_state() is get_state()

    def test_get_turbo_state_returns_dict(self):
        ts = get_turbo_state()
        assert isinstance(ts, dict)

    def test_get_turbo_state_is_singleton(self):
        assert get_turbo_state() is get_turbo_state()

    def test_get_turbo_state_has_expected_keys(self):
        ts = get_turbo_state()
        assert "status" in ts
        assert "progress" in ts
        assert "loss" in ts

    def test_get_turbo_lock_returns_lock(self):
        lock = get_turbo_lock()
        assert isinstance(lock, type(threading.Lock()))

    def test_get_turbo_lock_is_singleton(self):
        assert get_turbo_lock() is get_turbo_lock()


# ---------------------------------------------------------------------------
# get_turbo_status
# ---------------------------------------------------------------------------

class TestGetTurboStatus:
    def _patch_all(self):
        import domains.training.service as svc
        return svc

    def test_idle_status(self):
        svc = self._patch_all()
        with patch.dict(svc._turbo_state, {"status": "idle", "last_heartbeat": 0.0}):
            result = get_turbo_status()
        assert result["status"] == "idle"

    def test_running_recent_heartbeat(self):
        svc = self._patch_all()
        with patch.dict(svc._turbo_state, {
            "status": "running",
            "last_heartbeat": time.time(),
        }):
            result = get_turbo_status()
        assert result["status"] == "running"

    def test_running_stale_heartbeat_sets_error(self):
        svc = self._patch_all()
        with patch.dict(svc._turbo_state, {
            "status": "running",
            "last_heartbeat": time.time() - 60,
            "paused": True,
        }):
            result = get_turbo_status()
        assert result["status"] == "error"
        assert "no progress" in result["error"].lower()

    def test_complete_status_not_affected(self):
        svc = self._patch_all()
        with patch.dict(svc._turbo_state, {
            "status": "complete",
            "last_heartbeat": 0.0,
        }):
            result = get_turbo_status()
        assert result["status"] == "complete"

    def test_returns_copy_not_original(self):
        svc = self._patch_all()
        with patch.dict(svc._turbo_state, {"status": "idle", "last_heartbeat": 0.0}):
            result = get_turbo_status()
        result["status"] = "mutated"
        assert svc._turbo_state["status"] == "idle"

    def test_zero_heartbeat_not_stale(self):
        svc = self._patch_all()
        with patch.dict(svc._turbo_state, {
            "status": "running",
            "last_heartbeat": 0,
        }):
            result = get_turbo_status()
        assert result["status"] == "running"


# ---------------------------------------------------------------------------
# start_from_sessions_training
# ---------------------------------------------------------------------------

class TestStartFromSessionsTraining:
    def _make_state(self, running=False):
        return TrainingState(running=running)

    def test_sets_running_true(self):
        state = self._make_state()
        start_from_sessions_training(state, {})
        assert state.running is True

    def test_builds_config_with_defaults(self):
        state = self._make_state()
        config = start_from_sessions_training(state, {})
        assert config["method"] == "from-sessions"
        assert config["epochs"] == 5
        assert config["learning_rate"] == pytest.approx(3e-4)
        assert config["batch_size"] == 8
        assert config["n_embed"] == 128
        assert config["n_layer"] == 4
        assert config["n_head"] == 4
        assert config["block_size"] == 128
        assert config["dropout"] == 0.1
        assert config["soul_name"] == "chat-trained"
        assert config["min_pair_quality"] == 2.0
        assert config["max_pairs"] == 500
        assert config["checkpoint_name"] is None
        assert config["session_ids"] is None
        assert config["experiment_id"] is None
        assert "started_at" in config

    def test_custom_config_overrides(self):
        state = self._make_state()
        config = start_from_sessions_training(state, {
            "epochs": 10,
            "learning_rate": 0.001,
            "batch_size": 16,
            "n_embed": 256,
            "n_layer": 6,
            "n_head": 8,
            "block_size": 256,
            "dropout": 0.2,
            "soul_name": "custom",
            "min_pair_quality": 1.5,
            "max_pairs": 1000,
            "checkpoint_name": "my_model",
            "session_ids": ["s1", "s2"],
            "experiment_id": "exp_123",
        })
        assert config["epochs"] == 10
        assert config["learning_rate"] == 0.001
        assert config["batch_size"] == 16
        assert config["n_embed"] == 256
        assert config["n_layer"] == 6
        assert config["n_head"] == 8
        assert config["block_size"] == 256
        assert config["dropout"] == 0.2
        assert config["soul_name"] == "custom"
        assert config["min_pair_quality"] == 1.5
        assert config["max_pairs"] == 1000
        assert config["checkpoint_name"] == "my_model"
        assert config["session_ids"] == ["s1", "s2"]
        assert config["experiment_id"] == "exp_123"

    def test_raises_if_already_running(self):
        state = self._make_state(running=True)
        with pytest.raises(RuntimeError, match="already"):
            start_from_sessions_training(state, {})

    def test_stores_config_in_state(self):
        state = self._make_state()
        config = start_from_sessions_training(state, {"epochs": 7})
        assert state.config is config
        assert state.config["epochs"] == 7

    def test_started_at_is_float(self):
        state = self._make_state()
        config = start_from_sessions_training(state, {})
        assert isinstance(config["started_at"], float)


# ---------------------------------------------------------------------------
# cleanup_stream_state
# ---------------------------------------------------------------------------

class TestCleanupStreamState:
    def _mock_runtime(self):
        from types import SimpleNamespace
        jobs = {}

        class FakeRuntime:
            def get(self, task_id):
                return jobs.get(task_id)

            def sync(self, task_id):
                pass

        return FakeRuntime(), jobs

    def test_sets_running_false(self):
        rt, jobs = self._mock_runtime()
        state = {"running": True}
        with patch("domains.training.runtime_protocol.get_training_runtime", return_value=rt):
            cleanup_stream_state("t1", {}, state, lambda s, e: "")
        assert state["running"] is False

    def test_marks_interrupted_if_not_terminal(self):
        rt, jobs = self._mock_runtime()
        jobs["t1"] = {"status": "running", "error": None}
        state = {"running": True}
        with patch("domains.training.runtime_protocol.get_training_runtime", return_value=rt):
            cleanup_stream_state("t1", {}, state, lambda s, e: "")
        assert jobs["t1"]["status"] == "interrupted"

    def test_does_not_overwrite_terminal_status(self):
        rt, jobs = self._mock_runtime()
        jobs["t1"] = {"status": "completed", "error": None}
        state = {"running": True}
        with patch("domains.training.runtime_protocol.get_training_runtime", return_value=rt):
            cleanup_stream_state("t1", {}, state, lambda s, e: "")
        assert jobs["t1"]["status"] == "completed"

    def test_does_not_overwrite_failed_status(self):
        rt, jobs = self._mock_runtime()
        jobs["t1"] = {"status": "failed", "error": "bad"}
        state = {"running": True}
        with patch("domains.training.runtime_protocol.get_training_runtime", return_value=rt):
            cleanup_stream_state("t1", {}, state, lambda s, e: "")
        assert jobs["t1"]["status"] == "failed"

    def test_does_not_overwrite_cancelled_status(self):
        rt, jobs = self._mock_runtime()
        jobs["t1"] = {"status": "cancelled", "error": "user"}
        state = {"running": True}
        with patch("domains.training.runtime_protocol.get_training_runtime", return_value=rt):
            cleanup_stream_state("t1", {}, state, lambda s, e: "")
        assert jobs["t1"]["status"] == "cancelled"

    def test_custom_status(self):
        rt, jobs = self._mock_runtime()
        jobs["t1"] = {"status": "running", "error": None}
        state = {"running": True}
        with patch("domains.training.runtime_protocol.get_training_runtime", return_value=rt):
            cleanup_stream_state("t1", {}, state, lambda s, e: "", status="timeout")
        assert jobs["t1"]["status"] == "timeout"

    def test_custom_error_message(self):
        rt, jobs = self._mock_runtime()
        jobs["t1"] = {"status": "running", "error": None}
        state = {"running": True}
        with patch("domains.training.runtime_protocol.get_training_runtime", return_value=rt):
            cleanup_stream_state("t1", {}, state, lambda s, e: "", error="oops")
        assert jobs["t1"]["error"] == "oops"

    def test_existing_error_preserved(self):
        rt, jobs = self._mock_runtime()
        jobs["t1"] = {"status": "running", "error": "prior error"}
        state = {"running": True}
        with patch("domains.training.runtime_protocol.get_training_runtime", return_value=rt):
            cleanup_stream_state("t1", {}, state, lambda s, e: "")
        assert jobs["t1"]["error"] == "prior error"

    def test_no_job_still_sets_running_false(self):
        rt, jobs = self._mock_runtime()
        state = {"running": True}
        with patch("domains.training.runtime_protocol.get_training_runtime", return_value=rt):
            cleanup_stream_state("t1", {}, state, lambda s, e: "")
        assert state["running"] is False

    def test_finish_cm_not_called_directly(self):
        """cleanup_stream_state does not call finish_cm_fn (it only resets state)."""
        rt, jobs = self._mock_runtime()
        state = {"running": True}
        called_with = []
        with patch("domains.training.runtime_protocol.get_training_runtime", return_value=rt):
            cleanup_stream_state("t1", {}, state, lambda s, e: called_with.append((s, e)))
        assert len(called_with) == 0


# ---------------------------------------------------------------------------
# Edge cases and integration-like tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_finite_payload_deeply_nested(self):
        obj = {"a": [{"b": [{"c": float("inf")}]}]}
        result = _finite_payload(obj)
        assert result == {"a": [{"b": [{"c": None}]}]}

    def test_finite_payload_empty_structures(self):
        assert _finite_payload({}) == {}
        assert _finite_payload([]) == []
        assert _finite_payload(()) == []

    def test_finite_payload_nested_dataclass_in_list(self):
        @dataclass
        class Point:
            x: float = float("nan")
            y: float = 1.0

        result = _finite_payload([Point(), Point(y=float("inf"))])
        assert result == [{"x": None, "y": 1.0}, {"x": None, "y": None}]

    def test_parse_subtitle_real_world_srt(self):
        srt = (
            "1\n"
            "00:01:23,456 --> 00:01:25,789\n"
            "This is a subtitle line.\n"
            "\n"
            "2\n"
            "00:01:26,000 --> 00:01:28,000\n"
            "And another one.\n"
        )
        result = parse_subtitle_text(srt)
        assert result == ["This is a subtitle line.", "And another one."]

    def test_resolve_dataset_id_with_hyphens(self, tmp_path):
        ds_dir = tmp_path / "datasets" / "my-dataset-v2"
        ds_dir.mkdir(parents=True)
        (ds_dir / "corpus.jsonl").write_text("data")
        import domains.training.service as svc
        with patch.object(svc, "REPO_ROOT", tmp_path):
            result = resolve_dataset_path("my-dataset-v2")
        assert result.endswith("corpus.jsonl")

    def test_describe_checkpoint_with_zero_loss(self):
        desc = describe_checkpoint({"loss": 0.0})
        assert "0.00" in desc
        assert "learned well" in desc

    def test_training_state_config_not_shared(self):
        s1 = TrainingState()
        s2 = TrainingState()
        s1.config["key"] = "value"
        assert "key" not in s2.config

    def test_start_from_sessions_does_not_mutate_input(self):
        state = TrainingState()
        input_config = {"epochs": 3}
        start_from_sessions_training(state, input_config)
        assert "method" not in input_config
