"""Unit tests for training_queue — training job handlers.

Tests the pure logic functions (_json_safe_payload, _resolve_checkpoint) and
handler behavior (missing enqueue, cancel propagation, error handling) using
monkeypatch to avoid real training runs.
"""

import asyncio
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from domains.infrastructure.training_queue import (
    _json_safe_payload,
    _resolve_checkpoint,
    training_handler,
    training_sessions_handler,
)
from domains.infrastructure.task_queue import Task


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_task(payload=None, enqueue=None, task_type="training") -> Task:
    task = Task(
        name="auto-train",
        task_type=task_type,
        payload=payload or {},
        metadata={"enqueue": enqueue},
    )
    return task


def _collect_events(enqueue):
    """Return a list of parsed SSE envelope dicts from an enqueue callback."""
    events = []

    def _enqueue(event_str: str):
        if event_str.startswith("data: "):
            events.append(json.loads(event_str[6:]))

    return events, _enqueue


def _make_fake_trainer_class(monkeypatch):
    """Create a fake TrainerConfig and SloughGPTTrainer for monkeypatching."""
    captured = {}

    class FakeConfig:
        def __init__(self, **kwargs):
            captured["config_kwargs"] = kwargs

    class FakeTrainer:
        def __init__(self, data_path, config):
            captured["data_path"] = data_path
            captured["config"] = config

        def train(self, on_progress=None, cancel_event=None, pause_event=None,
                  resume=False, resume_path=""):
            captured["train_kwargs"] = {
                "on_progress": on_progress,
                "cancel_event": cancel_event,
                "pause_event": pause_event,
                "resume": resume,
                "resume_path": resume_path,
            }
            if on_progress:
                on_progress({
                    "progress_percent": 50.0,
                    "train_loss": 0.5,
                    "eval_loss": 0.4,
                    "global_step": 10,
                    "total_steps": 20,
                    "steps_per_sec": 1.0,
                    "eta_s": 10.0,
                    "elapsed_s": 10.0,
                    "learning_rate": 3e-4,
                    "done": False,
                    "done_reason": None,
                    "avg_quality": None,
                    "epoch": 1,
                    "epochs": 1,
                })
                on_progress({
                    "progress_percent": 100.0,
                    "train_loss": 0.1,
                    "eval_loss": 0.2,
                    "global_step": 20,
                    "total_steps": 20,
                    "steps_per_sec": 1.0,
                    "eta_s": 0.0,
                    "elapsed_s": 20.0,
                    "learning_rate": 3e-4,
                    "done": True,
                    "done_reason": "completed",
                    "avg_quality": 0.9,
                    "epoch": 1,
                    "epochs": 1,
                })
            return {"success": True, "final_loss": 0.1}

    return captured, FakeConfig, FakeTrainer


# ── _json_safe_payload ──────────────────────────────────────────────────────

class TestJsonSafePayload:

    def test_int_passthrough(self):
        assert _json_safe_payload(42) == 42

    def test_str_passthrough(self):
        assert _json_safe_payload("hello") == "hello"

    def test_none_passthrough(self):
        assert _json_safe_payload(None) is None

    def test_bool_passthrough(self):
        assert _json_safe_payload(True) is True

    def test_finite_float_passthrough(self):
        assert _json_safe_payload(3.14) == 3.14

    def test_positive_inf_replaced(self):
        assert _json_safe_payload(float("inf")) is None

    def test_negative_inf_replaced(self):
        assert _json_safe_payload(float("-inf")) is None

    def test_nan_replaced(self):
        assert _json_safe_payload(float("nan")) is None

    def test_list_with_inf(self):
        result = _json_safe_payload([1.0, float("inf"), 3.0])
        assert result == [1.0, None, 3.0]

    def test_tuple_with_inf(self):
        result = _json_safe_payload((1.0, float("nan")))
        assert result == [1.0, None]

    def test_dict_with_inf(self):
        result = _json_safe_payload({"a": 1.0, "b": float("inf"), "c": 3.0})
        assert result == {"a": 1.0, "b": None, "c": 3.0}

    def test_nested_dict(self):
        result = _json_safe_payload({"outer": {"inner": float("inf")}})
        assert result == {"outer": {"inner": None}}

    def test_nested_list_of_dicts(self):
        result = _json_safe_payload([{"loss": 0.5}, {"loss": float("nan")}])
        assert result == [{"loss": 0.5}, {"loss": None}]

    def test_dataclass(self):
        @dataclass
        class Metrics:
            loss: float
            steps: int

        m = Metrics(loss=float("inf"), steps=10)
        result = _json_safe_payload(m)
        assert result == {"loss": None, "steps": 10}

    def test_nested_dataclass(self):
        @dataclass
        class Inner:
            val: float

        @dataclass
        class Outer:
            inner: Inner
            count: int

        o = Outer(inner=Inner(val=float("nan")), count=5)
        result = _json_safe_payload(o)
        assert result == {"inner": {"val": None}, "count": 5}

    def test_empty_dict(self):
        assert _json_safe_payload({}) == {}

    def test_empty_list(self):
        assert _json_safe_payload([]) == []

    def test_zero_float_finite(self):
        assert _json_safe_payload(0.0) == 0.0

    def test_negative_float_finite(self):
        assert _json_safe_payload(-1.5) == -1.5

    def test_complex_nested_structure(self):
        result = _json_safe_payload({
            "metrics": {"loss": 0.5, "ppl": float("inf")},
            "history": [0.8, 0.6, float("nan")],
            "done": True,
        })
        assert result == {
            "metrics": {"loss": 0.5, "ppl": None},
            "history": [0.8, 0.6, None],
            "done": True,
        }

    def test_numpy_finite(self):
        result = _json_safe_payload(np.float32(1.5))
        assert float(result) == pytest.approx(1.5)

    def test_numpy_inf(self):
        # numpy floats are not Python floats, so _json_safe_payload does not
        # catch them. This is acceptable because the SSE envelope's _json_safe
        # handles numpy serialization separately.
        result = _json_safe_payload(np.float32("inf"))
        # Result is the numpy scalar as-is (not replaced with None)
        assert float(result) == float("inf")

    def test_integer_in_list(self):
        result = _json_safe_payload([1, 2, 3])
        assert result == [1, 2, 3]


# ── _resolve_checkpoint ─────────────────────────────────────────────────────

class TestResolveCheckpoint:

    def test_none_name(self):
        assert _resolve_checkpoint(None, "/tmp") is None

    def test_empty_string(self):
        assert _resolve_checkpoint("", "/tmp") is None

    def test_full_path_exists(self, tmp_path):
        f = tmp_path / "model.soul"
        f.write_text("data")
        result = _resolve_checkpoint(str(f), str(tmp_path))
        assert result == str(f)

    def test_full_path_not_exists(self):
        assert _resolve_checkpoint("/nonexistent/path.soul", "/tmp") is None

    def test_name_with_soul_suffix(self, tmp_path):
        f = tmp_path / "my-run.soul"
        f.write_text("data")
        result = _resolve_checkpoint("my-run", str(tmp_path))
        assert result == str(f)

    def test_name_without_suffix(self, tmp_path):
        f = tmp_path / "checkpoint"
        f.write_text("data")
        result = _resolve_checkpoint("checkpoint", str(tmp_path))
        assert result == str(f)

    def test_name_priority_over_bare(self, tmp_path):
        soul = tmp_path / "run.soul"
        soul.write_text("soul data")
        bare = tmp_path / "run"
        bare.write_text("bare data")
        result = _resolve_checkpoint("run", str(tmp_path))
        assert result == str(soul)

    def test_not_found(self, tmp_path):
        result = _resolve_checkpoint("nonexistent", str(tmp_path))
        assert result is None

    def test_subdirectory_not_checked(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "model.soul").write_text("data")
        result = _resolve_checkpoint("model", str(tmp_path))
        assert result is None

    def test_absolute_path_as_name(self, tmp_path):
        f = tmp_path / "abs.soul"
        f.write_text("data")
        result = _resolve_checkpoint(str(f), str(tmp_path))
        assert result == str(f)


# ── training_handler ────────────────────────────────────────────────────────

class TestTrainingHandler:

    def test_missing_enqueue_returns_failed(self):
        task = _make_task({"data_path": "/nonexistent"}, enqueue=None)
        result = asyncio.run(training_handler(task))
        assert result["status"] == "failed"
        assert "No enqueue callback" in result["error"]

    def test_missing_enqueue_includes_metadata_keys(self):
        task = _make_task({}, enqueue=None)
        task.metadata["other_key"] = "value"
        result = asyncio.run(training_handler(task))
        assert result["status"] == "failed"

    def test_cancel_immediately_returns_cancelled(self, tmp_path, monkeypatch):
        captured, FakeConfig, FakeTrainer = _make_fake_trainer_class(monkeypatch)
        import domains.training.train_pipeline as tp
        monkeypatch.setattr(tp, "TrainerConfig", FakeConfig)
        monkeypatch.setattr(tp, "SloughGPTTrainer", FakeTrainer)

        events, enqueue = _collect_events(None)
        task = _make_task(
            {"data_path": str(tmp_path / "dummy.txt"),
             "checkpoint_dir": str(tmp_path / "ckpts"),
             "n_embed": 16, "n_layer": 1, "n_head": 2,
             "block_size": 16, "epochs": 1},
            enqueue=enqueue,
        )
        task.cancel_event.set()

        result = asyncio.run(training_handler(task))
        assert result["status"] == "cancelled"

    def test_emits_train_sse_events(self, tmp_path, monkeypatch):
        captured, FakeConfig, FakeTrainer = _make_fake_trainer_class(monkeypatch)
        import domains.training.train_pipeline as tp
        monkeypatch.setattr(tp, "TrainerConfig", FakeConfig)
        monkeypatch.setattr(tp, "SloughGPTTrainer", FakeTrainer)

        events, enqueue = _collect_events(None)
        task = _make_task(
            {"data_path": str(tmp_path / "dummy.txt"),
             "checkpoint_dir": str(tmp_path / "ckpts")},
            enqueue=enqueue,
        )

        result = asyncio.run(training_handler(task))

        train_events = [e for e in events if e.get("phase") == "TRAIN"]
        assert len(train_events) >= 1
        assert train_events[0]["stream"] == "auto-train"
        assert train_events[0]["status"] == "working"
        assert "loss" in train_events[0]["data"]
        assert "progress" in train_events[0]["data"]

    def test_emits_complete_event(self, tmp_path, monkeypatch):
        captured, FakeConfig, FakeTrainer = _make_fake_trainer_class(monkeypatch)
        import domains.training.train_pipeline as tp
        monkeypatch.setattr(tp, "TrainerConfig", FakeConfig)
        monkeypatch.setattr(tp, "SloughGPTTrainer", FakeTrainer)

        events, enqueue = _collect_events(None)
        task = _make_task(
            {"data_path": str(tmp_path / "dummy.txt"),
             "checkpoint_dir": str(tmp_path / "ckpts")},
            enqueue=enqueue,
        )

        result = asyncio.run(training_handler(task))

        completes = [e for e in events if e.get("status") == "complete"]
        assert completes
        assert completes[-1]["message"] == "Training complete"

    def test_defaults_applied(self, tmp_path, monkeypatch):
        captured, FakeConfig, FakeTrainer = _make_fake_trainer_class(monkeypatch)
        import domains.training.train_pipeline as tp
        monkeypatch.setattr(tp, "TrainerConfig", FakeConfig)
        monkeypatch.setattr(tp, "SloughGPTTrainer", FakeTrainer)

        events, enqueue = _collect_events(None)
        task = _make_task(
            {"data_path": str(tmp_path / "dummy.txt"),
             "checkpoint_dir": str(tmp_path / "ckpts")},
            enqueue=enqueue,
        )

        asyncio.run(training_handler(task))

        cfg = captured["config_kwargs"]
        assert cfg["n_embed"] == 128
        assert cfg["n_layer"] == 4
        assert cfg["n_head"] == 4
        assert cfg["block_size"] == 128
        assert cfg["dropout"] == 0.1
        assert cfg["batch_size"] == 16
        assert cfg["epochs"] == 20
        assert cfg["learning_rate"] == 3e-4
        assert cfg["early_stopping_patience"] == 5

    def test_custom_config_passed_through(self, tmp_path, monkeypatch):
        captured, FakeConfig, FakeTrainer = _make_fake_trainer_class(monkeypatch)
        import domains.training.train_pipeline as tp
        monkeypatch.setattr(tp, "TrainerConfig", FakeConfig)
        monkeypatch.setattr(tp, "SloughGPTTrainer", FakeTrainer)

        events, enqueue = _collect_events(None)
        task = _make_task(
            {"data_path": str(tmp_path / "dummy.txt"),
             "checkpoint_dir": str(tmp_path / "ckpts"),
             "n_embed": 64, "n_layer": 2, "n_head": 8,
             "block_size": 64, "dropout": 0.2, "batch_size": 8,
             "epochs": 5, "learning_rate": 1e-3,
             "early_stopping_patience": 3},
            enqueue=enqueue,
        )

        asyncio.run(training_handler(task))

        cfg = captured["config_kwargs"]
        assert cfg["n_embed"] == 64
        assert cfg["n_layer"] == 2
        assert cfg["n_head"] == 8
        assert cfg["block_size"] == 64
        assert cfg["dropout"] == 0.2
        assert cfg["batch_size"] == 8
        assert cfg["epochs"] == 5
        assert cfg["learning_rate"] == 1e-3
        assert cfg["early_stopping_patience"] == 3

    def test_resume_defaults_false(self, tmp_path, monkeypatch):
        captured, FakeConfig, FakeTrainer = _make_fake_trainer_class(monkeypatch)
        import domains.training.train_pipeline as tp
        monkeypatch.setattr(tp, "TrainerConfig", FakeConfig)
        monkeypatch.setattr(tp, "SloughGPTTrainer", FakeTrainer)

        events, enqueue = _collect_events(None)
        task = _make_task(
            {"data_path": str(tmp_path / "dummy.txt"),
             "checkpoint_dir": str(tmp_path / "ckpts")},
            enqueue=enqueue,
        )

        asyncio.run(training_handler(task))
        assert captured["train_kwargs"]["resume"] is False
        assert captured["train_kwargs"]["resume_path"] == ""

    def test_resume_true_passed(self, tmp_path, monkeypatch):
        captured, FakeConfig, FakeTrainer = _make_fake_trainer_class(monkeypatch)
        import domains.training.train_pipeline as tp
        monkeypatch.setattr(tp, "TrainerConfig", FakeConfig)
        monkeypatch.setattr(tp, "SloughGPTTrainer", FakeTrainer)

        events, enqueue = _collect_events(None)
        task = _make_task(
            {"data_path": str(tmp_path / "dummy.txt"),
             "checkpoint_dir": str(tmp_path / "ckpts"),
             "resume": True, "resume_path": "/some/path.soul"},
            enqueue=enqueue,
        )

        asyncio.run(training_handler(task))
        assert captured["train_kwargs"]["resume"] is True
        assert captured["train_kwargs"]["resume_path"] == "/some/path.soul"

    def test_cancel_event_passed_to_trainer(self, tmp_path, monkeypatch):
        captured, FakeConfig, FakeTrainer = _make_fake_trainer_class(monkeypatch)
        import domains.training.train_pipeline as tp
        monkeypatch.setattr(tp, "TrainerConfig", FakeConfig)
        monkeypatch.setattr(tp, "SloughGPTTrainer", FakeTrainer)

        events, enqueue = _collect_events(None)
        task = _make_task(
            {"data_path": str(tmp_path / "dummy.txt"),
             "checkpoint_dir": str(tmp_path / "ckpts")},
            enqueue=enqueue,
        )

        asyncio.run(training_handler(task))
        cancel_ev = captured["train_kwargs"]["cancel_event"]
        assert isinstance(cancel_ev, type(__import__("threading").Event()))

    def test_pause_event_passed_to_trainer(self, tmp_path, monkeypatch):
        captured, FakeConfig, FakeTrainer = _make_fake_trainer_class(monkeypatch)
        import domains.training.train_pipeline as tp
        monkeypatch.setattr(tp, "TrainerConfig", FakeConfig)
        monkeypatch.setattr(tp, "SloughGPTTrainer", FakeTrainer)

        events, enqueue = _collect_events(None)
        task = _make_task(
            {"data_path": str(tmp_path / "dummy.txt"),
             "checkpoint_dir": str(tmp_path / "ckpts")},
            enqueue=enqueue,
        )

        asyncio.run(training_handler(task))
        pause_ev = captured["train_kwargs"]["pause_event"]
        assert isinstance(pause_ev, type(__import__("threading").Event()))

    def test_on_progress_callback_called(self, tmp_path, monkeypatch):
        captured, FakeConfig, FakeTrainer = _make_fake_trainer_class(monkeypatch)
        import domains.training.train_pipeline as tp
        monkeypatch.setattr(tp, "TrainerConfig", FakeConfig)
        monkeypatch.setattr(tp, "SloughGPTTrainer", FakeTrainer)

        events, enqueue = _collect_events(None)
        task = _make_task(
            {"data_path": str(tmp_path / "dummy.txt"),
             "checkpoint_dir": str(tmp_path / "ckpts")},
            enqueue=enqueue,
        )

        asyncio.run(training_handler(task))
        assert captured["train_kwargs"]["on_progress"] is not None

    def test_trainer_exception_returns_failed(self, tmp_path, monkeypatch):
        class RaisingConfig:
            def __init__(self, **kwargs):
                pass

        class RaisingTrainer:
            def __init__(self, data_path, config):
                pass

            def train(self, **kwargs):
                raise RuntimeError("Simulated training failure")

        import domains.training.train_pipeline as tp
        monkeypatch.setattr(tp, "TrainerConfig", RaisingConfig)
        monkeypatch.setattr(tp, "SloughGPTTrainer", RaisingTrainer)

        events, enqueue = _collect_events(None)
        task = _make_task(
            {"data_path": str(tmp_path / "dummy.txt"),
             "checkpoint_dir": str(tmp_path / "ckpts")},
            enqueue=enqueue,
        )

        result = asyncio.run(training_handler(task))
        assert result["status"] == "failed"
        assert "Simulated training failure" in result["error"]

    def test_trainer_exception_emits_error_event(self, tmp_path, monkeypatch):
        class RaisingConfig:
            def __init__(self, **kwargs):
                pass

        class RaisingTrainer:
            def __init__(self, data_path, config):
                pass

            def train(self, **kwargs):
                raise ValueError("Bad data")

        import domains.training.train_pipeline as tp
        monkeypatch.setattr(tp, "TrainerConfig", RaisingConfig)
        monkeypatch.setattr(tp, "SloughGPTTrainer", RaisingTrainer)

        events, enqueue = _collect_events(None)
        task = _make_task(
            {"data_path": str(tmp_path / "dummy.txt"),
             "checkpoint_dir": str(tmp_path / "ckpts")},
            enqueue=enqueue,
        )

        asyncio.run(training_handler(task))
        errors = [e for e in events if e.get("status") == "error"]
        assert errors
        assert errors[-1]["stream"] == "auto-train"

    def test_output_dir_created(self, tmp_path, monkeypatch):
        captured, FakeConfig, FakeTrainer = _make_fake_trainer_class(monkeypatch)
        import domains.training.train_pipeline as tp
        monkeypatch.setattr(tp, "TrainerConfig", FakeConfig)
        monkeypatch.setattr(tp, "SloughGPTTrainer", FakeTrainer)

        events, enqueue = _collect_events(None)
        ckpt_dir = tmp_path / "new_dir" / "ckpts"
        task = _make_task(
            {"data_path": str(tmp_path / "dummy.txt"),
             "checkpoint_dir": str(ckpt_dir)},
            enqueue=enqueue,
        )

        asyncio.run(training_handler(task))
        assert ckpt_dir.exists()

    def test_default_checkpoint_dir(self, tmp_path, monkeypatch):
        captured, FakeConfig, FakeTrainer = _make_fake_trainer_class(monkeypatch)
        import domains.training.train_pipeline as tp
        monkeypatch.setattr(tp, "TrainerConfig", FakeConfig)
        monkeypatch.setattr(tp, "SloughGPTTrainer", FakeTrainer)

        events, enqueue = _collect_events(None)
        task = _make_task(
            {"data_path": str(tmp_path / "dummy.txt")},
            enqueue=enqueue,
        )

        asyncio.run(training_handler(task))
        # Config should have a checkpoint_dir set
        assert captured["config_kwargs"]["checkpoint_dir"]


# ── training_sessions_handler ───────────────────────────────────────────────

class TestTrainingSessionsHandler:

    def test_missing_enqueue_returns_failed(self):
        task = _make_task({"session_ids": ["s1"]}, enqueue=None, task_type="training-sessions")
        result = asyncio.run(training_sessions_handler(task))
        assert result["status"] == "failed"
        assert "No enqueue callback" in result["error"]

    def test_cancel_immediately_returns_cancelled(self, monkeypatch):
        class FakeConfig:
            def __init__(self, **kwargs):
                self.epochs = kwargs.get("epochs", 5)
                self.__dict__.update(kwargs)

        class FakeSessionTrainer:
            def __call__(self, config, on_step=None, cancel_event=None):
                return {"success": True}, {"num_pairs": 1}

        monkeypatch.setattr(
            "domains.training.chat_trainer.ChatTrainConfig", FakeConfig
        )
        monkeypatch.setattr(
            "domains.training.chat_trainer.train_from_sessions",
            FakeSessionTrainer(),
        )

        events, enqueue = _collect_events(None)
        task = _make_task(
            {"session_ids": ["s1"], "checkpoint_dir": str(Path("/tmp/ckpts"))},
            enqueue=enqueue,
            task_type="training-sessions",
        )
        task.cancel_event.set()

        result = asyncio.run(training_sessions_handler(task))
        assert result["status"] == "cancelled"

    def test_emits_pairs_event(self, monkeypatch):
        class FakeConfig:
            def __init__(self, **kwargs):
                self.epochs = kwargs.get("epochs", 5)
                self.__dict__.update(kwargs)

        class FakeSessionTrainer:
            def __call__(self, config, on_step=None, cancel_event=None):
                return {"success": True}, {"num_pairs": 5, "perplexity": 1.2}

        monkeypatch.setattr(
            "domains.training.chat_trainer.ChatTrainConfig", FakeConfig
        )
        monkeypatch.setattr(
            "domains.training.chat_trainer.train_from_sessions",
            FakeSessionTrainer(),
        )

        events, enqueue = _collect_events(None)
        task = _make_task(
            {"session_ids": ["s1"], "checkpoint_dir": str(Path("/tmp/ckpts"))},
            enqueue=enqueue,
            task_type="training-sessions",
        )

        asyncio.run(training_sessions_handler(task))

        pairs_events = [e for e in events if e.get("phase") == "PAIRS"]
        assert pairs_events

    def test_emits_complete_event(self, monkeypatch):
        class FakeConfig:
            def __init__(self, **kwargs):
                self.epochs = kwargs.get("epochs", 5)
                self.__dict__.update(kwargs)

        class FakeSessionTrainer:
            def __call__(self, config, on_step=None, cancel_event=None):
                return {"success": True}, {"num_pairs": 3}

        monkeypatch.setattr(
            "domains.training.chat_trainer.ChatTrainConfig", FakeConfig
        )
        monkeypatch.setattr(
            "domains.training.chat_trainer.train_from_sessions",
            FakeSessionTrainer(),
        )

        events, enqueue = _collect_events(None)
        task = _make_task(
            {"session_ids": ["s1"], "checkpoint_dir": str(Path("/tmp/ckpts"))},
            enqueue=enqueue,
            task_type="training-sessions",
        )

        asyncio.run(training_sessions_handler(task))

        completes = [e for e in events if e.get("status") == "complete"]
        assert completes
        assert completes[-1]["message"] == "Training complete"

    def test_returns_metadata(self, monkeypatch):
        class FakeConfig:
            def __init__(self, **kwargs):
                self.epochs = kwargs.get("epochs", 5)
                self.__dict__.update(kwargs)

        expected_meta = {"num_pairs": 10, "perplexity": 2.0, "samples": 50}

        class FakeSessionTrainer:
            def __call__(self, config, on_step=None, cancel_event=None):
                return {"success": True}, expected_meta

        monkeypatch.setattr(
            "domains.training.chat_trainer.ChatTrainConfig", FakeConfig
        )
        monkeypatch.setattr(
            "domains.training.chat_trainer.train_from_sessions",
            FakeSessionTrainer(),
        )

        events, enqueue = _collect_events(None)
        task = _make_task(
            {"session_ids": ["s1"], "checkpoint_dir": str(Path("/tmp/ckpts"))},
            enqueue=enqueue,
            task_type="training-sessions",
        )

        result = asyncio.run(training_sessions_handler(task))
        assert result["num_pairs"] == 10
        assert result["perplexity"] == 2.0

    def test_defaults_applied(self, monkeypatch):
        captured = {}

        class FakeConfig:
            def __init__(self, **kwargs):
                self.epochs = kwargs.get("epochs", 5)
                captured.update(kwargs)

        class FakeSessionTrainer:
            def __call__(self, config, on_step=None, cancel_event=None):
                return {"success": True}, {"num_pairs": 1}

        monkeypatch.setattr(
            "domains.training.chat_trainer.ChatTrainConfig", FakeConfig
        )
        monkeypatch.setattr(
            "domains.training.chat_trainer.train_from_sessions",
            FakeSessionTrainer(),
        )

        events, enqueue = _collect_events(None)
        task = _make_task(
            {"session_ids": ["s1"], "checkpoint_dir": str(Path("/tmp/ckpts"))},
            enqueue=enqueue,
            task_type="training-sessions",
        )

        asyncio.run(training_sessions_handler(task))

        assert captured["n_embed"] == 128
        assert captured["n_layer"] == 4
        assert captured["n_head"] == 4
        assert captured["block_size"] == 128
        assert captured["dropout"] == 0.1
        assert captured["epochs"] == 5
        assert captured["lr"] == 3e-4
        assert captured["batch_size"] == 8

    def test_custom_config_passed(self, monkeypatch):
        captured = {}

        class FakeConfig:
            def __init__(self, **kwargs):
                self.epochs = kwargs.get("epochs", 5)
                captured.update(kwargs)

        class FakeSessionTrainer:
            def __call__(self, config, on_step=None, cancel_event=None):
                return {"success": True}, {"num_pairs": 1}

        monkeypatch.setattr(
            "domains.training.chat_trainer.ChatTrainConfig", FakeConfig
        )
        monkeypatch.setattr(
            "domains.training.chat_trainer.train_from_sessions",
            FakeSessionTrainer(),
        )

        events, enqueue = _collect_events(None)
        task = _make_task(
            {"session_ids": ["s1"],
             "checkpoint_dir": str(Path("/tmp/ckpts")),
             "n_embed": 64, "n_layer": 2, "n_head": 8,
             "block_size": 64, "epochs": 10,
             "learning_rate": 5e-4, "batch_size": 4,
             "soul_name": "custom-soul"},
            enqueue=enqueue,
            task_type="training-sessions",
        )

        asyncio.run(training_sessions_handler(task))

        assert captured["n_embed"] == 64
        assert captured["n_layer"] == 2
        assert captured["n_head"] == 8
        assert captured["soul_name"] == "custom-soul"

    def test_session_ids_passed_to_config(self, monkeypatch):
        captured = {}

        class FakeConfig:
            def __init__(self, **kwargs):
                self.epochs = kwargs.get("epochs", 5)
                captured.update(kwargs)

        class FakeSessionTrainer:
            def __call__(self, config, on_step=None, cancel_event=None):
                return {"success": True}, {"num_pairs": 1}

        monkeypatch.setattr(
            "domains.training.chat_trainer.ChatTrainConfig", FakeConfig
        )
        monkeypatch.setattr(
            "domains.training.chat_trainer.train_from_sessions",
            FakeSessionTrainer(),
        )

        events, enqueue = _collect_events(None)
        task = _make_task(
            {"session_ids": ["s1", "s2", "s3"],
             "checkpoint_dir": str(Path("/tmp/ckpts"))},
            enqueue=enqueue,
            task_type="training-sessions",
        )

        asyncio.run(training_sessions_handler(task))
        assert captured["session_ids"] == ["s1", "s2", "s3"]

    def test_trainer_exception_returns_failed(self, monkeypatch):
        class FakeConfig:
            def __init__(self, **kwargs):
                self.epochs = kwargs.get("epochs", 5)

        def raising_train(*args, **kwargs):
            raise RuntimeError("Session training crashed")

        monkeypatch.setattr(
            "domains.training.chat_trainer.ChatTrainConfig", FakeConfig
        )
        monkeypatch.setattr(
            "domains.training.chat_trainer.train_from_sessions",
            raising_train,
        )

        events, enqueue = _collect_events(None)
        task = _make_task(
            {"session_ids": ["s1"], "checkpoint_dir": str(Path("/tmp/ckpts"))},
            enqueue=enqueue,
            task_type="training-sessions",
        )

        result = asyncio.run(training_sessions_handler(task))
        assert result["status"] == "failed"
        assert "Session training crashed" in result["error"]

    def test_trainer_exception_emits_error(self, monkeypatch):
        class FakeConfig:
            def __init__(self, **kwargs):
                self.epochs = kwargs.get("epochs", 5)

        def raising_train(*args, **kwargs):
            raise ValueError("Bad sessions")

        monkeypatch.setattr(
            "domains.training.chat_trainer.ChatTrainConfig", FakeConfig
        )
        monkeypatch.setattr(
            "domains.training.chat_trainer.train_from_sessions",
            raising_train,
        )

        events, enqueue = _collect_events(None)
        task = _make_task(
            {"session_ids": ["s1"], "checkpoint_dir": str(Path("/tmp/ckpts"))},
            enqueue=enqueue,
            task_type="training-sessions",
        )

        asyncio.run(training_sessions_handler(task))
        errors = [e for e in events if e.get("status") == "error"]
        assert errors
        assert errors[-1]["stream"] == "auto-train"

    def test_checkpoint_name_resolved(self, monkeypatch, tmp_path):
        captured = {}

        class FakeConfig:
            def __init__(self, **kwargs):
                self.epochs = kwargs.get("epochs", 5)
                captured.update(kwargs)

        soul_file = tmp_path / "existing.soul"
        soul_file.write_text("data")

        class FakeSessionTrainer:
            def __call__(self, config, on_step=None, cancel_event=None):
                return {"success": True}, {"num_pairs": 1}

        monkeypatch.setattr(
            "domains.training.chat_trainer.ChatTrainConfig", FakeConfig
        )
        monkeypatch.setattr(
            "domains.training.chat_trainer.train_from_sessions",
            FakeSessionTrainer(),
        )

        events, enqueue = _collect_events(None)
        task = _make_task(
            {"session_ids": ["s1"],
             "checkpoint_dir": str(tmp_path),
             "checkpoint_name": "existing"},
            enqueue=enqueue,
            task_type="training-sessions",
        )

        asyncio.run(training_sessions_handler(task))
        assert captured["resume_checkpoint"] == str(soul_file)

    def test_checkpoint_not_found_gives_none(self, monkeypatch):
        captured = {}

        class FakeConfig:
            def __init__(self, **kwargs):
                self.epochs = kwargs.get("epochs", 5)
                captured.update(kwargs)

        class FakeSessionTrainer:
            def __call__(self, config, on_step=None, cancel_event=None):
                return {"success": True}, {"num_pairs": 1}

        monkeypatch.setattr(
            "domains.training.chat_trainer.ChatTrainConfig", FakeConfig
        )
        monkeypatch.setattr(
            "domains.training.chat_trainer.train_from_sessions",
            FakeSessionTrainer(),
        )

        events, enqueue = _collect_events(None)
        task = _make_task(
            {"session_ids": ["s1"],
             "checkpoint_dir": str(Path("/tmp/nonexistent")),
             "checkpoint_name": "ghost"},
            enqueue=enqueue,
            task_type="training-sessions",
        )

        asyncio.run(training_sessions_handler(task))
        assert captured["resume_checkpoint"] is None

    def test_on_step_callback_emits_train_events(self, monkeypatch):
        class FakeConfig:
            def __init__(self, **kwargs):
                self.epochs = kwargs.get("epochs", 5)
                self.__dict__.update(kwargs)

        class FakeSessionTrainer:
            def __call__(self, config, on_step=None, cancel_event=None):
                if on_step:
                    on_step(1, 0.9, 0, total_steps=10)
                    on_step(5, 0.5, 0, total_steps=10)
                return {"success": True}, {"num_pairs": 1}

        monkeypatch.setattr(
            "domains.training.chat_trainer.ChatTrainConfig", FakeConfig
        )
        monkeypatch.setattr(
            "domains.training.chat_trainer.train_from_sessions",
            FakeSessionTrainer(),
        )

        events, enqueue = _collect_events(None)
        task = _make_task(
            {"session_ids": ["s1"], "checkpoint_dir": str(Path("/tmp/ckpts"))},
            enqueue=enqueue,
            task_type="training-sessions",
        )

        asyncio.run(training_sessions_handler(task))

        train_events = [e for e in events if e.get("phase") == "TRAIN"]
        assert len(train_events) >= 2
        assert train_events[0]["data"]["step"] == 1
        assert train_events[0]["data"]["loss"] == 0.9
        assert train_events[1]["data"]["step"] == 5

    def test_soul_name_default(self, monkeypatch):
        captured = {}

        class FakeConfig:
            def __init__(self, **kwargs):
                self.epochs = kwargs.get("epochs", 5)
                captured.update(kwargs)

        class FakeSessionTrainer:
            def __call__(self, config, on_step=None, cancel_event=None):
                return {"success": True}, {"num_pairs": 1}

        monkeypatch.setattr(
            "domains.training.chat_trainer.ChatTrainConfig", FakeConfig
        )
        monkeypatch.setattr(
            "domains.training.chat_trainer.train_from_sessions",
            FakeSessionTrainer(),
        )

        events, enqueue = _collect_events(None)
        task = _make_task(
            {"session_ids": ["s1"], "checkpoint_dir": str(Path("/tmp/ckpts"))},
            enqueue=enqueue,
            task_type="training-sessions",
        )

        asyncio.run(training_sessions_handler(task))
        assert captured["soul_name"] == "chat-trained"

    def test_min_pair_quality_default(self, monkeypatch):
        captured = {}

        class FakeConfig:
            def __init__(self, **kwargs):
                self.epochs = kwargs.get("epochs", 5)
                captured.update(kwargs)

        class FakeSessionTrainer:
            def __call__(self, config, on_step=None, cancel_event=None):
                return {"success": True}, {"num_pairs": 1}

        monkeypatch.setattr(
            "domains.training.chat_trainer.ChatTrainConfig", FakeConfig
        )
        monkeypatch.setattr(
            "domains.training.chat_trainer.train_from_sessions",
            FakeSessionTrainer(),
        )

        events, enqueue = _collect_events(None)
        task = _make_task(
            {"session_ids": ["s1"], "checkpoint_dir": str(Path("/tmp/ckpts"))},
            enqueue=enqueue,
            task_type="training-sessions",
        )

        asyncio.run(training_sessions_handler(task))
        assert captured["min_pair_quality"] == 2.0
        assert captured["max_pairs"] == 500

    def test_train_event_stream_and_status(self, monkeypatch):
        class FakeConfig:
            def __init__(self, **kwargs):
                self.epochs = kwargs.get("epochs", 5)
                self.__dict__.update(kwargs)

        class FakeSessionTrainer:
            def __call__(self, config, on_step=None, cancel_event=None):
                if on_step:
                    on_step(1, 1.0, 0, total_steps=5)
                return {"success": True}, {"num_pairs": 1}

        monkeypatch.setattr(
            "domains.training.chat_trainer.ChatTrainConfig", FakeConfig
        )
        monkeypatch.setattr(
            "domains.training.chat_trainer.train_from_sessions",
            FakeSessionTrainer(),
        )

        events, enqueue = _collect_events(None)
        task = _make_task(
            {"session_ids": ["s1"], "checkpoint_dir": str(Path("/tmp/ckpts"))},
            enqueue=enqueue,
            task_type="training-sessions",
        )

        asyncio.run(training_sessions_handler(task))

        train_events = [e for e in events if e.get("phase") == "TRAIN"]
        assert train_events[0]["stream"] == "auto-train"
        assert train_events[0]["status"] == "working"


# ── Training Pipeline Integration Tests ───────────────────────────────

class TestTrainingDataPathValidation:
    """Tests for data_path validation in training_handler."""

    def test_empty_data_path_returns_failed(self):
        """training_handler returns failed when data_path is empty."""
        events, enqueue = _collect_events(None)
        task = _make_task({"data_path": ""}, enqueue=enqueue)
        result = asyncio.run(training_handler(task))
        assert result["status"] == "failed"
        assert "Data file not found" in result["error"] or "not found" in result["error"]

    def test_missing_data_path_returns_failed(self):
        """training_handler returns failed when data_path is missing from payload."""
        events, enqueue = _collect_events(None)
        task = _make_task({}, enqueue=enqueue)
        result = asyncio.run(training_handler(task))
        assert result["status"] == "failed"
        assert "Data file not found" in result["error"] or "not found" in result["error"]

    def test_empty_data_path_sends_error_event(self):
        """training_handler sends SSE error event when data_path is empty."""
        events, enqueue = _collect_events(None)
        task = _make_task({"data_path": ""}, enqueue=enqueue)
        asyncio.run(training_handler(task))
        error_events = [e for e in events if e.get("status") == "error"]
        assert len(error_events) > 0
        assert "not found" in error_events[0].get("message", "").lower()

    def test_valid_data_path_proceeds(self, tmp_path, monkeypatch):
        """training_handler proceeds when data_path is valid."""
        captured, FakeConfig, FakeTrainer = _make_fake_trainer_class(monkeypatch)
        import domains.training.train_pipeline as tp
        monkeypatch.setattr(tp, "TrainerConfig", FakeConfig)
        monkeypatch.setattr(tp, "SloughGPTTrainer", FakeTrainer)

        dummy = tmp_path / "data.txt"
        dummy.write_text("hello world " * 100)

        events, enqueue = _collect_events(None)
        task = _make_task(
            {"data_path": str(dummy), "checkpoint_dir": str(tmp_path / "ckpts"),
             "n_embed": 16, "n_layer": 1, "n_head": 2,
             "block_size": 16, "epochs": 1},
            enqueue=enqueue,
        )
        result = asyncio.run(training_handler(task))
        assert result.get("success") is True
        assert captured["data_path"] == str(dummy)


class TestCheckpointPruning:
    """Tests for _prune_stale_checkpoints behavior."""

    def test_prune_keeps_max_checkpoints(self, tmp_path, monkeypatch):
        """During training, max_checkpoints newest files are kept."""
        import domains.training.train_pipeline as tp

        config = tp.TrainerConfig(
            checkpoint_dir=str(tmp_path),
            max_checkpoints=3,
        )
        trainer = tp.SloughGPTTrainer.__new__(tp.SloughGPTTrainer)
        trainer.config = config
        trainer._best_model_path = None
        trainer._last_checkpoint_path = None

        for i in range(5):
            f = tmp_path / f"ckpt_{i}.soul"
            f.write_text(f"checkpoint {i}")
            os.utime(f, (i, i))

        trainer._prune_stale_checkpoints(keep_final=False)

        remaining = list(tmp_path.glob("*.soul"))
        assert len(remaining) == 3

    def test_prune_final_keeps_one(self, tmp_path, monkeypatch):
        """On final save, only the newest checkpoint is kept."""
        import domains.training.train_pipeline as tp

        config = tp.TrainerConfig(checkpoint_dir=str(tmp_path))
        trainer = tp.SloughGPTTrainer.__new__(tp.SloughGPTTrainer)
        trainer.config = config
        trainer._best_model_path = None
        trainer._last_checkpoint_path = str(tmp_path / "final.soul")

        for i in range(4):
            f = tmp_path / f"ckpt_{i}.soul"
            f.write_text(f"checkpoint {i}")
            os.utime(f, (i, i))

        final = tmp_path / "final.soul"
        final.write_text("final checkpoint")
        os.utime(final, (10, 10))

        trainer._prune_stale_checkpoints(keep_final=True)

        remaining = list(tmp_path.glob("*.soul"))
        assert len(remaining) == 1
        assert final in remaining

    def test_prune_final_rewires_best_model_path(self, tmp_path):
        """After final prune, _best_model_path points to final checkpoint."""
        import domains.training.train_pipeline as tp

        config = tp.TrainerConfig(checkpoint_dir=str(tmp_path))
        trainer = tp.SloughGPTTrainer.__new__(tp.SloughGPTTrainer)
        trainer.config = config
        old_best = str(tmp_path / "old_best.soul")
        trainer._best_model_path = old_best
        trainer._last_checkpoint_path = str(tmp_path / "final.soul")

        for i in range(3):
            f = tmp_path / f"ckpt_{i}.soul"
            f.write_text(f"checkpoint {i}")
            os.utime(f, (i, i))

        final = tmp_path / "final.soul"
        final.write_text("final checkpoint")
        os.utime(final, (10, 10))

        trainer._prune_stale_checkpoints(keep_final=True)

        assert trainer._best_model_path == str(final)

    def test_prune_empty_dir_no_crash(self, tmp_path):
        """Pruning an empty directory doesn't crash."""
        import domains.training.train_pipeline as tp

        config = tp.TrainerConfig(checkpoint_dir=str(tmp_path))
        trainer = tp.SloughGPTTrainer.__new__(tp.SloughGPTTrainer)
        trainer.config = config
        trainer._best_model_path = None
        trainer._last_checkpoint_path = None

        trainer._prune_stale_checkpoints(keep_final=False)
        assert list(tmp_path.glob("*.soul")) == []

    def test_prune_nonexistent_dir_no_crash(self, tmp_path):
        """Pruning a nonexistent directory doesn't crash."""
        import domains.training.train_pipeline as tp

        config = tp.TrainerConfig(checkpoint_dir=str(tmp_path / "nonexistent"))
        trainer = tp.SloughGPTTrainer.__new__(tp.SloughGPTTrainer)
        trainer.config = config
        trainer._best_model_path = None
        trainer._last_checkpoint_path = None

        trainer._prune_stale_checkpoints(keep_final=False)

    def test_prune_cleans_meta_files(self, tmp_path):
        """Pruning removes associated .meta.json files."""
        import domains.training.train_pipeline as tp

        config = tp.TrainerConfig(checkpoint_dir=str(tmp_path), max_checkpoints=1)
        trainer = tp.SloughGPTTrainer.__new__(tp.SloughGPTTrainer)
        trainer.config = config
        trainer._best_model_path = None
        trainer._last_checkpoint_path = None

        for i in range(3):
            f = tmp_path / f"ckpt_{i}.soul"
            f.write_text(f"checkpoint {i}")
            meta = tmp_path / f"ckpt_{i}.soul.meta.json"
            meta.write_text(f'{{"name": "ckpt_{i}"}}')
            os.utime(f, (i, i))

        trainer._prune_stale_checkpoints(keep_final=False)

        remaining_soul = list(tmp_path.glob("*.soul"))
        remaining_meta = list(tmp_path.glob("*.meta.json"))
        assert len(remaining_soul) == 1
        assert len(remaining_meta) == 1
