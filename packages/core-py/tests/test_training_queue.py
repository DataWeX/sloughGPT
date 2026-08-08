"""
Tests for the task-queue training handlers.

Covers ``training_handler`` (native method: task queue → SloughGPTTrainer)
and ``training_sessions_handler`` (chat-trained method: train_from_sessions).
These are the executor path that ``GET /auto-train/stream`` dispatches to via
``register_training_handlers()``.
"""

import asyncio
import json

import numpy as np
import pytest

from domains.infrastructure.task_queue import Task
from domains.infrastructure.training_queue import (
    training_handler,
    training_sessions_handler,
)


def _make_task(payload, enqueue=None) -> Task:
    task = Task(
        name="auto-train",
        task_type="training",
        payload=payload,
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


def _tiny_text(blocks: int = 20) -> str:
    """Repeated meaningful text so token ids vary and training converges fast."""
    sentence = (
        "the quick brown fox jumps over the lazy dog near the river "
        "while the sun sets behind the hills and the birds sing softly. "
    )
    return (sentence * blocks)


class TestTrainingHandler:
    """task queue handler for SloughGPTTrainer (native method)."""

    def _payload(self, tmp_path, data_path) -> dict:
        return {
            "data_path": data_path,
            "checkpoint_dir": str(tmp_path / "ckpts"),
            "n_embed": 32,
            "n_layer": 1,
            "n_head": 4,
            "block_size": 32,
            "dropout": 0.0,
            "batch_size": 4,
            "epochs": 1,
            "learning_rate": 1e-3,
            "early_stopping_patience": 0,
        }

    def test_missing_enqueue_returns_failed(self):
        task = _make_task({"data_path": "/nonexistent"}, enqueue=None)
        result = asyncio.run(training_handler(task))
        assert result["status"] == "failed"
        assert "No enqueue callback" in result["error"]

    def test_runs_real_training_emits_sse_and_writes_checkpoint(self, tmp_path):
        data_path = tmp_path / "input.txt"
        data_path.write_text(_tiny_text(20), encoding="utf-8")
        events, enqueue = _collect_events(None)
        task = _make_task(self._payload(tmp_path, str(data_path)), enqueue=enqueue)

        result = asyncio.run(training_handler(task))

        assert result.success is True
        assert result["success"] is True
        train_events = [e for e in events if e.get("phase") == "TRAIN"]
        assert train_events, "expected at least one TRAIN SSE event"
        first = train_events[0]
        assert first["stream"] == "auto-train"
        assert first["status"] == "working"
        assert "loss" in first["data"]
        assert "step" in first["data"]
        assert "progress" in first["data"]
        ckpts = list((tmp_path / "ckpts").glob("*.soul"))
        assert ckpts, "training should write a .soul checkpoint"

    def test_defaults_applied_with_minimal_payload(self, tmp_path, monkeypatch):
        import domains.training.train_pipeline as tp
        from domains.training.trainer_protocol import TrainResult

        captured = {}

        class FakeConfig:
            def __init__(self, **kwargs):
                captured["config_kwargs"] = kwargs

        class FakeTrainer:
            def __init__(self, data_path, config):
                captured["data_path"] = data_path

            def train(self, **kwargs):
                captured["train_kwargs"] = kwargs
                return TrainResult(success=True)

        monkeypatch.setattr(tp, "TrainerConfig", FakeConfig)
        monkeypatch.setattr(tp, "SloughGPTTrainer", FakeTrainer)

        data_path = tmp_path / "input.txt"
        data_path.write_text(_tiny_text(20), encoding="utf-8")
        events, enqueue = _collect_events(None)
        task = _make_task(
            {"data_path": str(data_path), "checkpoint_dir": str(tmp_path / "ckpts")},
            enqueue=enqueue,
        )
        result = asyncio.run(training_handler(task))

        assert result.success is True
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
        assert cfg["checkpoint_dir"] == str(tmp_path / "ckpts")
        assert captured["data_path"] == str(data_path)
        assert captured["train_kwargs"]["resume"] is False
        assert captured["train_kwargs"]["resume_path"] == ""

    def test_cancel_propagates_to_cancelled_result(self, tmp_path):
        data_path = tmp_path / "input.txt"
        data_path.write_text(_tiny_text(40), encoding="utf-8")
        events, enqueue = _collect_events(None)
        task = _make_task(self._payload(tmp_path, str(data_path)), enqueue=enqueue)
        task.cancel_event.set()

        result = asyncio.run(training_handler(task))

        assert result["status"] == "cancelled"
        completes = [e for e in events if e.get("status") == "complete"]
        assert completes
        assert completes[-1]["data"].get("cancelled") is True

    def test_trainer_failure_emits_error(self, tmp_path):
        events, enqueue = _collect_events(None)
        payload = self._payload(tmp_path, "/nonexistent/input.txt")
        task = _make_task(payload, enqueue=enqueue)

        result = asyncio.run(training_handler(task))

        assert result["status"] == "failed"
        errors = [e for e in events if e.get("status") == "error"]
        assert errors
        assert errors[-1]["stream"] == "auto-train"


class TestTrainingSessionsHandler:
    """task queue handler for train_from_sessions (chat-trained method)."""

    def _payload(self, tmp_path) -> dict:
        return {
            "checkpoint_dir": str(tmp_path / "ckpts"),
            "n_embed": 16,
            "n_layer": 1,
            "n_head": 2,
            "block_size": 16,
            "dropout": 0.0,
            "epochs": 1,
            "learning_rate": 1e-3,
            "batch_size": 8,
            "soul_name": "test-chat",
            "session_ids": ["s1"],
        }

    def _write_sessions(self, tmp_path, monkeypatch):
        sess_dir = tmp_path / "sessions"
        sess_dir.mkdir()
        (sess_dir / "s1.json").write_text(
            '{"messages": ['
            '{"role": "user", "content": "User message number one asking something interesting."},'
            '{"role": "assistant", "content": "Assistant responds helpfully with a detailed answer about topic one."},'
            '{"role": "user", "content": "User message number two asking something interesting."},'
            '{"role": "assistant", "content": "Assistant responds helpfully with a detailed answer about topic two."}'
            "]}",
            encoding="utf-8",
        )
        monkeypatch.setattr("domains.training.pair_extractor._SESSIONS_DIR", sess_dir)

    def test_missing_enqueue_returns_failed(self):
        task = _make_task({"session_ids": ["s1"]}, enqueue=None)
        result = asyncio.run(training_sessions_handler(task))
        assert result["status"] == "failed"
        assert "No enqueue callback" in result["error"]

    def test_runs_sessions_training_emits_events_and_checkpoint(self, tmp_path, monkeypatch):
        self._write_sessions(tmp_path, monkeypatch)
        events, enqueue = _collect_events(None)
        task = _make_task(self._payload(tmp_path), enqueue=enqueue)

        result = asyncio.run(training_sessions_handler(task))

        assert result["num_pairs"] >= 1
        assert "perplexity" in result or "samples" in result
        phases = {e.get("phase") for e in events}
        assert "PAIRS" in phases
        train_events = [e for e in events if e.get("phase") == "TRAIN"]
        assert train_events
        ckpts = list((tmp_path / "ckpts").glob("*.soul"))
        assert ckpts

    def test_cancel_propagates_to_cancelled(self, tmp_path, monkeypatch):
        self._write_sessions(tmp_path, monkeypatch)
        events, enqueue = _collect_events(None)
        task = _make_task(self._payload(tmp_path), enqueue=enqueue)
        task.cancel_event.set()

        result = asyncio.run(training_sessions_handler(task))

        assert result["status"] == "cancelled"
        completes = [e for e in events if e.get("status") == "complete"]
        assert completes
        assert completes[-1]["data"].get("cancelled") is True

    def test_no_pairs_raises_failed(self, tmp_path, monkeypatch):
        sess_dir = tmp_path / "sessions"
        sess_dir.mkdir()
        (sess_dir / "s1.json").write_text('{"messages": []}', encoding="utf-8")
        monkeypatch.setattr("domains.training.pair_extractor._SESSIONS_DIR", sess_dir)
        monkeypatch.setattr(
            "domains.training.pair_extractor.extract_pairs_from_corpus",
            lambda limit=None: [],
        )
        events, enqueue = _collect_events(None)
        task = _make_task(self._payload(tmp_path), enqueue=enqueue)

        result = asyncio.run(training_sessions_handler(task))

        assert result["status"] == "failed"
        errors = [e for e in events if e.get("status") == "error"]
        assert errors
