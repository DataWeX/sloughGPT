"""Tests for training.turbo — get_turbo_status, start_turbo_training, run_turbo_worker."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

from domains.training import state as _state_mod
from domains.training.state import (
    _state, _turbo_lock, _turbo_cancel_event, _turbo_pause_event, _turbo_state,
    CHECKPOINTS_DIR, TURBO_DIR, REPO_ROOT,
)
from domains.training.turbo import get_turbo_status, start_turbo_training, run_turbo_worker


def _reset_turbo():
    with _turbo_lock:
        _turbo_state.clear()
        _turbo_state.update({
            "status": "idle",
            "job_id": "",
            "global_step": 0,
            "total_steps": 0,
            "progress": 0.0,
            "loss": None,
            "learning_rate": None,
            "steps_per_sec": None,
            "eta_s": None,
            "elapsed_s": None,
            "avg_quality": None,
            "result": None,
            "error": None,
            "paused": False,
            "last_heartbeat": 0,
        })
    _turbo_cancel_event.clear()
    _turbo_pause_event.clear()
    _state.running = False


# ── get_turbo_status ───────────────────────────────────────────────────────


class TestGetTurboStatus:

    def setup_method(self):
        _reset_turbo()

    def test_idle_status(self):
        result = get_turbo_status()
        assert result["status"] == "idle"

    def test_running_status(self):
        with _turbo_lock:
            _turbo_state["status"] = "running"
            _turbo_state["last_heartbeat"] = time.time()
        result = get_turbo_status()
        assert result["status"] == "running"

    def test_stale_heartbeat_errors(self):
        with _turbo_lock:
            _turbo_state["status"] = "running"
            _turbo_state["last_heartbeat"] = time.time() - 60
        result = get_turbo_status()
        assert result["status"] == "error"
        assert "30 seconds" in result["error"]

    def test_returns_copy(self):
        result = get_turbo_status()
        result["status"] = "mutated"
        assert _turbo_state["status"] != "mutated"


# ── start_turbo_training ──────────────────────────────────────────────────


class TestStartTurboTraining:

    def setup_method(self):
        _reset_turbo()

    def test_raises_if_already_running(self):
        with _turbo_lock:
            _turbo_state["status"] = "running"
        with pytest.raises(RuntimeError, match="already running"):
            start_turbo_training({"data_path": "/tmp/nonexistent"})

    def test_raises_no_data(self):
        with pytest.raises(ValueError, match="No data_path"):
            start_turbo_training({})

    def test_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Data file not found"):
            start_turbo_training({"data_path": "/tmp/nonexistent.jsonl"})

    def test_raises_checkpoint_not_found(self):
        tmp = Path("/tmp/test_turbo_data")
        tmp.mkdir(exist_ok=True)
        (tmp / "data.jsonl").write_text("{}\n")
        try:
            with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
                start_turbo_training({
                    "data_path": str(tmp / "data.jsonl"),
                    "checkpoint_name": "nonexistent",
                })
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def test_raises_data_outside_allowed_dirs(self):
        tmp = Path("/tmp/outside_datasets")
        tmp.mkdir(exist_ok=True)
        (tmp / "data.jsonl").write_text("{}\n")
        try:
            with pytest.raises(ValueError, match="must be under datasets/"):
                start_turbo_training({"data_path": str(tmp / "data.jsonl")})
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    @patch("domains.training.runtime_protocol.get_training_runtime")
    def test_success(self, mock_get_rt):
        mock_rt = MagicMock()
        mock_get_rt.return_value = mock_rt
        datasets_dir = REPO_ROOT / "datasets"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        data_file = datasets_dir / "test_turbo_data.jsonl"
        data_file.write_text('{"text": "hello"}\n')
        try:
            result = start_turbo_training({"data_path": str(data_file)})
            assert "job_id" in result
            assert result["data_path"] == str(data_file)
            assert result["resume"] is False
            assert isinstance(result["cancel_event"], threading.Event)
            mock_rt.register.assert_called_once()
        finally:
            data_file.unlink(missing_ok=True)

    @patch("domains.training.runtime_protocol.get_training_runtime")
    def test_resume(self, mock_get_rt):
        mock_rt = MagicMock()
        mock_get_rt.return_value = mock_rt
        datasets_dir = REPO_ROOT / "datasets"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        data_file = datasets_dir / "test_turbo_resume.jsonl"
        data_file.write_text('{"text": "hello"}\n')
        ckpt_dir = CHECKPOINTS_DIR
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        soul_file = ckpt_dir / "test_ckpt.soul"
        soul_file.write_text("fake checkpoint")
        try:
            result = start_turbo_training({
                "data_path": str(data_file),
                "checkpoint_name": "test_ckpt",
            })
            assert result["resume"] is True
            assert "test_ckpt.soul" in result["resume_path"]
        finally:
            data_file.unlink(missing_ok=True)
            soul_file.unlink(missing_ok=True)

    @patch("domains.training.runtime_protocol.get_training_runtime")
    @patch("domains.training.turbo.resolve_dataset_path")
    def test_dataset_id_fallback(self, mock_resolve, mock_get_rt):
        mock_get_rt.return_value = MagicMock()
        datasets_dir = REPO_ROOT / "datasets"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        data_file = datasets_dir / "ds_fallback.jsonl"
        data_file.write_text('{"text": "x"}\n')
        mock_resolve.return_value = str(data_file)
        try:
            result = start_turbo_training({"dataset_id": "ds_fallback"})
            assert result["data_path"] == str(data_file)
            mock_resolve.assert_called_once_with("ds_fallback")
        finally:
            data_file.unlink(missing_ok=True)

    @patch("domains.training.runtime_protocol.get_training_runtime")
    def test_sets_state_running(self, mock_get_rt):
        mock_rt = MagicMock()
        mock_get_rt.return_value = mock_rt
        datasets_dir = REPO_ROOT / "datasets"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        data_file = datasets_dir / "state_check.jsonl"
        data_file.write_text('{}\n')
        try:
            start_turbo_training({"data_path": str(data_file)})
            with _turbo_lock:
                assert _turbo_state["status"] == "running"
                assert _turbo_state["global_step"] == 0
                assert _turbo_state["progress"] == 0.0
        finally:
            data_file.unlink(missing_ok=True)


# ── run_turbo_worker ───────────────────────────────────────────────────────


class TestRunTurboWorker:

    def setup_method(self):
        _reset_turbo()

    @patch("domains.training.turbo.update_job")
    def test_no_data_path_errors(self, mock_update):
        with _turbo_lock:
            _turbo_state["job_id"] = "test_job"
        run_turbo_worker({})
        mock_update.assert_called_once()
        assert mock_update.call_args[1].get("status") == "error" or mock_update.call_args[0][1] == "error"

    @patch("domains.training.turbo.update_job")
    def test_no_data_path_with_dataset_id_fails(self, mock_update):
        with _turbo_lock:
            _turbo_state["job_id"] = "test_job"
        run_turbo_worker({"dataset_id": "nonexistent"})
        mock_update.assert_called_once()

    @patch("domains.training.turbo.update_job")
    @patch("domains.training.runtime_protocol.get_training_runtime")
    def test_trainer_exception(self, mock_get_rt, mock_update):
        mock_rt = MagicMock()
        mock_get_rt.return_value = mock_rt
        with _turbo_lock:
            _turbo_state["job_id"] = "test_job"

        with patch("domains.training.train_pipeline.SloughGPTTrainer", side_effect=RuntimeError("boom")):
            run_turbo_worker({"data_path": "/tmp/fake.jsonl"})

        with _turbo_lock:
            assert _turbo_state["status"] == "error"
            assert "boom" in _turbo_state["error"]

    @patch("domains.training.turbo.update_job")
    @patch("domains.training.runtime_protocol.get_training_runtime")
    def test_trainer_result_error(self, mock_get_rt, mock_update):
        mock_rt = MagicMock()
        mock_get_rt.return_value = mock_rt
        with _turbo_lock:
            _turbo_state["job_id"] = "test_job"

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {"status": "error", "message": "train failed"}
        with patch("domains.training.train_pipeline.SloughGPTTrainer", return_value=mock_trainer):
            run_turbo_worker({"data_path": "/tmp/fake.jsonl"})

        with _turbo_lock:
            assert _turbo_state["status"] == "error"
            assert _turbo_state["error"] == "train failed"

    @patch("domains.training.turbo.update_job")
    @patch("domains.training.runtime_protocol.get_training_runtime")
    def test_cancel_during_training(self, mock_get_rt, mock_update):
        mock_rt = MagicMock()
        mock_get_rt.return_value = mock_rt
        with _turbo_lock:
            _turbo_state["job_id"] = "test_job"

        def set_cancel(*args, **kwargs):
            _turbo_cancel_event.set()
            return {"status": "ok"}

        mock_trainer = MagicMock()
        mock_trainer.train.side_effect = set_cancel
        with patch("domains.training.train_pipeline.SloughGPTTrainer", return_value=mock_trainer):
            run_turbo_worker({"data_path": "/tmp/fake.jsonl"})

        with _turbo_lock:
            assert _turbo_state["status"] == "error"
            assert _turbo_state["error"] == "Training cancelled"

    @patch("domains.training.turbo.update_job")
    @patch("domains.training.runtime_protocol.get_training_runtime")
    def test_successful_training(self, mock_get_rt, mock_update):
        mock_rt = MagicMock()
        mock_get_rt.return_value = mock_rt
        with _turbo_lock:
            _turbo_state["job_id"] = "test_job"

        TURBO_DIR.mkdir(parents=True, exist_ok=True)
        soul_file = TURBO_DIR / "result.soul"
        soul_file.write_text("fake")

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {"loss": 0.5}
        with patch("domains.training.train_pipeline.SloughGPTTrainer", return_value=mock_trainer):
            run_turbo_worker({"data_path": "/tmp/fake.jsonl"})

        with _turbo_lock:
            assert _turbo_state["status"] == "complete"
            assert _turbo_state["progress"] == 100.0
        soul_file.unlink(missing_ok=True)

    @patch("domains.training.turbo.update_job")
    @patch("domains.training.runtime_protocol.get_training_runtime")
    def test_on_progress_callback(self, mock_get_rt, mock_update):
        mock_rt = MagicMock()
        mock_get_rt.return_value = mock_rt
        with _turbo_lock:
            _turbo_state["job_id"] = "test_job"

        TURBO_DIR.mkdir(parents=True, exist_ok=True)

        def fake_train(on_progress=None, **kwargs):
            if on_progress:
                on_progress({
                    "global_step": 10,
                    "total_steps": 100,
                    "progress_percent": 10.0,
                    "train_loss": 0.8,
                    "learning_rate": 0.001,
                    "steps_per_sec": 5.0,
                    "eta_s": 18.0,
                    "elapsed_s": 2.0,
                    "avg_quality": 0.7,
                })
            return {"loss": 0.5}

        mock_trainer = MagicMock()
        mock_trainer.train.side_effect = fake_train
        with patch("domains.training.train_pipeline.SloughGPTTrainer", return_value=mock_trainer):
            run_turbo_worker({"data_path": "/tmp/fake.jsonl"})

        with _turbo_lock:
            assert _turbo_state["global_step"] == 10
            assert _turbo_state["total_steps"] == 100
            assert _turbo_state["loss"] == 0.8
            assert _turbo_state["learning_rate"] == 0.001
            assert _turbo_state["steps_per_sec"] == 5.0
            assert _turbo_state["eta_s"] == 18.0
            assert _turbo_state["elapsed_s"] == 2.0
            assert _turbo_state["avg_quality"] == 0.7

    @patch("domains.training.turbo.update_job")
    @patch("domains.training.runtime_protocol.get_training_runtime")
    def test_state_running_false_after_worker(self, mock_get_rt, mock_update):
        mock_rt = MagicMock()
        mock_get_rt.return_value = mock_rt
        with _turbo_lock:
            _turbo_state["job_id"] = "test_job"

        mock_trainer = MagicMock()
        mock_trainer.train.side_effect = RuntimeError("fail")
        with patch("domains.training.train_pipeline.SloughGPTTrainer", return_value=mock_trainer):
            run_turbo_worker({"data_path": "/tmp/fake.jsonl"})

        assert _state.running is False
