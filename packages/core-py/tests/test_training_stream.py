"""Tests for training.stream — SSE stream completion handling."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from domains.training.stream import process_training_completion, cleanup_stream_state


# ── process_training_completion ────────────────────────────────────────────


class TestProcessTrainingCompletion:

    @patch("domains.training.stream.update_job")
    @patch("domains.training.stream.log_experiment_param")
    @patch("domains.training.stream.log_experiment_metric")
    def test_complete_status_sets_completed(self, mock_metric, mock_param, mock_update):
        mock_update.return_value = {"train_loss": 0.5, "global_step": 100}
        finish_fn = MagicMock()
        ev = {"status": "complete"}
        process_training_completion(ev, "t1", {}, Path("/tmp"), finish_fn)
        mock_update.assert_called_once_with("t1", status="completed", error="", checkpoint=None)
        finish_fn.assert_called_once_with("complete", "")

    @patch("domains.training.stream.update_job")
    def test_failed_status_sets_failed(self, mock_update):
        mock_update.return_value = {}
        finish_fn = MagicMock()
        ev = {"status": "failed", "message": "OOM"}
        process_training_completion(ev, "t1", {}, Path("/tmp"), finish_fn)
        mock_update.assert_called_once_with("t1", status="failed", error="OOM", checkpoint=None)
        finish_fn.assert_called_once_with("failed", "OOM")

    @patch("domains.training.stream.update_job")
    def test_failed_uses_data_fallback(self, mock_update):
        mock_update.return_value = {}
        finish_fn = MagicMock()
        ev = {"status": "failed", "data": "crash reason"}
        process_training_completion(ev, "t1", {}, Path("/tmp"), finish_fn)
        mock_update.assert_called_once_with("t1", status="failed", error="crash reason", checkpoint=None)

    @patch("domains.training.stream.update_job")
    def test_failed_uses_default_message(self, mock_update):
        mock_update.return_value = {}
        finish_fn = MagicMock()
        ev = {"status": "failed"}
        process_training_completion(ev, "t1", {}, Path("/tmp"), finish_fn)
        call_args = mock_update.call_args
        assert call_args[1]["error"] == "training failed"

    @patch("domains.training.stream.update_job")
    def test_finds_checkpoint_from_soul_files(self, mock_update, tmp_path):
        mock_update.return_value = {}
        (tmp_path / "model_a.soul").touch()
        (tmp_path / "model_b.soul").touch()
        finish_fn = MagicMock()
        ev = {"status": "complete"}
        process_training_completion(ev, "t1", {}, tmp_path, finish_fn)
        call_args = mock_update.call_args
        assert call_args[1]["checkpoint"] == str(tmp_path / "model_b.soul")

    @patch("domains.training.stream.update_job")
    def test_checkpoint_none_when_no_soul_files(self, mock_update, tmp_path):
        mock_update.return_value = {}
        finish_fn = MagicMock()
        ev = {"status": "complete"}
        process_training_completion(ev, "t1", {}, tmp_path, finish_fn)
        assert mock_update.call_args[1]["checkpoint"] is None

    @patch("domains.training.stream.update_job")
    @patch("domains.training.stream.log_experiment_param")
    @patch("domains.training.stream.log_experiment_metric")
    def test_logs_experiment_on_complete(self, mock_metric, mock_param, mock_update):
        mock_update.return_value = {"train_loss": 0.3, "global_step": 50}
        finish_fn = MagicMock()
        config = {"experiment_id": "exp-1", "epochs": 5, "learning_rate": 0.001}
        ev = {"status": "complete"}
        process_training_completion(ev, "t1", config, Path("/tmp"), finish_fn)
        mock_metric.assert_called_once_with("exp-1", "final_train_loss", 0.3, 50)
        assert mock_param.call_count == 2

    @patch("domains.training.stream.update_job")
    @patch("domains.training.stream.log_experiment_param")
    @patch("domains.training.stream.log_experiment_metric")
    def test_no_experiment_logging_when_failed(self, mock_metric, mock_param, mock_update):
        mock_update.return_value = {}
        finish_fn = MagicMock()
        config = {"experiment_id": "exp-1"}
        ev = {"status": "failed"}
        process_training_completion(ev, "t1", config, Path("/tmp"), finish_fn)
        mock_metric.assert_not_called()
        mock_param.assert_not_called()


# ── cleanup_stream_state ──────────────────────────────────────────────────


class TestCleanupStreamState:

    @patch("domains.training.runtime_protocol.get_training_runtime")
    @patch("domains.training.stream.update_job")
    def test_sets_running_false(self, mock_update, mock_runtime):
        mock_runtime.return_value.get.return_value = None
        state = {"running": True}
        finish_fn = MagicMock()
        cleanup_stream_state("t1", {}, state, finish_fn)
        assert state["running"] is False

    @patch("domains.training.runtime_protocol.get_training_runtime")
    @patch("domains.training.stream.update_job")
    def test_updates_non_terminal_job(self, mock_update, mock_runtime):
        mock_runtime.return_value.get.return_value = {"status": "running", "error": None}
        state = {"running": True}
        finish_fn = MagicMock()
        cleanup_stream_state("t1", {}, state, finish_fn)
        mock_update.assert_called_once()

    @patch("domains.training.runtime_protocol.get_training_runtime")
    @patch("domains.training.stream.update_job")
    def test_skips_terminal_job(self, mock_update, mock_runtime):
        mock_runtime.return_value.get.return_value = {"status": "completed"}
        state = {"running": True}
        finish_fn = MagicMock()
        cleanup_stream_state("t1", {}, state, finish_fn)
        mock_update.assert_not_called()

    @patch("domains.training.runtime_protocol.get_training_runtime")
    @patch("domains.training.stream.update_job")
    def test_skips_when_no_job(self, mock_update, mock_runtime):
        mock_runtime.return_value.get.return_value = None
        state = {"running": True}
        finish_fn = MagicMock()
        cleanup_stream_state("t1", {}, state, finish_fn)
        mock_update.assert_not_called()

    @patch("domains.training.runtime_protocol.get_training_runtime")
    @patch("domains.training.stream.update_job")
    def test_custom_status_and_error(self, mock_update, mock_runtime):
        mock_runtime.return_value.get.return_value = {"status": "running"}
        state = {"running": True}
        finish_fn = MagicMock()
        cleanup_stream_state("t1", {}, state, finish_fn, status="cancelled", error="user cancel")
        call_args = mock_update.call_args
        assert call_args[1]["status"] == "cancelled"
