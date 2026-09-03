"""Tests for TrainingUX — programmatic training log formatter."""

from __future__ import annotations

import os
import sys
import time
from unittest.mock import MagicMock, call

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from domains.training.training_ux import TrainingUX


@pytest.fixture
def log():
    return MagicMock()


@pytest.fixture
def ux(log):
    return TrainingUX(log, total_params=124_439_808)


@pytest.fixture
def ux_no_params(log):
    return TrainingUX(log)


@pytest.fixture
def ux_structured(log):
    cb = MagicMock()
    return TrainingUX(log, total_params=1_000_000, on_structured=cb), cb


# ── _fmt_num ────────────────────────────────────────────────────────

class TestFmtNum:
    def test_normal_float(self, ux):
        assert ux._fmt_num(3.14159, 2) == "3.14"

    def test_small_number_scientific(self, ux):
        assert ux._fmt_num(0.0005, 2) == "5.0e-04"

    def test_zero(self, ux):
        assert ux._fmt_num(0, 2) == "0.00"

    def test_integer(self, ux):
        assert ux._fmt_num(42, 2) == "42.00"

    def test_string_input(self, ux):
        assert ux._fmt_num("abc", 2) == "abc"

    def test_none_input(self, ux):
        assert ux._fmt_num(None, 2) == "None"

    def test_high_precision(self, ux):
        assert ux._fmt_num(0.123456789, 6) == "0.123457"


# ── _fmt_eta ────────────────────────────────────────────────────────

class TestFmtEta:
    def test_seconds_only(self, ux):
        assert ux._fmt_eta(90) == "1:30"

    def test_hours_minutes_seconds(self, ux):
        assert ux._fmt_eta(3661) == "1:01:01"

    def test_exact_hour(self, ux):
        assert ux._fmt_eta(3600) == "1:00:00"

    def test_zero(self, ux):
        assert ux._fmt_eta(0) == "0:00"

    def test_none(self, ux):
        assert ux._fmt_eta(None) == "??:??"

    def test_negative(self, ux):
        assert ux._fmt_eta(-5) == "??:??"

    def test_large(self, ux):
        assert ux._fmt_eta(7200) == "2:00:00"


# ── _fmt_params ─────────────────────────────────────────────────────

class TestFmtParams:
    def test_millions(self, ux):
        assert ux._fmt_params(124_439_808) == "124.4M"

    def test_billions(self, ux):
        assert ux._fmt_params(2_500_000_000) == "2.5B"

    def test_thousands(self, ux):
        assert ux._fmt_params(15_000) == "15.0K"

    def test_small(self, ux):
        assert ux._fmt_params(42) == "42"

    def test_exact_million(self, ux):
        assert ux._fmt_params(1_000_000) == "1.0M"

    def test_exact_thousand(self, ux):
        assert ux._fmt_params(1_000) == "1.0K"


# ── on_config ───────────────────────────────────────────────────────

class TestOnConfig:
    def test_logs_header(self, ux, log):
        ux.on_config({"epochs": 10, "lr": 3e-4, "block_size": 128, "dataset": "shakespeare"})
        log.header.assert_called_once_with("Training")

    def test_logs_model_name(self, ux, log):
        ux.on_config({"model_name": "gpt2"})
        log.key_value.assert_any_call("Model", "gpt2")

    def test_logs_dataset(self, ux, log):
        ux.on_config({"dataset": "shakespeare"})
        log.key_value.assert_any_call("Dataset", "shakespeare")

    def test_logs_epochs(self, ux, log):
        ux.on_config({"epochs": 10})
        log.key_value.assert_any_call("Epochs", "10")

    def test_logs_lr(self, ux, log):
        ux.on_config({"lr": 0.0003})
        # lr is formatted with decimals=1e-4
        log.key_value.assert_any_call("Learning rate", ux._fmt_num(0.0003, decimals=1e-4))

    def test_logs_block_size(self, ux, log):
        ux.on_config({"block_size": 128})
        log.key_value.assert_any_call("Block size", "128")

    def test_logs_batch_size(self, ux, log):
        ux.on_config({"batch_size": 32})
        log.key_value.assert_any_call("Batch size", "32")

    def test_logs_params(self, ux, log):
        ux.on_config({})
        log.key_value.assert_any_call("Parameters", "124.4M")

    def test_no_params_when_not_set(self, ux_no_params, log):
        ux_no_params.on_config({})
        for c in log.key_value.call_args_list:
            assert c[0][0] != "Parameters"

    def test_sets_train_start(self, ux):
        ux.on_config({})
        assert ux._train_start is not None


# ── on_epoch_start ──────────────────────────────────────────────────

class TestOnEpochStart:
    def test_logs_section(self, ux, log):
        ux.on_epoch_start(1, 10, 500)
        log.section.assert_called_once_with("Epoch 1/10 (500 steps)")

    def test_sets_epoch_start(self, ux):
        ux.on_epoch_start(1, 10, 500)
        assert ux._epoch_start is not None


# ── on_progress ─────────────────────────────────────────────────────

class TestOnProgress:
    def test_logs_step(self, ux, log):
        ux.on_progress({"global_step": 100, "total_steps": 1000})
        log.info.assert_called_once()
        msg = log.info.call_args[0][0]
        assert "Step 100/1000" in msg

    def test_logs_loss(self, ux, log):
        ux.on_progress({"global_step": 100, "train_loss": 4.23})
        msg = log.info.call_args[0][0]
        assert "loss" in msg

    def test_logs_lr(self, ux, log):
        ux.on_progress({"global_step": 100, "learning_rate": 0.001})
        msg = log.info.call_args[0][0]
        assert "lr" in msg

    def test_logs_eta(self, ux, log):
        ux.on_progress({"global_step": 100, "eta_s": 120})
        msg = log.info.call_args[0][0]
        assert "ETA" in msg

    def test_logs_percent(self, ux, log):
        ux.on_progress({"global_step": 100, "progress_percent": 50})
        msg = log.info.call_args[0][0]
        assert "50%" in msg

    def test_logs_speed(self, ux, log):
        ux.on_progress({"global_step": 100, "steps_per_sec": 10.5})
        msg = log.info.call_args[0][0]
        assert "s/s" in msg

    def test_updates_last_loss(self, ux):
        ux.on_progress({"global_step": 100, "train_loss": 3.5})
        assert ux._last_loss == 3.5

    def test_increments_step_count(self, ux):
        assert ux._step_count == 0
        ux.on_progress({"global_step": 1})
        assert ux._step_count == 1
        ux.on_progress({"global_step": 2})
        assert ux._step_count == 2


# ── on_eval ─────────────────────────────────────────────────────────

class TestOnEval:
    def test_logs_eval(self, ux, log):
        ux.on_eval({"eval_loss": 3.87, "eval_ppl": 48.2, "global_step": 100})
        log.success.assert_called_once()
        msg = log.success.call_args[0][0]
        assert "Eval @100" in msg
        assert "loss" in msg
        assert "ppl" in msg

    def test_updates_best_loss(self, ux):
        ux.on_eval({"eval_loss": 3.87})
        assert ux._best_loss == 3.87

    def test_best_loss_improves(self, ux):
        ux.on_eval({"eval_loss": 3.87})
        ux.on_eval({"eval_loss": 3.50})
        assert ux._best_loss == 3.50

    def test_best_loss_does_not_worsen(self, ux):
        ux.on_eval({"eval_loss": 3.50})
        ux.on_eval({"eval_loss": 3.87})
        assert ux._best_loss == 3.50


# ── on_checkpoint ───────────────────────────────────────────────────

class TestOnCheckpoint:
    def test_logs_checkpoint(self, ux, log):
        ux.on_checkpoint({"path": "models/best.soul", "step": 500})
        log.step.assert_called_once_with("Checkpoint saved: models/best.soul")


# ── on_complete ─────────────────────────────────────────────────────

class TestOnComplete:
    def test_logs_header(self, ux, log):
        ux.on_complete({})
        log.header.assert_called_with("Results")

    def test_logs_steps(self, ux, log):
        ux.on_complete({"total_steps": 1000})
        log.key_value.assert_any_call("Steps", "1000")

    def test_logs_epochs(self, ux, log):
        ux.on_complete({"epochs_completed": 10})
        log.key_value.assert_any_call("Epochs", "10")

    def test_logs_best_loss(self, ux, log):
        ux.on_eval({"eval_loss": 3.5})  # set best_loss
        ux.on_complete({})
        log.key_value.assert_any_call("Best loss", "3.5000")

    def test_logs_final_loss(self, ux, log):
        ux.on_complete({"final_loss": 3.6})
        log.key_value.assert_any_call("Final loss", "3.6000")

    def test_logs_model_path(self, ux, log):
        ux.on_complete({"model_path": "models/shakespeare.soul"})
        log.key_value.assert_any_call("Model", "models/shakespeare.soul")

    def test_logs_duration(self, ux, log):
        ux._train_start = time.time() - 10  # 10 seconds ago
        ux.on_complete({})
        # Duration should be called with a formatted time
        duration_calls = [c for c in log.key_value.call_args_list if c[0][0] == "Duration"]
        assert len(duration_calls) == 1


# ── on_error ────────────────────────────────────────────────────────

class TestOnError:
    def test_logs_error(self, ux, log):
        ux.on_error({"error": "CUDA out of memory"})
        log.error.assert_called_once_with("Training failed: CUDA out of memory")

    def test_default_error_message(self, ux, log):
        ux.on_error({})
        log.error.assert_called_once_with("Training failed: Unknown error")


# ── on_cancel ───────────────────────────────────────────────────────

class TestOnCancel:
    def test_logs_cancel(self, ux, log):
        ux.on_cancel({"global_step": 500})
        log.warning.assert_called_once_with("Training cancelled at step 500")

    def test_default_step(self, ux, log):
        ux.on_cancel({})
        log.warning.assert_called_once_with("Training cancelled at step ?")


# ── _emit ───────────────────────────────────────────────────────────

class TestEmit:
    def test_emits_structured_data(self, ux_structured):
        ux, cb = ux_structured
        ux.on_config({"epochs": 10})
        cb.assert_called_once()
        data = cb.call_args[0][0]
        assert data["event"] == "config"
        assert data["epochs"] == 10

    def test_emit_silences_exceptions(self, log):
        bad_cb = MagicMock(side_effect=RuntimeError("boom"))
        ux = TrainingUX(log, on_structured=bad_cb)
        # Should not raise
        ux.on_config({"epochs": 1})

    def test_no_emit_when_no_callback(self, ux_no_params, log):
        # Should not raise when no callback is set
        ux_no_params.on_config({"epochs": 1})
