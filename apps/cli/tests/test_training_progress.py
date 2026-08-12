"""Tests for the CLI TrainingProgressBar component."""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.training_progress import TrainingProgressBar, SPARK_CHARS


def _info(**overrides):
    base = {
        "global_step": 10,
        "epoch": 1,
        "epochs": 3,
        "steps_per_epoch": 100,
        "progress_percent": 3,
        "train_loss": 4.5,
        "eval_loss": None,
        "learning_rate": 0.001,
        "done": False,
        "done_reason": None,
    }
    base.update(overrides)
    return base


class TestInit:
    def test_defaults(self):
        bar = TrainingProgressBar()
        assert bar.desc == "Training"
        assert bar.total_steps is None
        assert bar.last_line == ""
        assert bar.stats == {}

    def test_explicit_total(self):
        bar = TrainingProgressBar(total_steps=2450)
        assert bar.total_steps == 2450

    def test_width_minimum(self):
        assert TrainingProgressBar(width=2).width == 8


class TestTotalInference:
    def test_from_epochs_and_steps_per_epoch(self, monkeypatch):
        monkeypatch.setattr("utils.training_progress._is_terminal", lambda: False)
        bar = TrainingProgressBar()
        bar.update(_info(global_step=10, epochs=3, steps_per_epoch=100, progress_percent=3))
        assert bar.total_steps == 300

    def test_from_progress_percent(self, monkeypatch):
        monkeypatch.setattr("utils.training_progress._is_terminal", lambda: False)
        bar = TrainingProgressBar()
        bar.update(_info(global_step=50, progress_percent=25, epochs=0, steps_per_epoch=0))
        assert bar.total_steps == 200

    def test_explicit_total_wins(self, monkeypatch):
        monkeypatch.setattr("utils.training_progress._is_terminal", lambda: False)
        bar = TrainingProgressBar(total_steps=100)
        bar.update(_info(global_step=10, epochs=3, steps_per_epoch=100))
        assert bar.total_steps == 100

    def test_refines_downward_when_max_steps_caps(self, monkeypatch):
        monkeypatch.setattr("utils.training_progress._is_terminal", lambda: False)
        bar = TrainingProgressBar(total_steps=2500)
        bar.update(_info(global_step=1000, progress_percent=50))
        assert bar.total_steps == 2000


class TestStats:
    def test_best_eval_tracked(self, monkeypatch):
        monkeypatch.setattr("utils.training_progress._is_terminal", lambda: False)
        bar = TrainingProgressBar()
        bar.update(_info(global_step=10, train_loss=4.0, eval_loss=4.2))
        bar.update(_info(global_step=20, train_loss=3.5, eval_loss=3.8))
        assert bar.stats["best_eval"] == 3.8
        assert bar.stats["eval_loss"] == 3.8

    def test_best_eval_ignores_worse(self, monkeypatch):
        monkeypatch.setattr("utils.training_progress._is_terminal", lambda: False)
        bar = TrainingProgressBar()
        bar.update(_info(global_step=10, eval_loss=3.5))
        bar.update(_info(global_step=20, eval_loss=3.9))
        assert bar.stats["best_eval"] == 3.5

    def test_stats_snapshot(self, monkeypatch):
        monkeypatch.setattr("utils.training_progress._is_terminal", lambda: False)
        bar = TrainingProgressBar(total_steps=100)
        bar.update(_info(global_step=50, epoch=2, train_loss=3.0, learning_rate=0.0005))
        assert bar.stats["step"] == 50
        assert bar.stats["pct"] == 3
        assert bar.stats["epoch"] == 2
        assert bar.stats["epochs"] == 3
        assert bar.stats["train_loss"] == 3.0
        assert bar.stats["learning_rate"] == 0.0005
        assert bar.stats["total_steps"] == 100
        assert bar.stats["done"] is False

    def test_done_update_flags_stats(self, monkeypatch):
        monkeypatch.setattr("utils.training_progress._is_terminal", lambda: False)
        bar = TrainingProgressBar(total_steps=100)
        bar.update(_info(global_step=100, progress_percent=99, done=True))
        assert bar.stats["done"] is True
        assert bar.stats["pct"] == 100


class TestSparkline:
    def test_shrinking_losses_descend(self, monkeypatch):
        monkeypatch.setattr("utils.training_progress._is_terminal", lambda: False)
        bar = TrainingProgressBar()
        for i in range(5):
            bar.update(_info(global_step=i + 1, train_loss=5.0 - i))
        spark = bar._sparkline()
        assert len(spark) == 5
        assert all(spark[i] >= spark[i + 1] for i in range(len(spark) - 1))
        assert spark[-1] == SPARK_CHARS[0]

    def test_flat_losses_use_mid_bar(self, monkeypatch):
        monkeypatch.setattr("utils.training_progress._is_terminal", lambda: False)
        bar = TrainingProgressBar()
        for i in range(4):
            bar.update(_info(global_step=i + 1, train_loss=2.5))
        assert bar._sparkline() == "▄▄▄▄"

    def test_no_sparkline_for_single_loss(self, monkeypatch):
        monkeypatch.setattr("utils.training_progress._is_terminal", lambda: False)
        bar = TrainingProgressBar()
        bar.update(_info(global_step=1, train_loss=4.0))
        assert bar._sparkline() == ""


class TestRendering:
    def test_line_contains_stats(self, monkeypatch):
        monkeypatch.setattr("utils.training_progress._is_terminal", lambda: False)
        bar = TrainingProgressBar(total_steps=100)
        bar.update(_info(global_step=50, epoch=2, train_loss=3.25, eval_loss=3.5, learning_rate=0.0005))
        line = bar.last_line
        assert "step 50/100" in line
        assert "ep 2/3" in line
        assert "loss 3.2500" in line
        assert "eval 3.5000" in line
        assert "lr 5.00e-04" in line
        assert "it/s" in line
        assert "eta" in line
        assert "%" in line

    def test_non_tty_prints_line(self, monkeypatch, capsys):
        monkeypatch.setattr("utils.training_progress._is_terminal", lambda: False)
        bar = TrainingProgressBar(total_steps=100)
        bar.update(_info(global_step=50))
        out = capsys.readouterr().out
        assert "step 50/100" in out

    def test_tty_uses_carriage_return(self, monkeypatch, capsys):
        monkeypatch.setattr("utils.training_progress._is_terminal", lambda: True)
        bar = TrainingProgressBar(total_steps=100)
        bar.update(_info(global_step=50))
        bar.finish()
        out = capsys.readouterr().out
        assert "\r" in out
        assert "\n" in out  # finish moves to a fresh line

    def test_finish_renders_at_100(self, monkeypatch):
        monkeypatch.setattr("utils.training_progress._is_terminal", lambda: False)
        bar = TrainingProgressBar(total_steps=100)
        bar.update(_info(global_step=50, progress_percent=50))
        bar.finish()
        assert "[██" in bar.last_line
        assert "100%" in bar.last_line

    def test_throttle_skips_rapid_updates(self, monkeypatch, capsys):
        monkeypatch.setattr("utils.training_progress._is_terminal", lambda: False)
        fake_now = {"t": 100.0}
        monkeypatch.setattr("utils.training_progress.time", type("T", (), {"time": lambda: fake_now["t"]}))
        bar = TrainingProgressBar(total_steps=100)
        bar.update(_info(global_step=10, progress_percent=10))
        fake_now["t"] = 100.1  # within throttle window (0.5s non-tty)
        bar.update(_info(global_step=20, progress_percent=20))
        fake_now["t"] = 100.8  # past throttle window
        bar.update(_info(global_step=30, progress_percent=30))
        out = capsys.readouterr().out
        assert out.count("\n") == 2
