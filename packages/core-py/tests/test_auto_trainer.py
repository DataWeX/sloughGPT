"""Tests for AutoTrainer (background training from inference logs)."""

import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from domains.training.auto_trainer import AutoTrainer


class TestAutoTrainer:
    def test_init_defaults(self):
        """Default config values."""
        t = AutoTrainer()
        assert t.threshold == 10
        assert t.interval_s == 300
        assert t._conversation_count == 0
        assert t._total_trains == 0

    def test_init_custom(self):
        """Custom config values."""
        t = AutoTrainer(threshold=5, interval_s=60)
        assert t.threshold == 5
        assert t.interval_s == 60

    def test_start_stop(self):
        """Start and stop the background thread."""
        t = AutoTrainer(interval_s=9999)
        t.start()
        assert t._thread is not None
        assert t._thread.is_alive()
        t.stop()
        assert not t._thread.is_alive()

    def test_start_idempotent(self):
        """Calling start twice doesn't create a second thread."""
        t = AutoTrainer(interval_s=9999)
        t.start()
        first = t._thread
        t.start()
        assert t._thread is first
        t.stop()

    def test_status_structure(self):
        """status() returns all expected fields."""
        t = AutoTrainer(threshold=5, interval_s=60)
        s = t.status()
        assert "enabled" in s
        assert "threshold" in s
        assert "interval_s" in s
        assert "pending_conversations" in s
        assert "total_trains" in s
        assert "last_train" in s
        assert "last_loss" in s
        assert "last_checkpoint" in s
        assert s["threshold"] == 5
        assert s["interval_s"] == 60
        assert s["enabled"] is False

    def test_status_running(self):
        """status() shows enabled=True when running."""
        t = AutoTrainer(interval_s=9999)
        t.start()
        assert t.status()["enabled"] is True
        t.stop()

    @patch("domains.training.auto_trainer.AutoTrainer._do_train")
    def test_check_no_new_data(self, mock_train):
        """No training when no new files detected."""
        t = AutoTrainer()
        t._check_and_train()
        mock_train.assert_not_called()

    @patch("domains.training.auto_trainer.AutoTrainer._do_train")
    def test_check_below_threshold(self, mock_train):
        """No training when conversations below threshold."""
        t = AutoTrainer(threshold=10)
        t._sessions_mtime = 0
        t._logs_mtime = 0
        with patch.object(AutoTrainer, "_dir_mtime", return_value=100.0):
            t._check_and_train()
        mock_train.assert_not_called()
        assert t._conversation_count == 1

    @patch("domains.training.auto_trainer.AutoTrainer._do_train", return_value=True)
    def test_check_at_threshold(self, mock_train):
        """Training triggered at threshold."""
        t = AutoTrainer(threshold=3, interval_s=0)
        t._sessions_mtime = 0
        t._logs_mtime = 0
        counter = [100.0]
        def incr_mtime(_):
            counter[0] += 1
            return counter[0]
        with patch.object(AutoTrainer, "_dir_mtime", side_effect=incr_mtime):
            for _ in range(3):
                t._check_and_train()
        mock_train.assert_called()

    @patch("domains.training.auto_trainer.AutoTrainer._do_train", return_value=True)
    def test_check_interval_respected(self, mock_train):
        """Training not triggered if interval hasn't elapsed."""
        t = AutoTrainer(threshold=1, interval_s=9999)
        t._last_train_ts = time.time()
        t._sessions_mtime = 0
        t._logs_mtime = 0
        with patch.object(AutoTrainer, "_dir_mtime", return_value=100.0):
            t._check_and_train()
        # _do_train should NOT be called because interval hasn't elapsed
        # The conversation count is now 1 and threshold is 1,
        # but _last_train_ts is recent → interval check fails
        mock_train.assert_not_called()

    def test_dir_mtime_nonexistent(self, tmp_path):
        """dir_mtime returns 0 for nonexistent directory."""
        result = AutoTrainer._dir_mtime(tmp_path / "nope")
        assert result == 0

    def test_dir_mtime_empty_dir(self, tmp_path):
        """dir_mtime returns 0 for empty directory."""
        result = AutoTrainer._dir_mtime(tmp_path)
        assert result == 0

    def test_dir_mtime_with_files(self, tmp_path):
        """dir_mtime returns latest mtime."""
        (tmp_path / "a.txt").write_text("a")
        t1 = tmp_path.stat().st_mtime
        time.sleep(0.01)
        (tmp_path / "b.txt").write_text("b")
        result = AutoTrainer._dir_mtime(tmp_path)
        assert result > t1

    def test_stop_when_not_started(self):
        """stop() is safe to call when not started."""
        t = AutoTrainer()
        t.stop()  # Should not raise

    @patch("domains.training.pair_extractor.write_training_text")
    @patch("domains.training.pair_extractor.extract_pairs_from_sessions", return_value=[
        {"user_msg": "Hello", "assistant_msg": "Hi there!", "session_id": "s1"},
        {"user_msg": "Bye", "assistant_msg": "Goodbye!", "session_id": "s1"},
        {"user_msg": "Thanks", "assistant_msg": "You're welcome!", "session_id": "s1"},
        {"user_msg": "Test", "assistant_msg": "Result!", "session_id": "s1"},
        {"user_msg": "One", "assistant_msg": "More!", "session_id": "s1"},
    ])
    @patch("domains.training.pair_extractor.extract_pairs_from_logs", return_value=[])
    def test_do_train_success(self, mock_logs, mock_sessions, mock_write, tmp_path):
        """Successful training updates state."""
        mock_write.return_value = tmp_path / "train.txt"
        # Create .venv/bin/python3 stub so the venv check passes
        venv = tmp_path / ".venv" / "bin"
        venv.mkdir(parents=True)
        (venv / "python3").touch()
        (tmp_path / "models" / "auto-training").mkdir(parents=True, exist_ok=True)

        t = AutoTrainer(threshold=5, interval_s=0)
        t._conversation_count = 5

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout='{"success": true, "loss": 2.5, "steps": 10}\n',
                stderr="",
            )
            with patch("domains.training.auto_trainer._REPO_ROOT", tmp_path):
                result = t._do_train()

        assert result is True
        assert t._total_trains == 1
        assert t._conversation_count == 0
        assert t._last_train_loss == 2.5

    @patch("domains.training.pair_extractor.extract_pairs_from_sessions", return_value=[])
    @patch("domains.training.pair_extractor.extract_pairs_from_corpus", return_value=[])
    @patch("domains.training.pair_extractor.extract_pairs_from_logs", return_value=[])
    def test_do_train_insufficient_pairs(self, mock_logs, mock_corpus, mock_sessions):
        """Training skipped when fewer than 5 pairs found."""
        t = AutoTrainer()
        t._conversation_count = 5
        result = t._do_train()
        assert result is False
        assert t._conversation_count == 0

    def test_do_train_venv_missing(self, tmp_path, monkeypatch):
        """Training skipped when .venv Python doesn't exist."""
        monkeypatch.setattr("domains.training.auto_trainer._REPO_ROOT", tmp_path)
        t = AutoTrainer()
        t._conversation_count = 5

        with patch(
            "domains.training.pair_extractor.extract_pairs_from_sessions", return_value=[
                {"user_msg": f"Q{i}", "assistant_msg": f"A{i} long enough", "session_id": "s1"}
                for i in range(5)
            ]
        ), patch(
            "domains.training.pair_extractor.write_training_text", return_value=tmp_path / "train.txt"
        ):
            result = t._do_train()
        assert result is False

    def test_loop_logs_exception(self, caplog):
        """The monitoring loop logs exceptions from _check_and_train."""
        t = AutoTrainer()
        with patch.object(t, "_check_and_train", side_effect=RuntimeError("boom")), \
                patch.object(t._stop_event, "wait", side_effect=lambda *_: t._stop_event.set()):
            t._loop()
        assert t._stop_event.is_set()

    def test_check_no_new_data_returns_early(self):
        """_check_and_train returns immediately when mtimes unchanged."""
        t = AutoTrainer()
        with patch.object(AutoTrainer, "_dir_mtime", return_value=42.0):
            t._sessions_mtime = 42.0
            t._logs_mtime = 42.0
            t._corpus_mtime = 42.0
            t._conversation_count = 0
            t._check_and_train()
        assert t._conversation_count == 0

    @patch("domains.training.pair_extractor.extract_pairs_from_sessions", return_value=[])
    @patch("domains.training.pair_extractor.extract_pairs_from_corpus", return_value=[])
    @patch("domains.training.pair_extractor.extract_pairs_from_logs", return_value=[
        {"user_msg": f"Q{i}", "assistant_msg": f"A{i} long enough", "session_id": "s1"}
        for i in range(5)
    ])
    @patch("domains.training.pair_extractor.write_training_text")
    def test_do_train_falls_back_to_logs(self, mock_write, mock_logs, mock_corpus, mock_sessions, tmp_path):
        """Training falls back to response logs when sessions/corpus are empty."""
        mock_write.return_value = tmp_path / "train.txt"
        venv = tmp_path / ".venv" / "bin"
        venv.mkdir(parents=True)
        (venv / "python3").touch()
        (tmp_path / "models" / "auto-training").mkdir(parents=True, exist_ok=True)

        t = AutoTrainer(threshold=5, interval_s=0)
        t._conversation_count = 5
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout='{"success": true, "loss": 1.5, "steps": 5}\n',
                stderr="",
            )
            with patch("domains.training.auto_trainer._REPO_ROOT", tmp_path):
                result = t._do_train()
        assert result is True
        assert t._total_trains == 1

    @patch("domains.training.pair_extractor.extract_pairs_from_sessions", return_value=[])
    @patch("domains.training.pair_extractor.extract_pairs_from_corpus", return_value=[])
    @patch("domains.training.pair_extractor.extract_pairs_from_logs", return_value=[
        {"user_msg": f"Q{i}", "assistant_msg": f"A{i} long enough", "session_id": "s1"}
        for i in range(5)
    ])
    @patch("domains.training.pair_extractor.write_training_text")
    def test_do_train_subprocess_failed(self, mock_write, mock_logs, mock_corpus, mock_sessions, tmp_path):
        """Training reports False when the subprocess exits non-zero."""
        mock_write.return_value = tmp_path / "train.txt"
        venv = tmp_path / ".venv" / "bin"
        venv.mkdir(parents=True)
        (venv / "python3").touch()
        (tmp_path / "models" / "auto-training").mkdir(parents=True, exist_ok=True)

        t = AutoTrainer(threshold=5, interval_s=0)
        t._conversation_count = 5
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="train failed")
            with patch("domains.training.auto_trainer._REPO_ROOT", tmp_path):
                result = t._do_train()
        assert result is False
        assert t._total_trains == 0

    @patch("domains.training.pair_extractor.extract_pairs_from_sessions", return_value=[
        {"user_msg": f"Q{i}", "assistant_msg": f"A{i} long enough", "session_id": "s1"}
        for i in range(5)
    ])
    @patch("domains.training.pair_extractor.write_training_text")
    def test_do_train_store_failure_is_logged(self, mock_write, mock_sessions, tmp_path):
        """Store failures are logged without failing the training run."""
        mock_write.return_value = tmp_path / "train.txt"
        venv = tmp_path / ".venv" / "bin"
        venv.mkdir(parents=True)
        (venv / "python3").touch()
        (tmp_path / "models" / "auto-training").mkdir(parents=True, exist_ok=True)

        t = AutoTrainer(threshold=5, interval_s=0)
        t._conversation_count = 5
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout='{"success": true, "loss": 1.5, "steps": 5}\n',
                stderr="",
            )
            with patch("domains.training.auto_trainer._REPO_ROOT", tmp_path), \
                    patch("domains.training.quality_scorer.score_batch", side_effect=RuntimeError("db down")):
                result = t._do_train()
        assert result is True
        assert t._total_trains == 1

    @patch("domains.training.pair_extractor.extract_pairs_from_sessions", return_value=[
        {"user_msg": f"Q{i}", "assistant_msg": f"A{i} long enough", "session_id": "s1"}
        for i in range(5)
    ])
    @patch("domains.training.pair_extractor.write_training_text")
    def test_do_train_result_not_success(self, mock_write, mock_sessions, tmp_path):
        """Training reports False when the subprocess result is not successful."""
        mock_write.return_value = tmp_path / "train.txt"
        venv = tmp_path / ".venv" / "bin"
        venv.mkdir(parents=True)
        (venv / "python3").touch()
        (tmp_path / "models" / "auto-training").mkdir(parents=True, exist_ok=True)

        t = AutoTrainer(threshold=5, interval_s=0)
        t._conversation_count = 5
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout='{"success": false, "error": "loss diverged"}\n',
                stderr="",
            )
            with patch("domains.training.auto_trainer._REPO_ROOT", tmp_path):
                result = t._do_train()
        assert result is False
        assert t._total_trains == 0

    @patch("domains.training.pair_extractor.extract_pairs_from_sessions", return_value=[
        {"user_msg": f"Q{i}", "assistant_msg": f"A{i} long enough", "session_id": "s1"}
        for i in range(5)
    ])
    @patch("domains.training.pair_extractor.write_training_text")
    def test_do_train_subprocess_timeout(self, mock_write, mock_sessions, tmp_path):
        """Training reports False when the subprocess times out."""
        mock_write.return_value = tmp_path / "train.txt"
        venv = tmp_path / ".venv" / "bin"
        venv.mkdir(parents=True)
        (venv / "python3").touch()
        (tmp_path / "models" / "auto-training").mkdir(parents=True, exist_ok=True)

        t = AutoTrainer(threshold=5, interval_s=0)
        t._conversation_count = 5
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("hf_train.py", 300)), \
                patch("domains.training.auto_trainer._REPO_ROOT", tmp_path):
            result = t._do_train()
        assert result is False

    @patch("domains.training.pair_extractor.extract_pairs_from_sessions", return_value=[
        {"user_msg": f"Q{i}", "assistant_msg": f"A{i} long enough", "session_id": "s1"}
        for i in range(5)
    ])
    @patch("domains.training.pair_extractor.write_training_text")
    def test_do_train_subprocess_other_error(self, mock_write, mock_sessions, tmp_path):
        """Training reports False on an unexpected subprocess error."""
        mock_write.return_value = tmp_path / "train.txt"
        venv = tmp_path / ".venv" / "bin"
        venv.mkdir(parents=True)
        (venv / "python3").touch()
        (tmp_path / "models" / "auto-training").mkdir(parents=True, exist_ok=True)

        t = AutoTrainer(threshold=5, interval_s=0)
        t._conversation_count = 5
        with patch("subprocess.run", side_effect=OSError("no python")), \
                patch("domains.training.auto_trainer._REPO_ROOT", tmp_path):
            result = t._do_train()
        assert result is False


class TestAutoTrainerSingleton:
    def test_get_auto_trainer_creates_singleton(self, monkeypatch):
        """get_auto_trainer creates a singleton with env-driven config."""
        import domains.training.auto_trainer as at
        monkeypatch.setattr(at, "_auto_trainer", None)
        monkeypatch.setenv("SLO_AUTO_TRAIN_THRESHOLD", "7")
        monkeypatch.setenv("SLO_AUTO_TRAIN_INTERVAL", "120")
        trainer = at.get_auto_trainer()
        assert trainer.threshold == 7
        assert trainer.interval_s == 120
        assert at.get_auto_trainer() is trainer

    def test_start_auto_trainer_if_disabled(self, monkeypatch):
        """start_auto_trainer_if_enabled returns None when disabled."""
        import domains.training.auto_trainer as at
        monkeypatch.setattr(at, "_auto_trainer", None)
        monkeypatch.setenv("SLO_AUTO_TRAIN", "0")
        assert at.start_auto_trainer_if_enabled() is None

    def test_start_auto_trainer_if_enabled(self, monkeypatch):
        """start_auto_trainer_if_enabled starts the trainer when enabled."""
        import domains.training.auto_trainer as at
        t = AutoTrainer(interval_s=9999)
        monkeypatch.setattr(at, "_auto_trainer", t)
        monkeypatch.setenv("SLO_AUTO_TRAIN", "1")
        try:
            result = at.start_auto_trainer_if_enabled()
            assert result is t
            assert t._thread.is_alive()
        finally:
            t.stop()
            monkeypatch.setattr(at, "_auto_trainer", None)

    def test_stop_auto_trainer_stops_global(self, monkeypatch):
        """stop_auto_trainer stops the global trainer."""
        import domains.training.auto_trainer as at
        t = AutoTrainer(interval_s=9999)
        monkeypatch.setattr(at, "_auto_trainer", t)
        t.start()
        at.stop_auto_trainer()
        assert not t._thread.is_alive()
