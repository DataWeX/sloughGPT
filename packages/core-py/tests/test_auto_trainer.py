"""Tests for AutoTrainer (background training from inference logs)."""

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
    @patch("domains.training.pair_extractor.extract_pairs_from_logs", return_value=[])
    def test_do_train_insufficient_pairs(self, mock_logs, mock_sessions):
        """Training skipped when fewer than 5 pairs found."""
        t = AutoTrainer()
        result = t._do_train()
        assert result is False

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
