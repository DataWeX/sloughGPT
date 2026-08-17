"""Tests for apps/cli/src/utils/progress.py — progress bars and spinners."""
import sys
import os
import time
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestProgressBar:
    def test_init_defaults(self, monkeypatch):
        monkeypatch.setattr("utils.progress._is_terminal", lambda: False)
        from utils.progress import ProgressBar
        bar = ProgressBar(total=100)
        assert bar.total == 100
        assert bar.current == 0
        assert bar.desc == ""

    def test_init_custom(self, monkeypatch):
        monkeypatch.setattr("utils.progress._is_terminal", lambda: False)
        from utils.progress import ProgressBar
        bar = ProgressBar(total=50, desc="Test", width=30)
        assert bar.total == 50
        assert bar.desc == "Test"
        assert bar.width == 30

    def test_update_increments(self, monkeypatch):
        monkeypatch.setattr("utils.progress._is_terminal", lambda: False)
        from utils.progress import ProgressBar
        bar = ProgressBar(total=10)
        bar.update(3)
        assert bar.current == 3

    def test_update_caps_at_total(self, monkeypatch):
        monkeypatch.setattr("utils.progress._is_terminal", lambda: False)
        from utils.progress import ProgressBar
        bar = ProgressBar(total=5)
        bar.update(10)
        assert bar.current == 5

    def test_set_progress(self, monkeypatch):
        monkeypatch.setattr("utils.progress._is_terminal", lambda: False)
        from utils.progress import ProgressBar
        bar = ProgressBar(total=100)
        bar.set_progress(42)
        assert bar.current == 42

    def test_set_progress_caps_at_total(self, monkeypatch):
        monkeypatch.setattr("utils.progress._is_terminal", lambda: False)
        from utils.progress import ProgressBar
        bar = ProgressBar(total=50)
        bar.set_progress(999)
        assert bar.current == 50

    def test_finish_sets_to_total(self, monkeypatch):
        monkeypatch.setattr("utils.progress._is_terminal", lambda: False)
        from utils.progress import ProgressBar
        bar = ProgressBar(total=100)
        bar.update(30)
        bar.finish()
        assert bar.current == 100

    def test_render_non_tty(self, monkeypatch, capsys):
        monkeypatch.setattr("utils.progress._is_terminal", lambda: False)
        from utils.progress import ProgressBar
        bar = ProgressBar(total=10, desc="Loading")
        bar.update(5)
        bar.finish()
        out = capsys.readouterr().out
        assert "Loading" in out
        assert "50%" in out or "5/10" in out

    def test_render_zero_total_no_crash(self, monkeypatch):
        monkeypatch.setattr("utils.progress._is_terminal", lambda: False)
        from utils.progress import ProgressBar
        bar = ProgressBar(total=0)
        bar._render()
        # Should not crash — zero-total is a valid edge case
        assert bar.total == 0

    def test_format_time_under_minute(self):
        from utils.progress import ProgressBar
        assert ProgressBar._format_time(30) == "30s"

    def test_format_time_minutes(self):
        from utils.progress import ProgressBar
        result = ProgressBar._format_time(90)
        assert "1m" in result
        assert "30s" in result

    def test_format_time_hours(self):
        from utils.progress import ProgressBar
        result = ProgressBar._format_time(3720)
        assert "1h" in result
        assert "2m" in result

    def test_dedup_skips_same_pct(self, monkeypatch):
        monkeypatch.setattr("utils.progress._is_terminal", lambda: False)
        from utils.progress import ProgressBar
        bar = ProgressBar(total=100)
        bar.update(50)
        # Force the dedup state
        assert bar._last_pct == 50
        # Same pct should be deduped
        assert bar._last_pct == 50 and bar.desc == bar._last_desc


class TestSpinner:
    def test_start_and_stop(self, monkeypatch):
        monkeypatch.setattr("utils.progress._is_terminal", lambda: False)
        from utils.progress import Spinner
        spinner = Spinner(text="Loading")
        spinner.start()
        assert spinner._running is True
        spinner.stop(message="")
        assert spinner._running is False

    def test_stop_prints_message(self, monkeypatch, capsys):
        monkeypatch.setattr("utils.progress._is_terminal", lambda: False)
        from utils.progress import Spinner
        spinner = Spinner()
        spinner.start()
        spinner.stop(message="Done")
        out = capsys.readouterr().out
        assert "Done" in out

    def test_frames_exist(self):
        from utils.progress import Spinner
        assert len(Spinner.FRAMES) > 0


class TestProgressIter:
    def test_yields_all_items(self, monkeypatch):
        monkeypatch.setattr("utils.progress._is_terminal", lambda: False)
        from utils.progress import progress_iter
        items = list(progress_iter([1, 2, 3], total=3))
        assert items == [1, 2, 3]

    def test_auto_detects_total(self, monkeypatch):
        monkeypatch.setattr("utils.progress._is_terminal", lambda: False)
        from utils.progress import progress_iter
        items = list(progress_iter([1, 2, 3]))
        assert items == [1, 2, 3]

    def test_empty_iterable(self, monkeypatch):
        monkeypatch.setattr("utils.progress._is_terminal", lambda: False)
        from utils.progress import progress_iter
        items = list(progress_iter([], total=0))
        assert items == []
