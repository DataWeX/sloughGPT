"""Tests for ``core.tui`` — the live display and the metrics it paints.

Covers the two Aug-31 TUI fixes that shipped without a test:

1. ``LiveDisplay.update`` rewrites the screen with cursor addressing
   (``\\033[row;1H``) instead of ``\\n``-delimited writes, which scrolled the
   terminal during animation.
2. ``DevDashboard._collect_metrics`` reads ``/proc/stat`` + ``/proc/meminfo``
   on Linux; the old macOS-only path (``top -l``, ``vm_stat``) failed silently
   and left every gauge at ``0.0``.
"""

from __future__ import annotations

import contextlib
import io
import platform
import time

import core.tui as tui
import pytest
from core.tui import DevDashboard, LiveDisplay

LINUX = platform.system() == "Linux"


def _capture(render, *, rows: int = 10, cols: int = 80) -> str:
    """Run ``render(display)`` against a fixed-size terminal, return written bytes."""
    buf = io.StringIO()
    original_scrn = tui._scrn
    tui._scrn = lambda: (cols, rows)
    try:
        with contextlib.redirect_stdout(buf):
            with LiveDisplay() as display:
                render(display)
    finally:
        tui._scrn = original_scrn
    return buf.getvalue()


class TestLiveDisplay:
    def test_update_writes_no_newlines(self):
        """The rewrite's whole point: never emit \\n, so the terminal cannot scroll."""
        out = _capture(lambda d: d.update("alpha\nbeta\ngamma"))
        assert "\n" not in out

    def test_enter_emits_no_newlines(self):
        out = _capture(lambda d: d.update("one"))
        assert out.startswith(tui._HIDE_CURSOR + tui._CLEAR + tui._MOVE_HOME)

    def test_addresses_every_row(self):
        out = _capture(lambda d: d.update("alpha\nbeta"))
        assert "\033[1;1H" in out
        assert "\033[2;1H" in out
        assert out.split("\033[1;1H")[1].startswith("alpha")
        assert out.split("\033[2;1H")[1].startswith("beta")

    def test_each_row_is_terminated_by_clear_to_eol(self):
        out = _capture(lambda d: d.update("alpha\nbeta"))
        assert "\033[1;1Halpha\033[K" in out
        assert "\033[2;1Hbeta\033[K" in out

    def test_shrinking_frame_clears_the_rows_it_abandoned(self):
        """A 6-line frame followed by a 1-line frame must wipe rows 2..6."""
        buf = io.StringIO()
        original_scrn = tui._scrn
        tui._scrn = lambda: (80, 10)
        try:
            with contextlib.redirect_stdout(buf):
                with LiveDisplay() as display:
                    display.update("\n".join(f"row {i}" for i in range(1, 7)))
                    buf.seek(0)
                    buf.truncate()
                    display.update("only")
        finally:
            tui._scrn = original_scrn
        out = buf.getvalue()
        for row in range(2, 7):
            assert f"\033[{row};1H\033[K" in out, f"row {row} not cleared"

    def test_frame_longer_than_the_terminal_is_clamped(self):
        """10-row terminal ⇒ 9 usable rows; row 10 and beyond are never written."""
        out = _capture(lambda d: d.update("\n".join(f"row {i}" for i in range(1, 30))))
        assert "\033[1;1H" in out
        assert "\033[9;1H" in out
        assert "\033[10;1H" not in out
        assert "\n" not in out

    def test_repeated_updates_do_not_grow_state(self):
        buf = io.StringIO()
        original_scrn = tui._scrn
        tui._scrn = lambda: (80, 10)
        try:
            with contextlib.redirect_stdout(buf):
                with LiveDisplay() as display:
                    for i in range(5):
                        display.update(f"frame {i}\nsecond")
        finally:
            tui._scrn = original_scrn
        assert display._prev == "frame 4\nsecond"
        assert display._prev_line_count == 2


class TestCollectMetrics:
    def _dashboard(self) -> DevDashboard:
        dash = DevDashboard()
        dash._last_metrics_collect = 0.0  # defeat the 2s throttle
        return dash

    @pytest.mark.skipif(not LINUX, reason="/proc metrics only exist on Linux")
    def test_memory_comes_from_proc_meminfo(self):
        """Regression: the macOS-only path reported 0.0 on Linux."""
        dash = self._dashboard()
        dash._collect_metrics()
        memory = dash._metrics["memory"]
        assert 0 < memory <= 100

    @pytest.mark.skipif(not LINUX, reason="/proc metrics only exist on Linux")
    def test_cpu_is_bounded_after_two_samples(self):
        dash = self._dashboard()
        dash._collect_metrics()  # first call only stores the sample
        time.sleep(0.05)  # let /proc/stat tick forward
        dash._last_metrics_collect = 0.0
        dash._collect_metrics()
        assert 0 <= dash._metrics["cpu"] <= 100

    @pytest.mark.skipif(not LINUX, reason="/proc metrics only exist on Linux")
    def test_first_cpu_sample_is_stored_not_reported(self):
        dash = self._dashboard()
        dash._collect_metrics()
        assert dash._linux_cpu_sample is not None
        assert dash._metrics["cpu"] == 0.0

    def test_throttle_skips_a_back_to_back_collection(self):
        dash = self._dashboard()
        dash._last_metrics_collect = time.monotonic()
        dash._collect_metrics()
        assert dash._metrics == {"cpu": 0.0, "memory": 0.0, "disk": 0.0}
        assert dash._linux_cpu_sample is None

    def test_collect_never_raises(self):
        dash = self._dashboard()
        dash._collect_metrics()  # exceptions are swallowed; gauges stay in range
        for value in dash._metrics.values():
            assert 0 <= value <= 100
