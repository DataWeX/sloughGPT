"""
Tests for the terminal window manager (domains/shell/window_manager.py).

Tests the non-curses logic: Pane, Workspace, _compute_tiled_rects,
and WindowManager's public API (splits, close, focus, write, layout).
"""

import sys
from typing import Any, List, Tuple
import pytest

from domains.shell.window_manager import (
    WindowManager,
    Pane,
    Workspace,
    LayoutType,
    _compute_tiled_rects,
    get_window_manager,
    reset_window_manager,
)


# ── Pane ───────────────────────────────────────────────────────────


class TestPane:
    def test_init_defaults(self) -> None:
        p = Pane()
        assert p.title == "term"
        assert p.buffer == []
        assert p.scroll_offset == 0
        assert p.floating is False

    def test_init_with_title(self) -> None:
        p = Pane(title="test")
        assert p.title == "test"

    def test_init_with_buffer(self) -> None:
        p = Pane(buffer=["a", "b"])
        assert p.buffer == ["a", "b"]

    def test_write_appends_lines(self) -> None:
        p = Pane()
        p.write("hello\nworld")
        assert p.buffer == ["hello", "world"]

    def test_write_overflow_truncates(self) -> None:
        p = Pane(buffer=[f"line{i}" for i in range(1000)])
        p.write("new")
        assert len(p.buffer) == 1000  # max buffer
        assert p.buffer[-1] == "new"
        assert p.buffer[0] == "line1"  # oldest dropped

    def test_scroll_up(self) -> None:
        p = Pane(buffer=[f"l{i}" for i in range(50)])
        p.scroll_offset = 10
        p.scroll_up(3)
        assert p.scroll_offset == 7

    def test_scroll_up_clamps(self) -> None:
        p = Pane(buffer=[f"l{i}" for i in range(50)])
        p.scroll_offset = 2
        p.scroll_up(5)
        assert p.scroll_offset == 0

    def test_scroll_down(self) -> None:
        p = Pane(buffer=[f"l{i}" for i in range(50)])
        p.scroll_offset = 10
        p.scroll_down(5, pane_h=20)
        assert p.scroll_offset == 15

    def test_scroll_down_clamps(self) -> None:
        p = Pane(buffer=[f"l{i}" for i in range(50)])
        p.scroll_offset = 45
        p.scroll_down(10, pane_h=20)
        # max_offset = 50 - 18 = 32
        assert p.scroll_offset == 32

    def test_visible_lines(self) -> None:
        p = Pane(buffer=[f"l{i}" for i in range(50)])
        p.scroll_offset = 10
        lines = p.visible_lines(15)
        assert len(lines) == 13
        assert lines[0] == "l10"
        assert lines[-1] == "l22"

    def test_visible_lines_empty(self) -> None:
        p = Pane()
        assert p.visible_lines(10) == []

    def test_floating_default_false(self) -> None:
        p = Pane()
        assert p.floating is False

    def test_float_position_defaults(self) -> None:
        p = Pane()
        assert p.float_x == 0
        assert p.float_y == 0
        assert p.float_w == 40
        assert p.float_h == 12


# ── Workspace ──────────────────────────────────────────────────────


class TestWorkspace:
    def test_init(self) -> None:
        ws = Workspace("1")
        assert ws.name == "1"
        assert len(ws.panes) == 1
        assert ws.focus_idx == 0
        assert ws.layout == LayoutType.SPLIT_H

    def test_focused_pane(self) -> None:
        ws = Workspace("1")
        assert ws.focused_pane is ws.panes[0]

    def test_focused_pane_out_of_range(self) -> None:
        ws = Workspace("1")
        ws.focus_idx = 99
        assert ws.focused_pane is None

    def test_focused_pane_empty_panes(self) -> None:
        ws = Workspace("1")
        ws.panes.clear()
        assert ws.focused_pane is None

    def test_add_pane_increases_count(self) -> None:
        ws = Workspace("1")
        ws.add_pane()
        assert len(ws.panes) == 2

    def test_add_pane_focuses_new(self) -> None:
        ws = Workspace("1")
        ws.add_pane()
        assert ws.focus_idx == 1

    def test_add_pane_inserts_after_focus(self) -> None:
        ws = Workspace("1")
        ws.add_pane()
        ws.add_pane()
        ws.focus_idx = 0
        ws.add_pane()
        assert ws.focus_idx == 1

    def test_add_pane_custom_title(self) -> None:
        ws = Workspace("1")
        p = ws.add_pane(title="custom")
        assert p.title == "custom"

    def test_remove_pane_reduces_count(self) -> None:
        ws = Workspace("1")
        ws.add_pane()
        ws.remove_pane(1)
        assert len(ws.panes) == 1

    def test_remove_pane_updates_focus(self) -> None:
        ws = Workspace("1")
        ws.add_pane()
        ws.add_pane()
        ws.remove_pane(1)
        assert ws.focus_idx == 1

    def test_remove_pane_last_returns_none(self) -> None:
        ws = Workspace("1")
        assert ws.remove_pane(0) is None

    def test_next_layout_cycles(self) -> None:
        ws = Workspace("1")
        assert ws.layout == LayoutType.SPLIT_H
        ws.next_layout()
        assert ws.layout == LayoutType.SPLIT_V
        ws.next_layout()
        assert ws.layout == LayoutType.STACKED
        ws.next_layout()
        assert ws.layout == LayoutType.MONOCLE
        ws.next_layout()
        assert ws.layout == LayoutType.SPLIT_H

    def test_focus_next(self) -> None:
        ws = Workspace("1")
        ws.add_pane()
        ws.add_pane()
        ws.focus_idx = 0
        ws.focus_next()
        assert ws.focus_idx == 1

    def test_focus_next_wraps(self) -> None:
        ws = Workspace("1")
        ws.add_pane()
        ws.focus_idx = 1
        ws.focus_next()
        assert ws.focus_idx == 0

    def test_focus_prev(self) -> None:
        ws = Workspace("1")
        ws.add_pane()
        ws.focus_idx = 1
        ws.focus_prev()
        assert ws.focus_idx == 0

    def test_focus_prev_wraps(self) -> None:
        ws = Workspace("1")
        ws.focus_idx = 0
        ws.focus_prev()
        assert ws.focus_idx == 0

    def test_toggle_floating_on_focused(self) -> None:
        ws = Workspace("1")
        result = ws.toggle_floating()
        assert result is True
        assert ws.panes[0].floating is True

    def test_toggle_floating_off(self) -> None:
        ws = Workspace("1")
        ws.panes[0].floating = True
        result = ws.toggle_floating()
        assert result is False
        assert ws.panes[0].floating is False

    def test_toggle_floating_no_focus(self) -> None:
        ws = Workspace("1")
        ws.panes.clear()
        assert ws.toggle_floating() is False

    def test_move_floating(self) -> None:
        ws = Workspace("1")
        ws.panes[0].floating = True
        ws.move_floating(5, 3)
        assert ws.panes[0].float_x == 5
        assert ws.panes[0].float_y == 3

    def test_move_floating_non_floating_ignored(self) -> None:
        ws = Workspace("1")
        ws.move_floating(5, 3)
        assert ws.panes[0].float_x == 0

    def test_resize_floating(self) -> None:
        ws = Workspace("1")
        ws.panes[0].floating = True
        ws.resize_floating(10, 5)
        assert ws.panes[0].float_w == 50
        assert ws.panes[0].float_h == 17

    def test_resize_floating_minimum(self) -> None:
        ws = Workspace("1")
        ws.panes[0].floating = True
        ws.resize_floating(-100, -100)
        assert ws.panes[0].float_w == 10
        assert ws.panes[0].float_h == 4

    def test_set_layout(self) -> None:
        ws = Workspace("1")
        ws.set_layout(LayoutType.MONOCLE)
        assert ws.layout == LayoutType.MONOCLE


# ── _compute_tiled_rects ────────────────────────────────────────────


class TestComputeTiledRects:
    def test_empty_returns_empty(self) -> None:
        assert _compute_tiled_rects([], LayoutType.SPLIT_H, 0, 0, 80, 24) == []

    def test_single_pane_full_area(self) -> None:
        p = Pane()
        rects = _compute_tiled_rects([p], LayoutType.SPLIT_H, 0, 0, 80, 24)
        assert len(rects) == 1
        assert rects[0] == (0, 0, 80, 24)

    def test_split_h_even(self) -> None:
        p1, p2 = Pane(), Pane()
        rects = _compute_tiled_rects([p1, p2], LayoutType.SPLIT_H, 0, 0, 80, 24)
        assert len(rects) == 2
        assert rects[0] == (0, 0, 80, 12)
        assert rects[1] == (0, 12, 80, 12)

    def test_split_h_with_remainder(self) -> None:
        p1, p2, p3 = Pane(), Pane(), Pane()
        rects = _compute_tiled_rects([p1, p2, p3], LayoutType.SPLIT_H, 0, 0, 80, 10)
        assert len(rects) == 3
        # 10/3 = 3 each, remainder 1 → first pane gets extra
        assert rects[0] == (0, 0, 80, 4)
        assert rects[1] == (0, 4, 80, 3)
        assert rects[2] == (0, 7, 80, 3)

    def test_split_v_even(self) -> None:
        p1, p2 = Pane(), Pane()
        rects = _compute_tiled_rects([p1, p2], LayoutType.SPLIT_V, 0, 0, 80, 24)
        assert len(rects) == 2
        assert rects[0] == (0, 0, 40, 24)
        assert rects[1] == (40, 0, 40, 24)

    def test_split_v_with_remainder(self) -> None:
        p1, p2, p3 = Pane(), Pane(), Pane()
        rects = _compute_tiled_rects([p1, p2, p3], LayoutType.SPLIT_V, 0, 0, 10, 24)
        assert len(rects) == 3
        # 10/3 = 3 each, remainder 1 → first gets 4
        assert rects[0] == (0, 0, 4, 24)
        assert rects[1][0] == 4
        assert rects[2][0] == 7

    def test_monocle_all_same_rect(self) -> None:
        panes = [Pane(), Pane(), Pane()]
        rects = _compute_tiled_rects(panes, LayoutType.MONOCLE, 5, 5, 80, 24)
        assert len(rects) == 3
        assert all(r == (5, 5, 80, 24) for r in rects)

    def test_stacked_all_same_rect(self) -> None:
        panes = [Pane(), Pane(), Pane()]
        rects = _compute_tiled_rects(panes, LayoutType.STACKED, 10, 10, 80, 24)
        assert len(rects) == 3
        assert all(r == (10, 10, 80, 24) for r in rects)

    def test_offset_position(self) -> None:
        p = Pane()
        rects = _compute_tiled_rects([p], LayoutType.SPLIT_H, 10, 5, 80, 24)
        assert rects[0] == (10, 5, 80, 24)


# ── WindowManager ──────────────────────────────────────────────────


class TestWindowManager:
    def test_init_no_parent(self) -> None:
        wm = WindowManager()
        assert wm.pane_count == 1
        assert wm.focused_pane is not None

    def test_init_with_parent(self) -> None:
        parent = object()
        wm = WindowManager(parent_shell=parent)
        assert wm._parent_shell is parent

    def test_split_horizontal_increases_panes(self) -> None:
        wm = WindowManager()
        wm.split_horizontal()
        assert wm.pane_count == 2

    def test_split_vertical_increases_panes(self) -> None:
        wm = WindowManager()
        wm.split_vertical()
        assert wm.pane_count == 2

    def test_split_horizontal_sets_layout(self) -> None:
        wm = WindowManager()
        wm.split_horizontal()
        assert wm._workspace.layout == LayoutType.SPLIT_H

    def test_split_vertical_sets_layout(self) -> None:
        wm = WindowManager()
        wm.split_vertical()
        assert wm._workspace.layout == LayoutType.SPLIT_V

    def test_close_pane_reduces_count(self) -> None:
        wm = WindowManager()
        wm.split_horizontal()
        wm.close_pane()
        assert wm.pane_count == 1

    def test_close_pane_last_returns_none(self) -> None:
        wm = WindowManager()
        title = wm.close_pane()
        assert title is None
        assert wm.pane_count == 1  # can't close last pane

    def test_close_pane_returns_title(self) -> None:
        wm = WindowManager()
        wm.split_horizontal()
        title = wm.close_pane()
        assert title is not None

    def test_focus_next(self) -> None:
        wm = WindowManager()
        wm.split_horizontal()
        wm.split_horizontal()
        wm._workspace.focus_idx = 0
        wm.focus_next()
        assert wm._workspace.focus_idx == 1

    def test_focus_prev(self) -> None:
        wm = WindowManager()
        wm.split_horizontal()
        wm._workspace.focus_idx = 1
        wm.focus_prev()
        assert wm._workspace.focus_idx == 0

    def test_next_layout_cycles(self) -> None:
        wm = WindowManager()
        wm.next_layout()
        assert wm._workspace.layout == LayoutType.SPLIT_V

    def test_set_layout(self) -> None:
        wm = WindowManager()
        wm.set_layout(LayoutType.MONOCLE)
        assert wm._workspace.layout == LayoutType.MONOCLE

    def test_toggle_floating(self) -> None:
        wm = WindowManager()
        wm.toggle_floating()
        assert wm.focused_pane is not None
        assert wm.focused_pane.floating is True

    def test_switch_workspace(self) -> None:
        wm = WindowManager()
        wm.switch_workspace(2)
        assert wm._workspace.name == "3"

    def test_switch_workspace_same_ignored(self) -> None:
        wm = WindowManager()
        wm.switch_workspace(0)
        assert wm._workspace.name == "1"

    def test_switch_workspace_out_of_range(self) -> None:
        wm = WindowManager()
        wm.switch_workspace(99)
        assert wm._workspace.name == "1"
        wm.switch_workspace(-1)
        assert wm._workspace.name == "1"

    def test_move_pane_to_workspace(self) -> None:
        wm = WindowManager()
        wm.split_horizontal()
        wm.move_pane_to_workspace(1)
        assert wm.pane_count == 1
        assert len(wm._workspaces[1].panes) == 2

    def test_move_pane_last_does_not_move(self) -> None:
        wm = WindowManager()
        wm.move_pane_to_workspace(1)
        assert wm.pane_count == 1
        assert len(wm._workspaces[1].panes) == 1

    def test_move_focus_forward(self) -> None:
        wm = WindowManager()
        wm.split_horizontal()
        wm._workspace.focus_idx = 0
        wm.move_focus(1, 0)
        assert wm._workspace.focus_idx == 1

    def test_move_focus_backward(self) -> None:
        wm = WindowManager()
        wm.split_horizontal()
        wm._workspace.focus_idx = 1
        wm.move_focus(-1, 0)
        assert wm._workspace.focus_idx == 0

    def test_move_focus_down_moves_next(self) -> None:
        wm = WindowManager()
        wm.split_horizontal()
        wm._workspace.focus_idx = 0
        wm.move_focus(0, 1)
        assert wm._workspace.focus_idx == 1

    def test_move_focus_up_moves_prev(self) -> None:
        wm = WindowManager()
        wm.split_horizontal()
        wm._workspace.focus_idx = 1
        wm.move_focus(0, -1)
        assert wm._workspace.focus_idx == 0

    def test_move_focus_floating(self) -> None:
        wm = WindowManager()
        wm.toggle_floating()
        x0 = wm.focused_pane.float_x  # type: ignore[union-attr]
        y0 = wm.focused_pane.float_y  # type: ignore[union-attr]
        wm.move_focus(1, 0)
        assert wm.focused_pane.float_x == x0 + 2  # type: ignore[union-attr]
        assert wm.focused_pane.float_y == y0  # type: ignore[union-attr]

    def test_resize_focused_floating(self) -> None:
        wm = WindowManager()
        wm.toggle_floating()
        w0 = wm.focused_pane.float_w  # type: ignore[union-attr]
        wm.resize_focused(10, 0)
        assert wm.focused_pane.float_w == w0 + 10  # type: ignore[union-attr]

    def test_resize_focused_tiled_swaps_left(self) -> None:
        wm = WindowManager()
        wm.split_horizontal()
        # focus is on pane-1 (idx=1), swap left
        titles = [p.title for p in wm._workspace.panes]
        wm.resize_focused(-1, 0)
        assert wm._workspace.panes[0].title == titles[1]
        assert wm._workspace.panes[1].title == titles[0]
        assert wm._workspace.focus_idx == 0

    def test_pane_count_property(self) -> None:
        wm = WindowManager()
        assert wm.pane_count == 1
        wm.split_horizontal()
        assert wm.pane_count == 2


# ── LayoutType ─────────────────────────────────────────────────────


class TestLayoutType:
    def test_next_from_split_h(self) -> None:
        assert LayoutType.next(LayoutType.SPLIT_H) == LayoutType.SPLIT_V

    def test_next_from_split_v(self) -> None:
        assert LayoutType.next(LayoutType.SPLIT_V) == LayoutType.STACKED

    def test_next_from_stacked(self) -> None:
        assert LayoutType.next(LayoutType.STACKED) == LayoutType.MONOCLE

    def test_next_from_monocle(self) -> None:
        assert LayoutType.next(LayoutType.MONOCLE) == LayoutType.SPLIT_H

    def test_next_unknown_returns_split_v(self) -> None:
        # Unknown layout defaults to start of order list, so next is index 1 = SPLIT_V
        assert LayoutType.next("unknown") == LayoutType.SPLIT_V


# ── Singleton ──────────────────────────────────────────────────────


class TestSingleton:
    def test_get_window_manager_creates(self) -> None:
        reset_window_manager()
        wm = get_window_manager()
        assert isinstance(wm, WindowManager)

    def test_get_window_manager_reuses(self) -> None:
        reset_window_manager()
        wm1 = get_window_manager()
        wm2 = get_window_manager()
        assert wm1 is wm2

    def test_reset_window_manager(self) -> None:
        wm1 = get_window_manager()
        reset_window_manager()
        wm2 = get_window_manager()
        assert wm1 is not wm2

    def test_get_window_manager_passes_parent(self) -> None:
        reset_window_manager()
        parent = object()
        wm = get_window_manager(parent_shell=parent)
        assert wm._parent_shell is parent


# ── Window Manager Commands ──────────────────────────────────────────


class TestWMCommands:
    def make_wm(self) -> WindowManager:
        reset_window_manager()
        wm = get_window_manager()
        ws = wm._workspace
        ws.panes = [Pane(title="p1"), Pane(title="p2"), Pane(title="p3")]
        ws.focus_idx = 0
        return wm

    def test_grep_finds_matches(self) -> None:
        wm = self.make_wm()
        p = wm.focused_pane
        assert p is not None
        p.buffer = ["hello world", "foo bar", "hello again"]
        wm._execute_wm_command(":grep hello")
        assert wm._dirty
        # Buffer should contain match info
        assert any("hello" in line for line in p.buffer)

    def test_grep_no_matches(self) -> None:
        wm = self.make_wm()
        p = wm.focused_pane
        assert p is not None
        p.buffer = ["hello world"]
        wm._execute_wm_command(":grep zzz")
        assert wm._dirty
        assert any("No matches" in line for line in p.buffer)

    def test_rename_pane(self) -> None:
        wm = self.make_wm()
        p = wm.focused_pane
        assert p is not None
        wm._execute_wm_command(":rename my-title")
        assert p.title == "my-title"
        assert wm._dirty

    def test_focus_by_index(self) -> None:
        wm = self.make_wm()
        wm._execute_wm_command(":focus 3")
        assert wm._workspace.focus_idx == 2  # 0-indexed

    def test_focus_out_of_range(self) -> None:
        wm = self.make_wm()
        old = wm._workspace.focus_idx
        wm._execute_wm_command(":focus 99")
        assert wm._workspace.focus_idx == old  # unchanged

    def test_split_equal_tiles_panes(self) -> None:
        wm = self.make_wm()
        wm._execute_wm_command(":split-equal")
        assert wm._dirty
        # should have changed layout from default
        ws = wm._workspace
        assert ws.layout != LayoutType.SPLIT_H or len(ws.panes) > 1

    def test_split_equal_single_pane(self) -> None:
        reset_window_manager()
        wm = get_window_manager()
        ws = wm._workspace
        ws.panes = [Pane(title="only")]
        ws.focus_idx = 0
        wm._execute_wm_command(":tile")
        assert wm._dirty

    def test_pane_cmd_history(self) -> None:
        wm = self.make_wm()
        p = wm.focused_pane
        assert p is not None
        p._cmd_history.append("cmd1")
        p._cmd_history.append("cmd2")
        wm._execute_wm_command(":history")
        assert wm._dirty
        assert any("cmd1" in line for line in p.buffer)

    def test_pane_cmd_history_empty(self) -> None:
        wm = self.make_wm()
        p = wm.focused_pane
        assert p is not None
        p._cmd_history.clear()
        wm._execute_wm_command(":hist")
        assert wm._dirty
        assert any("no command history" in line.lower() for line in p.buffer)

    def test_write_no_filename(self) -> None:
        wm = self.make_wm()
        wm._execute_wm_command(":write")
        # Should flash usage, not crash

    def test_load_no_filename(self) -> None:
        wm = self.make_wm()
        wm._execute_wm_command(":load")
        # Should flash usage, not crash

    def test_help_topic_exists(self) -> None:
        wm = self.make_wm()
        p = wm.focused_pane
        wm._execute_wm_command(":help shell")
        assert wm._dirty
        assert any("Pane Shell" in line for line in (p.buffer or []))

    def test_help_topic_unknown(self) -> None:
        wm = self.make_wm()
        wm._execute_wm_command(":help nonexistent")
        assert wm._dirty

    def test_ls_works_in_pane(self) -> None:
        wm = self.make_wm()
        p = wm.focused_pane
        wm._execute_wm_command(":ls")
        assert wm._dirty
        # Should have listed something (at least the test file dir)
        assert any(line.strip() for line in (p.buffer or []))

    def test_pwd_in_pane(self) -> None:
        wm = self.make_wm()
        p = wm.focused_pane
        wm._execute_wm_command(":pwd")
        assert wm._dirty
        # Buffer should contain a path
        assert p and any("/" in line for line in p.buffer)

    def test_echo_in_pane(self) -> None:
        wm = self.make_wm()
        p = wm.focused_pane
        wm._execute_wm_command(":echo hello test")
        assert wm._dirty
        assert p and any("hello test" in line for line in p.buffer)

    def test_date_in_pane(self) -> None:
        wm = self.make_wm()
        p = wm.focused_pane
        wm._execute_wm_command(":date")
        assert wm._dirty
        assert p and any(len(line.strip()) > 5 for line in p.buffer)

    def test_quit_command(self) -> None:
        wm = self.make_wm()
        wm._execute_wm_command(":q")
        assert not wm._running

    def test_close_pane_via_command(self) -> None:
        wm = self.make_wm()
        n = len(wm._workspace.panes)
        wm._execute_wm_command(":close")
        assert len(wm._workspace.panes) == n - 1

    def test_split_h_via_command(self) -> None:
        wm = self.make_wm()
        n = len(wm._workspace.panes)
        wm._execute_wm_command(":split-h")
        assert len(wm._workspace.panes) == n + 1

    def test_split_v_via_command(self) -> None:
        wm = self.make_wm()
        n = len(wm._workspace.panes)
        wm._execute_wm_command(":split-v")
        assert len(wm._workspace.panes) == n + 1

    def test_float_command(self) -> None:
        wm = self.make_wm()
        p = wm.focused_pane
        assert p is not None
        wm._execute_wm_command(":float")
        assert p.floating

    def test_resize_command(self) -> None:
        wm = self.make_wm()
        wm._execute_wm_command(":resize")
        assert wm._resize_mode

    def test_monocle_command(self) -> None:
        wm = self.make_wm()
        wm._execute_wm_command(":monocle")
        assert wm._workspace.layout == LayoutType.MONOCLE

    def test_stacked_command(self) -> None:
        wm = self.make_wm()
        wm._execute_wm_command(":stacked")
        assert wm._workspace.layout == LayoutType.STACKED

    def test_reset_command(self) -> None:
        wm = self.make_wm()
        wm._workspace.panes = [Pane("a"), Pane("b")]
        wm._workspace.layout = LayoutType.MONOCLE
        wm._execute_wm_command(":reset")
        assert len(wm._workspace.panes) == 1
        assert wm._workspace.layout == LayoutType.SPLIT_H
