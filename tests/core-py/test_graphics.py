"""Tests for the terminal graphics engine."""

from __future__ import annotations

import io
import sys

import pytest

from domain.shell._internal.graphics import (
    Attr,
    BoxModel,
    BoxStyle,
    Cell,
    Color,
    DisplayMode,
    FontConfig,
    Framebuffer,
    GraphicsEngine,
    HardwareOutput,
    HardwareState,
    Layer,
    Pattern,
    Snapshot,
    SnapshotManager,
    TabManager,
    _pattern_char,
    compare_hardware_outputs,
    compare_hardware_states,
    get_mode_attr,
    rgb_to_ansi256,
)

# ── Cell ──────────────────────────────────────────────────────────────


class TestCell:
    def test_default(self):
        c = Cell()
        assert c.char == " "
        assert c.fg == Color.DEFAULT
        assert c.bg == Color.DEFAULT
        assert c.attr == Attr.NONE

    def test_custom(self):
        c = Cell("X", fg=Color.RED, bg=Color.BLUE, attr=Attr.BOLD)
        assert c.char == "X"
        assert c.fg == Color.RED
        assert c.bg == Color.BLUE
        assert c.attr == Attr.BOLD

    def test_equality(self):
        a = Cell("A", fg=1, bg=2, attr=3)
        b = Cell("A", fg=1, bg=2, attr=3)
        assert a == b

    def test_inequality(self):
        a = Cell("A", fg=1)
        b = Cell("B", fg=1)
        assert a != b

    def test_copy(self):
        original = Cell("X", fg=Color.GREEN, attr=Attr.UNDERLINE)
        copied = original.copy()
        assert copied == original
        copied.char = "Y"
        assert original.char == "X"

    def test_not_equal_to_non_cell(self):
        assert Cell().__eq__("not a cell") is NotImplemented


# ── Color ─────────────────────────────────────────────────────────────


class TestColor:
    def test_rgb_to_ansi256_black(self):
        assert rgb_to_ansi256(0, 0, 0) == 0

    def test_rgb_to_ansi256_white(self):
        assert rgb_to_ansi256(255, 255, 255) == 15

    def test_rgb_to_ansi256_gray(self):
        # A neutral input must produce a neutral index. This used to assert 34,
        # which is cube entry (0,0,95) -- blue -- for a grey request. The comment
        # on the old assertion acknowledged the confusion and encoded it anyway.
        idx = rgb_to_ansi256(128, 128, 128)
        assert 232 <= idx <= 255, f"grey must use the grey ramp, got {idx}"

    @pytest.mark.parametrize(
        "level,expect",
        [
            (0, 0),  # below the ramp's floor of 8
            (72, 238),  # mid grey
            (238, 255),  # the ramp's last entry is rgb 238
            (248, 255),  # past the ramp's top, clamps to its last entry
            (255, 15),  # white
        ],
    )
    def test_rgb_to_ansi256_greys_are_neutral(self, level, expect):
        """Every grey maps to the ramp (or the plain black/white ends), never into
        the tinted cube.

        Regression guard: the cube branch quantises each channel to six steps, so
        an earlier version sent rgb(72,72,72) to index 28 == (0,95,0), a green.
        Frames and surfaces are drawn in greys, so that failure tinted the whole UI.
        """
        assert rgb_to_ansi256(level, level, level) == expect

    @pytest.mark.parametrize("n", range(24))
    def test_rgb_to_ansi256_grey_ramp_round_trips(self, n):
        """Every entry of the grey ramp (232-255) is reachable and exact.

        The ramp spans rgb 8..238 in steps of 10. An earlier scale divided by 247
        instead of 230, which compressed the mapping so indices 254 and 255 could
        never be produced -- a value like rgb(238) is a perfectly ordinary frame
        colour and had no correct answer.
        """
        level = 8 + 10 * n
        assert rgb_to_ansi256(level, level, level) == 232 + n

    def test_rgb_to_ansi256_red(self):
        idx = rgb_to_ansi256(255, 0, 0)
        assert 16 <= idx <= 231  # in color cube


# ── DisplayMode ───────────────────────────────────────────────────────


class TestDisplayMode:
    def test_all_modes_have_attrs(self):
        for mode in DisplayMode:
            attr = get_mode_attr(mode)
            assert isinstance(attr, int)

    def test_normal_is_no_attr(self):
        assert get_mode_attr(DisplayMode.NORMAL) == Attr.NONE

    def test_bold(self):
        assert get_mode_attr(DisplayMode.BOLD) & Attr.BOLD

    def test_reverse(self):
        assert get_mode_attr(DisplayMode.REVERSE) & Attr.REVERSE

    def test_dim(self):
        assert get_mode_attr(DisplayMode.DIM) & Attr.DIM


# ── Pattern ───────────────────────────────────────────────────────────


class TestPattern:
    def test_solid(self):
        ch = _pattern_char(Pattern.SOLID, 0, 0)
        assert ch == "█"

    def test_dots_alternate(self):
        chars = set()
        for r in range(10):
            for c in range(10):
                chars.add(_pattern_char(Pattern.DOTS, r, c))
        assert len(chars) >= 2  # dots and spaces

    def test_checker_alternates(self):
        a = _pattern_char(Pattern.CHECKER, 0, 0)
        b = _pattern_char(Pattern.CHECKER, 0, 1)
        c = _pattern_char(Pattern.CHECKER, 1, 1)
        assert a != b
        assert a == c  # same parity: (0+0)%2 == (1+1)%2

    def test_none_is_space(self):
        assert _pattern_char(Pattern.NONE, 5, 5) == " "


# ── Framebuffer ───────────────────────────────────────────────────────


class TestFramebuffer:
    def test_creation(self):
        fb = Framebuffer(10, 20)
        assert fb.rows == 10
        assert fb.cols == 20

    def test_put_and_get(self):
        fb = Framebuffer(5, 5)
        fb.put(2, 3, "X", fg=Color.RED)
        cell = fb.get(2, 3)
        assert cell.char == "X"
        assert cell.fg == Color.RED

    def test_out_of_bounds_returns_default(self):
        fb = Framebuffer(5, 5)
        cell = fb.get(10, 10)
        assert cell == Cell()

    def test_out_of_bounds_put_noop(self):
        fb = Framebuffer(5, 5)
        fb.put(10, 10, "X")  # should not raise

    def test_write_str(self):
        fb = Framebuffer(5, 20)
        written = fb.write_str(0, 0, "Hello")
        assert written == 5
        assert fb.get(0, 0).char == "H"
        assert fb.get(0, 4).char == "o"

    def test_write_str_clips_at_edge(self):
        fb = Framebuffer(5, 10)
        written = fb.write_str(0, 8, "Hello")
        assert written == 2  # only "He" fits

    def test_clear(self):
        fb = Framebuffer(5, 5)
        fb.put(2, 2, "X")
        fb.clear()
        assert fb.get(2, 2).char == " "

    def test_fill(self):
        fb = Framebuffer(5, 5)
        fb.fill("#", fg=Color.GREEN)
        for r in range(5):
            for c in range(5):
                assert fb.get(r, c).char == "#"
                assert fb.get(r, c).fg == Color.GREEN

    def test_fill_pattern(self):
        fb = Framebuffer(10, 10)
        fb.fill_pattern(Pattern.DOTS, fg=Color.CYAN)
        # At least some cells should have dots
        dot_count = sum(1 for r in range(10) for c in range(10) if fb.get(r, c).char == "·")
        assert dot_count > 0

    def test_draw_rect(self):
        fb = Framebuffer(10, 10)
        fb.draw_rect(2, 3, 4, 3, char="X")
        # Inside the rect
        assert fb.get(2, 3).char == "X"
        assert fb.get(3, 4).char == "X"
        assert fb.get(4, 5).char == "X"
        # Outside the rect
        assert fb.get(1, 3).char == " "
        assert fb.get(2, 2).char == " "

    def test_draw_box_single(self):
        fb = Framebuffer(10, 20)
        fb.draw_box(0, 0, 10, 5, style=BoxStyle.SINGLE)
        # Corners
        assert fb.get(0, 0).char == "┌"
        assert fb.get(0, 9).char == "┐"
        assert fb.get(4, 0).char == "└"
        assert fb.get(4, 9).char == "┘"
        # Edges
        assert fb.get(0, 5).char == "─"
        assert fb.get(2, 0).char == "│"

    def test_draw_box_double(self):
        fb = Framebuffer(10, 20)
        fb.draw_box(0, 0, 10, 5, style=BoxStyle.DOUBLE)
        assert fb.get(0, 0).char == "╔"
        assert fb.get(0, 9).char == "╗"

    def test_draw_box_none(self):
        fb = Framebuffer(10, 10)
        fb.draw_box(0, 0, 5, 5, style=BoxStyle.NONE)
        assert fb.get(0, 0).char == " "

    def test_draw_box_too_small(self):
        fb = Framebuffer(10, 10)
        fb.draw_box(0, 0, 1, 1, style=BoxStyle.SINGLE)
        assert fb.get(0, 0).char == " "

    def test_resize_preserves_content(self):
        fb = Framebuffer(5, 5)
        fb.put(2, 2, "X")
        fb.resize(10, 10)
        assert fb.get(2, 2).char == "X"
        assert fb.rows == 10
        assert fb.cols == 10

    def test_resize_clips(self):
        fb = Framebuffer(10, 10)
        fb.put(8, 8, "X")
        fb.resize(5, 5)
        assert fb.get(8, 8) == Cell()  # out of bounds

    def test_composite(self):
        bg = Framebuffer(10, 10)
        bg.fill(" ", fg=Color.BLACK)
        fg = Framebuffer(5, 5)
        fg.write_str(0, 0, "HELLO", fg=Color.GREEN)
        bg.composite(fg, dest_row=2, dest_col=3)
        assert bg.get(2, 3).char == "H"
        assert bg.get(2, 7).char == "O"

    def test_diff(self):
        a = Framebuffer(3, 3)
        b = Framebuffer(3, 3)
        a.put(1, 1, "X")
        b.put(1, 1, "Y")
        changes = a.diff(b)
        assert len(changes) == 1
        assert changes[0][2].char == "X"  # what 'a' has

    def test_snapshot_and_restore(self):
        fb = Framebuffer(5, 5)
        fb.write_str(0, 0, "Hello")
        snap = fb.snapshot()
        fb.clear()
        assert fb.get(0, 0).char == " "
        fb.restore(snap)
        assert fb.get(0, 0).char == "H"


# ── Layer ─────────────────────────────────────────────────────────────


class TestLayer:
    def test_creation(self):
        layer = Layer("test", 10, 20, z=5)
        assert layer.name == "test"
        assert layer.z == 5
        assert layer.visible is True

    def test_write(self):
        layer = Layer("test", 10, 20)
        layer.write(0, 0, "Hello", fg=Color.RED)
        assert layer.framebuffer.get(0, 0).char == "H"

    def test_fill_pattern(self):
        layer = Layer("test", 10, 10)
        layer.fill_pattern(Pattern.CROSS)
        cross_count = sum(
            1 for r in range(10) for c in range(10) if layer.framebuffer.get(r, c).char == "┼"
        )
        assert cross_count > 0

    def test_resize(self):
        layer = Layer("test", 5, 5)
        layer.write(0, 0, "X")
        layer.resize(10, 10)
        assert layer.framebuffer.get(0, 0).char == "X"
        assert layer.framebuffer.rows == 10

    def test_clear(self):
        layer = Layer("test", 5, 5)
        layer.write(0, 0, "X")
        layer.clear()
        assert layer.framebuffer.get(0, 0).char == " "


# ── SnapshotManager ───────────────────────────────────────────────────


class TestSnapshotManager:
    def test_save_and_undo(self):
        mgr = SnapshotManager()
        fb = Framebuffer(5, 5)
        fb.write_str(0, 0, "A")
        mgr.save(fb)
        fb.write_str(0, 0, "B")
        snap = mgr.undo()
        assert snap is not None
        fb.restore(snap.framebuffer)
        assert fb.get(0, 0).char == "A"

    def test_undo_empty(self):
        mgr = SnapshotManager()
        assert mgr.undo() is None

    def test_redo(self):
        mgr = SnapshotManager()
        fb = Framebuffer(5, 5)
        fb.write_str(0, 0, "A")
        mgr.save(fb)
        fb.write_str(0, 0, "B")
        snap_a = mgr.undo()
        assert snap_a is not None
        mgr.push_redo(Snapshot(fb.snapshot()))
        snap_b = mgr.redo()
        assert snap_b is not None

    def test_redo_empty(self):
        mgr = SnapshotManager()
        assert mgr.redo() is None

    def test_max_history(self):
        mgr = SnapshotManager(max_history=3)
        for i in range(5):
            fb = Framebuffer(3, 3)
            fb.write_str(0, 0, str(i))
            mgr.save(fb)
        # Only last 3 should be kept
        count = 0
        while mgr.undo() is not None:
            count += 1
        assert count == 3

    def test_save_clears_redo(self):
        mgr = SnapshotManager()
        fb = Framebuffer(3, 3)
        mgr.save(fb)
        mgr.undo()
        mgr.push_redo(Snapshot(fb.snapshot()))
        assert mgr.can_redo
        mgr.save(fb)  # new save should clear redo
        assert not mgr.can_redo

    def test_can_undo_redo(self):
        mgr = SnapshotManager()
        assert not mgr.can_undo
        assert not mgr.can_redo
        fb = Framebuffer(3, 3)
        mgr.save(fb)
        assert mgr.can_undo
        assert not mgr.can_redo


# ── TabManager ────────────────────────────────────────────────────────


class TestTabManager:
    def test_add_region(self):
        mgr = TabManager()
        r = mgr.add_region("a", 0, 0, 10, 5, tab_order=1)
        assert r.name == "a"
        assert len(mgr.regions) == 1

    def test_focus_next(self):
        mgr = TabManager()
        mgr.add_region("a", 0, 0, 10, 5, tab_order=1)
        mgr.add_region("b", 5, 0, 10, 5, tab_order=2)
        r = mgr.focus_next()
        assert r is not None
        assert r.name == "a"
        r = mgr.focus_next()
        assert r is not None
        assert r.name == "b"
        r = mgr.focus_next()
        assert r is not None
        assert r.name == "a"  # wraps

    def test_focus_prev(self):
        mgr = TabManager()
        mgr.add_region("a", 0, 0, 10, 5, tab_order=1)
        mgr.add_region("b", 5, 0, 10, 5, tab_order=2)
        mgr.focus_next()  # a
        mgr.focus_next()  # b
        r = mgr.focus_prev()
        assert r is not None
        assert r.name == "a"

    def test_focus_by_name(self):
        mgr = TabManager()
        mgr.add_region("a", 0, 0, 10, 5)
        mgr.add_region("b", 5, 0, 10, 5)
        r = mgr.focus("b")
        assert r is not None
        assert r.name == "b"
        assert mgr.focused is r

    def test_focus_callback(self):
        mgr = TabManager()
        called = []
        mgr.add_region("a", 0, 0, 10, 5, on_focus=lambda: called.append("focus"))
        mgr.focus_next()
        assert "focus" in called

    def test_blur_callback(self):
        mgr = TabManager()
        called = []
        mgr.add_region("a", 0, 0, 10, 5, on_blur=lambda: called.append("blur"))
        mgr.add_region("b", 5, 0, 10, 5)
        mgr.focus_next()  # a
        mgr.focus_next()  # b (blurs a)
        assert "blur" in called

    def test_remove_region(self):
        mgr = TabManager()
        mgr.add_region("a", 0, 0, 10, 5)
        mgr.add_region("b", 5, 0, 10, 5)
        mgr.remove_region("a")
        assert len(mgr.regions) == 1
        assert mgr.regions[0].name == "b"

    def test_empty_no_crash(self):
        mgr = TabManager()
        assert mgr.focus_next() is None
        assert mgr.focus_prev() is None
        assert mgr.focused is None


# ── BoxModel ──────────────────────────────────────────────────────────


class TestBoxModel:
    def test_default_no_border(self):
        bm = BoxModel()
        assert bm.border_top == 0
        assert bm.border_bottom == 0

    def test_border_thickness(self):
        bm = BoxModel(border=BoxStyle.SINGLE)
        assert bm.border_top == 1
        assert bm.border_left == 1


# ── FontConfig ────────────────────────────────────────────────────────


class TestFontConfig:
    def test_default(self):
        fc = FontConfig()
        assert fc.size == 14
        assert fc.cells_per_char() == 1

    def test_double_width(self):
        fc = FontConfig(char_width=2)
        assert fc.cells_per_char() == 2


# ── HardwareState ─────────────────────────────────────────────────────


class TestHardwareState:
    def test_creation(self):
        hs = HardwareState(rows=5, cols=10)
        assert hs.rows == 5
        assert hs.cols == 10

    def test_get_cell(self):
        hs = HardwareState(
            rows=2,
            cols=3,
            cells=[
                [Cell("a"), Cell("b"), Cell("c")],
                [Cell("d"), Cell("e"), Cell("f")],
            ],
        )
        assert hs.get_cell(0, 0).char == "a"
        assert hs.get_cell(1, 2).char == "f"

    def test_get_cell_out_of_bounds(self):
        hs = HardwareState(rows=2, cols=2)
        assert hs.get_cell(10, 10).char == " "

    def test_text_at(self):
        hs = HardwareState(
            rows=1,
            cols=5,
            cells=[[Cell("H"), Cell("e"), Cell("l"), Cell("l"), Cell("o")]],
        )
        assert hs.text_at(0, 1, 3) == "ell"

    def test_timestamp(self):
        hs = HardwareState()
        assert hs.timestamp > 0


# ── HardwareOutput ────────────────────────────────────────────────────


class TestHardwareOutput:
    def test_default(self):
        ho = HardwareOutput()
        assert ho.raw_bytes == b""

    def test_save_and_load(self, tmp_path):
        ho = HardwareOutput(raw_bytes=b"\x1b[2J\x1b[HHello")
        path = str(tmp_path / "test.hwf")
        ho.save(path)
        loaded = HardwareOutput.load(path)
        assert loaded.raw_bytes == ho.raw_bytes


# ── compare_hardware_states ──────────────────────────────────────────


class TestCompareHardwareStates:
    def test_identical(self):
        cells = [[Cell("X")] * 3 for _ in range(3)]
        s1 = HardwareState(rows=3, cols=3, cells=cells)
        s2 = HardwareState(rows=3, cols=3, cells=[[Cell("X")] * 3 for _ in range(3)])
        diff = compare_hardware_states(s1, s2)
        assert diff.identical
        assert diff.summary == "States are identical"

    def test_single_cell_diff(self):
        s1 = HardwareState(rows=1, cols=3, cells=[[Cell("A"), Cell("B"), Cell("C")]])
        s2 = HardwareState(rows=1, cols=3, cells=[[Cell("A"), Cell("X"), Cell("C")]])
        diff = compare_hardware_states(s1, s2)
        assert not diff.identical
        assert len(diff.cell_diffs) == 1
        assert diff.cell_diffs[0].col == 1

    def test_color_diff(self):
        s1 = HardwareState(rows=1, cols=1, cells=[[Cell("A", fg=Color.RED)]])
        s2 = HardwareState(rows=1, cols=1, cells=[[Cell("A", fg=Color.BLUE)]])
        diff = compare_hardware_states(s1, s2)
        assert not diff.identical
        assert len(diff.cell_diffs) == 1

    def test_attr_diff(self):
        s1 = HardwareState(rows=1, cols=1, cells=[[Cell("A", attr=Attr.BOLD)]])
        s2 = HardwareState(rows=1, cols=1, cells=[[Cell("A", attr=Attr.NONE)]])
        diff = compare_hardware_states(s1, s2)
        assert not diff.identical

    def test_size_diff(self):
        s1 = HardwareState(rows=3, cols=3)
        s2 = HardwareState(rows=5, cols=5)
        diff = compare_hardware_states(s1, s2)
        assert not diff.identical
        assert diff.size_diff

    def test_cursor_diff(self):
        s1 = HardwareState(rows=1, cols=1, cursor_row=0, cursor_col=0)
        s2 = HardwareState(rows=1, cols=1, cursor_row=0, cursor_col=1)
        diff = compare_hardware_states(s1, s2)
        assert not diff.identical
        assert diff.cursor_diff

    def test_multiple_diffs(self):
        s1 = HardwareState(rows=2, cols=2, cells=[[Cell("A"), Cell("B")], [Cell("C"), Cell("D")]])
        s2 = HardwareState(rows=2, cols=2, cells=[[Cell("X"), Cell("B")], [Cell("C"), Cell("X")]])
        diff = compare_hardware_states(s1, s2)
        assert not diff.identical
        assert len(diff.cell_diffs) == 2


# ── compare_hardware_outputs ─────────────────────────────────────────


class TestCompareHardwareOutputs:
    def test_identical(self):
        a = HardwareOutput(raw_bytes=b"hello")
        b = HardwareOutput(raw_bytes=b"hello")
        result = compare_hardware_outputs(a, b)
        assert result["identical"]
        assert result["byte_diff_count"] == 0

    def test_different(self):
        a = HardwareOutput(raw_bytes=b"hello")
        b = HardwareOutput(raw_bytes=b"world")
        result = compare_hardware_outputs(a, b)
        assert not result["identical"]
        # 'hello' vs 'world': h!=w, e!=o, l!=r, l==l, o!=d = 4 diffs
        assert result["byte_diff_count"] == 4
        assert result["first_diff_offset"] == 0

    def test_different_lengths(self):
        a = HardwareOutput(raw_bytes=b"hi")
        b = HardwareOutput(raw_bytes=b"hello")
        result = compare_hardware_outputs(a, b)
        assert not result["identical"]
        # 'hi' vs 'hello': h==h, i!=e → first diff at 1
        assert result["first_diff_offset"] == 1

    def test_empty(self):
        a = HardwareOutput()
        b = HardwareOutput()
        result = compare_hardware_outputs(a, b)
        assert result["identical"]


# ── terminal writes ─────────────────────────────────────────────────


class _BrokenTerminal(io.TextIOBase):
    """A pty whose far end has gone away."""

    def __init__(self) -> None:
        self.attempts = 0

    def write(self, s: str) -> int:
        self.attempts += 1
        raise BrokenPipeError(32, "Broken pipe")

    def flush(self) -> None:
        raise BrokenPipeError(32, "Broken pipe")


class _TransientTerminal(io.TextIOBase):
    """Fails the first write with EAGAIN, then succeeds."""

    def __init__(self) -> None:
        self.seen: list[str] = []
        self.fail_next = True

    def write(self, s: str) -> int:
        if self.fail_next:
            self.fail_next = False
            raise OSError(11, "Resource temporarily unavailable")
        self.seen.append(s)
        return len(s)

    def flush(self) -> None:
        return


class _ClosedTerminal(io.TextIOBase):
    """A closed file. Raises ValueError, which is deliberately *not* an OSError."""

    def write(self, s: str) -> int:
        raise ValueError("I/O operation on closed file")

    def flush(self) -> None:
        raise ValueError("I/O operation on closed file")


def _with_terminal(term: io.TextIOBase, call) -> object:
    """Run *call* with *term* as stdout, always restoring pytest's own.

    Exceptions propagate -- a test that wants "did not raise" simply succeeds
    by returning. What matters is that the restore runs regardless, because a
    leaked stdout would poison every subsequent test in the session.
    """
    real = sys.stdout
    sys.stdout = term
    try:
        return call()
    finally:
        sys.stdout = real


def _engine_with(text: str) -> tuple[GraphicsEngine, Layer]:
    eng = GraphicsEngine()
    _with_terminal(_TransientTerminal(), eng.open)
    layer = eng.create_layer("test", z=0)
    layer.write(0, 0, text)
    return eng, layer


class TestTerminalWrite:
    """The engine must survive a terminal that has gone away.

    Removing curses from the screen-writer deleted the ``except
    curses.error: pass`` that used to wrap ``addstr``, and nothing replaced it
    -- a plain ``BrokenPipeError`` propagated out of ``render()`` and killed the
    caller. That is not a hypothetical: a pty vanishes when an SSH session
    drops, an emulator is closed, or output is piped into something that stops
    reading early (``| head``).

    Three properties, and they are independent -- losing any one loses content
    silently rather than loudly:

    1. a write failure must not raise;
    2. a *transient* failure must not latch the engine off, or one ``EAGAIN``
       would mute it permanently;
    3. a failed write must not commit the shadow, or the cells are marked as
       already on screen and are never emitted again.

    Property 3 is the same cache-coherence invariant ncurses violated: a
    single-writer model where the writer's memory of the screen advances
    independently of what actually reached the screen.
    """

    def test_dead_terminal_does_not_raise(self):
        eng, layer = _engine_with("hello")
        term = _BrokenTerminal()
        # Simply reaching the end is the assertion.
        _with_terminal(term, eng.render)
        _with_terminal(term, lambda: eng.set_cursor(1, 1))
        _with_terminal(term, lambda: eng.show_cursor(True))
        assert layer is not None
        assert term.attempts >= 1, "nothing was even attempted"

    def test_dead_terminal_latches_off(self):
        eng, _layer = _engine_with("hello")
        _with_terminal(_BrokenTerminal(), eng.render)
        assert eng._stdout_dead is True
        # Latched: no further syscalls, so a dead pipe is not hammered at
        # frame rate for the life of the process.
        term = _BrokenTerminal()
        _with_terminal(term, eng.render)
        assert term.attempts == 0, "the engine kept writing to a dead terminal"

    def test_open_resets_the_latch(self):
        # A new session may have a new terminal. Staying mute across open()
        # would render a perfectly good screen into nothing.
        eng, _layer = _engine_with("hello")
        _with_terminal(_BrokenTerminal(), eng.render)
        assert eng._stdout_dead is True
        _with_terminal(_TransientTerminal(), eng.open)
        assert eng._stdout_dead is False

    def test_transient_eagain_does_not_latch_and_retries(self):
        eng, _layer = _engine_with("retry-me")
        term = _TransientTerminal()
        _with_terminal(term, eng.render)  # EAGAIN on the first frame
        first = eng._stdout_dead
        _with_terminal(term, eng.render)  # must land this time

        assert first is False, "one EAGAIN muted the engine for good"
        assert any("retry-me" in s for s in term.seen), (
            "content never reached the terminal on retry"
        )

    def test_failed_write_does_not_commit_the_shadow(self):
        eng, _layer = _engine_with("shadow-test")
        term = _TransientTerminal()
        _with_terminal(term, eng.render)  # fails, so the shadow must not advance
        _with_terminal(term, eng.render)  # must re-emit the same content
        emitted = [s for s in term.seen if "shadow-test" in s]
        assert emitted, (
            "a failed write committed the shadow: the content is now marked "
            "on-screen and will never be emitted again"
        )

    def test_closed_file_is_caught(self):
        # ValueError is what a closed file raises and it is not an OSError.
        # Catching only OSError misses the easiest failure to hit in tests.
        eng, _layer = _engine_with("closed")
        _with_terminal(_ClosedTerminal(), eng.render)
        assert eng._stdout_dead is True

    def test_close_restores_the_terminal_even_after_a_failure(self):
        eng, _layer = _engine_with("x")
        _with_terminal(_BrokenTerminal(), eng.render)
        # close() must still *attempt* the restore: on a live terminal a
        # skipped restore strands the user in the alternate screen.
        term = _BrokenTerminal()
        _with_terminal(term, eng.close)
        assert term.attempts == 1, "restore was skipped"
        assert eng._alt_screen is False
