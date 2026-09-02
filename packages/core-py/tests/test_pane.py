"""Tests for domains.shell.pane — Rect, Border, Pane, PaneLayout, split, vsplit."""

from __future__ import annotations

import pytest
from domains.shell.pane import (
    Rect,
    Border,
    Pane,
    PaneLayout,
    split,
    vsplit,
)


# ── Rect ──────────────────────────────────────────────────────────────────────

class TestRect:
    def test_basic(self):
        r = Rect(top=0, left=0, rows=10, cols=20)
        assert r.top == 0
        assert r.rows == 10
        assert r.cols == 20

    def test_bool_true(self):
        assert bool(Rect(0, 0, 5, 5)) is True

    def test_bool_zero_size(self):
        assert bool(Rect(0, 0, 0, 5)) is False
        assert bool(Rect(0, 0, 5, 0)) is False

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="negative"):
            Rect(0, 0, -1, 5)
        with pytest.raises(ValueError, match="negative"):
            Rect(-1, 0, 5, 5)

    def test_inset(self):
        r = Rect(0, 0, 20, 40)
        inner = r.inset(top=2, left=3, bottom=4, right=5)
        assert inner == Rect(2, 3, 14, 32)

    def test_inset_clamps_to_zero(self):
        r = Rect(0, 0, 5, 5)
        inner = r.inset(top=10, left=10)
        assert inner.rows == 0
        assert inner.cols == 0


# ── Border ────────────────────────────────────────────────────────────────────

class TestBorder:
    def test_none(self):
        b = Border("none")
        assert b.top is False
        assert b.bottom is False
        assert b.left is False
        assert b.right is False
        assert b.is_empty is True

    def test_top(self):
        b = Border("top")
        assert b.top is True
        assert b.bottom is False
        assert b.is_empty is False

    def test_bottom(self):
        b = Border("bottom")
        assert b.bottom is True

    def test_horizontal(self):
        b = Border("horizontal")
        assert b.top is True
        assert b.bottom is True
        assert b.left is False

    def test_vertical(self):
        b = Border("vertical")
        assert b.left is True
        assert b.right is True
        assert b.top is False

    def test_all(self):
        b = Border("all")
        assert b.top is True
        assert b.bottom is True
        assert b.left is True
        assert b.right is True


# ── Pane ──────────────────────────────────────────────────────────────────────

class TestPane:
    def test_defaults(self):
        p = Pane(name="a")
        assert p.ratio == 0.0
        assert p.visible is True
        assert p.focusable is True

    def test_ratio_validation(self):
        with pytest.raises(ValueError, match="ratio"):
            Pane(name="a", ratio=1.5)
        with pytest.raises(ValueError, match="ratio"):
            Pane(name="a", ratio=-0.1)

    def test_fixed_negative_raises(self):
        with pytest.raises(ValueError, match="fixed"):
            Pane(name="a", fixed=-5)

    def test_border_properties(self):
        p = Pane(name="a", border=Border("all"))
        assert p.border_top == 1
        assert p.border_bottom == 1
        assert p.border_left == 1
        assert p.border_right == 1

    def test_border_none(self):
        p = Pane(name="a")
        assert p.border_top == 0
        assert p.border_bottom == 0

    def test_padding(self):
        p = Pane(name="a", padding=(1, 2, 3, 4))
        assert p.pad_top == 1
        assert p.pad_left == 2
        assert p.pad_bottom == 3
        assert p.pad_right == 4

    def test_outer_height(self):
        p = Pane(name="a", border=Border("horizontal"))
        assert p.outer_height(10) == 12  # 10 + top border + bottom border

    def test_outer_width(self):
        p = Pane(name="a", border=Border("vertical"))
        assert p.outer_width(10) == 12

    def test_content_rect(self):
        p = Pane(name="a", border=Border("all"), padding=(1, 2, 3, 4))
        r = Rect(0, 0, 20, 40)
        inner = p.content_rect(r)
        assert inner == Rect(2, 3, 14, 32)  # top=1+1, left=1+2, bottom=1+3, right=1+4


# ── PaneLayout ────────────────────────────────────────────────────────────────

class TestPaneLayout:
    def test_empty(self):
        layout = PaneLayout()
        assert layout.compute(20, 80) == {}
        assert layout.focus_name is None

    def test_single_pane(self):
        p = Pane(name="main", ratio=1.0)
        layout = PaneLayout([p])
        regions = layout.compute(20, 80)
        assert "main" in regions
        assert regions["main"].rows == 20

    def test_two_panes_ratio(self):
        p1 = Pane(name="top", ratio=0.5)
        p2 = Pane(name="bottom", ratio=0.5)
        layout = PaneLayout([p1, p2])
        regions = layout.compute(20, 80)
        total = sum(r.rows for r in regions.values())
        assert total == 20

    def test_fixed_pane(self):
        p1 = Pane(name="status", fixed=3)
        p2 = Pane(name="main", ratio=1.0)
        layout = PaneLayout([p1, p2])
        regions = layout.compute(20, 80)
        assert regions["status"].rows == 3
        assert regions["main"].rows == 17

    def test_border_consumes_rows(self):
        p1 = Pane(name="a", ratio=0.5, border=Border("bottom"))
        p2 = Pane(name="b", ratio=0.5)
        layout = PaneLayout([p1, p2])
        regions = layout.compute(20, 80)
        total = sum(r.rows for r in regions.values())
        assert total == 20

    def test_hidden_pane_skipped(self):
        p1 = Pane(name="a", ratio=0.5)
        p2 = Pane(name="b", ratio=0.5, visible=False)
        layout = PaneLayout([p1, p2])
        regions = layout.compute(20, 80)
        assert "b" not in regions
        assert regions["a"].rows == 20

    def test_min_rows(self):
        p1 = Pane(name="a", ratio=0.1, min_rows=8)
        p2 = Pane(name="b", ratio=0.9)
        layout = PaneLayout([p1, p2])
        regions = layout.compute(20, 80)
        assert regions["a"].rows >= 8

    def test_max_rows(self):
        p1 = Pane(name="a", ratio=1.0, max_rows=5)
        layout = PaneLayout([p1])
        regions = layout.compute(20, 80)
        assert regions["a"].rows == 5

    def test_zero_dimensions(self):
        p = Pane(name="a", ratio=1.0)
        layout = PaneLayout([p])
        assert layout.compute(0, 80) == {}
        assert layout.compute(20, 0) == {}


# ── Focus ─────────────────────────────────────────────────────────────────────

class TestFocus:
    def test_initial_focus(self):
        p1 = Pane(name="a")
        p2 = Pane(name="b")
        layout = PaneLayout([p1, p2])
        assert layout.focus_index == 0
        assert layout.focus_name == "a"

    def test_set_focus(self):
        p1 = Pane(name="a")
        p2 = Pane(name="b")
        layout = PaneLayout([p1, p2])
        assert layout.set_focus("b") is True
        assert layout.focus_name == "b"

    def test_set_focus_not_found(self):
        p1 = Pane(name="a")
        layout = PaneLayout([p1])
        assert layout.set_focus("nonexistent") is False

    def test_set_focus_not_focusable(self):
        p1 = Pane(name="a")
        p2 = Pane(name="b", focusable=False)
        layout = PaneLayout([p1, p2])
        assert layout.set_focus("b") is False
        assert layout.focus_name == "a"

    def test_focus_next(self):
        p1 = Pane(name="a")
        p2 = Pane(name="b")
        p3 = Pane(name="c")
        layout = PaneLayout([p1, p2, p3])
        layout.set_focus("a")
        assert layout.focus_next() == "b"
        assert layout.focus_next() == "c"
        assert layout.focus_next() == "a"  # wraps

    def test_focus_prev(self):
        p1 = Pane(name="a")
        p2 = Pane(name="b")
        p3 = Pane(name="c")
        layout = PaneLayout([p1, p2, p3])
        layout.set_focus("b")
        assert layout.focus_prev() == "a"
        assert layout.focus_prev() == "c"  # wraps

    def test_focus_skips_hidden(self):
        p1 = Pane(name="a")
        p2 = Pane(name="b", visible=False)
        p3 = Pane(name="c")
        layout = PaneLayout([p1, p2, p3])
        layout.set_focus("a")
        assert layout.focus_next() == "c"

    def test_focus_skips_unfocusable(self):
        p1 = Pane(name="a")
        p2 = Pane(name="b", focusable=False)
        p3 = Pane(name="c")
        layout = PaneLayout([p1, p2, p3])
        layout.set_focus("a")
        assert layout.focus_next() == "c"


# ── Visibility ────────────────────────────────────────────────────────────────

class TestVisibility:
    def test_set_visible(self):
        p = Pane(name="a")
        layout = PaneLayout([p])
        assert layout.set_visible("a", False) is True
        assert layout.is_visible("a") is False

    def test_set_visible_not_found(self):
        layout = PaneLayout()
        assert layout.set_visible("x", False) is False


# ── split / vsplit ────────────────────────────────────────────────────────────

class TestSplitHelpers:
    def test_split_basic(self):
        rects = split(20, 80, [0.5, 0.5])
        assert len(rects) == 2
        total = sum(r.rows for r in rects)
        assert total == 20

    def test_split_with_min_rows(self):
        rects = split(20, 80, [0.1, 0.9], min_rows=[5, 5])
        assert rects[0].rows >= 5
        assert rects[1].rows >= 5

    def test_split_with_borders(self):
        rects = split(20, 80, [0.5, 0.5], borders="top")
        assert len(rects) == 2
        total = sum(r.rows for r in rects)
        assert total == 20

    def test_vsplit_basic(self):
        rects = vsplit(20, 80, [0.5, 0.5])
        assert len(rects) == 2
        total = sum(r.cols for r in rects)
        assert total == 80
        # All at same top
        assert all(r.top == 0 for r in rects)

    def test_vsplit_with_min_cols(self):
        rects = vsplit(20, 100, [0.1, 0.9], min_cols=[20, 20])
        assert rects[0].cols >= 20
        assert rects[1].cols >= 20

    def test_vsplit_remainder(self):
        rects = vsplit(10, 100, [0.33, 0.33, 0.34])
        total = sum(r.cols for r in rects)
        assert total == 100
