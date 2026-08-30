"""Tests for domains.shell.pane — Rect, Border, Pane, PaneLayout, split, vsplit."""

import pytest
from domains.shell.pane import Rect, Border, Pane, PaneLayout, split, vsplit


# ── Rect ──────────────────────────────────────────────────────────────


class TestRect:
    def test_basic(self):
        r = Rect(0, 0, 10, 20)
        assert r.top == 0
        assert r.rows == 10
        assert r.cols == 20

    def test_bool_truthy(self):
        assert bool(Rect(0, 0, 1, 1)) is True

    def test_bool_falsy_zero_rows(self):
        assert bool(Rect(0, 0, 0, 10)) is False

    def test_bool_falsy_zero_cols(self):
        assert bool(Rect(0, 0, 10, 0)) is False

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="negative"):
            Rect(-1, 0, 10, 10)

    def test_inset(self):
        r = Rect(0, 0, 20, 80)
        inner = r.inset(top=2, left=4, bottom=2, right=4)
        assert inner == Rect(2, 4, 16, 72)

    def test_inset_clamps_to_zero(self):
        r = Rect(0, 0, 2, 4)
        inner = r.inset(top=3, left=5)
        assert inner.rows == 0
        assert inner.cols == 0


# ── Border ────────────────────────────────────────────────────────────


class TestBorder:
    def test_none(self):
        b = Border()
        assert b.is_empty is True
        assert b.top is False
        assert b.bottom is False
        assert b.left is False
        assert b.right is False

    def test_top(self):
        b = Border("top")
        assert b.top is True
        assert b.bottom is False

    def test_bottom(self):
        b = Border("bottom")
        assert b.bottom is True
        assert b.top is False

    def test_horizontal(self):
        b = Border("horizontal")
        assert b.top is True
        assert b.bottom is True
        assert b.left is False
        assert b.right is False

    def test_vertical(self):
        b = Border("vertical")
        assert b.left is True
        assert b.right is True
        assert b.top is False
        assert b.bottom is False

    def test_all(self):
        b = Border("all")
        assert b.top is True
        assert b.bottom is True
        assert b.left is True
        assert b.right is True


# ── Pane ──────────────────────────────────────────────────────────────


class TestPane:
    def test_defaults(self):
        p = Pane(name="test")
        assert p.ratio == 0.0
        assert p.fixed is None
        assert p.width_ratio == 1.0
        assert p.visible is True
        assert p.focusable is True
        assert p.border.is_empty is True
        assert p.padding == (0, 0, 0, 0)

    def test_ratio_out_of_range(self):
        with pytest.raises(ValueError, match="ratio"):
            Pane(name="test", ratio=1.5)

    def test_negative_fixed(self):
        with pytest.raises(ValueError, match="fixed"):
            Pane(name="test", fixed=-1)

    def test_negative_max_rows(self):
        with pytest.raises(ValueError, match="max_rows"):
            Pane(name="test", max_rows=-1)

    def test_border_top(self):
        p = Pane(name="t", border=Border("top"))
        assert p.border_top == 1
        assert p.border_bottom == 0
        assert p.outer_height(10) == 11

    def test_border_bottom(self):
        p = Pane(name="b", border=Border("bottom"))
        assert p.border_bottom == 1
        assert p.outer_height(10) == 11

    def test_border_horizontal(self):
        p = Pane(name="h", border=Border("horizontal"))
        assert p.outer_height(10) == 12
        assert p.outer_width(80) == 80

    def test_border_vertical(self):
        p = Pane(name="v", border=Border("vertical"))
        assert p.outer_height(10) == 10
        assert p.outer_width(80) == 82

    def test_border_all(self):
        p = Pane(name="a", border=Border("all"))
        assert p.outer_height(10) == 12
        assert p.outer_width(80) == 82

    def test_content_rect_with_border(self):
        p = Pane(name="t", border=Border("all"))
        rect = Rect(0, 0, 12, 82)
        cr = p.content_rect(rect)
        assert cr == Rect(1, 1, 10, 80)

    def test_content_rect_with_padding(self):
        p = Pane(name="p", padding=(1, 2, 1, 2))
        rect = Rect(0, 0, 10, 20)
        cr = p.content_rect(rect)
        assert cr == Rect(1, 2, 8, 16)

    def test_content_rect_border_and_padding(self):
        p = Pane(name="bp", border=Border("all"), padding=(1, 2, 1, 2))
        rect = Rect(0, 0, 14, 24)
        cr = p.content_rect(rect)
        assert cr == Rect(2, 3, 10, 18)


# ── PaneLayout ────────────────────────────────────────────────────────


class TestPaneLayout:
    def test_empty(self):
        layout = PaneLayout([])
        assert layout.compute(24, 80) == {}

    def test_zero_dimensions(self):
        layout = PaneLayout([Pane(name="a")])
        assert layout.compute(0, 80) == {}

    def test_single_pane_fills(self):
        layout = PaneLayout([Pane(name="main")])
        regions = layout.compute(24, 80)
        assert regions["main"].rows == 24
        assert regions["main"].cols == 80

    def test_two_pane_ratio(self):
        layout = PaneLayout([
            Pane(name="top", ratio=0.3),
            Pane(name="bot", ratio=0.7),
        ])
        regions = layout.compute(20, 80)
        total = regions["top"].rows + regions["bot"].rows
        assert total == 20

    def test_fixed_pane(self):
        layout = PaneLayout([
            Pane(name="bar", fixed=3),
            Pane(name="main"),
        ])
        regions = layout.compute(24, 80)
        assert regions["bar"].rows == 3
        assert regions["main"].rows == 21

    def test_width_ratio(self):
        layout = PaneLayout([Pane(name="a", width_ratio=0.5)])
        regions = layout.compute(24, 80)
        assert regions["a"].cols == 40
        assert regions["a"].left == 20

    # ── Border layout ──────────────────────────────────────────────────

    def test_border_horizontal_single_pane(self):
        p = Pane(name="a", border=Border("horizontal"))
        layout = PaneLayout([p])
        regions = layout.compute(10, 80)
        # 10 total - 2 border rows = 8 content, outer = 10
        assert regions["a"].rows == 10
        assert regions["a"].cols == 80

    def test_border_between_panes(self):
        layout = PaneLayout([
            Pane(name="top", ratio=0.5, border=Border("bottom")),
            Pane(name="bot", ratio=0.5),
        ])
        regions = layout.compute(20, 80)
        # border_bottom of "top" consumes 1 row from available space
        total = regions["top"].rows + regions["bot"].rows
        assert total == 20
        assert regions["top"].rows == 11  # 10 content + 1 border
        assert regions["bot"].rows == 9   # 9 content (1 row lost to border)

    def test_hidden_pane_skipped(self):
        layout = PaneLayout([
            Pane(name="top", visible=False),
            Pane(name="bot"),
        ])
        regions = layout.compute(20, 80)
        assert "top" not in regions
        assert regions["bot"].rows == 20

    # ── max_rows ───────────────────────────────────────────────────────

    def test_max_rows_clamps(self):
        layout = PaneLayout([
            Pane(name="a", ratio=1.0, max_rows=5),
            Pane(name="b", ratio=1.0),
        ])
        regions = layout.compute(20, 80)
        assert regions["a"].rows <= 5

    # ── Content regions ────────────────────────────────────────────────

    def test_content_regions_strip_border(self):
        p = Pane(name="a", border=Border("all"))
        layout = Layout = PaneLayout([p])
        cr = layout.content_regions(12, 82)
        assert cr["a"] == Rect(1, 1, 10, 80)

    def test_content_regions_strip_padding(self):
        p = Pane(name="a", padding=(1, 2, 1, 2))
        layout = PaneLayout([p])
        cr = layout.content_regions(10, 20)
        assert cr["a"] == Rect(1, 2, 8, 16)


# ── Focus ─────────────────────────────────────────────────────────────


class TestFocus:
    def test_initial_focus(self):
        layout = PaneLayout([Pane(name="a"), Pane(name="b")])
        assert layout.focus_index == 0
        assert layout.focus_name == "a"

    def test_set_focus(self):
        layout = PaneLayout([Pane(name="a"), Pane(name="b")])
        assert layout.set_focus("b") is True
        assert layout.focus_name == "b"

    def test_set_focus_not_found(self):
        layout = PaneLayout([Pane(name="a")])
        assert layout.set_focus("z") is False

    def test_set_focus_not_focusable(self):
        layout = PaneLayout([
            Pane(name="a"),
            Pane(name="b", focusable=False),
        ])
        assert layout.set_focus("b") is False
        assert layout.focus_name == "a"

    def test_set_focus_not_visible(self):
        layout = PaneLayout([
            Pane(name="a"),
            Pane(name="b", visible=False),
        ])
        assert layout.set_focus("b") is False
        assert layout.focus_name == "a"

    def test_focus_next(self):
        layout = PaneLayout([Pane(name="a"), Pane(name="b"), Pane(name="c")])
        assert layout.focus_next() == "b"
        assert layout.focus_next() == "c"
        assert layout.focus_next() == "a"  # wraps

    def test_focus_next_skips_unfocusable(self):
        layout = PaneLayout([
            Pane(name="a"),
            Pane(name="b", focusable=False),
            Pane(name="c"),
        ])
        assert layout.focus_next() == "c"
        assert layout.focus_next() == "a"  # wraps

    def test_focus_next_skips_hidden(self):
        layout = PaneLayout([
            Pane(name="a"),
            Pane(name="b", visible=False),
            Pane(name="c"),
        ])
        assert layout.focus_next() == "c"

    def test_focus_prev(self):
        layout = PaneLayout([Pane(name="a"), Pane(name="b"), Pane(name="c")])
        assert layout.focus_prev() == "c"
        assert layout.focus_prev() == "b"
        assert layout.focus_prev() == "a"  # wraps

    def test_focus_empty(self):
        layout = PaneLayout([])
        assert layout.focus_next() is None
        assert layout.focus_prev() is None
        assert layout.focus_name is None


# ── Visibility ────────────────────────────────────────────────────────


class TestVisibility:
    def test_set_visible(self):
        layout = PaneLayout([Pane(name="a"), Pane(name="b")])
        assert layout.set_visible("a", False) is True
        assert layout.is_visible("a") is False

    def test_set_visible_not_found(self):
        layout = PaneLayout([Pane(name="a")])
        assert layout.set_visible("z", False) is False

    def test_hidden_pane_not_in_regions(self):
        layout = PaneLayout([Pane(name="a"), Pane(name="b")])
        layout.set_visible("a", False)
        regions = layout.compute(20, 80)
        assert "a" not in regions
        assert regions["b"].rows == 20


# ── Split ─────────────────────────────────────────────────────────────


class TestSplit:
    def test_equal_split(self):
        rects = split(20, 80, [0.5, 0.5])
        assert len(rects) == 2
        assert rects[0].rows + rects[1].rows == 20

    def test_min_rows(self):
        rects = split(20, 80, [0.5, 0.5], min_rows=[5, 5])
        assert rects[0].rows >= 5
        assert rects[1].rows >= 5

    def test_single(self):
        rects = split(10, 80, [1.0])
        assert len(rects) == 1
        assert rects[0].rows == 10


# ── Vsplit ────────────────────────────────────────────────────────────


class TestVsplit:
    def test_equal_split(self):
        rects = vsplit(24, 80, [0.5, 0.5])
        assert len(rects) == 2
        assert rects[0].cols + rects[1].cols == 80
        assert rects[0].left == 0
        assert rects[1].left == rects[0].cols

    def test_three_way(self):
        rects = vsplit(24, 90, [1 / 3, 1 / 3, 1 / 3])
        total = sum(r.cols for r in rects)
        assert total == 90

    def test_min_cols(self):
        rects = vsplit(24, 80, [0.5, 0.5], min_cols=[30, 30])
        assert rects[0].cols >= 30
        assert rects[1].cols >= 30

    def test_single(self):
        rects = vsplit(24, 80, [1.0])
        assert len(rects) == 1
        assert rects[0].cols == 80
