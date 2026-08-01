"""Tests for domains/shell/pane.py — the pure layout engine."""

import pytest

from domains.shell.pane import Pane, PaneLayout, Rect, split


class TestRect:
    def test_positive_region(self):
        r = Rect(top=1, left=2, rows=10, cols=20)
        assert r.top == 1
        assert r.left == 2
        assert r.rows == 10
        assert r.cols == 20

    def test_negative_rows_rejected(self):
        with pytest.raises(ValueError):
            Rect(top=0, left=0, rows=-1, cols=10)

    def test_negative_cols_rejected(self):
        with pytest.raises(ValueError):
            Rect(top=0, left=0, rows=10, cols=-1)

    def test_truthiness_empty(self):
        assert not Rect(top=0, left=0, rows=0, cols=10)
        assert not Rect(top=0, left=0, rows=10, cols=0)
        assert Rect(top=0, left=0, rows=10, cols=10)


class TestPane:
    def test_ratio_bounds(self):
        with pytest.raises(ValueError):
            Pane("bad", ratio=1.5)
        with pytest.raises(ValueError):
            Pane("bad", ratio=-0.1)

    def test_negative_fixed_rejected(self):
        with pytest.raises(ValueError):
            Pane("bad", fixed=-2)


class TestPaneLayout:
    def test_three_panes_cover_height(self):
        layout = PaneLayout([
            Pane("console", ratio=0.3, min_rows=4),
            Pane("output", ratio=0.7, min_rows=6),
            Pane("input", fixed=1),
        ])
        regions = layout.compute(rows=40, cols=80)
        assert set(regions) == {"console", "output", "input"}
        total = sum(r.rows for r in regions.values())
        assert total == 40

    def test_input_is_single_row(self):
        layout = PaneLayout([Pane("a", ratio=0.5), Pane("input", fixed=1)])
        regions = layout.compute(rows=20, cols=50)
        assert regions["input"].rows == 1
        assert regions["input"].top == 19

    def test_regions_stack_vertically(self):
        layout = PaneLayout([
            Pane("top", ratio=0.5),
            Pane("bottom", ratio=0.5),
        ])
        regions = layout.compute(rows=20, cols=40)
        assert regions["top"].top == 0
        assert regions["bottom"].top == 10

    def test_full_width_by_default(self):
        layout = PaneLayout([Pane("a", ratio=1.0)])
        regions = layout.compute(rows=10, cols=30)
        assert regions["a"].cols == 30
        assert regions["a"].left == 0

    def test_width_ratio_centers_pane(self):
        layout = PaneLayout([Pane("a", ratio=1.0, width_ratio=0.5)])
        regions = layout.compute(rows=10, cols=40)
        assert regions["a"].cols == 20
        assert regions["a"].left == 10

    def test_zero_geometry_returns_empty(self):
        layout = PaneLayout([Pane("a", ratio=1.0)])
        assert layout.compute(rows=0, cols=0) == {}

    def test_fixed_panes_respected(self):
        layout = PaneLayout([Pane("a", fixed=3), Pane("b", ratio=1.0)])
        regions = layout.compute(rows=10, cols=10)
        assert regions["a"].rows == 3
        assert regions["b"].rows == 7

    def test_min_rows_floored(self):
        layout = PaneLayout([Pane("a", ratio=0.1, min_rows=5), Pane("b", ratio=0.9)])
        regions = layout.compute(rows=50, cols=10)
        assert regions["a"].rows >= 5

    def test_last_pane_absorbs_remainder(self):
        layout = PaneLayout([Pane("a", ratio=0.3), Pane("b", ratio=0.7)])
        regions = layout.compute(rows=31, cols=10)
        assert regions["a"].rows + regions["b"].rows == 31


class TestSplit:
    def test_split_bands_cover(self):
        rects = split(rows=30, cols=40, ratios=[0.3, 0.7])
        assert len(rects) == 2
        assert rects[0].top == 0
        assert rects[0].rows + rects[1].rows == 30
        assert rects[1].top == rects[0].rows

    def test_split_with_min_rows(self):
        rects = split(rows=30, cols=40, ratios=[0.1, 0.9], min_rows=[5, 5])
        assert rects[0].rows >= 5

    def test_split_contiguous(self):
        rects = split(rows=24, cols=30, ratios=[0.5, 0.25, 0.25])
        y = 0
        for r in rects:
            assert r.top == y
            y += r.rows
        assert y == 24
