"""Tests for domains.shell.pane — Rect, Pane, PaneLayout, split."""

import pytest
from domains.shell.pane import Rect, Pane, PaneLayout, split


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


class TestPane:
    def test_defaults(self):
        p = Pane(name="test")
        assert p.ratio == 0.0
        assert p.fixed is None
        assert p.width_ratio == 1.0

    def test_ratio_out_of_range(self):
        with pytest.raises(ValueError, match="ratio"):
            Pane(name="test", ratio=1.5)

    def test_negative_fixed(self):
        with pytest.raises(ValueError, match="fixed"):
            Pane(name="test", fixed=-1)


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
