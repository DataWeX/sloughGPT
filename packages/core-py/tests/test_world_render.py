"""Tests for domains.shell.world_render — RenderConfig, RenderDiff, RenderHistory, RenderAnalyzer.

Covers: dataclass defaults, diff computation, history CRUD, series analysis,
significant change detection, summary, diff summary text. Pure numpy tests,
no rendering dependencies.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_core_dir = str(Path(__file__).resolve().parents[2])
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

from domains.shell.world_render import (
    RenderConfig,
    RenderDiff,
    RenderHistory,
    RenderAnalyzer,
    MATERIAL_AIR,
    MATERIAL_GROUND,
    MATERIAL_FOOD,
    MATERIAL_TOXIC,
    MATERIAL_SIGNAL,
    MATERIAL_NEST,
    MATERIAL_WATER,
)


class TestMaterialConstants:
    def test_values_are_distinct(self):
        mats = [MATERIAL_AIR, MATERIAL_GROUND, MATERIAL_FOOD,
                MATERIAL_TOXIC, MATERIAL_SIGNAL, MATERIAL_NEST, MATERIAL_WATER]
        assert len(set(mats)) == 7

    def test_air_is_zero(self):
        assert MATERIAL_AIR == 0


class TestRenderConfig:
    def test_defaults(self):
        c = RenderConfig()
        assert c.width == 160
        assert c.height == 120
        assert c.samples == 16
        assert c.voxel_size == 1.0

    def test_custom(self):
        c = RenderConfig(width=320, height=240, samples=32)
        assert c.width == 320
        assert c.height == 240
        assert c.samples == 32

    def test_color_map_has_all_materials(self):
        c = RenderConfig()
        for mat in [MATERIAL_AIR, MATERIAL_GROUND, MATERIAL_FOOD,
                    MATERIAL_TOXIC, MATERIAL_SIGNAL, MATERIAL_NEST, MATERIAL_WATER]:
            assert mat in c.material_color_map

    def test_emission_map(self):
        c = RenderConfig()
        assert MATERIAL_SIGNAL in c.material_emission_map
        assert MATERIAL_FOOD in c.material_emission_map


class TestRenderDiff:
    def test_identical_images(self):
        img = np.ones((10, 10, 3), dtype=np.float32) * 0.5
        d = RenderDiff(img, img)
        assert d.mse == 0.0
        assert d.mae == 0.0
        assert d.max_diff == 0.0
        assert d.changed_pixels == 0
        assert d.change_ratio == 0.0

    def test_different_images(self):
        a = np.zeros((10, 10, 3), dtype=np.float32)
        b = np.ones((10, 10, 3), dtype=np.float32)
        d = RenderDiff(a, b)
        assert d.mse > 0
        assert d.mae > 0
        assert d.max_diff > 0
        assert d.changed_pixels == 100
        assert d.change_ratio == 1.0

    def test_mismatched_sizes(self):
        a = np.zeros((10, 10, 3), dtype=np.float32)
        b = np.ones((20, 20, 3), dtype=np.float32)
        d = RenderDiff(a, b)
        assert d.total_pixels == 100

    def test_summary(self):
        img = np.ones((10, 10, 3), dtype=np.float32) * 0.5
        d = RenderDiff(img, img)
        s = d.summary()
        assert "mse" in s
        assert "mae" in s
        assert "change_ratio" in s

    def test_heatmap(self):
        a = np.zeros((10, 10, 3), dtype=np.float32)
        b = np.ones((10, 10, 3), dtype=np.float32)
        d = RenderDiff(a, b)
        h = d.heatmap()
        assert h.shape == (10, 10)
        assert h.max() <= 1.0

    def test_diff_image(self):
        a = np.zeros((10, 10, 3), dtype=np.float32)
        b = np.ones((10, 10, 3), dtype=np.float32)
        d = RenderDiff(a, b)
        di = d.diff_image()
        assert di.shape == (10, 10, 3)
        assert di.max() <= 1.0


class TestRenderHistory:
    def test_add_and_get(self):
        h = RenderHistory()
        img = np.ones((5, 5, 3), dtype=np.float32)
        idx = h.add(img, tick=1)
        assert idx == 0
        assert h.get(0) is not None
        np.testing.assert_array_equal(h.get(0), img)

    def test_get_out_of_range(self):
        h = RenderHistory()
        assert h.get(0) is None
        assert h.get(-1) is None

    def test_max_entries(self):
        h = RenderHistory(max_entries=3)
        for i in range(5):
            h.add(np.ones((2, 2, 3), dtype=np.float32), tick=i)
        assert len(h) == 3
        assert h[0]["tick"] == 2

    def test_diff(self):
        h = RenderHistory()
        a = np.zeros((5, 5, 3), dtype=np.float32)
        b = np.ones((5, 5, 3), dtype=np.float32)
        h.add(a, tick=0)
        h.add(b, tick=1)
        d = h.diff(0, 1)
        assert d is not None
        assert d.mse > 0

    def test_diff_out_of_range(self):
        h = RenderHistory()
        assert h.diff(0, 1) is None

    def test_diff_latest(self):
        h = RenderHistory()
        assert h.diff_latest() is None
        h.add(np.zeros((5, 5, 3), dtype=np.float32))
        assert h.diff_latest() is None
        h.add(np.ones((5, 5, 3), dtype=np.float32))
        d = h.diff_latest()
        assert d is not None
        assert d.mse > 0

    def test_timeline(self):
        h = RenderHistory()
        h.add(np.ones((2, 2, 3), dtype=np.float32), tick=0)
        h.add(np.ones((2, 2, 3), dtype=np.float32), tick=1)
        tl = h.timeline()
        assert len(tl) == 2
        assert tl[0]["tick"] == 0

    def test_recent(self):
        h = RenderHistory()
        for i in range(5):
            h.add(np.ones((2, 2, 3), dtype=np.float32), tick=i)
        r = h.recent(2)
        assert len(r) == 2
        assert r[0]["tick"] == 3

    def test_len(self):
        h = RenderHistory()
        assert len(h) == 0
        h.add(np.ones((2, 2, 3), dtype=np.float32))
        assert len(h) == 1

    def test_getitem(self):
        h = RenderHistory()
        h.add(np.ones((2, 2, 3), dtype=np.float32), tick=42)
        entry = h[0]
        assert entry["tick"] == 42
        assert h[999] is None

    def test_clear(self):
        h = RenderHistory()
        h.add(np.ones((2, 2, 3), dtype=np.float32))
        h.clear()
        assert len(h) == 0

    def test_metadata(self):
        h = RenderHistory()
        h.add(np.ones((2, 2, 3), dtype=np.float32), metadata={"key": "val"})
        entry = h[0]
        assert entry["metadata"]["key"] == "val"

    def test_image_is_copied(self):
        h = RenderHistory()
        img = np.ones((2, 2, 3), dtype=np.float32)
        h.add(img)
        img[:] = 0
        np.testing.assert_array_equal(h.get(0), np.ones((2, 2, 3), dtype=np.float32))


class TestRenderAnalyzer:
    def test_analyze_empty(self):
        a = RenderAnalyzer()
        result = a.analyze_series()
        assert result["count"] == 0

    def test_analyze_series(self):
        h = RenderHistory()
        h.add(np.ones((2, 2, 3), dtype=np.float32) * 0.3, tick=0)
        h.add(np.ones((2, 2, 3), dtype=np.float32) * 0.7, tick=1)
        a = RenderAnalyzer(h)
        result = a.analyze_series()
        assert result["count"] == 2
        assert result["mean_trend"] > 0

    def test_detect_significant_changes(self):
        h = RenderHistory()
        h.add(np.zeros((10, 10, 3), dtype=np.float32), tick=0)
        h.add(np.ones((10, 10, 3), dtype=np.float32), tick=1)
        a = RenderAnalyzer(h)
        changes = a.detect_significant_changes(threshold=0.01)
        assert len(changes) == 1
        assert changes[0]["change_ratio"] > 0.01

    def test_detect_no_changes(self):
        h = RenderHistory()
        img = np.ones((10, 10, 3), dtype=np.float32) * 0.5
        h.add(img, tick=0)
        h.add(img.copy(), tick=1)
        a = RenderAnalyzer(h)
        changes = a.detect_significant_changes(threshold=0.1)
        assert len(changes) == 0

    def test_compare_ticks(self):
        h = RenderHistory()
        h.add(np.zeros((5, 5, 3), dtype=np.float32), tick=10)
        h.add(np.ones((5, 5, 3), dtype=np.float32), tick=20)
        a = RenderAnalyzer(h)
        d = a.compare_ticks(10, 20)
        assert d is not None
        assert d.mse > 0

    def test_compare_ticks_missing(self):
        h = RenderHistory()
        h.add(np.zeros((5, 5, 3), dtype=np.float32), tick=10)
        a = RenderAnalyzer(h)
        assert a.compare_ticks(10, 99) is None

    def test_summary(self):
        h = RenderHistory()
        h.add(np.zeros((10, 10, 3), dtype=np.float32), tick=0)
        h.add(np.ones((10, 10, 3), dtype=np.float32), tick=1)
        a = RenderAnalyzer(h)
        s = a.summary()
        assert s["count"] == 2
        assert s["significant_changes"] >= 1

    def test_summary_empty(self):
        a = RenderAnalyzer()
        s = a.summary()
        assert s["count"] == 0
        assert s["significant_changes"] == 0

    def test_render_diff_summary(self):
        h = RenderHistory()
        h.add(np.zeros((5, 5, 3), dtype=np.float32))
        h.add(np.ones((5, 5, 3), dtype=np.float32))
        a = RenderAnalyzer(h)
        text = a.render_diff_summary(0, 1)
        assert "MSE" in text
        assert "MAE" in text

    def test_render_diff_summary_out_of_range(self):
        a = RenderAnalyzer()
        text = a.render_diff_summary(0, 1)
        assert "Cannot compare" in text
