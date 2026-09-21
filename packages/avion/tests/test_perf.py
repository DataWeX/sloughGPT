"""Perf tests — marks, measure blocks, percentiles, slowest."""

import time

from arken.perf import PerformanceMarker


class TestMarkers:
    def test_mark_and_summary(self):
        p = PerformanceMarker()
        p.mark("goto", 10.0)
        p.mark("goto", 30.0)
        s = p.summary_for("goto")
        assert s.count == 2
        assert s.avg_ms == 20.0
        assert s.max_ms == 30.0
        assert s.p50_ms == 20.0

    def test_unknown_name_none(self):
        assert PerformanceMarker().summary_for("nope") is None

    def test_measure_block(self):
        p = PerformanceMarker()
        with p.measure("work"):
            time.sleep(0.01)
        s = p.summary_for("work")
        assert s.count == 1 and s.avg_ms >= 5.0

    def test_measure_records_on_error(self):
        p = PerformanceMarker()
        try:
            with p.measure("boom"):
                raise RuntimeError("x")
        except RuntimeError:
            pass
        assert p.summary_for("boom").count == 1

    def test_slowest_order(self):
        p = PerformanceMarker()
        p.mark("a", 5.0)
        p.mark("b", 50.0)
        p.mark("c", 20.0)
        assert [c.name for c in p.slowest(2)] == ["b", "c"]

    def test_report_shape(self):
        p = PerformanceMarker()
        p.mark("a", 5.0)
        r = p.report().to_dict()
        assert r["markers"][0]["name"] == "a"
        assert r["slowest"][0]["name"] == "a"
        assert r["total_ms"] >= 0

    def test_clear(self):
        p = PerformanceMarker()
        p.mark("a", 1.0)
        p.clear()
        assert p.checkpoints == [] and p.summary_for("a") is None
