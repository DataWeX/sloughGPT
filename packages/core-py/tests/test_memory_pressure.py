"""Tests for domains.infrastructure.memory_pressure — MemoryPressureMonitor."""

import gc
import time
import sys
from unittest.mock import MagicMock, patch
import pytest
from domains.infrastructure.memory_pressure import (
    MemoryPressureMonitor,
    PressureLevel,
    get_memory_pressure_monitor,
    reset_memory_pressure_monitor,
)

psutil = pytest.importorskip("psutil", reason="psutil not installed")


class TestPressureLevel:
    def test_enum_values(self):
        assert PressureLevel.NORMAL.value == "normal"
        assert PressureLevel.WARNING.value == "warning"
        assert PressureLevel.CRITICAL.value == "critical"
        assert PressureLevel.EMERGENCY.value == "emergency"


class TestMemoryPressureMonitor:
    def test_init_defaults(self):
        m = MemoryPressureMonitor()
        assert m._warning == 80.0
        assert m._critical == 90.0
        assert m._emergency == 95.0
        assert m._check_interval == 15.0

    def test_init_custom_thresholds(self):
        m = MemoryPressureMonitor(
            warning_threshold=70.0,
            critical_threshold=85.0,
            emergency_threshold=92.0,
        )
        assert m._warning == 70.0
        assert m._critical == 85.0
        assert m._emergency == 92.0

    def test_classify_normal(self):
        m = MemoryPressureMonitor()
        assert m._classify(50.0) == PressureLevel.NORMAL
        assert m._classify(79.9) == PressureLevel.NORMAL

    def test_classify_warning(self):
        m = MemoryPressureMonitor()
        assert m._classify(80.0) == PressureLevel.WARNING
        assert m._classify(89.9) == PressureLevel.WARNING

    def test_classify_critical(self):
        m = MemoryPressureMonitor()
        assert m._classify(90.0) == PressureLevel.CRITICAL
        assert m._classify(94.9) == PressureLevel.CRITICAL

    def test_classify_emergency(self):
        m = MemoryPressureMonitor()
        assert m._classify(95.0) == PressureLevel.EMERGENCY
        assert m._classify(100.0) == PressureLevel.EMERGENCY

    def test_check_respects_interval(self):
        m = MemoryPressureMonitor(check_interval_s=999)
        first = m.check()
        second = m.check()
        assert first == second

    def test_check_force_cleanup_bypasses_interval(self):
        m = MemoryPressureMonitor(check_interval_s=999)
        m.check()
        result = m.force_cleanup()
        assert isinstance(result, PressureLevel)

    def test_register_cleanup(self):
        m = MemoryPressureMonitor()
        cb = MagicMock()
        m.register_cleanup(cb)
        assert cb in m._cleanup_callbacks

    def test_cleanup_callbacks_called_on_critical(self):
        m = MemoryPressureMonitor(check_interval_s=0)
        cb = MagicMock()
        m.register_cleanup(cb)

        mock_mem = MagicMock()
        mock_mem.percent = 91.0
        mock_mem.available = 1024 * 1024 * 500

        with patch("psutil.virtual_memory", return_value=mock_mem):
            with patch.object(m, "_get_rss_mb", return_value=500.0):
                m.check()
        cb.assert_called()

    def test_allow_load_normal(self):
        m = MemoryPressureMonitor()
        mock_mem = MagicMock()
        mock_mem.percent = 50.0

        with patch("psutil.virtual_memory", return_value=mock_mem):
            assert m.allow_load() is True

    def test_allow_load_emergency_blocks(self):
        m = MemoryPressureMonitor()
        mock_mem = MagicMock()
        mock_mem.percent = 96.0

        with patch("psutil.virtual_memory", return_value=mock_mem):
            assert m.allow_load() is False
            assert m._loads_blocked == 1

    def test_allow_load_critical_allows(self):
        m = MemoryPressureMonitor()
        mock_mem = MagicMock()
        mock_mem.percent = 91.0

        with patch("psutil.virtual_memory", return_value=mock_mem):
            assert m.allow_load() is True

    def test_stats_returns_dict(self):
        m = MemoryPressureMonitor()
        stats = m.stats()
        assert "level" in stats
        assert "system_percent" in stats
        assert "available_mb" in stats
        assert "cleanup_count" in stats
        assert "loads_blocked" in stats
        assert "warning_threshold" in stats

    def test_force_gc_tracked(self):
        m = MemoryPressureMonitor(check_interval_s=0)
        m._force_gc()
        assert m._gc_forced == 1

    def test_level_transitions(self):
        m = MemoryPressureMonitor(check_interval_s=0)

        mock_mem_normal = MagicMock()
        mock_mem_normal.percent = 50.0
        mock_mem_normal.available = 8 * 1024 ** 3

        mock_mem_critical = MagicMock()
        mock_mem_critical.percent = 92.0
        mock_mem_critical.available = 500 * 1024 ** 2

        with patch("psutil.virtual_memory", return_value=mock_mem_normal):
            level = m.check()
            assert level == PressureLevel.NORMAL

        with patch("psutil.virtual_memory", return_value=mock_mem_critical):
            with patch.object(m, "_get_rss_mb", return_value=1000.0):
                level = m.check()
                assert level == PressureLevel.CRITICAL

        with patch("psutil.virtual_memory", return_value=mock_mem_normal):
            level = m.check()
            assert level == PressureLevel.NORMAL

    def test_singleton(self):
        reset_memory_pressure_monitor()
        m1 = get_memory_pressure_monitor()
        m2 = get_memory_pressure_monitor()
        assert m1 is m2
        reset_memory_pressure_monitor()

    def test_singleton_reset(self):
        reset_memory_pressure_monitor()
        m1 = get_memory_pressure_monitor()
        reset_memory_pressure_monitor()
        m2 = get_memory_pressure_monitor()
        assert m1 is not m2
        reset_memory_pressure_monitor()

    def test_malloc_trim_no_crash(self):
        m = MemoryPressureMonitor()
        m._malloc_trim()

    def test_clear_kv_caches_no_crash(self):
        m = MemoryPressureMonitor()
        m._clear_kv_caches()

    def test_release_all_idle_weights_no_crash(self):
        m = MemoryPressureMonitor()
        m._release_all_idle_weights()

    def test_cleanup_callbacks_error_handled(self):
        m = MemoryPressureMonitor(check_interval_s=0)
        bad_cb = MagicMock(side_effect=RuntimeError("boom"))
        good_cb = MagicMock()
        m.register_cleanup(bad_cb)
        m.register_cleanup(good_cb)

        mock_mem = MagicMock()
        mock_mem.percent = 92.0
        mock_mem.available = 500 * 1024 ** 2

        with patch("psutil.virtual_memory", return_value=mock_mem):
            with patch.object(m, "_get_rss_mb", return_value=500.0):
                m.check()
        good_cb.assert_called()

    def test_properties(self):
        m = MemoryPressureMonitor(
            warning_threshold=70.0,
            critical_threshold=85.0,
            emergency_threshold=92.0,
        )
        assert m.warning_threshold == 70.0
        assert m.critical_threshold == 85.0
        assert m.emergency_threshold == 92.0

    def test_cleanup_callbacks_called_on_emergency(self):
        m = MemoryPressureMonitor(check_interval_s=0)
        cb = MagicMock()
        m.register_cleanup(cb)

        mock_mem = MagicMock()
        mock_mem.percent = 96.0
        mock_mem.available = 100 * 1024 * 1024

        with patch("psutil.virtual_memory", return_value=mock_mem):
            with patch.object(m, "_get_rss_mb", return_value=500.0):
                m.check()
        cb.assert_called()
        assert m._cleanup_count >= 1

    def test_emergency_increments_cleanup_count(self):
        m = MemoryPressureMonitor(check_interval_s=0)
        assert m._cleanup_count == 0

        mock_mem = MagicMock()
        mock_mem.percent = 97.0
        mock_mem.available = 50 * 1024 * 1024

        with patch("psutil.virtual_memory", return_value=mock_mem):
            with patch.object(m, "_get_rss_mb", return_value=500.0):
                m.check()
        assert m._cleanup_count == 1

        # Reset the throttle so the second check triggers _on_emergency again
        m._last_emergency_logged = 0

        with patch("psutil.virtual_memory", return_value=mock_mem):
            with patch.object(m, "_get_rss_mb", return_value=500.0):
                m.check()
        assert m._cleanup_count == 2

    def test_cleanup_deduplication(self):
        m = MemoryPressureMonitor()
        cb = lambda: None  # noqa: E731
        m.register_cleanup(cb)
        m.register_cleanup(cb)
        m.register_cleanup(cb)
        assert len([c for c in m._cleanup_callbacks if c is cb]) == 1

    def test_stats_returns_dict_on_psutil_failure(self):
        m = MemoryPressureMonitor()
        with patch("psutil.virtual_memory", side_effect=ImportError("no psutil")):
            stats = m.stats()
            assert isinstance(stats, dict)
            assert stats["level"] == "normal"
            assert stats["system_percent"] == 0.0

    def test_configure_updates_warning(self):
        m = MemoryPressureMonitor()
        assert m._warning == 80.0
        m.configure(warning=70.0)
        assert m._warning == 70.0

    def test_configure_updates_critical(self):
        m = MemoryPressureMonitor()
        assert m._critical == 90.0
        m.configure(critical=85.0)
        assert m._critical == 85.0

    def test_configure_updates_emergency(self):
        m = MemoryPressureMonitor()
        assert m._emergency == 95.0
        m.configure(emergency=92.0)
        assert m._emergency == 92.0

    def test_configure_no_args_unchanged(self):
        m = MemoryPressureMonitor(warning_threshold=70.0, critical_threshold=85.0, emergency_threshold=92.0)
        m.configure()
        assert m._warning == 70.0
        assert m._critical == 85.0
        assert m._emergency == 92.0

    def test_configure_partial(self):
        m = MemoryPressureMonitor()
        m.configure(warning=75.0)
        assert m._warning == 75.0
        assert m._critical == 90.0  # unchanged
        assert m._emergency == 95.0  # unchanged

    def test_configure_affects_classify(self):
        m = MemoryPressureMonitor()
        assert m._classify(85.0) == PressureLevel.WARNING  # default: 80-90 is WARNING
        m.configure(warning=82.0)
        assert m._classify(85.0) == PressureLevel.NORMAL  # 85 < 82? No, 85 > 82
        # After configure(warning=82), 85.0 should be WARNING (82 <= 85 < 90)
        assert m._classify(85.0) == PressureLevel.WARNING
        m.configure(emergency=83.0)
        assert m._classify(85.0) == PressureLevel.EMERGENCY  # 85 >= 83
