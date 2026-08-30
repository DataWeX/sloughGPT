"""Tests for domains.infrastructure.model_server — IdleManager."""

import time
import threading
from unittest.mock import MagicMock
from domains.infrastructure.model_server import IdleManager


class TestIdleManager:
    def test_init(self):
        im = IdleManager()
        assert im._idle_timeout_s == 300.0
        assert im._check_interval_s == 30.0

    def test_register(self):
        im = IdleManager()
        im.register("model1")
        assert "model1" in im._models

    def test_unregister(self):
        im = IdleManager()
        im.register("model1")
        im.unregister("model1")
        assert "model1" not in im._models

    def test_touch(self):
        im = IdleManager()
        im.register("model1")
        result = im.touch("model1")
        assert isinstance(result, bool)

    def test_init_custom_values(self):
        im = IdleManager(idle_timeout_s=60.0, check_interval_s=5.0)
        assert im._idle_timeout_s == 60.0
        assert im._check_interval_s == 5.0

    def test_touch_returns_false_when_not_reloaded(self):
        im = IdleManager()
        im.register("model1")
        result = im.touch("model1")
        assert result is False

    def test_touch_unregistered_returns_false(self):
        im = IdleManager()
        result = im.touch("nonexistent")
        assert result is False

    def test_unregister_nonexistent(self):
        im = IdleManager()
        im.unregister("nonexistent")

    def test_register_multiple_models(self):
        im = IdleManager()
        im.register("m1")
        im.register("m2")
        im.register("m3")
        assert "m1" in im._models
        assert "m2" in im._models
        assert "m3" in im._models

    def test_unregister_one_of_many(self):
        im = IdleManager()
        im.register("m1")
        im.register("m2")
        im.unregister("m1")
        assert "m1" not in im._models
        assert "m2" in im._models

    def test_touch_updates_last_touch(self):
        im = IdleManager()
        im.register("model1")
        old_time = im._models["model1"]["last_touch"]
        time.sleep(0.01)
        im.touch("model1")
        new_time = im._models["model1"]["last_touch"]
        assert new_time >= old_time

    def test_register_with_unload_fn(self):
        called = []
        im = IdleManager()
        im.register("model1", unload_fn=lambda: called.append("unload"))
        assert "model1" in im._models
        assert im._models["model1"]["unload_fn"] is not None

    def test_register_with_reload_fn(self):
        called = []
        im = IdleManager()
        im.register("model1", reload_fn=lambda: called.append("reload"))
        assert im._models["model1"]["reload_fn"] is not None

    def test_is_idle_unloaded_false_by_default(self):
        im = IdleManager()
        im.register("model1")
        assert im.is_idle_unloaded("model1") is False

    def test_is_idle_unloaded_nonexistent(self):
        im = IdleManager()
        assert im.is_idle_unloaded("nonexistent") is False

    def test_is_reloading_nonexistent(self):
        im = IdleManager()
        assert im.is_reloading("nonexistent") is False

    def test_is_reloading_false_by_default(self):
        im = IdleManager()
        im.register("model1")
        assert im.is_reloading("model1") is False

    def test_get_idle_info(self):
        im = IdleManager(idle_timeout_s=300.0)
        im.register("model1")
        info = im.get_idle_info("model1")
        assert info is not None
        assert "last_request_age_s" in info
        assert "idle_timeout_s" in info
        assert "unloaded" in info
        assert "remaining_s" in info
        assert info["idle_timeout_s"] == 300.0
        assert info["unloaded"] is False

    def test_get_idle_info_nonexistent(self):
        im = IdleManager()
        info = im.get_idle_info("nonexistent")
        assert info is None

    def test_get_idle_info_remaining(self):
        im = IdleManager(idle_timeout_s=100.0)
        im.register("model1")
        info = im.get_idle_info("model1")
        assert info["remaining_s"] <= 100.0
        assert info["remaining_s"] >= 0.0

    def test_shutdown(self):
        im = IdleManager()
        im.register("model1")
        im.shutdown()
        assert im._running is False

    def test_shutdown_without_register(self):
        im = IdleManager()
        im.shutdown()

    def test_reset(self):
        im = IdleManager()
        im.register("m1")
        im.register("m2")
        im.reset()
        assert len(im._models) == 0
        assert im._running is False

    def test_reset_clears_all_models(self):
        im = IdleManager()
        for i in range(10):
            im.register(f"m{i}")
        im.reset()
        assert len(im._models) == 0

    def test_touch_async_returns_ok(self):
        im = IdleManager()
        im.register("model1")
        result = im.touch_async("model1")
        assert result == "ok"

    def test_touch_async_unregistered(self):
        im = IdleManager()
        result = im.touch_async("nonexistent")
        assert result == "ok"

    def test_register_overwrites_existing(self):
        im = IdleManager()
        im.register("model1")
        im.register("model1")
        assert "model1" in im._models

    def test_multiple_touches(self):
        im = IdleManager()
        im.register("model1")
        for _ in range(10):
            result = im.touch("model1")
            assert result is False

    def test_unregister_then_touch(self):
        im = IdleManager()
        im.register("model1")
        im.unregister("model1")
        result = im.touch("model1")
        assert result is False

    def test_shutdown_idempotent(self):
        im = IdleManager()
        im.register("model1")
        im.shutdown()
        im.shutdown()
        assert im._running is False

    def test_background_thread_started(self):
        im = IdleManager()
        im.register("model1")
        assert im._thread is not None
        assert im._thread.is_alive()

    def test_background_thread_not_started_without_register(self):
        im = IdleManager()
        assert im._thread is None

    def test_concurrent_touch(self):
        im = IdleManager()
        im.register("model1")
        results = []

        def toucher():
            for _ in range(20):
                im.touch("model1")

        threads = [threading.Thread(target=toucher) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert im._models["model1"]["last_touch"] > 0

    def test_concurrent_register_unregister(self):
        im = IdleManager()

        def worker(start):
            for i in range(start, start + 10):
                im.register(f"m{i}")
                im.unregister(f"m{i}")

        threads = [threading.Thread(target=worker, args=(i * 10,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def test_model_entry_structure(self):
        im = IdleManager()
        im.register("model1")
        entry = im._models["model1"]
        assert "last_touch" in entry
        assert "unload_fn" in entry
        assert "reload_fn" in entry
        assert "unloaded_at" in entry
        assert entry["unload_fn"] is None
        assert entry["reload_fn"] is None
        assert entry["unloaded_at"] is None

    def test_custom_idle_timeout(self):
        im = IdleManager(idle_timeout_s=10.0)
        im.register("model1")
        info = im.get_idle_info("model1")
        assert info["idle_timeout_s"] == 10.0

    def test_custom_check_interval(self):
        im = IdleManager(check_interval_s=1.0)
        assert im._check_interval_s == 1.0

    def test_touch_after_unregister_re_register(self):
        im = IdleManager()
        im.register("model1")
        im.unregister("model1")
        im.register("model1")
        result = im.touch("model1")
        assert result is False

    def test_many_models(self):
        im = IdleManager()
        for i in range(50):
            im.register(f"model_{i}")
        assert len(im._models) == 50
        for i in range(50):
            im.touch(f"model_{i}")
        assert len(im._models) == 50

    def test_shutdown_joins_thread(self):
        im = IdleManager(check_interval_s=0.1)
        im.register("model1")
        thread = im._thread
        im.shutdown()
        thread.join(timeout=2.0)
        assert not thread.is_alive()

    def test_reset_after_shutdown(self):
        im = IdleManager()
        im.register("model1")
        im.shutdown()
        im.reset()
        assert len(im._models) == 0

    def test_unregister_nonexistent_model(self):
        im = IdleManager()
        im.register("model1")
        im.unregister("model2")
        assert "model1" in im._models

    def test_get_idle_info_age_increases(self):
        im = IdleManager(idle_timeout_s=100.0)
        im.register("model1")
        info1 = im.get_idle_info("model1")
        time.sleep(0.01)
        info2 = im.get_idle_info("model1")
        assert info2["last_request_age_s"] >= info1["last_request_age_s"]

    def test_touch_resets_age(self):
        im = IdleManager(idle_timeout_s=100.0)
        im.register("model1")
        time.sleep(0.02)
        im.touch("model1")
        info = im.get_idle_info("model1")
        assert info["last_request_age_s"] < 0.05

    def test_multiple_sessions_different_models(self):
        im = IdleManager()
        im.register("fast_model")
        im.register("slow_model")
        im.touch("fast_model")
        im.touch("slow_model")
        info_fast = im.get_idle_info("fast_model")
        info_slow = im.get_idle_info("slow_model")
        assert info_fast is not None
        assert info_slow is not None

    def test_lifecycle_register_touch_unregister(self):
        im = IdleManager()
        im.register("model1")
        assert "model1" in im._models
        im.touch("model1")
        assert "model1" in im._models
        im.unregister("model1")
        assert "model1" not in im._models

    def test_default_idle_timeout(self):
        im = IdleManager()
        assert im._idle_timeout_s == 300.0

    def test_default_check_interval(self):
        im = IdleManager()
        assert im._check_interval_s == 30.0

    def test_touch_async_returns_ok_for_registered(self):
        im = IdleManager()
        im.register("m1")
        result = im.touch_async("m1")
        assert result == "ok"

    def test_is_idle_unloaded_false_after_touch(self):
        im = IdleManager()
        im.register("m1")
        im.touch("m1")
        assert im.is_idle_unloaded("m1") is False

    def test_get_idle_info_after_multiple_touches(self):
        im = IdleManager(idle_timeout_s=100.0)
        im.register("m1")
        for _ in range(5):
            im.touch("m1")
        info = im.get_idle_info("m1")
        assert info["remaining_s"] <= 100.0
        assert info["remaining_s"] >= 0.0

    def test_unregister_clears_from_models_dict(self):
        im = IdleManager()
        im.register("m1")
        im.register("m2")
        im.unregister("m1")
        assert "m1" not in im._models
        assert "m2" in im._models

    def test_register_preserves_existing_entry(self):
        im = IdleManager()
        im.register("m1", unload_fn=lambda: None)
        old_touch = im._models["m1"]["last_touch"]
        time.sleep(0.01)
        im.register("m1")
        new_touch = im._models["m1"]["last_touch"]
        assert new_touch >= old_touch

    def test_shutdown_multiple_times(self):
        im = IdleManager(check_interval_s=0.1)
        im.register("m1")
        im.shutdown()
        im.shutdown()
        assert im._running is False

    def test_get_idle_info_remaining_decreases(self):
        im = IdleManager(idle_timeout_s=1.0)
        im.register("m1")
        info1 = im.get_idle_info("m1")
        time.sleep(0.05)
        info2 = im.get_idle_info("m1")
        assert info2["remaining_s"] <= info1["remaining_s"]

    def test_model_entry_has_all_keys(self):
        im = IdleManager()
        im.register("m1")
        entry = im._models["m1"]
        expected_keys = {"last_touch", "unload_fn", "reload_fn", "unloaded_at"}
        assert set(entry.keys()) == expected_keys

    def test_touch_updates_last_touch_monotonic(self):
        im = IdleManager()
        im.register("m1")
        times = []
        for _ in range(5):
            im.touch("m1")
            times.append(im._models["m1"]["last_touch"])
        assert times == sorted(times)

    def test_register_with_both_callbacks(self):
        unload_called = []
        reload_called = []
        im = IdleManager()
        im.register(
            "m1",
            unload_fn=lambda: unload_called.append(True),
            reload_fn=lambda: reload_called.append(True),
        )
        entry = im._models["m1"]
        assert entry["unload_fn"] is not None
        assert entry["reload_fn"] is not None

    def test_many_models_thread_safety(self):
        im = IdleManager()
        errors = []

        def worker(start):
            try:
                for i in range(start, start + 10):
                    im.register(f"m{i}")
                    im.touch(f"m{i}")
                    im.unregister(f"m{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i * 10,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_reset_idempotent(self):
        im = IdleManager()
        im.register("m1")
        im.reset()
        im.reset()
        assert len(im._models) == 0
        assert im._running is False

    def test_touch_async_multiple_calls(self):
        im = IdleManager()
        im.register("m1")
        for _ in range(5):
            result = im.touch_async("m1")
            assert result == "ok"

    def test_get_idle_info_unloaded_state(self):
        im = IdleManager(idle_timeout_s=0.01)
        unload_called = []
        im.register("m1", unload_fn=lambda: unload_called.append(True))
        time.sleep(0.02)
        # Trigger the idle check by simulating what _check_loop does
        with im._lock:
            now = time.time()
            for model_id, entry in im._models.items():
                if entry["unloaded_at"] is not None:
                    continue
                age = now - entry["last_touch"]
                if age >= im._idle_timeout_s:
                    unload_fn = entry.get("unload_fn")
                    if unload_fn:
                        try:
                            unload_fn()
                            entry["unloaded_at"] = now
                        except Exception:
                            pass
        info = im.get_idle_info("m1")
        assert info is not None
        assert info["unloaded"] is True
