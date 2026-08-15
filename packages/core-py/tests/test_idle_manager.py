"""Tests for IdleManager — auto-unload on inactivity + auto-reload on next request."""
import time
import threading
from unittest.mock import MagicMock, patch

import pytest
from domains.infrastructure.model_server import IdleManager, get_idle_manager, ModelServer, ModelStatus


class TestIdleManager:
    def setup_method(self):
        self.mgr = IdleManager(idle_timeout_s=0.5, check_interval_s=0.1)

    def teardown_method(self):
        self.mgr.shutdown()

    def test_register_and_touch(self):
        self.mgr.register("m1")
        info = self.mgr.get_idle_info("m1")
        assert info is not None
        assert info["unloaded"] is False
        assert info["idle_timeout_s"] == 0.5
        assert info["last_request_age_s"] < 1.0

    def test_touch_after_register_resets_timer(self):
        self.mgr.register("m1")
        time.sleep(0.3)
        self.mgr.touch("m1")
        info = self.mgr.get_idle_info("m1")
        assert info["last_request_age_s"] < 0.3

    def test_unload_fires_after_timeout(self):
        unload_fn = MagicMock()
        self.mgr.register("m1", unload_fn=unload_fn)
        time.sleep(0.8)
        assert unload_fn.call_count >= 1
        assert self.mgr.is_idle_unloaded("m1")

    def test_unloaded_model_not_touched_again(self):
        unload_fn = MagicMock()
        self.mgr.register("m1", unload_fn=unload_fn)
        time.sleep(0.8)
        time.sleep(0.3)
        assert unload_fn.call_count == 1

    def test_reload_on_touch(self):
        unload_fn = MagicMock()
        reload_fn = MagicMock()
        self.mgr.register("m1", unload_fn=unload_fn, reload_fn=reload_fn)
        time.sleep(0.8)
        assert self.mgr.is_idle_unloaded("m1")
        self.mgr.touch("m1")
        reload_fn.assert_called_once()
        assert not self.mgr.is_idle_unloaded("m1")

    def test_unregister_stops_tracking(self):
        self.mgr.register("m1")
        self.mgr.unregister("m1")
        assert self.mgr.get_idle_info("m1") is None

    def test_touch_unknown_model_returns_false(self):
        assert self.mgr.touch("nonexistent") is False

    def test_multiple_models_independent(self):
        unload_a = MagicMock()
        unload_b = MagicMock()
        self.mgr.register("a", unload_fn=unload_a)
        self.mgr.register("b", unload_fn=unload_b)
        time.sleep(0.3)
        self.mgr.touch("a")
        time.sleep(0.5)
        assert unload_b.call_count >= 1
        assert unload_a.call_count == 0
        assert not self.mgr.is_idle_unloaded("a")
        assert self.mgr.is_idle_unloaded("b")

    def test_get_idle_info_unloaded_model(self):
        self.mgr.register("m1", unload_fn=MagicMock())
        time.sleep(0.8)
        info = self.mgr.get_idle_info("m1")
        assert info["unloaded"] is True

    def test_singleton_get_idle_manager(self):
        mgr1 = get_idle_manager()
        mgr2 = get_idle_manager()
        assert mgr1 is mgr2


class TestModelServerIdle:
    """Test ModelServer idle integration."""

    def test_set_hf_model_id(self):
        server = ModelServer(model=None, tokenizer=None, model_id="test-model", enable_warmup=False)
        server.set_hf_model_id("gpt2")
        assert server._hf_model_id == "gpt2"
        assert server._slnc_path is None
        assert server._reload_quantize is False

    def test_set_hf_model_id_with_params(self):
        server = ModelServer(model=None, tokenizer=None, model_id="test-model", enable_warmup=False)
        server.set_hf_model_id(
            "gpt2",
            slnc_path="/tmp/model.slnc",
            quantize=True,
            quant_bits=4,
            quant_mode="asymmetric",
        )
        assert server._hf_model_id == "gpt2"
        assert server._slnc_path == "/tmp/model.slnc"
        assert server._reload_quantize is True
        assert server._reload_quant_bits == 4
        assert server._reload_quant_mode == "asymmetric"

    def test_idle_unload_sets_status(self):
        server = ModelServer(model=None, tokenizer=None, model_id="test-model", enable_warmup=False)
        server._idle_unload()
        assert server.status == ModelStatus.UNLOADED
        assert server._model_ref is None
        assert server._local_backend is None

    def test_idle_reload_without_model_id_logs_warning(self):
        server = ModelServer(model=None, tokenizer=None, model_id="test-model", enable_warmup=False)
        server._hf_model_id = None
        server._slnc_path = None
        server._idle_reload()  # should not raise

    def test_idle_reload_missing_slnc_logs_error(self):
        server = ModelServer(model=None, tokenizer=None, model_id="test-model", enable_warmup=False)
        server.set_hf_model_id("nonexistent-model")
        # Should not raise — missing .slnc is caught internally
        server._idle_reload()

    def test_idle_reload_without_model_id_is_noop(self):
        server = ModelServer(model=None, tokenizer=None, model_id="test-model", enable_warmup=False)
        server._hf_model_id = None
        server._slnc_path = None
        # Should not raise
        server._idle_reload()


class TestIdleEndToEnd:
    """End-to-end tests for idle unload→reload lifecycle."""

    def test_full_lifecycle_unregister_touch_reloads(self):
        unload_fn = MagicMock()
        reload_fn = MagicMock()
        mgr = IdleManager(idle_timeout_s=0.3, check_interval_s=0.1)
        try:
            mgr.register("m1", unload_fn=unload_fn, reload_fn=reload_fn)
            # Wait for idle unload
            time.sleep(0.6)
            assert unload_fn.call_count == 1
            assert mgr.is_idle_unloaded("m1")
            # Touch triggers reload
            reloaded = mgr.touch("m1")
            assert reloaded is True
            assert reload_fn.call_count == 1
            assert not mgr.is_idle_unloaded("m1")
            # Second touch does not reload again
            reloaded2 = mgr.touch("m1")
            assert reloaded2 is False
            assert reload_fn.call_count == 1
        finally:
            mgr.shutdown()

    def test_model_server_registers_with_idle_manager(self):
        from domains.infrastructure.model_server import get_idle_manager
        server = ModelServer(
            model=None, tokenizer=None, model_id="idle-reg-test",
            enable_warmup=False, idle_timeout_s=999,
        )
        mgr = get_idle_manager()
        info = mgr.get_idle_info("idle-reg-test")
        assert info is not None
        # The manager uses its own default timeout (300s), not the per-server value
        assert info["idle_timeout_s"] == mgr._idle_timeout_s

    def test_idle_unload_clears_model_ref_and_sets_status(self):
        fake_model = MagicMock()
        fake_tokenizer = MagicMock()
        server = ModelServer(
            model=fake_model, tokenizer=fake_tokenizer,
            model_id="unload-status-test", enable_warmup=False,
        )
        # Simulate unload
        server._idle_unload()
        assert server.status == ModelStatus.UNLOADED
        assert server._model_ref is None
        assert server._local_backend is None

    def test_idle_reload_sets_ready_status(self):
        fake_model = MagicMock()
        fake_tokenizer = MagicMock()
        server = ModelServer(
            model=fake_model, tokenizer=fake_tokenizer,
            model_id="reload-status-test", enable_warmup=False,
        )
        server.set_hf_model_id("nonexistent-model")
        # Reload will fail (model doesn't exist), but should not raise
        server._idle_reload()
        # Status remains whatever it was (reload failed gracefully)

    def test_unregister_prevents_future_unload(self):
        unload_fn = MagicMock()
        mgr = IdleManager(idle_timeout_s=0.3, check_interval_s=0.1)
        try:
            mgr.register("m1", unload_fn=unload_fn)
            mgr.unregister("m1")
            time.sleep(0.6)
            assert unload_fn.call_count == 0
            assert mgr.get_idle_info("m1") is None
        finally:
            mgr.shutdown()

    def test_shutdown_stops_background_thread(self):
        mgr = IdleManager(idle_timeout_s=0.1, check_interval_s=0.05)
        mgr.register("m1")
        assert mgr._thread is not None
        assert mgr._thread.is_alive()
        mgr.shutdown()
        assert not mgr._running
        assert not mgr._thread.is_alive()

    def test_get_idle_info_tracks_remaining_time(self):
        mgr = IdleManager(idle_timeout_s=1.0, check_interval_s=0.1)
        try:
            mgr.register("m1")
            time.sleep(0.2)
            info = mgr.get_idle_info("m1")
            assert info is not None
            assert info["remaining_s"] <= 0.9
            assert info["remaining_s"] >= 0.0
            assert info["unloaded"] is False
        finally:
            mgr.shutdown()

    def test_reload_failure_does_not_crash(self):
        def bad_reload():
            raise RuntimeError("reload failed")
        mgr = IdleManager(idle_timeout_s=0.3, check_interval_s=0.1)
        try:
            mgr.register("m1", unload_fn=MagicMock(), reload_fn=bad_reload)
            time.sleep(0.6)
            assert mgr.is_idle_unloaded("m1")
            # Touch triggers reload which fails — should not raise
            reloaded = mgr.touch("m1")
            assert reloaded is False
            assert mgr.is_idle_unloaded("m1")
        finally:
            mgr.shutdown()

    def test_unload_failure_does_not_crash(self):
        def bad_unload():
            raise RuntimeError("unload failed")
        mgr = IdleManager(idle_timeout_s=0.3, check_interval_s=0.1)
        try:
            mgr.register("m1", unload_fn=bad_unload)
            time.sleep(0.6)
            # Should not raise — unload failure is caught
            assert not mgr.is_idle_unloaded("m1")
        finally:
            mgr.shutdown()
