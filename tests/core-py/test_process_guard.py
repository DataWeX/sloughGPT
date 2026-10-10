"""Tests for ProcessGuard — resolve_memory_limit_mb, health, callbacks."""

from unittest.mock import MagicMock, patch

import pytest

from domain.infrastructure._internal.process_guard import (
    ProcessGuard,
    create_model_guard,
    create_slo_guard,
    resolve_memory_limit_mb,
)


class TestResolveMemoryLimitMb:
    def test_configured_overrides_auto(self, tmp_path):
        f = tmp_path / "model.slnc"
        f.write_bytes(b"\x00" * 1024)
        result = resolve_memory_limit_mb(str(f), configured=2048.0)
        assert result == 2048.0

    def test_configured_zero_falls_back(self, tmp_path):
        f = tmp_path / "model.slnc"
        f.write_bytes(b"\x00" * (100 * 1024 * 1024))
        result = resolve_memory_limit_mb(str(f), configured=0.0)
        assert result == max(8192.0, 100.0 * 8.0)

    def test_no_slnc_no_config_returns_none(self):
        assert resolve_memory_limit_mb(None, None) is None

    def test_no_slnc_with_config_returns_config(self):
        assert resolve_memory_limit_mb(None, 4096.0) == 4096.0

    def test_slnc_file_size(self, tmp_path):
        f = tmp_path / "model.slnc"
        f.write_bytes(b"\x00" * (50 * 1024 * 1024))
        result = resolve_memory_limit_mb(str(f))
        assert result == max(8192.0, 50.0 * 8.0)

    def test_slnc_tiny_file_uses_floor(self, tmp_path):
        f = tmp_path / "tiny.slnc"
        f.write_bytes(b"\x00" * 1024)
        result = resolve_memory_limit_mb(str(f))
        assert result == 8192.0

    def test_slnc_nonexistent_returns_none(self):
        result = resolve_memory_limit_mb("/nonexistent/path.slnc")
        assert result is None


class TestProcessGuardInit:
    def test_defaults(self):
        guard = ProcessGuard()
        assert guard.worker_id == "guard"
        assert guard.generate_timeout == 120.0
        assert guard.stall_timeout == 120.0
        assert guard.max_restarts == 3
        assert guard.restart_delay == 1.0
        assert guard.memory_limit_mb == 4096.0

    def test_custom_params(self):
        guard = ProcessGuard(
            worker_id="my-worker",
            generate_timeout=60.0,
            max_restarts=5,
            restart_delay=2.0,
            memory_limit_mb=8192.0,
        )
        assert guard.worker_id == "my-worker"
        assert guard.generate_timeout == 60.0
        assert guard.max_restarts == 5
        assert guard.restart_delay == 2.0
        assert guard.memory_limit_mb == 8192.0

    def test_slnc_params_stored(self):
        guard = ProcessGuard(slnc_path="/path/to/model.slnc", model_id="gpt2")
        assert guard._config.slnc_path == "/path/to/model.slnc"
        assert guard._config.model_id == "gpt2"

    def test_hf_params_stored(self):
        guard = ProcessGuard(
            model_cls_path="transformers.AutoModel",
            model_kwargs={"pretrained_model_name_or_path": "gpt2"},
        )
        assert guard._config.hf_model_cls_path == "transformers.AutoModel"
        assert guard._config.hf_model_kwargs == {"pretrained_model_name_or_path": "gpt2"}


class TestProcessGuardAlive:
    def test_alive_false_when_no_worker(self):
        guard = ProcessGuard()
        assert guard.alive is False

    def test_alive_delegates_to_worker(self):
        guard = ProcessGuard()
        mock_worker = MagicMock()
        mock_worker.alive = True
        guard._worker = mock_worker
        assert guard.alive is True

    def test_alive_false_when_worker_dead(self):
        guard = ProcessGuard()
        mock_worker = MagicMock()
        mock_worker.alive = False
        guard._worker = mock_worker
        assert guard.alive is False


class TestProcessGuardHealth:
    def test_health_dict_structure(self):
        guard = ProcessGuard(worker_id="test-guard")
        health = guard.health()
        assert "alive" in health
        assert "worker_id" in health
        assert "mode" in health
        assert "model_id" in health
        assert "requests_served" in health
        assert "restart_count" in health
        assert "max_restarts" in health
        assert "exhausted" in health
        assert "memory_limit_mb" in health

    def test_health_worker_id(self):
        guard = ProcessGuard(worker_id="my-guard")
        assert guard.health()["worker_id"] == "my-guard"

    def test_health_exhausted_false(self):
        guard = ProcessGuard(max_restarts=3)
        assert guard.health()["exhausted"] is False

    def test_health_exhausted_true(self):
        guard = ProcessGuard(max_restarts=0)
        guard._restart_count = 0
        assert guard.health()["exhausted"] is True

    def test_health_requests_served(self):
        guard = ProcessGuard()
        guard._requests_served = 42
        assert guard.health()["requests_served"] == 42


class TestProcessGuardCallbacks:
    def test_on_crash_registers(self):
        guard = ProcessGuard()
        cb = MagicMock()
        guard.on_crash(cb)
        assert cb in guard._crash_callbacks

    def test_on_restart_registers(self):
        guard = ProcessGuard()
        cb = MagicMock()
        guard.on_restart(cb)
        assert cb in guard._restart_callbacks

    def test_multiple_crash_callbacks(self):
        guard = ProcessGuard()
        cb1, cb2 = MagicMock(), MagicMock()
        guard.on_crash(cb1)
        guard.on_crash(cb2)
        assert len(guard._crash_callbacks) == 2


class TestProcessGuardGenerateNotAlive:
    def test_generate_raises_when_not_alive(self):
        guard = ProcessGuard()
        with pytest.raises(RuntimeError, match="not alive"):
            guard.generate("hello")

    def test_generate_stream_raises_when_not_alive(self):
        guard = ProcessGuard()
        with pytest.raises(RuntimeError, match="not alive"):
            list(guard.generate_stream("hello"))


class TestProcessGuardRecoverFromStall:
    def test_stall_raises_when_budget_exhausted(self):
        guard = ProcessGuard(max_restarts=0)
        with pytest.raises(RuntimeError, match="restart budget exhausted"):
            guard._recover_from_stall()

    def test_stall_increments_restart_count(self):
        guard = ProcessGuard(max_restarts=3, restart_delay=0.0)
        guard._launch_worker = MagicMock()
        guard._recover_from_stall()
        assert guard._restart_count == 1


class TestCreateModelGuardFactory:
    def test_factory_sets_params(self):
        with patch.object(ProcessGuard, "start"):
            guard = create_model_guard("gpt2", device="cpu", max_restarts=5)
            assert guard._config.hf_model_kwargs["model_id"] == "gpt2"
            assert guard._config.hf_model_kwargs["device"] == "cpu"
            assert guard.max_restarts == 5
            assert guard.worker_id == "guard-gpt2"

    def test_factory_custom_worker_id(self):
        with patch.object(ProcessGuard, "start"):
            guard = create_model_guard("gpt2", worker_id="custom")
            assert guard.worker_id == "custom"


class TestCreateSloGuardFactory:
    def test_factory_sets_params(self):
        with patch.object(ProcessGuard, "start"):
            guard = create_slo_guard("/path/model.slnc", model_id="my-model")
            assert guard._config.slnc_path == "/path/model.slnc"
            assert guard._config.model_id == "my-model"
            assert guard.worker_id == "slo-guard-my-model"

    def test_factory_quantize_params(self):
        with patch.object(ProcessGuard, "start"):
            guard = create_slo_guard(
                "/m.slnc", quantize=True, quant_bits=4, quant_mode="asymmetric"
            )
            assert guard._config.quantize is True
            assert guard._config.quant_bits == 4
            assert guard._config.quant_mode == "asymmetric"


# ── Lifecycle tests (mocked worker) ────────────────────────────────────


class TestProcessGuardStartStop:
    def test_start_launches_worker(self):
        guard = ProcessGuard(max_restarts=1, restart_delay=0.0)
        mock_worker = MagicMock()
        mock_worker.alive = True
        guard._launch_worker = MagicMock()
        guard._worker = mock_worker
        guard.start()
        assert guard._monitor_thread is not None
        guard._stop_monitor.set()
        guard._worker.stop()

    def test_stop_clears_worker(self):
        guard = ProcessGuard()
        mock_worker = MagicMock()
        mock_worker.alive = True
        guard._launch_worker = MagicMock()
        guard._worker = mock_worker
        guard.start()
        guard.stop()
        assert guard._worker is None
        assert guard._monitor_thread is None

    def test_stop_without_worker(self):
        guard = ProcessGuard()
        guard.stop()
        assert guard._worker is None


class TestProcessGuardGenerate:
    def test_generate_returns_result(self):
        guard = ProcessGuard()
        mock_worker = MagicMock()
        mock_worker.alive = True
        mock_worker.generate.return_value = {"text": "hello", "tokens": 1}
        guard._worker = mock_worker
        result = guard.generate("test prompt")
        assert result["text"] == "hello"
        assert guard._requests_served == 1

    def test_generate_increments_requests(self):
        guard = ProcessGuard()
        mock_worker = MagicMock()
        mock_worker.alive = True
        mock_worker.generate.return_value = {}
        guard._worker = mock_worker
        guard.generate("a")
        guard.generate("b")
        assert guard._requests_served == 2

    def test_generate_stream_returns_tokens(self):
        guard = ProcessGuard()
        mock_worker = MagicMock()
        mock_worker.alive = True

        def fake_stream(prompt, **kwargs):
            yield "token1"
            yield "token2"
            return {"tokens_generated": 2}

        mock_worker.generate_stream = fake_stream
        guard._worker = mock_worker
        tokens = list(guard.generate_stream("test"))
        assert tokens == ["token1", "token2"]

    def test_generate_recover_from_stall(self):
        from domain.infrastructure._internal.model_worker import WorkerStreamStalledError

        guard = ProcessGuard(max_restarts=3, restart_delay=0.0)
        mock_worker = MagicMock()
        mock_worker.alive = True
        mock_worker.generate.side_effect = WorkerStreamStalledError("stall")
        guard._worker = mock_worker
        guard._launch_worker = MagicMock()
        with pytest.raises(WorkerStreamStalledError):
            guard.generate("test")
        assert guard._restart_count == 1


class TestProcessGuardRestartWorker:
    def test_restart_fires_crash_callbacks(self):
        guard = ProcessGuard(max_restarts=3, restart_delay=0.0)
        cb = MagicMock()
        guard.on_crash(cb)
        guard._launch_worker = MagicMock()
        guard._restart_worker_locked("crash", fire_callbacks=True)
        cb.assert_called_once_with(guard.worker_id)

    def test_restart_fires_restart_callbacks(self):
        guard = ProcessGuard(max_restarts=3, restart_delay=0.0)
        cb = MagicMock()
        guard.on_restart(cb)
        guard._launch_worker = MagicMock()
        guard._restart_worker_locked("restart", fire_callbacks=True)
        cb.assert_called_once_with(guard.worker_id)

    def test_restart_increments_count(self):
        guard = ProcessGuard(max_restarts=3, restart_delay=0.0)
        guard._launch_worker = MagicMock()
        guard._restart_worker_locked("test")
        assert guard._restart_count == 1

    def test_restart_callback_exception_does_not_propagate(self):
        guard = ProcessGuard(max_restarts=3, restart_delay=0.0)
        guard.on_crash(lambda wid: 1 / 0)  # type: ignore
        guard._launch_worker = MagicMock()
        guard._restart_worker_locked("crash", fire_callbacks=True)
        assert guard._restart_count == 1


class TestProcessGuardLoadAdapter:
    def test_load_adapter_not_alive(self):
        guard = ProcessGuard()
        with pytest.raises(RuntimeError, match="not alive"):
            guard.load_adapter("/path/adapter")

    def test_unload_adapter_not_alive(self):
        guard = ProcessGuard()
        with pytest.raises(RuntimeError, match="not alive"):
            guard.unload_adapter()


class TestProcessGuardMemoryMb:
    def test_memory_mb_no_worker(self):
        guard = ProcessGuard()
        assert guard._memory_mb() is None

    def test_memory_mb_with_worker(self):
        guard = ProcessGuard()
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 1024 * 1024 * 500
        mock_worker = MagicMock()
        mock_worker._process.pid = 12345
        guard._worker = mock_worker
        with patch.dict(
            "sys.modules", {"psutil": MagicMock(Process=MagicMock(return_value=mock_process))}
        ):
            result = guard._memory_mb()
        assert result == 500.0

    def test_memory_mb_psutil_not_installed(self):
        guard = ProcessGuard()
        mock_worker = MagicMock()
        mock_worker._process.pid = 12345
        guard._worker = mock_worker
        with patch.dict("sys.modules", {"psutil": None}):
            result = guard._memory_mb()
        assert result is None

    def test_memory_mb_no_process(self):
        guard = ProcessGuard()
        mock_worker = MagicMock()
        mock_worker._process = None
        guard._worker = mock_worker
        assert guard._memory_mb() is None


class TestProcessGuardProgressHeartbeat:
    """Progress heartbeat (card d3a4a079): make a livelocked worker visible.

    `alive` only proves the thread object exists. A worker that spins forever
    without raising never reaches _recover_from_stall's exception path, so
    health() called it healthy for the whole 90-minute 2026-10-05 wedge.
    """

    @staticmethod
    def _fake_live_worker(guard: ProcessGuard) -> None:
        """Stand in a live worker without spawning one (alive reads _worker)."""
        guard._worker = MagicMock(alive=True)

    @staticmethod
    def _open_stream(guard: ProcessGuard) -> None:
        """Put the guard into the in-flight-stream state the predicate needs."""
        guard._active_streams = 1

    def test_health_exposes_progress_fields(self):
        guard = ProcessGuard(worker_id="pg")
        health = guard.health()
        assert "last_progress_age_s" in health
        assert "progress_stall_timeout_s" in health
        assert "stalled" in health

    def test_fresh_guard_is_not_stalled(self):
        guard = ProcessGuard()
        self._fake_live_worker(guard)
        assert guard.health()["stalled"] is False

    def test_livelock_reports_stalled_while_alive_stays_true(self):
        """The whole point: alive=True and stalled=True at the same moment."""
        guard = ProcessGuard(progress_stall_timeout=0.05)
        self._fake_live_worker(guard)
        self._open_stream(guard)  # work must be in flight for silence to mean
        guard._last_progress -= 10.0  # silence far past the threshold
        health = guard.health()
        assert health["alive"] is True
        assert health["stalled"] is True
        # >= not >: age is rounded to 3dp, so a fast test lands on exactly 10.0.
        assert health["last_progress_age_s"] >= 10.0

    def test_alive_alone_no_longer_implies_healthy(self):
        """Regression pin for the outage: the two signals stay independent."""
        guard = ProcessGuard(progress_stall_timeout=0.05)
        self._fake_live_worker(guard)
        self._open_stream(guard)
        guard._last_progress -= 10.0
        assert guard.health()["alive"] is True
        assert guard.health()["stalled"] is True

    def test_idle_worker_is_never_stalled(self):
        """Regression pin for a defect in the first cut of this card.

        Without the active-stream gate, a healthy worker with ZERO traffic
        accumulated progress_age and reported stalled once the threshold
        elapsed — a false positive proven against a guard that had served no
        requests at all. Silence only means livelock while work is in flight.
        """
        guard = ProcessGuard(progress_stall_timeout=0.05)
        self._fake_live_worker(guard)
        guard._last_progress -= 10_000.0  # hours of pure idleness, no requests
        assert guard.health()["alive"] is True
        assert guard.health()["stalled"] is False

    def test_mark_progress_clears_the_stall(self):
        guard = ProcessGuard(progress_stall_timeout=0.05)
        self._fake_live_worker(guard)
        self._open_stream(guard)
        guard._last_progress -= 10.0
        assert guard.stalled is True
        guard._mark_progress()
        assert guard.stalled is False
        assert guard.health()["last_progress_age_s"] < 1.0

    def test_dead_worker_is_not_reported_stalled(self):
        """A dead worker is dead, not stalled — no live worker, no claim."""
        guard = ProcessGuard(progress_stall_timeout=0.05)
        guard._worker = None
        guard._last_progress -= 10.0
        assert guard.health()["alive"] is False
        assert guard.health()["stalled"] is False

    def test_threshold_is_configurable(self):
        guard = ProcessGuard(progress_stall_timeout=7.5)
        assert guard.health()["progress_stall_timeout_s"] == 7.5

    def test_default_threshold_outlives_a_single_generate(self):
        """Must exceed DEFAULT_GENERATE_TIMEOUT, or a merely-slow request gets
        misreported as a livelock."""
        from domain.infrastructure._internal.constants import (
            DEFAULT_GENERATE_TIMEOUT,
            DEFAULT_PROGRESS_STALL_TIMEOUT,
        )

        assert DEFAULT_PROGRESS_STALL_TIMEOUT > DEFAULT_GENERATE_TIMEOUT

    def test_generate_completion_marks_progress(self):
        """A completed request refreshes the heartbeat even with no stream."""
        guard = ProcessGuard(progress_stall_timeout=0.05)
        self._fake_live_worker(guard)
        guard._worker.generate = MagicMock(return_value={"text": "ok"})
        guard._last_progress -= 10.0
        assert guard.health()["last_progress_age_s"] >= 10.0
        assert guard.generate("hello") == {"text": "ok"}
        assert guard.health()["last_progress_age_s"] < 1.0
        assert guard.health()["requests_served"] == 1

    def test_streamed_tokens_mark_progress(self):
        guard = ProcessGuard(progress_stall_timeout=0.05)
        self._fake_live_worker(guard)

        def _three_tokens():
            yield "a"
            yield "b"
            yield "c"

        guard._worker.generate_stream = MagicMock(side_effect=lambda *a, **k: _three_tokens())
        stream = guard.generate_stream("hello")
        # Consuming one token starts the stream body: the gate is now armed.
        assert next(stream) == "a"
        assert guard._active_streams == 1
        guard._last_progress -= 10.0  # in flight AND silent -> livelock
        assert guard.stalled is True
        # The next token proves progress and clears it.
        assert next(stream) == "b"
        assert guard.stalled is False
        # Draining releases the counter, so silence afterwards is just idle.
        assert next(stream) == "c"
        assert list(stream) == []
        assert guard._active_streams == 0
        assert guard.stalled is False
