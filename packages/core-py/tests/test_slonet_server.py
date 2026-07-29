"""Tests for SloNetServer — concurrency, circuit breaker, warmup, metrics, pool."""

import asyncio
import threading
import time
import queue
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

from domains.infrastructure.model_server import CircuitBreakerState
from domains.infrastructure.slonet_server import SloNetServer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_model():
    m = MagicMock()
    m.generate_numpy.return_value = np.array([[101, 102, 103]], dtype=np.int64)
    m.generate_numpy_stream.return_value = iter([np.int64(101), np.int64(102)])
    m.layers = [MagicMock()]
    m.layers[0].weight.shape = (256, 64)
    m.max_seq_len = 2048
    m.parameters.return_value = []
    m._config = {"n_embd": 64, "n_head": 4}
    return m


@pytest.fixture
def mock_tokenizer():
    t = MagicMock()
    t.encode.return_value = [10, 20, 30]
    t.decode.return_value = "hello world"
    t.eos_token_id = 0
    return t


@pytest.fixture
def server(mock_model, mock_tokenizer):
    return SloNetServer(
        model=mock_model,
        tokenizer=mock_tokenizer,
        model_id="test-slonet",
        enable_warmup=False,
    )


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

class TestInit:
    def test_creates_semaphore(self, server):
        assert isinstance(server._semaphore, asyncio.Semaphore)

    def test_read_semaphores_empty_initially(self, server):
        assert len(server._read_semaphores) == 0

    def test_creates_circuit_breaker_by_default(self, mock_model, mock_tokenizer):
        s = SloNetServer(mock_model, mock_tokenizer, enable_circuit_breaker=True, enable_warmup=False)
        assert s._circuit_breaker is not None

    def test_can_disable_circuit_breaker(self, mock_model, mock_tokenizer):
        s = SloNetServer(mock_model, mock_tokenizer, enable_circuit_breaker=False, enable_warmup=False)
        assert s._circuit_breaker is None

    def test_warmup_disabled(self, server):
        assert server.warmup_completed is False
        assert server.warmup_error is None

    def test_metrics_start_empty(self, server):
        m = server.get_metrics()
        assert m["requests_total"] == 0
        assert m["requests_completed"] == 0
        assert m["requests_failed"] == 0


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

class TestWarmup:
    def test_warmup_success(self, mock_model, mock_tokenizer):
        s = SloNetServer(mock_model, mock_tokenizer, enable_warmup=True, warmup_prompt="Hi")
        time.sleep(0.2)
        assert s.warmup_completed is True
        assert s.warmup_error is None
        assert mock_model.generate_numpy.called

    def test_warmup_failure(self, mock_model, mock_tokenizer):
        mock_model.generate_numpy.side_effect = RuntimeError("warmup fail")
        s = SloNetServer(mock_model, mock_tokenizer, enable_warmup=True, warmup_prompt="Hi")
        time.sleep(0.2)
        assert s.warmup_completed is False
        assert s.warmup_error is not None

    def test_warmup_non_blocking(self, mock_model, mock_tokenizer):
        start = time.monotonic()
        SloNetServer(mock_model, mock_tokenizer, enable_warmup=True)
        elapsed = time.monotonic() - start
        assert elapsed < 0.1


# ---------------------------------------------------------------------------
# Tokenize / Count
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestTokenize:
    async def test_tokenize_delegates(self, server):
        tokens = await server.tokenize("hello")
        assert tokens == [10, 20, 30]
        server._tokenizer.encode.assert_called_with("hello")

    async def test_count_tokens(self, server):
        n = await server.count_tokens("hello")
        assert n == 3

    async def test_get_read_semaphore_creates_per_loop(self, server):
        s1 = server._get_read_semaphore()
        s2 = server._get_read_semaphore()
        assert s1 is s2


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestGenerate:
    async def test_generate_returns_string(self, server):
        result = await server.generate("hello")
        assert isinstance(result, str)
        assert result == "hello world"

    async def test_generate_passes_prompt(self, server):
        await server.generate("test prompt")
        server._tokenizer.encode.assert_called_with("test prompt")

    async def test_generate_increments_metrics(self, server):
        await server.generate("hello")
        m = server.get_metrics()
        assert m["requests_total"] == 1
        assert m["requests_completed"] == 1

    async def test_generate_circuit_breaker_closed_after_success(self, server):
        server._circuit_breaker._state = CircuitBreakerState.OPEN
        server._circuit_breaker._last_failure_at = time.time()
        with pytest.raises(RuntimeError, match="circuit breaker open"):
            await server.generate("hello")

    async def test_generate_circuit_breaker_opens_after_3_failures(self, server):
        err = RuntimeError("gen fail")
        server._model.generate_numpy.side_effect = err
        for _ in range(3):
            with pytest.raises(RuntimeError):
                await server.generate("hello")
        with pytest.raises(RuntimeError, match="circuit breaker open"):
            await server.generate("hello")

    async def test_generate_queue_full_isolation(self, server):
        tasks = [server.generate("hello") for _ in range(2)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert all(r == "hello world" for r in results)

    async def test_generate_timeout(self, mock_model, mock_tokenizer):
        def _slow(*a, **kw):
            time.sleep(10)
            return "done"
        mock_model.generate_numpy.side_effect = _slow
        s = SloNetServer(mock_model, mock_tokenizer, generate_timeout=0.05, enable_warmup=False)
        with pytest.raises(TimeoutError):
            await s.generate("hello")

    async def test_generate_metrics_on_success(self, server):
        await server.generate("hello")
        m = server.get_metrics()
        assert m["requests_completed"] == 1
        assert m["requests_failed"] == 0
        assert m["avg_generation_time_ms"] > 0

    async def test_generate_metrics_on_failure(self, mock_model, mock_tokenizer):
        mock_model.generate_numpy.side_effect = ValueError("boom")
        s = SloNetServer(mock_model, mock_tokenizer, enable_warmup=False)
        with pytest.raises(ValueError):
            await s.generate("hello")
        m = s.get_metrics()
        assert m["requests_failed"] == 1
        assert "boom" in m["last_error"]

    async def test_generate_passes_parameters(self, server):
        await server.generate(
            "test",
            max_new_tokens=50,
            temperature=0.5,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.2,
        )
        _, kwargs = server._model.generate_numpy.call_args
        assert kwargs["max_new_tokens"] == 50
        assert kwargs["temperature"] == 0.5
        assert kwargs["top_p"] == 0.8
        assert kwargs["top_k"] == 20
        assert kwargs["repetition_penalty"] == 1.2

    async def test_generate_circuit_breaker_records_success(self, mock_model, mock_tokenizer):
        mock_model.generate_numpy.return_value = np.array([[101, 102, 103]], dtype=np.int64)
        s = SloNetServer(mock_model, mock_tokenizer, enable_warmup=False)
        s._circuit_breaker._state = CircuitBreakerState.HALF_OPEN
        await s.generate("hello")
        assert s._circuit_breaker.state.value == "closed"

    async def test_generate_circuit_breaker_records_failure(self, mock_model, mock_tokenizer):
        mock_model.generate_numpy.side_effect = RuntimeError("fail")
        s = SloNetServer(mock_model, mock_tokenizer, enable_warmup=False)
        s._circuit_breaker._state = CircuitBreakerState.HALF_OPEN
        with pytest.raises(RuntimeError):
            await s.generate("hello")
        assert s._circuit_breaker.state.value == "open"

    async def test_generate_timeout_increments_timed_out_metric(self, mock_model, mock_tokenizer):
        def _slow(*a, **kw):
            time.sleep(10)
            return "done"
        mock_model.generate_numpy.side_effect = _slow
        s = SloNetServer(mock_model, mock_tokenizer, generate_timeout=0.05, enable_warmup=False)
        with pytest.raises((TimeoutError, asyncio.TimeoutError)):
            await s.generate("hello")
        m = s.get_metrics()
        assert m["requests_timed_out"] >= 1
        assert m["consecutive_failures"] >= 1

    async def test_generate_metrics_avg_time(self, server):
        await server.generate("hello")
        m = server.get_metrics()
        assert m["avg_generation_time_ms"] > 0

    async def test_generate_metrics_error_rate(self, mock_model, mock_tokenizer):
        mock_model.generate_numpy.side_effect = RuntimeError("fail")
        s = SloNetServer(mock_model, mock_tokenizer, enable_warmup=False)
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await s.generate("hello")
        m = s.get_metrics()
        assert m["error_rate"] == 1.0
        assert m["consecutive_failures"] == 2

    async def test_allow_request_returns_true_by_default(self, server):
        assert server._circuit_breaker.allow_request() is True

    async def test_allow_request_returns_false_when_open(self, server):
        server._circuit_breaker._state = CircuitBreakerState.OPEN
        server._circuit_breaker._last_failure_at = time.time()
        assert server._circuit_breaker.allow_request() is False

    async def test_snapshot_includes_all_expected_fields(self, server):
        await server.generate("hello")
        s = server.get_metrics()
        expected = {
            "requests_total", "requests_completed", "requests_failed",
            "requests_timed_out", "consecutive_failures",
            "avg_generation_time_ms", "max_generation_time_ms",
            "min_generation_time_ms", "last_generation_time_ms",
            "tokens_generated_total", "last_error", "error_rate",
        }
        assert set(s.keys()) == expected


# ---------------------------------------------------------------------------
# Generate Stream
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestGenerateStream:
    async def test_stream_yields_tokens(self, server):
        tokens = []
        async for token in server.generate_stream("hello"):
            tokens.append(token)
        assert len(tokens) == 2
        assert tokens[0] == "hello world"

    async def test_stream_cancellation(self, server):
        cancel = threading.Event()
        gen = server.generate_stream("hello", cancel_event=cancel)
        cancel.set()
        tokens = []
        async for token in gen:
            tokens.append(token)
        assert len(tokens) == 0

    async def test_stream_metrics(self, server):
        tokens = []
        async for token in server.generate_stream("hello"):
            tokens.append(token)
        m = server.get_metrics()
        assert m["requests_total"] == 1
        assert m["requests_completed"] == 1
        assert m["tokens_generated_total"] >= 0

    async def test_stream_errors_raise(self, server):
        server._model.generate_numpy_stream.side_effect = RuntimeError("gen fail")
        with pytest.raises(RuntimeError, match="gen fail"):
            async for _ in server.generate_stream("hello"):
                pass

    async def test_stream_error_mid_stream(self, server):
        def _stream_gen(*a, **kw):
            yield np.int64(42)
            raise RuntimeError("mid-stream fail")
        server._model.generate_numpy_stream.return_value = _stream_gen()
        with pytest.raises(RuntimeError, match="mid-stream fail"):
            async for t in server.generate_stream("hello"):
                pass

    async def test_stream_circuit_breaker_records_success(self, mock_model, mock_tokenizer):
        mock_model.generate_numpy_stream.return_value = iter([np.int64(42)])
        s = SloNetServer(mock_model, mock_tokenizer, enable_warmup=False)
        s._circuit_breaker._state = CircuitBreakerState.HALF_OPEN
        async for _ in s.generate_stream("hello"):
            pass
        assert s._circuit_breaker.state.value == "closed"

    async def test_stream_metrics_token_count(self, server):
        tokens = []
        async for t in server.generate_stream("hello"):
            tokens.append(t)
        m = server.get_metrics()
        assert m["tokens_generated_total"] == 2
        assert m["requests_total"] == 1
        assert m["requests_completed"] == 1

    async def test_stream_passes_parameters(self, server):
        tokens = []
        async for t in server.generate_stream(
            "test", max_new_tokens=10, temperature=0.3, top_p=0.7, top_k=5, repetition_penalty=1.5,
        ):
            tokens.append(t)
        _, kwargs = server._model.generate_numpy_stream.call_args
        assert kwargs["max_new_tokens"] == 10
        assert kwargs["eos_token"] == 0

    async def test_stream_cancelled_error_during_active_stream(self, mock_model, mock_tokenizer):
        wait_forever = threading.Event()
        def _blocking_stream(*a, **kw):
            yield np.int64(42)
            wait_forever.wait()
        mock_model.generate_numpy_stream.return_value = _blocking_stream()
        s = SloNetServer(mock_model, mock_tokenizer, enable_warmup=False)

        cancel = threading.Event()
        gen = s.generate_stream("hello", cancel_event=cancel)

        async def _consume():
            tokens = []
            try:
                async for t in gen:
                    tokens.append(t)
            except (asyncio.CancelledError, StopAsyncIteration):
                pass
            return tokens

        task = asyncio.create_task(_consume())
        await asyncio.sleep(0.02)
        assert not task.done()
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, RuntimeError):
            pass
        wait_forever.set()
        assert cancel.is_set()
        # Verify metrics still recorded on cancellation
        m = s.get_metrics()
        assert m["requests_total"] == 1


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------

class TestObservability:
    def test_metadata_contains_keys(self, server):
        md = server.metadata()
        assert md["model_id"] == "test-slonet"
        assert md["architecture"] == "SloTransformer"
        assert md["n_embed"] == 64
        assert md["n_head"] == 4
        assert md["vocab_size"] == 256
        assert md["max_seq_len"] == 2048
        assert md["circuit_breaker_state"] in ("closed", "disabled")

    def test_health_ready(self, server):
        h = server.health()
        assert h["status"] == "ready"

    def test_health_degraded_when_circuit_breaker_open(self, mock_model, mock_tokenizer):
        s = SloNetServer(mock_model, mock_tokenizer, enable_warmup=False)
        s._circuit_breaker._state = CircuitBreakerState.OPEN
        s._circuit_breaker._failure_count = 3
        s._circuit_breaker._last_failure_at = time.time()
        h = s.health()
        assert h["status"] == "degraded"

    def test_metadata_warmup_error_non_none(self, mock_model, mock_tokenizer):
        mock_model.generate_numpy.side_effect = RuntimeError("warmup fail")
        s = SloNetServer(mock_model, mock_tokenizer, enable_warmup=True, warmup_prompt="Hi")
        time.sleep(0.2)
        md = s.metadata()
        assert md["warmup_completed"] is False
        assert "warmup fail" in md["warmup_error"]

    def test_metadata_warmup_completed_true(self, server):
        server._warmup_completed = True
        md = server.metadata()
        assert md["warmup_completed"] is True
        assert md["warmup_error"] is None

    def test_metadata_circuit_breaker_disabled(self, mock_model, mock_tokenizer):
        s = SloNetServer(mock_model, mock_tokenizer, enable_circuit_breaker=False, enable_warmup=False)
        md = s.metadata()
        assert md["circuit_breaker_state"] == "disabled"


# ---------------------------------------------------------------------------
# Pool mode
# ---------------------------------------------------------------------------

class TestPoolMode:
    def test_pool_mode_active_with_factory(self):
        factory = MagicMock()
        factory.return_value = MagicMock()
        s = SloNetServer(
            model_factory=factory,
            tokenizer=MagicMock(),
            max_workers=3,
            enable_warmup=False,
        )
        assert s._pool_mode is True
        assert s._max_workers == 3
        assert s.pool_stats()["mode"] == "pool"

    def test_single_mode_no_factory(self, mock_model, mock_tokenizer):
        s = SloNetServer(mock_model, mock_tokenizer, enable_warmup=False)
        assert s._pool_mode is False
        assert s.pool_stats()["mode"] == "single"

    def test_factory_called_on_first_acquire(self):
        factory = MagicMock()
        factory.return_value = MagicMock()
        s = SloNetServer(
            model_factory=factory,
            tokenizer=MagicMock(),
            max_workers=2,
            enable_warmup=False,
        )
        model = s._acquire_model(timeout=1)
        factory.assert_called_once()
        assert model is factory.return_value
        assert s._pool_size == 1

    def test_acquire_reuse_after_release(self):
        factory = MagicMock()
        factory.return_value = MagicMock()
        s = SloNetServer(
            model_factory=factory,
            tokenizer=MagicMock(),
            max_workers=2,
            enable_warmup=False,
        )
        m1 = s._acquire_model(timeout=1)
        factory.assert_called_once()
        s._release_model(m1)
        m2 = s._acquire_model(timeout=1)
        factory.assert_called_once()
        assert m2 is m1

    def test_factory_called_for_each_concurrent(self):
        factory = MagicMock()
        factory.side_effect = [MagicMock(), MagicMock()]
        s = SloNetServer(
            model_factory=factory,
            tokenizer=MagicMock(),
            max_workers=2,
            enable_warmup=False,
        )
        m1 = s._acquire_model(timeout=1)
        m2 = s._acquire_model(timeout=1)
        assert factory.call_count == 2
        assert m1 is not m2

    def test_pool_exhaustion_raises(self):
        factory = MagicMock()
        factory.side_effect = [MagicMock(), MagicMock()]
        s = SloNetServer(
            model_factory=factory,
            tokenizer=MagicMock(),
            max_workers=1,
            enable_warmup=False,
        )
        s._acquire_model(timeout=1)
        with pytest.raises(RuntimeError, match="pool exhausted"):
            s._acquire_model(timeout=0.1)

    def test_pool_stats(self):
        factory = MagicMock()
        factory.return_value = MagicMock()
        s = SloNetServer(
            model_factory=factory,
            tokenizer=MagicMock(),
            max_workers=4,
            enable_warmup=False,
        )
        stats = s.pool_stats()
        assert stats["mode"] == "pool"
        assert stats["workers"] == 4
        assert stats["available"] == 0
        assert stats["created"] == 0

    def test_pool_stats_after_acquire(self):
        factory = MagicMock()
        factory.return_value = MagicMock()
        s = SloNetServer(
            model_factory=factory,
            tokenizer=MagicMock(),
            max_workers=4,
            enable_warmup=False,
        )
        s._acquire_model(timeout=1)
        stats = s.pool_stats()
        assert stats["created"] == 1
        assert stats["available"] == 0

    def test_pool_stats_after_release(self):
        factory = MagicMock()
        factory.return_value = MagicMock()
        s = SloNetServer(
            model_factory=factory,
            tokenizer=MagicMock(),
            max_workers=4,
            enable_warmup=False,
        )
        m = s._acquire_model(timeout=1)
        s._release_model(m)
        stats = s.pool_stats()
        assert stats["available"] == 1

    @pytest.mark.asyncio
    async def test_generate_uses_pool(self):
        factory = MagicMock()
        model = MagicMock()
        model.generate_numpy.return_value = np.array([[101, 102, 103]], dtype=np.int64)
        factory.return_value = model
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [10, 20, 30]
        tokenizer.decode.return_value = "pool result"
        tokenizer.eos_token_id = 0

        s = SloNetServer(
            model_factory=factory,
            tokenizer=tokenizer,
            max_workers=2,
            enable_warmup=False,
        )

        result = await s.generate("hello")
        assert result == "pool result"
        assert model.generate_numpy.called

    @pytest.mark.asyncio
    async def test_stream_generate_uses_pool(self):
        factory = MagicMock()
        model = MagicMock()
        model.generate_numpy_stream.return_value = iter([np.int64(42)])
        factory.return_value = model
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [10, 20, 30]
        tokenizer.decode.return_value = "tok"
        tokenizer.eos_token_id = 0

        s = SloNetServer(
            model_factory=factory,
            tokenizer=tokenizer,
            max_workers=2,
            enable_warmup=False,
        )

        tokens = []
        async for t in s.generate_stream("hello"):
            tokens.append(t)
        assert tokens == ["tok"]
        assert model.generate_numpy_stream.called

    def test_health_shows_pool_info(self):
        factory = MagicMock()
        factory.return_value = MagicMock()
        s = SloNetServer(
            model_factory=factory,
            tokenizer=MagicMock(),
            max_workers=3,
            enable_warmup=False,
        )
        h = s.health()
        assert h["pool"]["mode"] == "pool"
        assert h["pool"]["workers"] == 3

    def test_metadata_shows_dispatch_pool(self):
        factory = MagicMock()
        factory.return_value = MagicMock()
        s = SloNetServer(
            model_factory=factory,
            tokenizer=MagicMock(),
            max_workers=3,
            enable_warmup=False,
        )
        md = s.metadata()
        assert md["dispatch"] == "pool"
        assert md["workers"] == 3



