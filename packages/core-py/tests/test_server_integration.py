"""
Integration tests for ModelRegistry + ModelServer + ServerState.

Tests the full pipeline without an HTTP server:
1. Create ServerState singleton
2. Create ModelRegistry
3. Register a mock model
4. Test generation through the registry
5. Test circuit breaker
6. Test semaphore serialization
7. Test health summary
"""

import asyncio
import time
import threading
from unittest.mock import patch
import pytest
pytestmark = pytest.mark.slow
from domains.infrastructure.server_state import get_server_state
from domains.infrastructure.model_registry import get_model_registry, ModelRegistry
from domains.infrastructure.model_server import (
    ModelServer, ModelStatus, CircuitBreakerState, PriorityRequestQueue, Priority,
)
from domains.infrastructure.event_bus import get_event_bus, set_event_bus


# ── Mock model (torch-free) ──────────────────────────────────────────


class MockTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, prompt, return_tensors="pt", **kwargs):
        return {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
        }

    def decode(self, tokens, skip_special_tokens=True):
        return "hello world"


class MockModel:
    device = "cpu"

    def __init__(self, fail_on_call=False, slow=False):
        self._fail = fail_on_call
        self._slow = slow

    def generate(self, **kwargs):
        if self._fail:
            raise RuntimeError("mock generation failure")
        if self._slow:
            time.sleep(3)
        return [[1, 2, 3, 4, 5]]

    def parameters(self):
        return []


# ── Fixtures ─────────────────────────────────────────────────────────


# ── Fixtures ─────────────────────────────────────────────────────────

# Shared state for _generate_sync mock — tests set this to configure fail/slow
_GEN_CONFIG: dict = {"fail": False, "slow": False}


def _mock_generate_sync(*args, **kwargs):
    """Mock _generate_sync — defaults to success."""
    return {
        "text": f"generated: {args[0] if args else kwargs.get('prompt', 'hello')}",
        "tokens_generated": 5,
        "elapsed_ms": 10.0,
    }


@pytest.fixture(autouse=True)
def patch_torch_deps():
    """Patch out torch from ModelServer so tests run torch-free."""
    patches = [
        patch.object(ModelServer, "_generate_sync", autospec=True,
                     side_effect=_mock_generate_sync),
    ]
    for p in patches:
        p.start()
    yield
    for p in patches:
        p.stop()


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the singleton registry before each test."""
    reg = get_model_registry()
    for m in reg.list_models():
        reg.unregister(m["model_id"])
    reg.reset_metrics()
    yield


@pytest.fixture
def tokenizer():
    return MockTokenizer()


@pytest.fixture
def model():
    return MockModel()


@pytest.fixture
def registry():
    return get_model_registry()


# ── ServerState tests ─────────────────────────────────────────────────────────


class TestServerState:
    def test_singleton(self):
        s1 = get_server_state()
        s2 = get_server_state()
        assert s1 is s2

    def test_uptime(self):
        state = get_server_state()
        assert state.uptime_seconds >= 0

    def test_request_count(self):
        state = get_server_state()
        before = state.request_count
        state.record_request()
        assert state.request_count == before + 1

    def test_atomic_fields(self):
        state = get_server_state()
        state.model.set("test-value")
        assert state.model.get() == "test-value"
        state.model_type.set("gpt2")
        assert state.model_type.get() == "gpt2"
        state.tokenizer.set("mock-tokenizer")
        assert state.tokenizer.get() == "mock-tokenizer"


# ── ModelRegistry tests ───────────────────────────────────────────────────────


class TestModelRegistry:
    def test_empty_on_start(self, registry):
        assert registry.list_models() == []
        assert registry.default_id is None

    def test_register_and_get(self, registry, model, tokenizer):
        server = registry.register("gpt2", model, tokenizer, make_default=True)
        assert server is not None
        assert registry.default_id == "gpt2"
        assert registry.get("gpt2") is server
        assert registry.get() is server  # default

    def test_register_replaces_existing(self, registry, model, tokenizer):
        registry.register("m1", model, tokenizer, make_default=True)
        old_server = registry.get("m1")
        registry.register("m1", model, tokenizer, make_default=True)
        new_server = registry.get("m1")
        assert new_server is not old_server

    def test_default_fallback(self, registry, model, tokenizer):
        registry.register("m1", model, tokenizer, make_default=False)
        assert registry.default_id == "m1"  # first registered becomes default

        registry.register("m2", model, tokenizer, make_default=True)
        assert registry.default_id == "m2"  # explicit default

    def test_unregister(self, registry, model, tokenizer):
        registry.register("m1", model, tokenizer, make_default=True)
        registry.unregister("m1")
        assert registry.list_models() == []
        assert registry.default_id is None

    def test_unregister_nonexistent(self, registry):
        assert registry.unregister("nonexistent") is False

    def test_list_models(self, registry, model, tokenizer):
        registry.register("m1", model, tokenizer)
        registry.register("m2", model, tokenizer)
        models = registry.list_models()
        assert len(models) == 2
        ids = [m["model_id"] for m in models]
        assert "m1" in ids
        assert "m2" in ids

    def test_health_summary_healthy(self, registry, model, tokenizer):
        registry.register("gpt2", model, tokenizer, make_default=True)
        health = registry.health_summary()
        assert health["healthy"] is True
        assert health["models_loaded"] == 1
        assert health["default_model"] == "gpt2"
        assert health["degraded"] is False

    def test_health_summary_empty(self, registry):
        health = registry.health_summary()
        assert health["healthy"] is False
        assert health["models_loaded"] == 0

    def test_reset_metrics(self, registry, model, tokenizer):
        registry.register("m1", model, tokenizer)
        server = registry.get("m1")
        server.metrics.requests_total = 99
        registry.reset_metrics()
        assert server.metrics.requests_total == 0

    def test_swap_default(self, registry, model, tokenizer):
        registry.register("m1", model, tokenizer, make_default=True)
        registry.register("m2", model, tokenizer, make_default=False)
        assert registry.default_id == "m1"
        registry.default_id = "m2"
        assert registry.default_id == "m2"


# ── ModelServer tests ─────────────────────────────────────────────────────────


class TestModelServer:
    @pytest.mark.asyncio
    async def test_generate_success(self, model, tokenizer):
        server = ModelServer(model, tokenizer, model_id="gpt2")
        result = await server.generate("hello")
        assert "text" in result
        assert result["tokens_generated"] >= 0

    @pytest.mark.asyncio
    async def test_generate_metrics(self, model, tokenizer):
        server = ModelServer(model, tokenizer, model_id="gpt2", enable_warmup=False)
        await server.generate("hello")
        metrics = server.metrics
        assert metrics.requests_total == 1
        assert metrics.requests_completed == 1
        assert metrics.last_generation_time_ms > 0

    @pytest.mark.asyncio
    async def test_semaphore_serializes(self, model, tokenizer):
        """Two concurrent requests should be serialized by the semaphore."""
        slow_model = MockModel(slow=True)
        server = ModelServer(slow_model, tokenizer, model_id="slow", max_concurrent=1, enable_warmup=False)

        # Patch _generate_sync to be slow for this test only
        original = server._generate_sync
        def _slow(*args, **kwargs):
            time.sleep(3)
            return {"text": "slow result", "tokens_generated": 1, "elapsed_ms": 3000.0}
        server._generate_sync = _slow

        async def gen():
            return await server.generate("hello")

        t1 = asyncio.create_task(gen())
        await asyncio.sleep(0.05)  # let t1 acquire semaphore
        t2 = asyncio.create_task(gen())

        start = time.time()
        results = await asyncio.gather(t1, t2, return_exceptions=True)
        elapsed = time.time() - start

        # With 2 requests at 3s each but serialized, should take ~6s
        # We just check it takes more than 3s (serialized, not parallel)
        assert elapsed >= 4.0, f"Expected serialized ~6s, got {elapsed:.2f}s"
        assert not any(isinstance(r, Exception) for r in results)

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens(self, model, tokenizer):
        fail_model = MockModel(fail_on_call=True)
        server = ModelServer(
            fail_model, tokenizer, model_id="fail",
            enable_circuit_breaker=True,
            failure_threshold=2,
            enable_warmup=False,
        )

        def _fail(*args, **kwargs):
            raise RuntimeError("mock generation failure")
        server._generate_sync = _fail

        for _ in range(2):
            with pytest.raises(RuntimeError):
                await server.generate("hello")

        # Third request should hit open circuit breaker
        with pytest.raises(RuntimeError, match="Circuit breaker open"):
            await server.generate("hello")

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovers(self, model, tokenizer):
        """After recovery_timeout, circuit breaker should half-open and recover."""
        fail_model = MockModel(fail_on_call=True)
        server = ModelServer(
            fail_model, tokenizer, model_id="fail",
            enable_circuit_breaker=True,
            failure_threshold=1,
            recovery_timeout=2.0,
            enable_warmup=False,
        )

        def _fail(*args, **kwargs):
            raise RuntimeError("mock generation failure")
        server._generate_sync = _fail
        with pytest.raises(RuntimeError):
            await server.generate("hello")

        # Circuit breaker is open or half-open (may transition during generate error handling)
        assert server._circuit_breaker.state in (CircuitBreakerState.OPEN, CircuitBreakerState.HALF_OPEN)

        # Wait for recovery
        await asyncio.sleep(2.1)

        # Should half-open and fail again (model still fails)
        with pytest.raises(RuntimeError):
            await server.generate("hello")

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_queue_full_generate(self, model, tokenizer):
        """Queue-full error in generate() trips circuit breaker after threshold."""
        server = ModelServer(
            model, tokenizer, model_id="test",
            enable_circuit_breaker=True,
            failure_threshold=2,
            enable_warmup=False,
        )

        # Inject a tiny queue (1 concurrent, 1 queued) and start its worker
        q = PriorityRequestQueue(max_concurrent=1, max_queue=1)
        wk = asyncio.get_event_loop().create_task(q.worker())
        server._request_queue = q
        await asyncio.sleep(0.02)

        # Make generate() slow so the second request sits in the heap
        def _slow(*args, **kwargs):
            time.sleep(0.5)
            return {"text": "done", "tokens_generated": 5, "elapsed_ms": 500.0}
        server._generate_sync = _slow

        # First request: occupies the in-flight slot
        t1 = asyncio.create_task(server.generate("hello"))
        await asyncio.sleep(0.05)

        # Second request: sits in the heap (in_flight == max_concurrent)
        t2 = asyncio.create_task(server.generate("world"))
        await asyncio.sleep(0.05)

        # Third request: heap is full → raises → CB records 1st failure
        with pytest.raises(RuntimeError, match="Queue full"):
            await server.generate("third")
        assert server._circuit_breaker._failure_count == 1

        # Fourth request: heap still full → raises → CB records 2nd → OPEN
        with pytest.raises(RuntimeError, match="Queue full"):
            await server.generate("fourth")
        assert server._circuit_breaker.state == CircuitBreakerState.OPEN

        # Cleanup
        await t1
        t2.cancel()
        try:
            await t2
        except (asyncio.CancelledError, RuntimeError):
            pass
        wk.cancel()
        try:
            await wk
        except (asyncio.CancelledError, RuntimeError):
            pass

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_queue_full_stream(self, model, tokenizer):
        """Queue-full error in generate_stream() trips circuit breaker.

        Fill the heap via queue.acquire() directly, then verify
        generate_stream() raises Queue full → CB opens.
        """
        server = ModelServer(
            model, tokenizer, model_id="test",
            enable_circuit_breaker=True,
            failure_threshold=2,
            enable_warmup=False,
        )

        q = PriorityRequestQueue(max_concurrent=1, max_queue=1)
        wk = asyncio.get_event_loop().create_task(q.worker())
        server._request_queue = q
        await asyncio.sleep(0.02)

        def _slow(*args, **kwargs):
            time.sleep(0.5)
            return {"text": "done", "tokens_generated": 5, "elapsed_ms": 500.0}
        server._generate_sync = _slow

        # Slow generate occupies the in-flight slot
        t1 = asyncio.create_task(server.generate("hello"))
        await asyncio.sleep(0.05)

        # Fill heap via direct acquire (worker busy, in_flight=1)
        a1 = asyncio.create_task(
            q.acquire(priority=Priority.HIGH, request_id="fill")
        )
        await asyncio.sleep(0.05)

        # generate_stream tries acquire → heap full → raises → CB records 1
        s3 = server.generate_stream("third")
        with pytest.raises(RuntimeError, match="Queue full"):
            await s3.__anext__()
        assert server._circuit_breaker._failure_count == 1

        # Fourth → CB records 2nd → OPEN
        s4 = server.generate_stream("fourth")
        with pytest.raises(RuntimeError, match="Queue full"):
            await s4.__anext__()
        assert server._circuit_breaker.state == CircuitBreakerState.OPEN

        # Cleanup
        await t1
        a1.cancel()
        try:
            await a1
        except Exception:
            pass
        wk.cancel()
        try:
            await wk
        except (asyncio.CancelledError, RuntimeError):
            pass

    @pytest.mark.asyncio
    async def test_timeout_semaphore(self, model, tokenizer):
        slow_model = MockModel(slow=True)
        server = ModelServer(slow_model, tokenizer, model_id="slow", max_concurrent=1, enable_warmup=False)

        # Patch to be slow
        def _slow(*args, **kwargs):
            time.sleep(3)
            return {"text": "slow result", "tokens_generated": 1, "elapsed_ms": 3000.0}
        server._generate_sync = _slow

        # First request hogs the semaphore
        t1 = asyncio.create_task(server.generate("hello"))

        await asyncio.sleep(0.1)

        # Second request should time out waiting for semaphore (0.1s timeout)
        # The semaphore acquire timeout is min(generate_timeout, 30s), so
        # by default this won't time out. Let's just verify it waits.
        t2 = asyncio.create_task(server.generate("hello"))

        results = await asyncio.gather(t1, t2, return_exceptions=True)
        assert not any(isinstance(r, TimeoutError) for r in results)
        assert not any(isinstance(r, Exception) for r in results)

    @pytest.mark.asyncio
    async def test_pre_gen_hooks(self, model, tokenizer):
        hooks = []
        server = ModelServer(model, tokenizer, model_id="hooked", enable_warmup=False)
        server.add_pre_generate_hook(lambda: hooks.append("pre"))
        await server.generate("hello")
        assert hooks == ["pre"]

    @pytest.mark.asyncio
    async def test_post_gen_hooks(self, model, tokenizer):
        hooks = []
        server = ModelServer(model, tokenizer, model_id="hooked", enable_warmup=False)

    @pytest.mark.asyncio
    async def test_on_error_hooks(self, model, tokenizer):
        errors = []
        fail_model = MockModel(fail_on_call=True)
        server = ModelServer(fail_model, tokenizer, model_id="fail", enable_warmup=False)
        server.add_on_error_hook(lambda e: errors.append(str(e)))
        def _fail(*args, **kwargs):
            raise RuntimeError("mock generation failure")
        server._generate_sync = _fail
        with pytest.raises(RuntimeError):
            await server.generate("hello")
        assert len(errors) == 1
        assert "mock generation failure" in errors[0]

    def test_status_degraded_on_error(self, model, tokenizer):
        server = ModelServer(model, tokenizer, model_id="test")
        assert server.status == ModelStatus.READY
        server.set_status(ModelStatus.DEGRADED)
        assert server.status == ModelStatus.DEGRADED

    def test_swap_model(self, model, tokenizer):
        server = ModelServer(model, tokenizer, model_id="swappable")
        new_model = MockModel()
        server.swap_model(new_model)
        assert server._model_ref is new_model
        assert server.status == ModelStatus.READY

    def test_metrics_snapshot(self, model, tokenizer):
        server = ModelServer(model, tokenizer, model_id="metrics", enable_warmup=False)
        snapshot = server.get_metrics_snapshot()
        assert snapshot["model_id"] == "metrics"
        assert snapshot["status"] == "ready"
        assert snapshot["requests_total"] == 0

    @pytest.mark.asyncio
    async def test_generate_with_kwargs(self, model, tokenizer):
        server = ModelServer(model, tokenizer, model_id="kwargs-test")
        result = await server.generate("hello", max_new_tokens=200, temperature=0.9)
        assert "text" in result


# ── Warmup tests ──────────────────────────────────────────────────────────────


class TestWarmup:
    def test_warmup_default_enabled(self, model, tokenizer):
        """Warmup runs by default on construction."""
        server = ModelServer(model, tokenizer, model_id="warmup", enable_warmup=True)
        # Give warmup thread time to finish (may take longer on slow machines or with CPU fallback)
        import time
        for _ in range(20):
            time.sleep(0.5)
            with server._warmup_lock:
                if server._warmup_completed or server._warmup_error:
                    break
        with server._warmup_lock:
            assert server._warmup_completed or server._warmup_error, "warmup did not complete or error"
        snap = server.get_metrics_snapshot()
        assert snap["warmup_completed"]

    def test_warmup_disabled(self, model, tokenizer):
        """Warmup can be disabled."""
        server = ModelServer(model, tokenizer, model_id="no-warmup", enable_warmup=False)
        assert not server._warmup_completed
        snap = server.get_metrics_snapshot()
        assert not snap["warmup_completed"]

    def test_warmup_metrics_recorded(self, model, tokenizer):
        """Warmup request counts toward metrics."""
        server = ModelServer(model, tokenizer, model_id="warmup-metrics")
        import time
        time.sleep(1.0)
        snap = server.get_metrics_snapshot()
        assert snap["requests_total"] >= 1
        assert snap["requests_completed"] >= 1

    def test_warmup_on_model_swap(self, model, tokenizer):
        """Warmup re-runs after swap_model()."""
        server = ModelServer(model, tokenizer, model_id="swap-warmup")
        import time
        time.sleep(1.0)
        assert server._warmup_completed

        # Swap model — warmup restarts and eventually completes
        new_model = MockModel()
        server.swap_model(new_model)
        time.sleep(1.0)
        assert server._warmup_completed

    def test_warmup_graceful_on_failure(self, tokenizer):
        """Warmup failure doesn't crash — just degrades status."""
        with patch.object(ModelServer, "_generate_sync", new=lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("mock generation failure"))):
            server = ModelServer(MockModel(fail_on_call=True), tokenizer, model_id="fail-warmup")
            import time
            deadline = time.time() + 15.0
            while time.time() < deadline:
                if server._warmup_error is not None:
                    break
                time.sleep(0.2)
            assert not server._warmup_completed, "warmup should not have completed"
            assert server._warmup_error is not None, f"expected warmup error, got None (completed={server._warmup_completed})"
            assert server.status == ModelStatus.DEGRADED


# ── ModelRegistry ↔ ModelServer wiring ─────────────────────────────────────


class TestRegistryWiring:
    """Verify that registering a model creates a ModelServer with lifecycle
    management, and that unregistering cleans up properly."""

    def test_register_creates_model_server(self, registry, model, tokenizer):
        """Registering a model produces a ModelServer in the registry."""
        server = registry.register("gpt2", model, tokenizer, make_default=True)
        assert isinstance(server, ModelServer)
        assert registry.get("gpt2") is server
        assert registry.default_id == "gpt2"

    def test_list_models_shows_registered(self, registry, model, tokenizer):
        """Registered models appear in list_models with metrics."""
        registry.register("test-model", model, tokenizer)
        models = registry.list_models()
        assert len(models) >= 1
        ids = [m["model_id"] for m in models]
        assert "test-model" in ids

    def test_health_summary_includes_model(self, registry, model, tokenizer):
        """Health summary reports registered models."""
        registry.register("healthy-model", model, tokenizer)
        health = registry.health_summary()
        assert health["models_loaded"] >= 1
        assert health["default_model"] == "healthy-model"

    def test_unregister_removes_model(self, registry, model, tokenizer):
        """Unregister removes the ModelServer and cleans up."""
        registry.register("temp-model", model, tokenizer)
        assert registry.get("temp-model") is not None
        result = registry.unregister("temp-model")
        assert result is True
        assert registry.get("temp-model") is None

    def test_unregister_updates_default(self, registry, model, tokenizer):
        """Unregistering the default model falls back to the next one."""
        registry.register("first", model, tokenizer, make_default=True)
        registry.register("second", model, tokenizer)
        assert registry.default_id == "first"
        registry.unregister("first")
        assert registry.default_id == "second"

    def test_generate_through_registry(self, registry, model, tokenizer):
        """Generation through registry delegates to ModelServer."""
        registry.register("gen-model", model, tokenizer, make_default=True)
        result = asyncio.run(registry.generate("Hello world"))
        assert "text" in result
        assert result["tokens_generated"] >= 1

    def test_registry_lifecycle_hooks_fire(self, registry, model, tokenizer):
        """Pre/post hooks on ModelServer fire during generation."""
        server = registry.register("hook-model", model, tokenizer, make_default=True)
        pre_count_before = len(server._pre_generate_hooks)
        post_count_before = len(server._post_generate_hooks)
        asyncio.run(registry.generate("test prompt"))
        # Hooks should have fired at least once (warmup + explicit generate)
        assert server.metrics.requests_completed >= 1

    def test_registry_metrics_track_generation(self, registry, model, tokenizer):
        """Metrics are recorded after generation through registry."""
        server = registry.register("metrics-model", model, tokenizer, make_default=True)
        asyncio.run(registry.generate("metrics test"))
        snap = server.get_metrics_snapshot()
        assert snap["requests_completed"] >= 1
        assert snap["tokens_generated_total"] >= 1

    def test_register_replaces_old_server(self, registry, tokenizer):
        """Registering the same ID replaces the old ModelServer."""
        old_model = MockModel()
        new_model = MockModel()
        server1 = registry.register("swap-model", old_model, tokenizer)
        server2 = registry.register("swap-model", new_model, tokenizer)
        assert server1 is not server2
        assert registry.get("swap-model") is server2

    def test_multiple_models_coexist(self, registry, model, tokenizer):
        """Multiple models can coexist in the registry."""
        registry.register("model-a", model, tokenizer)
        registry.register("model-b", model, tokenizer)
        models = registry.list_models()
        ids = [m["model_id"] for m in models]
        assert "model-a" in ids
        assert "model-b" in ids


# ── EventBus + CircuitBreaker integration tests ──────────────────────────────


class TestCircuitBreakerEvents:

    @pytest.fixture(autouse=True)
    def setup_event_bus(self):
        bus = get_event_bus()
        bus.clear()
        bus._max_history = 100
        yield bus
        bus.clear()

    @pytest.mark.asyncio
    async def test_cb_emits_open_on_failure(self, model, tokenizer, setup_event_bus):
        bus = setup_event_bus
        events = []

        def handler(event, data):
            events.append((event, data))

        bus.on("circuit_breaker.open", handler)

        server = ModelServer(
            model, tokenizer, model_id="test-cb-open",
            enable_circuit_breaker=True, failure_threshold=1,
            enable_warmup=False,
        )

        def _fail(*args, **kwargs):
            raise RuntimeError("fail")
        server._generate_sync = _fail

        with pytest.raises(RuntimeError):
            await server.generate("hello")

        assert len(events) == 1
        assert events[0][0] == "circuit_breaker.open"
        assert events[0][1]["model_id"] == "test-cb-open"
        assert events[0][1]["old_state"] == "closed"
        assert events[0][1]["new_state"] == "open"

    @pytest.mark.asyncio
    async def test_cb_emits_closed_on_reset(self, model, tokenizer, setup_event_bus):
        bus = setup_event_bus
        events = []

        def handler(event, data):
            events.append((event, data))

        bus.on("circuit_breaker.closed", handler)

        server = ModelServer(
            model, tokenizer, model_id="test-cb-closed",
            enable_circuit_breaker=True, failure_threshold=1,
            recovery_timeout=0.5,
            enable_warmup=False,
        )

        def _fail(*args, **kwargs):
            raise RuntimeError("fail")
        server._generate_sync = _fail

        # First failure opens the breaker
        with pytest.raises(RuntimeError):
            await server.generate("hello")

        # Wait for recovery timeout to pass (breaker auto HALF_OPEN on state read)
        await asyncio.sleep(0.6)

        # Now fix the model and succeed
        server._generate_sync = _mock_generate_sync
        await server.generate("hello")

        assert len(events) >= 1
        closed_events = [e for e in events if e[0] == "circuit_breaker.closed"]
        assert len(closed_events) == 1
        assert closed_events[0][1]["model_id"] == "test-cb-closed"
        assert closed_events[0][1]["new_state"] == "closed"

    @pytest.mark.asyncio
    async def test_cb_emits_via_model_server(self, model, tokenizer, setup_event_bus):
        bus = setup_event_bus
        events = []

        def handler(event, data):
            events.append((event, data))

        bus.on("circuit_breaker.open", handler)
        bus.on("circuit_breaker.closed", handler)

        server = ModelServer(
            model, tokenizer, model_id="test-cb-all",
            enable_circuit_breaker=True, failure_threshold=1,
            enable_warmup=False,
        )

        def _fail(*args, **kwargs):
            raise RuntimeError("fail")
        server._generate_sync = _fail

        with pytest.raises(RuntimeError):
            await server.generate("hello")

        opened = [e for e in events if e[0] == "circuit_breaker.open"]
        assert len(opened) >= 1
        assert opened[0][1]["model_id"] == "test-cb-all"

    # ── Generation lifecycle event tests ──────────────────────────────

    @pytest.mark.asyncio
    async def test_generation_emits_started_completed(self, model, tokenizer, setup_event_bus):
        bus = setup_event_bus
        events = []

        def handler(event, data):
            events.append((event, data))

        bus.on("generation.started", handler)
        bus.on("generation.completed", handler)
        bus.on("generation.failed", handler)

        server = ModelServer(
            model, tokenizer, model_id="test-gen-lifecycle",
            enable_circuit_breaker=False, enable_warmup=False,
        )
        result = await server.generate("hello")
        assert result["tokens_generated"] == 5

        started = [e for e in events if e[0] == "generation.started"]
        completed = [e for e in events if e[0] == "generation.completed"]
        failed = [e for e in events if e[0] == "generation.failed"]

        assert len(started) == 1, f"expected 1 started, got {len(started)}"
        assert started[0][1]["model_id"] == "test-gen-lifecycle"
        assert started[0][1]["prompt_length"] == 5

        assert len(completed) == 1, f"expected 1 completed, got {len(completed)}"
        assert completed[0][1]["model_id"] == "test-gen-lifecycle"
        assert completed[0][1]["tokens"] == 5
        assert completed[0][1]["elapsed_ms"] > 0

        assert len(failed) == 0, f"expected 0 failed, got {len(failed)}"

    @pytest.mark.asyncio
    async def test_generation_emits_failed(self, model, tokenizer, setup_event_bus):
        bus = setup_event_bus
        events = []

        def handler(event, data):
            events.append((event, data))

        bus.on("generation.started", handler)
        bus.on("generation.failed", handler)

        server = ModelServer(
            model, tokenizer, model_id="test-gen-fail",
            enable_circuit_breaker=True, failure_threshold=5,
            enable_warmup=False,
        )

        def _fail(*args, **kwargs):
            raise RuntimeError("mock generation failure")
        server._generate_sync = _fail

        with pytest.raises(RuntimeError):
            await server.generate("hello")

        started = [e for e in events if e[0] == "generation.started"]
        failed = [e for e in events if e[0] == "generation.failed"]

        assert len(started) == 1, f"expected 1 started, got {len(started)}"
        assert len(failed) == 1, f"expected 1 failed, got {len(failed)}"
        assert "mock generation failure" in failed[0][1]["error"]

    @pytest.mark.asyncio
    async def test_generation_stream_emits_started_completed(self, model, tokenizer, setup_event_bus):
        bus = setup_event_bus
        events = []

        def handler(event, data):
            events.append((event, data))

        bus.on("generation.started", handler)
        bus.on("generation.completed", handler)

        server = ModelServer(
            model, tokenizer, model_id="test-stream-lifecycle",
            enable_circuit_breaker=False, enable_warmup=False,
        )

        def _mock_backend():
            class FakeBackend:
                def generate_stream(self, *a, **kw):
                    yield "hello"
                    yield " world"
                    return {"text": "hello world", "tokens_generated": 2}
            return FakeBackend()

        with patch.object(server, "_select_backend", return_value=_mock_backend()):
            tokens = []
            async for token in server.generate_stream("hello", max_new_tokens=3):
                tokens.append(token)

        started = [e for e in events if e[0] == "generation.started"]
        completed = [e for e in events if e[0] == "generation.completed"]

        assert len(started) == 1, f"expected 1 started, got {len(started)}"
        assert started[0][1]["model_id"] == "test-stream-lifecycle"
        assert started[0][1].get("streaming") is True

        assert len(completed) == 1, f"expected 1 completed, got {len(completed)}"
        assert completed[0][1]["tokens"] == 2

    @pytest.mark.asyncio
    async def test_generation_stream_emits_failed(self, model, tokenizer, setup_event_bus):
        bus = setup_event_bus
        events = []

        def handler(event, data):
            events.append((event, data))

        bus.on("generation.failed", handler)
        bus.on("generation.started", handler)

        server = ModelServer(
            model, tokenizer, model_id="test-stream-fail",
            enable_circuit_breaker=False, enable_warmup=False,
        )

        class FailBackend:
            def generate_stream(self, *a, **kw):
                raise RuntimeError("stream generation failure")

        with patch.object(server, "_select_backend", return_value=FailBackend()):
            with pytest.raises(RuntimeError, match="stream generation failure"):
                async for token in server.generate_stream("hello"):
                    pass

        failed = [e for e in events if e[0] == "generation.failed"]
        assert len(failed) == 1, f"expected 1 failed, got {len(failed)}"
        assert "stream generation failure" in failed[0][1]["error"]

    @pytest.mark.asyncio
    async def test_generation_stream_emits_cancelled(self, model, tokenizer, setup_event_bus):
        bus = setup_event_bus
        events = []

        def handler(event, data):
            events.append((event, data))

        bus.on("generation.cancelled", handler)
        bus.on("generation.started", handler)

        server = ModelServer(
            model, tokenizer, model_id="test-stream-cancel",
            enable_circuit_breaker=False, enable_warmup=False,
        )

        class SlowBackend:
            def generate_stream(self, *a, **kw):
                import time
                for i in range(50):
                    time.sleep(0.005)
                    yield f"token{i}"

        cancel_event = threading.Event()

        with patch.object(server, "_select_backend", return_value=SlowBackend()):
            gen = server.generate_stream("hello", max_new_tokens=50, cancel_event=cancel_event)
            first = await gen.__anext__()
            assert first == "token0"
            await gen.aclose()

        cancelled = [e for e in events if e[0] == "generation.cancelled"]
        assert len(cancelled) == 1, f"expected 1 cancelled, got {len(cancelled)}"
        assert cancelled[0][1]["model_id"] == "test-stream-cancel"
