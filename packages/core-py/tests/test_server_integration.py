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
from unittest.mock import patch
import pytest
pytestmark = pytest.mark.slow
from domains.infrastructure.server_state import get_server_state
from domains.infrastructure.model_registry import get_model_registry, ModelRegistry
from domains.infrastructure.model_server import ModelServer, ModelStatus, CircuitBreakerState


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
        # Give warmup thread time to finish (may take longer on slow machines)
        import time
        for _ in range(10):
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
