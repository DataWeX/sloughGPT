"""
Fast, torch-free unit tests for the uncovered branches of model_server.py.

Targets the branches that the slow integration suite does not reach:
PriorityRequestQueue cancellation paths, SessionKVCache edge cases,
torch.compile/inference-mode helpers, generate backends, LocalBackend
streaming, and ModelServer lifecycle/streaming error paths.

torch and transformers are NOT installed; the tests install minimal fake
modules into ``sys.modules`` for the duration of each test.
"""

import asyncio
import queue
import sys
import threading
import time
import types
import warnings
from contextlib import contextmanager
from threading import Lock
from unittest.mock import patch

import numpy as np
import pytest

import domains.infrastructure.model_server as model_server
from domains.infrastructure.model_server import (
    GenerateBackend,
    GuardBackend,
    LocalBackend,
    ModelMetrics,
    ModelServer,
    ModelStatus,
    Priority,
    PriorityRequestQueue,
    SessionKVCache,
    SESSION_KV_CACHE,
    CircuitBreaker,
    CircuitBreakerState,
    _cancelable_gen,
    _emit_gen_event,
    _has_mps,
    _is_intel_mac,
    _mps_oom_recovery,
    _schedule_gc,
)


# ── Minimal torch / transformers stand-ins ────────────────────────────


class FakeTensor:
    """Minimal 2D tensor stand-in supporting the ops model_server needs."""

    def __init__(self, data):
        self._data = data

    def to(self, device):
        return self

    def cpu(self):
        return self

    @property
    def shape(self):
        return (len(self._data), len(self._data[0]))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return FakeTensor(self._data[idx])
        return FakeTensor([self._data[idx]])

    def tolist(self):
        return self._data[0]


class FakeTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, prompt, return_tensors="pt", **kwargs):
        return {
            "input_ids": FakeTensor([[1, 2, 3]]),
            "attention_mask": FakeTensor([[1, 1, 1]]),
        }

    def decode(self, tokens, skip_special_tokens=True):
        return "hello world"


class FakeTextStreamer:
    def __init__(self, tokenizer, skip_prompt=True, timeout=None):
        self.text_queue = queue.Queue()
        self.stop_signal = object()

    def put(self, token):
        self.text_queue.put(token)

    def end(self):
        self.text_queue.put(self.stop_signal)


@contextmanager
def fake_torch(**kwargs):
    """Install a fake ``torch`` module into sys.modules for the test."""
    mod = types.ModuleType("torch")
    mod.tensor = lambda data: FakeTensor(data)
    mod.inference_mode = _inference_mode_cm

    mod.compile_calls = []

    def _compile(model, backend="inductor"):
        if kwargs.get("compile_raises"):
            raise RuntimeError("compile boom")
        mod.compile_calls.append((model, backend))
        return kwargs.get("compile_result", ("compiled", model))

    if kwargs.get("has_compile", True):
        mod.compile = _compile

    if kwargs.get("has_mps"):
        class _Mps:
            @staticmethod
            def empty_cache():
                if kwargs.get("mps_raises"):
                    raise RuntimeError("mps boom")

        mod.mps = _Mps()

    sys.modules["torch"] = mod
    try:
        yield
    finally:
        sys.modules.pop("torch", None)


@contextmanager
def fake_transformers():
    """Install a fake ``transformers`` module into sys.modules for the test."""
    tmod = types.ModuleType("transformers")
    tmod.TextIteratorStreamer = FakeTextStreamer
    tmod.StoppingCriteria = type("StoppingCriteria", (), {})
    sys.modules["transformers"] = tmod
    try:
        yield
    finally:
        sys.modules.pop("transformers", None)


@contextmanager
def _inference_mode_cm():
    yield


# ── Model stand-ins ───────────────────────────────────────────────────


class NpMockModel:
    def __init__(self, fail=False):
        self._fail = fail

    def parameters(self):
        return []

    def generate(self, **kwargs):
        if self._fail:
            raise RuntimeError("mock generation failure")
        return np.array([[1, 2, 3, 4, 5]])


class MpsModel(NpMockModel):
    def cpu(self):
        return self

    def to(self, device):
        return self


class StreamModel:
    def __init__(self, tokens, delay=0.0):
        self._tokens = tokens
        self._delay = delay

    def generate(self, **kwargs):
        streamer = kwargs.get("streamer")
        for i, t in enumerate(self._tokens):
            if self._delay:
                time.sleep(self._delay)
            streamer.put(t)
        streamer.end()
        return "dummy"


class FailStreamModel:
    def generate(self, **kwargs):
        streamer = kwargs["streamer"]
        streamer.put("tok")
        raise RuntimeError("stream boom")


class FakeGuard:
    def __init__(self, alive=True, result=None, stream_tokens=None, stream_error=None):
        self.alive = alive
        self._result = result or {}
        self._stream_tokens = stream_tokens if stream_tokens is not None else ["t1", "t2"]
        self._stream_error = stream_error
        self._crash = None
        self._restart = None

    def on_crash(self, cb):
        self._crash = cb

    def on_restart(self, cb):
        self._restart = cb

    def generate(self, **kwargs):
        result = {"text": "guard text", "tokens_generated": 3}
        result.update(self._result)
        return result

    def generate_stream(self, **kwargs):
        if self._stream_error is not None:
            raise self._stream_error

        def _gen():
            for t in self._stream_tokens:
                yield t

        return _gen()


def _fast_generate(*args, **kwargs):
    return {"text": "ok", "tokens_generated": 5, "elapsed_ms": 1.0}


def _make_server(model=None, tokenizer=None, **kwargs):
    return ModelServer(
        model=model if model is not None else NpMockModel(),
        tokenizer=tokenizer if tokenizer is not None else FakeTokenizer(),
        enable_warmup=False,
        **kwargs,
    )


# ── Module helpers ────────────────────────────────────────────────────


def test_emit_gen_event_exception():
    with patch.object(model_server, "_get_gen_bus", side_effect=Exception("bus down")):
        _emit_gen_event("test.event", {})


def test_is_intel_mac():
    assert _is_intel_mac() in (True, False)
    with patch("platform.system", side_effect=Exception("probe failed")):
        assert _is_intel_mac() is False


def test_has_mps_success_and_import_error():
    assert _has_mps() in (True, False)
    from domains.infrastructure import ml_types

    orig = ml_types.mps
    try:
        del ml_types.mps
        assert _has_mps() is False
    finally:
        ml_types.mps = orig


def test_mps_oom_recovery_available_and_failure():
    with patch.object(model_server, "_has_mps", return_value=True):
        _mps_oom_recovery()
    with patch.object(model_server, "_has_mps", return_value=True):
        with patch(
            "domains.infrastructure.ml_types.mps.empty_cache",
            side_effect=Exception("mps cleared failed"),
        ):
            _mps_oom_recovery()


def test_schedule_gc_thread_failure():
    with patch.object(model_server, "Thread", side_effect=Exception("thread boom")):
        _schedule_gc()


# ── torch.compile / inference mode ────────────────────────────────────


class _Param:
    numel_val = 10_000_000
    device = "cpu"

    def numel(self):
        return self.numel_val


class BigModel:
    device = "cpu"

    def __init__(self):
        self._params = [_Param()]

    def parameters(self):
        return self._params

    def generate(self, **kwargs):
        return "dummy"


class _BadParamsModel:
    def parameters(self):
        raise RuntimeError("params boom")

    def generate(self, **kwargs):
        return None


# ── PriorityRequestQueue edge cases ───────────────────────────────────


async def test_acquire_cancel_before_pop():
    q = PriorityRequestQueue(max_concurrent=1, max_queue=16)
    task = asyncio.create_task(q.acquire(priority=Priority.HIGH))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    d = await q.depth()
    assert d == [0, 0, 0]
    assert q.in_flight == 0


async def test_acquire_cancel_after_pop():
    q = PriorityRequestQueue(max_concurrent=1, max_queue=16)
    task = asyncio.create_task(q.acquire(priority=Priority.HIGH))
    await asyncio.sleep(0)
    async with q._lock:
        item = q._pop()
    q._in_flight += 1
    item.future.set_result(None)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert q.in_flight == 0


async def test_queue_depth_and_metrics_snapshot():
    q = PriorityRequestQueue(max_concurrent=1, max_queue=16)
    tasks = []
    for pr in (Priority.HIGH, Priority.MEDIUM, Priority.LOW):
        tasks.append(asyncio.create_task(q.acquire(priority=pr)))
    await asyncio.sleep(0)
    d = await q.depth()
    assert d == [1, 1, 1]
    qm = q.metrics_snapshot()
    assert qm.total_depth == 3
    assert qm.depth_high == 1 and qm.depth_medium == 1 and qm.depth_low == 1
    for t in tasks:
        t.cancel()
    for t in tasks:
        with pytest.raises(asyncio.CancelledError):
            await t


async def test_acquire_queue_full():
    q = PriorityRequestQueue(max_concurrent=1, max_queue=1)
    t1 = asyncio.create_task(q.acquire(priority=Priority.LOW, request_id="a"))
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="Queue full"):
        await q.acquire(priority=Priority.LOW, request_id="b")
    t1.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t1
    assert q.in_flight == 0


async def test_submit_queue_full():
    async def _blocker():
        await asyncio.sleep(10)
        return "x"

    q = PriorityRequestQueue(max_concurrent=1, max_queue=1)
    first = _blocker()
    t1 = asyncio.create_task(q.submit(first, request_id="a"))
    await asyncio.sleep(0)
    second = _blocker()
    with pytest.raises(RuntimeError, match="Queue full"):
        await q.submit(second, request_id="b")
    t1.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t1
    second.close()
    async with q._lock:
        for item in q._heap:
            if item.coro is not None:
                item.coro.close()
        q._heap.clear()


# ── SessionKVCache ────────────────────────────────────────────────────


def test_kv_cache_miss():
    c = SessionKVCache()
    pkv, prefix = c.get("s1", [1, 2, 3])
    assert pkv is None and prefix == 0


def test_kv_cache_prefix_hit_and_mismatch():
    c = SessionKVCache()
    c.store("s1", [1, 2, 3, 4], "pkv")
    pkv, prefix = c.get("s1", [1, 2, 5])
    assert pkv == "pkv" and prefix == 2
    pkv, prefix = c.get("s1", [9, 2, 3])
    assert pkv is None and prefix == 0


def test_kv_cache_lru_eviction():
    c = SessionKVCache(max_sessions=2, ttl=600)
    now = time.time()
    c._caches["a"] = ([1], "pkv_a", now - 2)
    c._caches["b"] = ([2], "pkv_b", now - 1)
    c.store("c", [3], "pkv_c")
    assert "a" not in c._caches
    assert set(c._caches) == {"b", "c"}


def test_kv_cache_size_and_clear():
    c = SessionKVCache()
    c.store("s1", [1, 2], "pkv")
    assert c.size == 1
    c.clear("s1")
    assert c.size == 0
    stats = c.stats()
    assert stats["max_sessions"] == 20
    assert stats["ttl_seconds"] == 600.0


def test_kv_cache_evict_expired():
    c = SessionKVCache(ttl=0.0)
    c._caches["old"] = ([1], "pkv", time.time() - 5)
    c.evict_expired()
    assert c.size == 0


# ── ModelMetrics ──────────────────────────────────────────────────────


def test_metrics_record_timeout():
    m = ModelMetrics()
    m.record_timeout()
    assert m.requests_timed_out == 1
    assert m.consecutive_failures == 1


def test_metrics_zero_properties():
    m = ModelMetrics()
    assert m.avg_generation_time_ms == 0.0
    assert m.error_rate == 0.0
    snap = m.snapshot()
    assert snap["requests_total"] == 0
    assert snap["min_generation_time_ms"] == 0.0
    assert snap["avg_generation_time_ms"] == 0.0


def test_metrics_computed_values():
    m = ModelMetrics(requests_completed=2, total_generation_time_ms=20.0)
    assert m.avg_generation_time_ms == 10.0
    m2 = ModelMetrics(requests_total=4, requests_failed=1)
    assert m2.error_rate == 0.25


def test_circuit_breaker_half_open_transition():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
    cb.record_failure()
    assert cb.state == CircuitBreakerState.OPEN
    time.sleep(0.02)
    assert cb.state == CircuitBreakerState.HALF_OPEN


def test_circuit_breaker_half_open_failure_reopens():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
    cb.record_failure()
    time.sleep(0.02)
    assert cb.state == CircuitBreakerState.HALF_OPEN
    cb.record_failure()
    assert cb.state == CircuitBreakerState.OPEN
    assert cb._failure_count == cb.failure_threshold


def test_circuit_breaker_state_change_callback():
    seen = []
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=30.0)
    cb._on_state_change = lambda o, n: seen.append((o, n))
    cb.record_failure()
    assert seen == [(CircuitBreakerState.CLOSED, CircuitBreakerState.OPEN)]
    cb.record_success()
    assert seen[-1] == (CircuitBreakerState.OPEN, CircuitBreakerState.CLOSED)


# ── GenerateBackend base ──────────────────────────────────────────────


def test_generate_backend_base():
    b = GenerateBackend()
    assert b.alive is True
    with pytest.raises(NotImplementedError):
        b.generate("p", 10, 0.7, 0.9, 50, 1.0)
    with pytest.raises(NotImplementedError):
        b.generate_stream("p", 10, 0.7, 0.9, 50, 1.0)


# ── GuardBackend ──────────────────────────────────────────────────────


def test_guard_backend_init_and_alive():
    gb = GuardBackend(None)
    assert gb.alive is False
    gb2 = GuardBackend(FakeGuard(alive=True))
    assert gb2.alive is True


def test_guard_backend_generate():
    guard = FakeGuard()
    gb = GuardBackend(guard)
    result = gb.generate(
        "p", 10, 0.7, 0.9, 50, 1.0,
        input_ids="skip", attention_mask="skip", extra=1,
    )
    assert result["text"] == "guard text"
    assert "elapsed_ms" in result


def test_guard_backend_generate_stream_and_cancel():
    guard = FakeGuard(stream_tokens=["x", "y"])
    gb = GuardBackend(guard)
    assert list(gb.generate_stream("p", 10, 0.7, 0.9, 50, 1.0)) == ["x", "y"]
    ce = threading.Event()
    assert list(gb.generate_stream("p", 10, 0.7, 0.9, 50, 1.0, cancel_event=ce)) == ["x", "y"]
    ce.set()
    assert list(gb.generate_stream("p", 10, 0.7, 0.9, 50, 1.0, cancel_event=ce)) == []


# ── LocalBackend ──────────────────────────────────────────────────────


def test_local_backend_alive():
    lb = LocalBackend(
        model=None, tokenizer=None, lock=Lock(), gen_lock=Lock(),
        device="cpu", tokenize_cache={},
    )
    assert lb.alive is False


def test_local_backend_generate():
    with fake_torch():
        lb = LocalBackend(
            model=NpMockModel(), tokenizer=FakeTokenizer(), lock=Lock(),
            gen_lock=Lock(), device="cpu", tokenize_cache={},
        )
        result = lb.generate("hello", 10, 0.7, 0.9, 50, 1.0, session_id="fresh-sess")
        assert result["text"] == "hello world"
        assert result["tokens_generated"] == 2


def test_local_backend_generate_session_kv():
    with fake_torch():
        lb = LocalBackend(
            model=NpMockModel(), tokenizer=FakeTokenizer(), lock=Lock(),
            gen_lock=Lock(), device="cpu", tokenize_cache={},
        )
        SESSION_KV_CACHE.store("unit-sess", [1, 2, 3], "pkv-obj")
        try:
            result = lb.generate("hello", 10, 0.7, 0.9, 50, 1.0, session_id="unit-sess")
            assert result["tokens_generated"] == 2
        finally:
            SESSION_KV_CACHE.clear("unit-sess")


def test_local_backend_generate_mps_fallback():
    with fake_torch():
        lb = LocalBackend(
            model=MpsModel(), tokenizer=FakeTokenizer(), lock=Lock(),
            gen_lock=Lock(), device="mps", tokenize_cache={},
        )
        result = lb.generate("hello", 10, 0.7, 0.9, 50, 1.0)
        assert result["tokens_generated"] == 2


def test_local_backend_generate_mps_cpu_fallback_failure():
    class _CpuRaisesModel(MpsModel):
        def cpu(self):
            raise RuntimeError("cpu boom")

    with fake_torch():
        lb = LocalBackend(
            model=_CpuRaisesModel(), tokenizer=FakeTokenizer(), lock=Lock(),
            gen_lock=Lock(), device="mps", tokenize_cache={},
        )
        result = lb.generate("hello", 10, 0.7, 0.9, 50, 1.0)
        assert result["tokens_generated"] == 2


def test_local_backend_generate_mps_restore_failure():
    class _ToRaisesModel(MpsModel):
        def to(self, device):
            raise RuntimeError("to boom")

    with fake_torch():
        lb = LocalBackend(
            model=_ToRaisesModel(), tokenizer=FakeTokenizer(), lock=Lock(),
            gen_lock=Lock(), device="mps", tokenize_cache={},
        )
        result = lb.generate("hello", 10, 0.7, 0.9, 50, 1.0)
        assert result["tokens_generated"] == 2


def test_local_backend_generate_kv_capture():
    class KVCaptureModel(NpMockModel):
        past_key_values = "captured"

    with fake_torch():
        lb = LocalBackend(
            model=KVCaptureModel(), tokenizer=FakeTokenizer(), lock=Lock(),
            gen_lock=Lock(), device="cpu", tokenize_cache={},
        )
        try:
            result = lb.generate("hello", 10, 0.7, 0.9, 50, 1.0, session_id="capture-sess")
            assert result["tokens_generated"] == 2
            assert SESSION_KV_CACHE._caches["capture-sess"][1] == "captured"
        finally:
            SESSION_KV_CACHE.clear("capture-sess")


def test_local_backend_generate_kv_capture_store_error():
    class KVCaptureModel(NpMockModel):
        _past_key_values = "captured"

    with fake_torch():
        lb = LocalBackend(
            model=KVCaptureModel(), tokenizer=FakeTokenizer(), lock=Lock(),
            gen_lock=Lock(), device="cpu", tokenize_cache={},
        )
        with patch.object(SESSION_KV_CACHE, "store", side_effect=RuntimeError("kv boom")):
            result = lb.generate("hello", 10, 0.7, 0.9, 50, 1.0, session_id="capture-sess")
        assert result["tokens_generated"] == 2


class _CancelCriteriaStreamModel:
    def generate(self, **kwargs):
        streamer = kwargs.get("streamer")
        sc = kwargs.get("stopping_criteria") or []
        for t in ["a", "b"]:
            if any(c([], 0) for c in sc):
                break
            streamer.put(t)
        streamer.end()
        return "dummy"


def test_local_backend_generate_stream_cancel_criteria_stops():
    with fake_torch(), fake_transformers():
        lb = LocalBackend(
            model=_CancelCriteriaStreamModel(), tokenizer=FakeTokenizer(),
            lock=Lock(), gen_lock=Lock(), device="cpu", tokenize_cache={},
        )
        ce = threading.Event()
        ce.set()
        assert list(lb.generate_stream(
            "hello", 10, 0.7, 0.9, 50, 1.0, cancel_event=ce,
        )) == []


def test_local_backend_generate_stream():
    with fake_torch(), fake_transformers():
        lb = LocalBackend(
            model=StreamModel(tokens=["tok1", "tok2"]), tokenizer=FakeTokenizer(),
            lock=Lock(), gen_lock=Lock(), device="cpu", tokenize_cache={},
        )
        assert list(lb.generate_stream("hello", 10, 0.7, 0.9, 50, 1.0)) == ["tok1", "tok2"]


def test_local_backend_generate_stream_cancel_criteria():
    with fake_torch(), fake_transformers():
        lb = LocalBackend(
            model=StreamModel(tokens=["tok1"]), tokenizer=FakeTokenizer(),
            lock=Lock(), gen_lock=Lock(), device="cpu", tokenize_cache={},
        )
        ce = threading.Event()
        assert list(lb.generate_stream("hello", 10, 0.7, 0.9, 50, 1.0, cancel_event=ce)) == ["tok1"]


def test_local_backend_generate_stream_session():
    with fake_torch(), fake_transformers():
        lb = LocalBackend(
            model=StreamModel(tokens=["tok1"]), tokenizer=FakeTokenizer(),
            lock=Lock(), gen_lock=Lock(), device="cpu", tokenize_cache={},
        )
        SESSION_KV_CACHE.store("stream-sess", [1, 2, 3], "pkv")
        try:
            assert list(lb.generate_stream(
                "hello", 10, 0.7, 0.9, 50, 1.0, session_id="stream-sess",
            )) == ["tok1"]
        finally:
            SESSION_KV_CACHE.clear("stream-sess")


def test_local_backend_generate_stream_slow_poll():
    with fake_torch(), fake_transformers():
        lb = LocalBackend(
            model=StreamModel(tokens=["tok"], delay=0.05), tokenizer=FakeTokenizer(),
            lock=Lock(), gen_lock=Lock(), device="cpu", tokenize_cache={},
        )
        assert list(lb.generate_stream("hello", 10, 0.7, 0.9, 50, 1.0)) == ["tok"]


def test_local_backend_generate_stream_error():
    with fake_torch(), fake_transformers():
        lb = LocalBackend(
            model=FailStreamModel(), tokenizer=FakeTokenizer(),
            lock=Lock(), gen_lock=Lock(), device="cpu", tokenize_cache={},
        )
        with pytest.raises(RuntimeError):
            list(lb.generate_stream("hello", 10, 0.7, 0.9, 50, 1.0))


# ── _cancelable_gen ───────────────────────────────────────────────────


def test_cancelable_gen():
    gen = (x for x in [1, 2, 3])
    ce = threading.Event()
    assert list(_cancelable_gen(gen, ce)) == [1, 2, 3]
    ce2 = threading.Event()
    ce2.set()
    assert list(_cancelable_gen((x for x in [1, 2, 3]), ce2)) == []


# ── ModelServer: device / backend / lifecycle ─────────────────────────


def test_resolved_device_sentinels():
    s = ModelServer(model=None, tokenizer=None, enable_warmup=False)
    assert s._device == "guard"
    assert s._resolved_device == "cpu"


def test_resolved_device_real():
    with fake_torch():
        s = ModelServer(model=NpMockModel(), tokenizer=FakeTokenizer(), enable_warmup=False)
        assert s._resolved_device == "cpu"


def test_check_device_parameters_path():
    with fake_torch():
        class ParamModel:
            def __init__(self):
                self._param = _Param()
                self._param.device = "cuda"

            def parameters(self):
                yield self._param

        s = ModelServer(model=ParamModel(), tokenizer=FakeTokenizer(), enable_warmup=False)
        assert s._device == "cuda"


def test_check_device_exception_path():
    class BadModel:
        def parameters(self):
            raise RuntimeError("boom")

    with fake_torch():
        s = ModelServer(model=BadModel(), tokenizer=FakeTokenizer(), enable_warmup=False)
        assert s._device == "unknown"


def test_guard_backend_wiring_and_select():
    guard = FakeGuard(alive=True)
    s = ModelServer(
        model=NpMockModel(), tokenizer=FakeTokenizer(),
        process_guard=guard, enable_warmup=False,
    )
    assert s._guard_backend is not None
    assert guard._crash is not None
    assert guard._restart is not None
    assert isinstance(s._select_backend(), GuardBackend)


def test_select_backend_guard_dead_falls_back():
    guard = FakeGuard(alive=False)
    s = ModelServer(
        model=NpMockModel(), tokenizer=FakeTokenizer(),
        process_guard=guard, enable_warmup=False,
    )
    assert isinstance(s._select_backend(), LocalBackend)


def test_swap_model():
    with fake_torch():
        s = _make_server()
        old = s._model_ref
        new = NpMockModel()
        s.swap_model(new)
        assert s._model_ref is new
        assert s._local_backend._model_ref is new
        assert s.status == ModelStatus.READY
        assert old is not new


def test_swap_model_rewarms_when_enabled():
    class _T:
        def __init__(self, target, daemon=False):
            captured["target"] = target
            captured["daemon"] = daemon

        def start(self):
            captured["started"] = True

    captured = {}
    with fake_torch(), patch.object(model_server, "Thread", _T):
        s = ModelServer(model=NpMockModel(), tokenizer=FakeTokenizer(), enable_warmup=True)
        s.swap_model(NpMockModel())
    assert captured.get("started") is True
    assert captured.get("daemon") is True


def test_drop_model_ref():
    s = _make_server()
    s.drop_model_ref()
    assert s._model_ref is None
    assert s._device == "guard"


def test_cleanup_kv_cache_no_model():
    s = ModelServer(model=None, tokenizer=None, enable_warmup=False)
    s._cleanup_kv_cache()


def test_cleanup_kv_cache_model():
    class _Resettable:
        def reset(self):
            self.called = True

    class _Clearable:
        def clear(self):
            self.called = True

    class CacheModel:
        def __init__(self):
            self.past_key_values = "x"
            self._past_key_values = "y"
            self.kv_cache = _Resettable()
            self._cache = _Clearable()

    with fake_torch():
        s = ModelServer(model=CacheModel(), tokenizer=FakeTokenizer(), enable_warmup=False)
        s._cleanup_kv_cache()
        assert s._model_ref.past_key_values is None


def test_cleanup_kv_cache_exception():
    class BadCacheModel:
        @property
        def past_key_values(self):
            return "x"

        @past_key_values.setter
        def past_key_values(self, value):
            raise RuntimeError("clear boom")

    with fake_torch():
        s = ModelServer(model=BadCacheModel(), tokenizer=FakeTokenizer(), enable_warmup=False)
        s._cleanup_kv_cache()


def test_cleanup_kv_cache_reset_exception():
    class _RaisesReset:
        def reset(self):
            raise RuntimeError("reset boom")

    class ResetFailModel:
        past_key_values = "x"
        kv_cache = _RaisesReset()

    with fake_torch():
        s = ModelServer(model=ResetFailModel(), tokenizer=FakeTokenizer(), enable_warmup=False)
        s._cleanup_kv_cache()


def test_warmup_thread_started():
    captured = {}

    class _T:
        def __init__(self, target, daemon=False):
            captured["target"] = target
            captured["daemon"] = daemon

        def start(self):
            captured["started"] = True

    with patch.object(model_server, "Thread", _T):
        ModelServer(model=NpMockModel(), tokenizer=FakeTokenizer(), enable_warmup=True)
    assert captured.get("started") is True
    assert captured.get("daemon") is True


def test_warmup_failure_logs_warning():
    s = ModelServer(model=NpMockModel(), tokenizer=FakeTokenizer(), enable_warmup=False)
    with patch.object(LocalBackend, "generate", side_effect=RuntimeError("boom")):
        with patch("domains.infrastructure.model_server.logger") as mock_log:
            s._run_warmup()
    assert s._warmup_error == "RuntimeError: boom"
    mock_log.warning.assert_called()


def test_warmup_failure_logs_debug_for_missing_module():
    s = ModelServer(model=NpMockModel(), tokenizer=FakeTokenizer(), enable_warmup=False)
    with patch.object(
        LocalBackend, "generate",
        side_effect=ModuleNotFoundError("No module named 'torch'"),
    ):
        with patch("domains.infrastructure.model_server.logger") as mock_log:
            s._run_warmup()
    mock_log.debug.assert_called()


def test_on_cb_state_change_exception():
    with patch(
        "domains.infrastructure.event_bus.get_event_bus",
        side_effect=Exception("bus down"),
    ):
        s = _make_server()
        s._on_cb_state_change(CircuitBreakerState.CLOSED, CircuitBreakerState.OPEN)


def test_status_degraded_when_open():
    s = _make_server()
    cb = s._circuit_breaker
    for _ in range(3):
        cb.record_failure()
    assert cb.state == CircuitBreakerState.OPEN
    assert s.status == ModelStatus.DEGRADED


def test_warmup_applies_compile():
    with fake_torch():
        s = ModelServer(
            model=BigModel(), tokenizer=FakeTokenizer(), model_id="big",
            enable_warmup=False,
        )
        with patch.object(LocalBackend, "generate", side_effect=_fast_generate):
            s._run_warmup()
        assert s._warmup_completed is True


# ── ModelServer: semaphores / tokenize ────────────────────────────────


def test_get_read_semaphore_no_concurrency():
    s = _make_server()
    s._max_concurrent = None
    coro = s._get_read_semaphore()
    with pytest.raises(StopIteration) as ei:
        coro.send(None)
    assert ei.value.value is None


async def test_get_read_semaphore_no_running_loop():
    s = _make_server()
    with patch.object(
        model_server.asyncio, "get_running_loop", side_effect=RuntimeError("no loop"),
    ):
        sem = await s._get_read_semaphore()
    assert sem is None


async def test_get_read_semaphore_loop_cache():
    s = _make_server()
    sem1 = await s._get_read_semaphore()
    sem2 = await s._get_read_semaphore()
    assert sem1 is sem2


async def test_run_queue_workers_exception():
    class _BrokenQueue:
        def worker(self):
            async def _w():
                raise RuntimeError("worker boom")

            return _w()

    s = _make_server()
    s._max_concurrent = 1
    await s._run_queue_workers(_BrokenQueue())


async def test_tokenize_ok():
    s = _make_server()
    out = await s.tokenize("hello")
    assert "input_ids" in out


async def test_tokenize_no_tokenizer():
    s = ModelServer(model=NpMockModel(), tokenizer=None, enable_warmup=False)
    with pytest.raises(RuntimeError):
        await s.tokenize("hello")


async def test_tokenize_timeout():
    s = _make_server()
    s._generate_timeout = 0.01
    sem = await s._get_read_semaphore()
    for _ in range(s._max_readers):
        await sem.acquire()
    with pytest.raises(TimeoutError):
        await s.tokenize("hello")


# ── ModelServer: generate ─────────────────────────────────────────────


async def test_ensure_queue_loop_none_rebind():
    s = _make_server()
    q1 = await s._ensure_queue()
    s._queue_loop = None
    q2 = await s._ensure_queue()
    assert q2 is q1


async def test_ensure_queue_loop_mismatch_rebuild():
    s = _make_server()
    q1 = await s._ensure_queue()
    s._queue_loop = object()
    q2 = await s._ensure_queue()
    assert q2 is not q1
    assert s._queue_loop is not None


async def test_get_metrics_snapshot():
    s = _make_server()
    base = s.get_metrics_snapshot()
    assert base["model_id"] is not None
    assert base["status"] == "ready"
    assert "queue_depth_total" not in base
    await s._ensure_queue()
    base2 = s.get_metrics_snapshot()
    assert "queue_depth_total" in base2
    assert "queue_avg_wait_ms" in base2


async def test_generate_circuit_breaker_open():
    s = _make_server()
    cb = s._circuit_breaker
    for _ in range(3):
        cb.record_failure()
    with pytest.raises(RuntimeError, match="Circuit breaker open"):
        await s.generate("hello")


async def test_generate_backend_error():
    s = _make_server()
    with patch.object(ModelServer, "_generate_sync", side_effect=RuntimeError("gen boom")):
        with pytest.raises(RuntimeError):
            await s.generate("hello")
    assert s.metrics.requests_failed >= 1
    assert s._circuit_breaker._failure_count >= 1


async def test_generate_queue_full():
    s = _make_server()
    q = await s._ensure_queue()
    q._max_queue = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(RuntimeError, match="Queue full"):
            await s.generate("hello")
    assert s.metrics.requests_failed == 1


def test_generate_sync_real_local_path():
    with fake_torch():
        s = _make_server()
        out = s._generate_sync("hello", 10, 0.7, 0.9, 50, 1.0)
        assert "text" in out
        assert out["tokens_generated"] == 2


async def test_generate_pre_hook_failure():
    s = _make_server()

    def bad_hook():
        raise RuntimeError("pre hook boom")

    s.add_pre_generate_hook(bad_hook)
    with patch.object(ModelServer, "_generate_sync", autospec=True, side_effect=_fast_generate):
        result = await s.generate("hello")
    assert result["tokens_generated"] == 5


async def test_generate_timeout():
    s = _make_server()
    s._generate_timeout = 0.05

    def _slow(*a, **k):
        time.sleep(1)
        return {"text": "x", "tokens_generated": 1}

    with patch.object(ModelServer, "_generate_sync", autospec=True, side_effect=_slow):
        with pytest.raises(TimeoutError):
            await s.generate("hello")
    assert s.metrics.requests_timed_out == 1


def test_generate_stream_sync_guard():
    guard = FakeGuard(stream_tokens=["tok1", "tok2"])
    s = ModelServer(
        model=NpMockModel(), tokenizer=FakeTokenizer(),
        process_guard=guard, enable_warmup=False,
    )
    streamer = s.generate_stream_sync("hello")
    out = []
    while True:
        item = streamer.text_queue.get()
        if item is streamer.stop_signal:
            break
        out.append(item)
    assert out == ["tok1", "tok2"]


def test_streamer_pump_error():
    def _bad_gen():
        yield "tok"
        raise RuntimeError("pump boom")

    streamer = ModelServer._wrap_generator_as_streamer(_bad_gen())
    out = []
    while True:
        item = streamer.text_queue.get()
        if item is streamer.stop_signal:
            break
        out.append(item)
    assert out == ["tok"]


def test_streamer_pump_stop_iteration():
    class _IterRaises:
        def __iter__(self):
            raise StopIteration

    streamer = ModelServer._wrap_generator_as_streamer(_IterRaises())
    assert streamer.text_queue.get() is streamer.stop_signal


# ── ModelServer: generate_stream ──────────────────────────────────────


async def test_generate_stream_cb_open():
    s = _make_server()
    cb = s._circuit_breaker
    for _ in range(3):
        cb.record_failure()
    ag = s.generate_stream("hello")
    with pytest.raises(RuntimeError):
        await ag.__anext__()


async def test_generate_stream_pre_hook_failure():
    guard = FakeGuard(stream_tokens=["a"])
    s = ModelServer(
        model=NpMockModel(), tokenizer=FakeTokenizer(),
        process_guard=guard, enable_warmup=False,
    )

    def bad_hook():
        raise RuntimeError("pre hook boom")

    s.add_pre_generate_hook(bad_hook)
    assert [t async for t in s.generate_stream("hello")] == ["a"]


async def test_generate_stream_queue_full():
    s = _make_server()
    q = await s._ensure_queue()
    q._max_queue = 0
    ag = s.generate_stream("hello")
    with pytest.raises(RuntimeError, match="Queue full"):
        await ag.__anext__()
    assert s.metrics.requests_failed == 1


async def test_generate_stream_success_cb():
    guard = FakeGuard(stream_tokens=["a", "b"])
    s = ModelServer(
        model=NpMockModel(), tokenizer=FakeTokenizer(),
        process_guard=guard, enable_warmup=False,
    )
    assert [t async for t in s.generate_stream("hello")] == ["a", "b"]
    assert s._circuit_breaker.state == CircuitBreakerState.CLOSED


async def test_generate_stream_timeout():
    guard = FakeGuard(stream_error=asyncio.TimeoutError("stream timeout"))
    s = ModelServer(
        model=NpMockModel(), tokenizer=FakeTokenizer(),
        process_guard=guard, enable_warmup=False,
    )
    ag = s.generate_stream("hello")
    with pytest.raises(asyncio.TimeoutError):
        async for _ in ag:
            pass
    assert s.metrics.requests_timed_out == 1


async def test_generate_stream_backend_error():
    guard = FakeGuard(stream_error=RuntimeError("backend boom"))
    s = ModelServer(
        model=NpMockModel(), tokenizer=FakeTokenizer(),
        process_guard=guard, enable_warmup=False,
    )
    ag = s.generate_stream("hello")
    with pytest.raises(RuntimeError):
        async for _ in ag:
            pass
    assert s.metrics.requests_failed == 1
    assert s._circuit_breaker._failure_count >= 1


async def test_generate_stream_post_hook_failure():
    guard = FakeGuard(stream_tokens=["tok"])
    s = ModelServer(
        model=NpMockModel(), tokenizer=FakeTokenizer(),
        process_guard=guard, enable_warmup=False,
    )

    def bad_hook():
        raise RuntimeError("post hook boom")

    s.add_post_generate_hook(bad_hook)
    assert [t async for t in s.generate_stream("hello")] == ["tok"]


async def test_generate_stream_empty_queue_poll():
    def _slow_gen():
        time.sleep(0.05)
        yield "a"
        time.sleep(0.05)
        yield "b"

    guard = FakeGuard(stream_tokens=None)
    guard.generate_stream = lambda **kwargs: _slow_gen()
    s = ModelServer(
        model=NpMockModel(), tokenizer=FakeTokenizer(),
        process_guard=guard, enable_warmup=False,
    )
    assert [t async for t in s.generate_stream("hello")] == ["a", "b"]


async def test_generate_stream_generator_exit():
    def _blocking_gen():
        yield "tok"
        time.sleep(60)

    guard = FakeGuard(stream_tokens=None)
    guard.generate_stream = lambda **kwargs: _blocking_gen()
    s = ModelServer(
        model=NpMockModel(), tokenizer=FakeTokenizer(),
        process_guard=guard, enable_warmup=False,
    )

    class _PumpThread(threading.Thread):
        def join(self, timeout=None):
            return None

    ce = threading.Event()
    with patch.object(model_server, "Thread", _PumpThread):
        ag = s.generate_stream("hello", cancel_event=ce)
        assert await ag.__anext__() == "tok"
        await ag.aclose()
    assert ce.is_set()


# ── ModelServer: error handling ───────────────────────────────────────


def test_on_generation_error_hook_failure():
    s = _make_server()

    def bad_hook(error):
        raise RuntimeError("error hook boom")

    s.add_on_error_hook(bad_hook)
    s._on_generation_error(RuntimeError("gen error"))
    assert s.status == ModelStatus.DEGRADED
