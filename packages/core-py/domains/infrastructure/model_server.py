"""
Composable model serving infrastructure with request lifecycle management.

Provides ModelServer — a wrapper around any HuggingFace-compatible model
that handles concurrency control, timeouts, error recovery, and metrics.

Architecture::

    Request → ModelServer.generate() / generate_stream()
                ├── select backend (GuardBackend or LocalBackend)
                ├── acquire semaphore (serializes concurrent access)
                ├── pre-generation hook (OOM check, cache warm)
                ├── backend.generate() / backend.generate_stream()
                ├── post-generation hook (KV cache reset)
                └── release semaphore

Backends:
    GuardBackend  — delegates to ProcessGuard subprocess (crash-isolated).
                    Falls back to LocalBackend when subprocess is dead.
    LocalBackend  — direct model.generate().
"""

import asyncio
import heapq
import logging
import time
import gc
import os
import functools
from threading import Lock, Thread, Event
from typing import Generator as GeneratorType

def _ensure_torch() -> bool:
    """Check if real PyTorch is available (needed for HF model inference)."""
    try:
        import torch as _real_torch  # noqa: F401
        return True
    except ImportError:
        return False
from typing import Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum, IntEnum

from domains.infrastructure.structured_log import StructuredLogger, LogContext

logger = StructuredLogger("slo.infrastructure.model_server")


# ---------------------------------------------------------------------------
# Priority levels for request scheduling
# ---------------------------------------------------------------------------

class Priority(IntEnum):
    """Request priority — lower number = higher priority (dequeued first)."""
    HIGH = 0    # interactive chat
    MEDIUM = 1  # generate / inference
    LOW = 2     # batch / background


@dataclass
class QueueMetrics:
    """Snapshot of priority queue state (thread-safe)."""
    depth_high: int = 0
    depth_medium: int = 0
    depth_low: int = 0
    total_depth: int = 0
    served: int = 0
    timed_out: int = 0
    avg_wait_ms: float = 0.0
    max_wait_ms: float = 0.0


# ---------------------------------------------------------------------------
# Priority request queue — async, priority-aware, with metrics
# ---------------------------------------------------------------------------

@dataclass(order=True)
class _QueueItem:
    """Item in the priority queue.  ``order=True`` makes ``heapq`` sort by
    ``(priority, enqueue_order)`` — lower priority first, FIFO within same
    priority."""
    priority: int          # Priority value (0=HIGH, 1=MEDIUM, 2=LOW)
    enqueue_order: int     # insertion order for FIFO within priority level
    coro: Any = field(compare=False)                # awaitable to execute
    future: asyncio.Future = field(compare=False)   # resolves when coro is done
    enqueued_at: float = field(compare=False, default_factory=time.time)
    request_id: str = field(compare=False, default="")


class PriorityRequestQueue:
    """Async request queue with 3 priority levels and bounded concurrency.

    Workers pull the highest-priority item (FIFO within priority level)
    and execute it, up to ``max_concurrent`` in-flight.
    """

    def __init__(self, max_concurrent: int = 2, max_queue: int = 128):
        self._max_concurrent = max_concurrent
        self._max_queue = max_queue
        self._heap: list[_QueueItem] = []
        self._order_counter = 0
        self._in_flight = 0
        self._lock = asyncio.Lock()
        self._wake_event = asyncio.Event()

        # Metrics
        self._served = 0
        self._total_wait = 0.0
        self._max_wait_s = 0.0
        self._metrics_lock = Lock()

    # --- Public API ---

    async def acquire(
        self,
        priority: Priority = Priority.MEDIUM,
        request_id: str = "",
    ) -> Callable[[], None]:
        """Reserve a slot for long-lived work (e.g. streaming).

        Returns a ``release()`` callable that the caller *must* invoke when
        the work completes (or is aborted).  Until release, the slot counts
        against ``max_concurrent``, blocking other submissions.

        Unlike :meth:`submit`, ``acquire`` does **not** execute a coroutine
        inside the queue worker — it only grants permission to proceed.
        The caller runs their work externally and calls ``release()``.
        """
        loop = asyncio.get_running_loop()
        grant: asyncio.Future = loop.create_future()

        async with self._lock:
            if len(self._heap) >= self._max_queue:
                logger.warning("Queue full on acquire", max_queue=self._max_queue,
                               request_id=request_id)
                raise RuntimeError(f"Queue full ({self._max_queue} items)")
            item = _QueueItem(
                priority=priority.value,
                enqueue_order=self._order_counter,
                coro=None,  # marker — no coroutine to execute
                future=grant,
                request_id=request_id or f"acq-{self._order_counter}",
            )
            self._order_counter += 1
            heapq.heappush(self._heap, item)

        logger.debug("Acquire enqueued", request_id=item.request_id,
                     priority=priority.name, queue_depth=len(self._heap))
        self._wake_event.set()

        released = False

        def _release() -> None:
            nonlocal released
            if not released:
                released = True
                self._in_flight -= 1
                self._wake_event.set()

        try:
            await grant  # blocks until worker pops this marker
        except (asyncio.CancelledError, Exception, GeneratorExit):
            if grant.done() and not grant.cancelled():
                # Worker already popped it — release the slot
                _release()
            else:
                # Marker still in heap — remove it
                async with self._lock:
                    self._heap = [x for x in self._heap if x.future is not grant]
                    heapq.heapify(self._heap)
            raise

        return _release

    async def submit(
        self,
        coro: Any,
        priority: Priority = Priority.MEDIUM,
        request_id: str = "",
    ) -> Any:
        """Submit an awaitable for execution, returning its result."""
        async with self._lock:
            if len(self._heap) >= self._max_queue:
                logger.warning("Queue full", max_queue=self._max_queue, request_id=request_id)
                raise RuntimeError(f"Queue full ({self._max_queue} items)")
            future: asyncio.Future = asyncio.get_running_loop().create_future()
            item = _QueueItem(
                priority=priority.value,
                enqueue_order=self._order_counter,
                coro=coro,
                future=future,
                request_id=request_id or f"req-{self._order_counter}",
            )
            self._order_counter += 1
            heapq.heappush(self._heap, item)

        logger.debug("Enqueued", request_id=item.request_id, priority=priority.name,
                      queue_depth=len(self._heap))
        self._wake_event.set()
        return await future

    def _pop(self) -> Optional[_QueueItem]:
        """Pop highest-priority item (caller must hold ``_lock``)."""
        return heapq.heappop(self._heap) if self._heap else None

    async def depth(self) -> list[int]:
        async with self._lock:
            d = [0, 0, 0]
            for item in self._heap:
                if item.priority < 3:
                    d[item.priority] += 1
            return d

    @property
    def in_flight(self) -> int:
        return self._in_flight

    def metrics_snapshot(self) -> QueueMetrics:
        with self._metrics_lock:
            depths = [0, 0, 0]
            for item in self._heap:
                if item.priority < 3:
                    depths[item.priority] += 1
            avg = (self._total_wait / max(self._served, 1)) * 1000
            return QueueMetrics(
                depth_high=depths[0], depth_medium=depths[1], depth_low=depths[2],
                total_depth=sum(depths), served=self._served,
                avg_wait_ms=avg, max_wait_ms=self._max_wait_s * 1000,
            )

    # --- Worker loop ---

    async def worker(self) -> None:
        """Pop and execute items (runs forever)."""
        while True:
            await self._wake_event.wait()

            item: Optional[_QueueItem] = None
            async with self._lock:
                if self._in_flight < self._max_concurrent and self._heap:
                    item = self._pop()

            if item is None:
                async with self._lock:
                    if not self._heap:
                        self._wake_event.clear()
                await asyncio.sleep(0.01)
                continue

            wait_s = time.time() - item.enqueued_at
            with self._metrics_lock:
                self._served += 1
                self._total_wait += wait_s
                if wait_s > self._max_wait_s:
                    self._max_wait_s = wait_s

            logger.debug("Dequeued", request_id=item.request_id,
                          wait_ms=round(wait_s * 1000, 1),
                          priority=Priority(item.priority).name)

            # Reservation marker — grant the slot, let caller manage in_flight
            if item.coro is None:
                self._in_flight += 1
                item.future.set_result(None)
                continue

            # Normal work item
            self._in_flight += 1
            try:
                result = await item.coro
                item.future.set_result(result)
            except Exception as e:
                if not item.future.done():
                    item.future.set_exception(e)
            finally:
                self._in_flight -= 1
                self._wake_event.set()


class SessionKVCache:
    """Thread-safe per-session KV cache for incremental cross-turn decoding.

    Stores ``(token_ids, past_key_values)`` per session so that follow-up
    messages can skip re-encoding the shared prompt prefix.

    Cache entries expire after ``ttl`` seconds and at most ``max_sessions``
    entries are kept (LRU eviction).
    """

    def __init__(self, max_sessions: int = 20, ttl: float = 600.0):
        self._caches: dict[str, Any] = {}
        self._max_sessions = max_sessions
        self._ttl = ttl
        self._lock = Lock()

    def get(self, session_id: str, current_ids: list[int]):
        """Return the cached ``past_key_values`` if ``current_ids`` shares a
        prefix with the stored token IDs, else ``None``.

        Also returns the prefix length so the caller can build the correct
        attention mask.
        """
        with self._lock:
            entry = self._caches.get(session_id)
        if entry is None:
            return None, 0
        cached_ids, cached_pkv, _ = entry
        prefix_len = 0
        for a, b in zip(cached_ids, current_ids):
            if a != b:
                break
            prefix_len += 1
        if prefix_len == 0:
            return None, 0
        return cached_pkv, prefix_len

    def store(self, session_id: str, token_ids: list[int], past_key_values: Any) -> None:
        """Store ``past_key_values`` keyed by session + token IDs."""
        with self._lock:
            self._evict_expired()
            if len(self._caches) >= self._max_sessions:
                oldest = min(self._caches, key=lambda k: self._caches[k][2])
                del self._caches[oldest]
            self._caches[session_id] = (token_ids, past_key_values, time.time())

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._caches.pop(session_id, None)

    def _evict_expired(self) -> None:
        now = time.time()
        stale = [k for k, v in self._caches.items() if now - v[2] > self._ttl]
        for k in stale:
            del self._caches[k]

    def evict_expired(self) -> None:
        with self._lock:
            self._evict_expired()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._caches)

    def stats(self) -> dict:
        with self._lock:
            return {
                "entries": len(self._caches),
                "max_sessions": self._max_sessions,
                "ttl_seconds": self._ttl,
            }


def _optimize_cpu_threads() -> None:
    """Set optimal compute / BLAS thread counts from ResourceManager."""
    from domains.infrastructure.resource_manager import get_resource_manager
    rm = get_resource_manager()
    rm.apply_blas_env()
    rm.apply_compute_limits()
    logger.debug(
        "Compute threads: %d  I/O threads: %d  (topology: %s)",
        rm.compute_threads, rm.io_threads, rm.topology.summary(),
    )


def _torch_compile_model(model, model_id: str) -> Any:
    """Apply ``torch.compile`` to the model.  Returns compiled model or original."""
    if not _ensure_torch():
        return model
    import torch
    if not hasattr(torch, "compile") or not callable(torch.compile):
        return model
    # Skip models too small to benefit (compile overhead > gain)
    try:
        n_params = sum(p.numel() for p in model.parameters())
        if n_params < 10_000_000:  # <10M params — not worth it
            logger.debug("Skipping torch.compile for %s (%d params)", model_id, n_params)
            return model
    except Exception:
        pass
    try:
        compiled = torch.compile(model, backend="aot_eager" if _is_intel_mac() else "inductor")
        logger.info("torch.compile applied to %s", model_id, extra={"tag": "MODEL"})
        return compiled
    except Exception as e:
        logger.warning("torch.compile failed for %s: %s", model_id, e, extra={"tag": "MODEL"})
        return model


def _inference_mode_generate(model, gen_kwargs: dict) -> Any:
    """Generate under ``torch.inference_mode()`` for ~5-15% speedup over no_grad."""
    if not _ensure_torch():
        return model.generate(**gen_kwargs)
    import torch
    with torch.inference_mode():
        return model.generate(**gen_kwargs)


def _is_intel_mac() -> bool:
    try:
        import platform
        return platform.system() == "Darwin" and platform.machine() == "x86_64"
    except Exception:
        return False


class ModelStatus(Enum):
    UNINITIALIZED = "uninitialized"
    LOADING = "loading"
    READY = "ready"
    DEGRADED = "degraded"
    ERROR = "error"
    UNLOADED = "unloaded"


@dataclass
class ModelMetrics:
    """Metrics collected per-model-session."""
    requests_total: int = 0
    requests_completed: int = 0
    requests_failed: int = 0
    requests_timed_out: int = 0
    total_generation_time_ms: float = 0.0
    max_generation_time_ms: float = 0.0
    min_generation_time_ms: float = float("inf")
    tokens_generated_total: int = 0
    last_generation_time_ms: float = 0.0
    last_error: Optional[str] = None
    last_error_at: Optional[float] = None
    consecutive_failures: int = 0

    def record_success(self, elapsed_ms: float, tokens: int) -> None:
        self.requests_completed += 1
        self.total_generation_time_ms += elapsed_ms
        self.max_generation_time_ms = max(self.max_generation_time_ms, elapsed_ms)
        self.min_generation_time_ms = min(self.min_generation_time_ms, elapsed_ms)
        self.last_generation_time_ms = elapsed_ms
        self.tokens_generated_total += tokens
        self.consecutive_failures = 0

    def record_failure(self, error: str) -> None:
        self.requests_failed += 1
        self.last_error = error
        self.last_error_at = time.time()
        self.consecutive_failures += 1

    def record_timeout(self) -> None:
        self.requests_timed_out += 1
        self.consecutive_failures += 1

    @property
    def avg_generation_time_ms(self) -> float:
        if self.requests_completed == 0:
            return 0.0
        return self.total_generation_time_ms / self.requests_completed

    @property
    def error_rate(self) -> float:
        if self.requests_total == 0:
            return 0.0
        return self.requests_failed / self.requests_total

    def snapshot(self) -> dict:
        return {
            "requests_total": self.requests_total,
            "requests_completed": self.requests_completed,
            "requests_failed": self.requests_failed,
            "requests_timed_out": self.requests_timed_out,
            "consecutive_failures": self.consecutive_failures,
            "avg_generation_time_ms": round(self.avg_generation_time_ms, 1),
            "max_generation_time_ms": round(self.max_generation_time_ms, 1),
            "min_generation_time_ms": round(self.min_generation_time_ms, 1) if self.min_generation_time_ms != float("inf") else 0.0,
            "last_generation_time_ms": round(self.last_generation_time_ms, 1),
            "tokens_generated_total": self.tokens_generated_total,
            "last_error": self.last_error,
            "error_rate": round(self.error_rate, 4),
        }


class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Simple circuit breaker to stop sending requests to a failing model."""
    failure_threshold: int = 3
    recovery_timeout: float = 30.0
    _state: CircuitBreakerState = CircuitBreakerState.CLOSED
    _failure_count: int = 0
    _last_failure_at: float = 0.0
    _lock: Lock = field(default_factory=Lock)

    @property
    def state(self) -> CircuitBreakerState:
        with self._lock:
            if self._state == CircuitBreakerState.OPEN:
                if time.time() - self._last_failure_at >= self.recovery_timeout:
                    self._state = CircuitBreakerState.HALF_OPEN
            return self._state

    def record_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            self._state = CircuitBreakerState.CLOSED

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_at = time.time()
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitBreakerState.OPEN

    def allow_request(self) -> bool:
        return self.state != CircuitBreakerState.OPEN


def _has_mps() -> bool:
    try:
        from domains.infrastructure.ml_types import mps as ml_mps
        return ml_mps.is_available()
    except ImportError:
        return False


def _mps_oom_recovery() -> None:
    """Clear MPS cache and potentially force CPU fallback."""
    try:
        from domains.infrastructure.ml_types import mps as ml_mps
        if _has_mps():
            ml_mps.empty_cache()
    except Exception:
        pass


def _schedule_gc() -> None:
    """Schedule gc.collect() in a background thread to avoid blocking the event loop."""
    try:
        Thread(target=gc.collect, daemon=True).start()
    except Exception:
        pass


# ── Generate Backends ─────────────────────────────────────────────────────────
#
# Strategy pattern: ModelServer delegates generation to a backend.  Each backend
# owns one generation path (guard subprocess or local model).  Backends are
# selected at request time, so a dead guard transparently falls back to local
# generation without restarting the server.

class GenerateBackend:
    """Base class for token generation backends.

    Subclasses implement ``generate()`` (non-streaming) and
    ``generate_stream()`` (streaming).  ModelServer delegates to whichever
    backend is selected for the current request.
    """

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        **kwargs: Any,
    ) -> dict:
        """Non-streaming generation.  Returns {"text": str, "tokens_generated": int}."""
        raise NotImplementedError

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        cancel_event=None,
        **kwargs: Any,
    ) -> GeneratorType[str, None, dict]:
        """Streaming generation.  Yields tokens, returns final result dict."""
        raise NotImplementedError

    @property
    def alive(self) -> bool:
        """Whether this backend is ready to serve requests."""
        return True


class GuardBackend(GenerateBackend):
    """Delegates generation to a ProcessGuard subprocess.

    When the subprocess is dead (MPS OOM, crash, etc.), ``alive`` returns
    False and the caller falls back to LocalBackend.
    """

    def __init__(self, guard: Any):
        self._guard = guard

    @property
    def alive(self) -> bool:
        return self._guard is not None and self._guard.alive

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        **kwargs: Any,
    ) -> dict:
        safe_kwargs = {k: v for k, v in kwargs.items()
                       if k not in ("input_ids", "attention_mask")}
        start = time.time()
        result = self._guard.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            **safe_kwargs,
        )
        result["elapsed_ms"] = round((time.time() - start) * 1000, 1)
        return result

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        cancel_event=None,
        **kwargs: Any,
    ) -> GeneratorType[str, None, dict]:
        safe_kwargs = {k: v for k, v in kwargs.items()
                       if k not in ("input_ids", "attention_mask")}
        gen = self._guard.generate_stream(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            **safe_kwargs,
        )
        if cancel_event is not None:
            gen = _cancelable_gen(gen, cancel_event)
        return gen


class LocalBackend(GenerateBackend):
    """Direct in-process model.generate()."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        lock: Lock,
        gen_lock: Lock,
        device: str,
        tokenize_cache: dict,
    ):
        self._model_ref = model
        self._tokenizer = tokenizer
        self._lock = lock
        self._gen_lock = gen_lock
        self._device = device
        self._tokenize_cache = tokenize_cache

    @property
    def alive(self) -> bool:
        return self._model_ref is not None

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        from domains.infrastructure.ml_types import no_grad as ml_no_grad

        inputs = _tokenize_cached(self._tokenizer, prompt, self._tokenize_cache)
        input_ids = inputs["input_ids"].to(self._device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)

        # Session KV cache: reuse cached prefix to skip re-encoding
        pkv = None
        if session_id is not None and self._model_ref is not None:
            pkv, prefix_len = SESSION_KV_CACHE.get(session_id, input_ids[0].tolist())
            if pkv is not None:
                logger.debug(
                    "Session[%s]: reuse KV cache for %d of %d tokens",
                    session_id, prefix_len, input_ids.shape[1],
                )
            else:
                logger.debug("Session[%s]: no cached KV — full encode", session_id)

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
        )
        if pkv is not None:
            gen_kwargs["past_key_values"] = pkv
        gen_kwargs.update(kwargs)

        # MPS workaround: model.generate() deadlocks when called from
        # an async context on MPS.  Move to CPU for generation.
        _cpu_fallback = False
        if self._device.startswith("mps"):
            try:
                with self._lock:
                    self._model_ref = self._model_ref.cpu()
                input_ids = input_ids.cpu()
                if attention_mask is not None:
                    attention_mask = attention_mask.cpu()
                gen_kwargs["input_ids"] = input_ids
                if attention_mask is not None:
                    gen_kwargs["attention_mask"] = attention_mask
                _cpu_fallback = True
            except Exception as e:
                logger.debug("OOM-to-CPU fallback failed: %s", e)

        with self._gen_lock:
            output = _inference_mode_generate(self._model_ref, gen_kwargs)

        if _cpu_fallback:
            try:
                with self._lock:
                    self._model_ref = self._model_ref.to(self._device)
            except Exception as e:
                logger.debug("Model restore to device failed: %s", e)

        _mps_empty_cache()

        tokens_generated = output.shape[1] - input_ids.shape[1]
        text = self._tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)

        # Store updated KV cache for next turn
        if session_id is not None and self._model_ref is not None:
            if pkv is not None:
                # Cache was passed — it's been extended in-place during generate
                SESSION_KV_CACHE.store(session_id, input_ids[0].tolist(), pkv)
            else:
                # No cache was passed (first turn or mismatch) — store it now
                # If we can get it from the model's internal state, do so
                try:
                    import torch
                    captured = getattr(self._model_ref, "_past_key_values", None) or \
                               getattr(self._model_ref, "past_key_values", None)
                    if captured is not None:
                        SESSION_KV_CACHE.store(session_id, input_ids[0].tolist(), captured)
                except Exception:
                    pass

        return {"text": text, "tokens_generated": tokens_generated}

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        cancel_event=None,
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> GeneratorType[str, None, dict]:
        """Stream via TextIteratorStreamer in background thread.

        Yields tokens as they arrive.  Returns final result dict when done.

        This is a **sync** generator — ``ModelServer.generate_stream()``
        wraps it in ``run_in_executor`` so it doesn't block the event loop.
        """
        from transformers import TextIteratorStreamer, StoppingCriteria
        import queue

        inputs = _tokenize_cached(self._tokenizer, prompt, self._tokenize_cache)
        input_ids = inputs["input_ids"].to(self._device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)

        # Session KV cache: reuse cached prefix to skip re-encoding
        pkv = None
        if session_id is not None and self._model_ref is not None:
            pkv, prefix_len = SESSION_KV_CACHE.get(session_id, input_ids[0].tolist())
            if pkv is not None:
                logger.debug(
                    "Session[%s]: reuse KV cache for %d of %d tokens",
                    session_id, prefix_len, input_ids.shape[1],
                )

        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, timeout=120.0)

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
            streamer=streamer,
        )
        if pkv is not None:
            gen_kwargs["past_key_values"] = pkv
        gen_kwargs.update(kwargs)

        if cancel_event is not None:
            class _CancelCriteria(StoppingCriteria):
                def __call__(self, input_ids_, scores_, **kwargs):
                    return cancel_event.is_set()
            gen_kwargs.setdefault("stopping_criteria", [])
            gen_kwargs["stopping_criteria"].append(_CancelCriteria())

        _error: list = []
        _pkv_holder: list = [pkv]  # capture the cache ref for after generation

        def _generate_inner():
            try:
                with self._gen_lock:
                    _inference_mode_generate(self._model_ref, gen_kwargs)
            except Exception as e:
                _error.append(e)

        thread = Thread(target=_generate_inner, daemon=True)
        thread.start()

        start = time.time()
        token_count = 0

        while thread.is_alive() or not streamer.text_queue.empty():
            if _error:
                raise _error[0]
            try:
                text = streamer.text_queue.get(timeout=0.02)
            except queue.Empty:
                time.sleep(0.01)
                continue
            if text == streamer.stop_signal:
                break
            if text:
                token_count += 1
                yield text

        thread.join(timeout=30)

        elapsed_ms = (time.time() - start) * 1000

        # Store updated KV cache after generation completes
        if session_id is not None and self._model_ref is not None:
            final_pkv = _pkv_holder[0]
            if final_pkv is not None:
                SESSION_KV_CACHE.store(session_id, input_ids[0].tolist(), final_pkv)

        return {"text": "", "tokens_generated": token_count, "elapsed_ms": elapsed_ms}


def _tokenize_cached(tokenizer, prompt: str, cache: dict) -> dict:
    """Tokenize with LRU cache (64 entries)."""
    if prompt in cache:
        ids, attn = cache[prompt]
        import torch
        result = {"input_ids": torch.tensor([ids])}
        if attn is not None:
            result["attention_mask"] = torch.tensor([attn])
        return result
    import torch
    inputs = tokenizer(prompt, return_tensors="pt")
    attn_list = None
    if inputs.get("attention_mask") is not None:
        attn_list = inputs["attention_mask"][0].tolist()
    cache[prompt] = (inputs["input_ids"][0].tolist(), attn_list)
    if len(cache) > 64:
        cache.pop(next(iter(cache)))
    return inputs


# Module-level session KV cache singleton
SESSION_KV_CACHE = SessionKVCache()


def _mps_empty_cache() -> None:
    """Free MPS cached memory if available."""
    try:
        import torch
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass


def _cancelable_gen(gen, cancel_event):
    """Wrap a generator so it stops when cancel_event is set."""
    for token in gen:
        if cancel_event.is_set():
            break
        yield token


class ModelServer:
    """Composable wrapper around a HuggingFace model for safe concurrent serving.

    Usage::

        server = ModelServer(model, tokenizer, model_id="gpt2")
        result = server.generate(prompt, max_new_tokens=100)
    """

    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        model_id: str = "unknown",
        max_concurrent: Optional[int] = None,
        generate_timeout: float = 120.0,
        enable_circuit_breaker: bool = True,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        process_guard: Optional[Any] = None,
        enable_warmup: bool = True,
        warmup_prompt: str = "Hello",
    ):
        # CPU threading optimization — uses ResourceManager for topology-aware tuning
        _optimize_cpu_threads()

        self.model_id = model_id
        self._tokenizer = tokenizer
        self._model_ref = model
        self._lock = Lock()  # protects model reference swap
        self._gen_lock = Lock()  # serializes model.generate() calls (HF not thread-safe)
        self._process_guard = process_guard  # optional ProcessGuard for bulk gen

        # Generate backends — strategy pattern
        self._guard_backend: Optional[GuardBackend] = (
            GuardBackend(process_guard) if process_guard is not None else None
        )
        self._local_backend = LocalBackend(
            model=model,
            tokenizer=tokenizer,
            lock=self._lock,
            gen_lock=self._gen_lock,
            device="cpu",  # updated by _check_device below
            tokenize_cache={},  # shared with ModelServer
        ) if model is not None else None

        # torch.compile flag — applied after warmup
        self._compiled = False

        # Warmup
        self._enable_warmup = enable_warmup
        self._warmup_prompt = warmup_prompt
        self._warmup_completed = False
        self._warmup_error: Optional[str] = None
        self._warmup_lock = Lock()

        # Priority request queue (replaces per-loop semaphore)
        if max_concurrent is None:
            from domains.infrastructure.resource_manager import get_resource_manager
            max_concurrent = get_resource_manager().concurrent_writes
        self._max_concurrent = max_concurrent
        self._request_queue: Optional[PriorityRequestQueue] = None
        self._queue_task: Optional[asyncio.Task] = None

        # Read/write separation — readers (tokenize, health) can run concurrently;
        # writers (generate) get exclusive access.
        self._read_semaphores: dict[int, asyncio.Semaphore] = {}
        self._max_readers = max_concurrent * 4 if max_concurrent else 4

        # Timeout
        self._generate_timeout = generate_timeout

        # Circuit breaker
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        ) if enable_circuit_breaker else None

        # Wire guard crash callbacks to circuit breaker
        if self._guard_backend is not None and self._circuit_breaker is not None:
            self._process_guard.on_crash(
                lambda wid: self._circuit_breaker.record_failure()
            )
            self._process_guard.on_restart(
                lambda wid: self._circuit_breaker.record_success()
            )

        # Metrics
        self._metrics_lock = Lock()
        self.metrics = ModelMetrics()

        # Status
        self._status = ModelStatus.READY
        self._status_lock = Lock()

        # Lifecycle hooks
        self._pre_generate_hooks: list[Callable[[], None]] = []
        self._post_generate_hooks: list[Callable[[], None]] = []
        self._on_error_hooks: list[Callable[[Exception], None]] = []

        # Tokenizer cache (LRU, 64 entries) — shared with LocalBackend
        self._tokenize_cache: dict = (
            self._local_backend._tokenize_cache if self._local_backend is not None else {}
        )

        # Device tracking
        self._device: Optional[str] = None
        self._check_device()

        # Register default post-generation hook for KV cache cleanup
        self.add_post_generate_hook(self._cleanup_kv_cache)

        # Background warmup
        if self._enable_warmup:
            Thread(target=self._run_warmup, daemon=True).start()

        # Queue workers are lazily started on first generate() call
        # (avoids asyncio loop issues with warmup threads)

    @property
    def _resolved_device(self) -> str:
        """Get a valid PyTorch device string, falling back to ``"cpu"`` when
        the stored device is a sentinel like ``"guard"`` or ``"unknown"``."""
        if self._device in ("guard", "unknown", None):
            return "cpu"
        return self._device

    def _run_warmup(self) -> None:
        """Send a short warmup request to prime the model (JIT, KV cache, etc.).

        Runs in a daemon thread so it never blocks startup. Warmup failures
        are logged but never raised — they don't prevent the model from serving.
        """
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.generate(
                    prompt=self._warmup_prompt,
                    max_new_tokens=5,
                    temperature=0.7,
                ))
            finally:
                loop.close()
            with self._warmup_lock:
                self._warmup_completed = True
            logger.info("ModelServer[%s]: warmup completed", self.model_id, extra={"tag": "MODEL"})
            # Apply torch.compile after warmup (enhances JIT cache)
            if not self._compiled and self._model_ref is not None:
                compiled = _torch_compile_model(self._model_ref, self.model_id)
                if compiled is not self._model_ref:
                    self._compiled = True
                    self._model_ref = compiled
                    if self._local_backend is not None:
                        self._local_backend._model_ref = compiled
        except Exception as e:
            with self._warmup_lock:
                self._warmup_error = f"{type(e).__name__}: {e}"
            # Don't warn for expected failures (missing torch on CPU-only, etc.)
            if "No module named" in str(e):
                logger.debug("ModelServer[%s]: warmup skipped: %s", self.model_id, e)
            else:
                logger.warning("ModelServer[%s]: warmup failed: %s", self.model_id, e, extra={"tag": "MODEL"})

    def _check_device(self) -> None:
        if self._model_ref is None:
            self._device = "guard"
            return
        if not _ensure_torch():
            self._device = "unknown"
            return
        try:
            if hasattr(self._model_ref, "device"):
                self._device = str(self._model_ref.device)
            elif hasattr(self._model_ref, "parameters"):
                p = next(self._model_ref.parameters(), None)
                if p is not None:
                    self._device = str(p.device)
        except Exception:
            self._device = "unknown"
        # Sync device to local backend
        if self._local_backend is not None:
            self._local_backend._device = self._resolved_device

    def _select_backend(self) -> GenerateBackend:
        """Pick the best available backend for the current request.

        Priority: GuardBackend (crash-isolated) > LocalBackend.
        """
        if self._guard_backend is not None and self._guard_backend.alive:
            return self._guard_backend
        return self._local_backend

    def drop_model_ref(self) -> None:
        """Release the in-memory model reference.

        When a ``ProcessGuard`` is active, the in-memory model is not needed
        for inference. Calling this method sets ``_model_ref = None`` so the
        model can be garbage collected, saving main-process memory.
        """
        with self._lock:
            self._model_ref = None
        if self._local_backend is not None:
            self._local_backend._model_ref = None
        self._device = "guard"
        logger.info("ModelServer[%s]: dropped in-memory model ref (guard mode)", self.model_id, extra={"tag": "MODEL"})

    def _cleanup_kv_cache(self) -> None:
        """Clear any KV cache tensors the model may have accumulated."""
        if self._model_ref is None:
            return
        try:
            import torch
            if hasattr(self._model_ref, "past_key_values"):
                self._model_ref.past_key_values = None
            for attr in ("_past_key_values", "kv_cache", "_cache"):
                if hasattr(self._model_ref, attr):
                    try:
                        obj = getattr(self._model_ref, attr)
                        if obj is not None:
                            if hasattr(obj, "reset"):
                                obj.reset()
                            elif hasattr(obj, "clear"):
                                obj.clear()
                    except Exception as e:
                        logger.debug("KV cache clear for %s failed: %s", attr, e)
            _schedule_gc()
        except Exception as e:
            logger.debug("KV cache cleanup failed: %s", e)

    # --- Lifecycle hooks ---

    def add_pre_generate_hook(self, hook: Callable[[], None]) -> None:
        self._pre_generate_hooks.append(hook)

    def add_post_generate_hook(self, hook: Callable[[], None]) -> None:
        self._post_generate_hooks.append(hook)

    def add_on_error_hook(self, hook: Callable[[Exception], None]) -> None:
        self._on_error_hooks.append(hook)

    # --- Status ---

    async def _ensure_queue(self) -> PriorityRequestQueue:
        """Lazily create the priority queue and worker tasks on first call.

        Workers run as asyncio tasks on the current event loop.
        """
        if self._request_queue is not None:
            return self._request_queue
        q = PriorityRequestQueue(
            max_concurrent=self._max_concurrent or 2,
            max_queue=256,
        )
        self._request_queue = q
        # Start workers (one per slot)
        loop = asyncio.get_running_loop()
        self._queue_task = loop.create_task(self._run_queue_workers(q))
        return q

    async def _run_queue_workers(self, q: PriorityRequestQueue) -> None:
        """Launch and manage queue worker tasks."""
        n_workers = min(self._max_concurrent or 2, 8)
        workers = [asyncio.create_task(q.worker()) for _ in range(n_workers)]
        try:
            await asyncio.gather(*workers)
        except Exception:
            pass  # workers run forever

    async def _get_read_semaphore(self) -> Optional[asyncio.Semaphore]:
        """Get a read semaphore for the current event loop.

        Read operations (tokenize, health) share this semaphore and can
        run concurrently up to ``_max_readers``.  Write operations
        (generate) use the exclusive ``_get_semaphore()`` instead.
        """
        if self._max_concurrent is None:
            return None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return None
        loop_id = id(loop)
        if loop_id not in self._read_semaphores:
            self._read_semaphores[loop_id] = asyncio.Semaphore(self._max_readers)
        return self._read_semaphores[loop_id]

    async def tokenize(self, text: str) -> dict:
        """Tokenize text without acquiring the write semaphore.

        Read-only operation — runs concurrently with other tokenizations
        and health checks.  Only blocks during model swap (lock(self._lock)).
        """
        semaphore = await self._get_read_semaphore()
        acquired = False
        if semaphore is not None:
            try:
                await asyncio.wait_for(
                    semaphore.acquire(),
                    timeout=min(self._generate_timeout, 10.0),
                )
                acquired = True
            except asyncio.TimeoutError:
                raise TimeoutError("Tokenize queued too long")
        try:
            with self._lock:
                tok = self._tokenizer
            if tok is None:
                raise RuntimeError("No tokenizer loaded")
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, lambda: tok(text, return_tensors="pt")
            )
        finally:
            if acquired and semaphore is not None:
                semaphore.release()

    @property
    def status(self) -> ModelStatus:
        with self._status_lock:
            if self._circuit_breaker and self._circuit_breaker.state == CircuitBreakerState.OPEN:
                return ModelStatus.DEGRADED
            return self._status

    def set_status(self, status: ModelStatus) -> None:
        with self._status_lock:
            self._status = status

    # --- Metrics ---

    def get_metrics_snapshot(self) -> dict:
        with self._metrics_lock:
            base = self.metrics.snapshot()
        with self._warmup_lock:
            warmup_ok = self._warmup_completed
            warmup_err = self._warmup_error
        base["model_id"] = self.model_id
        base["status"] = self.status.value
        base["device"] = self._device or "unknown"
        base["circuit_breaker"] = self._circuit_breaker.state.value if self._circuit_breaker else "disabled"
        base["warmup_completed"] = warmup_ok
        base["warmup_error"] = warmup_err
        # Priority queue metrics
        if self._request_queue is not None:
            qm = self._request_queue.metrics_snapshot()
            base["queue_depth_total"] = qm.total_depth
            base["queue_depth_high"] = qm.depth_high
            base["queue_depth_medium"] = qm.depth_medium
            base["queue_depth_low"] = qm.depth_low
            base["queue_served"] = qm.served
            base["queue_avg_wait_ms"] = round(qm.avg_wait_ms, 1)
            base["queue_max_wait_ms"] = round(qm.max_wait_ms, 1)
        return base

    # --- Core generation (async) ---

    async def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        session_id: Optional[str] = None,
        priority: Priority = Priority.MEDIUM,
        **kwargs: Any,
    ) -> dict:
        """Generate text with priority-aware request scheduling.

        If ``session_id`` is provided, the KV cache from previous
        generations is reused via :class:`SessionKVCache`, avoiding
        re-encoding the shared prompt prefix.

        Returns::

            {"text": str, "tokens_generated": int, "elapsed_ms": float}

        Raises ``TimeoutError`` if generation exceeds timeout.
        """
        with self._metrics_lock:
            self.metrics.requests_total += 1

        # Circuit breaker check
        if self._circuit_breaker and not self._circuit_breaker.allow_request():
            raise RuntimeError(
                f"Circuit breaker open for {self.model_id} "
                f"(state={self._circuit_breaker.state.value})"
            )

        # Pre-generation hooks (OOM check, cache warm)
        for hook in self._pre_generate_hooks:
            try:
                hook()
            except Exception as e:
                logger.warning("Pre-gen hook failed: %s", e, extra={"tag": "MODEL"})

        # Submit to priority queue
        async def _run() -> dict:
            start = time.time()
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._generate_sync,
                        prompt, max_new_tokens, temperature,
                        top_p, top_k, repetition_penalty,
                        session_id=session_id,
                        **kwargs,
                    ),
                    timeout=self._generate_timeout,
                )
                elapsed_ms = (time.time() - start) * 1000
                tokens = result.get("tokens_generated", 0)
                with self._metrics_lock:
                    self.metrics.record_success(elapsed_ms, tokens)
                if self._circuit_breaker:
                    self._circuit_breaker.record_success()
                logger.info("Generated", model_id=self.model_id, tokens=tokens,
                            elapsed_ms=round(elapsed_ms, 1), session_id=session_id)
                return result
            except asyncio.TimeoutError:
                with self._metrics_lock:
                    self.metrics.record_timeout()
                self._on_generation_error(RuntimeError("Generation timed out"))
                logger.warning("Generation timed out", model_id=self.model_id,
                               timeout=self._generate_timeout)
                raise TimeoutError(
                    f"Generation timed out after {self._generate_timeout}s "
                    f"for {self.model_id}"
                )
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                with self._metrics_lock:
                    self.metrics.record_failure(error_msg)
                if self._circuit_breaker:
                    self._circuit_breaker.record_failure()
                self._on_generation_error(e)
                _mps_oom_recovery()
                raise

        queue = await self._ensure_queue()
        try:
            return await queue.submit(_run(), priority=priority, request_id=f"gen-{id(prompt[:32])}")
        except RuntimeError as e:
            error_msg = f"{type(e).__name__}: {e}"
            with self._metrics_lock:
                self.metrics.record_failure(error_msg)
            if self._circuit_breaker:
                self._circuit_breaker.record_failure()
            raise

    def _generate_sync(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Synchronous generation — delegates to selected backend.

        Backend selection: prefers GuardBackend (crash-isolated), falls back
        to LocalBackend when guard is dead or absent.
        """
        backend = self._select_backend()
        is_local = isinstance(backend, LocalBackend)
        return backend.generate(
            prompt, max_new_tokens, temperature,
            top_p, top_k, repetition_penalty,
            session_id=session_id if is_local else None,
            **kwargs,
        )

    # --- Streaming generation ---

    def generate_stream_sync(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        cancel_event=None,
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Synchronous streaming generation — returns a text streamer.

        Delegates to the selected backend's generate_stream().
        """
        backend = self._select_backend()
        is_local = isinstance(backend, LocalBackend)
        gen = backend.generate_stream(
            prompt, max_new_tokens, temperature,
            top_p, top_k, repetition_penalty,
            cancel_event=cancel_event,
            session_id=session_id if is_local else None,
            **kwargs,
        )
        return self._wrap_generator_as_streamer(gen)

    # --- Async streaming generation ---

    async def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        cancel_event=None,
        session_id: Optional[str] = None,
        priority: Priority = Priority.MEDIUM,
        **kwargs: Any,
    ) -> Any:
        """Async streaming generation with full lifecycle management.

        Selects the best backend (guard or local), acquires semaphore,
        runs hooks, runs generation in a thread pool (non-blocking), and
        yields tokens.

        If ``session_id`` is provided, the KV cache from previous
        generations is reused via :class:`SessionKVCache`, avoiding
        re-encoding the shared prompt prefix.

        Args:
            cancel_event: Optional ``threading.Event`` to abort generation early.

        Yields:
            str tokens from the streamer.

        Raises ``TimeoutError`` if semaphore cannot be acquired.
        """
        with self._metrics_lock:
            self.metrics.requests_total += 1

        # Circuit breaker check
        if self._circuit_breaker and not self._circuit_breaker.allow_request():
            raise RuntimeError(
                f"Circuit breaker open for {self.model_id} "
                f"(state={self._circuit_breaker.state.value})"
            )

        # Pre-generation hooks
        for hook in self._pre_generate_hooks:
            try:
                hook()
            except Exception as e:
                logger.warning("Pre-gen hook failed: %s", e, extra={"tag": "MODEL"})

        # Acquire slot in the priority queue for admission control
        queue = await self._ensure_queue()
        try:
            _release = await queue.acquire(
                priority=priority,
                request_id=f"stream-{id(prompt[:32])}",
            )
        except RuntimeError as e:
            error_msg = f"{type(e).__name__}: {e}"
            with self._metrics_lock:
                self.metrics.record_failure(error_msg)
            if self._circuit_breaker:
                self._circuit_breaker.record_failure()
            raise

        # Select backend (guard if alive, else local)
        backend = self._select_backend()
        logger.debug("generate_stream[%s]: backend=%s session_id=%s",
                     self.model_id, type(backend).__name__, session_id)

        start = time.time()
        token_count = 0
        aborted = False
        pump_thread = None
        try:
            # Run the backend's sync generator in a thread pool so we
            # don't block the event loop during generation.
            loop = asyncio.get_event_loop()

            def _gen():
                return backend.generate_stream(
                    prompt, max_new_tokens, temperature,
                    top_p, top_k, repetition_penalty,
                    cancel_event=cancel_event, **kwargs,
                )

            # We can't directly await a generator, so we run the
            # first-token generation in a thread and then iterate
            # the sync generator's queue from the async context.
            #
            # Strategy: start a thread that pumps the sync generator
            # into a queue; the async generator reads from the queue.
            import queue as _queue
            q: _queue.Queue = _queue.Queue()
            _sentinel = object()

            is_local = isinstance(backend, LocalBackend)

            def _pump():
                logger.debug("_pump started: session_id=%s is_local=%s", session_id, is_local)
                try:
                    for token in backend.generate_stream(
                        prompt, max_new_tokens, temperature,
                        top_p, top_k, repetition_penalty,
                        cancel_event=cancel_event,
                        session_id=session_id if is_local else None,
                        **kwargs,
                    ):
                        q.put(token)
                except Exception as e:
                    q.put(e)
                finally:
                    q.put(_sentinel)

            pump_thread = Thread(target=_pump, daemon=True)
            pump_thread.start()

            while True:
                try:
                    item = q.get(timeout=0.02)
                except _queue.Empty:
                    await asyncio.sleep(0)
                    continue
                if item is _sentinel:
                    break
                if isinstance(item, Exception):
                    raise item
                token_count += 1
                yield item

            pump_thread.join(timeout=30)

            # Success — record metrics
            elapsed_ms = (time.time() - start) * 1000
            with self._metrics_lock:
                self.metrics.record_success(elapsed_ms, token_count)
            if self._circuit_breaker:
                self._circuit_breaker.record_success()
            logger.info("Streamed", model_id=self.model_id, tokens=token_count,
                        elapsed_ms=round(elapsed_ms, 1), session_id=session_id)

        except GeneratorExit:
            aborted = True
            logger.info("generate_stream[%s]: client disconnected mid-stream", self.model_id, extra={"tag": "MODEL"})
            if cancel_event is not None:
                cancel_event.set()
            # Join the pump thread so post-generation hooks don't race with it
            if pump_thread is not None and pump_thread.is_alive():
                pump_thread.join(timeout=10)
                if pump_thread.is_alive():
                    logger.warning("generate_stream[%s]: pump thread did not stop within 10s", self.model_id, extra={"tag": "MODEL"})
            return
        except asyncio.TimeoutError:
            with self._metrics_lock:
                self.metrics.record_timeout()
            self._on_generation_error(RuntimeError("Generation timed out"))
            raise
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            with self._metrics_lock:
                self.metrics.record_failure(error_msg)
            if self._circuit_breaker:
                self._circuit_breaker.record_failure()
            self._on_generation_error(e)
            _mps_oom_recovery()
            raise
        finally:
            # Post-generation hooks (KV cache reset, memory cleanup)
            for hook in self._post_generate_hooks:
                try:
                    hook()
                except Exception as e:
                    logger.warning("Post-gen hook failed: %s", e, extra={"tag": "MODEL"})
            # Release priority-queue slot
            _release()
            if aborted:
                logger.info("generate_stream[%s]: cleaned up after abort", self.model_id, extra={"tag": "MODEL"})

    @staticmethod
    def _wrap_generator_as_streamer(gen):
        """Wrap a generator as a TextIteratorStreamer-compatible object.

        Returns an object with ``text_queue`` and ``stop_signal`` so callers
        of ``generate_stream_sync`` can use the same polling pattern.
        """
        import queue
        q = queue.Queue()
        stop_signal = object()

        def _pump():
            try:
                for token in gen:
                    q.put(token)
            except StopIteration:
                pass
            except Exception as e:
                logger.debug("Streaming pump generator failed: %s", e)
            finally:
                q.put(stop_signal)

        import threading
        t = threading.Thread(target=_pump, daemon=True)
        t.start()

        streamer = type("_GenStreamer", (), {})()
        streamer.text_queue = q
        streamer.stop_signal = stop_signal
        return streamer

    # --- Error handling ---

    def _on_generation_error(self, error: Exception) -> None:
        self.set_status(ModelStatus.DEGRADED)
        for hook in self._on_error_hooks:
            try:
                hook(error)
            except Exception as e:
                logger.warning("Error hook failed: %s", e, extra={"tag": "MODEL"})

    # --- Model swap (hot-reload) ---

    def swap_model(self, new_model: Any) -> None:
        """Atomically swap the underlying model reference."""
        with self._lock:
            old = self._model_ref
            self._model_ref = new_model
        # Sync local backend
        if self._local_backend is not None:
            self._local_backend._model_ref = new_model
        # Clean up old model
        if old is not None and old is not new_model:
            try:
                del old
            except Exception:
                pass
            _schedule_gc()
        self._check_device()
        self.set_status(ModelStatus.READY)
        logger.info("ModelServer[%s]: model swapped", self.model_id, extra={"tag": "MODEL"})
        # Re-warmup with new model
        with self._warmup_lock:
            self._warmup_completed = False
            self._warmup_error = None
        if self._enable_warmup:
            Thread(target=self._run_warmup, daemon=True).start()
