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
from enum import Enum

logger = logging.getLogger("man.infrastructure.model_server")


def _optimize_cpu_threads() -> None:
    """Set optimal CPU thread count for PyTorch inference on Intel Mac."""
    import os
    if not _ensure_torch():
        return
    import torch
    # Detect physical cores — on this Intel Mac (i7-9750H) that's 6
    try:
        import multiprocessing
        n_cores = multiprocessing.cpu_count()
    except Exception:
        n_cores = 4
    # Reserve 1 thread for async overhead when >4 cores
    intra_ops = max(1, n_cores - 1) if n_cores > 4 else n_cores
    inter_ops = 2  # parallel graph partitions
    torch.set_num_threads(intra_ops)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(min(inter_ops, intra_ops))
    os.environ.setdefault("OMP_NUM_THREADS", str(intra_ops))
    os.environ.setdefault("MKL_NUM_THREADS", str(intra_ops))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    logger.debug("CPU threads: intra=%d inter=%d cores=%d", intra_ops, inter_ops, n_cores)


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
        logger.info("torch.compile applied to %s", model_id)
        return compiled
    except Exception as e:
        logger.warning("torch.compile failed for %s: %s", model_id, e)
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
        gc.collect()
        from domains.infrastructure.ml_types import mps as ml_mps
        if _has_mps():
            ml_mps.empty_cache()
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


class NumpyBackend(GenerateBackend):
    """Pure-NumPy backend — no PyTorch, no tokenizer, no MPS.

    Wraps a NumpyEngine instance for direct safetensors→numpy inference.
    Used when NumpyEngine is loaded as the primary model (zero PyTorch dep).

    Features:
      - Compression: weights compressed via VQ (4x memory savings)
      - KV cache: incremental decoding (faster generation)
      - Streaming: token-by-token async output
    """

    def __init__(self, engine: Any):
        self._engine = engine

    @property
    def alive(self) -> bool:
        return self._engine is not None

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
        """Generate text using NumpyEngine with KV cache."""
        text = self._engine.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            use_kv_cache=True,
        )
        input_len = len(self._engine.tokenizer.encode(prompt))
        output_len = len(self._engine.tokenizer.encode(text))
        tokens_generated = max(0, output_len - input_len)
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
        **kwargs: Any,
    ) -> GeneratorType[str, None, dict]:
        """Stream tokens from NumpyEngine — yields one token at a time.

        Uses the engine's generate_stream() async generator for true
        token-by-token streaming with KV cache support.
        """
        if cancel_event is not None and cancel_event.is_set():
            return {"text": "", "tokens_generated": 0}

        import asyncio

        # Collect tokens from async generator via a coroutine
        collected_tokens = []

        async def _collect():
            async for token in self._engine.generate_stream(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            ):
                if cancel_event is not None and cancel_event.is_set():
                    break
                collected_tokens.append(token)

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_collect())
        finally:
            loop.close()

        # Yield collected tokens synchronously
        token_count = 0
        for token in collected_tokens:
            token_count += 1
            yield token

        return {"text": "", "tokens_generated": token_count}

class LocalBackend(GenerateBackend):
    """Direct in-process model.generate()."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        lock: Lock,
        device: str,
        tokenize_cache: dict,
    ):
        self._model_ref = model
        self._tokenizer = tokenizer
        self._lock = lock
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
        **kwargs: Any,
    ) -> dict:
        from domains.infrastructure.ml_types import no_grad as ml_no_grad

        inputs = _tokenize_cached(self._tokenizer, prompt, self._tokenize_cache)
        input_ids = inputs["input_ids"].to(self._device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)

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
            except Exception:
                pass

        output = _inference_mode_generate(self._model_ref, gen_kwargs)

        if _cpu_fallback:
            try:
                with self._lock:
                    self._model_ref = self._model_ref.to(self._device)
            except Exception:
                pass

        _mps_empty_cache()

        tokens_generated = output.shape[1] - input_ids.shape[1]
        text = self._tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
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
        gen_kwargs.update(kwargs)

        if cancel_event is not None:
            class _CancelCriteria(StoppingCriteria):
                def __call__(self, input_ids_, scores_, **kwargs):
                    return cancel_event.is_set()
            gen_kwargs.setdefault("stopping_criteria", [])
            gen_kwargs["stopping_criteria"].append(_CancelCriteria())

        _error: list = []

        def _generate_inner():
            try:
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
        max_concurrent: int = 1,
        generate_timeout: float = 120.0,
        enable_circuit_breaker: bool = True,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        process_guard: Optional[Any] = None,
        enable_warmup: bool = True,
        warmup_prompt: str = "Hello",
        numpy_engine: Optional[Any] = None,
    ):
        # CPU threading optimization (torch.set_num_threads etc.)
        _optimize_cpu_threads()

        self.model_id = model_id
        self._tokenizer = tokenizer
        self._model_ref = model
        self._lock = Lock()  # protects model reference swap
        self._process_guard = process_guard  # optional ProcessGuard for bulk gen

        # Generate backends — strategy pattern
        self._guard_backend: Optional[GuardBackend] = (
            GuardBackend(process_guard) if process_guard is not None else None
        )
        self._numpy_backend: Optional[NumpyBackend] = (
            NumpyBackend(numpy_engine) if numpy_engine is not None else None
        )
        self._local_backend = LocalBackend(
            model=model,
            tokenizer=tokenizer,
            lock=self._lock,
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

        # Concurrency control (per-event-loop semaphore — handles warmup/test loops)
        self._max_concurrent = max_concurrent
        self._semaphores: dict[int, asyncio.Semaphore] = {}

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
            logger.info("ModelServer[%s]: warmup completed", self.model_id)
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
            logger.warning("ModelServer[%s]: warmup failed: %s", self.model_id, e)

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

        Priority: GuardBackend (crash-isolated) > NumpyBackend (no torch) > LocalBackend.
        """
        if self._guard_backend is not None and self._guard_backend.alive:
            return self._guard_backend
        if self._numpy_backend is not None and self._numpy_backend.alive:
            return self._numpy_backend
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
        logger.info("ModelServer[%s]: dropped in-memory model ref (guard mode)", self.model_id)

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
                    except Exception:
                        pass
            import gc
            gc.collect()
        except Exception:
            pass

    # --- Lifecycle hooks ---

    def add_pre_generate_hook(self, hook: Callable[[], None]) -> None:
        self._pre_generate_hooks.append(hook)

    def add_post_generate_hook(self, hook: Callable[[], None]) -> None:
        self._post_generate_hooks.append(hook)

    def add_on_error_hook(self, hook: Callable[[Exception], None]) -> None:
        self._on_error_hooks.append(hook)

    # --- Status ---

    async def _get_semaphore(self) -> Optional[asyncio.Semaphore]:
        """Get an asyncio semaphore for the current event loop.

        Each event loop gets its own semaphore so warmup (daemon thread with
        its own loop) and main-request ``asyncio.run()`` loops don't collide.
        """
        if self._max_concurrent is None:
            return None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return None
        loop_id = id(loop)
        if loop_id not in self._semaphores:
            self._semaphores[loop_id] = asyncio.Semaphore(self._max_concurrent)
        return self._semaphores[loop_id]

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
        base["semaphore_locked"] = any(
            s.locked() for s in self._semaphores.values()
        ) if self._semaphores else False
        base["warmup_completed"] = warmup_ok
        base["warmup_error"] = warmup_err
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
        **kwargs: Any,
    ) -> dict:
        """Generate text with full lifecycle management.

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
                logger.warning("Pre-gen hook failed: %s", e)

        # Acquire semaphore (serialize concurrent access)
        acquired = False
        semaphore = await self._get_semaphore()
        if semaphore is not None:
            try:
                await asyncio.wait_for(
                    semaphore.acquire(),
                    timeout=min(self._generate_timeout, 30.0),
                )
                acquired = True
            except asyncio.TimeoutError:
                with self._metrics_lock:
                    self.metrics.record_timeout()
                raise TimeoutError(
                    f"Request queued too long for {self.model_id} "
                    f"(semaphore timeout)"
                )
        else:
            acquired = True  # no semaphore — proceed

        try:
            # Run generation in thread pool
            start = time.time()
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    self._generate_sync,
                    prompt,
                    max_new_tokens,
                    temperature,
                    top_p,
                    top_k,
                    repetition_penalty,
                    **kwargs,
                ),
                timeout=self._generate_timeout,
            )
            elapsed_ms = (time.time() - start) * 1000
            tokens = result.get("tokens_generated", 0)

            with self._metrics_lock:
                self.metrics.record_success(elapsed_ms, tokens)

            # Circuit breaker: record success
            if self._circuit_breaker:
                self._circuit_breaker.record_success()

            return result

        except asyncio.TimeoutError:
            with self._metrics_lock:
                self.metrics.record_timeout()
            self._on_generation_error(RuntimeError("Generation timed out"))
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
        finally:
            if acquired and semaphore is not None:
                semaphore.release()
            # Post-generation hooks (KV cache reset, memory cleanup)
            for hook in self._post_generate_hooks:
                try:
                    hook()
                except Exception as e:
                    logger.warning("Post-gen hook failed: %s", e)

    def _generate_sync(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        **kwargs: Any,
    ) -> dict:
        """Synchronous generation — delegates to selected backend.

        Backend selection: prefers GuardBackend (crash-isolated), falls back
        to LocalBackend when guard is dead or absent.
        """
        backend = self._select_backend()
        return backend.generate(
            prompt, max_new_tokens, temperature,
            top_p, top_k, repetition_penalty, **kwargs,
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
        **kwargs: Any,
    ) -> Any:
        """Synchronous streaming generation — returns a text streamer.

        Delegates to the selected backend's generate_stream().
        """
        backend = self._select_backend()
        gen = backend.generate_stream(
            prompt, max_new_tokens, temperature,
            top_p, top_k, repetition_penalty,
            cancel_event=cancel_event, **kwargs,
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
        **kwargs: Any,
    ) -> Any:
        """Async streaming generation with full lifecycle management.

        Selects the best backend (guard or local), acquires semaphore,
        runs hooks, runs generation in a thread pool (non-blocking), and
        yields tokens.

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
                logger.warning("Pre-gen hook failed: %s", e)

        # Select backend (guard if alive, else local)
        backend = self._select_backend()

        # Acquire semaphore (serialize concurrent access, per-event-loop)
        acquired = False
        semaphore = await self._get_semaphore()
        if semaphore is not None:
            try:
                await asyncio.wait_for(
                    semaphore.acquire(),
                    timeout=min(self._generate_timeout, 30.0),
                )
                acquired = True
            except asyncio.TimeoutError:
                with self._metrics_lock:
                    self.metrics.record_timeout()
                raise TimeoutError(
                    f"Request queued too long for {self.model_id} "
                    f"(semaphore timeout)"
                )
        else:
            acquired = True

        start = time.time()
        token_count = 0
        aborted = False
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

            def _pump():
                try:
                    for token in backend.generate_stream(
                        prompt, max_new_tokens, temperature,
                        top_p, top_k, repetition_penalty,
                        cancel_event=cancel_event, **kwargs,
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

        except GeneratorExit:
            aborted = True
            logger.info("generate_stream[%s]: client disconnected mid-stream", self.model_id)
            if cancel_event is not None:
                cancel_event.set()
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
            if acquired and semaphore is not None:
                semaphore.release()
            # Post-generation hooks (KV cache reset, memory cleanup)
            for hook in self._post_generate_hooks:
                try:
                    hook()
                except Exception as e:
                    logger.warning("Post-gen hook failed: %s", e)
            if aborted:
                logger.info("generate_stream[%s]: cleaned up after abort", self.model_id)

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
            except Exception:
                pass
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
                logger.warning("Error hook failed: %s", e)

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
        gc.collect()
        self._check_device()
        self.set_status(ModelStatus.READY)
        logger.info("ModelServer[%s]: model swapped", self.model_id)
        # Re-warmup with new model
        with self._warmup_lock:
            self._warmup_completed = False
            self._warmup_error = None
        if self._enable_warmup:
            Thread(target=self._run_warmup, daemon=True).start()
