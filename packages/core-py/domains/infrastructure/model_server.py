"""
Composable model serving infrastructure with request lifecycle management.

Provides ModelServer — a wrapper around any HuggingFace-compatible model
that handles concurrency control, timeouts, error recovery, and metrics.

Architecture::

    Request → ModelServer.generate()
                ├── acquire semaphore (serializes concurrent access)
                ├── pre-generation hook (OOM check, cache warm)
                ├── model.generate() with timeout wrapper
                ├── post-generation hook (KV cache reset)
                └── release semaphore
"""

import asyncio
import logging
import time
import gc
import os
import functools
from threading import Lock, Thread, Event

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


def _cpu_fallback() -> str:
    """Force model to CPU; returns device string."""
    try:
        from domains.training.slonet_compat import torch
        for obj in gc.get_objects():
            if hasattr(obj, "device") and hasattr(obj, "to"):
                try:
                    if str(obj.device) != "cpu":
                        obj.to("cpu")
                except Exception:
                    pass
        logger.warning("Forced all torch tensors to CPU")
    except Exception:
        pass
    return "cpu"


class ModelServer:
    """Composable wrapper around a HuggingFace model for safe concurrent serving.

    Usage::

        server = ModelServer(model, tokenizer, model_id="gpt2")
        result = server.generate(prompt, max_new_tokens=100)
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        model_id: str = "unknown",
        max_concurrent: int = 1,
        generate_timeout: float = 120.0,
        enable_circuit_breaker: bool = True,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        process_guard: Optional[Any] = None,
        enable_warmup: bool = True,
        warmup_prompt: str = "Hello",
    ):
        self.model_id = model_id
        self._tokenizer = tokenizer
        self._model_ref = model
        self._lock = Lock()  # protects model reference swap
        self._process_guard = process_guard  # optional ProcessGuard for bulk gen

        # Wire guard crash callbacks to circuit breaker
        if self._process_guard is not None and self._circuit_breaker is not None:
            self._process_guard.on_crash(
                lambda wid: self._circuit_breaker.record_failure()
            )
            self._process_guard.on_restart(
                lambda wid: self._circuit_breaker.record_success()
            )

        # Warmup
        self._enable_warmup = enable_warmup
        self._warmup_prompt = warmup_prompt
        self._warmup_completed = False
        self._warmup_error: Optional[str] = None
        self._warmup_lock = Lock()

        # Concurrency control
        try:
            self._semaphore = asyncio.Semaphore(max_concurrent)
        except RuntimeError:
            # No event loop in this thread; run with unlimited concurrency
            self._semaphore = None

        # Timeout
        self._generate_timeout = generate_timeout

        # Circuit breaker
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        ) if enable_circuit_breaker else None

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

        # Tokenizer cache (LRU, 64 entries)
        self._tokenize_cache: dict = {}

        # Device tracking
        self._device: Optional[str] = None
        self._check_device()

        # Register default post-generation hook for KV cache cleanup
        self.add_post_generate_hook(self._cleanup_kv_cache)

        # Background warmup
        if self._enable_warmup:
            Thread(target=self._run_warmup, daemon=True).start()

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
        except Exception as e:
            with self._warmup_lock:
                self._warmup_error = f"{type(e).__name__}: {e}"
            logger.warning("ModelServer[%s]: warmup failed: %s", self.model_id, e)

    def _check_device(self) -> None:
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

    def _cleanup_kv_cache(self) -> None:
        """Clear any KV cache tensors the model may have accumulated."""
        try:
            import torch
            if hasattr(self._model_ref, "past_key_values"):
                self._model_ref.past_key_values = None
            # Clear model's internal cache dict if it exists
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
        base["semaphore_locked"] = self._semaphore.locked() if self._semaphore is not None else False
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
        if self._semaphore is not None:
            try:
                result = await asyncio.wait_for(
                    self._semaphore.acquire(),
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
            if acquired and self._semaphore is not None:
                self._semaphore.release()
            # Post-generation hooks (KV cache reset, memory cleanup)
            for hook in self._post_generate_hooks:
                try:
                    hook()
                except Exception as e:
                    logger.warning("Post-gen hook failed: %s", e)

    def _tokenize_cached(self, tokenizer, prompt: str) -> tuple:
        """Tokenize with LRU cache to avoid redundant tokenization.

        Keyed on prompt text only — each ModelServer has exactly one
        tokenizer, so the prompt is sufficient for cache identity.
        """
        if prompt in self._tokenize_cache:
            ids, attn = self._tokenize_cache[prompt]
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
        self._tokenize_cache[prompt] = (
            inputs["input_ids"][0].tolist(),
            attn_list,
        )
        if len(self._tokenize_cache) > 64:
            # Evict oldest
            self._tokenize_cache.pop(next(iter(self._tokenize_cache)))
        return inputs

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
        """Synchronous generation — runs in thread pool (or process guard)."""
        # Delegate to process guard if available (process-level isolation)
        if self._process_guard is not None:
            # Strip kwargs that should not be forwarded (input_ids, attention_mask
            # are built inside the worker)
            safe_kwargs = {k: v for k, v in kwargs.items()
                           if k not in ("input_ids", "attention_mask")}
            start = time.time()
            result = self._process_guard.generate(
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

        if not _ensure_torch():
            raise RuntimeError("torch is required for synchronous generation")

        with self._lock:
            model = self._model_ref
            tokenizer = self._tokenizer

        inputs = self._tokenize_cached(tokenizer, prompt)
        input_ids = inputs["input_ids"].to(self._device or "cpu")
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device or "cpu")

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        gen_kwargs.update(kwargs)

        import torch
        from domains.infrastructure.ml_types import no_grad as ml_no_grad
        with ml_no_grad():
            output = model.generate(**gen_kwargs)

        tokens_generated = output.shape[1] - input_ids.shape[1]
        text = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)

        return {"text": text, "tokens_generated": tokens_generated}

    # --- Streaming generation ---

    def generate_stream_sync(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        cancel_event=None,
        **kwargs: Any,
    ) -> Any:
        """Synchronous streaming generation — returns a text streamer.

        Runs in caller's thread. The caller is responsible for polling
        the streamer and for semaphore management.

        When a ``_process_guard`` is configured, delegates to the guard's
        ``generate_stream()`` which streams from a subprocess worker.

        Args:
            cancel_event: Optional ``threading.Event``. When set, a
                ``StoppingCriteria`` tells ``model.generate()`` to stop
                early (e.g. on client disconnect).
        """
        # Delegate to process guard if available
        if self._process_guard is not None:
            safe_kwargs = {k: v for k, v in kwargs.items()
                           if k not in ("input_ids", "attention_mask")}
            if cancel_event is not None:
                # Guard doesn't support cancel_event yet; wrap in a custom
                # generator that stops yielding when cancel is set
                gen = self._process_guard.generate_stream(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    **safe_kwargs,
                )
                return self._wrap_cancelable_streamer(gen, cancel_event)
            from typing import Generator
            gen = self._process_guard.generate_stream(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                **safe_kwargs,
            )
            return self._wrap_generator_as_streamer(gen)

        from transformers import TextIteratorStreamer, StoppingCriteria

        with self._lock:
            model = self._model_ref
            tokenizer = self._tokenizer

        inputs = self._tokenize_cached(tokenizer, prompt)
        input_ids = inputs["input_ids"].to(self._device or "cpu")
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device or "cpu")

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, timeout=self._generate_timeout)

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )
        gen_kwargs.update(kwargs)

        if cancel_event is not None:
            class _CancelCriteria(StoppingCriteria):
                def __call__(self, input_ids_, scores_, **kwargs):
                    return cancel_event.is_set()
            gen_kwargs.setdefault("stopping_criteria", [])
            gen_kwargs["stopping_criteria"].append(_CancelCriteria())

        import torch
        from domains.infrastructure.ml_types import no_grad as ml_no_grad
        with ml_no_grad():
            model.generate(**gen_kwargs)

        return streamer

    # --- Async streaming generation ---

    async def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        cancel_event=None,
        **kwargs: Any,
    ) -> Any:
        """Async streaming generation with full lifecycle management.

        Acquires semaphore, runs hooks, starts generation in background
        thread, and yields tokens from the streamer.

        When ``cancel_event`` is set (e.g. client disconnect), a stopping
        criteria tells ``model.generate()`` to stop early.

        Args:
            cancel_event: Optional ``threading.Event`` to abort generation early.

        Yields:
            str tokens from the streamer.

        Raises ``TimeoutError`` if semaphore cannot be acquired.
        """
        from transformers import StoppingCriteria

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

        # Acquire semaphore
        acquired = False
        if self._semaphore is not None:
            try:
                await asyncio.wait_for(
                    self._semaphore.acquire(),
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

        streamer = None
        thread = None
        start = 0.0
        token_count = 0
        aborted = False
        try:
            from transformers import TextIteratorStreamer
            import torch
            from threading import Thread
            import queue

            with self._lock:
                model = self._model_ref
                tokenizer = self._tokenizer

            inputs = self._tokenize_cached(tokenizer, prompt)
            input_ids = inputs["input_ids"].to(self._device or "cpu")
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self._device or "cpu")

            streamer = TextIteratorStreamer(
                tokenizer, skip_prompt=True, timeout=self._generate_timeout
            )

            gen_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                streamer=streamer,
            )
            gen_kwargs.update(kwargs)

            if cancel_event is not None:
                class _CancelCriteria(StoppingCriteria):
                    def __call__(self, input_ids_, scores_, **kwargs):
                        return cancel_event.is_set()
                gen_kwargs.setdefault("stopping_criteria", [])
                gen_kwargs["stopping_criteria"].append(_CancelCriteria())

            _error: list[Exception] = []

            def _generate_inner():
                try:
                    from domains.infrastructure.ml_types import no_grad as ml_no_grad
                    with ml_no_grad():
                        model.generate(**gen_kwargs)
                except Exception as e:
                    _error.append(e)

            thread = Thread(target=_generate_inner)
            thread.start()

            start = time.time()

            while thread.is_alive() or not streamer.text_queue.empty():
                if _error:
                    raise _error[0]
                try:
                    text = streamer.text_queue.get(timeout=0.02)
                except queue.Empty:
                    await asyncio.sleep(0)
                    continue
                if text == streamer.stop_signal:
                    break
                if text:
                    token_count += 1
                    yield text

            thread.join(timeout=30)

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
            if streamer is not None:
                try:
                    streamer.text_queue.put(streamer.stop_signal)
                except Exception:
                    pass
            if thread is not None and thread.is_alive():
                thread.join(timeout=5)
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
            if acquired and self._semaphore is not None:
                self._semaphore.release()
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

    @staticmethod
    def _wrap_cancelable_streamer(gen, cancel_event):
        """Wrap a generator as a cancelable TextIteratorStreamer."""
        import queue
        q = queue.Queue()
        stop_signal = object()

        def _pump():
            try:
                for token in gen:
                    if cancel_event.is_set():
                        break
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
