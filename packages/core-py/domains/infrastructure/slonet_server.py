import asyncio
import time
import threading
import queue
import logging
import copy
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Optional, List, Iterator, AsyncIterator, Dict, Any, Callable

import numpy as np

from domains.infrastructure.model_server import CircuitBreaker, ModelMetrics

logger = logging.getLogger("slo.infrastructure.slonet_server")


class SloNetServer:
    """
    Thread-safe ``SloTransformer`` wrapper with concurrency control, circuit
    breaker, warmup, and metrics. Mirrors ``ModelServer`` patterns.

    Two dispatch modes:
    - **Single model** (default): ``model`` param, serialized via ``asyncio.Semaphore(1)``
    - **Multi-worker pool**: ``model_factory`` + ``max_workers`` — each worker gets its
      own model copy for true parallel generation on CPU.

    Args:
        model: SloTransformer instance (single-model mode)
        model_factory: Zero-arg callable returning a SloTransformer (pool mode)
        tokenizer: Tokenizer with ``encode()`` / ``decode()`` / ``eos_token_id``
        model_id: Identifier for metrics/metadata
        max_workers: Pool size (only used with ``model_factory``)
        generate_timeout: Per-call generation timeout in seconds
        enable_circuit_breaker: If True, failures gate future requests
        enable_warmup: If True, background warmup on init
        warmup_prompt: Short prompt for warmup generation
    """

    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        model_factory: Optional[Callable[[], Any]] = None,
        model_id: str = "slonet",
        max_workers: int = 4,
        generate_timeout: float = 120.0,
        enable_circuit_breaker: bool = True,
        enable_warmup: bool = True,
        warmup_prompt: str = "Hello",
    ):
        self._tokenizer = tokenizer
        self._model_id = model_id
        self._generate_timeout = generate_timeout
        self._warmup_prompt = warmup_prompt
        self._max_workers = max_workers

        self._read_semaphores: dict[int, asyncio.Semaphore] = {}
        self._max_readers = max_workers * 4

        self._circuit_breaker = CircuitBreaker() if enable_circuit_breaker else None

        self._metrics_lock = threading.Lock()
        self._metrics = ModelMetrics()

        self._warmup_lock = threading.Lock()
        self._warmup_completed = False
        self._warmup_error: Optional[str] = None

        # Dispatch mode: pool or single-model
        if model_factory is not None:
            self._pool_mode = True
            self._model_factory = model_factory
            self._model_pool: queue.Queue = queue.Queue()
            self._executor = ThreadPoolExecutor(max_workers=max_workers)
            self._pool_lock = threading.Lock()
            self._pool_size = 0
        else:
            self._pool_mode = False
            self._model = model
            self._semaphore = asyncio.Semaphore(1)

        if enable_warmup:
            threading.Thread(target=self._run_warmup, daemon=True).start()

    # ------------------------------------------------------------------
    # Model pool management
    # ------------------------------------------------------------------

    def _acquire_model(self, timeout: float = 30.0) -> Any:
        if not self._pool_mode:
            return self._model
        try:
            return self._model_pool.get_nowait()
        except queue.Empty:
            with self._pool_lock:
                if self._pool_size < self._max_workers:
                    model = self._model_factory()
                    self._pool_size += 1
                    return model
            try:
                return self._model_pool.get(timeout=timeout)
            except queue.Empty:
                raise RuntimeError(
                    f"SloNet worker pool exhausted ({self._max_workers} workers, "
                    f"all busy after {timeout}s)"
                )

    def _release_model(self, model: Any) -> None:
        if self._pool_mode:
            self._model_pool.put(model)

    def pool_stats(self) -> dict:
        if not self._pool_mode:
            return {"mode": "single"}
        return {
            "mode": "pool",
            "workers": self._max_workers,
            "available": self._model_pool.qsize(),
            "created": self._pool_size,
        }

    # ------------------------------------------------------------------
    # Read semaphore helpers
    # ------------------------------------------------------------------

    def _get_read_semaphore(self) -> asyncio.Semaphore:
        loop_id = id(asyncio.get_event_loop())
        if loop_id not in self._read_semaphores:
            self._read_semaphores[loop_id] = asyncio.Semaphore(self._max_readers)
        return self._read_semaphores[loop_id]

    async def tokenize(self, text: str) -> List[int]:
        sem = self._get_read_semaphore()
        async with sem:
            return self._tokenizer.encode(text)

    async def count_tokens(self, text: str) -> int:
        sem = self._get_read_semaphore()
        async with sem:
            return len(self._tokenizer.encode(text))

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def _run_warmup(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            self._generate_sync(self._warmup_prompt, max_new_tokens=5, temperature=0.7)
            with self._warmup_lock:
                self._warmup_completed = True
        except Exception as e:
            with self._warmup_lock:
                self._warmup_error = str(e)
            logger.warning("SloNet warmup failed: %s", e)

    @property
    def warmup_completed(self) -> bool:
        return self._warmup_completed

    @property
    def warmup_error(self) -> Optional[str]:
        return self._warmup_error

    # ------------------------------------------------------------------
    # Sync generation (runs in thread pool)
    # ------------------------------------------------------------------

    def _generate_sync(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        if cancel_event and cancel_event.is_set():
            raise RuntimeError("Generation cancelled before start")
        model = self._acquire_model()
        try:
            tokens = self._tokenizer.encode(prompt)
            input_ids = np.array([tokens], dtype=np.int64)
            result = model.generate_numpy(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token=self._tokenizer.eos_token_id or 0,
            )
            return self._tokenizer.decode(result[0].tolist())
        finally:
            self._release_model(model)

    def _generate_stream_sync(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        cancel_event: Optional[threading.Event] = None,
    ) -> Iterator[str]:
        model = self._acquire_model()
        try:
            tokens = self._tokenizer.encode(prompt)
            input_ids = np.array([tokens], dtype=np.int64)
            eos_id = self._tokenizer.eos_token_id or 0

            for tok_id in model.generate_numpy_stream(
                input_ids,
                max_new_tokens=max_new_tokens,
                eos_token=eos_id,
            ):
                if cancel_event and cancel_event.is_set():
                    return
                decoded = self._tokenizer.decode([tok_id])
                if decoded:
                    yield decoded
        finally:
            self._release_model(model)

    # ------------------------------------------------------------------
    # Async generation (public API)
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        if cancel_event and cancel_event.is_set():
            raise asyncio.CancelledError("Generation cancelled before start")
        with self._metrics_lock:
            self._metrics.requests_total += 1

        if self._circuit_breaker and not self._circuit_breaker.allow_request():
            raise RuntimeError(f"SloNet circuit breaker open ({self._model_id})")

        start = time.monotonic()
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    self._generate_sync,
                    prompt, max_new_tokens,
                    temperature, top_p, top_k, repetition_penalty,
                    cancel_event,
                ),
                timeout=self._generate_timeout,
            )
            elapsed = (time.monotonic() - start) * 1000
            with self._metrics_lock:
                self._metrics.record_success(elapsed, 0)
            if self._circuit_breaker:
                self._circuit_breaker.record_success()
            return result
        except asyncio.TimeoutError:
            with self._metrics_lock:
                self._metrics.record_timeout()
            if self._circuit_breaker:
                self._circuit_breaker.record_failure()
            raise TimeoutError(
                f"SloNet generation timed out after {self._generate_timeout}s"
            )
        except Exception as e:
            with self._metrics_lock:
                self._metrics.record_failure(str(e))
            if self._circuit_breaker:
                self._circuit_breaker.record_failure()
            raise

    async def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        cancel_event: Optional[threading.Event] = None,
    ) -> AsyncIterator[str]:
        with self._metrics_lock:
            self._metrics.requests_total += 1

        if self._circuit_breaker and not self._circuit_breaker.allow_request():
            raise RuntimeError(f"SloNet circuit breaker open ({self._model_id})")

        q_buf: queue.Queue = queue.Queue()
        err_q: queue.Queue = queue.Queue()
        sentinel = object()

        def _pump() -> None:
            try:
                for token in self._generate_stream_sync(
                    prompt, max_new_tokens, temperature, top_p, top_k,
                    repetition_penalty, cancel_event,
                ):
                    q_buf.put(token)
            except Exception as e:
                err_q.put(e)
            finally:
                q_buf.put(sentinel)

        pump_thread = threading.Thread(target=_pump, daemon=True)
        pump_thread.start()

        start = time.monotonic()
        tokens = 0
        try:
            while True:
                try:
                    token = await asyncio.wait_for(
                        asyncio.to_thread(q_buf.get),
                        timeout=30.0,
                    )
                except asyncio.TimeoutError:
                    if not pump_thread.is_alive():
                        if not err_q.empty():
                            exc = err_q.get_nowait()
                            raise RuntimeError(f"SloNet stream error: {exc}")
                        while not q_buf.empty():
                            t = q_buf.get_nowait()
                            if t is sentinel:
                                break
                            yield t
                        break
                    if not err_q.empty():
                        exc = err_q.get_nowait()
                        raise RuntimeError(f"SloNet stream error: {exc}")
                    continue

                if token is sentinel:
                    if not err_q.empty():
                        exc = err_q.get_nowait()
                        raise RuntimeError(f"SloNet stream error: {exc}")
                    break

                if not err_q.empty():
                    exc = err_q.get_nowait()
                    raise RuntimeError(f"SloNet stream error: {exc}")

                tokens += 1
                yield token
        except asyncio.CancelledError:
            if cancel_event:
                cancel_event.set()
            pump_thread.join(timeout=10)
            raise
        finally:
            elapsed = (time.monotonic() - start) * 1000
            with self._metrics_lock:
                self._metrics.record_success(elapsed, tokens)
            if self._circuit_breaker:
                self._circuit_breaker.record_success()

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict:
        with self._metrics_lock:
            return self._metrics.snapshot()

    def metadata(self) -> dict:
        m = self._model if not self._pool_mode else None
        config = getattr(m, "_config", {}) if m else {}
        total_params = int(sum(p.data.size for p in m.parameters())) if m else 0
        n_layer = len([l for l in m.layers if hasattr(l, "forward_numpy")]) if m else 0
        n_embed = config.get("n_embd", config.get("hidden_size", 0))
        n_head = config.get("n_head", config.get("num_attention_heads", 0))
        vocab = int(m.layers[0].weight.shape[0]) if m else 0
        max_seq = int(m.max_seq_len) if m else 0
        cb_state = (
            self._circuit_breaker.state.value
            if self._circuit_breaker
            else "disabled"
        )
        with self._warmup_lock:
            wc = self._warmup_completed
            we = self._warmup_error
        return {
            "model_id": self._model_id,
            "architecture": "SloTransformer",
            "total_params": total_params,
            "n_layer": n_layer,
            "n_embed": n_embed,
            "n_head": n_head,
            "vocab_size": vocab,
            "max_seq_len": max_seq,
            "warmup_completed": wc,
            "warmup_error": we,
            "circuit_breaker_state": cb_state,
            "dispatch": "pool" if self._pool_mode else "single",
            "workers": self._max_workers if self._pool_mode else 1,
        }

    def health(self) -> dict:
        m = self.get_metrics()
        cb_state = (
            self._circuit_breaker.state.value
            if self._circuit_breaker
            else "disabled"
        )
        return {
            "status": "ready" if cb_state != "open" else "degraded",
            "model_id": self._model_id,
            "warmup": self.warmup_completed,
            "circuit_breaker": cb_state,
            "metrics": m,
            "pool": self.pool_stats(),
        }
