"""
Process-level crash guard for model inference workers.

Provides ProcessGuard — wraps model inference with automatic crash
detection, configurable restart policy, health monitoring, and lifecycle
callbacks.

Two execution modes:

  Thread (preferred — for autoload / PGQ context)::

      from .model_config import ModelConfig, ExecutionMode

      config = ModelConfig(slnc_path="models/gpt2.slnc", model_id="gpt2")
      guard = ProcessGuard(config, mode=ExecutionMode.THREAD)
      guard.start()
      result = guard.generate("Hello")

  Subprocess (for manual API load — full OS isolation)::

      config = ModelConfig(slnc_path="models/gpt2.slnc", model_id="gpt2")
      guard = ProcessGuard(config, mode=ExecutionMode.SUBPROCESS)
      guard.start()
      result = guard.generate("Hello")

Legacy construction (backward-compatible)::

      guard = ProcessGuard(slnc_path="models/gpt2.slnc", model_id="gpt2")
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, Generator
from typing import Any

from .constants import DEFAULT_GENERATE_TIMEOUT, DEFAULT_STALL_TIMEOUT, DEFAULT_STARTUP_TIMEOUT
from .model_config import ExecutionMode, ModelConfig

logger = logging.getLogger("slo.infrastructure.process_guard")


def resolve_memory_limit_mb(slnc_path: str | None, configured: float | None = None) -> float | None:
    """Resolve a guard worker memory limit from an explicit value or the model size."""
    import os

    if configured and configured > 0:
        return float(configured)
    if not slnc_path:
        return None
    try:
        size_mb = os.path.getsize(slnc_path) / (1024 * 1024)
    except OSError:
        return None
    return max(8192.0, size_mb * 8.0)


# ── Thread-mode worker ────────────────────────────────────────────────


class _ThreadWorker:
    """In-process worker that loads model in a thread and serves via Queue.

    Uses threading.Queue (not mp.Queue) so it works reliably when called
    from a PGQ ThreadPoolExecutor thread.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        worker_id: str = "worker",
        generate_timeout: float = DEFAULT_GENERATE_TIMEOUT,
        startup_timeout: float = DEFAULT_STARTUP_TIMEOUT,
    ):
        self._config = model_config
        self.worker_id = worker_id
        self._generate_timeout = generate_timeout
        self._startup_timeout = startup_timeout

        self._req_q: queue.Queue = queue.Queue()
        self._resp_q: queue.Queue = queue.Queue()
        self._hb_q: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._alive = False
        self._started_at: float = 0.0
        self._requests_served = 0
        self._errors = 0
        self._crashed = False
        self._crash_count = 0

    @property
    def alive(self) -> bool:
        return self._alive and self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        self._started_at = time.time()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"tw-{self.worker_id}"
        )
        self._thread.start()

        # Wait for ready signal
        deadline = time.time() + self._startup_timeout
        while time.time() < deadline:
            try:
                msg, val = self._hb_q.get(timeout=0.5)
                if msg == "ready":
                    self._alive = True
                    return
                elif msg == "dead":
                    raise RuntimeError(f"ThreadWorker[{self.worker_id}]: model load failed")
            except queue.Empty:
                if self._thread and not self._thread.is_alive():
                    raise RuntimeError(f"ThreadWorker[{self.worker_id}]: thread died during startup")
                continue
        raise RuntimeError(
            f"ThreadWorker[{self.worker_id}]: failed to start within {self._startup_timeout}s"
        )

    def stop(self, timeout: float = 10.0) -> None:
        self._alive = False
        try:
            self._req_q.put_nowait(("stop", None))
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def generate(self, prompt: str, **kwargs: Any) -> dict:
        if not self.alive:
            raise RuntimeError(f"ThreadWorker[{self.worker_id}] is not alive")
        session_id = _new_session_id()
        payload = {
            "max_new_tokens": kwargs.get("max_new_tokens", 100),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "top_k": kwargs.get("top_k", 50),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.0),
        }
        self._req_q.put_nowait(("generate", (session_id, prompt, payload)))

        deadline = time.time() + self._generate_timeout
        while time.time() < deadline:
            if not self.alive:
                self._crashed = True
                self._crash_count += 1
                raise RuntimeError(f"ThreadWorker[{self.worker_id}] crashed during generation")
            try:
                msg, *rest = self._resp_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if len(rest) == 1:
                msg_session, data = None, rest[0]
            else:
                msg_session, data = rest[0], rest[1]
            if msg_session is not None and msg_session != session_id:
                continue
            if msg == "result":
                self._requests_served += 1
                return data
            elif msg == "error":
                self._errors += 1
                raise RuntimeError(f"ThreadWorker generate error: {data}")

        self._errors += 1
        raise TimeoutError(
            f"ThreadWorker[{self.worker_id}] generation timed out after {self._generate_timeout}s"
        )

    def generate_stream(self, prompt: str, **kwargs: Any) -> Generator[str, None, dict]:
        if not self.alive:
            raise RuntimeError(f"ThreadWorker[{self.worker_id}] is not alive")
        session_id = _new_session_id()
        payload = {
            "max_new_tokens": kwargs.get("max_new_tokens", 100),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "top_k": kwargs.get("top_k", 50),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.0),
        }
        self._req_q.put_nowait(("generate_stream", (session_id, prompt, payload)))

        deadline = time.time() + self._generate_timeout
        final_result: dict = {}
        while time.time() < deadline:
            if not self.alive:
                self._crashed = True
                self._crash_count += 1
                raise RuntimeError(f"ThreadWorker[{self.worker_id}] crashed during streaming")
            try:
                msg, *rest = self._resp_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if len(rest) == 1:
                msg_session, data = None, rest[0]
            else:
                msg_session, data = rest[0], rest[1]
            if msg_session is not None and msg_session != session_id:
                continue
            if msg == "token":
                yield data
            elif msg == "result":
                self._requests_served += 1
                final_result = data
                break
            elif msg == "error":
                self._errors += 1
                raise RuntimeError(f"ThreadWorker stream error: {data}")
        return final_result

    def _run(self) -> None:
        """Worker thread entry point — loads model then processes requests."""
        try:
            if self._config.is_slo:
                self._run_slo()
            else:
                self._run_hf()
        except Exception as e:
            logger.error("ThreadWorker[%s]: died: %s", self.worker_id, e)
            try:
                self._hb_q.put_nowait(("dead", None))
            except Exception:
                pass
            self._alive = False

    def _run_slo(self) -> None:
        from domain.inference._internal.slonet_provider import SloNetChatProvider

        provider = SloNetChatProvider.from_slnc(
            self._config.slnc_path,
            model_id=self._config.model_id,
            quantize=self._config.quantize,
            quant_bits=self._config.quant_bits,
            quant_mode=self._config.quant_mode,
            quant_clip=self._config.quant_clip,
            free_quantized_originals=True,
        )
        self._hb_q.put_nowait(("ready", None))
        self._provider = provider
        self._request_loop()

    def _run_hf(self) -> None:
        from domain.infrastructure._internal.hf_model_worker import hf_model_loader

        model, tokenizer = hf_model_loader(
            model_id=self._config.hf_model_kwargs.get("model_id", self._config.model_id),
            device=self._config.hf_model_kwargs.get("device", "cpu"),
        )
        self._hb_q.put_nowait(("ready", None))
        self._provider = None  # HF uses model+tokenizer directly
        self._hf_model = model
        self._hf_tokenizer = tokenizer
        self._request_loop()

    def _request_loop(self) -> None:
        """Process requests from req_q until stop."""
        while True:
            try:
                cmd, payload = self._req_q.get(timeout=0.5)
            except queue.Empty:
                continue

            if cmd == "stop":
                break

            if cmd == "generate":
                session_id, prompt, kwargs = payload
                try:
                    result = self._generate_fn(prompt, **kwargs)
                    self._resp_q.put_nowait(("result", session_id, result))
                except Exception as e:
                    self._resp_q.put_nowait(("error", session_id, str(e)))

            elif cmd == "generate_stream":
                session_id, prompt, kwargs = payload
                try:
                    for tok in self._stream_fn(prompt, **kwargs):
                        self._resp_q.put_nowait(("token", session_id, tok))
                    self._resp_q.put_nowait(("result", session_id, {}))
                except Exception as e:
                    self._resp_q.put_nowait(("error", session_id, str(e)))

    def _generate_fn(self, prompt: str, **kwargs) -> dict:
        import numpy as np

        provider = self._provider
        token_ids = provider._tokenizer.encode(prompt)
        input_ids = np.array([token_ids], dtype=np.int64)
        result = provider._model.generate_numpy(
            input_ids,
            max_new_tokens=kwargs.get("max_new_tokens", 100),
            temperature=kwargs.get("temperature", 0.7),
            top_k=kwargs.get("top_k", 50),
            top_p=kwargs.get("top_p", 0.9),
            repetition_penalty=kwargs.get("repetition_penalty", 1.0),
            eos_token=provider._tokenizer.eos_token_id or 0,
        )
        generated = result[0].tolist()
        text = provider._tokenizer.decode(generated)
        return {"text": text, "tokens_generated": len(generated), "elapsed_ms": 0}

    def _stream_fn(self, prompt: str, **kwargs):
        import numpy as np

        provider = self._provider
        token_ids = provider._tokenizer.encode(prompt)
        input_ids = np.array([token_ids], dtype=np.int64)
        for tok_id in provider._model.generate_numpy_stream(
            input_ids,
            max_new_tokens=kwargs.get("max_new_tokens", 100),
            eos_token=provider._tokenizer.eos_token_id or 0,
            temperature=kwargs.get("temperature", 0.7),
            top_k=kwargs.get("top_k", 50),
            top_p=kwargs.get("top_p", 0.9),
            repetition_penalty=kwargs.get("repetition_penalty", 1.0),
        ):
            yield provider._tokenizer.decode([tok_id])


_session_counter = 0
_session_lock = threading.Lock()


def _new_session_id() -> int:
    global _session_counter
    with _session_lock:
        _session_counter += 1
        return _session_counter


# ── ProcessGuard ───────────────────────────────────────────────────────


class ProcessGuard:
    """Crash guard for model inference — thread or subprocess mode.

    Wraps model inference with auto-restart, health monitoring, and
    lifecycle callbacks. Execution mode determines whether inference
    runs in a thread (no IPC, for PGQ/autoload) or subprocess (full
    isolation, for manual API load).

    Usage::

        config = ModelConfig(slnc_path="models/gpt2.slnc", model_id="gpt2")
        guard = ProcessGuard(config, mode=ExecutionMode.THREAD)
        guard.start()
        result = guard.generate("Hello")
        guard.stop()
    """

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        mode: ExecutionMode = ExecutionMode.THREAD,
        worker_id: str = "guard",
        generate_timeout: float = DEFAULT_GENERATE_TIMEOUT,
        stall_timeout: float = DEFAULT_STALL_TIMEOUT,
        max_restarts: int = 3,
        restart_delay: float = 1.0,
        health_check_interval: float = 1.0,
        max_concurrent: int | None = None,
        memory_limit_mb: float | None = 4096.0,
        extra_sys_paths: list | None = None,
        # Legacy params (backward compat — used when model_config is None)
        slnc_path: str | None = None,
        model_id: str | None = None,
        quantize: bool = False,
        quant_bits: int = 8,
        quant_mode: str = "symmetric",
        quant_clip: float = 0.999,
        model_cls_path: str | None = None,
        model_kwargs: dict | None = None,
    ):
        # Build ModelConfig from legacy params if not provided
        if model_config is None:
            model_config = ModelConfig(
                slnc_path=slnc_path,
                model_id=model_id or "default",
                quantize=quantize,
                quant_bits=quant_bits,
                quant_mode=quant_mode,
                quant_clip=quant_clip,
                hf_model_cls_path=model_cls_path,
                hf_model_kwargs=model_kwargs or {},
            )

        self._config = model_config
        self._mode = mode
        self.worker_id = worker_id
        self.generate_timeout = generate_timeout
        self.stall_timeout = stall_timeout
        self.max_restarts = max_restarts
        self.restart_delay = restart_delay
        self.health_check_interval = health_check_interval
        self.memory_limit_mb = memory_limit_mb
        self._extra_sys_paths = extra_sys_paths or []

        self._worker: Any | None = None
        self._restart_count = 0
        self._requests_served = 0
        self._requests_served_lock = threading.Lock()
        self._crash_callbacks: list[Callable[[str], None]] = []
        self._restart_callbacks: list[Callable[[str], None]] = []
        self._monitor_thread: threading.Thread | None = None
        self._stop_monitor = threading.Event()
        self._restart_lock = threading.Lock()

        if max_concurrent is None:
            from .resource_manager import get_resource_manager

            max_concurrent = get_resource_manager().process_guard_concurrent
        self._semaphore = threading.Semaphore(max_concurrent)

    @property
    def alive(self) -> bool:
        return self._worker is not None and self._worker.alive

    def start(self) -> None:
        logger.info(
            "process_guard: starting (%s mode)",
            self._mode.value,
            extra={"worker_id": self.worker_id, "model_id": self._config.model_id},
        )
        self._launch_worker()
        self._stop_monitor.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name=f"guard-mon-{self.worker_id}"
        )
        self._monitor_thread.start()

    def stop(self) -> None:
        logger.info(
            "process_guard: stopping",
            extra={"worker_id": self.worker_id, "model_id": self._config.model_id},
        )
        self._stop_monitor.set()
        if self._worker is not None:
            self._worker.stop()
            self._worker = None
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=3.0)
            self._monitor_thread = None

    def generate(self, prompt: str, **kwargs: Any) -> dict:
        if not self.alive:
            raise RuntimeError(f"Guard worker [{self.worker_id}] is not alive")
        with self._semaphore:
            try:
                result = self._worker.generate(prompt, **kwargs)
            except (TimeoutError, Exception) as e:
                if "stall" in str(e).lower() or "timeout" in str(e).lower():
                    self._recover_from_stall()
                raise
        with self._requests_served_lock:
            self._requests_served += 1
        return result

    def generate_stream(self, prompt: str, **kwargs: Any) -> Generator[str, None, dict]:
        if not self.alive:
            raise RuntimeError(f"Guard worker [{self.worker_id}] is not alive")
        with self._semaphore:
            gen = self._worker.generate_stream(prompt, **kwargs)
            try:
                token = next(gen)
                while True:
                    yield token
                    token = next(gen)
            except StopIteration as e:
                return e.value if hasattr(e, "value") else {}
            except (TimeoutError, Exception) as e:
                if "stall" in str(e).lower() or "timeout" in str(e).lower():
                    self._recover_from_stall()
                raise

    def _recover_from_stall(self) -> None:
        """Restart a wedged worker (stalled queue writes / no messages)."""
        with self._restart_lock:
            if self._restart_count >= self.max_restarts:
                logger.error(
                    "ProcessGuard[%s]: worker stalled and restart budget exhausted",
                    self.worker_id,
                )
                raise RuntimeError(
                    f"ProcessGuard[{self.worker_id}]: worker restart budget exhausted "
                    f"({self.max_restarts} restarts)"
                )
            self._restart_worker_locked("stalled")

    def health(self) -> dict:
        return {
            "alive": self.alive,
            "worker_id": self.worker_id,
            "mode": self._mode.value,
            "model_id": self._config.model_id,
            "requests_served": self._requests_served,
            "restart_count": self._restart_count,
            "max_restarts": self.max_restarts,
            "exhausted": self._restart_count >= self.max_restarts,
            "memory_limit_mb": self.memory_limit_mb,
        }

    def on_crash(self, cb: Callable[[str], None]) -> None:
        with self._restart_lock:
            self._crash_callbacks.append(cb)

    def on_restart(self, cb: Callable[[str], None]) -> None:
        with self._restart_lock:
            self._restart_callbacks.append(cb)

    def load_adapter(self, adapter_path: str, merge: bool = False, timeout: float = 120.0) -> dict:
        if self._worker is None or not self.alive:
            raise RuntimeError("Worker is not alive — cannot load adapter.")
        if hasattr(self._worker, "load_adapter"):
            return self._worker.load_adapter(adapter_path, merge=merge, timeout=timeout)
        raise RuntimeError("Adapter loading not supported in this mode")

    def unload_adapter(self, timeout: float = 60.0) -> dict:
        if self._worker is None or not self.alive:
            raise RuntimeError("Worker is not alive — cannot unload adapter.")
        if hasattr(self._worker, "unload_adapter"):
            return self._worker.unload_adapter(timeout=timeout)
        raise RuntimeError("Adapter unloading not supported in this mode")

    def _memory_mb(self) -> float | None:
        if self._worker is None:
            return None
        try:
            import psutil
        except ImportError:
            return None
        proc_ref = getattr(self._worker, "_process", None)
        if proc_ref is None:
            return None
        try:
            proc = psutil.Process(proc_ref.pid)
            return proc.memory_info().rss / (1024 * 1024)
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            return None

    # ── Private ──────────────────────────────────────────────────────

    def _launch_worker(self) -> None:
        if self._worker is not None:
            self._worker.stop()

        if self._mode == ExecutionMode.THREAD:
            self._worker = _ThreadWorker(
                model_config=self._config,
                worker_id=self.worker_id,
                generate_timeout=self.generate_timeout,
            )
        else:
            from .model_worker import ModelWorkerProcess

            if self._config.is_slo:
                self._worker = ModelWorkerProcess(
                    slnc_path=self._config.slnc_path,
                    model_id=self._config.model_id,
                    worker_id=self.worker_id,
                    generate_timeout=self.generate_timeout,
                    stall_timeout=self.stall_timeout,
                    extra_sys_paths=self._extra_sys_paths,
                    quantize=self._config.quantize,
                    quant_bits=self._config.quant_bits,
                    quant_mode=self._config.quant_mode,
                    quant_clip=self._config.quant_clip,
                )
            else:
                self._worker = ModelWorkerProcess(
                    model_cls_path=self._config.hf_model_cls_path,
                    model_kwargs=self._config.hf_model_kwargs,
                    worker_id=self.worker_id,
                    generate_timeout=self.generate_timeout,
                    stall_timeout=self.stall_timeout,
                    extra_sys_paths=self._extra_sys_paths,
                )

        self._worker.start()
        logger.info(
            "process_guard: worker launched (%s)",
            self._mode.value,
            extra={"worker_id": self.worker_id, "model_id": self._config.model_id},
        )

    def _restart_worker(self, reason: str, fire_callbacks: bool = False) -> None:
        with self._restart_lock:
            self._restart_worker_locked(reason, fire_callbacks=fire_callbacks)

    def _restart_worker_locked(self, reason: str, fire_callbacks: bool = False) -> None:
        if fire_callbacks:
            for cb in self._crash_callbacks:
                try:
                    cb(self.worker_id)
                except Exception:
                    logger.exception("ProcessGuard crash callback failed")
        logger.info(
            "ProcessGuard[%s]: restarting worker (%s) (%d/%d)",
            self.worker_id,
            reason,
            self._restart_count + 1,
            self.max_restarts,
        )
        time.sleep(self.restart_delay)
        self._launch_worker()
        self._restart_count += 1
        if fire_callbacks:
            for cb in self._restart_callbacks:
                try:
                    cb(self.worker_id)
                except Exception:
                    logger.exception("ProcessGuard restart callback failed")

    def _monitor_loop(self) -> None:
        while not self._stop_monitor.is_set():
            time.sleep(self.health_check_interval)
            if self._stop_monitor.is_set():
                break
            with self._restart_lock:
                worker = self._worker
                restarts_left = self._restart_count < self.max_restarts
            if worker is None:
                continue
            if not worker.alive and restarts_left:
                self._restart_worker("died", fire_callbacks=True)
            elif not worker.alive and not restarts_left:
                logger.error(
                    "ProcessGuard[%s]: worker dead, restart budget exhausted",
                    self.worker_id,
                )


# ── Factory functions (backward compat) ────────────────────────────────


def create_model_guard(
    model_id: str,
    device: str = "cpu",
    worker_id: str | None = None,
    max_restarts: int = 3,
    restart_delay: float = 2.0,
    memory_limit_mb: float | None = None,
    generate_timeout: float = DEFAULT_GENERATE_TIMEOUT,
    stall_timeout: float = DEFAULT_STALL_TIMEOUT,
    max_concurrent: int | None = None,
) -> ProcessGuard:
    """Create a ProcessGuard for an HF model (legacy path)."""
    config = ModelConfig(
        hf_model_cls_path="domains.infrastructure.hf_model_worker.hf_model_loader",
        hf_model_kwargs={"model_id": model_id, "device": device},
        model_id=model_id,
    )
    guard = ProcessGuard(
        model_config=config,
        mode=ExecutionMode.SUBPROCESS,
        worker_id=worker_id or f"guard-{model_id.split('/')[-1]}",
        max_restarts=max_restarts,
        restart_delay=restart_delay,
        memory_limit_mb=memory_limit_mb,
        generate_timeout=generate_timeout,
        stall_timeout=stall_timeout,
        max_concurrent=max_concurrent,
    )
    guard.start()
    return guard


def create_slo_guard(
    slnc_path: str,
    model_id: str = "default",
    worker_id: str | None = None,
    max_restarts: int = 3,
    restart_delay: float = 2.0,
    memory_limit_mb: float | None = None,
    generate_timeout: float = DEFAULT_GENERATE_TIMEOUT,
    stall_timeout: float = DEFAULT_STALL_TIMEOUT,
    max_concurrent: int | None = None,
    quantize: bool = False,
    quant_bits: int = 8,
    quant_mode: str = "symmetric",
    quant_clip: float = 0.999,
) -> ProcessGuard:
    """Create a ProcessGuard for a SloNet model (pure NumPy)."""
    config = ModelConfig(
        slnc_path=slnc_path,
        model_id=model_id,
        quantize=quantize,
        quant_bits=quant_bits,
        quant_mode=quant_mode,
        quant_clip=quant_clip,
    )
    guard = ProcessGuard(
        model_config=config,
        mode=ExecutionMode.SUBPROCESS,
        worker_id=worker_id or f"slo-guard-{model_id.split('/')[-1]}",
        max_restarts=max_restarts,
        restart_delay=restart_delay,
        memory_limit_mb=memory_limit_mb,
        generate_timeout=generate_timeout,
        stall_timeout=stall_timeout,
        max_concurrent=max_concurrent,
    )
    guard.start()
    return guard
