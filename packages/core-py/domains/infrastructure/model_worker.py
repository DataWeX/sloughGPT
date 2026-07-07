"""
Process-level isolation for model inference.

Provides ModelWorkerProcess — runs any callable model in a child process
with Queue-based RPC, timeout, crash detection, and automatic restart.

Architecture::

    API Process                Worker Process
    ┌──────────────┐           ┌─────────────────┐
    │ ProcessGuard │───req_q──▶│ ModelWorkerProcess │
    │              │◀──resp_q──│  model.generate()  │
    │ health Q     │───hb_q──▶│  (heartbeat tick)  │
    └──────────────┘           └─────────────────┘
          │
          │ wraps
          ▼
    ┌──────────────┐
    │  ModelServer │  (unchanged)
    └──────────────┘
"""

import multiprocessing as mp
import time
import logging
import os
import queue
import gc
import traceback
import sys
import threading
from typing import Any, Optional, Callable, Generator
from dataclasses import dataclass, field

logger = logging.getLogger("man.infrastructure.model_worker")

_ctx = mp.get_context("spawn")  # spawn avoids fork-safety issues with torch


@dataclass
class WorkerHealth:
    """Health snapshot of a worker process."""
    pid: Optional[int] = None
    alive: bool = False
    started_at: float = 0.0
    last_heartbeat: float = 0.0
    requests_served: int = 0
    errors: int = 0
    crashed: bool = False
    crash_count: int = 0


def _worker_main(
    req_q: mp.Queue,
    resp_q: mp.Queue,
    hb_q: mp.Queue,
    model_cls_path: str,
    model_kwargs: dict,
    worker_id: str,
    extra_sys_paths: Optional[list] = None,
) -> None:
    """Entry point for the worker subprocess.

    Loads the model, then loops on request queue until ``stop`` sentinel.
    """
    if extra_sys_paths:
        for p in extra_sys_paths:
            if p not in sys.path:
                sys.path.insert(0, p)
    logger.info("Worker[%s]: started (pid=%d)", worker_id, os.getpid())

    model = None
    tokenizer = None

    try:
        # Dynamic import for the model class
        import importlib
        mod_path, cls_name = model_cls_path.rsplit(".", 1)
        module = importlib.import_module(mod_path)
        model_cls = getattr(module, cls_name)

        model = model_cls(**model_kwargs)
        # If model_cls returns (model, tokenizer) tuple, unpack
        if isinstance(model, (list, tuple)) and len(model) == 2:
            model, tokenizer = model

        logger.info("Worker[%s]: model loaded (type=%s)", worker_id, type(model).__name__)

        # If the model has a .tokenizer or ._tokenizer attr, extract it
        if tokenizer is None:
            tokenizer = getattr(model, "tokenizer", None) or getattr(model, "_tokenizer", None)

    except Exception as e:
        logger.error("Worker[%s]: model load failed: %s", worker_id, e)
        resp_q.put_nowait(("error", f"Model load failed: {e}"))
        hb_q.put_nowait(("dead", os.getpid()))
        return

    hb_q.put_nowait(("ready", os.getpid()))

    def _generate(
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        **gen_kwargs: Any,
    ) -> dict:
        inputs = tokenizer(prompt, return_tensors="pt")  # noqa: F821
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        device = getattr(model, "device", None)  # noqa: F821
        if device is not None and device != "cpu":
            input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

        gen_kwargs.update(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,  # noqa: F821
            eos_token_id=tokenizer.eos_token_id,  # noqa: F821
        )
        start = time.time()
        output_ids = model.generate(**gen_kwargs)  # noqa: F821
        elapsed_ms = (time.time() - start) * 1000
        generated = output_ids[0][input_ids.shape[1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True)  # noqa: F821
        return {
            "text": text,
            "tokens_generated": len(generated),
            "elapsed_ms": round(elapsed_ms, 1),
        }

    def _generate_stream(
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        **gen_kwargs: Any,
    ) -> None:
        """Generate tokens and stream each one through resp_q."""
        from transformers import TextIteratorStreamer

        inputs = tokenizer(prompt, return_tensors="pt")  # noqa: F821
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        device = getattr(model, "device", None)  # noqa: F821
        if device is not None and device != "cpu":
            input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True  # noqa: F821
        )

        gen_kwargs.update(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,  # noqa: F821
            eos_token_id=tokenizer.eos_token_id,  # noqa: F821
            streamer=streamer,
        )

        # Launch generation in background thread
        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)  # noqa: F821
        thread.start()

        tokens_generated = 0
        start = time.time()
        for text in streamer:
            try:
                resp_q.put_nowait(("token", text))
            except Exception:
                break
            tokens_generated += 1

        thread.join(timeout=10)
        elapsed_ms = (time.time() - start) * 1000

        try:
            resp_q.put_nowait(("result", {
                "text": "",
                "tokens_generated": tokens_generated,
                "elapsed_ms": round(elapsed_ms, 1),
            }))
        except Exception:
            pass

    requests_served = 0

    while True:
        try:
            cmd, payload = req_q.get(timeout=0.5)
        except queue.Empty:
            hb_q.put_nowait(("alive", os.getpid()))
            continue

        if cmd == "stop":
            logger.info("Worker[%s]: stop requested", worker_id)
            break

        if cmd == "generate":
            try:
                prompt, kwargs = payload
                result = _generate(prompt, **kwargs)
                try:
                    resp_q.put_nowait(("result", result))
                except Exception as put_e:
                    logger.error("Worker[%s]: resp_q.put_nowait failed: %s", worker_id, put_e)
                    resp_q.put_nowait(("error", f"put failed: {put_e}"))
                requests_served += 1
            except Exception as e:
                tb = traceback.format_exc()
                logger.error("Worker[%s]: generate error: %s\n%s", worker_id, e, tb)
                try:
                    resp_q.put_nowait(("error", f"{type(e).__name__}: {e}"))
                except Exception:
                    logger.error("Worker[%s]: failed to send error response", worker_id)

        if cmd == "generate_stream":
            try:
                prompt, kwargs = payload
                _generate_stream(prompt, **kwargs)
                requests_served += 1
            except Exception as e:
                tb = traceback.format_exc()
                logger.error("Worker[%s]: generate_stream error: %s\n%s", worker_id, e, tb)
                try:
                    resp_q.put_nowait(("error", f"{type(e).__name__}: {e}"))
                except Exception:
                    logger.error("Worker[%s]: failed to send error response", worker_id)

    # Cleanup
    del model
    del tokenizer
    gc.collect()
    hb_q.put_nowait(("dead", os.getpid()))
    logger.info("Worker[%s]: stopped", worker_id)


class ModelWorkerProcess:
    """Manages a model inference subprocess with Queue-based RPC.

    Usage::

        worker = ModelWorkerProcess(
            model_cls_path="transformers.AutoModelForCausalLM",
            model_kwargs={"from_pretrained": "gpt2"},
            worker_id="gpt2",
        )
        worker.start()
        result = worker.generate("Hello", max_new_tokens=50)
    """

    def __init__(
        self,
        model_cls_path: str,
        model_kwargs: dict,
        worker_id: str = "worker",
        generate_timeout: float = 120.0,
        extra_sys_paths: Optional[list] = None,
    ):
        self.model_cls_path = model_cls_path
        self.model_kwargs = model_kwargs
        self.worker_id = worker_id
        self._generate_timeout = generate_timeout
        self._extra_sys_paths = extra_sys_paths or []

        self._req_q: Optional[mp.Queue] = None
        self._resp_q: Optional[mp.Queue] = None
        self._hb_q: Optional[mp.Queue] = None
        self._process: Optional[mp.Process] = None
        self._health = WorkerHealth()
        self._started_at: float = 0.0
        self._lock = mp.Lock() if hasattr(mp, "Lock") else None  # type: ignore

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(self) -> None:
        """Launch the worker subprocess."""
        if self._process is not None and self._process.is_alive():
            logger.warning("Worker[%s]: already running", self.worker_id)
            return

        self._req_q = mp.Queue()
        self._resp_q = mp.Queue()
        self._hb_q = mp.Queue()

        self._health = WorkerHealth()
        self._started_at = time.time()

        self._process = _ctx.Process(
            target=_worker_main,
            args=(
                self._req_q,
                self._resp_q,
                self._hb_q,
                self.model_cls_path,
                self.model_kwargs,
                self.worker_id,
                self._extra_sys_paths,
            ),
            daemon=True,
        )
        self._process.start()

        self._health.pid = self._process.pid
        self._health.alive = True
        self._health.started_at = self._started_at

        # Wait for ready signal
        deadline = time.time() + 60.0
        ready = False
        while time.time() < deadline:
            try:
                msg, val = self._hb_q.get(timeout=0.5)
                if msg == "ready":
                    self._health.last_heartbeat = time.time()
                    ready = True
                    break
                elif msg == "dead":
                    break
            except queue.Empty:
                if not self._process.is_alive():
                    break
                # Poll heartbeat — worker sends "alive" every 0.5s once running
                continue

        if not ready:
            raise RuntimeError(
                f"Worker[{self.worker_id}]: failed to start within 60s"
            )

        logger.info(
            "Worker[%s]: ready (pid=%d)", self.worker_id, self._process.pid
        )

    def stop(self, timeout: float = 10.0) -> None:
        """Stop the worker subprocess."""
        if self._process is None:
            return
        try:
            self._req_q.put_nowait(("stop", None))
        except Exception:
            pass
        self._process.join(timeout=timeout)
        if self._process.is_alive():
            logger.warning("Worker[%s]: killing unresponsive process", self.worker_id)
            self._process.kill()
            self._process.join(timeout=5)
        self._health.alive = False
        self._cleanup_queues()
        logger.info("Worker[%s]: stopped", self.worker_id)

    def _cleanup_queues(self) -> None:
        for q_name in ("_req_q", "_resp_q", "_hb_q"):
            q = getattr(self, q_name, None)
            if q is not None:
                try:
                    q.close()
                except Exception:
                    pass
                try:
                    q.join_thread()
                except Exception:
                    pass

    # ── Health ─────────────────────────────────────────────────────────

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def health_check(self) -> WorkerHealth:
        """Return current health snapshot, polling heartbeat queue."""
        now = time.time()
        # Drain heartbeat queue
        while self._hb_q is not None:
            try:
                msg, val = self._hb_q.get_nowait()
                if msg == "alive":
                    self._health.last_heartbeat = now
                    self._health.alive = True
                elif msg == "dead":
                    self._health.alive = False
                    self._health.crashed = True
                elif msg == "ready":
                    self._health.last_heartbeat = now
            except queue.Empty:
                break
            except (OSError, ValueError):
                self._health.alive = False
                break

        # Double-check process state
        if self._process is not None:
            was_alive = self._health.alive
            proc_alive = self._process.is_alive()
            self._health.alive = proc_alive
            if was_alive and not self._health.alive:
                self._health.crashed = True
                self._health.crash_count += 1

        return WorkerHealth(
            pid=self._health.pid,
            alive=self._health.alive,
            started_at=self._health.started_at,
            last_heartbeat=self._health.last_heartbeat,
            requests_served=self._health.requests_served,
            errors=self._health.errors,
            crashed=self._health.crashed,
            crash_count=self._health.crash_count,
        )

    # ── Generate ───────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        **kwargs: Any,
    ) -> dict:
        """Send generate request to worker and wait for result.

        Returns::

            {"text": str, "tokens_generated": int, "elapsed_ms": float}

        Raises ``TimeoutError`` if generation exceeds timeout,
        ``RuntimeError`` if worker crashes.
        """
        if not self.alive:
            raise RuntimeError(f"Worker[{self.worker_id}] is not alive")

        payload = dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            **kwargs,
        )

        try:
            self._req_q.put_nowait(("generate", (prompt, payload)))
        except Exception as e:
            self._health.errors += 1
            raise RuntimeError(f"Failed to send request to worker: {e}")

        # Wait for response
        deadline = time.time() + self._generate_timeout
        while time.time() < deadline:
            # Check worker alive
            if not self.alive:
                self._health.crashed = True
                self._health.crash_count += 1
                raise RuntimeError(f"Worker[{self.worker_id}] crashed during generation")

            try:
                msg, data = self._resp_q.get(timeout=0.2)
            except queue.Empty:
                continue

            if msg == "result":
                self._health.requests_served += 1
                return data
            elif msg == "error":
                self._health.errors += 1
                raise RuntimeError(f"Worker generate error: {data}")
            else:
                logger.warning("Worker[%s]: unknown response type: %s", self.worker_id, msg)
                continue

        self._health.errors += 1
        raise TimeoutError(
            f"Worker[{self.worker_id}] generation timed out after {self._generate_timeout}s"
        )

    # ── Generate (streaming) ────────────────────────────────────────────

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        **kwargs: Any,
    ) -> Generator[str, None, dict]:
        """Send streaming generate request to worker, yield tokens.

        Yields:
            str: Each decoded token as it is produced.

        Returns:
            dict: Final result with ``tokens_generated`` and ``elapsed_ms``.
        """
        if not self.alive:
            raise RuntimeError(f"Worker[{self.worker_id}] is not alive")

        payload = dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            **kwargs,
        )

        try:
            self._req_q.put_nowait(("generate_stream", (prompt, payload)))
        except Exception as e:
            self._health.errors += 1
            raise RuntimeError(f"Failed to send stream request to worker: {e}")

        # Read tokens from response queue until result or error
        deadline = time.time() + self._generate_timeout
        final_result: dict = {}

        while time.time() < deadline:
            if not self.alive:
                self._health.crashed = True
                self._health.crash_count += 1
                raise RuntimeError(
                    f"Worker[{self.worker_id}] crashed during streaming generation"
                )

            try:
                msg, data = self._resp_q.get(timeout=0.2)
            except queue.Empty:
                continue

            if msg == "token":
                yield data
            elif msg == "result":
                self._health.requests_served += 1
                final_result = data
                break
            elif msg == "error":
                self._health.errors += 1
                raise RuntimeError(f"Worker generate_stream error: {data}")
            else:
                logger.warning(
                    "Worker[%s]: unknown response type: %s", self.worker_id, msg
                )
                continue

        return final_result

    # ── Context manager ────────────────────────────────────────────────

    def __enter__(self) -> "ModelWorkerProcess":
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()
