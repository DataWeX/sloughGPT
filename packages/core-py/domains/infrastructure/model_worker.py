"""
Process-level isolation for model inference.

Provides ModelWorkerProcess — runs any callable model in a child process
with Queue-based RPC, timeout, crash detection, and automatic restart.

Two worker backends:
  - ``_hf_worker_main``: Legacy HuggingFace/PyTorch path (deprecated)
  - ``_slo_worker_main``: SloNet pure-NumPy path (preferred)

Architecture::

    API Process                Worker Process
    ┌──────────────┐           ┌──────────────────────┐
    │ ProcessGuard │───req_q──▶│ ModelWorkerProcess   │
    │              │◀──resp_q──│  slo/generate_numpy  │
    │ health Q     │───hb_q──▶│  (heartbeat tick)    │
    └──────────────┘           └──────────────────────┘
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

logger = logging.getLogger("slo.infrastructure.model_worker")

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


def _worker_loop(
    req_q: mp.Queue,
    resp_q: mp.Queue,
    hb_q: mp.Queue,
    worker_id: str,
    generate_fn,
    stream_fn,
    cleanup_fn=None,
) -> None:
    """Shared request loop for both HF and SloNet workers.

    Args:
        req_q: Request queue (commands from parent)
        resp_q: Response queue (tokens/results to parent)
        hb_q: Heartbeat queue (health signals to parent)
        worker_id: Human-readable worker name
        generate_fn: ``fn(prompt, **kwargs) -> dict`` for non-streaming
        stream_fn: ``fn(prompt, resp_q, **kwargs)`` for streaming (puts tokens directly)
        cleanup_fn: Optional cleanup callable on shutdown
    """
    hb_q.put_nowait(("ready", os.getpid()))
    requests_served = 0

    while True:
        try:
            cmd, payload = req_q.get(timeout=0.5)
        except queue.Empty:
            hb_q.put_nowait(("alive", os.getpid()))
            continue

        if cmd == "stop":
            logger.info("Worker[%s]: stop requested", worker_id,
                extra={"tag": "INFRA"})
            break

        if cmd == "generate":
            try:
                prompt, kwargs = payload
                result = generate_fn(prompt, **kwargs)
                try:
                    resp_q.put_nowait(("result", result))
                except Exception as put_e:
                    logger.error("Worker[%s]: resp_q.put failed: %s", worker_id, put_e,
                        extra={"tag": "INFRA"})
                    resp_q.put_nowait(("error", f"put failed: {put_e}"))
                requests_served += 1
            except Exception as e:
                tb = traceback.format_exc()
                logger.error("Worker[%s]: generate error: %s\n%s", worker_id, e, tb,
                    extra={"tag": "INFRA"})
                try:
                    resp_q.put_nowait(("error", f"{type(e).__name__}: {e}"))
                except Exception:
                    pass

        if cmd == "generate_stream":
            try:
                prompt, kwargs = payload
                stream_fn(prompt, resp_q, **kwargs)
                requests_served += 1
            except Exception as e:
                tb = traceback.format_exc()
                logger.error("Worker[%s]: stream error: %s\n%s", worker_id, e, tb,
                    extra={"tag": "INFRA"})
                try:
                    resp_q.put_nowait(("error", f"{type(e).__name__}: {e}"))
                except Exception:
                    pass

    if cleanup_fn:
        try:
            cleanup_fn()
        except Exception:
            pass
    gc.collect()
    hb_q.put_nowait(("dead", os.getpid()))
    logger.info("Worker[%s]: stopped", worker_id,
        extra={"tag": "INFRA"})


# ---------------------------------------------------------------------------
# SloNet worker (pure NumPy — no PyTorch dependency)
# ---------------------------------------------------------------------------

def _slo_worker_main(
    req_q: mp.Queue,
    resp_q: mp.Queue,
    hb_q: mp.Queue,
    slnc_path: str,
    model_id: str,
    worker_id: str,
    quantize: bool = False,
    quant_bits: int = 8,
    quant_mode: str = "symmetric",
    quant_clip: float = 0.999,
    extra_sys_paths: Optional[list] = None,
) -> None:
    """Worker subprocess entry point for SloNet models (pure NumPy).

    Loads via ``SloNetChatProvider.from_slnc()`` and streams via
    ``generate_numpy_stream()``. Zero PyTorch dependency.
    """
    if extra_sys_paths:
        for p in extra_sys_paths:
            if p not in sys.path:
                sys.path.insert(0, p)

    logger.info("Worker[%s]: started (pid=%d, slo)", worker_id, os.getpid(),
        extra={"tag": "INFRA"})

    provider = None

    try:
        from domains.inference.slonet_provider import SloNetChatProvider

        provider = SloNetChatProvider.from_slnc(
            slnc_path,
            model_id=model_id,
            quantize=quantize,
            quant_bits=quant_bits,
            quant_mode=quant_mode,
            quant_clip=quant_clip,
        )
        logger.info("Worker[%s]: SloNet model loaded (%s)", worker_id, model_id,
            extra={"tag": "INFRA"})
    except Exception as e:
        logger.error("Worker[%s]: SloNet load failed: %s", worker_id, e,
            extra={"tag": "INFRA"})
        resp_q.put_nowait(("error", f"Model load failed: {e}"))
        hb_q.put_nowait(("dead", os.getpid()))
        return

    import numpy as _np

    def _generate(
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        **_kwargs: Any,
    ) -> dict:
        token_ids = provider._tokenizer.encode(prompt)
        input_ids = _np.array([token_ids], dtype=_np.int64)
        start = time.time()
        result = provider._model.generate_numpy(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token=provider._tokenizer.eos_token_id or 0,
        )
        elapsed_ms = (time.time() - start) * 1000
        generated = result[0].tolist()
        text = provider._tokenizer.decode(generated)
        return {
            "text": text,
            "tokens_generated": len(generated),
            "elapsed_ms": round(elapsed_ms, 1),
        }

    def _stream(
        prompt: str,
        resp_q_inner: mp.Queue,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        **_kwargs: Any,
    ) -> None:
        """Stream tokens via ``generate_numpy_stream()`` into resp_q."""
        token_ids = provider._tokenizer.encode(prompt)
        input_ids = _np.array([token_ids], dtype=_np.int64)
        start = time.time()
        tokens_generated = 0

        for tok_id in provider._model.generate_numpy_stream(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token=provider._tokenizer.eos_token_id or 0,
        ):
            decoded = provider._tokenizer.decode([tok_id])
            if decoded:
                try:
                    resp_q_inner.put_nowait(("token", decoded))
                except Exception:
                    break
                tokens_generated += 1

        elapsed_ms = (time.time() - start) * 1000
        try:
            resp_q_inner.put_nowait(("result", {
                "text": "",
                "tokens_generated": tokens_generated,
                "elapsed_ms": round(elapsed_ms, 1),
            }))
        except Exception:
            pass

    def _cleanup():
        nonlocal provider
        provider = None

    _worker_loop(req_q, resp_q, hb_q, worker_id, _generate, _stream, _cleanup)


# ---------------------------------------------------------------------------
# HF worker (legacy — requires PyTorch + transformers)
# ---------------------------------------------------------------------------

def _hf_worker_main(
    req_q: mp.Queue,
    resp_q: mp.Queue,
    hb_q: mp.Queue,
    model_cls_path: str,
    model_kwargs: dict,
    worker_id: str,
    extra_sys_paths: Optional[list] = None,
) -> None:
    """Worker subprocess entry point for HuggingFace models (legacy).

    Loads via dynamic import of ``model_cls_path`` and uses HF
    ``model.generate()`` / ``TextIteratorStreamer``. Requires PyTorch.
    """
    if extra_sys_paths:
        for p in extra_sys_paths:
            if p not in sys.path:
                sys.path.insert(0, p)
    logger.info("Worker[%s]: started (pid=%d, hf)", worker_id, os.getpid(),
        extra={"tag": "INFRA"})

    model = None
    tokenizer = None

    try:
        import importlib
        mod_path, cls_name = model_cls_path.rsplit(".", 1)
        module = importlib.import_module(mod_path)
        model_cls = getattr(module, cls_name)

        model = model_cls(**model_kwargs)
        if isinstance(model, (list, tuple)) and len(model) == 2:
            model, tokenizer = model

        logger.info("Worker[%s]: HF model loaded (type=%s)", worker_id, type(model).__name__,
            extra={"tag": "INFRA"})

        if tokenizer is None:
            tokenizer = getattr(model, "tokenizer", None) or getattr(model, "_tokenizer", None)

    except Exception as e:
        logger.error("Worker[%s]: HF model load failed: %s", worker_id, e,
            extra={"tag": "INFRA"})
        resp_q.put_nowait(("error", f"Model load failed: {e}"))
        hb_q.put_nowait(("dead", os.getpid()))
        return

    def _generate(
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        **gen_kwargs: Any,
    ) -> dict:
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        device = getattr(model, "device", None)
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
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        start = time.time()
        output_ids = model.generate(**gen_kwargs)
        elapsed_ms = (time.time() - start) * 1000
        generated = output_ids[0][input_ids.shape[1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        return {
            "text": text,
            "tokens_generated": len(generated),
            "elapsed_ms": round(elapsed_ms, 1),
        }

    def _stream(
        prompt: str,
        resp_q_inner: mp.Queue,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        **gen_kwargs: Any,
    ) -> None:
        from transformers import TextIteratorStreamer

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        device = getattr(model, "device", None)
        if device is not None and device != "cpu":
            input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
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
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )

        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()

        tokens_generated = 0
        start = time.time()
        for text_chunk in streamer:
            try:
                resp_q_inner.put_nowait(("token", text_chunk))
            except Exception:
                break
            tokens_generated += 1

        thread.join(timeout=10)
        elapsed_ms = (time.time() - start) * 1000

        try:
            resp_q_inner.put_nowait(("result", {
                "text": "",
                "tokens_generated": tokens_generated,
                "elapsed_ms": round(elapsed_ms, 1),
            }))
        except Exception:
            pass

    def _cleanup():
        nonlocal model, tokenizer
        del model
        del tokenizer
        model = None
        tokenizer = None

    _worker_loop(req_q, resp_q, hb_q, worker_id, _generate, _stream, _cleanup)


class ModelWorkerProcess:
    """Manages a model inference subprocess with Queue-based RPC.

    Two construction modes:

    SloNet (preferred — pure NumPy, no PyTorch)::

        worker = ModelWorkerProcess(
            slnc_path="models/gpt2.slnc",
            model_id="gpt2",
            worker_id="gpt2",
        )

    HF/Legacy (requires PyTorch)::

        worker = ModelWorkerProcess(
            model_cls_path="transformers.AutoModelForCausalLM",
            model_kwargs={"pretrained_model_name_or_path": "gpt2"},
            worker_id="gpt2",
        )
    """

    def __init__(
        self,
        worker_id: str = "worker",
        generate_timeout: float = 120.0,
        extra_sys_paths: Optional[list] = None,
        # SloNet mode (preferred)
        slnc_path: Optional[str] = None,
        model_id: Optional[str] = None,
        quantize: bool = False,
        quant_bits: int = 8,
        quant_mode: str = "symmetric",
        quant_clip: float = 0.999,
        # HF/Legacy mode
        model_cls_path: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
    ):
        self.worker_id = worker_id
        self._generate_timeout = generate_timeout
        self._extra_sys_paths = extra_sys_paths or []

        # SloNet params
        self._slnc_path = slnc_path
        self._model_id = model_id or "default"
        self._quantize = quantize
        self._quant_bits = quant_bits
        self._quant_mode = quant_mode
        self._quant_clip = quant_clip

        # HF params
        self._model_cls_path = model_cls_path
        self._model_kwargs = model_kwargs or {}

        # Determine backend
        self._use_slo = slnc_path is not None
        if not self._use_slo and model_cls_path is None:
            raise ValueError(
                "ModelWorkerProcess requires either slnc_path (SloNet) or "
                "model_cls_path (HF legacy)"
            )

        self._req_q: Optional[mp.Queue] = None
        self._resp_q: Optional[mp.Queue] = None
        self._hb_q: Optional[mp.Queue] = None
        self._process: Optional[mp.Process] = None
        self._health = WorkerHealth()
        self._started_at: float = 0.0

    @property
    def backend(self) -> str:
        """Worker backend: ``'slo'`` or ``'hf'``."""
        return "slo" if self._use_slo else "hf"

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(self) -> None:
        """Launch the worker subprocess."""
        if self._process is not None and self._process.is_alive():
            logger.warning("Worker[%s]: already running", self.worker_id,
                extra={"tag": "INFRA"})
            return

        self._req_q = _ctx.Queue()
        self._resp_q = _ctx.Queue()
        self._hb_q = _ctx.Queue()

        self._health = WorkerHealth()
        self._started_at = time.time()

        if self._use_slo:
            target = _slo_worker_main
            args = (
                self._req_q,
                self._resp_q,
                self._hb_q,
                self._slnc_path,
                self._model_id,
                self.worker_id,
                self._quantize,
                self._quant_bits,
                self._quant_mode,
                self._quant_clip,
                self._extra_sys_paths,
            )
        else:
            target = _hf_worker_main
            args = (
                self._req_q,
                self._resp_q,
                self._hb_q,
                self._model_cls_path,
                self._model_kwargs,
                self.worker_id,
                self._extra_sys_paths,
            )

        self._process = _ctx.Process(target=target, args=args, daemon=True)
        self._process.start()

        self._health.pid = self._process.pid
        self._health.alive = True
        self._health.started_at = self._started_at

        # Wait for ready signal
        deadline = time.time() + 120.0
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
                continue

        if not ready:
            raise RuntimeError(
                f"Worker[{self.worker_id}]: failed to start within 120s"
            )

        logger.info(
            "Worker[%s]: ready (pid=%d, backend=%s)",
            self.worker_id, self._process.pid, self.backend,
            extra={"tag": "INFRA"},
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
            logger.warning("Worker[%s]: killing unresponsive process", self.worker_id,
                extra={"tag": "INFRA"})
            self._process.kill()
            self._process.join(timeout=5)
        self._health.alive = False
        self._cleanup_queues()
        logger.info("Worker[%s]: stopped", self.worker_id,
            extra={"tag": "INFRA"})

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
                logger.warning("Worker[%s]: unknown response type: %s", self.worker_id, msg,
                    extra={"tag": "INFRA"})
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
                    "Worker[%s]: unknown response type: %s", self.worker_id, msg,
                    extra={"tag": "INFRA"},
                )
                continue

        return final_result

    # ── Context manager ────────────────────────────────────────────────

    def __enter__(self) -> "ModelWorkerProcess":
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()
