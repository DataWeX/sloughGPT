"""
Process-level isolation for model inference.

Provides ModelWorkerProcess — runs any callable model in a child process
with Queue-based RPC, timeout, crash detection, and automatic restart.

Two worker backends:
  - ``_hf_worker_main``: Legacy HuggingFace/PyTorch path (deprecated)
  - ``_slo_worker_main``: SloNet pure-NumPy path (preferred)

IPC protocol: every request carries a unique ``session_id`` (parent→worker
via req_q). The worker tags each resp_q message ``(type, session_id, data)``
so the parent can discard stale results from abandoned sessions. Worker-side
queue writes are bounded (``_STREAM_PUT_TIMEOUT_S``) so a full pipe cannot
block the feeder thread forever; the parent treats a silent worker as wedged
(``WorkerStreamStalledError`` after ``_STALL_TIMEOUT_S``) so the guard can
restart it.

Architecture::

from __future__ import annotations

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

import itertools
import multiprocessing as mp
import time
import logging
import os
import queue
import gc
import traceback
import sys
import threading
from typing import Any, Optional, Generator
from dataclasses import dataclass

from .constants import DEFAULT_GENERATE_TIMEOUT, DEFAULT_STARTUP_TIMEOUT

logger = logging.getLogger("slo.infrastructure.model_worker")

_ctx = mp.get_context("spawn")  # spawn avoids fork-safety issues with torch

# Worker-side cap on a single queue write. If the parent stops draining
# resp_q (abandoned stream) the pipe fills; a bounded put prevents the
# worker's feeder thread from blocking forever in anon_pipe_write.
_STREAM_PUT_TIMEOUT_S = 30.0

# Parent-side: if the worker produces no message for this long mid-request,
# treat it as wedged (stale resp_q data + blocked feeder) and restart it.
_STALL_TIMEOUT_S = 30.0

_session_ids = itertools.count(1)


def _new_session_id() -> str:
    """Monotonic per-request id tagging IPC messages in resp_q.

    Lets the parent discard messages belonging to abandoned/other sessions
    instead of treating stale queue data as its own response.
    """
    return f"req-{os.getpid()}-{next(_session_ids)}"


class WorkerStreamStalledError(RuntimeError):
    """Raised when a worker stops producing messages mid-request (wedge)."""


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
    adapter_fn=None,
    unload_fn=None,
) -> None:  # pragma: no cover — child process only; coverage recorded in parent
    """Shared request loop for both HF and SloNet workers.

    Args:
        req_q: Request queue (commands from parent). Payloads are
            ``(session_id, prompt, kwargs)`` triples.
        resp_q: Response queue (tokens/results to parent). Messages are
            ``(type, session_id, data)`` triples.
        hb_q: Heartbeat queue (health signals to parent)
        worker_id: Human-readable worker name
        generate_fn: ``fn(prompt, **kwargs) -> dict`` for non-streaming
        stream_fn: ``fn(prompt, resp_q, session_id=..., **kwargs)`` for
            streaming (puts tagged tokens directly)
        cleanup_fn: Optional cleanup callable on shutdown
        adapter_fn: ``fn(adapter_path, merge) -> dict`` for loading LoRA adapters
        unload_fn: ``fn() -> dict`` for unloading LoRA adapters
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
            session_id = None
            try:
                session_id, prompt, kwargs = payload
                result = generate_fn(prompt, **kwargs)
                resp_q.put_nowait(("result", session_id, result))
                requests_served += 1
            except Exception as e:
                tb = traceback.format_exc()
                logger.error("Worker[%s]: generate error: %s\n%s", worker_id, e, tb,
                    extra={"tag": "INFRA"})
                try:
                    resp_q.put_nowait(("error", session_id, f"{type(e).__name__}: {e}"))
                except Exception:
                    logger.error("Worker[%s]: generate error fallback also failed", worker_id,
                        extra={"tag": "INFRA"})

        if cmd == "generate_stream":
            session_id = None
            try:
                session_id, prompt, kwargs = payload
                stream_fn(prompt, resp_q, session_id=session_id, hb_q=hb_q, **kwargs)
                requests_served += 1
            except Exception as e:
                tb = traceback.format_exc()
                logger.error("Worker[%s]: stream error: %s\n%s", worker_id, e, tb,
                    extra={"tag": "INFRA"})
                try:
                    resp_q.put_nowait(("error", session_id, f"{type(e).__name__}: {e}"))
                except Exception:
                    logger.error("Worker[%s]: stream error fallback also failed", worker_id,
                        extra={"tag": "INFRA"})

        if cmd == "load_adapter":
            session_id = None
            try:
                session_id, adapter_path, merge = payload
                if adapter_fn is None:
                    resp_q.put_nowait(("error", session_id, "Adapter loading not supported by this worker."))
                else:
                    result = adapter_fn(adapter_path, merge)
                    resp_q.put_nowait(("result", session_id, result))
            except Exception as e:
                tb = traceback.format_exc()
                logger.error("Worker[%s]: load_adapter error: %s\n%s", worker_id, e, tb,
                    extra={"tag": "INFRA"})
                try:
                    resp_q.put_nowait(("error", session_id, f"{type(e).__name__}: {e}"))
                except Exception:
                    logger.error("Worker[%s]: load_adapter error fallback also failed", worker_id,
                        extra={"tag": "INFRA"})

        if cmd == "unload_adapter":
            session_id = None
            try:
                session_id, = payload
                if unload_fn is None:
                    resp_q.put_nowait(("error", session_id, "Adapter unloading not supported by this worker."))
                else:
                    result = unload_fn()
                    resp_q.put_nowait(("result", session_id, result))
            except Exception as e:
                tb = traceback.format_exc()
                logger.error("Worker[%s]: unload_adapter error: %s\n%s", worker_id, e, tb,
                    extra={"tag": "INFRA"})
                try:
                    resp_q.put_nowait(("error", session_id, f"{type(e).__name__}: {e}"))
                except Exception:
                    logger.error("Worker[%s]: unload_adapter error fallback also failed", worker_id,
                        extra={"tag": "INFRA"})

    if cleanup_fn:
        try:
            cleanup_fn()
        except Exception as e:
            logger.warning("Worker[%s]: cleanup_fn failed: %s", worker_id, e,
                extra={"tag": "INFRA"})
    gc.collect()
    hb_q.put_nowait(("dead", os.getpid()))
    logger.info("Worker[%s]: stopped", worker_id,
        extra={"tag": "INFRA"})


# ---------------------------------------------------------------------------
# SloNet worker (pure NumPy)
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
) -> None:  # pragma: no cover — child process only; coverage recorded in parent
    """Worker subprocess entry point for SloNet models (pure NumPy).

    Loads via ``SloNetChatProvider.from_slnc()`` and streams via
    ``generate_numpy_stream()``.
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
            free_quantized_originals=True,
        )
        logger.info("Worker[%s]: SloNet model loaded (%s)", worker_id, model_id,
            extra={"tag": "INFRA"})
    except Exception as e:
        logger.error("Worker[%s]: SloNet load failed: %s", worker_id, e,
            extra={"tag": "INFRA"})
        resp_q.put_nowait(("error", None, f"Model load failed: {e}"))
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

        # Send periodic heartbeats during non-streaming generation so the
        # parent's stall detector knows the worker is alive.  First call to
        # generate_numpy() may trigger JIT compilation or cache warming that
        # blocks for >120s with no intermediate messages.
        import threading as _threading
        _hb_stop = _threading.Event()

        def _hb_tick():
            while not _hb_stop.is_set():
                try:
                    hb_q.put_nowait(("alive", os.getpid()))
                except Exception as exc:
                    logger.debug("worker: heartbeat put failed: %s", exc)
                _hb_stop.wait(timeout=5.0)

        hb_thread = _threading.Thread(target=_hb_tick, daemon=True)
        hb_thread.start()
        try:
            result = provider._model.generate_numpy(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token=provider._tokenizer.eos_token_id or 0,
                extra_stop_ids=getattr(provider._tokenizer, "chat_stop_ids", lambda: ())(),
            )
        finally:
            _hb_stop.set()
            hb_thread.join(timeout=2.0)
        elapsed_ms = (time.time() - start) * 1000
        generated = result[0].tolist()
        text = provider._tokenizer.decode(generated)
        logger.debug(
            "worker generate",
            extra={
                "tag": "INFO",
                "context": {
                    "elapsed_ms": round(elapsed_ms, 1),
                    "prompt_len": len(token_ids),
                    "tokens": len(generated),
                    "text": text[:80],
                },
            },
        )
        return {
            "text": text,
            "tokens_generated": len(generated),
            "elapsed_ms": round(elapsed_ms, 1),
        }

    def _stream(
        prompt: str,
        resp_q_inner: mp.Queue,
        session_id: Optional[str] = None,
        hb_q: Optional[mp.Queue] = None,
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
            extra_stop_ids=getattr(provider._tokenizer, "chat_stop_ids", lambda: ())(),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        ):
            if hb_q is not None:
                try:
                    hb_q.put_nowait(("alive", os.getpid()))
                except Exception as exc:
                    logger.debug("worker: heartbeat put failed: %s", exc)
            decoded = provider._tokenizer.decode([tok_id])
            if decoded:
                try:
                    resp_q_inner.put(("token", session_id, decoded), timeout=_STREAM_PUT_TIMEOUT_S)
                except Exception as exc:
                    logger.debug("worker: token put failed for session %s: %s", session_id, exc)
                    break
                tokens_generated += 1

        elapsed_ms = (time.time() - start) * 1000
        try:
            resp_q_inner.put(("result", session_id, {
                "text": "",
                "tokens_generated": tokens_generated,
                "elapsed_ms": round(elapsed_ms, 1),
            }), timeout=_STREAM_PUT_TIMEOUT_S)
        except Exception as exc:
            logger.debug("worker: result put failed for session %s: %s", session_id, exc)

    def _cleanup():
        nonlocal provider
        provider = None

    def _load_adapter(adapter_path: str, merge: bool = False) -> dict:
        """Load a LoRA adapter into the running SloNet model (inside worker subprocess)."""
        from pathlib import Path
        import numpy as np

        p = Path(adapter_path)
        if not p.is_file():
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")

        adapter = np.load(str(p), allow_pickle=True)

        rank = int(adapter.get("_config/rank", [8])[0]) if "_config/rank" in adapter else 8
        alpha = float(adapter.get("_config/alpha", [16.0])[0]) if "_config/alpha" in adapter else 16.0

        target_modules = []
        n_modules = int(adapter.get("_config/target_modules", [0])[0]) if "_config/target_modules" in adapter else 0
        for i in range(n_modules):
            key = f"_config/target_module_{i}"
            if key in adapter:
                chars = adapter[key].tolist()
                target_modules.append("".join(chr(c) for c in chars))
        if not target_modules:
            target_modules = ["W_q", "W_k", "W_v", "W_o"]

        from domains.training.lora import LoRAConfig, apply_lora_to_model, count_lora_parameters
        from domains.training.hf_lora_finetune import load_lora_adapter, merge_lora_adapter

        model = provider._model
        lora_config = LoRAConfig(rank=rank, alpha=alpha, target_modules=target_modules)
        new_model = apply_lora_to_model(model, lora_config)
        load_lora_adapter(new_model, str(p))
        n_params = count_lora_parameters(new_model)

        result = {
            "status": "loaded",
            "model_type": "slonet",
            "adapter_path": str(p),
            "rank": rank,
            "alpha": alpha,
            "target_modules": target_modules,
            "n_params": n_params,
            "merged": False,
        }

        if merge:
            new_model = merge_lora_adapter(new_model)
            result["merged"] = True

        provider._model = new_model
        logger.info("Worker loaded LoRA adapter %s (rank=%d, alpha=%.1f, %d params)",
            p, rank, alpha, n_params, extra={"tag": "TRAIN"})
        return result

    def _unload_adapter() -> dict:
        """Unload LoRA adapter and revert to base model (inside worker subprocess)."""
        from pathlib import Path
        model = provider._model

        # SloNet named_modules() doesn't recurse — use _walk_slo_tree instead
        from domains.training.lora import _walk_slo_tree
        has_lora = False
        try:
            for _path, mod in _walk_slo_tree(model, []):
                if hasattr(mod, 'lora_A') or hasattr(mod, 'lora_B'):
                    has_lora = True
                    break
        except Exception as exc:
            logger.debug("worker: LoRA detection failed: %s", exc)

        if not has_lora:
            return {"status": "no_adapter", "message": "No LoRA adapter loaded."}

        slnc_path = getattr(provider, '_slnc_path', None)
        if slnc_path is None:
            raise RuntimeError("Cannot determine base model path.")

        from domains.inference.slonet_provider import SloNetChatProvider
        base_provider = SloNetChatProvider.from_slnc(slnc_path, model_id="base")
        provider._model = base_provider._model

        logger.info("Worker unloaded LoRA adapter, reverted to: %s", slnc_path, extra={"tag": "TRAIN"})
        return {"status": "unloaded", "model_type": "slonet", "message": f"Reverted to base: {Path(slnc_path).name}"}

    _worker_loop(req_q, resp_q, hb_q, worker_id, _generate, _stream, _cleanup, _load_adapter, _unload_adapter)


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
) -> None:  # pragma: no cover — child process only; coverage recorded in parent
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
        session_id: Optional[str] = None,
        hb_q: Optional[mp.Queue] = None,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        **gen_kwargs: Any,
    ) -> None:
        from domains.infrastructure.model_server import _TokenStreamer

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        device = getattr(model, "device", None)
        if device is not None and device != "cpu":
            input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

        streamer = _TokenStreamer(
            tokenizer, skip_prompt=True
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
            if hb_q is not None:
                try:
                    hb_q.put_nowait(("alive", os.getpid()))
                except Exception as exc:
                    logger.debug("worker: heartbeat put failed: %s", exc)
            try:
                resp_q_inner.put(("token", session_id, text_chunk), timeout=_STREAM_PUT_TIMEOUT_S)
            except Exception as exc:
                logger.debug("worker: HF token put failed for session %s: %s", session_id, exc)
                break
            tokens_generated += 1

        thread.join(timeout=10)
        elapsed_ms = (time.time() - start) * 1000

        try:
            resp_q_inner.put(("result", session_id, {
                "text": "",
                "tokens_generated": tokens_generated,
                "elapsed_ms": round(elapsed_ms, 1),
            }), timeout=_STREAM_PUT_TIMEOUT_S)
        except Exception as exc:
            logger.debug("worker: HF result put failed for session %s: %s", session_id, exc)

    def _cleanup():
        nonlocal model, tokenizer
        del model
        del tokenizer

    _worker_loop(req_q, resp_q, hb_q, worker_id, _generate, _stream, _cleanup)


class ModelWorkerProcess:
    """Manages a model inference subprocess with Queue-based RPC.

    Two construction modes:

    SloNet (preferred — pure NumPy)::

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
        generate_timeout: float = DEFAULT_GENERATE_TIMEOUT,
        startup_timeout: float = DEFAULT_STARTUP_TIMEOUT,
        stall_timeout: float = _STALL_TIMEOUT_S,
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
        self._startup_timeout = startup_timeout
        self._stall_timeout = stall_timeout
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
                "ModelWorkerProcess requires either slnc_path (SloNet) "
                "or model_cls_path (HF/Legacy)"
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

        # Wait for ready signal (use startup timeout, not generate timeout)
        deadline = time.time() + self._startup_timeout
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
                f"Worker[{self.worker_id}]: failed to start within {self._startup_timeout}s"
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
        except Exception as e:
            logger.warning("Worker[%s]: failed to send stop signal: %s", self.worker_id, e,
                extra={"tag": "INFRA"})
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
                except Exception as exc:
                    logger.debug("worker: queue close failed for %s: %s", q_name, exc)
                try:
                    q.join_thread()
                except Exception as exc:
                    logger.debug("worker: queue join_thread failed for %s: %s", q_name, exc)

    # ── Health ─────────────────────────────────────────────────────────

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def _drain_heartbeats(self, since: float) -> float:
        """Drain the heartbeat queue and return the latest heartbeat timestamp.

        If any heartbeats arrived after ``since``, the caller should reset its
        stall timer — the worker is alive and producing messages.
        """
        latest = since
        if self._hb_q is None:
            return latest
        now = time.time()
        while True:
            try:
                msg, _val = self._hb_q.get_nowait()
            except queue.Empty:
                break
            except (OSError, ValueError):
                break
            if msg == "alive":
                self._health.last_heartbeat = now
                self._health.alive = True
                latest = now
            elif msg == "dead":
                self._health.alive = False
                self._health.crashed = True
            elif msg == "ready":
                self._health.last_heartbeat = now
                latest = now
        return latest

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

        session_id = _new_session_id()
        payload = dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            **kwargs,
        )

        try:
            self._req_q.put_nowait(("generate", (session_id, prompt, payload)))
        except Exception as e:
            self._health.errors += 1
            raise RuntimeError(f"Failed to send request to worker: {e}")

        # Wait for response
        deadline = time.time() + self._generate_timeout
        last_activity = time.time()
        while time.time() < deadline:
            # Check worker alive
            if not self.alive:
                self._health.crashed = True
                self._health.crash_count += 1
                raise RuntimeError(f"Worker[{self.worker_id}] crashed during generation")

            # Drain heartbeat queue — worker pings during loading and long
            # generation runs.  If we see a heartbeat, the worker is alive
            # and the stall timer resets even though no resp_q message arrived.
            hb_time = self._drain_heartbeats(last_activity)
            if hb_time > last_activity:
                last_activity = hb_time

            try:
                msg, *rest = self._resp_q.get(timeout=0.2)
            except queue.Empty:
                if time.time() - last_activity > self._stall_timeout:
                    self._health.errors += 1
                    raise WorkerStreamStalledError(
                        f"Worker[{self.worker_id}] stalled for {self._stall_timeout}s "
                        f"during generation"
                    )
                continue
            last_activity = time.time()

            msg_session, data = self._split_resp(msg, rest)
            if msg_session is not None and msg_session != session_id:
                logger.warning(
                    "Worker[%s]: discarding stale response from session %s",
                    self.worker_id, msg_session, extra={"tag": "INFRA"},
                )
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

    @staticmethod
    def _split_resp(msg: str, rest: list) -> tuple[Optional[str], Any]:
        """Split a response message into ``(session_id, data)``.

        Back-compat: legacy 2-tuples ``(type, data)`` yield session_id None
        (always accepted); new 3-tuples ``(type, session_id, data)`` yield
        the tagged session id.
        """
        if len(rest) == 1:
            return None, rest[0]
        return rest[0], rest[1]

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

        session_id = _new_session_id()
        payload = dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            **kwargs,
        )

        try:
            self._req_q.put_nowait(("generate_stream", (session_id, prompt, payload)))
        except Exception as e:
            self._health.errors += 1
            raise RuntimeError(f"Failed to send stream request to worker: {e}")

        # Read tokens from response queue until result or error
        deadline = time.time() + self._generate_timeout
        final_result: dict = {}
        last_activity = time.time()

        while time.time() < deadline:
            if not self.alive:
                self._health.crashed = True
                self._health.crash_count += 1
                raise RuntimeError(
                    f"Worker[{self.worker_id}] crashed during streaming generation"
                )

            hb_time = self._drain_heartbeats(last_activity)
            if hb_time > last_activity:
                last_activity = hb_time

            try:
                msg, *rest = self._resp_q.get(timeout=0.2)
            except queue.Empty:
                if time.time() - last_activity > self._stall_timeout:
                    self._health.errors += 1
                    raise WorkerStreamStalledError(
                        f"Worker[{self.worker_id}] stalled for {self._stall_timeout}s "
                        f"during streaming generation"
                    )
                continue
            last_activity = time.time()

            msg_session, data = self._split_resp(msg, rest)
            if msg_session is not None and msg_session != session_id:
                logger.warning(
                    "Worker[%s]: discarding stale response from session %s",
                    self.worker_id, msg_session, extra={"tag": "INFRA"},
                )
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

    # ── Adapter loading ──────────────────────────────────────────────────

    def load_adapter(self, adapter_path: str, merge: bool = False, timeout: float = DEFAULT_GENERATE_TIMEOUT) -> dict:
        """Send load_adapter request to worker and wait for result.

        Args:
            adapter_path: Path to .npz adapter file.
            merge: Whether to merge LoRA into base weights.
            timeout: Maximum wait time in seconds.

        Returns:
            dict with status, rank, alpha, n_params, etc.
        """
        if not self.alive:
            raise RuntimeError(f"Worker[{self.worker_id}] is not alive")

        session_id = _new_session_id()
        try:
            self._req_q.put_nowait(("load_adapter", (session_id, adapter_path, merge)))
        except Exception as e:
            self._health.errors += 1
            raise RuntimeError(f"Failed to send load_adapter request: {e}")

        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.alive:
                self._health.crashed = True
                self._health.crash_count += 1
                raise RuntimeError(f"Worker[{self.worker_id}] crashed during adapter load")

            try:
                msg, *rest = self._resp_q.get(timeout=0.5)
            except queue.Empty:
                continue

            msg_session, data = self._split_resp(msg, rest)
            if msg_session is not None and msg_session != session_id:
                continue

            if msg == "result":
                self._health.requests_served += 1
                return data
            elif msg == "error":
                self._health.errors += 1
                raise RuntimeError(f"Worker load_adapter error: {data}")

        raise TimeoutError(f"Worker[{self.worker_id}] adapter load timed out after {timeout}s")

    def unload_adapter(self, timeout: float = 60.0) -> dict:
        """Send unload_adapter request to worker and wait for result.

        Args:
            timeout: Maximum wait time in seconds.

        Returns:
            dict with status and message.
        """
        if not self.alive:
            raise RuntimeError(f"Worker[{self.worker_id}] is not alive")

        session_id = _new_session_id()
        try:
            self._req_q.put_nowait(("unload_adapter", (session_id,)))
        except Exception as e:
            self._health.errors += 1
            raise RuntimeError(f"Failed to send unload_adapter request: {e}")

        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.alive:
                self._health.crashed = True
                self._health.crash_count += 1
                raise RuntimeError(f"Worker[{self.worker_id}] crashed during adapter unload")

            try:
                msg, *rest = self._resp_q.get(timeout=0.5)
            except queue.Empty:
                continue

            msg_session, data = self._split_resp(msg, rest)
            if msg_session is not None and msg_session != session_id:
                continue

            if msg == "result":
                self._health.requests_served += 1
                return data
            elif msg == "error":
                self._health.errors += 1
                raise RuntimeError(f"Worker unload_adapter error: {data}")

        raise TimeoutError(f"Worker[{self.worker_id}] adapter unload timed out after {timeout}s")

    # ── Context manager ────────────────────────────────────────────────

    def __enter__(self) -> "ModelWorkerProcess":
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()
