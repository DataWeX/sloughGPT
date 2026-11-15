"""local_api — in-process transport for shell commands (card ad4f5ec2).

Three-tier architecture (docs/PRODUCT_ENGINEERING.md):

* **model tier** — the shell binary, which owns the domain core in-process;
* **cloud tier** — the API server, for the web UI and remote clients;
* **clients** — consume the one descriptor-projected contract.

In local mode (the default) every path ``commands.py`` uses is served here
by calling ``domain/`` functions directly — same process, no HTTP, no
server required. ``MAN_API_URL`` / ``shell --remote`` demote the shell to
an HTTP client of a running API server (the remote tier).

Route templates mirror the API paths so the transport seam stays in one
place; response shapes match what the shell's renderers already consume.
All domain imports are lazy so booting the shell stays cheap.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

from domain.shared import find_repo_root

logger = logging.getLogger("slo.shell.local_api")

# ── Local job store ──────────────────────────────────────────────────
# Mirrors the API server's ``training_jobs`` records for jobs this process
# starts (only the shell's own training runs live here in local mode).

_LOCAL_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()

_TRAINER_DEFAULTS: dict[str, Any] = {
    "n_embed": 128,
    "n_layer": 4,
    "n_head": 4,
    "block_size": 128,
    "dropout": 0.1,
    "batch_size": 32,
    "epochs": 3,
    "lr": 1e-3,
    "max_steps": None,
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 1.0,
    "checkpoint_dir": "checkpoints",
    "checkpoint_interval": 500,
    "save_best_only": False,
    "max_checkpoints": 5,
    "scheduler_type": "cosine",
    "warmup_steps": 100,
    "min_lr": 1e-5,
    "weight_decay": 0.01,
    "use_lora": False,
    "lora_rank": 8,
    "lora_alpha": 16,
    "log_interval": 10,
    "eval_interval": 100,
    "feed_refresh_interval": 0.0,
    "device": None,
}


# ── Shared helpers ───────────────────────────────────────────────────


def _state():
    """The domain-owned ServerState singleton (model/provider/tokenizer)."""
    from domain.infrastructure.server_state import get_server_state

    return get_server_state()


def _current_provider():
    """The loaded in-process provider (model ref first, then provider ref)."""
    st = _state()
    return (st.model.get() if st else None) or (st.provider.get() if st else None)


def _publish_provider(provider: Any, model_id: str) -> None:
    """Publish a loaded provider to ServerState (mirrors the models controller)."""
    st = _state()
    if st is None or provider is None:
        return
    st.model.set(provider)
    st.provider.set(provider)
    st.model_type.set(model_id)


def _run_coro(coro):
    """Run a domain coroutine synchronously (the shell is a sync host)."""
    import asyncio

    return asyncio.run(coro)


def _unsupported(op: str) -> dict[str, Any]:
    """Structured degradation for endpoints whose orchestration is server-side."""
    return {
        "error": (
            f"{op} is not served in in-process mode yet — start the API server "
            f"(`api start`) and re-run the shell with --remote or MAN_API_URL"
        ),
        "error_type": "LocalUnsupported",
    }


# ── Route registry ───────────────────────────────────────────────────

_Routes = tuple  # (method, compiled pattern, handler)
_ROUTES: list[tuple[str, re.Pattern[str], Callable[..., Any]]] = []


def _route(method: str, template: str):
    """Register a local handler for an API path template (``/x/{id}``)."""

    pattern = re.compile("^" + re.sub(r"\{(\w+)\}", r"(?P<\1>[^/]+)", template) + "$")

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        _ROUTES.append((method.upper(), pattern, fn))
        return fn

    return deco


def _dispatch(
    method: str, path: str, data: dict[str, Any] | None = None
) -> dict[str, Any] | list[Any]:
    path_only, _, qs = str(path).partition("?")
    query = {k: v[0] for k, v in parse_qs(qs).items() if v}
    for m, rx, fn in _ROUTES:
        if m != method.upper():
            continue
        match = rx.match(path_only)
        if not match:
            continue
        try:
            return fn(match.groupdict(), query, data)
        except Exception as e:
            logger.debug("local %s %s failed: %s", method, path, e)
            return {"error": str(e) or type(e).__name__, "error_type": type(e).__name__}
    return {
        "error": (
            f"No in-process handler for {method} {path} — start the API server "
            f"and use --remote / MAN_API_URL to reach it"
        ),
        "error_type": "LocalDispatchError",
    }


def dispatch_get(path: str) -> dict[str, Any] | list[Any]:
    return _dispatch("GET", path)


def dispatch_post(path: str, data: dict[str, Any] | None = None) -> dict[str, Any] | list[Any]:
    return _dispatch("POST", path, data)


def dispatch_delete(path: str) -> dict[str, Any]:
    return _dispatch("DELETE", path)


# ── Health & status ──────────────────────────────────────────────────


@_route("GET", "/health")
def _get_health(params, query, data):
    st = _state()
    model_type = (st.model_type.get() if st else "") or ""
    provider = _current_provider()
    out: dict[str, Any] = {
        "status": "healthy",
        "mode": "local",
        "model_loaded": provider is not None,
        "model_loading": False,
        "device": "cpu",
    }
    if model_type:
        out["model_type"] = model_type
    soul = st.current_soul.get() if st else None
    soul_name = soul if isinstance(soul, str) else (getattr(soul, "name", "") if soul else "")
    if soul_name:
        out["soul_name"] = soul_name
    try:
        from domain.infrastructure.model_registry import get_model_registry

        out["registry"] = get_model_registry().health_summary()
    except Exception as e:
        logger.debug("local /health registry summary failed: %s", e)
    return out


@_route("GET", "/health/detailed")
def _get_health_detailed(params, query, data):
    detail = dict(_get_health(params, query, data))
    try:
        import psutil

        proc = psutil.Process()
        vm = psutil.virtual_memory()
        detail.update(
            {
                "cpu_percent": psutil.cpu_percent(interval=None),
                "cpu_count_logical": psutil.cpu_count(logical=True) or 1,
                "memory_percent": vm.percent,
                "memory_used_gb": round(vm.used / (1024**3), 2),
                "memory_total_gb": round(vm.total / (1024**3), 2),
                "disk_percent": psutil.disk_usage(str(find_repo_root(Path(__file__)))).percent,
                "uptime_seconds": round(time.time() - proc.create_time(), 1),
            }
        )
    except Exception as e:
        logger.debug("local /health/detailed metrics failed: %s", e)
    return detail


@_route("GET", "/info")
def _get_info(params, query, data):
    st = _state()
    model_type = (st.model_type.get() if st else "") or ""
    return {
        "api_version": "1.0.0",
        "mode": "local",
        "model": {"type": model_type or None, "loaded": _current_provider() is not None},
    }


@_route("GET", "/system/metrics")
def _get_system_metrics(params, query, data):
    logical = physical = None
    try:
        from domain.infrastructure.resource_manager import get_resource_manager

        rm = get_resource_manager()
        logical = rm.topology.logical_cores
        physical = rm.topology.physical_cores
    except Exception as e:
        logger.debug("local /system/metrics resource_manager unavailable: %s", e)
    import psutil

    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "memory_percent": psutil.virtual_memory().percent,
        "memory_used_gb": psutil.virtual_memory().used / (1024**3),
        "memory_total_gb": psutil.virtual_memory().total / (1024**3),
        "cpu_count_logical": logical or psutil.cpu_count(logical=True) or 1,
        "cpu_count_physical": physical or psutil.cpu_count(logical=False) or 1,
    }


@_route("GET", "/tokenizer/stats")
def _get_tokenizer_stats(params, query, data):
    from domain.training._internal.tokenizer_manager import get_tokenizer_manager

    s = get_tokenizer_manager().stats()
    return {
        "vocab_size": s.get("vocab_size", 0),
        "base_chars": s.get("base_chars", 0),
        "merged_subwords": s.get("merged_subwords", s.get("subwords", 0)),
        "special_tokens": s.get("special_tokens", {}),
        "total_merges": s.get("total_merges", s.get("total_merges_learned", 0)),
        "trained": s.get("trained", True),
    }


# ── Models ───────────────────────────────────────────────────────────


@_route("GET", "/models")
def _get_models(params, query, data):
    st = _state()
    model_type = (st.model_type.get() if st else "") or ""
    provider = _current_provider()
    entries: list[dict[str, Any]] = []
    if model_type:
        entries.append(
            {
                "model_id": model_type,
                "status": "loaded" if provider is not None else "registered",
                "device": "cpu",
                "loaded": provider is not None,
            }
        )
    try:
        from domain.infrastructure.model_registry import get_model_registry

        for m in get_model_registry().list_models():
            if m.get("model_id") != model_type:
                entries.append({**m, "loaded": m.get("status") in ("ready", "degraded", "loaded")})
    except Exception as e:
        logger.debug("local /models registry scan failed: %s", e)
    return entries


@_route("POST", "/models/load")
def _post_models_load(params, query, data):
    model_id = str((data or {}).get("model_id") or "").strip()
    if not model_id:
        raise ValueError("model_id is required")
    if model_id.endswith(".gguf"):
        return _unsupported("GGUF model loading")

    from domain.models import get_provider, setup_providers

    setup_providers(slonet_hf_id=model_id)
    provider = get_provider("slonet-native") or get_provider("slonet")
    # setup_providers logs-and-continues on failure — verify the provider
    # actually belongs to the requested model before publishing (mirrors
    # the API server's models controller).
    if provider is None or getattr(provider, "_model_id", None) != model_id:
        raise RuntimeError(f"SloNet provider for {model_id} not registered after load")
    _publish_provider(provider, model_id)
    return {"status": "loaded", "model_id": model_id, "type": "slonet", "device": "cpu"}


@_route("POST", "/models/unload")
def _post_models_unload(params, query, data):
    st = _state()
    model_id = (st.model_type.get() if st else "") or ""
    try:
        from domain.infrastructure.model_registry import get_model_registry

        registry = get_model_registry()
        model_id = model_id or (registry.default_id or "")
        if model_id:
            registry.unregister(model_id)
    except Exception as e:
        logger.debug("local unload registry.unregister failed: %s", e)
    try:
        from domain.models import clear_providers

        clear_providers()
    except Exception as e:
        logger.debug("local unload clear_providers failed: %s", e)
    if st:
        st.model.set(None)
        st.provider.set(None)
        st.model_type.set(None)
    return {"status": "unloaded", "model_id": model_id}


@_route("POST", "/models/precision")
def _post_models_precision(params, query, data):
    mode = str((data or {}).get("mode") or "auto")
    from domain.slolib import get_accelerator, set_accelerator_precision

    active = set_accelerator_precision(mode)
    acc = get_accelerator()
    return {
        "mode": active or mode,
        "requested": mode,
        "accelerator": acc.name,
        "device_type": acc.device_type,
    }


@_route("POST", "/models/quantize")
def _post_models_quantize(params, query, data):
    bits = int((data or {}).get("bits") or 8)
    mode = str((data or {}).get("mode") or "symmetric")
    if bits not in (4, 8):
        raise ValueError(f"bits must be 4 or 8, got {bits}")
    if mode not in ("symmetric", "asymmetric"):
        raise ValueError(f"mode must be symmetric or asymmetric, got {mode}")

    import numpy as np

    from domain.infrastructure.quantization import Quantine
    from domain.models import get_provider

    provider = get_provider("slonet")
    model_type = "slonet"
    if provider is None:
        provider = get_provider("hf-default")
        model_type = "huggingface"
    if provider is None:
        return {"error": "No model loaded", "error_type": "ModelNotLoaded"}
    model = getattr(provider, "_model", None)
    if model is None:
        return {"error": "Provider has no model", "error_type": "ModelNotLoaded"}

    if model_type == "slonet":
        from domain.infrastructure.quantization import walk_slo_linears

        layers = walk_slo_linears(model)
    else:
        from domain.infrastructure.quantization import walk_hf_linears

        layers = walk_hf_linears(model)

    engine = Quantine(bits=bits, mode=mode)
    tensor_infos: dict[str, Any] = {}
    quantized_count = 0
    for name, module in layers.items():
        weight = module.weight.data
        if hasattr(weight, "cpu"):  # torch tensor
            weight = weight.cpu().numpy().astype(np.float32).copy()
        else:
            weight = np.asarray(weight, dtype=np.float32).copy()
        info = engine.quantize(f"{name}.weight", weight)
        if not info.is_quantized:
            continue
        if model_type == "slonet":
            module.set_quantized_weight(info)
        else:
            from domain.infrastructure.quantization import QuantizedLinear

            ql = QuantizedLinear.from_linear(module, info)
            module._quant_info = info
            module._ql = ql
            module._orig_forward = module.forward
            module.forward = ql.make_torch_forward()
        tensor_infos[name] = info
        quantized_count += 1

    if quantized_count > 0 and getattr(provider, "_model_path", None):
        p = Path(str(provider._model_path))
        engine.save_weights(str(p.with_suffix(p.suffix + ".quant.npz")), tensor_infos)
    provider._quant_engine = engine

    avx2 = False
    try:
        from domain.infrastructure.quant_core.wrapper import HAS_AVX2

        avx2 = bool(HAS_AVX2)
    except Exception:
        pass
    return {
        "status": "ok",
        "quantized": True,
        "bits": bits,
        "mode": mode,
        "model_type": model_type,
        "layers_quantized": quantized_count,
        "total_layers": len(layers),
        "summary": engine.summary(),
        "avx2_enabled": avx2,
    }


@_route("POST", "/models/dequantize")
def _post_models_dequantize(params, query, data):
    from domain.models import get_provider

    provider = get_provider("slonet")
    model_type = "slonet"
    if provider is None:
        provider = get_provider("hf-default")
        model_type = "huggingface"
    if provider is None:
        return {"error": "No model loaded", "error_type": "ModelNotLoaded"}
    model = getattr(provider, "_model", None)
    if model is None:
        return {"error": "Provider has no model", "error_type": "ModelNotLoaded"}

    if model_type == "slonet":
        from domain.infrastructure.quantization import walk_slo_linears

        layers = walk_slo_linears(model)
        for _name, module in layers.items():
            module._quant_info = None
    else:
        from domain.infrastructure.quantization import walk_hf_linears

        layers = walk_hf_linears(model)
        for _name, module in layers.items():
            if hasattr(module, "_quant_info"):
                module._quant_info = None
                if hasattr(module, "_orig_forward"):
                    module.forward = module._orig_forward
                    del module._orig_forward
                if hasattr(module, "_ql"):
                    del module._ql
    if hasattr(provider, "_quant_engine"):
        try:
            del provider._quant_engine
        except Exception:
            provider._quant_engine = None
    return {"status": "ok", "layers_reset": len(layers), "model_type": model_type}


# ── Inference ────────────────────────────────────────────────────────


@_route("POST", "/inference/generate")
def _post_inference_generate(params, query, data):
    provider = _current_provider()
    if provider is None or not hasattr(provider, "generate"):
        return {"error": "No model loaded", "error_type": "ModelNotLoaded"}
    payload = data or {}
    text = provider.generate(
        str(payload.get("prompt") or ""),
        max_new_tokens=int(payload.get("max_new_tokens") or 100),
        temperature=float(payload.get("temperature") or 0.7),
    )
    st = _state()
    return {"text": text, "model_id": ((st.model_type.get() if st else "") or "")}


@_route("POST", "/chat")
def _post_chat(params, query, data):
    provider = _current_provider()
    if provider is None or not hasattr(provider, "chat"):
        return {"error": "No model loaded", "error_type": "ModelNotLoaded"}
    messages = (data or {}).get("messages") or []
    text = _run_coro(provider.chat(messages, max_tokens=512, temperature=0.7))
    return {"reply": text, "content": text}


def local_chat_stream(
    messages: list[dict[str, str]],
    session_id: str = "",
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> Iterator[Any]:
    """Stream a chat response from the in-process provider.

    Drives the provider's async token stream on a dedicated event loop and
    yields token strings (or ``{"error": ...}`` dicts), matching the shape
    the remote SSE path yields.
    """
    import asyncio

    provider = _current_provider()
    if provider is None or not hasattr(provider, "chat_stream"):
        yield {
            "error": "No model loaded — run 'models load <id>' first",
            "error_type": "ModelNotLoaded",
        }
        return

    loop = asyncio.new_event_loop()
    agen = None
    try:
        kwargs: dict[str, Any] = {"max_tokens": int(max_tokens), "temperature": float(temperature)}
        if session_id:
            kwargs["session_id"] = session_id
        agen = provider.chat_stream(messages, **kwargs)
        while True:
            try:
                token = loop.run_until_complete(agen.__anext__())
            except StopAsyncIteration:
                return
            except Exception as e:
                yield {"error": str(e) or type(e).__name__, "error_type": type(e).__name__}
                return
            if token:
                yield token
    finally:
        if agen is not None:
            try:
                loop.run_until_complete(agen.aclose())
            except Exception:
                pass
        try:
            loop.close()
        except Exception:
            pass


# ── Souls ────────────────────────────────────────────────────────────


def _soul_dict(soul: Any) -> dict[str, Any]:
    if is_dataclass(soul) and not isinstance(soul, type):
        return asdict(soul)
    if isinstance(soul, dict):
        return soul
    return {"name": getattr(soul, "name", str(soul))}


@_route("GET", "/souls")
def _get_souls(params, query, data):
    from domain.inference._internal.slo_manager import get_slo_manager

    return {"souls": [_soul_dict(s) for s in get_slo_manager().list_souls()]}


@_route("GET", "/souls/current")
def _get_souls_current(params, query, data):
    from domain.inference._internal.slo_manager import get_slo_manager

    soul = get_slo_manager().get_current_soul()
    return _soul_dict(soul) if soul is not None else {"name": "unknown"}


@_route("POST", "/souls/switch")
def _post_souls_switch(params, query, data):
    name = str((data or {}).get("name") or "").strip()
    if not name:
        raise ValueError("soul name is required")
    from domain.inference._internal.slo_manager import switch_soul

    return switch_soul(name)


# ── Knowledge ────────────────────────────────────────────────────────


@_route("GET", "/knowledge")
def _get_knowledge(params, query, data):
    from domain.knowledge import get_knowledge_engine

    res = get_knowledge_engine().list_all(top_k=200)
    if res is not None and res.success:
        return res.data or []
    logger.debug("local /knowledge failed: %s", getattr(res, "error", None))
    return []


@_route("GET", "/knowledge/search")
def _get_knowledge_search(params, query, data):
    from domain.knowledge import get_knowledge_engine

    q = query.get("query", "")
    res = get_knowledge_engine().query(q, top_k=20) if q else None
    results = list(res.data or []) if res is not None and res.success else []
    return {"results": results, "count": len(results)}


@_route("GET", "/knowledge/stats")
def _get_knowledge_stats(params, query, data):
    from domain.knowledge import get_knowledge_engine

    res = get_knowledge_engine().stats()
    if res is not None and res.success and isinstance(res.data, dict):
        return res.data
    return {"total_items": 0}


@_route("POST", "/knowledge")
def _post_knowledge(params, query, data):
    payload = data or {}
    content = str(payload.get("content") or "").strip()
    if not content:
        raise ValueError("content is required")
    from domain.knowledge import get_knowledge_engine

    res = get_knowledge_engine().store(
        content,
        topic=str(payload.get("topic") or "general"),
        source=str(payload.get("source") or "shell"),
    )
    if res is None or not res.success:
        return {
            "error": getattr(res, "error", None) or "knowledge store failed",
            "error_type": "KnowledgeError",
        }
    return {"status": "stored", **(res.data or {})}


# ── Datasets ─────────────────────────────────────────────────────────


@_route("GET", "/datasets")
def _get_datasets(params, query, data):
    from domain.training.engine import get_training_engine

    res = get_training_engine().list_datasets()
    raw = list(res.data or []) if getattr(res, "success", False) else []
    out: list[dict[str, Any]] = []
    for c in raw:
        if isinstance(c, dict):
            out.append(c)
            continue
        dtype = getattr(c, "dataset_type", "")
        dformat = getattr(c, "data_format", "")
        out.append(
            {
                "name": getattr(c, "name", ""),
                "id": getattr(c, "name", ""),
                "path": str(getattr(c, "path", "")),
                "type": str(getattr(dtype, "value", dtype)),
                "format": str(getattr(dformat, "value", dformat)),
                "max_samples": getattr(c, "max_samples", None),
            }
        )
    return {"datasets": out, "count": len(out)}


# ── Training jobs ────────────────────────────────────────────────────


def _job_summary(rec: dict[str, Any]) -> dict[str, Any]:
    """Public view of a local job record (underscore keys stripped)."""
    summary = {
        "id": rec.get("id"),
        "name": rec.get("name"),
        "status": rec.get("status", "unknown"),
        "model": rec.get("model", "slonet"),
        "dataset": rec.get("dataset", ""),
        "progress": rec.get("progress", 0),
        "current_epoch": rec.get("current_epoch", 0),
        "epochs": rec.get("epochs", 0),
        "global_step": rec.get("global_step", 0),
        "train_loss": rec.get("train_loss"),
        "eval_loss": rec.get("eval_loss"),
        "loss": rec.get("loss"),
        "checkpoint": rec.get("checkpoint"),
        "error": rec.get("error"),
        "started_at": rec.get("started_at"),
        "updated_at": rec.get("updated_at"),
    }
    return {k: v for k, v in summary.items() if v is not None}


def _purge_jobs() -> None:
    """Drop terminal job records older than 1h (bounds memory, mirrors server)."""
    now = time.time()
    terminal = ("completed", "failed", "stopped", "cancelled")
    with _JOBS_LOCK:
        stale = [
            jid
            for jid, rec in _LOCAL_JOBS.items()
            if rec.get("status") in terminal
            and now - float(rec.get("updated_at") or rec.get("started_at") or 0) > 3600
        ]
        for jid in stale:
            _LOCAL_JOBS.pop(jid, None)


def _finish_job(job_id: str, status: str, error: str | None = None) -> None:
    with _JOBS_LOCK:
        rec = _LOCAL_JOBS.get(job_id)
        if rec is None:
            return
        rec["status"] = status
        rec["updated_at"] = time.time()
        if error:
            rec["error"] = error
        rec.pop("_cancel_event", None)
        rec.pop("_pause_event", None)
    try:
        from domain.infrastructure.cancel_manager import get_cancel_manager

        get_cancel_manager().finish(job_id, error=error)
    except Exception as exc:
        logger.debug("cancel_manager.finish failed: %s", exc)


def _resolve_dataset(name: str) -> Path:
    """Resolve a dataset name to a training data file (repo-anchored)."""
    name = str(name or "").strip()
    if not name:
        raise ValueError("No dataset given")
    root = find_repo_root(Path(__file__).resolve())
    raw = Path(name)
    if raw.is_absolute():
        candidates = [raw]
    else:
        candidates = []
        for base in (root / "data", root / "datasets", root):
            candidates.extend([base / raw, base / f"{name}.jsonl", base / f"{name}.txt"])
    for c in candidates:
        if c.is_file():
            return c
        if c.is_dir():
            for leaf in ("input.txt", "corpus.jsonl", "train.txt", "data.jsonl"):
                p = c / leaf
                if p.is_file():
                    return p
    raise ValueError(f"Dataset not found: {name}")


def _run_quick_training(job_id: str) -> None:
    """Executor target: train a char-level SloNet model on a dataset."""
    from domain.training import SloughGPTTrainer

    with _JOBS_LOCK:
        rec = dict(_LOCAL_JOBS.get(job_id) or {})
    cancel_event = rec.get("_cancel_event") or threading.Event()
    pause_event = rec.get("_pause_event") or threading.Event()
    data_path = str(rec.get("data_path") or "")
    save_stem = str(rec.get("save_stem") or f"job_{job_id}")

    def on_progress(info: dict[str, Any]) -> None:
        with _JOBS_LOCK:
            r = _LOCAL_JOBS.get(job_id)
            if r is None:
                return
            r["progress"] = int(info.get("progress_percent", r.get("progress", 0)))
            r["current_epoch"] = int(info.get("epoch", r.get("current_epoch", 0)))
            r["global_step"] = int(info.get("global_step", 0))
            if info.get("total_steps") is not None:
                r["total_steps"] = int(info["total_steps"])
            if info.get("train_loss") is not None:
                r["train_loss"] = float(info["train_loss"])
            if info.get("eval_loss") is not None:
                r["eval_loss"] = float(info["eval_loss"])
                r["loss"] = r["eval_loss"]
            r["updated_at"] = time.time()

    try:
        trainer = SloughGPTTrainer(data_path=data_path, **dict(_TRAINER_DEFAULTS))
        trainer.train(on_progress=on_progress, cancel_event=cancel_event, pause_event=pause_event)
        if cancel_event.is_set():
            _finish_job(job_id, "cancelled")
            return
        out_path = find_repo_root(Path(__file__).resolve()) / "models" / f"{save_stem}_trained.soul"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save(str(out_path))
        with _JOBS_LOCK:
            r = _LOCAL_JOBS.get(job_id)
            if r is not None:
                r["progress"] = 100
                r["checkpoint"] = str(out_path)
                r["current_epoch"] = int(r.get("epochs") or _TRAINER_DEFAULTS["epochs"])
        _finish_job(job_id, "completed")
    except Exception as e:
        logger.warning("local quick training failed for %s: %s", job_id, e)
        _finish_job(job_id, "failed", error=str(e))


@_route("POST", "/training/quick")
def _post_training_quick(params, query, data):
    payload = data or {}
    dataset = str(payload.get("dataset") or "").strip()
    name = str(payload.get("name") or "").strip()
    dpath = _resolve_dataset(dataset)
    jid = f"job_{uuid.uuid4().hex[:8]}"
    now = time.time()
    rec: dict[str, Any] = {
        "id": jid,
        "name": name or f"quick-{dataset}",
        "model": "slonet",
        "dataset": dataset,
        "data_path": str(dpath),
        "status": "running",
        "progress": 0,
        "current_epoch": 0,
        "epochs": _TRAINER_DEFAULTS["epochs"],
        "global_step": 0,
        "train_loss": None,
        "eval_loss": None,
        "loss": None,
        "started_at": now,
        "updated_at": now,
        "save_stem": (re.sub(r"[^A-Za-z0-9._-]", "_", name or dataset)[:120] or "quick"),
        "_cancel_event": threading.Event(),
        "_pause_event": threading.Event(),
    }
    with _JOBS_LOCK:
        _LOCAL_JOBS[jid] = rec
    try:
        from domain.infrastructure.cancel_manager import OpType, get_cancel_manager

        mgr = get_cancel_manager()
        mgr.register(
            op_type=OpType.TRAINING,
            label=rec["name"],
            cancel_fn=rec["_cancel_event"].set,
            meta={"job_id": jid, "source": "shell-local"},
            op_id=jid,
        )
        mgr.start(jid)
    except Exception as exc:
        logger.debug("cancel_manager register failed: %s", exc)
    from domain.training._internal.executor import get_training_executor

    get_training_executor().submit(_run_quick_training, jid)
    return {
        "id": jid,
        "job_id": jid,
        "status": "running",
        "dataset": dataset,
        "data_path": str(dpath),
    }


@_route("GET", "/training/jobs")
def _get_training_jobs(params, query, data):
    _purge_jobs()
    with _JOBS_LOCK:
        return [_job_summary(rec) for rec in _LOCAL_JOBS.values()]


@_route("GET", "/training/jobs/{job_id}")
def _get_training_job(params, query, data):
    with _JOBS_LOCK:
        rec = _LOCAL_JOBS.get(params["job_id"])
        if rec is None:
            return {}
        return _job_summary(rec)


@_route("POST", "/training/jobs/{job_id}/stop")
def _post_training_job_stop(params, query, data):
    jid = params["job_id"]
    with _JOBS_LOCK:
        rec = _LOCAL_JOBS.get(jid)
        if rec is not None:
            status = rec.get("status", "unknown")
            if status not in ("running", "queued", "starting"):
                return {
                    "error": f"Job is not running (status: {status})",
                    "error_type": "BadRequest",
                }
            rec["status"] = "stopping"
            rec["updated_at"] = time.time()
            cancel_event = rec.get("_cancel_event")
    if rec is not None and cancel_event is not None:
        cancel_event.set()
    from domain.infrastructure.cancel_manager import get_cancel_manager
    from domain.training._internal.executor import get_training_executor

    ex = get_training_executor()
    known = rec is not None or ex.status(jid) is not None
    if not known:
        return {"error": "Job not found", "error_type": "NotFound"}
    ex.cancel(jid)
    try:
        get_cancel_manager().cancel(jid)
    except Exception as exc:
        logger.debug("cancel_manager.cancel failed: %s", exc)
    return {"status": "stopping", "job_id": jid}


@_route("DELETE", "/training/jobs/{job_id}")
def _delete_training_job(params, query, data):
    jid = params["job_id"]
    with _JOBS_LOCK:
        rec = _LOCAL_JOBS.pop(jid, None)
    if rec is None:
        return {"error": "Job not found", "error_type": "NotFound"}
    deleted_files: list[str] = []
    checkpoint = rec.get("checkpoint")
    if checkpoint:
        p = Path(str(checkpoint))
        if p.exists():
            try:
                p.unlink()
                deleted_files.append(str(p))
            except OSError as exc:
                logger.warning("could not delete checkpoint %s: %s", p, exc)
    return {"status": "deleted", "job_id": jid, "deleted_files": deleted_files}


@_route("POST", "/auto-train/start")
def _post_auto_train_start(params, query, data):
    payload = data or {}
    from domain.training._internal.auto_trainer import get_auto_trainer

    trainer = get_auto_trainer()
    trainer.start()
    try:
        status = trainer.status()
    except Exception as e:
        logger.debug("auto_trainer.status failed: %s", e)
        status = {}
    return {
        "status": "started",
        "soul_name": str(payload.get("soul_name") or ""),
        "teacher_model": str(payload.get("teacher_model") or ""),
        "detail": status,
    }


# ── Checkpoints ──────────────────────────────────────────────────────


@_route("GET", "/auto-train/checkpoints")
def _get_checkpoints(params, query, data):
    from domain.training._internal.checkpoints import list_checkpoints

    res = _run_coro(list_checkpoints())
    return {"checkpoints": res if isinstance(res, list) else []}


@_route("POST", "/auto-train/checkpoints/{name}/load")
def _post_checkpoint_load(params, query, data):
    from domain.training._internal.checkpoints import load_checkpoint

    res = _run_coro(load_checkpoint(params["name"]))
    if isinstance(res, dict):
        return res
    return {"status": "loaded", "name": params["name"]}


@_route("DELETE", "/auto-train/checkpoints/{name}")
def _delete_checkpoint(params, query, data):
    from domain.training._internal.checkpoints import delete_checkpoint

    res = _run_coro(delete_checkpoint(params["name"]))
    return {"status": "deleted", "name": params["name"], "deleted": res}


# ── Fine-tuned models ────────────────────────────────────────────────


def _finetuned_dir() -> Path:
    return find_repo_root(Path(__file__).resolve()) / "models" / "hf-finetuned"


def _resolve_finetuned(name: str) -> Path:
    base = _finetuned_dir().resolve()
    target = (base / name).resolve()
    if base not in target.parents:
        raise ValueError(f"Invalid fine-tuned model name: {name}")
    if not target.is_dir():
        raise ValueError(f"Fine-tuned model not found: {name}")
    return target


def _read_finetuned_metadata(model_dir: Path) -> dict[str, Any]:
    try:
        return json.loads((model_dir / "metadata.json").read_text())
    except Exception:
        return {}


@_route("GET", "/training/finetuned-models")
def _get_finetuned_models(params, query, data):
    base = _finetuned_dir()
    models: list[dict[str, Any]] = []
    if base.is_dir():
        for d in sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not d.is_dir():
                continue
            size_bytes = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            meta = _read_finetuned_metadata(d)
            display_name = d.name.split("_")[0].replace("--", "/")
            models.append(
                {
                    "name": d.name,
                    "model_path": str(d),
                    "size_mb": round(size_bytes / (1024 * 1024), 1),
                    "size_bytes": size_bytes,
                    "created_at": time.strftime(
                        "%Y-%m-%dT%H:%M:%S", time.localtime(d.stat().st_mtime)
                    ),
                    "model": meta.get("model") or display_name,
                    "dataset": meta.get("dataset")
                    or (d.name.split("_")[1] if "_" in d.name else ""),
                    "model_name": d.name,
                    "final_loss": meta.get("final_loss"),
                    "epochs": meta.get("epochs") or 0,
                }
            )
    return {"models": models}


@_route("POST", "/training/finetuned-models/{name}/load")
def _post_finetuned_load(params, query, data):
    name = params["name"]
    target = _resolve_finetuned(name)
    from domain.infrastructure.model_loader import load_model

    result = load_model(str(target), device="cpu")
    if not result.success:
        return {
            "error": result.error or "Failed to load fine-tuned model",
            "error_type": "LoadError",
        }
    _publish_provider(result.provider, result.model_id or name)
    return {
        "status": "loaded",
        "name": name,
        "model_path": str(target),
        "model_id": result.model_id,
    }


@_route("DELETE", "/training/finetuned-models/{name}")
def _delete_finetuned(params, query, data):
    name = params["name"]
    target = _resolve_finetuned(name)
    shutil.rmtree(str(target))
    return {"status": "deleted", "name": name}


# ── LoRA adapters ────────────────────────────────────────────────────


def _resolve_adapter_path(raw: str) -> Path:
    p = Path(raw)
    if p.is_file():
        return p
    root = find_repo_root(Path(__file__).resolve())
    for base in (root / "models", root / "data" / "user_adapters"):
        alt = base / raw
        if alt.is_file():
            return alt
    raise ValueError(f"Adapter not found: {raw}")


@_route("POST", "/training/load-adapter")
def _post_load_adapter(params, query, data):
    payload = data or {}
    adapter_path = _resolve_adapter_path(str(payload.get("adapter_path") or ""))
    merge = bool(payload.get("merge"))

    provider = _state().model.get() if _state() else None
    if provider is None:
        return {"error": "No model loaded — load a model first", "error_type": "ModelNotLoaded"}

    import numpy as np

    from domain.training._internal.hf_lora_finetune import (
        load_lora_adapter,
        merge_lora_adapter,
    )
    from domain.training._internal.lora import (
        LoRAConfig,
        apply_lora_to_model,
        count_lora_parameters,
        get_lora_parameters,
    )

    adapter = np.load(str(adapter_path), allow_pickle=True)
    rank = int(adapter["_config/rank"][0]) if "_config/rank" in adapter else 8
    alpha = float(adapter["_config/alpha"][0]) if "_config/alpha" in adapter else 16.0
    target_modules: list[str] = []
    n_modules = (
        int(adapter["_config/target_modules"][0]) if "_config/target_modules" in adapter else 0
    )
    for i in range(n_modules):
        key = f"_config/target_module_{i}"
        if key in adapter:
            target_modules.append("".join(chr(int(c)) for c in adapter[key].tolist()))
    if not target_modules:
        target_modules = ["W_q", "W_k", "W_v", "W_o"]

    model = getattr(provider, "_model", provider)
    if model is None:
        return {"error": "No model loaded", "error_type": "ModelNotLoaded"}
    if not get_lora_parameters(model):
        model = apply_lora_to_model(
            model, LoRAConfig(rank=rank, alpha=alpha, target_modules=target_modules)
        )
        if hasattr(provider, "_model"):
            provider._model = model
    load_lora_adapter(model, str(adapter_path))
    if merge:
        model = merge_lora_adapter(model)
        if hasattr(provider, "_model"):
            provider._model = model
    n_params = count_lora_parameters(model) if not merge else 0
    return {
        "status": "loaded",
        "adapter_path": str(adapter_path),
        "merged": merge,
        "rank": rank,
        "alpha": alpha,
        "target_modules": target_modules,
        "n_params": n_params,
        "message": f"Adapter loaded: {adapter_path.name}",
    }


@_route("POST", "/training/unload-adapter")
def _post_unload_adapter(params, query, data):
    provider = _state().model.get() if _state() else None
    if provider is None:
        return {"error": "No model loaded", "error_type": "ModelNotLoaded"}
    from domain.training._internal.hf_lora_finetune import unload_lora_adapter

    unload_lora_adapter(provider)
    return {"status": "unloaded", "message": "LoRA adapter unloaded"}


# ── Operations ───────────────────────────────────────────────────────


def _opt_optype(query: dict[str, str]):
    t = query.get("type") or ""
    if not t:
        return None
    from domain.infrastructure.cancel_manager import OpType

    return OpType(t)


@_route("GET", "/operations")
def _get_operations(params, query, data):
    from domain.infrastructure.cancel_manager import get_cancel_manager

    mgr = get_cancel_manager()
    op_type = _opt_optype(query)
    return {
        "operations": [op.to_dict() for op in mgr.list_all(op_type=op_type)],
        "counts": mgr.count(op_type=op_type),
    }


@_route("POST", "/cancel/{op_id}")
def _cancel_operation(params, query, data):
    from domain.infrastructure.cancel_manager import get_cancel_manager

    mgr = get_cancel_manager()
    op_id = params["op_id"]
    if mgr.get(op_id) is None:
        return {"error": "Operation not found", "error_type": "NotFound"}
    ok = mgr.cancel(op_id)
    return {"cancelled": bool(ok), "id": op_id}


@_route("POST", "/cancel-all")
def _cancel_all(params, query, data):
    from domain.infrastructure.cancel_manager import get_cancel_manager

    ids = get_cancel_manager().cancel_all(op_type=_opt_optype(query))
    return {"cancelled": ids, "count": len(ids)}


# ── Heavy orchestration (server-side today) ──────────────────────────


@_route("POST", "/training/distill")
def _post_training_distill(params, query, data):
    return _unsupported("Knowledge distillation")


@_route("POST", "/training/lora-finetune")
def _post_training_lora_finetune(params, query, data):
    return _unsupported("LoRA fine-tuning")
