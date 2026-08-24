"""
Health Controller - Business logic for system health
"""
import json
import time
from typing import Dict, Any, Tuple, Optional
import psutil
from datetime import datetime

_health_start_time = datetime.now()


def _get_executor_stats() -> Optional[Dict[str, Any]]:
    """Get TrainingExecutor pool stats if available."""
    try:
        from domains.training.executor import get_training_executor, _instance
        if _instance is None:
            return None
        ex = get_training_executor()
        return {
            "active_jobs": ex.active_count(),
            "max_workers": ex._max_workers,
            "total_tracked": len(ex._jobs),
        }
    except Exception:
        return None


def _get_process_guard_status() -> Optional[Dict[str, Any]]:
    """Get ProcessGuard status from the ModelsController if available."""
    try:
        from controllers.models import get_models_controller
        ctrl = get_models_controller()
        return ctrl.get_process_guard_status()
    except Exception:
        return None


def _get_mps_monitor_info() -> Optional[Dict[str, Any]]:
    """Get MPS GPU memory monitor status if available."""
    try:
        from domains.infrastructure.mps_monitor import get_mps_monitor
        mon = get_mps_monitor()
        return {
            "usage": round(mon.get_usage(), 3),
            "locked_to_cpu": mon.is_locked_to_cpu(),
        }
    except Exception:
        return None


def _is_model_loading() -> bool:
    """Check if the server is currently loading a model in the background.

    Returns True when the model isn't loaded yet but the server has been
    running for less than 90s (the typical model load window).
    """
    try:
        import state as server_state
        if server_state.model is not None:
            return False
        if server_state.provider is not None:
            return False
        uptime = (datetime.now() - _health_start_time).total_seconds()
        return uptime < 90
    except Exception:
        return False


def _is_app_ready() -> bool:
    """True only when the app has finished startup (routers registered).

    Health must not report the model as loaded until the lifecycle reaches
    RUNNING — otherwise clients see ``model_loaded: true`` during the startup
    window and hit routes that are not registered yet (404 "Not Found").

    Reads from ``STARTUP_PHASE`` (set by the startup orchestrator) instead of
    creating the lifecycle singleton, which would race with
    ``StartupOrchestrator._init_lifecycle()`` when health routes are queried
    pre-lifespan.
    """
    try:
        from startup_progress import STARTUP_PHASE
        return STARTUP_PHASE.get("phase") in ("running", "ready")
    except Exception:
        return True


def _get_model_info() -> Tuple[bool, Optional[str]]:
    """Get model info from registry, controller, or server_state."""
    loaded, model_type, _ = _get_model_info_with_registry()
    if loaded and not _is_app_ready():
        return False, model_type
    return loaded, model_type


def _get_model_info_with_registry() -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """Get model info and registry health in a single query."""
    registry_health: Dict[str, Any] = {}

    # Check ModelRegistry first (most authoritative)
    try:
        from domains.infrastructure.model_registry import get_model_registry
        registry = get_model_registry()
        registry_health = registry.health_summary()
        if registry_health.get("healthy") and registry_health.get("default_model"):
            return True, registry_health["default_model"], registry_health
    except ImportError:
        pass

    # Fallback: check models controller
    try:
        from controllers.models import get_models_controller
        ctrl = get_models_controller()
        current = ctrl.get_current_model()
        if current:
            return True, current.get("model_id"), registry_health
    except ImportError:
        pass

    # Fallback: check server_state (used by autoload in lifespan)
    try:
        import state as server_state
        if server_state.model is not None:
            return True, server_state.model_type, registry_health
        if server_state.provider is not None:
            return True, server_state.model_type, registry_health
    except ImportError:
        pass

    return False, None, registry_health


def _get_model_device() -> Optional[str]:
    """Resolve the active model's device string for health reporting.

    Order: models controller first (authoritative — it holds the resolved
    device after availability validation in ``_resolve_device``), then the
    ModelRegistry default model, then ``None``.

    Returns:
        The resolved device string (e.g. ``"cpu"``, ``"mps"``, ``"cuda"``),
        or None when no model is loaded.
    """
    try:
        from controllers.models import get_models_controller
        ctrl = get_models_controller()
        current = ctrl.get_current_model()
        if current and current.get("device"):
            return current["device"]
    except ImportError:
        pass
    try:
        from domains.infrastructure.model_registry import get_model_registry
        registry = get_model_registry()
        health = registry.health_summary()
        if health.get("healthy"):
            for m in health.get("models", []):
                if m.get("is_default") and m.get("device"):
                    return m["device"]
    except Exception:
        pass
    return None


def _get_lifecycle_info() -> Dict[str, Any]:
    """Get lifecycle phase and profile info from the lifecycle manager."""
    try:
        from domains.infrastructure.lifecycle import get_lifecycle_manager
        mgr = get_lifecycle_manager()
        return {
            "phase": mgr.phase.value,
            "profile": mgr.get_profile().value,
            "is_running": mgr.is_running(),
            "is_draining": mgr.is_draining(),
            "uptime": round(mgr.uptime_seconds, 1),
            "in_flight": mgr.in_flight_count,
        }
    except Exception as exc:
        return {
            "phase": "unavailable",
            "profile": "unknown",
            "error": str(exc),
        }


def _get_inference_stats() -> Dict[str, Any]:
    """Get inference stats from models controller"""
    try:
        from controllers.models import get_models_controller
        ctrl = get_models_controller()
        return ctrl.get_inference_stats()
    except ImportError:
        return {}
    except Exception:
        return {}


def _get_quantization_info() -> Dict[str, Any]:
    """Get quantization status from the active provider."""
    try:
        from domains.models.provider import get_provider
        provider = get_provider("slonet-native")
        if provider is None:
            provider = get_provider("slonet")
        if provider is None:
            provider = get_provider("hf-default")
        if provider is not None and hasattr(provider, 'quantization_report'):
            return provider.quantization_report()
        return {}
    except Exception:
        return {}


def _get_kv_session_info() -> Dict[str, Any]:
    """Get cross-turn KV cache session stats from the active provider.

    Surfaces ``SloNetChatProvider.session_stats()`` (active sessions, cached
    tokens, TTL). Returns ``{"enabled": False}`` when no provider exposes it.
    """
    try:
        from domains.models.provider import get_provider
        provider = get_provider("slonet-native")
        if provider is None:
            provider = get_provider("slonet")
        if provider is not None and hasattr(provider, "session_stats"):
            stats = provider.session_stats()
            stats["enabled"] = True
            return stats
        return {"enabled": False}
    except Exception:
        return {"enabled": False}


def _build_status_message(
    model_loaded: bool,
    model_type: Optional[str],
    model_loading: bool,
    current_soul: Optional[str],
    request_count: int,
    error_count: int,
    lifecycle: Dict[str, Any],
) -> str:
    """Build a human-readable status message incorporating lifecycle phase."""
    phase = lifecycle.get("phase", "unknown")
    profile = lifecycle.get("profile", "unknown")

    if lifecycle.get("is_draining"):
        return f"Draining — completing {lifecycle.get('in_flight', 0)} in-flight requests."

    if not lifecycle.get("is_running", False):
        return f"Starting — phase={phase}, profile={profile}."

    if model_loaded:
        msg = (
            f"Ready — {model_type or 'model'} loaded"
            + (f" with {current_soul} personality" if current_soul else "")
            + f". Served {request_count} requests."
            + (f" {error_count} errors." if error_count > 0 else "")
        )
    elif model_loading:
        msg = "Loading model weights — this takes about a minute on first start."
    else:
        msg = f"Server running, no model loaded. Profile: {profile}."
    return msg


def _get_resource_allocation() -> Dict[str, Any]:
    """Get CPU topology resource allocation."""
    try:
        from domains.infrastructure.resource_manager import get_resource_manager
        rm = get_resource_manager()
        return {
            "mode": rm.mode,
            "compute_threads": rm.compute_threads,
            "io_threads": rm.io_threads,
            "omp_num_threads": rm.omp_num_threads,
            "mkl_num_threads": rm.mkl_num_threads,
            "openblas_num_threads": rm.openblas_num_threads,
            "numexpr_num_threads": rm.numexpr_num_threads,
            "inference_pool_size": rm.inference_pool_size,
            "train_pool_size": rm.train_pool_size,
            "task_queue_workers": rm.task_queue_workers,
            "dataloader_workers": rm.dataloader_workers,
            "concurrent_reads": rm.concurrent_reads,
            "concurrent_writes": rm.concurrent_writes,
            "process_guard_concurrent": rm.process_guard_concurrent,
        }
    except Exception:
        return {}


_cached_process: Optional["psutil.Process"] = None


def _get_process() -> "psutil.Process":
    """Return a cached psutil.Process instance for the current process."""
    global _cached_process
    if _cached_process is None:
        _cached_process = psutil.Process()
    return _cached_process


def _get_process_info() -> Dict[str, Any]:
    """Compute real process metrics for the current server process.

    Uses psutil for file descriptors/threads/memory and the stdlib ``gc``
    module for garbage-collector generation counts. Returns ``{}`` (the
    frontend falls back to placeholder values) when any read fails.

    Returns:
        dict with keys open_files, threads, process_cpu_percent,
        process_memory_percent, rss_mb, gc_gen0, gc_gen1, gc_gen2.

    Side effects:
        - reads process stats via psutil and gc (no mutation)
    """
    try:
        import gc
        proc = _get_process()
        info: Dict[str, Any] = {
            "threads": proc.num_threads(),
            "process_cpu_percent": round(proc.cpu_percent(interval=None), 1),
            "process_memory_percent": round(proc.memory_percent(), 1),
            "rss_mb": proc.memory_info().rss // (1024 * 1024),
        }
        try:
            gen0, gen1, gen2 = gc.get_count()
            info["gc_gen0"] = gen0
            info["gc_gen1"] = gen1
            info["gc_gen2"] = gen2
        except Exception:
            pass
        return info
    except Exception:
        return {}


class HealthController:
    """Controller for system health"""

    _CACHE_TTL = 2.0  # seconds

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._cache_time: float = 0.0
        # Warm up psutil cpu_percent — first call always returns 0.0
        try:
            psutil.cpu_percent(interval=None)
            psutil.Process().cpu_percent(interval=None)
        except Exception:
            pass

    def get_basic_health(self) -> Dict[str, Any]:
        """Get basic health status with flow-based summary."""
        model_loaded, model_type = _get_model_info()
        inference_stats = _get_inference_stats()
        lifecycle = _get_lifecycle_info()
        model_loading = not model_loaded and _is_model_loading()
        result: Dict[str, Any] = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_loaded": model_loaded,
            "model_loading": model_loading,
            "model_type": model_type,
            "device": _get_model_device(),
            "is_inferencing": inference_stats.get("is_inferencing", False),
            "inference_count": inference_stats.get("inference_count", 0),
            "lifecycle": lifecycle,
        }

        # Status message from lifecycle + model state
        result["status_message"] = _build_status_message(
            model_loaded, model_type, model_loading, None, 0, 0, lifecycle,
        )

        if model_loaded:
            try:
                import state as server_state
                if server_state.model is not None:
                    model = server_state.model
                    if hasattr(model, "parameters") and callable(getattr(model, "parameters", None)):
                        result["num_parameters"] = sum(p.numel() for p in model.parameters())
                elif server_state.provider is not None:
                    meta = server_state.provider.metadata()
                    if meta and meta.get("total_params"):
                        result["num_parameters"] = meta["total_params"]
            except Exception:
                pass

        # Quantization status
        quant_info = _get_quantization_info()
        if quant_info:
            result["quantization"] = quant_info

        # Cross-turn KV cache session stats
        kv_sessions = _get_kv_session_info()
        if kv_sessions.get("enabled"):
            result["kv_sessions"] = kv_sessions

        # Training executor pool status
        executor_stats = _get_executor_stats()
        if executor_stats:
            result["training_pool"] = executor_stats

        # Resource allocation from CPU topology
        result["resource_allocation"] = _get_resource_allocation()

        # Idle manager status
        try:
            from domains.infrastructure.model_server import get_idle_manager
            idle_mgr = get_idle_manager()
            import state as server_state
            model_id = server_state.model_type or "unknown"
            idle_info = idle_mgr.get_idle_info(model_id)
            if idle_info:
                result["idle"] = idle_info
        except Exception:
            pass

        return result

    def get_detailed_health(self) -> Dict[str, Any]:
        """Get detailed health with system metrics and GPU info (cached up to CACHE_TTL seconds)."""
        now = time.monotonic()
        if self._cache and (now - self._cache_time) < self._CACHE_TTL:
            return self._cache

        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        model_loaded, model_type, registry_health = _get_model_info_with_registry()
        inference_stats = _get_inference_stats()
        model_loading = not model_loaded and _is_model_loading()
        uptime = (datetime.now() - _health_start_time).total_seconds()

        gpu_info: Dict[str, Any] = {}
        try:
            from domains.slolib.gpu import get_accelerator
            acc = get_accelerator()
            gpu_info = {
                "backend": acc.name,
                "device_type": acc.device_type,
                "vram_gb": round(acc.vram_gb(), 2),
                "tier": acc.compute_tier,
                "memory_hint": json.dumps(acc.memory_hint()),
            }
        except Exception as e:
            gpu_info = {"backend": "unknown", "error": str(e)}

        # Add ServerState counters if available
        try:
            from domains.infrastructure.server_state import get_server_state
            ss = get_server_state()
            request_count = ss.request_count
            error_count = ss.error_count
            current_soul = ss.current_soul.get()
            avg_latency = ss.get_avg_latency()
            p95_latency = ss.get_p95_latency()
            requests_per_min = ss.get_requests_per_minute()
            path_latencies = ss.get_path_latencies(5)
            recent_errors = ss.get_error_history(5)
            inference_count = ss.inference_count
            total_tokens = ss.total_tokens
            tokens_per_sec = ss.get_tokens_per_second()
            avg_tokens_per_req = ss.get_avg_tokens_per_request()
            health_score = ss.get_health_score(cpu_percent=cpu, memory_percent=mem.percent)
            model_metrics = ss.get_model_metrics()
            model_events = ss.get_model_events(10)
            ss.record_trend_snapshots()
            health_history = ss.get_health_history(20)
            memory_history = ss.get_memory_history(10)
            rate_violations = ss.get_rate_limit_violations(5)
        except Exception:
            request_count = 0
            error_count = 0
            current_soul = None
            avg_latency = 0.0
            p95_latency = 0.0
            requests_per_min = 0.0
            path_latencies = []
            recent_errors = []
            inference_count = 0
            total_tokens = 0
            tokens_per_sec = 0.0
            avg_tokens_per_req = 0.0
            health_score = {"score": 0, "status": "unknown"}
            model_metrics = []
            model_events = []
            health_history = []
            memory_history = []
            rate_violations = []

        lifecycle = _get_lifecycle_info()

        result = {
            "status": "healthy" if lifecycle.get("is_running", False) else lifecycle.get("phase", "unknown"),
            "uptime_seconds": uptime,
            "timestamp": datetime.now().isoformat(),
            "request_count": request_count,
            "error_count": error_count,
            "avg_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency,
            "requests_per_minute": requests_per_min,
            "path_latencies": path_latencies,
            "recent_errors": recent_errors,
            "inference_count": inference_count,
            "total_tokens": total_tokens,
            "tokens_per_sec": tokens_per_sec,
            "avg_tokens_per_request": avg_tokens_per_req,
            "health_score": health_score,
            "model_metrics": model_metrics,
            "model_events": model_events,
            "health_history": health_history,
            "memory_history": memory_history,
            "rate_violations": rate_violations,
            "system": {
                "cpu_percent": round(cpu, 1),
                "memory_percent": round(mem.percent, 1),
                "memory_available_mb": mem.available // (1024 * 1024),
                **_get_process_info(),
            },
            "gpu": gpu_info,
            "mps_monitor": _get_mps_monitor_info(),
            "model_loaded": model_loaded,
            "model_loading": model_loading,
            "model_type": model_type,
            "device": _get_model_device(),
            "soul": current_soul,
            "inference": inference_stats,
            "registry": registry_health,
            "quantization": _get_quantization_info(),
            "kv_sessions": _get_kv_session_info(),
            "lifecycle": lifecycle,
            "training_pool": _get_executor_stats(),
            "resource_allocation": _get_resource_allocation(),
            "process_guard": _get_process_guard_status(),
            "status_message": _build_status_message(
                model_loaded, model_type, model_loading, current_soul,
                request_count, error_count, lifecycle,
            ),
        }
        self._cache = result
        self._cache_time = now
        return result

    def get_liveness(self) -> Dict[str, Any]:
        """Kubernetes liveness probe"""
        return {"status": "alive"}

    def get_readiness(self) -> Dict[str, Any]:
        """Kubernetes readiness probe"""
        return {"status": "ready"}


_health_controller: Optional[HealthController] = None


def get_health_controller() -> HealthController:
    global _health_controller
    if _health_controller is None:
        _health_controller = HealthController()
    return _health_controller
