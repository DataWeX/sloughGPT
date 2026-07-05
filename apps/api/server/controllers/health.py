"""
Health Controller - Business logic for system health
"""
import json
import time
from typing import Dict, Any, Tuple, Optional
import psutil
from datetime import datetime


def _get_model_info() -> Tuple[bool, Optional[str]]:
    """Get model info from registry, controller, or server_state."""
    # Check ModelRegistry first (most authoritative)
    try:
        from domains.infrastructure.model_registry import get_model_registry
        registry = get_model_registry()
        health = registry.health_summary()
        if health["healthy"] and health["default_model"]:
            return True, health["default_model"]
    except ImportError:
        pass

    # Fallback: check models controller
    try:
        from controllers.models import get_models_controller
        ctrl = get_models_controller()
        current = ctrl.get_current_model()
        if current:
            return True, current.get("model_id")
    except ImportError:
        pass

    # Fallback: check server_state (used by autoload in lifespan)
    try:
        import state as server_state
        if server_state.model is not None:
            return True, server_state.model_type
    except ImportError:
        pass

    return False, None


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


class HealthController:
    """Controller for system health"""

    _CACHE_TTL = 2.0  # seconds

    def __init__(self):
        self._start_time = datetime.now()
        self._cache: Dict[str, Any] = {}
        self._cache_time: float = 0.0

    def get_basic_health(self) -> Dict[str, Any]:
        """Get basic health status with flow-based summary."""
        model_loaded, model_type = _get_model_info()
        inference_stats = _get_inference_stats()
        result: Dict[str, Any] = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_loaded": model_loaded,
            "model_type": model_type,
            "is_inferencing": inference_stats.get("is_inferencing", False),
            "inference_count": inference_stats.get("inference_count", 0),
        }

        # Flow-based summary sentence
        try:
            from domains.infrastructure.server_state import get_server_state
            ss = get_server_state()
            hs = ss.get_health_score()
            result["summary"] = hs.get("summary", "")
        except Exception:
            result["summary"] = ""

        if model_loaded:
            try:
                import state as server_state
                if server_state.model is not None:
                    model = server_state.model
                    if hasattr(model, "parameters") and callable(getattr(model, "parameters", None)):
                        result["num_parameters"] = sum(p.numel() for p in model.parameters())
            except Exception:
                pass

        return result

    def get_detailed_health(self) -> Dict[str, Any]:
        """Get detailed health with system metrics and GPU info (cached up to CACHE_TTL seconds)."""
        now = time.monotonic()
        if self._cache and (now - self._cache_time) < self._CACHE_TTL:
            return self._cache

        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        model_loaded, model_type = _get_model_info()
        inference_stats = _get_inference_stats()
        uptime = (datetime.now() - self._start_time).total_seconds()

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
        except Exception:
            gpu_info = {"backend": "unknown", "error": str(Exception)}

        # Add ModelRegistry health if available
        registry_health: Dict[str, Any] = {}
        try:
            from domains.infrastructure.model_registry import get_model_registry
            reg = get_model_registry()
            registry_health = reg.health_summary()
        except Exception:
            pass

        # Add ServerState counters if available
        try:
            from domains.infrastructure.server_state import get_server_state
            ss = get_server_state()
            request_count = ss.request_count
            error_count = ss.error_count
            current_soul = ss.current_soul.get()
            avg_latency = ss.get_avg_latency()
            requests_per_min = ss.get_requests_per_minute()
            path_latencies = ss.get_path_latencies(5)
            recent_errors = ss.get_error_history(5)
            inference_count = ss.inference_count
            total_tokens = ss.total_tokens
            tokens_per_sec = ss.get_tokens_per_second()
            avg_tokens_per_req = ss.get_avg_tokens_per_request()
            health_score = ss.get_health_score()
            model_metrics = ss.get_model_metrics()
            model_events = ss.get_model_events(10)
            health_history = ss.get_health_history(20)
            memory_history = ss.get_memory_history(10)
            rate_violations = ss.get_rate_limit_violations(5)
        except Exception:
            request_count = 0
            error_count = 0
            current_soul = None
            avg_latency = 0.0
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

        result = {
            "status": "healthy",
            "uptime_seconds": uptime,
            "timestamp": datetime.now().isoformat(),
            "request_count": request_count,
            "error_count": error_count,
            "avg_latency_ms": avg_latency,
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
            },
            "gpu": gpu_info,
            "model_loaded": model_loaded,
            "model_type": model_type,
            "soul": current_soul,
            "inference": inference_stats,
            "registry": registry_health,
            "status_message": (
                f"Ready — {model_type or 'model'} loaded"
                + (f" with {current_soul} personality" if current_soul else "")
                + f". Served {request_count} requests."
                + (f" {error_count} errors." if error_count > 0 else "")
            ) if model_loaded else "Server running, no model loaded.",
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
