"""
Health Controller - Business logic for system health
"""
import json
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
    
    def __init__(self):
        self._start_time = datetime.now()
    
    def get_basic_health(self) -> Dict[str, Any]:
        """Get basic health status"""
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
        if model_loaded:
            try:
                import state as server_state
                if server_state.model is not None:
                    import torch
                    model = server_state.model
                    if isinstance(model, torch.nn.Module):
                        result["num_parameters"] = sum(p.numel() for p in model.parameters())
            except Exception:
                pass
        return result
    
    def get_detailed_health(self) -> Dict[str, Any]:
        """Get detailed health with system metrics and GPU info."""
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

        return {
            "status": "healthy",
            "uptime_seconds": uptime,
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu_percent": round(cpu, 1),
                "memory_percent": round(mem.percent, 1),
                "memory_available_mb": mem.available // (1024 * 1024),
            },
            "gpu": gpu_info,
            "model_loaded": model_loaded,
            "model_type": model_type,
            "inference": inference_stats,
            "registry": registry_health,
        }
    
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