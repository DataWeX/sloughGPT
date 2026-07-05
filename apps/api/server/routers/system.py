"""
System Router - Host metrics and system information
"""
from fastapi import APIRouter
import psutil
import platform
import time

router = APIRouter(prefix="/system", tags=["system"])

_metrics_cache = {"data": None, "ts": 0.0}
_METRICS_TTL = 2.0  # seconds


@router.get("/metrics")
async def get_metrics():
    """Get system metrics (cached for 2s)"""
    now = time.monotonic()
    if _metrics_cache["data"] is None or (now - _metrics_cache["ts"]) > _METRICS_TTL:
        _metrics_cache["data"] = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_gb": psutil.virtual_memory().used / (1024**3),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
        }
        _metrics_cache["ts"] = now
    return _metrics_cache["data"]


@router.get("/info")
async def get_info():
    """Get system info"""
    return {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": psutil.cpu_count(),
    }


@router.get("/disk")
async def get_disk():
    """Get disk usage"""
    disk = psutil.disk_usage('/')
    return {
        "total_gb": disk.total / (1024**3),
        "used_gb": disk.used / (1024**3),
        "free_gb": disk.free / (1024**3),
        "percent": disk.percent,
    }
