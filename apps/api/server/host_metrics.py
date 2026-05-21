"""Host CPU/RAM samples for ``GET /info`` (runs blocking psutil in a thread)."""

from __future__ import annotations

import asyncio
import platform
from typing import Any, Dict, Optional


def sample_host_metrics_sync() -> Dict[str, Any]:
    import os

    import psutil

    cpu_percent = psutil.cpu_percent(interval=0.1)
    vm = psutil.virtual_memory()
    phys = psutil.cpu_count(logical=False)
    try:
        process_rss_bytes = int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        process_rss_bytes = None
    return {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "python_version": platform.python_version(),
        "cpu_count_logical": int(psutil.cpu_count(logical=True) or 1),
        "cpu_count_physical": int(phys) if phys is not None else None,
        "cpu_percent": round(float(cpu_percent), 2),
        "memory_total_bytes": int(vm.total),
        "memory_used_bytes": int(vm.used),
        "memory_percent": round(float(vm.percent), 2),
        "process_rss_bytes": process_rss_bytes,
    }


async def sample_host_metrics_async() -> Optional[Dict[str, Any]]:
    try:
        return await asyncio.to_thread(sample_host_metrics_sync)
    except Exception:
        return None
