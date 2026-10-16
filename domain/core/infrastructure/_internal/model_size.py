"""
Shared model size calculator.

Uses ``downcraft`` (generic HTTP downloader) for cache health checks
and HuggingFace Hub API for file sizes.  No hardcoded size data.

Priority (always live data, never hardcoded):
1. Local HF cache — sum of weight files on disk, only if download is complete
2. HuggingFace Hub API — sum of ``.safetensors`` + ``.bin`` sibling file sizes (real bytes)
3. Returns ``None`` when size cannot be determined
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

logger = logging.getLogger("slo.infrastructure.model_size")

_SIZE_CACHE_TTL = 300  # 5 minutes
_size_cache: dict[str, tuple[float, float | None]] = {}
_size_cache_lock = threading.Lock()

# Upper bound for a single hub metadata request when a caller supplies a
# deadline (the hub default of 15s is far too slow for a request path).
_HUB_REQUEST_TIMEOUT_S = 15.0

try:
    from .hf_hub import (
        find_cached_model_dir,
        get_cache_dir,
        is_download_complete,
    )
except ImportError:
    logger.warning(
        "downcraft not available — cache completeness checks disabled", extra={"tag": "INFRA"}
    )

    def is_download_complete(model_id: str) -> bool:
        return False

    def get_cache_dir(model_id: str) -> str:
        return f"~/.cache/huggingface/hub/models--{model_id.replace('/', '--')}/"

    def find_cached_model_dir(model_id: str):
        return None


def _sum_weight_files(cache_dir: Path) -> float | None:
    """Sum sizes of all weight files in a cache directory (safetensors + bin > 1KB)."""
    total_bytes = 0
    for ext in ("*.safetensors", "*.bin"):
        for f in cache_dir.rglob(ext):
            try:
                if f.stat().st_size > 1_000:
                    total_bytes += f.stat().st_size
            except OSError:
                continue
    if total_bytes > 0:
        return round(total_bytes / (1024**3), 2)
    return None


def _get_hub_file_size_gb(model_id: str, *, timeout: float | None = None) -> float | None:
    """Get total model weight file size from HuggingFace Hub API (siblings listing).

    Args:
        model_id: HuggingFace model ID.
        timeout: Optional per-request timeout in seconds. Callers bounding a
            cold path pass the remaining budget so an unreachable hub cannot
            stall them (``requests`` applies it to connect and read
            separately, so wall time is at most ~2x this value).
    """
    try:
        from .hf_hub import fetch_model_info

        info = (
            fetch_model_info(model_id)
            if timeout is None
            else fetch_model_info(model_id, timeout=timeout)
        )
        if not info or not info.get("siblings"):
            return None
        total = 0
        for sib in info.get("siblings") or []:
            if not isinstance(sib, dict):
                continue
            name = sib.get("rfilename", "")
            if name.endswith((".safetensors", ".bin")) and sib.get("size"):
                total += sib.get("size")
        if total > 0:
            return round(total / (1024**3), 2)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.debug("sibling estimate failed: %s", exc)
    return None


def compute_model_size_gb(model_id: str, *, hub_deadline: float | None = None) -> float | None:
    """Get actual model size in GB from real file sizes only.

    Priority:
    1. Local HF cache — sum of weight files on disk, only if complete
    2. HuggingFace Hub API — sum of ``.safetensors`` + ``.bin`` sibling file sizes
    3. Returns ``None`` when size cannot be determined

    Results are cached for 5 minutes to avoid repeated API calls.

    Args:
        hub_deadline: Optional ``time.monotonic()`` deadline for starting NEW
            hub requests. When given, no hub call starts after the deadline and
            any call that does start gets only the remaining budget as its
            timeout — this is what keeps the ``/models/hf`` cold path (up to 50
            uncached models) bounded instead of stacking 15s timeouts.
            Budget-skipped results are deliberately NOT cached: "we ran out of
            time" is not "the size is unknown", and a later warm call should
            still be able to fetch it.
    """
    now = time.monotonic()
    with _size_cache_lock:
        if model_id in _size_cache:
            ts, val = _size_cache[model_id]
            if now - ts < _SIZE_CACHE_TTL:
                return val

    # 1. Local HF cache — only if download is truly complete
    if is_download_complete(model_id):
        cache_dir = Path(find_cached_model_dir(model_id) or get_cache_dir(model_id))
        cache_size = _sum_weight_files(cache_dir)
        if cache_size is not None:
            with _size_cache_lock:
                _size_cache[model_id] = (now, cache_size)
            return cache_size

    # 2. HuggingFace Hub API — real file sizes from repo sibling listing
    if hub_deadline is not None:
        remaining = hub_deadline - time.monotonic()
        if remaining <= 0:
            return None  # budget exhausted — skip, do not poison the cache
        result = _get_hub_file_size_gb(model_id, timeout=min(_HUB_REQUEST_TIMEOUT_S, remaining))
    else:
        result = _get_hub_file_size_gb(model_id)
    with _size_cache_lock:
        _size_cache[model_id] = (now, result)
    return result


_cached_check_cache: dict[str, tuple[float, bool]] = {}


def is_model_cached(model_id: str, deep_check: bool = False) -> bool:
    """Check if a model is fully downloaded to local HF cache.

    Args:
        model_id: HuggingFace model ID
        deep_check: If True, verifies every expected weight file exists
            via Hub API (slower but more accurate).

    Results are cached for 5 minutes (unless deep_check is requested).
    """
    if deep_check:
        return is_download_complete(model_id, deep_check=True)

    now = time.monotonic()
    with _size_cache_lock:
        if model_id in _cached_check_cache:
            ts, val = _cached_check_cache[model_id]
            if now - ts < _SIZE_CACHE_TTL:
                return val

    result = is_download_complete(model_id)
    with _size_cache_lock:
        _cached_check_cache[model_id] = (now, result)
    return result


def format_size_gb(size_gb: float | None, decimals: int = 2) -> str:
    if size_gb is None:
        return "—"
    return f"{size_gb:.{decimals}f} GB"


def format_size_mb(size_gb: float | None) -> float | None:
    if size_gb is None:
        return None
    return round(size_gb * 1024, 1)
