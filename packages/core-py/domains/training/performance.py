"""Performance Optimization Module for SloughGPT

Device detection and environment setup for training/inference.
"""

from __future__ import annotations


def get_optimal_device() -> str:
    """Auto-detect best available device (degrades to CPU).

    Delegates to ``ml_types.auto_device()`` — platform-based detection
    (MPS via Apple Silicon, CUDA via CuPy) with no torch import.
    """
    from domains.infrastructure.ml_types import auto_device
    return auto_device()


def get_device_name() -> str:
    """Get human-readable device name."""
    device = get_optimal_device()
    if device == "cuda":
        try:
            import cupy as cp
            props = cp.cuda.runtime.getDeviceProperties(0)
            return str(props["name"])
        except Exception:
            return "CUDA"
    if device == "mps":
        return "Apple Silicon (MPS)"
    return "CPU"


def setup_device_environment():
    """Setup optimal device environment variables.

    No-op on the numpy SloNet stack — torch CUDA knobs are not used.
    """
    return None
