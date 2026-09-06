"""
Centralized version information for SloughGPT.

All version constants live here. Import from this module rather than
hardcoding version strings elsewhere.

Includes per-feature version registry that maps backend routers to
their version numbers, matching the strui component versioning system.
"""

import importlib.metadata

APP_NAME = "SloughGPT"
API_VERSION = "1.0.0"

try:
    PKG_VERSION = importlib.metadata.version("sloughgpt")
except importlib.metadata.PackageNotFoundError:
    PKG_VERSION = "0.3.0"

try:
    import torch

    TORCH_VERSION = torch.__version__
except Exception:
    TORCH_VERSION = "unknown"

try:
    import pydantic

    PYDANTIC_VERSION = pydantic.__version__
except Exception:
    PYDANTIC_VERSION = "unknown"

# ── Per-feature version registry ────────────────────────────────────
# Maps feature domains to their backend router prefix and version.
# These correspond 1:1 with the strui FEATURE_VERSIONS manifest.
# Bump the version here when the backend feature changes; the frontend
# strui versions.ts should be updated to match.
FEATURE_VERSIONS = {
    "core": {"backend": "/", "api": "1.0.0"},
    "chat": {"backend": "/", "api": "1.0.0"},
    "models": {"backend": "/models", "api": "1.0.0"},
    "training": {"backend": "/training", "api": "1.0.0"},
    "tools": {"backend": "/agents", "api": "1.0.0"},
    "knowledge": {"backend": "/knowledge", "api": "1.0.0"},
    "layout": {"backend": "/system", "api": "1.0.0"},
    "health": {"backend": "/health", "api": "1.0.0"},
    "feedback": {"backend": "/feedback", "api": "1.0.0"},
    "datasets": {"backend": "/datasets", "api": "1.0.0"},
    "souls": {"backend": "/souls", "api": "1.0.0"},
    "memory": {"backend": "/memory", "api": "1.0.0"},
    "multimodal": {"backend": "/multimodal", "api": "1.0.0"},
    "vector": {"backend": "/vector", "api": "1.0.0"},
    "security": {"backend": "/security", "api": "1.0.0"},
    "vm": {"backend": "/vm", "api": "1.0.0"},
    "shell": {"backend": "/shell", "api": "1.0.0"},
    "infer": {"backend": "/infer", "api": "1.0.0"},
    "tokenizer": {"backend": "/tokenizer", "api": "1.0.0"},
    "benchmark": {"backend": "/benchmark", "api": "1.0.0"},
}


def version_info() -> dict:
    """Return all version constants as a dict for API responses."""
    return {
        "app": APP_NAME,
        "api": API_VERSION,
        "package": PKG_VERSION,
        "torch": TORCH_VERSION,
        "pydantic": PYDANTIC_VERSION,
        "features": FEATURE_VERSIONS,
    }
