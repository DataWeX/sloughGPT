"""
Shared constants for infrastructure modules.

Centralizes magic numbers that were previously hardcoded across multiple files.
Import from here instead of duplicating literal values.
"""

from __future__ import annotations

# Default timeout for model.generate() calls (seconds).
# Used by ProcessGuard, ModelServer, ModelWorkerProcess, ModelRegistry,
# SloNetServer, InferenceClient, and InferenceEngine.
DEFAULT_GENERATE_TIMEOUT: float = 120.0

# Default stall timeout for worker process monitoring (seconds).
# Used by ProcessGuard and ModelWorkerProcess.
DEFAULT_STALL_TIMEOUT: float = 120.0

# Default timeout for worker subprocess startup (seconds).
# Larger than generate_timeout because model loading can take minutes.
# Used by ModelWorkerProcess.
DEFAULT_STARTUP_TIMEOUT: float = 300.0

# Default idle timeout before model is unloaded (seconds).
# Used by IdleManager and ModelServer.
DEFAULT_IDLE_TIMEOUT: float = 300.0

# Maximum number of model load retries on transient failure.
# Used by _autoload_model in startup.py.
DEFAULT_LOAD_MAX_RETRIES: int = 2

# Base delay between model load retries (seconds, doubles each attempt).
# Used by _autoload_model in startup.py.
DEFAULT_LOAD_RETRY_DELAY: float = 5.0
