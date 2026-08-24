"""
Shared constants for infrastructure modules.

Centralizes magic numbers that were previously hardcoded across multiple files.
Import from here instead of duplicating literal values.
"""

# Default timeout for model.generate() calls (seconds).
# Used by ProcessGuard, ModelServer, ModelWorkerProcess, ModelRegistry,
# SloNetServer, InferenceClient, and InferenceEngine.
DEFAULT_GENERATE_TIMEOUT: float = 120.0

# Default stall timeout for worker process monitoring (seconds).
# Used by ProcessGuard and ModelWorkerProcess.
DEFAULT_STALL_TIMEOUT: float = 120.0
