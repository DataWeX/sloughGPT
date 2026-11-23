"""
Shared constants for infrastructure modules.

Centralizes magic numbers that were previously hardcoded across multiple files.
Import from here instead of duplicating literal values.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("slo.infrastructure.constants")

# Default timeout for model.generate() calls (seconds).
# Used by ProcessGuard, ModelServer, ModelWorkerProcess, ModelRegistry,
# SloNetServer, InferenceClient, and InferenceEngine.
DEFAULT_GENERATE_TIMEOUT: float = 120.0

# Prefill token cap (card 20261102_001622): the maximum prompt length, in
# tokens, that generate_numpy will prefill. Measured on Qwen2.5-0.5B (CPU,
# numpy fallback): 2048 tokens = 37s, 3840 = 100s, 4096 = 127-138s — the
# 120s request budget bites at ~3,980 tokens, and 8k OOM-kills a 23GB box
# before any timeout. 3072 leaves ~25-40% of the budget for decode + jitter.
# Over cap = structured error (E_PROMPT_TOO_LONG), NEVER silent truncation:
# dropping tokens from a user prompt is a product decision made at the chat
# seam (drop oldest turns, announced), not inside a kernel. Env-overridable
# like SLO_PREFILL_FALLBACK_BUDGET_MB.
DEFAULT_PREFILL_MAX_TOKENS: int = 3072

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


def prefill_max_tokens() -> int:
    """Effective prefill token cap — ``SLO_PREFILL_MAX_TOKENS`` or default.

    Env-tunable like ``SLO_PREFILL_FALLBACK_BUDGET_MB`` (the right cap
    depends on the machine: a 23 GB box and a 128 GB box do not share a
    number). Fails SAFE on malformed input — a bad value in a unit file must
    not take the server down at import time (the SLO_PREFILL_FALLBACK_
    BUDGET_MB precedent, test_prefill_fallback.py) — and a non-positive cap
    would reject every prompt, so it falls back too.
    """
    raw = os.environ.get("SLO_PREFILL_MAX_TOKENS", str(DEFAULT_PREFILL_MAX_TOKENS))
    try:
        cap = int(raw)
    except ValueError:
        logger.warning(
            "SLO_PREFILL_MAX_TOKENS=%r is not a number; using %d",
            raw,
            DEFAULT_PREFILL_MAX_TOKENS,
            extra={"tag": "INFRA"},
        )
        return DEFAULT_PREFILL_MAX_TOKENS
    if cap <= 0:
        logger.warning(
            "SLO_PREFILL_MAX_TOKENS=%r is not positive; using %d",
            raw,
            DEFAULT_PREFILL_MAX_TOKENS,
            extra={"tag": "INFRA"},
        )
        return DEFAULT_PREFILL_MAX_TOKENS
    return cap
