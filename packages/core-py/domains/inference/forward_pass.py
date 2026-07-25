"""
forward_pass.py — Unified forward pass interface for all transformer engines.

Any model loaded into the NPU must implement ForwardPassable:
  - forward_pass(input_ids) -> ForwardPassResult

Both SloTransformer (numpy) and TransformerEngine (C) implement this.
The NPU calls forward_pass() without knowing which backend is running.
"""

from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class ForwardPassResult:
    """Result of a single forward pass through any transformer engine."""
    logits: np.ndarray           # (batch, seq_len, vocab_size)
    forward_time_ms: float = 0.0
    model_name: str = ""
    cached_tokens: int = 0       # how many tokens were in KV cache before this call
    engine: str = "unknown"      # "numpy" or "c"

    @property
    def shape(self):
        return list(self.logits.shape)


@runtime_checkable
class ForwardPassable(Protocol):
    """Any transformer that can run a forward pass on token IDs."""

    def forward_pass(self, input_ids: np.ndarray) -> ForwardPassResult:
        """
        Run a forward pass through the model.

        Args:
            input_ids: int64 array of shape (batch, seq_len) containing token IDs

        Returns:
            ForwardPassResult with logits array of shape (batch, seq_len, vocab_size)
        """
        ...


def timed_forward(model: ForwardPassable, input_ids: np.ndarray,
                  model_name: str = "") -> ForwardPassResult:
    """Run forward_pass() with automatic timing."""
    t0 = time.monotonic()
    result = model.forward_pass(input_ids)
    result.forward_time_ms = (time.monotonic() - t0) * 1000
    result.model_name = model_name
    return result
