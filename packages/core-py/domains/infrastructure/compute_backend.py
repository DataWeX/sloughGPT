"""
ComputeBackend — abstract protocol for pluggable inference backends.

Every backend (numpy, torch, C extension, triton) implements this protocol.
The rest of the system (providers, serving, weight loading) talks only to
ComputeBackend, never to a specific framework.

Usage:
    backend = NumpyBE.from_weights(weights_dict, arch_config)
    logits = backend.forward(token_ids)
    for tok in backend.generate_stream(token_ids, max_tokens=100):
        ...
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

import numpy as np

from domains.infrastructure.arch_config import ArchConfig


class ComputeBackend(abc.ABC):
    """Abstract compute backend for transformer inference.

    Subclasses implement tensor operations using their framework
    (numpy, torch, C extension, triton). The protocol is deliberately
    minimal — only the operations the inference hot path actually uses.
    """

    # ── Construction ─────────────────────────────────────────────────────

    @classmethod
    @abc.abstractmethod
    def from_weights(cls, weights: Dict[str, np.ndarray], arch: ArchConfig) -> "ComputeBackend":
        """Create a backend from pre-loaded weight arrays and architecture config.

        Args:
            weights: {tensor_name: np.ndarray} — all model weights.
            arch: Architecture config (head count, dim, norm type, etc.).

        Returns:
            Initialized backend ready for inference.
        """
        ...

    @abc.abstractmethod
    def warmup(self, seq_len: int = 1) -> None:
        """Run a warmup pass to trigger JIT compilation / memory allocation."""
        ...

    # ── Tensor primitives ───────────────────────────────────────────────

    @abc.abstractmethod
    def matmul(self, a: Any, b: Any) -> Any:
        """Matrix multiply: a @ b."""
        ...

    @abc.abstractmethod
    def softmax(self, x: Any, axis: int = -1) -> Any:
        """Softmax along axis."""
        ...

    @abc.abstractmethod
    def rmsnorm(self, x: Any, weight: Any, eps: float = 1e-6) -> Any:
        """RMS normalization: (x / sqrt(mean(x^2) + eps)) * weight."""
        ...

    @abc.abstractmethod
    def silu(self, x: Any) -> Any:
        """SiLU activation: x * sigmoid(x)."""
        ...

    @abc.abstractmethod
    def gelu(self, x: Any) -> Any:
        """GELU activation."""
        ...

    @abc.abstractmethod
    def rope(self, x: Any, cos: Any, sin: Any) -> Any:
        """Apply rotary position embeddings.

        Args:
            x: (batch, seq, heads, head_dim) or (batch, seq, dim)
            cos: precomputed cosine cache
            sin: precomputed sine cache
        """
        ...

    @abc.abstractmethod
    def repeat_kv(self, x: Any, n_reps: int) -> Any:
        """Expand KV heads for GQA: (batch, kv_heads, seq, dim) → (batch, n_heads, seq, dim)."""
        ...

    @abc.abstractmethod
    def argmax(self, x: Any) -> int:
        """Return index of max value along last axis."""
        ...

    @abc.abstractmethod
    def clip(self, x: Any, lo: Any, hi: Any) -> Any:
        """Clip values to [lo, hi]."""
        ...

    # ── Array conversion ────────────────────────────────────────────────

    @abc.abstractmethod
    def from_numpy(self, arr: np.ndarray) -> Any:
        """Convert numpy array to backend's native tensor type."""
        ...

    @abc.abstractmethod
    def to_numpy(self, tensor: Any) -> np.ndarray:
        """Convert backend's native tensor to numpy array."""
        ...

    # ── Inference ───────────────────────────────────────────────────────

    @abc.abstractmethod
    def forward(self, token_ids: np.ndarray, **kwargs) -> np.ndarray:
        """Run a single forward pass. Returns logits array.

        Args:
            token_ids: (batch, seq_len) token ids
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        ...

    @abc.abstractmethod
    def generate_stream(
        self,
        token_ids: np.ndarray,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        eos_token: Optional[int] = None,
        extra_stop_ids: Optional[Sequence[int]] = None,
    ) -> Iterator[int]:
        """Generate tokens one at a time (streaming).

        Yields:
            token_id: next generated token
        """
        ...

    @abc.abstractmethod
    def generate(
        self,
        token_ids: np.ndarray,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        eos_token: Optional[int] = None,
        extra_stop_ids: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate tokens to completion. Returns (token_ids, metrics).

        Returns:
            token_ids: (batch, prompt_len + generated) all token ids
            metrics: dict with n_tokens, ttft_ms, tokens_per_sec, etc.
        """
        ...

    # ── Metadata ────────────────────────────────────────────────────────

    @abc.abstractmethod
    def backend_name(self) -> str:
        """Return backend identifier (e.g. 'numpy', 'torch', 'native_c')."""
        ...

    @abc.abstractmethod
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        ...

    @abc.abstractmethod
    def n_layers(self) -> int:
        """Return number of transformer layers."""
        ...


# ── Registry ─────────────────────────────────────────────────────────────

_BACKENDS: Dict[str, type] = {}


def _auto_register() -> None:
    """Auto-register built-in backends."""
    try:
        from domains.infrastructure.numpy_backend import NumpyBE
        _BACKENDS["numpy"] = NumpyBE
    except ImportError:
        pass
    try:
        from domains.infrastructure.vector_backend import VectorBE
        _BACKENDS["vector"] = VectorBE
    except ImportError:
        pass


_auto_register()


def register_backend(name: str, cls: type) -> None:
    """Register a compute backend by name."""
    _BACKENDS[name] = cls


def get_backend(name: str) -> type:
    """Get a registered backend class by name."""
    if name not in _BACKENDS:
        raise KeyError(f"Unknown compute backend: {name!r}. Available: {list(_BACKENDS)}")
    return _BACKENDS[name]


def create_backend(name: str, weights: Dict[str, np.ndarray], arch: ArchConfig) -> ComputeBackend:
    """Create a backend instance by name."""
    return get_backend(name).from_weights(weights, arch)
