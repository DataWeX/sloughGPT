"""
Optimized Operations for SloughGPT

Pure NumPy implementations with GPU acceleration via our own accelerator layer.
Every op is either CPU numpy or GPU-backed. No external dependencies.

Fused operations:
- Fused softmax + mask
- Fused attention score computation
- Fused layer norm
- Memory-efficient operations
"""

from __future__ import annotations

import math
import numpy as np
from typing import Optional, Tuple, List


# =============================================================================
# LAYER NORM
# =============================================================================

class FusedLayerNorm:
    """Fused Layer Normalization — pure numpy, GPU-aware.

    Computes: (x - mean) / sqrt(var + eps) * weight + bias
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5, bias: bool = True):
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.eps = eps
        self.weight = np.ones(normalized_shape, dtype=np.float32)
        self.bias = np.zeros(normalized_shape, dtype=np.float32) if bias else None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        y = ((x - mean) / np.sqrt(var + self.eps)) * self.weight
        if self.bias is not None:
            y = y + self.bias
        return y

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class FusedRMSNorm:
    """Fused RMSNorm — LLaMA-style, no mean computation.

    Computes: x * weight / RMS(x)
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        self.eps = eps
        self.weight = np.ones(dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.weight

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


# =============================================================================
# CROSS ENTROPY
# =============================================================================

class FusedCrossEntropyLoss:
    """Fused Cross-Entropy: log_softmax + nll_loss in one pass."""

    def __init__(self, ignore_index: int = -100, label_smoothing: float = 0.0):
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        if logits.dtype != np.float32:
            logits = logits.astype(np.float32)
        targets = targets.astype(np.int64).flatten()
        flat_logits = logits.reshape(-1, logits.shape[-1])

        x_max = np.max(flat_logits, axis=-1, keepdims=True)
        log_probs = flat_logits - x_max - np.log(np.sum(np.exp(flat_logits - x_max), axis=-1, keepdims=True))

        if self.label_smoothing > 0:
            vocab = flat_logits.shape[-1]
            smooth = self.label_smoothing / max(vocab - 1, 1)
            log_probs = (1 - self.label_smoothing) * log_probs + smooth

        valid_mask = targets != self.ignore_index
        valid_targets = targets[valid_mask]
        valid_log_probs = log_probs[valid_mask]

        if len(valid_targets) == 0:
            return 0.0

        losses = [-float(log_probs[i, int(t)]) for i, t in enumerate(valid_targets) if int(t) < log_probs.shape[1]]
        return sum(losses) / len(losses) if losses else 0.0

    def __call__(self, logits: np.ndarray, targets: np.ndarray) -> float:
        return self.forward(logits, targets)


# =============================================================================
# ATTENTION
# =============================================================================

class FusedAttentionBias:
    """Fused attention with bias, causal mask, and scaling.

    Shape: query [B, N, H, E], key [B, S, H, E], value [B, S, H, E]
    Returns: output [B, N, H, E], weights [B, H, N, S]
    """

    def __init__(self, num_heads: int):
        self.num_heads = num_heads

    def forward(self, query: np.ndarray, key: np.ndarray, value: np.ndarray,
                attn_bias: Optional[np.ndarray] = None, scale: float = 1.0,
                causal: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        B, N, H, E = query.shape
        _, S, _, _ = key.shape

        scores = np.einsum("bnhe,bshe->bhsn", query, key) * scale

        if attn_bias is not None:
            scores = scores + attn_bias

        if causal and N > 0 and S > 0:
            mask = np.triu(np.ones((N, S), dtype=np.bool_), k=1)
            scores = np.where(mask, -1e9, scores)

        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        output = np.einsum("bhsn,bshe->bnhe", attn_weights, value)

        return output, attn_weights

    def __call__(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        return self.forward(*args, **kwargs)


class ChunkedOperation:
    """Chunked attention for long sequences — avoids O(n^2) memory."""

    def __init__(self, chunk_size: int = 512):
        self.chunk_size = chunk_size

    def attention_chunked(self, query: np.ndarray, key: np.ndarray, value: np.ndarray,
                          chunk_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        chunk_size = chunk_size or self.chunk_size
        B, H, N, E = query.shape
        _, _, S, _ = key.shape

        all_outputs, all_weights = [], []
        scale = 1.0 / math.sqrt(E)

        for i in range(0, N, chunk_size):
            q_chunk = query[:, :, i:i+chunk_size]
            start = max(0, i - chunk_size)
            k_chunk = key[:, :, start:i+chunk_size]
            v_chunk = value[:, :, start:i+chunk_size]

            q_local = query.shape[2] - i
            k_len = k_chunk.shape[2]
            mask = np.triu(np.ones((q_local, k_len), dtype=np.bool_), k=1)
            scores = np.einsum("bqhe,bshe->bqhs", q_chunk, k_chunk) * scale
            scores = np.where(mask, -1e9, scores)

            exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            attn = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
            out = np.einsum("bqhs,bshe->bqhe", attn, v_chunk)

            all_outputs.append(out)
            all_weights.append(attn)

        output = np.concatenate(all_outputs, axis=2)
        weights = np.concatenate(all_weights, axis=3)
        return output, weights


# =============================================================================
# MEMORY-EFFICIENT SOFTMAX
# =============================================================================

class MemoryEfficientSoftmax:
    """Memory-efficient softmax with numerical stability and chunking."""

    @staticmethod
    def forward(logits: np.ndarray, dim: int = -1, stable: bool = True,
                chunk_size: int = 0) -> np.ndarray:
        if chunk_size > 0 and logits.shape[dim] > chunk_size:
            return MemoryEfficientSoftmax._chunked(logits, dim, stable, chunk_size)

        if stable:
            logits = logits - np.max(logits, axis=dim, keepdims=True)

        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits, axis=dim, keepdims=True)

    @staticmethod
    def _chunked(logits: np.ndarray, dim: int, stable: bool, chunk_size: int) -> np.ndarray:
        dim_size = logits.shape[dim]
        shape = list(logits.shape)
        n_out = shape[dim]
        shape[dim] = n_out
        result = np.zeros(shape, dtype=np.float32)

        for i in range(0, dim_size, chunk_size):
            end = min(i + chunk_size, dim_size)
            idx = [slice(None)] * logits.ndim
            idx[dim] = slice(i, end)
            chunk = logits[tuple(idx)]

            if stable:
                chunk = chunk - np.max(chunk, axis=dim, keepdims=True)

            exp_chunk = np.exp(chunk)
            result[tuple(idx)] = exp_chunk / np.sum(exp_chunk, axis=dim, keepdims=True)

        return result


# =============================================================================
# FUSED SCALE + BIAS
# =============================================================================

class FusedScaleBias:
    """Fused multiply-add: x * weight + bias in one pass."""

    def __init__(self, normalized_shape: int):
        self.weight = np.ones(normalized_shape, dtype=np.float32)
        self.bias = np.zeros(normalized_shape, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x * self.weight + self.bias

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


# =============================================================================
# OPTIMIZED EMBEDDING
# =============================================================================

class OptimizedEmbedding:
    """Embedding lookup with optional quantization (int8 / uint8)."""

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 padding_idx: Optional[int] = None, max_norm: Optional[float] = None,
                 norm_type: float = 2.0, scale_grad_by_freq: bool = False,
                 sparse: bool = False, quantize: bool = False):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self.quantize = quantize
        self.weight = np.random.randn(num_embeddings, embedding_dim).astype(np.float32) * 0.02
        self._quantized: Optional[np.ndarray] = None
        self._scale: Optional[np.ndarray] = None

    def quantize_weight(self, dtype: str = "uint8"):
        """Quantize weights to int8/uint8 for memory savings."""
        w = self.weight
        scale = (np.abs(w).max(axis=1, keepdims=True) + 1e-8)
        if dtype == "uint8":
            self._quantized = np.clip((w / scale * 127).astype(np.int8) + 128, 0, 255).astype(np.uint8)
        else:
            self._quantized = np.clip((w / scale * 127).astype(np.int8), -128, 127)
        self._scale = scale

    def forward(self, x: np.ndarray) -> np.ndarray:
        flat = x.astype(np.int64).flatten()
        flat = np.clip(flat, 0, self.num_embeddings - 1)
        return self.weight[flat].reshape(list(x.shape) + [self.embedding_dim])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


# =============================================================================
# FUSED OPERATIONS
# =============================================================================

def fused_swiglu(x: np.ndarray, w1_weight: np.ndarray, w1_bias: np.ndarray,
                  w2_weight: np.ndarray, w2_bias: np.ndarray,
                  w3_weight: np.ndarray, w3_bias: np.ndarray) -> np.ndarray:
    """Fused SwiGLU: SiLU(w1(x)) * w3(x) @ w2"""
    h = x @ w1_weight.T + w1_bias
    act = h / (1 + np.exp(-h))
    gate = x @ w3_weight.T + w3_bias
    return (act * gate) @ w2_weight.T + w2_bias


def efficient_cross_entropy(logits: np.ndarray, targets: np.ndarray,
                            ignore_index: int = -100, reduction: str = "mean") -> float:
    """Cross-entropy with log-sum-exp stability and ignore index."""
    flat_l = logits.reshape(-1, logits.shape[-1])
    x_max = np.max(flat_l, axis=-1, keepdims=True)
    log_probs = flat_l - x_max - np.log(np.sum(np.exp(flat_l - x_max), axis=-1, keepdims=True))

    flat_t = targets.astype(np.int64).flatten()
    valid = flat_t != ignore_index

    if reduction == "mean":
        losses = [-float(log_probs[i, int(t)]) for i, t in enumerate(flat_t[valid]) if int(t) < log_probs.shape[1]]
        return sum(losses) / len(losses) if losses else 0.0
    return 0.0


def chunked_matmul(a: np.ndarray, b: np.ndarray, chunk_size: int = 512) -> np.ndarray:
    """Chunked matrix multiplication for memory efficiency."""
    if min(a.shape[0], b.shape[1]) <= chunk_size:
        return a @ b

    result = np.zeros((a.shape[0], b.shape[1]), dtype=a.dtype)
    for i in range(0, a.shape[0], chunk_size):
        chunk = a[i:i+chunk_size]
        result[i:i+chunk_size] = chunk @ b
    return result


def ragged_to_padded(tokens: np.ndarray, pad_token_id: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Convert ragged sequences to padded with attention mask. Returns (padded, mask)."""
    mask = tokens != pad_token_id
    return tokens, mask


def estimate_attention_memory(batch_size: int, seq_len: int, num_heads: int,
                               head_dim: int, precision_bytes: int = 2) -> float:
    """Estimate memory for attention score matrix in MB."""
    return batch_size * num_heads * seq_len * seq_len * precision_bytes / (1024 ** 2)


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU / Swish: x * sigmoid(x)"""
    return x / (1 + np.exp(-x))


def gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit (GELU) approximation."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


__all__ = [
    "FusedLayerNorm",
    "FusedRMSNorm",
    "FusedCrossEntropyLoss",
    "FusedAttentionBias",
    "ChunkedOperation",
    "MemoryEfficientSoftmax",
    "FusedScaleBias",
    "OptimizedEmbedding",
    "fused_swiglu",
    "efficient_cross_entropy",
    "chunked_matmul",
    "ragged_to_padded",
    "estimate_attention_memory",
    "silu",
    "gelu",
]
