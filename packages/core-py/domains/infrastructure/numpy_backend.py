"""
NumpyBE — NumPy compute backend for pugeeq.

Implements ComputeBackend using numpy_ops. This is the reference
implementation — slow-ish but correct. The production hot path
(SloTransformer.generate_numpy) stays inlined for speed.

NumpyBE exists for:
  - Correctness testing (compare inlined vs backend output)
  - Framework swapping (torch/C backends implement same protocol)
  - Training loops that need autograd (numpy autograd via slo.autograd)
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from domains.infrastructure.compute_backend import ComputeBackend, register_backend
from domains.infrastructure.arch_config import ArchConfig
from domains.infrastructure.numpy_ops import (
    softmax as _softmax,
    rmsnorm as _rmsnorm,
    silu as _silu,
    gelu as _gelu,
)


class NumpyBE(ComputeBackend):
    """NumPy compute backend.

    Stores pre-loaded weight arrays and exposes tensor primitives.
    Does NOT own the generation loop — that lives in the provider/model.
    """

    def __init__(self, weights: Dict[str, np.ndarray], arch: ArchConfig):
        self._weights = weights
        self._arch = arch
        self._warmed = False
        # Pre-build flat lookup: canonical_name → np.ndarray
        # Handles both {i}-placeholder keys and direct keys
        self._flat = self._build_flat_lookup(weights, arch)

    @staticmethod
    def _build_flat_lookup(weights: Dict[str, np.ndarray], arch: ArchConfig) -> Dict[str, np.ndarray]:
        """Build a flat lookup dict: canonical name (with layer idx) → weight array.

        Converts weight map entries like "layers.{i}.q.weight" → "model.layers.{i}.self_attn.q_proj.weight"
        into direct lookups: "layers.0.q.weight" → weights["model.layers.0.self_attn.q_proj.weight"]
        """
        flat = {}
        W = arch.weight_map
        for canonical, mapped in W.items():
            if "{i}" in canonical:
                for layer in range(arch.n_layers):
                    actual = mapped.replace("{i}", str(layer))
                    if actual in weights:
                        flat[f"{canonical.replace('{i}', str(layer))}"] = weights[actual]
            else:
                if mapped in weights:
                    flat[canonical] = weights[mapped]
        # Also add direct keys from weights dict (e.g. model.lm_head.weight)
        for key in weights:
            if key not in flat:
                flat[key] = weights[key]
        return flat

    # ── Construction ─────────────────────────────────────────────────────

    @classmethod
    def from_weights(cls, weights: Dict[str, np.ndarray], arch: ArchConfig) -> "NumpyBE":
        return cls(weights, arch)

    def warmup(self, seq_len: int = 1) -> None:
        if self._warmed:
            return
        dummy = np.zeros((1, seq_len), dtype=np.int64)
        self.forward(dummy)
        self._warmed = True

    # ── Tensor primitives ───────────────────────────────────────────────

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a @ b

    def softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        return _softmax(x, axis=axis)

    def rmsnorm(self, x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        return _rmsnorm(x, weight, eps)

    def layer_norm(self, x: np.ndarray, weight: np.ndarray, bias: Optional[np.ndarray] = None, eps: float = 1e-5) -> np.ndarray:
        eps = np.dtype(x.dtype).type(eps)
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        out = (x - mean) / np.sqrt(var + eps) * weight
        if bias is not None:
            out = out + bias.astype(x.dtype)
        return out

    def silu(self, x: np.ndarray) -> np.ndarray:
        return _silu(x)

    def gelu(self, x: np.ndarray) -> np.ndarray:
        return _gelu(x)

    def rope(self, x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
        """Apply rotary position embeddings.

        Expects cos/sin precomputed at the caller's positions.
        x: (batch, seq, heads, head_dim) — last dim is even-split for rotation.
        cos/sin: broadcastable to x shape.
        """
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

    def repeat_kv(self, x: np.ndarray, n_reps: int) -> np.ndarray:
        """Expand KV heads for GQA: (batch, kv_heads, seq, dim) → (batch, n_heads, seq, dim)."""
        if n_reps <= 1:
            return x
        bs, nkv, sl, d = x.shape
        return np.broadcast_to(
            x[:, :, np.newaxis, :, :],
            (bs, nkv, n_reps, sl, d),
        ).reshape(bs, nkv * n_reps, sl, d)

    def argmax(self, x: np.ndarray) -> int:
        return int(np.argmax(x))

    def clip(self, x: np.ndarray, lo: Any, hi: Any) -> np.ndarray:
        return np.clip(x, lo, hi)

    # ── Array conversion ────────────────────────────────────────────────

    def from_numpy(self, arr: np.ndarray) -> np.ndarray:
        return arr

    def to_numpy(self, tensor: np.ndarray) -> np.ndarray:
        return tensor

    # ── Inference ───────────────────────────────────────────────────────

    def forward(self, token_ids: np.ndarray, **kwargs) -> np.ndarray:
        """Run a forward pass through all transformer blocks.

        This delegates to the architecture-generic forward function.
        For max speed in production, use SloTransformer.generate_numpy directly.
        """
        if token_ids.ndim == 1:
            token_ids = token_ids.reshape(1, -1)

        W = self._arch.weight_map
        arch = self._arch
        flat = self._flat

        def w(name: str) -> np.ndarray:
            if name in flat:
                return flat[name]
            raise KeyError(f"Weight '{name}' not found")

        def wn(name: str) -> Optional[np.ndarray]:
            return flat.get(name)

        norm_fn = self.rmsnorm if arch.norm == "rms_norm" else None
        seq_len = token_ids.shape[1]
        x = w("embed.token")[token_ids]
        if arch.positional == "absolute":
            x = x + w("embed.pos")[:seq_len]

        mask = np.triu(np.full((seq_len, seq_len), -1e10, dtype=np.float32), k=1)

        # Resolve layer weights and run blocks
        for i in range(arch.n_layers):
            # Attention norm
            nw = wn(f"layers.{i}.attn_norm.weight")
            nb = wn(f"layers.{i}.attn_norm.bias")
            if nw is not None:
                if norm_fn is self.rmsnorm:
                    h = self.rmsnorm(x, nw, eps=getattr(arch, 'norm_eps', 1e-6))
                else:
                    h = self.layer_norm(x, nw, nb)

            # QKV
            qw = w(f"layers.{i}.q.weight")
            kw = w(f"layers.{i}.k.weight")
            vw = w(f"layers.{i}.v.weight")
            q = self.matmul(h, qw.T)
            k = self.matmul(h, kw.T)
            v = self.matmul(h, vw.T)

            # RoPE
            if arch.positional == "rope" and arch.head_dim:
                q = q.reshape(1, seq_len, arch.n_head, arch.head_dim)
                k = k.reshape(1, seq_len, arch.n_kv_head, arch.head_dim)
                t = np.arange(0, seq_len, dtype=np.float32)
                freqs = 1.0 / (arch.rope_base ** (np.arange(0, arch.head_dim, 2, dtype=np.float32) / arch.head_dim))
                emb = np.outer(t, freqs)
                cos = np.cos(emb)[:, np.newaxis, :]
                sin = np.sin(emb)[:, np.newaxis, :]
                q = self.rope(q, cos, sin)
                k = self.rope(k, cos, sin)
                q = q.reshape(1, seq_len, -1)
                k = k.reshape(1, seq_len, -1)

            # GQA expand
            if arch.n_kv_head < arch.n_head:
                reps = arch.n_head // arch.n_kv_head
                k_r = k.reshape(1, seq_len, arch.n_kv_head, arch.head_dim)
                v_r = v.reshape(1, seq_len, arch.n_kv_head, arch.head_dim)
                # (1, seq, kv_heads, dim) → (1, seq, kv_heads, reps, dim) → (1, seq, kv_heads*reps, dim)
                k = np.broadcast_to(k_r[:, :, :, np.newaxis, :],
                                    (1, seq_len, arch.n_kv_head, reps, arch.head_dim)).reshape(1, seq_len, -1)
                v = np.broadcast_to(v_r[:, :, :, np.newaxis, :],
                                    (1, seq_len, arch.n_kv_head, reps, arch.head_dim)).reshape(1, seq_len, -1)

            # Attention
            scale = math.sqrt(arch.head_dim)
            q = q.reshape(1, seq_len, arch.n_head, -1).transpose(0, 2, 1, 3)
            k = k.reshape(1, seq_len, arch.n_head, -1).transpose(0, 2, 1, 3)
            v = v.reshape(1, seq_len, arch.n_head, -1).transpose(0, 2, 1, 3)
            attn = self.matmul(q, k.transpose(0, 1, 3, 2)) / scale
            attn = attn + mask[:seq_len, :seq_len]
            attn = self.softmax(attn, axis=-1)
            out = self.matmul(attn, v)
            out = out.transpose(0, 2, 1, 3).reshape(1, seq_len, -1)

            # Output projection
            ow = w(f"layers.{i}.o_proj.weight")
            out = self.matmul(out, ow.T)
            x = x + out

            # FFN norm
            fnw = wn(f"layers.{i}.ff_norm.weight")
            fnb = wn(f"layers.{i}.ff_norm.bias")
            if fnw is not None:
                if norm_fn is self.rmsnorm:
                    h = self.rmsnorm(x, fnw, eps=getattr(arch, 'norm_eps', 1e-6))
                else:
                    h = self.layer_norm(x, fnw, fnb)

            # FFN (SwiGLU or GELU)
            if arch.activation == "swiglu":
                gw = w(f"layers.{i}.ffn.gate.weight")
                uw = w(f"layers.{i}.ffn.up.weight")
                dw = w(f"layers.{i}.ffn.down.weight")
                gate = self.silu(self.matmul(h, gw.T))
                up = self.matmul(h, uw.T)
                x = x + self.matmul(gate * up, dw.T)
            else:
                gw = w(f"layers.{i}.ffn.weight")
                up = self.matmul(h, gw.T)
                x = x + self.matmul(self.gelu(up), w(f"layers.{i}.ffn.down.weight").T)

        # Final norm
        fnw = wn("final_norm.weight")
        fnb = wn("final_norm.bias")
        if fnw is not None:
            if norm_fn is self.rmsnorm:
                x = self.rmsnorm(x, fnw, eps=getattr(arch, 'norm_eps', 1e-6))
            else:
                x = self.layer_norm(x, fnw, fnb)

        # LM head — tied to embed tokens unless separate lm_head.weight exists
        embed_key = W.get("embed.token", "embed.token")
        embed_weight = flat.get(embed_key)
        if embed_weight is None:
            embed_weight = flat.get("model.embed_tokens.weight")
        lm_head_key = "lm_head.weight"
        lm_head_weight = flat.get(lm_head_key)
        if lm_head_weight is None:
            lm_head_weight = flat.get("model.lm_head.weight")
        if lm_head_weight is not None:
            embed_weight = lm_head_weight
        logits = self.matmul(x, embed_weight.T)

        return logits

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
        """Generate tokens one at a time using the backend's forward method."""
        if token_ids.ndim == 1:
            token_ids = token_ids.reshape(1, -1)

        stop_ids = set()
        if eos_token is not None:
            stop_ids.add(eos_token)
        if extra_stop_ids:
            stop_ids.update(extra_stop_ids)

        generated = []
        for _ in range(max_new_tokens):
            logits = self.forward(token_ids)
            next_logits = logits[0, -1, :]

            # Repetition penalty
            if repetition_penalty != 1.0 and generated:
                prev = np.array(generated, dtype=np.int64)
                next_logits[prev] /= repetition_penalty

            # Temperature
            if temperature > 1e-6:
                next_logits = next_logits / temperature

            # Top-k
            if top_k is not None and top_k > 0:
                kth = np.partition(next_logits, -top_k)[-top_k]
                next_logits[next_logits < kth] = -1e10

            # Top-p
            if top_p is not None and 0 < top_p < 1.0:
                sorted_idx = np.argsort(-next_logits)
                sorted_logits = next_logits[sorted_idx]
                cum_probs = np.cumsum(np.exp(sorted_logits - sorted_logits.max()))
                cum_probs /= cum_probs[-1]
                cutoff = np.searchsorted(cum_probs, top_p)
                if cutoff < len(sorted_idx):
                    next_logits[sorted_idx[cutoff + 1:]] = -1e10

            # Greedy or sampling
            if temperature < 1e-6:
                tok = int(np.argmax(next_logits))
            else:
                probs = np.exp(next_logits - next_logits.max())
                probs = probs / probs.sum()
                tok = int(np.random.choice(len(probs), p=probs))

            if tok in stop_ids:
                break

            generated.append(tok)
            token_ids = np.concatenate([token_ids, np.array([[tok]], dtype=np.int64)], axis=1)
            yield tok

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
        """Generate to completion. Returns (all_token_ids, metrics)."""
        import time

        t_start = time.perf_counter()
        prompt_len = token_ids.shape[1] if token_ids.ndim > 1 else len(token_ids)
        all_tokens = list(token_ids.flatten())

        for tok in self.generate_stream(
            token_ids, max_new_tokens, temperature, top_k, top_p,
            repetition_penalty, eos_token, extra_stop_ids,
        ):
            all_tokens.append(tok)

        t_end = time.perf_counter()
        n_generated = len(all_tokens) - prompt_len
        dt = t_end - t_start

        result_ids = np.array([all_tokens], dtype=np.int64)
        metrics = {
            "n_tokens": n_generated,
            "prompt_tokens": prompt_len,
            "total_tokens": len(all_tokens),
            "t_start": t_start,
            "t_end": t_end,
            "decode_ms": dt * 1000,
            "tokens_per_sec": n_generated / dt if dt > 0 else 0.0,
        }

        return result_ids, metrics

    # ── Metadata ────────────────────────────────────────────────────────

    def backend_name(self) -> str:
        return "numpy"

    def vocab_size(self) -> int:
        embed_w = self._flat.get("embed.token")
        if embed_w is None:
            embed_w = self._flat.get("model.embed_tokens.weight")
        return embed_w.shape[0] if embed_w is not None else 0

    def n_layers(self) -> int:
        return self._arch.n_layers

    # ── Helpers ─────────────────────────────────────────────────────────

    def get_weight(self, name: str) -> Optional[np.ndarray]:
        """Direct weight access for debugging / testing."""
        return self._weights.get(name)

    def weight_names(self) -> List[str]:
        """List all weight tensor names."""
        return list(self._weights.keys())


# Register numpy as the default backend
register_backend("numpy", NumpyBE)


# ── Factory ──────────────────────────────────────────────────────────────

def create_backend_from_slnc(slnc_path: str, backend_name: str = "numpy") -> ComputeBackend:
    """Create a ComputeBackend from an SLNC model file.

    This is the main entry point for loading a model into any backend.
    Infrastructure (weight loading, arch detection) is handled here;
    the backend only receives weights + arch config.

    Args:
        slnc_path: Path to .slnc model file.
        backend_name: Backend to use ("numpy", "torch", "native_c", etc.).

    Returns:
        Initialized ComputeBackend ready for inference.
    """
    from domains.infrastructure.slnc.parser import SLNCParser
    from domains.infrastructure.arch_config import build_arch
    from domains.infrastructure.compute_backend import get_backend

    parser = SLNCParser(slnc_path)
    config = parser.config
    weight_keys = set(parser._tensor_map.keys())
    arch = build_arch(config.get("_name_or_path", "model"), config, weight_keys)

    weights = parser.get_weights_dict_parallel()

    backend_cls = get_backend(backend_name)
    backend = backend_cls.from_weights(weights, arch)

    # Keep parser alive for mmap-backed backends
    if hasattr(backend, '_parser'):
        backend._parser = parser

    return backend
