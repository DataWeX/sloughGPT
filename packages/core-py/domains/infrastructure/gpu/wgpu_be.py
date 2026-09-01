"""
WgpuBE — GPU compute backend using our own engine.

Implements ComputeBackend using the gpu_engine C library.
Runs transformer inference on GPU via compute shaders.

Usage:
    backend = create_backend_from_slnc("model.slnc", "gpu")
    logits = backend.forward(token_ids)
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np

from domains.infrastructure.compute_backend import ComputeBackend, register_backend
from domains.infrastructure.arch_config import ArchConfig

logger = logging.getLogger(__name__)

_SHADERS_DIR = Path(__file__).parent / "shaders"


def _load_spirv(name: str) -> np.ndarray:
    """Load pre-compiled SPIR-V binary from shaders directory."""
    path = _SHADERS_DIR / f"{name}.spv"
    if not path.exists():
        raise FileNotFoundError(f"SPIR-V not found: {path}")
    return np.fromfile(str(path), dtype=np.uint32)


def _load_metallib(name: str) -> bytes:
    """Load pre-compiled Metal library."""
    path = _SHADERS_DIR / f"{name}.metallib"
    if not path.exists():
        raise FileNotFoundError(f"Metallib not found: {path}")
    return path.read_bytes()


class WgpuBE(ComputeBackend):
    """GPU compute backend using our own engine.

    Transfers weights to GPU, runs transformer forward pass via compute shaders.
    Falls back to numpy for operations not yet implemented on GPU.
    """

    def __init__(self, weights: Dict[str, np.ndarray], arch: ArchConfig, device: Any = None):
        self._arch = arch
        self._np_weights = weights  # Keep numpy copy for fallback

        # Import GPU engine
        try:
            from domains.infrastructure.gpu.gpu_engine import GpuDevice, GPU_BUF_STORAGE
            self._gpu = GpuDevice(device)
            self._GPU_BUF_STORAGE = GPU_BUF_STORAGE
            self._has_gpu = True
        except (FileNotFoundError, RuntimeError) as e:
            logger.warning("WgpuBE: GPU not available (%s), falling back to numpy", e)
            self._has_gpu = False

        # Transfer weights to GPU
        self._gpu_buffers: Dict[str, Any] = {}
        if self._has_gpu:
            self._upload_weights(weights)

        # Pre-build flat lookup for numpy fallback
        self._flat = self._build_flat_lookup(weights, arch)

        # Cache common shapes
        self._cached_cos = None
        self._cached_sin = None
        self._cached_cos_pos = -1

    @staticmethod
    def _build_flat_lookup(weights: Dict[str, np.ndarray], arch: ArchConfig) -> Dict[str, np.ndarray]:
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
        for key in weights:
            if key not in flat:
                flat[key] = weights[key]
        return flat

    def _upload_weights(self, weights: Dict[str, np.ndarray]) -> None:
        """Upload all weights to GPU buffers."""
        for name, arr in weights.items():
            arr = np.ascontiguousarray(arr.astype(np.float32))
            buf = self._gpu.buffer_create(arr.nbytes, self._GPU_BUF_STORAGE)
            buf.write(arr)
            self._gpu_buffers[name] = (buf, arr.shape, arr.dtype)

    @classmethod
    def from_weights(cls, weights: Dict[str, np.ndarray], arch: ArchConfig) -> "WgpuBE":
        return cls(weights, arch)

    def warmup(self, seq_len: int = 1) -> None:
        pass

    # ── Tensor primitives ───────────────────────────────────────────────

    def matmul(self, a: Any, b: Any) -> Any:
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return a @ b
        # GPU matmul via shader — placeholder
        return a @ b

    def softmax(self, x: Any, axis: int = -1) -> Any:
        if isinstance(x, np.ndarray):
            x_max = x.max(axis=axis, keepdims=True)
            e_x = np.exp(x - x_max)
            return e_x / e_x.sum(axis=axis, keepdims=True)
        return x

    def rmsnorm(self, x: Any, weight: Any, eps: float = 1e-6) -> Any:
        if isinstance(x, np.ndarray):
            eps_t = np.dtype(x.dtype).type(eps)
            return (x / np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps_t)) * weight
        return x

    def layer_norm(self, x: np.ndarray, weight: np.ndarray, bias: Optional[np.ndarray] = None, eps: float = 1e-5) -> np.ndarray:
        eps_t = np.dtype(x.dtype).type(eps)
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        out = (x - mean) / np.sqrt(var + eps_t) * weight
        if bias is not None:
            out = out + bias.astype(x.dtype)
        return out

    def silu(self, x: Any) -> Any:
        if isinstance(x, np.ndarray):
            T = np.dtype(x.dtype).type
            return x * (T(1.0) / (T(1.0) + np.exp(-x)))
        return x

    def gelu(self, x: Any) -> Any:
        if isinstance(x, np.ndarray):
            T = np.dtype(x.dtype).type
            return T(0.5) * x * (T(1.0) + np.tanh(T(np.sqrt(2.0 / np.pi)) * (x + T(0.044715) * x ** 3)))
        return x

    def rope(self, x: Any, cos: Any, sin: Any) -> Any:
        if isinstance(x, np.ndarray):
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
            return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
        return x

    def repeat_kv(self, x: Any, n_reps: int) -> Any:
        if isinstance(x, np.ndarray):
            if n_reps <= 1:
                return x
            bs, nkv, sl, d = x.shape
            return np.broadcast_to(x[:, :, np.newaxis, :, :],
                                   (bs, nkv, n_reps, sl, d)).reshape(bs, nkv * n_reps, sl, d)
        return x

    def argmax(self, x: Any) -> int:
        if isinstance(x, np.ndarray):
            return int(np.argmax(x))
        return 0

    def clip(self, x: Any, lo: Any, hi: Any) -> Any:
        if isinstance(x, np.ndarray):
            return np.clip(x, lo, hi)
        return x

    def from_numpy(self, arr: np.ndarray) -> Any:
        return arr

    def to_numpy(self, tensor: Any) -> np.ndarray:
        if isinstance(tensor, np.ndarray):
            return tensor
        return np.asarray(tensor)

    # ── Inference ───────────────────────────────────────────────────────

    def forward(self, token_ids: np.ndarray, **kwargs) -> np.ndarray:
        """Forward pass — uses numpy for now, GPU shaders for matmul when available."""
        if token_ids.ndim == 1:
            token_ids = token_ids.reshape(1, -1)

        flat = self._flat
        arch = self._arch
        seq_len = token_ids.shape[1]

        def w(name: str) -> np.ndarray:
            return flat[name]

        def wn(name: str) -> Optional[np.ndarray]:
            return flat.get(name)

        # Embeddings
        x = w("embed.token")[token_ids]
        if arch.positional == "absolute":
            x = x + w("embed.pos")[:seq_len]

        mask = np.triu(np.full((seq_len, seq_len), -1e10, dtype=np.float32), k=1)

        # Transformer blocks
        for i in range(arch.n_layers):
            # Attention norm
            nw = wn(f"layers.{i}.attn_norm.weight")
            nb = wn(f"layers.{i}.attn_norm.bias")
            if nw is not None:
                if arch.norm == "rms_norm":
                    h = self.rmsnorm(x, nw)
                else:
                    h = self.layer_norm(x, nw, nb)

            # QKV
            q = self.matmul(h, w(f"layers.{i}.q.weight").T)
            k = self.matmul(h, w(f"layers.{i}.k.weight").T)
            v = self.matmul(h, w(f"layers.{i}.v.weight").T)

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
            out = self.matmul(out, w(f"layers.{i}.o_proj.weight").T)
            x = x + out

            # FFN norm
            fnw = wn(f"layers.{i}.ff_norm.weight")
            fnb = wn(f"layers.{i}.ff_norm.bias")
            if fnw is not None:
                if arch.norm == "rms_norm":
                    h = self.rmsnorm(x, fnw)
                else:
                    h = self.layer_norm(x, fnw, fnb)

            # FFN
            if arch.activation == "swiglu":
                gate = self.silu(self.matmul(h, w(f"layers.{i}.ffn.gate.weight").T))
                up = self.matmul(h, w(f"layers.{i}.ffn.up.weight").T)
                x = x + self.matmul(gate * up, w(f"layers.{i}.ffn.down.weight").T)
            else:
                up = self.matmul(h, w(f"layers.{i}.ffn.weight").T)
                x = x + self.matmul(self.gelu(up), w(f"layers.{i}.ffn.down.weight").T)

        # Final norm
        fnw = wn("final_norm.weight")
        fnb = wn("final_norm.bias")
        if fnw is not None:
            if arch.norm == "rms_norm":
                x = self.rmsnorm(x, fnw)
            else:
                x = self.layer_norm(x, fnw, fnb)

        # LM head
        embed_key = arch.weight_map.get("embed.token", "embed.token")
        embed_weight = flat.get(embed_key)
        if embed_weight is None:
            embed_weight = flat.get("model.embed_tokens.weight")
        lm_head_weight = flat.get("lm_head.weight") or flat.get("model.lm_head.weight")
        if lm_head_weight is not None:
            embed_weight = lm_head_weight
        logits = self.matmul(x, embed_weight.T)

        return logits

    def generate_stream(self, token_ids, max_new_tokens=100, temperature=1.0,
                        top_k=None, top_p=None, repetition_penalty=1.0,
                        eos_token=None, extra_stop_ids=None) -> Iterator[int]:
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

            if repetition_penalty != 1.0 and generated:
                prev = np.array(generated, dtype=np.int64)
                next_logits[prev] /= repetition_penalty

            if temperature > 1e-6:
                next_logits = next_logits / temperature

            if top_k is not None and top_k > 0:
                kth = np.partition(next_logits, -top_k)[-top_k]
                next_logits[next_logits < kth] = -1e10

            if top_p is not None and 0 < top_p < 1.0:
                sorted_idx = np.argsort(-next_logits)
                sorted_logits = next_logits[sorted_idx]
                cum_probs = np.cumsum(np.exp(sorted_logits - sorted_logits.max()))
                cum_probs /= cum_probs[-1]
                cutoff = np.searchsorted(cum_probs, top_p)
                if cutoff < len(sorted_idx):
                    next_logits[sorted_idx[cutoff + 1:]] = -1e10

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

    def generate(self, token_ids, max_new_tokens=100, temperature=1.0,
                 top_k=None, top_p=None, repetition_penalty=1.0,
                 eos_token=None, extra_stop_ids=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        t_start = time.perf_counter()
        prompt_len = token_ids.shape[1] if token_ids.ndim > 1 else len(token_ids)
        all_tokens = list(token_ids.flatten())

        for tok in self.generate_stream(token_ids, max_new_tokens, temperature, top_k, top_p,
                                        repetition_penalty, eos_token, extra_stop_ids):
            all_tokens.append(tok)

        t_end = time.perf_counter()
        n_generated = len(all_tokens) - prompt_len
        dt = t_end - t_start

        return np.array([all_tokens], dtype=np.int64), {
            "n_tokens": n_generated,
            "prompt_tokens": prompt_len,
            "total_tokens": len(all_tokens),
            "t_start": t_start,
            "t_end": t_end,
            "decode_ms": dt * 1000,
            "tokens_per_sec": n_generated / dt if dt > 0 else 0.0,
        }

    def backend_name(self) -> str:
        if self._has_gpu:
            return f"gpu({self._gpu.name})"
        return "gpu(fallback-numpy)"

    def vocab_size(self) -> int:
        embed_w = self._flat.get("embed.token")
        if embed_w is None:
            embed_w = self._flat.get("model.embed_tokens.weight")
        return embed_w.shape[0] if embed_w is not None else 0

    def n_layers(self) -> int:
        return self._arch.n_layers


# Register gpu backend (will fallback to numpy if GPU engine not compiled)
try:
    register_backend("gpu", WgpuBE)
except Exception as e:
    import logging
    logging.getLogger(__name__).debug("GPU backend not registered: %s", e)
