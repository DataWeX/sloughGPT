"""
VectorBE — CPU compute backend with real multiprocessing.

Shared memory architecture: weights live in shared memory once,
workers read directly — zero pickle overhead per dispatch.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import multiprocessing.shared_memory as shm
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from domains.infrastructure.arch_config import ArchConfig
from domains.infrastructure.compute_backend import ComputeBackend, register_backend

logger = logging.getLogger(__name__)

_N_PROC = min(8, mp.cpu_count() or 4)


# ── Worker functions (module-level for pickling) ─────────────────────────

def _matmul_worker(args):
    """Matmul on a chunk of rows: a[start:end] @ b, full output written to shared memory."""
    shm_name_a, shape_a, dtype_a, start, end, shm_name_b, shape_b, dtype_b, out_shm, out_shape = args

    block_a = shm.SharedMemory(name=shm_name_a)
    a = np.ndarray(shape_a, dtype=dtype_a, buffer=block_a.buf)

    block_b = shm.SharedMemory(name=shm_name_b)
    b = np.ndarray(shape_b, dtype=dtype_b, buffer=block_b.buf)

    block_out = shm.SharedMemory(name=out_shm)
    out = np.ndarray(out_shape, dtype=np.float32, buffer=block_out.buf)

    out[start:end] = a[start:end] @ b

    block_a.close()
    block_b.close()
    block_out.close()
    return True


def _softmax_worker(args):
    """Softmax on a chunk of rows, full output written to shared memory."""
    shm_name_x, shape_x, dtype_x, start, end, out_shm, out_shape = args

    block_x = shm.SharedMemory(name=shm_name_x)
    x = np.ndarray(shape_x, dtype=dtype_x, buffer=block_x.buf)

    block_out = shm.SharedMemory(name=out_shm)
    out = np.ndarray(out_shape, dtype=np.float32, buffer=block_out.buf)

    for i in range(start, end):
        row = x[i]
        mx = row.max()
        e = np.exp(row - mx)
        out[i] = e / e.sum()

    block_x.close()
    block_out.close()
    return True


def _rmsnorm_worker(args):
    """RMSNorm on a chunk of rows, full output written to shared memory."""
    shm_name_x, shape_x, dtype_x, shm_name_w, shape_w, eps, start, end, out_shm, out_shape = args

    block_x = shm.SharedMemory(name=shm_name_x)
    x = np.ndarray(shape_x, dtype=dtype_x, buffer=block_x.buf)

    block_w = shm.SharedMemory(name=shm_name_w)
    w = np.ndarray(shape_w, dtype=np.float32, buffer=block_w.buf)

    block_out = shm.SharedMemory(name=out_shm)
    out = np.ndarray(out_shape, dtype=np.float32, buffer=block_out.buf)

    cols = shape_x[1]
    for i in range(start, end):
        row = x[i]
        ss = float(np.dot(row, row))
        inv = 1.0 / np.sqrt(ss / cols + eps)
        out[i] = row * inv * w

    block_x.close()
    block_w.close()
    block_out.close()
    return True


class VectorBE(ComputeBackend):
    """CPU compute backend with real multiprocessing via shared memory.

    Weights live in shared memory once. Workers read directly.
    Zero pickle overhead per dispatch.
    """

    def __init__(self, weights: Dict[str, np.ndarray], arch: ArchConfig):
        self._arch = arch
        self._n_proc = _N_PROC
        self._pool = mp.Pool(_N_PROC)

        # Put all weights into shared memory
        self._shm_blocks: Dict[str, shm.SharedMemory] = {}
        self._weight_names: List[str] = []
        self._weight_shapes: List[Tuple] = []
        self._weight_dtypes: List[np.dtype] = []

        for name, arr in weights.items():
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            block = shm.SharedMemory(create=True, size=arr.nbytes)
            shared_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=block.buf)
            shared_arr[:] = arr[:]
            self._shm_blocks[name] = block
            self._weight_names.append(name)
            self._weight_shapes.append(arr.shape)
            self._weight_dtypes.append(arr.dtype)

        # Build lookup: name -> (shm_name, shape, dtype)
        self._w_info: Dict[str, Tuple[str, Tuple, np.dtype]] = {}
        for name, block in self._shm_blocks.items():
            idx = self._weight_names.index(name)
            self._w_info[name] = (block.name, self._weight_shapes[idx], self._weight_dtypes[idx])

    @classmethod
    def from_weights(cls, weights: Dict[str, np.ndarray], arch: ArchConfig) -> VectorBE:
        return cls(weights, arch)

    def warmup(self, seq_len: int = 1) -> None:
        self.forward(np.zeros((1, seq_len), dtype=np.int64))

    def __del__(self):
        if hasattr(self, '_pool') and self._pool:
            self._pool.close()
        for block in getattr(self, '_shm_blocks', {}).values():
            try:
                block.close()
                block.unlink()
            except Exception as e:
                logger.debug("Failed to close shared memory block: %s", e, extra={"tag": "INFRA"})

    def _get_w(self, name: str) -> np.ndarray:
        """Read a weight tensor from shared memory."""
        shm_name, shape, dtype = self._w_info[name]
        block = shm.SharedMemory(name=shm_name)
        arr = np.ndarray(shape, dtype=dtype, buffer=block.buf).copy()
        block.close()
        return arr

    def _make_out_shm(self, shape, dtype=np.float32) -> Tuple[shm.SharedMemory, np.ndarray]:
        arr = np.empty(shape, dtype=dtype)
        block = shm.SharedMemory(create=True, size=arr.nbytes)
        shared = np.ndarray(shape, dtype=dtype, buffer=block.buf)
        return block, shared

    # ── Tensor primitives ───────────────────────────────────────────────

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if a.ndim == 2 and b.ndim == 2:
            m = a.shape[0]
            n = b.shape[1]

            # Put a and b in shared memory
            a_shm = shm.SharedMemory(create=True, size=a.nbytes)
            np.copyto(np.ndarray(a.shape, dtype=a.dtype, buffer=a_shm.buf), a)

            b_shm = shm.SharedMemory(create=True, size=b.nbytes)
            np.copyto(np.ndarray(b.shape, dtype=b.dtype, buffer=b_shm.buf), b)

            # Output shared memory — full shape for all workers
            out_shm, out_arr = self._make_out_shm((m, n))

            # Dispatch chunks — each worker writes to out[start:end]
            chunk = max(1, m // self._n_proc)
            tasks = []
            for start in range(0, m, chunk):
                end = min(start + chunk, m)
                tasks.append((a_shm.name, a.shape, a.dtype,
                              start, end,
                              b_shm.name, b.shape, b.dtype,
                              out_shm.name, (m, n)))

            self._pool.map(_matmul_worker, tasks)

            result = out_arr.copy()

            a_shm.close(); a_shm.unlink()
            b_shm.close(); b_shm.unlink()
            out_shm.close(); out_shm.unlink()

            return result
        return a @ b

    def softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        if x.ndim == 2 and axis == -1:
            rows, cols = x.shape

            x_shm = shm.SharedMemory(create=True, size=x.nbytes)
            np.copyto(np.ndarray(x.shape, dtype=x.dtype, buffer=x_shm.buf), x)

            out_shm, out_arr = self._make_out_shm((rows, cols))

            chunk = max(1, rows // self._n_proc)
            tasks = []
            for start in range(0, rows, chunk):
                end = min(start + chunk, rows)
                tasks.append((x_shm.name, x.shape, x.dtype,
                              start, end,
                              out_shm.name, (rows, cols)))

            self._pool.map(_softmax_worker, tasks)

            result = out_arr.copy()
            x_shm.close(); x_shm.unlink()
            out_shm.close(); out_shm.unlink()
            return result

        mx = x.max(axis=axis, keepdims=True)
        e = np.exp(x - mx)
        return e / e.sum(axis=axis, keepdims=True)

    def rmsnorm(self, x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        if x.ndim == 2:
            rows, cols = x.shape

            x_shm = shm.SharedMemory(create=True, size=x.nbytes)
            np.copyto(np.ndarray(x.shape, dtype=x.dtype, buffer=x_shm.buf), x)

            w_shm = shm.SharedMemory(create=True, size=weight.nbytes)
            np.copyto(np.ndarray(weight.shape, dtype=weight.dtype, buffer=w_shm.buf), weight)

            out_shm, out_arr = self._make_out_shm((rows, cols))

            chunk = max(1, rows // self._n_proc)
            tasks = []
            for start in range(0, rows, chunk):
                end = min(start + chunk, rows)
                tasks.append((x_shm.name, x.shape, x.dtype,
                              w_shm.name, weight.shape, eps,
                              start, end,
                              out_shm.name, (rows, cols)))

            self._pool.map(_rmsnorm_worker, tasks)

            result = out_arr.copy()
            x_shm.close(); x_shm.unlink()
            w_shm.close(); w_shm.unlink()
            out_shm.close(); out_shm.unlink()
            return result

        ss = np.sum(x * x, axis=-1, keepdims=True)
        return x * (1.0 / np.sqrt(ss / x.shape[-1] + eps)) * weight

    def silu(self, x: np.ndarray) -> np.ndarray:
        return x * (1.0 / (1.0 + np.exp(-x)))

    def gelu(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))

    def rope(self, x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        c = np.broadcast_to(cos, x1.shape)
        s = np.broadcast_to(sin, x1.shape)
        return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)

    def repeat_kv(self, x: np.ndarray, n_reps: int) -> np.ndarray:
        if n_reps == 1:
            return x
        b, h, seq, d = x.shape
        return np.broadcast_to(x[:, :, None, :, :], (b, h, n_reps, seq, d)).reshape(b, h * n_reps, seq, d)

    def argmax(self, x: np.ndarray) -> int:
        return int(np.argmax(x))

    def clip(self, x: np.ndarray, lo: Any, hi: Any) -> np.ndarray:
        return np.clip(x, lo, hi)

    def from_numpy(self, arr: np.ndarray) -> np.ndarray:
        return arr

    def to_numpy(self, tensor: np.ndarray) -> np.ndarray:
        return tensor

    # ── Forward pass ────────────────────────────────────────────────────

    def forward(self, token_ids: np.ndarray, **kwargs) -> np.ndarray:
        arch = self._arch
        n_heads = arch.n_head
        n_kv_heads = arch.n_kv_head
        head_dim = arch.head_dim

        # Read weights from shared memory
        embed_w = self._get_w("embed_tokens.weight")
        x = embed_w[token_ids]
        rope_cos, rope_sin = self._build_rope_cache(x.shape[1], head_dim)

        for i in range(arch.n_layers):
            x = self._forward_layer(x, i, rope_cos, rope_sin, n_heads, n_kv_heads, head_dim)

        norm_w = self._get_w("model.norm.weight")
        x = self.rmsnorm(x, norm_w)
        lm_w = self._get_w("lm_head.weight")
        return x @ lm_w.T

    def _forward_layer(self, x, idx, rope_cos, rope_sin, n_heads, n_kv_heads, head_dim):
        p = f"model.layers.{idx}"
        residual = x

        ln1_w = self._get_w(f"{p}.input_layernorm.weight")
        x = self.rmsnorm(x, ln1_w)

        q_w = self._get_w(f"{p}.self_attn.q_proj.weight")
        k_w = self._get_w(f"{p}.self_attn.k_proj.weight")
        v_w = self._get_w(f"{p}.self_attn.v_proj.weight")

        q = x @ q_w.T
        k = x @ k_w.T
        v = x @ v_w.T

        b, seq, _ = x.shape
        q = q.reshape(b, seq, n_heads, head_dim).transpose((0, 2, 1, 3))
        k = k.reshape(b, seq, n_kv_heads, head_dim).transpose((0, 2, 1, 3))
        v = v.reshape(b, seq, n_kv_heads, head_dim).transpose((0, 2, 1, 3))

        # RoPE: cos/sin are (seq_len, head_dim//2), broadcast to (1, 1, seq, head_dim//2)
        cos = rope_cos[np.newaxis, np.newaxis, :, :]
        sin = rope_sin[np.newaxis, np.newaxis, :, :]
        q = self.rope(q, cos, sin)
        k = self.rope(k, cos, sin)
        k = self.repeat_kv(k, n_heads // n_kv_heads)
        v = self.repeat_kv(v, n_heads // n_kv_heads)

        scale = head_dim ** -0.5
        scores = (q @ np.swapaxes(k, -2, -1)) * scale
        scores = self.softmax(scores, axis=-1)
        attn = scores @ v

        o_w = self._get_w(f"{p}.self_attn.o_proj.weight")
        out = attn.reshape(b, seq, n_heads * head_dim) @ o_w.T
        x = residual + out

        residual = x
        ln2_w = self._get_w(f"{p}.post_attention_layernorm.weight")
        x = self.rmsnorm(x, ln2_w)

        gate_w = self._get_w(f"{p}.mlp.gate_proj.weight")
        up_w = self._get_w(f"{p}.mlp.up_proj.weight")
        down_w = self._get_w(f"{p}.mlp.down_proj.weight")

        gate = x @ gate_w.T
        up = x @ up_w.T
        x = self.silu(gate) * up
        x = x @ down_w.T
        return residual + x

    def _build_rope_cache(self, seq_len, head_dim):
        t = np.arange(seq_len, dtype=np.float32)
        freqs = 1.0 / (10000.0 ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
        angles = np.outer(t, freqs)
        return np.cos(angles), np.sin(angles)

    # ── Generation ──────────────────────────────────────────────────────

    def generate_stream(self, token_ids, max_new_tokens=100, temperature=1.0,
                        top_k=None, top_p=None, repetition_penalty=1.0,
                        eos_token=None, extra_stop_ids=None):
        generated = []
        for _ in range(max_new_tokens):
            logits = self.forward(token_ids)[:, -1, :]
            if repetition_penalty != 1.0 and generated:
                logits[:, np.array(generated, dtype=np.int64)] /= repetition_penalty
            if temperature <= 0.0:
                next_token = int(np.argmax(logits[0]))
            else:
                logits = logits[0] / temperature
                if top_k is not None:
                    idx = np.argpartition(logits, -top_k)[-top_k:]
                    full = np.full_like(logits, -np.inf)
                    full[idx] = logits[idx]
                    logits = full
                probs = np.exp(logits - logits.max())
                probs /= probs.sum()
                next_token = int(np.random.choice(len(probs), p=probs))
            if next_token == eos_token or (extra_stop_ids and next_token in extra_stop_ids):
                break
            generated.append(next_token)
            token_ids = np.concatenate([token_ids, [[next_token]]], axis=1)
            yield next_token

    def generate(self, token_ids, max_new_tokens=100, temperature=1.0,
                 top_k=None, top_p=None, repetition_penalty=1.0,
                 eos_token=None, extra_stop_ids=None):
        t0 = time.perf_counter()
        gen = list(self.generate_stream(token_ids, max_new_tokens, temperature,
                                         top_k, top_p, repetition_penalty,
                                         eos_token, extra_stop_ids))
        ttft = (time.perf_counter() - t0) * 1000
        all_ids = np.concatenate([token_ids, np.array([gen], dtype=np.int64)], axis=1)
        return all_ids, {
            "n_tokens": len(gen),
            "ttft_ms": ttft,
            "tokens_per_sec": len(gen) / (ttft / 1000) if ttft > 0 else 0,
        }

    def backend_name(self) -> str:
        return f"vector({_N_PROC}proc,shm)"

    def vocab_size(self) -> int:
        return self._arch.vocab_size

    def n_layers(self) -> int:
        return self._arch.n_layers


register_backend("vector", VectorBE)
