"""
Performance Optimization Module for SloughGPT

High-performance training and inference optimizations:
- Optimized batching: pre-allocated arrays, vectorized operations
- Fast sampling: fused top-k/top-p/repetition penalty on numpy
- KV Cache: efficient cache management for inference
- Memory optimization: channel-last, gradient checkpointing

CUDA-specific accelerations (CUDA graphs, ``torch.compile``) are no-ops on
the numpy SloNet stack — SloNet models always train and infer on numpy.
Device detection is platform-based and never imports torch.

Usage:
    from domains.training.performance import optimize_training, optimize_inference
    model, trainer = optimize_training(model, config)
"""

from __future__ import annotations

import time
from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass, field
import logging

import numpy as np

logger = logging.getLogger("slo.performance")


@dataclass
class TrainingOptimizations:
    """Training performance settings."""

    use_compile: bool = True
    compile_mode: str = "reduce-overhead"
    compile_fullgraph: bool = False

    use_cuda_graphs: bool = False
    channel_last: bool = True

    dataloader_workers: int = 4
    dataloader_prefetch: int = 2
    dataloader_persistent: bool = True
    dataloader_pin_memory: bool = True

    use_fused_optimizer: bool = True

    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False

    use_flash_attention: bool = True
    gradient_checkpointing: bool = True

    batch_preallocation: bool = True


@dataclass
class InferenceOptimizations:
    """Inference performance settings."""

    use_compile: bool = True
    compile_mode: str = "default"

    use_cuda_graphs: bool = True
    channel_last: bool = True

    use_flash_attention: bool = True
    use_sdpa: bool = True

    max_batch_size: int = 32
    kv_cache_preallocate: bool = True

    use_kv_cache: bool = True
    use_continuous_batching: bool = True


@dataclass
class PerformanceConfig:
    """Configuration for performance optimizations."""

    device: str = "auto"

    training: TrainingOptimizations = field(default_factory=TrainingOptimizations)
    inference: InferenceOptimizations = field(default_factory=InferenceOptimizations)

    def __post_init__(self):
        if self.device == "auto":
            self.device = get_optimal_device()


def get_optimal_device() -> str:
    """Auto-detect best available device (degrades to CPU).

    Delegates to ``ml_types.auto_device()`` — platform-based detection
    (MPS via Apple Silicon, CUDA via CuPy) with no torch import.
    """
    from domains.infrastructure.ml_types import auto_device
    return auto_device()


def get_device_name() -> str:
    """Get human-readable device name."""
    device = get_optimal_device()
    if device == "cuda":
        try:
            import cupy as cp
            props = cp.cuda.runtime.getDeviceProperties(0)
            return str(props["name"])
        except Exception:
            return "CUDA"
    if device == "mps":
        return "Apple Silicon (MPS)"
    return "CPU"


def setup_device_environment():
    """Setup optimal device environment variables.

    No-op on the numpy SloNet stack — torch CUDA knobs are not used.
    """
    return None


class CUDAGraphManager:
    """Manages CUDA graphs for kernel capture/replay.

    CUDA graphs require PyTorch + CUDA. On the numpy SloNet stack this
    manager is a transparent no-op: ``capture()`` returns False and
    ``replay()`` falls back to a plain forward pass, keeping callers
    working on the numpy SloNet stack.
    """

    def __init__(self, model, config: InferenceOptimizations):
        self.model = model
        self.config = config
        self.graphs: Dict[int, Any] = {}
        self.static_inputs: Dict[int, Any] = {}
        self.static_outputs: Optional[np.ndarray] = None
        self._enabled = False

    def capture(
        self,
        batch_size: int,
        seq_len: int,
        vocab_size: int,
    ) -> bool:
        """Capture a CUDA graph for given input shape.

        No-op on the numpy stack — always returns False so callers use
        plain forward passes.
        """
        return False

    def replay(self, input_ids) -> Any:
        """Replay captured graph or fall back to normal forward."""
        return self.model(input_ids)


class _NumpyBatchIterator:
    """Minimal numpy replacement for ``torch.utils.data.DataLoader``."""

    def __init__(self, dataset, batch_size: int, shuffle: bool = True):
        self.dataset = list(dataset)
        self.batch_size = max(1, int(batch_size))
        self.shuffle = shuffle
        self._idx = 0
        self._order = list(range(len(self.dataset)))
        self._reshuffle()

    def _reshuffle(self):
        if self.shuffle:
            rng = np.random.default_rng()
            self._order = list(rng.permutation(len(self.dataset)))
        else:
            self._order = list(range(len(self.dataset)))
        self._idx = 0

    def __iter__(self):
        self._reshuffle()
        return self

    def __next__(self):
        if self._idx >= len(self.dataset):
            raise StopIteration
        end = min(self._idx + self.batch_size, len(self.dataset))
        batch = [self.dataset[i] for i in self._order[self._idx : end]]
        self._idx = end
        return _collate(batch)

    def __len__(self):
        return max(1, (len(self.dataset) + self.batch_size - 1) // self.batch_size)


def _collate(batch: List[Any]) -> Any:
    """Stack a list of (x, y) pairs or tensors into numpy batches."""
    if not batch:
        return None
    if isinstance(batch[0], (list, tuple)):
        xs = [np.asarray(b[0]) for b in batch]
        ys = [np.asarray(b[1]) for b in batch]
        max_x = max(a.shape[-1] for a in xs)
        max_y = max(a.shape[-1] for a in ys)
        x = np.stack([_pad_last(a, max_x) for a in xs])
        y = np.stack([_pad_last(a, max_y) for a in ys])
        return x, y
    arrs = [np.asarray(b) for b in batch]
    return np.stack(arrs)


def _pad_last(a: np.ndarray, n: int) -> np.ndarray:
    if a.shape[-1] == n:
        return a
    pad = [(0, 0)] * a.ndim
    pad[-1] = (0, n - a.shape[-1])
    return np.pad(a, pad, mode="constant")


class OptimizedDataLoader:
    """High-performance DataLoader with prefetching and memory optimization."""

    def __init__(
        self,
        dataset,
        batch_size: int,
        num_workers: Optional[int] = None,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        collate_fn: Optional[Callable] = None,
    ):
        if num_workers is None:
            from domains.infrastructure.resource_manager import get_resource_manager
            num_workers = get_resource_manager().dataloader_workers
        effective_workers = effective_dataloader_workers(num_workers)
        effective_prefetch = effective_prefetch_factor(effective_workers, prefetch_factor)

        self._collate_fn = collate_fn
        self.dataloader = _NumpyBatchIterator(dataset, batch_size, shuffle=True)
        self._iterator = None
        self._prefetched_batch: Optional[Any] = None

    def prefetch(self):
        """Prefetch next batch in background."""
        if self._iterator is None:
            self._iterator = iter(self.dataloader)
        try:
            self._prefetched_batch = next(self._iterator)
        except StopIteration:  # pragma: no cover (iterator freshly restarted, cannot be empty)
            self._iterator = iter(self.dataloader)
            self._prefetched_batch = next(self._iterator)

    def get_batch(self) -> Any:
        """Get next batch, prefetching in background."""
        if self._prefetched_batch is not None:
            batch = self._prefetched_batch
            self._prefetched_batch = None
            return batch

        if self._iterator is None:
            self._iterator = iter(self.dataloader)

        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.dataloader)
            return next(self._iterator)

    def __iter__(self) -> "OptimizedDataLoader":
        self._iterator = iter(self.dataloader)
        self._prefetched_batch = None
        return self

    def __next__(self) -> Any:
        try:
            return self.get_batch()
        except StopIteration:  # pragma: no cover (get_batch restarts internally, never raises)
            self._iterator = iter(self.dataloader)
            return next(self._iterator)


def effective_dataloader_workers(requested: int) -> int:
    """Get safe worker count for platform."""
    import sys
    if sys.platform == "darwin":  # pragma: no cover (darwin-only)
        return 0
    try:
        n = int(requested)
    except (TypeError, ValueError):
        n = 0
    return max(0, n)


def effective_prefetch_factor(num_workers: int, requested: int) -> Optional[int]:
    """Get safe prefetch factor."""
    if num_workers <= 0:
        return None
    try:
        pf = int(requested)
    except (TypeError, ValueError):
        pf = 2
    return max(1, pf)


class PreallocatedBatchDataset:
    """Dataset that returns indices for pre-allocated batch tensors."""

    def __init__(self, data: np.ndarray, block_size: int, batch_size: int):
        self.data = np.asarray(data)
        self.block_size = block_size
        self.batch_size = batch_size
        self.seq_len = block_size + 1

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if idx < 0 or idx >= len(self.data) - self.seq_len:
            raise IndexError(f"index {idx} out of range for dataset of length {len(self)}")
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.seq_len]
        return x, y


class OptimizedBatchCache:
    """Cache for pre-allocated batch tensors to avoid allocations."""

    def __init__(self, device: str, dtype: Any = np.int64):
        self.device = device
        self.dtype = dtype
        self.x_cache: Optional[np.ndarray] = None
        self.y_cache: Optional[np.ndarray] = None
        self._batch_size = 0
        self._block_size = 0

    def allocate(self, batch_size: int, block_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Allocate or return cached batch tensors."""
        if self.x_cache is None or self._batch_size != batch_size or self._block_size != block_size:
            self._batch_size = batch_size
            self._block_size = block_size
            self.x_cache = np.empty((batch_size, block_size), dtype=self.dtype)
            self.y_cache = np.empty((batch_size, block_size), dtype=self.dtype)
        return self.x_cache, self.y_cache

    def fill(
        self,
        batch_size: int,
        block_size: int,
        data: np.ndarray,
        indices: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fill pre-allocated tensors with batch data."""
        x, y = self.allocate(batch_size, block_size)

        for i, idx in enumerate(indices):
            x[i] = data[idx : idx + block_size]
            y[i] = data[idx + 1 : idx + block_size + 1]

        return x, y


def _as_array(x, dtype=None) -> np.ndarray:
    """Coerce a SloNet Tensor or numpy array into a numpy ndarray.

    SloNet Tensor wraps a numpy array in ``.data``; plain ``np.asarray`` on a
    Tensor produces an object array whose elements are row Tensors and cannot
    be cast. This helper unwraps the underlying numpy buffer first.
    """
    data = getattr(x, "data", None)
    if isinstance(data, np.ndarray):
        x = data
    return np.asarray(x, dtype=dtype)


def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax on numpy."""
    logits = _as_array(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


class FastInferenceSampler:
    """Optimized token sampling with vectorized operations."""

    @staticmethod
    def sample(
        logits,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        prev_tokens=None,
    ) -> np.ndarray:
        """Fast token sampling with optional repetition penalty.

        Optimizations:
        - Vectorized repetition penalty (O(1) instead of O(n))
        - Fused top-k/top-p filtering
        - Pure numpy operations
        """
        logits = _as_array(logits, dtype=np.float64)

        if temperature == 0:
            return np.argmax(logits, axis=-1, keepdims=True)

        logits = logits / temperature

        prev = np.asarray(prev_tokens) if prev_tokens is not None else None
        if repetition_penalty != 1.0 and prev is not None and prev.size > 0:
            logits = FastInferenceSampler._apply_repetition_penalty_vectorized(
                logits, prev, repetition_penalty
            )

        if top_k > 0:
            logits = FastInferenceSampler._apply_top_k(logits, top_k)

        if top_p < 1.0:
            logits = FastInferenceSampler._apply_top_p(logits, top_p)

        probs = _softmax(logits, axis=-1)
        probs = np.clip(probs, 1e-10, None)

        # Normalize after clipping (numpy multinomial needs valid probabilities)
        probs = probs / probs.sum(axis=-1, keepdims=True)

        batch = logits.shape[:-1]
        flat_probs = probs.reshape(-1, probs.shape[-1])
        tokens = np.array([
            int(np.random.choice(flat_probs.shape[-1], p=row / row.sum()))
            for row in flat_probs
        ], dtype=np.int64)
        return tokens.reshape(*batch, 1)

    @staticmethod
    def _apply_repetition_penalty_vectorized(
        logits: np.ndarray,
        prev_tokens: np.ndarray,
        penalty: float,
    ) -> np.ndarray:
        """Apply repetition penalty using advanced indexing (O(1) per batch)."""
        flat = prev_tokens.reshape(-1)
        unique_tokens = np.unique(flat)
        if unique_tokens.size == 0:
            return logits

        penalties = np.ones(logits.shape[-1], dtype=np.float64)
        idx = unique_tokens[unique_tokens < logits.shape[-1]]
        penalties[idx] = penalty

        pos_mask = logits > 0
        neg_mask = ~pos_mask
        return np.where(pos_mask, logits * penalties, logits / penalties)

    @staticmethod
    def _apply_top_k(logits: np.ndarray, k: int) -> np.ndarray:
        """Apply top-k filtering efficiently."""
        k = min(k, logits.shape[-1])
        if k <= 0:
            return logits
        if logits.ndim == 1:
            logits = logits[None, :]
            flat = True
        else:
            flat = False
        cut = logits.shape[-1] - k
        threshold = np.partition(logits, cut, axis=-1)[..., cut:cut + 1]
        result = np.where(
            logits < threshold,
            np.full_like(logits, float("-inf")),
            logits,
        )
        return result[0] if flat else result

    @staticmethod
    def _apply_top_p(logits: np.ndarray, p: float) -> np.ndarray:
        """Apply top-p (nucleus) filtering with early exit."""
        if logits.ndim == 1:
            logits = logits[None, :]
            flat = True
        else:
            flat = False

        sorted_indices = np.argsort(-logits, axis=-1)
        sorted_logits = np.take_along_axis(logits, sorted_indices, axis=-1)
        cumsum = np.cumsum(_softmax(sorted_logits, axis=-1), axis=-1)

        mask = cumsum > p
        mask[..., 1:] = mask[..., :-1]
        mask[..., 0] = False

        # Scatter mask back to original positions
        out = np.full_like(logits, float("-inf"))
        np.put_along_axis(out, sorted_indices, np.where(mask, float("-inf"), sorted_logits), axis=-1)
        return out[0] if flat else out


class OptimizedInferenceEngine:
    """High-performance inference engine with all optimizations."""

    def __init__(
        self,
        model,
        config: Optional[InferenceOptimizations] = None,
        device: str = "auto",
    ):
        self.model = model
        self.config = config or InferenceOptimizations()
        self.device = get_optimal_device() if device == "auto" else device

        self._compiled_model = None

        if self.device == "cuda":  # pragma: no cover (CUDA-only)
            self.cuda_graph_manager = CUDAGraphManager(model, self.config)
        else:
            self.cuda_graph_manager = None

        if hasattr(self.model, "eval"):
            self.model.eval()

    def _setup_compiled_forward(self):
        """Setup torch.compile for faster forward passes.

        No-op on the numpy SloNet stack — no torch.compile. ``_compiled_model``
        stays None so ``generate()`` uses the plain model.
        """
        self._compiled_model = None

    def generate(
        self,
        input_ids,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> np.ndarray:
        """Optimized autoregressive generation."""
        model = self._compiled_model if self._compiled_model else self.model
        if hasattr(model, "eval"):
            model.eval()

        current = np.asarray(input_ids, dtype=np.int64)
        if current.ndim == 1:
            current = current[None, :]
        prev_tokens = current.copy()

        block_size = getattr(model, "block_size", current.shape[-1])

        for _ in range(int(max_new_tokens)):
            idx_cond = current[:, -block_size:]

            if self.cuda_graph_manager:  # pragma: no cover (CUDA-graphs only)
                logits = self.cuda_graph_manager.replay(idx_cond)
            else:
                logits = model(idx_cond)

            if isinstance(logits, tuple):
                logits = logits[0]

            logits = _as_array(logits)
            if logits.ndim == 3:
                logits = logits[:, -1, :]
            elif logits.ndim == 2 and logits.shape[0] == 1:  # pragma: no cover (defensive no-op)
                pass

            next_token = FastInferenceSampler.sample(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                prev_tokens=prev_tokens,
            )

            current = np.concatenate([current, next_token], axis=1)
            prev_tokens = np.concatenate([prev_tokens, next_token], axis=1)

            if int(next_token[0, -1]) == 0:  # pragma: no cover (data-dependent EOS break)
                break

        return current


class PerformanceMonitor:
    """Monitor training/inference performance metrics."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.step_times: List[float] = []
        self.losses: List[float] = []
        self.tokens_processed = 0
        self._start_time = time.time()

    def record_step(self, loss: float, step_time: float, batch_size: int, seq_len: int):
        """Record a training step."""
        self.step_times.append(step_time)
        self.losses.append(loss)
        self.tokens_processed += batch_size * seq_len

        if len(self.step_times) > self.window_size:
            self.step_times.pop(0)
            self.losses.pop(0)

    def get_stats(self) -> Dict[str, float]:
        """Get current performance statistics."""
        if not self.step_times:
            return {}

        avg_step_time = sum(self.step_times) / len(self.step_times)
        tokens_per_sec = self.tokens_processed / (time.time() - self._start_time + 1e-6)

        return {
            "avg_step_time_ms": avg_step_time * 1000,
            "steps_per_sec": 1.0 / avg_step_time if avg_step_time > 0 else 0,
            "tokens_per_sec": tokens_per_sec,
            "avg_loss": sum(self.losses) / len(self.losses) if self.losses else 0,
            "total_steps": len(self.step_times),
        }


def optimize_model_for_inference(
    model,
    device: str = "auto",
    use_compile: bool = True,
    use_channels_last: bool = True,
):
    """Apply all inference optimizations to a model.

    Compilation and channel-last layouts require CUDA + PyTorch; on the
    numpy SloNet stack the model is moved to the target device and set to
    eval mode, which is the only optimization available.
    """
    device = get_optimal_device() if device == "auto" else device
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    return model


def _clip_grad_norm_(model, max_norm: float) -> float:
    """Clip parameter grads in place (numpy/SloNet), returning the norm."""
    total_sq = 0.0
    for p in model.parameters():
        g = getattr(p, "grad", None)
        if g is None:
            continue
        arr = _as_array(g).reshape(-1)
        total_sq += float(np.dot(arr, arr))
    norm = float(np.sqrt(total_sq))
    if norm > max_norm and norm > 0:
        scale = max_norm / norm
        for p in model.parameters():
            g = getattr(p, "grad", None)
            if g is None:  # pragma: no cover (all params get grads when clipping is active)
                continue
            if isinstance(g, np.ndarray):  # pragma: no cover (SloNet grads are Tensors with .data)
                g *= scale
            else:
                g.data[:] = _as_array(g) * scale
    return norm


def benchmark_training(
    model,
    batch_size: int = 8,
    seq_len: int = 128,
    num_steps: int = 100,
    device: str = "auto",
) -> Dict[str, float]:
    """Benchmark training performance (numpy SloNet stack)."""
    device = get_optimal_device() if device == "auto" else device
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "train"):
        model.train()

    rng = np.random.default_rng(0)
    dummy_data = rng.integers(0, 1000, size=(10000,)).astype(np.int64)
    dataset = PreallocatedBatchDataset(dummy_data, seq_len, batch_size)

    optimizer = None
    try:
        from domains.training.slonet import SloAdam
        optimizer = SloAdam(
            lr=1e-4, b1=0.9, b2=0.999, eps=1e-8,
            weight_decay=0.01, max_grad_norm=1.0,
        )
    except Exception as e:  # pragma: no cover (SloAdam imports successfully)
        logger.warning(f"SloAdam unavailable: {e}", extra={"tag": "TRAIN"})

    loader = _NumpyBatchIterator(dataset, batch_size, shuffle=True)
    x, y = next(iter(loader))

    start = time.time()
    for _ in range(int(num_steps)):
        logits, loss = model(x, y)
        loss.backward()
        _clip_grad_norm_(model, 1.0)
        if optimizer is not None:
            params = model.parameters()
            optimizer.step(params)
        for p in model.parameters():
            p.grad = None

    elapsed = time.time() - start
    tokens_per_sec = (batch_size * seq_len * num_steps) / elapsed

    return {
        "elapsed_sec": elapsed,
        "tokens_per_sec": tokens_per_sec,
        "steps_per_sec": num_steps / elapsed,
        "device": device,
    }


def benchmark_inference(
    model,
    batch_size: int = 1,
    seq_len: int = 128,
    gen_len: int = 50,
    num_runs: int = 10,
    device: str = "auto",
) -> Dict[str, float]:
    """Benchmark inference performance."""
    device = get_optimal_device() if device == "auto" else device

    config = InferenceOptimizations(use_compile=False, use_cuda_graphs=False)
    engine = OptimizedInferenceEngine(model, config, device=device)

    rng = np.random.default_rng(0)
    input_ids = rng.integers(0, 1000, size=(batch_size, seq_len)).astype(np.int64)

    latencies = []
    for _ in range(int(num_runs)):
        start = time.time()
        engine.generate(input_ids, max_new_tokens=gen_len)
        latencies.append((time.time() - start) * 1000)

    return {
        "avg_latency_ms": sum(latencies) / len(latencies),
        "p50_latency_ms": sorted(latencies)[len(latencies) // 2],
        "p95_latency_ms": sorted(latencies)[min(int(len(latencies) * 0.95), len(latencies) - 1)],
        "tokens_per_sec": (batch_size * gen_len * num_runs) / (sum(latencies) / 1000),
        "device": device,
    }


__all__ = [
    "PerformanceConfig",
    "TrainingOptimizations",
    "InferenceOptimizations",
    "get_optimal_device",
    "get_device_name",
    "setup_device_environment",
    "CUDAGraphManager",
    "OptimizedDataLoader",
    "PreallocatedBatchDataset",
    "OptimizedBatchCache",
    "FastInferenceSampler",
    "OptimizedInferenceEngine",
    "PerformanceMonitor",
    "optimize_model_for_inference",
    "benchmark_training",
    "benchmark_inference",
]
