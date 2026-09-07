"""Performance Optimization Module for SloughGPT

Device detection and environment setup for training/inference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


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


@dataclass
class TrainingOptimizations:
    """Training-specific performance knobs."""
    use_compile: bool = True
    compile_mode: str = "reduce-overhead"
    compile_fullgraph: bool = False
    use_cuda_graphs: bool = False
    channel_last: bool = True
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True
    dataloader_workers: int = 4
    dataloader_prefetch: int = 2
    dataloader_persistent: bool = True
    dataloader_pin_memory: bool = True
    use_bf16: bool = False
    use_fused_optimizer: bool = True
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False
    batch_preallocation: bool = True
    gradient_accumulation_steps: int = 1

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class InferenceOptimizations:
    """Inference-specific performance knobs."""
    use_compile: bool = True
    compile_mode: str = "default"
    channel_last: bool = True
    use_cuda_graphs: bool = True
    use_flash_attention: bool = True
    use_sdpa: bool = True
    use_kv_cache: bool = True
    kv_cache_preallocate: bool = True
    max_batch_size: int = 32
    use_dynamic_batching: bool = True
    use_continuous_batching: bool = True

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class PerformanceConfig:
    """Top-level performance configuration."""
    device: str = "auto"
    training: TrainingOptimizations = field(default_factory=TrainingOptimizations)
    inference: InferenceOptimizations = field(default_factory=InferenceOptimizations)

    def __post_init__(self):
        if self.device == "auto":
            self.device = get_optimal_device()

    def to_dict(self) -> dict:
        return {
            "device": self.device,
            "training": self.training.to_dict(),
            "inference": self.inference.to_dict(),
        }


class CUDAGraphManager:
    """Manages CUDA graphs for repeated inference patterns."""

    def __init__(self, model: object, config: InferenceOptimizations | None = None):
        self._model = model
        self._config = config or InferenceOptimizations()
        self._captured = False

    def capture(self, batch_size: int = 1, seq_len: int = 128, hidden: int = 256) -> bool:
        """Capture a CUDA graph for the given input shape. Returns False (no GPU)."""
        return False

    def replay(self, input_ids):
        """Replay the captured graph — delegates to model."""
        if callable(self._model):
            return self._model(input_ids)
        return input_ids

    @property
    def is_captured(self) -> bool:
        return self._captured


class FastInferenceSampler:
    """Fast sampling utilities for inference (numpy-only, no torch)."""

    @staticmethod
    def sample(
        logits,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        prev_tokens=None,
        recent_tokens=None,
    ):
        """Sample from logits with temperature, top-k, top-p, and repetition penalty.

        Returns ndarray of shape matching input logits.
        """
        import numpy as np

        logits = np.asarray(logits, dtype=np.float64)
        single_dim = logits.ndim == 1
        if single_dim:
            logits = logits.reshape(1, -1)

        if temperature == 0:
            result = np.argmax(logits, axis=-1, keepdims=True)
            return result

        scaled = logits / max(temperature, 1e-8)

        tokens = prev_tokens if prev_tokens is not None else recent_tokens
        if repetition_penalty != 1.0 and tokens is not None:
            tokens = np.asarray(tokens)
            scaled = FastInferenceSampler._apply_repetition_penalty_vectorized(
                scaled, tokens, repetition_penalty
            )

        if top_k > 0:
            scaled = FastInferenceSampler._apply_top_k(scaled, top_k)

        if top_p < 1.0:
            scaled = FastInferenceSampler._apply_top_p(scaled, top_p)

        probs = np.exp(scaled - scaled.max(axis=-1, keepdims=True))
        probs = probs / probs.sum(axis=-1, keepdims=True)
        result = np.array([
            np.random.choice(p.shape[-1], p=p) for p in probs
        ]).reshape(-1, 1)
        return result

    @staticmethod
    def _apply_top_k(logits, k: int):
        """Mask all but top-k logits to -inf."""
        import numpy as np
        logits = np.asarray(logits, dtype=np.float64)
        if k <= 0 or k >= logits.shape[-1]:
            return logits
        if logits.ndim == 1:
            threshold = np.sort(logits)[-k]
            result = logits.copy()
            result[logits < threshold] = -np.inf
            return result
        result = logits.copy()
        for i in range(result.shape[0]):
            row = result[i]
            threshold = np.sort(row)[-k]
            result[i][row < threshold] = -np.inf
        return result

    @staticmethod
    def _apply_top_p(logits, p: float):
        """Nucleus sampling: mask tokens outside cumulative probability p."""
        import numpy as np
        logits = np.asarray(logits, dtype=np.float64)
        if logits.ndim == 1:
            sorted_indices = np.argsort(-logits)
            sorted_logits = logits[sorted_indices]
            cum_probs = np.cumsum(np.exp(sorted_logits - sorted_logits.max()))
            cum_probs = cum_probs / cum_probs[-1]
            mask = cum_probs <= p
            mask[0] = True
            result = np.full_like(logits, -np.inf)
            result[sorted_indices[mask]] = sorted_logits[mask]
            return result
        result = logits.copy()
        for i in range(result.shape[0]):
            row = result[i]
            sorted_indices = np.argsort(-row)
            sorted_logits = row[sorted_indices]
            cum_probs = np.cumsum(np.exp(sorted_logits - sorted_logits.max()))
            cum_probs = cum_probs / cum_probs[-1]
            mask = cum_probs <= p
            mask[0] = True
            result[i] = np.full_like(row, -np.inf)
            result[i, sorted_indices[mask]] = sorted_logits[mask]
        return result

    @staticmethod
    def _apply_repetition_penalty_vectorized(logits, recent_tokens, penalty: float):
        """Penalize recently generated tokens."""
        import numpy as np
        recent_tokens = np.asarray(recent_tokens)
        if len(recent_tokens) == 0:
            return logits
        result = logits.copy()
        for t in recent_tokens:
            t = int(t)
            if result.ndim == 1:
                if 0 <= t < len(result):
                    result[t] = result[t] / penalty if result[t] > 0 else result[t] * penalty
            else:
                for i in range(result.shape[0]):
                    if 0 <= t < result.shape[1]:
                        val = result[i, t]
                        result[i, t] = val / penalty if val > 0 else val * penalty
        return result


class PerformanceMonitor:
    """Tracks latency, throughput, and resource usage during training/inference."""

    def __init__(self, window_size: int = 100):
        self._records = []
        self._window_size = window_size
        self._losses = []
        self._step_times = []
        self._tokens_processed = 0

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def step_times(self):
        return list(self._step_times[-self._window_size:])

    @property
    def losses(self):
        return list(self._losses[-self._window_size:])

    @property
    def tokens_processed(self) -> int:
        return self._tokens_processed

    def record(self, name: str, duration: float, tokens: int = 0):
        self._records.append({"name": name, "duration": duration, "tokens": tokens})

    def record_step(self, loss: float = 0.0, step_time: float = 0.0, batch_size: int = 1, seq_len: int = 1):
        self._losses.append(loss)
        self._step_times.append(step_time)
        self._tokens_processed += batch_size * seq_len
        if len(self._step_times) > self._window_size:
            self._step_times = self._step_times[-self._window_size:]
            self._losses = self._losses[-self._window_size:]

    def get_stats(self) -> dict:
        if not self._step_times:
            return {}
        avg_step_time = sum(self._step_times) / len(self._step_times)
        avg_loss = sum(self._losses) / len(self._losses) if self._losses else 0.0
        tokens_per_sec = 0.0
        steps_per_sec = 0.0
        if avg_step_time > 0:
            steps_per_sec = 1.0 / avg_step_time
            tokens_per_sec = self._tokens_processed / (avg_step_time * len(self._step_times))
        return {
            "avg_step_time_ms": avg_step_time * 1000,
            "tokens_per_sec": tokens_per_sec,
            "steps_per_sec": steps_per_sec,
            "avg_loss": avg_loss,
            "total_steps": len(self._step_times),
        }

    def summary(self) -> dict:
        if not self._records:
            return {"total_calls": 0}
        total_time = sum(r["duration"] for r in self._records)
        total_tokens = sum(r["tokens"] for r in self._records)
        return {
            "total_calls": len(self._records),
            "total_time_s": total_time,
            "total_tokens": total_tokens,
            "tokens_per_sec": total_tokens / max(total_time, 1e-9),
        }

    def reset(self):
        self._records.clear()
        self._losses.clear()
        self._step_times.clear()


class OptimizedBatchCache:
    """LRU cache for pre-batched training data."""

    def __init__(self, max_size: int = 64):
        self._cache = {}
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def get(self, key):
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, key, value):
        if len(self._cache) >= self._max_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = value

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(total, 1),
        }


def _softmax(x, axis=-1):
    """Numerically stable softmax."""
    import numpy as np
    x_max = x.max(axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / e_x.sum(axis=axis, keepdims=True)


def _collate(batch):
    """Collate a list of (x, y) tuples into batched arrays.

    If arrays have different shapes, pad shorter ones with zeros.
    """
    import numpy as np
    if not batch:
        return np.array([]), np.array([])
    if isinstance(batch[0], tuple):
        xs = [np.asarray(b[0]) for b in batch]
        ys = [np.asarray(b[1]) for b in batch]
        # If all same shape, just stack
        if all(x.shape == xs[0].shape for x in xs) and all(y.shape == ys[0].shape for y in ys):
            return np.stack(xs, axis=0), np.stack(ys, axis=0)
        # Pad to max shape
        max_x = max(x.shape for x in xs)
        max_y = max(y.shape for y in ys)
        padded_x = np.zeros((len(xs),) + max_x, dtype=xs[0].dtype)
        padded_y = np.zeros((len(ys),) + max_y, dtype=ys[0].dtype)
        for i, (x, y) in enumerate(zip(xs, ys)):
            padded_x[i, : x.shape[0]] = x
            padded_y[i, : y.shape[0]] = y
        return padded_x, padded_y
    return np.stack(batch, axis=0)


def _pad_last(a, target_len: int = 0, pad_value=0):
    """Pad 1D array to target length (or its own length = no-op)."""
    import numpy as np
    a = np.asarray(a)
    if target_len <= 0:
        target_len = len(a)
    if len(a) >= target_len:
        return a
    result = np.full(target_len, pad_value, dtype=a.dtype)
    result[: len(a)] = a
    return result


class _NumpyBatchIterator:
    """Iterates over data (list or array) in batches."""

    def __init__(self, data, batch_size: int, shuffle: bool = True):
        import numpy as np
        self._data = data
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._rng = np.random.default_rng()
        self._indices = np.arange(len(data))

    def __iter__(self):
        if self._shuffle:
            self._rng.shuffle(self._indices)
        for start in range(0, len(self._indices), self._batch_size):
            batch_indices = self._indices[start : start + self._batch_size]
            yield [self._data[int(i)] for i in batch_indices]

    def __len__(self):
        import numpy as np
        return int(np.ceil(len(self._data) / self._batch_size))


class OptimizedDataLoader:
    """Simple data loader backed by numpy arrays."""

    def __init__(self, dataset, batch_size: int = 32, shuffle: bool = True):
        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle

    def __iter__(self):
        return _NumpyBatchIterator(self._dataset, self._batch_size, self._shuffle).__iter__()

    def __len__(self):
        return len(_NumpyBatchIterator(self._dataset, self._batch_size, self._shuffle))


class PreallocatedBatchDataset:
    """Language modeling dataset that pre-allocates (x, y) shifted pairs."""

    def __init__(self, data, block_size: int = 10, batch_size: int = 8):
        import numpy as np
        self._data = np.asarray(data)
        self._block_size = block_size
        self._batch_size = batch_size
        # Number of valid (x, y) pairs: data must have room for block_size + 1
        n = len(data) - block_size - 1
        self._n_samples = max(0, n)

    def __len__(self):
        return self._n_samples

    def __getitem__(self, idx):
        import numpy as np
        if idx < 0 or idx >= self._n_samples:
            raise IndexError(f"index {idx} out of range for {self._n_samples} samples")
        x = self._data[idx : idx + self._block_size]
        y = self._data[idx + 1 : idx + 1 + self._block_size]
        return x, y

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class OptimizedBatchCache:
    """Reusable cache for pre-allocated (x, y) batch tensors."""

    def __init__(self, device: str = "cpu"):
        self._device = device
        self._cache = {}

    def _key(self, batch_size: int, block_size: int):
        return (batch_size, block_size)

    def allocate(self, batch_size: int, block_size: int):
        """Allocate or reuse cached (x, y) arrays."""
        import numpy as np
        key = self._key(batch_size, block_size)
        if key in self._cache:
            return self._cache[key]
        x = np.zeros((batch_size, block_size), dtype=np.float64)
        y = np.zeros((batch_size, block_size), dtype=np.float64)
        self._cache[key] = (x, y)
        return x, y

    def fill(self, batch_size: int, block_size: int, data, indices):
        """Fill and return (x, y) from data at given indices."""
        import numpy as np
        x, y = self.allocate(batch_size, block_size)
        data = np.asarray(data)
        indices = np.asarray(indices)
        for i, idx in enumerate(indices):
            x[i] = data[idx : idx + block_size]
            y[i] = data[idx + 1 : idx + 1 + block_size]
        return x, y

    def clear(self):
        self._cache.clear()


class OptimizedInferenceEngine:
    """Inference engine with batched generation and KV cache support."""

    def __init__(self, model=None, config: InferenceOptimizations | None = None):
        self._model = model
        self._config = config or InferenceOptimizations()
        self._monitor = PerformanceMonitor()
        self._cuda_graph_manager = CUDAGraphManager(model, config) if config and config.use_cuda_graphs else None

    @property
    def cuda_graph_manager(self):
        return self._cuda_graph_manager

    def generate(self, prompt_ids, max_new_tokens: int = 100, temperature: float = 1.0, **kwargs):
        """Generate tokens from prompt."""
        import numpy as np
        prompt_ids = np.asarray(prompt_ids)
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.reshape(1, -1)
        batch_size, seq_len = prompt_ids.shape
        # Stub: return prompt with a few extra tokens appended
        extra = np.zeros((batch_size, max_new_tokens), dtype=prompt_ids.dtype)
        return np.concatenate([prompt_ids, extra], axis=1)

    @property
    def monitor(self) -> PerformanceMonitor:
        return self._monitor


def _as_array(x):
    """Convert input to numpy array if needed."""
    import numpy as np
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _clip_grad_norm_(parameters, max_norm: float = 1.0):
    """Clip gradient norms (no-op for numpy — stub for API compat)."""
    return 0.0


def effective_dataloader_workers(max_workers: int = 4) -> int:
    """Return safe number of dataloader workers (caps at cpu_count, 0 for invalid)."""
    import os
    try:
        n = int(max_workers)
    except (TypeError, ValueError):
        return 0
    if n <= 0:
        return 0
    cpu_count = os.cpu_count() or 2
    return min(n, cpu_count)


def effective_prefetch_factor(workers: int = 2, default: int = 2) -> int | None:
    """Return a safe prefetch factor given worker count.

    Returns None if workers <= 0, otherwise returns max(1, default).
    """
    try:
        w = int(workers)
    except (TypeError, ValueError):
        return None
    if w <= 0:
        return None
    try:
        d = int(default)
    except (TypeError, ValueError):
        d = 2
    return max(1, d)


def benchmark_inference(model, input_ids, n_tokens: int = 50, **kwargs) -> dict:
    """Benchmark inference throughput — not implemented (returns zeros).

    TODO: implement real wall-clock timing around model forward passes.
    """
    logger.warning("benchmark_inference called but not implemented — returning zeros")
    return {
        "n_tokens": n_tokens,
        "tokens_per_sec": 0.0,
        "latency_ms": 0.0,
    }


def benchmark_training(model, dataset, n_steps: int = 10, **kwargs) -> dict:
    """Benchmark training throughput — not implemented (returns zeros).

    TODO: implement real wall-clock timing around training steps.
    """
    logger.warning("benchmark_training called but not implemented — returning zeros")
    return {
        "n_steps": n_steps,
        "steps_per_sec": 0.0,
        "loss": 0.0,
    }


def optimize_model_for_inference(model, config: InferenceOptimizations | None = None):
    """Optimize model for inference (no-op stub — actual optimization is device-specific)."""
    return model
