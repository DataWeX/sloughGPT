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
    from domain.infrastructure._internal.ml_types import auto_device

    return auto_device()


def get_device_name() -> str:
    """Get human-readable device name."""
    device = get_optimal_device()
    if device == "cuda":
        try:
            import cupy as cp

            props = cp.cuda.runtime.getDeviceProperties(0)
            raw = props["name"]
            # cupy returns the model name as bytes; str(b'...') would render
            # the repr "b'NVIDIA …'" — decode instead (correctness bug fix).
            return raw.decode() if isinstance(raw, bytes) else str(raw)
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
        # Graph capture needs torch + a real CUDA device; this module is the
        # numpy SloNet stack, so graphs are never enabled — capture() is False.
        self._enabled = False

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
        result = np.array([np.random.choice(p.shape[-1], p=p) for p in probs]).reshape(-1, 1)
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
        """Penalize recently generated tokens.

        HF semantics: positive logits divide by the penalty, negative
        multiply — both move the token's probability down.  ``recent_tokens``
        arrives as anything array-like (``sample()`` passes ``prev_tokens``
        verbatim, e.g. a ``(1, 3)`` row), so ravel before iterating.
        """
        import numpy as np

        recent_tokens = np.asarray(recent_tokens).ravel()
        if recent_tokens.size == 0:
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
        # Live view of the backing window (tests/callers record into it);
        # record_step() trims both lists to the window size.
        return self._step_times

    @property
    def losses(self):
        return self._losses

    @property
    def tokens_processed(self) -> int:
        return self._tokens_processed

    def record(self, name: str, duration: float, tokens: int = 0):
        self._records.append({"name": name, "duration": duration, "tokens": tokens})

    def record_step(
        self, loss: float = 0.0, step_time: float = 0.0, batch_size: int = 1, seq_len: int = 1
    ):
        self._losses.append(loss)
        self._step_times.append(step_time)
        self._tokens_processed += batch_size * seq_len
        if len(self._step_times) > self._window_size:
            self._step_times = self._step_times[-self._window_size :]
            self._losses = self._losses[-self._window_size :]

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


class _LRUBatchCache:
    """Legacy LRU cache for pre-batched training data (superseded by OptimizedBatchCache)."""

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
    An empty batch collates to ``None`` (nothing to stack).
    """
    import numpy as np

    if not batch:
        return None
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
        for i, (x, y) in enumerate(zip(xs, ys, strict=False)):
            padded_x[i, : x.shape[0]] = x
            padded_y[i, : y.shape[0]] = y
        return padded_x, padded_y
    return np.stack(batch, axis=0)


def _pad_last(a, target_len: int = 0, pad_value=0):
    """Pad the last axis to target length (or its own length = no-op).

    Documented 1D contract: ``_pad_last([1, 2], 4) -> [1, 2, 0, 0]``.
    Higher-rank inputs pad the trailing axis, so 1D behaviour is unchanged.
    """
    import numpy as np

    a = np.asarray(a)
    if a.ndim == 0:
        return a
    if target_len <= 0:
        target_len = a.shape[-1]
    if a.shape[-1] >= target_len:
        return a
    pad_width = [(0, 0)] * a.ndim
    pad_width[-1] = (0, target_len - a.shape[-1])
    return np.pad(a, pad_width, constant_values=pad_value)


class _NumpyBatchIterator:
    """Iterates over data (list or array) in batches.

    A real iterator: ``next(it)`` works directly (not just ``next(iter(it))``),
    and ``len()`` is the batch count — an empty dataset is one (empty) batch,
    so ``len`` never reports 0 batches for a dataset that is being iterated.
    """

    def __init__(self, data, batch_size: int, shuffle: bool = True):
        import numpy as np

        self._data = data
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._rng = np.random.default_rng()
        self._indices = np.arange(len(data))
        self._gen = None

    def __iter__(self):
        if self._shuffle:
            self._rng.shuffle(self._indices)
        for start in range(0, len(self._indices), self._batch_size):
            batch_indices = self._indices[start : start + self._batch_size]
            yield [self._data[int(i)] for i in batch_indices]

    def __next__(self):
        if self._gen is None:
            self._gen = self.__iter__()
        return next(self._gen)

    def __len__(self):
        import numpy as np

        return max(1, int(np.ceil(len(self._data) / self._batch_size)))


class _RestartingBatchIterator:
    """Iterator that starts a fresh epoch whenever the source is exhausted.

    ``OptimizedDataLoader`` must never terminate: one pass over the dataset
    is one epoch, and the next ``next()`` continues into a new one.
    """

    def __init__(self, source):
        self._source = source
        self._gen = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._gen is None:
            self._gen = self._source()
            return next(self._gen)  # empty dataset -> StopIteration here
        try:
            return next(self._gen)
        except StopIteration:
            self._gen = self._source()
            return next(self._gen)


class OptimizedDataLoader:
    """Simple data loader backed by numpy arrays.

    Batches are numpy arrays (``batch.tolist()`` works).  Iteration restarts
    at end-of-dataset (infinite epoch stream), and ``prefetch()`` /
    ``get_batch()`` expose a pull-style window over the same stream.
    Default worker count comes from the resource manager.
    """

    def __init__(self, dataset, batch_size: int = 32, shuffle: bool = True, workers=None):
        from domain.infrastructure._internal.resource_manager import ResourceManager

        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle
        # Underlying batch source — one object across epochs so shuffle order
        # carries over; `self.dataloader` is the live handle callers poke at.
        self.dataloader = _NumpyBatchIterator(dataset, batch_size, shuffle)
        if workers is None:
            workers = ResourceManager().dataloader_workers
        self.workers = effective_dataloader_workers(workers)
        self._stream = None
        self._buffer = None

    def _new_epoch(self):
        """One epoch's worth of collated batches (ndarrays), shuffle-aware."""
        for batch in iter(self.dataloader):
            out = _collate(batch)
            if out is None:
                return
            yield out

    def _batch_array(self):
        """Next batch from the pull stream (wraps at end of data)."""
        if self._stream is None:
            self._stream = _RestartingBatchIterator(self._new_epoch)
        return next(self._stream)

    def __iter__(self):
        return _RestartingBatchIterator(self._new_epoch)

    def __len__(self):
        return len(self.dataloader)

    def prefetch(self):
        """Fill the one-slot buffer with the next batch."""
        self._buffer = self._batch_array()

    def get_batch(self):
        """Return the prefetched (or next) batch, wrapping at end of data."""
        if self._buffer is not None:
            batch, self._buffer = self._buffer, None
            return batch
        return self._batch_array()


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
        self._cuda_graph_manager = (
            CUDAGraphManager(model, config) if config and config.use_cuda_graphs else None
        )

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


def _as_array(x, dtype=None):
    """Convert input to numpy array if needed.

    slonet ``Tensor`` (and similar wrappers) carry their values in ``.data`` —
    ``np.asarray()`` on them yields an object-dtype array of Tensor reprs, not
    the numbers.  ``dtype`` converts the result when given.
    """
    import numpy as np

    if isinstance(x, np.ndarray):
        out = x
    else:
        data = getattr(x, "data", None)
        if isinstance(data, np.ndarray):
            out = data
        else:
            out = np.asarray(x)
    if dtype is not None:
        out = out.astype(dtype, copy=False)
    return out


def _clip_grad_norm_(parameters, max_norm: float = 1.0) -> float:
    """Clip the global gradient norm to ``max_norm``; return the pre-clip norm.

    ``parameters`` may be a model (anything with ``.parameters()``) or an
    iterable of parameters.  Gradients are scaled in place.  Returns 0.0 when
    nothing carries a gradient.
    """
    import numpy as np

    if hasattr(parameters, "parameters"):
        params = list(parameters.parameters())
    else:
        params = list(parameters)

    grads = []
    for p in params:
        g = getattr(p, "grad", None)
        if g is None:
            continue
        arr = _as_array(g)
        if arr.size == 0:
            continue
        grads.append(arr)

    if not grads:
        return 0.0

    total_sq = 0.0
    for arr in grads:
        flat = arr.reshape(-1)
        total_sq += float(np.dot(flat, flat))
    total_norm = float(np.sqrt(total_sq))
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        for arr in grads:
            arr *= clip_coef
    return total_norm


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


def benchmark_inference(model, input_ids=None, n_tokens: int = 50, **kwargs) -> dict:
    """Benchmark inference latency/throughput with wall-clock timing.

    Runs ``num_runs`` timed greedy-decode windows of ``gen_len`` forward steps
    each (real model work, not a stub).

    Contract keys: ``avg_latency_ms``, ``p50_latency_ms``, ``p95_latency_ms``,
    ``tokens_per_sec``, ``device`` — plus the legacy ``n_tokens`` /
    ``latency_ms``.
    """
    import time

    import numpy as np

    batch_size = int(kwargs.get("batch_size", 1))
    seq_len = int(kwargs.get("seq_len", 8))
    gen_len = int(kwargs.get("gen_len", max(1, min(int(n_tokens), 8))))
    runs = max(1, int(kwargs.get("num_runs", kwargs.get("n_runs", 5))))

    device = get_optimal_device()
    vocab = int(getattr(model, "vocab_size", 50257) or 50257)

    if input_ids is not None:
        x = np.asarray(input_ids, dtype=np.int64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
    else:
        x = np.random.default_rng(0).integers(
            0, max(vocab, 1), size=(max(batch_size, 1), max(seq_len, 1))
        )
        x = x.astype(np.int64)

    limits = [
        v for v in (getattr(model, "block_size", None), getattr(model, "max_seq_len", None)) if v
    ]
    limit = max(min(limits) if limits else 1024, x.shape[1])

    def _step(ctx):
        """One decode step: forward, take the greedy token, grow the context."""
        out = model(ctx)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        arr = _as_array(logits)
        nxt = np.asarray(arr[:, -1, :], dtype=np.float64).argmax(axis=-1).reshape(-1, 1)
        nxt = nxt.astype(np.int64)
        if ctx.shape[1] + 1 > limit:
            ctx = ctx[:, -(limit - 1) :] if limit > 1 else ctx
        return np.concatenate([ctx, nxt], axis=1)

    latencies: list[float] = []
    generated = 0
    started = time.perf_counter()
    for _ in range(runs):
        t0 = time.perf_counter()
        ctx = x
        try:
            for _g in range(max(gen_len, 1)):
                ctx = _step(ctx)
                generated += ctx.shape[0]
        except Exception as exc:
            # Keep the wall time actually spent — a benchmark that swallows
            # its failure must still report the time it burned.
            latencies.append(time.perf_counter() - t0)
            logger.warning("benchmark_inference step failed: %s", exc)
            break
        latencies.append(time.perf_counter() - t0)
    elapsed = time.perf_counter() - started

    if not latencies:
        latencies = [elapsed]
    latencies = [l if l > 0 else 1e-9 for l in latencies]
    arr = np.asarray(latencies, dtype=np.float64) * 1000.0
    avg_ms = float(arr.mean())
    tokens_per_sec = generated / elapsed if elapsed > 0 else 0.0

    return {
        "avg_latency_ms": avg_ms,
        "p50_latency_ms": float(np.percentile(arr, 50)),
        "p95_latency_ms": float(np.percentile(arr, 95)),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "device": device,
        # legacy keys
        "n_tokens": generated,
        "latency_ms": round(avg_ms, 2),
    }


def benchmark_training(model, dataset=None, n_steps: int | None = None, **kwargs) -> dict:
    """Benchmark training throughput with wall-clock timing.

    Runs ``num_steps`` real forward(+loss) steps on a synthetic or supplied
    token batch.

    Contract keys: ``elapsed_sec``, ``tokens_per_sec``, ``steps_per_sec``,
    ``device`` — plus the legacy ``n_steps`` / ``loss``.
    """
    import time

    import numpy as np

    batch_size = int(kwargs.get("batch_size", 8))
    seq_len = int(kwargs.get("seq_len", kwargs.get("block_size", 64)))
    if n_steps is None:
        n_steps = kwargs.get("num_steps", kwargs.get("n_steps", 10))
    n_steps = int(n_steps)

    device = get_optimal_device()
    vocab = int(getattr(model, "vocab_size", 0) or 50257)

    x = None
    if dataset is not None:
        try:
            flat = np.asarray(dataset, dtype=np.int64).reshape(-1)
            if flat.size >= batch_size * seq_len:
                x = flat[: batch_size * seq_len].reshape(batch_size, seq_len)
        except Exception:
            x = None
    if x is None:
        x = np.random.default_rng(0).integers(
            0, max(vocab, 1), size=(max(batch_size, 1), max(seq_len, 1))
        )
        x = x.astype(np.int64)
    y = np.roll(x, -1, axis=1)  # language-modelling targets: next token

    # Warmup (not timed) so the first step doesn't pay one-time costs.
    try:
        model(x, y)
    except Exception as exc:
        logger.warning("benchmark_training warmup failed: %s", exc)

    start = time.perf_counter()
    completed = 0
    last_loss = 0.0
    for _ in range(max(n_steps, 0)):
        try:
            out = model(x, y)
        except Exception as exc:
            logger.warning("benchmark_training step failed: %s", exc)
            break
        completed += 1
        if isinstance(out, (tuple, list)) and len(out) > 1 and out[1] is not None:
            try:
                last_loss = float(np.asarray(_as_array(out[1])).reshape(-1)[0])
            except Exception:
                last_loss = 0.0
        zero_grad = getattr(model, "zero_grad", None)
        if callable(zero_grad):
            try:
                zero_grad()
            except Exception:
                pass
    elapsed = time.perf_counter() - start

    steps_per_sec = completed / elapsed if elapsed > 0 else 0.0
    tokens_per_sec = completed * x.shape[0] * x.shape[1] / elapsed if elapsed > 0 else 0.0

    return {
        "elapsed_sec": elapsed,
        "tokens_per_sec": round(tokens_per_sec, 1),
        "steps_per_sec": round(steps_per_sec, 2),
        "device": device,
        # legacy keys
        "n_steps": completed,
        "loss": round(last_loss, 4),
    }


def optimize_model_for_inference(model, config: InferenceOptimizations | None = None, **kwargs):
    """Optimize model for inference (no-op stub — actual optimization is device-specific).

    Accepts a legacy ``device=``/``use_…`` keyword set; the model is returned
    unchanged.
    """
    return model
