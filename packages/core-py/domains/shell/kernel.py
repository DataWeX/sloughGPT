"""
AI-Native Kernel — the mother code.

Unified kernel combining core process/memory/device management with neural
capabilities (NPU, KV cache, embeddings, gradient accumulation, batch processing).

All kernel imports go through this module:
    from domains.shell.kernel import Kernel, Process, ProcessState, DaitRuntime, Resource

Supporting modules (kernel_process, kernel_memory, etc.) are imported by this
file — they contain the subsystem implementations. kernel.py is the canonical
source of the Kernel class.
"""

from __future__ import annotations

import time
import math
import logging
import threading
import numpy as np
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Any, Callable

from .kernel_process import Process, ProcessState, Priority, TensorRef
from .kernel_memory import TensorMemory, MemoryBlock
from .kernel_scheduler import Scheduler
from .kernel_syscall import SyscallTable, SyscallResult, SyscallNumber, build_default_syscall_table
from .kernel_devices import DeviceManager, DeviceDriver, DeviceType, DeviceState, DeviceHandle
from .kernel_interrupts import InterruptManager, InterruptType, Interrupt

logger = logging.getLogger("slo.kernel")


# ---------------------------------------------------------------------------
# Null Device
# ---------------------------------------------------------------------------

class NullDevice(DeviceDriver):
    """A null /dev/null device that discards writes and returns empty on read."""
    def __init__(self):
        super().__init__("null", DeviceType.CUSTOM)

    def read(self, **kwargs) -> bytes:
        return b""

    def write(self, data: Any) -> bool:
        return True


# ---------------------------------------------------------------------------
# Neural enums
# ---------------------------------------------------------------------------

class NeuralOp(IntEnum):
    """Neural operation types."""
    NONE = 0
    EMBEDDING = 1
    ATTENTION = 2
    LINEAR = 3
    NORM = 4
    ACTIVATION = 5
    POOLING = 6
    CONVOLUTION = 7
    LOSS = 8
    OPTIMIZER_STEP = 9


class NeuralState(IntEnum):
    """Neural process states (extends ProcessState)."""
    IDLE = 0
    LOADING_WEIGHTS = 1
    COMPUTING = 2
    WAITING_INPUT = 3
    COMPLETE = 4
    FAILED = 5
    BACKPROPAGATING = 6
    OPTIMIZING = 7


class NeuralProcessType(IntEnum):
    """Types of neural processes."""
    INFERENCE = 0
    TRAINING = 1
    GENERATION = 2
    ATTENTION = 3


class NeuralMemoryType(IntEnum):
    """Types of neural memory."""
    KV_CACHE = 0
    EMBEDDING = 1
    ACTIVATION = 2
    GRADIENT = 3
    WAITING_GRADIENT = 4
    BACKPROPAGATING = 5
    OPTIMIZING = 6
    COMPLETE = 7
    FAILED = 8


class CacheStrategy(IntEnum):
    """KV cache eviction strategies."""
    LRU = 0
    LFU = 1
    FIFO = 2
    PRIORITY = 3


# ---------------------------------------------------------------------------
# Neural Process
# ---------------------------------------------------------------------------

@dataclass
class NeuralProcess:
    """
    Neural process — an inference or training job managed by the kernel.

    Extends Process with neural-network-specific state: model reference,
    operation type, tensor buffers, and gradient tracking.
    """
    process: Process
    neural_state: NeuralState = NeuralState.IDLE
    neural_type: NeuralProcessType = NeuralProcessType.INFERENCE
    op_type: NeuralOp = NeuralOp.NONE
    model_name: str = ""
    model_ref: Any = None
    weights_loaded: bool = False

    # Input/output tensors
    input_tensors: dict[str, np.ndarray] = field(default_factory=dict)
    output_tensors: dict[str, np.ndarray] = field(default_factory=dict)

    # Gradient tracking
    gradients: dict[str, np.ndarray] = field(default_factory=dict)
    gradient_norm: float = 0.0
    learning_rate: float = 0.001

    # Timing
    compute_start: float | None = None
    compute_end: float | None = None
    forward_time_ms: float = 0.0

    # Token tracking
    token_count: int = 0
    generated_text: str = ""
    tokens_per_second: float = 0.0

    # Loss and attention
    loss: float = 0.0
    attention_converged: bool = False

    # Error tracking
    last_error: str | None = None
    retry_count: int = 0
    max_retries: int = 3

    @property
    def pid(self) -> int:
        return self.process.pid

    @property
    def name(self) -> str:
        return self.process.name

    @property
    def state(self) -> ProcessState:
        return self.process.state

    @property
    def is_computing(self) -> bool:
        return self.neural_state in (
            NeuralState.COMPUTING,
            NeuralState.BACKPROPAGATING,
            NeuralState.OPTIMIZING,
        )

    @property
    def compute_time_ms(self) -> float:
        if self.compute_start is None:
            return 0.0
        end = self.compute_end or time.time()
        return (end - self.compute_start) * 1000

    def transition_neural(self, state: NeuralState) -> None:
        old = self.neural_state
        self.neural_state = state
        if state == NeuralState.COMPUTING:
            self.compute_start = time.time()
            self.compute_end = None
        elif state in (NeuralState.COMPLETE, NeuralState.FAILED):
            self.compute_end = time.time()
        logger.debug("Neural pid=%d: %s -> %s", self.pid, old.name, state.name)

    def record_tokens(self, token_ids: list[int], text: str) -> None:
        self.token_count = len(token_ids)
        self.generated_text = text
        if self.forward_time_ms > 0:
            self.tokens_per_second = self.token_count / (self.forward_time_ms / 1000.0)

    def set_loss(self, loss: float) -> None:
        self.loss = loss
        self.process.metadata["loss"] = loss

    def record_attention(self, patterns: list[np.ndarray]) -> None:
        entropies = []
        for pattern in patterns:
            if pattern.ndim >= 2:
                if pattern.ndim == 3:
                    flat = pattern.reshape(-1, pattern.shape[-1])
                else:
                    flat = pattern
                row_sums = flat.sum(axis=-1, keepdims=True)
                row_sums = np.where(row_sums > 0, row_sums, 1.0)
                normalized = flat / row_sums
                log_p = np.log(normalized + 1e-10)
                entropy = -(normalized * log_p).sum(axis=-1).mean()
                entropies.append(float(entropy))
        if entropies:
            avg_entropy = sum(entropies) / len(entropies)
            self.attention_converged = avg_entropy < 0.5

    def record_gradients(self, grads: dict[str, np.ndarray]) -> None:
        self.gradients.update(grads)
        total = 0.0
        for g in grads.values():
            total += float(np.sum(g ** 2))
        self.gradient_norm = math.sqrt(total)

    def start_timing(self) -> None:
        self.process.transition(ProcessState.RUNNING)

    def stop_timing(self, result: Any = None) -> None:
        self.process.result = result
        self.process.transition(ProcessState.ZOMBIE)

    def store_gradient(self, name: str, grad: np.ndarray) -> None:
        self.gradients[name] = grad
        self.gradient_norm += float(np.sum(grad ** 2))

    def clear_gradients(self) -> None:
        self.gradients.clear()
        self.gradient_norm = 0.0

    def status_line(self) -> str:
        base = self.process.status_line()
        return f"{base} [{self.neural_state.name}] model={self.model_name}"


# ---------------------------------------------------------------------------
# Neural KV Cache
# ---------------------------------------------------------------------------

@dataclass
class KVCacheEntry:
    """A single cached key-value pair for one layer."""
    layer_idx: int
    keys: np.ndarray | None = None
    values: np.ndarray | None = None
    seq_len: int = 0
    last_access: float = 0.0
    access_count: int = 0


class NeuralKVCache:
    """
    Per-process KV cache for transformer inference.

    Stores keys and values from attention layers so that past context
    doesn't need to be recomputed on each forward pass.
    """

    def __init__(self, num_layers: int = 12, head_dim: int = 64,
                 max_positions: int = 512, **kwargs):
        self._num_layers = num_layers
        self._head_dim = head_dim
        self._max_positions = max_positions
        self._num_heads = kwargs.get('num_heads', 8)
        self._entries: dict[int, KVCacheEntry] = {}
        self._position = 0
        self._lock = threading.Lock()
        self._total_tokens_cached = 0
        self._evictions = 0

    @property
    def total_tokens_cached(self) -> int:
        return self._total_tokens_cached

    @property
    def evictions(self) -> int:
        return self._evictions

    def initialize(self, num_heads: int) -> None:
        self._num_heads = num_heads
        with self._lock:
            for i in range(self._num_layers):
                keys = np.zeros((num_heads, self._max_positions, self._head_dim))
                values = np.zeros((num_heads, self._max_positions, self._head_dim))
                self._entries[i] = KVCacheEntry(
                    layer_idx=i, keys=keys, values=values,
                    seq_len=0, last_access=time.time(),
                )

    def get_position(self) -> int:
        return self._position

    def memory_bytes(self) -> int:
        total = 0
        for entry in self._entries.values():
            if entry.keys is not None:
                total += entry.keys.nbytes
            if entry.values is not None:
                total += entry.values.nbytes
        return total

    def update(self, layer_idx: int, k: np.ndarray,
               v: np.ndarray) -> int:
        with self._lock:
            if layer_idx not in self._entries:
                if layer_idx < 0 or layer_idx >= self._num_layers:
                    raise ValueError(f"Layer {layer_idx} out of range (0-{self._num_layers - 1})")
                num_heads = k.shape[0] if k.ndim >= 2 else 1
                keys = np.zeros((num_heads, self._max_positions, self._head_dim))
                values = np.zeros((num_heads, self._max_positions, self._head_dim))
                self._entries[layer_idx] = KVCacheEntry(
                    layer_idx=layer_idx, keys=keys, values=values)

            entry = self._entries[layer_idx]
            pos = self._position
            if entry.keys is not None and pos < entry.keys.shape[1]:
                entry.keys[:, pos, :] = k
                entry.values[:, pos, :] = v
            entry.seq_len = max(entry.seq_len, pos + 1)
            entry.last_access = time.time()
            entry.access_count += 1
            return pos

    def advance(self, n: int = 1) -> None:
        self._position += n

    def get(self, layer_idx: int, start: int = 0,
            end: int | None = None) -> tuple[np.ndarray | None, np.ndarray | None]:
        with self._lock:
            entry = self._entries.get(layer_idx)
            if entry is None or entry.keys is None:
                return None, None
            entry.last_access = time.time()
            entry.access_count += 1
            if end is None:
                end = entry.seq_len
            return entry.keys[:, start:end, :], entry.values[:, start:end, :]

    def reset(self, layer_idx: int | None = None) -> None:
        with self._lock:
            if layer_idx is not None:
                self._entries.pop(layer_idx, None)
            else:
                self._entries.clear()
                self._position = 0
                self._total_tokens_cached = 0

    def stats(self) -> dict:
        with self._lock:
            return {
                "layers_cached": len(self._entries),
                "total_tokens": self._total_tokens_cached,
                "evictions": self._evictions,
                "memory_bytes": self.memory_bytes(),
                "memory_mb": self.memory_bytes() / (1024 * 1024),
                "max_positions": self._max_positions,
                "position": self._position,
            }


# ---------------------------------------------------------------------------
# Neural Embedding Store
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingEntry:
    """A stored embedding with metadata."""
    id: str
    vector: np.ndarray
    text: str
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class NeuralEmbeddingStore:
    """
    In-memory embedding store with cosine similarity search.

    Stores text embeddings and supports nearest-neighbor queries.
    Used for knowledge retrieval, memory, and semantic search.
    """

    def __init__(self, vocab_size: int = 1000, embed_dim: int = 64,
                 dim: int | None = None, max_entries: int = 100_000):
        self._vocab_size = vocab_size
        self._embed_dim = embed_dim
        self._dim = dim or embed_dim
        self._max_entries = max_entries
        self._entries: dict[str, EmbeddingEntry] = {}
        self._embeddings = np.random.randn(vocab_size, embed_dim).astype(np.float32) * 0.01
        self._lock = threading.Lock()
        self._dirty = True

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def size(self) -> int:
        return len(self._entries)

    def lookup(self, ids: np.ndarray) -> np.ndarray:
        return self._embeddings[ids]

    def update(self, ids: np.ndarray, vecs: np.ndarray) -> int:
        count = 0
        for i, idx in enumerate(ids):
            if 0 <= idx < self._vocab_size and i < vecs.shape[0]:
                self._embeddings[idx] = vecs[i]
                count += 1
        return count

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def nearest(self, query: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
        norms = np.linalg.norm(self._embeddings, axis=1)
        valid = norms > 1e-10
        if not np.any(valid):
            return []
        normalized = np.zeros_like(self._embeddings)
        normalized[valid] = self._embeddings[valid] / norms[valid, np.newaxis]
        q_norm = np.linalg.norm(query)
        if q_norm < 1e-10:
            return []
        q_normalized = query / q_norm
        scores = normalized @ q_normalized
        scores[~valid] = -np.inf
        order = np.lexsort((-np.arange(len(scores)), scores))
        top_indices = order[::-1][:k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def add(self, id: str, vector: np.ndarray, text: str,
            metadata: dict | None = None) -> None:
        with self._lock:
            self._entries[id] = EmbeddingEntry(
                id=id,
                vector=vector / (np.linalg.norm(vector) + 1e-10),
                text=text,
                metadata=metadata or {},
            )
            self._dirty = True

    def search(self, query: np.ndarray, top_k: int = 5) -> list[tuple[str, float, str]]:
        results = self.nearest(query, k=top_k)
        out = []
        ids = list(self._entries.keys())
        for idx, score in results:
            if idx < len(ids):
                id_ = ids[idx]
                entry = self._entries.get(id_)
                if entry:
                    out.append((id_, score, entry.text))
        return out

    def stats(self) -> dict:
        return {
            "vocab_size": self._vocab_size,
            "embed_dim": self._embed_dim,
            "entries": len(self._entries),
            "max_entries": self._max_entries,
        }


# ---------------------------------------------------------------------------
# Neural Devices (kernel-level hardware abstractions)
# ---------------------------------------------------------------------------

class NeuralEngineDevice(DeviceDriver):
    """
    Neural engine device — wraps an inference/training engine as a device.

    Processes interact with the engine through the standard device API:
    open -> write(input) -> read(output) -> close.
    """

    def __init__(self, name: str = "neural_engine"):
        super().__init__(name, DeviceType.INFERENCE)
        self._models: dict[str, Any] = {}
        self._request_count = 0
        self._lock = threading.Lock()

    def load_model(self, name: str, model: Any) -> None:
        self._models[name] = model

    def unload_model(self, name: str) -> None:
        self._models.pop(name, None)

    def info(self) -> dict:
        return {
            "name": self._name,
            "model_names": list(self._models.keys()),
            "models_loaded": len(self._models),
            "request_count": self._request_count,
        }

    def read(self, offset: int = 0, size: int = -1) -> Any:
        with self._lock:
            return self._models

    def write(self, data: Any) -> bool:
        with self._lock:
            self._request_count += 1
            return True

    def ioctl(self, command: str, *args: Any, **kwargs: Any) -> Any:
        if command == "forward":
            model_name = args[0] if args else None
            inputs = args[1] if len(args) > 1 else None
            model = self._models.get(model_name)
            if model is None:
                return SyscallResult(success=False, error=f"Model '{model_name}' not found")
            output = model(inputs)
            return SyscallResult(success=True, value={"output": output})

        elif command == "generate":
            model_name = args[0] if args else None
            prompt = args[1] if len(args) > 1 else None
            max_tokens = kwargs.get("max_tokens", 10)
            model = self._models.get(model_name)
            if model is None:
                return SyscallResult(success=False, error=f"Model '{model_name}' not found")
            if hasattr(model, "generate_numpy"):
                tokens = model.generate_numpy(prompt, max_tokens=max_tokens)
            else:
                tokens = []
            return SyscallResult(success=True, value={"token_count": len(tokens), "tokens": tokens})

        elif command == "attention":
            q, k, v = args[0], args[1], args[2]
            d_k = q.shape[-1]
            scale = 1.0 / math.sqrt(d_k)
            scores = np.matmul(q, np.swapaxes(k, -2, -1)) * scale
            attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            attn_weights = attn_weights / (np.sum(attn_weights, axis=-1, keepdims=True) + 1e-10)
            output = np.matmul(attn_weights, v)
            return SyscallResult(success=True, value={
                "output": output,
                "attention_weights": attn_weights,
            })

        elif command == "loss":
            pred, tgt = args[0], args[1]
            loss_fn = kwargs.get("loss_fn", "mse")
            if loss_fn == "cross_entropy":
                clipped = np.clip(pred, 1e-7, 1.0 - 1e-7)
                loss = -np.sum(tgt * np.log(clipped)) / pred.shape[0]
            else:
                loss = float(np.mean((pred - tgt) ** 2))
            return SyscallResult(success=True, value={"loss": loss})

        return None


class TokenizerDevice(DeviceDriver):
    """
    Tokenizer device — wraps a tokenizer as a device.

    Supports encode and decode operations through ioctl.
    """

    def __init__(self, name: str = "tokenizer", tokenizer: Any = None):
        super().__init__(name, DeviceType.CUSTOM)
        self._tokenizer = tokenizer
        self._tokenize_count = 0
        self._total_tokens = 0

    def read(self, offset: int = 0, size: int = -1) -> Any:
        if self._tokenizer is None:
            return None
        return getattr(self._tokenizer, "vocab_size", 0)

    def write(self, data: Any) -> bool:
        if self._tokenizer is None or not isinstance(data, str):
            return False
        try:
            tokens = self._tokenizer.encode(data) if hasattr(self._tokenizer, "encode") else []
            self._tokenize_count += 1
            self._total_tokens += len(tokens)
            return True
        except Exception:
            return False

    def ioctl(self, command: str, *args: Any, **kwargs: Any) -> Any:
        if command == "encode":
            text = args[0] if args else ""
            if self._tokenizer and hasattr(self._tokenizer, "encode"):
                tokens = self._tokenizer.encode(text)
                return SyscallResult(success=True, value={"tokens": tokens, "encoding": "custom"})
            tokens = list(text.encode("utf-8"))
            return SyscallResult(success=True, value={"tokens": tokens, "encoding": "byte-level"})

        elif command == "decode":
            token_ids = args[0] if args else []
            if self._tokenizer and hasattr(self._tokenizer, "decode"):
                text = self._tokenizer.decode(token_ids)
                return SyscallResult(success=True, value={"text": text})
            text = bytes(token_ids).decode("utf-8", errors="replace")
            return SyscallResult(success=True, value={"text": text})

        return None

    def info(self) -> dict:
        base = super().info()
        base["tokenize_count"] = self._tokenize_count
        base["total_tokens"] = self._total_tokens
        return base


class EmbeddingStoreDevice(DeviceDriver):
    """
    Embedding store device — manages named NeuralEmbeddingStore instances.

    Supports create, lookup, update, and nearest operations through ioctl.
    """

    def __init__(self, name: str = "embedding-store", store: Any = None):
        super().__init__(name, DeviceType.STORAGE)
        self._stores: dict[str, NeuralEmbeddingStore] = {}
        if store is not None:
            self._stores["default"] = store
        self._query_count = 0

    def get_store(self, name: str) -> NeuralEmbeddingStore | None:
        return self._stores.get(name)

    def create_store(self, name: str, vocab_size: int = 1000,
                     embed_dim: int = 64) -> NeuralEmbeddingStore:
        store = NeuralEmbeddingStore(vocab_size=vocab_size, embed_dim=embed_dim)
        self._stores[name] = store
        return store

    def read(self, offset: int = 0, size: int = -1) -> Any:
        return {name: s.stats() for name, s in self._stores.items()}

    def write(self, data: Any) -> bool:
        return False

    def ioctl(self, command: str, *args: Any, **kwargs: Any) -> Any:
        store_name = kwargs.get("store_name", "default")

        if command == "create":
            vocab_size = kwargs.get("vocab_size", 1000)
            embed_dim = kwargs.get("embed_dim", 64)
            self.create_store(store_name, vocab_size, embed_dim)
            return SyscallResult(success=True)

        elif command == "lookup":
            ids = args[0] if args else []
            store = self._stores.get(store_name)
            if store is None:
                return SyscallResult(success=False, error=f"Store '{store_name}' not found")
            vecs = store.lookup(np.array(ids))
            return SyscallResult(success=True, value={"vectors": vecs, "shape": vecs.shape})

        elif command == "update":
            ids = args[0] if args else []
            vecs = args[1] if len(args) > 1 else None
            store = self._stores.get(store_name)
            if store is None:
                return SyscallResult(success=False, error=f"Store '{store_name}' not found")
            count = store.update(np.array(ids), vecs)
            return SyscallResult(success=True, value={"updated": count})

        elif command == "nearest":
            query = args[0] if args else None
            k = kwargs.get("k", 5)
            store = self._stores.get(store_name)
            if store is None:
                return SyscallResult(success=False, error=f"Store '{store_name}' not found")
            results = store.nearest(query, k=k)
            return SyscallResult(success=True, value={"nearest": results})

        return None

    def info(self) -> dict:
        base = super().info()
        base["stores"] = list(self._stores.keys())
        return base


# ---------------------------------------------------------------------------
# Neural Interrupts & Syscalls
# ---------------------------------------------------------------------------

class NeuralInterrupt:
    """Neural-specific interrupt types and helpers."""

    @staticmethod
    def inference_done(pid: int, result: Any = None) -> Interrupt:
        return Interrupt(
            vector=InterruptType.INFERENCE_DONE,
            source_pid=pid,
            data=result,
        )

    @staticmethod
    def training_step(pid: int, loss: float = 0.0, step: int = 0) -> Interrupt:
        return Interrupt(
            vector=InterruptType.TRAINING_STEP,
            source_pid=pid,
            data={"loss": loss, "step": step},
        )

    @staticmethod
    def gradient_update(pid: int, grad_norm: float = 0.0) -> Interrupt:
        return Interrupt(
            vector=InterruptType.GRADIENT_UPDATE,
            source_pid=pid,
            data={"grad_norm": grad_norm},
        )

    @staticmethod
    def data_ready(pid: int, batch_size: int = 0) -> Interrupt:
        return Interrupt(
            vector=InterruptType.DATA_READY,
            source_pid=pid,
            data={"batch_size": batch_size},
        )


class NeuralSyscall:
    """Neural-specific syscall numbers and handlers."""

    TOKENIZE = 20000
    GENERATE = 20001

    @staticmethod
    def forward(neural_proc: NeuralProcess, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        neural_proc.transition_neural(NeuralState.COMPUTING)
        try:
            if neural_proc.model_ref is not None and hasattr(neural_proc.model_ref, "forward"):
                outputs = neural_proc.model_ref.forward(inputs)
            else:
                outputs = inputs
            neural_proc.output_tensors = outputs
            neural_proc.transition_neural(NeuralState.COMPLETE)
            return outputs
        except Exception as e:
            neural_proc.last_error = str(e)
            neural_proc.transition_neural(NeuralState.FAILED)
            raise

    @staticmethod
    def backward(neural_proc: NeuralProcess,
                 grad_output: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        neural_proc.transition_neural(NeuralState.BACKPROPAGATING)
        try:
            if neural_proc.model_ref is not None and hasattr(neural_proc.model_ref, "backward"):
                grads = neural_proc.model_ref.backward(grad_output)
            else:
                grads = {}
            for name, grad in grads.items():
                neural_proc.store_gradient(name, grad)
            neural_proc.transition_neural(NeuralState.COMPLETE)
            return grads
        except Exception as e:
            neural_proc.last_error = str(e)
            neural_proc.transition_neural(NeuralState.FAILED)
            raise

    @staticmethod
    def embed(store: NeuralEmbeddingStore, text: str) -> np.ndarray:
        if hasattr(store, "_dim"):
            vec = np.random.randn(store._dim).astype(np.float32)
            vec /= np.linalg.norm(vec) + 1e-10
            return vec
        return np.zeros(384)

    @staticmethod
    def attention(device: Any, q: np.ndarray,
                  k: np.ndarray, v: np.ndarray,
                  mask: np.ndarray | None = None) -> np.ndarray:
        if hasattr(device, '_compute_attention'):
            return device._compute_attention(q, k, v, mask)
        return np.zeros_like(q)


# ---------------------------------------------------------------------------
# Gradient Accumulator
# ---------------------------------------------------------------------------

class GradientAccumulator:
    """
    Accumulates gradients across mini-batches before applying an optimizer step.

    Supports gradient clipping and averaging.
    """

    def __init__(self, max_grad_norm: float = 1.0, accumulation_steps: int = 1):
        self._max_grad_norm = max_grad_norm
        self._accumulation_steps = accumulation_steps
        self._accumulated: dict[str, np.ndarray] = {}
        self._step_count = 0
        self._total_norm = 0.0
        self._lock = threading.Lock()

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def ready(self) -> bool:
        return self._step_count >= self._accumulation_steps

    def accumulate(self, gradients: dict[str, np.ndarray]) -> bool:
        with self._lock:
            self._step_count += 1
            for name, grad in gradients.items():
                if name in self._accumulated:
                    self._accumulated[name] += grad / self._accumulation_steps
                else:
                    self._accumulated[name] = grad / self._accumulation_steps
            total_sq = sum(float(np.sum(g ** 2)) for g in self._accumulated.values())
            self._total_norm = math.sqrt(total_sq)
            return self.ready

    def get_clipped_gradients(self) -> dict[str, np.ndarray]:
        with self._lock:
            if self._total_norm > self._max_grad_norm:
                scale = self._max_grad_norm / (self._total_norm + 1e-10)
                return {k: v * scale for k, v in self._accumulated.items()}
            return dict(self._accumulated)

    def reset(self) -> None:
        with self._lock:
            self._accumulated.clear()
            self._step_count = 0
            self._total_norm = 0.0

    def stats(self) -> dict:
        with self._lock:
            return {
                "step_count": self._step_count,
                "accumulation_steps": self._accumulation_steps,
                "total_norm": self._total_norm,
                "max_grad_norm": self._max_grad_norm,
                "ready": self.ready,
                "param_groups": len(self._accumulated),
            }


# ---------------------------------------------------------------------------
# Batch Processor
# ---------------------------------------------------------------------------

@dataclass
class BatchRequest:
    """A request to process in a batch."""
    id: str
    inputs: dict[str, np.ndarray]
    callback: Callable | None = None
    priority: int = 0
    created_at: float = field(default_factory=time.time)


@dataclass
class BatchResult:
    """Result of processing a batch."""
    id: str
    outputs: dict[str, np.ndarray]
    elapsed_ms: float = 0.0
    error: str | None = None


class BatchProcessor:
    """
    Batches multiple inference requests for efficient GPU/CPU utilization.
    """

    def __init__(self, max_batch_size: int = 32, max_wait_ms: float = 10.0,
                 process_fn: Callable | None = None):
        self._max_batch_size = max_batch_size
        self._max_wait_ms = max_wait_ms
        self._process_fn = process_fn
        self._queue: list[BatchRequest] = []
        self._lock = threading.Lock()
        self._total_batches = 0
        self._total_requests = 0
        self._total_errors = 0

    @property
    def queue_size(self) -> int:
        with self._lock:
            return len(self._queue)

    def submit(self, request: BatchRequest) -> bool:
        with self._lock:
            if len(self._queue) >= self._max_batch_size * 2:
                return False
            self._queue.append(request)
            self._total_requests += 1
            return True

    def process_batch(self) -> list[BatchResult]:
        with self._lock:
            if not self._queue:
                return []
            batch = self._queue[:self._max_batch_size]
            self._queue = self._queue[self._max_batch_size:]

        results = []
        self._total_batches += 1
        start = time.time()

        for req in batch:
            try:
                if self._process_fn is not None:
                    outputs = self._process_fn(req.inputs)
                else:
                    outputs = req.inputs
                results.append(BatchResult(
                    id=req.id,
                    outputs=outputs,
                    elapsed_ms=(time.time() - start) * 1000,
                ))
            except Exception as e:
                self._total_errors += 1
                results.append(BatchResult(
                    id=req.id,
                    outputs={},
                    error=str(e),
                    elapsed_ms=(time.time() - start) * 1000,
                ))

        for req in batch:
            if req.callback is not None:
                try:
                    res = next((r for r in results if r.id == req.id), None)
                    req.callback(res)
                except Exception:
                    pass

        return results

    def flush(self) -> list[BatchResult]:
        all_results = []
        while True:
            results = self.process_batch()
            if not results:
                break
            all_results.extend(results)
        return all_results

    def stats(self) -> dict:
        with self._lock:
            return {
                "queue_size": len(self._queue),
                "max_batch_size": self._max_batch_size,
                "total_batches": self._total_batches,
                "total_requests": self._total_requests,
                "total_errors": self._total_errors,
            }


# ---------------------------------------------------------------------------
# Multi-Head Attention Device
# ---------------------------------------------------------------------------

class MultiHeadAttentionDevice(DeviceDriver):
    """Multi-head attention device — wraps an attention mechanism."""

    def __init__(self, name: str = "mha-device", num_heads: int = 8,
                 head_dim: int = 64):
        super().__init__(name, DeviceType.CUSTOM)
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._compute_count = 0

    def _compute_attention(self, q: np.ndarray, k: np.ndarray,
                           v: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        self._compute_count += 1
        d_k = q.shape[-1]
        scale = 1.0 / math.sqrt(d_k)
        scores = np.matmul(q, k.transpose(-2, -1)) * scale
        if mask is not None:
            scores = np.where(mask, scores, -1e9)
        attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = attn_weights / (np.sum(attn_weights, axis=-1, keepdims=True) + 1e-10)
        return np.matmul(attn_weights, v)

    def read(self, offset: int = 0, size: int = -1) -> Any:
        return {"compute_count": self._compute_count}

    def write(self, data: Any) -> bool:
        return False

    def ioctl(self, command: str, *args: Any, **kwargs: Any) -> Any:
        if command == "attention" and len(args) >= 3:
            return self._compute_attention(args[0], args[1], args[2],
                                           args[3] if len(args) > 3 else None)
        return None

    def info(self) -> dict:
        base = super().info()
        base["num_heads"] = self._num_heads
        base["head_dim"] = self._head_dim
        base["compute_count"] = self._compute_count
        return base


# ---------------------------------------------------------------------------
# Unified Kernel — core + neural
# ---------------------------------------------------------------------------

class Kernel:
    """
    AI-native unified kernel — the mother code.

    Combines core process/memory/device management with neural capabilities:
    NPU device, KV cache, embedding stores, gradient accumulation,
    batch processing, attention, and neural syscalls.

    Single entry point for all kernel operations.
    """

    def __init__(self):
        # Core subsystems
        self._scheduler = Scheduler()
        self._memory = TensorMemory()
        self._devices = DeviceManager()
        self._interrupts = InterruptManager()
        self._syscall_table = build_default_syscall_table()

        # Process tracking
        self._next_pid = 1
        self._processes: dict[int, Process] = {}
        self._lock = threading.Lock()

        # Lifecycle
        self._boot_time: float | None = None
        self._running = False
        self._tick_count = 0

        # Hooks
        self._on_tick: list = []
        self._on_process_done: list = []

        # Neural state
        self._neural_processes: dict[int, NeuralProcess] = {}
        self._kv_caches: dict[str, NeuralKVCache] = {}
        self._embedding_stores: dict[str, NeuralEmbeddingStore] = {}
        self._gradient_accumulator = GradientAccumulator()
        self._batch_processor = BatchProcessor()
        self._attention_device = MultiHeadAttentionDevice()
        self._engine = NeuralEngineDevice()
        self._tokenizer_device = TokenizerDevice()
        self._embedding_device = EmbeddingStoreDevice()
        self._next_neural_id = 1

        # Wire up default interrupt handlers
        self._interrupts.vector.register(
            InterruptType.PROCESS_DONE, self._handle_process_done
        )
        self._interrupts.vector.register(
            InterruptType.MEMORY_FULL, self._handle_memory_full
        )
        self._interrupts.vector.register(
            InterruptType.DEVICE_ERROR, self._handle_device_error
        )

        # Register neural syscalls
        self._register_neural_syscalls()

    # --- Lifecycle ---

    def boot(self) -> str:
        if self._running:
            return "Already booted"
        self._boot_time = time.time()
        self._running = True

        # Register built-in devices
        self._devices.register(NullDevice())

        # Boot init process (completes immediately — it's just the boot marker)
        init_proc = self.spawn_process(
            "kernel-init",
            Priority.CRITICAL,
            entry=lambda: "booted",
        )

        msg = f"Kernel booted (pid={init_proc.pid}, memory={self._memory.capacity // (1024 * 1024)}MB)"
        logger.info(msg)
        return msg

    def spawn_shell(self, shell_class: Any = None, **kwargs: Any) -> Process:
        """Spawn the interactive shell as a kernel process.

        If shell_class is None, lazily imports ShellREPL. The shell runs
        as a NORMAL priority process — it gets scheduled by the kernel's
        tick loop.
        """
        if shell_class is None:
            from .repl import ShellREPL
            shell_class = ShellREPL

        def _shell_entry():
            shell = shell_class(**kwargs)
            shell.run()

        proc = self.spawn_process(
            "shell",
            priority=Priority.NORMAL,
            entry=_shell_entry,
        )
        return proc

    def spawn_kernel_shell(self, stdin_fn=None, stdout_fn=None) -> Process:
        """Spawn a simple kernel shell process.

        A minimal command loop that processes commands via the kernel's
        built-in dispatch. Commands: help, meminfo, procs, halt.
        """
        kernel = self

        def _kernel_shell_entry():
            prompt = "ai-compteur> "
            if stdout_fn:
                stdout_fn(prompt)

            while True:
                if stdin_fn:
                    line = stdin_fn()
                else:
                    break

                line = line.strip()
                if not line:
                    if stdout_fn:
                        stdout_fn(prompt)
                    continue

                parts = line.split()
                cmd = parts[0].lower()
                args = parts[1:]

                if cmd == "help":
                    cmds = "help, meminfo, procs, run, ls, cat, halt"
                    if stdout_fn:
                        stdout_fn(f"commands: {cmds}\n")
                elif cmd == "meminfo":
                    info = kernel._memory.stats()
                    if stdout_fn:
                        stdout_fn(f"blocks: {info.get('block_count', 0)}\n")
                elif cmd == "procs":
                    for p in kernel.list_processes():
                        if stdout_fn:
                            stdout_fn(f"  pid={p.pid} {p.name} {p.state.name}\n")
                elif cmd == "run":
                    if not args:
                        if stdout_fn:
                            stdout_fn("usage: run <program.asm>\n")
                    else:
                        prog_name = args[0]
                        if stdout_fn:
                            stdout_fn(f"loading {prog_name}...\n")
                        try:
                            from .vm import DiskProgramLoader, FlatFS, BlockDevice
                            if not hasattr(kernel, '_block_device'):
                                kernel._block_device = BlockDevice()
                                kernel._fs = FlatFS(kernel._block_device)
                            loader = DiskProgramLoader(kernel._fs)
                            result = loader.run(prog_name, stdout_fn=stdout_fn)
                            if stdout_fn:
                                stdout_fn(f"done ({result['steps']} steps)\n")
                        except Exception as e:
                            if stdout_fn:
                                stdout_fn(f"error: {e}\n")
                elif cmd == "ls":
                    if not hasattr(kernel, '_fs'):
                        if stdout_fn:
                            stdout_fn("no filesystem mounted\n")
                    else:
                        files = kernel._fs.list_files()
                        if not files:
                            if stdout_fn:
                                stdout_fn("(empty)\n")
                        else:
                            for f in files:
                                if stdout_fn:
                                    stdout_fn(f"  {f}\n")
                elif cmd == "cat":
                    if not args:
                        if stdout_fn:
                            stdout_fn("usage: cat <file>\n")
                    elif not hasattr(kernel, '_fs'):
                        if stdout_fn:
                            stdout_fn("no filesystem mounted\n")
                    else:
                        try:
                            data = kernel._fs.read(args[0])
                            if stdout_fn:
                                stdout_fn(data.decode('utf-8', errors='replace').rstrip('\x00') + "\n")
                        except Exception as e:
                            if stdout_fn:
                                stdout_fn(f"error: {e}\n")
                elif cmd in ("halt", "exit", "quit"):
                    if stdout_fn:
                        stdout_fn("shutting down...\n")
                    break
                else:
                    if stdout_fn:
                        stdout_fn(f"unknown: {cmd}\n")

                if stdout_fn:
                    stdout_fn(prompt)

        proc = self.spawn_process(
            "kernel-shell",
            priority=Priority.NORMAL,
            entry=_kernel_shell_entry,
        )
        return proc

    def spawn_vm_process(self, name: str, source: str,
                         stdin_fn=None, stdout_fn=None,
                         priority: Priority = Priority.NORMAL,
                         use_syscalls: bool = False) -> Process:
        """Spawn a process that runs VM assembly code.

        Creates a VirtualSystem, loads the assembled program, and executes
        it in a background thread. I/O goes through the console device.
        If use_syscalls=True, wires SYSCALL instruction to kernel's syscall table.
        """
        from .vm import VirtualSystem, set_syscall_handler

        output_log: list[str] = []
        kernel = self

        def _handle_syscall(num, args):
            from .kernel_syscall import SyscallNumber
            if num == SyscallNumber.CONSOLE_WRITE:
                val = args[0]
                if stdout_fn:
                    stdout_fn(str(val) + "\n")
                else:
                    output_log.append(str(val))
                return 0
            elif num == SyscallNumber.CONSOLE_READ:
                if stdin_fn:
                    return stdin_fn()
                return ""
            elif num == SyscallNumber.EXIT:
                return -1
            elif num == SyscallNumber.MALLOC:
                block = kernel._memory.allocate(
                    shape=(args[0],) if args[0] else (1,),
                    dtype="float32",
                )
                return block.block_id
            elif num == SyscallNumber.FREE:
                kernel._memory.free_block(args[0])
                return 0
            elif num == SyscallNumber.UPTIME:
                return int(kernel.uptime * 1000)
            elif num == SyscallNumber.STATS:
                return kernel.info()
            return 0

        def _vm_entry():
            handler = _handle_syscall if use_syscalls else None
            vs = VirtualSystem(
                stdin_fn=stdin_fn,
                stdout_fn=stdout_fn or (lambda v: output_log.append(str(v))),
                syscall_handler=handler,
            )
            vs.load_program(source)
            vs.run()

        proc = self.spawn_process(
            name,
            priority=priority,
            entry=_vm_entry,
            metadata={"source": source, "output_log": output_log},
        )
        return proc

    def shutdown(self) -> str:
        if not self._running:
            return "Already shut down"
        self._running = False

        # Stop all processes
        for proc in list(self._processes.values()):
            if proc.is_active:
                proc.transition(ProcessState.STOPPED)

        # Free all process memory
        for pid in list(self._processes.keys()):
            self._memory.free_pid(pid)

        msg = f"Kernel shut down (uptime={self.uptime:.1f}s, ticks={self._tick_count})"
        logger.info(msg)
        return msg

    @property
    def uptime(self) -> float:
        if self._boot_time is None:
            return 0.0
        return time.time() - self._boot_time

    @property
    def running(self) -> bool:
        return self._running

    @property
    def tick_count(self) -> int:
        return self._tick_count

    # --- Process management ---

    def spawn_process(self, name: str, priority: Priority = Priority.NORMAL,
                      entry: Any = None, args: tuple = (), metadata: dict | None = None,
                      depends_on: list[int] | None = None) -> Process:
        with self._lock:
            pid = self._next_pid
            self._next_pid += 1

        proc = Process(
            pid=pid,
            name=name,
            priority=priority,
            entry=entry,
            args=args,
            metadata=metadata or {},
        )
        if depends_on:
            proc.metadata["depends_on"] = depends_on

        with self._lock:
            self._processes[pid] = proc

        self._scheduler.add(proc)
        logger.debug("Spawned pid=%d name=%s priority=%s", pid, name, priority.name)
        return proc

    def create_process(self, name: str, priority: Priority = Priority.NORMAL,
                       depends_on: list[int] | None = None) -> int:
        """Create a process and return its PID. Backward-compatible wrapper."""
        proc = self.spawn_process(name, priority, depends_on=depends_on)
        return proc.pid

    def kill_process(self, pid: int) -> bool:
        proc = self._processes.get(pid)
        if proc is None:
            return False
        proc.transition(ProcessState.STOPPED)
        self._scheduler.remove(pid)
        self._memory.free_pid(pid)
        self._interrupts.signal_process_done(pid)
        logger.debug("Killed pid=%d", pid)
        return True

    def get_process(self, pid: int) -> Process | None:
        return self._processes.get(pid)

    def list_processes(self) -> list[Process]:
        return list(self._processes.values())

    # --- Memory ---

    @property
    def memory(self) -> TensorMemory:
        return self._memory

    def alloc_tensor(self, shape: tuple, dtype: str = "float32") -> dict:
        """Allocate a tensor block in kernel memory. Returns block metadata."""
        block = self._memory.allocate(shape, dtype)
        return {
            "block_id": block.block_id,
            "shape": block.shape,
            "dtype": block.dtype,
            "size_bytes": block.size_bytes,
        }

    def free_tensor(self, block_id: int) -> bool:
        """Free a tensor block by ID."""
        return self._memory.free_block(block_id)

    # --- Devices ---

    @property
    def devices(self) -> DeviceManager:
        return self._devices

    def register_device(self, device: DeviceDriver) -> bool:
        return self._devices.register(device)

    def unregister_device(self, name: str) -> bool:
        return self._devices.unregister(name)

    def open_device(self, name: str) -> Any:
        """Open a device by name, returns a DeviceHandle."""
        return self._devices.open(name)

    def close_device(self, fd: Any) -> bool:
        """Close a device handle. Accepts DeviceHandle or int fd."""
        if isinstance(fd, DeviceHandle):
            return self._devices.close(fd.fd)
        return self._devices.close(fd)

    # --- Interrupts ---

    @property
    def interrupts(self) -> InterruptManager:
        return self._interrupts

    # --- Syscalls ---

    @property
    def syscall_table(self) -> SyscallTable:
        return self._syscall_table

    def syscall(self, number: Any, *args: Any, caller: Process | None = None, **kwargs: Any) -> Any:
        """Dispatch a syscall, handling both base and neural syscall numbers."""
        if caller is None:
            for proc in self._processes.values():
                caller = proc
                break
        if caller is None:
            caller = Process(pid=0, name="kernel", state=ProcessState.RUNNING)

        # Check if this is a known base syscall number
        try:
            sn = SyscallNumber(number)
        except ValueError:
            sn = None

        if sn is not None:
            # Handle TENSOR_ALLOC directly
            if sn == SyscallNumber.TENSOR_ALLOC:
                shape, dtype = args[0], args[1] if len(args) > 1 else "float32"
                info = self.alloc_tensor(shape, dtype)
                return SyscallResult(success=True, value=info)
            return self._syscall_table.dispatch(sn, caller, *args, **kwargs)

        # Custom neural syscall — dispatch directly through table
        return self._syscall_table.dispatch(number, caller, *args, **kwargs)

    # --- Scheduler ---

    @property
    def scheduler(self) -> Scheduler:
        return self._scheduler

    # --- Tick ---

    def tick(self) -> dict:
        """Advance the kernel by one tick.

        If the scheduled process has an entry function and hasn't been started
        yet, launches it in a background thread. When the entry completes,
        the process transitions to ZOMBIE.
        """
        if not self._running:
            return {"current_pid": None, "tick_count": self._tick_count}

        self._tick_count += 1
        proc = self._scheduler.tick()

        # Launch process entry function if present and not yet started
        if proc is not None and proc.entry is not None and proc._thread is None:
            proc.transition(ProcessState.RUNNING)
            proc.started_at = time.time()

            def _run_proc(p: Process):
                try:
                    result = p.entry(*p.args, **p.kwargs)
                    p.result = result
                except Exception as exc:
                    p.error = str(exc)
                    logger.error("Process %d (%s) crashed: %s", p.pid, p.name, exc)
                finally:
                    p.finished_at = time.time()
                    p.cpu_time_ms = (p.finished_at - (p.started_at or p.created_at)) * 1000
                    p.transition(ProcessState.ZOMBIE)
                    self._scheduler.complete(p.pid)
                    for cb in self._on_process_done:
                        try:
                            cb(p)
                        except Exception:
                            pass

            t = threading.Thread(target=_run_proc, args=(proc,), daemon=True, name=f"proc-{proc.pid}")
            proc._thread = t
            t.start()

        # Fire timer interrupt
        self._interrupts.vector.fire(Interrupt(
            vector=InterruptType.TIMER,
            data={"tick": self._tick_count},
        ))

        # Process pending interrupts
        self._interrupts.vector.process_pending()

        return {
            "current_pid": proc.pid if proc else None,
            "tick_count": self._tick_count,
        }

    # --- Hooks ---

    def on_tick(self, callback) -> None:
        self._on_tick.append(callback)

    def on_process_done(self, callback) -> None:
        self._on_process_done.append(callback)

    # --- Interrupt handlers ---

    def _handle_process_done(self, interrupt: Interrupt) -> None:
        pid = interrupt.source_pid
        if pid is not None:
            proc = self._processes.get(pid)
            if proc is not None:
                proc.result = interrupt.data
                for cb in self._on_process_done:
                    try:
                        cb(proc)
                    except Exception:
                        logger.exception("on_process_done callback failed")

    def _handle_memory_full(self, interrupt: Interrupt) -> None:
        logger.warning("Memory full interrupt fired")

    def _handle_device_error(self, interrupt: Interrupt) -> None:
        logger.error("Device error interrupt: %s", interrupt.data)

    # --- Run loop ---

    def run(self, max_ticks: int = 100) -> list[dict]:
        """Run the kernel for up to max_ticks, returning tick results."""
        results = []
        for _ in range(max_ticks):
            if not self._running:
                break
            result = self.tick()
            results.append(result)
            if not self._processes:
                break
        return results

    def run_program(self, source: str, trace: bool = False) -> dict:
        """Run a VM assembly program through the kernel's device bus.

        Creates a VM CPU, wires it to the kernel's devices, and executes
        the assembled program. Returns output, trace, and step count.
        """
        from .vm import CPU, Assembler, DeviceBus as VMBus

        vm_bus = VMBus()
        if hasattr(self._devices, '_table'):
            for name, dev in self._devices._table._devices.items():
                vm_bus.register(name, dev)
        elif hasattr(self._devices, '_devices'):
            for name, dev in self._devices._devices.items():
                vm_bus.register(name, dev)

        cpu = CPU(devices=vm_bus)
        cpu._tracing = trace
        assembler = Assembler()
        instructions = assembler.assemble(source)
        cpu.load_program(instructions)
        output = cpu.run()

        return {
            "output": output,
            "steps": cpu._step_count,
            "trace": cpu.get_trace() if trace else [],
            "regs": {f"R{i}": v for i, v in enumerate(cpu.regs) if v != 0},
        }

    # --- Info ---

    def info(self) -> dict:
        """Return a snapshot of kernel state."""
        return {
            "uptime_s": self.uptime,
            "running": self._running,
            "tick_count": self._tick_count,
            "process_count": len(self._processes),
            "memory": self._memory.stats(),
            "devices": self._devices.stats(),
            "interrupts": self._interrupts.stats(),
            "syscalls": self._syscall_table.stats(),
        }

    def stats(self) -> dict:
        return {
            "uptime": self.uptime,
            "running": self._running,
            "tick_count": self._tick_count,
            "process_count": len(self._processes),
            "scheduler": self._scheduler.stats(),
            "memory": self._memory.stats(),
            "devices": self._devices.stats(),
            "interrupts": self._interrupts.stats(),
            "syscalls": self._syscall_table.stats(),
        }

    # ===================================================================
    # Neural capabilities
    # ===================================================================

    @property
    def engine(self) -> NeuralEngineDevice:
        return self._engine

    @property
    def tokenizer_device(self) -> TokenizerDevice:
        return self._tokenizer_device

    @property
    def embedding_device(self) -> EmbeddingStoreDevice:
        return self._embedding_device

    @property
    def embedding_store(self) -> NeuralEmbeddingStore:
        stores = list(self._embedding_stores.values())
        return stores[0] if stores else NeuralEmbeddingStore()

    @property
    def kv_caches(self) -> dict[str, NeuralKVCache]:
        return self._kv_caches

    @property
    def gradient_accumulator(self) -> GradientAccumulator:
        return self._gradient_accumulator

    @property
    def batch_processor(self) -> BatchProcessor:
        return self._batch_processor

    # --- Neural process management ---

    def create_neural_process(self, name: str,
                              neural_type: NeuralProcessType = NeuralProcessType.INFERENCE,
                              model_name: str = "", priority: Priority = Priority.NORMAL,
                              **kwargs) -> NeuralProcess:
        proc = self.spawn_process(name, priority)
        neural = NeuralProcess(process=proc, model_name=model_name)
        neural.neural_type = neural_type
        with self._lock:
            self._neural_processes[proc.pid] = neural
        return neural

    def get_neural_process(self, pid: int) -> NeuralProcess | None:
        return self._neural_processes.get(pid)

    def list_neural_processes(self) -> list[NeuralProcess]:
        return list(self._neural_processes.values())

    # --- Tokenization ---

    def tokenize(self, text: str) -> list[int]:
        result = self._tokenizer_device.ioctl("encode", text)
        if result and hasattr(result, 'value') and result.value:
            return result.value.get("tokens", [])
        return list(text.encode("utf-8"))

    def detokenize(self, tokens: list[int]) -> str:
        result = self._tokenizer_device.ioctl("decode", tokens)
        if result and hasattr(result, 'value') and result.value:
            return result.value.get("text", "")
        return bytes(tokens).decode("utf-8", errors="replace")

    # --- Embeddings ---

    def embed(self, ids: np.ndarray, store_name: str = "default") -> np.ndarray | None:
        store = self._embedding_stores.get(store_name)
        if store is None:
            return None
        return store.lookup(ids)

    def embed_text(self, text: str) -> np.ndarray:
        store = list(self._embedding_stores.values())[0] if self._embedding_stores else NeuralEmbeddingStore()
        return NeuralSyscall.embed(store, text)

    def create_embedding_store(self, name: str, vocab_size: int = 1000,
                               embed_dim: int = 64) -> NeuralEmbeddingStore:
        store = NeuralEmbeddingStore(vocab_size=vocab_size, embed_dim=embed_dim)
        self._embedding_stores[name] = store
        return store

    # --- KV Cache ---

    def create_kv_cache(self, name: str, num_layers: int = 6,
                        head_dim: int = 32, **kwargs) -> NeuralKVCache:
        cache = NeuralKVCache(num_layers=num_layers, head_dim=head_dim,
                              max_positions=kwargs.get('max_positions', 512))
        self._kv_caches[name] = cache
        return cache

    def get_kv_cache(self, name: str) -> NeuralKVCache | None:
        return self._kv_caches.get(name)

    def remove_kv_cache(self, name: str) -> None:
        with self._lock:
            self._kv_caches.pop(name, None)

    # --- Generation ---

    def generate(self, model_name: str, prompt: str, max_tokens: int = 10, **kwargs: Any) -> dict[str, Any] | None:
        """Generate text using a loaded model via the engine."""
        result = self._engine.ioctl("generate", model_name, prompt, max_tokens=max_tokens, **kwargs)
        if result and hasattr(result, 'value') and result.value:
            return result.value
        return None

    # --- Forward/Backward ---

    def forward(self, neural_proc: NeuralProcess,
                inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return NeuralSyscall.forward(neural_proc, inputs)

    def backward(self, neural_proc: NeuralProcess,
                 grad_output: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return NeuralSyscall.backward(neural_proc, grad_output)

    # --- Attention ---

    def attention(self, q: np.ndarray, k: np.ndarray, v: np.ndarray,
                  mask: np.ndarray | None = None) -> np.ndarray:
        return NeuralSyscall.attention(self._attention_device, q, k, v, mask)

    # --- Neural syscall dispatch ---

    def neural_syscall(self, proc: NeuralProcess, op: str, *args: Any, **kwargs: Any) -> Any:
        if op == "forward":
            return NeuralSyscall.forward(proc, *args, **kwargs)
        elif op == "backward":
            return NeuralSyscall.backward(proc, *args, **kwargs)
        elif op == "embed":
            return NeuralSyscall.embed(self._embedding_stores.get("default", NeuralEmbeddingStore()), *args, **kwargs)
        return None

    def _register_neural_syscalls(self) -> None:
        table = self._syscall_table

        def _handle_tokenize(caller: Any, text: str, **kwargs: Any) -> dict:
            tokens = self.tokenize(text)
            return {"tokens": tokens}

        def _handle_generate(caller: Any, text: str, model_name: str = "",
                             **kwargs: Any) -> dict:
            result = self._engine.ioctl("generate", model_name, text, **kwargs)
            if result and hasattr(result, 'value') and result.value:
                return result.value
            return {"token_count": 0, "tokens": []}

        table.register(NeuralSyscall.TOKENIZE, "neural_tokenize",
                        _handle_tokenize, description="Neural tokenize")
        table.register(NeuralSyscall.GENERATE, "neural_generate",
                        _handle_generate, description="Neural generate")

    # --- Device registration ---

    def register_devices(self) -> None:
        """Register all neural devices with the kernel's device manager."""
        self.register_device(self._engine)
        self.register_device(self._tokenizer_device)
        self.register_device(self._embedding_device)
        self.register_device(self._attention_device)

    # --- Cleanup ---

    def cleanup_pid(self, pid: int) -> None:
        """Clean up all neural resources for a process."""
        with self._lock:
            self._neural_processes.pop(pid, None)
        self.memory.free_pid(pid)

    # --- Neural stats ---

    def neural_stats(self) -> dict:
        return {
            "neural_processes": len(self._neural_processes),
            "kv_caches": len(self._kv_caches),
            "embedding_stores": len(self._embedding_stores),
            "gradient_accumulator": self._gradient_accumulator.stats(),
            "batch_processor": self._batch_processor.stats(),
            "attention_device": self._attention_device.info(),
            "engine": self._engine.info(),
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_kernel: Kernel | None = None


def get_kernel() -> Kernel:
    global _kernel
    if _kernel is None:
        _kernel = Kernel()
    return _kernel


def reset_kernel() -> Kernel:
    global _kernel
    if _kernel is not None and _kernel.running:
        _kernel.shutdown()
    _kernel = Kernel()
    return _kernel
