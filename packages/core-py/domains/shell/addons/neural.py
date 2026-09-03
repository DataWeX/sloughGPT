"""
Neural addon — all neural capabilities for the kernel.

Provides: enums, NeuralProcess, KV cache, embedding store, neural devices,
gradient accumulator, batch processor, attention device, neural interrupts/syscalls.

Install via:
    from domains.shell.addons import neural
    kernel.install_addon(neural)
"""
from __future__ import annotations

import time
import math
import logging
import threading
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..kernel_process import Process, ProcessState
from ..kernel_devices import DeviceDriver, DeviceType
from ..kernel_interrupts import InterruptType, Interrupt
from ..kernel_syscall import SyscallResult

logger = logging.getLogger("slo.kernel.neural")


# --- Neural enums ---

class NeuralOp(IntEnum):
    NONE = 0; EMBEDDING = 1; ATTENTION = 2; LINEAR = 3; NORM = 4
    ACTIVATION = 5; POOLING = 6; CONVOLUTION = 7; LOSS = 8; OPTIMIZER_STEP = 9

class NeuralState(IntEnum):
    IDLE = 0; LOADING_WEIGHTS = 1; COMPUTING = 2; WAITING_INPUT = 3
    COMPLETE = 4; FAILED = 5; BACKPROPAGATING = 6; OPTIMIZING = 7

class NeuralProcessType(IntEnum):
    INFERENCE = 0; TRAINING = 1; GENERATION = 2; ATTENTION = 3

class NeuralMemoryType(IntEnum):
    KV_CACHE = 0; EMBEDDING = 1; ACTIVATION = 2; GRADIENT = 3
    WAITING_GRADIENT = 4; BACKPROPAGATING = 5; OPTIMIZING = 6
    COMPLETE = 7; FAILED = 8

class CacheStrategy(IntEnum):
    LRU = 0; LFU = 1; FIFO = 2; PRIORITY = 3


# --- Neural Process ---

@dataclass
class NeuralProcess:
    process: Process
    neural_state: NeuralState = NeuralState.IDLE
    neural_type: NeuralProcessType = NeuralProcessType.INFERENCE
    op_type: NeuralOp = NeuralOp.NONE
    model_name: str = ""
    model_ref: Any = None
    weights_loaded: bool = False
    input_tensors: dict[str, np.ndarray] = field(default_factory=dict)
    output_tensors: dict[str, np.ndarray] = field(default_factory=dict)
    gradients: dict[str, np.ndarray] = field(default_factory=dict)
    gradient_norm: float = 0.0
    learning_rate: float = 0.001
    compute_start: float | None = None
    compute_end: float | None = None
    forward_time_ms: float = 0.0
    token_count: int = 0
    generated_text: str = ""
    tokens_per_second: float = 0.0
    loss: float = 0.0
    attention_converged: bool = False
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
        return self.neural_state in (NeuralState.COMPUTING, NeuralState.BACKPROPAGATING, NeuralState.OPTIMIZING)

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
                flat = pattern.reshape(-1, pattern.shape[-1]) if pattern.ndim == 3 else pattern
                row_sums = flat.sum(axis=-1, keepdims=True)
                row_sums = np.where(row_sums > 0, row_sums, 1.0)
                normalized = flat / row_sums
                log_p = np.log(normalized + 1e-10)
                entropy = -(normalized * log_p).sum(axis=-1).mean()
                entropies.append(float(entropy))
        if entropies:
            self.attention_converged = sum(entropies) / len(entropies) < 0.5

    def record_gradients(self, grads: dict[str, np.ndarray]) -> None:
        self.gradients.update(grads)
        self.gradient_norm = math.sqrt(sum(float(np.sum(g ** 2)) for g in grads.values()))

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


# --- KV Cache ---

@dataclass
class KVCacheEntry:
    layer_idx: int
    keys: np.ndarray | None = None
    values: np.ndarray | None = None
    seq_len: int = 0
    last_access: float = 0.0
    access_count: int = 0


class NeuralKVCache:
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
                self._entries[i] = KVCacheEntry(
                    layer_idx=i,
                    keys=np.zeros((num_heads, self._max_positions, self._head_dim)),
                    values=np.zeros((num_heads, self._max_positions, self._head_dim)),
                    seq_len=0, last_access=time.time())

    def get_position(self) -> int:
        return self._position

    def memory_bytes(self) -> int:
        total = 0
        for e in self._entries.values():
            if e.keys is not None:
                total += e.keys.nbytes
            if e.values is not None:
                total += e.values.nbytes
        return total

    def update(self, layer_idx: int, k: np.ndarray, v: np.ndarray) -> int:
        with self._lock:
            if layer_idx not in self._entries:
                if layer_idx < 0 or layer_idx >= self._num_layers:
                    raise ValueError(f"Layer {layer_idx} out of range (0-{self._num_layers - 1})")
                nh = k.shape[0] if k.ndim >= 2 else 1
                self._entries[layer_idx] = KVCacheEntry(
                    layer_idx=layer_idx,
                    keys=np.zeros((nh, self._max_positions, self._head_dim)),
                    values=np.zeros((nh, self._max_positions, self._head_dim)))
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
            return {"layers_cached": len(self._entries), "total_tokens": self._total_tokens_cached,
                    "evictions": self._evictions, "memory_bytes": self.memory_bytes(),
                    "memory_mb": self.memory_bytes() / (1024 * 1024),
                    "max_positions": self._max_positions, "position": self._position}


# --- Embedding Store ---

@dataclass
class EmbeddingEntry:
    id: str
    vector: np.ndarray
    text: str
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class NeuralEmbeddingStore:
    def __init__(self, vocab_size: int = 1000, embed_dim: int = 64,
                 dim: int | None = None, max_entries: int = 100_000):
        self._vocab_size = vocab_size
        self._embed_dim = embed_dim
        self._dim = dim or embed_dim
        self._max_entries = max_entries
        self._entries: dict[str, EmbeddingEntry] = {}
        self._embeddings = np.random.randn(vocab_size, embed_dim).astype(np.float32) * 0.01
        self._lock = threading.Lock()

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
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
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
        scores = normalized @ (query / q_norm)
        scores[~valid] = -np.inf
        order = np.lexsort((-np.arange(len(scores)), scores))
        return [(int(idx), float(scores[idx])) for idx in order[::-1][:k]]

    def add(self, id: str, vector: np.ndarray, text: str,
            metadata: dict | None = None) -> None:
        with self._lock:
            self._entries[id] = EmbeddingEntry(
                id=id, vector=vector / (np.linalg.norm(vector) + 1e-10),
                text=text, metadata=metadata or {})

    def search(self, query: np.ndarray, top_k: int = 5) -> list[tuple[str, float, str]]:
        results = self.nearest(query, k=top_k)
        ids = list(self._entries.keys())
        out = []
        for idx, score in results:
            if idx < len(ids):
                entry = self._entries.get(ids[idx])
                if entry:
                    out.append((ids[idx], score, entry.text))
        return out

    def stats(self) -> dict:
        return {"vocab_size": self._vocab_size, "embed_dim": self._embed_dim,
                "entries": len(self._entries), "max_entries": self._max_entries}


# --- Neural Devices ---

class NeuralEngineDevice(DeviceDriver):
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
        return {"name": self._name, "model_names": list(self._models.keys()),
                "models_loaded": len(self._models), "request_count": self._request_count}

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
            return SyscallResult(success=True, value={"output": model(inputs)})
        elif command == "generate":
            model_name = args[0] if args else None
            prompt = args[1] if len(args) > 1 else None
            max_tokens = kwargs.get("max_tokens", 10)
            model = self._models.get(model_name)
            if model is None:
                return SyscallResult(success=False, error=f"Model '{model_name}' not found")
            tokens = model.generate_numpy(prompt, max_tokens=max_tokens) if hasattr(model, "generate_numpy") else []
            return SyscallResult(success=True, value={"token_count": len(tokens), "tokens": tokens})
        elif command == "attention":
            q, k, v = args[0], args[1], args[2]
            d_k = q.shape[-1]
            scale = 1.0 / math.sqrt(d_k)
            scores = np.matmul(q, np.swapaxes(k, -2, -1)) * scale
            attn = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            attn = attn / (np.sum(attn, axis=-1, keepdims=True) + 1e-10)
            return SyscallResult(success=True, value={"output": np.matmul(attn, v), "attention_weights": attn})
        elif command == "loss":
            pred, tgt = args[0], args[1]
            if kwargs.get("loss_fn") == "cross_entropy":
                clipped = np.clip(pred, 1e-7, 1.0 - 1e-7)
                loss = -np.sum(tgt * np.log(clipped)) / pred.shape[0]
            else:
                loss = float(np.mean((pred - tgt) ** 2))
            return SyscallResult(success=True, value={"loss": loss})
        return None


class TokenizerDevice(DeviceDriver):
    def __init__(self, name: str = "tokenizer", tokenizer: Any = None):
        super().__init__(name, DeviceType.CUSTOM)
        self._tokenizer = tokenizer
        self._tokenize_count = 0
        self._total_tokens = 0

    def read(self, offset: int = 0, size: int = -1) -> Any:
        return getattr(self._tokenizer, "vocab_size", 0) if self._tokenizer else None

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
                return SyscallResult(success=True, value={"tokens": self._tokenizer.encode(text), "encoding": "custom"})
            return SyscallResult(success=True, value={"tokens": list(text.encode("utf-8")), "encoding": "byte-level"})
        elif command == "decode":
            token_ids = args[0] if args else []
            if self._tokenizer and hasattr(self._tokenizer, "decode"):
                return SyscallResult(success=True, value={"text": self._tokenizer.decode(token_ids)})
            return SyscallResult(success=True, value={"text": bytes(token_ids).decode("utf-8", errors="replace")})
        return None

    def info(self) -> dict:
        base = super().info()
        base["tokenize_count"] = self._tokenize_count
        base["total_tokens"] = self._total_tokens
        return base


class EmbeddingStoreDevice(DeviceDriver):
    def __init__(self, name: str = "embedding-store", store: Any = None):
        super().__init__(name, DeviceType.STORAGE)
        self._stores: dict[str, NeuralEmbeddingStore] = {}
        if store is not None:
            self._stores["default"] = store

    def get_store(self, name: str) -> NeuralEmbeddingStore | None:
        return self._stores.get(name)

    def create_store(self, name: str, vocab_size: int = 1000, embed_dim: int = 64) -> NeuralEmbeddingStore:
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
            self.create_store(store_name, kwargs.get("vocab_size", 1000), kwargs.get("embed_dim", 64))
            return SyscallResult(success=True)
        store = self._stores.get(store_name)
        if store is None:
            return SyscallResult(success=False, error=f"Store '{store_name}' not found")
        if command == "lookup":
            vecs = store.lookup(np.array(args[0] if args else []))
            return SyscallResult(success=True, value={"vectors": vecs, "shape": vecs.shape})
        elif command == "update":
            count = store.update(np.array(args[0] if args else []), args[1] if len(args) > 1 else None)
            return SyscallResult(success=True, value={"updated": count})
        elif command == "nearest":
            return SyscallResult(success=True, value={"nearest": store.nearest(args[0], k=kwargs.get("k", 5))})
        return None

    def info(self) -> dict:
        base = super().info()
        base["stores"] = list(self._stores.keys())
        return base


class MultiHeadAttentionDevice(DeviceDriver):
    def __init__(self, name: str = "mha-device", num_heads: int = 8, head_dim: int = 64):
        super().__init__(name, DeviceType.CUSTOM)
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._compute_count = 0

    def _compute_attention(self, q: np.ndarray, k: np.ndarray,
                           v: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        self._compute_count += 1
        d_k = q.shape[-1]
        scores = np.matmul(q, np.swapaxes(k, -2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = np.where(mask, scores, -1e9)
        attn = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = attn / (np.sum(attn, axis=-1, keepdims=True) + 1e-10)
        return np.matmul(attn, v)

    def read(self, offset: int = 0, size: int = -1) -> Any:
        return {"compute_count": self._compute_count}

    def write(self, data: Any) -> bool:
        return False

    def ioctl(self, command: str, *args: Any, **kwargs: Any) -> Any:
        if command == "attention" and len(args) >= 3:
            return self._compute_attention(args[0], args[1], args[2], args[3] if len(args) > 3 else None)
        return None

    def info(self) -> dict:
        base = super().info()
        base["num_heads"] = self._num_heads
        base["head_dim"] = self._head_dim
        base["compute_count"] = self._compute_count
        return base


# --- Neural Interrupts & Syscalls ---

class NeuralInterrupt:
    @staticmethod
    def inference_done(pid: int, result: Any = None) -> Interrupt:
        return Interrupt(vector=InterruptType.INFERENCE_DONE, source_pid=pid, data=result)

    @staticmethod
    def training_step(pid: int, loss: float = 0.0, step: int = 0) -> Interrupt:
        return Interrupt(vector=InterruptType.TRAINING_STEP, source_pid=pid, data={"loss": loss, "step": step})

    @staticmethod
    def gradient_update(pid: int, grad_norm: float = 0.0) -> Interrupt:
        return Interrupt(vector=InterruptType.GRADIENT_UPDATE, source_pid=pid, data={"grad_norm": grad_norm})

    @staticmethod
    def data_ready(pid: int, batch_size: int = 0) -> Interrupt:
        return Interrupt(vector=InterruptType.DATA_READY, source_pid=pid, data={"batch_size": batch_size})


class NeuralSyscall:
    TOKENIZE = 20000
    GENERATE = 20001

    @staticmethod
    def forward(neural_proc: NeuralProcess, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        neural_proc.transition_neural(NeuralState.COMPUTING)
        try:
            outputs = neural_proc.model_ref.forward(inputs) if neural_proc.model_ref and hasattr(neural_proc.model_ref, "forward") else inputs
            neural_proc.output_tensors = outputs
            neural_proc.transition_neural(NeuralState.COMPLETE)
            return outputs
        except Exception as e:
            neural_proc.last_error = str(e)
            neural_proc.transition_neural(NeuralState.FAILED)
            raise

    @staticmethod
    def backward(neural_proc: NeuralProcess, grad_output: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        neural_proc.transition_neural(NeuralState.BACKPROPAGATING)
        try:
            grads = neural_proc.model_ref.backward(grad_output) if neural_proc.model_ref and hasattr(neural_proc.model_ref, "backward") else {}
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
    def attention(device: Any, q: np.ndarray, k: np.ndarray,
                  v: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        return device._compute_attention(q, k, v, mask) if hasattr(device, '_compute_attention') else np.zeros_like(q)


# --- Gradient Accumulator ---

class GradientAccumulator:
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
                self._accumulated[name] = self._accumulated.get(name, np.zeros_like(grad)) + grad / self._accumulation_steps
            self._total_norm = math.sqrt(sum(float(np.sum(g ** 2)) for g in self._accumulated.values()))
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
            return {"step_count": self._step_count, "accumulation_steps": self._accumulation_steps,
                    "total_norm": self._total_norm, "max_grad_norm": self._max_grad_norm,
                    "ready": self.ready, "param_groups": len(self._accumulated)}


# --- Batch Processor ---

@dataclass
class BatchRequest:
    id: str
    inputs: dict[str, np.ndarray]
    callback: Callable | None = None
    priority: int = 0
    created_at: float = field(default_factory=time.time)


@dataclass
class BatchResult:
    id: str
    outputs: dict[str, np.ndarray]
    elapsed_ms: float = 0.0
    error: str | None = None


class BatchProcessor:
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
                outputs = self._process_fn(req.inputs) if self._process_fn else req.inputs
                results.append(BatchResult(id=req.id, outputs=outputs, elapsed_ms=(time.time() - start) * 1000))
            except Exception as e:
                self._total_errors += 1
                results.append(BatchResult(id=req.id, outputs={}, error=str(e), elapsed_ms=(time.time() - start) * 1000))
        for req in batch:
            if req.callback is not None:
                try:
                    req.callback(next((r for r in results if r.id == req.id), None))
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
            return {"queue_size": len(self._queue), "max_batch_size": self._max_batch_size,
                    "total_batches": self._total_batches, "total_requests": self._total_requests,
                    "total_errors": self._total_errors}


# --- NeuralKernel convenience class ---
# NOTE: NeuralKernel is defined in kernel.py (avoids circular import).
# Re-exported here for backward compatibility.
def __getattr__(name: str):
    if name == "NeuralKernel":
        from ..kernel import NeuralKernel
        return NeuralKernel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# --- Addon setup ---

def _install_facade(kernel: Any) -> None:
    """Attach all neural facade functions and property descriptors to the kernel."""
    from .neural_bindings import (
        engine as _engine_prop,
        tokenizer_device as _tok_prop,
        embedding_device as _emb_prop,
        kv_caches as _kv_prop,
        gradient_accumulator as _grad_prop,
        batch_processor as _batch_prop,
        embedding_store,
        create_neural_process,
        get_neural_process,
        list_neural_processes,
        tokenize,
        detokenize,
        embed,
        embed_text,
        create_embedding_store,
        create_kv_cache,
        get_kv_cache,
        remove_kv_cache,
        generate,
        forward,
        backward,
        attention,
        neural_syscall,
        register_devices,
        cleanup_pid,
        neural_stats,
    )

    from ..kernel import Kernel

    # Install property descriptors on the Kernel class (shared by all instances)
    for name, prop in [
        ("engine", _engine_prop),
        ("tokenizer_device", _tok_prop),
        ("embedding_device", _emb_prop),
        ("kv_caches", _kv_prop),
        ("gradient_accumulator", _grad_prop),
        ("batch_processor", _batch_prop),
    ]:
        setattr(Kernel, name, prop)

    # Install bound functions on the kernel instance
    _facade_fns = {
        "embedding_store": lambda: embedding_store(kernel),
        "create_neural_process": lambda *a, **kw: create_neural_process(kernel, *a, **kw),
        "get_neural_process": lambda pid: get_neural_process(kernel, pid),
        "list_neural_processes": lambda: list_neural_processes(kernel),
        "tokenize": lambda text: tokenize(kernel, text),
        "detokenize": lambda tokens: detokenize(kernel, tokens),
        "embed": lambda *a, **kw: embed(kernel, *a, **kw),
        "embed_text": lambda text: embed_text(kernel, text),
        "create_embedding_store": lambda *a, **kw: create_embedding_store(kernel, *a, **kw),
        "create_kv_cache": lambda *a, **kw: create_kv_cache(kernel, *a, **kw),
        "get_kv_cache": lambda name: get_kv_cache(kernel, name),
        "remove_kv_cache": lambda name: remove_kv_cache(kernel, name),
        "generate": lambda *a, **kw: generate(kernel, *a, **kw),
        "forward": lambda *a, **kw: forward(kernel, *a, **kw),
        "backward": lambda *a, **kw: backward(kernel, *a, **kw),
        "attention": lambda *a, **kw: attention(kernel, *a, **kw),
        "neural_syscall": lambda *a, **kw: neural_syscall(kernel, *a, **kw),
        "register_devices": lambda: register_devices(kernel),
        "cleanup_pid": lambda pid: cleanup_pid(kernel, pid),
        "neural_stats": lambda: neural_stats(kernel),
    }
    for attr_name, fn in _facade_fns.items():
        setattr(kernel, attr_name, fn)


def setup(kernel: Any) -> None:
    """Install neural capabilities on the kernel."""
    kernel._neural_processes = {}
    kernel._kv_caches = {}
    kernel._embedding_stores = {}
    kernel._gradient_accumulator = GradientAccumulator()
    kernel._batch_processor = BatchProcessor()
    kernel._attention_device = MultiHeadAttentionDevice()
    kernel._engine = NeuralEngineDevice()
    kernel._tokenizer_device = TokenizerDevice()
    kernel._embedding_device = EmbeddingStoreDevice()
    kernel._next_neural_id = 1

    def _handle_tokenize(caller: Any, text: str, **kw: Any) -> dict:
        from .neural_bindings import tokenize as _tokenize
        return {"tokens": _tokenize(kernel, text)}

    def _handle_generate(caller: Any, text: str, model_name: str = "", **kw: Any) -> dict:
        result = kernel._engine.ioctl("generate", model_name, text, **kw)
        if result and hasattr(result, 'value') and result.value:
            return result.value
        return {"token_count": 0, "tokens": []}

    kernel._syscall_table.register(NeuralSyscall.TOKENIZE, "neural_tokenize", _handle_tokenize, description="Neural tokenize")
    kernel._syscall_table.register(NeuralSyscall.GENERATE, "neural_generate", _handle_generate, description="Neural generate")
    kernel.register_device(kernel._engine)
    kernel.register_device(kernel._tokenizer_device)
    kernel.register_device(kernel._embedding_device)
    kernel.register_device(kernel._attention_device)
    kernel._addons["neural"] = True

    _install_facade(kernel)
    logger.debug("Neural addon installed")
