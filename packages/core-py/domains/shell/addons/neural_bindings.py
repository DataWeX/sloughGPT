"""
Neural facade — all 28 delegating methods extracted from kernel.py.

These functions are attached to the kernel instance during addon setup(),
making the neural addon fully self-contained. The kernel retains only thin
property wrappers for attribute-access compatibility.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..kernel_process import Priority


class Property:
    """Instance-attachable property descriptor for addon-injected attributes."""

    def __init__(self, attr_name: str, require_addon: bool = True,
                 default_factory: Any = None):
        self._attr = attr_name
        self._require = require_addon
        self._default_factory = default_factory

    def __set_name__(self, owner: Any, name: str) -> None:
        self._name = name

    def __get__(self, obj: Any, objtype: Any = None) -> Any:
        if obj is None:
            return self
        if self._require:
            obj._require_addon("neural")
        return getattr(obj, self._attr, None) if not self._default_factory else (
            getattr(obj, self._attr, None) or self._default_factory()
        )


# ---------------------------------------------------------------------------
# Property instances (installed on kernel class during setup)
# ---------------------------------------------------------------------------

engine = Property("_engine")
tokenizer_device = Property("_tokenizer_device")
embedding_device = Property("_embedding_device")
kv_caches = Property("_kv_caches")
gradient_accumulator = Property("_gradient_accumulator")
batch_processor = Property("_batch_processor")


def embedding_store(kernel: Any) -> Any:
    """Return the first embedding store, or an empty one."""
    from .neural import NeuralEmbeddingStore
    kernel._require_addon("neural")
    stores = list(kernel._embedding_stores.values())
    return stores[0] if stores else NeuralEmbeddingStore()


# ---------------------------------------------------------------------------
# Neural process management
# ---------------------------------------------------------------------------

def create_neural_process(kernel: Any, name: str, neural_type: Any = None,
                          model_name: str = "", priority: Priority = Priority.NORMAL,
                          **kwargs) -> Any:
    from .neural import NeuralProcess, NeuralProcessType
    kernel._require_addon("neural")
    proc = kernel.spawn_process(name, priority)
    neural = NeuralProcess(process=proc, model_name=model_name)
    neural.neural_type = neural_type or NeuralProcessType.INFERENCE
    with kernel._lock:
        kernel._neural_processes[proc.pid] = neural
    return neural


def get_neural_process(kernel: Any, pid: int) -> Any:
    kernel._require_addon("neural")
    return kernel._neural_processes.get(pid)


def list_neural_processes(kernel: Any) -> list:
    kernel._require_addon("neural")
    return list(kernel._neural_processes.values())


# ---------------------------------------------------------------------------
# Tokenization / embedding
# ---------------------------------------------------------------------------

def tokenize(kernel: Any, text: str) -> list[int]:
    kernel._require_addon("neural")
    result = kernel._tokenizer_device.ioctl("encode", text)
    if result and hasattr(result, 'value') and result.value:
        return result.value.get("tokens", [])
    return list(text.encode("utf-8"))


def detokenize(kernel: Any, tokens: list[int]) -> str:
    kernel._require_addon("neural")
    result = kernel._tokenizer_device.ioctl("decode", tokens)
    if result and hasattr(result, 'value') and result.value:
        return result.value.get("text", "")
    return bytes(tokens).decode("utf-8", errors="replace")


def embed(kernel: Any, ids: np.ndarray, store_name: str = "default") -> np.ndarray | None:
    kernel._require_addon("neural")
    store = kernel._embedding_stores.get(store_name)
    if store is None:
        return None
    return store.lookup(ids)


def embed_text(kernel: Any, text: str) -> np.ndarray:
    from .neural import NeuralEmbeddingStore, NeuralSyscall
    kernel._require_addon("neural")
    store = list(kernel._embedding_stores.values())[0] if kernel._embedding_stores else NeuralEmbeddingStore()
    return NeuralSyscall.embed(store, text)


def create_embedding_store(kernel: Any, name: str, vocab_size: int = 1000,
                           embed_dim: int = 64) -> Any:
    from .neural import NeuralEmbeddingStore
    kernel._require_addon("neural")
    store = NeuralEmbeddingStore(vocab_size=vocab_size, embed_dim=embed_dim)
    kernel._embedding_stores[name] = store
    return store


# ---------------------------------------------------------------------------
# KV cache management
# ---------------------------------------------------------------------------

def create_kv_cache(kernel: Any, name: str, num_layers: int = 6,
                    head_dim: int = 32, **kwargs: Any) -> Any:
    from .neural import NeuralKVCache
    kernel._require_addon("neural")
    cache = NeuralKVCache(num_layers=num_layers, head_dim=head_dim,
                          max_positions=kwargs.get('max_positions', 512))
    kernel._kv_caches[name] = cache
    return cache


def get_kv_cache(kernel: Any, name: str) -> Any:
    kernel._require_addon("neural")
    return kernel._kv_caches.get(name)


def remove_kv_cache(kernel: Any, name: str) -> None:
    kernel._require_addon("neural")
    with kernel._lock:
        kernel._kv_caches.pop(name, None)


# ---------------------------------------------------------------------------
# Generation / forward / backward / attention
# ---------------------------------------------------------------------------

def generate(kernel: Any, model_name: str, prompt: str,
             max_tokens: int = 10, **kwargs: Any) -> dict[str, Any] | None:
    kernel._require_addon("neural")
    result = kernel._engine.ioctl("generate", model_name, prompt,
                                  max_tokens=max_tokens, **kwargs)
    if result and hasattr(result, 'value') and result.value:
        return result.value
    return None


def forward(kernel: Any, neural_proc: Any,
            inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    from .neural import NeuralSyscall
    return NeuralSyscall.forward(neural_proc, inputs)


def backward(kernel: Any, neural_proc: Any,
             grad_output: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    from .neural import NeuralSyscall
    return NeuralSyscall.backward(neural_proc, grad_output)


def attention(kernel: Any, q: np.ndarray, k: np.ndarray, v: np.ndarray,
              mask: np.ndarray | None = None) -> np.ndarray:
    from .neural import NeuralSyscall
    return NeuralSyscall.attention(kernel._attention_device, q, k, v, mask)


# ---------------------------------------------------------------------------
# Generic neural syscall
# ---------------------------------------------------------------------------

def neural_syscall(kernel: Any, proc: Any, op: str,
                   *args: Any, **kwargs: Any) -> Any:
    from .neural import NeuralSyscall, NeuralEmbeddingStore
    if op == "forward":
        return NeuralSyscall.forward(proc, *args, **kwargs)
    elif op == "backward":
        return NeuralSyscall.backward(proc, *args, **kwargs)
    elif op == "embed":
        return NeuralSyscall.embed(
            kernel._embedding_stores.get("default", NeuralEmbeddingStore()),
            *args, **kwargs)
    return None


# ---------------------------------------------------------------------------
# Device registration / cleanup / stats
# ---------------------------------------------------------------------------

def register_devices(kernel: Any) -> None:
    if "neural" in kernel._addons:
        kernel.register_device(kernel._engine)
        kernel.register_device(kernel._tokenizer_device)
        kernel.register_device(kernel._embedding_device)
        kernel.register_device(kernel._attention_device)


def cleanup_pid(kernel: Any, pid: int) -> None:
    if "neural" in kernel._addons:
        with kernel._lock:
            kernel._neural_processes.pop(pid, None)
    kernel.memory.free_pid(pid)


def neural_stats(kernel: Any) -> dict:
    kernel._require_addon("neural")
    return {
        "neural_processes": len(kernel._neural_processes),
        "kv_caches": len(kernel._kv_caches),
        "embedding_stores": len(kernel._embedding_stores),
        "gradient_accumulator": kernel._gradient_accumulator.stats(),
        "batch_processor": kernel._batch_processor.stats(),
        "attention_device": kernel._attention_device.info(),
        "engine": kernel._engine.info(),
    }
