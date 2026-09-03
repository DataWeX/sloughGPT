"""
NPU (Neural Processing Unit) device — bridges VM device bus to real inference.

NPUDevice wraps a model provider behind a standard device interface:
  forward(), generate(), embed(), tokenize(), detokenize(), train_step()
  plus profiling, checkpointing, quantization, batch processing, and
  attention map extraction.

NPUModel holds provider + metadata for a loaded model.
"""

from __future__ import annotations

import os
import time
import struct
import logging
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Any

from .kernel_devices import DeviceDriver, DeviceType, DeviceState
from .kernel_syscall import SyscallResult
from domains.inference.forward_pass import ForwardPassResult

try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

logger = logging.getLogger("slo.kernel.npu")


# ---------------------------------------------------------------------------
# NPUModel — lightweight model wrapper
# ---------------------------------------------------------------------------

@dataclass
class NPUModel:
    """Loaded model metadata. Holds a provider for actual inference."""
    name: str = ""
    provider: Any = None
    config: dict[str, Any] = field(default_factory=dict)
    loaded_at: float = 0.0
    inference_count: int = 0
    total_tokens: int = 0
    total_forward_ms: float = 0.0


# ---------------------------------------------------------------------------
# _HuggingFaceProvider — wraps transformers model for kernel integration
# ---------------------------------------------------------------------------

class _HuggingFaceProvider:
    """Wraps a HuggingFace model + tokenizer behind the kernel's provider interface."""

    def __init__(self, model: Any, tokenizer: Any, model_id: str, device: str = "cpu"):
        self._model = model
        self._tokenizer = tokenizer
        self._model_id = model_id
        self._device = device

    def metadata(self) -> dict:
        return {
            "model_id": self._model_id,
            "device": self._device,
            "vocab_size": getattr(self._tokenizer, "vocab_size", 0),
        }

    def __call__(self, inputs: dict) -> Any:
        raise RuntimeError("HuggingFace torch backend is not supported — use numpy or .slnc providers")

    def generate_numpy(self, prompt: str, max_tokens: int = 20,
                       temperature: float = 1.0, **kwargs) -> list[int]:
        raise RuntimeError("HuggingFace torch backend is not supported — use numpy or .slnc providers")

    def tokenize(self, text: str) -> list[int]:
        return self._tokenizer.encode(text)

    def detokenize(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(token_ids)


# ---------------------------------------------------------------------------
# NPUGraph — compute graph (model)
# ---------------------------------------------------------------------------

class NPUGraph:
    """Compute graph — the model itself.

    Load file → creates graph. Submit input → executes.
    """

    def __init__(self, name: str, provider: Any, config: dict):
        self._name = name
        self._provider = provider
        self._config = config
        self._stats = {"inferences": 0, "tokens": 0, "total_ms": 0.0}

    def __call__(self, input_data, **kwargs):
        """Submit input → graph executes."""
        if isinstance(input_data, str):
            return self._text(input_data, **kwargs)
        elif isinstance(input_data, (list, np.ndarray)):
            return self._tokens(input_data, **kwargs)
        return {"data": input_data}

    def _text(self, text: str, **kwargs):
        """Text → tokenize → forward → detokenize."""
        mode = kwargs.get("mode", "generate")
        max_tokens = kwargs.get("max_tokens", 100)

        if mode == "tokenize":
            tokens = self._provider.tokenize(text)
            return {"tokens": tokens, "count": len(tokens)}
        elif mode == "embed":
            emb = self._provider.embed(text)
            return {"embedding": emb, "shape": list(emb.shape)}

        # generate
        t0 = time.time()
        tokens = self._provider.tokenize(text)
        ids = np.array([tokens], dtype=np.int64)
        gen = self._provider.generate_numpy(ids, max_new_tokens=max_tokens)
        result = self._provider.detokenize(gen[0].tolist())
        ms = (time.time() - t0) * 1000
        self._stats["inferences"] += 1
        self._stats["tokens"] += len(gen[0])
        self._stats["total_ms"] += ms
        return {"text": result, "tokens": len(gen[0]), "ms": round(ms, 2)}

    def _tokens(self, tokens, **kwargs):
        """Tokens → forward → logits."""
        ids = np.array([tokens], dtype=np.int64) if isinstance(tokens, list) else tokens
        t0 = time.time()
        inner = getattr(self._provider, "_model", self._provider)
        if hasattr(inner, "forward_pass"):
            fpr = inner.forward_pass(ids)
            logits = fpr.logits if hasattr(fpr, "logits") else fpr
        else:
            logits = self._provider.forward_numpy(ids)
        ms = (time.time() - t0) * 1000
        self._stats["inferences"] += 1
        self._stats["total_ms"] += ms
        return {"logits": logits, "shape": list(logits.shape), "ms": round(ms, 2)}

    @property
    def stats(self):
        return self._stats.copy()

    @property
    def config(self):
        return self._config.copy()


# ---------------------------------------------------------------------------
# NPUDevice — hardware only
# ---------------------------------------------------------------------------

class NPUDevice(DeviceDriver):
    """NPU hardware device.

    Assembly interface:
        DEV_CALL R1, R0, load, path, name      → creates graph
        DEV_CALL R2, R0, call, name, input     → executes graph
        DEV_CALL R3, R0, unload, name          → removes graph
        DEV_CALL R4, R0, info                  → device info
    """

    def __init__(self, name: str = "npu", capabilities: list | None = None):
        super().__init__(name, DeviceType.INFERENCE)
        self._graphs: dict[str, NPUGraph] = {}
        self._default_graph: str = ""
        self._open_count: int = 0
        self._lock = threading.Lock()

    # ── Device lifecycle ──────────────────────────────────────────────────

    def open(self) -> bool:
        self._open_count += 1
        self._state = DeviceState.OPEN
        return True

    def close(self) -> bool:
        self._state = DeviceState.CLOSED
        return True

    def info(self) -> dict:
        return {
            "device": self.name,
            "graphs": len(self._graphs),
            "names": list(self._graphs.keys()),
            "default": self._default_graph,
        }

    # ── Graph management ──────────────────────────────────────────────────

    def load(self, path: str = "", name: str = "", **kwargs) -> NPUGraph:
        """Load file → creates graph on device."""
        if not path:
            raise ValueError("no path")

        key = name or path.rsplit("/", 1)[-1]

        if path.endswith(".slnc"):
            graph = self._load_slnc(path, key, **kwargs)
        elif path.endswith((".npy", ".npz")):
            graph = self._load_numpy(path, key)
        elif path.endswith(".py"):
            graph = self._load_python(path, key)
        elif path.endswith((".csv", ".json", ".parquet")):
            graph = self._load_dataset(path, key)
        else:
            raise ValueError(f"unsupported: {path}")

        self._graphs[key] = graph
        if not self._default_graph:
            self._default_graph = key
        return graph

    def unload(self, name: str) -> bool:
        """Remove graph from device."""
        if name in self._graphs:
            del self._graphs[name]
            if self._default_graph == name:
                self._default_graph = next(iter(self._graphs), "")
            return True
        return False

    def __call__(self, graph_name: str, input_data, **kwargs):
        """Execute graph on device."""
        name = graph_name or self._default_graph
        graph = self._graphs.get(name)
        if graph is None:
            raise ValueError(f"graph '{name}' not loaded")
        return graph(input_data, **kwargs)

    def ioctl(self, command: str, *args: Any) -> SyscallResult | dict:
        if command == "INFO":
            return SyscallResult.ok(self.info())
        elif command == "LOAD":
            path, name = (args[0] if len(args) > 0 else ""), (args[1] if len(args) > 1 else "")
            graph = self.load(path, name)
            return SyscallResult.ok({"graph": graph._name})
        elif command == "UNLOAD":
            ok = self.unload(args[0] if args else "")
            return SyscallResult.ok({"unloaded": ok})
        elif command == "CALL":
            name = args[0] if len(args) > 0 else ""
            inp = args[1] if len(args) > 1 else ""
            result = self(name, inp)
            return SyscallResult.ok(result)
        else:
            raise ValueError(f"unknown ioctl: {command}")

    # ── Backend loaders ───────────────────────────────────────────────────

    def _load_slnc(self, path: str, name: str, **kwargs) -> NPUGraph:
        backend = kwargs.get("backend")
        if backend == "c":
            try:
                provider = self._load_c(name, path)
            except Exception:
                provider = self._load_numpy(name, path)
        elif backend == "numpy":
            provider = self._load_numpy(name, path)
        else:
            try:
                provider = self._load_c(name, path)
            except Exception:
                provider = self._load_numpy(name, path)
        config = provider.metadata() if hasattr(provider, "metadata") else {}
        return NPUGraph(name, provider, config)

    def _load_numpy(self, name: str, path: str):
        from domains.inference.slonet_provider import SlonetChatProvider
        return SlonetChatProvider.from_slnc(path, model_id=name)

    def _load_c(self, name: str, path: str):
        from domains.inference.ct_provider import CTransformProvider
        return CTransformProvider.from_slnc(path, model_id=name)

    def _load_numpy(self, path: str, name: str) -> NPUGraph:
        arr = np.load(path, allow_pickle=False)
        provider = {"type": "numpy", "array": arr}
        config = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
        return NPUGraph(name, provider, config)

    def _load_python(self, path: str, name: str) -> NPUGraph:
        import importlib.util
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        provider = {"type": "python", "module": module}
        return NPUGraph(name, provider, {"type": "python_module"})

    def _load_dataset(self, path: str, name: str) -> NPUGraph:
        import pandas as pd
        if path.endswith(".csv"):
            data = pd.read_csv(path)
        elif path.endswith(".json"):
            data = pd.read_json(path)
        else:
            data = pd.read_parquet(path)
        provider = {"type": "dataset", "data": data}
        config = {"rows": len(data), "columns": list(data.columns)}
        return NPUGraph(name, provider, config)
