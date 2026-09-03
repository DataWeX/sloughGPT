"""
NPUDevice — hardware device driver.

Uses TensorDevice for computation.
Adds: model management, I/O, assembly interface (ioctl).
"""

from __future__ import annotations

import time
import logging
import threading
import numpy as np
from typing import Any

from .kernel_devices import DeviceDriver, DeviceType, DeviceState
from .kernel_syscall import SyscallResult
from .tensor_device import TensorDevice

logger = logging.getLogger("slo.kernel.npu")


class NPUDevice(DeviceDriver):
    """NPU hardware device — loads models, executes them.

    Uses TensorDevice for computation.
    Adds: model management, I/O, assembly interface (ioctl).
    """

    def __init__(self, name: str = "npu"):
        super().__init__(name, DeviceType.INFERENCE)
        self._compute = TensorDevice()  # Pure compute engine
        self._models: dict[str, Any] = {}
        self._default_model: str = ""
        self._checkpoints: dict[str, dict] = {}
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
            "compute_ops": self._compute.list_ops(),
            "models": len(self._models),
            "names": list(self._models.keys()),
            "default": self._default_model,
            "checkpoints": list(self._checkpoints.keys()),
        }

    # ── Compute access (for direct tensor ops) ────────────────────────────

    def compute(self, op: str, *args, **kwargs):
        """Direct access to TensorDevice for pure compute."""
        return self._compute(op, *args, **kwargs)

    # ── Model management ──────────────────────────────────────────────────

    def load(self, path: str = "", name: str = "", **kwargs) -> Any:
        """Load model from file."""
        if not path:
            raise ValueError("no path")

        key = name or path.rsplit("/", 1)[-1]

        if path.endswith(".slnc"):
            provider = self._load_slnc(path, key, **kwargs)
        elif path.endswith((".npy", ".npz")):
            provider = self._load_numpy(path, key)
        elif path.endswith(".py"):
            provider = self._load_python(path, key)
        elif path.endswith((".csv", ".json", ".parquet")):
            provider = self._load_dataset(path, key)
        else:
            raise ValueError(f"unsupported: {path}")

        self._models[key] = provider
        if not self._default_model:
            self._default_model = key
        return provider

    def unload(self, name: str) -> bool:
        """Unload model."""
        if name in self._models:
            del self._models[name]
            if self._default_model == name:
                self._default_model = next(iter(self._models), "")
            return True
        return False

    def __call__(self, model_name: str, input_data, **kwargs):
        """Execute model — input goes in, output comes out."""
        name = model_name or self._default_model
        provider = self._models.get(name)
        if provider is None:
            raise ValueError(f"model '{name}' not loaded")

        # Text input
        if isinstance(input_data, str):
            return self._execute_text(provider, input_data, **kwargs)
        # Token input
        elif isinstance(input_data, (list, np.ndarray)):
            return self._execute_tokens(provider, input_data, **kwargs)
        return {"data": input_data}

    def _execute_text(self, provider, text: str, **kwargs):
        """Text → tokenize → model → output."""
        mode = kwargs.get("mode", "generate")
        max_tokens = kwargs.get("max_tokens", 100)

        if mode == "tokenize":
            tokens = provider.tokenize(text)
            return {"tokens": tokens, "count": len(tokens)}
        elif mode == "embed":
            emb = provider.embed(text)
            return {"embedding": emb, "shape": list(emb.shape)}

        # generate
        t0 = time.time()
        tokens = provider.tokenize(text)
        ids = np.array([tokens], dtype=np.int64)
        gen = provider.generate_numpy(ids, max_new_tokens=max_tokens)
        result = provider.detokenize(gen[0].tolist())
        ms = (time.time() - t0) * 1000
        return {"text": result, "tokens": len(gen[0]), "ms": round(ms, 2)}

    def _execute_tokens(self, provider, tokens, **kwargs):
        """Tokens → model → logits."""
        ids = np.array([tokens], dtype=np.int64) if isinstance(tokens, list) else tokens
        t0 = time.time()
        inner = getattr(provider, "_model", provider)
        if hasattr(inner, "forward_pass"):
            fpr = inner.forward_pass(ids)
            logits = fpr.logits if hasattr(fpr, "logits") else fpr
        else:
            logits = provider.forward_numpy(ids)
        ms = (time.time() - t0) * 1000
        return {"logits": logits, "shape": list(logits.shape), "ms": round(ms, 2)}

    # ── Advanced operations ───────────────────────────────────────────────

    def batch(self, model_name: str, inputs: list, **kwargs) -> dict:
        """Execute multiple inputs."""
        name = model_name or self._default_model
        provider = self._models.get(name)
        if provider is None:
            raise ValueError(f"model '{name}' not loaded")

        t0 = time.time()
        results = [self(name, inp, **kwargs) for inp in inputs]
        ms = (time.time() - t0) * 1000
        return {
            "results": results,
            "count": len(results),
            "total_ms": round(ms, 2),
            "avg_ms": round(ms / len(results), 2) if results else 0,
        }

    def pipeline(self, model_names: list[str], input_data, **kwargs) -> dict:
        """Chain models — output of one feeds into next."""
        t0 = time.time()
        data = input_data
        trace = []

        for name in model_names:
            provider = self._models.get(name)
            if provider is None:
                raise ValueError(f"model '{name}' not loaded")
            t1 = time.time()
            data = self(name, data, **kwargs)
            ms = (time.time() - t1) * 1000
            trace.append({"model": name, "ms": round(ms, 2)})

        total_ms = (time.time() - t0) * 1000
        return {
            "output": data,
            "trace": trace,
            "total_ms": round(total_ms, 2),
        }

    def profile(self, model_name: str, seq_len: int = 512,
                batch_sizes: list[int] | None = None) -> dict:
        """Benchmark model performance."""
        name = model_name or self._default_model
        provider = self._models.get(name)
        if provider is None:
            raise ValueError(f"model '{name}' not loaded")

        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8]

        profiles = []
        for bs in batch_sizes:
            prompt = "a" * seq_len
            t0 = time.time()
            for _ in range(bs):
                self(name, prompt)
            ms = (time.time() - t0) * 1000
            toks_per_sec = (bs * seq_len) / (ms / 1000) if ms > 0 else 0
            profiles.append({
                "batch_size": bs,
                "seq_len": seq_len,
                "latency_ms": round(ms / bs, 2),
                "tokens_per_sec": round(toks_per_sec, 1),
            })

        return {"model": name, "profiles": profiles}

    def quantize(self, model_name: str, bits: int = 8) -> dict:
        """Compress model weights."""
        name = model_name or self._default_model
        provider = self._models.get(name)
        if provider is None:
            raise ValueError(f"model '{name}' not loaded")

        inner = getattr(provider, "_model", None)
        if inner is None or not hasattr(inner, "_params"):
            return {"error": "model does not support quantization"}

        n_quantized = 0
        for key, arr in inner._params.items():
            if not isinstance(arr, np.ndarray):
                continue
            if not hasattr(inner, "_original_weights"):
                inner._original_weights = {}
            inner._original_weights[key] = arr.copy()
            scale = np.max(np.abs(arr)) / (127 if bits == 8 else 7)
            quant = np.clip(np.round(arr / scale), -(128 if bits == 8 else 8),
                            127 if bits == 8 else 7).astype(np.int8)
            inner._params[key] = quant
            n_quantized += 1

        return {"bits": bits, "params_quantized": n_quantized}

    def checkpoint_save(self, model_name: str, path: str) -> dict:
        """Save model state."""
        name = model_name or self._default_model
        provider = self._models.get(name)
        if provider is None:
            raise ValueError(f"model '{name}' not loaded")

        state = {"name": name, "type": type(provider).__name__}
        self._checkpoints[name] = state
        return {"saved": name, "path": path}

    def checkpoint_load(self, model_name: str, path: str) -> dict:
        """Load model state."""
        name = model_name or self._default_model
        state = self._checkpoints.get(name)
        if state is None:
            return {"error": f"no checkpoint for '{name}'"}
        return {"loaded": name, "state": state}

    def memory(self) -> dict:
        """Memory usage."""
        import psutil
        proc = psutil.Process()
        rss = proc.memory_info().rss / (1024 * 1024)
        model_mem = {}
        for name, provider in self._models.items():
            inner = getattr(provider, "_model", None)
            if inner and hasattr(inner, "_params"):
                mem = sum(
                    p.nbytes for p in inner._params.values()
                    if isinstance(p, np.ndarray)
                ) / (1024 * 1024)
                model_mem[name] = round(mem, 2)
        return {
            "rss_mb": round(rss, 2),
            "models": model_mem,
            "total_model_mb": round(sum(model_mem.values()), 2),
        }

    # ── Ioctl dispatch ────────────────────────────────────────────────────

    def ioctl(self, command: str, *args: Any) -> SyscallResult | dict:
        """All operations go through ioctl."""
        try:
            # Compute ops (direct to TensorDevice)
            if command.startswith("COMPUTE:"):
                op = command[8:]  # Remove "COMPUTE:" prefix
                result = self._compute(op, *args)
                return SyscallResult.ok({"result": result})

            # Device ops
            elif command == "INFO":
                return SyscallResult.ok(self.info())

            elif command == "LOAD":
                path = args[0] if len(args) > 0 else ""
                name = args[1] if len(args) > 1 else ""
                provider = self.load(path, name)
                return SyscallResult.ok({"model": name or path.rsplit("/", 1)[-1]})

            elif command == "UNLOAD":
                name = args[0] if args else ""
                ok = self.unload(name)
                return SyscallResult.ok({"unloaded": ok})

            elif command == "CALL":
                name = args[0] if len(args) > 0 else ""
                inp = args[1] if len(args) > 1 else ""
                result = self(name, inp)
                return SyscallResult.ok(result)

            elif command == "BATCH":
                name = args[0] if len(args) > 0 else ""
                inputs = args[1] if len(args) > 1 else []
                result = self.batch(name, inputs)
                return SyscallResult.ok(result)

            elif command == "PIPE":
                names = args[0] if len(args) > 0 else []
                inp = args[1] if len(args) > 1 else ""
                result = self.pipeline(names, inp)
                return SyscallResult.ok(result)

            elif command == "PROFILE":
                name = args[0] if len(args) > 0 else ""
                seq_len = args[1] if len(args) > 1 else 512
                result = self.profile(name, int(seq_len))
                return SyscallResult.ok(result)

            elif command == "QUANTIZE":
                name = args[0] if len(args) > 0 else ""
                bits = args[1] if len(args) > 1 else 8
                result = self.quantize(name, int(bits))
                return SyscallResult.ok(result)

            elif command == "CHECKPOINT_SAVE":
                name = args[0] if len(args) > 0 else ""
                path = args[1] if len(args) > 1 else ""
                result = self.checkpoint_save(name, path)
                return SyscallResult.ok(result)

            elif command == "CHECKPOINT_LOAD":
                name = args[0] if len(args) > 0 else ""
                path = args[1] if len(args) > 1 else ""
                result = self.checkpoint_load(name, path)
                return SyscallResult.ok(result)

            elif command == "MEMORY":
                return SyscallResult.ok(self.memory())

            else:
                return SyscallResult.fail(f"unknown ioctl: {command}")

        except Exception as e:
            return SyscallResult.fail(f"ioctl {command} failed: {e}")

    # ── Backend loaders ───────────────────────────────────────────────────

    def _load_slnc(self, path: str, name: str, **kwargs):
        backend = kwargs.get("backend")
        if backend == "c":
            try:
                return self._load_c(name, path)
            except Exception:
                return self._load_numpy(name, path)
        elif backend == "numpy":
            return self._load_numpy(name, path)
        else:
            try:
                return self._load_c(name, path)
            except Exception:
                return self._load_numpy(name, path)

    def _load_numpy(self, name: str, path: str):
        from domains.inference.slonet_provider import SlonetChatProvider
        return SlonetChatProvider.from_slnc(path, model_id=name)

    def _load_c(self, name: str, path: str):
        from domains.inference.ct_provider import CTransformProvider
        return CTransformProvider.from_slnc(path, model_id=name)

    def _load_numpy_array(self, path: str, name: str):
        arr = np.load(path, allow_pickle=False)
        return {"type": "numpy", "array": arr, "shape": list(arr.shape)}

    def _load_python(self, path: str, name: str):
        import importlib.util
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return {"type": "python", "module": module}

    def _load_dataset(self, path: str, name: str):
        import pandas as pd
        if path.endswith(".csv"):
            data = pd.read_csv(path)
        elif path.endswith(".json"):
            data = pd.read_json(path)
        else:
            data = pd.read_parquet(path)
        return {"type": "dataset", "data": data, "rows": len(data)}
