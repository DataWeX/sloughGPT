"""
NPU (Neural Processing Unit) device — the one NPU driver.

There is exactly ONE NPU implementation in the shell: this class. The former
standalone ``npu_device.NPUDevice`` is a re-export of it (compat shim), and
the VM reaches it through ``NPUVMDevice`` (vm_devices) — so a model loaded
from any seam (VM ``DEV_CALL``, ``/dev/npu``, CLI, shell) is visible from
every seam: one device, one registry, one ioctl vocabulary.

The model file (.slnc) already contains the graph.
NPUDevice loads/unloads/executes models via ioctl.

Two faces of one contract (see tests/core-py/test_device_contract.py):
  - ``call(method, *args)``  — guest/VM face: lowercase name, returns the
    value, raises DeviceFault on failure (inherited from DeviceDriver).
  - ``ioctl(command, *args)`` — kernel face: uppercase command, returns a
    SyscallResult, never raises.

ioctl commands:
  INFO                         → device info
  LIST_COMMANDS                → all accepted commands
  LOAD   path, name            → load model
  UNLOAD name                  → unload model
  CALL   name, input[, max_tokens] → execute model
  BATCH  name, [inputs]        → execute multiple inputs
  PIPE   [names], input        → chain models (alias of PIPELINE)
  PIPELINE [names], input      → chain models
  PROFILE name, seq_len        → benchmark
  QUANTIZE name, bits          → compress weights
  CHECKPOINT_SAVE name, path   → save state
  CHECKPOINT_LOAD name, path   → load state
  MEMORY                       → memory usage
  COMPUTE op, args...          → passthrough to TensorDevice
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

from .ioctl import IoctlCommand
from .kernel_devices import DeviceDriver, DeviceState, DeviceType
from .kernel_syscall import SyscallResult
from .tensor_device import TensorDevice

logger = logging.getLogger("slo.kernel.npu")


class NPUModel:
    """A loaded model: the device's handle on a model file and its space.

    Wraps one provider plus its config. Calling executes against the loaded
    weights without mutating them — reads, never writes (file-cache style).
    Models are owned by an :class:`NPUDevice`; use ``device.load()`` to
    create one and ``device.model(name)`` to fetch it.
    """

    def __init__(self, name: str, provider: Any, device: NPUDevice, config: dict | None = None):
        self._name = name
        self._provider = provider
        self._device = device
        self._config = dict(config) if config else {"name": name}

    @property
    def name(self) -> str:
        return self._name

    @property
    def provider(self) -> Any:
        return self._provider

    @property
    def config(self) -> dict:
        return dict(self._config)

    def __call__(self, input_data, **kwargs):
        """Execute the model — input goes in, output comes out (no writes)."""
        return self._device.execute(self._name, input_data, **kwargs)

    def __repr__(self) -> str:
        return f"NPUModel(name={self._name!r}, provider={type(self._provider).__name__})"


class NPUDevice(DeviceDriver):
    """NPU hardware device — loads models, executes them.

    One driver, both seams: ``call()`` (guest/VM) and ``ioctl()`` (kernel)
    are the same operation with opposite conventions, translated once in
    DeviceDriver.call / this ioctl dispatch.
    """

    # One operation, two names: the kernel vocabulary historically said PIPE,
    # the standalone one PIPELINE. Callers of either keep working.
    _ALIASES = {"PIPE": "PIPELINE"}

    # The accepted command set never changes at runtime, so it is declared
    # once and copied out — building it per call put a sorted(set) on the
    # hot INFO path of every DEV_CALL.
    _COMMANDS = (
        "BATCH",
        "CALL",
        "CHECKPOINT_LOAD",
        "CHECKPOINT_SAVE",
        "COMPUTE",
        "DETOKENIZE",
        "EMBED",
        "FORWARD",
        "INFO",
        "LIST_COMMANDS",
        "LOAD",
        "MEMORY",
        "PIPE",
        "PIPELINE",
        "PROFILE",
        "QUANTIZE",
        "TOKENIZE",
        "UNLOAD",
    )

    def __init__(self, name: str = "npu", capabilities: list | None = None):
        super().__init__(name, DeviceType.INFERENCE)
        self._models: dict[str, Any] = {}  # name → provider
        self._default_model: str = ""
        self._checkpoints: dict[str, dict] = {}
        self._loaded_models: dict[str, NPUModel] = {}
        self._open_count: int = 0
        self._lock = threading.Lock()
        self._compute = TensorDevice()  # Pure compute engine (COMPUTE passthrough)
        # TensorDevice's op table is static: resolve its command list once
        # instead of re-sorting it on every info() (the per-call build was a
        # ~10x cost on the INFO path once the two faces merged).
        self._compute_ops: tuple[str, ...] = tuple(self._compute.list_commands())

    # ── Device lifecycle ──────────────────────────────────────────────────

    def open(self) -> bool:
        self._open_count += 1
        self._state = DeviceState.OPEN
        return True

    def close(self) -> bool:
        self._state = DeviceState.CLOSED
        return True

    def info(self) -> dict:
        """Device info — the union of both historical faces.

        ``device`` is the kernel-face key the fd/DeviceBus seams read;
        ``name``/``type``/``compute_ops`` are the standalone-face keys the
        /dev seam and list surfaces read. One dict, both vocabularies.
        """
        return {
            "name": self._name,
            "device": self._name,
            "type": "npu",
            "state": self._state.name,
            "compute_ops": list(self._compute_ops),
            "models": len(self._models),
            "names": list(self._models.keys()),
            "default": self._default_model,
            "checkpoints": list(self._checkpoints.keys()),
            "models_loaded": list(self._loaded_models.keys()),
        }

    def list_commands(self) -> list[str]:
        """List all accepted ioctl commands (PIPE included as PIPE alias)."""
        return list(self._COMMANDS)

    # ── Model management ──────────────────────────────────────────────────

    def load(self, path: str = "", name: str = "", **kwargs) -> Any:
        """Load model from file — graph is already inside."""
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
        self._loaded_models[key] = NPUModel(
            key, provider, self, {"name": key, "provider": type(provider).__name__}
        )
        if not self._default_model:
            self._default_model = key
        return provider

    def unload(self, name: str) -> bool:
        """Unload model."""
        if name in self._models:
            del self._models[name]
            self._loaded_models.pop(name, None)
            if self._default_model == name:
                self._default_model = next(iter(self._models), "")
            return True
        return False

    def model(self, name: str = "") -> NPUModel:
        """Fetch a loaded model (builds lazily if the provider is loaded)."""
        key = name or self._default_model
        m = self._loaded_models.get(key)
        if m is None:
            provider = self._models.get(key)
            if provider is None:
                raise ValueError(f"model '{key}' not loaded")
            m = NPUModel(key, provider, self, {"name": key})
            self._loaded_models[key] = m
        return m

    # ── Provider-level ops (reachable from BOTH seams) ───────────────────

    def _provider(self, model_name: str = "") -> Any:
        """Resolve a loaded provider — the default model when name is empty."""
        name = model_name or self._default_model
        provider = self._models.get(name)
        if provider is None:
            raise ValueError(f"model '{name}' not loaded")
        return provider

    def tokenize(self, model_name: str, text: str) -> dict:
        """Text → token ids (VM ``tokenize`` / ioctl ``TOKENIZE``)."""
        ids = self._provider(model_name).tokenize(text)
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        ids = list(ids)
        return {"token_ids": ids, "count": len(ids)}

    def detokenize(self, model_name: str, token_ids) -> dict:
        """Token ids → text (VM ``detokenize`` / ioctl ``DETOKENIZE``)."""
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        elif not isinstance(token_ids, list):
            token_ids = list(token_ids)
        return {"text": self._provider(model_name).detokenize(token_ids)}

    def embed(self, model_name: str, text: str, layer: int = -1) -> dict:
        """Text → embedding vector (VM ``embed`` / ioctl ``EMBED``).

        ``layer`` is only forwarded when the caller actually asked for a
        non-default one, so a provider whose ``embed`` takes just the text
        still works.
        """
        provider = self._provider(model_name)
        emb = provider.embed(text) if layer == -1 else provider.embed(text, layer=layer)
        return {"embedding": emb, "shape": list(np.asarray(emb).shape)}

    def forward(self, model_name: str, input_ids) -> dict:
        """Token ids → logits (VM ``forward`` / ioctl ``FORWARD``)."""
        ids = np.asarray(input_ids, dtype=np.int64)
        if ids.ndim == 1:
            ids = ids[None, :]  # a bare id list is one sequence
        logits = self._forward_logits(self._provider(model_name), ids)
        return {"logits": logits, "shape": list(np.asarray(logits).shape)}

    @staticmethod
    def _forward_logits(provider: Any, ids: np.ndarray) -> Any:
        """Logits from whichever protocol the provider actually speaks.

        Three generations of provider coexist:
          * raw wrappers expose ``forward_numpy``;
          * chat providers (``SloNetChatProvider``) keep the engine private
            behind ``_get_model()`` and expose no numpy forward at all;
          * both engines promise ``forward_pass(input_ids) -> ForwardPassResult``
            — the interface ``forward_pass.py`` exists to describe.

        So: prefer the cheap public numpy call, else resolve the engine and
        use its forward_pass, unwrapping to the bare logits.
        """
        fn = getattr(provider, "forward_numpy", None)
        if callable(fn):
            return fn(ids)
        inner = getattr(provider, "_model", None)
        if inner is None:
            get = getattr(provider, "_get_model", None)
            inner = get() if callable(get) else None
        fp = getattr(inner if inner is not None else provider, "forward_pass", None)
        if callable(fp):
            result = fp(ids)
            return getattr(result, "logits", result)
        raise ValueError(
            f"provider {type(provider).__name__} exposes neither forward_numpy nor forward_pass"
        )

    # ── Execution ─────────────────────────────────────────────────────────

    def __call__(self, *args, model_name: str = "", input_data=None, **kwargs):
        """Execute a model — accepts both historical positional shapes.

        Canonical (kernel face):      ``npu("model", input)``
        Standalone keyword shape:     ``npu(input, model_name="model")``
        Single positional:            ``npu(input)`` → the default model
        """
        if model_name and args:
            # standalone shape: npu(input, model_name="m") — first positional is the input
            return self.execute(model_name, args[0], **kwargs)
        if len(args) == 2:
            return self.execute(args[0], args[1], **kwargs)
        if len(args) == 1:
            return self.execute("", args[0], **kwargs)
        if input_data is not None:
            return self.execute(model_name, input_data, **kwargs)
        raise TypeError("__call__() missing input_data")

    def execute(self, model_name: str, input_data, **kwargs):
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

        # generate — two provider protocols, one result shape.
        t0 = time.time()
        gen_numpy = getattr(provider, "generate_numpy", None)
        if callable(gen_numpy):
            # Low-level protocol: token ids in, token ids out (mocks and
            # raw-model wrappers expose this; ``tokens`` is the sequence
            # length they return, prompt included — as it always was).
            tokens = provider.tokenize(text)
            ids = np.array([tokens], dtype=np.int64)
            gen = gen_numpy(ids, max_new_tokens=max_tokens)
            result = provider.detokenize(gen[0].tolist())
            count = len(gen[0])
        elif callable(getattr(provider, "generate", None)):
            # Public chat-provider protocol (SloNetChatProvider): prompt in,
            # text out. The inner model's generate_numpy is private, so the
            # device must not reach for it — go through generate(). The
            # answer echoes the prompt, so the generated-token count is the
            # continuation alone.
            result = provider.generate(text, max_tokens=max_tokens)
            cont = result[len(text) :] if result.startswith(text) else result
            count = len(provider.tokenize(cont)) if cont else 0
        else:
            raise ValueError(
                f"provider {type(provider).__name__} exposes neither generate_numpy nor generate"
            )
        ms = (time.time() - t0) * 1000
        return {"text": result, "tokens": count, "ms": round(ms, 2)}

    def _execute_tokens(self, provider, tokens, **kwargs):
        """Tokens → model → logits."""
        ids = np.array([tokens], dtype=np.int64) if isinstance(tokens, list) else tokens
        t0 = time.time()
        logits = self._forward_logits(provider, ids)
        ms = (time.time() - t0) * 1000
        return {"logits": logits, "shape": list(np.asarray(logits).shape), "ms": round(ms, 2)}

    # ── Advanced operations ───────────────────────────────────────────────

    def batch(self, model_name: str, inputs: list, **kwargs) -> dict:
        """Execute multiple inputs."""
        name = model_name or self._default_model
        provider = self._models.get(name)
        if provider is None:
            raise ValueError(f"model '{name}' not loaded")

        t0 = time.time()
        results = [self.execute(name, inp, **kwargs) for inp in inputs]
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
            data = self.execute(name, data, **kwargs)
            ms = (time.time() - t1) * 1000
            trace.append({"model": name, "ms": round(ms, 2)})

        total_ms = (time.time() - t0) * 1000
        return {
            "output": data,
            "trace": trace,
            "total_ms": round(total_ms, 2),
        }

    def profile(
        self, model_name: str, seq_len: int = 512, batch_sizes: list[int] | None = None
    ) -> dict:
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
                self.execute(name, prompt)
            ms = (time.time() - t0) * 1000
            toks_per_sec = (bs * seq_len) / (ms / 1000) if ms > 0 else 0
            profiles.append(
                {
                    "batch_size": bs,
                    "seq_len": seq_len,
                    "latency_ms": round(ms / bs, 2),
                    "tokens_per_sec": round(toks_per_sec, 1),
                }
            )

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
            quant = np.clip(
                np.round(arr / scale), -(128 if bits == 8 else 8), 127 if bits == 8 else 7
            ).astype(np.int8)
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
                mem = sum(p.nbytes for p in inner._params.values() if isinstance(p, np.ndarray)) / (
                    1024 * 1024
                )
                model_mem[name] = round(mem, 2)
        return {
            "rss_mb": round(rss, 2),
            "models": model_mem,
            "total_model_mb": round(sum(model_mem.values()), 2),
        }

    # ── Ioctl dispatch ────────────────────────────────────────────────────

    def ioctl(self, command: str | IoctlCommand, *args: Any) -> SyscallResult:
        """All operations go through ioctl. Never raises — reports."""
        try:
            # Normalize: enum members and the PIPE/PIPELINE alias collapse to
            # one command string, so one dispatch table serves both faces.
            cmd = command.value if isinstance(command, IoctlCommand) else command
            cmd = self._ALIASES.get(cmd, cmd)

            if cmd == "INFO":
                return SyscallResult.ok(self.info())

            elif cmd == "LIST_COMMANDS":
                return SyscallResult.ok(self.list_commands())

            elif cmd == "LOAD":
                if len(args) < 1:
                    return SyscallResult.fail("LOAD requires path")
                path = args[0]
                name = args[1] if len(args) > 1 else ""
                self.load(path, name)
                return SyscallResult.ok({"model": name or path.rsplit("/", 1)[-1]})

            elif cmd == "UNLOAD":
                if len(args) < 1:
                    return SyscallResult.fail("UNLOAD requires name")
                ok = self.unload(args[0])
                return SyscallResult.ok({"unloaded": ok})

            elif cmd == "CALL":
                if len(args) < 2:
                    return SyscallResult.fail("CALL requires name and input")
                name, inp = args[0], args[1]
                extra = {"max_tokens": int(args[2])} if len(args) > 2 else {}
                result = self.execute(name, inp, **extra)
                return SyscallResult.ok(result)

            elif cmd == "BATCH":
                if len(args) < 2:
                    return SyscallResult.fail("BATCH requires name and inputs")
                result = self.batch(args[0], args[1])
                return SyscallResult.ok(result)

            elif cmd == "TOKENIZE":
                if len(args) < 2:
                    return SyscallResult.fail("TOKENIZE requires name and text")
                return SyscallResult.ok(self.tokenize(args[0], args[1]))

            elif cmd == "DETOKENIZE":
                if len(args) < 2:
                    return SyscallResult.fail("DETOKENIZE requires name and token ids")
                return SyscallResult.ok(self.detokenize(args[0], args[1]))

            elif cmd == "EMBED":
                if len(args) < 2:
                    return SyscallResult.fail("EMBED requires name and text")
                layer = int(args[2]) if len(args) > 2 else -1
                return SyscallResult.ok(self.embed(args[0], args[1], layer))

            elif cmd == "FORWARD":
                if len(args) < 2:
                    return SyscallResult.fail("FORWARD requires name and input ids")
                return SyscallResult.ok(self.forward(args[0], args[1]))

            elif cmd == "PIPELINE":
                if len(args) < 2:
                    return SyscallResult.fail("PIPELINE requires names and input")
                result = self.pipeline(args[0], args[1])
                return SyscallResult.ok(result)

            elif cmd == "PROFILE":
                if len(args) < 1:
                    return SyscallResult.fail("PROFILE requires name")
                seq_len = args[1] if len(args) > 1 else 512
                result = self.profile(args[0], int(seq_len))
                return SyscallResult.ok(result)

            elif cmd == "QUANTIZE":
                if len(args) < 1:
                    return SyscallResult.fail("QUANTIZE requires name")
                bits = args[1] if len(args) > 1 else 8
                result = self.quantize(args[0], int(bits))
                return SyscallResult.ok(result)

            elif cmd == "CHECKPOINT_SAVE":
                if len(args) < 2:
                    return SyscallResult.fail("CHECKPOINT_SAVE requires name and path")
                result = self.checkpoint_save(args[0], args[1])
                return SyscallResult.ok(result)

            elif cmd == "CHECKPOINT_LOAD":
                if len(args) < 2:
                    return SyscallResult.fail("CHECKPOINT_LOAD requires name and path")
                result = self.checkpoint_load(args[0], args[1])
                return SyscallResult.ok(result)

            elif cmd == "MEMORY":
                return SyscallResult.ok(self.memory())

            elif cmd == "COMPUTE":
                if len(args) < 1:
                    return SyscallResult.fail("COMPUTE requires op")
                op = args[0]
                op_args = args[1:] if len(args) > 1 else ()
                return self._compute.ioctl(op, *op_args)

            else:
                # Both phrase-sets are pinned by tests on the two historical
                # faces: kernel asserts "unknown ioctl", standalone asserts
                # "unknown command". One message, both dialects.
                return SyscallResult.fail(f"unknown command: {command} (unknown ioctl)")

        except Exception as e:
            return SyscallResult.fail(f"ioctl {command} failed: {e}")

    # ── Backend loaders ───────────────────────────────────────────────────

    def _load_slnc(self, path: str, name: str, **kwargs):
        backend = kwargs.get("backend")
        if backend == "c":
            try:
                return self._load_c(name, path)
            except Exception:
                return self._load_numpy(path, name)
        elif backend == "numpy":
            return self._load_numpy(path, name)
        else:
            try:
                return self._load_c(name, path)
            except Exception:
                return self._load_numpy(path, name)

    def _load_numpy(self, path: str, name: str):
        """Load via the numpy-backend provider (file path first, like every
        sibling loader — the old (name, path) order made callers pass the
        model key as the file to open)."""
        from domain.inference._internal.slonet_provider import SlonetChatProvider

        return SlonetChatProvider.from_slnc(path, model_id=name)

    def _load_c(self, name: str, path: str):
        from domain.inference._internal.ct_provider import CTransformProvider

        return CTransformProvider.from_slnc(path, model_id=name)

    def _load_numpy_array(self, path: str, name: str):
        arr = np.load(path, allow_pickle=False)
        return {"type": "numpy", "array": arr, "shape": list(arr.shape)}

    def _load_python(self, path: str, name: str):
        import importlib.util

        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot import a python module from {path}")
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
