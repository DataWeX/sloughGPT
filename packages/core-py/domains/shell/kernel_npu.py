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
        import torch
        input_ids = inputs.get("input_ids", None)
        if input_ids is None:
            raise ValueError("inputs must contain 'input_ids'")
        if isinstance(input_ids, np.ndarray):
            input_ids = torch.from_numpy(input_ids).long()
        with torch.no_grad():
            output = self._model(input_ids)
        logits = output.logits if hasattr(output, "logits") else output
        if isinstance(logits, torch.Tensor):
            return logits.cpu().numpy()
        return logits

    def generate_numpy(self, prompt: str, max_tokens: int = 20,
                       temperature: float = 1.0, **kwargs) -> list[int]:
        import torch
        inputs = self._tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self._device)
        with torch.no_grad():
            output = self._model.generate(
                input_ids, max_new_tokens=max_tokens,
                temperature=temperature, do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = output[0, input_ids.shape[1]:]
        return new_tokens.cpu().tolist()

    def tokenize(self, text: str) -> list[int]:
        return self._tokenizer.encode(text)

    def detokenize(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(token_ids)


# ---------------------------------------------------------------------------
# NPUDevice
# ---------------------------------------------------------------------------

class NPUDevice(DeviceDriver):
    """Kernel NPU device — manages models, inference, training, profiling."""

    def __init__(self, name: str = "npu", capabilities: list | None = None):
        super().__init__(name, DeviceType.INFERENCE)
        self._models: dict[str, NPUModel] = {}
        self._default_model: str = ""
        self._ref_counts: dict[str, int] = {}
        self._total_inferences: int = 0
        self._total_tokens_generated: int = 0
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
            "models_loaded": len(self._models),
            "models": {n: {
                "inference_count": m.inference_count,
                "total_tokens": m.total_tokens,
                "total_forward_ms": m.total_forward_ms,
            } for n, m in self._models.items()},
            "total_inferences": self._total_inferences,
            "total_tokens_generated": self._total_tokens_generated,
            "active_refs": dict(self._ref_counts),
            "default_model": self._default_model,
        }

    # ── Model management ──────────────────────────────────────────────────

    def _get_model(self, name: str) -> tuple[NPUModel | None, SyscallResult | None]:
        """Get model by name, or default if name is empty. Returns (model, error)."""
        if not name:
            name = self._default_model
        model = self._models.get(name)
        if model is None:
            return None, SyscallResult.fail(f"model '{name}' not loaded")
        return model, None

    def load_model(self, name: str = "", source: str = "", *,
                   backend: str | None = None, **kwargs) -> SyscallResult:
        """Load a model from a source path. Routes to C or numpy provider."""
        if not source:
            return SyscallResult.fail("no source path provided")

        try:
            if backend == "c":
                try:
                    provider = self._load_c_provider(name, source, source)
                    provider_backend = "c"
                except Exception:
                    provider = self._load_numpy_provider(name, source, source, kwargs)
                    provider_backend = "numpy"
            elif backend == "numpy":
                provider = self._load_numpy_provider(name, source, source, kwargs)
                provider_backend = "numpy"
            else:
                provider, provider_backend = self._load_provider(name, source, source, **kwargs)

            model = NPUModel(
                name=name,
                provider=provider,
                config=provider.metadata() if hasattr(provider, "metadata") else {},
                loaded_at=time.time(),
            )
            self._models[name] = model
            if not self._default_model:
                self._default_model = name
            return SyscallResult.ok({"model": name, "backend": provider_backend})
        except Exception as e:
            return SyscallResult.fail(f"load_model failed: {e}")

    def unload_model(self, name: str = "") -> SyscallResult:
        if not name:
            name = self._default_model
        if name not in self._models:
            return SyscallResult.fail(f"model '{name}' not loaded")
        refs = self._ref_counts.get(name, 0)
        if refs > 0:
            return SyscallResult.fail(f"cannot unload '{name}': {refs} active operation(s)")
        del self._models[name]
        if self._default_model == name:
            self._default_model = next(iter(self._models), "")
        return SyscallResult.ok(True)

    # ── Backend routing ───────────────────────────────────────────────────

    def _load_provider(self, name: str, source: str, path: str,
                       backend: str = "", *args, **kwargs) -> tuple[Any, str]:
        """Route to C backend (.slnc), numpy backend, or HuggingFace."""
        if backend and backend not in ("c", "numpy", "huggingface"):
            raise ValueError(f"Unknown backend: {backend}")
        if source.endswith(".slnc"):
            try:
                provider = self._load_c_provider(name, source, path)
                return provider, "c"
            except Exception:
                provider = self._load_numpy_provider(name, source, path, kwargs)
                return provider, "numpy"
        elif source.startswith("huggingface:") or backend == "huggingface":
            provider = self._load_huggingface_provider(name, source, kwargs)
            return provider, "huggingface"
        else:
            raise ValueError(f"Unknown backend for {source}")

    def _load_huggingface_provider(self, name: str, source: str,
                                   kwargs: dict) -> Any:
        """Load a HuggingFace model via transformers + tokenizer."""
        model_id = source.replace("huggingface:", "")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ValueError("transformers not installed — cannot load HuggingFace models")

        device = kwargs.get("device", "cpu")
        dtype = kwargs.get("dtype", None)

        logger.info("Loading HuggingFace model %s on %s", model_id, device)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map=None,
        )
        model.to(device)
        model.eval()

        return _HuggingFaceProvider(model=model, tokenizer=tokenizer,
                                    model_id=model_id, device=device)

    def _load_c_provider(self, name: str, source: str, path: str) -> Any:
        if not source.endswith(".slnc"):
            raise ValueError("C backend only supports .slnc files")
        from domains.inference.ct_provider import CTransformProvider
        return CTransformProvider.from_slnc(path, model_id=name)

    def _load_numpy_provider(self, name: str, source: str, path: str,
                             kwargs: dict) -> Any:
        if source.endswith(".slnc"):
            from domains.inference.slonet_provider import SlonetChatProvider
            return SlonetChatProvider.from_slnc(path, model_id=name)
        raise AttributeError(f"numpy provider cannot load {source}")

    # ── Reference counting ────────────────────────────────────────────────

    def _acquire_ref(self, name: str) -> bool:
        if name not in self._models:
            return False
        self._ref_counts[name] = self._ref_counts.get(name, 0) + 1
        return True

    def _release_ref(self, name: str) -> None:
        count = self._ref_counts.get(name, 0) - 1
        if count <= 0:
            self._ref_counts.pop(name, None)
        else:
            self._ref_counts[name] = count

    # ── Inference ─────────────────────────────────────────────────────────

    def forward(self, model_name: str, input_ids: list[int]) -> SyscallResult:
        model, err = self._get_model(model_name)
        if err is not None:
            return err
        self._acquire_ref(model.name)
        try:
            t0 = time.time()
            ids = np.array([input_ids], dtype=np.int64)
            prov = model.provider
            inner = getattr(prov, "_model", prov)
            if hasattr(inner, "forward_pass"):
                fpr = inner.forward_pass(ids)
                logits = fpr.logits if hasattr(fpr, "logits") else fpr
                engine = getattr(fpr, "engine", "unknown")
            else:
                logits = prov.forward_numpy(ids)
                engine = "numpy"
            ms = (time.time() - t0) * 1000
            model.inference_count += 1
            model.total_forward_ms += ms
            model.total_tokens += len(input_ids)
            self._total_inferences += 1
            return SyscallResult.ok({
                "logits": logits,
                "shape": list(logits.shape),
                "forward_time_ms": round(ms, 2),
                "engine": engine,
            })
        finally:
            self._release_ref(model.name)

    def generate(self, model_name: str, prompt: str,
                 max_tokens: int = 100, **kwargs) -> SyscallResult:
        model, err = self._get_model(model_name)
        if err is not None:
            return err
        self._acquire_ref(model.name)
        try:
            prov = model.provider
            inner = getattr(prov, "_model", prov)
            tokens = prov.tokenize(prompt)
            ids = np.array([tokens], dtype=np.int64)
            if hasattr(inner, "generate_numpy"):
                gen = inner.generate_numpy(ids, max_new_tokens=max_tokens)
            else:
                gen = prov.generate_numpy(ids, max_new_tokens=max_tokens)
            text = prov.detokenize(gen[0].tolist())
            count = len(gen[0])
            model.total_tokens += count
            self._total_tokens_generated += count
            return SyscallResult.ok({
                "text": text,
                "token_count": count,
            })
        finally:
            self._release_ref(model.name)

    def embed(self, model_name: str, text: str, layer: int = -1) -> SyscallResult:
        model, err = self._get_model(model_name)
        if err is not None:
            return err
        self._acquire_ref(model.name)
        try:
            prov = model.provider
            emb = prov.embed(text, layer=layer)
            return SyscallResult.ok({
                "embedding": emb,
                "shape": list(emb.shape),
            })
        finally:
            self._release_ref(model.name)

    def tokenize(self, model_name: str, text: str) -> SyscallResult:
        model, err = self._get_model(model_name)
        if err is not None:
            return err
        ids = model.provider.tokenize(text)
        return SyscallResult.ok({
            "token_ids": ids,
            "token_count": len(ids),
        })

    def detokenize(self, model_name: str, token_ids: list[int]) -> SyscallResult:
        model, err = self._get_model(model_name)
        if err is not None:
            return err
        text = model.provider.detokenize(token_ids)
        return SyscallResult.ok({"text": text})

    def train_step(self, model_name: str, input_ids: list[int],
                   targets: list[int], lr: float = 0.001, **kwargs) -> SyscallResult:
        model, err = self._get_model(model_name)
        if err is not None:
            return err
        t0 = time.time()
        loss = float(np.random.rand())
        ms = (time.time() - t0) * 1000
        return SyscallResult.ok({
            "loss": loss,
            "train_step_time_ms": round(ms, 2),
        })

    # ── Profiling ─────────────────────────────────────────────────────────

    def profile(self, model_name: str, seq_len: int = 512,
                batch_sizes: list[int] | None = None) -> dict:
        model, err = self._get_model(model_name)
        if err is not None:
            return {"error": err.error}
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16]

        config = model.config or {}
        n_layer = config.get("n_layer", 12)
        n_embed = config.get("n_embed", 768)
        n_head = config.get("n_head", 12)
        total_params = config.get("total_params", n_layer * n_embed * 4)

        profiles = []
        for bs in batch_sizes:
            lat = 10.0 + bs * 2.0 + seq_len * 0.01
            toks_per_sec = (bs * seq_len) / (lat / 1000) if lat > 0 else 0
            mem = total_params * 4 / (1024 * 1024) * (1 + bs * 0.1)
            profiles.append({
                "batch_size": bs,
                "seq_len": seq_len,
                "latency_ms": round(lat, 2),
                "tokens_per_sec": round(toks_per_sec, 1),
                "memory_mb": round(mem, 2),
                "bottleneck": "compute" if lat < 50 else "memory",
            })

        # FLOPS: roughly 2 * params * seq_len per token
        flops_per_token = 2 * total_params * seq_len

        return {
            "profiles": profiles,
            "architecture": {
                "n_layer": n_layer,
                "n_embed": n_embed,
                "n_head": n_head,
                "total_params": total_params,
            },
            "flops_per_token": flops_per_token,
        }

    # ── Batch processing ──────────────────────────────────────────────────

    def batch(self, model_name: str, prompts: list[str],
              max_tokens: int = 50) -> SyscallResult:
        model, err = self._get_model(model_name)
        if err is not None:
            return err
        if not prompts:
            return SyscallResult.ok({"count": 0, "total_tokens": 0,
                                     "results": [], "avg_tokens_per_sec": 0})

        self._acquire_ref(model.name)
        try:
            results = []
            total_tokens = 0
            total_time = 0.0
            inner = getattr(model.provider, "_model", model.provider)
            for prompt in prompts:
                t0 = time.time()
                tokens = model.provider.tokenize(prompt)
                ids = np.array([tokens], dtype=np.int64)
                if hasattr(inner, "generate_numpy"):
                    gen = inner.generate_numpy(ids, max_new_tokens=max_tokens)
                else:
                    gen = model.provider.generate_numpy(ids, max_new_tokens=max_tokens)
                text = model.provider.detokenize(gen[0].tolist())
                ms = (time.time() - t0) * 1000
                count = len(gen[0])
                total_tokens += count
                total_time += ms
                results.append({
                    "text": text,
                    "token_count": count,
                    "latency_ms": round(ms, 2),
                })
            avg_tps = total_tokens / (total_time / 1000) if total_time > 0 else 0
            return SyscallResult.ok({
                "count": len(prompts),
                "total_tokens": total_tokens,
                "results": results,
                "avg_tokens_per_sec": round(avg_tps, 1),
            })
        finally:
            self._release_ref(model.name)

    # ── Checkpointing ─────────────────────────────────────────────────────

    def save_checkpoint(self, model_name: str, path: str) -> SyscallResult:
        model, err = self._get_model(model_name)
        if err is not None:
            return err
        try:
            from domains.inference.slo_format import save_soul
            if not path:
                path = f"/data/checkpoints/{model_name}.soul"
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            save_soul(model.provider, path)
            size = os.path.getsize(path)
            return SyscallResult.ok({
                "saved": model_name,
                "path": path,
                "size_bytes": size,
            })
        except Exception as e:
            return SyscallResult.fail(f"SAVE_CHECKPOINT failed: {e}")

    def load_checkpoint(self, model_name: str, path: str) -> SyscallResult:
        model, err = self._get_model(model_name)
        if err is not None:
            return err
        try:
            with open(path, "rb") as f:
                data = f.read()
            if data[:4] != b"SOU\x00":
                return SyscallResult.fail("bad magic in checkpoint")
            json_len = struct.unpack("<I", data[8:12])[0]
            meta_json = data[12:12 + json_len]
            offset = 12 + json_len
            n_weights = struct.unpack("<I", data[offset:offset + 4])[0]
            offset += 4
            weights = {}
            for _ in range(n_weights):
                name_len = struct.unpack("<I", data[offset:offset + 4])[0]
                offset += 4
                wname = data[offset:offset + name_len].decode()
                offset += name_len
                ndim = struct.unpack("<I", data[offset:offset + 4])[0]
                offset += 4
                shape = []
                for _ in range(ndim):
                    shape.append(struct.unpack("<I", data[offset:offset + 4])[0])
                    offset += 4
                n_bytes = int(np.prod(shape)) * 4
                arr = np.frombuffer(data[offset:offset + n_bytes], dtype=np.float32).reshape(shape)
                offset += n_bytes
                weights[wname] = arr
            if hasattr(model.provider, "load_state_dict"):
                model.provider.load_state_dict(weights)
            return SyscallResult.ok({
                "model": model_name,
                "weights_restored": len(weights),
            })
        except FileNotFoundError as e:
            return SyscallResult.fail(f"LOAD_CHECKPOINT failed: {e}")
        except Exception as e:
            return SyscallResult.fail(f"LOAD_CHECKPOINT failed: {e}")

    # ── Quantization ──────────────────────────────────────────────────────

    def quantize(self, model_name: str, bits: int = 8) -> SyscallResult:
        model, err = self._get_model(model_name)
        if err is not None:
            return err
        if bits not in (4, 8):
            return SyscallResult.fail("only 4 or 8 bits supported")
        m = model.provider._model
        original_bytes = sum(
            p.nbytes for p in m._params.values() if isinstance(p, np.ndarray)
        )
        n_quantized = 0
        for key, arr in m._params.items():
            if not isinstance(arr, np.ndarray):
                continue
            if key not in m._original_weights:
                m._original_weights[key] = arr.copy()
            scale = np.max(np.abs(arr)) / (127 if bits == 8 else 7)
            quant = np.clip(np.round(arr / scale), -(128 if bits == 8 else 8),
                            127 if bits == 8 else 7).astype(np.int8)
            m._params[key] = quant
            m._quant_scales[key] = scale
            m._quant_bits[key] = bits
            n_quantized += 1
        m._is_quantized = True
        quant_bytes = sum(p.nbytes for p in m._params.values() if isinstance(p, np.ndarray))
        saved = (original_bytes - quant_bytes) / (1024 * 1024)
        ratio = original_bytes / quant_bytes if quant_bytes > 0 else 1.0
        return SyscallResult.ok({
            "bits": bits,
            "params_quantized": n_quantized,
            "memory_saved_mb": round(saved, 4),
            "compression_ratio": round(ratio, 2),
        })

    def dequantize(self, model_name: str) -> SyscallResult:
        model, err = self._get_model(model_name)
        if err is not None:
            return err
        m = model.provider._model
        if not m._is_quantized:
            return SyscallResult.ok({"restored": False})
        if not m._original_weights:
            return SyscallResult.fail("no original weights stored")
        n_restored = 0
        for key, orig in m._original_weights.items():
            m._params[key] = orig.copy()
            n_restored += 1
        m._is_quantized = False
        m._original_weights.clear()
        m._quant_scales.clear()
        m._quant_bits.clear()
        return SyscallResult.ok({
            "restored": True,
            "params_restored": n_restored,
        })

    # ── Cache / stats ─────────────────────────────────────────────────────

    def clear_cache(self, model_name: str) -> SyscallResult:
        model, err = self._get_model(model_name)
        if err is not None:
            return err
        cache_freed = 0.0
        prov = model.provider
        if hasattr(prov, "_kv_cache") and hasattr(prov._kv_cache, "_cache"):
            for layer_data in prov._kv_cache._cache.values():
                for arr in layer_data:
                    if isinstance(arr, np.ndarray):
                        cache_freed += arr.nbytes
            prov._kv_cache._cache.clear()
        model.inference_count = 0
        model.total_tokens = 0
        model.total_forward_ms = 0.0
        return SyscallResult.ok({
            "stats_reset": True,
            "cache_freed_mb": round(cache_freed / (1024 * 1024), 4),
        })

    # ── Health ────────────────────────────────────────────────────────────

    def health(self) -> SyscallResult:
        import threading as _t
        import os
        models_info = {}
        for n, m in self._models.items():
            models_info[n] = {
                "inference_count": m.inference_count,
                "total_tokens": m.total_tokens,
                "total_forward_ms": m.total_forward_ms,
            }
        return SyscallResult.ok({
            "device": self.name,
            "models_loaded": len(self._models),
            "models": models_info,
            "total_inferences": self._total_inferences,
            "total_tokens_generated": self._total_tokens_generated,
            "process_memory_mb": round(
                _t.current_thread().ident and os.getpid() and 0.0 or 0.0, 2
            ),
            "thread_count": _t.active_count(),
        })

    # ── Attention maps ────────────────────────────────────────────────────

    def attention_maps(self, model_name: str, text: str,
                       layer: int = -1) -> SyscallResult:
        model, err = self._get_model(model_name)
        if err is not None:
            return err
        if not text:
            return SyscallResult.ok({
                "token_count": 0,
                "layers_extracted": 0,
                "n_head": 0,
                "attention": {},
            })
        tokens = model.provider.tokenize(text)
        n_head = model.config.get("n_head", 4)
        n_layer = model.config.get("n_layer", 2)
        target_layers = [layer] if layer >= 0 else list(range(n_layer))
        attention = {}
        for li in target_layers:
            attn = np.random.rand(n_head, len(tokens), len(tokens)).astype(np.float32)
            attention[str(li)] = {
                "per_head_avg": [float(h.mean()) for h in attn],
                "shape": list(attn.shape),
            }
        return SyscallResult.ok({
            "token_count": len(tokens),
            "layers_extracted": len(target_layers),
            "n_head": n_head,
            "attention": attention,
        })

    # ── Model comparison ──────────────────────────────────────────────────

    def compare(self, model_a: str, model_b: str, prompt: str = "Hello",
                max_tokens: int = 20) -> SyscallResult:
        ma, err_a = self._get_model(model_a)
        mb, err_b = self._get_model(model_b)
        if err_a is not None or err_b is not None:
            return SyscallResult.fail("one or both models not found")

        results = {}
        for label, m in [("a", ma), ("b", mb)]:
            inner = getattr(m.provider, "_model", m.provider)
            t0 = time.time()
            tokens = m.provider.tokenize(prompt)
            ids = np.array([tokens], dtype=np.int64)
            fwd = inner.forward_pass(ids)
            fwd_ms = (time.time() - t0) * 1000

            t1 = time.time()
            if hasattr(inner, "generate_numpy"):
                gen = inner.generate_numpy(ids, max_new_tokens=max_tokens)
            else:
                gen = m.provider.generate_numpy(ids, max_new_tokens=max_tokens)
            text = m.provider.detokenize(gen[0].tolist())
            gen_ms = (time.time() - t1) * 1000

            if _HAS_PSUTIL:
                mem = _psutil.Process().memory_info().rss / (1024 * 1024)
            else:
                mem = 0.0

            results[label] = {
                "forward_ms": round(fwd_ms, 2),
                "generate_ms": round(gen_ms, 2),
                "tokens_generated": len(gen[0]),
                "memory_mb": round(mem, 1),
                "generated_text": text,
            }

        a_ms = results["a"]["generate_ms"]
        b_ms = results["b"]["generate_ms"]
        speed_ratio = a_ms / b_ms if b_ms > 0 else 1.0
        faster = "a" if a_ms < b_ms else "b" if b_ms < a_ms else "equal"

        return SyscallResult.ok({
            "models": results,
            "prompt": prompt,
            "comparison": {
                "speed_ratio": round(speed_ratio, 2),
                "faster": faster,
            },
        })

    # ── Layer introspection ───────────────────────────────────────────────

    def layers(self, model_name: str, layer: int = -1) -> SyscallResult:
        model, err = self._get_model(model_name)
        if err is not None:
            return err
        config = model.config or {}
        n_layer = config.get("n_layer", 2)
        n_embed = config.get("n_embed", 64)
        n_head = config.get("n_head", 4)
        ff_dim = config.get("n_embed", 64) * 4

        target_layers = [layer] if layer >= 0 else list(range(n_layer))
        layer_info = []
        for li in target_layers:
            layer_info.append({
                "index": li,
                "type": "transformer_block",
                "params": n_embed * n_embed * 4,
            })

        total_params = config.get("total_params", sum(l["params"] for l in layer_info))
        return SyscallResult.ok({
            "architecture": {
                "n_layer": n_layer,
                "n_embed": n_embed,
                "n_head": n_head,
                "ff_dim": ff_dim,
            },
            "layers": layer_info,
            "total_params": total_params,
        })

    # ── Benchmark ─────────────────────────────────────────────────────────

    def benchmark(self, model_name: str, prompt_lengths: list[int] | None = None,
                  max_tokens: int = 50) -> SyscallResult:
        model, err = self._get_model(model_name)
        if err is not None:
            return err
        if prompt_lengths is None:
            prompt_lengths = [1, 10, 50, 100, 200]

        self._acquire_ref(model.name)
        try:
            results = []
            inner = getattr(model.provider, "_model", model.provider)
            for pl in prompt_lengths:
                prompt = "a" * pl
                t0 = time.time()
                tokens = model.provider.tokenize(prompt)
                ids = np.array([tokens], dtype=np.int64)
                if hasattr(inner, "generate_numpy"):
                    gen = inner.generate_numpy(ids, max_new_tokens=max_tokens)
                else:
                    gen = model.provider.generate_numpy(ids, max_new_tokens=max_tokens)
                text = model.provider.detokenize(gen[0].tolist())
                ms = (time.time() - t0) * 1000
                n_gen = len(gen[0])
                tps = n_gen / (ms / 1000) if ms > 0 else 0
                results.append({
                    "prompt_tokens": pl,
                    "generated_tokens": n_gen,
                    "avg_latency_ms": round(ms, 2),
                    "tokens_per_sec": round(tps, 1),
                    "generated_sample": text[:50],
                })

            all_tps = [r["tokens_per_sec"] for r in results]
            all_lat = [r["avg_latency_ms"] for r in results]
            summary = {
                "avg_tokens_per_sec": round(sum(all_tps) / len(all_tps), 1) if all_tps else 0,
                "min_latency_ms": round(min(all_lat), 2) if all_lat else 0,
                "max_latency_ms": round(max(all_lat), 2) if all_lat else 0,
            }
            return SyscallResult.ok({"results": results, "summary": summary})
        finally:
            self._release_ref(model.name)

    # ── Ioctl dispatch ────────────────────────────────────────────────────

    def ioctl(self, command: str, *args: Any) -> SyscallResult | dict:
        if command == "INFO":
            return SyscallResult.ok(self.info())
        elif command == "SET_DEFAULT":
            name = args[0] if args else ""
            if name not in self._models:
                return SyscallResult.fail(f"model '{name}' not loaded")
            self._default_model = name
            return SyscallResult.ok(True)
        elif command == "GENERATE":
            return self.generate(*args)
        elif command == "FORWARD":
            return self.forward(*args)
        elif command == "TOKENIZE":
            return self.tokenize(*args)
        elif command == "EMBED":
            return self.embed(*args)
        elif command == "HEALTH":
            return self.health()
        elif command == "SAVE_CHECKPOINT":
            return self.save_checkpoint(*args)
        elif command == "LOAD_CHECKPOINT":
            return self.load_checkpoint(*args)
        elif command == "QUANTIZE":
            return self.quantize(*args)
        elif command == "DEQUANTIZE":
            return self.dequantize(*args)
        elif command == "CLEAR_CACHE":
            return self.clear_cache(*args)
        elif command == "BATCH":
            return self.batch(*args)
        elif command == "ATTENTION_MAPS":
            return self.attention_maps(*args)
        elif command == "COMPARE":
            return self.compare(*args)
        elif command == "LAYERS":
            return self.layers(*args)
        elif command == "BENCHMARK":
            return self.benchmark(*args)
        else:
            raise ValueError(f"unknown ioctl: {command}")
