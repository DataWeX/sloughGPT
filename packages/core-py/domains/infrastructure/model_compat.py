"""
Universal model compatibility layer.

Wraps any model backend (SloEngine, InferenceEngine, HFModelProvider,
SloTransformer, or raw HF PyTorch) into a single interface.

No new inference engines — wraps existing ones.

Usage:
    from domains.infrastructure.model_compat import wrap_model

    provider = wrap_model(some_engine, model_id="gpt2")
    async for token in provider.chat_stream([{"role": "user", "content": "Hi"}]):
        print(token)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger("man.infrastructure.model_compat")


# ── Model type detection ────────────────────────────────────────────────────


class ModelType(str, Enum):
    SLO_ENGINE = "slo_engine"           # domains.core.soul.SloEngine
    INFERENCE_ENGINE = "inference_engine"  # domains.inference.engine.InferenceEngine
    HF_PROVIDER = "hf_provider"         # domains.models.provider.HFModelProvider
    HF_TORCH = "hf_torch"              # Raw HuggingFace PyTorch model
    SLO_TRANSFORMER = "slonet"         # domains.training.slonet.SloTransformer
    NUMPY_ENGINE = "numpy_engine"       # domains.infrastructure.numpy_engine.NumpyEngine
    EXTERNAL_API = "external"           # External API (OpenAI, etc.)
    UNKNOWN = "unknown"


def detect_model_type(model: Any) -> ModelType:
    """Detect what kind of model/engine we're dealing with."""
    if model is None:
        return ModelType.UNKNOWN

    class_name = type(model).__name__
    module = type(model).__module__ or ""

    # SloEngine (main generation engine)
    if class_name == "SloEngine":
        return ModelType.SLO_ENGINE

    # InferenceEngine
    if class_name == "InferenceEngine":
        return ModelType.INFERENCE_ENGINE

    # HFModelProvider
    if class_name == "HFModelProvider":
        return ModelType.HF_PROVIDER

    # SloTransformer (pure NumPy autograd)
    if class_name == "SloTransformer":
        return ModelType.SLO_TRANSFORMER

    # HuggingFace PyTorch model
    if "transformers" in module or class_name in (
        "GPT2LMHeadModel", "GPT2ForCausalLM",
        "Qwen2ForCausalLM", "LlamaForCausalLM",
        "PreTrainedModel", "PreTrainedLMModel",
    ):
        return ModelType.HF_TORCH

    # SloNet family
    if "slonet" in module.lower() or class_name.startswith("Slo"):
        return ModelType.SLO_TRANSFORMER

    # NumpyEngine (pure NumPy inference)
    if class_name == "NumpyEngine":
        return ModelType.NUMPY_ENGINE

    # Check for PyTorch model indicators
    if hasattr(model, "parameters") and callable(model.parameters):
        try:
            params = list(model.parameters())
            if params and hasattr(params[0], "numpy"):
                return ModelType.HF_TORCH
        except Exception:
            pass

    return ModelType.UNKNOWN


# ── Compatibility interface ──────────────────────────────────────────────────


class UniversalModel(Protocol):
    """Protocol that any wrapped model satisfies."""

    @property
    def model_type(self) -> ModelType: ...

    @property
    def model_id(self) -> str: ...

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> str: ...

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> Iterator: ...

    def tokenize(self, text: str) -> List[int]: ...

    def detokenize(self, ids: List[int]) -> str: ...

    def info(self) -> Dict[str, Any]: ...


# ── Wrappers (thin delegation to existing engines) ──────────────────────────


class _SloEngineWrapper:
    """Wraps SloEngine — the main generation engine."""

    def __init__(self, engine: Any, model_id: str = "slonet"):
        self._engine = engine
        self._model_id = model_id

    @property
    def model_type(self) -> ModelType:
        return ModelType.SLO_ENGINE

    @property
    def model_id(self) -> str:
        return self._model_id

    def generate_text(self, prompt: str, max_new_tokens: int = 100,
                      temperature: float = 0.8, top_k: int = 40, **kw) -> str:
        result = self._engine.generate(
            prompt, max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=top_k,
        )
        if isinstance(result, tuple):
            return result[0]
        return result

    def generate_stream(self, prompt: str, max_new_tokens: int = 100,
                        temperature: float = 0.8, top_k: int = 40, **kw):
        # SloEngine.generate() is sync; stream character-by-character
        text = self.generate_text(prompt, max_new_tokens, temperature, top_k)
        for char in text:
            yield char

    def tokenize(self, text: str) -> List[int]:
        if hasattr(self._engine, "tokenizer") and hasattr(self._engine.tokenizer, "encode"):
            return self._engine.tokenizer.encode(text)
        return list(text.encode("utf-8"))

    def detokenize(self, ids: List[int]) -> str:
        if hasattr(self._engine, "tokenizer") and hasattr(self._engine.tokenizer, "decode"):
            return self._engine.tokenizer.decode(ids)
        return "".join(chr(i) if i < 128 else "?" for i in ids)

    def info(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type.value,
            "model_id": self._model_id,
            "engine": "SloEngine",
            "has_soul": hasattr(self._engine, "soul") and self._engine.soul is not None,
        }


class _InferenceEngineWrapper:
    """Wraps InferenceEngine — lower-level inference."""

    def __init__(self, engine: Any, model_id: str = "inference"):
        self._engine = engine
        self._model_id = model_id

    @property
    def model_type(self) -> ModelType:
        return ModelType.INFERENCE_ENGINE

    @property
    def model_id(self) -> str:
        return self._model_id

    def generate_text(self, prompt: str, max_new_tokens: int = 100,
                      temperature: float = 0.8, top_k: int = 40, **kw) -> str:
        # InferenceEngine.generate_single() is sync
        return self._engine.generate_single(
            prompt, max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=top_k,
        )

    def generate_stream(self, prompt: str, max_new_tokens: int = 100,
                        temperature: float = 0.8, top_k: int = 40, **kw):
        import asyncio
        async def _agen():
            async for token in self._engine.generate_stream(
                prompt, max_new_tokens=max_new_tokens,
                temperature=temperature, top_k=top_k,
            ):
                yield token

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already in async context — return async gen
            return _agen()
        else:
            # Sync context — run and yield
            text = self.generate_text(prompt, max_new_tokens, temperature, top_k)
            for char in text:
                yield char

    def tokenize(self, text: str) -> List[int]:
        if hasattr(self._engine, "tokenizer") and hasattr(self._engine.tokenizer, "encode"):
            return self._engine.tokenizer.encode(text)
        return []

    def detokenize(self, ids: List[int]) -> str:
        if hasattr(self._engine, "tokenizer") and hasattr(self._engine.tokenizer, "decode"):
            return self._engine.tokenizer.decode(ids)
        return ""

    def info(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type.value,
            "model_id": self._model_id,
            "engine": "InferenceEngine",
        }


class _HFProviderWrapper:
    """Wraps HFModelProvider — HuggingFace model provider."""

    def __init__(self, provider: Any, model_id: str = "hf"):
        self._provider = provider
        self._model_id = model_id

    @property
    def model_type(self) -> ModelType:
        return ModelType.HF_PROVIDER

    @property
    def model_id(self) -> str:
        return self._model_id

    def generate_text(self, prompt: str, max_new_tokens: int = 100,
                      temperature: float = 0.8, top_k: int = 40, **kw) -> str:
        import asyncio
        messages = [{"role": "user", "content": prompt}]
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # Can't await in running loop — use sync fallback
                return self._provider._generate_sync(messages, max_new_tokens, temperature)
        except RuntimeError:
            pass
        return asyncio.run(self._provider.chat(messages, max_new_tokens=max_new_tokens, temperature=temperature))

    def generate_stream(self, prompt: str, max_new_tokens: int = 100,
                        temperature: float = 0.8, top_k: int = 40, **kw):
        messages = [{"role": "user", "content": prompt}]

        async def _stream():
            async for token in self._provider.chat_stream(messages, max_new_tokens=max_new_tokens, temperature=temperature):
                yield token

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            return _stream()
        else:
            text = self.generate_text(prompt, max_new_tokens, temperature, top_k)
            for char in text:
                yield char

    def tokenize(self, text: str) -> List[int]:
        if hasattr(self._provider, "tokenizer") and hasattr(self._provider.tokenizer, "encode"):
            return self._provider.tokenizer.encode(text)
        return []

    def detokenize(self, ids: List[int]) -> str:
        if hasattr(self._provider, "tokenizer") and hasattr(self._provider.tokenizer, "decode"):
            return self._provider.tokenizer.decode(ids)
        return ""

    def info(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type.value,
            "model_id": self._model_id,
            "engine": "HFModelProvider",
            "model_server": getattr(self._provider, "_server", None) is not None,
        }


class _HFTorchWrapper:
    """Wraps a raw HuggingFace PyTorch model."""

    def __init__(self, model: Any, tokenizer: Any, model_id: str, device: str = "cpu"):
        self._model = model
        self._tokenizer = tokenizer
        self._model_id = model_id
        self._device = device

    @property
    def model_type(self) -> ModelType:
        return ModelType.HF_TORCH

    @property
    def model_id(self) -> str:
        return self._model_id

    def generate_text(self, prompt: str, max_new_tokens: int = 100,
                      temperature: float = 0.8, top_k: int = 40, **kw) -> str:
        import torch
        input_ids = self._tokenizer.encode(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            output = self._model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                top_k=top_k if top_k > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = output[0][input_ids.shape[-1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def generate_stream(self, prompt: str, max_new_tokens: int = 100,
                        temperature: float = 0.8, top_k: int = 40, **kw):
        from transformers import TextIteratorStreamer
        import threading

        input_ids = self._tokenizer.encode(prompt, return_tensors="pt").to(self._device)
        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)

        def _gen():
            import torch
            with torch.no_grad():
                self._model.generate(
                    input_ids, max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else None,
                    top_k=top_k if top_k > 0 else None,
                    do_sample=temperature > 0,
                    pad_token_id=self._tokenizer.eos_token_id,
                    streamer=streamer,
                )

        thread = threading.Thread(target=_gen, daemon=True)
        thread.start()
        for text in streamer:
            if text:
                yield text

    def tokenize(self, text: str) -> List[int]:
        return self._tokenizer.encode(text)

    def detokenize(self, ids: List[int]) -> str:
        return self._tokenizer.decode(ids)

    def info(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type.value,
            "model_id": self._model_id,
            "device": self._device,
            "params": sum(p.numel() for p in self._model.parameters()),
        }


class _SloTransformerWrapper:
    """Wraps SloTransformer — pure NumPy autograd model.

    SloTransformer.generate() expects token IDs (np.array), not strings.
    This wrapper handles string→token conversion if a tokenizer is provided.
    Without a tokenizer, prompts must be pre-tokenized (pass np.array).
    """

    def __init__(self, model: Any, tokenizer: Any = None, model_id: str = "slonet"):
        self._model = model
        self._tokenizer = tokenizer
        self._model_id = model_id

    @property
    def model_type(self) -> ModelType:
        return ModelType.SLO_TRANSFORMER

    @property
    def model_id(self) -> str:
        return self._model_id

    def _to_ids(self, prompt) -> np.ndarray:
        """Convert prompt to token IDs."""
        if isinstance(prompt, np.ndarray):
            return prompt
        if isinstance(prompt, list):
            return np.array(prompt, dtype=np.int64)
        # String prompt — need tokenizer
        if self._tokenizer and hasattr(self._tokenizer, "encode"):
            ids = self._tokenizer.encode(prompt)
            return np.array(ids, dtype=np.int64)
        # No tokenizer — encode as UTF-8 bytes (last resort)
        return np.array(list(prompt.encode("utf-8")), dtype=np.int64)

    def generate_text(self, prompt: str, max_new_tokens: int = 100,
                      temperature: float = 0.8, top_k: int = 40, **kw) -> str:
        if hasattr(self._model, "generate"):
            ids = self._to_ids(prompt)
            result = self._model.generate(
                ids, max_new_tokens=max_new_tokens,
                temperature=temperature, top_k=top_k,
            )
            # generate() returns string if tokenizer attached, else np.array
            if isinstance(result, str):
                return result
            if self._tokenizer and hasattr(self._tokenizer, "decode"):
                return self._tokenizer.decode(result.tolist())
            return str(result)
        return ""

    def generate_stream(self, prompt: str, max_new_tokens: int = 100,
                        temperature: float = 0.8, top_k: int = 40, **kw):
        text = self.generate_text(prompt, max_new_tokens, temperature, top_k)
        for char in text:
            yield char

    def tokenize(self, text: str) -> List[int]:
        if self._tokenizer and hasattr(self._tokenizer, "encode"):
            return self._tokenizer.encode(text)
        return list(text.encode("utf-8"))

    def detokenize(self, ids: List[int]) -> str:
        if self._tokenizer and hasattr(self._tokenizer, "decode"):
            return self._tokenizer.decode(ids)
        return "".join(chr(i) if i < 128 else "?" for i in ids)

    def info(self) -> Dict[str, Any]:
        info = {
            "model_type": self.model_type.value,
            "model_id": self._model_id,
            "has_tokenizer": self._tokenizer is not None,
        }
        if hasattr(self._model, "vocab_size"):
            info["vocab_size"] = self._model.vocab_size
        if hasattr(self._model, "block_size"):
            info["max_context"] = self._model.block_size
        return info


class _NumpyEngineWrapper:
    """Wrapper for NumpyEngine — delegates to its own generate() with KV cache and streaming."""

    def __init__(self, engine: Any, model_id: str):
        self._engine = engine
        self._model_id = model_id

    @property
    def model_type(self) -> ModelType:
        return ModelType.NUMPY_ENGINE

    def generate_text(self, prompt: str, max_new_tokens: int = 100,
                      temperature: float = 0.7, **kwargs) -> str:
        """Generate text using NumpyEngine with KV cache."""
        return self._engine.generate(prompt, max_new_tokens=max_new_tokens,
                                     temperature=temperature, use_kv_cache=True)

    def generate_stream(self, prompt: str, max_new_tokens: int = 100,
                        temperature: float = 0.7, **kwargs) -> Iterator[str]:
        """Stream tokens using NumpyEngine's async generator."""
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            async def _collect():
                tokens = []
                async for token in self._engine.generate_stream(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                ):
                    tokens.append(token)
                return tokens

            tokens = loop.run_until_complete(_collect())
            for token in tokens:
                yield token
        finally:
            loop.close()

    def info(self) -> Dict[str, Any]:
        return self._engine.info()


# ── Factory ──────────────────────────────────────────────────────────────────


def wrap_model(
    model: Any,
    tokenizer: Any = None,
    model_id: str = "unknown",
    device: str = "cpu",
) -> UniversalModel:
    """
    Wrap any model/engine into the UniversalModel interface.

    Auto-detects type and applies the right wrapper.
    Delegates to existing engines — no new inference logic.
    """
    model_type = detect_model_type(model)
    logger.info("Detected: %s for %s", model_type.value, model_id)

    if model_type == ModelType.SLO_ENGINE:
        return _SloEngineWrapper(model, model_id)
    elif model_type == ModelType.INFERENCE_ENGINE:
        return _InferenceEngineWrapper(model, model_id)
    elif model_type == ModelType.HF_PROVIDER:
        return _HFProviderWrapper(model, model_id)
    elif model_type == ModelType.HF_TORCH:
        return _HFTorchWrapper(model, tokenizer, model_id, device)
    elif model_type == ModelType.SLO_TRANSFORMER:
        return _SloTransformerWrapper(model, tokenizer, model_id)
    elif model_type == ModelType.NUMPY_ENGINE:
        return _NumpyEngineWrapper(model, model_id)
    else:
        logger.warning("Unknown model type %s, using generic wrapper", type(model).__name__)
        return _GenericWrapper(model, tokenizer, model_id)


class _GenericWrapper:
    """Generic wrapper for unknown types — tries common interfaces."""

    def __init__(self, model: Any, tokenizer: Any, model_id: str):
        self._model = model
        self._tokenizer = tokenizer
        self._model_id = model_id

    @property
    def model_type(self) -> ModelType:
        return ModelType.UNKNOWN

    @property
    def model_id(self) -> str:
        return self._model_id

    def generate_text(self, prompt: str, max_new_tokens: int = 100,
                      temperature: float = 0.8, top_k: int = 40, **kw) -> str:
        if hasattr(self._model, "generate"):
            try:
                return self._model.generate(prompt, max_new_tokens=max_new_tokens)
            except TypeError:
                pass
        return ""

    def generate_stream(self, prompt: str, max_new_tokens: int = 100,
                        temperature: float = 0.8, top_k: int = 40, **kw):
        text = self.generate_text(prompt, max_new_tokens, temperature, top_k)
        for char in text:
            yield char

    def tokenize(self, text: str) -> List[int]:
        if self._tokenizer and hasattr(self._tokenizer, "encode"):
            return self._tokenizer.encode(text)
        return list(text.encode("utf-8"))

    def detokenize(self, ids: List[int]) -> str:
        if self._tokenizer and hasattr(self._tokenizer, "decode"):
            return self._tokenizer.decode(ids)
        return "".join(chr(i) if i < 128 else "?" for i in ids)

    def info(self) -> Dict[str, Any]:
        return {"model_type": "unknown", "model_id": self._model_id}
