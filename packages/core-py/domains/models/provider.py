"""
ModelProvider Protocol — plug-and-play interface for any model backend.

A ModelProvider is anything that can:
- Chat (messages in → text out, streaming or blocking)
- Embed (text → vector)
- Report its identity and capabilities

No inheritance required — structural typing (Protocol).
Any class with the right methods IS a ModelProvider.

ProviderRouter chains message processors + a text provider:
  messages → processor 1 → processor 2 → ... → text provider → tokens

Each MessageProcessor transforms messages before they reach the text provider.
Processors are composable, swappable, and testable independently.

HFModelProvider wraps a HuggingFace model+tokenizer as a ModelProvider.
"""

import asyncio
from typing import Protocol, AsyncIterator, Optional, List, Dict, Any, runtime_checkable
from dataclasses import dataclass, field
import logging

from domains.inference.prompt_formatter import PromptFormatter
from domains.training.slonet_compat import torch

logger = logging.getLogger("man.models.provider")


@dataclass
class ModelCapabilities:
    """What a model can do."""
    chat: bool = False
    streaming: bool = False
    embedding: bool = False
    vision: bool = False
    functions: bool = False


ChatMessage = Dict[str, str]  # {"role": "user"|"assistant"|"system", "content": str}


@runtime_checkable
class ModelProvider(Protocol):
    """Standard interface every model backend implements.

    Usage:
        provider: ModelProvider = LocalSoulNetModel()
        async for chunk in provider.chat_stream([{"role": "user", "content": "Hello"}]):
            print(chunk)
    """

    @property
    def model_id(self) -> str:
        """Unique identifier for this model (e.g. 'gpt2', 'llama-7b')."""
        ...

    @property
    def capabilities(self) -> ModelCapabilities:
        """What this model supports."""
        ...

    async def chat_stream(
        self,
        messages: List[ChatMessage],
        max_tokens: int = 512,
        temperature: float = 0.8,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream a chat response token by token.

        Args:
            messages: Conversation history [{"role": "...", "content": "..."}]
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Yields:
            Each token/partial as it's generated.
        """
        ...
        yield  # pragma: no cover

    async def chat(
        self,
        messages: List[ChatMessage],
        max_tokens: int = 512,
        temperature: float = 0.8,
        **kwargs,
    ) -> str:
        """Blocking chat — returns complete response."""
        chunks = []
        async for chunk in self.chat_stream(messages, max_tokens, temperature, **kwargs):
            chunks.append(chunk)
        return "".join(chunks)

    def embed(self, text: str) -> List[float]:
        """Convert text to embedding vector.

        Args:
            text: Input text

        Returns:
            Embedding vector as float list
        """
        ...

    @property
    def metadata(self) -> Dict[str, Any]:
        """Arbitrary metadata about this model (size, quantization, etc.)."""
        return {}


# =============================================================================
# Convenience — wrap any provider into a single call
# =============================================================================

_providers: Dict[str, ModelProvider] = {}


def register_provider(name: str, provider: ModelProvider) -> None:
    """Register a model provider by name."""
    _providers[name] = provider


def get_provider(name: str) -> Optional[ModelProvider]:
    """Get a registered provider by name."""
    return _providers.get(name)


def list_providers() -> List[str]:
    """List all registered provider names."""
    return list(_providers.keys())


# =============================================================================
# MessageProcessor — transform messages before they reach the text provider
# =============================================================================

class MessageProcessor(Protocol):
    """Transforms a message list before generation.

    Processors form a pipeline: each one receives the output of the previous.
    Return the (possibly modified) message list for the next processor.

    Usage:
        class MyProcessor:
            async def process(self, messages: list) -> list:
                for m in messages:
                    m["content"] = m["content"].upper()
                return messages
    """

    async def process(self, messages: list) -> list:
        """Transform messages in-place or return a new list."""
        ...


class VisionProcessor:
    """Extracts images from messages, captions via vision provider, injects as text.

    Runs before the text provider so the LLM sees image descriptions as text.
    """

    def __init__(self, provider_name: str = "multimodal"):
        self._provider_name = provider_name

    def _extract_images(self, messages: list) -> list:
        images = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        images.append(url)
            if isinstance(content, str):
                import re
                for m in re.finditer(r'data:image/\w+;base64,([^"]+)', content):
                    images.append(m.group(0))
        return images

    def _ensure_provider(self) -> bool:
        if get_provider(self._provider_name) is not None:
            return True
        try:
            from domains.multimodal.manager import get_multimodal_manager, initialize_multimodal
            initialize_multimodal()
            mgr = get_multimodal_manager()
            if hasattr(mgr, 'initialize'):
                mgr.initialize(vision_model="slonet")
            return get_provider(self._provider_name) is not None
        except Exception as e:
            logger.warning(f"Failed to init multimodal: {e}")
            return False

    async def _caption(self, img_data: str) -> str:
        vision = get_provider(self._provider_name)
        if vision is not None:
            try:
                msg = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": img_data}}]}]
                text = ""
                async for token in vision.chat_stream(msg, max_tokens=30, temperature=0.8):
                    text += token
                if text.strip():
                    return text.strip()
            except Exception:
                pass
        try:
            import base64
            clean = img_data.split(",")[1] if "," in img_data else img_data
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(base64.b64decode(clean))).convert("RGB")
            from domains.multimodal.manager import get_multimodal_manager
            mgr = get_multimodal_manager()
            return mgr.caption_image(img).text
        except Exception as e:
            logger.warning(f"Caption failed: {e}")
            return "[image]"

    async def process(self, messages: list) -> list:
        images = self._extract_images(messages)
        if not images:
            return messages
        if not self._ensure_provider():
            logger.warning("Vision provider unavailable, keeping images as-is")
            return messages

        captions = []
        for i, img in enumerate(images):
            cap = await self._caption(img)
            captions.append(cap)
            logger.info(f"Image {i+1} captioned: {cap[:60]}")

        caption_block = "\n".join(f"[Image: {c}]" for c in captions)

        result = []
        injected = False
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                content = "\n".join(text_parts)

            if not injected and msg.get("role") == "user":
                content = f"{content}\n{caption_block}"
                injected = True

            result.append({"role": msg["role"], "content": content})

        return result


class KnowledgeProcessor:
    """Injects knowledge/context into system messages."""

    def __init__(self, knowledge: Optional[List[str]] = None):
        self._knowledge = knowledge or []

    def set_knowledge(self, knowledge: List[str]) -> None:
        self._knowledge = knowledge

    async def process(self, messages: list) -> list:
        if not self._knowledge:
            return messages
        k_text = "\n".join(f"- {k}" for k in self._knowledge)
        return [{"role": "system", "content": f"Knowledge context:\n{k_text}"}] + messages


# =============================================================================
# SloTransformerProvider — wrap SloTransformer (pure NumPy) as ModelProvider
# =============================================================================

class SloTransformerProvider:
    """Wraps a pure NumPy ``SloTransformer`` as a ``ModelProvider``.

    Uses character-level vocab (stoi/itos) for encoding/decoding just like
    the auto-train pipeline. Works with any ``.slo`` checkpoint exported
    by the auto-train router.

    Unlike ``SloNetProvider`` (which wraps PyTorch ``SloughGPTModel``),
    this has zero PyTorch dependency — pure NumPy via SloNet autograd.

    Args:
        model: A loaded ``SloTransformer`` instance
        stoi: char-to-index dict
        itos: index-to-char dict
        model_id_str: Optional name (defaults to ``soultransformer``)
    """

    def __init__(self, model, stoi: dict, itos: dict, model_id_str: str = "soultransformer"):
        self._model = model
        self._stoi = stoi
        # itos may have string keys from JSON — normalise to int
        self._itos = {int(k): v for k, v in itos.items()}
        self._model_id_str = model_id_str
        self._bos = stoi.get(" ", 0)
        self._eos = stoi.get("<PAD>", 0)

    @property
    def model_id(self) -> str:
        return self._model_id_str

    @property
    def capabilities(self):
        return ModelCapabilities(chat=True, streaming=True, embedding=False, vision=False)

    def _encode(self, text: str) -> list:
        return [self._stoi.get(c, self._bos) for c in text.lower()]

    def _decode(self, ids: list) -> str:
        return "".join(self._itos.get(i, "?") for i in ids)

    def _messages_to_prompt(self, messages: list) -> str:
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        return "\n".join(parts)

    async def chat_stream(
        self,
        messages: list,
        max_tokens: int = 512,
        temperature: float = 0.8,
        cancel_event=None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream tokens from the SloTransformer, chunked for responsiveness.

        Generates ``min(8, max_tokens)`` tokens at a time via
        ``run_in_executor`` to avoid blocking the event loop.
        """
        prompt = self._messages_to_prompt(messages)
        input_ids = self._encode(prompt)
        if not input_ids:
            input_ids = [self._bos]
        import numpy as np
        inp = np.array([input_ids], dtype=np.int64)
        import asyncio
        loop = asyncio.get_event_loop()
        chunk_size = min(8, max_tokens)
        generated = 0
        while generated < max_tokens:
            if cancel_event is not None and cancel_event.is_set():
                break
            to_gen = min(chunk_size, max_tokens - generated)

            def _gen():
                return self._model.generate(inp, max_new_tokens=to_gen, temperature=temperature)

            try:
                out = await loop.run_in_executor(None, _gen)
            except Exception as e:
                logger.warning(f"SloTransformer generation error: {e}")
                break
            if out is None:
                break
            out_ids = out.data.flatten().tolist()
            text = self._decode(out_ids)
            if text:
                yield text
            generated += len(out_ids)
            inp = np.array([input_ids + out_ids], dtype=np.int64)
            if self._eos in out_ids:
                break

    async def chat(self, messages, max_tokens=512, temperature=0.8, **kwargs):
        chunks = []
        async for chunk in self.chat_stream(messages, max_tokens, temperature, **kwargs):
            chunks.append(chunk)
        return "".join(chunks)

    def embed(self, text: str) -> list:
        return []

    @property
    def metadata(self) -> dict:
        return {
            "model_id": self._model_id_str,
            "vocab_size": len(self._stoi),
            "type": "soultransformer",
            "n_layer": self._model.n_layer,
            "n_embed": self._model.n_embed,
            "n_head": self._model.n_head,
        }

    @classmethod
    def load_from_sou(cls, path: str, model_id_str: str = "") -> "SloTransformerProvider":
        """Load a trained SloTransformer from a ``.slo`` checkpoint.

        Handles both auto-train checkpoints (LSTM-based) and SloTransformer
        checkpoints. Auto-detects architecture from the state dict keys.

        Args:
            path: Path to ``.slo`` checkpoint file
            model_id_str: Optional provider name (defaults to filename stem)

        Returns:
            SloTransformerProvider ready to use in the provider pipeline.
        """
        from domains.inference import load_soul

        soul, sd = load_soul(path)
        if isinstance(sd, dict) and "tok_emb.weight" not in sd:
            sd = sd.get("weights", sd)
            if not isinstance(sd, dict):
                sd = sd.state_dict() if hasattr(sd, "state_dict") else {}

        # Build default char vocab
        chars = ["<PAD>", "<UNK>"] + list(" abcdefghijklmnopqrstuvwxyz0123456789.,!?-'")
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}

        # Override vocab from soul metadata if available
        meta_vocab = getattr(soul, "vocab_size", None) or (soul.metadata or {}).get("vocab_size")
        if meta_vocab and meta_vocab > len(chars):
            extra = ["_"] * (meta_vocab - len(chars))
            chars = chars + extra
            stoi = {ch: i for i, ch in enumerate(chars)}
            itos = {i: ch for i, ch in enumerate(chars)}

        vocab = sd["tok_emb.weight"].shape[0]
        n_embed = sd["tok_emb.weight"].shape[1]

        # Detect n_layer from state dict keys
        n_layer = 1
        for key in sd:
            if key.startswith("blocks."):
                idx = int(key.split(".")[1])
                n_layer = max(n_layer, idx + 1)

        # Detect n_head
        n_head = 8
        q_w = sd.get("blocks.0.attn.q_proj.weight")
        if q_w is not None:
            head_dim = n_embed // 8
            if head_dim > 0:
                detected = q_w.shape[0] // head_dim
                if detected >= 1:
                    n_head = detected

        from domains.training.slonet import SloTransformer

        model = SloTransformer(
            vocab_size=vocab,
            n_embed=n_embed,
            n_layer=n_layer,
            n_head=n_head,
            dropout=0.0,
            tie_weights=False,
        )
        model.load_state_dict(sd, strict=False)

        logger.info(
            "Loaded SloTransformer from %s (vocab=%d, "
            "n_embed=%d, n_layer=%d, n_head=%d)",
            path, vocab, n_embed, n_layer, n_head,
        )
        return cls(model, stoi, itos, model_id_str=model_id_str)


# =============================================================================
# HFModelProvider — wrap a HuggingFace model+tokenizer as a ModelProvider
# =============================================================================

class HFModelProvider:
    """Wraps a HuggingFace model+tokenizer as a ModelProvider.

    .. deprecated::
        Use ``NumpyEngine`` (no PyTorch) or ``InferenceEngine`` (full features)
        instead. This engine will be removed in a future version.

    Args:
        model: HuggingFace PreTrainedModel
        tokenizer: HuggingFace PreTrainedTokenizer
        model_id_str: Optional name (defaults to 'hf-model')
        model_server: Optional ModelServer for lifecycle-managed generation.
            If provided, the provider delegates ``model.generate()`` through the
            server's semaphore, circuit breaker, timeout, and lifecycle hooks.
    """

    def __init__(self, model, tokenizer, model_id_str: str = "hf-model",
                 model_server=None):
        self._model = model
        self._tokenizer = tokenizer
        self._model_id_str = model_id_str
        self._formatter = PromptFormatter(tokenizer=tokenizer)
        self._server = model_server
        self._quant_engine = None

    def quantization_report(self) -> dict:
        """Get quantization error report (if quantized).

        Returns:
            Dict with per-tensor error metrics and aggregate summary.
            Empty dict if model was not quantized.
        """
        if self._quant_engine is None:
            return {"quantized": False}
        summary = self._quant_engine.summary()
        return {
            "quantized": True,
            "bits": summary.get("bits", 0),
            "mode": summary.get("mode", "symmetric"),
            "summary": summary,
            "per_tensor": self._quant_engine.error_report(),
        }

    @property
    def model_id(self) -> str:
        return self._model_id_str

    @property
    def capabilities(self):
        return ModelCapabilities(chat=True, streaming=True, embedding=False, vision=False)

    async def chat_stream(
        self,
        messages: List[ChatMessage],
        max_tokens: int = 512,
        temperature: float = 0.8,
        cancel_event=None,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        prompt = self._formatter.messages_to_prompt(messages)

        if self._server is not None:
            is_first = True
            async for text in self._server.generate_stream(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=kwargs.pop("top_p", 0.9),
                cancel_event=cancel_event,
                session_id=session_id,
                **kwargs,
            ):
                cleaned = self._formatter.clean_chunk(text, first=is_first)
                is_first = False
                if cleaned:
                    yield cleaned
            return

        # Fallback: direct generation without ModelServer
        from transformers import TextIteratorStreamer
        from threading import Thread
        import queue
        import asyncio
        import gc

        inputs = self._tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]

        if hasattr(self._model, "device"):
            input_ids = input_ids.to(self._model.device)

        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=kwargs.pop("top_p", 0.9),
            top_k=kwargs.pop("top_k", 50),
            repetition_penalty=kwargs.pop("repetition_penalty", 1.2),
            pad_token_id=self._tokenizer.eos_token_id,
            streamer=streamer,
        )
        gen_kwargs.update(kwargs)

        _error: list[Exception] = []

        def _generate():
            try:
                with torch.no_grad():
                    self._model.generate(**gen_kwargs)  # noqa: F821
            except Exception as e:
                _error.append(e)

        thread = Thread(target=_generate)
        thread.start()
        is_first = True

        while thread.is_alive() or not streamer.text_queue.empty():
            if _error:
                raise _error[0]
            try:
                text = streamer.text_queue.get(timeout=0.01)
            except queue.Empty:
                await asyncio.sleep(0)
                continue
            if text == streamer.stop_signal:
                break
            if text:
                yield self._formatter.clean_chunk(text, first=is_first)
                is_first = False

        thread.join(timeout=15)

        try:
            del input_ids, gen_kwargs, streamer
        except Exception:
            pass

    async def chat(
        self,
        messages: List[ChatMessage],
        max_tokens: int = 512,
        temperature: float = 0.8,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        if self._server is not None:
            prompt = self._formatter.messages_to_prompt(messages)
            result = await self._server.generate(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=kwargs.pop("top_p", 0.9),
                session_id=session_id,
                **kwargs,
            )
            return result.get("text", "")
        # Fallback: collect from streaming
        chunks = []
        async for chunk in self.chat_stream(messages, max_tokens, temperature, session_id=session_id, **kwargs):
            chunks.append(chunk)
        return "".join(chunks)

    def embed(self, text: str) -> List[float]:
        return []

    @property
    def metadata(self) -> Dict[str, Any]:
        try:
            total = sum(p.numel() for p in self._model.parameters())
            return {
                "model_id": self._model_id_str,
                "parameters": total,
                "device": str(self._model.device),
            }
        except Exception:
            return {"model_id": self._model_id_str}


# =============================================================================
# InferenceEngineProvider — wraps InferenceEngine as a ModelProvider
# =============================================================================


class InferenceEngineProvider:
    """ModelProvider backed by InferenceEngine (KV cache, streaming, sampling).

    When a ``model_server`` is provided, delegates lifecycle management
    (semaphore, circuit breaker, timeout, warmup) to the ``ModelServer``,
    keeping this class as a thin message-formatting wrapper — same pattern
    as ``HFModelProvider``.

    Without a ``model_server``, falls back to direct ``InferenceEngine``
    generation (no lifecycle management).

    Args:
        engine: InferenceEngine instance.
        tokenizer: Optional tokenizer for chat template formatting.
        model_id_str: Identifier for this provider.
        model_server: Optional ``ModelServer`` for lifecycle management.
    """

    def __init__(
        self,
        engine,
        tokenizer=None,
        model_id_str: str = "inference-engine",
        model_server=None,
    ):
        self._engine = engine
        self._tokenizer = tokenizer
        self._model_id_str = model_id_str
        self._formatter = PromptFormatter(tokenizer=tokenizer)
        self._server = model_server

    @property
    def model_id(self) -> str:
        return self._model_id_str

    @property
    def capabilities(self):
        return ModelCapabilities(chat=True, streaming=True, embedding=False, vision=False)

    async def chat_stream(
        self,
        messages: List[ChatMessage],
        max_tokens: int = 512,
        temperature: float = 0.8,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        prompt = self._formatter.messages_to_prompt(messages)
        cancel_event = kwargs.get("cancel_event")

        if self._server is not None:
            is_first = True
            async for text in self._server.generate_stream(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=kwargs.get("top_p", 0.9),
                top_k=kwargs.get("top_k", 40),
                cancel_event=cancel_event,
                session_id=session_id,
            ):
                cleaned = self._formatter.clean_chunk(text, first=is_first)
                is_first = False
                if cleaned:
                    yield cleaned
            return

        # Fallback: direct engine generation (no lifecycle management)
        is_first = True
        async for token in self._engine.generate_stream(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=kwargs.get("top_p", 0.9),
            top_k=kwargs.get("top_k", 40),
            cancel_event=cancel_event,
        ):
            cleaned = self._formatter.clean_chunk(token, first=is_first)
            is_first = False
            if cleaned:
                yield cleaned

    async def chat(
        self,
        messages: List[ChatMessage],
        max_tokens: int = 512,
        temperature: float = 0.8,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        import time as _time
        t0 = _time.monotonic()
        if self._server is not None:
            prompt = self._formatter.messages_to_prompt(messages)
            result = await self._server.generate(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=kwargs.get("top_p", 0.9),
                top_k=kwargs.get("top_k", 40),
                session_id=session_id,
            )
            text = result.get("text", "")
            try:
                from domains.infrastructure.metrics import get_metrics_collector
                get_metrics_collector().record_inference(_time.monotonic() - t0, tokens=result.get("tokens_generated", 0))
            except Exception:
                pass
            return text

        chunks = []
        async for chunk in self.chat_stream(messages, max_tokens, temperature, session_id=session_id, **kwargs):
            chunks.append(chunk)
        try:
            from domains.infrastructure.metrics import get_metrics_collector
            get_metrics_collector().record_inference(_time.monotonic() - t0, tokens=len(chunks))
        except Exception:
            pass
        return "".join(chunks)

    def embed(self, text: str) -> List[float]:
        return []

    def get_metrics_snapshot(self) -> dict:
        if self._server is not None:
            return self._server.get_metrics_snapshot()
        return {
            "model_id": self._model_id_str,
            "status": "ready",
            "device": "cpu",
        }

    @property
    def metadata(self) -> Dict[str, Any]:
        try:
            stats = self._engine.get_stats()
            base = self.get_metrics_snapshot()
            base["stats"] = stats
            return base
        except Exception:
            return self.get_metrics_snapshot()


# =============================================================================
# setup_providers — wire up default vision→text routing
# =============================================================================


def discover_checkpoints(checkpoint_dir: str = "models/auto-training") -> List[Dict[str, Any]]:
    """Scan a directory for trainable SloNet checkpoints.

    Returns list of dicts with ``path``, ``name``, ``steps``, ``vocab_size``.
    Only includes checkpoints with valid stoi/itos vocab.
    """
    import json
    from pathlib import Path
    results = []
    base = Path(checkpoint_dir)
    if not base.exists():
        return results
    for pt in sorted(base.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            ckpt = torch.load(str(pt), map_location="cpu", weights_only=False)
            tok = ckpt.get("tokenizer", {})
            stoi = tok.get("stoi") or ckpt.get("stoi")
            if stoi is None:
                continue
            info = {
                "path": str(pt),
                "name": pt.stem,
                "vocab_size": len(stoi),
                "steps": ckpt.get("total_steps", 0),
            }
            results.append(info)
        except Exception:
            continue
    return results


def setup_providers(hf_model=None, hf_tokenizer=None, hf_model_id: str = "gpt2",
                    inference_engine=None,
                    slonet_hf_id: Optional[str] = None,
                    model_registry=None) -> None:
    """Register all model providers and wire up the default router.

    Call after the HF model is loaded. Registers:
    - ``"hf-default"``: ``HFModelProvider`` wrapping the loaded model (if provided)
    - ``"slonet-native"``: ``SloNetChatProvider`` (pure NumPy SloTransformer, no PyTorch runtime)
      (if ``slonet_hf_id`` given)
    - ``"inference-engine"``: ``InferenceEngineProvider`` wrapping the InferenceEngine
      (if ``inference_engine`` given)
    - ``"default"``: ``ProviderRouter`` with VisionProcessor + text provider

    The text provider is chosen in this priority order:
    1. ``slonet-native`` (pure NumPy SloTransformer, no PyTorch at inference time)
    2. ``inference-engine`` (InferenceEngine with KV cache)
    3. ``hf-default`` (HuggingFace model)

    Args:
        model_registry: Optional ``ModelRegistry``. If provided, the ``hf-default``
            provider uses the registered ModelServer for lifecycle-managed generation
            (semaphore, timeout, circuit breaker, pre/post hooks).
    """
    if hf_model is not None and hf_tokenizer is not None:
        # Look up ModelServer from registry
        model_server = model_registry.get(hf_model_id) if model_registry else None
        hf_provider = HFModelProvider(
            hf_model, hf_tokenizer,
            model_id_str=hf_model_id,
            model_server=model_server,
        )
        register_provider("hf-default", hf_provider)
        logger.info("Registered hf-default provider: %s (model_server=%s)", hf_model_id, model_server is not None)

    text_provider_name = "hf-default" if hf_model is not None else None

    if slonet_hf_id:
        try:
            from domains.inference.slonet_provider import SloNetChatProvider
            slonet_provider = SloNetChatProvider(hf_model_id=slonet_hf_id)
            register_provider("slonet-native", slonet_provider)
            text_provider_name = "slonet-native"
            logger.info("Registered slonet-native provider: %s", slonet_hf_id)
        except Exception as e:
            logger.warning("Failed to load slonet-native provider %s: %s", slonet_hf_id, e)

    if inference_engine is not None:
        try:
            # Look up ModelServer from registry (same pattern as hf-default)
            model_server = model_registry.get("inference-engine") if model_registry else None
            ie_provider = InferenceEngineProvider(
                inference_engine,
                tokenizer=hf_tokenizer,
                model_id_str="inference-engine",
                model_server=model_server,
            )
            register_provider("inference-engine", ie_provider)
            if text_provider_name is None:
                text_provider_name = "inference-engine"
            # Register in ModelRegistry for lifecycle visibility
            if model_registry is not None:
                model_registry.register_engine(
                    "inference-engine", ie_provider, make_default=False,
                )
                logger.info("Registered inference-engine in ModelRegistry")
            logger.info("Registered inference-engine provider")
        except Exception as e:
            logger.warning("Failed to register inference-engine provider: %s", e)

    router = ProviderRouter()
    router.add_processor(VisionProcessor("multimodal"))
    if text_provider_name:
        router.set_text_provider(text_provider_name)

    # Don't override SloNet if already active as "default" — auto_train registers
    # SloTransformerProvider directly as "default"; replacing it would cause
    # chat to silently fall back to HF and produce empty responses.
    existing = _providers.get("default")
    _is_slonet = existing is not None and type(existing).__name__ in ("SloTransformerProvider", "SloNetChatProvider")
    if not _is_slonet:
        register_provider("default", router)
        logger.info("Registered default provider router (processors=%s, text=%s)",
                    [type(p).__name__ for p in router._processors],
                    text_provider_name)
    else:
        logger.info("SloNet provider active as default — skipping ProviderRouter override")


# =============================================================================
# Tool use — let the text model call other providers mid-conversation
# =============================================================================

@dataclass
class ToolDef:
    """A tool the text model can call during generation.

    Args:
        name: Tool name (e.g. ``describe_image``). Matched in output.
        provider_name: Provider to call when this tool is invoked.
        description: Prompt fragment telling the model how to use it.
    """
    name: str
    provider_name: str
    description: str = ""

    TOOL_RE = r'\[\[TOOL:\s*(\w+)\s*\]\]\s*(\S+)'

    @classmethod
    def parse_call(cls, text: str) -> Optional[tuple[str, str]]:
        """Parse ``[[TOOL: name]] arg`` from text. Returns (name, arg) or None."""
        import re
        m = re.search(cls.TOOL_RE, text)
        if m:
            return m.group(1), m.group(2)
        return None


_BUILTIN_TOOLS = [
    ToolDef(
        name="describe_image",
        provider_name="multimodal",
        description=(
            "To see an image, output: [[TOOL: describe_image]] <base64_image_data>\n"
            "I will describe it and you can continue."
        ),
    ),
]


class ToolUseProcessor:
    """Injects tool-use instructions into the system prompt.

    Tells the model it can call tools by outputting ``[[TOOL: name]] arg``.
    Works with any text model — no special function-calling support needed.
    """

    def __init__(self, tools: Optional[List[ToolDef]] = None):
        self._tools = tools or _BUILTIN_TOOLS

    async def process(self, messages: list) -> list:
        tool_descriptions = "\n".join(
            f"- {t.description}" for t in self._tools
        )
        tool_prompt = (
            "You have access to these tools:\n"
            f"{tool_descriptions}\n"
        )

        has_system = any(m.get("role") == "system" for m in messages)
        if has_system:
            for m in messages:
                if m.get("role") == "system":
                    m["content"] = f"{m['content']}\n\n{tool_prompt}"
                    break
        else:
            messages.insert(0, {"role": "system", "content": tool_prompt})
        return messages

    def match_tool(self, text: str) -> Optional[tuple[str, str, str]]:
        """Check if generated text contains a tool call.

        Returns (tool_name, argument, full_match_text) or None.
        """
        for tool in self._tools:
            result = ToolDef.parse_call(text)
            if result and result[0] == tool.name:
                return (result[0], result[1], f"[[TOOL: {result[0]}]] {result[1]}")
        return None


# =============================================================================
# ProviderRouter — processor pipeline + text provider + tool loop
# =============================================================================

class ProviderRouter:
    """Routes messages through a processor pipeline to a text provider.

    Implements ModelProvider protocol (duck-typed) so it can be used
    anywhere a ModelProvider is expected.

    Messages flow through:
      messages → [processor, ...] → text provider → tokens

    If a ToolUseProcessor is in the chain, the router runs a tool loop:
      generate → check for [[TOOL: name]] → execute tool → continue generation

    Usage:
        router = ProviderRouter()
        router.add_processor(VisionProcessor("multimodal"))
        router.add_processor(ToolUseProcessor())
        router.set_text_provider("hf-default")
        register_provider("default", router)
    """

    def __init__(self):
        self._processors: List[MessageProcessor] = []
        self._text_name: Optional[str] = None
        self._model_id_str = "router-v1"
        self._max_tool_rounds = 3

    def add_processor(self, processor: MessageProcessor) -> "ProviderRouter":
        """Add a message processor to the pipeline (appended to chain)."""
        self._processors.append(processor)
        return self

    def set_text_provider(self, name: str) -> None:
        """Set the provider that generates the final response."""
        self._text_name = name

    @property
    def model_id(self) -> str:
        return self._model_id_str

    @property
    def capabilities(self):
        return ModelCapabilities(chat=True, streaming=True, embedding=False, vision=True)

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "processors": [type(p).__name__ for p in self._processors],
            "text_provider": self._text_name,
            "max_tool_rounds": self._max_tool_rounds,
        }

    def _find_tool_processor(self) -> Optional[ToolUseProcessor]:
        for p in self._processors:
            if isinstance(p, ToolUseProcessor):
                return p
        return None

    async def chat_stream(
        self,
        messages: List[ChatMessage],
        max_tokens: int = 512,
        temperature: float = 0.8,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Run messages through processors, then stream from text provider.

        If a ToolUseProcessor is configured, runs a tool loop:
        generate → detect [[TOOL: name]] → execute → continue.
        """
        msgs = list(messages)
        for processor in self._processors:
            try:
                msgs = await processor.process(msgs)
            except Exception as e:
                logger.warning(f"Processor {type(processor).__name__} failed: {e}")

        if not self._text_name:
            yield "no text provider configured"
            return
        text_provider = get_provider(self._text_name)
        if text_provider is None:
            yield f"text provider '{self._text_name}' not found"
            return

        tool_proc = self._find_tool_processor()
        tool_round = 0

        while tool_round <= self._max_tool_rounds:
            generated = ""
            async for token in text_provider.chat_stream(msgs, max_tokens, temperature, **kwargs):
                generated += token
                yield token

            if tool_proc is None or tool_round >= self._max_tool_rounds:
                break

            match = tool_proc.match_tool(generated)
            if match is None:
                break

            tool_name, tool_arg, match_text = match
            logger.info(f"Tool call detected: {tool_name}({tool_arg[:40]})")
            yield f"\n[Running tool: {tool_name}...]\n"

            # Execute the tool
            tool_result = await self._execute_tool(tool_name, tool_arg)
            logger.info(f"Tool result: {tool_result[:60] if tool_result else 'empty'}")

            # Append result as a system message and continue
            msgs.append({"role": "assistant", "content": generated.replace(match_text, "").strip()})
            msgs.append({"role": "system", "content": f"Tool {tool_name} returned: {tool_result}"})
            msgs.append({"role": "user", "content": "Continue where you left off."})
            tool_round += 1

    async def _execute_tool(self, tool_name: str, arg: str) -> str:
        """Execute a tool call by routing to the appropriate provider."""
        if tool_name == "describe_image":
            provider = get_provider("multimodal")
            if provider is None:
                try:
                    from domains.multimodal.manager import get_multimodal_manager
                    mgr = get_multimodal_manager()
                    mgr.initialize(vision_model="slonet")
                    provider = get_provider("multimodal")
                except Exception:
                    pass
            if provider is not None:
                try:
                    result = ""
                    async for token in provider.chat_stream(
                        [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": arg}}]}],
                        max_tokens=30, temperature=0.8,
                    ):
                        result += token
                    return result.strip() or "[no description]"
                except Exception as e:
                    return f"[tool error: {e}]"

            # Fallback: direct manager call
            try:
                import base64
                from PIL import Image
                import io
                clean = arg.split(",")[1] if "," in arg else arg
                img = Image.open(io.BytesIO(base64.b64decode(clean))).convert("RGB")
                from domains.multimodal.manager import get_multimodal_manager
                mgr = get_multimodal_manager()
                return mgr.caption_image(img).text
            except Exception as e:
                return f"[tool error: {e}]"

        logger.warning(f"Unknown tool: {tool_name}")
        return f"[unknown tool: {tool_name}]"

    async def chat(
        self,
        messages: List[ChatMessage],
        max_tokens: int = 512,
        temperature: float = 0.8,
        **kwargs,
    ) -> str:
        chunks = []
        async for chunk in self.chat_stream(messages, max_tokens, temperature, **kwargs):
            chunks.append(chunk)
        return "".join(chunks)

    def embed(self, text: str) -> List[float]:
        return []


__all__ = [
    "ModelProvider",
    "ModelCapabilities",
    "ChatMessage",
    "MessageProcessor",
    "VisionProcessor",
    "KnowledgeProcessor",
    "register_provider",
    "get_provider",
    "list_providers",
    "ProviderRouter",
    "HFModelProvider",
    "InferenceEngineProvider",
    "setup_providers",
    "discover_checkpoints",
]
