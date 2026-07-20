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

SloNetChatProvider is the primary torch-free inference engine.
"""

import asyncio
from typing import Protocol, AsyncIterator, Optional, List, Dict, Any, runtime_checkable
from dataclasses import dataclass
import logging

from domains.inference.prompt_formatter import PromptFormatter

logger = logging.getLogger("slo.models.provider")


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
            logger.warning(f"Failed to init multimodal: {e}", extra={"tag": "MODEL"})
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
            logger.warning(f"Caption failed: {e}", extra={"tag": "MODEL"})
            return "[image]"

    async def process(self, messages: list) -> list:
        images = self._extract_images(messages)
        if not images:
            return messages
        if not self._ensure_provider():
            logger.warning("Vision provider unavailable, keeping images as-is", extra={"tag": "MODEL"})
            return messages

        captions = []
        for i, img in enumerate(images):
            cap = await self._caption(img)
            captions.append(cap)
            logger.info(f"Image {i+1} captioned: {cap[:60]}", extra={"tag": "MODEL"})

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
                logger.warning(f"SloTransformer generation error: {e}", extra={"tag": "MODEL"})
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
            extra={"tag": "MODEL"},
        )
        return cls(model, stoi, itos, model_id_str=model_id_str)


# =============================================================================
# setup_providers — wire up default vision→text routing
# =============================================================================


def setup_providers(slonet_hf_id: Optional[str] = None,
                    slonet_provider=None,
                    model_registry=None,
                    quantize: bool = False,
                    quant_bits: int = 8,
                    quant_mode: str = "symmetric") -> None:
    """Register all model providers and wire up the default router.

    Registers:
    - ``"slonet-native"``: ``SloNetChatProvider`` (pure NumPy, no PyTorch)
      (if ``slonet_hf_id`` given OR ``slonet_provider`` given)
    - ``"default"``: ``ProviderRouter`` with VisionProcessor + text provider

    Args:
        slonet_provider: Optional pre-loaded ``SloNetChatProvider``. When provided,
            skips re-loading the SLNC file (saves ~6s). Takes priority over
            ``slonet_hf_id``.
        model_registry: Optional ``ModelRegistry`` for lifecycle management.
    """
    text_provider_name = None

    # Prefer pre-loaded provider (avoids duplicate SLNC load)
    if slonet_provider is not None:
        register_provider("slonet-native", slonet_provider)
        text_provider_name = "slonet-native"
        logger.info("Registered slonet-native provider: %s (pre-loaded)",
                    getattr(slonet_provider, '_model_id', '?'), extra={"tag": "MODEL"})
    elif slonet_hf_id:
        try:
            from domains.inference.slonet_provider import SloNetChatProvider
            from domains.infrastructure.safetensors_loader import _get_model_dir
            _cache_dir = _get_model_dir(slonet_hf_id)
            _slnc = _cache_dir / "model.slnc"
            if not _slnc.exists():
                raise FileNotFoundError(f"No .slnc file for {slonet_hf_id} at {_slnc}")
            slonet_provider = SloNetChatProvider.from_slnc(
                str(_slnc),
                model_id=slonet_hf_id,
                quantize=quantize,
                quant_bits=quant_bits,
                quant_mode=quant_mode,
            )
            register_provider("slonet-native", slonet_provider)
            text_provider_name = "slonet-native"
            logger.info("Registered slonet-native provider: %s (quant=%s)",
                        slonet_hf_id, f"int{quant_bits}" if quantize else "none", extra={"tag": "MODEL"})
        except Exception as e:
            logger.warning("Failed to load slonet-native provider %s: %s", slonet_hf_id, e, extra={"tag": "MODEL"})

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
                    text_provider_name, extra={"tag": "MODEL"})
    else:
        logger.info("SloNet provider active as default — skipping ProviderRouter override", extra={"tag": "MODEL"})


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
        router.set_text_provider("slonet-native")
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
                logger.warning(f"Processor {type(processor).__name__} failed: {e}", extra={"tag": "MODEL"})

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
            logger.info(f"Tool call detected: {tool_name}({tool_arg[:40]})", extra={"tag": "MODEL"})
            yield f"\n[Running tool: {tool_name}...]\n"

            # Execute the tool
            tool_result = await self._execute_tool(tool_name, tool_arg)
            logger.info(f"Tool result: {tool_result[:60] if tool_result else 'empty'}", extra={"tag": "MODEL"})

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

        logger.warning(f"Unknown tool: {tool_name}", extra={"tag": "MODEL"})
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
    "setup_providers",
]
