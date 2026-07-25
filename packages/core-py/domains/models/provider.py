"""
Model providers — register, retrieve, and serve model backends.

Core abstractions:
- ModelProvider: protocol every backend satisfies (chat, embed, capabilities)
- MessageProcessor: protocol for message transforms (vision, knowledge, tools, personality)
- ProviderRouter: chains processors → provider, implements ModelProvider itself
- Provider/processor registries: named lookup for DI

Built-in processors:
- VisionProcessor: captions images, injects as text
- KnowledgeProcessor: injects knowledge context
- ToolUseProcessor: injects tool-call instructions, detects [[TOOL: name]] in output
- PersonalityProcessor: injects soul/personality traits into system prompt
- StyleProcessor: adjusts formality/directness/verbosity
"""

import asyncio
import re
from typing import AsyncIterator, Optional, List, Dict, Any, Protocol, runtime_checkable, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger("slo.models.provider")


# =============================================================================
# Types
# =============================================================================

ChatMessage = Dict[str, str]  # {"role": "user"|"assistant"|"system", "content": str}


@dataclass
class ModelCapabilities:
    """What a model can do."""
    chat: bool = False
    streaming: bool = False
    embedding: bool = False
    vision: bool = False
    functions: bool = False


# =============================================================================
# ModelProvider — protocol every backend must satisfy
# =============================================================================

@runtime_checkable
class ModelProvider(Protocol):
    """Interface for model backends.

    Satisfied by: SloTransformerProvider, SloNetChatProvider,
    MultimodalEngine, ProviderRouter.

    Callers always do:
        provider = get_provider("default")
        async for token in provider.chat_stream(messages, **kwargs): ...
    """

    @property
    def model_id(self) -> str: ...

    @property
    def capabilities(self) -> ModelCapabilities: ...

    async def chat_stream(
        self,
        messages: List[ChatMessage],
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.2,
        cancel_event=None,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]: ...

    async def chat(
        self,
        messages: List[ChatMessage],
        max_tokens: int = 512,
        temperature: float = 0.8,
        **kwargs,
    ) -> str: ...

    def embed(self, text: str) -> List[float]: ...

    @property
    def metadata(self) -> Dict[str, Any]: ...


# =============================================================================
# MessageProcessor — protocol for message transformers
# =============================================================================

@runtime_checkable
class MessageProcessor(Protocol):
    """Transforms message lists before they reach the provider.

    Concrete processors: VisionProcessor, KnowledgeProcessor,
    ToolUseProcessor, PersonalityProcessor, StyleProcessor.
    """

    async def process(self, messages: list) -> list: ...


# =============================================================================
# Provider registry
# =============================================================================

_providers: Dict[str, Any] = {}


def register_provider(name: str, provider) -> None:
    """Register a model provider by name."""
    _providers[name] = provider


def get_provider(name: str):
    """Get a registered provider by name."""
    return _providers.get(name)


def list_providers() -> List[str]:
    """List all registered provider names."""
    return list(_providers.keys())


# =============================================================================
# Processor registry
# =============================================================================

_processors: Dict[str, Any] = {}


def register_processor(name: str, processor) -> None:
    """Register a message processor by name."""
    _processors[name] = processor


def get_processor(name: str):
    """Get a registered processor by name."""
    return _processors.get(name)


def list_processors() -> List[str]:
    """List all registered processor names."""
    return list(_processors.keys())


async def apply_processors(messages: list, processors: list) -> list:
    """Run messages through a list of processors sequentially."""
    for proc in processors:
        try:
            messages = await proc.process(messages)
        except Exception as e:
            logger.warning(f"Processor {type(proc).__name__} failed: {e}", extra={"tag": "MODEL"})
    return messages


# =============================================================================
# VisionProcessor — caption images, inject as text
# =============================================================================

class VisionProcessor:
    """Extracts images from messages, captions via vision provider, injects as text.

    Runs before the text provider so the LLM sees image descriptions.
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
            except Exception as e:
                logger.debug("Vision chat_stream caption failed: %s", e)
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
            logger.warning("Vision provider unavailable, stripping images from messages", extra={"tag": "MODEL"})
            result = []
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, list):
                    text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                    content = "\n".join(text_parts)
                    if text_parts and msg.get("role") == "user":
                        content = f"{content}\n[Image attached but model does not support vision]"
                result.append({"role": msg["role"], "content": content})
            return result

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


# =============================================================================
# KnowledgeProcessor — inject knowledge context
# =============================================================================

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
# ToolUseProcessor — tool-call injection + detection
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

    TOOL_RE = re.compile(r'\[\[TOOL:\s*(\w+)\s*\]\]\s*(\S+)')

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

    def match_tool(self, text: str) -> Optional[Tuple[str, str, str]]:
        """Check if generated text contains a tool call.

        Returns (tool_name, argument, full_match_text) or None.
        """
        m = self.TOOL_RE.search(text)
        if m:
            tool_name, tool_arg = m.group(1), m.group(2)
            for tool in self._tools:
                if tool.name == tool_name:
                    return (tool_name, tool_arg, m.group(0))
        return None


# =============================================================================
# PersonalityProcessor — inject soul/personality traits
# =============================================================================

class PersonalityProcessor:
    """Injects personality traits into the system prompt.

    Maps all 10 PersonalityCore traits to descriptive adjectives based on weight thresholds.
    Traits with value < 0.3 or > 0.7 produce the strongest descriptions.
    """

    TRAIT_ADJECTIVES = {
        "warmth": {0.0: "neutral", 0.3: "reserved", 0.5: "friendly", 0.7: "warm", 0.9: "very warm and empathetic"},
        "creativity": {0.0: "factual", 0.3: "practical", 0.5: "balanced", 0.7: "creative", 0.9: "highly creative and imaginative"},
        "empathy": {0.0: "detached", 0.3: "observant", 0.5: "understanding", 0.7: "empathetic", 0.9: "deeply empathetic and compassionate"},
        "formality": {0.0: "casual", 0.3: "relaxed", 0.5: "professional", 0.7: "formal", 0.9: "highly formal and precise"},
        "humor": {0.0: "serious", 0.3: "dry", 0.5: "witty", 0.7: "humorous", 0.9: "very humorous and playful"},
        "patience": {0.0: "brisk", 0.3: "efficient", 0.5: "patient", 0.7: "thorough", 0.9: "extremely patient and methodical"},
        "confidence": {0.0: "cautious", 0.3: "measured", 0.5: "confident", 0.7: "assertive", 0.9: "very confident and decisive"},
        "curiosity": {0.0: "direct", 0.3: "interested", 0.5: "curious", 0.7: "inquisitive", 0.9: "deeply curious and exploratory"},
        "directness": {0.0: "indirect", 0.3: "gentle", 0.5: "balanced", 0.7: "direct", 0.9: "very direct and to the point"},
        "optimism": {0.0: "realistic", 0.3: "grounded", 0.5: "optimistic", 0.7: "positive", 0.9: "very optimistic and encouraging"},
    }

    def __init__(self, traits: Optional[Dict[str, float]] = None):
        self._traits = traits or {}

    def set_traits(self, traits: Dict[str, float]) -> None:
        self._traits = traits

    def _describe_trait(self, name: str, value: float) -> str:
        adjectives = self.TRAIT_ADJECTIVES.get(name, {})
        if not adjectives:
            return ""
        best_threshold = max((t for t in adjectives if t <= value), default=min(adjectives))
        return adjectives[best_threshold]

    async def process(self, messages: list) -> list:
        if not self._traits:
            return messages

        descriptions = []
        for trait, value in self._traits.items():
            desc = self._describe_trait(trait, value)
            if desc:
                descriptions.append(desc)

        if not descriptions:
            return messages

        personality_line = "Be " + ", ".join(descriptions) + " in your responses."
        personality_msg = {"role": "system", "content": f"Personality: {personality_line}"}

        has_system = any(m.get("role") == "system" for m in messages)
        if has_system:
            for i, m in enumerate(messages):
                if m.get("role") == "system":
                    messages[i] = {"role": "system", "content": f"{m['content']}\n\n{personality_line}"}
                    break
        else:
            messages.insert(0, personality_msg)
        return messages


# =============================================================================
# StyleProcessor — adjust response style
# =============================================================================

class StyleProcessor:
    """Adjusts formality, directness, and verbosity of responses.

    Injects style instructions into the system prompt based on
    configurable style weights.
    """

    def __init__(self, formality: float = 0.5, directness: float = 0.5, verbosity: float = 0.5):
        self._formality = formality
        self._directness = directness
        self._verbosity = verbosity

    def set_style(self, formality: float = 0.5, directness: float = 0.5, verbosity: float = 0.5) -> None:
        self._formality = formality
        self._directness = directness
        self._verbosity = verbosity

    async def process(self, messages: list) -> list:
        instructions = []

        if self._formality > 0.7:
            instructions.append("Use formal language and proper grammar.")
        elif self._formality < 0.3:
            instructions.append("Use casual, conversational language.")

        if self._directness > 0.7:
            instructions.append("Be direct and concise. Get to the point quickly.")
        elif self._directness < 0.3:
            instructions.append("Be thorough and provide context before conclusions.")

        if self._verbosity > 0.7:
            instructions.append("Provide detailed, comprehensive answers.")
        elif self._verbosity < 0.3:
            instructions.append("Keep answers brief and to the point.")

        if not instructions:
            return messages

        style_text = " ".join(instructions)
        style_msg = {"role": "system", "content": f"Style: {style_text}"}

        has_system = any(m.get("role") == "system" for m in messages)
        if has_system:
            for i, m in enumerate(messages):
                if m.get("role") == "system":
                    messages[i] = {"role": "system", "content": f"{m['content']}\n\n{style_text}"}
                    break
        else:
            messages.insert(0, style_msg)
        return messages


# =============================================================================
# ProviderRouter — processor pipeline + text provider
# =============================================================================

class ProviderRouter:
    """Routes messages through a processor pipeline to a text provider.

    Implements ModelProvider protocol so it can be registered as "default".

    Pipeline: messages → [processors...] → text provider → tokens

    If ToolUseProcessor is in the chain, runs a tool loop:
      generate → detect [[TOOL: name]] → execute → continue

    Usage:
        router = ProviderRouter()
        router.add_processor(VisionProcessor("multimodal"))
        router.add_processor(KnowledgeProcessor())
        router.add_processor(ToolUseProcessor())
        router.add_processor(PersonalityProcessor())
        router.set_text_provider("slonet-native")
        register_provider("default", router)
    """

    def __init__(self):
        self._processors: List[MessageProcessor] = []
        self._text_name: Optional[str] = None
        self._model_id_str = "router-v1"
        self._max_tool_rounds = 3

    def add_processor(self, processor: MessageProcessor) -> "ProviderRouter":
        self._processors.append(processor)
        return self

    def set_text_provider(self, name: str) -> None:
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
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.2,
        cancel_event=None,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Run messages through processors, then stream from text provider."""
        msgs = list(messages)
        for processor in self._processors:
            try:
                msgs = await processor.process(msgs)
            except Exception as e:
                logger.warning(f"Processor {type(processor).__name__} failed: {e}", extra={"tag": "MODEL"})

        if not self._text_name:
            yield "No text model configured. Please load a model first."
            return
        text_provider = get_provider(self._text_name)
        if text_provider is None:
            yield f"Text model '{self._text_name}' is not available. Please load a model."
            return

        tool_proc = self._find_tool_processor()
        tool_round = 0

        while tool_round <= self._max_tool_rounds:
            if cancel_event is not None and cancel_event.is_set():
                return

            generated = ""
            async for token in text_provider.chat_stream(
                msgs, max_tokens=max_tokens, temperature=temperature,
                top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty,
                cancel_event=cancel_event, session_id=session_id, **kwargs,
            ):
                generated += token
                yield token

            if tool_proc is None or tool_round >= self._max_tool_rounds:
                break

            match = tool_proc.match_tool(generated)
            if match is None:
                break

            if cancel_event is not None and cancel_event.is_set():
                return

            tool_name, tool_arg, match_text = match
            logger.info(f"Tool call detected: {tool_name}({tool_arg[:40]})", extra={"tag": "MODEL"})
            yield f"\n[Running tool: {tool_name}...]\n"

            tool_result = await self._execute_tool(tool_name, tool_arg, cancel_event=cancel_event)
            logger.info(f"Tool result: {tool_result[:60] if tool_result else 'empty'}", extra={"tag": "MODEL"})

            msgs.append({"role": "assistant", "content": generated.replace(match_text, "").strip()})
            msgs.append({"role": "system", "content": f"Tool {tool_name} returned: {tool_result}"})
            msgs.append({"role": "user", "content": "Continue where you left off."})
            tool_round += 1

    async def _execute_tool(self, tool_name: str, arg: str, cancel_event=None) -> str:
        """Execute a tool call by routing to the appropriate provider."""
        if cancel_event is not None and cancel_event.is_set():
            return "[cancelled]"

        if tool_name == "describe_image":
            provider = get_provider("multimodal")
            if provider is None:
                try:
                    from domains.multimodal.manager import get_multimodal_manager
                    mgr = get_multimodal_manager()
                    mgr.initialize(vision_model="slonet")
                    provider = get_provider("multimodal")
                except Exception as e:
                    logger.debug("Failed to init multimodal for tool: %s", e)
            if provider is not None:
                try:
                    result = ""
                    async for token in provider.chat_stream(
                        [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": arg}}]}],
                        max_tokens=30, temperature=0.8,
                        cancel_event=cancel_event,
                    ):
                        result += token
                    return result.strip() or "[no description]"
                except Exception as e:
                    return f"[tool error: {e}]"

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


# =============================================================================
# SloTransformerProvider — pure NumPy SloTransformer inference
# =============================================================================

class SloTransformerProvider:
    """Wraps a pure NumPy SloTransformer as an async chat provider.

    Zero PyTorch dependency. Works with any .slo checkpoint.
    """

    def __init__(self, model, stoi: dict, itos: dict, model_id_str: str = "soultransformer"):
        self._model = model
        self._stoi = stoi
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
        session_id: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        prompt = self._messages_to_prompt(messages)
        input_ids = self._encode(prompt)
        if not input_ids:
            input_ids = [self._bos]
        import numpy as np
        inp = np.array([input_ids], dtype=np.int64)
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
        """Load a trained SloTransformer from a .slo checkpoint."""
        from domains.inference import load_soul

        soul, sd = load_soul(path)
        if isinstance(sd, dict) and "tok_emb.weight" not in sd:
            sd = sd.get("weights", sd)
            if not isinstance(sd, dict):
                sd = sd.state_dict() if hasattr(sd, "state_dict") else {}

        chars = ["<PAD>", "<UNK>"] + list(" abcdefghijklmnopqrstuvwxyz0123456789.,!?-'")
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}

        meta_vocab = getattr(soul, "vocab_size", None) or (soul.metadata or {}).get("vocab_size")
        if meta_vocab and meta_vocab > len(chars):
            extra = ["_"] * (meta_vocab - len(chars))
            chars = chars + extra
            stoi = {ch: i for i, ch in enumerate(chars)}
            itos = {i: ch for i, ch in enumerate(chars)}

        vocab = sd["tok_emb.weight"].shape[0]
        n_embed = sd["tok_emb.weight"].shape[1]

        n_layer = 1
        for key in sd:
            if key.startswith("blocks."):
                idx = int(key.split(".")[1])
                n_layer = max(n_layer, idx + 1)

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
# setup_providers — wire up default provider + full processor pipeline
# =============================================================================

def setup_providers(
    slonet_hf_id: Optional[str] = None,
    slonet_provider=None,
    model_registry=None,
    quantize: bool = False,
    quant_bits: int = 8,
    quant_mode: str = "symmetric",
    personality_traits: Optional[Dict[str, float]] = None,
) -> None:
    """Register providers and build the default processor pipeline.

    Registers:
    - ``"slonet-native"``: SloNetChatProvider (if slonet_hf_id or slonet_provider given)
    - ``"default"``: ProviderRouter with processor chain → text provider

    Processor chain (in order):
    1. VisionProcessor — caption images
    2. ToolUseProcessor — tool-call instructions
    3. PersonalityProcessor — soul trait injection
    4. StyleProcessor — formality/directness/verbosity

    KnowledgeProcessor is NOT in the router pipeline — knowledge is per-request
    and handled by the caller (inference.py) via ``apply_processors()``.

    Args:
        slonet_hf_id: HuggingFace model ID for SloNet model
        slonet_provider: Pre-loaded provider (skips re-loading)
        model_registry: Optional ModelRegistry for lifecycle management
        quantize: Whether to quantize the model
        quant_bits: Quantization bit width
        quant_mode: Quantization mode
        personality_traits: Optional personality traits dict
    """
    text_provider_name = None

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

    # Build default ProviderRouter with full processor pipeline
    existing = _providers.get("default")
    _is_slonet = existing is not None and type(existing).__name__ in ("SloTransformerProvider", "SloNetChatProvider")
    if not _is_slonet:
        router = ProviderRouter()
        vision_proc = VisionProcessor("multimodal")
        tool_proc = ToolUseProcessor()
        personality_proc = PersonalityProcessor(traits=personality_traits)
        style_proc = StyleProcessor()
        router.add_processor(vision_proc)
        router.add_processor(tool_proc)
        router.add_processor(personality_proc)
        router.add_processor(style_proc)
        if text_provider_name:
            router.set_text_provider(text_provider_name)
        register_provider("default", router)
        # Register processors in the registry for lookup by routers
        register_processor("vision", vision_proc)
        register_processor("tool_use", tool_proc)
        register_processor("personality", personality_proc)
        register_processor("style", style_proc)
        logger.info("Registered default router (processors=%s, text=%s)",
                    [type(p).__name__ for p in router._processors],
                    text_provider_name, extra={"tag": "MODEL"})
    else:
        logger.info("SloNet provider active as default — skipping router override", extra={"tag": "MODEL"})


def update_personality_traits(traits: Dict[str, float]) -> None:
    """Update processors in the default router with new soul traits.

    PersonalityProcessor receives all traits.
    StyleProcessor receives formality/directness if present in the trait dict.

    Called when a soul is switched to keep the processor pipeline in sync.
    """
    router = _providers.get("default")
    if router is None or not isinstance(router, ProviderRouter):
        return
    for proc in router._processors:
        if isinstance(proc, PersonalityProcessor):
            proc.set_traits(traits)
            logger.info("Updated personality traits: %s", list(traits.keys()), extra={"tag": "MODEL"})
        elif isinstance(proc, StyleProcessor):
            formality = traits.get("formality", 0.5)
            directness = traits.get("directness", 0.5)
            proc.set_style(formality=formality, directness=directness)
            logger.info("Updated style: formality=%.2f directness=%.2f", formality, directness, extra={"tag": "MODEL"})


__all__ = [
    # Types
    "ChatMessage",
    "ModelCapabilities",
    # Protocols
    "ModelProvider",
    "MessageProcessor",
    # Registries
    "register_provider",
    "get_provider",
    "list_providers",
    "register_processor",
    "get_processor",
    "list_processors",
    "apply_processors",
    # Providers
    "SloTransformerProvider",
    # Processors
    "VisionProcessor",
    "KnowledgeProcessor",
    "ToolUseProcessor",
    "ToolDef",
    "PersonalityProcessor",
    "StyleProcessor",
    # Router
    "ProviderRouter",
    # Setup
    "setup_providers",
    "update_personality_traits",
]
