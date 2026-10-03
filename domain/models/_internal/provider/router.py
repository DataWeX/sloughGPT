"""ProviderRouter — processor pipeline + text provider, __slots__, module-level."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from .protocols import ChatMessage, MessageProcessor, ModelCapabilities
from .registry import get_provider

logger = logging.getLogger("slo.models.provider")


class ProviderRouter:
    """Routes messages through a processor pipeline to a text provider."""

    __slots__ = ("_processors", "_text_name", "_model_id_str", "_max_tool_rounds")

    def __init__(self):
        self._processors: list[MessageProcessor] = []
        self._text_name: str | None = None
        self._model_id_str = "router-v1"
        self._max_tool_rounds = 3

    def add_processor(self, processor: MessageProcessor) -> ProviderRouter:
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
    def metadata(self) -> dict[str, Any]:
        return {
            "processors": [type(p).__name__ for p in self._processors],
            "text_provider": self._text_name,
            "max_tool_rounds": self._max_tool_rounds,
        }

    def _find_tool_processor(self):
        from .processors.tool_use import ToolUseProcessor

        for p in self._processors:
            if isinstance(p, ToolUseProcessor):
                return p
        return None

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.85,
        top_k: int = 40,
        repetition_penalty: float = 1.15,
        cancel_event=None,
        session_id: str | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        msgs = list(messages)
        for processor in self._processors:
            try:
                msgs = await processor.process(msgs)
            except Exception as e:
                logger.warning(
                    "Processor %s failed: %s", type(processor).__name__, e, extra={"tag": "MODEL"}
                )
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
                msgs,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                cancel_event=cancel_event,
                session_id=session_id,
                **kwargs,
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
            logger.info(
                "Tool call detected: %s(%s)", tool_name, tool_arg[:40], extra={"tag": "MODEL"}
            )
            yield f"\n[Running tool: {tool_name}...]\n"
            tool_result = await self._execute_tool(tool_name, tool_arg, cancel_event=cancel_event)
            logger.info(
                "Tool result: %s",
                tool_result[:60] if tool_result else "empty",
                extra={"tag": "MODEL"},
            )
            msgs.append({"role": "assistant", "content": generated.replace(match_text, "").strip()})
            msgs.append({"role": "system", "content": f"Tool {tool_name} returned: {tool_result}"})
            msgs.append({"role": "user", "content": "Continue where you left off."})
            tool_round += 1

    async def _execute_tool(self, tool_name: str, arg: str, cancel_event=None) -> str:
        if cancel_event is not None and cancel_event.is_set():
            return "[cancelled]"
        if tool_name == "describe_image":
            provider = get_provider("multimodal")
            if provider is None:
                try:
                    from domain.multimodal._internal.manager import get_multimodal_manager

                    mgr = get_multimodal_manager()
                    mgr.initialize(vision_model="slonet")
                    provider = get_provider("multimodal")
                except Exception as e:
                    logger.debug("Failed to init multimodal for tool: %s", e)
            if provider is not None:
                try:
                    result = ""
                    async for token in provider.chat_stream(
                        [
                            {
                                "role": "user",
                                "content": [{"type": "image_url", "image_url": {"url": arg}}],
                            }
                        ],
                        max_tokens=30,
                        temperature=0.8,
                        cancel_event=cancel_event,
                    ):
                        result += token
                    return result.strip() or "[no description]"
                except Exception as e:
                    return f"[tool error: {e}]"
            try:
                import base64
                import io

                from PIL import Image

                clean = arg.split(",")[1] if "," in arg else arg
                img = Image.open(io.BytesIO(base64.b64decode(clean))).convert("RGB")
                from domain.multimodal._internal.manager import get_multimodal_manager

                mgr = get_multimodal_manager()
                return mgr.caption_image(img).text
            except Exception as e:
                return f"[tool error: {e}]"
        logger.warning("Unknown tool: %s", tool_name, extra={"tag": "MODEL"})
        return f"[unknown tool: {tool_name}]"

    async def chat(
        self, messages: list[ChatMessage], max_tokens: int = 512, temperature: float = 0.7, **kwargs
    ) -> str:
        msgs = list(messages)
        for processor in self._processors:
            try:
                msgs = await processor.process(msgs)
            except Exception as e:
                logger.warning(
                    "Processor %s failed: %s", type(processor).__name__, e, extra={"tag": "MODEL"}
                )
        if not self._text_name:
            return "No text model configured. Please load a model first."
        text_provider = get_provider(self._text_name)
        if text_provider is None:
            return f"Text model '{self._text_name}' is not available. Please load a model."
        return await text_provider.chat(
            msgs, max_tokens=max_tokens, temperature=temperature, **kwargs
        )

    def embed(self, text: str) -> list[float]:
        return []
