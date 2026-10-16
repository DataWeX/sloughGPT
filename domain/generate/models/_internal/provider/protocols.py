"""Provider protocols — single source for ModelProvider/MessageProcessor."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

ChatMessage = dict[str, str]


@dataclass(slots=True)
class ModelCapabilities:
    """What a model can do."""

    chat: bool = False
    streaming: bool = False
    embedding: bool = False
    vision: bool = False
    functions: bool = False


@runtime_checkable
class ModelProvider(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def capabilities(self) -> ModelCapabilities: ...

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
    ) -> AsyncIterator[str]: ...

    async def chat(
        self,
        messages: list[ChatMessage],
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> str: ...

    def embed(self, text: str) -> list[float]: ...

    @property
    def metadata(self) -> dict[str, Any]: ...


@runtime_checkable
class MessageProcessor(Protocol):
    async def process(self, messages: list) -> list: ...
