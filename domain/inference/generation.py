"""generation — the one generation entry: inference folded into generation.

Every model call in the product answers the same contract here::

    chat(messages, *, max_tokens, temperature) -> str

Three transports sit behind it, one implementation each:

    hosted(base_url, api_key, model)  the canonical ApiProvider wire
                                      (httpx, retries, usage tracking,
                                      OpenAI-compatible endpoints — incl.
                                      keyless local Ollama/LM Studio/vLLM)
    soul(soul_path)                   in-process SloNet checkpoint
    engine(host, port)                standalone InferenceEngine over its
                                      own TCP protocol (InferenceClient)

Plus ``extract_json`` — the tolerant outermost-first reply parser.

Scripts and surfaces import THIS; they never carry wire code. That is
the whole point of the entry: one hosted implementation, one contract,
one place a wire change lands.
"""

from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LanguageClient(Protocol):
    """The one contract. Messages are ``[{role, content}]`` dicts."""

    model_id: str

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> str: ...


def extract_json(text: str) -> Any:
    """First parseable JSON object or array in a reply — prose tolerated.

    Outermost-first: whichever opener (``{``/``[``) appears earliest wins,
    so an array wrapping objects stays an array. Falls back to the other
    pair, then to None. Callers never see a ``json.JSONDecodeError``.
    """
    pairs = sorted(
        (("{", "}"), ("[", "]")),
        key=lambda pair: idx if (idx := text.find(pair[0])) != -1 else len(text),
    )
    for open_ch, close_ch in pairs:
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start == -1 or end <= start:
            continue
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, (dict, list)):
            return parsed
    return None


class _Hosted:
    """The entry face over the canonical ApiProvider — one wire, stable id."""

    def __init__(self, base_url: str, provider: Any) -> None:
        self.model_id = f"chat-api:{base_url}"
        self._provider = provider

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> str:
        return await self._provider.chat(messages, max_tokens=max_tokens, temperature=temperature)


def hosted(
    base_url: str,
    api_key: str | None = None,
    model: str | None = None,
    timeout: float = 120.0,
) -> LanguageClient:
    """A hosted OpenAI-compatible endpoint via the canonical ApiProvider.

    Keyless endpoints are allowed (local servers take no Authorization
    header); when a key is present it rides the Bearer header and
    nothing else.
    """
    from domain.inference._internal.api_provider import ApiProvider

    kwargs: dict[str, Any] = {}
    if model:
        kwargs["model"] = model
    provider = ApiProvider(
        api_key=api_key or "",
        api_url=base_url,
        timeout=timeout,
        **kwargs,
    )
    return _Hosted(base_url, provider)


def soul(soul_path: str) -> LanguageClient:
    """An in-process SloNet checkpoint — our engine, numpy sampling."""
    from domain.inference._internal.slonet_provider import SloNetChatProvider

    return SloNetChatProvider.from_soul(soul_path, model_id=soul_path)


def engine(host: str = "127.0.0.1", port: int = 9100) -> LanguageClient:
    """A standalone InferenceEngine over its own TCP protocol."""
    from domain.infrastructure._internal.inference_client import InferenceClient

    client = InferenceClient(host=host, port=port)
    client.connect()
    return client
