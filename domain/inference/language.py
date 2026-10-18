"""language — the one language-model API every surface projects from.

Three transports, one shape: every client answers
``chat(messages, *, max_tokens, temperature) -> str`` with
OpenAI-shaped message dicts.

    hosted(base_url, api_key, model)  hosted chat-completions wire shape
                                      (Authorization: Bearer), stdlib only
    soul(soul_path)                   in-process SloNet checkpoint
    engine(host, port)                standalone engine over its own
                                      TCP protocol (InferenceClient)

Plus ``extract_json`` — the tolerant "first JSON object/array in a
reply" helper scripts kept re-implementing with their own json imports.

Wire code and tolerance live HERE; scripts and surfaces call it. That
is the whole point of this module: no adapter in avion, no _chat in a
script, no per-surface Authorization header.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
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


class _HostedClient:
    """Chat-completions over HTTP — the de-facto hosted wire shape.

    The key reaches the Authorization header and nothing else: errors
    carry the status only, and ``model_id`` carries the URL only.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str | None,
        timeout: float,
    ) -> None:
        self.model_id = f"chat-api:{base_url}"
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._model = model
        self._timeout = timeout

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> str:
        payload: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if self._model:
            payload["model"] = self._model
        req = urllib.request.Request(
            self._url,
            data=json.dumps(payload).encode(),
            headers=self._headers,
            method="POST",
        )

        def _post() -> str:
            with urllib.request.urlopen(req, None, self._timeout) as resp:
                data = json.loads(resp.read())
            choices = data.get("choices") or []
            if choices and isinstance(choices[0], dict):
                message = choices[0].get("message") or {}
                content = message.get("content") if isinstance(message, dict) else None
                if content is not None:
                    return str(content)
            raise RuntimeError("chat endpoint returned no message content")

        try:
            return await asyncio.to_thread(_post)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"HTTP {exc.code}") from None


def hosted(
    base_url: str,
    api_key: str | None = None,
    model: str | None = None,
    timeout: float = 120.0,
) -> LanguageClient:
    """A hosted chat-completions endpoint (``base_url`` carries /v1)."""
    return _HostedClient(base_url, api_key, model, timeout)


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
