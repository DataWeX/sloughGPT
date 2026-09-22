from __future__ import annotations

"""
Generic API Provider for External AI Services

Supports any OpenAI-compatible API endpoint:
- OpenAI (api.openai.com)
- Ollama (localhost:11434/v1)
- LM Studio (localhost:1234/v1)
- vLLM (localhost:8000/v1)
- text-generation-webui (localhost:5000/v1)
- Together.ai, Groq, etc.

Usage:
    from domain.inference._internal.api_provider import ApiProvider

    provider = ApiProvider(
        api_key="sk-...",
        api_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
    )

    # Non-streaming
    response = await provider.chat([{"role": "user", "content": "Hello"}])

    # Streaming
    async for token in provider.chat_stream([{"role": "user", "content": "Hello"}]):
        print(token, end="", flush=True)
"""

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from domain.models._internal.provider import ModelCapabilities

logger = logging.getLogger("slo.api_provider")


class ApiProvider:
    """
    Generic API provider implementing the ModelProvider protocol.

    Works with any OpenAI-compatible /v1/chat/completions endpoint.
    """

    def __init__(
        self,
        api_key: str,
        api_url: str,
        model: str = "gpt-4o-mini",
        timeout: float = 60.0,
        max_retries: int = 2,
    ):
        self.api_key = api_key
        self.api_url = api_url.rstrip("/")
        self.model_name = model
        self.timeout = timeout
        self.max_retries = max_retries

        if not self.api_key:
            raise ValueError("API key required")
        if not self.api_url:
            raise ValueError("API URL required")

    @property
    def model_id(self) -> str:
        return self.model_name

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            chat=True,
            embedding=False,
            streaming=True,
            vision=True,
            functions=True,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_name,
            "provider": "api",
            "api_url": self.api_url,
            "streaming": True,
        }

    async def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> str:
        """Non-streaming chat completion."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }

        # Add optional parameters
        if "top_k" in kwargs and kwargs["top_k"] > 0:
            payload["top_k"] = kwargs["top_k"]
        if "stop" in kwargs:
            payload["stop"] = kwargs["stop"]

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.api_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

                return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            logger.error(
                "API chat failed with status %d: %s", e.response.status_code, e.response.text
            )
            raise
        except Exception as e:
            logger.error("API chat failed: %s", e)
            raise

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        cancel_event=None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Streaming chat completion via SSE."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
        }

        if "top_k" in kwargs and kwargs["top_k"] > 0:
            payload["top_k"] = kwargs["top_k"]
        if "stop" in kwargs:
            payload["stop"] = kwargs["stop"]

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.api_url}/chat/completions",
                    headers=headers,
                    json=payload,
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if cancel_event and cancel_event.is_set():
                            break

                        if not line:
                            continue

                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break

                            try:
                                data = json.loads(data_str)
                                delta = data.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content")
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                logger.warning("Failed to parse SSE data: %s", data_str)
                                continue

        except httpx.HTTPStatusError as e:
            logger.error(
                "API stream failed with status %d: %s", e.response.status_code, e.response.text
            )
            raise
        except Exception as e:
            logger.error("API stream failed: %s", e)
            raise

    def embed(self, text: str) -> list[float]:
        """Embedding not supported by this provider."""
        return []

    def test_connection(self) -> dict[str, Any]:
        """Test connection to the API endpoint."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client(timeout=10.0) as client:
                # Try to list models as a health check
                response = client.get(
                    f"{self.api_url}/models",
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()

                models = [m.get("id", "") for m in data.get("data", [])]
                model_available = self.model_name in models

                return {
                    "status": "connected",
                    "api_url": self.api_url,
                    "model": self.model_name,
                    "model_available": model_available,
                    "available_models": models[:10],  # First 10 models
                }
        except httpx.HTTPStatusError as e:
            return {
                "status": "error",
                "error": f"HTTP {e.response.status_code}: {e.response.text[:200]}",
                "api_url": self.api_url,
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "api_url": self.api_url,
            }


def _native_provider_name(registry: dict[str, Any]) -> str | None:
    """Return the first registered native text provider name, if any."""
    for name in ("native-c", "slonet-native"):
        if name in registry:
            return name
    return None


def configure_api_provider(
    enabled: bool,
    *,
    api_url: str = "",
    api_key: str = "",
    model: str = "gpt-4o-mini",
    timeout: float = 60.0,
    max_retries: int = 2,
) -> dict[str, Any]:
    """Register ``ApiProvider`` into the runtime registry for live chat routing.

    When ``enabled`` is True and both ``api_url`` and ``api_key`` are present,
    an ``ApiProvider`` is registered under the name ``"api"`` and the default
    router's text provider is pointed at it — the processor pipeline
    (personality/style/tool-use) stays intact. When disabled, ``"api"`` is
    de-registered and chat falls back to the most recently registered native
    provider (``native-c`` then ``slonet-native``).

    Returns a status dict with ``registered``, ``text_provider`` and any
    ``error``. Never raises for missing credentials.
    """
    from domain.models._internal.provider.registry import (
        _providers,
        register_provider,
    )
    from domain.models._internal.provider.router import ProviderRouter

    router = _providers.get("default")
    current = router._text_name if isinstance(router, ProviderRouter) else None

    if not enabled:
        _providers.pop("api", None)
        native = _native_provider_name(_providers)
        if isinstance(router, ProviderRouter) and native and current == "api":
            router.set_text_provider(native)
        return {
            "enabled": False,
            "registered": False,
            "error": None,
            "text_provider": native if current == "api" else current,
        }

    if not api_url or not api_key:
        return {
            "enabled": True,
            "registered": False,
            "error": "api_url and api_key are required",
            "text_provider": current,
        }

    provider = ApiProvider(
        api_key=api_key,
        api_url=api_url,
        model=model,
        timeout=timeout,
        max_retries=max_retries,
    )
    register_provider("api", provider)
    if isinstance(router, ProviderRouter):
        router.set_text_provider("api")
    return {
        "enabled": True,
        "registered": True,
        "error": None,
        "text_provider": "api",
        "api": {"model": provider.model_name, "api_url": provider.api_url},
    }


__all__ = ["ApiProvider", "configure_api_provider"]
