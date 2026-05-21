"""HTTP client for TUI: status checks, streaming chat, text generation.

Encapsulates API URL and all endpoint calls in a single ``TuiApiClient`` class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

import httpx


@dataclass(frozen=True)
class ApiJsonResult:
    """JSON body from a GET, or error metadata on transport failure."""

    status_code: int
    payload: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# Backward-compat alias
HealthFetchResult = ApiJsonResult


class TuiApiClient:
    """HTTP client for all TUI API interactions.

    Wraps base URL + all endpoints as methods so screens never construct URLs.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8000):
        self.base_url = f"http://{host}:{port}"

    def _get(self, path: str, *, timeout: float = 5.0) -> ApiJsonResult:
        url = self.base_url + path
        try:
            with httpx.Client(timeout=timeout) as client:
                r = client.get(url)
                payload: Optional[Dict[str, Any]] = None
                try:
                    payload = r.json()
                except ValueError:
                    payload = None
                return ApiJsonResult(status_code=r.status_code, payload=payload, error=None)
        except httpx.HTTPError as e:
            return ApiJsonResult(status_code=0, payload=None, error=str(e))

    def _stream_sse(
        self,
        path: str,
        payload: Dict[str, Any],
        *,
        timeout: float = 60.0,
    ) -> Iterator[str]:
        """POST to endpoint, yield ``data.token`` from SSE events until complete/error."""
        import json

        url = self.base_url + path
        try:
            with httpx.Client(timeout=httpx.Timeout(timeout)) as client:
                with client.stream("POST", url, json=payload) as resp:
                    if resp.status_code != 200:
                        yield f"[Error {resp.status_code}]"
                        return
                    for line in resp.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        try:
                            event = json.loads(line[6:])
                            status = event.get("status", "")
                            data = event.get("data", {})
                            if status == "error":
                                yield f"[Error: {data.get('error', 'unknown')}]"
                                return
                            token = data.get("token", "")
                            if token:
                                yield token
                            if status in ("complete", "done"):
                                return
                        except json.JSONDecodeError:
                            continue
        except httpx.HTTPError as e:
            yield f"[Connection error: {e}]"

    # ── Health ──

    def fetch_health(self, *, timeout: float = 5.0) -> ApiJsonResult:
        """GET /health."""
        return self._get("/health", timeout=timeout)

    def fetch_metrics(self, *, timeout: float = 5.0) -> ApiJsonResult:
        """GET /metrics."""
        return self._get("/metrics", timeout=timeout)

    def fetch_health_detailed(self, *, timeout: float = 5.0) -> ApiJsonResult:
        """GET /health/detailed."""
        return self._get("/health/detailed", timeout=timeout)

    # ── Chat ──

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.8,
        max_tokens: int = 256,
        model: str = "gpt2",
        timeout: float = 60.0,
    ) -> Iterator[str]:
        """POST /chat/stream, yield tokens."""
        payload = {
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "use_context_core": False,
        }
        yield from self._stream_sse("/chat/stream", payload, timeout=timeout)

    def generate_text(
        self,
        prompt: str,
        *,
        temperature: float = 0.8,
        max_tokens: int = 128,
        model: str = "gpt2",
        timeout: float = 30.0,
    ) -> Iterator[str]:
        """POST /generate/stream, yield tokens."""
        payload = {
            "prompt": prompt,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        yield from self._stream_sse("/generate/stream", payload, timeout=timeout)


# ── Backward-compat aliases for existing tests ──

def _client_for(base_url: str) -> TuiApiClient:
    """Parse host:port from base_url for tests that pass URLs directly."""
    import re
    m = re.match(r"http://([^:]+):(\d+)", base_url)
    if m:
        return TuiApiClient(host=m.group(1), port=int(m.group(2)))
    return TuiApiClient()


def fetch_health(base_url: str, *, timeout: float = 5.0) -> ApiJsonResult:
    """Backward-compat: delegates to TuiApiClient."""
    return _client_for(base_url).fetch_health(timeout=timeout)


def fetch_metrics(base_url: str, *, timeout: float = 5.0) -> ApiJsonResult:
    return _client_for(base_url).fetch_metrics(timeout=timeout)


def fetch_health_detailed(base_url: str, *, timeout: float = 5.0) -> ApiJsonResult:
    return _client_for(base_url).fetch_health_detailed(timeout=timeout)


def stream_chat(base_url: str, messages, *, temperature=0.8, max_tokens=256, model="gpt2", timeout=60.0) -> Iterator[str]:
    return _client_for(base_url).stream_chat(messages, temperature=temperature, max_tokens=max_tokens, model=model, timeout=timeout)


def generate_text(base_url: str, prompt: str, *, temperature=0.8, max_tokens=128, model="gpt2", timeout=30.0) -> Iterator[str]:
    return _client_for(base_url).generate_text(prompt, temperature=temperature, max_tokens=max_tokens, model=model, timeout=timeout)
