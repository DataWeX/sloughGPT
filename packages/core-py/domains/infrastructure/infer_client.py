"""
InferClient — Language-agnostic Python client for the /infer API.

Usage:
    from domains.infrastructure.infer_client import InferClient

    client = InferClient("http://localhost:8000")

    # Generate
    result = client.generate("Hello, world!")
    print(result.text)

    # Stream
    for token in client.generate_stream("Tell me a joke"):
        print(token, end="", flush=True)

    # Embed
    vec = client.embed("hello world")
    print(f"Dimensions: {len(vec)}")

    # Tokenize
    tokens = client.tokenize("hello world")
    print(tokens.ids)

    # Health
    health = client.health()
    print(health.model_id)

Requires: requests (stdlib fallback for basic HTTP)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

logger = logging.getLogger("slo.infer_client")

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    import urllib.request
    import urllib.error
    _HAS_REQUESTS = False


# --- Response types ---


@dataclass
class InferResult:
    """Generation result."""
    text: str
    model: str = ""
    tokens_generated: int = 0
    elapsed_ms: float = 0


@dataclass
class EmbedResult:
    """Embedding result."""
    embedding: List[float]
    dimensions: int = 0
    model: str = ""


@dataclass
class TokenizeResult:
    """Tokenization result."""
    tokens: List[str]
    ids: List[int]
    count: int = 0


@dataclass
class DetokenizeResult:
    """Detokenization result."""
    text: str
    count: int = 0


@dataclass
class HealthResult:
    """Engine health status."""
    status: str
    model_loaded: bool = False
    model_id: Optional[str] = None
    engine_type: Optional[str] = None
    has_streaming: bool = True
    has_embedding: bool = False


@dataclass
class InfoResult:
    """Model metadata."""
    model_id: str = ""
    model_type: str = ""
    num_parameters: int = 0
    vocab_size: int = 0
    max_context: int = 0
    num_layers: int = 0
    has_tokenizer: bool = False
    has_streaming: bool = True
    has_embedding: bool = False
    extra: dict = field(default_factory=dict)


# --- Client ---


class InferClient:
    """Language-agnostic client for the /infer API.

    Supports both streaming and non-streaming generation, embedding,
    tokenization, and health checks.

    Args:
        base_url: Server URL (e.g. "http://localhost:8000")
        timeout: Request timeout in seconds (default 120)
        api_key: Optional API key for authentication
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 120, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key

    def _url(self, path: str) -> str:
        return f"{self.base_url}/infer{path}"

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _post(self, path: str, data: dict) -> dict:
        """POST request, return parsed JSON."""
        url = self._url(path)
        if _HAS_REQUESTS:
            resp = _requests.post(url, json=data, headers=self._headers(), timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        else:
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode(),
                headers=self._headers(),
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())

    def _get(self, path: str) -> dict:
        """GET request, return parsed JSON."""
        url = self._url(path)
        headers = self._headers()
        if _HAS_REQUESTS:
            resp = _requests.get(url, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        else:
            req = urllib.request.Request(url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())

    # --- Generate ---

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.2,
        model: Optional[str] = None,
    ) -> InferResult:
        """Generate text from a prompt (non-streaming).

        Args:
            prompt: Input text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            repetition_penalty: Repetition penalty
            model: Optional model name override

        Returns:
            InferResult with generated text
        """
        data = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
        }
        if model:
            data["model"] = model
        resp = self._post("", data)
        return InferResult(
            text=resp.get("text", ""),
            model=resp.get("model", ""),
            tokens_generated=resp.get("tokens_generated", 0),
            elapsed_ms=resp.get("elapsed_ms", 0),
        )

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.2,
        model: Optional[str] = None,
    ) -> Iterator[str]:
        """Stream generated tokens one at a time.

        Yields individual token strings as they are generated.
        """
        data = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
        }
        if model:
            data["model"] = model

        url = self._url("/stream")
        if _HAS_REQUESTS:
            with _requests.post(url, json=data, headers=self._headers(), timeout=self.timeout, stream=True) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if not payload or payload.strip() == "[DONE]":
                        break
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if event.get("status") == "error":
                        raise RuntimeError(event.get("data", {}).get("error", "Unknown error"))
                    if event.get("status") == "complete":
                        break
                    token = event.get("data", {}).get("token", "")
                    if token:
                        yield token
        else:
            import urllib.request
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode(),
                headers=self._headers(),
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if not payload or payload.strip() == "[DONE]":
                        break
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if event.get("status") == "error":
                        raise RuntimeError(event.get("data", {}).get("error", "Unknown error"))
                    if event.get("status") == "complete":
                        break
                    token = event.get("data", {}).get("token", "")
                    if token:
                        yield token

    # --- Embed ---

    def embed(self, text: str, model: Optional[str] = None) -> List[float]:
        """Get embedding vector for text.

        Args:
            text: Input text
            model: Optional model name override

        Returns:
            List of floats (embedding vector)
        """
        data = {"text": text}
        if model:
            data["model"] = model
        resp = self._post("/embed", data)
        return resp.get("embedding", [])

    def embed_result(self, text: str, model: Optional[str] = None) -> EmbedResult:
        """Get embedding with full metadata.

        Args:
            text: Input text to embed
            model: Optional model name override

        Returns:
            EmbedResult with embedding vector, dimensions, and model name
        """
        data = {"text": text}
        if model:
            data["model"] = model
        resp = self._post("/embed", data)
        return EmbedResult(
            embedding=resp.get("embedding", []),
            dimensions=resp.get("dimensions", 0),
            model=resp.get("model", ""),
        )

    # --- Tokenize ---

    def tokenize(self, text: str, model: Optional[str] = None) -> TokenizeResult:
        """Tokenize text into token IDs and strings.

        Args:
            text: Input text
            model: Optional model name override

        Returns:
            TokenizeResult with tokens, ids, count
        """
        data = {"text": text}
        if model:
            data["model"] = model
        resp = self._post("/tokenize", data)
        return TokenizeResult(
            tokens=resp.get("tokens", []),
            ids=resp.get("ids", []),
            count=resp.get("count", 0),
        )

    def detokenize(self, ids: List[int], model: Optional[str] = None) -> DetokenizeResult:
        """Convert token IDs back to text.

        Args:
            ids: List of token IDs
            model: Optional model name override

        Returns:
            DetokenizeResult with text
        """
        data = {"ids": ids}
        if model:
            data["model"] = model
        resp = self._post("/detokenize", data)
        return DetokenizeResult(
            text=resp.get("text", ""),
            count=resp.get("count", 0),
        )

    # --- Health / Info ---

    def health(self) -> HealthResult:
        """Get engine health status.

        Returns:
            HealthResult with status, model_loaded, model_id, engine_type,
            has_streaming, and has_embedding fields
        """
        resp = self._get("/health")
        return HealthResult(
            status=resp.get("status", "unknown"),
            model_loaded=resp.get("model_loaded", False),
            model_id=resp.get("model_id"),
            engine_type=resp.get("engine_type"),
            has_streaming=resp.get("has_streaming", True),
            has_embedding=resp.get("has_embedding", False),
        )

    def info(self) -> InfoResult:
        """Get loaded model metadata.

        Returns:
            InfoResult with model_id, model_type, num_parameters, vocab_size,
            max_context, num_layers, has_tokenizer, has_streaming, has_embedding,
            and extra fields
        """
        resp = self._get("/info")
        return InfoResult(
            model_id=resp.get("model_id", ""),
            model_type=resp.get("model_type", ""),
            num_parameters=resp.get("num_parameters", 0),
            vocab_size=resp.get("vocab_size", 0),
            max_context=resp.get("max_context", 0),
            num_layers=resp.get("num_layers", 0),
            has_tokenizer=resp.get("has_tokenizer", False),
            has_streaming=resp.get("has_streaming", True),
            has_embedding=resp.get("has_embedding", False),
            extra=resp.get("extra", {}),
        )

    # --- Convenience ---

    def is_ready(self) -> bool:
        """Check if the engine has a model loaded and is ready.

        Returns:
            True if model is loaded and status is "ready", False otherwise
            (including on connection errors)
        """
        try:
            h = self.health()
            return h.model_loaded and h.status == "ready"
        except Exception:
            return False

    def __repr__(self) -> str:
        return f"InferClient(base_url={self.base_url!r})"
