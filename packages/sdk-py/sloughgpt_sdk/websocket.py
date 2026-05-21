"""
SloughGPT SDK - WebSocket Client
WebSocket client for real-time streaming (matches server ``/ws/generate``).
"""

import json
import threading
from typing import Optional, Callable, Dict, Any, List, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    pass


@dataclass
class WebSocketMessage:
    """A message received via WebSocket."""

    type: str
    data: Any
    raw: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WebSocketMessage":
        """Create from dictionary."""
        return cls(
            type=data.get("type", "unknown"),
            data=data.get("data", ""),
            raw=data,
        )


class WebSocketClient:
    """
    WebSocket client for SloughGPT ``/ws/generate``.

    The server expects:
    1. First message: ``{"api_key": "..."}`` or ``{"token": "<jwt>"}``
    2. Then each generation: ``{"prompt": str, "max_tokens": int, "temperature": float}``

    Example::

        ws = WebSocketClient("http://localhost:8000", api_key="your-key")
        ws.connect()
        ws.on("token", lambda m: print(m.data, end="", flush=True))
        ws.on("complete", lambda m: print())
        ws.send_generate("Hello", max_new_tokens=50)
        # ... wait for stream ...
        ws.close()
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        jwt_token: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/").replace("http://", "ws://").replace("https://", "wss://")
        self.api_key = api_key
        self.jwt_token = jwt_token
        self._ws: Any = None
        self._connected = False
        self._handlers: Dict[str, List[Callable]] = {}
        self._recv_thread: Optional[threading.Thread] = None
        self._recv_stop = threading.Event()

    def connect(self, timeout: int = 30) -> bool:
        """Connect and complete auth handshake."""
        try:
            import websocket
        except ImportError:
            raise ImportError("websocket-client package required: pip install websocket-client") from None

        if not self.api_key and not self.jwt_token:
            raise ValueError("/ws/generate requires api_key or jwt_token")

        url = f"{self.base_url}/ws/generate"
        self._ws = websocket.create_connection(url, timeout=timeout)

        auth: Dict[str, str] = (
            {"api_key": self.api_key} if self.api_key else {"token": self.jwt_token or ""}
        )
        self._ws.send(json.dumps(auth))
        resp = json.loads(self._ws.recv())
        if resp.get("status") != "authenticated":
            err = resp.get("error", str(resp))
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
            raise ConnectionError(err)

        self._connected = True
        self._recv_stop.clear()
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()
        return True

    def _recv_loop(self) -> None:
        while self._connected and self._ws is not None and not self._recv_stop.is_set():
            try:
                raw = self._ws.recv()
                if not raw:
                    continue
                data = json.loads(raw)
                self._dispatch(data)
            except Exception:
                self._connected = False
                break

    def _dispatch(self, data: Dict[str, Any]) -> None:
        if data.get("status") == "error" and "token" not in data:
            msg = WebSocketMessage(type="error", data=data.get("error", ""), raw=data)
            for h in self._handlers.get("error", []) + self._handlers.get("*", []):
                h(msg)
            return

        if "token" in data:
            msg = WebSocketMessage(type="token", data=data.get("token"), raw=data)
            for h in self._handlers.get("token", []) + self._handlers.get("*", []):
                h(msg)

        if data.get("done") is True and data.get("status") == "done":
            msg = WebSocketMessage(type="complete", data=data.get("text", ""), raw=data)
            for h in self._handlers.get("complete", []) + self._handlers.get("*", []):
                h(msg)

    def on(self, event_type: str, handler: Callable[[WebSocketMessage], None]) -> None:
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def send_generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Send a generation request (server field: ``max_tokens``)."""
        if not self._connected or not self._ws:
            raise ConnectionError("Not connected. Call connect() first.")

        max_tokens = int(kwargs.pop("max_tokens", max_new_tokens))
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if model is not None:
            payload["model"] = model
        payload.update(kwargs)
        self._ws.send(json.dumps(payload))

    def send_chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> None:
        """Not supported by ``/ws/generate``; formats messages as a single prompt."""
        lines = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            lines.append(f"{role}: {content}")
        prompt = "\n".join(lines) + "\nassistant:"
        self.send_generate(prompt, **kwargs)

    def send_ping(self) -> None:
        if self._connected and self._ws:
            try:
                self._ws.send(json.dumps({"type": "ping"}))
            except Exception:
                pass

    def close(self) -> None:
        self._connected = False
        self._recv_stop.set()
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    @property
    def is_connected(self) -> bool:
        return self._connected


class StreamIterator:
    """Iterator for streaming tokens."""

    def __init__(self, generator: "StreamGenerator"):
        self.generator = generator
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self) -> str:
        import time

        while self._index >= len(self.generator._buffer) and not self.generator._complete:
            time.sleep(0.01)

        if self._index >= len(self.generator._buffer):
            raise StopIteration

        token = self.generator._buffer[self._index]
        self._index += 1
        return token

    def __aiter__(self):
        return self

    async def __anext__(self):
        import asyncio

        while self._index >= len(self.generator._buffer) and not self.generator._complete:
            await asyncio.sleep(0.01)

        if self._index >= len(self.generator._buffer):
            raise StopAsyncIteration

        token = self.generator._buffer[self._index]
        self._index += 1
        return token


class StreamGenerator:
    """
    Context manager for streaming responses via WebSocket.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        jwt_token: Optional[str] = None,
    ):
        self.client = WebSocketClient(base_url, api_key=api_key, jwt_token=jwt_token)
        self._buffer: List[str] = []
        self._complete = False

    def __enter__(self) -> "StreamGenerator":
        self.client.connect()

        def on_token(msg: WebSocketMessage) -> None:
            if msg.type == "token" and msg.data:
                self._buffer.append(str(msg.data))

        def on_complete(_msg: WebSocketMessage) -> None:
            self._complete = True

        self.client.on("token", on_token)
        self.client.on("complete", on_complete)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.client.close()

    def generate(self, prompt: str, **kwargs: Any) -> StreamIterator:
        self._buffer = []
        self._complete = False
        self.client.send_generate(prompt, **kwargs)
        return StreamIterator(self)

    def get_full_text(self) -> str:
        return "".join(self._buffer)
