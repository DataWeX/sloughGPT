"""Inference client — connects to a standalone InferenceEngine.

Implements the ModelProvider protocol so it can be registered as
"slonet-native" in the provider registry. The API server uses this
client instead of the in-process SloNetChatProvider.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import threading
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

from domains.infrastructure.inference_protocol import (
    HEADER_SIZE,
    decode_header,
    encode_message,
)

logger = logging.getLogger(__name__)


class InferenceClient:
    """Client that connects to a standalone InferenceEngine via TCP.

    Args:
        host: Engine host address
        port: Engine port
        connect_timeout: Seconds to wait for connection
        generate_timeout: Seconds to wait for a generate response
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9100,
        connect_timeout: float = 10.0,
        generate_timeout: float = 120.0,
        restart_fn=None,
    ):
        self.host = host
        self.port = port
        self.connect_timeout = connect_timeout
        self.generate_timeout = generate_timeout
        self._restart_fn = restart_fn

        self._socket: Optional[socket.socket] = None
        self._lock = threading.Lock()
        self._model_id = "unknown"
        self._loaded = False

        self._kv_states: Dict[str, Any] = {}
        self._kv_last_access: Dict[str, float] = {}
        self._kv_lock = threading.Lock()
        self._kv_max_sessions = 64
        self._kv_ttl = 3600.0

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def _meta(self) -> dict:
        return {"model_id": self._model_id, "loaded": self._loaded}

    @property
    def capabilities(self):
        from domains.models.provider import ModelCapabilities
        return ModelCapabilities(chat=True, streaming=True, embedding=False, vision=False)

    def connect(self) -> bool:
        """Connect to the inference engine. Returns True if successful."""
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(self.connect_timeout)
            self._socket.connect((self.host, self.port))
            self._socket.settimeout(None)
            resp = self._send_and_recv({"type": "health"})
            if resp and resp.get("type") == "health_ok":
                self._model_id = resp.get("model_id", "unknown")
                self._loaded = resp.get("loaded", False)
                logger.info(
                    "InferenceClient: connected (model=%s, loaded=%s)",
                    self._model_id, self._loaded,
                )
                return True
            return False
        except Exception as e:
            logger.warning("InferenceClient: connection failed: %s", e)
            self._socket = None
            return False

    def disconnect(self) -> None:
        """Disconnect from the engine."""
        if self._socket is not None:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None

    def health(self) -> dict:
        """Synchronous health check."""
        try:
            return self._send_and_recv({"type": "health"}) or {}
        except Exception as e:
            return {"type": "error", "message": str(e)}

    def reload(self, model_id: str, slnc_path: Optional[str] = None) -> dict:
        """Hot-reload: swap the model in the engine without restarting."""
        try:
            return self._send_and_recv({
                "type": "reload",
                "model_id": model_id,
                "slnc_path": slnc_path,
            }, timeout=300.0) or {}
        except Exception as e:
            return {"type": "error", "message": str(e)}

    async def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.8,
        **kwargs,
    ) -> str:
        """Non-streaming chat — returns complete response string."""
        req_id = str(uuid.uuid4())
        msg = {
            "type": "generate",
            "id": req_id,
            "messages": messages,
            "params": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_k": kwargs.get("top_k"),
                "top_p": kwargs.get("top_p"),
                "repetition_penalty": kwargs.get("repetition_penalty", 1.0),
                "session_id": kwargs.get("session_id"),
            },
        }
        resp = await asyncio.to_thread(self._send_and_recv, msg, self.generate_timeout)
        if resp is None:
            return "[Error: No response from inference engine]"
        if resp.get("type") == "error":
            return f"[Error: {resp.get('message', 'unknown')}]"
        return resp.get("text", "")

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.8,
        cancel_event=None,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Streaming chat — yields token strings."""
        req_id = str(uuid.uuid4())

        msg = {
            "type": "stream_start",
            "id": req_id,
            "messages": messages,
            "params": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_k": kwargs.get("top_k"),
                "top_p": kwargs.get("top_p"),
                "repetition_penalty": kwargs.get("repetition_penalty", 1.0),
                "session_id": session_id,
            },
        }
        self._send_message(msg)

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()
        done = asyncio.Event()

        def _reader():
            try:
                while not done.is_set():
                    resp = self._recv_message()
                    if resp is None:
                        break
                    rtype = resp.get("type")
                    if rtype == "token" and resp.get("id") == req_id:
                        loop.call_soon_threadsafe(queue.put_nowait, resp.get("token", ""))
                    elif rtype == "stream_done" and resp.get("id") == req_id:
                        loop.call_soon_threadsafe(queue.put_nowait, None)
                        break
                    elif rtype == "error" and resp.get("id") == req_id:
                        loop.call_soon_threadsafe(
                            queue.put_nowait, f"[Error: {resp.get('message', '')}]"
                        )
                        break
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, f"[Error: {e}]")
            finally:
                loop.call_soon_threadsafe(done.set)

        reader = threading.Thread(target=_reader, daemon=True, name=f"client-reader-{req_id[:8]}")
        reader.start()

        try:
            while True:
                token = await queue.get()
                if token is None:
                    break
                if cancel_event is not None and cancel_event.is_set():
                    try:
                        self._send_message({"type": "stream_stop", "id": req_id})
                    except Exception:
                        pass
                    break
                yield token
        finally:
            done.set()

    def _send_and_recv(self, msg: dict, timeout: float = 30.0) -> Optional[dict]:
        """Send a message and wait for a response (thread-safe).

        If the connection is lost, attempts one reconnect before failing.
        """
        with self._lock:
            if self._socket is None:
                self._try_reconnect()
            if self._socket is None:
                return None
            try:
                self._send_message(msg)
                self._socket.settimeout(timeout)
                try:
                    return self._recv_message()
                finally:
                    self._socket.settimeout(None)
            except Exception as e:
                logger.warning("InferenceClient: send/recv error: %s", e)
                self.disconnect()
                # One reconnect attempt
                if self._try_reconnect():
                    try:
                        self._send_message(msg)
                        self._socket.settimeout(timeout)
                        try:
                            return self._recv_message()
                        finally:
                            self._socket.settimeout(None)
                    except Exception:
                        self.disconnect()
                return None

    def _try_reconnect(self) -> bool:
        """Attempt to reconnect to the engine. Must hold _lock."""
        self.disconnect()
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(self.connect_timeout)
            self._socket.connect((self.host, self.port))
            self._socket.settimeout(None)
            resp = self._send_and_recv_unlocked({"type": "health"})
            if resp and resp.get("type") == "health_ok":
                self._model_id = resp.get("model_id", "unknown")
                self._loaded = resp.get("loaded", False)
                logger.info("InferenceClient: reconnected (model=%s)", self._model_id)
                return True
        except Exception as e:
            logger.warning("InferenceClient: reconnect failed: %s", e)
            self._socket = None
        # If direct reconnect failed and we have a restart callback, try restarting the engine
        if self._restart_fn is not None:
            logger.info("InferenceClient: attempting engine restart via callback")
            try:
                new_client = self._restart_fn()
                if new_client is not None:
                    self.host = new_client.host
                    self.port = new_client.port
                    self._socket = new_client._socket
                    new_client._socket = None
                    self._model_id = new_client._model_id
                    self._loaded = new_client._loaded
                    logger.info("InferenceClient: engine restarted and reconnected")
                    return True
            except Exception as e:
                logger.warning("InferenceClient: restart callback failed: %s", e)
        return False

    def _send_and_recv_unlocked(self, msg: dict) -> Optional[dict]:
        """Send/recv without lock (for use inside _try_reconnect which holds lock)."""
        try:
            self._send_message(msg)
            return self._recv_message()
        except Exception:
            return None

    def _send_message(self, msg: dict) -> None:
        """Send a length-prefixed JSON message."""
        data = encode_message(msg)
        self._socket.sendall(data)

    def _recv_message(self) -> Optional[dict]:
        """Read a length-prefixed JSON message from the socket."""
        header = self._recv_exact(HEADER_SIZE)
        if header is None:
            return None
        length = decode_header(header)
        if length > 100 * 1024 * 1024:
            return None
        payload = self._recv_exact(length)
        if payload is None:
            return None
        import json
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None

    def _recv_exact(self, n: int) -> Optional[bytes]:
        """Read exactly n bytes from the socket."""
        buf = bytearray()
        while len(buf) < n:
            try:
                chunk = self._socket.recv(n - len(buf))
            except (OSError, ValueError):
                return None
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)
