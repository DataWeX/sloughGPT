"""Tests for inference protocol, engine, and client."""

from __future__ import annotations

import json
import socket
import struct
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from domains.infrastructure.inference_protocol import (
    HEADER_SIZE,
    decode_header,
    encode_message,
)


# ── Protocol wire format ──────────────────────────────────────────────

class TestProtocolWireFormat:
    def test_encode_decode_roundtrip(self):
        msg = {"type": "health"}
        raw = encode_message(msg)
        header = raw[:HEADER_SIZE]
        length = decode_header(header)
        payload = raw[HEADER_SIZE:]
        assert length == len(payload)
        decoded = json.loads(payload)
        assert decoded == msg

    def test_encode_large_message(self):
        msg = {"type": "generate", "prompt": "x" * 100_000}
        raw = encode_message(msg)
        length = decode_header(raw[:HEADER_SIZE])
        assert length == len(raw) - HEADER_SIZE

    def test_header_is_4_bytes(self):
        assert HEADER_SIZE == 4

    def test_decode_header(self):
        raw = struct.pack("!I", 42) + b"test"
        assert decode_header(raw[:4]) == 42

    def test_encode_preserves_json(self):
        msg = {"type": "token", "id": "abc-123", "token": "hello"}
        raw = encode_message(msg)
        decoded = json.loads(raw[HEADER_SIZE:])
        assert decoded["token"] == "hello"
        assert decoded["id"] == "abc-123"


# ── Engine message dispatch ───────────────────────────────────────────

class TestEngineDispatch:
    """Test InferenceEngine message handling without loading a real model."""

    def _make_engine(self):
        from domains.infrastructure.inference_engine import InferenceEngine
        engine = InferenceEngine.__new__(InferenceEngine)
        engine.model_id = "test-model"
        engine.slnc_path = None
        engine.host = "127.0.0.1"
        engine.port = 0
        engine.quantize = False
        engine.quant_bits = 8
        engine.quant_mode = "symmetric"
        engine.quant_clip = 0.999
        engine._provider = None
        engine._server_socket = None
        engine._thread = None
        engine._ready = threading.Event()
        engine._stop = threading.Event()
        engine._active_streams = {}
        engine._active_streams_lock = threading.Lock()
        return engine

    def _send_recv(self, sock, msg):
        sock.sendall(encode_message(msg))
        header = sock.recv(HEADER_SIZE)
        if not header:
            return None
        length = decode_header(header)
        payload = sock.recv(length)
        return json.loads(payload)

    def test_health_when_no_provider(self):
        engine = self._make_engine()
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]

        def _accept():
            client, _ = server.accept()
            msg_raw = client.recv(1024)
            # Decode the message
            length = struct.unpack("!I", msg_raw[:4])[0]
            msg = json.loads(msg_raw[4:4+length])
            # Handle
            engine._handle_health(client, msg)
            client.close()

        t = threading.Thread(target=_accept, daemon=True)
        t.start()

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(("127.0.0.1", port))
        resp = self._send_recv(sock, {"type": "health"})
        sock.close()
        server.close()

        assert resp["type"] == "health_ok"
        assert resp["model_id"] == "test-model"
        assert resp["loaded"] is False

    def test_generate_when_no_provider_returns_error(self):
        engine = self._make_engine()
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]

        def _accept():
            client, _ = server.accept()
            msg_raw = client.recv(4096)
            length = struct.unpack("!I", msg_raw[:4])[0]
            msg = json.loads(msg_raw[4:4+length])
            engine._handle_generate(client, msg)
            client.close()

        t = threading.Thread(target=_accept, daemon=True)
        t.start()

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(("127.0.0.1", port))
        resp = self._send_recv(sock, {
            "type": "generate",
            "id": "req-1",
            "messages": [{"role": "user", "content": "hi"}],
            "params": {"max_new_tokens": 10},
        })
        sock.close()
        server.close()

        assert resp["type"] == "error"
        assert resp["id"] == "req-1"
        assert "NoneType" in resp["message"] or "has no attribute" in resp["message"]

    def test_unknown_message_type(self):
        engine = self._make_engine()
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]

        def _accept():
            client, _ = server.accept()
            msg_raw = client.recv(1024)
            length = struct.unpack("!I", msg_raw[:4])[0]
            msg = json.loads(msg_raw[4:4+length])
            engine._dispatch(client, msg)
            client.close()

        t = threading.Thread(target=_accept, daemon=True)
        t.start()

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(("127.0.0.1", port))
        resp = self._send_recv(sock, {"type": "bogus", "id": "x"})
        sock.close()
        server.close()

        assert resp["type"] == "error"
        assert "Unknown" in resp["message"]

    def test_stream_stop_marks_inactive(self):
        engine = self._make_engine()
        with engine._active_streams_lock:
            engine._active_streams["test-id"] = True
        engine._handle_stream_stop({"id": "test-id"})
        with engine._active_streams_lock:
            assert engine._active_streams.get("test-id") is False


# ── Client reconnection ───────────────────────────────────────────────

class TestInferenceClient:
    def test_client_kv_properties(self):
        from domains.infrastructure.inference_client import InferenceClient
        client = InferenceClient(host="127.0.0.1", port=9999)
        assert client.model_id == "unknown"
        assert client._meta["loaded"] is False
        assert client._kv_max_sessions == 64
        assert client._kv_ttl == 3600.0

    def test_client_capabilities(self):
        from domains.infrastructure.inference_client import InferenceClient
        client = InferenceClient()
        caps = client.capabilities
        assert caps.chat is True
        assert caps.streaming is True
        assert caps.embedding is False

    def test_client_health_when_disconnected(self):
        from domains.infrastructure.inference_client import InferenceClient
        client = InferenceClient(host="127.0.0.1", port=1)
        resp = client.health()
        assert resp.get("type") == "error" or resp == {}

    def test_client_disconnect_is_idempotent(self):
        from domains.infrastructure.inference_client import InferenceClient
        client = InferenceClient()
        client.disconnect()
        client.disconnect()
        assert client._socket is None

    def test_send_recv_exact(self):
        from domains.infrastructure.inference_client import InferenceClient
        client = InferenceClient()
        s1, s2 = socket.socketpair()
        client._socket = s1
        s2.sendall(b"hello")
        result = client._recv_exact(5)
        assert result == b"hello"
        s1.close()
        s2.close()

    def test_recv_exact_short_read(self):
        from domains.infrastructure.inference_client import InferenceClient
        client = InferenceClient()
        s1, s2 = socket.socketpair()
        client._socket = s1
        s2.sendall(b"ab")
        result = client._recv_exact(5)
        assert result is None
        s1.close()
        s2.close()

    def test_recv_message(self):
        from domains.infrastructure.inference_client import InferenceClient
        client = InferenceClient()
        s1, s2 = socket.socketpair()
        client._socket = s1
        msg = {"type": "health_ok", "model_id": "test"}
        raw = encode_message(msg)
        s2.sendall(raw)
        result = client._recv_message()
        assert result["type"] == "health_ok"
        s1.close()
        s2.close()
