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
        s2.close()
        result = client._recv_exact(5)
        assert result is None
        s1.close()

    def test_recv_message(self):
        from domains.infrastructure.inference_client import InferenceClient
        client = InferenceClient()
        msg = {"type": "health_ok", "model_id": "test"}
        raw = encode_message(msg)

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]

        def _send():
            conn, _ = server.accept()
            conn.sendall(raw)
            time.sleep(0.1)
            conn.close()

        t = threading.Thread(target=_send, daemon=True)
        t.start()

        client._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client._socket.connect(("127.0.0.1", port))
        result = client._recv_message()
        client._socket.close()
        server.close()
        t.join(timeout=2)

        assert result["type"] == "health_ok"


# ── Concurrent connections ────────────────────────────────────────────

class TestEngineConcurrency:
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
        engine._pid_file = None
        return engine

    def test_concurrent_health_requests(self):
        engine = self._make_engine()
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(8)
        port = server.getsockname()[1]

        def _handler():
            while not engine._stop.is_set():
                try:
                    client, _ = server.accept()
                except OSError:
                    break
                msg_raw = client.recv(1024)
                if not msg_raw:
                    client.close()
                    continue
                length = struct.unpack("!I", msg_raw[:4])[0]
                msg = json.loads(msg_raw[4:4+length])
                engine._handle_health(client, msg)
                client.close()

        t = threading.Thread(target=_handler, daemon=True)
        t.start()

        results = []

        def _do_health(i):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(("127.0.0.1", port))
            sock.sendall(encode_message({"type": "health"}))
            header = sock.recv(4)
            length = struct.unpack("!I", header)[0]
            payload = sock.recv(length)
            resp = json.loads(payload)
            results.append((i, resp["type"]))
            sock.close()

        threads = [threading.Thread(target=_do_health, args=(i,)) for i in range(5)]
        for t2 in threads:
            t2.start()
        for t2 in threads:
            t2.join(timeout=5)

        engine._stop.set()
        server.close()
        t.join(timeout=2)

        assert len(results) == 5
        assert all(r[1] == "health_ok" for r in results)

    def test_active_streams_tracking(self):
        engine = self._make_engine()
        with engine._active_streams_lock:
            engine._active_streams["s1"] = True
            engine._active_streams["s2"] = True
        assert len(engine._active_streams) == 2

        engine._handle_stream_stop({"id": "s1"})
        with engine._active_streams_lock:
            assert engine._active_streams["s1"] is False
            assert engine._active_streams["s2"] is True

        engine._handle_stream_stop({"id": "s2"})
        with engine._active_streams_lock:
            assert engine._active_streams["s2"] is False


# ── Client reconnection with restart callback ─────────────────────────

class TestClientRestart:
    def test_restart_callback_called_on_failure(self):
        from domains.infrastructure.inference_client import InferenceClient

        call_count = [0]

        def fake_restart():
            call_count[0] += 1
            new_client = InferenceClient.__new__(InferenceClient)
            new_client.host = "127.0.0.1"
            new_client.port = 1
            new_client.connect_timeout = 1.0
            new_client.generate_timeout = 5.0
            new_client._restart_fn = None
            new_client._socket = None
            new_client._lock = threading.Lock()
            new_client._model_id = "restarted"
            new_client._loaded = True
            new_client._kv_states = {}
            new_client._kv_last_access = {}
            new_client._kv_max_sessions = 64
            new_client._kv_ttl = 3600.0
            return None  # restart also fails

        client = InferenceClient(host="127.0.0.1", port=1, restart_fn=fake_restart)
        client._try_reconnect()

        assert call_count[0] == 1

    def test_no_restart_callback(self):
        from domains.infrastructure.inference_client import InferenceClient
        client = InferenceClient(host="127.0.0.1", port=1)
        result = client._try_reconnect()
        assert result is False


# ── Full integration: engine + client over TCP ─────────────────────────

class MockProvider:
    """Minimal provider that simulates generate and chat_stream."""

    def __init__(self):
        self._loaded = True
        self._model_id = "mock-model"

    @property
    def _meta(self):
        return {"quantized": False}

    def _generate_sync(self, messages, max_tokens=512, temperature=0.8,
                       top_k=None, top_p=None, repetition_penalty=1.0,
                       session_id=None):
        prompt = messages[-1]["content"] if messages else ""
        return f"echo:{prompt[:50]}"

    async def chat_stream(self, messages, max_tokens=512, temperature=0.7, **kwargs):
        prompt = messages[-1]["content"] if messages else ""
        for word in f"stream:{prompt[:20]}".split(":")[-1].split():
            yield word + " "


class TestFullIntegration:
    """Spin up an engine with a mock provider, connect a client, run requests."""

    def _start_engine_with_mock(self):
        from domains.infrastructure.inference_engine import InferenceEngine
        engine = InferenceEngine.__new__(InferenceEngine)
        engine.model_id = "mock-model"
        engine.slnc_path = None
        engine.host = "127.0.0.1"
        engine.port = 0
        engine.quantize = False
        engine.quant_bits = 8
        engine.quant_mode = "symmetric"
        engine.quant_clip = 0.999
        engine._provider = MockProvider()
        engine._server_socket = None
        engine._thread = None
        engine._ready = threading.Event()
        engine._stop = threading.Event()
        engine._active_streams = {}
        engine._active_streams_lock = threading.Lock()
        engine._pid_file = None

        # Bind and start accept loop manually
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(4)
        port = server.getsockname()[1]
        engine._server_socket = server

        def _accept_loop():
            engine._ready.set()
            while not engine._stop.is_set():
                try:
                    client, addr = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                t = threading.Thread(
                    target=engine._handle_client, args=(client, addr), daemon=True
                )
                t.start()

        engine._thread = threading.Thread(target=_accept_loop, daemon=True, name="mock-engine")
        engine._thread.start()
        return engine, port

    def test_health_roundtrip(self):
        engine, port = self._start_engine_with_mock()
        try:
            from domains.infrastructure.inference_client import InferenceClient
            client = InferenceClient(host="127.0.0.1", port=port, connect_timeout=2.0)
            assert client.connect() is True
            assert client.model_id == "mock-model"
            health = client.health()
            assert health["type"] == "health_ok"
            assert health["loaded"] is True
            client.disconnect()
        finally:
            engine.stop()

    def test_generate_roundtrip(self):
        engine, port = self._start_engine_with_mock()
        try:
            from domains.infrastructure.inference_client import InferenceClient
            client = InferenceClient(host="127.0.0.1", port=port, connect_timeout=2.0)
            client.connect()
            import asyncio
            result = asyncio.run(client.chat(
                [{"role": "user", "content": "hello world"}],
                max_tokens=10,
            ))
            assert result == "echo:hello world"
            client.disconnect()
        finally:
            engine.stop()

    def test_stream_roundtrip(self):
        engine, port = self._start_engine_with_mock()
        try:
            from domains.infrastructure.inference_client import InferenceClient
            client = InferenceClient(host="127.0.0.1", port=port, connect_timeout=2.0)
            client.connect()
            import asyncio

            async def _collect():
                tokens = []
                async for t in client.chat_stream(
                    [{"role": "user", "content": "test"}],
                    max_tokens=5,
                ):
                    tokens.append(t)
                return tokens

            tokens = asyncio.run(_collect())
            assert len(tokens) > 0
            assert "".join(tokens).strip() != ""
            client.disconnect()
        finally:
            engine.stop()

    def test_multiple_clients(self):
        engine, port = self._start_engine_with_mock()
        try:
            from domains.infrastructure.inference_client import InferenceClient
            import asyncio

            results = []

            def _do_request(i):
                c = InferenceClient(host="127.0.0.1", port=port, connect_timeout=2.0)
                c.connect()
                r = asyncio.run(c.chat(
                    [{"role": "user", "content": f"msg{i}"}],
                    max_tokens=5,
                ))
                results.append((i, r))
                c.disconnect()

            threads = [threading.Thread(target=_do_request, args=(i,)) for i in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

            assert len(results) == 3
            for i, r in results:
                assert r == f"echo:msg{i}"
        finally:
            engine.stop()

    def test_client_auto_reconnect(self):
        engine, port = self._start_engine_with_mock()
        from domains.infrastructure.inference_client import InferenceClient

        # Shared state for restart fn to find the new engine
        new_port = [None]

        def _restart():
            if new_port[0] is None:
                return None
            c = InferenceClient.__new__(InferenceClient)
            c.host = "127.0.0.1"
            c.port = new_port[0]
            c.connect_timeout = 2.0
            c.generate_timeout = 5.0
            c._restart_fn = None
            c._socket = None
            c._lock = threading.Lock()
            c._model_id = "mock-model"
            c._loaded = True
            c._kv_states = {}
            c._kv_last_access = {}
            c._kv_max_sessions = 64
            c._kv_ttl = 3600.0
            if c.connect():
                return c
            return None

        client = InferenceClient(host="127.0.0.1", port=port, connect_timeout=2.0,
                                 restart_fn=_restart)
        client.connect()

        # Kill engine
        engine.stop()
        time.sleep(0.2)

        # Restart on new port
        engine2, port2 = self._start_engine_with_mock()
        new_port[0] = port2
        try:
            import asyncio
            result = asyncio.run(client.chat(
                [{"role": "user", "content": "reconnect"}],
                max_tokens=5,
            ))
            assert result == "echo:reconnect"
        finally:
            client.disconnect()
            engine2.stop()
