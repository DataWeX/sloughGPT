"""Tests for domains.infrastructure.inference_client — InferenceClient.

Covers: connect, disconnect, health, reload, chat, chat_stream, reconnect,
restart callback, send/recv, message framing.
Socket is mocked to avoid real network calls.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from threading import Event
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

_core_dir = str(Path(__file__).resolve().parents[2])
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

from domains.infrastructure.inference_protocol import HEADER_SIZE, encode_message
from domains.infrastructure.inference_client import InferenceClient


def _make_health_ok(model_id="m1", loaded=True):
    return encode_message({"type": "health_ok", "model_id": model_id, "loaded": loaded})


def _make_response(msg: dict):
    return encode_message(msg)


def _mock_socket():
    """Create a mock socket that tracks sent data and queued responses."""
    sock = MagicMock()
    sock.recv = MagicMock(return_value=b"")
    sock.sendall = MagicMock()
    sock.settimeout = MagicMock()
    return sock


class TestInit:
    def test_defaults(self):
        c = InferenceClient()
        assert c.host == "127.0.0.1"
        assert c.port == 9100
        assert c.connect_timeout == 10.0
        assert c.generate_timeout == 120.0
        assert c.model_id == "unknown"

    def test_custom_params(self):
        c = InferenceClient(host="10.0.0.1", port=8080, connect_timeout=5.0, generate_timeout=60.0)
        assert c.host == "10.0.0.1"
        assert c.port == 8080


class TestMeta:
    def test_meta(self):
        c = InferenceClient()
        c._model_id = "gpt2"
        c._loaded = True
        assert c._meta == {"model_id": "gpt2", "loaded": True}


class TestCapabilities:
    def test_capabilities(self):
        c = InferenceClient()
        caps = c.capabilities
        assert caps.chat is True
        assert caps.streaming is True
        assert caps.embedding is False


class TestConnect:
    @patch("socket.socket")
    def test_connect_success(self, mock_cls):
        sock = _mock_socket()
        mock_cls.return_value = sock
        health_resp = _make_health_ok("gpt2", True)
        header = health_resp[:HEADER_SIZE]
        payload = health_resp[HEADER_SIZE:]
        sock.recv.side_effect = [header, payload]

        c = InferenceClient()
        assert c.connect() is True
        assert c.model_id == "gpt2"
        assert c._loaded is True

    @patch("socket.socket")
    def test_connect_failure_bad_response(self, mock_cls):
        sock = _mock_socket()
        mock_cls.return_value = sock
        bad_resp = _make_response({"type": "error"})
        header = bad_resp[:HEADER_SIZE]
        payload = bad_resp[HEADER_SIZE:]
        sock.recv.side_effect = [header, payload]

        c = InferenceClient()
        assert c.connect() is False

    @patch("socket.socket")
    def test_connect_failure_exception(self, mock_cls):
        sock = _mock_socket()
        mock_cls.return_value = sock
        sock.connect.side_effect = OSError("refused")

        c = InferenceClient()
        assert c.connect() is False
        assert c._socket is None


class TestDisconnect:
    def test_disconnect_closes_socket(self):
        c = InferenceClient()
        sock = _mock_socket()
        c._socket = sock
        c.disconnect()
        sock.close.assert_called_once()
        assert c._socket is None

    def test_disconnect_no_socket(self):
        c = InferenceClient()
        c.disconnect()  # should not raise


class TestHealth:
    @patch("socket.socket")
    def test_health_ok(self, mock_cls):
        sock = _mock_socket()
        mock_cls.return_value = sock
        resp = _make_response({"type": "health_ok", "model_id": "m1", "loaded": True})
        sock.recv.side_effect = [resp[:HEADER_SIZE], resp[HEADER_SIZE:]]

        c = InferenceClient()
        c._socket = sock
        result = c.health()
        assert result["type"] == "health_ok"

    def test_health_no_socket(self):
        c = InferenceClient()
        result = c.health()
        assert result == {}


class TestReload:
    @patch("socket.socket")
    def test_reload_ok(self, mock_cls):
        sock = _mock_socket()
        mock_cls.return_value = sock
        resp = _make_response({"type": "reload_ok", "model_id": "new_model"})
        sock.recv.side_effect = [resp[:HEADER_SIZE], resp[HEADER_SIZE:]]

        c = InferenceClient()
        c._socket = sock
        result = c.reload("new_model", "/path/to/model.slnc")
        assert result["type"] == "reload_ok"

    def test_reload_no_socket(self):
        c = InferenceClient()
        result = c.reload("m1")
        assert result == {}


class TestChat:
    @pytest.mark.asyncio
    async def test_chat_success(self):
        c = InferenceClient()
        c._send_and_recv = MagicMock(return_value={"type": "generate_done", "text": "hello world"})
        result = await c.chat([{"role": "user", "content": "hi"}])
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_chat_no_response(self):
        c = InferenceClient()
        c._send_and_recv = MagicMock(return_value=None)
        result = await c.chat([{"role": "user", "content": "hi"}])
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_chat_error_response(self):
        c = InferenceClient()
        c._send_and_recv = MagicMock(return_value={"type": "error", "message": "oom"})
        result = await c.chat([{"role": "user", "content": "hi"}])
        assert "oom" in result


class TestSendRecv:
    def test_send_and_recv_no_socket(self):
        c = InferenceClient()
        c._socket = None
        result = c._send_and_recv({"type": "health"})
        assert result is None

    @patch("socket.socket")
    def test_send_and_recv_success(self, mock_cls):
        sock = _mock_socket()
        mock_cls.return_value = sock
        resp = _make_response({"type": "ok"})
        sock.recv.side_effect = [resp[:HEADER_SIZE], resp[HEADER_SIZE:]]

        c = InferenceClient()
        c._socket = sock
        result = c._send_and_recv({"type": "test"})
        assert result == {"type": "ok"}

    def test_send_and_recv_reconnect_success(self):
        c = InferenceClient()
        c._socket = MagicMock()
        c._socket.sendall.side_effect = OSError("broken pipe")
        c._try_reconnect = MagicMock(return_value=True)
        result = c._send_and_recv({"type": "test"})
        assert result is None

    def test_send_and_recv_reconnect_fails(self):
        c = InferenceClient()
        c._socket = MagicMock()
        c._socket.sendall.side_effect = OSError("broken pipe")
        c._try_reconnect = MagicMock(return_value=False)
        result = c._send_and_recv({"type": "test"})
        assert result is None


class TestReconnect:
    @patch("socket.socket")
    def test_reconnect_direct_success(self, mock_cls):
        sock = _mock_socket()
        mock_cls.return_value = sock
        health = _make_health_ok("m2")
        sock.recv.side_effect = [health[:HEADER_SIZE], health[HEADER_SIZE:]]

        c = InferenceClient(host="10.0.0.1", port=9100)
        c._socket = None
        assert c._try_reconnect() is True
        assert c.model_id == "m2"

    def test_reconnect_with_restart_callback(self):
        c = InferenceClient()
        c._socket = MagicMock()
        c._socket.connect.side_effect = OSError("refused")

        new_client = InferenceClient(host="new", port=9999)
        new_client._model_id = "restarted"
        new_client._loaded = True
        new_client._socket = _mock_socket()

        c._restart_fn = MagicMock(return_value=new_client)
        assert c._try_reconnect() is True
        assert c.model_id == "restarted"

    @patch("socket.socket")
    def test_reconnect_all_fail(self, mock_cls):
        c = InferenceClient()
        sock = _mock_socket()
        mock_cls.return_value = sock
        sock.connect.side_effect = OSError("refused")
        assert c._try_reconnect() is False


class TestMessageFraming:
    def test_send_message(self):
        c = InferenceClient()
        c._socket = _mock_socket()
        c._send_message({"type": "test", "data": 42})
        c._socket.sendall.assert_called_once()
        sent = c._socket.sendall.call_args[0][0]
        assert len(sent) >= HEADER_SIZE

    def test_recv_message_success(self):
        c = InferenceClient()
        msg = {"type": "response", "text": "hello"}
        encoded = _make_response(msg)
        sock = _mock_socket()
        sock.recv.side_effect = [encoded[:HEADER_SIZE], encoded[HEADER_SIZE:]]
        c._socket = sock
        result = c._recv_message()
        assert result == msg

    def test_recv_message_none_header(self):
        c = InferenceClient()
        sock = _mock_socket()
        sock.recv.return_value = b""
        c._socket = sock
        result = c._recv_message()
        assert result is None

    def test_recv_exact(self):
        c = InferenceClient()
        data = b"hello world"
        sock = _mock_socket()
        sock.recv.side_effect = [data[:5], data[5:]]
        c._socket = sock
        result = c._recv_exact(len(data))
        assert result == data

    def test_recv_exact_disconnect(self):
        c = InferenceClient()
        sock = _mock_socket()
        sock.recv.return_value = b""
        c._socket = sock
        result = c._recv_exact(10)
        assert result is None
