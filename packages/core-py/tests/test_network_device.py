"""Tests for shell.network_device — NetworkDevice socket operations and ioctl."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from domains.shell.network_device import NetworkDevice
from domains.shell.kernel_syscall import SyscallResult


@pytest.fixture
def dev():
    return NetworkDevice(name="test-net")


# ── Basics ────────────────────────────────────────────────────────────────


class TestNetworkDeviceBasics:

    def test_name(self, dev):
        assert dev.name == "test-net"

    def test_default_name(self):
        d = NetworkDevice()
        assert d.name == "network"

    def test_info(self, dev):
        info = dev.info()
        assert info["name"] == "test-net"
        assert info["type"] == "network"
        assert info["open_sockets"] == 0

    def test_list_commands(self, dev):
        cmds = dev.list_commands()
        assert "TCP_CONNECT" in cmds
        assert "TCP_LISTEN" in cmds
        assert "TCP_SEND" in cmds
        assert "DNS_RESOLVE" in cmds
        assert "HTTP_GET" in cmds
        assert "INFO" in cmds
        assert cmds == sorted(cmds)


# ── ioctl ─────────────────────────────────────────────────────────────────


class TestNetworkDeviceIoctl:

    def test_ioctl_unknown_command(self, dev):
        result = dev.ioctl("NONEXISTENT")
        assert isinstance(result, SyscallResult)
        assert not result.success
        assert "unknown command" in result.error

    def test_ioctl_tcp_connect(self, dev):
        with patch.object(dev, "tcp_connect", return_value=1):
            result = dev.ioctl("TCP_CONNECT", "localhost", 80)
            assert result.success
            assert result.value == 1

    def test_ioctl_tcp_listen(self, dev):
        with patch.object(dev, "tcp_listen", return_value=1):
            result = dev.ioctl("TCP_LISTEN", "0.0.0.0", 8080)
            assert result.success

    def test_ioctl_tcp_send(self, dev):
        with patch.object(dev, "tcp_send", return_value=5):
            result = dev.ioctl("TCP_SEND", 1, b"hello")
            assert result.success
            assert result.value == 5

    def test_ioctl_tcp_recv(self, dev):
        with patch.object(dev, "tcp_recv", return_value=b"data"):
            result = dev.ioctl("TCP_RECV", 1)
            assert result.success

    def test_ioctl_tcp_close(self, dev):
        with patch.object(dev, "tcp_close", return_value=True):
            result = dev.ioctl("TCP_CLOSE", 1)
            assert result.success

    def test_ioctl_dns_resolve(self, dev):
        with patch.object(dev, "dns_resolve", return_value="1.2.3.4"):
            result = dev.ioctl("DNS_RESOLVE", "example.com")
            assert result.success
            assert result.value == "1.2.3.4"

    def test_ioctl_http_get(self, dev):
        with patch.object(dev, "http_get", return_value=b"html"):
            result = dev.ioctl("HTTP_GET", "http://example.com")
            assert result.success

    def test_ioctl_http_post(self, dev):
        with patch.object(dev, "http_post", return_value=b"ok"):
            result = dev.ioctl("HTTP_POST", "http://example.com", b"data")
            assert result.success

    def test_ioctl_exception(self, dev):
        with patch.object(dev, "tcp_connect", side_effect=RuntimeError("boom")):
            result = dev.ioctl("TCP_CONNECT", "localhost", 80)
            assert not result.success
            assert "ioctl error" in result.error


# ── call interface ────────────────────────────────────────────────────────


class TestNetworkDeviceCall:

    def test_call_success(self, dev):
        with patch.object(dev, "dns_resolve", return_value="1.2.3.4"):
            assert dev.call("DNS_RESOLVE", "example.com") == "1.2.3.4"

    def test_call_failure_raises(self, dev):
        with pytest.raises(Exception, match="unknown command"):
            dev.call("NONEXISTENT")


# ── Socket operations ─────────────────────────────────────────────────────


class TestNetworkDeviceSockets:

    def test_tcp_connect(self, dev):
        with patch("socket.socket") as mock_cls:
            mock_sock = MagicMock()
            mock_cls.return_value = mock_sock
            fd = dev.tcp_connect("localhost", 80)
            assert fd == 1
            mock_sock.connect.assert_called_once_with(("localhost", 80))
            assert fd in dev._sockets

    def test_tcp_listen(self, dev):
        with patch("socket.socket") as mock_cls:
            mock_sock = MagicMock()
            mock_cls.return_value = mock_sock
            fd = dev.tcp_listen("0.0.0.0", 8080)
            assert fd == 1
            mock_sock.bind.assert_called_once_with(("0.0.0.0", 8080))
            mock_sock.listen.assert_called_once()

    def test_tcp_send(self, dev):
        mock_sock = MagicMock()
        mock_sock.send.return_value = 5
        dev._sockets[1] = mock_sock
        result = dev.tcp_send(1, b"hello")
        assert result == 5

    def test_tcp_send_bad_fd(self, dev):
        with pytest.raises(ValueError, match="bad fd"):
            dev.tcp_send(99, b"data")

    def test_tcp_recv(self, dev):
        mock_sock = MagicMock()
        mock_sock.recv.return_value = b"data"
        dev._sockets[1] = mock_sock
        result = dev.tcp_recv(1)
        assert result == b"data"

    def test_tcp_recv_bad_fd(self, dev):
        with pytest.raises(ValueError, match="bad fd"):
            dev.tcp_recv(99)

    def test_tcp_close(self, dev):
        mock_sock = MagicMock()
        dev._sockets[1] = mock_sock
        assert dev.tcp_close(1) is True
        mock_sock.close.assert_called_once()
        assert 1 not in dev._sockets

    def test_tcp_close_bad_fd(self, dev):
        assert dev.tcp_close(99) is False

    def test_dns_resolve(self, dev):
        with patch("socket.gethostbyname", return_value="1.2.3.4"):
            assert dev.dns_resolve("example.com") == "1.2.3.4"

    def test_http_get(self, dev):
        mock_response = MagicMock()
        mock_response.read.return_value = b"html"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_response):
            result = dev.http_get("http://example.com")
            assert result == b"html"

    def test_http_post(self, dev):
        mock_response = MagicMock()
        mock_response.read.return_value = b"ok"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_response):
            result = dev.http_post("http://example.com", b"data")
            assert result == b"ok"
