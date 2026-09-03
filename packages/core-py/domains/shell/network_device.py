"""
NetworkDevice — standalone network hardware.

Socket operations with clean ioctl interface.
"""

from __future__ import annotations

import socket
import time
from typing import Any

from .kernel_syscall import SyscallResult


class NetworkDevice:
    """Standalone network hardware — socket operations.

    Has clean ioctl interface for assembly.
    Has function calls for direct use.
    """

    def __init__(self, name: str = "network"):
        self._name = name
        self._ops = {
            "TCP_CONNECT": self._tcp_connect,
            "TCP_LISTEN": self._tcp_listen,
            "TCP_ACCEPT": self._tcp_accept,
            "TCP_SEND": self._tcp_send,
            "TCP_RECV": self._tcp_recv,
            "TCP_CLOSE": self._tcp_close,
            "UDP_SEND": self._udp_send,
            "UDP_RECV": self._udp_recv,
            "DNS_RESOLVE": self._dns_resolve,
            "HTTP_GET": self._http_get,
            "HTTP_POST": self._http_post,
            "INFO": self._info,
        }
        self._sockets: dict[int, socket.socket] = {}
        self._next_fd: int = 1

    @property
    def name(self) -> str:
        return self._name

    def info(self) -> dict:
        return {
            "name": self._name,
            "type": "network",
            "open_sockets": len(self._sockets),
        }

    def call(self, method: str, *args: Any) -> Any:
        """VM Device interface — delegates to ioctl."""
        result = self.ioctl(method, *args)
        if result.success:
            return result.value
        raise Exception(result.error)

    # ── ioctl interface ───────────────────────────────────────────────────

    def ioctl(self, command: str, *args: Any) -> SyscallResult:
        """Clean ioctl interface — type-safe, documented."""
        try:
            fn = self._ops.get(command)
            if fn is None:
                return SyscallResult.fail(f"unknown command: {command}")
            result = fn(*args)
            return SyscallResult.ok(result)
        except Exception as e:
            return SyscallResult.fail(f"ioctl error: {e}")

    def list_commands(self) -> list[str]:
        """List all available commands."""
        return sorted(self._ops.keys())

    # ── Function calls (direct use) ───────────────────────────────────────

    def tcp_connect(self, host: str, port: int) -> int:
        """TCP connect, return fd."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        fd = self._next_fd
        self._next_fd += 1
        self._sockets[fd] = sock
        return fd

    def tcp_listen(self, host: str, port: int, backlog: int = 5) -> int:
        """TCP listen, return fd."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen(backlog)
        fd = self._next_fd
        self._next_fd += 1
        self._sockets[fd] = sock
        return fd

    def tcp_accept(self, fd: int) -> tuple[int, tuple[str, int]]:
        """TCP accept, return (new_fd, (host, port))."""
        if fd not in self._sockets:
            raise ValueError(f"bad fd: {fd}")
        sock, addr = self._sockets[fd].accept()
        new_fd = self._next_fd
        self._next_fd += 1
        self._sockets[new_fd] = sock
        return new_fd, addr

    def tcp_send(self, fd: int, data: bytes) -> int:
        """TCP send, return bytes sent."""
        if fd not in self._sockets:
            raise ValueError(f"bad fd: {fd}")
        return self._sockets[fd].send(data)

    def tcp_recv(self, fd: int, size: int = 4096) -> bytes:
        """TCP receive."""
        if fd not in self._sockets:
            raise ValueError(f"bad fd: {fd}")
        return self._sockets[fd].recv(size)

    def tcp_close(self, fd: int) -> bool:
        """TCP close."""
        if fd not in self._sockets:
            return False
        self._sockets[fd].close()
        del self._sockets[fd]
        return True

    def udp_send(self, host: str, port: int, data: bytes) -> int:
        """UDP send, return bytes sent."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sent = sock.sendto(data, (host, port))
        sock.close()
        return sent

    def udp_recv(self, host: str, port: int, size: int = 4096) -> bytes:
        """UDP receive."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        data, _ = sock.recvfrom(size)
        sock.close()
        return data

    def dns_resolve(self, hostname: str) -> str:
        """DNS resolve hostname to IP."""
        return socket.gethostbyname(hostname)

    def http_get(self, url: str) -> bytes:
        """Simple HTTP GET."""
        import urllib.request
        with urllib.request.urlopen(url) as response:
            return response.read()

    def http_post(self, url: str, data: bytes) -> bytes:
        """Simple HTTP POST."""
        import urllib.request
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req) as response:
            return response.read()

    # ── Private methods (ioctl handlers) ──────────────────────────────────

    def _tcp_connect(self, *args):
        return self.tcp_connect(args[0], args[1])

    def _tcp_listen(self, *args):
        host, port = args[0], args[1]
        backlog = args[2] if len(args) > 2 else 5
        return self.tcp_listen(host, port, backlog)

    def _tcp_accept(self, *args):
        return self.tcp_accept(args[0])

    def _tcp_send(self, *args):
        return self.tcp_send(args[0], args[1])

    def _tcp_recv(self, *args):
        fd = args[0]
        size = args[1] if len(args) > 1 else 4096
        return self.tcp_recv(fd, size)

    def _tcp_close(self, *args):
        return self.tcp_close(args[0])

    def _udp_send(self, *args):
        return self.udp_send(args[0], args[1], args[2])

    def _udp_recv(self, *args):
        size = args[2] if len(args) > 2 else 4096
        return self.udp_recv(args[0], args[1], size)

    def _dns_resolve(self, *args):
        return self.dns_resolve(args[0])

    def _http_get(self, *args):
        return self.http_get(args[0])

    def _http_post(self, *args):
        return self.http_post(args[0], args[1])

    def _info(self, *args):
        return self.info()
