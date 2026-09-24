"""Socket / port helpers."""

from __future__ import annotations

import socket

__all__ = ["find_available_port"]


def find_available_port(host: str = "", start_port: int = 8000, max_attempts: int = 10) -> int:
    """Find a free TCP port at or after *start_port*.

    Args:
        host: Bind host (empty string for all interfaces)
        start_port: First port to try
        max_attempts: How many consecutive ports to probe

    Returns:
        An available port number

    Raises:
        RuntimeError: If no port in the range could be bound
    """
    for port in range(start_port, start_port + max_attempts):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            sock.close()
            return port
        except OSError:
            continue
    raise RuntimeError(
        f"Could not find available port in range {start_port}-{start_port + max_attempts}"
    )
