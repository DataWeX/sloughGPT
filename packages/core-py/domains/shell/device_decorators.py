"""
Device decorator — adds ioctl to any device without inheritance.

Usage:
    @with_ioctl({
        "MATMUL": "matmul",
        "RELU": "relu",
        "SOFTMAX": "softmax",
    })
    class TensorDevice:
        def matmul(self, a, b): ...
        def relu(self, a): ...
        def softmax(self, a): ...

    # Now TensorDevice has ioctl
    dev = TensorDevice()
    dev.ioctl("MATMUL", a, b)  # calls dev.matmul(a, b)
"""

from __future__ import annotations
from typing import Any, Callable


def with_ioctl(command_map: dict[str, str]):
    """Decorator that adds ioctl to a device class.

    Args:
        command_map: Maps ioctl command names to method names.
                     Example: {"MATMUL": "matmul", "RELU": "relu"}
    """
    def decorator(cls):
        def ioctl(self, command: str, *args: Any) -> Any:
            method_name = command_map.get(command)
            if method_name is None:
                raise ValueError(f"unknown command: {command}")
            method = getattr(self, method_name, None)
            if method is None:
                raise ValueError(f"method not found: {method_name}")
            return method(*args)

        def list_commands(self) -> list[str]:
            return sorted(command_map.keys())

        cls.ioctl = ioctl
        cls.list_commands = list_commands
        return cls

    return decorator


def add_ioctl_command(cls, command: str, method_name: str):
    """Add a single command to an existing device class."""
    if not hasattr(cls, '_ioctl_commands'):
        cls._ioctl_commands = {}

    cls._ioctl_commands[command] = method_name

    # Add ioctl if not exists
    if not hasattr(cls, 'ioctl'):
        def ioctl(self, command: str, *args: Any) -> Any:
            if not hasattr(self, '_ioctl_commands'):
                raise ValueError("no commands registered")
            method_name = self._ioctl_commands.get(command)
            if method_name is None:
                raise ValueError(f"unknown command: {command}")
            method = getattr(self, method_name, None)
            if method is None:
                raise ValueError(f"method not found: {method_name}")
            return method(*args)

        cls.ioctl = ioctl

    # Add list_commands if not exists
    if not hasattr(cls, 'list_commands'):
        def list_commands(self) -> list[str]:
            return sorted(self._ioctl_commands.keys()) if hasattr(self, '_ioctl_commands') else []

        cls.list_commands = list_commands

    return cls
