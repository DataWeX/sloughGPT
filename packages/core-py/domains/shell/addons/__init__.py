from __future__ import annotations

"""
Shell Kernel Addons.

Each addon is a module with a setup(kernel) function that registers
capabilities on the kernel. Addons are installed via kernel.install_addon().

Usage:
    from domains.shell.addons import neural, filesystem, shell_ui
    kernel.install_addon(neural)
    kernel.install_addon(filesystem)
    kernel.install_addon(shell_ui)
"""

from .base import Addon  # noqa: F401
from . import neural  # noqa: F401
from . import filesystem  # noqa: F401
from . import shell_ui  # noqa: F401

__all__ = ["Addon", "neural", "filesystem", "shell_ui"]
