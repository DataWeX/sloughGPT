"""
AI-Native Kernel Core — DEPRECATED.

This module is now a re-export shim. The unified Kernel class lives in kernel.py.

Migration:
    from domains.shell.kernel import Kernel
"""
from .kernel import Kernel, NullDevice, get_kernel, reset_kernel  # noqa: F401
