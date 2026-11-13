"""Addon protocol for kernel extensions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..kernel import Kernel


@runtime_checkable
class Addon(Protocol):
    """Protocol for kernel addons.

    Any module with a setup(kernel) function satisfies this protocol.
    """

    def setup(self, kernel: Kernel) -> None: ...
