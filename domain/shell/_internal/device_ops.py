"""Device op tables — the thin layer under every device interface.

A device's ops are its vocabulary: a defined set it implements, reached the
same way no matter which device it is, the way a processor's ISA is a defined
opcode set rather than a per-program dialect.

Before this module every device carried its own copy of the same dispatch
body::

    fn = self._ops.get(method)
    if fn is None:
        raise DeviceFault(f"<Device>: unknown op: {method}")
    return fn(*args)

Eight devices in the shell had that body verbatim, so the "what does a device
accept, and what does it do about an op it has never heard of" answer was
reimplemented per device instead of being a property of the interface.

:class:`OpTable` owns all of it — the registry, the lookup, the unknown-op
fault, the listing. It is a ``Mapping`` so every existing ``self._ops.get(...)``,
``in self._ops`` and ``list(self._ops.keys())`` keeps working unchanged:
adding a device means declaring a table, never copying a dispatcher.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

__all__ = ["OpTable", "dispatch_ops"]


def dispatch_ops(ops: Mapping[str, Callable], owner: str, method: str, *args: Any, **kwargs: Any) -> Any:
    """Run one op from a table, or fault with the owner's name.

    Shared by :class:`OpTable` and by ``vm.Device.call`` so a device that
    still carries a plain ``dict`` of ops dispatches identically to one that
    carries a table — one dispatcher, not one per device.
    """
    fn = ops.get(method)
    if fn is None:
        from .vm import DeviceFault  # deferred: vm.py imports this module

        raise DeviceFault(f"{owner}: unknown op: {method}")
    return fn(*args, **kwargs)


class OpTable(Mapping[str, Callable]):
    """A named, dispatchable set of device operations.

    Args:
        owner: the name that prefixes the unknown-op fault. Devices pass the
            string their fault already used (``"TensorDevice"``,
            ``f"EngineDevice({self._name})"``) so the message a guest sees is
            unchanged by this consolidation.
        ops: initial name -> callable entries.
    """

    def __init__(self, owner: str, ops: dict[str, Callable] | None = None):
        self._owner = owner
        self._ops: dict[str, Callable] = dict(ops or {})

    @property
    def owner(self) -> str:
        return self._owner

    def __getitem__(self, name: str) -> Callable:
        return self._ops[name]

    def __iter__(self):
        return iter(self._ops)

    def __len__(self) -> int:
        return len(self._ops)

    def register(self, name: str, fn: Callable) -> None:
        """Add or replace an op."""
        self._ops[name] = fn

    def unregister(self, name: str) -> bool:
        """Remove an op; True if it was there."""
        return self._ops.pop(name, None) is not None

    def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Dispatch one op by name.

        Unknown ops raise DeviceFault — the type the VM reports as a device
        error — so a guest program gets a fault naming the device and the op
        instead of a leaked AttributeError or a silent None.
        """
        return dispatch_ops(self._ops, self._owner, method, *args, **kwargs)

    def __repr__(self) -> str:
        return f"OpTable({self._owner!r}, {sorted(self._ops)})"
