"""
TensorArena — device-shared slot registry for VM bus values.

The VM register file holds machine-representable values only. The
FileDevice already sets the precedent: a register receives an ``int``
fd, never the host file object — the driver maps fd -> host object.
TensorArena extends that same discipline to array-producing devices:
``cycles render`` puts the image into an arena and the register
receives the slot id; consumers call :func:`resolve` to map the id
back to the array at the driver boundary.

Sharing one arena across devices is what makes slots composable from
assembly::

    DEV_CALL R1, cycles, render       # R1 = slot id (int), not an ndarray
    DEV_CALL R2, multimodal, embed, R1

Python callers may still pass a live ndarray anywhere — :func:`resolve`
passes it through unchanged — so legacy code keeps working while the
bus itself stays scalar.

Ownership (card a3b39114 follow-up): slots are freed explicitly via
``free()`` or wholesale via ``clear()``; no refcounting yet.
"""

from __future__ import annotations

import numpy as np

from .vm import DeviceFault


class TensorArena:
    """Slot registry mapping machine-representable ids to host arrays."""

    def __init__(self):
        self._slots: dict[int, np.ndarray] = {}
        self._next: int = 0

    def put(self, arr: np.ndarray) -> int:
        """Store an array; return its slot id (monotonic, never reused)."""
        handle = self._next
        self._next += 1
        self._slots[handle] = arr
        return handle

    def get(self, handle: int) -> np.ndarray:
        """Fetch the array for a slot id; unknown ids raise DeviceFault."""
        try:
            return self._slots[handle]
        except (KeyError, TypeError):
            raise DeviceFault(f"tensor arena: bad slot {handle!r}") from None

    def free(self, handle: int) -> None:
        """Release a slot; freeing an unknown slot raises DeviceFault."""
        try:
            del self._slots[handle]
        except (KeyError, TypeError):
            raise DeviceFault(f"tensor arena: bad slot {handle!r}") from None

    def clear(self) -> None:
        """Release every slot (e.g. between rendered frames)."""
        self._slots.clear()

    def __len__(self) -> int:
        return len(self._slots)


def resolve(value, arena: TensorArena):
    """Map a bus value back to its host array at the driver boundary.

    Accepts every value shape the bus may legitimately carry:

    - ``None``            -> ``None`` (optional arguments)
    - ``np.ndarray``      -> passthrough (legacy live-object callers)
    - ``int`` / np integer -> arena slot lookup
    - anything else       -> ``DeviceFault`` (not a bus value)
    """
    if value is None or isinstance(value, np.ndarray):
        return value
    if isinstance(value, (int, np.integer)):
        return arena.get(int(value))
    raise DeviceFault(f"expected ndarray or arena slot, got {type(value).__name__}")
