"""
DeviceSystem — central device registry wrapping VM DeviceBus.

Any component (shell, VM, managers, loggers) registers through DeviceSystem.
When assembly runs, DeviceSystem bridges registered devices into the VM's bus.

Architecture:
  DeviceSystem (singleton)
    ├── _devices: dict[str, Device]     — component-facing registry
    ├── _bus: DeviceBus                  — VM-facing bridge
    └── _metadata: dict[str, dict]       — registration context
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from domains.shell.vm import Device, DeviceBus, DeviceFault

logger = logging.getLogger("slo.shell.device_system")


class DeviceSystem:
    """Central device registry. Wraps VM DeviceBus internally.

    Components register devices here. The internal DeviceBus is exposed
    for VM wiring — assembly programs access devices through the bus.
    """

    def __init__(self):
        self._bus = DeviceBus()
        self._devices: dict[str, Device] = {}
        self._metadata: dict[str, dict] = {}

    def register(self, name: str, device: Device, *,
                 registered_by: str = "", **meta) -> None:
        """Register a device. Also registers on internal VM DeviceBus.

        Args:
            name: device name (used as DEV_OPEN target in assembly)
            device: Device instance
            registered_by: identifier of the registering component
            **meta: additional metadata (description, version, etc.)

        Side effects:
            - adds device to internal dicts and VM bus
        """
        self._devices[name] = device
        self._bus.register(name, device)
        self._metadata[name] = {
            "description": getattr(device, "description", ""),
            "registered_by": registered_by,
            **meta,
        }
        logger.debug("registered device %s (by %s)", name, registered_by or "unknown")

    def unregister(self, name: str) -> None:
        """Remove a device from the registry.

        Args:
            name: device name to remove

        Side effects:
            - removes device from internal dicts
            - NOTE: DeviceBus has no unregister, so the bus keeps a stale
              reference until a new device is registered under the same name
        """
        self._devices.pop(name, None)
        self._metadata.pop(name, None)
        logger.debug("unregistered device %s", name)

    def get(self, name: str) -> Device | None:
        """Look up a device by name."""
        return self._devices.get(name)

    def call(self, name: str, method: str, *args) -> Any:
        """Call a device method by name. Convenience for non-VM callers.

        Args:
            name: registered device name
            method: method name to call
            *args: arguments forwarded to device.call()

        Returns:
            result from device.call(method, *args)

        Raises:
            DeviceFault: if device not found or method unknown
        """
        dev = self._devices.get(name)
        if dev is None:
            raise DeviceFault(f"no such device: {name}")
        return dev.call(method, *args)

    def list_devices(self) -> list[str]:
        """Return sorted list of registered device names."""
        return sorted(self._devices.keys())

    def info(self, name: str) -> dict:
        """Get device info dict. Returns empty dict if not found."""
        dev = self._devices.get(name)
        if dev is None:
            return {}
        return dev.info()

    def metadata(self, name: str) -> dict:
        """Get registration metadata for a device."""
        return self._metadata.get(name, {})

    @property
    def bus(self) -> DeviceBus:
        """Expose the internal DeviceBus for VM wiring.

        The bus contains all registered devices. Pass to VMRunner:
            runner = VMRunner(devices=device_system.bus)
        """
        return self._bus

    def __len__(self) -> int:
        return len(self._devices)

    def __contains__(self, name: str) -> bool:
        return name in self._devices


# ── Singleton ────────────────────────────────────────────────────────────────

_instance: Optional[DeviceSystem] = None


def get_device_system() -> DeviceSystem:
    """Get or create the global DeviceSystem singleton."""
    global _instance
    if _instance is None:
        _instance = DeviceSystem()
    return _instance


def reset_device_system() -> None:
    """Reset the singleton (for testing)."""
    global _instance
    _instance = None
