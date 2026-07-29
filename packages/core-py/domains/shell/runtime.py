"""
Shell Runtime — DaitRuntime + Resource.

DaitRuntime is the top-level runtime orchestrator that boots the kernel,
init system, devices, VFS, and neural capabilities.

Resource is a file metadata dataclass used for disk scanning.

The deprecated Kernel/Process/ProcessState classes that lived here have been
removed — they were superseded by the unified Kernel in kernel.py.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("slo.shell.kernel")

_REPO_ROOT = None  # lazy


@dataclass
class Resource:
    """File metadata for disk scanning (models, datasets, souls)."""
    name: str
    kind: str
    path: str
    size_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def size_str(self) -> str:
        if self.size_bytes >= 1073741824:
            return f"{self.size_bytes / 1073741824:.1f}G"
        if self.size_bytes >= 1048576:
            return f"{self.size_bytes / 1048576:.1f}M"
        if self.size_bytes >= 1024:
            return f"{self.size_bytes / 1024:.1f}K"
        return f"{self.size_bytes}B"


class DaitRuntime:
    """Top-level runtime — orchestrates kernel, init, devices, VFS, and neural."""

    def __init__(self):
        from .kernel import Kernel
        self.kernel = Kernel()
        self._model_loaded: bool = False
        self._model_name: str = ""
        self._current_soul: str = ""
        self._boot_complete: bool = False
        self._boot_time: float = 0.0
        self._init: Any = None
        self._devices: Any = None
        self._vfs: Any = None
        self._device_system: Any = None

    def boot(self, shell_run: Callable[[str], str] | None = None) -> str:
        """Boot the full runtime: addons, kernel, VFS, devices, init system.

        Args:
            shell_run: Optional shell command executor for init services.

        Returns:
            Boot log from the init system.
        """
        from .init import get_init_system
        from .devices import create_default_devices
        from .device_system import get_device_system

        self._init = get_init_system()
        self._devices = create_default_devices(get_kernel=lambda: self.kernel)

        # Install addons before kernel boot — addons may depend on kernel state
        from .addons import neural, filesystem, shell_ui
        self.kernel.install_addon(neural)
        self.kernel.install_addon(filesystem)
        self.kernel.install_addon(shell_ui)

        # Boot kernel (sets _running, _boot_time)
        self._boot_time = time.time()
        self.kernel.boot()

        # Wire VFS — set_devices is not done by setup() (only set_kernel is)
        self._vfs = self.kernel.vfs
        self._vfs.set_devices(self._devices)

        # Wire DeviceSystem — register all shell devices
        self._device_system = get_device_system()
        for name in self._devices.names:
            dev = self._devices.get(name)
            self._device_system.register(name, dev, registered_by="shell")

        # Bridge NPU to DeviceSystem so assembly programs can access it
        from .vm_devices import NPUVMDevice
        from .kernel_npu import NPUDevice
        npu_device = NPUDevice(name="npu")
        self._device_system.register("npu", NPUVMDevice(npu_device), registered_by="kernel")

        boot_log = self._init.boot(target_runlevel=3, shell_run=shell_run)
        self._detect_health()
        self._boot_complete = True
        return boot_log

    def shutdown(self) -> str:
        """Shut down the runtime: init system, then kernel.

        Returns:
            Shutdown log from the init system.
        """
        from .init import get_init_system
        self._boot_complete = False
        shutdown_log = get_init_system().shutdown()
        self.kernel.shutdown()
        return shutdown_log

    def _detect_health(self) -> None:
        try:
            import requests
            from .config import get_api_base
            r = requests.get(f"{get_api_base()}/health", timeout=2)
            if r.status_code == 200:
                data = r.json()
                self._model_loaded = data.get("model_loaded", False)
                self._model_name = data.get("model_type", "")
        except Exception:
            self._model_loaded = False

    @property
    def status_summary(self) -> str:
        try:
            import psutil
            proc = psutil.Process()
            mem = proc.memory_info()
            mem_str = f"rss={mem.rss / 1048576:.0f}M vms={mem.vms / 1048576:.0f}M"
        except Exception:
            mem_str = "psutil not available"
        lines = (
            f"Kernel uptime: {self.kernel.uptime:.0f}s\n"
            f"Processes: {len(self.kernel.list_processes())}\n"
            f"Model: {'loaded' if self._model_loaded else 'not loaded'} ({self._model_name})\n"
            f"Soul: {self._current_soul or 'default'}\n"
            f"Memory: {mem_str}"
        )
        if self._boot_complete and self._init is not None:
            lines += f"\n{self._init.status_summary}"
        return lines

    @property
    def init_system(self):
        return self._init

    @property
    def devices(self):
        return self._devices

    @property
    def vfs(self):
        return self._vfs
