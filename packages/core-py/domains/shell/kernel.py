"""
Shell Runtime Kernel — core process, memory, and resource manager.

Initializes all subsystems, tracks running processes (training jobs,
inference sessions), manages virtual memory (context windows), and exposes
resources (models, datasets, souls, knowledge) as a virtual filesystem.
"""

from __future__ import annotations

import time
import json
import logging
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("slo.shell.kernel")

_REPO_ROOT = Path(__file__).resolve().parents[4]


class ProcessState:
    IDLE = "idle"
    RUNNING = "running"
    SLEEPING = "sleeping"
    STOPPED = "stopped"
    ZOMBIE = "zombie"


@dataclass
class Process:
    pid: int
    name: str
    state: str = ProcessState.IDLE
    created_at: float = field(default_factory=time.time)
    cpu_usage: float = 0.0
    memory_kb: int = 0
    thread: threading.Thread | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def uptime(self) -> float:
        return time.time() - self.created_at

    @property
    def status_line(self) -> str:
        return f"[{self.pid:4d}] {self.state:<10} {self.name:<30} {self.uptime:.1f}s"


@dataclass
class Resource:
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


class Kernel:
    """Core kernel — manages processes, memory, and resource listing."""

    def __init__(self):
        self._processes: dict[int, Process] = {}
        self._next_pid: int = 1
        self._resources: dict[str, list[Resource]] = {}
        self._context_memory: dict[str, Any] = {}
        self._boot_time = time.time()
        self._running = False

    def boot(self) -> None:
        self._running = True
        self._init_process = self.spawn("kernel", "kernel-init")
        logger.info("Kernel booted (pid=%d)", self._init_process.pid, extra={"tag": "INFRA"})

    def shutdown(self) -> None:
        self._running = False
        for proc in list(self._processes.values()):
            if proc.state == ProcessState.RUNNING:
                proc.state = ProcessState.STOPPED
        logger.info("Kernel shut down", extra={"tag": "INFRA"})

    @property
    def uptime(self) -> float:
        return time.time() - self._boot_time

    def spawn(self, name: str, kind: str = "task", metadata: dict | None = None) -> Process:
        pid = self._next_pid
        self._next_pid += 1
        proc = Process(pid=pid, name=name, metadata=metadata or {})
        self._processes[pid] = proc
        logger.debug("Spawned pid=%d name=%s kind=%s", pid, name, kind)
        return proc

    def kill(self, pid: int) -> bool:
        proc = self._processes.get(pid)
        if proc is None:
            return False
        proc.state = ProcessState.STOPPED
        del self._processes[pid]
        return True

    def list_processes(self) -> list[Process]:
        return list(self._processes.values())

    def get_process(self, pid: int) -> Process | None:
        return self._processes.get(pid)

    def scan_resources(self) -> dict[str, list[Resource]]:
        resources: dict[str, list[Resource]] = {}

        models_dir = _REPO_ROOT / "models"
        if models_dir.is_dir():
            models = []
            for f in models_dir.iterdir():
                if f.suffix in (".soul", ".pt", ".safetensors", ".gguf"):
                    models.append(Resource(f.stem, "model", str(f), f.stat().st_size))
            for d in models_dir.iterdir():
                if d.is_dir() and d.name not in ("hf-finetuned",):
                    size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                    models.append(Resource(d.name, "model-dir", str(d), size))
            resources["models"] = models

        finetuned_dir = models_dir / "hf-finetuned"
        if finetuned_dir.is_dir():
            finetuned = []
            for entry in finetuned_dir.iterdir():
                if entry.is_dir():
                    cfg = entry / "training_config.json"
                    meta = {}
                    if cfg.is_file():
                        try:
                            meta = json.loads(cfg.read_text())
                        except Exception:
                            pass
                    size = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
                    finetuned.append(Resource(entry.name, "finetuned", str(entry), size, meta))
            resources["finetuned"] = finetuned

        datasets_dir = _REPO_ROOT / "datasets"
        if datasets_dir.is_dir():
            datasets = []
            for d in datasets_dir.iterdir():
                if d.is_dir():
                    size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                    datasets.append(Resource(d.name, "dataset", str(d), size))
            resources["datasets"] = datasets

        souls_dir = _REPO_ROOT / "models"
        souls = []
        for f in souls_dir.glob("*.soul"):
            souls.append(Resource(f.stem, "soul", str(f), f.stat().st_size))
        for f in souls_dir.glob("auto-training/*.soul"):
            souls.append(Resource(f.stem, "soul-checkpoint", str(f), f.stat().st_size))
        resources["souls"] = souls

        self._resources = resources
        return resources

    def get_resource(self, kind: str, name: str) -> Resource | None:
        resources = self._resources.get(kind, [])
        for r in resources:
            if r.name == name:
                return r
        return None

    def store_context(self, key: str, value: Any) -> None:
        self._context_memory[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        return self._context_memory.get(key, default)

    def clear_context(self) -> None:
        self._context_memory.clear()

    @property
    def memory_usage_str(self) -> str:
        try:
            import psutil
            proc = psutil.Process()
            mem = proc.memory_info()
            return f"rss={mem.rss / 1048576:.0f}M vms={mem.vms / 1048576:.0f}M"
        except ImportError:
            return "psutil not available"


class DaitRuntime:
    """Top-level runtime — wraps Kernel + resource management."""

    def __init__(self):
        self.kernel = Kernel()
        self._model_loaded: bool = False
        self._model_name: str = ""
        self._current_soul: str = ""
        self._boot_complete: bool = False
        self._boot_time: float = 0.0
        self._init: "InitSystem" = None  # noqa: F821  # lazy import below
        self._devices: "DeviceManager" = None  # noqa: F821  # lazy import below
        self._vfs: "VFS" = None  # noqa: F821  # lazy import below

    def boot(self, shell_run: Callable[[str], str] | None = None) -> str:
        from .init import get_init_system
        from .devices import create_default_devices
        from .vfs import get_vfs
        self._init = get_init_system()
        self._devices = create_default_devices(get_kernel=lambda: self.kernel)
        self._vfs = get_vfs()
        self._vfs.set_devices(self._devices)
        self._vfs.set_kernel(self.kernel)
        self._boot_time = time.time()
        self.kernel.boot()
        boot_log = self._init.boot(target_runlevel=3, shell_run=shell_run)
        self._detect_health()
        self._boot_complete = True
        return boot_log

    def shutdown(self) -> str:
        from .init import get_init_system
        self._boot_complete = False
        shutdown_log = get_init_system().shutdown()
        self.kernel.shutdown()
        return shutdown_log

    def _detect_health(self) -> None:
        try:
            import requests
            r = requests.get("http://localhost:8000/health", timeout=2)
            if r.status_code == 200:
                data = r.json()
                self._model_loaded = data.get("model_loaded", False)
                self._model_name = data.get("model_type", "")
        except Exception:
            self._model_loaded = False

    @property
    def status_summary(self) -> str:
        lines = (
            f"Kernel uptime: {self.kernel.uptime:.0f}s\n"
            f"Processes: {len(self.kernel.list_processes())}\n"
            f"Model: {'loaded' if self._model_loaded else 'not loaded'} ({self._model_name})\n"
            f"Soul: {self._current_soul or 'default'}\n"
            f"Memory: {self.kernel.memory_usage_str}"
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
