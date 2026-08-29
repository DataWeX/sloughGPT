"""
Shell Runtime — DaitRuntime + Resource.

DaitRuntime is the top-level runtime orchestrator that boots the kernel,
init system, devices, VFS, and neural capabilities. It also manages the
API server lifecycle independently — the shell connects to the API, it
does NOT own it.

Resource is a file metadata dataclass used for disk scanning.

The deprecated Kernel/Process/ProcessState classes that lived here have been
removed — they were superseded by the unified Kernel in kernel.py.
"""

from __future__ import annotations

import os
import sys
import time
import json
import shlex
import signal
import logging
import threading
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from domains.shared import find_repo_root, find_server_python

logger = logging.getLogger("slo.shell.runtime")

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


def _default_api_url() -> str:
    return os.environ.get("API_BASE", "http://localhost:8000")


def _probe_api(api_url: str, timeout: float = 2.0) -> dict:
    """Check if the API server is reachable. Returns status dict."""
    import urllib.request
    try:
        req = urllib.request.Request(f"{api_url}/health", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = json.loads(resp.read().decode())
            # Health endpoint wraps in {"status":"success","data":{...}}; unwrap.
            data = raw.get("data", raw)
            return {
                "available": True,
                "status": data.get("status", "unknown"),
                "model_loaded": data.get("model_loaded", False),
                "model_id": data.get("model_type") or data.get("model_id"),
                "engine_type": data.get("engine_type"),
            }
    except Exception as e:
        return {
            "available": False,
            "error": str(e),
        }


_shared_proc: subprocess.Popen | None = None
_shared_lock = threading.Lock()
_shared_started_at: float = 0.0


class APIServerProcess:
    """Manages the API server as a subprocess (start/stop/status).

    The underlying subprocess is shared at the module level so that every
    DaitRuntime instance sees the same process.  Only the first call to
    ``start()`` spawns a new process; subsequent calls return immediately.
    """

    def __init__(self, api_url: str = ""):
        self._api_url = api_url or _default_api_url()
        from urllib.parse import urlparse
        self._port = urlparse(self._api_url).port or 8000

    # ── Public API ──────────────────────────────────────────────────────

    def status(self) -> dict:
        """Check API availability and return status dict."""
        global _shared_started_at
        result = _probe_api(self._api_url)
        with _shared_lock:
            result["running"] = _shared_proc is not None
            result["uptime"] = time.time() - _shared_started_at if _shared_started_at else 0
        return result

    def start(self, timeout: float = 180.0) -> dict:
        """Launch the API server subprocess and wait for it to become healthy.

        Three-phase flow:
        1. Health probe -- if a server is already healthy, return immediately.
        2. Port detection -- if something occupies the port but isn't healthy yet,
           wait for it (server is still loading from a previous invocation).
        3. Spawn subprocess -- start ``apps.api.server.main`` and poll health
           until the timeout elapses.

        Args:
            timeout: seconds to wait for the API to become healthy (default 180s,
            covers model loading on slow hardware).

        Returns:
            ``{"ok": True, "message": ...}`` on success,
            ``{"ok": False, "error": ...}`` on failure.

        Side effects:
            - Spawns ``apps.api.server.main`` as a shared subprocess (singleton).
            - Sets ``_shared_proc`` / ``_shared_started_at`` module globals.
        """
        global _shared_proc, _shared_started_at

        # ── Phase 0: already spawned by this process? ────────────────
        with _shared_lock:
            if _shared_proc is not None:
                if _shared_proc.poll() is None:
                    return {"ok": True, "message": "already running"}
                # Previous process exited -- clear stale handle
                _shared_proc = None
                _shared_started_at = 0.0

        # ── Phase 1: is a healthy server already listening? ───────────
        probe = _probe_api(self._api_url)
        if probe.get("available"):
            model_id = probe.get("model_id")
            return {"ok": True, "message": f"connected ({model_id})" if model_id else "connected"}

        # ── Phase 2: port occupied but not healthy -> loading server? ──
        import socket as _sock
        _port_busy = False
        try:
            with _sock.create_connection(("127.0.0.1", self._port), timeout=1.0):
                _port_busy = True
        except (ConnectionRefusedError, OSError, TimeoutError):
            pass

        if _port_busy:
            logger.debug("Port %d occupied but unhealthy -- waiting for server to finish loading", self._port, extra={"tag": "START"})
            deadline = time.time() + min(timeout, 60.0)
            while time.time() < deadline:
                probe = _probe_api(self._api_url)
                if probe.get("available"):
                    model_id = probe.get("model_id")
                    return {"ok": True, "message": f"connected ({model_id})" if model_id else "connected"}
                time.sleep(1.0)
            return {"ok": False, "error": f"Port {self._port} occupied but server not healthy after {timeout:.0f}s"}

        # ── Phase 3: port is free -> spawn the server ─────────────────
        repo_root = find_repo_root(Path(__file__).resolve())
        server_python = find_server_python(repo_root)
        cmd = [server_python, "-m", "apps.api.server.main"]
        logger.debug("Starting API server: %s (cwd=%s)", " ".join(cmd), repo_root, extra={"tag": "START"})

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(repo_root),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            )
        except Exception as e:
            return {"ok": False, "error": f"Failed to launch: {e}"}

        with _shared_lock:
            _shared_proc = proc
            _shared_started_at = time.time()

        # Stream stderr into LogBuffer — never print to terminal.
        def _log_stderr():
            if _shared_proc and _shared_proc.stderr:
                try:
                    from domains.shell.log_buffer import get_log_buffer, LogEntry
                    buf = get_log_buffer()
                    for line in _shared_proc.stderr:
                        stripped = line.rstrip()
                        if not stripped:
                            continue
                        level = "INFO"
                        upper = stripped.upper()
                        if "WRN" in upper or "WARNING" in upper:
                            level = "WARNING"
                        elif "ERR" in upper or "ERROR" in upper or "CRITICAL" in upper:
                            level = "ERROR"
                        elif "DBG" in upper or "DEBUG" in upper:
                            level = "DEBUG"
                        buf.append(LogEntry(
                            timestamp=time.time(),
                            level=level,
                            source="api.server",
                            message=stripped,
                        ))
                except Exception as e:
                    logger.warning("stderr capture thread failed: %s", e)
        threading.Thread(target=_log_stderr, daemon=True).start()

        # Poll health until ready
        deadline = time.time() + timeout
        model_id = None
        _probe_attempts = 0
        while time.time() < deadline:
            probe = _probe_api(self._api_url)
            if probe.get("available"):
                model_id = probe.get("model_id")
                logger.debug("API server ready after %d probe(s)", _probe_attempts)
                break
            _probe_attempts += 1
            time.sleep(0.5)
        else:
            logger.warning("API server not ready after %d probes (%.0fs)", _probe_attempts, timeout)
            return {"ok": False, "error": f"Timed out waiting for API ({timeout:.0f}s)"}

        return {"ok": True, "message": f"ready ({model_id})" if model_id else "ready"}

    def stop(self) -> dict:
        """Stop the API server process."""
        global _shared_proc, _shared_started_at

        with _shared_lock:
            proc = _shared_proc
            _shared_proc = None
            _shared_started_at = 0.0

        if proc is None:
            return {"ok": True, "message": "not running"}

        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:
                proc.terminate()
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                else:
                    proc.kill()
                proc.wait(timeout=5)
            except Exception as e:
                logger.warning("Failed to SIGKILL API server (pid=%d): %s", proc.pid, e)
        except Exception as e:
            logger.warning("Failed to stop API server (pid=%d): %s", proc.pid, e)

        return {"ok": True, "message": "stopped"}

    @property
    def is_running(self) -> bool:
        """Check whether the API server is running.

        First checks the local subprocess handle (started via ``start()``).
        If that is None, probes the HTTP health endpoint — the server may
        have been started externally (e.g. ``ensure_server()``).
        """
        with _shared_lock:
            if _shared_proc is not None:
                return _shared_proc.poll() is None
        return _probe_api(self._api_url).get("available", False)

    # ── Internal ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"APIServerProcess(url={self._api_url}, running={self.is_running})"


class DaitRuntime:
    """Top-level runtime — orchestrates kernel, init, devices, VFS, neural, and API connection."""

    def __init__(self, api_url: str = ""):
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
        self._api = APIServerProcess(api_url=api_url)

    # ── Boot / Shutdown ─────────────────────────────────────────────────

    def boot(self, shell_run: Callable[[str], str] | None = None) -> tuple[str, dict]:
        """Boot the full runtime: addons, kernel, VFS, devices, init system.

        Args:
            shell_run: Optional shell command executor for init services.

        Returns:
            (boot_log, api_status) tuple.
        """
        from .init import get_init_system
        from .devices import create_default_devices
        from .device_system import get_device_system

        self._init = get_init_system()
        self._devices = create_default_devices(get_kernel=lambda: self.kernel)

        # Install addons before kernel boot
        from .addons import neural, filesystem, shell_ui
        self.kernel.install_addon(neural)
        self.kernel.install_addon(filesystem)
        self.kernel.install_addon(shell_ui)

        self._boot_time = time.time()
        self.kernel.boot()

        self._vfs = self.kernel.vfs
        self._vfs.set_devices(self._devices)

        self._device_system = get_device_system()
        for name in self._devices.names:
            dev = self._devices.get(name)
            self._device_system.register(name, dev, registered_by="shell")

        from .vm_devices import NPUVMDevice
        from .kernel_npu import NPUDevice
        npu_device = NPUDevice(name="npu")
        self._device_system.register("npu", NPUVMDevice(npu_device), registered_by="kernel")

        boot_log = self._init.boot(target_runlevel=3, shell_run=shell_run)
        self._boot_complete = True
        return boot_log, self.api_status

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

    # ── API Server ──────────────────────────────────────────────────────

    @property
    def api(self) -> APIServerProcess:
        return self._api

    @property
    def api_status(self) -> dict:
        """Check API availability. Returns status dict with 'available' key."""
        result = self._api.status()
        self._model_loaded = result.get("model_loaded", False)
        self._model_name = result.get("model_id", "") or result.get("model_id", "")
        return result

    # ── Status ──────────────────────────────────────────────────────────

    @property
    def status_summary(self) -> str:
        try:
            import psutil
            proc = psutil.Process()
            mem = proc.memory_info()
            mem_str = f"rss={mem.rss / 1048576:.0f}M vms={mem.vms / 1048576:.0f}M"
        except Exception:
            mem_str = "psutil not available"
        api = self.api_status
        api_str = f"API: {'✓' if api.get('available') else '✗'} ({api.get('model_id') or 'not connected'})"
        lines = (
            f"Kernel uptime: {self.kernel.uptime:.0f}s\n"
            f"Processes: {len(self.kernel.list_processes())}\n"
            f"{api_str}\n"
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
