"""
Shell Init System — service definitions, runlevels, dependency ordering, respawn.

Provides:
  - ServiceDefinition: declarative service metadata
  - ServiceManager: lifecycle control (start/stop/restart/respawn)
  - InitSystem: multi-runlevel boot orchestration
  - Default built-in services
"""

from __future__ import annotations

import os
import sys
import time
import json
import logging
import shlex
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("slo.shell.init")


# ── Constants ──────────────────────────────────────────────────────────────

SERVICES_DIR = Path.home() / ".config" / "sloughgpt" / "services"
BUILTIN_SERVICES: dict[str, dict[str, Any]] = {
    "kernel": {
        "command": "",
        "deps": [],
        "respawn": False,
        "runlevel": 1,
        "description": "Core kernel — process and resource manager",
        "builtin": True,
    },
    "agent-orchestrator": {
        "command": "",
        "deps": [],
        "runlevel": 3,
        "description": "Multi-agent orchestration daemon",
        "builtin": True,
    },
    "knowledge-worker": {
        "command": "",
        "deps": [],
        "runlevel": 3,
        "description": "Background knowledge ingestion worker",
        "builtin": True,
    },
}

SERVICE_STATES = ["stopped", "starting", "running", "stopping", "failed", "crashed"]


# ── Data models ────────────────────────────────────────────────────────────


@dataclass
class ServiceDef:
    """Declarative service definition (loaded from JSON or built-in)."""
    name: str
    command: str = ""
    deps: list[str] = field(default_factory=list)
    respawn: bool = True
    max_respawns: int = 3
    respawn_delay: float = 2.0
    runlevel: int = 2
    timeout: float = 30.0
    health_check: str = ""
    description: str = ""
    builtin: bool = False


@dataclass
class ServiceInstance:
    """Running instance of a service."""
    definition: ServiceDef
    state: str = "stopped"
    pid: int = 0
    started_at: float = 0.0
    respawn_count: int = 0
    process: subprocess.Popen | None = None
    log: list[str] = field(default_factory=list)

    @property
    def uptime(self) -> float:
        if self.state in ("running", "starting") and self.started_at:
            return time.time() - self.started_at
        return 0.0


# ── Service Manager ────────────────────────────────────────────────────────


class ServiceManager:
    """Manages service lifecycle for a single service."""

    def __init__(self, definition: ServiceDef):
        self.defn = definition
        self.instance = ServiceInstance(definition=definition)
        self._lock = threading.Lock()
        self._stop_requested = False

    def start(self, shell_run: Callable[[str], str] | None = None) -> bool:
        """Start the service. Returns True if process launched successfully.

        Does NOT wait for health check — call wait_until_healthy() separately.
        """
        with self._lock:
            if self.instance.state in ("running", "starting"):
                return True
            self._stop_requested = False
            self.instance.state = "starting"
            self.instance.started_at = time.time()

        if self.defn.builtin:
            with self._lock:
                self.instance.state = "running"
            self._log("built-in service registered")
            return True

        if not self.defn.command:
            with self._lock:
                self.instance.state = "failed"
            self._log("no command defined")
            return False

        try:
            proc = subprocess.Popen(
                shlex.split(self.defn.command),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            )
            with self._lock:
                self.instance.process = proc
                self.instance.pid = proc.pid
                self.instance.state = "running"

            self._log(f"started (pid={proc.pid})")
            threading.Thread(target=self._wait, daemon=True).start()
            return True
        except Exception as e:
            with self._lock:
                self.instance.state = "failed"
            self._log(f"start failed: {e}")
            return False

    def wait_until_healthy(self) -> bool:
        """Single health check attempt. Returns True if healthy, False otherwise."""
        if not self.defn.health_check:
            return True
        try:
            subprocess.run(
                shlex.split(self.defn.health_check),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            with self._lock:
                self.instance.state = "running"
            return True
        except Exception:
            return False

    def stop(self) -> bool:
        """Stop the service."""
        with self._lock:
            self._stop_requested = True
            if self.instance.state not in ("running", "starting", "crashed"):
                return True
            self.instance.state = "stopping"
            proc = self.instance.process

        if proc and proc.poll() is None:
            try:
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(proc.pid), 15)
                else:
                    proc.terminate()
                proc.wait(timeout=10)
            except Exception as e:
                logger.debug("process terminate failed: %s", e)
                try:
                    proc.kill()
                except Exception as e:
                    logger.debug("process kill failed: %s", e)

        with self._lock:
            self.instance.state = "stopped"
            self.instance.process = None
            self.instance.pid = 0
        self._log("stopped")
        return True

    def restart(self, shell_run: Callable[[str], str] | None = None) -> bool:
        """Restart the service."""
        self.stop()
        time.sleep(0.5)
        return self.start(shell_run)

    @property
    def is_alive(self) -> bool:
        """Check if the underlying process is still alive."""
        proc = self.instance.process
        if proc is None:
            return self.instance.state == "running" and self.defn.builtin
        return proc.poll() is None

    def _wait(self) -> None:
        """Background thread: wait for process, handle respawn."""
        proc = self.instance.process
        if proc is None:
            return
        try:
            proc.wait()
        except Exception as e:
            logger.debug("process wait failed: %s", e)

        with self._lock:
            was_requested = self._stop_requested
            self.instance.process = None
            self.instance.pid = 0

        if was_requested:
            return

        if self.defn.respawn and self.instance.respawn_count < self.defn.max_respawns:
            self.instance.respawn_count += 1
            self._log(f"crashed — respawning ({self.instance.respawn_count}/{self.defn.max_respawns})")
            time.sleep(self.defn.respawn_delay)
            self.start()
        else:
            with self._lock:
                self.instance.state = "crashed"
            if self.instance.respawn_count >= self.defn.max_respawns:
                self._log("max respawns reached — giving up")
            else:
                self._log("process exited")

    def _log(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}"
        self.instance.log.append(entry)
        logger.debug("[%s] %s", self.defn.name, msg)

    def status_line(self, max_name: int = 24) -> str:
        name = self.defn.name.ljust(max_name)
        state = self.instance.state.ljust(10)
        pid = str(self.instance.pid) if self.instance.pid else "-"
        uptime_s = f"{self.instance.uptime:.0f}s" if self.instance.uptime > 0 else "-"
        respawns = f"r{self.instance.respawn_count}" if self.instance.respawn_count else ""
        return f"  {name} {state} pid={pid} uptime={uptime_s} {respawns}"


# ── Init System ────────────────────────────────────────────────────────────


class InitSystem:
    """Multi-runlevel init system — boot, service lifecycle, shutdown."""

    def __init__(self):
        self._managers: dict[str, ServiceManager] = {}
        self._boot_time: float = 0.0
        self._boot_complete: bool = False
        self._current_runlevel: int = 0
        self._lock = threading.Lock()
        self._load_definitions()

    def _load_definitions(self) -> None:
        """Load service definitions from built-in + user config."""
        for name, cfg in BUILTIN_SERVICES.items():
            self._managers[name] = ServiceManager(ServiceDef(name=name, **cfg))

        if SERVICES_DIR.is_dir():
            for f in sorted(SERVICES_DIR.glob("*.json")):
                try:
                    data = json.loads(f.read_text())
                    name = f.stem
                    if name not in self._managers:
                        self._managers[name] = ServiceManager(ServiceDef(name=name, **data))
                except Exception as e:
                    logger.warning("Failed to load service %s: %s", f.name, e, extra={"tag": "INFRA"})

    def boot(self, target_runlevel: int = 3, shell_run: Callable[[str], str] | None = None) -> str:
        """Boot through runlevels up to target_runlevel. Returns boot log."""
        self._boot_time = time.time()

        rl_names = {1: "boot-critical", 2: "core", 3: "optional"}
        output: list[str] = []

        for rl in range(1, target_runlevel + 1):
            self._current_runlevel = rl
            services = [m for m in self._managers.values() if m.defn.runlevel == rl]
            if not services:
                continue

            label = rl_names.get(rl, f"runlevel {rl}")
            output.append(f"  ── {label} ──")

            ordered = self._resolve_deps(services)

            for mgr in ordered:
                output.append(f"    {mgr.defn.name}...")

                deps_ok = True
                for dep_name in mgr.defn.deps:
                    dep = self._managers.get(dep_name)
                    if dep and dep.instance.state != "running":
                        dep_start = dep.start(shell_run)
                        if not dep_start:
                            output[-1] = f"    {mgr.defn.name}... dependency {dep_name} failed"
                            output.append("      └─ ✗ dependency failed")
                            deps_ok = False
                            break

                if not deps_ok:
                    continue

                ok = mgr.start(shell_run)

                if ok and mgr.defn.health_check:
                    deadline = time.time() + mgr.defn.timeout
                    while time.time() < deadline:
                        if mgr.instance.state in ("failed", "crashed"):
                            break
                        if mgr.wait_until_healthy():
                            break
                        time.sleep(0.5)
                    else:
                        with self._lock:
                            mgr.instance.state = "failed"

                if ok:
                    result = f"      └─ ✓ {mgr.defn.description or 'started'}"
                else:
                    result = "      └─ ✗ failed"
                output.append(result)

            output.append("")

        self._current_runlevel = target_runlevel
        self._boot_complete = True
        elapsed = time.time() - self._boot_time
        final = f"  Boot complete in {elapsed:.1f}s (runlevel {target_runlevel})"
        output.append("")
        output.append(final)
        return "\n".join(output)

    def _resolve_deps(self, services: list[ServiceManager]) -> list[ServiceManager]:
        """Simple topological sort by dependency ordering."""
        by_name = {m.defn.name: m for m in services}
        ordered: list[ServiceManager] = []
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            for dep in by_name[name].defn.deps:
                if dep in by_name:
                    visit(dep)
            ordered.append(by_name[name])

        for mgr in services:
            visit(mgr.defn.name)

        return ordered

    def shutdown(self) -> str:
        """Graceful shutdown — stop services in reverse order."""
        lines = ["\n  ── shutdown ──"]
        all_services = list(self._managers.values())
        # Stop in reverse runlevel order
        for rl in sorted({m.defn.runlevel for m in all_services}, reverse=True):
            for mgr in all_services:
                if mgr.defn.runlevel == rl and mgr.instance.state in ("running", "starting", "crashed"):
                    mgr.stop()
                    lines.append(f"    {mgr.defn.name}... stopped")
        self._boot_complete = False
        self._current_runlevel = 0
        lines.append("  System halted.")
        return "\n".join(lines)

    def get_manager(self, name: str) -> ServiceManager | None:
        return self._managers.get(name)

    @property
    def services(self) -> list[ServiceManager]:
        return list(self._managers.values())

    @property
    def runlevel(self) -> int:
        return self._current_runlevel

    @property
    def uptime(self) -> float:
        return time.time() - self._boot_time if self._boot_time else 0.0

    @property
    def status_summary(self) -> str:
        lines = [f"  Runlevel: {self._current_runlevel}"]
        lines.append(f"  Uptime: {self.uptime:.0f}s")
        lines.append(f"  Services: {len(self._managers)} ({sum(1 for m in self._managers.values() if m.instance.state == 'running')} running)")
        return "\n".join(lines)

    def service_table(self) -> str:
        if not self._managers:
            return "  No services defined"
        max_name = max(len(m.defn.name) for m in self._managers.values())
        lines = []
        for mgr in sorted(self._managers.values(), key=lambda m: m.defn.runlevel):
            lines.append(mgr.status_line(max_name))
        return "\n".join(lines)


# ── Singleton ──────────────────────────────────────────────────────────────

_init_system: InitSystem | None = None


def get_init_system() -> InitSystem:
    global _init_system
    if _init_system is None:
        _init_system = InitSystem()
    return _init_system


def reset_init_system() -> None:
    global _init_system
    _init_system = None
