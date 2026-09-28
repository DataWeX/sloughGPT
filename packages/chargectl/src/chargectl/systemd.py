"""systemd — is the enforcement unit installed, enabled, and actually running?

Probing the charge nodes says nothing about whether the thing that *keeps* them
is alive: a threshold decays on reboot, on AC re-plug, and on driver reload, so
"the unit is installed" and "the unit is running" are different questions.

Every subprocess call here is bounded and every failure comes back as a field —
``chargectl probe`` has to survive a machine with no systemd at all.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .policy import default_state_path

UNIT_NAME = "chargectl"
SYSTEM_UNIT_DIRS: tuple[Path, ...] = (
    Path("/etc/systemd/system"),
    Path("/usr/lib/systemd/system"),
    Path("/lib/systemd/system"),
)
USER_UNIT_DIR = Path("~/.config/systemd/user")
RUN_DIR = Path("/run/systemd/system")
SYSTEM_STATE_PATH = Path("/var/lib/chargectl/state.json")
TIMEOUT = 4.0

Runner = Callable[[Sequence[str]], str | None]


def _run(argv: Sequence[str]) -> str | None:
    """Run ``argv`` and return stripped stdout (even on a non-zero exit)."""
    try:
        proc = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    out = (proc.stdout or "").strip()
    return out or None


def systemd_available() -> bool:
    """True when pid 1 is systemd *and* ``systemctl`` is on PATH."""
    return shutil.which("systemctl") is not None and RUN_DIR.is_dir()


def parse_exec_start(text: str) -> str | None:
    """First active ``ExecStart=`` binary in a unit file, without its arguments.

    Quoted paths survive — a checkout under a directory with a space in it
    (``/home/me/Default Project/app``) has to be quoted in the unit file.
    """
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("ExecStart="):
            continue
        body = line[len("ExecStart=") :].strip()
        if not body:
            continue
        try:
            parts = shlex.split(body, comments=False, posix=True)
        except ValueError:  # pragma: no cover - unbalanced quotes
            parts = body.split()
        if parts:
            return parts[0]
    return None


@dataclass(frozen=True)
class ServiceStatus:
    """What systemd says about the chargectl unit on this machine."""

    systemd: bool
    scope: str | None
    unit_installed: bool
    unit_path: str | None
    enabled: str | None
    active: str | None
    pid: int | None
    exec_start: str | None
    reason: str

    @property
    def running(self) -> bool:
        return self.active == "active"


def _find_unit(unit: str, dirs: Sequence[Path]) -> Path | None:
    for directory in dirs:
        candidate = directory / f"{unit}.service"
        try:
            if candidate.is_file():
                return candidate
        except OSError:  # pragma: no cover - unreadable mount
            continue
    return None


def service_status(
    unit: str = UNIT_NAME,
    *,
    scope: str = "auto",
    unit_dirs: Sequence[Path] | None = None,
    runner: Runner | None = None,
    systemd: bool | None = None,
) -> ServiceStatus:
    """Report install + enable + active state for ``unit``.

    ``scope`` is ``"system"``, ``"user"``, or ``"auto"`` (system unit first, then
    a user unit — a dev box usually has one or the other). Failures degrade to
    ``None`` fields; this never raises.
    """
    run = runner if runner is not None else _run
    has_systemd = systemd_available() if systemd is None else systemd
    if not has_systemd:
        return ServiceStatus(
            systemd=False,
            scope=None,
            unit_installed=False,
            unit_path=None,
            enabled=None,
            active=None,
            pid=None,
            exec_start=None,
            reason="no systemd (pid 1 is not systemd, or systemctl is missing)",
        )

    if unit_dirs is not None:
        dirs = list(unit_dirs)
    elif scope == "system":
        dirs = list(SYSTEM_UNIT_DIRS)
    elif scope == "user":
        dirs = [Path(USER_UNIT_DIR).expanduser()]
    else:
        dirs = list(SYSTEM_UNIT_DIRS) + [Path(USER_UNIT_DIR).expanduser()]

    path = _find_unit(unit, dirs)
    found_scope = None
    if path is not None:
        found_scope = "user" if USER_UNIT_DIR.expanduser() in path.parents else "system"

    prefix = ["systemctl"] + (["--user"] if found_scope == "user" else [])
    enabled = active = pid_raw = None
    if path is not None:
        enabled = run([*prefix, "is-enabled", unit])
        active = run([*prefix, "is-active", unit])
        pid_raw = run([*prefix, "show", "-p", "MainPID", "--value", unit])
    pid: int | None = None
    if pid_raw and pid_raw.isdigit() and int(pid_raw) > 0:
        pid = int(pid_raw)

    exec_start = None
    if path is not None:
        try:
            exec_start = parse_exec_start(path.read_text(encoding="utf-8"))
        except OSError:  # pragma: no cover - unreadable unit
            exec_start = None

    if path is None:
        reason = "not installed — run `sudo make charge-svc` (or `make charge-svc-user`)"
    elif active == "active":
        reason = (
            f"running ({found_scope} unit, pid {pid})" if pid else f"running ({found_scope} unit)"
        )
    else:
        reason = f"installed but not running ({active or 'unknown state'})"

    return ServiceStatus(
        systemd=True,
        scope=found_scope,
        unit_installed=path is not None,
        unit_path=str(path) if path is not None else None,
        enabled=enabled,
        active=active,
        pid=pid,
        exec_start=exec_start,
        reason=reason,
    )


def daemon_state_path() -> Path:
    """Where the running daemon records state, from the app's point of view.

    A user unit writes the default ``~/.config/chargectl/state.json``. The system
    unit runs as root and writes ``/var/lib/chargectl/state.json``. The API has to
    find either one, or "managed" silently never lights up after a system install.
    """
    default = default_state_path()
    if default.is_file():
        return default
    if SYSTEM_STATE_PATH.is_file():
        return SYSTEM_STATE_PATH
    return default
