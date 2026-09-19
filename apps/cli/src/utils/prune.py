"""prune — garbage-collect stale sloughGPT server processes.

Complements ``ensure_server()`` (which dedups at startup): a process can be
alive yet never serve (killed worker, stuck boot, zombie socket). Those
squat gigabytes forever because nothing reaps them. This module finds our
own server processes whose ports don't answer ``/health`` and terminates
them, so the next start resumes on a clean port instead of stacking
another full backend beside the corpse.

Safety rules (all enforced):
- only processes whose command line matches our own server invocations
  (``cli.py serve`` or ``uvicorn … main:app``);
- never the current process, nor any of its ancestors;
- only when ``/health`` does NOT answer (healthy servers are kept);
- SIGTERM first, SIGKILL fallback after a grace period.
"""

from __future__ import annotations

import logging
import os
import urllib.error
import urllib.request

logger = logging.getLogger("slo.cli.prune")

_DEFAULT_PORT = 8000
_HEALTH_TIMEOUT = 3
_TERM_GRACE_SECONDS = 5


def _is_own_server(cmdline: list[str]) -> bool:
    """Whether this command line is one of our server processes."""
    joined = " ".join(cmdline)
    if "cli.py" in joined and "serve" in cmdline:
        return True
    if cmdline and cmdline[0].endswith("uvicorn") and "main:app" in joined:
        return True
    if "uvicorn" in cmdline and "main:app" in joined:
        return True
    return False


def _proc_port(cmdline: list[str], default: int = _DEFAULT_PORT) -> int:
    """Extract --port from a server command line (default 8000)."""
    for i, part in enumerate(cmdline):
        if part == "--port" and i + 1 < len(cmdline):
            try:
                return int(cmdline[i + 1])
            except (ValueError, TypeError):
                return default
        if part.startswith("--port="):
            try:
                return int(part.split("=", 1)[1])
            except (ValueError, TypeError):
                return default
    return default


def _health_ok(host: str, port: int, timeout: int = _HEALTH_TIMEOUT) -> bool:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def _protected_pids() -> set[int]:
    """Current process + ancestors — never terminate these."""
    protected = {os.getpid()}
    try:
        import psutil

        proc = psutil.Process()
        while True:
            parent = proc.parent()
            if parent is None:
                break
            protected.add(parent.pid)
            proc = parent
    except Exception:
        pass
    return protected


def _has_healthy_server_child(proc, host: str) -> bool:
    """Whether any child process is a healthy server (supervisor case).

    A ``cli.py serve`` parent may supervise a uvicorn child on a different
    port than the parent's own args imply — killing it would orphan a
    working server, so such parents are kept.
    """
    try:
        import psutil

        children = proc.children(recursive=False)
    except (AttributeError, psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False
    for child in children:
        try:
            info = getattr(child, "info", None)
            cmdline = info.get("cmdline") if isinstance(info, dict) else child.cmdline()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        if not cmdline or not _is_own_server(cmdline):
            continue
        if "cli.py" in " ".join(cmdline):
            continue
        if _health_ok(host, _proc_port(cmdline)):
            return True
    return False
    """Current process + ancestors — never terminate these."""
    protected = {os.getpid()}
    try:
        import psutil

        proc = psutil.Process()
        while True:
            parent = proc.parent()
            if parent is None:
                break
            protected.add(parent.pid)
            proc = parent
    except Exception:
        pass
    return protected


def find_stale_servers(
    host: str = "127.0.0.1", port: int | None = None
) -> list[dict]:
    """List our server processes that don't answer ``/health``.

    Args:
        host: Host to probe for health.
        port: If given, only consider servers bound for that port.

    Returns:
        List of ``{"pid", "port", "cmdline", "age_seconds"}`` dicts.
    """
    import psutil

    protected = _protected_pids()
    stale = []
    for proc in psutil.process_iter(["pid", "cmdline", "create_time"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            if not cmdline or not _is_own_server(cmdline):
                continue
            if proc.info["pid"] in protected:
                continue
            proc_port = _proc_port(cmdline)
            if port is not None and proc_port != port:
                continue
            if _health_ok(host, proc_port):
                continue
            try:
                if _has_healthy_server_child(proc, host):
                    continue
            except Exception:
                pass
            import time

            stale.append(
                {
                    "pid": proc.info["pid"],
                    "port": proc_port,
                    "cmdline": " ".join(cmdline)[:160],
                    "age_seconds": time.time() - (proc.info.get("create_time") or time.time()),
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return stale


def prune_stale_servers(
    host: str = "127.0.0.1",
    port: int | None = None,
    grace_seconds: int = _TERM_GRACE_SECONDS,
) -> dict:
    """Terminate stale servers. Returns ``{"killed": [...], "failed": [...]}``.

    Never raises — a best-effort janitor that must not break startup.
    """
    import psutil

    killed: list[dict] = []
    failed: list[dict] = []
    for entry in find_stale_servers(host=host, port=port):
        try:
            proc = psutil.Process(entry["pid"])
            proc.terminate()
            try:
                proc.wait(timeout=grace_seconds)
            except psutil.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=grace_seconds)
            killed.append(entry)
            logger.info(
                "Pruned stale server pid=%s port=%s age=%.0fs",
                entry["pid"],
                entry["port"],
                entry["age_seconds"],
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as exc:
            failed.append({**entry, "error": str(exc)})
        except Exception as exc:  # never break callers (startup paths)
            logger.debug("Prune failed for pid=%s: %s", entry["pid"], exc)
            failed.append({**entry, "error": str(exc)})
    return {"killed": killed, "failed": failed}
