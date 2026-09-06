"""Shared CLI helper functions — used by commands/ modules."""
import re
import sys
import os
import time
import logging
import threading
from pathlib import Path

logger = logging.getLogger("slo.cli.helpers")


def chat_repository_root() -> Path:
    """Repository root (delegates to shared utility)."""
    from domains.shared import find_repo_root
    return find_repo_root(Path(__file__).resolve())


def chat_uvicorn_bind_host(client_host: str) -> str:
    if client_host in ("localhost", "127.0.0.1"):
        return "127.0.0.1"
    return client_host


def chat_find_available_port(bind_host: str, start_port: int, max_attempts: int = 10) -> int:
    from domains.shared import find_available_port
    return find_available_port(host=bind_host, start_port=start_port, max_attempts=max_attempts)


def chat_wait_for_health(base_url: str, timeout_sec: float = 45.0) -> bool:
    import requests
    deadline = time.monotonic() + timeout_sec
    url = f"{base_url.rstrip('/')}/health"
    while time.monotonic() < deadline:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(0.25)
    return False


def train_export_stem_slug(part: str, fallback: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", (part or "").strip()).strip("-")
    return s[:64] or fallback


def train_export_default_stem(model_name: str, dataset_label: str) -> str:
    from datetime import datetime
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    return f"{train_export_stem_slug(model_name, 'model')}-{train_export_stem_slug(dataset_label, 'data')}-{stamp}"


def local_soul_candidate_paths(models_dir: Path, *, default_name: str = "sloughgpt.soul") -> list[Path]:
    default = models_dir / default_name
    out: list[Path] = []
    if default.exists():
        out.append(default)
    if models_dir.is_dir():
        others = sorted(
            (p for p in models_dir.glob("*.soul") if p != default),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        out.extend(others)
    return out


def ensure_server(host: str = "127.0.0.1", port: int = 8000, auto_start: bool = True) -> tuple[str, object | None]:
    """Check if the API server is running; optionally auto-start it.

    Singleton pattern: if a server is already running on *port*, reuse it.
    If the port is occupied by a loading server (unhealthy), wait for it.
    Only spawns a new subprocess if the port is genuinely free.

    Server stderr is routed to the LogBuffer (via ``LogBufferHandler``) so
    shell line-mode can display a status badge and the ``logs`` command
    without polluting the terminal.

    Returns:
        (base_url, proc_or_None) -- proc is the subprocess handle (or None
        if the server was already running).
    """
    import socket
    import subprocess
    import requests

    base_url = f"http://{host}:{port}"

    # Already healthy?
    try:
        r = requests.get(f"{base_url}/health", timeout=3)
        if r.status_code == 200:
            return base_url, None
    except requests.RequestException:
        pass

    if not auto_start:
        return base_url, None

    # Port occupied? Check if it's a loading server
    _port_busy = False
    try:
        with socket.create_connection((host, port), timeout=1.0):
            _port_busy = True
    except (ConnectionRefusedError, OSError, TimeoutError):
        pass

    if _port_busy:
        # Something is listening but not healthy yet -- wait for it
        if chat_wait_for_health(base_url, timeout_sec=120):
            return base_url, None
        return base_url, None

    # Port is free -> spawn the server via main.py
    repo = chat_repository_root()
    marker = repo / "apps" / "api" / "server" / "main.py"
    if not marker.is_file():
        return base_url, None

    from domains.shared import find_server_python
    server_python = find_server_python(repo)
    cmd = [server_python, "-m", "apps.api.server.main"]

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
    except OSError:
        return base_url, None

    # Stream stderr in background — route into LogBuffer for the shell's
    # status badge and ``logs`` command, never print to terminal.
    def _log_stderr():
        if proc.stderr:
            try:
                from domains.shell.log_buffer import get_log_buffer, LogEntry
                buf = get_log_buffer()
                for line in proc.stderr:
                    line = line.rstrip()
                    if not line:
                        continue
                    # Classify level from common uvicorn/logging prefixes
                    level = "INFO"
                    upper = line.upper()
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
                        message=line,
                    ))
            except (AttributeError, OSError):
                # LogBuffer unavailable — discard silently
                pass
    threading.Thread(target=_log_stderr, daemon=True).start()

    # Wait for health -- generous timeout covers model loading
    if chat_wait_for_health(base_url, timeout_sec=180):
        return base_url, proc

    # Timed out -- leave the server running (it may still be loading)
    return base_url, None
