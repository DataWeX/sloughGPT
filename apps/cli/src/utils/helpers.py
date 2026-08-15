"""Shared CLI helper functions — used by commands/ modules."""
import re
import sys
import os
import time
from pathlib import Path


def chat_repository_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


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
        except Exception:
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
    """Check if server is running, optionally auto-start it.

    Returns (base_url, server_proc_or_None).
    If auto_start=True and server is down, spawns uvicorn and waits for health.
    """
    import subprocess
    import tempfile
    import requests
    from requests.exceptions import ConnectionError as RequestsConnectionError

    base_url = f"http://{host}:{port}"

    def _reachable() -> bool:
        try:
            r = requests.get(f"{base_url}/health", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    if _reachable():
        return base_url, None

    if not auto_start:
        return base_url, None

    repo = chat_repository_root()
    marker = repo / "apps" / "api" / "server" / "main.py"
    if not marker.is_file():
        return base_url, None

    bind_host = chat_uvicorn_bind_host(host)
    try:
        listen_port = chat_find_available_port(bind_host, port)
    except RuntimeError:
        return base_url, None

    log_f = tempfile.NamedTemporaryFile(prefix="sloughgpt-", suffix=".log", delete=False)
    log_path = log_f.name
    log_f.close()

    cmd = [
        sys.executable, "-m", "uvicorn", "main:app",
        "--app-dir", str(repo / "apps" / "api" / "server"),
        "--host", bind_host, "--port", str(listen_port),
    ]

    try:
        with open(log_path, "wb") as out:
            proc = subprocess.Popen(cmd, cwd=str(repo), stdout=out, stderr=subprocess.STDOUT)
    except OSError:
        try:
            os.unlink(log_path)
        except OSError:
            pass
        return base_url, None

    actual_url = f"http://{host}:{listen_port}"
    if chat_wait_for_health(actual_url):
        return actual_url, proc

    # Failed to start
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
    try:
        os.unlink(log_path)
    except OSError:
        pass
    return actual_url, None
