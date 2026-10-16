#!/usr/bin/env python3
"""PGQ-Engine launcher: one entry point -> supervised processes -> one URL.

WanGP-style serve (card 85e15cc4): one Python entry point starts the API
(:8000) and the Vite web UI (:5173) as supervised children of a single PGQ
Engine, waits until both ports accept, and prints the one URL to open.

Why a PGQ Engine and not a bare Popen: the notebook's old Serve cell spawned
uvicorn with stdout=DEVNULL, no readiness wait, and never launched the web UI
at all, so a dead server looked alive and there was no URL to open. PGQ owns
the process primitive (Engine.run_subprocess -> SubprocessProcess, forkserver
start), and Engine.run_background() pumps dispatch on a thread, so the PGQ
core stays event-loop-free (AGENTS sync-core / async-seam).

Logs: PGQ's capture_output buffers until process exit — useless for a server —
so each child points its own fd 1/2 at logs/serve_ui-{api,web}.log before
starting the stack: live, tail-able, and captured tracebacks land there too.

Signals: Engine.__init__ installs SIGINT/SIGTERM handlers; under IPython the
kernel owns SIGINT, so start() restores the previous handlers immediately.

Degrade honestly: no Node/vite -> API only with the reason; Google Colab ->
best-effort kernel port forwarding, else the same degrade line.

CLI:
    python scripts/serve_ui.py            # start, print URL, supervise until stopped
    python scripts/serve_ui.py --help
    python scripts/serve_ui.py --no-web   # API only
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "apps" / "web"
VITE_BIN = WEB_DIR / "node_modules" / ".bin" / "vite"
LOG_DIR = ROOT / "logs"

API_PORT = int(os.environ.get("SLO_API_PORT", "8000"))
WEB_PORT = int(os.environ.get("WEB_PORT", "5173"))
READY_TIMEOUT = 90.0

# --------------------------------------------------------------------------
# Supervised targets.
# Module level and picklable: SubprocessProcess preflights the payload with
# ForkingPickler before allocating anything (forkserver rebuilds the child
# from pickled state), so no closures or lambdas may cross here.
# --------------------------------------------------------------------------


def _open_child_log(log_path: str):
    """Append this child's fd 1/2 to log_path (live, tail-able output)."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logf = open(path, "a", buffering=1)
    os.dup2(logf.fileno(), 1)
    os.dup2(logf.fileno(), 2)
    return logf


def _serve_api(log_path: str) -> int:
    """Run the FastAPI server in this supervised child.

    Module form (``-m apps.api.server.main``) so the repo root lands on
    sys.path — the same invocation dev-stack.sh health-checks green.
    """
    _open_child_log(log_path)
    cmd = [str(ROOT / "scripts" / "python"), "-m", "apps.api.server.main"]
    return int(subprocess.run(cmd, cwd=str(ROOT), env=dict(os.environ)).returncode)


def _serve_web(log_path: str, vite_bin: str) -> int:
    """Run the Vite dev server in this supervised child."""
    _open_child_log(log_path)
    cmd = [vite_bin, "--host", "--strictPort", "--port", str(WEB_PORT)]
    return int(subprocess.run(cmd, cwd=str(WEB_DIR), env=dict(os.environ)).returncode)


# --------------------------------------------------------------------------
# Environment probes
# --------------------------------------------------------------------------


def web_status() -> tuple[bool, str]:
    """(can_serve_web, reason_if_not)."""
    if not shutil.which("node"):
        return False, "no Node on PATH"
    if not VITE_BIN.is_file():
        rel = VITE_BIN.relative_to(ROOT)
        return False, f"vite missing ({rel} — run npm install in apps/web)"
    return True, ""


def _under_ipython() -> bool:
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    return get_ipython() is not None


def _colab_forward(port: int) -> str | None:
    """Best-effort Colab kernel-port forwarding; None when not on Colab."""
    try:
        from google.colab import output as colab_output  # type: ignore
    except Exception:
        return None
    try:
        colab_output.serve_kernel_port_if_connected(port)
        return f"Colab: check the 'Ports' tab — :{port} forwarding requested"
    except Exception:
        return None


def _port_open(port: int, timeout: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _wait_port(port: int, proc, timeout: float) -> bool:
    """Poll until port accepts or the supervised process dies first."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc is not None and proc.is_done:
            return False
        if _port_open(port):
            return True
        time.sleep(0.25)
    return False


def _tail(path: Path, lines: int = 12) -> str:
    try:
        return "".join(path.read_text(encoding="utf-8", errors="replace").splitlines(True)[-lines:])
    except OSError:
        return ""


# --------------------------------------------------------------------------
# Launcher state (module level so the notebook can start/stop by name)
# --------------------------------------------------------------------------

_ENGINE = None
_PROCS: dict[str, object] = {}
_LAST_INFO: dict = {}


def _announce(info: dict) -> None:
    lines = ["", "── SloughGPT stack is up " + "─" * 36]
    if info.get("web"):
        lines.append(f"  Open:  {info['url']}")
    else:
        lines.append(f"  API:   {info['api_url']}   (docs: {info['api_url']}/docs)")
        lines.append(f"  Web UI: unavailable — {info['reason']}")
    if info.get("web"):
        lines.append(f"  API:   {info['api_url']}   (docs: {info['api_url']}/docs)")
    lines.append("  Logs:  logs/serve_ui-api.log  logs/serve_ui-web.log")
    lines.append("  Stop:  serve_ui.stop()  (notebook) / Ctrl-C (CLI)")
    if info.get("colab_note"):
        lines.append(f"  {info['colab_note']}")
    lines.append("")
    print("\n".join(lines))


def start(ready_timeout: float = READY_TIMEOUT, with_web: bool = True) -> dict:
    """Start the supervised stack; return {url, api_url, web, api, reason, ...}.

    Blocks until the API port accepts (and the web port, when web is
    available). Raises RuntimeError with the child's log tail when a child
    dies before it becomes ready — never silently pretends to be up.
    """
    global _ENGINE, _PROCS, _LAST_INFO
    if _ENGINE is not None:
        return _LAST_INFO

    # Self-contained path setup: domain/ + the repo's shared toolchain roots,
    # independent of whatever PYTHONPATH the caller happened to carry.
    for extra in (
        str(ROOT),
        str(ROOT / "packages" / "core-py"),
        str(ROOT / "apps" / "api" / "server"),
    ):
        if extra not in sys.path:
            sys.path.insert(0, extra)

    from domain.infrastructure._internal.pugqeep.engine import Engine

    engine = Engine(name="serve_ui")
    # Dispatch routes ungrouped processes into a tree; with zero trees
    # dispatch() keeps returning 0 and children sit at CREATED forever (the
    # real-launch gate caught this — engine tests always create a tree first).
    # Plain Tree, not guarded: no auto-restart, the degrade must stay honest.
    engine.tree("serve")
    if _under_ipython():
        # Engine.__init__ installs SIGINT/SIGTERM handlers; in a notebook the
        # kernel owns SIGINT — hand it straight back.
        engine.restore_signal_handlers()
    _ENGINE = engine

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    api_log = LOG_DIR / "serve_ui-api.log"
    _PROCS["api"] = engine.run_subprocess(
        _serve_api,
        str(api_log),
        name="serve-ui-api",
        env={"SLO_API_PORT": str(API_PORT)},
        capture_output=False,
    )

    can_web, reason = web_status()
    if with_web and can_web:
        _PROCS["web"] = engine.run_subprocess(
            _serve_web,
            str(LOG_DIR / "serve_ui-web.log"),
            str(VITE_BIN),
            name="serve-ui-web",
            capture_output=False,
        )
    engine.run_background()

    if not _wait_port(API_PORT, _PROCS["api"], ready_timeout):
        tail = _tail(api_log)
        raise RuntimeError(
            f"API did not accept on :{API_PORT} within {ready_timeout:.0f}s "
            f"(status={_PROCS['api'].status}).\n--- {api_log} ---\n{tail}"
        )

    web_up = False
    if with_web and can_web:
        if not _wait_port(WEB_PORT, _PROCS.get("web"), ready_timeout):
            reason = f"web UI died before :{WEB_PORT} accepted (see logs/serve_ui-web.log)"
        else:
            web_up = True

    _LAST_INFO = {
        "url": f"http://localhost:{WEB_PORT}" if web_up else f"http://localhost:{API_PORT}",
        "api_url": f"http://localhost:{API_PORT}",
        "api": True,
        "web": web_up,
        "reason": reason,
        "colab_note": _colab_forward(API_PORT) if not web_up else None,
        "api_log": str(api_log),
        "web_log": str(LOG_DIR / "serve_ui-web.log"),
    }
    _announce(_LAST_INFO)
    return _LAST_INFO


def stop() -> int:
    """Cancel every supervised child and halt the engine; returns count."""
    global _ENGINE, _PROCS, _LAST_INFO
    if _ENGINE is None:
        return 0
    cancelled = _ENGINE.cancel_all()
    _ENGINE.stop()
    _ENGINE = None
    _PROCS = {}
    _LAST_INFO = {}
    return int(cancelled)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="serve_ui",
        description="One entry point -> PGQ-supervised API (:8000) + web UI (:5173) -> one URL.",
    )
    parser.add_argument(
        "--ready-timeout",
        type=float,
        default=READY_TIMEOUT,
        help=f"seconds to wait for each port to accept (default {READY_TIMEOUT:.0f})",
    )
    parser.add_argument("--no-web", action="store_true", help="start the API only")
    args = parser.parse_args(argv)

    try:
        start(ready_timeout=args.ready_timeout, with_web=not args.no_web)
    except RuntimeError as exc:
        print(f"serve_ui: {exc}", file=sys.stderr)
        stop()
        return 1

    try:
        while True:
            time.sleep(0.5)
            health_running = bool(_ENGINE is not None and _ENGINE.health()["running"])
            done = {k: p.is_done for k, p in _PROCS.items()}
            if all(done.values()) and not health_running:
                # Engine stopped underneath us: Ctrl-C path (its SIGINT handler
                # stops the engine and cancels the children) -> clean exit.
                print("serve_ui: stack stopped.")
                stop()
                return 0
            crashed = [k for k, k_done in done.items() if k_done and health_running]
            if crashed:
                name = crashed[0]
                log = LOG_DIR / f"serve_ui-{name}.log"
                print(f"serve_ui: {name} exited unexpectedly.\n{_tail(log)}", file=sys.stderr)
                stop()
                return 1
    except KeyboardInterrupt:
        # Only reachable where the engine's handlers were restored (notebook);
        # in a plain CLI the engine handler owns Ctrl-C and the loop exits.
        print("serve_ui: interrupted.")
        stop()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
