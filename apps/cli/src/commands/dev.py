"""
Dev commands - Development server, health checks, and API status.
"""
import re
import subprocess
import sys
import os
import time
import signal
import threading
import webbrowser
from pathlib import Path
from collections import deque

from domains.logging import get_global
from domains.shared import find_server_python
from utils.formatting import format_time

log = get_global()

# Component loggers — each gets its own dock in output
api_log = log.child("api")
web_log = log.child("web")
build_log = log.child("build")
mobile_log = log.child("mobile")


_LOG_BUF = 500  # max lines kept per panel


class StatusBlock:
    """Manages in-place updating of a block of status lines.

    Tries to use /dev/tty for cursor manipulation regardless of stdout.
    Falls back to logging only once if /dev/tty is unavailable.
    """

    def __init__(self, logger):
        self._log = logger
        self._lines: list[str] = []
        self._tty = None
        self._is_tty = False
        # Try /dev/tty for direct terminal access
        try:
            self._tty = open('/dev/tty', 'w')
            self._is_tty = True
        except (OSError, FileNotFoundError):
            pass
        self._non_tty_logged = False

    def _write(self, text: str) -> None:
        if self._tty:
            try:
                self._tty.write(text)
                self._tty.flush()
            except (OSError, ValueError):
                pass

    def update(self, *lines: str) -> None:
        """Update the block with new lines, clearing previous output if TTY."""
        if self._is_tty and self._lines:
            n = len(self._lines)
            # Move up n lines, clear each
            self._write(f"\033[{n}A")
            for _ in range(n):
                self._write("\033[2K")
                self._write("\033[1B")
            # Move back up to start
            self._write(f"\033[{n}A")

        self._lines = list(lines)

        if self._is_tty:
            for line in lines:
                self._write(line + "\n")
            self._tty.flush()
        elif not self._non_tty_logged:
            for line in lines:
                self._log.info(line)
            self._non_tty_logged = True


def _kill_port(port: int):
    """Kill process running on port."""
    import shlex
    result = subprocess.run(
        shlex.split(f"lsof -ti:{port}"),
        capture_output=True, text=True,
    )
    for pid in result.stdout.strip().split():
        if pid.isdigit():
            subprocess.run(["kill", "-9", pid], capture_output=True)


def _check_port(port: int) -> bool:
    """Check if a port is responding."""
    try:
        import urllib.request
        urllib.request.urlopen(f"http://localhost:{port}/", timeout=1)
        return True
    except Exception:
        return False


def _is_eaddrinuse(lines: deque) -> bool:
    """Check if output contains EADDRINUSE error."""
    for line in lines:
        if "EADDRINUSE" in line or "address already in use" in line.lower():
            return True
    return False


def _handle_eaddrinuse(port: int, service: str = "web"):
    """Display helpful EADDRINUSE error with remediation steps."""
    log.blank()
    log.key_value(service, f"port {port} in use")

    # Find what's using the port
    import shlex
    result = subprocess.run(
        shlex.split(f"lsof -ti:{port}"),
        capture_output=True, text=True,
    )
    pids = [pid for pid in result.stdout.strip().split() if pid.isdigit()]

    if pids:
        for pid in pids:
            try:
                with open(f"/proc/{pid}/comm", "r") as f:
                    proc_name = f.read().strip()
            except (FileNotFoundError, PermissionError):
                proc_name = "unknown"
            log.key_value("pid", f"{pid} ({proc_name})")
        log.blank()

    log.command(f"lsof -ti:{port} | xargs kill -9", "kill")
    log.command(f"PORT=3001 slo dev --web-port 3001", "or use another port")


def _extract_error_lines(lines: deque, max_lines: int = 40) -> list[str]:
    """Extract the most useful error lines from captured output.

    Shows the actual error instead of shutdown noise. Priority:
    1. Lines with CRITICAL/ERROR/exception/traceback/failed keywords
    2. Lines showing startup phase progress
    3. Lines showing lifecycle transitions
    4. Last N lines as fallback
    """
    all_lines = list(lines)
    if not all_lines:
        return ["(no output captured)"]

    # Noise to suppress — shutdown hook completions, uvicorn boilerplate
    noise_re = re.compile(
        r"(Shutdown hook .* completed in|Application shutdown complete|"
        r"Finished server process|Waiting for application (startup|shutdown)|"
        r"Application startup complete|Uvicorn running on)",
    )

    # Phase 1: Find actual errors and critical messages
    error_keywords = re.compile(
        r"(CRITICAL|ERROR|exception|traceback|failed|error|crashed|timed out|refused|exit code|ModuleNotFoundError|ImportError)",
        re.IGNORECASE,
    )
    error_lines = [l for l in all_lines if error_keywords.search(l)]

    # Phase 2: Find startup phase progress lines
    phase_keywords = re.compile(
        r"(Phase \d|lifecycle|startup|registering_routers|model_load|ready|preparing|Starting SloughGPT)",
        re.IGNORECASE,
    )
    phase_lines = [l for l in all_lines if phase_keywords.search(l) and not noise_re.search(l)]

    # Deduplicate while preserving order
    seen = set()
    result = []
    for l in error_lines + phase_lines:
        if l not in seen:
            seen.add(l)
            result.append(l)

    if result:
        return result[-max_lines:]

    # Fallback: last N non-noise lines
    useful = [l for l in all_lines if not noise_re.search(l)]
    return useful[-max_lines:] if useful else all_lines[-max_lines:]


def _check_api_ready(port: int) -> bool:
    """Check if API health endpoint responds."""
    try:
        import urllib.request
        urllib.request.urlopen(f"http://localhost:{port}/health", timeout=3)
        return True
    except Exception:
        return False


def _check_web_ready(port: int) -> bool:
    """Check if web frontend is responding."""
    try:
        import urllib.request
        urllib.request.urlopen(f"http://localhost:{port}/", timeout=3)
        return True
    except Exception:
        return False


def _is_port_bound(port: int) -> bool:
    """Check if a port is bound (in use) by any process."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def _find_free_port(start: int) -> int:
    """Find the next free port starting from *start*."""
    for offset in range(100):
        candidate = start + offset
        if not _is_port_bound(candidate):
            return candidate
    return start  # fallback


def _get_startup_progress(port: int) -> dict | None:
    """Fetch startup progress from /health/startup-progress."""
    try:
        import urllib.request
        import json
        resp = urllib.request.urlopen(f"http://localhost:{port}/health/startup-progress", timeout=2)
        data = json.loads(resp.read())
        return data.get("data", data)
    except Exception:
        return None


def _wait_for_api_with_progress(port: int, timeout: int = 90) -> bool:
    """Wait for API with live spinner showing startup phases."""
    from utils.progress import Spinner

    phase_names = {
        "initializing": "Initializing",
        "task_queue": "Task queue",
        "config": "Config",
        "loading_model": "Loading model",
        "wandb_server": "W&B metrics",
        "multimodal": "Multimodal",
        "model_registry": "Model registry",
        "registering_routers": "Routes",
        "ready": "Ready",
    }

    spinner = Spinner(text="Waiting for API")
    spinner.start()

    for elapsed in range(timeout):
        if _check_api_ready(port):
            spinner.stop("API ready")
            return True

        progress = _get_startup_progress(port)
        if progress:
            phase = progress.get("phase", "initializing")
            step = progress.get("step", 0)
            total = progress.get("total", 8)
            name = phase_names.get(phase, phase)
            spinner.text = f"[{step}/{total}] {name}"
        else:
            spinner.text = "Waiting for API"

        time.sleep(1)

    spinner.stop("Timed out")
    return False


def _read_stream(stream, lines: deque, stop: threading.Event, echo: bool = True, echo_event: threading.Event = None):
    """Read lines from a subprocess stream into a deque until stop is set.

    Args:
        stream: subprocess stdout/stderr pipe.
        lines: deque to accumulate lines (for later inspection on failure).
        stop: threading.Event to signal shutdown.
        echo: if True, print each line to stdout in real-time.
        echo_event: if provided, suppress echo until this event is set.
                    Useful for suppressing output during progress bar display.
    """
    waiting = echo_event is not None and not echo_event.is_set()
    try:
        for line in iter(stream.readline, ""):
            if stop.is_set():
                break
            if line:
                clean = line.rstrip("\n\r")
                lines.append(clean)
                # Check if echo should be enabled now
                if waiting and echo_event and echo_event.is_set():
                    waiting = False
                if echo and not waiting:
                    print(clean, flush=True)
            else:
                break
    except ValueError:
        pass
    finally:
        stream.close()


def _repo_root() -> Path:
    """Get the repository root from this file's location."""
    from domains.shared import find_repo_root
    return find_repo_root(str(Path(__file__).resolve()))


def cmd_dev(args):
    """Start API and Web servers with a live TUI dashboard."""
    # ── Pre-flight: check if model needs download ─────────
    _preflight_model_check(args)

    root = _repo_root()
    model = getattr(args, "model", None) or os.environ.get("SLOUGHGT_MODEL_PATH", "")
    api_port = getattr(args, "port", 8000)
    web_port = getattr(args, "web_port", 3000)
    watch_web = getattr(args, "watch_web", False)

    # ── In-place status block during startup ──────────────
    from domains.logging.cli_logger import _c as ansi_c, _A
    status_block = StatusBlock(log)
    api_status = "starting"
    web_status = "starting"

    def _update_startup_status():
        api_color = _A.GREEN if api_status == "ready" else _A.YELLOW
        web_color = _A.GREEN if web_status == "ready" else _A.YELLOW
        status_block.update(
            ansi_c("  SloughGPT", _A.BOLD, log._colors),
            f"  API: http://localhost:{api_port}  {ansi_c(api_status, api_color, log._colors)}",
            f"  Web: http://localhost:{web_port}  {ansi_c(web_status, web_color, log._colors)}",
        )

    _update_startup_status()

    status = {"api": "starting", "web": "starting", "api_ready": False, "web_ready": False}
    api_lines: deque = deque(maxlen=_LOG_BUF)
    web_lines: deque = deque(maxlen=_LOG_BUF)

    # ── Start API ────────────────────────────────────────
    env = os.environ.copy()
    env["FORCE_COLOR"] = "1"
    if model:
        env["SLOUGHGT_MODEL_PATH"] = model

    python = Path(find_server_python(root))
    api_proc = subprocess.Popen(
        [str(python), "-m", "uvicorn", "apps.api.server.main:app",
         "--host", "0.0.0.0", "--port", str(api_port), "--reload"],
        cwd=str(root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    stop_event = threading.Event()
    api_thread = threading.Thread(
        target=_read_stream, args=(api_proc.stdout, api_lines, stop_event), daemon=True
    )
    api_thread.start()

    # ── Start Web ────────────────────────────────────────
    web_cwd = root / "apps/web"

    if watch_web:
        web_proc = subprocess.Popen(
            ["npx", "nodemon", "--watch", "app", "--watch", "components",
             "--watch", "lib", "--watch", "hooks", "-e", "ts,tsx,js,jsx",
             "npm", "run", "dev"],
            cwd=str(web_cwd),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    else:
        web_proc = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=str(web_cwd),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

    web_thread = threading.Thread(
        target=_read_stream, args=(web_proc.stderr, web_lines, stop_event), daemon=True
    )
    web_thread.start()

    # ── Wait for readiness (async poll) ──────────────────
    def _poll_services():
        for _ in range(90):
            if status["api_ready"] and status["web_ready"]:
                break
            if not status["api_ready"] and _check_api_ready(api_port):
                status["api_ready"] = True
                status["api"] = "ready"
                api_status = "ready"
                _update_startup_status()
            if not status["web_ready"] and _check_port(web_port):
                status["web_ready"] = True
                status["web"] = "ready"
                web_status = "ready"
                _update_startup_status()
            # Check if web process died
            if not status["web_ready"] and web_proc.poll() is not None:
                if _is_eaddrinuse(web_lines):
                    status["web"] = "eaddrinuse"
                    web_status = "eaddrinuse"
                else:
                    status["web"] = "error"
                    web_status = "error"
                _update_startup_status()
                break
            time.sleep(0.5)

    poll_thread = threading.Thread(target=_poll_services, daemon=True)
    poll_thread.start()

    # ── TUI Dashboard ────────────────────────────────────
    from core.tui import DevDashboard, TabConfig

    dashboard = DevDashboard(
        title="SloughGPT Dev Server",
        tabs=[
            TabConfig("api", "API", api_lines, port=api_port, url_path="/docs"),
            TabConfig("web", "Web", web_lines, port=web_port),
        ],
        info={
            "Repository": str(root),
            "Model": model or "default",
            "Python": str(python),
        },
    )

    shutdown = [False]
    eaddrinuse_port = [None]  # Track EADDRINUSE for error display

    def _stop_check() -> bool:
        if shutdown[0]:
            return True
        dashboard.set_status("api", status["api"])
        dashboard.set_status("web", status["web"])
        if status["web"] == "eaddrinuse":
            eaddrinuse_port[0] = web_port
            return True
        if status["api"] == "error" and status["web"] == "error":
            return True
        return False

    def _signal_handler(sig, frame):
        shutdown[0] = True

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        dashboard.serve(stop_check=_stop_check)
    finally:
        stop_event.set()
        _cleanup(api_proc, web_proc, api_port, web_port)
        if eaddrinuse_port[0]:
            _handle_eaddrinuse(eaddrinuse_port[0], "web")
        else:
            _print_summary(api_lines, web_lines, status, api_port, web_port)


def _cleanup(api_proc, web_proc, api_port, web_port):
    """Terminate both subprocesses and free ports."""
    for proc in [api_proc, web_proc]:
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=2)
                except Exception:
                    pass
    _kill_port(api_port)
    _kill_port(web_port)


def _print_summary(api_lines, web_lines, status, api_port=8000, web_port=3000):
    """Print a clean shutdown summary to the normal console."""
    log.blank()
    log.header("Dev Server Stopped")

    st = status.get("api", "error")
    color = "green" if st == "ready" else "red"
    log.status("API Server", f"http://localhost:{api_port}", color if color == "green" else "error")

    st = status.get("web", "error")
    color = "green" if st == "ready" else "red"
    log.status("Web Server", f"http://localhost:{web_port}", color if color == "green" else "error")

    log.blank()
    log.info(f"API logs: {len(api_lines)} lines")
    log.info(f"Web logs: {len(web_lines)} lines")
    log.success("Done")


def cmd_serve(args):
    """Start HTTP inference server.

    With --web: starts full FastAPI server + Next.js web UI and opens browser.
    With --mobile: starts FastAPI server + React Native metro bundler.
    Without flags: starts the full FastAPI server only (API-only mode).

    Before starting, checks if the autoload model needs downloading and
    prompts the user for confirmation if it's over 50 MB.
    """
    # ── Pre-flight: check if model needs download ─────────
    _preflight_model_check(args)

    web = getattr(args, "web", False)
    mobile = getattr(args, "mobile", False)

    if web and mobile:
        log.error("Cannot use --web and --mobile together. Pick one.")
        return

    if mobile:
        _cmd_api_and_mobile(args)
        return

    if web:
        _cmd_api_and_web(args)
        return

    # ── API-only mode ────────────────────────────────────
    _cmd_api_only(args)


def _preflight_model_check(args):
    """Check if the configured autoload model is cached; prompt if download needed.

    Respects ``--auto-download`` flag and ``SLO_AUTO_DOWNLOAD`` env var.
    """
    model_id = (
        getattr(args, "model", None)
        or os.environ.get("SLO_AUTOLOAD_MODEL", "")
        or os.environ.get("SLOUGHGT_MODEL_PATH", "")
    )
    if not model_id:
        return

    # Only check HuggingFace IDs (local paths don't need download)
    if "/" in model_id or not Path(model_id).exists():
        from core.permissions import PermissionsManager

        pm = PermissionsManager(auto_yes=getattr(args, "auto_download", False))
        if not pm.confirm_autoload_download(model_id):
            log.info("Server start cancelled")
            raise SystemExit(0)


def _cmd_api_only(args):
    """Start FastAPI server only (no web frontend)."""
    root = _repo_root()
    api_port = getattr(args, "port", 8000)

    # ── Reuse existing healthy API or find free port ─────────────
    api_reused = False
    if _check_api_ready(api_port):
        api_reused = True
    elif _is_port_bound(api_port):
        api_port = _find_free_port(api_port + 1)

    # ── In-place status block ───────────────────────────────────
    from domains.logging.cli_logger import _c as ansi_c, _A
    status = StatusBlock(log)
    api_status = "ok (reusing)" if api_reused else "starting"

    def _update_status():
        api_color = _A.GREEN if "ok" in api_status else _A.YELLOW
        status.update(
            ansi_c("  SloughGPT API", _A.BOLD, log._colors),
            f"  API: http://{args.host}:{api_port}  {ansi_c(api_status, api_color, log._colors)}",
        )

    _update_status()

    env = os.environ.copy()
    env["FORCE_COLOR"] = "1"
    model = getattr(args, "model", None) or os.environ.get("SLOUGHGT_MODEL_PATH", "")
    if model:
        env["SLOUGHGT_MODEL_PATH"] = model
    for k in ("SLO_AUTOLOAD_MODEL", "SLO_API_PORT", "HF_TOKEN"):
        if k in os.environ:
            env[k] = os.environ[k]

    api_lines: deque = deque(maxlen=_LOG_BUF)
    stop_event = threading.Event()

    api_proc = None
    if not api_reused:
        python = Path(find_server_python(root))
        api_proc = subprocess.Popen(
            [str(python), "-m", "uvicorn", "apps.api.server.main:app",
             "--host", args.host, "--port", str(api_port)],
            cwd=str(root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        api_thread = threading.Thread(
            target=_read_stream, args=(api_proc.stdout, api_lines, stop_event), daemon=True
        )
        api_thread.start()

    if not api_reused:
        api_status = "waiting..."
        _update_status()
        api_ready = False
        for _ in range(90):
            if _check_api_ready(api_port):
                api_ready = True
                break
            time.sleep(1)
        if not api_ready:
            api_status = "error"
            _update_status()
            api_log.info("Relevant output:")
            for line in _extract_error_lines(api_lines):
                api_log.info(f"  | {line}")
            stop_event.set()
            return

    api_status = "ok"
    status.update(
        ansi_c("  SloughGPT API", _A.BOLD, log._colors),
        f"  API: http://{args.host}:{api_port}",
        "",
        f"  {ansi_c('ok', _A.GREEN, log._colors)} Ready",
        "",
        f"  Press Ctrl+C to stop",
    )

    shutdown = [False]

    def _sig_handler(sig, frame):
        if shutdown[0]:
            return
        shutdown[0] = True
        status.update(
            ansi_c("  SloughGPT API", _A.BOLD, log._colors),
            f"  API: http://{args.host}:{api_port}",
            "",
            f"  Shutting down...",
        )
        stop_event.set()
        if api_proc is not None and api_proc.poll() is None:
            try:
                api_proc.terminate()
                api_proc.wait(timeout=5)
            except Exception:
                api_proc.kill()
        status.update(
            ansi_c("  SloughGPT API", _A.BOLD, log._colors),
            f"  API: http://{args.host}:{api_port}",
            "",
            f"  {ansi_c('ok', _A.GREEN, log._colors)} Stopped",
        )

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    try:
        if api_proc is not None:
            api_proc.wait()
        else:
            while not shutdown[0]:
                time.sleep(1)
    except KeyboardInterrupt:
        if not shutdown[0]:
            shutdown[0] = True
            stop_event.set()
            if api_proc is not None:
                try:
                    api_proc.terminate()
                    api_proc.wait(timeout=5)
                except Exception:
                    api_proc.kill()
            status.update(
                ansi_c("  SloughGPT API", _A.BOLD, log._colors),
                f"  API: http://{args.host}:{api_port}",
                "",
                f"  {ansi_c('ok', _A.GREEN, log._colors)} Stopped",
            )


def _cmd_api_and_mobile(args):
    """Start FastAPI server + React Native metro bundler."""
    root = _repo_root()
    api_port = getattr(args, "port", 8000)
    mobile_root = root / "apps" / "mobile"

    if not (mobile_root / "package.json").is_file():
        mobile_log.error(f"App not found at {mobile_root}")
        return

    # ── Reuse existing healthy API or find free port ─────────────
    api_reused = False
    if _check_api_ready(api_port):
        api_reused = True
    elif _is_port_bound(api_port):
        api_port = _find_free_port(api_port + 1)

    # ── In-place status block ───────────────────────────────────
    from domains.logging.cli_logger import _c as ansi_c, _A
    status = StatusBlock(log)
    api_status = "ok (reusing)" if api_reused else "starting"
    mobile_status = "starting"

    def _update_status():
        api_color = _A.GREEN if "ok" in api_status else _A.YELLOW
        mobile_color = _A.GREEN if "ok" in mobile_status else _A.YELLOW
        status.update(
            ansi_c("  SloughGPT", _A.BOLD, log._colors),
            f"  API:    http://{args.host}:{api_port}  {ansi_c(api_status, api_color, log._colors)}",
            f"  Mobile: metro bundler (8081)  {ansi_c(mobile_status, mobile_color, log._colors)}",
        )

    _update_status()

    # ── Build env ─────────────────────────────────────────
    env = os.environ.copy()
    env["FORCE_COLOR"] = "1"
    env["GIO_USE_PORTAL"] = "0"

    nvm_dir = os.environ.get("NVM_DIR", os.path.expanduser("~/.nvm"))
    nvm_bin = os.path.join(nvm_dir, "versions", "node")
    if os.path.isdir(nvm_bin):
        for entry in os.listdir(nvm_bin):
            candidate = os.path.join(nvm_bin, entry, "bin")
            if os.path.isdir(candidate) and candidate not in env.get("PATH", ""):
                env["PATH"] = candidate + os.pathsep + env.get("PATH", "")

    model = getattr(args, "model", None) or os.environ.get("SLOUGHGT_MODEL_PATH", "")
    if model:
        env["SLOUGHGT_MODEL_PATH"] = model
    for k in ("SLO_AUTOLOAD_MODEL", "SLO_API_PORT", "HF_TOKEN"):
        if k in os.environ:
            env[k] = os.environ[k]

    env["API_BASE_URL"] = f"http://{args.host}:{api_port}"

    # ── Start FastAPI server ──────────────────────────────
    api_lines: deque = deque(maxlen=_LOG_BUF)
    mobile_lines: deque = deque(maxlen=_LOG_BUF)
    stop_event = threading.Event()

    api_proc = None
    if not api_reused:
        python = Path(find_server_python(root))
        api_proc = subprocess.Popen(
            [str(python), "-m", "uvicorn", "apps.api.server.main:app",
             "--host", args.host, "--port", str(api_port)],
            cwd=str(root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        api_thread = threading.Thread(
            target=_read_stream, args=(api_proc.stdout, api_lines, stop_event), daemon=True
        )
        api_thread.start()

    # ── Start React Native metro bundler ──────────────────
    mobile_env = {**env, "PORT": "8081"}
    mobile_proc = subprocess.Popen(
        ["npx", "react-native", "start"],
        cwd=str(mobile_root),
        env=mobile_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    mobile_thread = threading.Thread(
        target=_read_stream, args=(mobile_proc.stderr, mobile_lines, stop_event), daemon=True
    )
    mobile_thread.start()

    # ── Wait for API readiness ────────────────────────────
    if not api_reused:
        api_status = "waiting..."
        _update_status()
        api_ready = False
        for _ in range(90):
            if _check_api_ready(api_port):
                api_ready = True
                break
            time.sleep(1)
        if not api_ready:
            api_status = "error"
            _update_status()
            api_log.info("Relevant output:")
            for line in _extract_error_lines(api_lines):
                api_log.info(f"  | {line}")
            stop_event.set()
            _cleanup(api_proc, mobile_proc, api_port, 8081)
            return

    api_status = "ok"
    _update_status()

    # ── Wait for metro bundler ────────────────────────────
    mobile_status = "waiting..."
    _update_status()
    for _ in range(30):
        if _check_port(8081):
            mobile_status = "ok"
            _update_status()
            break
        if mobile_proc.poll() is not None:
            mobile_status = "error"
            _update_status()
            mobile_log.info("Relevant metro output:")
            for line in _extract_error_lines(mobile_lines):
                mobile_log.info(f"  | {line}")
            stop_event.set()
            _cleanup(api_proc, mobile_proc, api_port, 8081)
            return
        time.sleep(1)
    else:
        mobile_status = "timeout"
        _update_status()

    # ── Ready ─────────────────────────────────────────────
    api_url = f"http://{args.host}:{api_port}"

    status.update(
        ansi_c("  SloughGPT", _A.BOLD, log._colors),
        f"  API:    {api_url}",
        f"  Mobile: metro bundler (8081)",
        "",
        f"  {ansi_c('ok', _A.GREEN, log._colors)} All services ready",
        "",
        f"  Press Ctrl+C to stop",
    )

    # ── Signal handlers ───────────────────────────────────
    shutdown = [False]

    def _sig_handler(sig, frame):
        if shutdown[0]:
            return
        shutdown[0] = True
        status.update(
            ansi_c("  SloughGPT", _A.BOLD, log._colors),
            f"  API:    {api_url}",
            f"  Mobile: metro bundler (8081)",
            "",
            f"  Shutting down...",
        )
        stop_event.set()
        _cleanup(api_proc, mobile_proc, api_port, 8081)
        status.update(
            ansi_c("  SloughGPT", _A.BOLD, log._colors),
            f"  API:    {api_url}",
            f"  Mobile: metro bundler (8081)",
            "",
            f"  {ansi_c('ok', _A.GREEN, log._colors)} Stopped",
        )

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    # ── Monitor loop ──────────────────────────────────────
    try:
        while not shutdown[0]:
            time.sleep(2)
            if api_proc is not None and api_proc.poll() is not None:
                api_status = "error"
                _update_status()
                break
            if mobile_proc.poll() is not None:
                mobile_status = "restarting..."
                _update_status()
                mobile_proc = subprocess.Popen(
                    ["npx", "react-native", "start"],
                    cwd=str(mobile_root),
                    env=mobile_env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                mobile_thread = threading.Thread(
                    target=_read_stream, args=(mobile_proc.stderr, mobile_lines, stop_event), daemon=True
                )
                mobile_thread.start()
                mobile_status = "ok"
                _update_status()
    except KeyboardInterrupt:
        pass
    finally:
        if not shutdown[0]:
            stop_event.set()
            _cleanup(api_proc, mobile_proc, api_port, 8081)


def _cmd_api_and_web(args):
    """Start full FastAPI server + Next.js web frontend with browser auto-open."""
    root = _repo_root()
    api_port = getattr(args, "port", 8000)
    web_port = getattr(args, "web_port", 3000)

    # ── Reuse existing healthy services or find free ports ────────
    api_reused = False
    web_reused = False
    if _check_api_ready(api_port):
        api_reused = True
    elif _is_port_bound(api_port):
        api_port = _find_free_port(api_port + 1)

    if _check_web_ready(web_port):
        web_reused = True
    elif _is_port_bound(web_port):
        web_port = _find_free_port(web_port + 1)

    # ── In-place status block ───────────────────────────────────
    from domains.logging.cli_logger import _c as ansi_c, _A
    status = StatusBlock(log)
    api_status = "ok (reusing)" if api_reused else "starting"
    web_status = "ok (reusing)" if web_reused else "starting"

    def _update_status():
        api_color = _A.GREEN if "ok" in api_status else _A.YELLOW
        web_color = _A.GREEN if "ok" in web_status else _A.YELLOW
        status.update(
            ansi_c("  SloughGPT", _A.BOLD, log._colors),
            f"  API: http://{args.host}:{api_port}  {ansi_c(api_status, api_color, log._colors)}",
            f"  Web: http://localhost:{web_port}  {ansi_c(web_status, web_color, log._colors)}",
        )

    _update_status()

    # ── Build env with model overrides ──────────────────────────
    env = os.environ.copy()
    env["FORCE_COLOR"] = "1"

    # Suppress GNOME keyring warnings (epiphany secret storage)
    env["GIO_USE_PORTAL"] = "0"

    # Ensure nvm-installed node/npx/npm are on PATH for subprocesses
    nvm_dir = os.environ.get("NVM_DIR", os.path.expanduser("~/.nvm"))
    nvm_bin = os.path.join(nvm_dir, "versions", "node")
    if os.path.isdir(nvm_bin):
        for entry in os.listdir(nvm_bin):
            candidate = os.path.join(nvm_bin, entry, "bin")
            if os.path.isdir(candidate) and candidate not in env.get("PATH", ""):
                env["PATH"] = candidate + os.pathsep + env.get("PATH", "")

    model = getattr(args, "model", None) or os.environ.get("SLOUGHGT_MODEL_PATH", "")
    if model:
        env["SLOUGHGT_MODEL_PATH"] = model
    # Pass through training-relevant env vars
    for k in ("SLO_AUTOLOAD_MODEL", "SLO_API_PORT", "HF_TOKEN"):
        if k in os.environ:
            env[k] = os.environ[k]

    # Set NEXTAUTH_URL to suppress NextAuth warning
    if "NEXTAUTH_URL" not in env:
        env["NEXTAUTH_URL"] = f"http://localhost:{web_port}"

    # ── Stream buffers ──────────────────────────────────────────
    api_lines: deque = deque(maxlen=_LOG_BUF)
    web_lines: deque = deque(maxlen=_LOG_BUF)
    stop_event = threading.Event()
    api_ready_event = threading.Event()  # suppress echo until API is ready

    # ── Start FastAPI server ─────────────────────────────────────
    python = Path(find_server_python(root))
    api_proc = None
    if not api_reused:
        api_proc = subprocess.Popen(
            [str(python), "-m", "uvicorn", "apps.api.server.main:app",
             "--host", args.host, "--port", str(api_port)],
            cwd=str(root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        api_thread = threading.Thread(
            target=_read_stream, args=(api_proc.stdout, api_lines, stop_event),
            kwargs={"echo_event": api_ready_event}, daemon=True,
        )
        api_thread.start()

    # ── Build standalone if needed ──────────────────────────────
    web_root = root / "apps" / "web"
    standalone_dir = web_root / ".next" / "standalone"
    server_js_candidates = [
        standalone_dir / "server.js",
        standalone_dir / "apps" / "web" / "server.js",
    ]
    server_js = next((p for p in server_js_candidates if p.is_file()), server_js_candidates[0])

    if not server_js.is_file():
        build_log.step("Building Next.js standalone (first time)...")
        # Force-clean .next to avoid stale/locked artifacts on macOS
        next_cache = web_root / ".next"
        if next_cache.is_dir():
            subprocess.run(["rm", "-rf", str(next_cache)], check=False)
        build_env = {**env, "NEXT_TELEMETRY_DISABLED": "1"}
        build_proc = subprocess.Popen(
            ["npx", "next", "build"],
            cwd=str(web_root),
            env=build_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        build_lines = deque(maxlen=200)
        build_thread = threading.Thread(
            target=_read_stream, args=(build_proc.stdout, build_lines, stop_event), daemon=True
        )
        build_thread.start()
        build_proc.wait()
        if build_proc.returncode != 0:
            build_log.error("Next.js build failed")
            build_log.info("Relevant build output:")
            for line in _extract_error_lines(build_lines):
                build_log.info(f"  | {line}")
            stop_event.set()
            return
        build_log.success("Build complete")

    # ── Copy static assets for standalone ───────────────────────
    standalone_root = server_js.parent
    static_src = web_root / ".next" / "static"
    static_dst = standalone_root / ".next" / "static"
    if static_src.is_dir() and not static_dst.is_dir():
        import shutil
        static_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(static_src, static_dst)

    public_src = web_root / "public"
    public_dst = standalone_root / "public"
    if public_src.is_dir() and not public_dst.is_dir():
        import shutil
        for dirpath, dirnames, filenames in os.walk(public_src, followlinks=False):
            rel = os.path.relpath(dirpath, public_src)
            dst_dir = public_dst / rel
            dst_dir.mkdir(parents=True, exist_ok=True)
            for f in filenames:
                src_file = os.path.join(dirpath, f)
                if os.path.islink(src_file) and not os.path.exists(src_file):
                    continue
                shutil.copy2(src_file, dst_dir / f)

    # ── Start Web frontend ───────────────────────────────────────
    web_env = {
        **env,
        "PORT": str(web_port),
        "HOSTNAME": "0.0.0.0",
        "NEXT_PUBLIC_API_URL": os.environ.get("NEXT_PUBLIC_API_URL", f"http://{args.host}:{api_port}"),
    }

    web_proc = None
    if not web_reused:
        if server_js.is_file():
            web_proc = subprocess.Popen(
                ["node", "server.js"],
                cwd=str(server_js.parent.resolve()),
                env=web_env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        else:
            web_proc = subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=str(web_root.resolve()),
                env=web_env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )

        web_thread = threading.Thread(
            target=_read_stream, args=(web_proc.stderr, web_lines, stop_event),
            kwargs={"echo_event": api_ready_event}, daemon=True,
        )
        web_thread.start()

    # ── Wait for API readiness ────────────────────────────────────
    if not api_reused:
        api_status = "waiting..."
        _update_status()
        api_ready = False
        for _ in range(90):
            if _check_api_ready(api_port):
                api_ready = True
                break
            time.sleep(1)
        if not api_ready:
            api_status = "error"
            _update_status()
            api_log.info("Relevant output:")
            for line in _extract_error_lines(api_lines):
                api_log.info(f"  | {line}")
            stop_event.set()
            _cleanup(api_proc, web_proc, api_port, web_port)
            return

    # API ready — enable echo on stream threads
    api_ready_event.set()
    api_status = "ok"
    _update_status()

    # ── Wait for web readiness ────────────────────────────────────
    web_status = "waiting..."
    _update_status()
    for _ in range(60):
        if _check_port(web_port):
            web_status = "ok"
            _update_status()
            break
        # Check if web process died
        if web_proc is not None and web_proc.poll() is not None:
            if _is_eaddrinuse(web_lines):
                _handle_eaddrinuse(web_port, "web")
            else:
                web_status = "error"
                _update_status()
                web_log.info("Relevant web output:")
                for line in _extract_error_lines(web_lines):
                    web_log.info(f"  | {line}")
            stop_event.set()
            _cleanup(api_proc, web_proc, api_port, web_port)
            return
        time.sleep(1)
    else:
        web_status = "timeout"
        _update_status()

    # ── Ready ────────────────────────────────────────────────────
    web_url = f"http://localhost:{web_port}"
    api_url = f"http://{args.host}:{api_port}"

    status.update(
        ansi_c("  SloughGPT", _A.BOLD, log._colors),
        f"  API: {api_url}  {ansi_c('ok', _A.GREEN, log._colors)}",
        f"  Web: {web_url}  {ansi_c('ok', _A.GREEN, log._colors)}",
        "",
        f"  Press Ctrl+C to stop",
    )

    # ── Auto-open browser ────────────────────────────────────────
    def _open_browser():
        time.sleep(1.5)
        try:
            webbrowser.open(web_url)
        except Exception:
            pass

    browser_thread = threading.Thread(target=_open_browser, daemon=True)
    browser_thread.start()

    # ── Signal handlers for clean shutdown ──────────────────────
    shutdown = [False]

    def _sig_handler(sig, frame):
        if shutdown[0]:
            return
        shutdown[0] = True
        log.blank()
        log.info("Shutting down...")
        stop_event.set()
        _cleanup(api_proc, web_proc, api_port, web_port)
        log.success("Stopped")

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    # ── Monitor loop: restart crashed web, detect API death ─────
    try:
        while not shutdown[0]:
            time.sleep(2)
            # API crashed (only if we own the process)
            if api_proc is not None and api_proc.poll() is not None:
                log.error(f"API server exited (code {api_proc.returncode})")
                break
            # Web crashed — restart it (unless EADDRINUSE)
            if web_proc.poll() is not None:
                if _is_eaddrinuse(web_lines):
                    _handle_eaddrinuse(web_port, "web")
                    break
                log.warning(f"Web server exited (code {web_proc.returncode}), restarting...")
                web_cwd = str(server_js.parent.resolve()) if server_js.is_file() else str(web_root.resolve())
                web_proc = subprocess.Popen(
                    ["node", "server.js"] if server_js.is_file() else ["npm", "run", "dev"],
                    cwd=web_cwd,
                    env=web_env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                web_thread = threading.Thread(
                    target=_read_stream, args=(web_proc.stderr, web_lines, stop_event), daemon=True
                )
                web_thread.start()
    except KeyboardInterrupt:
        pass
    finally:
        if not shutdown[0]:
            shutdown[0] = True
            stop_event.set()
            _cleanup(api_proc, web_proc, api_port, web_port)
            log.success("Stopped")


def cmd_health(args):
    """Check API health status."""
    import requests

    base_url = f"http://{args.host}:{args.port}"

    log.header("API Health Check")
    log.key_value("Endpoint", f"{base_url}/health")

    try:
        import time
        start = time.time()
        response = requests.get(f"{base_url}/health", timeout=5)
        elapsed = time.time() - start

        if response.status_code == 200:
            data = response.json()
            log.success(f"Healthy ({format_time(elapsed)})")
            log.blank()

            for key, value in data.items():
                log.key_value(key, str(value))
        else:
            log.error(f"Unhealthy (HTTP {response.status_code})")
            log.info(response.text)
    except requests.ConnectionError:
        log.error("API not reachable")
        log.info(f"Is the server running on {base_url}?")
    except Exception as e:
        log.error(f"Health check failed: {e}")


def cmd_api_status(args):
    """Show detailed API status."""
    import requests

    base_url = f"http://{args.host}:{args.port}"

    log.header("SloughGPT API Status")

    endpoints = [
        ("Health", f"{base_url}/health"),
        ("Detailed Health", f"{base_url}/health/detailed"),
        ("Rate Limit", f"{base_url}/rate-limit/status"),
        ("Cache Stats", f"{base_url}/cache/stats"),
        ("Security", f"{base_url}/security/keys"),
    ]

    for name, url in endpoints:
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                log.status(name, "OK", "ok")
            else:
                log.status(name, f"HTTP {r.status_code}", "warn")
        except Exception:
            log.status(name, "Not reachable", "error")

    # Check metrics
    try:
        r = requests.get(f"{base_url}/metrics", timeout=5)
        if r.status_code == 200:
            data = r.json()
            log.blank()
            log.section("Metrics")
            log.key_value("WebSocket Connections", str(data.get("websocket_connections", "N/A")))
            log.key_value("Active Clients", str(data.get("active_clients", "N/A")))
            log.key_value("CPU", f"{data.get('system', {}).get('cpu_percent', 'N/A')}%")
            log.key_value("Memory", f"{data.get('system', {}).get('memory_percent', 'N/A')}%")
    except Exception:
        log.info("  (metrics endpoint not available)")


def cmd_api_test(args):
    """Test API endpoints."""
    import requests
    import time

    base_url = f"http://{args.host}:{args.port}"

    log.header("API Endpoint Tests")

    # Test generation
    log.step("Testing /generate...")
    try:
        start = time.time()
        r = requests.post(
            f"{base_url}/generate",
            json={"prompt": "Hello world", "max_new_tokens": 10},
            timeout=30,
        )
        elapsed = time.time() - start
        if r.status_code == 200:
            log.success(f"Generation OK ({format_time(elapsed)})")
        else:
            log.error(f"Generation failed ({r.status_code})")
    except Exception as e:
        log.error(f"Generation: {e}")

    # Test health
    log.step("Testing /health...")
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        if r.status_code == 200:
            log.success("Health OK")
        else:
            log.error(f"Health failed ({r.status_code})")
    except Exception as e:
        log.error(f"Health: {e}")


def cmd_api_auth(args):
    """Test API authentication."""
    import requests

    base_url = f"http://{args.host}:{args.port}"

    log.header("API Authentication Test")

    log.step("Testing generate without auth...")
    try:
        r = requests.post(f"{base_url}/generate", json={"prompt": "Hello", "max_new_tokens": 5}, timeout=10)
        if r.status_code == 200:
            log.status("No Auth", "Open (200)", "ok")
        else:
            log.status("No Auth", f"Protected ({r.status_code})", "warn")
    except Exception as e:
        log.error(str(e))

    log.step("Testing token endpoint...")
    try:
        r = requests.post(f"{base_url}/auth/token", json={"api_key": "test-key"}, timeout=10)
        if r.status_code == 401:
            log.status("Token", "Rejected bad key (401)", "ok")
        elif r.status_code == 200:
            log.status("Token", "Accepted", "info")
        else:
            log.status("Token", f"HTTP {r.status_code}", "warn")
    except Exception as e:
        log.info(f"No auth endpoint: {e}")

    log.step("Testing verify endpoint...")
    try:
        r = requests.post(f"{base_url}/auth/verify", headers={"Authorization": "Bearer invalid"}, timeout=10)
        log.status("Verify", f"HTTP {r.status_code}", "ok" if r.status_code in (401, 403) else "warn")
    except Exception as e:
        log.info(f"No verify endpoint: {e}")


def cmd_hf_serve(args):
    """Serve a HuggingFace model via API."""
    import requests

    base_url = f"http://{args.host}:{args.port}"

    log.header("Serving HuggingFace Model")
    log.key_value("Model", args.model)
    log.key_value("API", base_url)

    try:
        response = requests.post(
            f"{base_url}/models/load",
            json={"model_id": args.model, "mode": args.mode, "device": args.device},
            timeout=120,
        )
        if response.ok:
            log.success(f"Model loaded: {response.json()}")
        else:
            log.error(f"Failed: {response.text}")
    except Exception as e:
        log.error(f"API error: {e}")
        log.info("Make sure the API server is running: python3 cli.py dev")

