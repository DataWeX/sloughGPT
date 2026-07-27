"""
Dev commands - Development server, health checks, and API status.
"""
import subprocess
import sys
import os
import time
import signal
import threading
import webbrowser
from pathlib import Path
from collections import deque

from core.printer import printer
from utils.formatting import format_time


_LOG_BUF = 500  # max lines kept per panel


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


def _check_api_ready(port: int) -> bool:
    """Check if API health endpoint responds."""
    try:
        import urllib.request
        urllib.request.urlopen(f"http://localhost:{port}/health", timeout=3)
        return True
    except Exception:
        return False


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
    """Wait for API with live ProgressBar showing startup phases."""
    from utils.progress import ProgressBar

    bar = ProgressBar(total=timeout, desc="Starting API", width=30, show_eta=True)
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

    for elapsed in range(timeout):
        if _check_api_ready(port):
            bar.set_progress(timeout)
            bar.desc = "API ready"
            bar.finish()
            return True

        progress = _get_startup_progress(port)
        if progress:
            phase = progress.get("phase", "initializing")
            step = progress.get("step", 0)
            total = progress.get("total", 9)
            msg = progress.get("message", "")
            name = phase_names.get(phase, phase)
            bar.desc = f"[{step}/{total}] {name}"
            bar.set_progress(min(elapsed, timeout - 1))
        else:
            bar.desc = "Waiting for API"
            bar.set_progress(min(elapsed, timeout - 1))

        time.sleep(1)

    bar.desc = "Timed out"
    bar.finish()
    return False


def _read_stream(stream, lines: deque, stop: threading.Event, echo: bool = True):
    """Read lines from a subprocess stream into a deque until stop is set.

    Args:
        stream: subprocess stdout/stderr pipe.
        lines: deque to accumulate lines (for later inspection on failure).
        stop: threading.Event to signal shutdown.
        echo: if True, print each line to stdout in real-time.
    """
    from datetime import datetime

    try:
        for line in iter(stream.readline, ""):
            if stop.is_set():
                break
            if line:
                clean = line.rstrip("\n\r")
                lines.append(clean)
                if echo:
                    # Add timestamp to all output
                    ts = datetime.now().strftime("%H:%M:%S")
                    # Clear any progress bar on current line, then print log
                    sys.stdout.write(f"\r\033[2K\033[90m│\033[0m {ts} {clean}\n")
                    sys.stdout.flush()
            else:
                break
    except ValueError:
        pass
    finally:
        stream.close()


def _repo_root() -> Path:
    """Get the repository root from this file's location."""
    return Path(__file__).resolve().parent.parent.parent.parent.parent


def cmd_dev(args):
    """Start API and Web servers with a live TUI dashboard."""
    # ── Pre-flight: check if model needs download ─────────
    _preflight_model_check(args)

    root = _repo_root()
    model = getattr(args, "model", None) or os.environ.get("SLOUGHGT_MODEL_PATH", "")
    api_port = getattr(args, "port", 8000)
    web_port = getattr(args, "web_port", 3000)
    watch_web = getattr(args, "watch_web", False)

    status = {"api": "starting", "web": "starting", "api_ready": False, "web_ready": False}
    api_lines: deque = deque(maxlen=_LOG_BUF)
    web_lines: deque = deque(maxlen=_LOG_BUF)

    # Kill existing
    printer.step("Stopping existing servers...")
    for port in [api_port, web_port]:
        _kill_port(port)
    time.sleep(0.5)

    # ── Start API ────────────────────────────────────────
    printer.step(f"Starting API on port {api_port}...")
    env = os.environ.copy()
    if model:
        env["SLOUGHGT_MODEL_PATH"] = model

    python = Path(sys.executable)
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
    printer.step(f"Starting Web on port {web_port}...")
    web_cwd = root / "apps/web"

    if watch_web:
        web_proc = subprocess.Popen(
            ["npx", "nodemon", "--watch", "app", "--watch", "components",
             "--watch", "lib", "--watch", "hooks", "-e", "ts,tsx,js,jsx",
             "npm", "run", "dev"],
            cwd=str(web_cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    else:
        web_proc = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=str(web_cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    web_thread = threading.Thread(
        target=_read_stream, args=(web_proc.stdout, web_lines, stop_event), daemon=True
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
            if not status["web_ready"] and _check_port(web_port):
                status["web_ready"] = True
                status["web"] = "ready"
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

    def _stop_check() -> bool:
        if shutdown[0]:
            return True
        dashboard.set_status("api", status["api"])
        dashboard.set_status("web", status["web"])
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
    printer.blank()
    printer.header("Dev Server Stopped")

    st = status.get("api", "error")
    color = "green" if st == "ready" else "red"
    printer.status("API Server", f"http://localhost:{api_port}", color if color == "green" else "error")

    st = status.get("web", "error")
    color = "green" if st == "ready" else "red"
    printer.status("Web Server", f"http://localhost:{web_port}", color if color == "green" else "error")

    printer.blank()
    printer.info(f"API logs: {len(api_lines)} lines")
    printer.info(f"Web logs: {len(web_lines)} lines")
    printer.success("Done")


def cmd_serve(args):
    """Start HTTP inference server.

    With --web: starts full FastAPI server + Next.js web UI and opens browser.
    Without --web: starts the full FastAPI server only (API-only mode).

    Before starting, checks if the autoload model needs downloading and
    prompts the user for confirmation if it's over 50 MB.
    """
    # ── Pre-flight: check if model needs download ─────────
    _preflight_model_check(args)

    web = getattr(args, "web", False)

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
            printer.info("Server start cancelled")
            raise SystemExit(0)


def _cmd_api_only(args):
    """Start FastAPI server only (no web frontend)."""
    root = _repo_root()
    api_port = getattr(args, "port", 8000)

    _kill_port(api_port)
    time.sleep(0.3)

    printer.header("Starting SloughGPT API Server")
    printer.key_value("API", f"http://{args.host}:{api_port}")
    printer.key_value("Docs", f"http://{args.host}:{api_port}/docs")

    env = os.environ.copy()
    model = getattr(args, "model", None) or os.environ.get("SLOUGHGT_MODEL_PATH", "")
    if model:
        env["SLOUGHGT_MODEL_PATH"] = model
    for k in ("SLO_AUTOLOAD_MODEL", "SLO_API_PORT", "HF_TOKEN"):
        if k in os.environ:
            env[k] = os.environ[k]

    api_lines: deque = deque(maxlen=_LOG_BUF)
    stop_event = threading.Event()

    python = Path(sys.executable)
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

    if not _wait_for_api_with_progress(api_port):
        printer.error("API failed to start within 90s")
        printer.info("Last output:")
        for line in list(api_lines)[-20:]:
            printer.info(f"  | {line}")
        stop_event.set()
        _kill_port(api_port)
        return

    printer.success(f"API ready at http://{args.host}:{api_port}")
    printer.info("Press Ctrl+C to stop")

    shutdown = [False]

    def _sig_handler(sig, frame):
        if shutdown[0]:
            return
        shutdown[0] = True
        print()
        printer.info("Shutting down...")
        stop_event.set()
        if api_proc.poll() is None:
            try:
                api_proc.terminate()
                api_proc.wait(timeout=5)
            except Exception:
                api_proc.kill()
        _kill_port(api_port)
        printer.success("Stopped")

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    try:
        api_proc.wait()
    except KeyboardInterrupt:
        if not shutdown[0]:
            shutdown[0] = True
            stop_event.set()
            try:
                api_proc.terminate()
                api_proc.wait(timeout=5)
            except Exception:
                api_proc.kill()
            _kill_port(api_port)
            printer.success("Stopped")


def _cmd_api_and_web(args):
    """Start full FastAPI server + Next.js web frontend with browser auto-open."""
    root = _repo_root()
    api_port = getattr(args, "port", 8000)
    web_port = getattr(args, "web_port", 3000)

    # Kill existing processes on these ports
    _kill_port(api_port)
    _kill_port(web_port)
    time.sleep(0.5)

    printer.header("Starting SloughGPT — API + Web")
    printer.key_value("API", f"http://{args.host}:{api_port}")
    printer.key_value("Web", f"http://localhost:{web_port}")

    # ── Build env with model overrides ──────────────────────────
    env = os.environ.copy()

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

    # ── Start FastAPI server ─────────────────────────────────────
    python = Path(sys.executable)
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

    # ── Build standalone if needed ──────────────────────────────
    web_root = root / "apps" / "web"
    standalone_dir = web_root / ".next" / "standalone"
    server_js = standalone_dir / "server.js"

    if not server_js.is_file():
        printer.step("Building Next.js standalone (first time)...")
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
            printer.error("Next.js build failed")
            printer.info("Build output (last 20 lines):")
            for line in list(build_lines)[-20:]:
                printer.info(f"  | {line}")
            stop_event.set()
            _kill_port(api_port)
            return
        printer.success("Build complete")

    # ── Copy static assets for standalone ───────────────────────
    static_src = web_root / ".next" / "static"
    static_dst = standalone_dir / ".next" / "static"
    if static_src.is_dir() and not static_dst.is_dir():
        import shutil
        static_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(static_src, static_dst)

    public_src = web_root / "public"
    public_dst = standalone_dir / "public"
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
        "NEXT_PUBLIC_API_URL": f"http://{args.host}:{api_port}",
    }

    if server_js.is_file():
        printer.step(f"Starting Web (standalone) on port {web_port}...")
        web_proc = subprocess.Popen(
            ["node", "server.js"],
            cwd=str(standalone_dir),
            env=web_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    else:
        printer.step(f"Starting Web (dev) on port {web_port}...")
        web_proc = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=str(web_root),
            env=web_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    web_thread = threading.Thread(
        target=_read_stream, args=(web_proc.stdout, web_lines, stop_event), daemon=True
    )
    web_thread.start()

    # ── Wait for API readiness ────────────────────────────────────
    if not _wait_for_api_with_progress(api_port):
        printer.error("API failed to start within 90s")
        printer.info("Last API output:")
        for line in list(api_lines)[-20:]:
            printer.info(f"  | {line}")
        stop_event.set()
        _cleanup(api_proc, web_proc, api_port, web_port)
        return

    printer.success("API ready")

    # ── Wait for web readiness ────────────────────────────────────
    printer.step("Waiting for web frontend...")
    for _ in range(60):
        if _check_port(web_port):
            break
        # Check if web process died
        if web_proc.poll() is not None:
            printer.error(f"Web server exited with code {web_proc.returncode}")
            printer.info("Last web output:")
            for line in list(web_lines)[-20:]:
                printer.info(f"  | {line}")
            stop_event.set()
            _cleanup(api_proc, web_proc, api_port, web_port)
            return
        time.sleep(1)
    else:
        printer.warning("Web frontend did not respond within 60s")
        printer.info("API is still running — web may need manual start")

    # ── Ready ────────────────────────────────────────────────────
    web_url = f"http://localhost:{web_port}"
    api_url = f"http://{args.host}:{api_port}"

    print()
    printer.success("All services ready!")
    print()
    printer.info(f"  API:      {api_url}")
    printer.info(f"  API Docs: {api_url}/docs")
    printer.info(f"  Web UI:   {web_url}")
    print()
    printer.info("Press Ctrl+C to stop")
    print()

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
        print()
        printer.info("Shutting down...")
        stop_event.set()
        _cleanup(api_proc, web_proc, api_port, web_port)
        printer.success("Stopped")

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    # ── Monitor loop: restart crashed web, detect API death ─────
    try:
        while not shutdown[0]:
            time.sleep(2)
            # API crashed
            if api_proc.poll() is not None:
                printer.error(f"API server exited (code {api_proc.returncode})")
                break
            # Web crashed — restart it
            if web_proc.poll() is not None:
                printer.warning(f"Web server exited (code {web_proc.returncode}), restarting...")
                web_proc = subprocess.Popen(
                    ["node", "server.js"] if server_js.is_file() else ["npm", "run", "dev"],
                    cwd=str(standalone_dir if server_js.is_file() else web_root),
                    env=web_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                web_thread = threading.Thread(
                    target=_read_stream, args=(web_proc.stdout, web_lines, stop_event), daemon=True
                )
                web_thread.start()
    except KeyboardInterrupt:
        pass
    finally:
        if not shutdown[0]:
            shutdown[0] = True
            stop_event.set()
            _cleanup(api_proc, web_proc, api_port, web_port)
            printer.success("Stopped")


def cmd_health(args):
    """Check API health status."""
    import requests

    base_url = f"http://{args.host}:{args.port}"

    printer.header("API Health Check")
    printer.key_value("Endpoint", f"{base_url}/health")

    try:
        import time
        start = time.time()
        response = requests.get(f"{base_url}/health", timeout=5)
        elapsed = time.time() - start

        if response.status_code == 200:
            data = response.json()
            printer.success(f"Healthy ({format_time(elapsed)})")
            printer.blank()

            for key, value in data.items():
                printer.key_value(key, str(value))
        else:
            printer.error(f"Unhealthy (HTTP {response.status_code})")
            printer.info(response.text)
    except requests.ConnectionError:
        printer.error("API not reachable")
        printer.info(f"Is the server running on {base_url}?")
    except Exception as e:
        printer.error(f"Health check failed: {e}")


def cmd_api_status(args):
    """Show detailed API status."""
    import requests

    base_url = f"http://{args.host}:{args.port}"

    printer.header("SloughGPT API Status")

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
                printer.status(name, "OK", "ok")
            else:
                printer.status(name, f"HTTP {r.status_code}", "warn")
        except Exception:
            printer.status(name, "Not reachable", "error")

    # Check metrics
    try:
        r = requests.get(f"{base_url}/metrics", timeout=5)
        if r.status_code == 200:
            data = r.json()
            printer.blank()
            printer.section("Metrics")
            printer.key_value("WebSocket Connections", str(data.get("websocket_connections", "N/A")))
            printer.key_value("Active Clients", str(data.get("active_clients", "N/A")))
            printer.key_value("CPU", f"{data.get('system', {}).get('cpu_percent', 'N/A')}%")
            printer.key_value("Memory", f"{data.get('system', {}).get('memory_percent', 'N/A')}%")
    except Exception:
        printer.info("  (metrics endpoint not available)")


def cmd_api_test(args):
    """Test API endpoints."""
    import requests
    import time

    base_url = f"http://{args.host}:{args.port}"

    printer.header("API Endpoint Tests")

    # Test generation
    printer.step("Testing /generate...")
    try:
        start = time.time()
        r = requests.post(
            f"{base_url}/generate",
            json={"prompt": "Hello world", "max_new_tokens": 10},
            timeout=30,
        )
        elapsed = time.time() - start
        if r.status_code == 200:
            printer.success(f"Generation OK ({format_time(elapsed)})")
        else:
            printer.error(f"Generation failed ({r.status_code})")
    except Exception as e:
        printer.error(f"Generation: {e}")

    # Test health
    printer.step("Testing /health...")
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        if r.status_code == 200:
            printer.success("Health OK")
        else:
            printer.error(f"Health failed ({r.status_code})")
    except Exception as e:
        printer.error(f"Health: {e}")


def cmd_api_auth(args):
    """Test API authentication."""
    import requests

    base_url = f"http://{args.host}:{args.port}"

    printer.header("API Authentication Test")

    printer.step("Testing generate without auth...")
    try:
        r = requests.post(f"{base_url}/generate", json={"prompt": "Hello", "max_new_tokens": 5}, timeout=10)
        if r.status_code == 200:
            printer.status("No Auth", "Open (200)", "ok")
        else:
            printer.status("No Auth", f"Protected ({r.status_code})", "warn")
    except Exception as e:
        printer.error(str(e))

    printer.step("Testing token endpoint...")
    try:
        r = requests.post(f"{base_url}/auth/token", json={"api_key": "test-key"}, timeout=10)
        if r.status_code == 401:
            printer.status("Token", "Rejected bad key (401)", "ok")
        elif r.status_code == 200:
            printer.status("Token", "Accepted", "info")
        else:
            printer.status("Token", f"HTTP {r.status_code}", "warn")
    except Exception as e:
        printer.info(f"No auth endpoint: {e}")

    printer.step("Testing verify endpoint...")
    try:
        r = requests.post(f"{base_url}/auth/verify", headers={"Authorization": "Bearer invalid"}, timeout=10)
        printer.status("Verify", f"HTTP {r.status_code}", "ok" if r.status_code in (401, 403) else "warn")
    except Exception as e:
        printer.info(f"No verify endpoint: {e}")


def cmd_hf_serve(args):
    """Serve a HuggingFace model via API."""
    import requests

    base_url = f"http://{args.host}:{args.port}"

    printer.header("Serving HuggingFace Model")
    printer.key_value("Model", args.model)
    printer.key_value("API", base_url)

    try:
        response = requests.post(
            f"{base_url}/models/load",
            json={"model_id": args.model, "mode": args.mode, "device": args.device},
            timeout=120,
        )
        if response.ok:
            printer.success(f"Model loaded: {response.json()}")
        else:
            printer.error(f"Failed: {response.text}")
    except Exception as e:
        printer.error(f"API error: {e}")
        printer.info("Make sure the API server is running: python3 cli.py dev")

