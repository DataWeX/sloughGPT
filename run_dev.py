#!/usr/bin/env python3
"""
SloughGPT Dev Server - Simple CLI tool to start API + Web
Activity log style output (INF/WRN/ERR)
"""

import os
import sys
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
API_PORT = 8000
WEB_PORT = 3000


def log(level: str, msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def kill_port(port: int):
    subprocess.run(f"lsof -ti:{port} 2>/dev/null | xargs kill -9 2>/dev/null", shell=True)


def check_port(port: int) -> bool:
    try:
        import urllib.request
        urllib.request.urlopen(f"http://localhost:{port}/", timeout=1)
        return True
    except:
        return False


def find_free_port(start: int = 3000, max_attempts: int = 10) -> int:
    import socket
    for port in range(start, start + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    return start


def check_api_ready(port: int) -> bool:
    """Check if API is ready by hitting /health endpoint."""
    try:
        import urllib.request
        urllib.request.urlopen(f"http://localhost:{port}/health", timeout=3)
        return True
    except:
        return False


def main():
    model = os.environ.get("SLOUGHGT_MODEL_PATH", "/Users/mac/models/llama3.2-1b-q8_0.gguf")
    watch_web = "--watch-web" in sys.argv

    # Find Python with torch - use system python
    # Check .venv first, but prefer one with torch
    python = ROOT / ".venv" / "bin" / "python"
    if not python.exists():
        python = ROOT / "venv" / "bin" / "python"

    # Fall back to system python which has torch
    if not python.exists():
        python = Path("/usr/bin/python3")
    
    # Ensure we use python3
    if not str(python).endswith("python3"):
        python = Path("/usr/bin/python3")

    # Verify torch is available, otherwise use system python
    test_result = subprocess.run([str(python), "-c", "import torch"], capture_output=True)
    if test_result.returncode != 0:
        log("WRN", "torch not in venv, using system python")
        python = Path("/usr/bin/python3")

    log("INF", "=== SloughGPT Starting ===")
    log("INF", f"Model: {model}")
    log("INF", f"Root: {ROOT}")
    log("INF", f"Python: {python}")

    # Stop existing
    log("INF", "Stopping existing servers...")
    for port in [API_PORT, WEB_PORT]:
        kill_port(port)
    time.sleep(1)

    # Start API
    log("INF", f"Starting API server on port {API_PORT}...")
    env = os.environ.copy()
    env["SLOUGHGT_MODEL_PATH"] = model

    api_proc = subprocess.Popen(
        [str(python), "-m", "uvicorn", "apps.api.server.main:app", "--host", "0.0.0.0", "--port", str(API_PORT), "--reload"],
        cwd=ROOT,
        env=env,
    )
    api_pid = api_proc.pid

    # Wait for API (max 30s) - check health endpoint, not just port
    for i in range(30):
        if check_api_ready(API_PORT):
            log("INF", f"API ready → http://localhost:{API_PORT}")
            break
        time.sleep(1)
    else:
        log("ERR", "API failed to become ready")
        return

    # Start Web server
    log("INF", f"Starting Web server on port {WEB_PORT}{' (watching for changes)...' if watch_web else ''}...")
    
    web_cwd = ROOT / "apps/web"
    
    if watch_web:
        # Use nodemon to watch for changes in web folder
        web_proc = subprocess.Popen(
            ["npx", "nodemon", "--watch", "app", "--watch", "components", "--watch", "lib", "--watch", "hooks", "-e", "ts,tsx,js,jsx", "npm", "run", "dev"],
            cwd=web_cwd,
        )
    else:
        web_proc = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=web_cwd,
        )

    # Wait for Web (max 60s)
    for i in range(60):
        if check_port(WEB_PORT):
            log("INF", f"Web ready → http://localhost:{WEB_PORT}")
            break
        time.sleep(1)
    else:
        log("WRN", "Web may not have started")

    log("INF", "")
    log("INF", "=== SloughGPT Running ===")
    log("INF", f"API:  http://localhost:{API_PORT}")
    log("INF", f"Web:  http://localhost:{WEB_PORT}")
    log("INF", "")

    # Wait for ctrl+c
    try:
        input()
    except (KeyboardInterrupt, EOFError):
        pass

    log("INF", "Shutting down...")
    subprocess.run(f"kill {api_pid} 2>/dev/null", shell=True)
    web_proc.terminate()
    log("INF", "Done")


if __name__ == "__main__":
    main()
