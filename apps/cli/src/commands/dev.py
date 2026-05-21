"""
Dev commands - Development server, health checks, and API status.
"""
import subprocess
import sys
import os
import time
from pathlib import Path
from typing import Optional

from core.printer import printer
from utils.formatting import format_time


def _kill_port(port: int):
    """Kill process running on port."""
    subprocess.run(f"lsof -ti:{port} 2>/dev/null | xargs kill -9 2>/dev/null", shell=True)


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


def cmd_dev(args):
    """Start API and Web servers in development mode."""
    from ..cli import _chat_repository_root

    root = _chat_repository_root()
    model = getattr(args, "model", None) or os.environ.get("SLOUGHGT_MODEL_PATH", "")

    printer.header("Starting Development Servers")
    printer.key_value("Repository", str(root))
    if model:
        printer.key_value("Model", model)

    api_port = getattr(args, "port", 8000)
    web_port = getattr(args, "web_port", 3000)
    watch_web = getattr(args, "watch_web", False)

    # Find python with torch
    import sys as _sys
    python = Path(_sys.executable)

    # Kill existing processes
    printer.step("Stopping existing servers...")
    _kill_port(api_port)
    _kill_port(web_port)
    time.sleep(1)

    # Start API
    printer.step(f"Starting API server on port {api_port}...")
    env = os.environ.copy()
    if model:
        env["SLOUGHGT_MODEL_PATH"] = model

    api_proc = subprocess.Popen(
        [str(python), "-m", "uvicorn", "apps.api.server.main:app",
         "--host", "0.0.0.0", "--port", str(api_port), "--reload"],
        cwd=str(root),
        env=env,
    )

    # Wait for API
    for i in range(30):
        if _check_api_ready(api_port):
            printer.success(f"API ready → http://localhost:{api_port}")
            break
        time.sleep(1)
    else:
        printer.error("API failed to become ready")
        sys.exit(1)

    # Start Web
    printer.step(f"Starting Web server on port {web_port}...")
    web_cwd = root / "apps/web"
    web_proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=str(web_cwd),
    )

    # Wait for Web
    for i in range(60):
        if _check_port(web_port):
            printer.success(f"Web ready → http://localhost:{web_port}")
            break
        time.sleep(1)
    else:
        printer.warning("Web may not have started")

    printer.blank()
    printer.header("SloughGPT Running")
    printer.key_value("API", f"http://localhost:{api_port}")
    printer.key_value("Web", f"http://localhost:{web_port}")
    printer.key_value("Docs", f"http://localhost:{api_port}/docs")
    printer.blank()
    printer.info("Press Ctrl+C to stop")

    try:
        input()
    except (KeyboardInterrupt, EOFError):
        pass

    printer.step("Shutting down...")
    api_proc.terminate()
    web_proc.terminate()
    printer.success("Done")


def cmd_serve(args):
    """Start lightweight HTTP inference server."""
    import json
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import torch

    printer.header("Starting Inference Server")
    printer.key_value("Host", f"{args.host}:{args.port}")

    model_path = args.model or "models/sloughgpt.pt"
    model = None
    stoi = {}
    itos = {}

    if Path(model_path).exists():
        printer.step(f"Loading model from {model_path}...")
        try:
            checkpoint = torch.load(model_path, weights_only=False, map_location="cpu")
            stoi = checkpoint.get("stoi", {})
            itos = checkpoint.get("itos", {})
            printer.success("Model loaded")
        except Exception as e:
            printer.warning(f"Could not load model: {e}")
    else:
        printer.warning(f"Model not found: {model_path}")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok", "model": "sloughgpt"}).encode())
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path == "/generate":
                content_length = int(self.headers["Content-Length"])
                body = self.rfile.read(content_length)
                data = json.loads(body)

                prompt = data.get("prompt", "")
                max_tokens = data.get("max_tokens", 100)
                temperature = data.get("temperature", 0.8)

                text = f"Generated: {prompt[:50]}..."

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"text": text}).encode())
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            printer.info(f"{args[0]}")

    printer.info(f"Server ready on http://{args.host}:{args.port}")
    printer.info("Press Ctrl+C to stop")

    server = HTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
        printer.info("Server stopped")
        server.shutdown()


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
        pass


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


def register(subparsers):
    """Register dev commands with argparse subparser."""
    # Dev server
    dev_parser = subparsers.add_parser(
        "dev",
        help="Start API and Web servers (orchestrates uvicorn + npm)",
    )
    dev_parser.add_argument("--model", default=None, help="Model path (SLOUGHGT_MODEL_PATH)")
    dev_parser.add_argument("--web-port", type=int, default=3000, help="Web dev server port")
    dev_parser.add_argument("--watch-web", action="store_true", help="Watch web files for changes")
    dev_parser.set_defaults(func=cmd_dev)

    # Serve
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start lightweight HTTP inference server",
    )
    serve_parser.add_argument("--host", default="localhost", help="Bind address")
    serve_parser.add_argument("--port", type=int, default=8080, help="Listen port")
    serve_parser.add_argument("--model", metavar="PATH", help="Model to preload")
    serve_parser.set_defaults(func=cmd_serve)

    # Health
    health_parser = subparsers.add_parser(
        "health",
        help="Check API health",
    )
    health_parser.set_defaults(func=cmd_health)

    # API status
    api_status_parser = subparsers.add_parser(
        "api-status",
        help="Show detailed API status",
    )
    api_status_parser.set_defaults(func=cmd_api_status)

    # API test
    api_test_parser = subparsers.add_parser(
        "api-test",
        help="Test API endpoints",
    )
    api_test_parser.add_argument("--auth", action="store_true", help="Test authentication")
    api_test_parser.set_defaults(func=cmd_api_test)

    # API auth
    api_auth_parser = subparsers.add_parser(
        "api-auth",
        help="Test API authentication",
    )
    api_auth_parser.set_defaults(func=cmd_api_auth)

    # HF serve
    hf_serve_parser = subparsers.add_parser(
        "hf-serve",
        help="Serve a HuggingFace model via API",
    )
    hf_serve_parser.add_argument("model", help="Model name (e.g. gpt2)")
    hf_serve_parser.add_argument("--mode", choices=["api", "local"], default="local", help="Load mode")
    hf_serve_parser.add_argument("--device", default="auto", help="Device (auto, cuda, cpu, mps)")
    hf_serve_parser.set_defaults(func=cmd_hf_serve)

    # Docker
    docker_parser = subparsers.add_parser(
        "docker",
        help="Docker compose workflows (start, stop, status, logs, build, shell)",
    )
    docker_sub = docker_parser.add_subparsers(dest="docker_cmd", metavar="SUBCOMMAND")

    def _docker_compose_file():
        from ..cli import _chat_repository_root
        return _chat_repository_root() / "infra" / "docker" / "docker-compose.yml"

    def _docker_action(action: str, a):
        import subprocess
        compose = _docker_compose_file()
        if not compose.is_file():
            printer.error(f"Compose file not found: {compose}")
            return

        if action == "start":
            profile = []
            if getattr(a, "dev", False):
                profile = ["--profile", "dev"]
            elif getattr(a, "gpu", False):
                profile = ["--profile", "gpu"]
            printer.step("Starting Docker services...")
            subprocess.run(["docker", "compose", "-f", str(compose), "up", "-d", *profile])
            printer.success("Services started")
            subprocess.run(["docker", "compose", "-f", str(compose), "ps"])

        elif action == "stop":
            printer.step("Stopping Docker services...")
            subprocess.run(["docker", "compose", "-f", str(compose), "down"])
            printer.success("Services stopped")

        elif action == "status":
            subprocess.run(["docker", "compose", "-f", str(compose), "ps"])

        elif action == "logs":
            cmd = ["docker", "compose", "-f", str(compose), "logs", "-f"]
            if getattr(a, "service", None):
                cmd.append(a.service)
            subprocess.run(cmd)

        elif action == "build":
            cmd = ["docker", "compose", "-f", str(compose), "build"]
            if getattr(a, "no_cache", False):
                cmd.append("--no-cache")
            printer.step("Building Docker images...")
            subprocess.run(cmd)
            printer.success("Build complete")

        elif action == "shell":
            service = getattr(a, "service", "api")
            subprocess.run(["docker", "compose", "-f", str(compose), "exec", service, "/bin/bash"])

    docker_start = docker_sub.add_parser("start", help="Start services")
    docker_start.add_argument("--gpu", action="store_true", help="Use GPU")
    docker_start.add_argument("--dev", action="store_true", help="Development mode")
    docker_start.set_defaults(func=lambda a: _docker_action("start", a))

    docker_stop = docker_sub.add_parser("stop", help="Stop services")
    docker_stop.set_defaults(func=lambda a: _docker_action("stop", a))

    docker_status = docker_sub.add_parser("status", help="Show status")
    docker_status.set_defaults(func=lambda a: _docker_action("status", a))

    docker_logs = docker_sub.add_parser("logs", help="Show logs")
    docker_logs.add_argument("service", nargs="?", help="Service name")
    docker_logs.set_defaults(func=lambda a: _docker_action("logs", a))

    docker_build = docker_sub.add_parser("build", help="Build images")
    docker_build.add_argument("--no-cache", action="store_true", help="Build without cache")
    docker_build.set_defaults(func=lambda a: _docker_action("build", a))

    docker_shell = docker_sub.add_parser("shell", help="Shell into container")
    docker_shell.add_argument("service", default="api", nargs="?", help="Service name")
    docker_shell.set_defaults(func=lambda a: _docker_action("shell", a))
