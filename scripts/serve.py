#!/usr/bin/env python3
"""Unified one-script launcher: Python API + Rust gateway + web UI on one origin.

Architecture (this repo):
  - Backend / core infra: Python FastAPI sidecar (apps/api/server, :8000 default)
  - Core gateway: Rust Axum edge (apps/gateway) — serves static SPA and
    byte-relays API to the sidecar on ONE origin (:8080 default)
  - Frontend: Vite/React app (apps/web). Production build: dist-vite/ served by
    gateway (same-origin, NEXT_PUBLIC_API_URL=). Dev build served by Vite (:5173)
    can proxy API paths, but the unified single-origin path is gateway.

Modes:
  unified (default): API sidecar + Rust gateway serving dist-vite on MAN_GATEWAY_PORT.
    -> one URL/IP for UI + API. Requires gateway binary and web build (or --build).
  dev:              API sidecar + Vite dev server (:5173) with proxy to API.
    -> one dev origin on :5173 (matches existing serve_ui behavior).

This script degrades honestly when tools/build artifacts are missing and prints
actionable steps. It does not require the PGQ engine package at import time for
basic process supervision; if unavailable it falls back to subprocess management.

Usage:
  python3 scripts/serve.py                  # unified if ready, else explains
  python3 scripts/serve.py --mode dev       # API + Vite dev (:5173 proxy)
  python3 scripts/serve.py --mode unified   # API + gateway (:8080)
  python3 scripts/serve.py --build-web      # build web (dist-vite) if node/npm present
  python3 scripts/serve.py --build-gateway  # cargo build --release -p slough-gateway
  python3 scripts/serve.py --api-only
  python3 scripts/serve.py --ready-timeout 120
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "apps" / "web"
GATEWAY_DIR = ROOT / "apps" / "gateway"
LOG_DIR = ROOT / "logs"

API_PORT = int(os.environ.get("SLO_API_PORT", "8000"))
WEB_PORT = int(os.environ.get("WEB_PORT", "5173"))
GATEWAY_PORT = int(os.environ.get("MAN_GATEWAY_PORT", "8080"))
READY_TIMEOUT = 90.0


@dataclass
class ProcSpec:
    name: str
    cmd: list[str]
    cwd: Path
    log: Path
    port: int | None = None
    env: dict[str, str] | None = None


class Stack:
    def __init__(self) -> None:
        self.procs: dict[str, subprocess.Popen] = {}
        self.specs: dict[str, ProcSpec] = {}

    def start(self, spec: ProcSpec) -> subprocess.Popen:
        spec.log.parent.mkdir(parents=True, exist_ok=True)
        logf = open(spec.log, "a", buffering=1)
        env = os.environ.copy()
        if spec.env:
            env.update(spec.env)
        # start new session so we can signal the whole group
        p = subprocess.Popen(
            spec.cmd,
            cwd=str(spec.cwd),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.procs[spec.name] = p
        self.specs[spec.name] = spec
        return p

    def stop(self) -> None:
        for name, p in list(self.procs.items()):
            if p.poll() is None:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                except Exception:
                    try:
                        p.terminate()
                    except Exception:
                        pass
        deadline = time.monotonic() + 12.0
        for p in self.procs.values():
            try:
                remaining = max(0.1, deadline - time.monotonic())
                p.wait(timeout=remaining)
            except Exception:
                pass
        for name, p in list(self.procs.items()):
            if p.poll() is None:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except Exception:
                    try:
                        p.kill()
                    except Exception:
                        pass
                try:
                    p.wait(timeout=2)
                except Exception:
                    pass

    def alive(self, name: str) -> bool:
        p = self.procs.get(name)
        return p is not None and p.poll() is None


def tail(path: Path, lines: int = 20) -> str:
    try:
        return "".join(path.read_text(encoding="utf-8", errors="replace").splitlines(True)[-lines:])
    except Exception:
        return ""


def port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            return s.connect_ex((host, port)) == 0
        except Exception:
            return False


def wait_port(port: int, proc: subprocess.Popen | None, timeout: float, name: str) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        if port_open(port):
            return True
        time.sleep(0.25)
    return False


def resolve_python() -> str | None:
    # Prefer project resolver
    resolver = ROOT / "scripts" / "python"
    if resolver.exists() and os.access(resolver, os.X_OK):
        try:
            r = subprocess.run([str(resolver), "--resolve"], cwd=str(ROOT), capture_output=True, text=True, timeout=10)
            p = (r.stdout or "").strip().splitlines()
            if p and p[-1] and Path(p[-1]).exists():
                return p[-1]
        except Exception:
            pass
    # common fallbacks
    for cand in [os.environ.get("SLO_PYTHON"), shutil.which("python3"), shutil.which("python")]:
        if cand and Path(cand).exists() if "/" in cand else cand:
            return cand
    # conda env locations
    for cand in [
        Path.home() / "miniconda3/envs/sloughgpt/bin/python",
        Path.home() / "anaconda3/envs/sloughgpt/bin/python",
        Path("/opt/conda/envs/sloughgpt/bin/python"),
        ROOT / ".venv/bin/python",
    ]:
        if cand.exists():
            return str(cand)
    return None


def gateway_binary() -> Path | None:
    for p in [
        GATEWAY_DIR / "target/release/slough-gateway",
        GATEWAY_DIR / "target/debug/slough-gateway",
        ROOT / "target/release/slough-gateway",
    ]:
        if p.is_file() and os.access(p, os.X_OK):
            return p
    found = shutil.which("slough-gateway")
    return Path(found) if found else None


def web_built() -> bool:
    return (WEB_DIR / "dist-vite" / "index.html").is_file()


def has_vite() -> bool:
    return shutil.which("node") is not None and (WEB_DIR / "node_modules" / ".bin" / "vite").is_file()


def run_cmd(cmd: list[str], cwd: Path) -> int:
    print(f"+ {' '.join(cmd)} (cwd={cwd})")
    p = subprocess.run(cmd, cwd=str(cwd))
    return p.returncode


def build_web() -> bool:
    if not shutil.which("node") or not shutil.which("npm"):
        print("build-web: node/npm not found")
        return False
    if not (ROOT / "node_modules").exists():
        print("build-web: root node_modules missing (run npm ci/install at repo root)")
    env = os.environ.copy()
    env["NEXT_PUBLIC_API_URL"] = ""  # same-origin build for gateway
    cmd = ["npm", "run", "build:vite"]
    print("+ NEXT_PUBLIC_API_URL= npm run build:vite (apps/web)")
    rc = subprocess.run(cmd, cwd=str(WEB_DIR), env=env).returncode
    return rc == 0 and web_built()


def build_gateway() -> bool:
    if not shutil.which("cargo"):
        print("build-gateway: cargo not found")
        return False
    rc = run_cmd(["cargo", "build", "--release", "--locked", "-p", "slough-gateway"], GATEWAY_DIR)
    return rc == 0 and gateway_binary() is not None


def print_banner(info: dict) -> None:
    print("\n── SloughGPT unified stack " + "─" * 34)
    for k, v in info.items():
        print(f"  {k}: {v}")
    print("  logs: logs/serve-*.log")
    print("  stop: Ctrl+C")
    print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Unified launcher: Python API + Rust gateway + web UI")
    ap.add_argument("--mode", choices=["auto", "unified", "dev"], default="auto",
                    help="unified=gateway single-origin, dev=vite+proxy, auto=pick best available")
    ap.add_argument("--api-only", action="store_true", help="start only Python API sidecar")
    ap.add_argument("--build-web", action="store_true", help="build web dist-vite (same-origin)")
    ap.add_argument("--build-gateway", action="store_true", help="build rust gateway release binary")
    ap.add_argument("--ready-timeout", type=float, default=READY_TIMEOUT)
    ap.add_argument("--gateway-port", type=int, default=GATEWAY_PORT)
    ap.add_argument("--api-port", type=int, default=API_PORT)
    ap.add_argument("--web-port", type=int, default=WEB_PORT)
    ap.add_argument("--no-web", action="store_true", help="(dev mode) do not start vite")
    args = ap.parse_args(argv)

    if args.build_web:
        ok = build_web()
        print("build-web:", "ok" if ok else "failed")
        if not ok:
            return 1
    if args.build_gateway:
        ok = build_gateway()
        print("build-gateway:", "ok" if ok else "failed")
        if not ok:
            return 1

    py = resolve_python()
    gw = gateway_binary()
    built = web_built()
    vite_ok = has_vite()

    mode = args.mode
    if mode == "auto":
        if args.api_only:
            mode = "api"
        elif gw and built:
            mode = "unified"
        elif vite_ok:
            mode = "dev"
        elif py:
            mode = "api"
        else:
            mode = "none"

    if mode == "unified" and (not gw or not built):
        # explain and degrade
        missing = []
        if not gw:
            missing.append("gateway binary (build: cargo build --release --locked -p slough-gateway in apps/gateway, or pass --build-gateway)")
        if not built:
            missing.append("web build apps/web/dist-vite (build: cd apps/web && npm run build:vite, or pass --build-web)")
        print("unified mode unavailable: " + "; ".join(missing))
        if vite_ok:
            print("falling back to dev mode (vite+proxy on :%d)" % args.web_port)
            mode = "dev"
        elif py:
            print("falling back to api-only")
            mode = "api"
        else:
            mode = "none"

    if mode == "none":
        print("No runnable components found.")
        print(f"  python: {py or 'MISSING (need conda env sloughgpt / .venv / SLO_PYTHON)'}")
        print(f"  gateway: {gw or 'MISSING (need Rust toolchain + cargo build)'}")
        print(f"  web build: {'ok' if built else 'MISSING (need node/npm + npm run build:vite)'}")
        print(f"  vite dev: {'ok' if vite_ok else 'MISSING (need node_modules in apps/web)'}")
        print("\nSingle-origin unified target: Python API (:8000) + Rust gateway (:%d) serving dist-vite + proxying API." % args.gateway_port)
        return 1

    if mode in ("unified", "dev", "api") and not py:
        print("Python interpreter not found; cannot start API/core infra.")
        print("Set SLO_PYTHON or install conda env 'sloughgpt' / project venv. See README.md.")
        return 1

    stack = Stack()
    stopping = {"v": False}

    def _stop(*_a):
        if stopping["v"]:
            return
        stopping["v"] = True
        print("\nStopping...")
        stack.stop()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        # 1) Python API/core
        api_spec = ProcSpec(
            name="api",
            cmd=[py, "-m", "apps.api.server.main"],
            cwd=ROOT,
            log=LOG_DIR / "serve-api.log",
            port=args.api_port,
            env={"SLO_API_PORT": str(args.api_port)},
        )
        print(f"Starting Python API/core on :{args.api_port} ...")
        stack.start(api_spec)
        if not wait_port(args.api_port, stack.procs["api"], args.ready_timeout, "api"):
            print(f"API failed to become ready on :{args.api_port}")
            print(tail(api_spec.log))
            _stop()
            return 1
        print(f"  API ready: http://127.0.0.1:{args.api_port}")

        if args.api_only or mode == "api":
            info = {"api": f"http://127.0.0.1:{args.api_port}", "mode": "api-only"}
            print_banner(info)
            while not stopping["v"]:
                if not stack.alive("api"):
                    print("API exited unexpectedly")
                    print(tail(api_spec.log))
                    _stop()
                    return 1
                time.sleep(0.5)
            return 0

        # 2) Unified gateway (single origin)
        if mode == "unified":
            gw_env = {
                "MAN_GATEWAY_PORT": str(args.gateway_port),
                "MAN_CORE_URL": f"http://127.0.0.1:{args.api_port}",
                "MAN_STATIC_DIR": str(WEB_DIR / "dist-vite"),
            }
            gw_spec = ProcSpec(
                name="gateway",
                cmd=[str(gw)],
                cwd=ROOT,
                log=LOG_DIR / "serve-gateway.log",
                port=args.gateway_port,
                env=gw_env,
            )
            print(f"Starting Rust gateway on :{args.gateway_port} (static={gw_env['MAN_STATIC_DIR']}, core={gw_env['MAN_CORE_URL']}) ...")
            stack.start(gw_spec)
            if not wait_port(args.gateway_port, stack.procs["gateway"], args.ready_timeout, "gateway"):
                print(f"Gateway failed to become ready on :{args.gateway_port}")
                print(tail(gw_spec.log))
                _stop()
                return 1
            url = f"http://127.0.0.1:{args.gateway_port}"
            info = {
                "url (single origin, UI+API)": url,
                "api (sidecar)": f"http://127.0.0.1:{args.api_port}",
                "gateway": str(gw),
                "static": str(WEB_DIR / "dist-vite"),
                "mode": "unified",
            }
            # Colab hint
            try:
                from google.colab import output as colab_output  # type: ignore
                colab_output.serve_kernel_port_as_window(args.gateway_port)
                info["colab"] = f"port {args.gateway_port} forwarded"
            except Exception:
                pass
            print_banner(info)
            while not stopping["v"]:
                for name in ("api", "gateway"):
                    if not stack.alive(name):
                        print(f"{name} exited unexpectedly")
                        print(tail(stack.specs[name].log))
                        _stop()
                        return 1
                time.sleep(0.5)
            return 0

        # 3) Dev mode: Vite dev server proxies API paths (one dev origin)
        if mode == "dev" and not args.no_web:
            if not vite_ok:
                print("Vite dev server unavailable (need node + apps/web/node_modules).")
                print(f"API is running at http://127.0.0.1:{args.api_port}")
                while not stopping["v"] and stack.alive("api"):
                    time.sleep(0.5)
                _stop()
                return 0
            vite_bin = str(WEB_DIR / "node_modules" / ".bin" / "vite")
            web_spec = ProcSpec(
                name="web",
                cmd=[vite_bin, "--host", "0.0.0.0", "--port", str(args.web_port)],
                cwd=WEB_DIR,
                log=LOG_DIR / "serve-web.log",
                port=args.web_port,
                env={"SLO_API_URL": f"http://127.0.0.1:{args.api_port}"},
            )
            print(f"Starting Vite dev server on :{args.web_port} (proxy -> API :{args.api_port}) ...")
            stack.start(web_spec)
            if not wait_port(args.web_port, stack.procs["web"], args.ready_timeout, "web"):
                print(f"Web failed to become ready on :{args.web_port}")
                print(tail(web_spec.log))
                # keep API running
            else:
                print(f"  Web ready: http://0.0.0.0:{args.web_port}")
            info = {
                "url (dev, UI+proxied API)": f"http://localhost:{args.web_port}",
                "api (sidecar)": f"http://127.0.0.1:{args.api_port}",
                "mode": "dev",
                "note": "Vite proxy forwards backend paths to API (apps/web/vite.config.ts). For true gateway single-origin use --mode unified with built gateway+dist-vite.",
            }
            print_banner(info)
            while not stopping["v"]:
                if not stack.alive("api"):
                    print("API exited unexpectedly")
                    print(tail(api_spec.log))
                    _stop()
                    return 1
                if "web" in stack.procs and not stack.alive("web") and not stopping["v"]:
                    print("Web exited unexpectedly")
                    print(tail(web_spec.log))
                    _stop()
                    return 1
                time.sleep(0.5)
            return 0

        # fallback: api running
        print_banner({"api": f"http://127.0.0.1:{args.api_port}", "mode": "api"})
        while not stopping["v"] and stack.alive("api"):
            time.sleep(0.5)
        _stop()
        return 0

    except Exception as e:
        print(f"serve error: {e}", file=sys.stderr)
        _stop()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
