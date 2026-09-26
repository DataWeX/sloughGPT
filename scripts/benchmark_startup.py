"""Startup benchmark — time from process spawn to ``GET /health`` answering.

Measures the two numbers that decide whether ``sloughgpt serve`` reports
``ok`` or a bogus ``error``:

* ``time_to_health_s`` — seconds until ``/health`` first returns 2xx.
  This is bounded above by the CLI's ``API_STARTUP_TIMEOUT``; if it ever
  approaches that budget the CLI kills a healthy server.
* ``time_to_ready_s`` — seconds until the log line
  ``Startup complete — server ready for requests`` (Stage READY done, model
  loaded, training restored).
* ``preload_warnings`` — count of ``Preload import failed`` lines. Must be 0:
  every warning is a cold first-time import left for the background
  model-load thread, which is the concurrent-import race the prewarm list
  exists to prevent.

Usage:
    python scripts/benchmark_startup.py                 # boot, measure, print
    python scripts/benchmark_startup.py --json-out out.json
    python scripts/benchmark_startup.py --record        # bench + persist
    python scripts/benchmark_startup.py --reuse         # measure a running API

Exit code is 1 when ``time_to_health_s`` exceeds ``API_STARTUP_TIMEOUT``, so it
can gate CI the same way the CLI does.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
# Running a file under scripts/ puts scripts/ (not the repo root) on
# sys.path, so ``domain``/``commands`` would not import. Bootstrap it.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
READY_LOG = "Startup complete — server ready for requests"
WARNING_LOG = "Preload import failed"


def _cli_budget() -> int:
    """Import ``API_STARTUP_TIMEOUT`` from the CLI (single source of truth)."""
    sys.path.insert(0, str(REPO_ROOT / "apps" / "cli" / "src"))
    from commands.dev import API_STARTUP_TIMEOUT  # noqa: PLC0415

    return int(API_STARTUP_TIMEOUT)


def _get(url: str, timeout: float = 3.0) -> bool:
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except (urllib.error.URLError, OSError):
        return False


def _wait_until(check, deadline_s: float) -> float | None:
    start = time.time()
    while time.time() - start < deadline_s:
        if check():
            return time.time() - start
        time.sleep(0.25)
    return None


def _count(text: str, needle: str) -> int:
    return sum(1 for line in text.splitlines() if needle in line)


def run_once(port: int, budget: int, log_path: Path) -> dict:
    env = {**os.environ, "FORCE_COLOR": "0"}
    with open(log_path, "wb") as fh:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "apps.api.server.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    t0 = time.time()
    t_health = None
    t_ready = None
    try:
        # One loop so both markers are timestamps from the same spawn instant.
        while True:
            now = time.time()
            if t_health is None and _get(f"http://127.0.0.1:{port}/health"):
                t_health = now - t0
            if t_ready is None and READY_LOG in log_path.read_text(errors="replace"):
                t_ready = now - t0
            if t_health is not None and t_ready is not None:
                break
            if t_health is None and now - t0 > budget:
                break
            if now - t0 > max(budget, 600):
                break
            time.sleep(0.25)
    finally:
        _stop(proc)
        text = log_path.read_text(errors="replace")

    return {
        "time_to_health_s": round(t_health, 2) if t_health is not None else None,
        "time_to_ready_s": round(t_ready, 2) if t_ready is not None else None,
        "preload_warnings": _count(text, WARNING_LOG),
        "api_starting_timeout_s": budget,
        "reached_health": t_health is not None,
    }


def measure_running(port: int, budget: int) -> dict:
    t = _wait_until(lambda: _get(f"http://127.0.0.1:{port}/health"), 5.0)
    return {
        "time_to_health_s": round(t, 2) if t is not None else None,
        "time_to_ready_s": None,
        "preload_warnings": None,
        "api_starting_timeout_s": budget,
        "reached_health": t is not None,
    }


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=25)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=10)
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--budget", type=int, default=None, help="override API_STARTUP_TIMEOUT")
    ap.add_argument("--json-out", default=None, help="write raw metrics JSON here")
    ap.add_argument("--record", action="store_true", help="persist via benchmark_results record")
    ap.add_argument("--reuse", action="store_true", help="measure an already-running API")
    args = ap.parse_args()

    budget = args.budget if args.budget is not None else _cli_budget()

    if args.reuse:
        metrics = measure_running(args.port, budget)
    else:
        log_path = Path("/tmp/opencode/startup_bench.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if _get(f"http://127.0.0.1:{args.port}/health", timeout=1.0):
            print(f"[ERR] port {args.port} already serving; use --reuse or --port", file=sys.stderr)
            return 2
        metrics = run_once(args.port, budget, log_path)

    payload = {
        "time_to_health_s": metrics["time_to_health_s"],
        "time_to_ready_s": metrics["time_to_ready_s"],
        "preload_warnings": metrics["preload_warnings"],
        "api_starting_timeout_s": metrics["api_starting_timeout_s"],
    }
    print(json.dumps(payload, indent=2))

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
        print(f"[WROTE] {args.json_out}")

    rc = 0
    if not metrics["reached_health"]:
        print(f"[FAIL] /health never answered within {budget}s", file=sys.stderr)
        rc = 1
    elif metrics["time_to_health_s"] is not None and metrics["time_to_health_s"] > budget:
        print(
            f"[FAIL] time_to_health {metrics['time_to_health_s']}s >= budget {budget}s "
            "— CLI would report error and kill a healthy server",
            file=sys.stderr,
        )
        rc = 1
    if metrics["preload_warnings"]:
        print(f"[FAIL] {metrics['preload_warnings']} preload warning(s)", file=sys.stderr)
        rc = 1

    if args.record and rc == 0:
        json_file = args.json_out or "/tmp/opencode/startup_bench.json"
        if not args.json_out:
            Path(json_file).write_text(json.dumps(payload, indent=2))
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        import benchmark_results  # noqa: PLC0415

        rc = benchmark_results.do_record(
            SimpleNamespace(
                kind="startup",
                json_file=json_file,
                url=f"http://127.0.0.1:{args.port}",
                runs=1,
                model=None,
                vs="previous",
            )
        )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
