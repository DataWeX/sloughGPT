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
    python scripts/benchmark_startup.py --via serve_ui  # same boot through the launcher
    python scripts/benchmark_startup.py --json-out out.json
    python scripts/benchmark_startup.py --record        # bench + persist
    python scripts/benchmark_startup.py --reuse         # measure a running API

``--via`` selects the boot path:

* ``uvicorn`` (default) — a raw ``python -m uvicorn`` child: the baseline.
* ``serve_ui`` — ``scripts/serve_ui.py``, i.e. what the notebook's §Serve cell
  runs: the same API child spawned and supervised by PGQ (readiness probe,
  log in ``logs/serve_ui-api.log``, bounded TERM→KILL teardown — the app's
  own PGQ engine swallows SIGTERM, so the wrapper escalates).

The web service is not started: this measures API boot and teardown, which is
what a notebook's time-to-URL depends on.

Both paths report ``stop_latency_s`` (stop() call → ports released); it is
gated only for ``serve_ui``, because the baseline path's teardown is not part
of what the launcher owns.

Exit code is 1 when ``time_to_health_s`` exceeds ``API_STARTUP_TIMEOUT``, so it
can gate CI the same way the CLI does; also on a preload warning or a
``serve_ui`` stop slower than ``STOP_BUDGET_S``.
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
# The launcher's own teardown contract: wrapper escalates TERM→KILL at 5s,
# PGQ's terminate_grace is 10s, stop() reaps within 12s. Gate = that reap
# bound + margin, so a stop past its own contract fails; measured ~5s.
STOP_BUDGET_S = 14.0


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


def _spawn_uvicorn(port: int, log_path: Path) -> subprocess.Popen:
    """Baseline boot: a raw uvicorn child — how the API is started by hand or by the CLI."""
    env = {**os.environ, "FORCE_COLOR": "0"}
    with open(log_path, "wb") as fh:
        return subprocess.Popen(
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


def _serve_ui_module(port: int) -> tuple[object, Path]:
    """Import ``scripts/serve_ui.py`` configured for *port*, with a private log.

    ``SLO_API_PORT`` is read at import time by the launcher, so it must be set
    before the module is imported. ``LOG_DIR`` is then re-pointed at a
    benchmark-owned directory and this run's child log deleted: the launcher
    appends (``open(path, "a")``), so a log left by an earlier run still holds
    its ``Startup complete`` line and ``Preload import failed`` warnings —
    reading those would report someone else's boot as this one.
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    os.environ["SLO_API_PORT"] = str(port)
    import serve_ui  # noqa: PLC0415

    log_dir = Path(f"/tmp/opencode/startup_bench_logs/{port}")
    log_dir.mkdir(parents=True, exist_ok=True)
    api_log = log_dir / "serve_ui-api.log"
    api_log.unlink(missing_ok=True)
    serve_ui.LOG_DIR = log_dir
    return serve_ui, api_log


def _read(path: Path) -> str:
    """Log text, or ``""`` when the log does not exist yet (serve_ui creates it late)."""
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def run_once(port: int, budget: int, log_path: Path, via: str = "uvicorn") -> dict:
    t0 = time.time()
    proc: subprocess.Popen | None = None
    stop_fn = None
    boot_error = None

    if via == "serve_ui":
        mod, log_path = _serve_ui_module(port)
        # Bind stop() before start(): it is idempotent and also reaps an engine
        # left half-started by a boot that raised, so teardown never leaks.
        stop_fn = mod.stop
        try:
            mod.start(ready_timeout=float(budget), with_web=False)
        except Exception as exc:  # a failed boot is a metric here, not a traceback
            boot_error = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
    else:
        proc = _spawn_uvicorn(port, log_path)

    t_health = None
    t_ready = None
    stop_s = None
    try:
        if boot_error is None:
            # One loop so both markers are timestamps from the same spawn instant.
            while True:
                now = time.time()
                if t_health is None and _get(f"http://127.0.0.1:{port}/health"):
                    t_health = now - t0
                if t_ready is None and READY_LOG in _read(log_path):
                    t_ready = now - t0
                if t_health is not None and t_ready is not None:
                    break
                if t_health is None and now - t0 > budget:
                    break
                if now - t0 > max(budget, 600):
                    break
                time.sleep(0.25)
    finally:
        t_stop = time.time()
        if stop_fn is not None:
            stop_fn()
        elif proc is not None:
            _stop(proc)
        stop_s = round(time.time() - t_stop, 2)
        text = _read(log_path)

    return {
        "boot": via,
        "time_to_health_s": round(t_health, 2) if t_health is not None else None,
        "time_to_ready_s": round(t_ready, 2) if t_ready is not None else None,
        "stop_latency_s": stop_s,
        "preload_warnings": _count(text, WARNING_LOG),
        "api_starting_timeout_s": budget,
        "reached_health": t_health is not None,
        "boot_error": boot_error,
    }


def measure_running(port: int, budget: int) -> dict:
    t = _wait_until(lambda: _get(f"http://127.0.0.1:{port}/health"), 5.0)
    return {
        "boot": "reuse",
        "time_to_health_s": round(t, 2) if t is not None else None,
        "time_to_ready_s": None,
        "stop_latency_s": None,
        "preload_warnings": None,
        "api_starting_timeout_s": budget,
        "reached_health": t is not None,
        "boot_error": None,
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
    ap.add_argument(
        "--via",
        choices=("uvicorn", "serve_ui"),
        default="uvicorn",
        help="boot path: raw uvicorn child (baseline) or the PGQ launcher the notebook uses",
    )
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
        metrics = run_once(args.port, budget, log_path, via=args.via)

    payload = {
        "boot": metrics["boot"],
        "time_to_health_s": metrics["time_to_health_s"],
        "time_to_ready_s": metrics["time_to_ready_s"],
        "stop_latency_s": metrics["stop_latency_s"],
        "preload_warnings": metrics["preload_warnings"],
        "api_starting_timeout_s": metrics["api_starting_timeout_s"],
    }
    if metrics.get("boot_error"):
        print(f"[NOTE] boot failed via {args.via}: {metrics['boot_error']}", file=sys.stderr)
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
    stop_s = metrics["stop_latency_s"]
    if stop_s is not None and stop_s > STOP_BUDGET_S:
        if args.via == "serve_ui":
            print(
                f"[FAIL] stop_latency {stop_s}s > {STOP_BUDGET_S}s — "
                "the launcher's stop was not graceful",
                file=sys.stderr,
            )
            rc = 1
        else:
            print(
                f"[NOTE] baseline stop took {stop_s}s (> {STOP_BUDGET_S}s); "
                "reported, not gated — see serve_ui's signal re-arm",
                file=sys.stderr,
            )

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
