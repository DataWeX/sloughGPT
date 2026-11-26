"""Runner for the UX_FLOWS journeys — boots avion against a live stack.

Usage:
    .venv/bin/python -m journeys                # all journeys
    .venv/bin/python -m journeys --flow 2-write
    .venv/bin/python -m journeys --list
    .venv/bin/python -m journeys --strict-errors
    .venv/bin/python -m journeys --record       # record run + diff vs previous run

Env:
    SLO_WEB_URL         default http://localhost:5175
    SLO_API_URL         default http://localhost:8000
    SLO_JOURNEY_BROWSER firefox (default) | chromium
    SLO_JOURNEY_SHOTS   screenshot dir (default ~/.cache/slog-journeys/shots/ux)
    SLO_JOURNEY_REPORT  report path (default ~/.cache/slog-journeys/ux-flows-report.json)
    SLO_JOURNEY_RUNS    recorded-run history root (default ~/.cache/slog-journeys/runs)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

from journeys.flows import FLOWS, Flow

WEB = os.environ.get("SLO_WEB_URL", "http://localhost:5175")
API = os.environ.get("SLO_API_URL", "http://localhost:8000")
BROWSER = os.environ.get("SLO_JOURNEY_BROWSER", "firefox")
_SHOTS_DEFAULT = os.path.join(
    os.environ.get("SLO_JOURNEY_CACHE", os.path.expanduser("~/.cache/slog-journeys")),
    "shots",
    "ux",
)
SHOTS = os.environ.get("SLO_JOURNEY_SHOTS", _SHOTS_DEFAULT)
REPORT = os.environ.get(
    "SLO_JOURNEY_REPORT",
    os.path.join(
        os.environ.get("SLO_JOURNEY_CACHE", os.path.expanduser("~/.cache/slog-journeys")),
        "ux-flows-report.json",
    ),
)


def _http_ok(url: str, timeout: float = 5) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


def _shot_hook(page: Any, shots_dir: Path, counter: dict[str, int]):
    """Per-step checkpoint screenshot for Task.on_step (observational)."""
    from avion.recording.compare import shot_name

    seen: dict[str, int] = {}

    async def hook(step_name: str, status: str) -> None:
        counter["n"] += 1
        filename, _key = shot_name(counter["n"], step_name, seen)
        try:
            await page.screenshot(path=str(shots_dir / filename))
        except Exception:
            pass  # a dead page must never fail the journey

    return hook


def _finalize_run(run_dir: Path, results: list[dict[str, Any]]) -> None:
    """Write run.json, diff vs the previous run, prune history (bounded)."""
    from avion.recording.compare import (
        COMPARISON_NAME,
        RUN_META,
        compare_runs,
        git_sha,
        list_runs,
        prune_runs,
        render_summary,
        runs_root,
    )

    meta = {
        "id": run_dir.name,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "web": WEB,
        "browser": BROWSER,
        "git": git_sha(),
        "flows": [
            {"id": r["task"], "status": r["status"], "duration_s": r.get("duration_s")}
            for r in results
        ],
        "passed": sum(1 for r in results if r["status"] == "passed"),
        "failed": sum(1 for r in results if r["status"] != "passed"),
    }
    (run_dir / RUN_META).write_text(json.dumps(meta, indent=2))

    runs = [p for p in list_runs(runs_root()) if p != run_dir]
    if runs:
        previous = runs[-1]  # list_runs is oldest-first; runs[-1] is the newest other
        try:
            cmp = compare_runs(previous, run_dir)
            (run_dir / COMPARISON_NAME).write_text(json.dumps(cmp.to_dict(), indent=2))
            print("\n" + render_summary(cmp), flush=True)
            print(f"comparison: {run_dir / COMPARISON_NAME}", flush=True)
        except (OSError, ValueError) as exc:
            print(f"comparison skipped: {exc}", flush=True)
    else:
        print(f"\nrecorded run {run_dir.name} (first run — nothing to compare yet)", flush=True)

    removed = prune_runs(runs_root())
    if removed:
        print(f"pruned {len(removed)} old run(s) — history stays bounded", flush=True)


async def run(flows: list[Flow], headed: bool, strict_errors: bool, record: bool = False) -> int:
    from avion import Avion
    from avion.backends.playwright import PlaywrightBackend
    from avion.core.task import Task

    os.makedirs(SHOTS, exist_ok=True)
    backend = PlaywrightBackend(headless=not headed, browser_type=BROWSER)
    console_errors: list[str] = []
    network_errors: list[str] = []

    results: list[dict[str, Any]] = []
    run_dir: Path | None = None
    if record:
        from avion.recording.compare import new_run_id, runs_root

        run_dir = runs_root() / new_run_id()
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"recording run -> {run_dir}", flush=True)

    def persist() -> dict[str, Any]:
        """Write the report after every flow — a crash mid-run must not lose results."""
        report = {
            "started_web": WEB,
            "browser": BROWSER,
            "flows": results,
            "passed": sum(1 for r in results if r["status"] == "passed"),
            "failed": sum(1 for r in results if r["status"] != "passed"),
            "console_error_count": len(console_errors),
            "network_error_count": len(network_errors),
            "console_errors_sample": console_errors[:20],
            "network_errors_sample": network_errors[:20],
        }
        with open(REPORT, "w") as f:
            json.dump(report, f, indent=2)
        return report

    async with Avion(base_url=WEB) as a:
        await a.start(backend)
        page = backend.page
        page.on(
            "console",
            lambda m: console_errors.append(m.text[:300]) if m.type == "error" else None,
        )
        page.on("pageerror", lambda e: console_errors.append(f"pageerror: {e}"[:300]))
        page.on(
            "response",
            lambda r: (
                network_errors.append(f"{r.status} {r.url[:160]}") if r.status >= 400 else None
            ),
        )

        for flow in flows:
            t0 = time.perf_counter()
            task = Task(name=flow.id, description=flow.label, steps=flow.build())
            shot_counter = {"n": 0}
            if run_dir is not None:
                shots_dir = run_dir / flow.id / "shots"
                shots_dir.mkdir(parents=True, exist_ok=True)
                task.on_step = _shot_hook(page, shots_dir, shot_counter)
            result = await a.run_task(task)
            dur = time.perf_counter() - t0
            rec = result.to_dict()
            rec["label"] = flow.label
            rec["url"] = flow.url
            rec["spec"] = flow.spec
            rec["duration_s"] = round(dur, 2)
            if run_dir is not None:
                flow_dir = run_dir / flow.id
                flow_dir.mkdir(parents=True, exist_ok=True)
                (flow_dir / "result.json").write_text(json.dumps(rec, indent=2))
            if result.status.value != "passed":
                try:
                    shot = os.path.join(SHOTS, f"{flow.id}-FAIL.png")
                    await page.screenshot(path=shot)
                    rec["screenshot"] = shot
                except Exception:
                    pass
            results.append(rec)
            mark = "PASS" if result.status.value == "passed" else "FAIL"
            print(
                f"[{mark}] {flow.id:14s} {flow.label} ({dur:.1f}s) {result.error}".rstrip(),
                flush=True,
            )
            persist()

        try:
            print(a._reporter.report("terminal"))
        except Exception:
            pass

        await a.stop()

    report = persist()
    print(
        f"\n{report['passed']}/{len(results)} journeys passed | "
        f"console_errors={report['console_error_count']} "
        f"network_errors={report['network_error_count']}",
        flush=True,
    )
    print(f"report: {REPORT}", flush=True)
    if run_dir is not None:
        _finalize_run(run_dir, results)

    failed = report["failed"]
    if strict_errors and (console_errors or network_errors):
        print("strict-errors: console/network errors present → failing")
        failed = max(failed, 1)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the UX_FLOWS user journeys against a live stack.")
    ap.add_argument("--flow", action="append", help="run only these flow ids (repeatable)")
    ap.add_argument("--list", action="store_true", help="list flow ids and exit")
    ap.add_argument("--headed", action="store_true", help="show the browser window")
    ap.add_argument("--strict-errors", action="store_true", help="fail on console/network errors")
    ap.add_argument(
        "--record",
        action="store_true",
        help="record steps+screenshots into a run dir and diff vs the previous run",
    )
    args = ap.parse_args(argv)

    if args.list:
        for f in FLOWS:
            print(f"{f.id:14s} {f.label:32s} {f.spec}")
        return 0

    flows = FLOWS
    if args.flow:
        wanted = set(args.flow)
        flows = [f for f in FLOWS if f.id in wanted or f.id.split("-", 1)[-1] in wanted]
        if not flows:
            print(f"no flows match {sorted(wanted)}")
            return 2

    if not _http_ok(f"{API}/health"):
        print(f"API not reachable at {API}/health — start uvicorn first")
        return 2
    if not _http_ok(f"{WEB}/"):
        print(f"web not reachable at {WEB} — start vite first")
        return 2

    return asyncio.run(
        run(flows, headed=args.headed, strict_errors=args.strict_errors, record=args.record)
    )
