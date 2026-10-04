"""Runner for the UX_FLOWS journeys — boots avion against a live stack.

Usage:
    .venv/bin/python -m domain.journeys                # all journeys
    .venv/bin/python -m domain.journeys --flow 2-write
    .venv/bin/python -m domain.journeys --list
    .venv/bin/python -m domain.journeys --strict-errors
    .venv/bin/python -m domain.journeys --dry-run          # step plans, no browser
    .venv/bin/python -m domain.journeys --json             # machine-readable report

Env:
    SLO_WEB_URL         default http://localhost:3000 (matches scripts/dev-stack.sh)
    SLO_API_URL         default http://localhost:8000
    SLO_JOURNEY_BROWSER firefox (default) | chromium
    SLO_JOURNEY_SHOTS   screenshot dir (default ~/.cache/slog-journeys/shots/ux)
    SLO_JOURNEY_REPORT  report path (default ~/.cache/slog-journeys/ux-flows-report.json)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import urllib.request
from typing import Any

from domain.journeys.flows import FLOWS, Flow

WEB = os.environ.get("SLO_WEB_URL", "http://localhost:3000")
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


def describe_steps(flow: Flow) -> list[str]:
    """Step names for a flow, in order (optional steps marked)."""
    return [
        f"{s.name}{' (optional)' if getattr(s, 'optional', False) else ''}" for s in flow.build()
    ]


def flow_is_api_gated(flow: Flow) -> bool:
    """True if the flow sends a prompt (needs the LLM API to reply)."""
    return any(s.name.startswith("send") for s in flow.build())


def render_summary(report: dict[str, Any], report_path: str = "") -> str:
    """Plain-ASCII terminal summary ('gauge') for a journey run."""
    flows = report.get("flows", [])
    passed = report.get("passed", 0)
    total = len(flows)
    width = 56
    rows = ["=" * width, "  JOURNEY RUN", "=" * width]
    for rec in flows:
        fid = str(rec.get("task", "?"))
        dur = rec.get("duration_s", 0)
        ok = rec.get("status") == "passed"
        mark = "PASS" if ok else "FAIL"
        suffix = "" if ok else f"   {str(rec.get('error', ''))[:40]}"
        rows.append(f"  [{mark}] {fid:<14} {dur:>7}s{suffix}")
    rows.append("-" * width)
    rows.append(
        f"  {passed}/{total} passed | "
        f"console={report.get('console_error_count', 0)} "
        f"network={report.get('network_error_count', 0)}"
    )
    if report_path:
        rows.append(f"  report: {report_path}")
    return "\n".join(rows)


def report_to_json(report: dict[str, Any]) -> str:
    """Stable machine-readable report (sorted keys) for CI piping."""
    return json.dumps(report, indent=2, sort_keys=True)


async def run(flows: list[Flow], headed: bool, strict_errors: bool, json_output: bool = False) -> int:
    from avion import Avion
    from avion.backends.playwright import PlaywrightBackend
    from avion.core.task import Task

    os.makedirs(SHOTS, exist_ok=True)
    backend = PlaywrightBackend(headless=not headed, browser_type=BROWSER)
    console_errors: list[str] = []
    network_errors: list[str] = []

    results: list[dict[str, Any]] = []

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

    a = Avion(base_url=WEB)
    await a.start(backend)
    try:
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
            result = await a.run_task(task)
            dur = time.perf_counter() - t0
            rec = result.to_dict()
            rec["label"] = flow.label
            rec["url"] = flow.url
            rec["spec"] = flow.spec
            rec["duration_s"] = round(dur, 2)
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

    finally:
        await a.stop()

    report = persist()
    if json_output:
        print(report_to_json(report))
    else:
        print(render_summary(report, REPORT), flush=True)

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
        "--web-only",
        action="store_true",
        help="skip the API health gate (run web-only journeys when the LLM API is down)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print each journey's step plan without launching a browser",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="emit the report as JSON to stdout instead of the ASCII gauge",
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

    if args.dry_run:
        api_gated = 0
        for f in flows:
            gated = flow_is_api_gated(f)
            api_gated += 1 if gated else 0
            tag = "[api] " if gated else "[web] "
            print(f"{tag}[{f.id}] {f.label}")
            print(f"   url:  {f.url}")
            print(f"   spec: {f.spec}")
            for i, name in enumerate(describe_steps(f), 1):
                print(f"   {i:2d}. {name}")
            print()
        print(
            f"dry-run: {len(flows)} journeys planned "
            f"({api_gated} api-gated, {len(flows) - api_gated} web-only), no browser launched"
        )
        return 0

    if not args.web_only:
        if not _http_ok(f"{API}/health"):
            print(f"API not reachable at {API}/health — start uvicorn first (or pass --web-only)")
            return 2
    if not _http_ok(f"{WEB}/"):
        print(f"web not reachable at {WEB} — start the web dev server first")
        return 2

    return asyncio.run(
        run(flows, headed=args.headed, strict_errors=args.strict_errors, json_output=args.json)
    )
