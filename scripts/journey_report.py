"""Per-journey traceability + test-status report (observable journey results).

Every flow in domain/journeys (web + mobile) mapped to its UX_FLOWS spec,
implementation files, asserting tests, and owner. `--run` executes each
flow's mapped suites (mobile → jest, web → vitest, repo-root → pytest) and
prints a status table; a JSON report is written for history/automation.

Usage:
    ./scripts/python scripts/journey_report.py            # link check only (fast)
    ./scripts/python scripts/journey_report.py --run      # also execute mapped tests
    ./scripts/python scripts/journey_report.py --flow m1-organize --run
    ./scripts/python scripts/journey_report.py --json out.json

Exit code: 0 when all links resolve (and, with --run, all tests pass).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from domain.journeys.flows import ALL_FLOWS, Flow  # noqa: E402

_DEFAULT_REPORT = (
    Path(os.environ.get("SLO_JOURNEY_CACHE", os.path.expanduser("~/.cache/slog-journeys")))
    / "journey-report.json"
)


def _check_links(flow: Flow, repo: Path) -> dict:
    spec_path = repo / "docs" / "UX_FLOWS.md"
    text = spec_path.read_text() if spec_path.is_file() else ""
    missing: list[str] = []
    if flow.spec not in text:
        missing.append(f"spec:{flow.spec}")
    for feat in flow.features:
        if feat not in text:
            missing.append(f"feature:{feat}")
    for path in (*flow.impl, *flow.tests):
        if not (repo / path).is_file():
            missing.append(path)
    return {"ok": not missing, "missing": missing}


def _runners(flow: Flow) -> list[tuple[str, str, list[str]]]:
    """(cwd, binary, args) commands for a flow's mapped tests (may be several)."""
    if not flow.tests:
        return []
    mobile = [t for t in flow.tests if t.startswith("apps/mobile/")]
    web = [t for t in flow.tests if t.startswith("apps/web/")]
    py = [t for t in flow.tests if t not in mobile and t not in web]
    cmds: list[tuple[str, str, list[str]]] = []
    if mobile:
        rel = [str(Path(t).relative_to("apps/mobile")) for t in mobile]
        cmds.append((str(_REPO / "apps/mobile"), "npx", ["jest", "--silent", *rel]))
    if web:
        rel = [str(Path(t).relative_to("apps/web")) for t in web]
        cmds.append((str(_REPO / "apps/web"), "npx", ["vitest", "run", *rel]))
    if py:
        cmds.append((str(_REPO), str(_REPO / "scripts/python"), ["-m", "pytest", "-q", *py]))
    return cmds


def _run_tests(flow: Flow) -> dict:
    runners = _runners(flow)
    if not runners:
        return {"status": "unmapped", "passed": None, "seconds": 0.0}
    started = time.monotonic()
    tail = ""
    for cwd, binary, args in runners:
        try:
            proc = subprocess.run(
                [binary, *args],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except subprocess.TimeoutExpired:
            return {
                "status": "fail",
                "passed": False,
                "seconds": round(time.monotonic() - started, 1),
                "tail": "timeout after 600s",
            }
        if proc.returncode:
            return {
                "status": "fail",
                "passed": False,
                "seconds": round(time.monotonic() - started, 1),
                "tail": (proc.stdout + proc.stderr)[-2000:],
            }
        tail = ""
    return {
        "status": "pass",
        "passed": True,
        "seconds": round(time.monotonic() - started, 1),
        "tail": tail,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow", action="append", help="only these flow ids (repeatable)")
    parser.add_argument("--run", action="store_true", help="execute mapped test suites")
    parser.add_argument("--json", default=str(_DEFAULT_REPORT), help="report output path")
    args = parser.parse_args()

    flows = [f for f in ALL_FLOWS if not args.flow or f.id in args.flow]
    if not flows:
        print(f"no flows matched: {args.flow}", file=sys.stderr)
        return 2

    results = []
    failures = 0
    for flow in flows:
        entry = {
            "id": flow.id,
            "label": flow.label,
            "surface": flow.surface,
            "owner": flow.owner,
            "spec": flow.spec,
            "features": list(flow.features),
            "impl": list(flow.impl),
            "tests": list(flow.tests),
        }
        entry["links"] = _check_links(flow, _REPO)
        if args.run:
            entry["tests"] = _run_tests(flow)  # type: ignore[assignment]
        bad_links = not entry["links"]["ok"]
        bad_tests = args.run and entry["tests"]["status"] == "fail"  # type: ignore[index]
        if bad_links or bad_tests:
            failures += 1
        results.append(entry)

    id_w = max(len(f.id) for f in flows) + 2
    print(f"{'ID':<{id_w}}{'SURFACE':<9}{'LINKS':<8}{'OWNER':<14}{'TESTS'}")
    print("-" * (id_w + 40))
    for entry in results:
        links = "ok" if entry["links"]["ok"] else "MISSING"
        if args.run:
            t = entry["tests"]  # type: ignore[assignment]
            tests = {
                "pass": f"PASS ({t['seconds']}s)",
                "fail": "FAIL",
                "unmapped": "—",
            }[t["status"]]
        else:
            tests = f"{len(entry['tests'])} mapped"  # type: ignore[arg-type]
        print(f"{entry['id']:<{id_w}}{entry['surface']:<9}{links:<8}{entry['owner']:<14}{tests}")
        if entry["links"]["missing"]:
            for m in entry["links"]["missing"]:
                print(f"    missing: {m}")
        if args.run and entry["tests"]["status"] == "fail":  # type: ignore[index]
            tail = entry["tests"].get("tail", "")  # type: ignore[union-attr]
            for line in tail.splitlines()[-6:]:
                print(f"    {line}")

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "ran": args.run,
        "failures": failures,
        "flows": results,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n{len(results)} journeys, {failures} failing — report: {out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
