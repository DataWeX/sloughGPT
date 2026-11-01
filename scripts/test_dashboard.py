#!/usr/bin/env python3
"""test_dashboard.py — Generate an HTML dashboard from pytest results.

Usage:
    ./scripts/python scripts/test_dashboard.py                    # run tests + report
    ./scripts/python scripts/test_dashboard.py --no-run            # report from last run
    ./scripts/python scripts/test_dashboard.py tests/test_*.py     # specific tests

Output: test_dashboard.html (open in browser)
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "test_dashboard.html"


def run_pytest(patterns: list[str]) -> dict:
    """Run pytest with JSON report, return parsed results."""
    cmd = [
        sys.executable, "-m", "pytest",
        "--tb=no", "-q", "--no-header",
        "-o", "junit_family=xunit2",
        "--junitxml=/tmp/pytest_dashboard.xml",
    ]
    if patterns:
        cmd.extend(patterns)

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    elapsed = time.time() - t0

    # Parse stdout summary line: "56 passed in 3.57s" or "1 failed, 55 passed..."
    output = proc.stdout + proc.stderr
    passed = failed = errors = skipped = 0
    total_time = elapsed

    m = re.search(r"(\d+) passed", output)
    if m:
        passed = int(m.group(1))
    m = re.search(r"(\d+) failed", output)
    if m:
        failed = int(m.group(1))
    m = re.search(r"(\d+) error", output)
    if m:
        errors = int(m.group(1))
    m = re.search(r"(\d+) skipped", output)
    if m:
        skipped = int(m.group(1))
    m = re.search(r"in ([\d.]+)s", output)
    if m:
        total_time = float(m.group(1))

    # Parse JUnit XML for per-test detail
    tests = parse_junit_xml("/tmp/pytest_dashboard.xml")

    return {
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "total": passed + failed + errors + skipped,
        "time": total_time,
        "tests": tests,
        "raw_output": output[-3000:],
    }


def parse_junit_xml(path: str) -> list[dict]:
    """Parse JUnit XML into per-test dicts."""
    import xml.etree.ElementTree as ET

    tests = []
    try:
        tree = ET.parse(path)
        for testcase in tree.iter("testcase"):
            classname = testcase.get("classname", "")
            name = testcase.get("name", "")
            time_s = float(testcase.get("time", "0"))

            status = "passed"
            message = ""
            child = testcase.find("failure")
            if child is None:
                child = testcase.find("error")
            if child is not None:
                status = "failed"
                message = (child.get("message") or "")[:200]
            elif testcase.find("skipped") is not None:
                status = "skipped"

            # Extract file from classname (e.g. "tests.test_cognition_engine.TestX")
            file_match = re.split(r"\.Test", classname)
            test_file = file_match[0].replace(".", "/") + ".py" if file_match else classname

            tests.append({
                "file": test_file,
                "class": classname,
                "name": name,
                "status": status,
                "time": time_s,
                "message": message,
            })
    except Exception as e:
        print(f"Warning: could not parse JUnit XML: {e}", file=sys.stderr)

    return tests


def build_matrix(tests: list[dict]) -> dict:
    """Build file × status matrix."""
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "failed": 0, "skipped": 0})
    for t in tests:
        matrix[t["file"]][t["status"]] += 1
    return dict(matrix)


def sparkline(values: list[float], width: int = 40) -> str:
    """ASCII sparkline."""
    if not values:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    rng = hi - lo if hi > lo else 1
    # Downsample to width
    if len(values) > width:
        step = len(values) / width
        values = [values[int(i * step)] for i in range(width)]
    return "".join(blocks[int((v - lo) / rng * (len(blocks) - 1))] for v in values)


def bar(pct: float, width: int = 20) -> str:
    """ASCII bar."""
    filled = int(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


def generate_html(results: dict) -> str:
    """Generate HTML dashboard."""
    tests = results["tests"]
    matrix = build_matrix(tests)
    total = results["total"] or 1
    pass_rate = results["passed"] / total * 100 if total else 0

    # Per-file stats
    file_stats = []
    for f, counts in sorted(matrix.items()):
        fp = counts["passed"]
        ff = counts["failed"]
        fs = counts["skipped"]
        ft = fp + ff + fs
        rate = fp / ft * 100 if ft else 0
        file_stats.append({"file": f, "passed": fp, "failed": ff, "skipped": fs, "total": ft, "rate": rate})

    # Timing distribution
    times = [t["time"] for t in tests if t["time"] > 0]
    time_spark = sparkline(times) if times else ""

    # Build rows
    matrix_rows = ""
    for fs in file_stats:
        status_cls = "ok" if fs["rate"] >= 99 else ("warn" if fs["rate"] >= 80 else "bad")
        matrix_rows += f"""
        <tr class="{status_cls}">
            <td class="file">{fs['file']}</td>
            <td class="num">{fs['passed']}</td>
            <td class="num">{fs['failed']}</td>
            <td class="num">{fs['skipped']}</td>
            <td class="num">{fs['total']}</td>
            <td class="bar-cell">
                <div class="bar-track"><div class="bar-fill" style="width:{fs['rate']:.0f}%"></div></div>
                <span class="pct">{fs['rate']:.0f}%</span>
            </td>
        </tr>"""

    # Failed tests detail
    failed_rows = ""
    for t in tests:
        if t["status"] == "failed":
            failed_rows += f"""
            <tr>
                <td class="file">{t['file']}</td>
                <td>{t['class']}</td>
                <td>{t['name']}</td>
                <td class="msg">{t['message']}</td>
            </tr>"""

    # Slowest tests
    slowest = sorted([t for t in tests if t["time"] > 0], key=lambda t: t["time"], reverse=True)[:10]
    slow_rows = ""
    max_t = slowest[0]["time"] if slowest else 1
    for t in slowest:
        pct = t["time"] / max_t * 100
        slow_rows += f"""
        <tr>
            <td class="file">{t['file']}</td>
            <td>{t['name']}</td>
            <td class="num">{t['time']:.3f}s</td>
            <td class="bar-cell">
                <div class="bar-track"><div class="bar-fill slow" style="width:{pct:.0f}%"></div></div>
            </td>
        </tr>"""

    now = time.strftime("%Y-%m-%d %H:%M:%S")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>sloughGPT Test Dashboard</title>
<style>
  :root {{
    --bg: #0d1117; --card: #161b22; --border: #30363d;
    --text: #c9d1d9; --dim: #8b949e; --green: #3fb950;
    --red: #f85149; --yellow: #d29922; --blue: #58a6ff;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:var(--bg); color:var(--text); font:14px/1.5 'SF Mono',Consolas,monospace; padding:24px; }}
  h1 {{ font-size:20px; margin-bottom:4px; color:var(--blue); }}
  .subtitle {{ color:var(--dim); font-size:12px; margin-bottom:24px; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; margin-bottom:24px; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:8px; padding:16px; }}
  .card .label {{ color:var(--dim); font-size:11px; text-transform:uppercase; letter-spacing:1px; }}
  .card .value {{ font-size:28px; font-weight:bold; margin-top:4px; }}
  .card .value.ok {{ color:var(--green); }}
  .card .value.bad {{ color:var(--red); }}
  .card .value.warn {{ color:var(--yellow); }}
  .card .sub {{ color:var(--dim); font-size:11px; margin-top:2px; }}
  .section {{ background:var(--card); border:1px solid var(--border); border-radius:8px; padding:16px; margin-bottom:16px; }}
  .section h2 {{ font-size:14px; color:var(--blue); margin-bottom:12px; text-transform:uppercase; letter-spacing:1px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th {{ text-align:left; color:var(--dim); font-size:11px; text-transform:uppercase; letter-spacing:1px; padding:6px 8px; border-bottom:1px solid var(--border); }}
  td {{ padding:6px 8px; border-bottom:1px solid #21262d; }}
  td.file {{ color:var(--blue); }}
  td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  td.msg {{ color:var(--red); font-size:12px; max-width:400px; overflow:hidden; text-overflow:ellipsis; }}
  .bar-cell {{ display:flex; align-items:center; gap:8px; }}
  .bar-track {{ flex:1; height:8px; background:#21262d; border-radius:4px; overflow:hidden; }}
  .bar-fill {{ height:100%; background:var(--green); border-radius:4px; }}
  .bar-fill.slow {{ background:var(--yellow); }}
  .pct {{ min-width:36px; text-align:right; font-size:12px; color:var(--dim); }}
  tr.ok td:first-child {{ border-left:3px solid var(--green); }}
  tr.warn td:first-child {{ border-left:3px solid var(--yellow); }}
  tr.bad td:first-child {{ border-left:3px solid var(--red); }}
  .sparkline {{ font-size:18px; color:var(--blue); letter-spacing:-1px; }}
  .empty {{ color:var(--dim); font-style:italic; padding:12px 0; }}
</style>
</head>
<body>
<h1>sloughGPT Test Dashboard</h1>
<div class="subtitle">Generated {now} &middot; {results['time']:.1f}s</div>

<div class="cards">
  <div class="card">
    <div class="label">Total Tests</div>
    <div class="value">{results['total']}</div>
    <div class="sub">across {len(file_stats)} files</div>
  </div>
  <div class="card">
    <div class="label">Pass Rate</div>
    <div class="value {'ok' if pass_rate >= 99 else 'warn' if pass_rate >= 80 else 'bad'}">{pass_rate:.1f}%</div>
    <div class="sub">{results['passed']} passed</div>
  </div>
  <div class="card">
    <div class="label">Failed</div>
    <div class="value {'ok' if results['failed'] == 0 else 'bad'}">{results['failed']}</div>
    <div class="sub">{results['errors']} errors</div>
  </div>
  <div class="card">
    <div class="label">Skipped</div>
    <div class="value">{results['skipped']}</div>
    <div class="sub">&nbsp;</div>
  </div>
  <div class="card">
    <div class="label">Duration</div>
    <div class="value">{results['time']:.1f}s</div>
    <div class="sub">{time_spark or '&nbsp;'}</div>
  </div>
</div>

<div class="section">
  <h2>Test Matrix</h2>
  <table>
    <thead><tr><th>File</th><th>Pass</th><th>Fail</th><th>Skip</th><th>Total</th><th>Confidence</th></tr></thead>
    <tbody>{matrix_rows or '<tr><td colspan="6" class="empty">No tests found</td></tr>'}</tbody>
  </table>
</div>

{"<div class='section'><h2>Failures</h2><table><thead><tr><th>File</th><th>Class</th><th>Test</th><th>Message</th></tr></thead><tbody>" + failed_rows + "</tbody></table></div>" if failed_rows else ""}

<div class="section">
  <h2>Slowest Tests</h2>
  <table>
    <thead><tr><th>File</th><th>Test</th><th>Time</th><th></th></tr></thead>
    <tbody>{slow_rows or '<tr><td colspan="4" class="empty">No timing data</td></tr>'}</tbody>
  </table>
</div>

</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate test dashboard HTML")
    parser.add_argument("patterns", nargs="*", help="pytest patterns (default: all)")
    parser.add_argument("--no-run", action="store_true", help="Don't run tests, show placeholder")
    parser.add_argument("-o", "--output", default=str(OUTPUT), help="Output HTML path")
    args = parser.parse_args()

    if args.no_run:
        results = {
            "passed": 0, "failed": 0, "errors": 0, "skipped": 0,
            "total": 0, "time": 0, "tests": [],
            "raw_output": "--no-run specified",
        }
    else:
        print(f"Running pytest {' '.join(args.patterns) or '(all)'}...", file=sys.stderr)
        results = run_pytest(args.patterns)

    html = generate_html(results)
    out = Path(args.output)
    out.write_text(html, encoding="utf-8")
    print(f"Dashboard: {out}")
    print(f"  {results['passed']} passed, {results['failed']} failed, {results['total']} total in {results['time']:.1f}s")


if __name__ == "__main__":
    main()
