#!/usr/bin/env python3
"""
Comprehensive Training System — Full Test Runner.

Runs all training-related tests: unit tests, integration tests, and
user journey tests (requires running servers).

Usage:
    # Run all fast unit tests (no servers needed)
    python scripts/run_training_tests.py --unit

    # Run all tests including journeys (servers required)
    python scripts/run_training_tests.py --all

    # Run only user journey tests (servers required)
    python scripts/run_training_tests.py --journeys

    # Run E2E training trigger tests (servers required)
    python scripts/run_training_tests.py --e2e

    # Run with DevTools follower output
    python scripts/run_training_tests.py --journeys --devtools

    # Run with verbose output
    python scripts/run_training_tests.py --unit -v
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
PYTEST = REPO_ROOT / ".venv" / "bin" / "python" / "-m" / "pytest"
if not (REPO_ROOT / ".venv" / "bin" / "python" / "-m" / "pytest").exists():
    PYTEST = [sys.executable, "-m", "pytest"]
else:
    PYTEST = [str(REPO_ROOT / ".venv" / "bin" / "python"), "-m", "pytest"]

UNIT_TEST_FILE = "packages/core-py/tests/test_comprehensive_trainer_unit.py"
JOURNEY_TEST_FILE = "packages/core-py/tests/test_comprehensive_training_journeys.py"
E2E_TEST_FILE = "packages/core-py/tests/test_e2e_training_trigger.py"
INTEGRATION_TEST_FILE = "packages/core-py/tests/test_computer_use_training_integration.py"


def run_pytest(args: list[str], label: str) -> tuple[bool, float]:
    """Run pytest with given args, return (success, duration)."""
    cmd = PYTEST + args
    print(f"\n{'='*60}")
    print(f"Running: {label}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    duration = time.time() - t0

    success = result.returncode == 0
    status = "PASSED" if success else "FAILED"
    print(f"\n{label}: {status} ({duration:.1f}s)")
    return success, duration


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive training tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--journeys", action="store_true", help="Run user journey tests")
    parser.add_argument("--e2e", action="store_true", help="Run E2E training trigger tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--devtools", action="store_true", help="Enable DevTools follower output")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    if not any([args.unit, args.journeys, args.e2e, args.integration, args.all]):
        args.unit = True

    results = []
    total_start = time.time()

    # Unit tests
    if args.unit or args.all:
        pytest_args = [UNIT_TEST_FILE, "-x", "-v" if args.verbose else "-q"]
        ok, dur = run_pytest(pytest_args, "Unit Tests")
        results.append({"suite": "unit", "passed": ok, "duration_s": dur})

    # Journey tests
    if args.journeys or args.all:
        pytest_args = [JOURNEY_TEST_FILE, "-v" if args.verbose else "-q"]
        if args.devtools:
            pytest_args.extend(["-s"])
        ok, dur = run_pytest(pytest_args, "User Journey Tests")
        results.append({"suite": "journeys", "passed": ok, "duration_s": dur})

    # E2E tests
    if args.e2e or args.all:
        pytest_args = [E2E_TEST_FILE, "-v" if args.verbose else "-q"]
        if args.devtools:
            pytest_args.extend(["-s"])
        ok, dur = run_pytest(pytest_args, "E2E Training Trigger Tests")
        results.append({"suite": "e2e", "passed": ok, "duration_s": dur})

    # Integration tests
    if args.integration or args.all:
        pytest_args = [INTEGRATION_TEST_FILE, "-v" if args.verbose else "-q"]
        if args.devtools:
            pytest_args.extend(["-s"])
        ok, dur = run_pytest(pytest_args, "Integration Tests")
        results.append({"suite": "integration", "passed": ok, "duration_s": dur})

    total_duration = time.time() - total_start
    all_passed = all(r["passed"] for r in results)

    if args.json:
        print(json.dumps({
            "results": results,
            "total_duration_s": round(total_duration, 1),
            "all_passed": all_passed,
        }, indent=2))
    else:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for r in results:
            status = "ok" if r["passed"] else "FAIL"
            print(f"  [{status}] {r['suite']}: {r['duration_s']:.1f}s")
        print(f"  Total: {total_duration:.1f}s")
        print(f"  Result: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
        print(f"{'='*60}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
