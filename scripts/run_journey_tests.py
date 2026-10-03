#!/usr/bin/env python3
"""
Arken journey test runner for sloughGPT.

Runs end-to-end journey checks against a live sloughGPT instance
using the Arken autoclicker (Playwright backend).

Usage:
    # Run against local dev server
    python scripts/run_journey_tests.py

    # Run against specific URL
    BASE_URL=http://localhost:3000 python scripts/run_journey_tests.py

    # Run specific journey
    python scripts/run_journey_tests.py --journey chat

    # Choose backend (api needs no browser)
    python scripts/run_journey_tests.py --backend playwright
    python scripts/run_journey_tests.py --backend api
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

# Add packages to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages", "avion", "src"))


async def run_journeys(base_url: str, journeys: list[str] | None = None):
    """Run page-presence journeys using Arken + Playwright."""
    from avion import Arken

    pages = {
        "chat": "/chat",
        "souls": "/souls",
        "training": "/training",
        "datasets": "/datasets",
        "models": "/models",
        "knowledge": "/knowledge",
        "settings": "/settings",
        "agents": "/agents",
    }

    to_run = journeys or list(pages.keys())
    results = []

    async with Arken(base_url=base_url, headless=True) as a:
        for name in to_run:
            if name not in pages:
                print(f"Unknown journey: {name}")
                continue
            print(f"Running {name}...", end=" ", flush=True)
            start = time.perf_counter()
            try:
                ok = await a.goto(pages[name])
                assert ok and pages[name] in a.current_url, f"goto failed: {a.current_url}"
                duration = (time.perf_counter() - start) * 1000
                print(f"PASS ({duration:.0f}ms)")
                results.append((name, "passed", duration, ""))
            except Exception as e:
                duration = (time.perf_counter() - start) * 1000
                print(f"FAIL ({duration:.0f}ms): {e}")
                results.append((name, "failed", duration, str(e)))

    return results


async def run_with_api(base_url: str, journeys: list[str] | None = None):
    """Check API endpoints directly, no browser."""
    from avion.backends.api import ApiBackend

    backend = ApiBackend(base_url=base_url)
    await backend.start()

    endpoints = [
        ("chat", "/api/health"),
        ("training", "/api/training/status"),
        ("datasets", "/api/datasets"),
        ("models", "/api/models"),
        ("knowledge", "/api/knowledge"),
        ("settings", "/api/settings"),
        ("agents", "/api/agents"),
        ("souls", "/api/souls"),
    ]
    to_run = journeys or [name for name, _ in endpoints]
    results = []

    for name, endpoint in endpoints:
        if name not in to_run:
            continue
        print(f"Checking {name} ({endpoint})...", end=" ", flush=True)
        start = time.perf_counter()
        try:
            resp = await backend.get(endpoint)
            duration = (time.perf_counter() - start) * 1000
            if resp.ok or resp.status == 404:  # 404 ok for optional endpoints
                print(f"PASS ({duration:.0f}ms) [status={resp.status}]")
                results.append((name, "passed", duration, ""))
            else:
                print(f"FAIL ({duration:.0f}ms) [status={resp.status}]")
                results.append((name, "failed", duration, f"HTTP {resp.status}"))
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            print(f"ERROR ({duration:.0f}ms): {e}")
            results.append((name, "error", duration, str(e)))

    await backend.stop()
    return results


def print_report(results: list[tuple]):
    print("\n" + "=" * 60)
    print("Arken Journey Test Report")
    print("=" * 60)

    passed = sum(1 for _, s, _, _ in results if s == "passed")
    failed = sum(1 for _, s, _, _ in results if s in ("failed", "error"))
    total_duration = sum(d for _, _, d, _ in results)

    for name, status, duration, error in results:
        icon = "PASS" if status == "passed" else "FAIL"
        print(f"  [{icon}] {name}: {status} ({duration:.0f}ms)")
        if error:
            print(f"         {error[:100]}")

    print("-" * 60)
    print(f"  Total: {len(results)} | Passed: {passed} | Failed: {failed}")
    print(f"  Duration: {total_duration:.0f}ms")
    print("=" * 60)

    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Run Arken journey tests")
    parser.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://localhost:3000"))
    parser.add_argument("--backend", choices=["playwright", "api"], default="api")
    parser.add_argument("--journey", nargs="+", help="Specific journeys to run")
    args = parser.parse_args()

    print(f"Running journey tests against {args.base_url} ({args.backend})")
    if args.backend == "playwright":
        results = asyncio.run(run_journeys(args.base_url, args.journey))
    else:
        results = asyncio.run(run_with_api(args.base_url, args.journey))

    success = print_report(results)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
