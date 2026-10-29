#!/usr/bin/env python3
"""
Screenshot headers across all pages using avion.

Runs against the shared avion computer-use library (card e26dc68c) instead of
raw Playwright, so browser-driver fixes land once in packages/avion.

Usage:
    python3 scripts/screenshot_headers.py [--output DIR]
"""

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_AVION_SRC = os.path.join(REPO_ROOT, "packages", "avion", "src")
if os.path.isdir(_AVION_SRC) and _AVION_SRC not in sys.path:
    sys.path.insert(0, _AVION_SRC)

from avion import SyncRunner  # noqa: E402
from avion.backends.playwright import PlaywrightBackend  # noqa: E402
from avion.network import MockResponse, MockRule, NetworkMocker  # noqa: E402

BASE_URL = "http://localhost:3010"

PAGES = [
    ("home", "/"),
    ("chat", "/chat"),
    ("models", "/models"),
    ("training", "/training"),
    ("datasets", "/datasets"),
    ("settings", "/settings"),
    ("monitoring", "/monitoring"),
    ("knowledge", "/knowledge"),
    ("memory", "/memory"),
    ("shell", "/shell"),
    ("vm", "/vm"),
    ("companion", "/companion"),
    ("agents", "/agents"),
    ("collections", "/collections"),
    ("souls", "/souls"),
    ("tokenizer", "/tokenizer"),
    ("token-tree", "/token-tree"),
    ("infer", "/infer"),
    ("self-train", "/self-train"),
    ("lora-eval", "/lora-eval"),
    ("meta-weights", "/meta-weights"),
    ("rate-limit", "/rate-limit"),
    ("benchmark", "/benchmark"),
    ("feedback", "/feedback"),
    ("images", "/images"),
    ("files", "/files"),
    ("docstore", "/docstore"),
    ("kb", "/kb"),
    ("vector", "/vector"),
    ("workflow", "/workflow"),
    ("world", "/world"),
    ("registry", "/registry"),
    ("security", "/security"),
    ("admin", "/admin"),
    ("session", "/session"),
    ("evaluate", "/evaluate"),
    ("experiments", "/experiments"),
    ("export", "/export"),
    ("adapters", "/adapters"),
    ("learn", "/learn"),
    ("voice", "/voice"),
    ("multimodal", "/multimodal"),
    ("compare", "/compare"),
    ("kanban", "/kanban"),
    ("auto-train", "/auto-train"),
]

VIEWPORTS = [
    ("desktop", 1280, 800),
    ("tablet", 768, 1024),
    ("mobile", 375, 667),
]

MOCK_ROUTES = [
    ("**/health", '{"status":"healthy","model_loaded":false}'),
    ("**/models", '{"models":[]}'),
    ("**/datasets", '{"datasets":[]}'),
    ("**/system/**", '{"cpu_percent":45,"memory_percent":62}'),
    ("**/knowledge/**", "[]"),
    ("**/knowledge", "[]"),
]


def is_server_running():
    try:
        req = urllib.request.Request(BASE_URL, method="HEAD")
        urllib.request.urlopen(req, timeout=3)
        return True
    except Exception:
        return False


def setup_mock_routes(runner, backend, server_alive=False):
    """Fulfil canned JSON for API endpoints when no server is up.

    Playwright's ``page.route("**/health")`` glob becomes an avion
    NetworkMocker rule matched as a URL substring (``**`` stripped).
    """
    if server_alive:
        return
    mocker = NetworkMocker()
    for i, (pattern, body) in enumerate(MOCK_ROUTES):
        mocker.add_rule(
            MockRule(
                name=f"mock_{i}_{pattern}",
                url_pattern=pattern.replace("**", ""),
                response=MockResponse(
                    status=200,
                    body=body,
                    headers={"content-type": "application/json"},
                ),
            )
        )
    runner.run(backend.enable_network_mock(mocker))


def wait_for_app(runner, backend, timeout_s=10):
    """Wait for Next.js client-side hydration to finish."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            ready = runner.run(backend.evaluate("() => document.readyState === 'complete'"))
            if ready:
                runner.run(backend.wait_for_timeout(500))  # extra hydration settle
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


def screenshot_page(output_dir: Path, runner, backend, name, path):
    """Take a full-page screenshot of a route."""
    url = f"{BASE_URL}{path}"
    print(f"  [{name}] {url} ... ", end="", flush=True)
    try:
        runner.run(backend.navigate(url, wait_until="domcontentloaded", timeout=15.0))
        wait_for_app(runner, backend)

        full_path = output_dir / f"page-{name}.png"
        runner.run(backend.screenshot(path=str(full_path)))
        print(f"OK -> {full_path.name}")
        return True
    except Exception as e:
        print(f"FAIL: {type(e).__name__}")
        return False


def screenshot_viewports(output_dir: Path, runner, backend):
    """Take screenshots at different viewport sizes."""
    print("\n  Responsive (home page):")
    for vp_name, w, h in VIEWPORTS:
        runner.run(backend.set_viewport_size(w, h))
        try:
            runner.run(backend.navigate(BASE_URL, wait_until="domcontentloaded", timeout=15.0))
            wait_for_app(runner, backend)
            path = output_dir / f"page-home-{vp_name}.png"
            runner.run(backend.screenshot(path=str(path)))
            print(f"    {vp_name} ({w}x{h}) -> {path.name}")
        except Exception as e:
            print(f"    {vp_name} ({w}x{h}) FAIL: {type(e).__name__}")


def screenshot_headers(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    server_alive = is_server_running()
    if not server_alive:
        print(
            "Server not detected at :3010 — run `make web` first, or use --output for mock-only mode"
        )
    else:
        print("Server detected at :3010 — using live API responses")

    runner = SyncRunner()
    backend = PlaywrightBackend(headless=True)
    runner.run(backend.start())
    try:
        runner.run(backend.set_viewport_size(1280, 800))

        setup_mock_routes(runner, backend, server_alive=server_alive)

        print("  Page screenshots:")
        success = 0
        total = 0
        for name, path in PAGES:
            total += 1
            if screenshot_page(output_dir, runner, backend, name, path):
                success += 1

        screenshot_viewports(output_dir, runner, backend)
    finally:
        runner.run(backend.stop())
        runner.close()

    print(f"\n  {success}/{total} pages succeeded. Screenshots in {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Screenshot headers across all pages")
    parser.add_argument("--output", default="cypress/screenshots/headers", help="Output directory")
    args = parser.parse_args()

    output = Path(args.output)
    if not output.is_absolute():
        output = Path(__file__).resolve().parent.parent / args.output

    print(f"Taking header screenshots -> {output}/")
    screenshot_headers(output)
