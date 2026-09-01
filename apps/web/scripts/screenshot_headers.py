#!/usr/bin/env python3
"""
Screenshot headers across all pages using Playwright.

Usage:
    python3 scripts/screenshot_headers.py [--output DIR]
"""

import argparse
import time
import urllib.request
from pathlib import Path
from playwright.sync_api import sync_playwright

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


def setup_mock_routes(page, server_alive=False):
    if server_alive:
        return
    for pattern, body in MOCK_ROUTES:
        page.route(pattern, lambda route, b=body: route.fulfill(
            status=200, content_type="application/json", body=b,
        ))


def wait_for_app(page, timeout_s=10):
    """Wait for Next.js client-side hydration to finish."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            ready = page.evaluate("() => document.readyState === 'complete'")
            if ready:
                page.wait_for_timeout(500)  # extra hydration settle
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


def screenshot_page(output_dir: Path, page, name: str, path: str):
    """Take a full-page screenshot of a route."""
    url = f"{BASE_URL}{path}"
    print(f"  [{name}] {url} ... ", end="", flush=True)
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=15000)
        wait_for_app(page)

        full_path = output_dir / f"page-{name}.png"
        page.screenshot(path=str(full_path), full_page=True)
        print(f"OK -> {full_path.name}")
        return True
    except Exception as e:
        print(f"FAIL: {type(e).__name__}")
        return False


def screenshot_viewports(output_dir: Path, page):
    """Take screenshots at different viewport sizes."""
    print("\n  Responsive (home page):")
    for vp_name, w, h in VIEWPORTS:
        page.set_viewport_size({"width": w, "height": h})
        try:
            page.goto(BASE_URL, wait_until="domcontentloaded", timeout=15000)
            wait_for_app(page)
            path = output_dir / f"page-home-{vp_name}.png"
            page.screenshot(path=str(path), full_page=True)
            print(f"    {vp_name} ({w}x{h}) -> {path.name}")
        except Exception as e:
            print(f"    {vp_name} ({w}x{h}) FAIL: {type(e).__name__}")


def screenshot_headers(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    server_alive = is_server_running()
    if not server_alive:
        print("Server not detected at :3010 — run `make web` first, or use --output for mock-only mode")
    else:
        print("Server detected at :3010 — using live API responses")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 800})
        page = context.new_page()

        setup_mock_routes(page, server_alive=server_alive)

        print("  Page screenshots:")
        success = 0
        total = 0
        for name, path in PAGES:
            total += 1
            if screenshot_page(output_dir, page, name, path):
                success += 1

        screenshot_viewports(output_dir, page)

        page.close()
        context.close()
        browser.close()

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
