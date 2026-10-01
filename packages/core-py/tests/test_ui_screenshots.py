"""
UI screenshot tests — the app's headers across pages + basic interactions.

Card ec2ae5c1 (ask 2026-08-30): "computer use in component tests of basic
UI interactions so we can screenshot them... run it on our headers across
pages".

Mechanics
---------
- Playwright (Python, sync API) lives in the shared ``.venv`` already.
  Browser: ``~/.cache/ms-playwright`` chromium when present, otherwise the
  **flatpak chromium** via its export shim (the ms-playwright cache was
  wiped once on 2026-10-01; the flatpak path needs no download).
- This playwright build has no ``expect().to_have_screenshot``, so
  ``expect_snapshot()`` does it manually: first run writes the baseline,
  later runs byte-compare and, on drift, save ``*.actual.png`` +
  ``*.diff.png`` (PIL) beside the baseline and fail.
- What is screenshotted, and why ONLY these elements:
  * ``header.sl-app-mobile-header`` (brand + Open-menu button) — the one
    header whose content is fully static on every route. The desktop
    top-bar header carries a time greeting and LIVE health status
    ("Good morning", "CPU 100%", "1 req") — pixel-baselining it would rot
    hourly, so desktop gets a visibility check instead of pixels.
  * the mobile menu drawer (``[role=dialog]``) after tapping Open menu.
- Baselines live in ``__screenshots__/`` next to this file and are
  committed. A missing baseline is created on first run; a mismatch fails
  and writes a diff image beside it. Same machine + same headless chromium
  renders deterministically — on a different machine, delete the stale
  PNGs to regenerate.
- Marked ``slow`` + ``e2e`` so the default ``pytest`` run
  (``-m "not slow"``) deselects this file; browsers only launch when
  explicitly asked. Use ``-c pytest.ini`` so the ROOT config (which
  registers the ``e2e`` marker) wins over the nested core-py one.

Run
---
    .venv/bin/python -m pytest -c pytest.ini \\
        packages/core-py/tests/test_ui_screenshots.py -m slow -v

Web server must be up: ``http://localhost:5173`` by default (systemd
``slough-web`` unit). ``dev-stack.sh`` serves :3000 — override with
``SLO_WEB_URL=http://localhost:3000``.
"""

import os
from pathlib import Path

import pytest
from PIL import Image, ImageChops
from playwright.sync_api import Page, expect, sync_playwright

pytestmark = [pytest.mark.slow, pytest.mark.e2e]

WEB_URL = os.environ.get("SLO_WEB_URL", "http://localhost:5173")

# (path, baseline-name) — stable, logged-out routes. `/models` is excluded
# on purpose: it redirects to /developer.
PAGES = [
    ("/", "home"),
    ("/chat", "chat"),
    ("/settings", "settings"),
    ("/training", "training"),
    ("/shortcuts", "shortcuts"),
]

SNAP_DIR = "__screenshots__"

MOBILE_HEADER = "header.sl-app-mobile-header"

# Fallback when ~/.cache/ms-playwright is absent (it was wiped once on
# 2026-10-01): the system flatpak chromium, driven through its export shim
# so `flatpak run` sets up the sandbox libs. Verified launchable + able to
# load the app.
FLATPAK_SHIM = os.path.expanduser(
    "~/.local/share/flatpak/exports/bin/org.chromium.Chromium"
)


def _launch(browser_type):
    """Prefer the playwright-managed chromium; fall back to flatpak."""
    if os.path.isdir(os.path.expanduser("~/.cache/ms-playwright")):
        return browser_type.launch(headless=True)
    if os.path.exists(FLATPAK_SHIM):
        return browser_type.launch(
            executable_path=FLATPAK_SHIM, headless=True, args=["--no-sandbox"]
        )
    pytest.fail(
        "No chromium available: ~/.cache/ms-playwright is missing and the "
        "flatpak shim is absent. Run: .venv/bin/python -m playwright install chromium"
    )


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        b = _launch(p.chromium)
        yield b
        b.close()


@pytest.fixture(scope="module")
def page(browser):
    """Desktop page, 1280x720."""
    ctx = browser.new_context(viewport={"width": 1280, "height": 720})
    pg = ctx.new_page()
    yield pg
    ctx.close()


@pytest.fixture(scope="module")
def mobile_page(browser):
    """Mobile page, 375x667 — where the mobile header renders."""
    ctx = browser.new_context(viewport={"width": 375, "height": 667})
    pg = ctx.new_page()
    yield pg
    ctx.close()


def go(page: Page, path: str) -> None:
    """Navigate and wait until the app shell has rendered."""
    page.goto(f"{WEB_URL}{path}", wait_until="load", timeout=20000)
    # The health SSE stream prints "Connecting..." until its first event;
    # the shell is only stable once that resolves.
    try:
        page.wait_for_function(
            "() => !document.body.innerText.includes('Connecting...')",
            timeout=10000,
        )
    except Exception:
        pass  # banner may resolve late; the header checks below govern
    # Boot splash: a full-screen z-[9999] overlay that would both cover
    # screenshots and intercept clicks if still mounted.
    try:
        page.wait_for_selector(
            "div.fixed.inset-0.z-\\[9999\\]",
            state="detached",
            timeout=10000,
        )
    except Exception:
        pass  # overlay may not render at all on warm loads
    expect(page.locator("header:visible").first).to_be_visible(timeout=15000)


def snap_path(name: str) -> Path:
    return Path(__file__).parent / SNAP_DIR / name


def settle(page: Page) -> None:
    """Determinism: fonts loaded + transitions done before any capture."""
    page.evaluate("() => document.fonts.ready.then(() => true)")
    page.wait_for_timeout(350)


def expect_snapshot(capture, name: str) -> None:
    """Screenshot assertion (this playwright build has no
    ``expect().to_have_screenshot``): first run writes the baseline, later
    runs byte-compare. On drift, save ``*.actual.png`` + ``*.diff.png``
    beside the baseline and fail with the paths."""
    path = snap_path(name)
    actual = capture()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(actual)
        return
    if actual == path.read_bytes():
        return
    actual_path = path.with_name(f"{path.stem}.actual.png")
    actual_path.write_bytes(actual)
    try:
        expected = Image.open(path).convert("RGB")
        got = Image.open(actual_path).convert("RGB")
        if got.size == expected.size:
            diff = ImageChops.difference(expected, got)
            if diff.getbbox() is not None:
                diff.save(path.with_name(f"{path.stem}.diff.png"))
    except Exception:
        pass  # diagnostics are best-effort; the byte mismatch already fails
    raise AssertionError(
        f"screenshot drifted from baseline {path.name} "
        f"(actual + diff saved beside it; delete the baseline to regenerate)"
    )


class TestHeaders:
    """The shared header on every stable route, at both breakpoints."""

    @pytest.mark.parametrize("path,name", PAGES)
    def test_mobile_header_stable(self, mobile_page: Page, path: str, name: str):
        go(mobile_page, path)
        header = mobile_page.locator(MOBILE_HEADER)
        # Semantic guard first: pixels alone can pass vacuously if the
        # locator drifts to an empty element.
        expect(header).to_be_visible()
        expect(header).to_contain_text("sloughGPT")
        settle(mobile_page)
        expect_snapshot(
            lambda: header.screenshot(animations="disabled"),
            f"header_mobile_{name}.png",
        )

    # `/chat` is deliberately excluded from the desktop list: it renders
    # without the desktop top bar (immersive layout — verified live).
    @pytest.mark.parametrize("path,name", [p for p in PAGES if p[0] != "/chat"])
    def test_desktop_topbar_present(self, page: Page, path: str, name: str):
        """Desktop top bar renders on every route (visibility, not pixels —
        its status line and greeting are live/time-dependent by design)."""
        go(page, path)
        expect(page.locator("header:visible").first).to_be_visible()


class TestInteractions:
    """Basic UI interactions the card asks to capture."""

    def test_mobile_menu_opens(self, mobile_page: Page):
        go(mobile_page, "/")
        button = mobile_page.get_by_role("button", name="Open menu")
        expect(button).to_be_visible()
        button.click()
        drawer = mobile_page.locator("[role=dialog]")
        expect(drawer).to_be_visible(timeout=5000)
        # And the drawer really is the navigation:
        expect(drawer.get_by_role("link", name="Chat").first).to_be_visible()
        settle(mobile_page)
        # Drawer only (not the full page): the top bar shows live status.
        # The consciousness-status row (model + calm/busy + dot) is LIVE
        # data — masked to a constant so the baseline can't rot when other
        # sessions use the model.
        status = drawer.get_by_label("Consciousness status")
        expect_snapshot(
            lambda: drawer.screenshot(animations="disabled", mask=[status]),
            "interaction_mobile_menu_open.png",
        )

    def test_sidebar_link_navigates(self, page: Page):
        """A nav link actually routes — no pixels, pure behavior."""
        go(page, "/")
        link = page.locator("aside:visible a[href='/chat']").first
        expect(link).to_be_visible(timeout=5000)
        link.click()
        expect(page).to_have_url(f"{WEB_URL}/chat", timeout=10000)
        expect(page.locator("header:visible").first).to_be_visible()
