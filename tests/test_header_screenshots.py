"""Header screenshots across pages — card ec2ae5c1 (ask 2026-08-30).

The ask: "computer use in component tests of basic UI interactions so we
can screenshot them ... run it on our headers across pages." The capture
half already existed as ``apps/web/scripts/screenshot_headers.py`` (manual,
45 pages, no assertions, hardcoded :3010). This module drives the SAME
script code as a gated test:

- headers across a representative page slice + the three responsive
  viewports,
- one basic UI interaction (click through to /chat) with a post-click
  screenshot and URL assertion,
- real assertions: PNG integrity, non-blank pixels, viewport dimensions,
- skips (never fails) without a live web stack — same contract as the
  route-level journey tests (packages/core-py/tests/test_user_journeys.py).

Slow-marked browser capture: runs in ``-m ""`` sweeps against a live stack
(``scripts/dev-stack.sh`` — web :5173, or ``SLO_WEB_URL``).
"""

import importlib.util
import os
import sys
import urllib.request
from pathlib import Path

import pytest

# avion lives in the shared source tree (packages/avion/src), not
# site-packages — the path must be in place BEFORE importorskip below,
# otherwise the module skips itself ("No module named 'avion'").
_AVION_SRC = Path(__file__).resolve().parents[1] / "packages" / "avion" / "src"
if _AVION_SRC.is_dir() and str(_AVION_SRC) not in sys.path:
    sys.path.insert(0, str(_AVION_SRC))

pytest.importorskip("avion")
pytest.importorskip("playwright")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "apps" / "web" / "scripts" / "screenshot_headers.py"
WEB_URL = os.environ.get("SLO_WEB_URL") or "http://localhost:5173"

# Representative slice of the script's 45 pages: home + the primary nav
# destinations, enough to catch header regressions without a 3-minute test.
SUBSET = [
    ("home", "/"),
    ("chat", "/chat"),
    ("models", "/models"),
    ("training", "/training"),
    ("settings", "/settings"),
    ("monitoring", "/monitoring"),
    ("knowledge", "/knowledge"),
    ("shell", "/shell"),
]

MIN_BYTES = 3_000
MIN_UNIQUE_COLORS = 30


def _web_is_ready() -> bool:
    try:
        req = urllib.request.Request(WEB_URL)
        with urllib.request.urlopen(req, timeout=5) as resp:
            return 200 <= resp.status < 400
    except Exception:
        return False


def _api_is_ready() -> bool:
    try:
        req = urllib.request.Request("http://localhost:8000/health")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not _web_is_ready(),
        reason=(
            "header screenshot tests need a live web app "
            f"({WEB_URL} — scripts/dev-stack.sh or SLO_WEB_URL)"
        ),
    ),
]


def _load_script():
    """Import apps/web/scripts/screenshot_headers.py as a module (reuse, not copy)."""
    spec = importlib.util.spec_from_file_location("screenshot_headers_script", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.BASE_URL = WEB_URL  # script hardcodes :3010 (Cypress/Next port); live stack is :5173
    return mod


def _assert_real_screenshot(path: Path, *, min_width: int = 0) -> None:
    """PNG exists, is plausibly sized, and is not a blank/error shell."""
    from PIL import Image

    assert path.exists(), f"missing screenshot: {path.name}"
    size = path.stat().st_size
    assert size > MIN_BYTES, f"{path.name} only {size}B — blank or failed capture"
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"{path.name} is not a PNG"
    with Image.open(path) as im:
        assert im.width >= min_width, f"{path.name} width {im.width} < {min_width}"
        colors = im.convert("RGB").getcolors(maxcolors=1_000_001)
    if colors is not None:
        assert len(colors) >= MIN_UNIQUE_COLORS, (
            f"{path.name} has only {len(colors)} unique colors — blank render"
        )


@pytest.fixture()
def browser():
    """Script module + started runner/backend, torn down after the test."""
    mod = _load_script()
    from avion import SyncRunner
    from avion.backends.playwright import PlaywrightBackend

    runner = SyncRunner()
    backend = PlaywrightBackend(headless=True)
    runner.run(backend.start())
    try:
        runner.run(backend.set_viewport_size(1280, 800))
        # Fulfil canned API JSON when the API is down so pages settle
        # instead of hanging on fetches (script helper, reused as-is).
        mod.setup_mock_routes(runner, backend, server_alive=_api_is_ready())
        yield mod, runner, backend
    finally:
        runner.run(backend.stop())
        runner.close()


@pytest.mark.timeout(300)
def test_headers_captured_across_pages(browser, tmp_path):
    """Headers render on the page slice + all three responsive viewports."""
    mod, runner, backend = browser

    failed = []
    for name, path in SUBSET:
        if not mod.screenshot_page(tmp_path, runner, backend, name, path):
            failed.append(name)
    assert not failed, f"capture failed for pages: {failed}"

    for name, _ in SUBSET:
        _assert_real_screenshot(tmp_path / f"page-{name}.png", min_width=1000)

    mod.screenshot_viewports(tmp_path, runner, backend)
    for vp_name, min_w in (("desktop", 1200), ("tablet", 700), ("mobile", 350)):
        _assert_real_screenshot(tmp_path / f"page-home-{vp_name}.png", min_width=min_w)


@pytest.mark.timeout(300)
def test_basic_interaction_click_to_chat(browser, tmp_path):
    """Basic UI interaction: click the nav link to /chat, then screenshot."""
    from avion import ElementLocator

    mod, runner, backend = browser

    assert mod.screenshot_page(tmp_path, runner, backend, "home", "/"), (
        "home did not load before interacting"
    )

    loc = ElementLocator.css('a[href="/chat"]')
    element = runner.run(backend.wait_for(loc, timeout=15.0))
    runner.run(backend.click(element))
    runner.run(backend.wait_for_timeout(800))

    pathname = runner.run(backend.evaluate("() => window.location.pathname"))
    assert pathname == "/chat", f"click did not navigate: {pathname}"

    shot = tmp_path / "interaction-chat.png"
    runner.run(backend.screenshot(path=str(shot)))
    _assert_real_screenshot(shot, min_width=1000)
