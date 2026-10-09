"""
User Journey Tests — avion browser automation.

Runs all web UI flow tests headlessly. For CI and local verification.

This file used to drive raw ``playwright.sync_api``. It now drives the shared
avion computer-use library (card e26dc68c) so locator/backend bugs are fixed
once, in ``packages/avion``, instead of once per caller. Test names,
assertions and counts are unchanged: 103 tests.

Usage:
    SLO_WEB_URL=http://localhost:5173 scripts/python -m pytest \
        packages/core-py/tests/test_user_journeys.py -x -v

Requirements:
    .venv/bin/playwright install chromium
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip(
    "playwright.sync_api",
    reason="playwright not installed in this env — journey suite skipped (card 6e826226)",
)
from avion import Arken, ElementLocator, PageControls, SyncRunner

# Web lives on vite :5173 (scripts/dev-stack.sh; card e47e19ee retired the
# stale :3000 default). Override with SLO_WEB_URL for non-standard setups.
BASE = os.environ.get("SLO_WEB_URL") or "http://localhost:5173"
API = os.environ.get("SLO_API_URL") or "http://localhost:8000"
RESULTS = []

# One persistent loop for the whole module: Playwright objects bind to the
# loop that created them, so every call must go through the same runner.
RUNNER = SyncRunner()

# ── API readiness helpers ─────────────────────────────────────────────


def _api_is_ready() -> bool:
    """Check if the API is responding to health checks."""
    try:
        req = urllib.request.Request(f"{API}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def _wait_for_api(timeout: int = 60) -> bool:
    """Block until the API is ready or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _api_is_ready():
            return True
        time.sleep(1)
    return False


def _web_is_ready() -> bool:
    """Check if the web app answers without an HTTP client error."""
    try:
        req = urllib.request.Request(BASE)
        with urllib.request.urlopen(req, timeout=5) as resp:
            return 200 <= resp.status < 400
    except Exception:
        return False


# Route-level journeys drive a real browser against a live stack. Skip the
# whole module when one isn't running instead of failing every test, so the
# default offline suite stays green. Start one first if you want these:
#   scripts/dev-stack.sh   (web :5173, api :8000)  or  ./sloughgpt serve --web
pytestmark = pytest.mark.skipif(
    not (_web_is_ready() and _api_is_ready()),
    reason=(
        "route-level journeys need a live stack "
        "(web :5173 + api :8000 — scripts/dev-stack.sh or sloughgpt serve --web)"
    ),
)


def _api_has_model() -> bool:
    """Check if the API has a model loaded (or loading)."""
    try:
        req = urllib.request.Request(f"{API}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            d = data.get("data", data)
            return d.get("model_loaded") or d.get("model_loading")
    except Exception:
        return False


@pytest.fixture(scope="session", autouse=True)
def ensure_servers_ready():
    """Block until both API and web server are ready before any test runs."""
    # Live-stack journey tests: skip (don't error) when no server is up —
    # CI runs core-py as a pure unit job with no ./sloughgpt serve process.
    timeout = 5 if os.environ.get("CI") else 90
    if not _wait_for_api(timeout=timeout):
        pytest.skip(
            f"API at {API} not ready — journey tests need a live stack "
            "(start it with: FORCE_COLOR=1 ./sloughgpt serve --web)"
        )
    # Give the web server a moment to compile after API is up
    time.sleep(3)


# ── Test infrastructure ───────────────────────────────────────────────


def ok(name: str, passed: bool, detail: str = ""):
    RESULTS.append({"test": name, "passed": passed, "detail": detail})
    mark = "ok" if passed else "ERR"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))


@pytest.fixture(scope="module")
def session():
    """One avion session for the module, started on the shared runner."""
    s = Arken(base_url=BASE, headless=True)
    RUNNER.run(s.start())
    yield s
    RUNNER.run(s.stop())


@pytest.fixture(scope="session", autouse=True)
def close_runner():
    """Close the runner's loop after every module has torn down."""
    yield
    RUNNER.close()


# ── avion helpers ─────────────────────────────────────────────────────
# Thin sync shims over the async avion API. They add no browser
# capabilities of their own — every call lands in packages/avion.


def _backend(s: Arken):
    b = s.backend
    assert b is not None, "session not started"
    return b


def _loc(sel) -> ElementLocator:
    return sel if isinstance(sel, ElementLocator) else ElementLocator.css(str(sel))


def goto(s: Arken, path: str, *, wait_until: str = "load", timeout: float = 20.0) -> None:
    backend = _backend(s)
    # wait_until/timeout are a PageControls capability, not core Backend —
    # probe before passing them (CLI/API backends have no such knobs).
    assert isinstance(backend, PageControls), "backend lacks page controls"
    RUNNER.run(backend.navigate(f"{BASE}{path}", wait_until=wait_until, timeout=timeout))


def wait_fn(s: Arken, expression: str, timeout: float = 10.0) -> bool:
    """Poll a JS predicate (Playwright wait_for_function equivalent)."""
    backend = _backend(s)
    if not isinstance(backend, PageControls):
        return False
    return RUNNER.run(backend.wait_for_function(expression, timeout))


def find(s: Arken, sel, timeout: float = 10.0):
    """First match or None; waits up to timeout for it to appear."""
    return RUNNER.run(s.find_optional(_loc(sel), timeout=timeout))


def find_all(s: Arken, sel) -> list:
    return RUNNER.run(s.find_all(_loc(sel)))


def count(s: Arken, sel) -> int:
    return len(find_all(s, sel))


def wait_present(s: Arken, sel, timeout: float = 10.0) -> bool:
    """Wait until an element exists and is visible (locator.wait_for)."""
    deadline = time.time() + timeout
    loc = _loc(sel)
    while time.time() < deadline:
        if RUNNER.run(s.find_optional(loc, timeout=1.0)) is not None:
            return True
        time.sleep(0.2)
    return False


def click(s: Arken, sel, *, force: bool = False, timeout: float = 10.0) -> None:
    el = find(s, sel, timeout=timeout)
    assert el is not None, f"element not found: {_loc(sel).describe()}"
    RUNNER.run(s.click(el, force=force))


def fill(s: Arken, sel, value: str, *, force: bool = False) -> None:
    RUNNER.run(s.fill(_loc(sel), value, force=force))


def input_value(s: Arken, sel) -> str:
    el = find(s, sel, timeout=5.0)
    return str(el.value or "") if el is not None else ""


def focus(s: Arken, sel) -> None:
    el = find(s, sel)
    assert el is not None, f"element not found: {_loc(sel).describe()}"
    RUNNER.run(_backend(s).focus(el))


def press_on(s: Arken, sel, key: str) -> None:
    el = find(s, sel)
    assert el is not None, f"element not found: {_loc(sel).describe()}"
    RUNNER.run(_backend(s).press_element(el, key))


def press_key(s: Arken, key: str) -> None:
    RUNNER.run(s.press(key))


def body_text(s: Arken) -> str:
    el = find(s, "body", timeout=5.0)
    return el.text if el is not None else ""


def go(s: Arken, path: str) -> str:
    """Navigate and return body text.

    Waits for:
    1. Page load (20s timeout)
    2. "Connecting..." text to disappear (15s timeout)
    3. Additional settle time for SSE streams to deliver first events
    """
    goto(s, path, wait_until="load", timeout=20.0)

    # Wait for "Connecting..." to disappear — means the health SSE stream
    # delivered its first event OR the fallback HTTP poll succeeded.
    wait_fn(s, "() => !document.body.innerText.includes('Connecting...')", 15.0)

    # Extra settle: SSE streams fire every 3s, allow 1 full cycle for
    # downstream components (status bar, KPI grid) to populate.
    time.sleep(1)

    # If page shows error boundary ("Something went wrong"), retry once
    body = body_text(s)
    if "Something went wrong" in body and "Try again" in body:
        try:
            click(s, "button:has-text('Try again')", timeout=5.0)
            time.sleep(2)
            body = body_text(s)
        except Exception:
            pass

    return body


# ── Dashboard ─────────────────────────────────────────────────


class TestDashboard:
    def test_loads(self, session: Arken):
        body = go(session, "/")
        ok("dashboard_loads", len(body) > 50, f"len={len(body)}")
        assert len(body) > 50

    def test_has_nav(self, session: Arken):
        go(session, "/")
        links = count(session, ElementLocator.role("link"))
        ok("dashboard_has_nav", links > 3, f"links={links}")
        assert links > 3


# ── Navigation ────────────────────────────────────────────────

ROUTES = [
    ("/chat", "chat"),
    ("/training", "training"),
    ("/datasets", "datasets"),
    ("/models", "models"),
    ("/agents", "agents"),
    ("/souls", "souls"),
    ("/knowledge", "knowledge"),
    ("/monitoring", "monitoring"),
    ("/settings", "settings"),
    ("/planner", "planner"),
    ("/benchmark", "benchmark"),
    ("/tokenizer", "tokenizer"),
    ("/errors", "errors"),
    ("/security", "security"),
    ("/shell", "shell"),
    ("/feedback", "feedback"),
    ("/files", "files"),
    ("/adapters", "adapters"),
    # Tools pages
    ("/brainstorm", "brainstorm"),
    ("/decide", "decide"),
    ("/explain", "explain"),
    ("/rewrite", "rewrite"),
    ("/translate", "translate"),
    ("/wellness", "wellness"),
    ("/writing", "writing"),
    # Consciousness pages
    ("/consciousness/dashboard", "consciousness_dashboard"),
    ("/consciousness/health", "consciousness_health"),
    ("/consciousness/training", "consciousness_training"),
    ("/consciousness/settings", "consciousness_settings"),
    ("/consciousness/playground", "consciousness_playground"),
    # Training sub-pages
    ("/training/analytics", "training_analytics"),
    ("/training/presets", "training_presets"),
    ("/training/runs", "training_runs"),
    ("/training/compare", "training_compare"),
    ("/training/trends", "training_trends"),
    # Other
    ("/shortcuts", "shortcuts"),
    ("/phoneme", "phoneme"),
]


class TestNavigation:
    @pytest.mark.parametrize("path,name", ROUTES)
    def test_route(self, session: Arken, path: str, name: str):
        body = go(session, path)
        ok(f"nav_{name}", len(body) > 50, f"len={len(body)}")
        assert len(body) > 50, f"{path} returned empty body"


# ── Chat ──────────────────────────────────────────────────────

CHAT_INPUT = "textarea:visible, input[type='text']:visible"


class TestChat:
    def test_loads(self, session: Arken):
        body = go(session, "/chat")
        ok("chat_loads", len(body) > 50, f"len={len(body)}")
        assert len(body) > 50

    def test_has_input(self, session: Arken):
        go(session, "/chat")
        inputs = count(session, CHAT_INPUT)
        ok("chat_has_input", inputs > 0)
        assert inputs > 0

    def test_type_message(self, session: Arken):
        go(session, "/chat")
        if count(session, CHAT_INPUT) == 0:
            ok("chat_type_message", False, "no input found")
            pytest.skip("no input")
        fill(session, CHAT_INPUT, "Hello test message", force=True)
        val = input_value(session, CHAT_INPUT)
        ok("chat_type_message", "test message" in val, f"val={val[:40]}")
        assert "test message" in val


# ── Training ──────────────────────────────────────────────────


class TestTraining:
    def test_loads(self, session: Arken):
        body = go(session, "/training")
        ok("training_loads", "train" in body.lower(), f"len={len(body)}")
        assert "train" in body.lower()

    def test_job_detail_loads(self, session: Arken):
        """Job detail page renders for a sample job ID (shows job info or not-found)."""
        body = go(session, "/training/job/test-job-id")
        has_content = len(body) > 50
        ok("training_job_detail_loads", has_content, f"len={len(body)}")
        assert has_content

    def test_job_detail_back_link(self, session: Arken):
        """Job detail page has a back link to training list."""
        go(session, "/training/job/test-job-id")
        time.sleep(1)
        back = count(session, 'a[href="/training"]')
        ok("training_job_detail_back_link", back > 0)
        assert back > 0


# ── Settings ──────────────────────────────────────────────────


class TestSettings:
    def test_loads(self, session: Arken):
        body = go(session, "/settings")
        ok("settings_loads", len(body) > 50, f"len={len(body)}")
        assert len(body) > 50


# ── Planner ───────────────────────────────────────────────────


class TestPlanner:
    def test_loads(self, session: Arken):
        body = go(session, "/planner")
        ok("planner_loads", len(body) > 50, f"len={len(body)}")
        assert len(body) > 50


# ── Models ────────────────────────────────────────────────────


class TestModels:
    def test_loads(self, session: Arken):
        body = go(session, "/models")
        # Page might show "Connecting..." if API is down — that's still a valid page load
        has_content = len(body) > 50
        ok("models_loads", has_content, f"len={len(body)}")
        assert has_content


# ── Monitoring ────────────────────────────────────────────────


class TestMonitoring:
    def test_loads(self, session: Arken):
        body = go(session, "/monitoring")
        has_metrics = any(w in body.lower() for w in ["cpu", "memory", "monitor", "health", "gpu"])
        ok("monitoring_loads", has_metrics, f"len={len(body)}")
        assert has_metrics


# ── Knowledge ─────────────────────────────────────────────────


class TestKnowledge:
    def test_loads(self, session: Arken):
        body = go(session, "/knowledge")
        has_kw = any(w in body.lower() for w in ["knowledge", "memory", "fact", "search"])
        ok("knowledge_loads", has_kw, f"len={len(body)}")
        assert has_kw


# ── Datasets Import ───────────────────────────────────────────

HF_DIALOG = '[role="dialog"] [role="radio"][aria-label^="HuggingFace:"]'
HF_INPUT = "input[placeholder='username/dataset-name']"


def _open_datasets(s: Arken, settle: float = 1.0) -> None:
    go(s, "/datasets")
    time.sleep(settle)
    click(s, ElementLocator.role("button", "Add file"), force=True)
    time.sleep(1)


class TestDatasetsImport:
    def test_loads(self, session: Arken):
        body = go(session, "/datasets")
        ok("datasets_loads", len(body) > 50, f"len={len(body)}")
        assert len(body) > 50

    def test_import_button_exists(self, session: Arken):
        go(session, "/datasets")
        time.sleep(1)
        btn = count(session, ElementLocator.role("button", "Add file"))
        ok("datasets_import_button", btn > 0)
        assert btn > 0

    def test_import_dialog_opens(self, session: Arken):
        _open_datasets(session)
        dialog = count(session, ElementLocator.role("dialog"))
        ok("datasets_import_dialog_opens", dialog > 0)
        assert dialog > 0

    def test_hf_radio_exists(self, session: Arken):
        _open_datasets(session)
        hf = count(session, HF_DIALOG)
        ok("datasets_hf_source_exists", hf > 0)
        assert hf > 0
        press_key(session, "Escape")
        time.sleep(0.3)

    def test_hf_radio_clicks(self, session: Arken):
        _open_datasets(session)
        focus(session, HF_DIALOG)
        time.sleep(0.2)
        press_on(session, HF_DIALOG, "Space")
        time.sleep(1)
        inp = count(session, HF_INPUT)
        ok("datasets_hf_radio_clicks", inp > 0)
        assert inp > 0
        press_key(session, "Escape")
        time.sleep(0.3)

    def test_hf_input_fills(self, session: Arken):
        _open_datasets(session)
        focus(session, HF_DIALOG)
        time.sleep(0.2)
        press_on(session, HF_DIALOG, "Space")
        time.sleep(1)
        fill(session, HF_INPUT, "heptapod/titanic")
        time.sleep(0.3)
        val = input_value(session, HF_INPUT)
        ok("datasets_hf_input_fills", val == "heptapod/titanic")
        assert val == "heptapod/titanic"
        press_key(session, "Escape")
        time.sleep(0.3)

    def test_hf_import_attempt(self, session: Arken):
        # Reload to clear any stale state from prior tests
        goto(session, "/datasets", wait_until="load", timeout=20.0)
        wait_fn(session, "() => !document.body.innerText.includes('Connecting...')", 10.0)
        time.sleep(2)
        click(session, ElementLocator.role("button", "Add file"), force=True)
        time.sleep(2)
        focus(session, HF_DIALOG)
        time.sleep(0.2)
        press_on(session, HF_DIALOG, "Space")
        time.sleep(1)
        fill(session, HF_INPUT, "heptapod/titanic")
        time.sleep(0.5)
        # Try clicking Import, but don't fail if dialog blocks it
        try:
            click(session, '[role="dialog"] button:text-is("Import")', force=True, timeout=3.0)
        except Exception:
            pass
        # handleImport flips the button to "Importing..." synchronously — poll for it
        reacted = wait_present(session, '[role="dialog"] :text("Importing...")', 5.0)
        body = body_text(session)
        success = reacted or "heptapod/titanic" in body
        ok("datasets_hf_import_attempt", success, f"reacted={reacted}, body_snippet={body[-200:]}")
        assert success


# ── Redirects ─────────────────────────────────────────────────

REDIRECTS = [
    ("/companion", "souls"),
    ("/evaluate", "benchmark"),
    ("/memory", "knowledge"),
    ("/collections", "datasets"),
    ("/self-train", "training"),
    ("/admin", "settings"),
    ("/images", "files"),
    ("/session", "shell"),
]


class TestRedirects:
    @pytest.mark.parametrize("old,expected", REDIRECTS)
    def test_redirect(self, session: Arken, old: str, expected: str):
        body = go(session, old)
        ok(
            f"redirect_{old.replace('/', '_')}",
            len(body) > 50,
            f"{old} -> /{expected}, len={len(body)}",
        )
        assert len(body) > 50, f"Redirect {old} failed"


# ── Tools Pages — Interactive Flows ──────────────────────────


class TestToolsFlows:
    def test_brstorm_has_input_and_suggestions(self, session: Arken):
        body = go(session, "/brainstorm")
        has_input = count(session, "textarea:visible") > 0
        has_suggestions = "Name ideas" in body or "Weekend" in body
        ok(
            "brainstorm_input_and_suggestions",
            has_input and has_suggestions,
            f"input={has_input}, suggestions={has_suggestions}",
        )
        assert has_input and has_suggestions

    def test_decide_has_two_options(self, session: Arken):
        go(session, "/decide")
        inputs = count(session, "input:visible")
        ok("decide_has_options", inputs >= 2, f"inputs={inputs}")
        assert inputs >= 2

    def test_explain_has_difficulty_buttons(self, session: Arken):
        go(session, "/explain")
        wait_present(session, ":text-is('Level')", 10.0)
        body = body_text(session)
        has_simple = "Simple" in body
        has_moderate = "Moderate" in body
        ok(
            "explain_has_difficulty",
            has_simple and has_moderate,
            f"simple={has_simple}, moderate={has_moderate}",
        )
        assert has_simple and has_moderate

    def test_rewrite_has_action_buttons(self, session: Arken):
        body = go(session, "/rewrite")
        has_grammar = "Fix Grammar" in body
        has_shorter = "Make Shorter" in body
        ok(
            "rewrite_has_actions",
            has_grammar and has_shorter,
            f"grammar={has_grammar}, shorter={has_shorter}",
        )
        assert has_grammar and has_shorter

    def test_translate_has_language_selector(self, session: Arken):
        go(session, "/translate")
        wait_present(session, ":text('EN→ES')", 10.0)
        body = body_text(session)
        has_pair = "EN→ES" in body
        has_translate_btn = "Translate" in body
        ok(
            "translate_has_selector",
            has_pair and has_translate_btn,
            f"pair={has_pair}, btn={has_translate_btn}",
        )
        assert has_pair and has_translate_btn

    def test_wellness_has_options(self, session: Arken):
        body = go(session, "/wellness")
        has_sleep = "Sleep" in body
        has_meditate = "Meditat" in body
        ok(
            "wellness_has_options",
            has_sleep and has_meditate,
            f"sleep={has_sleep}, meditate={has_meditate}",
        )
        assert has_sleep and has_meditate

    def test_writing_has_tones_and_types(self, session: Arken):
        body = go(session, "/writing")
        has_friendly = "Friendly" in body
        has_email = "Email" in body
        ok(
            "writing_has_tones_types",
            has_friendly and has_email,
            f"friendly={has_friendly}, email={has_email}",
        )
        assert has_friendly and has_email


# ── Consciousness Pages ──────────────────────────────────────


class TestConsciousnessFlows:
    def test_dashboard_loads(self, session: Arken):
        body = go(session, "/consciousness/dashboard")
        ok("consciousness_dashboard_loads", len(body) > 50, f"len={len(body)}")
        assert len(body) > 50

    def test_health_loads(self, session: Arken):
        body = go(session, "/consciousness/health")
        ok("consciousness_health_loads", len(body) > 50, f"len={len(body)}")
        assert len(body) > 50

    def test_playground_loads(self, session: Arken):
        body = go(session, "/consciousness/playground")
        ok("consciousness_playground_loads", len(body) > 50, f"len={len(body)}")
        assert len(body) > 50


# ── Route smoke (consolidated from legacy comprehensive journeys) ────

SMOKE_ROUTES = [
    "/training/runs",
    "/files",
    "/adapters",
    "/agents",
    "/souls",
    "/shell",
    "/benchmark",
    "/tokenizer",
    "/errors",
    "/security",
    "/feedback",
    "/auto-train",
    "/consciousness/debug",
    "/consciousness/testing",
    "/consciousness/analytics",
    "/consciousness/insights",
    "/consciousness/monitor",
    "/consciousness/benchmark",
    "/consciousness/versions",
    "/infer",
    "/evaluate",
    "/compare",
    "/export",
    "/vector",
    "/multimodal",
    "/workflow",
    "/memory",
    "/vm",
]


class TestRouteSmoke:
    """Every app route loads without crashing (no error boundary, non-empty)."""

    @pytest.mark.parametrize("route", SMOKE_ROUTES)
    def test_route_loads(self, session: Arken, route: str):
        body = go(session, route)
        crashed = "Something went wrong" in body
        not_found = "This page could not be found" in body
        passed = len(body) > 50 and not crashed and not not_found
        ok(f"route_loads[{route}]", passed, f"len={len(body)}, crash={crashed}, 404={not_found}")
        assert passed, f"Route {route} crashed,404, or empty (len={len(body)})"


# ── Results ───────────────────────────────────────────────────


@pytest.fixture(scope="session", autouse=True)
def save_results():
    yield
    out = Path(__file__).parent / "test_results" / "user_journey_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(RESULTS, indent=2))

    total = len(RESULTS)
    passed = sum(1 for r in RESULTS if r["passed"])
    failed = total - passed
    print(f"\n{'=' * 50}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 50}")
    if failed:
        for r in RESULTS:
            if not r["passed"]:
                print(f"  ERR {r['test']}: {r['detail']}")
