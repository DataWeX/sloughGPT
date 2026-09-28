"""
User Journey Tests — Playwright browser automation.

Runs all web UI flow tests headlessly. Marked `browser` — deselected from
default/CI runs; run with:  pytest -m browser tests/test_user_journeys.py

Requirements:
    .venv/bin/playwright install chromium
"""

import json
import os
import time
import urllib.request
from pathlib import Path

import pytest
from playwright.sync_api import Page, sync_playwright

pytestmark = [pytest.mark.browser, pytest.mark.live]

BASE = os.environ.get("SLO_WEB_URL") or "http://localhost:3000"
API = os.environ.get("SLO_API_URL") or "http://localhost:8000"
RESULTS = []

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
    assert _wait_for_api(timeout=90), (
        f"API at {API} not ready after 90s. "
        "Start the server with: FORCE_COLOR=1 ./sloughgpt serve --web"
    )
    # Give the web server a moment to compile after API is up
    time.sleep(3)


# ── Test infrastructure ───────────────────────────────────────────────


def ok(name: str, passed: bool, detail: str = ""):
    RESULTS.append({"test": name, "passed": passed, "detail": detail})
    mark = "ok" if passed else "ERR"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        yield b
        b.close()


@pytest.fixture(scope="module")
def page(browser):
    ctx = browser.new_context(viewport={"width": 1280, "height": 720})
    pg = ctx.new_page()
    yield pg
    ctx.close()


def go(page: Page, path: str) -> str:
    """Navigate and return body text.

    Waits for:
    1. Page load (20s timeout)
    2. "Connecting..." text to disappear (15s timeout)
    3. Additional settle time for SSE streams to deliver first events
    """
    page.goto(f"{BASE}{path}", wait_until="load", timeout=20000)

    # Wait for "Connecting..." to disappear — means the health SSE stream
    # delivered its first event OR the fallback HTTP poll succeeded.
    try:
        page.wait_for_function(
            "() => !document.body.innerText.includes('Connecting...')",
            timeout=15000,
        )
    except Exception:
        pass

    # Extra settle: SSE streams fire every 3s, allow 1 full cycle for
    # downstream components (status bar, KPI grid) to populate.
    time.sleep(1)

    # If page shows error boundary ("Something went wrong"), retry once
    body = page.inner_text("body")
    if "Something went wrong" in body and "Try again" in body:
        try:
            page.locator("button:has-text('Try again')").first.click(timeout=5000)
            time.sleep(2)
            body = page.inner_text("body")
        except Exception:
            pass

    return body


# ── Dashboard ─────────────────────────────────────────────────


class TestDashboard:
    def test_loads(self, page: Page):
        body = go(page, "/")
        ok("dashboard_loads", len(body) > 50, f"len={len(body)}")
        assert len(body) > 50

    def test_has_nav(self, page: Page):
        go(page, "/")
        links = page.get_by_role("link")
        ok("dashboard_has_nav", links.count() > 3, f"links={links.count()}")
        assert links.count() > 3


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
    ("/training/runs", "training_runs"),
    # Other
    ("/shortcuts", "shortcuts"),
    ("/phoneme", "phoneme"),
]


class TestNavigation:
    @pytest.mark.parametrize("path,name", ROUTES)
    def test_route(self, page: Page, path: str, name: str):
        body = go(page, path)
        ok(f"nav_{name}", len(body) > 50, f"len={len(body)}")
        assert len(body) > 50, f"{path} returned empty body"


# ── Chat ──────────────────────────────────────────────────────


class TestChat:
    def test_loads(self, page: Page):
        body = go(page, "/chat")
        ok("chat_loads", len(body) > 50, f"len={len(body)}")
        assert len(body) > 50

    def test_has_input(self, page: Page):
        go(page, "/chat")
        inp = page.locator("textarea:visible, input[type='text']:visible").first
        ok("chat_has_input", inp.count() > 0)
        assert inp.count() > 0

    def test_type_message(self, page: Page):
        go(page, "/chat")
        inp = page.locator("textarea:visible, input[type='text']:visible").first
        if inp.count() == 0:
            ok("chat_type_message", False, "no input found")
            pytest.skip("no input")
        inp.fill("Hello test message", force=True)
        val = inp.input_value()
        ok("chat_type_message", "test message" in val, f"val={val[:40]}")
        assert "test message" in val


# ── Training ──────────────────────────────────────────────────


class TestTraining:
    def test_loads(self, page: Page):
        body = go(page, "/training")
        ok("training_loads", "train" in body.lower(), f"len={len(body)}")
        assert "train" in body.lower()

    def test_job_detail_loads(self, page: Page):
        """Job detail page renders for a sample job ID (shows job info or not-found)."""
        body = go(page, "/training/job/test-job-id")
        has_content = len(body) > 50
        ok("training_job_detail_loads", has_content, f"len={len(body)}")
        assert has_content

    def test_job_detail_back_link(self, page: Page):
        """Job detail page has a back link to training list."""
        go(page, "/training/job/test-job-id")
        time.sleep(1)
        back = page.locator('a[href="/training"]').first
        ok("training_job_detail_back_link", back.count() > 0)
        assert back.count() > 0


# ── Settings ──────────────────────────────────────────────────


class TestSettings:
    def test_loads(self, page: Page):
        body = go(page, "/settings")
        ok("settings_loads", len(body) > 50, f"len={len(body)}")
        assert len(body) > 50


# ── Planner ───────────────────────────────────────────────────


class TestPlanner:
    def test_loads(self, page: Page):
        body = go(page, "/planner")
        ok("planner_loads", len(body) > 50, f"len={len(body)}")
        assert len(body) > 50


# ── Models ────────────────────────────────────────────────────


class TestModels:
    def test_loads(self, page: Page):
        body = go(page, "/models")
        # Page might show "Connecting..." if API is down — that's still a valid page load
        has_content = len(body) > 50
        ok("models_loads", has_content, f"len={len(body)}")
        assert has_content


# ── Monitoring ────────────────────────────────────────────────


class TestMonitoring:
    def test_loads(self, page: Page):
        body = go(page, "/monitoring")
        has_metrics = any(w in body.lower() for w in ["cpu", "memory", "monitor", "health", "gpu"])
        ok("monitoring_loads", has_metrics, f"len={len(body)}")
        assert has_metrics


# ── Knowledge ─────────────────────────────────────────────────


class TestKnowledge:
    def test_loads(self, page: Page):
        body = go(page, "/knowledge")
        has_kw = any(w in body.lower() for w in ["knowledge", "memory", "fact", "search"])
        ok("knowledge_loads", has_kw, f"len={len(body)}")
        assert has_kw


# ── Datasets Import ───────────────────────────────────────────


class TestDatasetsImport:
    def test_loads(self, page: Page):
        body = go(page, "/datasets")
        ok("datasets_loads", len(body) > 50, f"len={len(body)}")
        assert len(body) > 50

    def test_import_button_exists(self, page: Page):
        go(page, "/datasets")
        time.sleep(1)
        btn = page.get_by_role("button", name="Add file").first
        ok("datasets_import_button", btn.count() > 0)
        assert btn.count() > 0

    def test_import_dialog_opens(self, page: Page):
        go(page, "/datasets")
        time.sleep(1)
        page.get_by_role("button", name="Add file").first.click(force=True)
        time.sleep(1)
        dialog = page.get_by_role("dialog")
        ok("datasets_import_dialog_opens", dialog.count() > 0)
        assert dialog.count() > 0

    def test_hf_radio_exists(self, page: Page):
        go(page, "/datasets")
        time.sleep(1)
        page.get_by_role("button", name="Add file").first.click(force=True)
        time.sleep(1)
        hf = page.locator('[role="dialog"] [role="radio"][aria-label^="HuggingFace:"]')
        ok("datasets_hf_source_exists", hf.count() > 0)
        assert hf.count() > 0
        page.keyboard.press("Escape")
        time.sleep(0.3)

    def test_hf_radio_clicks(self, page: Page):
        go(page, "/datasets")
        time.sleep(1)
        page.get_by_role("button", name="Add file").first.click(force=True)
        time.sleep(1)
        hf_radio = page.locator('[role="dialog"] [role="radio"][aria-label^="HuggingFace:"]').first
        hf_radio.focus()
        time.sleep(0.2)
        hf_radio.press("Space")
        time.sleep(1)
        inp = page.locator("input[placeholder='username/dataset-name']")
        ok("datasets_hf_radio_clicks", inp.count() > 0)
        assert inp.count() > 0
        page.keyboard.press("Escape")
        time.sleep(0.3)

    def test_hf_input_fills(self, page: Page):
        go(page, "/datasets")
        time.sleep(1)
        page.get_by_role("button", name="Add file").first.click(force=True)
        time.sleep(1)
        hf_radio = page.locator('[role="dialog"] [role="radio"][aria-label^="HuggingFace:"]').first
        hf_radio.focus()
        time.sleep(0.2)
        hf_radio.press("Space")
        time.sleep(1)
        inp = page.locator("input[placeholder='username/dataset-name']")
        inp.fill("heptapod/titanic")
        time.sleep(0.3)
        ok("datasets_hf_input_fills", inp.input_value() == "heptapod/titanic")
        assert inp.input_value() == "heptapod/titanic"
        page.keyboard.press("Escape")
        time.sleep(0.3)

    def test_hf_import_attempt(self, page: Page):
        # Reload to clear any stale state from prior tests
        page.goto(f"{BASE}/datasets", wait_until="load", timeout=20000)
        try:
            page.wait_for_function(
                "() => !document.body.innerText.includes('Connecting...')", timeout=10000
            )
        except Exception:
            pass
        time.sleep(2)
        page.get_by_role("button", name="Add file").first.click(force=True)
        time.sleep(2)
        hf_radio = page.locator('[role="dialog"] [role="radio"][aria-label^="HuggingFace:"]').first
        hf_radio.focus()
        time.sleep(0.2)
        hf_radio.press("Space")
        time.sleep(1)
        page.locator("input[placeholder='username/dataset-name']").fill("heptapod/titanic")
        time.sleep(0.5)
        # Try clicking Import, but don't fail if dialog blocks it
        try:
            page.get_by_role("dialog").get_by_role("button", name="Import", exact=True).click(force=True, timeout=3000)
        except Exception:
            pass
        # handleImport flips the button to "Importing..." synchronously — poll for it
        reacted = False
        try:
            page.get_by_role("dialog").get_by_text("Importing...", exact=False).first.wait_for(timeout=5000)
            reacted = True
        except Exception:
            pass
        body = page.inner_text("body")
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
    def test_redirect(self, page: Page, old: str, expected: str):
        body = go(page, old)
        ok(
            f"redirect_{old.replace('/', '_')}",
            len(body) > 50,
            f"{old} -> /{expected}, len={len(body)}",
        )
        assert len(body) > 50, f"Redirect {old} failed"


# ── Tools Pages — Interactive Flows ──────────────────────────


class TestToolsFlows:
    def test_brstorm_has_input_and_suggestions(self, page: Page):
        body = go(page, "/brainstorm")
        has_input = page.locator("textarea:visible").count() > 0
        has_suggestions = "Name ideas" in body or "Weekend" in body
        ok(
            "brainstorm_input_and_suggestions",
            has_input and has_suggestions,
            f"input={has_input}, suggestions={has_suggestions}",
        )
        assert has_input and has_suggestions

    def test_decide_has_two_options(self, page: Page):
        go(page, "/decide")
        inputs = page.locator("input:visible")
        ok("decide_has_options", inputs.count() >= 2, f"inputs={inputs.count()}")
        assert inputs.count() >= 2

    def test_explain_has_difficulty_buttons(self, page: Page):
        go(page, "/explain")
        page.get_by_text("Level", exact=True).first.wait_for(state="visible", timeout=10000)
        body = page.inner_text("body")
        has_simple = "Simple" in body
        has_moderate = "Moderate" in body
        ok(
            "explain_has_difficulty",
            has_simple and has_moderate,
            f"simple={has_simple}, moderate={has_moderate}",
        )
        assert has_simple and has_moderate

    def test_rewrite_has_action_buttons(self, page: Page):
        body = go(page, "/rewrite")
        has_grammar = "Fix Grammar" in body
        has_shorter = "Make Shorter" in body
        ok(
            "rewrite_has_actions",
            has_grammar and has_shorter,
            f"grammar={has_grammar}, shorter={has_shorter}",
        )
        assert has_grammar and has_shorter

    def test_translate_has_language_selector(self, page: Page):
        go(page, "/translate")
        page.get_by_text("EN→ES").first.wait_for(state="visible", timeout=10000)
        body = page.inner_text("body")
        has_pair = "EN→ES" in body
        has_translate_btn = "Translate" in body
        ok(
            "translate_has_selector",
            has_pair and has_translate_btn,
            f"pair={has_pair}, btn={has_translate_btn}",
        )
        assert has_pair and has_translate_btn

    def test_wellness_has_options(self, page: Page):
        body = go(page, "/wellness")
        has_sleep = "Sleep" in body
        has_meditate = "Meditat" in body
        ok(
            "wellness_has_options",
            has_sleep and has_meditate,
            f"sleep={has_sleep}, meditate={has_meditate}",
        )
        assert has_sleep and has_meditate

    def test_writing_has_tones_and_types(self, page: Page):
        body = go(page, "/writing")
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
    def test_dashboard_loads(self, page: Page):
        body = go(page, "/consciousness/dashboard")
        ok("consciousness_dashboard_loads", len(body) > 50, f"len={len(body)}")
        assert len(body) > 50

    def test_health_loads(self, page: Page):
        body = go(page, "/consciousness/health")
        ok("consciousness_health_loads", len(body) > 50, f"len={len(body)}")
        assert len(body) > 50

    def test_playground_loads(self, page: Page):
        body = go(page, "/consciousness/playground")
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
    def test_route_loads(self, page: Page, route: str):
        body = go(page, route)
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
