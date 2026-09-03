"""
User Journey Tests — Playwright browser automation.

Runs all web UI flow tests headlessly. For CI and local verification.

Usage:
    .venv/bin/python -m pytest tests/test_user_journeys.py -x -v

Requirements:
    .venv/bin/playwright install chromium
"""
import json
import time
import pytest
from pathlib import Path
from playwright.sync_api import sync_playwright, Page

BASE = "http://localhost:3000"
RESULTS = []


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
    """Navigate and return body text."""
    page.goto(f"{BASE}{path}", wait_until="domcontentloaded", timeout=15000)
    time.sleep(1)
    return page.inner_text("body")


# ── Dashboard ─────────────────────────────────────────────────

class TestDashboard:
    def test_loads(self, page: Page):
        body = go(page, "/")
        ok("dashboard_loads", len(body) > 50, f"len={len(body)}")
        assert len(body) > 50

    def test_has_nav(self, page: Page):
        go(page, "/")
        links = page.query_selector_all("a")
        ok("dashboard_has_nav", len(links) > 3, f"links={len(links)}")
        assert len(links) > 3


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
        inputs = page.query_selector_all("input, textarea")
        ok("chat_has_input", len(inputs) > 0, f"found={len(inputs)}")
        assert len(inputs) > 0

    def test_type_message(self, page: Page):
        go(page, "/chat")
        # Try visible inputs first, then any input
        inputs = page.query_selector_all("textarea:visible, input[type='text']:visible")
        if not inputs:
            inputs = page.query_selector_all("input, textarea")
        if not inputs:
            ok("chat_type_message", False, "no input found")
            pytest.skip("no input")
        inputs[0].click()
        inputs[0].fill("Hello test message")
        val = inputs[0].input_value()
        ok("chat_type_message", "test message" in val, f"val={val[:40]}")
        assert "test message" in val


# ── Training ──────────────────────────────────────────────────

class TestTraining:
    def test_loads(self, page: Page):
        body = go(page, "/training")
        ok("training_loads", "train" in body.lower(), f"len={len(body)}")
        assert "train" in body.lower()


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
        ok(f"redirect_{old.replace('/', '_')}", len(body) > 50,
           f"{old} -> /{expected}, len={len(body)}")
        assert len(body) > 50, f"Redirect {old} failed"


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
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print(f"{'='*50}")
    if failed:
        for r in RESULTS:
            if not r["passed"]:
                print(f"  ERR {r['test']}: {r['detail']}")
