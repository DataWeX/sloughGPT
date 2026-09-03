"""
User Journey Tests — end-to-end browser tests using Playwright.

Tests the main user flows by navigating the web UI, clicking elements,
filling forms, and verifying expected behavior.

Usage:
    .venv/bin/python -m pytest tests/test_user_journeys.py -x -v

Requirements:
    .venv/bin/playwright install chromium
"""
import time
import json
import pytest
from pathlib import Path
from playwright.sync_api import sync_playwright, Page, expect

BASE_URL = "http://localhost:3000"
RESULTS = []


def record(test_name: str, passed: bool, detail: str = ""):
    RESULTS.append({"test": test_name, "passed": passed, "detail": detail})


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        yield browser
        browser.close()


@pytest.fixture(scope="module")
def page(browser):
    ctx = browser.new_context(viewport={"width": 1280, "height": 720})
    pg = ctx.new_page()
    yield pg
    ctx.close()


# ═══════════════════════════════════════════════════════════════
# Journey 1: Dashboard
# ═══════════════════════════════════════════════════════════════

class TestDashboard:
    def test_dashboard_loads(self, page: Page):
        page.goto(BASE_URL, wait_until="networkidle")
        assert page.title(), "Page should have a title"
        # Dashboard should have some visible content
        body = page.inner_text("body")
        assert len(body) > 50, f"Dashboard body too short: {len(body)} chars"
        record("dashboard_loads", True, f"title={page.title()}")

    def test_dashboard_has_nav(self, page: Page):
        page.goto(BASE_URL, wait_until="networkidle")
        # Sidebar or nav should exist
        nav = page.query_selector("nav, [role='navigation'], aside")
        has_nav = nav is not None
        if not has_nav:
            # Fallback: check if any links exist
            links = page.query_selector_all("a")
            has_nav = len(links) > 3
        record("dashboard_has_nav", has_nav)
        assert has_nav, "Dashboard should have navigation"


# ═══════════════════════════════════════════════════════════════
# Journey 2: Navigation — all sidebar routes load
# ═══════════════════════════════════════════════════════════════

NAV_ROUTES = [
    ("/chat", "Chat"),
    ("/training", "Training"),
    ("/datasets", "Datasets"),
    ("/models", "Models"),
    ("/agents", "Agents"),
    ("/souls", "Souls"),
    ("/knowledge", "Knowledge"),
    ("/monitoring", "Monitoring"),
    ("/settings", "Settings"),
    ("/planner", "Planner"),
    ("/benchmark", "Benchmark"),
    ("/tokenizer", "Tokenizer"),
    ("/errors", "Errors"),
    ("/security", "Security"),
    ("/shell", "Shell"),
    ("/feedback", "Feedback"),
    ("/files", "Files"),
]


class TestNavigation:
    @pytest.mark.parametrize("route,name", NAV_ROUTES)
    def test_route_loads(self, page: Page, route: str, name: str):
        resp = page.goto(f"{BASE_URL}{route}", wait_until="networkidle")
        time.sleep(0.5)
        body = page.inner_text("body")
        has_content = len(body) > 50
        is_error = resp.status >= 400 or "404" in body[:200] or "not found" in body[:200].lower()
        record(f"nav_{name.lower()}", has_content and not is_error,
               f"route={route}, status={resp.status}, len={len(body)}")
        assert has_content and not is_error, f"Route {route} failed: status={resp.status}"


# ═══════════════════════════════════════════════════════════════
# Journey 3: Chat — page loads, input exists
# ═══════════════════════════════════════════════════════════════

class TestChat:
    def test_chat_page_loads(self, page: Page):
        page.goto(f"{BASE_URL}/chat", wait_until="networkidle")
        body = page.inner_text("body")
        assert "chat" in body.lower() or len(body) > 100, "Chat page should have content"
        record("chat_page_loads", True, f"len={len(body)}")

    def test_chat_has_input(self, page: Page):
        page.goto(f"{BASE_URL}/chat", wait_until="networkidle")
        # Look for input/textarea
        inputs = page.query_selector_all("input, textarea")
        has_input = len(inputs) > 0
        record("chat_has_input", has_input, f"inputs_found={len(inputs)}")
        assert has_input, "Chat page should have an input field"

    def test_chat_type_message(self, page: Page):
        page.goto(f"{BASE_URL}/chat", wait_until="networkidle")
        inputs = page.query_selector_all("input, textarea")
        if inputs:
            inputs[0].fill("Hello, this is a test message")
            val = inputs[0].input_value()
            record("chat_type_message", "test message" in val, f"value={val[:50]}")
            assert "test message" in val, "Typed text should appear in input"
        else:
            record("chat_type_message", False, "no input found")
            pytest.skip("No chat input found")


# ═══════════════════════════════════════════════════════════════
# Journey 4: Training page
# ═══════════════════════════════════════════════════════════════

class TestTraining:
    def test_training_page_loads(self, page: Page):
        page.goto(f"{BASE_URL}/training", wait_until="networkidle")
        body = page.inner_text("body")
        assert "train" in body.lower(), "Training page should mention training"
        record("training_page_loads", True, f"len={len(body)}")


# ═══════════════════════════════════════════════════════════════
# Journey 5: Settings page
# ═══════════════════════════════════════════════════════════════

class TestSettings:
    def test_settings_page_loads(self, page: Page):
        page.goto(f"{BASE_URL}/settings", wait_until="networkidle")
        body = page.inner_text("body")
        has_settings = "setting" in body.lower() or "config" in body.lower() or len(body) > 100
        record("settings_page_loads", has_settings, f"len={len(body)}")
        assert has_settings, "Settings page should load"


# ═══════════════════════════════════════════════════════════════
# Journey 6: Planner
# ═══════════════════════════════════════════════════════════════

class TestPlanner:
    def test_planner_page_loads(self, page: Page):
        page.goto(f"{BASE_URL}/planner", wait_until="networkidle")
        body = page.inner_text("body")
        has_planner = "planner" in body.lower() or "board" in body.lower() or "card" in body.lower() or len(body) > 100
        record("planner_page_loads", has_planner, f"len={len(body)}")
        assert has_planner, "Planner page should load"


# ═══════════════════════════════════════════════════════════════
# Journey 7: Models
# ═══════════════════════════════════════════════════════════════

class TestModels:
    def test_models_page_loads(self, page: Page):
        page.goto(f"{BASE_URL}/models", wait_until="networkidle")
        body = page.inner_text("body")
        assert "model" in body.lower(), "Models page should mention models"
        record("models_page_loads", True, f"len={len(body)}")


# ═══════════════════════════════════════════════════════════════
# Journey 8: Monitoring
# ═══════════════════════════════════════════════════════════════

class TestMonitoring:
    def test_monitoring_page_loads(self, page: Page):
        page.goto(f"{BASE_URL}/monitoring", wait_until="networkidle")
        body = page.inner_text("body")
        has_metrics = any(w in body.lower() for w in ["cpu", "memory", "monitor", "health", "gpu"])
        record("monitoring_page_loads", has_metrics, f"len={len(body)}")
        assert has_metrics, "Monitoring page should show system metrics"


# ═══════════════════════════════════════════════════════════════
# Journey 9: Knowledge
# ═══════════════════════════════════════════════════════════════

class TestKnowledge:
    def test_knowledge_page_loads(self, page: Page):
        page.goto(f"{BASE_URL}/knowledge", wait_until="networkidle")
        body = page.inner_text("body")
        has_knowledge = any(w in body.lower() for w in ["knowledge", "memory", "fact", "search"])
        record("knowledge_page_loads", has_knowledge, f"len={len(body)}")
        assert has_knowledge, "Knowledge page should load"


# ═══════════════════════════════════════════════════════════════
# Journey 10: Legacy redirects
# ═══════════════════════════════════════════════════════════════

REDIRECT_ROUTES = [
    ("/companion", "/souls"),
    ("/evaluate", "/benchmark"),
    ("/memory", "/knowledge"),
    ("/collections", "/datasets"),
    ("/self-train", "/training"),
    ("/admin", "/settings"),
    ("/images", "/files"),
    ("/session", "/shell"),
]


class TestRedirects:
    @pytest.mark.parametrize("old_route,expected", REDIRECT_ROUTES)
    def test_legacy_redirect(self, page: Page, old_route: str, expected: str):
        page.goto(f"{BASE_URL}{old_route}", wait_until="networkidle")
        time.sleep(0.5)
        body = page.inner_text("body")
        has_content = len(body) > 50
        record(f"redirect_{old_route.replace('/', '_')}", has_content,
               f"{old_route} -> {expected}, content={has_content}")
        assert has_content, f"Redirect {old_route} -> {expected} should load content"


# ═══════════════════════════════════════════════════════════════
# Results summary
# ═══════════════════════════════════════════════════════════════

@pytest.fixture(scope="session", autouse=True)
def save_results():
    yield
    results_path = Path(__file__).parent / "test_results" / "user_journey_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(RESULTS, indent=2))

    total = len(RESULTS)
    passed = sum(1 for r in RESULTS if r["passed"])
    failed = total - passed
    print(f"\n{'='*60}")
    print(f"User Journey Results: {passed}/{total} passed, {failed} failed")
    print(f"{'='*60}")
    if failed:
        for r in RESULTS:
            if not r["passed"]:
                print(f"  err {r['test']}: {r['detail']}")
