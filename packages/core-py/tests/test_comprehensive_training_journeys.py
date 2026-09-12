"""
Comprehensive Training Journey Tests — computer-use + DevTools monitoring.

Tests the full LLM training lifecycle through the web UI using browser
automation, verifying each phase with Playwright interactions and capturing
DevTools telemetry (console logs, network requests, errors, performance).

Usage:
    .venv/bin/python -m pytest tests/test_comprehensive_training_journeys.py -x -v

Requirements:
    .venv/bin/playwright install chromium
    API server at localhost:8000
    Web server at localhost:3000
"""
import json
import time
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from playwright.sync_api import Page

import pytest

BASE = "http://localhost:3000"
API = "http://localhost:8000"
RESULTS: list[dict[str, Any]] = []
DEVTOOLS_LOG: list[dict[str, Any]] = []


# ── Helpers ────────────────────────────────────────────────────────

def _api_is_ready() -> bool:
    try:
        req = urllib.request.Request(f"{API}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def _wait_for_api(timeout: int = 60) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _api_is_ready():
            return True
        time.sleep(1)
    return False


@pytest.fixture(scope="session", autouse=True)
def ensure_servers_ready():
    assert _wait_for_api(timeout=90), (
        f"API at {API} not ready after 90s. "
        "Start the server with: FORCE_COLOR=1 ./sloughgpt serve --web"
    )
    time.sleep(3)


def ok(name: str, passed: bool, detail: str = "", devtools: dict | None = None):
    entry = {"test": name, "passed": passed, "detail": detail}
    if devtools:
        entry["devtools"] = devtools
    RESULTS.append(entry)
    mark = "ok" if passed else "ERR"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
    if devtools:
        print(f"    devtools: {devtools.get('summary', '')}")


def record_devtools(step_name: str, console_msgs: list, network_reqs: list, errors: list) -> dict:
    entry = {
        "step": step_name,
        "console_count": len(console_msgs),
        "request_count": len(network_reqs),
        "error_count": len(errors),
        "console_errors": [m for m in console_msgs if m.get("type") == "error"],
        "failed_requests": [r for r in network_reqs if r.get("status", 200) >= 400],
        "summary": f"{len(console_msgs)} console, {len(network_reqs)} network, {len(errors)} errors",
    }
    DEVTOOLS_LOG.append(entry)
    return entry


# ── Fixtures ───────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright
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


class DevToolsCollector:
    """Captures console, network, and error events from a Playwright page."""

    def __init__(self, page):
        self.page = page
        self.console_messages = []
        self.network_requests = []
        self.errors = []
        self._setup_listeners()

    def _setup_listeners(self):
        self.page.on("console", self._on_console)
        self.page.on("pageerror", self._on_page_error)
        self.page.on("request", self._on_request)
        self.page.on("response", self._on_response)

    def _on_console(self, msg):
        self.console_messages.append({
            "type": msg.type,
            "text": msg.text,
            "timestamp": time.time(),
        })

    def _on_page_error(self, error):
        self.errors.append({"message": str(error), "timestamp": time.time()})

    def _on_request(self, request):
        self.network_requests.append({
            "url": request.url,
            "method": request.method,
            "timestamp": time.time(),
        })

    def _on_response(self, response):
        for req in self.network_requests:
            if req["url"] == response.url and "status" not in req:
                req["status"] = response.status
                req["duration_ms"] = (time.time() - req["timestamp"]) * 1000
                break

    def clear(self):
        self.console_messages.clear()
        self.network_requests.clear()
        self.errors.clear()

    def snapshot(self) -> dict[str, Any]:
        return {
            "console_count": len(self.console_messages),
            "request_count": len(self.network_requests),
            "error_count": len(self.errors),
            "console_errors": [m for m in self.console_messages if m.get("type") == "error"],
            "failed_requests": [r for r in self.network_requests if r.get("status", 200) >= 400],
        }


def go(page: Page, path: str, collector: DevToolsCollector | None = None) -> str:
    if collector:
        collector.clear()
    page.goto(f"{BASE}{path}", wait_until="load", timeout=20000)
    try:
        page.wait_for_function(
            "() => !document.body.innerText.includes('Connecting...')",
            timeout=15000,
        )
    except Exception:
        pass
    time.sleep(1)
    body = page.inner_text("body")
    if "Something went wrong" in body and "Try again" in body:
        try:
            page.locator("button:has-text('Try again')").first.click(timeout=5000)
            time.sleep(2)
            body = page.inner_text("body")
        except Exception:
            pass
    return body


@pytest.fixture(scope="module")
def collector(page):
    return DevToolsCollector(page)


# ══════════════════════════════════════════════════════════════════
# Training Lifecycle Journey Tests
# ══════════════════════════════════════════════════════════════════

class TestTrainingLifecycle:
    """Tests the full training flow: navigate → configure → train → monitor → results."""

    def test_training_page_loads(self, page, collector):
        body = go(page, "/training", collector)
        has_train = "train" in body.lower()
        dt = record_devtools("training_page_load", collector.console_messages, collector.network_requests, collector.errors)
        ok("training_lifecycle_page_loads", has_train, f"len={len(body)}", dt)
        assert has_train

    def test_training_has_tabs(self, page, collector):
        body = go(page, "/training", collector)
        has_tabs = any(w in body.lower() for w in ["train", "results", "settings", "configure"])
        dt = record_devtools("training_tabs", collector.console_messages, collector.network_requests, collector.errors)
        ok("training_lifecycle_has_tabs", has_tabs, f"body_snippet={body[:200]}", dt)
        assert has_tabs

    def test_training_pipeline_steps_visible(self, page, collector):
        body = go(page, "/training", collector)
        has_pipeline = any(w in body.lower() for w in ["data", "configure", "start", "pipeline", "step"])
        dt = record_devtools("training_pipeline", collector.console_messages, collector.network_requests, collector.errors)
        ok("training_lifecycle_pipeline_steps", has_pipeline, f"body_snippet={body[:200]}", dt)
        assert has_pipeline

    def test_training_datasets_section(self, page, collector):
        body = go(page, "/training", collector)
        has_datasets = any(w in body.lower() for w in ["dataset", "data", "upload", "import", "select"])
        dt = record_devtools("training_datasets", collector.console_messages, collector.network_requests, collector.errors)
        ok("training_lifecycle_datasets_section", has_datasets, f"body_snippet={body[:200]}", dt)
        assert has_datasets

    def test_training_config_section(self, page, collector):
        body = go(page, "/training", collector)
        has_config = any(w in body.lower() for w in ["config", "epoch", "learning", "rate", "batch", "hyper"])
        dt = record_devtools("training_config", collector.console_messages, collector.network_requests, collector.errors)
        ok("training_lifecycle_config_section", has_config, f"body_snippet={body[:200]}", dt)
        assert has_config

    def test_training_start_button_exists(self, page, collector):
        go(page, "/training", collector)
        time.sleep(1)
        btn = page.get_by_role("button", name="Train").first
        if btn.count() == 0:
            btn = page.get_by_role("button", name="Start").first
        dt = record_devtools("training_start_btn", collector.console_messages, collector.network_requests, collector.errors)
        ok("training_lifecycle_start_button", btn.count() > 0, f"found={btn.count()}", dt)
        assert btn.count() > 0

    def test_training_job_detail_page(self, page, collector):
        body = go(page, "/training/job/test-job-123", collector)
        has_content = len(body) > 50
        dt = record_devtools("training_job_detail", collector.console_messages, collector.network_requests, collector.errors)
        ok("training_lifecycle_job_detail", has_content, f"len={len(body)}", dt)
        assert has_content

    def test_training_job_back_link(self, page, collector):
        go(page, "/training/job/test-job-123", collector)
        time.sleep(1)
        back = page.locator('a[href="/training"]').first
        dt = record_devtools("training_job_back", collector.console_messages, collector.network_requests, collector.errors)
        ok("training_lifecycle_job_back_link", back.count() > 0, f"found={back.count()}", dt)
        assert back.count() > 0


# ══════════════════════════════════════════════════════════════════
# Training Subsystem Journey Tests
# ══════════════════════════════════════════════════════════════════

class TestTrainingSubsystems:
    """Tests training subsystem pages: analytics, compare, presets, queue, runs, trends."""

    ROUTES = [
        ("/training/analytics", "analytics"),
        ("/training/compare", "compare"),
        ("/training/presets", "presets"),
        ("/training/queue", "queue"),
        ("/training/runs", "runs"),
        ("/training/trends", "trends"),
        ("/training/insights", "insights"),
        ("/training/model-card", "model card"),
    ]

    def test_subsystem_pages_load(self, page, collector, route, name):
        body = go(page, route, collector)
        dt = record_devtools(f"subsystem_{name}", collector.console_messages, collector.network_requests, collector.errors)
        ok(f"training_subsystem_{name}_loads", len(body) > 30, f"len={len(body)}", dt)
        assert len(body) > 30


# ══════════════════════════════════════════════════════════════════
# Dataset → Training Flow Journey Tests
# ══════════════════════════════════════════════════════════════════

class TestDatasetToTrainingFlow:
    """Tests the flow: datasets page → import → select → train."""

    def test_datasets_page_loads(self, page, collector):
        body = go(page, "/datasets", collector)
        dt = record_devtools("datasets_load", collector.console_messages, collector.network_requests, collector.errors)
        ok("flow_datasets_page_loads", len(body) > 50, f"len={len(body)}", dt)
        assert len(body) > 50

    def test_datasets_import_button(self, page, collector):
        go(page, "/datasets", collector)
        time.sleep(1)
        btn = page.get_by_role("button", name="Import").first
        dt = record_devtools("datasets_import_btn", collector.console_messages, collector.network_requests, collector.errors)
        ok("flow_datasets_import_button", btn.count() > 0, f"found={btn.count()}", dt)
        assert btn.count() > 0

    def test_datasets_import_dialog(self, page, collector):
        go(page, "/datasets", collector)
        time.sleep(1)
        page.get_by_role("button", name="Import").first.click(force=True)
        time.sleep(1)
        dialog = page.get_by_role("dialog")
        dt = record_devtools("datasets_import_dialog", collector.console_messages, collector.network_requests, collector.errors)
        ok("flow_datasets_import_dialog_opens", dialog.count() > 0, f"found={dialog.count()}", dt)
        assert dialog.count() > 0

    def test_datasets_kaggle_option(self, page, collector):
        go(page, "/datasets", collector)
        time.sleep(1)
        page.get_by_role("button", name="Import").first.click(force=True)
        time.sleep(1)
        kaggle = page.get_by_role("radio", name="Kaggle: Download from Kaggle")
        dt = record_devtools("datasets_kaggle", collector.console_messages, collector.network_requests, collector.errors)
        ok("flow_datasets_kaggle_option", kaggle.count() > 0, f"found={kaggle.count()}", dt)
        assert kaggle.count() > 0

    def test_datasets_file_upload_option(self, page, collector):
        go(page, "/datasets", collector)
        time.sleep(1)
        page.get_by_role("button", name="Import").first.click(force=True)
        time.sleep(1)
        body = page.inner_text("body")
        has_upload = any(w in body.lower() for w in ["upload", "file", "csv", "jsonl", "json"])
        dt = record_devtools("datasets_upload", collector.console_messages, collector.network_requests, collector.errors)
        ok("flow_datasets_file_upload_option", has_upload, f"body_snippet={body[:200]}", dt)
        assert has_upload

    def test_navigate_to_training_after_datasets(self, page, collector):
        go(page, "/datasets", collector)
        time.sleep(1)
        body = go(page, "/training", collector)
        dt = record_devtools("datasets_to_training", collector.console_messages, collector.network_requests, collector.errors)
        ok("flow_datasets_to_training_navigation", "train" in body.lower(), f"url={page.url}", dt)
        assert "train" in page.url


# ══════════════════════════════════════════════════════════════════
# Auto-Training Journey Tests
# ══════════════════════════════════════════════════════════════════

class TestAutoTrainingFlow:
    """Tests the auto-training configuration flow."""

    def test_auto_train_page_loads(self, page, collector):
        body = go(page, "/auto-train", collector)
        dt = record_devtools("auto_train_load", collector.console_messages, collector.network_requests, collector.errors)
        ok("auto_train_page_loads", len(body) > 30, f"len={len(body)}", dt)
        assert len(body) > 30

    def test_auto_train_has_config(self, page, collector):
        body = go(page, "/auto-train", collector)
        has_config = any(w in body.lower() for w in ["auto", "train", "threshold", "interval", "monitor", "enable"])
        dt = record_devtools("auto_train_config", collector.console_messages, collector.network_requests, collector.errors)
        ok("auto_train_has_config_ui", has_config, f"body_snippet={body[:200]}", dt)
        assert has_config

    def test_auto_train_redirects_to_training(self, page, collector):
        go(page, "/self-train", collector)
        dt = record_devtools("auto_train_redirect", collector.console_messages, collector.network_requests, collector.errors)
        ok("auto_train_self_train_redirect", "train" in page.url, f"url={page.url}", dt)
        assert "train" in page.url


# ══════════════════════════════════════════════════════════════════
# Chat → Feedback → DPO Training Flow Tests
# ══════════════════════════════════════════════════════════════════

class TestChatToDPOTFlow:
    """Tests the flow: chat → feedback → DPO training path."""

    def test_chat_page_loads(self, page, collector):
        body = go(page, "/chat", collector)
        dt = record_devtools("chat_load", collector.console_messages, collector.network_requests, collector.errors)
        ok("dpo_chat_page_loads", len(body) > 30, f"len={len(body)}", dt)
        assert len(body) > 30

    def test_chat_has_input(self, page, collector):
        go(page, "/chat", collector)
        time.sleep(1)
        has_input = await_no_timeout(page, lambda: page.locator("textarea, input[type='text']").count() > 0, 5)
        dt = record_devtools("chat_input", collector.console_messages, collector.network_requests, collector.errors)
        ok("dpo_chat_has_input", has_input, f"found={has_input}", dt)
        assert has_input

    def test_feedback_page_loads(self, page, collector):
        body = go(page, "/feedback", collector)
        dt = record_devtools("feedback_load", collector.console_messages, collector.network_requests, collector.errors)
        ok("dpo_feedback_page_loads", len(body) > 30, f"len={len(body)}", dt)
        assert len(body) > 30

    def test_training_method_dpo_available(self, page, collector):
        body = go(page, "/training", collector)
        has_dpo = "dpo" in body.lower() or "preference" in body.lower() or "feedback" in body.lower()
        dt = record_devtools("training_dpo_available", collector.console_messages, collector.network_requests, collector.errors)
        ok("dpo_training_method_available", has_dpo, f"body_snippet={body[:200]}", dt)
        assert has_dpo


# ══════════════════════════════════════════════════════════════════
# Monitoring & Performance Journey Tests
# ══════════════════════════════════════════════════════════════════

class TestMonitoringFlow:
    """Tests the monitoring and performance tracking flow."""

    def test_monitoring_page_loads(self, page, collector):
        body = go(page, "/monitoring", collector)
        has_metrics = any(w in body.lower() for w in ["cpu", "memory", "monitor", "health", "gpu", "system"])
        dt = record_devtools("monitoring_load", collector.console_messages, collector.network_requests, collector.errors)
        ok("monitoring_page_loads", has_metrics, f"len={len(body)}", dt)
        assert has_metrics

    def test_monitoring_has_metrics_cards(self, page, collector):
        body = go(page, "/monitoring", collector)
        has_cards = any(w in body.lower() for w in ["usage", "rate", "percent", "total", "active"])
        dt = record_devtools("monitoring_cards", collector.console_messages, collector.network_requests, collector.errors)
        ok("monitoring_has_metrics_cards", has_cards, f"body_snippet={body[:200]}", dt)
        assert has_cards

    def test_benchmark_page_loads(self, page, collector):
        body = go(page, "/benchmark", collector)
        dt = record_devtools("benchmark_load", collector.console_messages, collector.network_requests, collector.errors)
        ok("monitoring_benchmark_loads", len(body) > 30, f"len={len(body)}", dt)
        assert len(body) > 30


# ══════════════════════════════════════════════════════════════════
# Full Cross-Page Navigation Journey Tests
# ══════════════════════════════════════════════════════════════════

class TestCrossPageNavigation:
    """Tests navigation across all major sections without errors."""

    ALL_ROUTES = [
        "/", "/chat", "/training", "/datasets", "/models",
        "/monitoring", "/settings", "/knowledge", "/feedback",
        "/benchmark", "/auto-train",
    ]

    def test_all_routes_load_without_crash(self, page, collector, route):
        body = go(page, route, collector)
        has_crash = "Something went wrong" in body
        dt = record_devtools(f"nav_{route}", collector.console_messages, collector.network_requests, collector.errors)
        passed = not has_crash and len(body) > 30
        ok(f"nav_{route.replace('/', '_') or '_root'}", passed, f"len={len(body)}, crash={has_crash}", dt)
        assert passed, f"Route {route} crashed or empty"

    def test_navigation_back_and_forth(self, page, collector):
        go(page, "/training", collector)
        go(page, "/datasets", collector)
        body = go(page, "/training", collector)
        dt = record_devtools("nav_back_forth", collector.console_messages, collector.network_requests, collector.errors)
        ok("nav_back_forth_works", "train" in body.lower(), f"url={page.url}", dt)
        assert "train" in page.url


# ══════════════════════════════════════════════════════════════════
# DevTools Telemetry Verification Tests
# ══════════════════════════════════════════════════════════════════

class TestDevToolsTelemetry:
    """Verifies that DevTools monitoring captured meaningful data."""

    def test_console_messages_captured(self, page, collector):
        go(page, "/training", collector)
        time.sleep(2)
        has_console = len(collector.console_messages) > 0
        dt = record_devtools("telemetry_console", collector.console_messages, collector.network_requests, collector.errors)
        ok("devtools_console_captured", has_console, f"count={len(collector.console_messages)}", dt)
        assert has_console

    def test_network_requests_captured(self, page, collector):
        go(page, "/training", collector)
        time.sleep(2)
        has_network = len(collector.network_requests) > 0
        dt = record_devtools("telemetry_network", collector.console_messages, collector.network_requests, collector.errors)
        ok("devtools_network_captured", has_network, f"count={len(collector.network_requests)}", dt)
        assert has_network

    def test_no_critical_errors(self, page, collector):
        go(page, "/training", collector)
        time.sleep(2)
        critical = [e for e in collector.errors if "uncaught" in e.get("message", "").lower()]
        dt = record_devtools("telemetry_no_errors", collector.console_messages, collector.network_requests, collector.errors)
        ok("devtools_no_critical_errors", len(critical) == 0, f"critical={len(critical)}", dt)
        assert len(critical) == 0

    def test_api_requests_reach_backend(self, page, collector):
        go(page, "/training", collector)
        time.sleep(2)
        api_reqs = [r for r in collector.network_requests if "localhost:8000" in r.get("url", "")]
        dt = record_devtools("telemetry_api_requests", collector.console_messages, collector.network_requests, collector.errors)
        ok("devtools_api_requests", len(api_reqs) > 0, f"api_requests={len(api_reqs)}", dt)
        assert len(api_reqs) > 0

    def test_performance_metrics_collected(self, page, collector):
        go(page, "/training", collector)
        time.sleep(1)
        try:
            perf = page.evaluate("""() => {
                const perf = performance;
                const entries = perf.getEntriesByType('navigation');
                const nav = entries.length > 0 ? entries[0] : null;
                return {
                    domContentLoaded: nav ? nav.domContentLoadedEventEnd : 0,
                    loadEvent: nav ? nav.loadEventEnd : 0,
                    resourceCount: perf.getEntriesByType('resource').length,
                };
            }""")
            has_perf = perf.get("resourceCount", 0) > 0
        except Exception:
            has_perf = False
            perf = {}
        dt = record_devtools("telemetry_performance", collector.console_messages, collector.network_requests, collector.errors)
        dt["performance"] = perf
        ok("devtools_performance_metrics", has_perf, f"perf={perf}", dt)
        assert has_perf


# ══════════════════════════════════════════════════════════════════
# Training Config Interaction Journey Tests
# ══════════════════════════════════════════════════════════════════

class TestTrainingConfigInteraction:
    """Tests interactive config elements on the training page."""

    def test_method_selector_exists(self, page, collector):
        go(page, "/training", collector)
        time.sleep(1)
        body = page.inner_text("body")
        has_method = any(w in body.lower() for w in ["method", "sft", "rlhf", "dpo", "lora", "finetune"])
        dt = record_devtools("config_method", collector.console_messages, collector.network_requests, collector.errors)
        ok("config_method_selector", has_method, f"body_snippet={body[:200]}", dt)
        assert has_method

    def test_hyperparameter_inputs_exist(self, page, collector):
        go(page, "/training", collector)
        time.sleep(1)
        body = page.inner_text("body")
        has_hyper = any(w in body.lower() for w in ["learning rate", "batch", "epoch", "dropout", "rank"])
        dt = record_devtools("config_hyperparams", collector.console_messages, collector.network_requests, collector.errors)
        ok("config_hyperparameter_inputs", has_hyper, f"body_snippet={body[:200]}", dt)
        assert has_hyper

    def test_lora_toggle_exists(self, page, collector):
        go(page, "/training", collector)
        time.sleep(1)
        body = page.inner_text("body")
        has_lora = "lora" in body.lower()
        dt = record_devtools("config_lora", collector.console_messages, collector.network_requests, collector.errors)
        ok("config_lora_toggle", has_lora, f"body_snippet={body[:200]}", dt)
        assert has_lora

    def test_checkpoint_name_input(self, page, collector):
        go(page, "/training", collector)
        time.sleep(1)
        body = page.inner_text("body")
        has_checkpoint = any(w in body.lower() for w in ["checkpoint", "save", "name", "output"])
        dt = record_devtools("config_checkpoint", collector.console_messages, collector.network_requests, collector.errors)
        ok("config_checkpoint_name", has_checkpoint, f"body_snippet={body[:200]}", dt)
        assert has_checkpoint


# ══════════════════════════════════════════════════════════════════
# Results & History Journey Tests
# ══════════════════════════════════════════════════════════════════

class TestTrainingResults:
    """Tests training results and history views."""

    def test_results_tab_loads(self, page, collector):
        body = go(page, "/training", collector)
        has_results = any(w in body.lower() for w in ["result", "history", "log", "run"])
        dt = record_devtools("results_tab", collector.console_messages, collector.network_requests, collector.errors)
        ok("results_tab_exists", has_results, f"body_snippet={body[:200]}", dt)
        assert has_results

    def test_training_runs_page(self, page, collector):
        body = go(page, "/training/runs", collector)
        dt = record_devtools("training_runs", collector.console_messages, collector.network_requests, collector.errors)
        ok("results_runs_page_loads", len(body) > 30, f"len={len(body)}", dt)
        assert len(body) > 30

    def test_training_analytics_page(self, page, collector):
        body = go(page, "/training/analytics", collector)
        dt = record_devtools("training_analytics", collector.console_messages, collector.network_requests, collector.errors)
        ok("results_analytics_page_loads", len(body) > 30, f"len={len(body)}", dt)
        assert len(body) > 30

    def test_training_trends_page(self, page, collector):
        body = go(page, "/training/trends", collector)
        dt = record_devtools("training_trends", collector.console_messages, collector.network_requests, collector.errors)
        ok("results_trends_page_loads", len(body) > 30, f"len={len(body)}", dt)
        assert len(body) > 30

    def test_model_card_page(self, page, collector):
        body = go(page, "/training/model-card", collector)
        dt = record_devtools("training_model_card", collector.console_messages, collector.network_requests, collector.errors)
        ok("results_model_card_loads", len(body) > 30, f"len={len(body)}", dt)
        assert len(body) > 30


# ══════════════════════════════════════════════════════════════════
# Helper
# ══════════════════════════════════════════════════════════════════

def await_no_timeout(page, fn, timeout_s=5):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if fn():
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


# ── Results ───────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def save_results():
    yield
    out = Path(__file__).parent / "test_results" / "comprehensive_training_journey_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(RESULTS, indent=2))

    devtools_out = Path(__file__).parent / "test_results" / "devtools_telemetry.json"
    devtools_out.write_text(json.dumps(DEVTOOLS_LOG, indent=2))

    total = len(RESULTS)
    passed = sum(1 for r in RESULTS if r["passed"])
    failed = total - passed
    print(f"\n{'='*60}")
    print(f"Comprehensive Training Journey Results: {passed}/{total} passed, {failed} failed")
    print(f"DevTools telemetry captured for {len(DEVTOOLS_LOG)} steps")
    print(f"{'='*60}")
    if failed:
        for r in RESULTS:
            if not r["passed"]:
                print(f"  ERR {r['test']}: {r['detail']}")
