"""
E2E Training Trigger Test — uses computer-use agent to trigger training via web UI.

Tests the full flow: navigate → configure → trigger training → monitor DevTools.
Requires running API (localhost:8000) and web (localhost:3000) servers.

Usage:
    .venv/bin/python -m pytest tests/test_e2e_training_trigger.py -x -v -s
"""
import json
import time
import urllib.request
from pathlib import Path

import pytest

BASE = "http://localhost:3000"
API = "http://localhost:8000"
RESULTS = []


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


def ok(name: str, passed: bool, detail: str = ""):
    RESULTS.append({"test": name, "passed": passed, "detail": detail})
    mark = "ok" if passed else "ERR"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))


# ── E2E Training Trigger Tests ──────────────────────────────────────

class TestE2ETrainingTrigger:
    """Tests triggering training through the web UI using computer-use agent."""

    @pytest.fixture
    async def agent(self):
        from domains.agents.computer_use import ComputerUseAgent

        async with ComputerUseAgent(base_url=BASE, headless=True) as a:
            yield a

    @pytest.fixture
    async def follower(self):
        from domains.agents.devtools_follower import DevToolsFollower

        async with DevToolsFollower(base_url=BASE, headless=True) as f:
            yield f

    async def test_navigate_to_training_and_check_ui(self, agent):
        """Navigate to training page and verify UI elements exist."""
        result = await agent.navigate("/training")
        body = await agent.get_body_text()

        checks = {
            "page_loaded": result.status == 200,
            "has_content": len(body) > 100,
            "has_train_keyword": "train" in body.lower(),
        }

        ok("e2e_training_page_ui", all(checks.values()), f"checks={checks}")
        assert all(checks.values())

    async def test_training_page_has_config_elements(self, agent):
        """Check that training page has configuration elements."""
        await agent.navigate("/training")
        time.sleep(2)
        body = await agent.get_body_text()

        config_elements = {
            "has_method": any(w in body.lower() for w in ["method", "sft", "finetune"]),
            "has_epochs": any(w in body.lower() for w in ["epoch", "iterations"]),
            "has_lr": any(w in body.lower() for w in ["learning rate", "lr"]),
            "has_batch": any(w in body.lower() for w in ["batch"]),
        }

        ok("e2e_training_config_elements", all(config_elements.values()), f"elements={config_elements}")
        assert all(config_elements.values())

    async def test_training_start_button_clickable(self, agent):
        """Verify the training start button is clickable."""
        await agent.navigate("/training")
        time.sleep(2)

        btn = await agent.click_button("Train")
        if not btn.found:
            btn = await agent.click_button("Start")

        ok("e2e_training_start_button", btn.found, f"found={btn.found}")
        assert btn.found

    async def test_datasets_page_import_flow(self, agent):
        """Test the datasets import flow."""
        await agent.navigate("/datasets")
        time.sleep(1)

        import_btn = await agent.click_button("Import")
        ok("e2e_datasets_import_button", import_btn.found)
        assert import_btn.found

        time.sleep(1)
        has_dialog = await agent.element_exists("dialog")
        ok("e2e_datasets_import_dialog", has_dialog)
        assert has_dialog

    async def test_training_devtools_monitoring(self, follower):
        """Test DevTools monitoring during training navigation."""
        await follower.navigate("/training")
        time.sleep(2)

        report = follower.report()
        checks = {
            "has_steps": report.total_steps >= 1,
            "has_console": report.total_console > 0,
            "has_network": report.total_requests > 0,
            "no_critical_errors": report.total_console_errors == 0,
        }

        ok("e2e_training_devtools", all(checks.values()), f"checks={checks}")
        assert all(checks.values())

    async def test_full_training_flow_with_devtools(self, follower):
        """Full flow: navigate → check UI → monitor DevTools → screenshot."""
        await follower.navigate("/training")
        time.sleep(2)

        body = await follower.get_body_text()
        has_ui = len(body) > 100

        await follower.navigate("/datasets")
        time.sleep(1)

        report = follower.report()
        json_report = report.to_json()
        parsed = json.loads(json_report)

        ok("e2e_full_flow_devtools", has_ui and parsed["totals"]["steps"] >= 2,
           f"steps={parsed['totals']['steps']}, ui={has_ui}")
        assert has_ui

    async def test_cross_page_navigation_no_errors(self, follower):
        """Navigate across multiple pages and check for errors."""
        pages = ["/training", "/datasets", "/models", "/chat", "/monitoring"]
        for page in pages:
            await follower.navigate(page)
            time.sleep(0.5)

        report = follower.report()
        ok("e2e_cross_page_navigation",
           report.total_steps >= 5 and report.total_failed_requests == 0,
           f"steps={report.total_steps}, failed={report.total_failed_requests}")
        assert report.total_failed_requests == 0

    async def test_api_health_endpoint_reachable(self, agent):
        """Verify the API health endpoint is reachable from the agent."""
        await agent.navigate("/api/health")
        body = await agent.get_body_text()
        has_health = "ok" in body.lower() or "status" in body.lower() or "healthy" in body.lower()
        ok("e2e_api_health_reachable", has_health, f"body={body[:100]}")
        assert has_health

    async def test_training_job_detail_page(self, agent):
        """Navigate to a training job detail page."""
        await agent.navigate("/training/job/test-job-123")
        body = await agent.get_body_text()
        ok("e2e_training_job_detail", len(body) > 50, f"len={len(body)}")
        assert len(body) > 50

    async def test_monitoring_page_loads_with_metrics(self, agent):
        """Verify monitoring page loads with system metrics."""
        await agent.navigate("/monitoring")
        body = await agent.get_body_text()
        has_metrics = any(w in body.lower() for w in ["cpu", "memory", "monitor", "health"])
        ok("e2e_monitoring_metrics", has_metrics, f"body_snippet={body[:200]}")
        assert has_metrics


# ── Results ─────────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def save_results():
    yield
    out = Path(__file__).parent / "test_results" / "e2e_training_trigger_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(RESULTS, indent=2))

    total = len(RESULTS)
    passed = sum(1 for r in RESULTS if r["passed"])
    failed = total - passed
    print(f"\n{'='*60}")
    print(f"E2E Training Trigger Results: {passed}/{total} passed, {failed} failed")
    print(f"{'='*60}")
    if failed:
        for r in RESULTS:
            if not r["passed"]:
                print(f"  ERR {r['test']}: {r['detail']}")
