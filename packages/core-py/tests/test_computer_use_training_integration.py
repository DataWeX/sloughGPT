"""
Integration tests for Computer-Use Agent + Comprehensive Trainer.

Tests the full pipeline: agent navigates UI → triggers training → monitors DevTools.
Requires running API (localhost:8000) and web (localhost:3000) servers.

Usage:
    .venv/bin/python -m pytest tests/test_computer_use_training_integration.py -x -v -s
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


# ── Computer-Use Agent Integration Tests ────────────────────────────

class TestComputerUseAgentIntegration:
    """Tests the ComputerUseAgent against a live server."""

    @pytest.fixture
    async def agent(self):
        from domains.agents.computer_use import ComputerUseAgent

        async with ComputerUseAgent(base_url=BASE, headless=True) as a:
            yield a

    async def test_agent_navigates_to_training(self, agent):
        result = await agent.navigate("/training")
        body = await agent.get_body_text()
        ok("agent_navigates_training", result.status == 200 and len(body) > 50,
           f"status={result.status}, len={len(body)}")
        assert result.status == 200

    async def test_agent_navigates_to_datasets(self, agent):
        result = await agent.navigate("/datasets")
        body = await agent.get_body_text()
        ok("agent_navigates_datasets", result.status == 200 and len(body) > 50,
           f"status={result.status}, len={len(body)}")
        assert result.status == 200

    async def test_agent_clicks_import_button(self, agent):
        await agent.navigate("/datasets")
        time.sleep(1)
        result = await agent.click_button("Import")
        ok("agent_clicks_import", result.found, f"found={result.found}")
        assert result.found

    async def test_agent_detects_dialog(self, agent):
        await agent.navigate("/datasets")
        time.sleep(1)
        await agent.click_button("Import")
        time.sleep(1)
        exists = await agent.element_exists("dialog")
        ok("agent_detects_dialog", exists, f"exists={exists}")
        assert exists

    async def test_agent_devtools_captures_console(self, agent):
        await agent.navigate("/training")
        time.sleep(2)
        report = agent.devtools_report()
        ok("agent_devtools_console", report.total_console > 0,
           f"console={report.total_console}")
        assert report.total_console > 0

    async def test_agent_devtools_captures_network(self, agent):
        await agent.navigate("/training")
        time.sleep(2)
        report = agent.devtools_report()
        ok("agent_devtools_network", report.total_requests > 0,
           f"requests={report.total_requests}")
        assert report.total_requests > 0

    async def test_agent_devtools_no_critical_errors(self, agent):
        await agent.navigate("/training")
        time.sleep(2)
        critical = [e for e in agent._errors if "uncaught" in e.get("message", "").lower()]
        ok("agent_devtools_no_critical", len(critical) == 0,
           f"critical={len(critical)}")
        assert len(critical) == 0

    async def test_agent_screenshot(self, agent, tmp_path):
        await agent.navigate("/training")
        path = str(tmp_path / "training_screenshot.png")
        await agent.take_screenshot(path)
        exists = Path(path).exists()
        ok("agent_screenshot", exists, f"path={path}")
        assert exists

    async def test_agent_step_recording(self, agent):
        await agent.navigate("/training")
        s1 = agent.step("navigate to training")
        s2 = agent.step("check page content")
        ok("agent_step_recording", s1["step"] == 1 and s2["step"] == 2)
        assert s1["step"] == 1

    async def test_agent_full_log(self, agent):
        await agent.navigate("/training")
        time.sleep(1)
        log = agent.get_full_log()
        ok("agent_full_log", len(log) > 0, f"entries={len(log)}")
        assert len(log) > 0


# ── DevTools Follower Integration Tests ─────────────────────────────

class TestDevToolsFollowerIntegration:
    """Tests the DevToolsFollower against a live server."""

    @pytest.fixture
    async def follower(self):
        from domains.agents.devtools_follower import DevToolsFollower

        async with DevToolsFollower(base_url=BASE, headless=True) as f:
            yield f

    async def test_follower_navigate(self, follower):
        record = await follower.navigate("/training")
        ok("follower_navigate", record.success and record.step == 1,
           f"step={record.step}, success={record.success}")
        assert record.success

    async def test_follower_click_button(self, follower):
        await follower.navigate("/datasets")
        time.sleep(1)
        record = await follower.click_button("Import")
        ok("follower_click", record.step > 0, f"step={record.step}")
        assert record.step > 0

    async def test_follower_multi_step(self, follower):
        await follower.navigate("/training")
        await follower.navigate("/datasets")
        await follower.navigate("/models")
        report = follower.report()
        ok("follower_multi_step", report.total_steps >= 3,
           f"steps={report.total_steps}")
        assert report.total_steps >= 3

    async def test_follower_report_json(self, follower):
        await follower.navigate("/training")
        time.sleep(1)
        report = follower.report()
        json_str = report.to_json()
        parsed = json.loads(json_str)
        ok("follower_report_json", "steps" in parsed and "totals" in parsed)
        assert "steps" in parsed

    async def test_follower_devtools_summary(self, follower):
        await follower.navigate("/training")
        time.sleep(1)
        report = follower.report()
        has_summary = "Console:" in report.summary and "Network:" in report.summary
        ok("follower_devtools_summary", has_summary, f"summary={report.summary[:100]}")
        assert has_summary

    async def test_follower_screenshot(self, follower, tmp_path):
        await follower.navigate("/training")
        path = str(tmp_path / "follower_screenshot.png")
        await follower.take_screenshot(path)
        ok("follower_screenshot", Path(path).exists())
        assert Path(path).exists()

    async def test_follower_clear(self, follower):
        await follower.navigate("/training")
        time.sleep(1)
        follower.clear()
        report = follower.report()
        ok("follower_clear", report.total_steps == 0, f"steps={report.total_steps}")
        assert report.total_steps == 0


# ── Cross-System Integration Tests ──────────────────────────────────

class TestCrossSystemIntegration:
    """Tests that agent + trainer + DevTools work together."""

    @pytest.fixture
    async def agent(self):
        from domains.agents.computer_use import ComputerUseAgent

        async with ComputerUseAgent(base_url=BASE, headless=True) as a:
            yield a

    async def test_training_page_has_expected_elements(self, agent):
        await agent.navigate("/training")
        body = await agent.get_body_text()
        checks = {
            "has_train_keyword": "train" in body.lower(),
            "has_page_content": len(body) > 100,
            "has_navigation": await agent.element_exists("link"),
        }
        all_ok = all(checks.values())
        ok("cross_training_elements", all_ok, f"checks={checks}")
        assert all_ok

    async def test_datasets_page_has_import(self, agent):
        await agent.navigate("/datasets")
        time.sleep(1)
        has_import = await agent.element_exists("button", "Import")
        ok("cross_datasets_import", has_import)
        assert has_import

    async def test_full_flow_navigate_check_screenshot(self, agent, tmp_path):
        await agent.navigate("/training")
        time.sleep(1)
        body = await agent.get_body_text()
        has_content = len(body) > 50

        await agent.navigate("/datasets")
        time.sleep(1)
        body2 = await agent.get_body_text()
        has_content2 = len(body2) > 50

        path = str(tmp_path / "full_flow.png")
        await agent.take_screenshot(path)

        ok("cross_full_flow", has_content and has_content2 and Path(path).exists())
        assert has_content and has_content2

    async def test_api_requests_reach_backend(self, agent):
        await agent.navigate("/training")
        time.sleep(2)
        api_reqs = [r for r in agent._network_requests if "localhost:8000" in r.get("url", "")]
        ok("cross_api_requests", len(api_reqs) > 0, f"api_reqs={len(api_reqs)}")
        assert len(api_reqs) > 0


# ── Results ─────────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def save_results():
    yield
    out = Path(__file__).parent / "test_results" / "computer_use_integration_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(RESULTS, indent=2))

    total = len(RESULTS)
    passed = sum(1 for r in RESULTS if r["passed"])
    failed = total - passed
    print(f"\n{'='*60}")
    print(f"Computer-Use Integration Results: {passed}/{total} passed, {failed} failed")
    print(f"{'='*60}")
    if failed:
        for r in RESULTS:
            if not r["passed"]:
                print(f"  ERR {r['test']}: {r['detail']}")
