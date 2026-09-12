"""
Tests for the UI Journey Testing Library (legacy wrapper).

Usage:
    .venv/bin/python -m pytest tests/test_ui_journey_library.py -x -v
"""
import pytest


class TestJourneyCore:
    """Tests for the core Journey class."""

    def test_journey_creation(self):
        from domains.testing import Journey, SiteConfig

        config = SiteConfig(name="Test", base_url="http://localhost:3000")
        journey = Journey(config)
        assert journey.config.name == "Test"

    def test_journey_execution(self):
        from domains.testing import Journey, SiteConfig

        config = SiteConfig(name="Test", base_url="http://localhost:3000")
        journey = Journey(config)
        result = journey.run([
            journey.goto("/"),
            journey.check_body("test"),
        ])
        assert result.passed
        assert result.passed_count == 2

    def test_journey_result_to_dict(self):
        from domains.testing import Journey, SiteConfig

        config = SiteConfig(name="Test", base_url="http://localhost:3000")
        journey = Journey(config)
        result = journey.run([journey.goto("/")])
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["passed"] is True

    def test_report_generation(self):
        from domains.testing import Journey, SiteConfig

        config = SiteConfig(name="Test", base_url="http://localhost:3000")
        journey = Journey(config)
        journey.journey("test1", [journey.goto("/")]) if hasattr(journey, "journey") else journey.run([journey.goto("/")])
        report = journey.report()
        assert "Test" in report
        assert "Report" in report


class TestStepResult:
    """Tests for StepResult."""

    def test_step_result(self):
        from domains.testing import StepResult

        result = StepResult(name="test", passed=True, detail="ok")
        assert result.passed
        assert result.name == "test"

    def test_step_result_failed(self):
        from domains.testing import StepResult

        result = StepResult(name="test", passed=False, detail="failed")
        assert not result.passed


class TestJourneySteps:
    """Tests for individual journey steps."""

    def test_goto_step(self):
        from domains.testing import Journey, SiteConfig

        config = SiteConfig(name="Test", base_url="http://localhost:3000")
        journey = Journey(config)
        step = journey.goto("/training")
        result = step()
        assert result.passed
        assert "training" in result.detail

    def test_wait_for_step(self):
        from domains.testing import Journey, SiteConfig

        config = SiteConfig(name="Test", base_url="http://localhost:3000")
        journey = Journey(config)
        step = journey.wait_for("train")
        result = step()
        assert result.passed

    def test_check_body_step(self):
        from domains.testing import Journey, SiteConfig

        config = SiteConfig(name="Test", base_url="http://localhost:3000")
        journey = Journey(config)
        step = journey.check_body("training")
        result = step()
        assert result.passed

    def test_click_button_step(self):
        from domains.testing import Journey, SiteConfig

        config = SiteConfig(name="Test", base_url="http://localhost:3000")
        journey = Journey(config)
        step = journey.click_button("Start")
        result = step()
        assert result.passed

    def test_check_api_step(self):
        from domains.testing import Journey, SiteConfig

        config = SiteConfig(name="Test", base_url="http://localhost:3000")
        journey = Journey(config)
        step = journey.check_api("/health")
        result = step()
        assert result.passed

    def test_check_no_console_errors_step(self):
        from domains.testing import Journey, SiteConfig

        config = SiteConfig(name="Test", base_url="http://localhost:3000")
        journey = Journey(config)
        step = journey.check_no_console_errors()
        result = step()
        assert result.passed

    def test_check_no_network_errors_step(self):
        from domains.testing import Journey, SiteConfig

        config = SiteConfig(name="Test", base_url="http://localhost:3000")
        journey = Journey(config)
        step = journey.check_no_network_errors()
        result = step()
        assert result.passed

    def test_custom_step(self):
        from domains.testing import Journey, SiteConfig, StepResult

        config = SiteConfig(name="Test", base_url="http://localhost:3000")
        journey = Journey(config)

        def my_check():
            return StepResult(name="custom", passed=True, detail="custom check")

        step = journey.custom("my_check", my_check)
        result = step()
        assert result.passed
        assert result.name == "custom"


class TestJourneyResults:
    """Tests for journey result handling."""

    def test_all_passed(self):
        from domains.testing import Journey, SiteConfig

        config = SiteConfig(name="Test", base_url="http://localhost:3000")
        journey = Journey(config)
        result = journey.run([
            journey.goto("/"),
            journey.goto("/chat"),
        ])
        assert result.passed
        assert result.failed_count == 0

    def test_one_failed(self):
        from domains.testing import Journey, SiteConfig, StepResult

        config = SiteConfig(name="Test", base_url="http://localhost:3000")
        journey = Journey(config)

        def failing_step():
            return StepResult(name="fail", passed=False, detail="failed")

        step = journey.custom("failing_step", failing_step)
        result = journey.run([step])
        assert not result.passed
        assert result.failed_count == 1

    def test_multiple_results(self):
        from domains.testing import Journey, SiteConfig

        config = SiteConfig(name="Test", base_url="http://localhost:3000")
        journey = Journey(config)
        journey.run([journey.goto("/")])
        journey.run([journey.goto("/chat")])
        assert len(journey.results) == 2


class TestAssertions:
    """Tests for assertion helpers."""

    def test_assert_page_loads(self):
        from domains.testing import assert_page_loads

        result = assert_page_loads("x" * 100)
        assert result.passed

    def test_assert_page_loads_short(self):
        from domains.testing import assert_page_loads

        result = assert_page_loads("short")
        assert not result.passed

    def test_assert_body_contains(self):
        from domains.testing import assert_body_contains

        result = assert_body_contains("Hello World", "World")
        assert result.passed

    def test_assert_no_errors(self):
        from domains.testing import assert_no_errors

        result = assert_no_errors([], [])
        assert result.passed

    def test_assert_no_errors_with_errors(self):
        from domains.testing import assert_no_errors

        result = assert_no_errors(["error1"], [])
        assert not result.passed

    def test_assert_api_healthy(self):
        from domains.testing import assert_api_healthy

        result = assert_api_healthy(200)
        assert result.passed

    def test_assert_api_unhealthy(self):
        from domains.testing import assert_api_healthy

        result = assert_api_healthy(500)
        assert not result.passed
