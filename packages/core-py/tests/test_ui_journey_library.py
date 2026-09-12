"""
Tests for the UI Journey Testing Library.

Verifies that the testing library works correctly.

Usage:
    .venv/bin/python -m pytest tests/test_ui_journey_library.py -x -v
"""
import pytest


class TestUITestSuite:
    """Tests for UITestSuite."""

    def test_suite_creation(self):
        from domains.testing import UITestSuite

        suite = UITestSuite(base_url="http://localhost:3000")
        assert suite.base_url == "http://localhost:3000"

    def test_journey_execution(self):
        from domains.testing import UITestSuite

        suite = UITestSuite()
        result = suite.journey("test_journey", [
            suite.goto("/"),
            suite.check_body("test"),
        ])
        assert result.passed
        assert result.passed_count == 2

    def test_step_result(self):
        from domains.testing import StepResult

        result = StepResult(name="test", passed=True, detail="ok")
        assert result.passed
        assert result.name == "test"

    def test_journey_result_to_dict(self):
        from domains.testing import UITestSuite

        suite = UITestSuite()
        result = suite.journey("test", [suite.goto("/")])
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["name"] == "test"
        assert d["passed"] is True

    def test_report_generation(self):
        from domains.testing import UITestSuite

        suite = UITestSuite()
        suite.journey("test1", [suite.goto("/")])
        suite.journey("test2", [suite.goto("/chat")])
        report = suite.report()
        assert "UI Journey Test Report" in report
        assert "test1" in report
        assert "test2" in report


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

    def test_assert_body_contains_case_insensitive(self):
        from domains.testing import assert_body_contains

        result = assert_body_contains("Hello World", "hello")
        assert result.passed

    def test_assert_no_errors(self):
        from domains.testing import assert_no_errors

        result = assert_no_errors([], [])
        assert result.passed

    def test_assert_no_errors_with_errors(self):
        from domains.testing import assert_no_errors

        result = assert_no_errors(["error1"], [])
        assert not result.passed

    def test_assert_element_exists(self):
        from domains.testing import assert_element_exists

        result = assert_element_exists("button", True)
        assert result.passed

    def test_assert_element_not_exists(self):
        from domains.testing import assert_element_exists

        result = assert_element_exists("button", False)
        assert not result.passed

    def test_assert_api_healthy(self):
        from domains.testing import assert_api_healthy

        result = assert_api_healthy(200)
        assert result.passed

    def test_assert_api_unhealthy(self):
        from domains.testing import assert_api_healthy

        result = assert_api_healthy(500)
        assert not result.passed


class TestJourneySteps:
    """Tests for individual journey steps."""

    def test_goto_step(self):
        from domains.testing import UITestSuite

        suite = UITestSuite()
        step = suite.goto("/training")
        result = step()
        assert result.passed
        assert "training" in result.detail

    def test_wait_for_step(self):
        from domains.testing import UITestSuite

        suite = UITestSuite()
        step = suite.wait_for("train")
        result = step()
        assert result.passed

    def test_check_body_step(self):
        from domains.testing import UITestSuite

        suite = UITestSuite()
        step = suite.check_body("training")
        result = step()
        assert result.passed

    def test_click_button_step(self):
        from domains.testing import UITestSuite

        suite = UITestSuite()
        step = suite.click_button("Start")
        result = step()
        assert result.passed

    def test_check_api_step(self):
        from domains.testing import UITestSuite

        suite = UITestSuite()
        step = suite.check_api("/health")
        result = step()
        assert result.passed

    def test_check_no_console_errors_step(self):
        from domains.testing import UITestSuite

        suite = UITestSuite()
        step = suite.check_no_console_errors()
        result = step()
        assert result.passed

    def test_check_no_network_errors_step(self):
        from domains.testing import UITestSuite

        suite = UITestSuite()
        step = suite.check_no_network_errors()
        result = step()
        assert result.passed


class TestJourneyResults:
    """Tests for journey result handling."""

    def test_all_passed(self):
        from domains.testing import UITestSuite

        suite = UITestSuite()
        result = suite.journey("test", [
            suite.goto("/"),
            suite.goto("/chat"),
        ])
        assert result.passed
        assert result.failed_count == 0

    def test_one_failed(self):
        from domains.testing import UITestSuite
        from domains.testing import StepResult

        suite = UITestSuite()

        def failing_step():
            return StepResult(name="fail", passed=False, detail="failed")

        result = suite.journey("test", [failing_step()])
        assert not result.passed
        assert result.failed_count == 1

    def test_multiple_results(self):
        from domains.testing import UITestSuite

        suite = UITestSuite()
        suite.journey("test1", [suite.goto("/")])
        suite.journey("test2", [suite.goto("/chat")])
        assert len(suite.results) == 2
