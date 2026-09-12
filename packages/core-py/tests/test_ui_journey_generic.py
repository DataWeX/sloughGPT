"""
Tests for the Generic UI Journey Testing Library.

Usage:
    .venv/bin/python -m pytest tests/test_ui_journey_generic.py -x -v
"""
import pytest


class TestGenericCore:
    """Tests for the generic core library."""

    def test_site_config_creation(self):
        from domains.testing import SiteConfig

        config = SiteConfig(
            name="Test App",
            base_url="http://localhost:3000",
        )
        assert config.name == "Test App"
        assert config.base_url == "http://localhost:3000"

    def test_page_creation(self):
        from domains.testing import Page

        page = Page(
            name="Home",
            path="/",
            checks=["welcome", "home"],
        )
        assert page.name == "Home"
        assert page.path == "/"
        assert len(page.checks) == 2

    def test_create_site_config(self):
        from domains.testing import create_site_config

        config = create_site_config(
            name="My App",
            base_url="http://localhost:3000",
            pages={
                "home": ("/", ["welcome"]),
                "about": ("/about", ["about us"]),
            },
        )
        assert config.name == "My App"
        assert len(config.pages) == 2
        assert "home" in config.pages

    def test_journey_creation(self):
        from domains.testing import Journey, SiteConfig

        config = SiteConfig(name="Test", base_url="http://localhost:3000")
        journey = Journey(config)
        assert journey.config.name == "Test"

    def test_journey_run(self):
        from domains.testing import Journey, SiteConfig

        config = SiteConfig(name="Test", base_url="http://localhost:3000")
        journey = Journey(config)
        result = journey.run([
            journey.goto("/"),
            journey.check_body("test"),
        ])
        assert result.passed
        assert result.passed_count == 2

    def test_journey_report(self):
        from domains.testing import Journey, SiteConfig

        config = SiteConfig(name="Test", base_url="http://localhost:3000")
        journey = Journey(config)
        journey.run([journey.goto("/")])
        report = journey.report()
        assert "Test" in report
        assert "Journey Test Report" in report

    def test_step_result(self):
        from domains.testing import StepResult

        result = StepResult(name="test", passed=True, detail="ok")
        assert result.passed
        assert result.name == "test"

    def test_journey_result_to_dict(self):
        from domains.testing import Journey, SiteConfig

        config = SiteConfig(name="Test", base_url="http://localhost:3000")
        journey = Journey(config)
        result = journey.run([journey.goto("/")])
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["name"] == "journey"
        assert d["passed"] is True


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


class TestStepBuilder:
    """Tests for step builder."""

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


class TestSloughGPTConfig:
    """Tests for sloughGPT-specific config."""

    def test_sloughgpt_site_config(self):
        from domains.testing.sloughgpt import SLOUGHPGPT_SITE

        assert SLOUGHPGPT_SITE.name == "sloughGPT"
        assert SLOUGHPGPT_SITE.base_url == "http://localhost:3000"
        assert len(SLOUGHPGPT_SITE.pages) > 0

    def test_sloughgpt_pages(self):
        from domains.testing.sloughgpt import SLOUGHPGPT_PAGES

        assert "training" in SLOUGHPGPT_PAGES
        assert "chat" in SLOUGHPGPT_PAGES
        assert "datasets" in SLOUGHPGPT_PAGES

    def test_sloughgpt_training_pages(self):
        from domains.testing.sloughgpt import SLOUGHPGPT_TRAINING_PAGES

        assert "queue" in SLOUGHPGPT_TRAINING_PAGES
        assert "runs" in SLOUGHPGPT_TRAINING_PAGES
        assert "presets" in SLOUGHPGPT_TRAINING_PAGES

    def test_sloughgpt_journey(self):
        from domains.testing.sloughgpt import SloughGPTJourney

        journey = SloughGPTJourney()
        assert journey.config.name == "sloughGPT"


class TestPrebuiltJourneys:
    """Tests for pre-built journey tests."""

    def test_page_journey(self):
        from domains.testing.journeys import PageJourney
        from domains.testing.sloughgpt import SLOUGHPGPT_SITE

        journey = PageJourney(SLOUGHPGPT_SITE)
        result = journey.test_page("training")
        assert result.passed

    def test_navigation_journey(self):
        from domains.testing.journeys import NavigationJourney
        from domains.testing.sloughgpt import SLOUGHPGPT_SITE

        journey = NavigationJourney(SLOUGHPGPT_SITE)
        result = journey.test_sidebar_navigation()
        assert result.passed

    def test_full_suite(self):
        from domains.testing.journeys import FullSuiteJourney
        from domains.testing.sloughgpt import SLOUGHPGPT_SITE

        suite = FullSuiteJourney(SLOUGHPGPT_SITE)
        results = suite.run_all()
        assert len(results) > 0


class TestChromeDevTools:
    """Tests for ChromeDevTools integration."""

    def test_browser_creation(self):
        from domains.testing.chrome_devtools import ChromeDevToolsBrowser

        browser = ChromeDevToolsBrowser(base_url="http://localhost:3000")
        assert browser.base_url == "http://localhost:3000"

    def test_browser_state(self):
        from domains.testing.chrome_devtools import BrowserState

        state = BrowserState()
        assert state.page_id is None
        assert state.url == ""

    def test_browser_commands(self):
        from domains.testing.chrome_devtools import ChromeDevToolsBrowser

        browser = ChromeDevToolsBrowser()
        commands = browser.get_commands()
        assert isinstance(commands, list)

    def test_browser_report(self):
        from domains.testing.chrome_devtools import ChromeDevToolsBrowser

        browser = ChromeDevToolsBrowser(base_url="http://localhost:3000")
        report = browser.report()
        assert "ChromeDevTools Browser Report" in report
