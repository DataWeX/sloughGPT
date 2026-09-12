"""
Tests for ChromeDevTools and sloughGPT Journey Test Helpers.

Usage:
    .venv/bin/python -m pytest tests/test_ui_journey_chrome_devtools.py -x -v
"""
import pytest


class TestChromeDevToolsBrowser:
    """Tests for ChromeDevToolsBrowser."""

    def test_browser_creation(self):
        from domains.testing.chrome_devtools import ChromeDevToolsBrowser

        browser = ChromeDevToolsBrowser(base_url="http://localhost:3000")
        assert browser.base_url == "http://localhost:3000"

    def test_browser_state(self):
        from domains.testing.chrome_devtools import BrowserState

        state = BrowserState()
        assert state.page_id is None
        assert state.url == ""
        assert state.errors == []

    def test_commands_tracking(self):
        from domains.testing.chrome_devtools import ChromeDevToolsBrowser

        browser = ChromeDevToolsBrowser()
        assert len(browser.commands) == 0

    def test_get_commands(self):
        from domains.testing.chrome_devtools import ChromeDevToolsBrowser

        browser = ChromeDevToolsBrowser()
        commands = browser.get_commands()
        assert isinstance(commands, list)

    def test_report_generation(self):
        from domains.testing.chrome_devtools import ChromeDevToolsBrowser

        browser = ChromeDevToolsBrowser(base_url="http://localhost:3000")
        report = browser.report()
        assert "ChromeDevTools Browser Report" in report
        assert "http://localhost:3000" in report


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

    def test_sloughgpt_api_endpoints(self):
        from domains.testing.sloughgpt import SLOUGHPGPT_API_ENDPOINTS

        assert "health" in SLOUGHPGPT_API_ENDPOINTS

    def test_page_check_creation(self):
        from domains.testing import Page

        check = Page(
            name="Test",
            path="/test",
            checks=["test"],
        )
        assert check.name == "Test"
        assert check.path == "/test"

    def test_sloughgpt_journey(self):
        from domains.testing.sloughgpt import SloughGPTJourney

        journey = SloughGPTJourney()
        assert journey.config.name == "sloughGPT"


class TestJourneyIntegration:
    """Integration tests for the journey testing library."""

    def test_import_all_modules(self):
        from domains.testing import Journey, SiteConfig, Page
        from domains.testing.chrome_devtools import ChromeDevToolsBrowser
        from domains.testing.sloughgpt import SloughGPTJourney, SLOUGHPGPT_SITE
        from domains.testing.journeys import PageJourney, FullSuiteJourney

        assert Journey is not None
        assert ChromeDevToolsBrowser is not None
        assert SloughGPTJourney is not None
        assert PageJourney is not None
        assert FullSuiteJourney is not None

    def test_uisuite_with_chrome_devtools(self):
        from domains.testing import Journey, SiteConfig
        from domains.testing.chrome_devtools import ChromeDevToolsBrowser

        config = SiteConfig(name="Test", base_url="http://localhost:3000")
        journey = Journey(config)
        browser = ChromeDevToolsBrowser(base_url="http://localhost:3000")

        assert journey.config.base_url == browser.base_url

    def test_sloughgpt_journey_results(self):
        from domains.testing.sloughgpt import SloughGPTJourney

        journey = SloughGPTJourney()
        assert len(journey.journey.results) == 0
