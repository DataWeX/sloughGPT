"""
Pre-built Journey Tests for Common Web Applications.

Provides reusable journey tests that work with any site configuration.

Usage:
    from domains.testing.journeys import PageJourney, NavigationJourney, FullSuiteJourney
    from domains.testing.sloughgpt import SLOUGHPGPT_SITE

    journey = PageJourney(SLOUGHPGPT_SITE)
    result = journey.test_page("training")
"""
from __future__ import annotations

from typing import Dict, List, Optional

from . import Journey, JourneyResult, Page, SiteConfig


class PageJourney:
    """Journey tests for individual pages."""

    def __init__(self, config: SiteConfig):
        self.config = config
        self.journey = Journey(config)

    def test_page(self, page_key: str) -> JourneyResult:
        """Test a single page."""
        page = self.config.pages.get(page_key)
        if not page:
            return JourneyResult(
                name=f"page_{page_key}",
                steps=[],
            )

        steps = [
            self.journey.goto(page.path),
            self.journey.check_body(page_key),
        ]

        for check in page.checks:
            steps.append(self.journey.wait_for(check))

        steps.append(self.journey.check_no_console_errors())

        return self.journey.run(steps, name=f"page_{page_key}")

    def test_all_pages(self) -> JourneyResult:
        """Test all pages in the config."""
        steps = []
        for key, page in self.config.pages.items():
            steps.append(self.journey.goto(page.path))
            steps.append(self.journey.check_body(key))

        return self.journey.run(steps, name="all_pages")


class NavigationJourney:
    """Journey tests for navigation flows."""

    def __init__(self, config: SiteConfig):
        self.config = config
        self.journey = Journey(config)

    def test_sidebar_navigation(self) -> JourneyResult:
        """Test navigation via sidebar."""
        steps = []
        for key, page in self.config.pages.items():
            steps.append(self.journey.goto(page.path))
            steps.append(self.journey.wait_for(page.checks[0] if page.checks else key))

        return self.journey.run(steps, name="sidebar_navigation")

    def test_back_and_forth(self, page1: str, page2: str) -> JourneyResult:
        """Test navigating back and forth between pages."""
        p1 = self.config.pages.get(page1)
        p2 = self.config.pages.get(page2)

        if not p1 or not p2:
            return JourneyResult(name="back_and_forth", steps=[])

        steps = [
            self.journey.goto(p1.path),
            self.journey.wait_for(p1.checks[0] if p1.checks else page1),
            self.journey.goto(p2.path),
            self.journey.wait_for(p2.checks[0] if p2.checks else page2),
            self.journey.goto(p1.path),
            self.journey.wait_for(p1.checks[0] if p1.checks else page1),
        ]

        return self.journey.run(steps, name=f"back_forth_{page1}_{page2}")


class APIJourney:
    """Journey tests for API health checks."""

    def __init__(self, config: SiteConfig):
        self.config = config
        self.journey = Journey(config)

    def test_health(self) -> JourneyResult:
        """Test API health endpoint."""
        if not self.config.api_url:
            return JourneyResult(name="api_health", steps=[])

        steps = [
            self.journey.check_api("/health"),
        ]

        return self.journey.run(steps, name="api_health")

    def test_all_endpoints(self, endpoints: Dict[str, str]) -> JourneyResult:
        """Test multiple API endpoints."""
        steps = []
        for name, path in endpoints.items():
            steps.append(self.journey.check_api(path))

        return self.journey.run(steps, name="api_endpoints")


class FullSuiteJourney:
    """Complete test suite for a site."""

    def __init__(self, config: SiteConfig):
        self.config = config
        self.page_journey = PageJourney(config)
        self.nav_journey = NavigationJourney(config)
        self.api_journey = APIJourney(config)

    def run_all(self) -> List[JourneyResult]:
        """Run all journey tests."""
        results = []

        # Page tests
        results.append(self.page_journey.test_all_pages())

        # Navigation tests
        results.append(self.nav_journey.test_sidebar_navigation())

        # API tests
        results.append(self.api_journey.test_health())

        return results

    def report(self) -> str:
        """Run all tests and generate a report."""
        results = self.run_all()
        lines = [f"{self.config.name} Full Suite Report", "=" * 50]

        total_passed = 0
        total_failed = 0

        for result in results:
            status = "PASS" if result.passed else "FAIL"
            lines.append(f"\n[{status}] {result.name} ({result.total_duration_s:.2f}s)")

            for step in result.steps:
                step_status = "ok" if step.passed else "ERR"
                lines.append(f"  [{step_status}] {step.name} — {step.detail}")

            total_passed += result.passed_count
            total_failed += result.failed_count

        lines.append(f"\n{'=' * 50}")
        lines.append(f"Total: {total_passed + total_failed} steps, {total_passed} passed, {total_failed} failed")

        return "\n".join(lines)
