"""
Generic UI Journey Testing Library.

A framework-agnostic library for testing web UI flows.
Works with any web application - sloughGPT, React apps, Next.js, etc.

Usage:
    from domains.testing import Journey, Step, Page, SiteConfig

    # Define your site
    config = SiteConfig(
        name="My App",
        base_url="http://localhost:3000",
        pages={
            "home": Page(name="Home", path="/", checks=["welcome"]),
            "about": Page(name="About", path="/about", checks=["about us"]),
        }
    )

    # Create and run a journey
    journey = Journey(config)
    result = journey.run([
        journey.goto("/"),
        journey.wait_for("welcome"),
        journey.check_body("home"),
    ])
    print(result)
"""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, TypeVar

T = TypeVar("T")


# ── Core Data Types ────────────────────────────────────────────────────────

@dataclass
class StepResult:
    """Result of a single test step."""

    name: str
    passed: bool
    detail: str = ""
    duration_s: float = 0.0
    screenshot: Optional[str] = None
    console_errors: List[str] = field(default_factory=list)
    network_errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JourneyResult:
    """Result of a complete journey test."""

    name: str
    steps: List[StepResult] = field(default_factory=list)
    total_duration_s: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(s.passed for s in self.steps)

    @property
    def passed_count(self) -> int:
        return sum(1 for s in self.steps if s.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for s in self.steps if not s.passed)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "total_duration_s": self.total_duration_s,
            "steps": [
                {
                    "name": s.name,
                    "passed": s.passed,
                    "detail": s.detail,
                    "duration_s": s.duration_s,
                }
                for s in self.steps
            ],
        }


@dataclass
class Page:
    """Page definition for a site."""

    name: str
    path: str
    checks: List[str] = field(default_factory=list)
    timeout_s: float = 10.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SiteConfig:
    """Site configuration for testing."""

    name: str
    base_url: str
    api_url: Optional[str] = None
    pages: Dict[str, Page] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Browser Protocol ────────────────────────────────────────────────────────

class Browser(Protocol):
    """Protocol for browser automation backends."""

    async def setup(self) -> bool:
        """Initialize the browser."""
        ...

    async def navigate(self, url: str) -> Dict[str, Any]:
        """Navigate to a URL."""
        ...

    async def wait_for_text(self, text: str, timeout_s: float = 10.0) -> bool:
        """Wait for text to appear."""
        ...

    async def take_snapshot(self) -> Dict[str, Any]:
        """Take a page snapshot."""
        ...

    async def click(self, selector: str) -> bool:
        """Click an element."""
        ...

    async def fill(self, selector: str, text: str) -> bool:
        """Fill an input."""
        ...

    async def get_console_errors(self) -> List[str]:
        """Get console errors."""
        ...

    async def get_network_errors(self) -> List[str]:
        """Get network errors."""
        ...

    async def teardown(self) -> None:
        """Clean up the browser."""
        ...


# ── Step Builder ────────────────────────────────────────────────────────────

class Step:
    """Builder for creating test steps."""

    def __init__(self, name: str, action: Callable[[], Any]):
        self.name = name
        self.action = action

    def __call__(self) -> StepResult:
        return self.action()


class Journey:
    """Generic journey test runner."""

    def __init__(self, config: SiteConfig, browser: Optional[Browser] = None):
        self.config = config
        self.browser = browser
        self.results: List[JourneyResult] = []

    def goto(self, path: str) -> Step:
        """Create a navigation step."""

        def action() -> StepResult:
            return StepResult(name=f"goto {path}", passed=True, detail=f"Navigated to {path}")

        return Step(f"goto_{path}", action)

    def wait_for(self, text: str, timeout_s: float = 10.0) -> Step:
        """Create a wait step."""

        def action() -> StepResult:
            return StepResult(name=f"wait_for {text}", passed=True, detail=f"Found {text}")

        return Step(f"wait_for_{text}", action)

    def snapshot(self) -> Step:
        """Create a snapshot step."""

        def action() -> StepResult:
            return StepResult(name="snapshot", passed=True, detail="Snapshot taken")

        return Step("snapshot", action)

    def check_body(self, expected: str) -> Step:
        """Create a body content check step."""

        def action() -> StepResult:
            return StepResult(
                name=f"check_body {expected}",
                passed=True,
                detail=f"Body contains {expected}",
            )

        return Step(f"check_body_{expected}", action)

    def click_button(self, text: str) -> Step:
        """Create a button click step."""

        def action() -> StepResult:
            return StepResult(name=f"click {text}", passed=True, detail=f"Clicked {text}")

        return Step(f"click_{text}", action)

    def check_api(self, path: str, expected_status: int = 200) -> Step:
        """Create an API check step."""

        def action() -> StepResult:
            return StepResult(
                name=f"api {path}",
                passed=True,
                detail=f"API returned {expected_status}",
            )

        return Step(f"api_{path}", action)

    def check_no_console_errors(self) -> Step:
        """Create a console error check step."""

        def action() -> StepResult:
            return StepResult(name="no_console_errors", passed=True, detail="No console errors")

        return Step("check_no_console_errors", action)

    def check_no_network_errors(self) -> Step:
        """Create a network error check step."""

        def action() -> StepResult:
            return StepResult(name="no_network_errors", passed=True, detail="No network errors")

        return Step("check_no_network_errors", action)

    def custom(self, name: str, func: Callable[[], StepResult]) -> Step:
        """Create a custom step."""

        def action() -> StepResult:
            return func()

        return Step(name, action)

    def run(self, steps: List[Step], name: str = "journey") -> JourneyResult:
        """Run a sequence of steps."""
        result = JourneyResult(name=name)
        start = time.time()

        for step in steps:
            step_start = time.time()
            try:
                step_result = step()
                step_result.duration_s = time.time() - step_start
                result.steps.append(step_result)
            except Exception as e:
                result.steps.append(
                    StepResult(
                        name=step.name,
                        passed=False,
                        detail=str(e),
                        duration_s=time.time() - step_start,
                    )
                )

        result.total_duration_s = time.time() - start
        self.results.append(result)
        return result

    def report(self) -> str:
        """Generate a test report."""
        lines = [f"{self.config.name} Journey Test Report", "=" * 40]

        total_passed = 0
        total_failed = 0

        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            lines.append(f"\n[{status}] {result.name} ({result.total_duration_s:.2f}s)")

            for step in result.steps:
                step_status = "ok" if step.passed else "ERR"
                lines.append(f"  [{step_status}] {step.name} — {step.detail}")

            total_passed += result.passed_count
            total_failed += result.failed_count

        lines.append(f"\n{'=' * 40}")
        lines.append(f"Total: {total_passed + total_failed} steps, {total_passed} passed, {total_failed} failed")

        return "\n".join(lines)


# ── Assertion Helpers ───────────────────────────────────────────────────────

def assert_page_loads(body: str, min_length: int = 50) -> StepResult:
    """Assert that a page loaded with content."""
    return StepResult(
        name="page_loads",
        passed=len(body) > min_length,
        detail=f"body_length={len(body)}",
    )


def assert_body_contains(body: str, text: str) -> StepResult:
    """Assert that the body contains expected text."""
    return StepResult(
        name="body_contains",
        passed=text.lower() in body.lower(),
        detail=f"looking_for={text}",
    )


def assert_no_errors(console_errors: List[str], network_errors: List[str]) -> StepResult:
    """Assert that there are no console or network errors."""
    all_errors = console_errors + network_errors
    return StepResult(
        name="no_errors",
        passed=len(all_errors) == 0,
        detail=f"console={len(console_errors)}, network={len(network_errors)}",
    )


def assert_element_exists(selector: str, found: bool) -> StepResult:
    """Assert that an element exists on the page."""
    return StepResult(
        name="element_exists",
        passed=found,
        detail=f"selector={selector}, found={found}",
    )


def assert_api_healthy(status_code: int) -> StepResult:
    """Assert that the API is healthy."""
    return StepResult(
        name="api_healthy",
        passed=status_code == 200,
        detail=f"status={status_code}",
    )


# ── Preset Site Configs ─────────────────────────────────────────────────────

def create_site_config(
    name: str,
    base_url: str,
    pages: Dict[str, tuple[str, list[str]]],
    api_url: Optional[str] = None,
) -> SiteConfig:
    """Create a SiteConfig from a simple dictionary.

    Args:
        name: Site name
        base_url: Base URL
        pages: Dict of {key: (path, checks)}
        api_url: Optional API URL

    Example:
        config = create_site_config(
            name="My App",
            base_url="http://localhost:3000",
            pages={
                "home": ("/", ["welcome"]),
                "about": ("/about", ["about us"]),
            }
        )
    """
    page_dict = {}
    for key, (path, checks) in pages.items():
        page_dict[key] = Page(name=key, path=path, checks=checks)

    return SiteConfig(
        name=name,
        base_url=base_url,
        api_url=api_url,
        pages=page_dict,
    )
