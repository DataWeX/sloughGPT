"""Backward-compatibility shim — imports from the new ``domain.testing`` package."""

from domain.testing import (
    Browser,
    Journey,
    JourneyResult,
    Page,
    SiteConfig,
    Step,
    StepResult,
    assert_api_healthy,
    assert_body_contains,
    assert_element_exists,
    assert_no_errors,
    assert_page_loads,
    create_site_config,
)

__all__ = [
    "StepResult",
    "JourneyResult",
    "Page",
    "SiteConfig",
    "Browser",
    "Step",
    "Journey",
    "assert_page_loads",
    "assert_body_contains",
    "assert_no_errors",
    "assert_element_exists",
    "assert_api_healthy",
    "create_site_config",
]
