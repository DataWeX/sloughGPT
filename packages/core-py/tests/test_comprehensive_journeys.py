"""
Comprehensive UI Journey Tests for sloughGPT.

Tests all pages, navigation flows, and API health using the internal
UI journey testing library.

Usage:
    # Requires running servers:
    #   API: localhost:8000
    #   Web: localhost:3000
    FORCE_COLOR=1 .venv/bin/python -m pytest tests/test_comprehensive_journeys.py -x -v

    # Dry-run (no servers needed):
    .venv/bin/python -m pytest tests/test_comprehensive_journeys.py -x -v -k "not live"
"""

from __future__ import annotations

import pytest

from domain.testing._internal.journeys import (
    APIJourney,
    FullSuiteJourney,
    NavigationJourney,
    PageJourney,
)
from domain.testing._internal.sloughgpt import (
    SLOUGHPGPT_API_ENDPOINTS,
    SLOUGHPGPT_PAGES,
    SLOUGHPGPT_SITE,
    SloughGPTJourney,
)

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def journey():
    return SloughGPTJourney()


@pytest.fixture(scope="module")
def page_journey():
    return PageJourney(SLOUGHPGPT_SITE)


@pytest.fixture(scope="module")
def nav_journey():
    return NavigationJourney(SLOUGHPGPT_SITE)


@pytest.fixture(scope="module")
def api_journey():
    return APIJourney(SLOUGHPGPT_SITE)


# ── Config Tests (no servers) ──────────────────────────────────────────────


class TestSiteConfig:
    """Verify site configuration is correct."""

    def test_config_name(self):
        assert SLOUGHPGPT_SITE.name == "sloughGPT"

    def test_config_base_url(self):
        assert SLOUGHPGPT_SITE.base_url == "http://localhost:3000"

    def test_config_api_url(self):
        assert SLOUGHPGPT_SITE.api_url == "http://localhost:8000"

    def test_all_pages_registered(self):
        assert len(SLOUGHPGPT_PAGES) >= 100

    def test_core_pages_exist(self):
        for key in ["dashboard", "chat", "training", "models", "settings"]:
            assert key in SLOUGHPGPT_PAGES, f"Missing core page: {key}"

    def test_training_subpages_exist(self):
        for key in [
            "training/runs",
            "training/presets",
            "training/analytics",
            "training/compare",
            "training/trends",
            "training/insights",
            "training/model-card",
            "training/grid-search",
        ]:
            assert key in SLOUGHPGPT_PAGES, f"Missing training page: {key}"

    def test_consciousness_subpages_exist(self):
        for key in [
            "consciousness/dashboard",
            "consciousness/debug",
            "consciousness/testing",
            "consciousness/analytics",
            "consciousness/insights",
            "consciousness/monitor",
        ]:
            assert key in SLOUGHPGPT_PAGES, f"Missing consciousness page: {key}"

    def test_api_endpoints_configured(self):
        assert "health" in SLOUGHPGPT_API_ENDPOINTS
        assert "training_status" in SLOUGHPGPT_API_ENDPOINTS

    def test_pages_have_valid_paths(self):
        for key, (path, checks) in SLOUGHPGPT_PAGES.items():
            assert path == "" or path.startswith("/"), (
                f"Page {key} path must be empty or start with /: {path}"
            )
            assert len(checks) > 0, f"Page {key} must have at least one check"

    def test_duplicate_page_keys_no_overlap(self):
        keys = list(SLOUGHPGPT_PAGES.keys())
        assert len(keys) == len(set(keys)), "Duplicate page keys found"


# ── Page Journey Tests (live) ──────────────────────────────────────────────


@pytest.mark.live
class TestAllPages:
    """Test every page loads without errors."""

    def test_dashboard(self, journey):
        result = journey.run(
            [
                journey.goto("/"),
                journey.check_body("chat"),
                journey.check_no_console_errors(),
            ],
            name="page_dashboard",
        )
        assert result.passed

    def test_chat(self, journey):
        result = journey.run(
            [
                journey.goto("/chat"),
                journey.check_body("chat"),
                journey.check_no_console_errors(),
            ],
            name="page_chat",
        )
        assert result.passed

    def test_training(self, journey):
        result = journey.run(
            [
                journey.goto("/training"),
                journey.check_body("train"),
                journey.check_no_console_errors(),
            ],
            name="page_training",
        )
        assert result.passed

    def test_training_runs(self, journey):
        result = journey.run(
            [
                journey.goto("/training/runs"),
                journey.check_no_console_errors(),
            ],
            name="page_training_runs",
        )
        assert result.passed

    def test_training_presets(self, journey):
        result = journey.run(
            [
                journey.goto("/training/presets"),
                journey.check_no_console_errors(),
            ],
            name="page_training_presets",
        )
        assert result.passed

    def test_training_analytics(self, journey):
        result = journey.run(
            [
                journey.goto("/training/analytics"),
                journey.check_no_console_errors(),
            ],
            name="page_training_analytics",
        )
        assert result.passed

    def test_training_compare(self, journey):
        result = journey.run(
            [
                journey.goto("/training/compare"),
                journey.check_no_console_errors(),
            ],
            name="page_training_compare",
        )
        assert result.passed

    def test_training_trends(self, journey):
        result = journey.run(
            [
                journey.goto("/training/trends"),
                journey.check_no_console_errors(),
            ],
            name="page_training_trends",
        )
        assert result.passed

    def test_training_insights(self, journey):
        result = journey.run(
            [
                journey.goto("/training/insights"),
                journey.check_no_console_errors(),
            ],
            name="page_training_insights",
        )
        assert result.passed

    def test_training_model_card(self, journey):
        result = journey.run(
            [
                journey.goto("/training/model-card"),
                journey.check_no_console_errors(),
            ],
            name="page_training_model_card",
        )
        assert result.passed

    def test_training_grid_search(self, journey):
        result = journey.run(
            [
                journey.goto("/training/grid-search"),
                journey.check_no_console_errors(),
            ],
            name="page_training_grid_search",
        )
        assert result.passed

    def test_datasets(self, journey):
        result = journey.run(
            [
                journey.goto("/datasets"),
                journey.check_body("dataset"),
                journey.check_no_console_errors(),
            ],
            name="page_datasets",
        )
        assert result.passed

    def test_files(self, journey):
        result = journey.run(
            [
                journey.goto("/files"),
                journey.check_no_console_errors(),
            ],
            name="page_files",
        )
        assert result.passed

    def test_models(self, journey):
        result = journey.run(
            [
                journey.goto("/models"),
                journey.check_body("model"),
                journey.check_no_console_errors(),
            ],
            name="page_models",
        )
        assert result.passed

    def test_adapters(self, journey):
        result = journey.run(
            [
                journey.goto("/adapters"),
                journey.check_no_console_errors(),
            ],
            name="page_adapters",
        )
        assert result.passed

    def test_agents(self, journey):
        result = journey.run(
            [
                journey.goto("/agents"),
                journey.check_body("agent"),
                journey.check_no_console_errors(),
            ],
            name="page_agents",
        )
        assert result.passed

    def test_souls(self, journey):
        result = journey.run(
            [
                journey.goto("/souls"),
                journey.check_no_console_errors(),
            ],
            name="page_souls",
        )
        assert result.passed

    def test_knowledge(self, journey):
        result = journey.run(
            [
                journey.goto("/knowledge"),
                journey.check_no_console_errors(),
            ],
            name="page_knowledge",
        )
        assert result.passed

    def test_monitoring(self, journey):
        result = journey.run(
            [
                journey.goto("/monitoring"),
                journey.check_no_console_errors(),
            ],
            name="page_monitoring",
        )
        assert result.passed

    def test_settings(self, journey):
        result = journey.run(
            [
                journey.goto("/settings"),
                journey.check_no_console_errors(),
            ],
            name="page_settings",
        )
        assert result.passed

    def test_shell(self, journey):
        result = journey.run(
            [
                journey.goto("/shell"),
                journey.check_no_console_errors(),
            ],
            name="page_shell",
        )
        assert result.passed

    def test_planner(self, journey):
        result = journey.run(
            [
                journey.goto("/planner"),
                journey.check_no_console_errors(),
            ],
            name="page_planner",
        )
        assert result.passed

    def test_benchmark(self, journey):
        result = journey.run(
            [
                journey.goto("/benchmark"),
                journey.check_no_console_errors(),
            ],
            name="page_benchmark",
        )
        assert result.passed

    def test_tokenizer(self, journey):
        result = journey.run(
            [
                journey.goto("/tokenizer"),
                journey.check_no_console_errors(),
            ],
            name="page_tokenizer",
        )
        assert result.passed

    def test_errors(self, journey):
        result = journey.run(
            [
                journey.goto("/errors"),
                journey.check_no_console_errors(),
            ],
            name="page_errors",
        )
        assert result.passed

    def test_security(self, journey):
        result = journey.run(
            [
                journey.goto("/security"),
                journey.check_no_console_errors(),
            ],
            name="page_security",
        )
        assert result.passed

    def test_feedback(self, journey):
        result = journey.run(
            [
                journey.goto("/feedback"),
                journey.check_no_console_errors(),
            ],
            name="page_feedback",
        )
        assert result.passed

    def test_consciousness_dashboard(self, journey):
        result = journey.run(
            [
                journey.goto("/consciousness/dashboard"),
                journey.check_no_console_errors(),
            ],
            name="page_consciousness_dashboard",
        )
        assert result.passed

    def test_consciousness_debug(self, journey):
        result = journey.run(
            [
                journey.goto("/consciousness/debug"),
                journey.check_no_console_errors(),
            ],
            name="page_consciousness_debug",
        )
        assert result.passed

    def test_consciousness_testing(self, journey):
        result = journey.run(
            [
                journey.goto("/consciousness/testing"),
                journey.check_no_console_errors(),
            ],
            name="page_consciousness_testing",
        )
        assert result.passed

    def test_consciousness_analytics(self, journey):
        result = journey.run(
            [
                journey.goto("/consciousness/analytics"),
                journey.check_no_console_errors(),
            ],
            name="page_consciousness_analytics",
        )
        assert result.passed

    def test_consciousness_insights(self, journey):
        result = journey.run(
            [
                journey.goto("/consciousness/insights"),
                journey.check_no_console_errors(),
            ],
            name="page_consciousness_insights",
        )
        assert result.passed

    def test_consciousness_monitor(self, journey):
        result = journey.run(
            [
                journey.goto("/consciousness/monitor"),
                journey.check_no_console_errors(),
            ],
            name="page_consciousness_monitor",
        )
        assert result.passed

    def test_consciousness_playground(self, journey):
        result = journey.run(
            [
                journey.goto("/consciousness/playground"),
                journey.check_no_console_errors(),
            ],
            name="page_consciousness_playground",
        )
        assert result.passed

    def test_consciousness_benchmark(self, journey):
        result = journey.run(
            [
                journey.goto("/consciousness/benchmark"),
                journey.check_no_console_errors(),
            ],
            name="page_consciousness_benchmark",
        )
        assert result.passed

    def test_consciousness_versions(self, journey):
        result = journey.run(
            [
                journey.goto("/consciousness/versions"),
                journey.check_no_console_errors(),
            ],
            name="page_consciousness_versions",
        )
        assert result.passed

    def test_infer(self, journey):
        result = journey.run(
            [
                journey.goto("/infer"),
                journey.check_no_console_errors(),
            ],
            name="page_infer",
        )
        assert result.passed

    def test_evaluate(self, journey):
        result = journey.run(
            [
                journey.goto("/evaluate"),
                journey.check_no_console_errors(),
            ],
            name="page_evaluate",
        )
        assert result.passed

    def test_compare(self, journey):
        result = journey.run(
            [
                journey.goto("/compare"),
                journey.check_no_console_errors(),
            ],
            name="page_compare",
        )
        assert result.passed

    def test_export(self, journey):
        result = journey.run(
            [
                journey.goto("/export"),
                journey.check_no_console_errors(),
            ],
            name="page_export",
        )
        assert result.passed

    def test_vector(self, journey):
        result = journey.run(
            [
                journey.goto("/vector"),
                journey.check_no_console_errors(),
            ],
            name="page_vector",
        )
        assert result.passed

    def test_multimodal(self, journey):
        result = journey.run(
            [
                journey.goto("/multimodal"),
                journey.check_no_console_errors(),
            ],
            name="page_multimodal",
        )
        assert result.passed

    def test_workflow(self, journey):
        result = journey.run(
            [
                journey.goto("/workflow"),
                journey.check_no_console_errors(),
            ],
            name="page_workflow",
        )
        assert result.passed

    def test_kanban(self, journey):
        result = journey.run(
            [
                journey.goto("/kanban"),
                journey.check_no_console_errors(),
            ],
            name="page_kanban",
        )
        assert result.passed

    def test_memory(self, journey):
        result = journey.run(
            [
                journey.goto("/memory"),
                journey.check_no_console_errors(),
            ],
            name="page_memory",
        )
        assert result.passed

    def test_vm(self, journey):
        result = journey.run(
            [
                journey.goto("/vm"),
                journey.check_no_console_errors(),
            ],
            name="page_vm",
        )
        assert result.passed


# ── Navigation Journey Tests (live) ────────────────────────────────────────


@pytest.mark.live
class TestNavigationFlows:
    """Test navigation between related pages."""

    def test_dashboard_to_chat(self, journey):
        result = journey.run(
            [
                journey.goto("/"),
                journey.check_body("chat"),
                journey.goto("/chat"),
                journey.check_body("chat"),
            ],
            name="nav_dashboard_chat",
        )
        assert result.passed

    def test_training_flow(self, journey):
        result = journey.run(
            [
                journey.goto("/training"),
                journey.check_body("train"),
                journey.goto("/training/runs"),
                journey.goto("/training/presets"),
                journey.goto("/training/analytics"),
            ],
            name="nav_training_flow",
        )
        assert result.passed

    def test_model_management_flow(self, journey):
        result = journey.run(
            [
                journey.goto("/models"),
                journey.check_body("model"),
                journey.goto("/adapters"),
                journey.goto("/export"),
            ],
            name="nav_model_flow",
        )
        assert result.passed

    def test_data_management_flow(self, journey):
        result = journey.run(
            [
                journey.goto("/datasets"),
                journey.check_body("dataset"),
                journey.goto("/files"),
                journey.goto("/docstore"),
            ],
            name="nav_data_flow",
        )
        assert result.passed

    def test_consciousness_flow(self, journey):
        result = journey.run(
            [
                journey.goto("/consciousness/dashboard"),
                journey.goto("/consciousness/debug"),
                journey.goto("/consciousness/testing"),
                journey.goto("/consciousness/analytics"),
            ],
            name="nav_consciousness_flow",
        )
        assert result.passed

    def test_admin_flow(self, journey):
        result = journey.run(
            [
                journey.goto("/settings"),
                journey.goto("/security"),
                journey.goto("/audit-trail"),
                journey.goto("/users"),
            ],
            name="nav_admin_flow",
        )
        assert result.passed

    def test_inference_flow(self, journey):
        result = journey.run(
            [
                journey.goto("/infer"),
                journey.goto("/evaluate"),
                journey.goto("/benchmark"),
                journey.goto("/compare"),
            ],
            name="nav_inference_flow",
        )
        assert result.passed


# ── API Journey Tests (live) ───────────────────────────────────────────────


@pytest.mark.live
class TestAPIHealth:
    """Test API health endpoints."""

    def test_health(self, api_journey):
        result = api_journey.test_health()
        assert result.passed

    def test_all_endpoints(self, api_journey):
        result = api_journey.test_all_endpoints(SLOUGHPGPT_API_ENDPOINTS)
        assert result.passed


# ── Full Suite (live) ──────────────────────────────────────────────────────


@pytest.mark.live
class TestFullSuite:
    """Run the complete test suite."""

    def test_full_suite(self):
        suite = FullSuiteJourney(SLOUGHPGPT_SITE)
        results = suite.run_all()
        for result in results:
            assert result.passed, f"Suite failed: {result.name}"

    def test_full_report(self):
        suite = FullSuiteJourney(SLOUGHPGPT_SITE)
        report = suite.report()
        assert "sloughGPT" in report
        assert "Total:" in report
