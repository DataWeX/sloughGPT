"""
Tests for the UI Journey Testing Library (legacy wrapper).

Usage:
    .venv/bin/python -m pytest tests/test_ui_journey_library.py -x -v
"""


class TestJourneyCore:
    """Tests for the core Journey class."""



    def test_journey_result_to_dict(self):
        from domain.testing import Journey, SiteConfig

        config = SiteConfig(name="Test", base_url="http://localhost:5173")
        journey = Journey(config)
        result = journey.run([journey.goto("/")])
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["passed"] is True

    def test_report_generation(self):
        from domain.testing import Journey, SiteConfig

        config = SiteConfig(name="Test", base_url="http://localhost:5173")
        journey = Journey(config)
        journey.journey("test1", [journey.goto("/")]) if hasattr(
            journey, "journey"
        ) else journey.run([journey.goto("/")])
        report = journey.report()
        assert "Test" in report
        assert "Report" in report


class TestStepResult:
    """Tests for StepResult."""


    def test_step_result_failed(self):
        from domain.testing import StepResult

        result = StepResult(name="test", passed=False, detail="failed")
        assert not result.passed


class TestJourneySteps:
    """Tests for individual journey steps."""






    def test_check_no_console_errors_step(self):
        from domain.testing import Journey, SiteConfig

        config = SiteConfig(name="Test", base_url="http://localhost:5173")
        journey = Journey(config)
        step = journey.check_no_console_errors()
        result = step()
        assert result.passed

    def test_check_no_network_errors_step(self):
        from domain.testing import Journey, SiteConfig

        config = SiteConfig(name="Test", base_url="http://localhost:5173")
        journey = Journey(config)
        step = journey.check_no_network_errors()
        result = step()
        assert result.passed


class TestJourneyResults:
    """Tests for journey result handling."""

    def test_all_passed(self):
        from domain.testing import Journey, SiteConfig

        config = SiteConfig(name="Test", base_url="http://localhost:5173")
        journey = Journey(config)
        result = journey.run(
            [
                journey.goto("/"),
                journey.goto("/chat"),
            ]
        )
        assert result.passed
        assert result.failed_count == 0

    def test_one_failed(self):
        from domain.testing import Journey, SiteConfig, StepResult

        config = SiteConfig(name="Test", base_url="http://localhost:5173")
        journey = Journey(config)

        def failing_step():
            return StepResult(name="fail", passed=False, detail="failed")

        step = journey.custom("failing_step", failing_step)
        result = journey.run([step])
        assert not result.passed
        assert result.failed_count == 1

    def test_multiple_results(self):
        from domain.testing import Journey, SiteConfig

        config = SiteConfig(name="Test", base_url="http://localhost:5173")
        journey = Journey(config)
        journey.run([journey.goto("/")])
        journey.run([journey.goto("/chat")])
        assert len(journey.results) == 2


