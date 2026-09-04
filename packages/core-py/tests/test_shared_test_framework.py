"""Tests for TestFramework — test runner and benchmark utilities."""
from __future__ import annotations

from domains.shared.test_framework import (
    BenchmarkRunner,
    TestFramework,
    TestResult,
    TestSuite,
    mark_test,
)


class TestTestResult:
    def test_creation(self):
        r = TestResult(name="test1", status="passed", execution_time=0.1)
        assert r.name == "test1"
        assert r.status == "passed"
        assert r.error_message is None


class TestTestSuite:
    def test_creation(self):
        s = TestSuite(name="suite", tests=[], total_tests=0, passed_tests=0, failed_tests=0, skipped_tests=0, total_execution_time=0.0)
        assert s.name == "suite"
        assert s.total_tests == 0


class TestTestFramework:
    def test_register_and_run(self):
        fw = TestFramework("my_tests")
        fw.register(lambda: None)
        suite = fw.run()
        assert suite.total_tests == 1
        assert suite.passed_tests == 1

    def test_catches_failure(self):
        fw = TestFramework()
        fw.register(lambda: (_ for _ in ()).throw(ValueError("bad")))
        suite = fw.run()
        assert suite.failed_tests == 1
        assert "bad" in suite.tests[0].error_message

    def test_get_summary(self):
        fw = TestFramework()
        fw.register(lambda: None)
        suite = fw.run()
        summary = fw.get_summary(suite)
        assert summary["passed"] == 1
        assert summary["pass_rate"] == 100.0


class TestMarkTest:
    def test_marks_function(self):
        @mark_test
        def my_test():
            pass
        assert getattr(my_test, "_is_test", False) is True


class TestBenchmarkRunner:
    def test_run_benchmark(self):
        runner = BenchmarkRunner()
        result = runner.run_benchmark("empty", lambda: None, iterations=10)
        assert result["name"] == "empty"
        assert result["iterations"] == 10
        assert result["mean_time"] >= 0
        assert len(runner.results) == 1
