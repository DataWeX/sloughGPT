"""Tests for internal test framework — TestFramework, BenchmarkRunner, mark_test.

Covers:
  - TestResult and TestSuite dataclass structure
  - TestFramework register/run/get_summary
  - BenchmarkRunner.run_benchmark timing
  - mark_test decorator
"""

import time
import pytest
from domains.shared.test_framework import (
    TestFramework,
    TestResult,
    TestSuite,
    BenchmarkRunner,
    mark_test,
)


class TestTestResult:
    def test_dataclass_fields(self):
        r = TestResult(name="t1", status="passed", execution_time=0.1)
        assert r.name == "t1"
        assert r.status == "passed"
        assert r.execution_time == 0.1
        assert r.error_message is None
        assert r.metrics == {}
        assert r.details == {}

    def test_with_error(self):
        r = TestResult(name="t2", status="failed", execution_time=0.01, error_message="boom")
        assert r.error_message == "boom"

    def test_not_a_pytest_test(self):
        assert TestResult.__test__ is False


class TestTestSuite:
    def test_dataclass_fields(self):
        s = TestSuite(
            name="suite",
            tests=[],
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            skipped_tests=0,
            total_execution_time=0.0,
        )
        assert s.name == "suite"
        assert s.coverage_percentage == 0.0


class TestTestFramework:
    def test_register_and_run(self):
        fw = TestFramework("test_suite")
        fw.register(lambda: None)
        fw.register(lambda: None)
        suite = fw.run()
        assert suite.total_tests == 2
        assert suite.passed_tests == 2
        assert suite.failed_tests == 0
        assert suite.total_execution_time >= 0

    def test_failed_test(self):
        fw = TestFramework()
        fw.register(lambda: (_ for _ in ()).throw(ValueError("bad")))
        suite = fw.run()
        assert suite.failed_tests == 1
        assert suite.passed_tests == 0
        assert "bad" in suite.tests[0].error_message

    def test_empty_suite(self):
        fw = TestFramework()
        suite = fw.run()
        assert suite.total_tests == 0
        assert suite.passed_tests == 0
        assert suite.skipped_tests == 0

    def test_mixed_results(self):
        fw = TestFramework()
        fw.register(lambda: None)
        fw.register(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        fw.register(lambda: None)
        suite = fw.run()
        assert suite.passed_tests == 2
        assert suite.failed_tests == 1
        assert suite.total_tests == 3

    def test_result_timing(self):
        fw = TestFramework()
        fw.register(lambda: time.sleep(0.01))
        suite = fw.run()
        assert suite.tests[0].execution_time >= 0.01

    def test_suite_name(self):
        fw = TestFramework("my_suite")
        suite = fw.run()
        assert suite.name == "my_suite"


class TestGetSummary:
    def test_summary_keys(self):
        fw = TestFramework("summary_test")
        fw.register(lambda: None)
        fw.register(lambda: (_ for _ in ()).throw(RuntimeError("x")))
        suite = fw.run()
        summary = fw.get_summary(suite)
        assert summary["name"] == "summary_test"
        assert summary["total"] == 2
        assert summary["passed"] == 1
        assert summary["failed"] == 1
        assert summary["skipped"] == 0
        assert summary["execution_time"] >= 0
        assert 0 <= summary["pass_rate"] <= 100

    def test_pass_rate_calculation(self):
        fw = TestFramework()
        fw.register(lambda: None)
        fw.register(lambda: None)
        suite = fw.run()
        summary = fw.get_summary(suite)
        assert summary["pass_rate"] == 100.0


class TestMarkTest:
    def test_sets_flag(self):
        @mark_test
        def my_test():
            pass
        assert my_test._is_test is True

    def test_preserves_function(self):
        @mark_test
        def my_test():
            return 42
        assert my_test() == 42


class TestBenchmarkRunner:
    def test_run_benchmark(self):
        br = BenchmarkRunner()
        result = br.run_benchmark("fast_fn", lambda: None, iterations=10)
        assert result["name"] == "fast_fn"
        assert result["iterations"] == 10
        assert result["mean_time"] >= 0
        assert result["min_time"] >= 0
        assert result["max_time"] >= result["min_time"]
        assert result["total_time"] >= 0

    def test_results_accumulate(self):
        br = BenchmarkRunner()
        br.run_benchmark("a", lambda: None, iterations=5)
        br.run_benchmark("b", lambda: None, iterations=10)
        assert len(br.results) == 2

    def test_benchmark_timing(self):
        br = BenchmarkRunner()
        result = br.run_benchmark("slow_fn", lambda: time.sleep(0.005), iterations=3)
        assert result["mean_time"] >= 0.004
        assert result["min_time"] >= 0.003
