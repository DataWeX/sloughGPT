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


# =============================================================================
# TestResult
# =============================================================================

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

    def test_with_metrics(self):
        r = TestResult(
            name="t3", status="passed", execution_time=0.05,
            metrics={"memory_mb": 12.5, "cpu_pct": 45.0}
        )
        assert r.metrics["memory_mb"] == 12.5
        assert r.metrics["cpu_pct"] == 45.0

    def test_with_details(self):
        r = TestResult(
            name="t4", status="failed", execution_time=0.01,
            error_message="assertion error",
            details={"line": 42, "file": "test.py"}
        )
        assert r.details["line"] == 42
        assert r.details["file"] == "test.py"

    def test_status_values(self):
        for status in ("passed", "failed", "skipped", "error"):
            r = TestResult(name="t", status=status, execution_time=0.0)
            assert r.status == status

    def test_execution_time_zero(self):
        r = TestResult(name="t", status="passed", execution_time=0.0)
        assert r.execution_time == 0.0

    def test_execution_time_negative(self):
        r = TestResult(name="t", status="passed", execution_time=-0.1)
        assert r.execution_time == -0.1

    def test_name_empty(self):
        r = TestResult(name="", status="passed", execution_time=0.0)
        assert r.name == ""

    def test_error_message_empty_string(self):
        r = TestResult(name="t", status="failed", execution_time=0.0, error_message="")
        assert r.error_message == ""

    def test_metrics_default_factory(self):
        r1 = TestResult(name="a", status="passed", execution_time=0.0)
        r2 = TestResult(name="b", status="passed", execution_time=0.0)
        assert r1.metrics is not r2.metrics  # separate dicts

    def test_details_default_factory(self):
        r1 = TestResult(name="a", status="passed", execution_time=0.0)
        r2 = TestResult(name="b", status="passed", execution_time=0.0)
        assert r1.details is not r2.details

    def test_equality(self):
        r1 = TestResult(name="t", status="passed", execution_time=0.1)
        r2 = TestResult(name="t", status="passed", execution_time=0.1)
        assert r1 == r2

    def test_inequality(self):
        r1 = TestResult(name="t1", status="passed", execution_time=0.1)
        r2 = TestResult(name="t2", status="passed", execution_time=0.1)
        assert r1 != r2


# =============================================================================
# TestSuite
# =============================================================================

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

    def test_with_tests(self):
        r1 = TestResult(name="t1", status="passed", execution_time=0.1)
        r2 = TestResult(name="t2", status="failed", execution_time=0.2)
        s = TestSuite(
            name="s", tests=[r1, r2],
            total_tests=2, passed_tests=1, failed_tests=1,
            skipped_tests=0, total_execution_time=0.3
        )
        assert len(s.tests) == 2
        assert s.total_tests == 2

    def test_coverage_percentage_default(self):
        s = TestSuite(
            name="s", tests=[], total_tests=0,
            passed_tests=0, failed_tests=0, skipped_tests=0,
            total_execution_time=0.0
        )
        assert s.coverage_percentage == 0.0

    def test_coverage_percentage_custom(self):
        s = TestSuite(
            name="s", tests=[], total_tests=10,
            passed_tests=10, failed_tests=0, skipped_tests=0,
            total_execution_time=0.5, coverage_percentage=85.5
        )
        assert s.coverage_percentage == 85.5

    def test_not_a_pytest_test(self):
        assert TestSuite.__test__ is False

    def test_skipped_tests(self):
        s = TestSuite(
            name="s", tests=[], total_tests=5,
            passed_tests=3, failed_tests=1, skipped_tests=1,
            total_execution_time=0.1
        )
        assert s.skipped_tests == 1

    def test_total_execution_time(self):
        s = TestSuite(
            name="s", tests=[], total_tests=0,
            passed_tests=0, failed_tests=0, skipped_tests=0,
            total_execution_time=1.23
        )
        assert s.total_execution_time == 1.23

    def test_tests_list_is_mutable(self):
        r = TestResult(name="t", status="passed", execution_time=0.0)
        s = TestSuite(
            name="s", tests=[r], total_tests=1,
            passed_tests=1, failed_tests=0, skipped_tests=0,
            total_execution_time=0.0
        )
        r2 = TestResult(name="t2", status="failed", execution_time=0.0)
        s.tests.append(r2)
        assert len(s.tests) == 2


# =============================================================================
# TestFramework
# =============================================================================

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

    def test_default_suite_name(self):
        fw = TestFramework()
        suite = fw.run()
        assert suite.name == "TestSuite"

    def test_result_names(self):
        fw = TestFramework()
        fw.register(lambda: None)
        fw.register(lambda: None)
        suite = fw.run()
        assert suite.tests[0].name == "<lambda>"
        assert suite.tests[1].name == "<lambda>"

    def test_results_in_suite(self):
        fw = TestFramework()
        fw.register(lambda: None)
        suite = fw.run()
        assert len(suite.tests) == 1
        assert suite.tests[0].status == "passed"

    def test_multiple_runs(self):
        fw = TestFramework()
        fw.register(lambda: None)
        fw.run()
        fw.register(lambda: (_ for _ in ()).throw(RuntimeError("x")))
        suite = fw.run()
        assert suite.total_tests == 2
        assert suite.failed_tests == 1

    def test_exception_types(self):
        fw = TestFramework()
        fw.register(lambda: (_ for _ in ()).throw(ValueError("v")))
        fw.register(lambda: (_ for _ in ()).throw(RuntimeError("r")))
        fw.register(lambda: (_ for _ in ()).throw(TypeError("t")))
        suite = fw.run()
        assert suite.failed_tests == 3
        errors = [t.error_message for t in suite.tests]
        assert "v" in errors[0]
        assert "r" in errors[1]
        assert "t" in errors[2]

    def test_fast_execution(self):
        fw = TestFramework()
        fw.register(lambda: None)
        suite = fw.run()
        assert suite.total_execution_time < 1.0

    def test_tests_list_populated(self):
        fw = TestFramework()
        fw.register(lambda: None)
        fw.register(lambda: (_ for _ in ()).throw(RuntimeError()))
        suite = fw.run()
        assert len(suite.tests) == 2
        assert isinstance(suite.tests[0], TestResult)

    def test_result_status_passed(self):
        fw = TestFramework()
        fw.register(lambda: None)
        suite = fw.run()
        assert suite.tests[0].status == "passed"
        assert suite.tests[0].error_message is None

    def test_result_status_failed(self):
        fw = TestFramework()
        fw.register(lambda: (_ for _ in ()).throw(AssertionError()))
        suite = fw.run()
        assert suite.tests[0].status == "failed"
        assert suite.tests[0].error_message is not None

    def test_framework_stores_all_results_in_suite(self):
        fw = TestFramework()
        fw.register(lambda: None)
        fw.register(lambda: None)
        suite = fw.run()
        assert len(suite.tests) == 2

    def test_suite_total_matches_registered(self):
        fw = TestFramework()
        for i in range(10):
            fw.register(lambda: None)
        suite = fw.run()
        assert suite.total_tests == 10
        assert len(suite.tests) == 10


# =============================================================================
# GetSummary
# =============================================================================

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

    def test_pass_rate_zero(self):
        fw = TestFramework()
        fw.register(lambda: (_ for _ in ()).throw(RuntimeError("x")))
        suite = fw.run()
        summary = fw.get_summary(suite)
        assert summary["pass_rate"] == 0.0

    def test_pass_rate_partial(self):
        fw = TestFramework()
        fw.register(lambda: None)
        fw.register(lambda: None)
        fw.register(lambda: (_ for _ in ()).throw(RuntimeError()))
        fw.register(lambda: None)
        suite = fw.run()
        summary = fw.get_summary(suite)
        assert summary["pass_rate"] == 75.0

    def test_summary_execution_time_non_negative(self):
        fw = TestFramework()
        fw.register(lambda: None)
        suite = fw.run()
        summary = fw.get_summary(suite)
        assert summary["execution_time"] >= 0

    def test_summary_empty_suite(self):
        fw = TestFramework("empty")
        suite = fw.run()
        summary = fw.get_summary(suite)
        assert summary["total"] == 0
        assert summary["pass_rate"] == 0.0  # 0 / max(1, 0) * 100 = 0

    def test_summary_has_all_keys(self):
        fw = TestFramework()
        fw.register(lambda: None)
        suite = fw.run()
        summary = fw.get_summary(suite)
        expected_keys = {"name", "total", "passed", "failed", "skipped", "execution_time", "pass_rate"}
        assert set(summary.keys()) == expected_keys

    def test_summary_name_matches(self):
        fw = TestFramework("custom_name")
        suite = fw.run()
        summary = fw.get_summary(suite)
        assert summary["name"] == "custom_name"


# =============================================================================
# MarkTest
# =============================================================================

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

    def test_preserves_docstring(self):
        @mark_test
        def my_test():
            """My docstring."""
            pass
        assert my_test.__doc__ == "My docstring."

    def test_preserves_name(self):
        @mark_test
        def my_test():
            pass
        assert my_test.__name__ == "my_test"

    def test_mark_with_args(self):
        @mark_test
        def my_test(a, b):
            return a + b
        assert my_test(3, 4) == 7

    def test_mark_with_kwargs(self):
        @mark_test
        def my_test(x=10):
            return x * 2
        assert my_test() == 20
        assert my_test(x=5) == 10

    def test_mark_preserves_exception(self):
        @mark_test
        def my_test():
            raise ValueError("test error")
        with pytest.raises(ValueError, match="test error"):
            my_test()

    def test_mark_is_decorator(self):
        @mark_test
        def f():
            pass
        assert callable(f)
        assert f._is_test is True

    def test_mark_on_class_method(self):
        class MyClass:
            @mark_test
            def my_method(self):
                return 99
        obj = MyClass()
        assert obj.my_method() == 99
        assert obj.my_method._is_test is True

    def test_mark_with_return_none(self):
        @mark_test
        def my_test():
            return None
        assert my_test() is None


# =============================================================================
# BenchmarkRunner
# =============================================================================

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

    def test_benchmark_keys(self):
        br = BenchmarkRunner()
        result = br.run_benchmark("test", lambda: None, iterations=1)
        expected_keys = {"name", "iterations", "mean_time", "min_time", "max_time", "total_time"}
        assert set(result.keys()) == expected_keys

    def test_benchmark_single_iteration(self):
        br = BenchmarkRunner()
        result = br.run_benchmark("single", lambda: None, iterations=1)
        assert result["iterations"] == 1
        assert result["mean_time"] == result["min_time"] == result["max_time"]

    def test_benchmark_total_equals_sum(self):
        br = BenchmarkRunner()
        result = br.run_benchmark("sum_test", lambda: None, iterations=5)
        assert abs(result["total_time"] - result["mean_time"] * result["iterations"]) < 1e-10

    def test_benchmark_results_stored(self):
        br = BenchmarkRunner()
        br.run_benchmark("a", lambda: None, iterations=3)
        assert len(br.results) == 1
        assert br.results[0]["name"] == "a"

    def test_benchmark_with_computation(self):
        br = BenchmarkRunner()
        result = br.run_benchmark("compute", lambda: sum(range(100)), iterations=10)
        assert result["mean_time"] >= 0
        assert result["iterations"] == 10

    def test_benchmark_min_le_mean(self):
        br = BenchmarkRunner()
        result = br.run_benchmark("test", lambda: time.sleep(0.001), iterations=5)
        assert result["min_time"] <= result["mean_time"]

    def test_benchmark_mean_le_max(self):
        br = BenchmarkRunner()
        result = br.run_benchmark("test", lambda: time.sleep(0.001), iterations=5)
        assert result["mean_time"] <= result["max_time"]

    def test_benchmark_empty_results_initially(self):
        br = BenchmarkRunner()
        assert br.results == []

    def test_benchmark_iterations_correct(self):
        br = BenchmarkRunner()
        result = br.run_benchmark("test", lambda: None, iterations=7)
        assert result["iterations"] == 7

    def test_benchmark_name_preserved(self):
        br = BenchmarkRunner()
        result = br.run_benchmark("my_benchmark_name", lambda: None, iterations=1)
        assert result["name"] == "my_benchmark_name"
