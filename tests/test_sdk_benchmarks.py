"""Coverage for sloughgpt_sdk.benchmarks."""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "sdk-py"))

from sloughgpt_sdk.benchmarks import (  # noqa: E402
    Benchmark,
    BenchmarkResult,
    LoadTestResult,
    LoadTester,
    Profiler,
    benchmark_cache_operations,
    percentile,
)


class TestPercentile:
    def test_empty_returns_zero(self):
        assert percentile([], 95) == 0.0

    def test_returns_sorted_position(self):
        assert percentile([10.0, 20.0, 30.0], 50) == 20.0

    def test_index_capped_at_max(self):
        data = [1.0, 2.0, 3.0]
        assert percentile(data, 100) == 3.0
        assert percentile(data, 101) == 3.0


class TestBenchmarkResult:
    def test_str_format(self):
        r = BenchmarkResult(
            name="op", iterations=10, total_time_ms=100.0, avg_time_ms=10.0,
            min_time_ms=5.0, max_time_ms=20.0, median_time_ms=9.0,
            std_dev_ms=2.0, ops_per_second=100.0, p95_ms=15.0, p99_ms=18.0,
        )
        s = str(r)
        assert "op" in s
        assert "Iterations: 10" in s
        assert "Ops/sec: 100.00" in s

    def test_to_dict(self):
        r = BenchmarkResult(
            name="op", iterations=10, total_time_ms=100.0, avg_time_ms=10.0,
            min_time_ms=5.0, max_time_ms=20.0, median_time_ms=9.0,
            std_dev_ms=2.0, ops_per_second=100.0, p95_ms=15.0, p99_ms=18.0,
        )
        d = r.to_dict()
        assert d["name"] == "op"
        assert d["p95_ms"] == 15.0


class TestLoadTestResult:
    def test_to_dict_truncates_errors(self):
        r = LoadTestResult(
            name="load", concurrent_workers=2, total_requests=5,
            successful_requests=3, failed_requests=2, total_time_ms=100.0,
            requests_per_second=50.0, avg_latency_ms=10.0, min_latency_ms=1.0,
            max_latency_ms=20.0, median_latency_ms=9.0, p95_latency_ms=15.0,
            p99_latency_ms=18.0, success_rate=0.6,
            errors=[f"e{i}" for i in range(20)],
        )
        d = r.to_dict()
        assert d["success_rate"] == 0.6
        assert len(d["errors"]) == 10


class TestBenchmarkRun:
    def test_run_deterministic_stats(self):
        fakes = [0.0, 0.01, 0.02, 0.02, 0.04, 0.04, 0.07]
        with patch("sloughgpt_sdk.benchmarks.time.perf_counter", side_effect=fakes):
            result = Benchmark().run("op", lambda: None, iterations=3, warmup=1)
        assert result.name == "op"
        assert result.iterations == 3
        assert result.total_time_ms == pytest.approx(60.0)
        assert result.avg_time_ms == pytest.approx(20.0)
        assert result.min_time_ms == pytest.approx(10.0)
        assert result.max_time_ms == pytest.approx(30.0)
        assert result.median_time_ms == pytest.approx(20.0)
        assert result.std_dev_ms == pytest.approx(10.0)
        assert result.ops_per_second == pytest.approx(50.0)
        assert result.p95_ms == pytest.approx(30.0)
        assert result.p99_ms == pytest.approx(30.0)

    def test_run_forwards_args_and_kwargs(self):
        seen = []

        def grab(*a, **k):
            seen.append((a, k))

        with patch("sloughgpt_sdk.benchmarks.time.perf_counter", side_effect=[0.0, 0.01, 0.02, 0.02, 0.04]):
            Benchmark().run(
                "op", func=grab, iterations=2, warmup=1,
                args=(1, 2), kwargs={"x": 3},
            )
        assert seen == [((1, 2), {"x": 3})] * 3

    def test_run_single_iteration_zero_stdev(self):
        with patch("sloughgpt_sdk.benchmarks.time.perf_counter", side_effect=[0.0, 0.01, 0.03]):
            result = Benchmark().run(
                "op", func=lambda: None, iterations=1, warmup=0, kwargs={},
            )
        assert result.std_dev_ms == 0

    def test_compare_sorts_by_avg(self):
        def slow():
            import time
            time.sleep(0.002)

        def fast():
            import time
            time.sleep(0.0005)

        results = Benchmark().compare("cmp", {"slow": slow, "fast": fast}, iterations=1)
        assert [r.name for r in results] == ["cmp - fast", "cmp - slow"]
        assert results[0].avg_time_ms <= results[1].avg_time_ms

    def test_memory_benchmark(self):
        report = Benchmark().memory_benchmark("allocs", func=lambda: [0] * 100, iterations=5)
        assert report["name"] == "allocs"
        assert report["iterations"] == 5
        assert "avg_time_ms" in report
        assert "memory_current_mb" in report
        assert "memory_peak_mb" in report


class TestLoadTester:
    def test_load_test_all_success(self):
        calls = []

        def ok():
            calls.append(1)

        result = LoadTester().load_test(
            name="bench", request_func=ok,
            concurrent_workers=3, requests_per_worker=4,
        )
        assert len(calls) == 12
        assert result.concurrent_workers == 3
        assert result.total_requests == 12
        assert result.successful_requests == 12
        assert result.failed_requests == 0
        assert result.success_rate == pytest.approx(1.0)
        assert result.requests_per_second > 0
        assert result.avg_latency_ms >= 0
        assert result.errors == []

    def test_load_test_with_failures(self):
        def flaky():
            raise RuntimeError("boom")

        result = LoadTester().load_test(
            name="bench", request_func=flaky,
            concurrent_workers=2, requests_per_worker=2,
        )
        assert result.total_requests == 4
        assert result.failed_requests == 4
        assert result.success_rate == 0
        assert result.errors == ["boom", "boom", "boom", "boom"]

    def test_stress_test(self):
        result = LoadTester().stress_test(
            name="stress", request_func=lambda: None,
            duration_seconds=0.01, target_rps=1000000,
        )
        assert result["name"] == "stress"
        assert result["total_requests"] >= 1
        assert result["successful"] == result["total_requests"]
        assert "p95_latency_ms" in result
        assert result["errors"] == []

    def test_stress_test_errors(self):
        def bad():
            raise ValueError("nope")

        result = LoadTester().stress_test(
            name="stress", request_func=bad,
            duration_seconds=0.01, target_rps=1000000,
        )
        assert result["failed"] == result["total_requests"]
        assert result["errors"] == ["nope"] or len(result["errors"]) >= 1

    def test_stress_test_pacing_sleeps(self):
        result = LoadTester().stress_test(
            name="pacing", request_func=lambda: None,
            duration_seconds=0.4, target_rps=10,
        )
        assert result["total_requests"] >= 2
        assert result["requests_per_second"] > 0
        assert result["successful"] == result["total_requests"]


class TestProfiler:
    def test_profile_decorator(self):
        prof = Profiler()

        @prof.profile("add")
        def add(a, b):
            return a + b

        assert add(1, 2) == 3
        assert add(4, 5) == 9
        report = prof.get_report()
        assert "add" in report
        assert report["add"]["calls"] == 2
        assert report["add"]["total_ms"] == pytest.approx(
            report["add"]["avg_ms"] * 2)
        assert report["add"]["min_ms"] <= report["add"]["avg_ms"] <= report["add"]["max_ms"]

    def test_profiler_context_manager(self):
        prof = Profiler()
        with prof:
            @prof.profile("ctx")
            def work():
                return "x"

            assert work() == "x"
        assert prof.get_report()["ctx"]["calls"] == 1

    def test_profiler_propagates_exceptions(self):
        prof = Profiler()

        @prof.profile("err")
        def boom():
            raise ValueError("no")

        with pytest.raises(ValueError, match="no"):
            boom()
        assert "err" not in prof.get_report()

    def test_print_report(self, capsys):
        prof = Profiler()

        @prof.profile("printme")
        def f():
            return 1

        f()
        prof.print_report()
        out = capsys.readouterr().out
        assert "PROFILING REPORT" in out
        assert "printme" in out


def test_benchmark_cache_operations():
    result = benchmark_cache_operations()
    assert result.name == "Cache SET"
    assert result.iterations == 10000


def test_load_test_worker_future_error():
    import sloughgpt_sdk.benchmarks as benchmarks_mod

    real = benchmarks_mod.time.perf_counter
    state = {"calls": 0}

    def flaky():
        state["calls"] += 1
        if state["calls"] == 2:
            raise RuntimeError("boom")
        return real()

    with patch.object(benchmarks_mod.time, "perf_counter", flaky):
        result = LoadTester().load_test(
            name="future-error",
            request_func=lambda: None,
            concurrent_workers=2,
            requests_per_worker=1,
        )
    assert result.errors
    assert "boom" in result.errors[0]


class TestBenchmarkMainEntry:
    def test_main_block_runs(self, capsys):
        import sloughgpt_sdk.benchmarks as benchmarks_mod

        src = Path(benchmarks_mod.__file__).read_text()
        ns = {"__name__": "__main__", "__file__": str(benchmarks_mod.__file__)}
        exec(compile(src, str(benchmarks_mod.__file__), "exec"), ns)
        out = capsys.readouterr().out
        assert "CACHE BENCHMARKS" in out