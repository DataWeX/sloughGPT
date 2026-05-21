"""
SloughGPT SDK - Benchmarks
Performance benchmarking utilities.
"""

import time
import statistics
import threading
import asyncio
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    name: str
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    median_time_ms: float
    std_dev_ms: float
    ops_per_second: float
    p95_ms: Optional[float] = None
    p99_ms: Optional[float] = None
    
    def __str__(self):
        return (
            f"{self.name}:\n"
            f"  Iterations: {self.iterations}\n"
            f"  Total: {self.total_time_ms:.2f}ms\n"
            f"  Avg: {self.avg_time_ms:.4f}ms\n"
            f"  Median: {self.median_time_ms:.4f}ms\n"
            f"  Min/Max: {self.min_time_ms:.4f}/{self.max_time_ms:.4f}ms\n"
            f"  Std Dev: {self.std_dev_ms:.4f}ms\n"
            f"  Ops/sec: {self.ops_per_second:.2f}"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "iterations": self.iterations,
            "total_time_ms": self.total_time_ms,
            "avg_time_ms": self.avg_time_ms,
            "min_time_ms": self.min_time_ms,
            "max_time_ms": self.max_time_ms,
            "median_time_ms": self.median_time_ms,
            "std_dev_ms": self.std_dev_ms,
            "ops_per_second": self.ops_per_second,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
        }


@dataclass
class LoadTestResult:
    """Result of a load test."""
    name: str
    concurrent_workers: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time_ms: float
    requests_per_second: float
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    success_rate: float
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "concurrent_workers": self.concurrent_workers,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_time_ms": self.total_time_ms,
            "requests_per_second": self.requests_per_second,
            "avg_latency_ms": self.avg_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "median_latency_ms": self.median_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "success_rate": self.success_rate,
            "errors": self.errors[:10],
        }


def percentile(data: List[float], p: float) -> float:
    """Calculate percentile of data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    index = int(len(sorted_data) * p / 100)
    return sorted_data[min(index, len(sorted_data) - 1)]


class Benchmark:
    """
    Benchmarking utility for SDK operations.
    
    Example:
    
    ```python
    from sloughgpt_sdk.benchmarks import Benchmark, BenchmarkResult
    
    bench = Benchmark()
    
    # Benchmark a simple operation
    result = bench.run(
        name="String concatenation",
        func=lambda: "hello" + "world",
        iterations=10000
    )
    
    print(result)
    ```
    """
    
    def run(
        self,
        name: str,
        func: Callable,
        iterations: int = 1000,
        warmup: int = 10,
        args: tuple = (),
        kwargs: dict = None,
    ) -> BenchmarkResult:
        """
        Run a benchmark.
        
        Args:
            name: Name of the benchmark.
            func: Function to benchmark.
            iterations: Number of iterations.
            warmup: Number of warmup runs.
            args: Positional arguments for func.
            kwargs: Keyword arguments for func.
        
        Returns:
            BenchmarkResult with timing statistics.
        """
        kwargs = kwargs or {}
        times = []
        
        for _ in range(warmup):
            func(*args, **kwargs)
        
        start = time.perf_counter()
        for _ in range(iterations):
            iter_start = time.perf_counter()
            func(*args, **kwargs)
            iter_end = time.perf_counter()
            times.append((iter_end - iter_start) * 1000)
        
        total_time = sum(times)
        
        return BenchmarkResult(
            name=name,
            iterations=iterations,
            total_time_ms=total_time,
            avg_time_ms=statistics.mean(times),
            min_time_ms=min(times),
            max_time_ms=max(times),
            median_time_ms=statistics.median(times),
            std_dev_ms=statistics.stdev(times) if len(times) > 1 else 0,
            ops_per_second=iterations / (total_time / 1000),
            p95_ms=percentile(times, 95),
            p99_ms=percentile(times, 99),
        )
    
    def compare(
        self,
        name: str,
        funcs: Dict[str, Callable],
        iterations: int = 1000,
    ) -> List[BenchmarkResult]:
        """
        Compare multiple implementations.
        
        Returns:
            List of BenchmarkResult, sorted by fastest.
        """
        results = []
        for func_name, func in funcs.items():
            result = self.run(
                name=f"{name} - {func_name}",
                func=func,
                iterations=iterations,
            )
            results.append(result)
        
        return sorted(results, key=lambda r: r.avg_time_ms)
    
    def memory_benchmark(
        self,
        name: str,
        func: Callable,
        iterations: int = 100,
    ) -> Dict[str, Any]:
        """Benchmark memory usage."""
        import tracemalloc
        
        tracemalloc.start()
        
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            result = func()
            times.append(time.perf_counter() - start)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return {
            "name": name,
            "iterations": iterations,
            "avg_time_ms": statistics.mean(times) * 1000,
            "memory_current_mb": current / 1024 / 1024,
            "memory_peak_mb": peak / 1024 / 1024,
        }


class LoadTester:
    """
    Load testing utility.
    
    Example:
    
    ```python
    from sloughgpt_sdk.benchmarks import LoadTester
    
    tester = LoadTester()
    
    result = tester.load_test(
        name="API Load Test",
        request_func=lambda: requests.get("http://localhost:8000/health"),
        concurrent_users=10,
        requests_per_user=100,
    )
    
    print(f"Requests/sec: {result.requests_per_second}")
    print(f"Success rate: {result.success_rate * 100}%")
    ```
    """
    
    def load_test(
        self,
        name: str,
        request_func: Callable,
        concurrent_workers: int = 10,
        requests_per_worker: int = 100,
    ) -> LoadTestResult:
        """
        Run a load test.
        
        Args:
            name: Name of the test.
            request_func: Function that makes a request.
            concurrent_workers: Number of concurrent workers.
            requests_per_worker: Requests per worker.
        
        Returns:
            LoadTestResult with statistics.
        """
        latencies = []
        errors = []
        successful = 0
        failed = 0
        
        def worker():
            nonlocal successful, failed
            for _ in range(requests_per_worker):
                start = time.perf_counter()
                try:
                    request_func()
                    latencies.append((time.perf_counter() - start) * 1000)
                    successful += 1
                except Exception as e:
                    failed += 1
                    errors.append(str(e)[:100])
        
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=concurrent_workers) as executor:
            futures = [executor.submit(worker) for _ in range(concurrent_workers)]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    errors.append(str(e)[:100])
        
        total_time = (time.perf_counter() - start_time) * 1000
        total_requests = successful + failed
        
        if latencies:
            latencies_sorted = sorted(latencies)
            return LoadTestResult(
                name=name,
                concurrent_workers=concurrent_workers,
                total_requests=total_requests,
                successful_requests=successful,
                failed_requests=failed,
                total_time_ms=total_time,
                requests_per_second=total_requests / (total_time / 1000) if total_time > 0 else 0,
                avg_latency_ms=statistics.mean(latencies),
                min_latency_ms=min(latencies),
                max_latency_ms=max(latencies),
                median_latency_ms=statistics.median(latencies),
                p95_latency_ms=percentile(latencies, 95),
                p99_latency_ms=percentile(latencies, 99),
                success_rate=successful / total_requests if total_requests > 0 else 0,
                errors=errors,
            )
        else:
            return LoadTestResult(
                name=name,
                concurrent_workers=concurrent_workers,
                total_requests=total_requests,
                successful_requests=successful,
                failed_requests=failed,
                total_time_ms=total_time,
                requests_per_second=0,
                avg_latency_ms=0,
                min_latency_ms=0,
                max_latency_ms=0,
                median_latency_ms=0,
                p95_latency_ms=0,
                p99_latency_ms=0,
                success_rate=0,
                errors=errors,
            )
    
    def stress_test(
        self,
        name: str,
        request_func: Callable,
        duration_seconds: int = 60,
        target_rps: int = 100,
    ) -> Dict[str, Any]:
        """
        Run a stress test for a duration.
        
        Args:
            name: Name of the test.
            request_func: Function that makes a request.
            duration_seconds: Test duration.
            target_rps: Target requests per second.
        
        Returns:
            Summary dictionary.
        """
        results = []
        errors = []
        request_count = 0
        start_time = time.time()
        interval = 1.0 / target_rps
        
        while time.time() - start_time < duration_seconds:
            next_request_time = start_time + (request_count * interval)
            sleep_time = next_request_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            req_start = time.perf_counter()
            try:
                request_func()
                results.append((time.perf_counter() - req_start) * 1000)
            except Exception as e:
                errors.append(str(e)[:100])
            request_count += 1
        
        elapsed = time.time() - start_time
        
        return {
            "name": name,
            "duration_seconds": elapsed,
            "total_requests": len(results) + len(errors),
            "successful": len(results),
            "failed": len(errors),
            "requests_per_second": len(results) / elapsed if elapsed > 0 else 0,
            "avg_latency_ms": statistics.mean(results) if results else 0,
            "p95_latency_ms": percentile(results, 95) if results else 0,
            "errors": errors[:10],
        }


class Profiler:
    """Simple profiler for SDK operations."""
    
    def __init__(self):
        """Initialize profiler."""
        self._timings: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
    
    def __enter__(self):
        """Enter context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        pass
    
    def profile(self, name: str):
        """Decorator for profiling a function."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    elapsed = (time.perf_counter() - start) * 1000
                    with self._lock:
                        if name not in self._timings:
                            self._timings[name] = []
                        self._timings[name].append(elapsed)
                    return result
                except Exception as e:
                    raise
            return wrapper
        return decorator
    
    def get_report(self) -> Dict[str, Any]:
        """Get profiling report."""
        report = {}
        for name, times in self._timings.items():
            report[name] = {
                "calls": len(times),
                "total_ms": sum(times),
                "avg_ms": statistics.mean(times),
                "min_ms": min(times),
                "max_ms": max(times),
                "median_ms": statistics.median(times),
            }
        return report
    
    def print_report(self):
        """Print profiling report."""
        print("\n" + "=" * 60)
        print("PROFILING REPORT")
        print("=" * 60)
        
        for name, stats in self.get_report().items():
            print(f"\n{name}:")
            print(f"  Calls: {stats['calls']}")
            print(f"  Total: {stats['total_ms']:.2f}ms")
            print(f"  Avg: {stats['avg_ms']:.4f}ms")
            print(f"  Min/Max: {stats['min_ms']:.4f}/{stats['max_ms']:.4f}ms")
        
        print("=" * 60)


def benchmark_cache_operations():
    """Benchmark cache operations."""
    from sloughgpt_sdk.cache import InMemoryCache
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    cache = InMemoryCache()
    
    bench = Benchmark()
    
    result = bench.run(
        name="Cache SET",
        func=lambda: cache.set("key", "value"),
        iterations=10000,
    )
    
    bench.run(
        name="Cache GET",
        func=lambda: cache.get("key"),
        iterations=10000,
    )
    
    bench.run(
        name="Cache MISS",
        func=lambda: cache.get("nonexistent"),
        iterations=10000,
    )
    
    return result


def benchmark_api_key_operations():
    """Benchmark API key operations."""
    import tempfile
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sloughgpt_sdk.auth import APIKeyManager, KeyTier
    
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    manager = APIKeyManager(storage_path=temp_file.name)
    
    key, _ = manager.create_key(name="Test", tier=KeyTier.PRO)
    
    bench = Benchmark()
    
    bench.run(
        name="API Key Validation",
        func=lambda: manager.validate_key(key),
        iterations=1000,
    )
    
    bench.run(
        name="API Key Creation",
        func=lambda: manager.create_key(name="Temp"),
        iterations=100,
    )
    
    bench.run(
        name="Usage Stats",
        func=lambda: manager.get_usage_stats(key[:20] + "xxx"),
        iterations=1000,
    )


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("Running SDK Benchmarks...\n")
    
    print("=" * 60)
    print("CACHE BENCHMARKS")
    print("=" * 60)
    benchmark_cache_operations()
    
    print("\n" + "=" * 60)
    print("API KEY BENCHMARKS")
    print("=" * 60)
    benchmark_api_key_operations()
