"""benchmark — Benchmark domain for performance testing.

Public API:
    BenchmarkDomain, BenchmarkResult, get_benchmark_domain, reset_benchmark_domain
"""

from domain.benchmark._internal.domain import (
    BenchmarkDomain,
    BenchmarkResult,
    get_benchmark_domain,
    reset_benchmark_domain,
)

__all__ = [
    "BenchmarkDomain",
    "BenchmarkResult",
    "get_benchmark_domain",
    "reset_benchmark_domain",
]
