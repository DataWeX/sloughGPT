"""Backward-compatibility shim — imports from the new ``domain.shared`` package."""

from domain.shared import (
    TestFramework,
    TestResult,
    TestSuite,
    BenchmarkRunner,
    test_decorator,
    find_available_port,
    find_repo_root,
    find_server_python,
)

__all__ = [
    "TestFramework",
    "TestResult",
    "TestSuite",
    "BenchmarkRunner",
    "test_decorator",
    "find_available_port",
    "find_repo_root",
    "find_server_python",
]
