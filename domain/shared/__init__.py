"""shared — Shared utilities, types, constants across all domains.

Public API:
    TestFramework, TestResult, TestSuite, BenchmarkRunner, test_decorator
    find_available_port, find_repo_root, find_server_python
"""

from domain.shared._internal.test_framework import (
    TestFramework,
    TestResult,
    TestSuite,
    BenchmarkRunner,
    mark_test as test_decorator,
)
from domain.shared._internal.utils import (
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
