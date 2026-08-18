"""
Shared Components

This domain contains shared utilities, types, constants,
and common functionality used across all domains.
"""

from .test_framework import TestFramework, TestResult, TestSuite, BenchmarkRunner, mark_test as test_decorator
from .utils import find_available_port, find_repo_root, find_server_python

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
