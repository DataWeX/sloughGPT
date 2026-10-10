"""shared — Shared utilities, types, constants across all domains.

Public API:
    TestFramework, TestResult, TestSuite, BenchmarkRunner, test_decorator
    find_available_port, find_repo_root, find_server_python, data_root
    utc_now_iso, to_iso, parse_iso, normalize_iso, is_valid_iso, get_timestamp,
    repair_iso — read-path repair for broken "+00:00Z" values
    normalize_local_iso — read-path repair treating naive input as server-local
    generate_id, hash_string, format_size, format_time, load_json, save_json
    merge_dicts, clamp, validate_config, retry, Timer, Cache, RateLimiter
"""

from domain.shared._internal.concurrency import Cache, RateLimiter, Timer, retry
from domain.shared._internal.formatting import format_size, format_time
from domain.shared._internal.hashing import hash_string
from domain.shared._internal.ids import generate_id
from domain.shared._internal.jsonio import load_json, save_json
from domain.shared._internal.net import find_available_port
from domain.shared._internal.ops import clamp, merge_dicts, validate_config
from domain.shared._internal.paths import data_root, find_repo_root, find_server_python
from domain.shared._internal.test_framework import (
    BenchmarkRunner,
    TestFramework,
    TestResult,
    TestSuite,
)
from domain.shared._internal.test_framework import (
    mark_test as test_decorator,
)
from domain.shared._internal.timestamps import (
    get_timestamp,
    is_valid_iso,
    normalize_iso,
    normalize_local_iso,
    parse_iso,
    repair_iso,
    to_iso,
    utc_now_iso,
)

__all__ = [
    "BenchmarkRunner",
    "Cache",
    "RateLimiter",
    "TestFramework",
    "TestResult",
    "TestSuite",
    "Timer",
    "clamp",
    "data_root",
    "find_available_port",
    "find_repo_root",
    "find_server_python",
    "format_size",
    "format_time",
    "generate_id",
    "get_timestamp",
    "hash_string",
    "is_valid_iso",
    "load_json",
    "merge_dicts",
    "normalize_iso",
    "normalize_local_iso",
    "parse_iso",
    "repair_iso",
    "retry",
    "save_json",
    "test_decorator",
    "to_iso",
    "utc_now_iso",
    "validate_config",
]
