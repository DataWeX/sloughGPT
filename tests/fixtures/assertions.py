"""Custom assertion helpers for test diagnostics.

These give better error messages than plain assert, making failures
easier to diagnose without --tb=long.

Usage:
    from tests.fixtures.assertions import (
        assert_contains, assert_not_contains, assert_json_structure,
        assert_importable, assert_no_exceptions, assert_changed,
    )

    def test_api_response(response):
        assert_json_structure(response, ["status", "data", "data.models"])
        assert_contains(response.text, "success")
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Optional, Sequence


def assert_contains(haystack: str, needle: str, msg: str = "") -> None:
    """Assert that haystack contains needle with a clear message."""
    if needle not in haystack:
        excerpt = haystack[:200] + "..." if len(haystack) > 200 else haystack
        detail = f"\n  looking for: {needle!r}\n  in: {excerpt!r}" if not msg else f"\n  {msg}"
        raise AssertionError(f"String does not contain expected substring.{detail}")


def assert_not_contains(haystack: str, needle: str, msg: str = "") -> None:
    """Assert that haystack does NOT contain needle."""
    if needle in haystack:
        excerpt = haystack[:200] + "..." if len(haystack) > 200 else haystack
        detail = f"\n  unexpected: {needle!r}\n  in: {excerpt!r}" if not msg else f"\n  {msg}"
        raise AssertionError(f"String contains unexpected substring.{detail}")


def assert_json_structure(data: Any, paths: list[str], msg: str = "") -> None:
    """Assert that nested keys exist in a dict/JSON structure.

    Paths use dot notation: "data.models.0.name" means data["data"]["models"][0]["name"]
    """
    for path in paths:
        parts = path.split(".")
        current = data
        for part in parts:
            try:
                if isinstance(current, list):
                    current = current[int(part)]
                elif isinstance(current, dict):
                    current = current[part]
                else:
                    raise AssertionError(
                        f"Cannot traverse {path!r}: "
                        f"expected dict/list at '{part}', got {type(current).__name__}"
                        + (f"\n  {msg}" if msg else "")
                    )
            except (KeyError, IndexError, ValueError) as e:
                raise AssertionError(
                    f"Missing path {path!r} at '{part}': {e}"
                    + (f"\n  {msg}" if msg else "")
                )


def assert_importable(module_name: str, msg: str = "") -> None:
    """Assert that a module can be imported."""
    try:
        importlib.import_module(module_name)
    except ImportError as e:
        raise AssertionError(
            f"Cannot import {module_name!r}: {e}"
            + (f"\n  {msg}" if msg else "")
        )


def assert_no_exceptions(func, *args, _msg: str = "", **kwargs) -> Any:
    """Assert that calling func doesn't raise. Returns the result."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        raise AssertionError(
            f"Expected no exception, got {type(e).__name__}: {e}"
            + (f"\n  {_msg}" if _msg else "")
        )


def assert_changed(file_path: str, expect_diff: bool = True) -> None:
    """Assert that a file has uncommitted changes (git diff)."""
    import subprocess
    result = subprocess.run(
        ["git", "diff", "--quiet", "--", file_path],
        capture_output=True,
        timeout=5,
    )
    has_changes = result.returncode != 0
    if expect_diff and not has_changes:
        raise AssertionError(f"Expected {file_path} to have changes, but it's clean.")
    if not expect_diff and has_changes:
        raise AssertionError(f"Expected {file_path} to be clean, but it has changes.")


def assert_response_ok(response, status_code: int = 200, msg: str = "") -> None:
    """Assert HTTP response has expected status and success envelope."""
    assert response.status_code == status_code, (
        f"Expected status {status_code}, got {response.status_code}"
        + (f"\n  {msg}" if msg else "")
    )


def assert_response_error(response, status_code: int, msg: str = "") -> None:
    """Assert HTTP response has error status."""
    assert response.status_code == status_code, (
        f"Expected error status {status_code}, got {response.status_code}"
        + (f"\n  {msg}" if msg else "")
    )


def assert_has_keys(data: dict, keys: Sequence[str], msg: str = "") -> None:
    """Assert that a dict has all expected keys."""
    missing = [k for k in keys if k not in data]
    if missing:
        raise AssertionError(
            f"Missing keys: {missing}\n  has: {list(data.keys())}"
            + (f"\n  {msg}" if msg else "")
        )


def assert_no_keys(data: dict, keys: Sequence[str], msg: str = "") -> None:
    """Assert that a dict does NOT have any of the given keys."""
    present = [k for k in keys if k in data]
    if present:
        raise AssertionError(
            f"Unexpected keys present: {present}"
            + (f"\n  {msg}" if msg else "")
        )


def assert_type(value: Any, expected_type: type, msg: str = "") -> None:
    """Assert value is of expected type."""
    if not isinstance(value, expected_type):
        raise AssertionError(
            f"Expected type {expected_type.__name__}, got {type(value).__name__}: {value!r}"
            + (f"\n  {msg}" if msg else "")
        )


def assert_length(value: Any, expected: int, msg: str = "") -> None:
    """Assert len(value) == expected."""
    actual = len(value)
    if actual != expected:
        raise AssertionError(
            f"Expected length {expected}, got {actual}: {value!r}"
            + (f"\n  {msg}" if msg else "")
        )


def assert_in_range(value: float, low: float, high: float, msg: str = "") -> None:
    """Assert value is within [low, high]."""
    if not (low <= value <= high):
        raise AssertionError(
            f"Expected {low} <= {value} <= {high}"
            + (f"\n  {msg}" if msg else "")
        )


def assert_file_exists(path: str | Path, msg: str = "") -> None:
    """Assert that a file exists."""
    p = Path(path)
    if not p.exists():
        raise AssertionError(f"File not found: {p}" + (f"\n  {msg}" if msg else ""))
    if not p.is_file():
        raise AssertionError(f"Not a file: {p}" + (f"\n  {msg}" if msg else ""))


def assert_dir_exists(path: str | Path, msg: str = "") -> None:
    """Assert that a directory exists."""
    p = Path(path)
    if not p.exists():
        raise AssertionError(f"Directory not found: {p}" + (f"\n  {msg}" if msg else ""))
    if not p.is_dir():
        raise AssertionError(f"Not a directory: {p}" + (f"\n  {msg}" if msg else ""))
