"""Test isolation — run a single test in complete isolation.

Usage:
    from tests.fixtures.isolation import isolate_test

    # In a test:
    def test_something():
        result = isolate_test("tests/test_foo.py::test_bar")
        assert result.passed
        assert "expected output" in result.stdout

    # As a script:
    python -m tests.fixtures.isolation tests/test_foo.py::test_bar
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class IsolationResult:
    """Result of running a test in isolation."""
    nodeid: str
    passed: bool
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration: float = 0.0
    env_overrides: dict = field(default_factory=dict)

    @property
    def output(self) -> str:
        return self.stdout + "\n" + self.stderr

    def assert_passed(self) -> None:
        if not self.passed:
            raise AssertionError(
                f"Test {self.nodeid} failed (exit={self.exit_code}):\n"
                f"{self.output[-1000:]}"
            )


def isolate_test(
    nodeid: str,
    timeout: int = 60,
    extra_args: Optional[list[str]] = None,
    env: Optional[dict[str, str]] = None,
    use_tmpdir: bool = False,
) -> IsolationResult:
    """Run a single test in a clean subprocess with optional env overrides.

    This ensures:
    - No shared state from other tests
    - Clean environment variables
    - Captured stdout/stderr
    - Timeout protection
    """
    cmd = [
        sys.executable, "-m", "pytest",
        nodeid,
        "-v", "--tb=short", "-q",
        "--no-header",
        "--forked",  # if pytest-forked is available
    ]
    if extra_args:
        cmd.extend(extra_args)

    # Build environment
    run_env = os.environ.copy()
    run_env["PYTHONDONTWRITEBYTECODE"] = "1"
    if env:
        run_env.update(env)

    # Optionally run in a temp directory
    cwd = str(REPO_ROOT)
    if use_tmpdir:
        cwd = tempfile.mkdtemp(prefix="test-isolation-")

    try:
        t0 = time.time()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=run_env,
            cwd=cwd,
        )
        duration = time.time() - t0

        return IsolationResult(
            nodeid=nodeid,
            passed=proc.returncode == 0,
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            duration=duration,
            env_overrides=env or {},
        )
    except subprocess.TimeoutExpired:
        return IsolationResult(
            nodeid=nodeid,
            passed=False,
            exit_code=-1,
            stderr=f"TIMEOUT after {timeout}s",
            duration=timeout,
        )


def isolate_test_clean(
    nodeid: str,
    timeout: int = 60,
    extra_args: Optional[list[str]] = None,
) -> IsolationResult:
    """Run a test with minimal env — no shared fixtures, no conftest magic."""
    clean_env = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(REPO_ROOT / "apps" / "cli" / "src"),
    }
    return isolate_test(
        nodeid,
        timeout=timeout,
        extra_args=extra_args,
        env=clean_env,
        use_tmpdir=True,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m tests.fixtures.isolation <test_nodeid>")
        print("Example: python -m tests.fixtures.isolation tests/test_foo.py::test_bar")
        sys.exit(1)

    nodeid = sys.argv[1]
    result = isolate_test(nodeid, extra_args=sys.argv[2:])

    print(f"\n{'=' * 60}")
    print(f"  Test: {nodeid}")
    print(f"  Result: {'PASSED' if result.passed else 'FAILED'} ({result.duration:.1f}s)")
    print(f"{'=' * 60}")
    if result.stdout:
        print(f"\nstdout:\n{result.stdout}")
    if result.stderr:
        print(f"\nstderr:\n{result.stderr}")
    sys.exit(result.exit_code)
