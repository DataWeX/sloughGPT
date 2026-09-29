"""Contract: docs/TESTING.md must match the real test infrastructure.

Guards the drift class found in the 2026-09-29 audit of TESTING.md:
- commands that fail locally (`-n auto` without pytest-xdist installed),
- example/test-file paths that do not exist (e.g. test_inference.py under
  packages/core-py, the dead brainstorm tree),
- the stale self-reported coverage table (100%/95% claims),
- markers defined in pytest.ini but undocumented,
- npm run commands whose scripts do not exist,
- a stale contract-suite count.

Every fix that changes the real infrastructure must update TESTING.md in the
same change, and vice versa.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_DOC = _REPO / "docs" / "TESTING.md"

# Backticked tokens with these prefixes must exist in the repo.
_PATH_PREFIXES = (
    "apps/web/",
    "packages/core-py/",
    "tests/",
    ".husky/",
    ".github/workflows/",
    ".pre-commit-config.yaml",
    "pytest.ini",
)
# Tokens that are glob examples, commands, or placeholders — not paths.
_SKIP_SUBSTRINGS = ("*", " ", "=", "\t")


def _doc_text() -> str:
    return _DOC.read_text(encoding="utf-8")


def test_referenced_paths_exist() -> None:
    text = _doc_text()
    tokens = re.findall(r"`([^`\n]+)`", text)
    missing: list[str] = []
    for token in tokens:
        token = token.strip().rstrip(".,;:")
        if any(s in token for s in _SKIP_SUBSTRINGS):
            continue
        if not token.startswith(_PATH_PREFIXES):
            continue
        if not (_REPO / token).exists():
            missing.append(token)
    assert not missing, (
        f"TESTING.md references paths that do not exist — truth-up the doc: {missing}"
    )


def test_no_stale_claims() -> None:
    text = _doc_text()
    forbidden = {
        "-n auto": "fails locally (pytest-xdist not in the venv; CI installs it)",
        "| 100% |": "stale self-reported coverage table",
        "| 95%+ |": "stale self-reported coverage table",
        "brainstorm": "dead example tree (dir no longer exists)",
    }
    hits = [f"{s!r} ({why})" for s, why in forbidden.items() if s in text]
    assert not hits, f"TESTING.md contains stale claims: {hits}"


def test_pytest_ini_markers_documented() -> None:
    ini = (_REPO / "pytest.ini").read_text(encoding="utf-8")
    match = re.search(r"^markers\s*=\s*\n((?:[ \t]+\w+:.*\n)+)", ini, re.MULTILINE)
    assert match, "root pytest.ini has no markers block — fix this test's parser"
    names = re.findall(r"^[ \t]+(\w+):", match.group(1), re.MULTILINE)
    assert names, "root pytest.ini markers block parsed empty"
    text = _doc_text()
    undocumented = [n for n in names if not re.search(rf"`{n}`", text)]
    assert not undocumented, (
        f"markers defined in pytest.ini but not documented in TESTING.md: {undocumented}"
    )


def test_run_commands_map_to_real_scripts() -> None:
    text = _doc_text()
    web_scripts = __import__("json").loads(
        (_REPO / "apps" / "web" / "package.json").read_text(encoding="utf-8")
    )["scripts"]
    root_scripts = __import__("json").loads((_REPO / "package.json").read_text(encoding="utf-8"))[
        "scripts"
    ]
    checks = (
        ("npm test", web_scripts, "test"),
        ("npm run test:lib", web_scripts, "test:lib"),
        ("npm run lint", root_scripts, "lint"),
        ("npm run typecheck", root_scripts, "typecheck"),
    )
    problems = []
    for referenced, scripts, key in checks:
        if referenced not in text:
            problems.append(f"doc no longer mentions {referenced!r}")
        elif key not in scripts:
            problems.append(f"doc documents {referenced!r} but script {key!r} is gone")
    assert not problems, problems


@pytest.mark.parametrize("case", ["doc_count_matches_collected"], ids=lambda c: c)
def test_contract_suite_count_matches_doc(case: str) -> None:
    assert case == "doc_count_matches_collected"
    match = re.search(r"contract suite: (\d+) tests", _doc_text())
    assert match, "TESTING.md no longer states the contract-suite count"
    doc_count = int(match.group(1))

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        (str(_REPO), str(_REPO / "packages" / "core-py"), str(_REPO / "apps" / "api" / "server"))
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/contract/",
            "--collect-only",
            "-q",
            "-o",
            "testpaths=",
            "-p",
            "no:cacheprovider",
        ],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"collect failed:\n{proc.stdout}\n{proc.stderr}"
    found = re.search(r"(\d+) tests? collected", proc.stdout)
    assert found, f"could not parse collect count from:\n{proc.stdout}"
    actual = int(found.group(1))
    assert doc_count == actual, (
        f"TESTING.md says contract suite: {doc_count} tests but pytest collects "
        f"{actual} — update the count in TESTING.md"
    )
