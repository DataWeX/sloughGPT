"""Conformance guard for OpenCode skills.

Per https://opencode.ai/docs/skills/ every ``SKILL.md`` must start with a
YAML frontmatter block carrying ``name`` and ``description``, where ``name``
matches its directory and the pattern ``^[a-z0-9]+(-[a-z0-9]+)*$`` and
``description`` is 1-1024 characters.

Why this is worth a test: a skill that violates this still *loads* when
asked for by name (OpenCode falls back to the directory name and parses
leniently), but it drops out of the auto-discovered ``<available_skills>``
list — so the agent never learns it exists. Two skills (``cleanup-agent``
with no frontmatter at all, ``component-state-completeness`` with an
unquoted ``:`` in its description) were silently undiscoverable this way
until this guard's first run caught them. The failure is invisible from
inside a session, so it needs to be caught by CI instead.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
FRONTMATTER_RE = re.compile(r"\A---\s*\n(?P<body>.*?)\n---\s*$", re.S | re.M)


def skill_roots() -> list[Path]:
    """The directories opencode is told to scan.

    Read from ``opencode.json`` ``skills.paths`` so adding a path to the
    config brings it under this guard automatically; falls back to the
    default project location when unset or unreadable.
    """
    config = REPO_ROOT / "opencode.json"
    paths: list[str] = []
    if config.exists():
        try:
            data = json.loads(config.read_text(encoding="utf-8"))
            paths = list((data.get("skills") or {}).get("paths") or [])
        except (ValueError, AttributeError, TypeError):
            paths = []
    return [REPO_ROOT / p for p in (paths or [".opencode/skills"])]


def skill_files() -> list[Path]:
    """Every ``<root>/<name>/SKILL.md`` opencode would discover."""
    files: list[Path] = []
    for root in skill_roots():
        if root.is_dir():
            files.extend(sorted(root.glob("*/SKILL.md")))
    return files


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def frontmatter(path: Path) -> dict:
    """Parse the frontmatter strictly — stricter than opencode's parser."""
    match = FRONTMATTER_RE.match(path.read_text(encoding="utf-8"))
    assert match, f"{rel(path)} must START with a `---` frontmatter block"
    data = yaml.safe_load(match.group("body"))
    assert isinstance(data, dict), f"{rel(path)} frontmatter is not a mapping"
    return data


def test_skills_are_actually_discovered():
    """Non-vacuity: an empty scan must fail, not pass every loop below."""
    roots = skill_roots()
    assert all(r.is_dir() for r in roots), f"missing skills dir in {roots}"
    assert skill_files(), f"no SKILL.md files found under {[str(r) for r in roots]}"


@pytest.mark.parametrize("path", skill_files(), ids=lambda p: p.parent.name)
def test_frontmatter_parses_and_is_required(path: Path):
    data = frontmatter(path)
    assert data.get("name"), f"{rel(path)} is missing required `name`"
    assert data.get("description"), f"{rel(path)} is missing required `description`"


@pytest.mark.parametrize("path", skill_files(), ids=lambda p: p.parent.name)
def test_name_matches_directory_and_pattern(path: Path):
    name = frontmatter(path).get("name")
    assert name == path.parent.name, (
        f"{rel(path)} declares name {name!r} but lives in {path.parent.name!r} — "
        "opencode requires them to match"
    )
    assert isinstance(name, str) and NAME_RE.match(name), (
        f"{rel(path)} name {name!r} must be lowercase alphanumeric with single "
        "hyphen separators (^[a-z0-9]+(-[a-z0-9]+)*$)"
    )
    assert 1 <= len(name) <= 64, f"{rel(path)} name {name!r} not in 1..64 chars"


@pytest.mark.parametrize("path", skill_files(), ids=lambda p: p.parent.name)
def test_description_within_length_bounds(path: Path):
    description = frontmatter(path).get("description")
    assert isinstance(description, str), f"{rel(path)} description must be a string"
    assert 1 <= len(description) <= 1024, (
        f"{rel(path)} description is {len(description)} chars, must be 1..1024"
    )


def test_no_stray_skill_files():
    """A SKILL.md at the root of a skills dir is never discovered."""
    for root in skill_roots():
        stray = root / "SKILL.md"
        assert not stray.exists(), (
            f"{rel(stray)} sits directly in the skills dir; opencode only finds "
            "<skill-name>/SKILL.md, so move it into a named subdirectory"
        )


def test_skill_names_are_unique_across_roots():
    """Duplicate names across locations shadow each other per the docs."""
    seen: dict[str, list[Path]] = {}
    for path in skill_files():
        seen.setdefault(path.parent.name, []).append(path)
    duplicates = {name: paths for name, paths in seen.items() if len(paths) > 1}
    assert not duplicates, f"skill names must be unique across roots: {duplicates}"
