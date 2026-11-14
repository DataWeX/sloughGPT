#!/usr/bin/env python3
"""Dockerfile build-context gate (card 20261027_070731_infra-dockerfile-rot-gate).

Every ``COPY`` source in every Dockerfile must (a) exist on disk and (b) survive
``.dockerignore``. A path excluded from the build context is invisible to the
builder, so the build dies with ``not found: excluded from dockerignore`` — the
exact rot that kept ``infra/docker/Dockerfile.test`` unbuildable for months
while nothing ever built it (its ``COPY tests/`` vs ``.dockerignore: tests/``,
excluded since d1b7e7484, 2026-06-21).

Matching follows Docker's own rules (moby patternmatcher / Go ``filepath.Match``):
patterns are rooted at the context root — ``node_modules`` matches only
``<root>/node_modules``, hence the explicit ``**/node_modules`` on line 31 of
``.dockerignore`` — ``!`` re-includes the exact path, a trailing ``/`` matches
directories only, and a source is excluded when it or any of its parent
directories matches. ``COPY --from=<stage>`` sources are skipped: they read
from an earlier stage, never from the context.

Known, accepted rot lives in ``dockerfile_context_baseline.json`` as strict
xfails with **exact-set** matching: fix the file and the entry goes stale (the
gate fails until the entry is deleted); add new rot and it fails too. The same
ratchet as ``page-baseline.json`` and ``command_parity_baseline.json``.

Usage: ``python scripts/check_dockerfile_contexts.py`` (stdlib only, exit 0
clean / 1 on new rot or a stale baseline entry).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = Path(__file__).resolve().parent / "dockerfile_context_baseline.json"


# ── Discovery ────────────────────────────────────────────────────────────────
def discover_dockerfiles() -> list[str]:
    """Every tracked file whose basename starts with ``Dockerfile``.

    ``git ls-files`` first (never walks ``.git``/``node_modules``); explicit
    globs as a fallback when git is unavailable.
    """
    files: list[str] = []
    try:
        out = subprocess.run(
            ["git", "ls-files"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        files = [line for line in out.splitlines() if Path(line).name.startswith("Dockerfile")]
    except (OSError, subprocess.CalledProcessError):
        files = []
    if not files:
        for pattern in (
            "Dockerfile*",
            "infra/docker/Dockerfile*",
            "apps/api/server/Dockerfile*",
            "apps/web/Dockerfile*",
            "buildroot/Dockerfile*",
        ):
            files.extend(str(p.relative_to(REPO_ROOT)) for p in sorted(REPO_ROOT.glob(pattern)))
    return sorted(set(files))


# ── .dockerignore semantics ─────────────────────────────────────────────────
def _segment_to_regex(segment: str) -> str:
    """Translate one non-``**`` glob segment (Go filepath.Match flavour)."""
    out: list[str] = []
    i = 0
    while i < len(segment):
        char = segment[i]
        if char == "*":
            out.append("[^/]*")
        elif char == "?":
            out.append("[^/]")
        elif char == "[":
            j = i + 1
            if j < len(segment) and segment[j] in "!^":
                j += 1
            if j < len(segment) and segment[j] == "]":
                j += 1
            while j < len(segment) and segment[j] != "]":
                j += 1
            if j >= len(segment):
                out.append(re.escape(char))
            else:
                cls = segment[i : j + 1]
                if cls.startswith("[!"):
                    cls = "[^" + cls[2:]
                out.append(cls)
                i = j
        else:
            out.append(re.escape(char))
        i += 1
    return "".join(out)


def _pattern_to_regex(pattern: str) -> re.Pattern[str]:
    parts: list[str] = []
    segments = pattern.split("/")
    for index, segment in enumerate(segments):
        if segment == "**":
            # ``**/`` = zero or more leading directories; bare ``**`` = rest
            parts.append("(?:[^/]+/)*" if index < len(segments) - 1 else ".*")
            if index == len(segments) - 1 and index > 0 and segments[index - 1] == "**":
                pass
        else:
            translated = _segment_to_regex(segment)
            if index < len(segments) - 1:
                translated += "/"
            parts.append(translated)
    regex = "".join(parts)
    # ``a/**/b`` produced ``a/(?:[^/]+/)*b`` — collapse the stray separator a
    # bare ``**`` in the middle would otherwise leave behind.
    regex = regex.replace("//", "/")
    if regex.startswith("/"):
        regex = regex[1:]
    return re.compile(regex + r"\Z")


class IgnoreRules:
    """Ordered ``.dockerignore`` rules evaluated last-match-wins."""

    def __init__(self, root: Path) -> None:
        self._rules: list[tuple[bool, str, bool, re.Pattern[str]]] = []
        path = root / ".dockerignore"
        if not path.is_file():
            return
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            negate = line.startswith("!")
            if negate:
                line = line[1:].strip()
            if not line:
                continue
            dir_only = line.endswith("/")
            if dir_only:
                line = line[:-1]
            line = line.lstrip("/")
            if not line:
                continue
            self._rules.append((negate, line, dir_only, _pattern_to_regex(line)))

    def _last_match(self, rel: str, is_dir: bool) -> tuple[bool, str] | None:
        hit: tuple[bool, str] | None = None
        for negate, pattern, dir_only, rx in self._rules:
            if dir_only and not is_dir:
                continue
            if rx.match(rel):
                hit = (negate, pattern)
        return hit

    def excluded(self, rel: str) -> str | None:
        """Return the pattern excluding *rel*, or ``None`` when it survives.

        An explicit ``!`` on the exact path re-includes it even under an
        excluded parent (dockerignore, unlike gitignore, allows that); short
        of that, matching any parent directory excludes the whole subtree.
        """
        parts = [p for p in rel.split("/") if p not in ("", ".")]
        if not parts:
            return None
        full = "/".join(parts)
        leaf_is_dir = (REPO_ROOT / full).is_dir()
        hit = self._last_match(full, leaf_is_dir)
        if hit is not None:
            negate, pattern = hit
            return None if negate else pattern
        for i in range(1, len(parts)):
            prefix = "/".join(parts[:i])
            hit = self._last_match(prefix, True)
            if hit is not None and not hit[0]:
                return hit[1]
        return None


# ── COPY parsing ─────────────────────────────────────────────────────────────
def copy_sources(line: str) -> list[str]:
    """Context sources of a ``COPY`` line (``--from=`` stages are not context)."""
    body = re.sub(r"(?i)^\s*COPY\s+", "", line.strip())
    if body.lstrip().startswith("["):
        try:
            array = json.loads(body)
        except json.JSONDecodeError:
            return []
        return [str(item) for item in array[:-1]] if len(array) > 1 else []
    tokens = body.split()
    sources: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("--"):
            if "=" not in token:
                i += 1  # value travels as its own token (--chown user:group)
            i += 1
            continue
        sources.append(token)
        i += 1
    return sources[:-1] if len(sources) > 1 else []


def _normalise(source: str) -> str:
    source = source.strip().strip('"').strip("'")
    while source.startswith("./"):
        source = source[2:]
    # ``tests/`` and ``tests`` are the same context path — one canonical key.
    return source.lstrip("/").rstrip("/")


def _is_glob(source: str) -> bool:
    return any(char in source for char in "*?[")


# ── The gate ─────────────────────────────────────────────────────────────────
def check_file(rel: str, rules: IgnoreRules) -> dict[str, str]:
    """``{source: reason}`` for every context source that would fail the build."""
    path = REPO_ROOT / rel
    problems: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"<unreadable>": f"cannot read {rel}: {exc}"}
    for line in text.splitlines():
        if not re.match(r"(?i)^\s*COPY\b", line):
            continue
        # ``COPY --from=<stage>`` reads an earlier stage, never the context —
        # its sources (/usr/local/..., /build/target/...) are not context paths.
        if re.search(r"(?i)--from(?:=|\s)", line):
            continue
        for raw in copy_sources(line):
            source = _normalise(raw)
            if not source or source.startswith("--"):
                continue
            if _is_glob(source):
                matches = sorted(p for p in REPO_ROOT.glob(source) if p.is_file() or p.is_dir())
                if not matches:
                    problems[source] = "glob matches nothing on disk"
                    continue
                if all(rules.excluded(str(p.relative_to(REPO_ROOT))) for p in matches):
                    problems[source] = "every match excluded by .dockerignore"
                continue
            target = REPO_ROOT / source
            if not target.exists():
                problems[source] = "missing on disk"
                continue
            pattern = rules.excluded(source)
            if pattern is not None:
                problems[source] = f"excluded by .dockerignore ({pattern})"
    return problems


def load_baseline() -> dict[str, dict]:
    if not BASELINE_PATH.is_file():
        return {}
    data = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    entries = data.get("entries", data)
    return {k: v for k, v in entries.items() if isinstance(v, dict)}


def main() -> int:
    rules = IgnoreRules(REPO_ROOT)
    baseline = load_baseline()
    dockerfiles = discover_dockerfiles()
    if not dockerfiles:
        print("check_dockerfile_contexts: no Dockerfiles found — discovery broken")
        return 1

    failures: list[str] = []
    stale: list[str] = []
    baselined: list[str] = []

    for rel in dockerfiles:
        problems = check_file(rel, rules)
        expected = set(baseline.get(rel, {}).get("paths", []))
        actual = set(problems)
        if actual == expected:
            if expected:
                baselined.append(rel)
            continue
        for source in sorted(actual - expected):
            failures.append(f"  {rel}: {source} — {problems[source]}")
        for source in sorted(expected - actual):
            stale.append(f"  {rel}: {source} — baseline entry no longer reproduces")

    print(f"check_dockerfile_contexts: {len(dockerfiles)} Dockerfiles scanned")
    if baselined:
        print(f"baselined (known rot, {len(baselined)}):")
        for rel in baselined:
            reason = baseline[rel].get("reason", "")
            print(f"  {rel}: {reason}")
    if stale:
        print("STALE baseline entries — the rot is fixed, delete these entries:")
        print("\n".join(stale))
    if failures:
        print("NEW build-context rot:")
        print("\n".join(failures))
    if failures or stale:
        return 1
    print("OK — every COPY source exists and survives .dockerignore")
    return 0


if __name__ == "__main__":
    sys.exit(main())
