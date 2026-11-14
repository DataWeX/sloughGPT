#!/usr/bin/env python3
"""Static gate: every COPY source in every Dockerfile must survive .dockerignore.

A build context that silently excludes a COPY source costs a full CI run to
discover (the `.dockerignore` incident: `packages/strui/`, `apps/web/`,
`apps/cli/` excludes made the context arrive at 2B — 60 consecutive red `build`
runs before the COPY sources were audited). This walks all Dockerfiles, parses
their COPY instructions, and checks each source against the root .dockerignore
using Docker's patternmatcher semantics (trailing-slash = whole subtree,
`**`/`*` globs, parent-walk: any matching ancestor excludes the source).

Usage:
    python3 scripts/check_dockerignore.py [repo_root]

Exit 0 = all COPY sources reachable; exit 1 = at least one excluded source.
"""

from __future__ import annotations

import fnmatch
import os
import re
import sys

DOCKERFILE_PATTERNS = (
    "Dockerfile",
    "Dockerfile.*",
    "*.Dockerfile",
    "docker/Dockerfile*",
    "infra/docker/Dockerfile*",
    "**/Dockerfile",
    "**/Dockerfile.*",
)

# Accepted remainders (each with a ruling): sources that ARE excluded and known.
# The gate exits non-zero only for anything NEW — a ratchet, not permanent red.
KNOWN_EXCLUSIONS = {
    # 2026-10-23 ruling (card 0464c3d4): local deploy tooling, never built by
    # CI's build job; un-excluding costs 76MB of context. Revisit if a workflow
    # ever builds Dockerfile.test.
    ("infra/docker/Dockerfile.test", "tests"),
}


def find_dockerfiles(root: str) -> list[str]:
    hits: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        rel = "" if rel == "." else rel
        # Never descend into trees that never reach the build context anyway.
        dirnames[:] = [d for d in dirnames if d not in {".git", "node_modules"}]
        for name in filenames:
            if not (name == "Dockerfile" or name.startswith("Dockerfile.")):
                continue
            path = os.path.normpath(os.path.join(rel, name)) if rel else name
            hits.append(path)
    return sorted(hits)


def dockerignore_patterns(root: str) -> list[str]:
    path = os.path.join(root, ".dockerignore")
    patterns: list[str] = []
    if not os.path.exists(path):
        return patterns
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        patterns.append(line.lstrip("/"))
    return patterns


def parse_copy_sources(dockerfile: str, text: str) -> list[str]:
    """Return context-relative COPY sources (multi-line aware, flags stripped)."""
    sources: list[str] = []
    # Join line continuations so `COPY a \\\n b` parses as one instruction.
    joined = re.sub(r"\\\s*\n", " ", text)
    for raw in joined.splitlines():
        line = raw.strip()
        if not re.match(r"^COPY\b", line):
            continue
        # COPY --from=<stage|image> ... is out of context: skip entirely.
        if re.match(r"^COPY\s+--from=", line):
            continue
        tokens = re.findall(r"[^\s]+", line)[1:]  # drop COPY
        args: list[str] = []
        for tok in tokens:
            if tok.startswith("--"):
                continue  # --chown=, --chmod=, --link, ...
            args.append(tok.strip("'\""))
        if len(args) < 2:
            continue
        # Last token = destination; everything before = sources.
        for src in args[:-1]:
            if src.startswith("http://") or src.startswith("https://"):
                continue
            if src == ".":
                continue  # whole context — trivially present by definition
            sources.append(src)
    return sources


def _pattern_matches(pattern: str, candidate: str) -> bool:
    pattern = pattern.rstrip()
    if pattern.endswith("/"):
        pattern = pattern[:-1]
    if not pattern:
        return False
    # Docker patternmatcher: `dir/` and `dir` both prune the whole subtree,
    # `**` spans segments, `*`/`?` are fnmatch within a segment.
    pat = pattern if "**" in pattern else pattern
    if fnmatch.fnmatch(candidate, pat):
        return True
    if "**" not in pattern and fnmatch.fnmatch(candidate, pattern + "/*"):
        # A plain `dir` pattern excludes everything below dir too only when the
        # pattern names a directory prefix — Docker matches `dir` against each
        # path element, so `dir` excludes `dir/anything`.
        pass
    return False


def is_excluded(source: str, patterns: list[str]) -> str | None:
    """Return the offending pattern if the source would be excluded, else None."""
    src = source.rstrip("/")
    if not src:
        return None
    # Strip glob tails for parent-walk probes (package-lock.json* -> its prefix).
    probe = re.split(r"[*?\[]", src, maxsplit=1)[0].rstrip("/")
    candidates = []
    cur = probe
    while cur and cur != ".":
        candidates.append(cur)
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    for pattern in patterns:
        pat = pattern.rstrip("/")
        for cand in candidates:
            if fnmatch.fnmatch(cand, pat) or fnmatch.fnmatch(cand, pat.rstrip("/")):
                return pattern
            # `dir` / `dir/` excludes everything beneath it.
            if not any(ch in pat for ch in "*?[") and (cand == pat or cand.startswith(pat + "/")):
                return pattern
        # `**/name` style: match the last segment too.
        if pat.startswith("**/") and any(fnmatch.fnmatch(cand, pat[3:]) for cand in candidates):
            return pattern
        # Wildcard source already truncated by probe split — match prefix dir.
        if any(ch in pattern for ch in "*?[") and any(
            cand.startswith(re.split(r"[*?\[]", pattern, maxsplit=1)[0].rstrip("/"))
            for cand in candidates
            if re.split(r"[*?\[]", pattern, maxsplit=1)[0].rstrip("/")
        ):
            return pattern
    return None


def main(argv: list[str]) -> int:
    root = os.path.abspath(argv[1] if len(argv) > 1 else ".")
    patterns = dockerignore_patterns(root)
    dockerfiles = find_dockerfiles(root)
    total = 0
    failures: list[tuple[str, str, str]] = []
    for df in dockerfiles:
        text = open(os.path.join(root, df), encoding="utf-8", errors="ignore").read()
        for src in parse_copy_sources(df, text):
            total += 1
            offender = is_excluded(src, patterns)
            if offender is not None:
                failures.append((df, src, offender))
    print(f"checked {total} COPY sources across {len(dockerfiles)} Dockerfiles")
    unexpected: list[str] = []
    known: list[str] = []
    for df, src, offender in failures:
        line = f"  FAIL {df}: COPY {src!r} excluded by pattern {offender!r}"
        if (df, src.rstrip("/")) in KNOWN_EXCLUSIONS:
            known.append(f"  known remainder: {df}: {src!r} ({offender!r})")
        else:
            unexpected.append(line)
    for line in known:
        print(line)
    for line in unexpected:
        print(line)
    if unexpected:
        print(f"{len(unexpected)} unexpected excluded COPY source(s) — build would 404 here")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
