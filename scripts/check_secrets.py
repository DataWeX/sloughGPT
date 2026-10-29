#!/usr/bin/env python3
"""Secret scanner: pre-commit hook + CI gate for hardcoded secrets in tracked files.

Usage:
    scripts/check_secrets.py <file>...            (pre-commit: staged files)
    git ls-files -z | xargs -0 scripts/check_secrets.py   (CI: whole tree)

Placeholder values (CHANGE_ME, test fixtures, etc.) are allowlisted below so
the gate flags real secrets only. Tune the allowlist, never the patterns.
"""

import re
import sys

PATTERNS = [
    (
        r'(?i)(api[_-]?key|secret[_-]?key|password|token)\s*[=:]\s*["\'][A-Za-z0-9+/=_-]{16,}["\']',
        "hardcoded secret",
    ),
    (r"(?i)bearer\s+[A-Za-z0-9+/=_-]{20,}", "hardcoded bearer token"),
    (r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----", "private key"),
    (r"sk-[A-Za-z0-9]{20,}", "OpenAI API key"),
    (r"ghp_[A-Za-z0-9]{36}", "GitHub PAT"),
    (r"xox[bpsa]-[A-Za-z0-9-]+", "Slack token"),
]

# Explicit placeholders, docs-instructions and test fixtures. A matched line
# containing any of these is not a finding (case-insensitive substring).
ALLOWLIST = [
    "change_me",
    "changeme",
    "generate-with",
    "openssl rand",
    "placeholder",
    "example",
    "dummy",
    "no-such",
    "fake-",
    "test-token",
    "xxxx",
    "redacted",
    "<your",
    "your-",
]


def _allowed(line: str) -> bool:
    low = line.lower()
    return any(marker in low for marker in ALLOWLIST)


def main():
    failed = False
    for path in sys.argv[1:]:
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except OSError:
            continue
        for i, line in enumerate(content.splitlines(), 1):
            if _allowed(line):
                continue
            for pattern, desc in PATTERNS:
                if re.search(pattern, line):
                    print(f"\033[31m{path}:{i}: {desc}\033[0m {line.strip()[:80]}")
                    failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
