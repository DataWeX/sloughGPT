#!/usr/bin/env python3
"""Pre-commit hook: catch hardcoded secrets in committed files."""
import re
import sys

PATTERNS = [
    (r'(?i)(api[_-]?key|secret[_-]?key|password|token)\s*[=:]\s*["\'][A-Za-z0-9+/=_-]{16,}["\']', "hardcoded secret"),
    (r'(?i)bearer\s+[A-Za-z0-9+/=_-]{20,}', "hardcoded bearer token"),
    (r'-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----', "private key"),
    (r'sk-[A-Za-z0-9]{20,}', "OpenAI API key"),
    (r'ghp_[A-Za-z0-9]{36}', "GitHub PAT"),
    (r'xox[bpsa]-[A-Za-z0-9-]+', "Slack token"),
]

def main():
    failed = False
    for path in sys.argv[1:]:
        try:
            content = open(path).read()
        except Exception:
            continue
        for i, line in enumerate(content.splitlines(), 1):
            for pattern, desc in PATTERNS:
                if re.search(pattern, line):
                    print(f"\033[31m{path}:{i}: {desc}\033[0m {line.strip()[:80]}")
                    failed = True
    return 1 if failed else 0

if __name__ == "__main__":
    raise SystemExit(main())
