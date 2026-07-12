#!/usr/bin/env python3
"""Pre-commit hook: warn on TODO/FIXME/HACK in committed Python files."""
import sys

BLOCKERS = ("FIXME", "HACK")
WARNINGS = ("TODO",)

def main():
    failed = False
    for path in sys.argv[1:]:
        try:
            lines = open(path).readlines()
        except Exception:
            continue
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                for tag in BLOCKERS:
                    if tag in stripped:
                        print(f"\033[31m{path}:{i}: {tag}\033[0m {stripped}")
                        failed = True
                for tag in WARNINGS:
                    if tag in stripped:
                        print(f"\033[33m{path}:{i}: {tag}\033[0m {stripped}")
    return 1 if failed else 0

if __name__ == "__main__":
    raise SystemExit(main())
