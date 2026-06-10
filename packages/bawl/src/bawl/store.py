"""
store — save/load pages as JSONL. Zero deps (stdlib json).
Works with existing apps via stdin/stdout piping.

Usage:
    from bawl import save, load, dumps, loads

    save(page)                       # write to stdout
    save(page, "output.jsonl")       # append to file
    save(page, "data.jsonl", append=False)  # overwrite

    for p in load("data.jsonl"):
        print(p.title)

    line = dumps(page)               # string -> pipe to another app
    for p in loads(line):
        print(p.text)
"""

import json
import sys
from pathlib import Path
from typing import Iterator, Union

from .parse import Page

_PATH = Union[str, Path]


def dumps_json_array(pages: list[Page]) -> str:
    """Serialize multiple Pages as a JSON array string.

    Args:
        pages: List of Page objects.

    Returns:
        JSON array string with pretty-print (2-space indent).
    """
    return json.dumps([p.to_dict() for p in pages], ensure_ascii=False, indent=2)


def save_json_array(pages: list[Page], path: str = "-") -> None:
    """Save a list of Pages as a JSON array (not JSONL).

    Args:
        pages: List of Page objects.
        path: File path, or "-" for stdout.

    Side effects:
        - Writes to stdout or creates a file on disk.
    """
    data = dumps_json_array(pages)
    if path == "-":
        sys.stdout.write(data)
        sys.stdout.write("\n")
        sys.stdout.flush()
    else:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            f.write(data)
            f.write("\n")


def dumps(page: Page) -> str:
    """Serialize a Page to a single JSONL line (dict + newline).

    Args:
        page: Page object to serialize.

    Returns:
        JSON string ending with \\n.
    """
    return json.dumps(page.to_dict(), ensure_ascii=False) + "\n"


def save(page: Page, path: _PATH = "-", append: bool = True) -> None:
    """Save a page as JSONL.

    Args:
        page: Page to save.
        path: File path, or "-" for stdout.
        append: If True (default) appends to file; if False, overwrites.

    Side effects:
        - Writes to stdout or creates/appends a file on disk.
        - Creates parent directories if they don't exist.
    """
    line = dumps(page)
    if path == "-":
        sys.stdout.write(line)
        sys.stdout.flush()
    else:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a" if append else "w") as f:
            f.write(line)


def loads(data: str) -> Iterator[Page]:
    """Parse a JSONL string into an iterator of Pages.

    Skips blank lines and lines that fail JSON parsing.

    Args:
        data: JSONL string (one JSON object per line).

    Yields:
        Page objects.
    """
    for line in data.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            yield Page.from_dict(json.loads(line))
        except json.JSONDecodeError:
            continue


def load(path: _PATH = "-") -> Iterator[Page]:
    """Load Pages from a JSONL file path or stdin.

    Args:
        path: File path, or "-" for stdin.

    Yields:
        Page objects.

    Side effects:
        - Opens and reads a file if path != "-".
    """
    f: Iterator[str]
    if path == "-":
        f = sys.stdin
    else:
        f = open(path)
    try:
        yield from loads("".join(f))
    finally:
        if path != "-":
            f.close()
