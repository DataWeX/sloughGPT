"""AML parser — tokenises and parses ``.aml`` text into an ``AmlDocument``.

Grammar (simplified)::

    document    := header? block*
    header      := '@aml' VERSION NEWLINE
    block       := TAG NAME? ('{' body '}')? | TAG NAME '=' value
    body        := (assignment | list_item | nested_block)*
    assignment  := IDENT '=' value
    list_item   := '-' value
    nested_block:= TAG NAME? '{' body '}'

The parser is line-oriented for robustness: each line is classified and
processed independently.  Brace nesting is tracked for block scoping.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from aml.schema import AmlBlock, AmlDocument, AmlValue

# ── token patterns ────────────────────────────────────────────────────

RE_COMMENT    = re.compile(r"^\s*#")
RE_HEADER     = re.compile(r"^@aml\s+(\d+\.\d+)")
RE_BLANK      = re.compile(r"^\s*$")

# @tag name { ... }  or  @tag {  — handles both single-line and multi-line
RE_TAG_WITH_BRACE = re.compile(r"^\s*(@\w+)(?:\s+([^{]+))?\s*\{(.*)\}\s*$")
# @tag name {  (multi-line, opening brace at end)
RE_TAG_OPEN = re.compile(r"^\s*(@\w+)(?:\s+([^{]+))?\s*\{\s*$")
# @tag name = value
RE_INLINE_EQ  = re.compile(r"^\s*(@\w+)\s+(\S+)\s*=\s*(.+)$")
#     key = value
RE_ASSIGN     = re.compile(r"^\s+(\w+)\s*=\s*(.+)$")
# - item
RE_LIST_ITEM  = re.compile(r"^\s*-\s+(.+)$")


# ── value parser ──────────────────────────────────────────────────────

def _parse_value(raw: str) -> Any:
    """Parse a value token into a Python object.

    Handles: quoted strings, numbers, booleans, null, inline lists.
    """
    stripped = raw.strip()

    # inline list: [a, b, c]
    if stripped.startswith("[") and stripped.endswith("]"):
        inner = stripped[1:-1].strip()
        if not inner:
            return []
        return [_parse_value(item.strip()) for item in _split_csv(inner)]

    # quoted string
    if (stripped.startswith('"') and stripped.endswith('"')) or \
       (stripped.startswith("'") and stripped.endswith("'")):
        return stripped[1:-1]

    # null
    if stripped.lower() in ("null", "none", ""):
        return None

    # bool
    if stripped.lower() in ("true", "yes", "on"):
        return True
    if stripped.lower() in ("false", "no", "off"):
        return False

    # int
    try:
        return int(stripped)
    except ValueError:
        pass

    # float
    try:
        return float(stripped)
    except ValueError:
        pass

    # unquoted string
    return stripped


def _split_csv(s: str) -> list[str]:
    """Split a comma-separated value string respecting quotes."""
    parts: list[str] = []
    in_quote: Optional[str] = None
    current: list[str] = []
    for ch in s:
        if ch in ('"', "'") and (in_quote is None or in_quote == ch):
            if in_quote == ch:
                in_quote = None
            else:
                in_quote = ch
            current.append(ch)
        elif ch == "," and in_quote is None:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return parts


# ── main parser ───────────────────────────────────────────────────────

def parse(source: str) -> AmlDocument:
    """Parse an AML string into an ``AmlDocument``."""
    doc = AmlDocument()
    lines = source.split("\n")
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        # skip blank lines and comments
        if RE_BLANK.match(line) or RE_COMMENT.match(line):
            i += 1
            continue

        # header: @aml 1.0
        m = RE_HEADER.match(line)
        if m:
            doc.version = m.group(1)
            i += 1
            continue

        # inline assignment: @tag name = value
        m = RE_INLINE_EQ.match(line)
        if m:
            tag = m.group(1).lstrip("@")
            name = m.group(2)
            val = _parse_value(m.group(3))
            doc.blocks.append(AmlBlock(tag=tag, name=name, body=val, line=i + 1))
            i += 1
            continue

        # block: @tag name { ... }  (single or multi-line)
        m = RE_TAG_OPEN.match(line)
        if not m:
            m = RE_TAG_WITH_BRACE.match(line)
        if m:
            tag = m.group(1).lstrip("@")
            name = m.group(2)
            if name:
                name = name.strip()
            # single-line block: @tag name { key = val }
            if RE_TAG_WITH_BRACE.match(line) and "{" in line and "}" in line:
                inner = line.split("{", 1)[1].rsplit("}", 1)[0].strip()
                metadata: dict[str, Any] = {}
                if inner:
                    for part in inner.split(","):
                        part = part.strip()
                        am = RE_ASSIGN.match("    " + part)
                        if am:
                            metadata[am.group(1)] = _parse_value(am.group(2))
                doc.blocks.append(AmlBlock(tag=tag, name=name, metadata=metadata, line=i + 1))
                i += 1
                continue
            # multi-line block: parse body until matching }
            block, i = _parse_block_body(lines, i + 1, tag, name, doc)
            doc.blocks.append(block)
            continue

        # skip unrecognized lines
        i += 1

    return doc


def _parse_block_body(lines: list[str], start: int, tag: str,
                      name: Optional[str], doc: AmlDocument
                      ) -> tuple[AmlBlock, int]:
    """Parse the body of a block delimited by ``{`` ... ``}``.

    Returns (block, next_line_index).
    """
    n = len(lines)
    i = start
    depth = 1
    metadata: dict[str, Any] = {}
    list_items: list[str] = []

    while i < n and depth > 0:
        bl = lines[i].rstrip()

        # skip blank / comment
        if RE_BLANK.match(bl) or RE_COMMENT.match(bl):
            i += 1
            continue

        # list item: - value
        m = RE_LIST_ITEM.match(bl)
        if m:
            list_items.append(_parse_value(m.group(1).strip()))
            i += 1
            continue

        # assignment: key = value
        m = RE_ASSIGN.match(bl)
        if m:
            key = m.group(1)
            val = _parse_value(m.group(2))
            metadata[key] = val
            i += 1
            continue

        # nested block: @tag name {  (inner recursion handles its own braces)
        nm = RE_INLINE_EQ.match(bl)
        if nm:
            i += 1
            continue

        # multi-line nested block: @tag name {
        nm = RE_TAG_OPEN.match(bl)
        if nm:
            nested_tag = nm.group(1).lstrip("@")
            nested_name = nm.group(2)
            if nested_name:
                nested_name = nested_name.strip()
            nested, i = _parse_block_body(lines, i + 1, nested_tag, nested_name, doc)
            doc.blocks.append(nested)
            continue

        # single-line nested block: @tag name { key = val }
        nm = RE_TAG_WITH_BRACE.match(bl)
        if nm and "{" in bl and "}" in bl:
            nested_tag = nm.group(1).lstrip("@")
            nested_name = nm.group(2)
            if nested_name:
                nested_name = nested_name.strip()
            inner = nm.group(3).strip()
            nested_meta: dict[str, Any] = {}
            if inner:
                for part in inner.split(","):
                    part = part.strip()
                    am = RE_ASSIGN.match("    " + part)
                    if am:
                        nested_meta[am.group(1)] = _parse_value(am.group(2))
            doc.blocks.append(AmlBlock(tag=nested_tag, name=nested_name,
                                        metadata=nested_meta, line=i + 1))
            i += 1
            continue

        # track brace depth for non-nested lines
        depth += bl.count("{") - bl.count("}")
        if depth <= 0:
            i += 1
            break

        i += 1

    # build body
    body: Any = None
    if list_items:
        body = list_items

    return AmlBlock(tag=tag, name=name, body=body, metadata=metadata,
                     line=start), i


def parse_file(path: str) -> AmlDocument:
    """Parse an AML file from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return parse(f.read())
