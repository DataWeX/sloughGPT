"""AML serializer — converts ``AmlDocument`` or plain Python dicts back to AML text.

Produces clean, human-readable AML markup.
"""

from __future__ import annotations

from typing import Any, Optional

from aml.schema import AmlBlock, AmlDocument


def serialize(doc: AmlDocument, *, indent: str = "    ") -> str:
    """Serialize an ``AmlDocument`` to AML markup text.

    Args:
        doc: the document to serialize
        indent: indentation string for nested content (default 4 spaces)

    Returns:
        AML markup string with trailing newline.
    """
    lines: list[str] = []

    # header
    lines.append(f"@aml {doc.version}")
    lines.append("")

    for block in doc.blocks:
        lines.append(_serialize_block(block, indent))
        lines.append("")

    return "\n".join(lines) + "\n"


def _serialize_block(block: AmlBlock, indent: str) -> str:
    """Serialize a single block."""
    lines: list[str] = []
    tag = f"@{block.tag}"
    header = f"{tag} {block.name}" if block.name else tag

    # inline body (no metadata, simple value)
    if block.body is not None and not block.metadata:
        val = _format_value(block.body)
        if isinstance(block.body, (str, int, float, bool, type(None))):
            lines.append(f"{header} = {val}")
            return "\n".join(lines)

    # block body
    lines.append(f"{header} {{")

    if block.body is not None:
        if isinstance(block.body, list):
            for item in block.body:
                lines.append(f"{indent}- {_format_value(item, in_list=True)}")
        elif isinstance(block.body, str):
            for part in block.body.split("\n"):
                lines.append(f"{indent}{part}")

    for key, val in block.metadata.items():
        lines.append(f"{indent}{key} = {_format_value(val)}")

    lines.append("}")
    return "\n".join(lines)


def _format_value(val: Any, in_list: bool = False) -> str:
    """Format a Python value for AML output."""
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, list):
        inner = ", ".join(_format_value(item) for item in val)
        return f"[{inner}]"
    if isinstance(val, str):
        # quote if contains special chars that would break parsing
        needs_quote = any(c in val for c in "={}\n\t") or val != val.strip()
        if needs_quote or (not in_list and " " in val):
            escaped = val.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        return val
    return str(val)


def serialize_file(doc: AmlDocument, path: str) -> None:
    """Write an ``AmlDocument`` to an AML file on disk."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(serialize(doc))


def dict_to_aml(data: dict[str, Any], *, version: str = "1.0") -> str:
    """Convert a plain dict to AML markup.

    Top-level keys become block tags.  Values become block bodies or metadata.

    Example::

        dict_to_aml({
            "knowledge:mitochondria": {
                "content": "The mitochondria is the powerhouse.",
                "topic": "biology",
            }
        })

    Produces::

        @aml 1.0

        @knowledge mitochondria {
            content = "The mitochondria is the powerhouse."
            topic = "biology"
        }
    """
    doc = AmlDocument(version=version)

    for key, val in data.items():
        tag = key
        name = None
        if ":" in key:
            tag, name = key.split(":", 1)

        if isinstance(val, dict):
            body = val.pop("body", None)
            meta = val if val else {}
            doc.blocks.append(AmlBlock(tag=tag, name=name, body=body, metadata=meta))
        elif isinstance(val, list):
            doc.blocks.append(AmlBlock(tag=tag, name=name, body=val))
        else:
            doc.blocks.append(AmlBlock(tag=tag, name=name, body=val))

    return serialize(doc)
