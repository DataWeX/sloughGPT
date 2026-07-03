"""AML data model — the in-memory representation of an AML document.

An AML document is a list of typed blocks.  Each block has:
  - tag:      the type directive (e.g. ``@knowledge``, ``@config``)
  - name:     optional identifier
  - body:     string literal, list, map, or nested blocks
  - metadata: key-value pairs from ``key = value`` lines inside the block

Top-level directives (``@aml``, ``@import``, ``@comment``) are also blocks
with tag starting with ``@``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union


@dataclass
class AmlValue:
    """A scalar value extracted from AML source."""
    raw: str
    type: str = "string"  # string | int | float | bool | null

    def python(self) -> Any:
        """Convert to native Python value."""
        if self.type == "int":
            return int(self.raw)
        if self.type == "float":
            return float(self.raw)
        if self.type == "bool":
            return self.raw.lower() in ("true", "yes", "on", "1")
        if self.type == "null":
            return None
        return self.raw


@dataclass
class AmlBlock:
    """A single block in an AML document.

    Examples::

        @knowledge mitochondria {
            content = "The mitochondria is the powerhouse."
            topic = "biology"
            tags = ["cell", "energy"]
        }

    Produces::

        AmlBlock(
            tag="knowledge",
            name="mitochondria",
            body=None,
            metadata={"content": "The mitochondria...", "topic": "biology",
                       "tags": ["cell", "energy"]},
        )
    """
    tag: str
    name: Optional[str] = None
    body: Optional[Union[str, list, dict]] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    line: int = 0


@dataclass
class AmlDocument:
    """Top-level container for a parsed AML document.

    Attributes:
        version: AML spec version from ``@aml X.Y`` directive
        blocks:  ordered list of all blocks
        errors:  parse errors (non-fatal, document still usable)
    """
    version: str = "1.0"
    blocks: list[AmlBlock] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def by_tag(self, tag: str) -> list[AmlBlock]:
        """Return all blocks matching *tag* (without ``@`` prefix)."""
        return [b for b in self.blocks if b.tag == tag]

    def by_name(self, name: str) -> Optional[AmlBlock]:
        """Return the first block with matching *name*, or None."""
        for b in self.blocks:
            if b.name == name:
                return b
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict suitable for JSON serialization."""
        out: dict[str, Any] = {"aml": self.version}
        for b in self.blocks:
            key = b.tag
            if b.name:
                key = f"{b.tag}:{b.name}"
            entry: dict[str, Any] = {}
            if b.body is not None:
                entry["body"] = b.body
            if b.metadata:
                entry.update(b.metadata)
            if key in out:
                if not isinstance(out[key], list):
                    out[key] = [out[key]]
                out[key].append(entry)
            else:
                out[key] = entry
        return out
