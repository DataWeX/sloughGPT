"""What an environment looked like at one moment — the model's input.

An observation is the *only* thing a driver above the port should need in order
to decide what to do next. It is deliberately small and backend-agnostic:

    url, title, accessibility_tree, screenshot, viewport, extra

Every field except ``url`` may be absent, and absence is **declared** rather
than faked. A backend that cannot screenshot yields ``screenshot=None`` and the
field shows up in :meth:`Observation.absent` — never a zero-byte PNG that reads
as "the page was blank". That distinction is the whole point: evidence that
cannot tell "missing" from "empty" is not evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

_OBSERVATION_FIELDS = (
    "title",
    "accessibility_tree",
    "screenshot",
    "viewport",
    "extra",
)


@dataclass(frozen=True)
class Observation:
    """A snapshot of an environment, with per-field absence tracked."""

    url: str
    title: str | None = None
    accessibility_tree: Mapping[str, Any] | None = None
    screenshot: bytes | None = None
    viewport: tuple[int, int] | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.url, str):
            raise TypeError("Observation.url must be a string")
        object.__setattr__(self, "extra", dict(self.extra))

    # -- absence ------------------------------------------------------------
    def absent(self) -> tuple[str, ...]:
        """Names of the fields this environment could not provide."""
        return tuple(name for name in _OBSERVATION_FIELDS if getattr(self, name) in (None, {}))

    def has(self, name: str) -> bool:
        if name not in _OBSERVATION_FIELDS:
            raise KeyError(f"unknown observation field {name!r}")
        return name not in self.absent()

    # -- projections --------------------------------------------------------
    def to_dict(self, *, include_screenshot_bytes: bool = False) -> dict[str, Any]:
        """JSON-safe projection.

        Screenshot bytes are described, not embedded, unless explicitly asked
        for — a trajectory record should stay diffable and small.
        """
        shot: dict[str, Any] | None = None
        if self.screenshot is not None:
            shot = {"bytes": len(self.screenshot)}
            if include_screenshot_bytes:
                import base64

                shot["b64"] = base64.b64encode(self.screenshot).decode("ascii")
        return {
            "url": self.url,
            "title": self.title,
            "accessibility_tree": dict(self.accessibility_tree)
            if self.accessibility_tree is not None
            else None,
            "screenshot": shot,
            "viewport": list(self.viewport) if self.viewport else None,
            "extra": dict(self.extra),
            "absent": list(self.absent()),
        }

    def with_extra(self, **values: Any) -> Observation:
        """Copy with additional adapter-specific facts (never replaces core fields)."""
        overlap = set(values) & set(_OBSERVATION_FIELDS)
        if overlap:
            raise ValueError(
                f"{sorted(overlap)} are core observation fields — build a new "
                "Observation instead of smuggling them through extra"
            )
        return replace(self, extra={**self.extra, **values})

    # -- model-facing text --------------------------------------------------
    def describe(self, *, max_tree_chars: int = 1400) -> str:
        """Compact textual form for a model prompt (tree truncated, absence named)."""
        lines = [f"url: {self.url}"]
        if self.title:
            lines.append(f"title: {self.title}")
        if self.accessibility_tree is not None:
            import json

            tree = json.dumps(self.accessibility_tree, ensure_ascii=False)
            if len(tree) > max_tree_chars:
                tree = tree[:max_tree_chars] + "…(truncated)"
            lines.append(f"tree: {tree}")
        if self.screenshot is not None:
            lines.append(f"screenshot: {len(self.screenshot)} bytes")
        if self.viewport:
            lines.append(f"viewport: {self.viewport[0]}x{self.viewport[1]}")
        for key, value in self.extra.items():
            lines.append(f"{key}: {value}")
        missing = self.absent()
        if missing:
            lines.append(f"unavailable: {', '.join(missing)}")
        return "\n".join(lines)
