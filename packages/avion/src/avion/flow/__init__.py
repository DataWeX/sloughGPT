"""Flow map — turns a recorded event stream into a navigable step graph.

This is the model half of the "map out flows" feature. It consumes the events
that ``EventRecorder`` already produces (including the ``SELECT`` events the
element-picker records) and builds a :class:`FlowGraph` of ordered steps with
edges, ready to render as a timeline, a flowchart, or drive a side-panel viewer.

Rendering is deliberately out of scope here — this module is pure data plus two
serialisation helpers (``to_json`` and ``to_mermaid``) so it stays importable in
a headless test with no UI dependency.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Any, Iterable

# Event types that represent a *navigation* — these open a new flow "page".
_NAVIGATION = {"navigate"}

# Event types that are pure noise for a flow map (lifecycle bookkeeping).
_SKIP = {"session_started", "session_ended"}


@dataclass
class FlowStep:
    """One node in the flow: a recorded event with its resolution detail."""

    seq: int
    type: str
    name: str
    success: bool = True
    error: str = ""
    duration_ms: float = 0.0
    timestamp: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def is_pick(self) -> bool:
        """True when this step was produced by the element-picker (SELECT)."""
        return self.type == "select"

    @property
    def is_navigation(self) -> bool:
        return self.type in _NAVIGATION

    @property
    def label(self) -> str:
        """A short human label for the flow node."""
        return self.name or self.type

    def to_dict(self) -> dict[str, Any]:
        d = {
            "seq": self.seq,
            "type": self.type,
            "name": self.name,
            "label": self.label,
            "success": self.success,
            "duration_ms": self.duration_ms,
        }
        if self.error:
            d["error"] = self.error
        if self.data:
            # Screenshots land in data as raw bytes; the JSON interchange
            # must stay real JSON — bytes become base64 strings here rather
            # than falling through json's default=str mangling ("b'\\x89PN..").
            d["data"] = {
                k: base64.b64encode(v).decode("ascii") if isinstance(v, (bytes, bytearray)) else v
                for k, v in self.data.items()
            }
        return d


@dataclass
class FlowGraph:
    """An ordered list of steps plus the edges that connect them.

    Edges are implicit (step N-1 -> step N) except at navigation boundaries,
    which start a new segment — that's what a viewer renders as separate pages.
    """

    steps: list[FlowStep] = field(default_factory=list)
    session_name: str = ""

    def add(self, step: FlowStep) -> None:
        self.steps.append(step)

    @property
    def picks(self) -> list[FlowStep]:
        """Only the element-picker steps — what the user selected."""
        return [s for s in self.steps if s.is_pick]

    @property
    def failed(self) -> list[FlowStep]:
        return [s for s in self.steps if not s.success]

    def segments(self) -> list[list[FlowStep]]:
        """Split steps into per-navigation segments (for a page-by-page view)."""
        out: list[list[FlowStep]] = [[]]
        for s in self.steps:
            if s.is_navigation and out[-1]:
                out.append([])
            out[-1].append(s)
        return [seg for seg in out if seg]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session": self.session_name,
            "step_count": len(self.steps),
            "pick_count": len(self.picks),
            "failed_count": len(self.failed),
            "steps": [s.to_dict() for s in self.steps],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_mermaid(self) -> str:
        """Render the flow as a mermaid flowchart (for docs / a web viewer)."""
        lines = ["flowchart TD"]
        prev = None
        for s in self.steps:
            node_id = f"s{s.seq}"
            shape_open, shape_close = ("[", "]") if not s.is_pick else ("{{", "}}")
            label = s.label.replace('"', "'")
            lines.append(f'    {node_id}{shape_open}"{label}"{shape_close}')
            if prev:
                lines.append(f"    {prev} --> {node_id}")
            prev = node_id
        # Colour failures red so a broken flow is obvious at a glance.
        for s in self.failed:
            lines.append(f"    style s{s.seq} fill:#fee2e2,stroke:#dc2626")
        return "\n".join(lines)


def build_flow(events: Iterable[Any], session_name: str = "") -> FlowGraph:
    """Build a FlowGraph from an EventRecorder event stream (or raw dicts).

    Accepts :class:`Event` objects or the dicts produced by ``Event.to_dict`` /
    ``EventRecorder.save``, so both live runs and saved JSON can be visualised.
    """
    graph = FlowGraph(session_name=session_name)
    for ev in events:
        if isinstance(ev, dict):
            etype = ev.get("type", "")
            if etype in _SKIP:
                continue
            graph.add(
                FlowStep(
                    seq=int(ev.get("seq", len(graph.steps))),
                    type=etype,
                    name=ev.get("name", ""),
                    success=bool(ev.get("success", True)),
                    error=ev.get("error", ""),
                    duration_ms=float(ev.get("duration_ms", 0.0) or 0.0),
                    timestamp=float(ev.get("timestamp", 0.0) or 0.0),
                    data=ev.get("data", {}) or {},
                )
            )
        else:  # an Event object
            if ev.type.value in _SKIP:
                continue
            graph.add(
                FlowStep(
                    seq=ev.seq,
                    type=ev.type.value,
                    name=ev.name,
                    success=ev.success,
                    error=ev.error,
                    duration_ms=ev.duration_ms,
                    timestamp=ev.timestamp,
                    data=ev.data or {},
                )
            )
    return graph


def load_flow(path: str) -> FlowGraph:
    """Load a flow from a JSON file saved by ``EventRecorder.save``."""
    with open(path) as f:
        payload = json.load(f)
    return build_flow(payload.get("events", []), session_name=payload.get("session", ""))


# The execution half: recorded events -> the flow completed on a live
# browser avion owns (import at the end — complete imports the session,
# which must not be pulled in before the pure map above).
from avion.flow.complete import FlowOutcome, complete_flow  # noqa: E402,F401

# Imported last so viewer.py's TYPE_CHECKING-only import of FlowGraph (and the
# names above) never races package initialisation.
from avion.flow.viewer import render_html, save_html  # noqa: E402

__all__ = [
    "FlowGraph",
    "FlowStep",
    "FlowOutcome",
    "build_flow",
    "complete_flow",
    "load_flow",
    "render_html",
    "save_html",
]
