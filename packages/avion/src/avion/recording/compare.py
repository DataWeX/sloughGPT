"""Run-vs-run journey comparison — step traces + screenshot pixels.

A *run* is one recorded pass of the journey suite (``python -m journeys
--record``)::

    <runs_root>/<run-id>/            # run-id sorts chronologically
      run.json                       # meta: id, started, git sha, browser, flows
      <flow-id>/
        result.json                  # TaskResult.to_dict() (holds "steps" trace)
        shots/NN-<step-slug>.png     # per-step checkpoint screenshots
      comparison.json                # written when a previous run existed

Ownership: this module owns the run-directory layout and the diff — it sits
beside the recording artifact schema it reads (placement ruling, card
20261125_193135). ``journeys/runner.py`` writes runs through these helpers
and calls ``compare_runs``; it never reimplements the layout or the diff.

Comparison is **advisory first** (no CI gate yet): ``compare_runs`` reports,
``render_summary`` prints a concise human summary (raw JSON one file away),
``prune_runs`` keeps the history bounded. The CLI exit codes follow ``diff(1)``
— 0 identical, 1 changed, 2 error — so a future gate comes for free.

Stdlib-only at import time — Pillow is probed lazily inside ``diff_shot``
(same capability rule as avion.recording.artifact / avion.vision).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from avion.recording.artifact import new_recording_id

#: How many recorded runs to keep (bounded history — no silent disk growth).
KEEP_RUNS = 10
#: Per-channel delta (0-255) ignored as noise when comparing screenshot pixels.
DEFAULT_PIXEL_TOL = 8
#: Fraction of pixels that must differ before a screenshot counts as changed.
DEFAULT_CHANGED_FRAC = 0.005

RUN_META = "run.json"
RESULT_NAME = "result.json"
COMPARISON_NAME = "comparison.json"
SHOTS_DIR = "shots"
_INDEX_PREFIX = re.compile(r"^\d+-")


def runs_root() -> Path:
    """Where recorded runs live (``SLO_JOURNEY_RUNS``, else the journey cache)."""
    override = os.environ.get("SLO_JOURNEY_RUNS")
    if override:
        return Path(override)
    cache = os.environ.get("SLO_JOURNEY_CACHE", os.path.expanduser("~/.cache/slog-journeys"))
    return Path(cache) / "runs"


def new_run_id() -> str:
    """Sortable-unique run id (same generator as recording artifacts)."""
    return new_recording_id()


def git_sha() -> str:
    """Best-effort short HEAD of the checkout running the journeys."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        return out.strip()
    except Exception:
        return ""


def slugify(step_name: str) -> str:
    """Filesystem-safe slug for a step name."""
    return re.sub(r"[^a-z0-9]+", "-", step_name.lower()).strip("-")[:40] or "step"


def shot_name(index: int, step_name: str, seen: dict[str, int]) -> tuple[str, str]:
    """Return ``(filename, diff-key)`` for a step checkpoint screenshot.

    The filename carries a zero-padded index so humans can read the order;
    the diff-key drops it (and disambiguates repeated step names with a
    ``-2`` suffix) so inserting a step never renumber-matches every shot.
    """
    slug = slugify(step_name)
    n = seen.get(slug, 0) + 1
    seen[slug] = n
    key = slug if n == 1 else f"{slug}-{n}"
    return f"{index:02d}-{key}.png", key


def list_runs(root: Path | None = None) -> list[Path]:
    """Completed run dirs (contain ``run.json``), oldest first by id."""
    root = root or runs_root()
    if not root.is_dir():
        return []
    return sorted(
        (p for p in root.iterdir() if p.is_dir() and (p / RUN_META).exists()),
        key=lambda p: p.name,
    )


def prune_runs(root: Path | None = None, keep: int = KEEP_RUNS) -> list[Path]:
    """Drop the oldest runs beyond ``keep``; returns what was removed.

    Only directories carrying ``run.json`` are ever considered — anything
    else in the runs root is untouched.
    """
    root = root or runs_root()
    runs = list_runs(root)
    excess = runs[:-keep] if keep > 0 and len(runs) > keep else []
    for path in excess:
        shutil.rmtree(path, ignore_errors=True)
    return excess


def _flow_shots(flow_dir: Path) -> dict[str, Path]:
    """Map diff-key → path for a flow's checkpoint screenshots."""
    shots = flow_dir / SHOTS_DIR
    if not shots.is_dir():
        return {}
    out: dict[str, Path] = {}
    for path in sorted(shots.iterdir()):
        if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
            out[_INDEX_PREFIX.sub("", path.stem)] = path
    return out


def _load_run(run_dir: Path) -> dict[str, Any]:
    """Read a run dir into ``{id, flows: {flow_id: {result, shots}}}``."""
    if not run_dir.is_dir():
        raise FileNotFoundError(f"not a run directory: {run_dir}")
    flows: dict[str, Any] = {}
    for path in sorted(run_dir.iterdir()):
        if not path.is_dir():
            continue
        result: dict[str, Any] | None = None
        result_path = path / RESULT_NAME
        if result_path.exists():
            try:
                result = json.loads(result_path.read_text())
            except (OSError, json.JSONDecodeError):
                result = None
        flows[path.name] = {"result": result, "shots": _flow_shots(path)}
    return {"id": run_dir.name, "flows": flows}


def diff_steps(
    old_steps: list[dict[str, Any]], new_steps: list[dict[str, Any]]
) -> list[dict[str, str]]:
    """Sequence-diff two step traces.

    Returns change dicts ``{kind, step, detail}`` where kind is one of
    ``added`` / ``removed`` / ``moved`` / ``status``. A name that left one
    position and appeared at another is reported as ``moved``, never as an
    add+remove pair. Status changes pair steps by position within matching
    name blocks (``SequenceMatcher`` alignment).
    """
    old_names = [s.get("name", "?") for s in old_steps]
    new_names = [s.get("name", "?") for s in new_steps]
    old_status = [s.get("status", "") for s in old_steps]
    new_status = [s.get("status", "") for s in new_steps]

    added: list[str] = []
    removed: list[str] = []
    changes: list[dict[str, str]] = []
    matcher = SequenceMatcher(a=old_names, b=new_names, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                if old_status[i1 + k] != new_status[j1 + k]:
                    changes.append(
                        {
                            "kind": "status",
                            "step": old_names[i1 + k],
                            "detail": (
                                f"{old_status[i1 + k] or '?'} -> {new_status[j1 + k] or '?'}"
                            ),
                        }
                    )
        elif tag == "insert":
            added.extend(new_names[j1:j2])
        elif tag == "delete":
            removed.extend(old_names[i1:i2])
        else:  # replace — renamed window: old side removed, new side added
            removed.extend(old_names[i1:i2])
            added.extend(new_names[j1:j2])

    moved = sorted(set(added) & set(removed))
    for name in moved:
        changes.append({"kind": "moved", "step": name, "detail": "position changed"})
    added = [n for n in added if n not in set(moved)]
    removed = [n for n in removed if n not in set(moved)]
    head = [{"kind": "added", "step": n, "detail": ""} for n in added]
    head += [{"kind": "removed", "step": n, "detail": ""} for n in removed]
    return head + changes


def diff_shot(
    name: str,
    old: Path | None,
    new: Path | None,
    *,
    pixel_tol: int = DEFAULT_PIXEL_TOL,
    changed_frac: float = DEFAULT_CHANGED_FRAC,
) -> dict[str, Any]:
    """Compare one checkpoint screenshot (or note it as added/removed).

    Statuses: ``added`` / ``removed`` / ``changed`` / ``unchanged`` /
    ``size-changed`` / ``unavailable``. ``fraction`` is the share of pixels
    whose per-channel delta exceeds ``pixel_tol`` (1.0 for size changes).
    Degrades to ``unavailable`` when Pillow is missing or a file is
    unreadable — never raises.
    """
    if old is None:
        return {"shot": name, "status": "added", "fraction": 1.0, "note": ""}
    if new is None:
        return {"shot": name, "status": "removed", "fraction": 1.0, "note": ""}
    try:
        from PIL import Image, ImageChops
    except ImportError:
        return {"shot": name, "status": "unavailable", "fraction": 0.0, "note": "Pillow missing"}
    try:
        with Image.open(old) as a, Image.open(new) as b:
            if a.size != b.size:
                return {
                    "shot": name,
                    "status": "size-changed",
                    "fraction": 1.0,
                    "note": f"{a.size[0]}x{a.size[1]} -> {b.size[0]}x{b.size[1]}",
                }
            diff = ImageChops.difference(a.convert("RGB"), b.convert("RGB"))
            r, g, bch = diff.split()

            def _mask(band: Any) -> Any:
                return band.point(lambda p: 255 if p > pixel_tol else 0)

            mask = ImageChops.lighter(ImageChops.lighter(_mask(r), _mask(g)), _mask(bch))
            changed_px = mask.histogram()[255]
            total = a.size[0] * a.size[1]
            fraction = changed_px / total if total else 0.0
    except OSError as exc:
        return {"shot": name, "status": "unavailable", "fraction": 0.0, "note": str(exc)}
    status = "changed" if fraction > changed_frac else "unchanged"
    return {"shot": name, "status": status, "fraction": round(fraction, 4), "note": ""}


_SHOT_CHANGE_STATUSES = ("changed", "added", "removed", "size-changed")


@dataclass
class FlowDiff:
    """One flow's comparison: status transition + step + screenshot changes."""

    flow: str
    old_status: str | None
    new_status: str | None
    steps: list[dict[str, str]] = field(default_factory=list)
    shots: list[dict[str, Any]] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return (
            self.old_status != self.new_status
            or bool(self.steps)
            or any(s["status"] in _SHOT_CHANGE_STATUSES for s in self.shots)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "flow": self.flow,
            "old_status": self.old_status,
            "new_status": self.new_status,
            "changed": self.changed,
            "steps": self.steps,
            "shots": self.shots,
        }


@dataclass
class RunComparison:
    """Full run-vs-run comparison."""

    old_id: str
    new_id: str
    flows: list[FlowDiff] = field(default_factory=list)
    flows_added: list[str] = field(default_factory=list)
    flows_removed: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.flows_added or self.flows_removed or any(f.changed for f in self.flows))

    def to_dict(self) -> dict[str, Any]:
        return {
            "old": self.old_id,
            "new": self.new_id,
            "changed": self.changed,
            "flows_added": self.flows_added,
            "flows_removed": self.flows_removed,
            "flows": [f.to_dict() for f in self.flows],
        }


def compare_runs(
    old_dir: Path | str,
    new_dir: Path | str,
    *,
    pixel_tol: int = DEFAULT_PIXEL_TOL,
    changed_frac: float = DEFAULT_CHANGED_FRAC,
) -> RunComparison:
    """Compare two recorded runs: step traces + checkpoint screenshots."""
    old = _load_run(Path(old_dir))
    new = _load_run(Path(new_dir))
    old_flows = old["flows"]
    new_flows = new["flows"]

    flows: list[FlowDiff] = []
    for flow_id in sorted(set(old_flows) & set(new_flows)):
        old_flow = old_flows[flow_id]
        new_flow = new_flows[flow_id]
        old_result = old_flow.get("result") or {}
        new_result = new_flow.get("result") or {}
        steps = diff_steps(old_result.get("steps", []), new_result.get("steps", []))
        old_shots = old_flow["shots"]
        new_shots = new_flow["shots"]
        shots = [
            diff_shot(
                key,
                old_shots.get(key),
                new_shots.get(key),
                pixel_tol=pixel_tol,
                changed_frac=changed_frac,
            )
            for key in sorted(set(old_shots) | set(new_shots))
        ]
        flows.append(
            FlowDiff(
                flow=flow_id,
                old_status=old_result.get("status"),
                new_status=new_result.get("status"),
                steps=steps,
                shots=shots,
            )
        )

    return RunComparison(
        old_id=old["id"],
        new_id=new["id"],
        flows=flows,
        flows_added=sorted(set(new_flows) - set(old_flows)),
        flows_removed=sorted(set(old_flows) - set(new_flows)),
    )


def render_summary(cmp: RunComparison) -> str:
    """Concise human summary — changed flows only, raw diff one file away."""
    lines = [f"compare {cmp.old_id} -> {cmp.new_id}"]
    n_changed = sum(1 for f in cmp.flows if f.changed)
    detail = f"  flows: {len(cmp.flows)} compared, {n_changed} changed"
    if cmp.flows_added:
        detail += f", +{len(cmp.flows_added)} added ({', '.join(cmp.flows_added)})"
    if cmp.flows_removed:
        detail += f", -{len(cmp.flows_removed)} removed ({', '.join(cmp.flows_removed)})"
    lines.append(detail)

    for flow in (f for f in cmp.flows if f.changed):
        parts: list[str] = []
        if flow.old_status != flow.new_status:
            parts.append(f"status {flow.old_status} -> {flow.new_status}")
        for change in flow.steps:
            text = f"step {change['kind']} {change['step']}"
            if change["detail"]:
                text += f" ({change['detail']})"
            parts.append(text)
        shot_changes = [s for s in flow.shots if s["status"] in _SHOT_CHANGE_STATUSES]
        for shot in shot_changes[:5]:
            text = f"shot {shot['shot']} {shot['status']}"
            if shot["status"] == "changed":
                text += f" {shot['fraction'] * 100:.1f}%"
            elif shot["note"]:
                text += f" ({shot['note']})"
            parts.append(text)
        if len(shot_changes) > 5:
            parts.append(f"+{len(shot_changes) - 5} more shots")
        lines.append(f"  [!] {flow.flow}: " + "; ".join(parts))

    if not cmp.changed:
        lines.append("  no differences")
    unavailable = sum(1 for f in cmp.flows for s in f.shots if s["status"] == "unavailable")
    if unavailable:
        lines.append(f"  note: {unavailable} screenshot comparison(s) unavailable")
    return "\n".join(lines)
