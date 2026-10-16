"""Spec-content gate: UX_FLOWS.md sections carry their own ui-gate declarations.

Re-expression of card 15c5fb8d after the journey-harness cut (cards 92af1aa0 /
433ccdb9: "whatever it still wants must be re-expressed against
docs/UX_FLOWS.md, not the deleted registry"). No flow registry, no library,
no harness: a flow section declares the user-visible copy it requires (or
forbids) in an HTML comment invisible to rendered readers:

    <!-- ui-gate: must-see: "Start training" | "Quick Import";
                  must-not-see: "Show advanced options" -->

This test fails when the declared copy disappears from apps/web non-test
source — the drift class that hid the missing training glosses and the
4-vs-3-step wizard change. Adding a gate to a new section needs no code
change here, only a comment in the spec.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "docs" / "UX_FLOWS.md"
WEB = REPO / "apps" / "web"

GATE_RE = re.compile(r"<!--\s*ui-gate:\s*(?P<body>.*?)\s*-->", re.S)
CLAUSE_RE = re.compile(
    r"(?P<kind>must-see|must-not-see):\s*(?P<items>.*?)(?=;?\s*(?:must-see|must-not-see|scope)\s*:|$)",
    re.S,
)
SCOPE_RE = re.compile(r"scope:\s*(?P<paths>[^;]+)")
ITEM_RE = re.compile(r'"([^"]+)"')
HEADING_RE = re.compile(r"^###\s+(.*)$", re.M)

SKIP_SUBSTRINGS = ("node_modules", "/dist/", ".test.", ".gen.", "__snapshots__")


def _web_source(scope: list[str] | None = None) -> str:
    """Concatenate non-test web source, optionally scoped to feature dirs.

    Scoping matters: a gate for the training wizard must not go green because
    the phrase 'Start training' happens to appear on an unrelated page (it
    does — consciousness/api-explorer uses it as an API description).
    """
    roots = [WEB / s.strip() for s in scope] if scope else [WEB]
    chunks: list[str] = []
    for root in roots:
        for path in sorted(root.rglob("*.ts*")):
            text = str(path)
            if any(s in text for s in SKIP_SUBSTRINGS):
                continue
            chunks.append(path.read_text(errors="ignore"))
    return "\n".join(chunks)


def _spec_gates() -> list[dict]:
    text = SPEC.read_text()
    headings = [(m.start(), m.group(1).strip()) for m in HEADING_RE.finditer(text)]
    gates: list[dict] = []
    for m in GATE_RE.finditer(text):
        section = next((h for pos, h in reversed(headings) if pos < m.start()), "<no heading>")
        must_see: list[str] = []
        must_not_see: list[str] = []
        body = m.group("body")
        must_see: list[str] = []
        must_not_see: list[str] = []
        for clause in CLAUSE_RE.finditer(body):
            items = ITEM_RE.findall(clause.group("items"))
            if clause.group("kind") == "must-see":
                must_see.extend(items)
            else:
                must_not_see.extend(items)
        scope_match = SCOPE_RE.search(body)
        scope = [p.strip() for p in scope_match.group("paths").split("|")] if scope_match else None
        gates.append(
            {
                "section": section,
                "must_see": must_see,
                "must_not_see": must_not_see,
                "scope": scope,
            }
        )
    return gates


def test_the_spec_actually_declares_gates() -> None:
    """The gate must never silently vanish: at least one ui-gate block exists."""
    gates = _spec_gates()
    assert gates, (
        "docs/UX_FLOWS.md carries no <!-- ui-gate: ... --> declarations — the "
        "content gate is gone. Every flow section with user-visible copy "
        "declares its required copy in a ui-gate comment."
    )


def test_declared_must_see_copy_exists_in_the_app() -> None:
    missing: list[str] = []
    for gate in _spec_gates():
        source = _web_source(gate["scope"])
        for item in gate["must_see"]:
            if item not in source:
                missing.append(f"{gate['section']}: missing {item!r}")
    assert not missing, (
        "UX spec copy vanished from apps/web (drift): "
        + "; ".join(missing)
        + ". Restore the copy or amend docs/UX_FLOWS.md deliberately."
    )


def test_declared_must_not_see_copy_is_absent() -> None:
    present: list[str] = []
    for gate in _spec_gates():
        source = _web_source(gate["scope"])
        for item in gate["must_not_see"]:
            if item in source:
                present.append(f"{gate['section']}: forbidden {item!r} is back")
    assert not present, (
        "UX spec forbids copy that exists in apps/web: "
        + "; ".join(present)
        + ". Remove it or amend the spec decision deliberately (the gate exists "
        "so these decisions cannot be re-taken by drift)."
    )


def test_gate_sections_reference_real_spec_headings() -> None:
    """A ui-gate comment must sit under a real ### section of the spec."""
    text = SPEC.read_text()
    headings = {m.group(1).strip() for m in HEADING_RE.finditer(text)}
    orphans = [g["section"] for g in _spec_gates() if g["section"] not in headings]
    assert not orphans, f"ui-gate comments under nonexistent sections: {orphans}"
