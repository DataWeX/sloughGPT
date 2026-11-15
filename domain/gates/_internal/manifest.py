"""Router-manifest gate core — the boot projection stays honest.

Reuses `scripts/gen_router_manifest.py` primitives directly (port, don't
rebuild): `scan_modules()` is the on-disk truth (public modules exposing a
module-level `router`), `reconcile()` applies the manifest rules (existing
order preserved — Starlette matches routes in registration order), and the
gate compares the rendered expectation against the committed file. This is
the `--check` half of that script, as a function.
"""

from __future__ import annotations

import sys
from pathlib import Path

from domain.gates import GateReport

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import gen_router_manifest as _manifest  # noqa: E402  (sibling script, not a package)


def check() -> GateReport:
    """Manifest matches what the generator would write (stale/added/renamed)."""
    on_disk = _manifest.scan_modules()
    if not on_disk:
        return GateReport(
            name="router-manifest",
            ok=False,
            findings=[f"no router modules found in {_manifest.ROUTERS_DIR}"],
            known=[],
        )
    current = _manifest.read_manifest() or []
    result, stale, added = _manifest.reconcile(current, on_disk)
    expected = _manifest.render(result)
    actual = (
        _manifest.MANIFEST_PATH.read_text(encoding="utf-8")
        if _manifest.MANIFEST_PATH.is_file()
        else None
    )
    if actual == expected:
        return GateReport(name="router-manifest", ok=True, findings=[], known=[])
    findings: list[str] = []
    for name in stale:
        findings.append(f"stale entry (module gone): {name}")
    for name in added:
        findings.append(f"unregistered module: {name}")
    if not findings:
        findings.append("manifest content differs from generated output")
    findings.append("run: python scripts/gen_router_manifest.py")
    return GateReport(name="router-manifest", ok=False, findings=findings, known=[])
