"""Generated TS contract bindings — the `ts-rs` half of the projection (card 25ebfc0f).

NymVPN annotates its Rust types and lets `cargo test` emit `src/types/tauri.ts`,
so a contract change that isn't regenerated fails at *build* time instead of in
front of a user. `scripts/gen_ts_contracts.py` is our equivalent: one descriptor
(`ToolSpec`) -> HTTP route (`create_router`) -> typed TS, emitted once per
frontend consumer (web, SDK).

Four properties are asserted here:

1. **Freshness** — every committed artifact matches what the registry produces
   right now. Runs in a *subprocess* on purpose: `create_router` writes to the
   process-global `_CONTRACTS` registry as a side effect of projection, so other
   tests in the same session (e.g. `test_contract_projection.py`, which projects
   14 ad-hoc doctor specs) leave rows behind that the committed files never
   contained. In-process comparison would be order-dependent; a fresh
   interpreter reads exactly what boot would.
2. **Completeness** — every registry row carries the fields the frontend
   `ContractDescriptor` type requires (`operation_id`, `module`), so the
   generated file type-checks rather than merely parsing.
3. **Generator owns the bytes** — every output is excluded from Prettier.
   lint-staged runs `prettier --write` over staged `.ts`; requoting the artifact
   (JSON double quotes -> single) makes it drift from its own source of truth and
   turns the freshness gate red on every commit. This is not hypothetical — it
   happened, and it is why `.prettierignore` carries one rule per output.
4. **Outputs agree** — all consumers read byte-identical files, so a capability
   cannot mean two things in two frontends.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "gen_ts_contracts.py"
PRETTIERIGNORE = ROOT / ".prettierignore"

# Import the list rather than restating it: adding an output to the generator
# must extend this gate for free, or the new artifact ships ungated.
sys.path.insert(0, str(ROOT / "scripts"))
from gen_ts_contracts import OUTPUTS  # noqa: E402

# The interpreter roots gen_ts_contracts.py needs to import `routers.*` and
# everything those modules pull in. Mirrors the invocation in the script's
# own docstring.
_PYTHONPATH = [
    ROOT,
    ROOT / "packages" / "mogdb" / "src",
    ROOT / "packages" / "downcraft",
    ROOT / "packages" / "core-py",
    ROOT / "apps" / "api" / "server",
]


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(str(p) for p in _PYTHONPATH)
    # AGENTS.md: a stray `tests` package in ~/.local shadows the repo's tests/.
    env["PYTHONNOUSERSITE"] = "1"
    return env


def test_ts_contract_projection_is_fresh():
    """Every committed artifact matches the live descriptor registry."""
    assert SCRIPT.is_file(), f"missing generator: {SCRIPT}"
    missing = [p for p in OUTPUTS if not p.is_file()]
    assert not missing, (
        "missing generated contract binding(s): "
        f"{', '.join(str(p.relative_to(ROOT)) for p in missing)} — run "
        "python scripts/gen_ts_contracts.py"
    )

    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        cwd=ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, (
        f"contract bindings are stale or the generator failed\n"
        f"stdout: {proc.stdout}\n"
        f"stderr: {proc.stderr}"
    )


def test_every_output_is_exempt_from_prettier():
    """lint-staged must not reformat what the generator owns (see module docstring)."""
    assert PRETTIERIGNORE.is_file(), "missing root .prettierignore"
    ignored = {
        line.strip()
        for line in PRETTIERIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    for output in OUTPUTS:
        rel = str(output.relative_to(ROOT))
        assert rel in ignored, (
            f"{rel} is generated but not in .prettierignore — "
            "prettier will requote it on commit and the freshness gate will fail"
        )


def test_outputs_are_byte_identical():
    """One descriptor, one artifact: no consumer may read a different contract."""
    texts = {str(p.relative_to(ROOT)): p.read_text(encoding="utf-8") for p in OUTPUTS}
    unique = set(texts.values())
    assert len(unique) == 1, f"outputs disagree: {sorted(texts)}"


def test_registry_rows_carry_frontend_descriptor_fields():
    """Every projected row satisfies the frontend's ContractDescriptor shape.

    `operation_id` and `module` are what let a client key a read and say which
    router declared the capability; without them the artifact fails
    `satisfies Record<string, ContractDescriptor>` at typecheck.
    """
    # Importing the pilot registers its own contract as a projection side effect.
    from routers.contracts import router  # noqa: F401  (import registers)
    from infrastructure.contract import get_contracts

    rows = get_contracts()
    assert rows, "no projected contracts registered — create_router did not run"

    for row in rows:
        name = row["name"]
        assert row.get("operation_id"), f"{name} has no operation_id"
        assert row.get("module"), f"{name} has no module"
        for field in ("method", "path", "version", "auth_scope", "idempotent", "params"):
            assert field in row, f"{name} missing descriptor field {field!r}"


def test_operation_ids_are_unique_across_crossings():
    """Two crossings may not claim one operation_id — OpenAPI keys on it."""
    from infrastructure.contract import get_contracts

    rows = get_contracts()
    ids = [row["operation_id"] for row in rows]
    assert len(ids) == len(set(ids)), f"duplicate operation_id: {sorted(ids)}"


def test_generated_file_is_marked_generated_and_self_contained():
    """Header warns off hand-edits; no relative imports, so any tree can host it."""
    for output in OUTPUTS:
        text = output.read_text(encoding="utf-8")
        rel = output.relative_to(ROOT)
        assert "GENERATED by scripts/gen_ts_contracts.py" in text, rel
        assert "do not edit by hand" in text, rel
        assert "export interface ContractDescriptor" in text, rel
        # Self-contained is what lets the same bytes land in web, SDK and mobile.
        assert "from './types'" not in text, f"{rel} imports ./types — not portable"
