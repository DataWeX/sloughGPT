"""Shared-core contract: every domain producer declares what it produced.

``save_soul(..., record=...)`` is the only moment provenance is knowable — it
is stamped into the header and sidecar at write time and never rewritten, so
no reader can reconstruct who produced a file from its bytes afterwards. The
value surfaces as the CLI's Provenance column and in
``GET /training/checkpoints``.

That is why this needs a contract test rather than ordinary coverage: when a
producer drops ``record=``, the call still succeeds, every unit test still
passes, and the column silently reverts to ``-`` for every file that producer
writes. Nothing else in the suite notices a value that went missing.

The invariant: a ``save_soul(...)`` call anywhere under ``domain/`` must pass
``record=`` bound to a ``SOUL_PROVENANCE_*`` constant — the declared
vocabulary, not a hand-rolled string, so no module grows a private spelling.

``scripts/`` is deliberately exempt: ``benchmark_soul_paths.py`` and
``benchmark_slonet.py`` write throwaway fixtures with ``weights_only=True``,
and a fixture is not an identity-bearing artifact.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOMAIN_DIR = REPO_ROOT / "domain"
SLO_FORMAT = DOMAIN_DIR / "inference" / "_internal" / "slo_format.py"


def _save_soul_calls() -> list[tuple[Path, ast.Call]]:
    """Every bare ``save_soul(...)`` call under domain/.

    Only ``ast.Name`` is matched: ``self.save_soul(...)`` is the engine's
    wrapper, whose inner call lives in ``soul.py`` and is counted there, so
    matching attributes would double-report or demand ``record=`` from a
    method that forwards it.
    """
    calls: list[tuple[Path, ast.Call]] = []
    for py in sorted(DOMAIN_DIR.rglob("*.py")):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "save_soul"
            ):
                calls.append((py, node))
    return calls


def _record_keyword(call: ast.Call) -> ast.keyword | None:
    return next((kw for kw in call.keywords if kw.arg == "record"), None)


def test_every_domain_producer_declares_its_provenance() -> None:
    violations: list[str] = []
    for py, call in _save_soul_calls():
        where = f"{py.relative_to(REPO_ROOT)}:{call.lineno}"
        kw = _record_keyword(call)
        if kw is None:
            violations.append(f"{where}: save_soul(...) has no record= kwarg")
        elif not (isinstance(kw.value, ast.Name) and kw.value.id.startswith("SOUL_PROVENANCE_")):
            violations.append(
                f"{where}: record= must bind a SOUL_PROVENANCE_* constant, "
                f"got {ast.unparse(kw.value)}"
            )
    assert not violations, (
        "A producer that omits record= writes a file whose provenance is "
        "unknowable forever — the call still succeeds, so no other test "
        "notices. Declare it from SOUL_PROVENANCE:\n" + "\n".join(violations)
    )


def test_at_least_the_three_known_producers_are_wired() -> None:
    """Guard against the walk silently matching nothing.

    If a refactor renames the call or moves it behind a wrapper, the test
    above would pass on an empty set. Pinning the known three keeps the
    contract honest: an empty inventory is a failure, not a green run.
    """
    expected = {
        "domain/core/_internal/soul.py",
        "domain/feedback/_internal/lora_eval.py",
        "domain/training/_internal/train_pipeline.py",
    }
    found = {str(py.relative_to(REPO_ROOT)) for py, _ in _save_soul_calls()}
    missing = expected - found
    assert not missing, f"producers no longer calling save_soul directly: {sorted(missing)}"


def test_every_provenance_constant_is_registered() -> None:
    """A constant defined but left out of SOUL_PROVENANCE would be rejected
    by save_soul at write time — i.e. discovered only after a checkpoint
    failed to save. Catch it at import-of-the-test instead."""
    from domain.inference import SOUL_PROVENANCE

    declared = dict(
        re.findall(
            r'^(SOUL_PROVENANCE_[A-Z_]+)\s*=\s*"([^"]+)"',
            SLO_FORMAT.read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
    )
    assert declared, f"no SOUL_PROVENANCE_* constants found in {SLO_FORMAT.name}"
    unregistered = {name: value for name, value in declared.items() if value not in SOUL_PROVENANCE}
    assert not unregistered, (
        "SOUL_PROVENANCE_* constants whose value is missing from the "
        f"SOUL_PROVENANCE tuple (save_soul would reject them): {unregistered}"
    )
