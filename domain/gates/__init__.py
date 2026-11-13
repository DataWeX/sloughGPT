"""Static gates as owned product code (card 20261112_060632).

Three gates already enforce invariants in CI — command parity (the Command
Rule), domain baseline drift (the catalogue), and the boot router manifest
(the endpoint rule's projection check).  Until now their only consumer was
pytest or a hand-typed `scripts/* --check`; this package extracts each
gate's *check core* into a pure function returning a `GateReport`, so the
CLI (`sloughgpt gates`) and Site Doctor can run them as product features.

Two disciplines hold for everything under `domain/gates/`:

* **read-only** — a gate inspects, it never mutates (Site Doctor's probe
  rule: a probe can never change what it is measuring). Pinned by AST in
  `tests/core-py/test_gates_product_surface.py`.
* **no test-runner dependency** — cores are stdlib + domain imports only;
  pytest suites are thin wrappers over the same functions (Test Rule: the
  product must not depend on the runner).
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["GateReport", "run_gates"]


@dataclass(frozen=True)
class GateReport:
    """Outcome of one gate: findings are blocking, known are baselined debt."""

    name: str
    ok: bool
    findings: list[str] = field(default_factory=list)
    known: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "ok" if self.ok else "FAIL"
        tail = f" ({len(self.known)} known)" if self.known else ""
        return f"{self.name}: {status}{tail}"


def run_gates() -> list[GateReport]:
    """Run every in-scope gate; each is static, read-only, and <5s.

    The gate set is closed by the intake filter (static + read-only +
    pass/fail + fast): adding a gate means extending this tuple *and*
    `tests/core-py/test_gates_product_surface.py`'s expected set together.
    """
    from domain.gates._internal import baseline, manifest, parity

    return [parity.check(), baseline.check(), manifest.check()]
