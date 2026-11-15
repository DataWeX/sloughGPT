"""The adapter composition gate — the training-adapters ruling (ADR-002), mechanically enforced.

One gate, three layers (AGENTS.md Test Rule clause 4 — seam gate):

1. **Fail-fast registration** — a broken adapter spec raises at ``register()``
   and never enters the registry (parametrized over the failure modes).
2. **Static contract composition** — walks the whole registry asserting the
   declared in/out contracts compose with the ONE trainer seam (``BatchArray``),
   and that adapters never import the loop (AST pin — static, never executed,
   ``test_command_parity.py`` shape).
3. **One dynamic synthetic smoke** — a tiny fixture batch through the REAL AML
   pipeline (dedup → quality → snapshot → ``BatchArray``), proving runtime shape.

Rules from the ruling: 100% registry coverage from day one (an empty registry
fails — no vacuous pass), no grandfather baseline, synthetic only — never real
credentials or datasets, ONE parametrized gate — not per-adapter tests.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from domain.training._internal.adapter_registry import (
    KNOWN_RECORD_CONTRACTS,
    KNOWN_TASKS,
    AdapterError,
    AdapterRegistry,
    AdapterSpec,
    BatchArray,
    all_specs,
    boot,
)

_BROKEN_DUPLICATE = "text-pretrain"  # the day-one entry; used for the duplicate pin


@pytest.fixture(scope="session")
def registry():
    """The whole registry, populated via boot() — shared by layer (b)."""
    boot()
    return all_specs()


# ── layer (a): fail-fast at registration ───────────────────────────────


class TestFailFastRegistration:
    def test_unknown_task_rejected(self):
        reg = AdapterRegistry()
        with pytest.raises(AdapterError, match="task"):
            reg.register(
                AdapterSpec(
                    name="mystery",
                    task="knowledge-store",
                    consumes=("text-records",),
                    produces=BatchArray,
                    build=lambda: None,
                    module=__name__,
                )
            )

    def test_duplicate_name_rejected(self):
        reg = AdapterRegistry()
        spec = AdapterSpec(
            name="dup",
            task="text-pretrain",
            consumes=("text-records",),
            produces=BatchArray,
            build=lambda: None,
            module=__name__,
        )
        reg.register(spec)
        with pytest.raises(AdapterError, match="duplicate"):
            reg.register(spec)

    def test_empty_consumes_rejected(self):
        reg = AdapterRegistry()
        with pytest.raises(AdapterError, match="consumes"):
            reg.register(
                AdapterSpec(
                    name="no-input",
                    task="text-pretrain",
                    consumes=(),
                    produces=BatchArray,
                    build=lambda: None,
                    module=__name__,
                )
            )

    def test_non_batcharray_produces_rejected(self):
        reg = AdapterRegistry()
        with pytest.raises(AdapterError, match="produces"):
            reg.register(
                AdapterSpec(
                    name="scalar",
                    task="text-pretrain",
                    consumes=("text-records",),
                    produces=int,
                    build=lambda: None,
                    module=__name__,
                )
            )

    def test_non_callable_build_rejected(self):
        reg = AdapterRegistry()
        with pytest.raises(AdapterError, match="build"):
            reg.register(
                AdapterSpec(
                    name="no-build",
                    task="text-pretrain",
                    consumes=("text-records",),
                    produces=BatchArray,
                    build=None,  # type: ignore[arg-type]
                    module=__name__,
                )
            )

    def test_non_spec_rejected(self):
        reg = AdapterRegistry()
        with pytest.raises(AdapterError, match="AdapterSpec"):
            reg.register({"name": "dict-as-spec"})  # type: ignore[arg-type]

    def test_boot_is_idempotent(self):
        boot()
        boot()  # module cache → no duplicate registration
        names = [s.name for s in all_specs()]
        assert names.count(_BROKEN_DUPLICATE) == 1


# ── layer (b): static contract composition over the whole registry ────


class TestStaticComposition:
    def test_registry_nonempty(self, registry):
        """Day-one coverage: an empty registry must never pass the gate."""
        assert registry, "registry is empty — composition gate would be vacuous"

    @pytest.mark.parametrize("field", ("name", "task", "consumes", "produces", "build", "module"))
    def test_spec_complete(self, registry, field):
        for spec in registry:
            assert getattr(spec, field) is not None, f"{spec.name}.{field} is None"

    def test_tasks_are_closed_set(self, registry):
        for spec in registry:
            assert spec.task in KNOWN_TASKS, f"{spec.name}: unknown task {spec.task!r}"

    def test_produces_satisfies_batch_array(self, registry):
        for spec in registry:
            for dunder in ("__len__", "__getitem__", "__iter__"):
                assert hasattr(spec.produces, dunder), (
                    f"{spec.name}: produces {spec.produces!r} lacks {dunder} — "
                    "not BatchArray-conformant"
                )

    def test_consumes_within_closed_contracts(self, registry):
        for spec in registry:
            assert isinstance(spec.consumes, tuple) and spec.consumes, spec.name
            for contract in spec.consumes:
                assert contract in KNOWN_RECORD_CONTRACTS, (
                    f"{spec.name}: undeclared record contract {contract!r}"
                )

    def test_adapters_never_import_the_loop(self, registry):
        """AST pin — static, never executed: adapters stay loop-free."""
        for spec in registry:
            module = importlib.import_module(spec.module)
            source = Path(module.__file__).read_text(encoding="utf-8")
            tree = ast.parse(source)
            for node in ast.walk(tree):
                imported: list[str] = []
                if isinstance(node, ast.Import):
                    imported = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    imported = [node.module or "", *(alias.name for alias in node.names)]
                for name in imported:
                    assert "train_pipeline" not in name, (
                        f"{spec.name}: imports the loop ({name}) — adapters never "
                        "import the training loop (ADR-002)"
                    )
                    assert name != "SloughGPTTrainer", (
                        f"{spec.name}: imports SloughGPTTrainer directly — exactly "
                        "ONE seam, via BatchArray (ADR-002)"
                    )


# ── layer (c): one dynamic synthetic smoke through the real AML pipeline ──


class TestAMLSmoke:
    GOOD_1 = (
        "The mitochondria is the powerhouse of the cell. It converts nutrients "
        "into ATP through oxidative phosphorylation, a process studied extensively "
        "in molecular biology since the mid twentieth century."
    )
    GOOD_2 = (
        "Photosynthesis converts sunlight into chemical energy stored in glucose. "
        "Chloroplasts in plant cells contain chlorophyll, the pigment that absorbs "
        "light most efficiently in the blue and red portions of the spectrum."
    )

    def test_fixture_through_real_pipeline(self, tmp_path: Path):
        from domain.training._internal.cruncher import Cruncher, SnapshotStore
        from domain.training._internal.text_pretrain_adapter import build

        store = SnapshotStore(tmp_path / "crunched")
        cruncher = Cruncher("gate-smoke", store=store)

        # dedup: GOOD_1 repeated; quality: "hi" too short → both stages drop.
        records = [self.GOOD_1, self.GOOD_2, self.GOOD_1, "hi"]
        ref = cruncher.crunch(records, sources=["gate-fixture"])

        assert cruncher.result.records_kept == 2, "dedup/quality stages did not run"
        assert cruncher.result.dropped_duplicate == 1
        assert cruncher.result.dropped_quality == 1

        arr = build(ref)

        # runtime shape: BatchArray, sized, re-iterable, random-access (shufflable)
        assert isinstance(arr, BatchArray)
        assert len(arr) == 2
        first_pass = list(arr)
        assert first_pass == list(arr), "BatchArray must be re-iterable (epochs)"
        assert set(first_pass) == {self.GOOD_1, self.GOOD_2}
        shuffled = [arr[i] for i in (1, 0)]
        assert shuffled == [self.GOOD_2, self.GOOD_1], "random access (shuffle) broken"
