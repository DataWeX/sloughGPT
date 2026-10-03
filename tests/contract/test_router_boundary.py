"""Shared-core contract: routers translate at the edge, never into the core.

Enforces the invariant that a FastAPI router module must not import another
router's private ``_instance`` (``from routers.x import _instance``). Doing so
creates a second, silent coupling to the shared kernel that breaks like the
``TrainingEngine.get_executor`` 500 when internals move. Consumers that need a
router capability must use its public facade (e.g.
``routers.inference.build_session_metadata_index`` / ``handle_chat``).

BFF endpoints (``routers/mobile.py``) may shape responses for their shell, but
ride the same shared handlers/accessors the web client calls.
"""

from __future__ import annotations

import re
from pathlib import Path

ROUTERS_DIR = Path(__file__).resolve().parents[2] / "apps" / "api" / "server" / "routers"

_PRIVATE_INSTANCE = re.compile(
    r"from routers\.(?P<module>[a-zA-Z_][a-zA-Z0-9_]*) import\s*.*?\b(?:_instance(?:\s+as\s+\w+)?)"
)
_INSTANCE_ATTR = re.compile(r"routers\.(?P<module>[a-zA-Z_][a-zA-Z0-9_]*)\._instance\b")


def _router_file_violations() -> list[str]:
    bad: list[str] = []
    for py in sorted(ROUTERS_DIR.glob("*.py")):
        src = py.read_text(encoding="utf-8")
        for i, line in enumerate(src.splitlines(), 1):
            if _PRIVATE_INSTANCE.search(line):
                bad.append(f"{py.name}:{i}: {line.strip()}")
            if _INSTANCE_ATTR.search(line):
                bad.append(f"{py.name}:{i}: {line.strip()}")
    return bad


def test_routers_never_import_another_routers_private_instance() -> None:
    violations = _router_file_violations()
    assert violations == [], (
        "Routers must not reach into another router's private _instance "
        "(shared-core contract). Use the router's public facade instead.\n" + "\n".join(violations)
    )


def test_inference_exposes_public_facades_over_its_instance() -> None:
    """The sanctioned pattern: public functions delegating to the singleton."""
    src = (ROUTERS_DIR / "inference.py").read_text(encoding="utf-8")
    for facade in (
        "def build_session_metadata_index()",
        "async def handle_chat(req: ChatRequest)",
    ):
        assert facade in src, f"inference router must keep the public facade: {facade}"


TESTS_DIR = Path(__file__).resolve().parents[2] / "apps" / "api" / "server" / "tests"

# Legacy patch debt: ``patch("domains.<...>")`` targets that never take effect.
# Production code resolves state through the ``domain.*`` shims or the canonical
# ``domain.*._internal`` modules — never through the legacy ``domains.*``
# namespace (the 49 shims are its only importers). A patch aimed at
# ``domains.*`` therefore patches an object nothing binds, so the mock is inert
# and the test passes vacuously. This is the recurring bug class fixed in
# output_buffer / event_buffer / SLNCCompiler wiring.
#
# Each entry is ``<file>|<patch target>``. The contract only allows the set to
# SHRINK (zero new inert patches); every line here is an entry to re-point at
# the module the router actually binds (``domain.*`` or ``domain.*._internal``)
# as part of the on-going legacy sweep.
_LEGACY_PATCH_TARGETS: frozenset[str] = frozenset()


def test_no_new_legacy_domains_patch_targets() -> None:
    """Mock targets must bind the module the router actually resolves.

    ``patch("domains.*")`` patches the legacy namespace that only the shims
    import from, so the mock never intercepts the router's binding. The pinned
    baseline below may only shrink; adding a target is a contract violation.
    """
    targets = {
        f"{py.name}|{re.sub(r'patch\(\s*"', 'patch("', t)}"
        for py in TESTS_DIR.glob("test_*.py")
        for t in re.findall(
            r'patch\(\s*"domains\.[^"]*"', py.read_text(encoding="utf-8"), re.DOTALL
        )
    }
    violations = sorted(targets - _LEGACY_PATCH_TARGETS)
    assert not violations, (
        "New inert legacy patch target(s) added. Patch the module the router "
        "actually binds (canonical domain.* / domain.*._internal).\n" + "\n".join(violations)
    )


# ── Build Order #4: routers must not reach into domain.*._internal ──────────
#
# Routers translate at the edge and call feature engines / public facades
# (TokenizerEngine, TrainingEngine, VoiceEngine, …). Importing
# ``domain.<ns>._internal.*`` from a router re-couples HTTP handlers to
# private modules and defeats the engine boundary. Baseline may only shrink;
# any new ``file|module`` pair is a contract violation.
_INTERNAL_IMPORT = re.compile(r"^\s*(?:from|import)\s+(domain\.[\w.]*_internal(?:\.[\w.]*)?)", re.M)

# Pinned baseline (44 entries after rewiring tokenizer / token_tree /
# cloud_training onto domain.training engines, settings.py onto the
# domain.settings / domain.training / domain.inference facades, and
# consciousness.py onto the domain.cognition facade).
# Re-generate when reducing:
#   python -c "import re,pathlib; ..."
_ROUTER_INTERNAL_IMPORTS: frozenset[str] = frozenset(
    {
        "agents.py|domain.agents._internal.multi",
        "agents.py|domain.agents._internal.run_history",
        "agents.py|domain.agents._internal.system",
        "agents.py|domain.api._internal.sse_envelope",
        "benchmark.py|domain.feedback._internal.response_tracker",
        "benchmark.py|domain.infrastructure._internal.errors",
        "dashboard.py|domain.infrastructure._internal.event_buffer",
        "dashboard.py|domain.settings._internal.persistent",
        "dashboard.py|domain.training._internal.outcome_tracker",
        "dashboard.py|domain.training._internal.service",
        "files.py|domain.cognition._internal.rag_service",
        "lora_eval.py|domain.feedback._internal.lora_eval",
        "lora_eval.py|domain.feedback._internal.per_user_lora",
        "memory.py|domain.memory._internal.config",
        "memory.py|domain.memory._internal.consolidation",
        "memory.py|domain.memory._internal.service",
        "memory.py|domain.memory._internal.task_memory",
        "model_stack.py|domain.training._internal.cache_tags",
        "models.py|domain.models._internal.provider",
        "models.py|domain.slolib._internal.gpu",
        "models.py|domain.training._internal.export",
        "registry.py|domain.infrastructure._internal",
        "self_train.py|domain.infrastructure._internal.errors",
        "shell.py|domain.shell._internal.io",
        "shell.py|domain.shell._internal.repl",
        "shell.py|domain.shell._internal.runtime",
        "status.py|domain.inference._internal.native.engine",
        "system.py|domain.infrastructure._internal.output_buffer",
        "system.py|domain.training._internal",
        "tenants.py|services.auth._internal.models",
        "tenants.py|services.auth._internal.repositories",
        "tokens.py|services.billing._internal.token_service",
        "tools.py|domain.models._internal.provider",
        "users.py|services.auth._internal.models",
        "users.py|services.auth._internal.repositories",
        "vm.py|domain.shell._internal.vm",
        "vm.py|domain.shell._internal.vm_permissions",
        "vm.py|domain.shell._internal.vm_training_bridge",
        "workspaces.py|services.auth._internal.models",
        "workspaces.py|services.auth._internal.repositories",
        "workspaces.py|domain.dataset._internal.repository",
        "workspaces.py|domain.learner._internal.knowledge",
        "world_render.py|domain.shell._internal.simulation",
        "world_render.py|domain.shell._internal.world_render",
    }
)


def _current_router_internal_imports() -> set[str]:
    found: set[str] = set()
    for py in sorted(ROUTERS_DIR.glob("*.py")):
        for m in _INTERNAL_IMPORT.finditer(py.read_text(encoding="utf-8")):
            found.add(f"{py.name}|{m.group(1)}")
    return found


def test_no_new_router_internal_imports() -> None:
    """Routers must call feature engines, not domain.*._internal modules.

    The pinned baseline may only shrink; adding a new ``file|module`` pair
    re-couples an HTTP handler to a private module (Build Order #4).
    """
    current = _current_router_internal_imports()
    violations = sorted(current - _ROUTER_INTERNAL_IMPORTS)
    assert not violations, (
        "New domain.*._internal import(s) in a router. Route through the "
        "feature engine / public facade instead.\n" + "\n".join(violations)
    )
