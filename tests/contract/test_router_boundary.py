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
_LEGACY_PATCH_TARGETS: frozenset[str] = frozenset(
    {
        'test_agents_router.py|patch("domains.agents.system.AgentSystem.execute"',
        'test_cloud_training_router.py|patch("domains.training.cloud.get_provider"',
        'test_feedback_controller_wiring.py|patch("domains.training.service.get_state"',
        'test_infer_router.py|patch("domains.models.provider.get_provider"',
        'test_kb_router.py|patch("domains.cognitive.rag_service.get_rag_service"',
        'test_learner_router.py|patch("domains.learner.get_learner"',
        'test_lora_eval_router.py|patch("domains.feedback.lora_eval.get_lora_evaluator"',
        'test_lora_eval_router.py|patch("domains.feedback.per_user_lora.get_per_user_lora"',
        'test_meta_weights_router.py|patch("domains.feedback.get_meta_weight_manager"',
        'test_multimodal_router.py|patch("domains.training.executor.get_training_executor"',
        'test_multimodal_router.py|patch("domains.training.video_trainer.list_video_checkpoints"',
        'test_plugins_router.py|patch("domains.plugins.get_plugin_manager"',
        'test_process_guard_controller.py|patch("domains.infrastructure.model_registry.get_model_registry"',
        'test_process_guard_controller.py|patch("domains.infrastructure.model_resolver.get_model_dir"',
        'test_process_guard_controller.py|patch("domains.infrastructure.process_guard.ProcessGuard"',
        'test_registry_router.py|patch("domains.infrastructure.model_registry.get_model_registry"',
        'test_souls_router.py|patch("domains.context.managers.get_trait_config"',
        'test_souls_router.py|patch("domains.inference.slo_manager.get_slo_manager"',
        'test_unified_training_routes.py|patch("domains.training.service.get_turbo_status"',
        'test_unified_training_routes.py|patch("domains.training.service.run_turbo_worker"',
        'test_vector_router.py|patch("domains.inference.vector_store.create_vector_store"',
        'test_workflow_router.py|patch("domains.feedback.get_feedback_workflow"',
        'test_world_render_router.py|patch("domains.shell.simulation.Simulation"',
        'test_world_render_router.py|patch("domains.shell.world_render.NeuralRenderBridge"',
        'test_world_render_router.py|patch("domains.shell.world_render.RenderBridge"',
    }
)


def test_no_new_legacy_domains_patch_targets() -> None:
    """Mock targets must bind the module the router actually resolves.

    ``patch("domains.*")`` patches the legacy namespace that only the shims
    import from, so the mock never intercepts the router's binding. The pinned
    baseline below may only shrink; adding a target is a contract violation.
    """
    targets = {
        f"{py.name}|{t}"
        for py in TESTS_DIR.glob("test_*.py")
        for t in re.findall(r'patch\("domains\.[^"]*"', py.read_text(encoding="utf-8"))
    }
    violations = sorted(targets - _LEGACY_PATCH_TARGETS)
    assert not violations, (
        "New inert legacy patch target(s) added. Patch the module the router "
        "actually binds (canonical domain.* / domain.*._internal).\n" + "\n".join(violations)
    )
