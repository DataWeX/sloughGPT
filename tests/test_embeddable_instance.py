"""Embeddable core facade — SloughInstance lifecycle (card 20261116_095706).

Phase-1 contract: an instance any host can start, call contracts on
in-process, and stop — with no FastAPI/uvicorn in the loop. Covers the
lifecycle state machine, in-process contract invocation through the shared
ToolRegistry, idempotence, and the profile/model_loader seam.

Each test that needs a running instance wires the real singletons (task
queue, routine runner, registry) and stops them in teardown — one instance
per process at a time is the documented phase-1 constraint.
"""

from __future__ import annotations

import pytest

from domain.infrastructure import SloughInstance, create_instance


@pytest.fixture()
def running_instance():
    inst = SloughInstance.create(profile="quick")
    assert inst.start(), "quick-profile instance must start"
    assert inst.state == "running"
    yield inst
    assert inst.stop(), "instance must stop cleanly"


def test_quick_instance_start_invoke_stop(running_instance):
    """Full happy path: wired, contracts listed, contract executed."""
    contracts = running_instance.list_contracts()
    names = {c["name"] for c in contracts}
    assert "calculator" in names, f"expected calculator in {sorted(names)}"
    result = running_instance.invoke("calculator", {"expression": "2 + 2"})
    assert result.success, f"calculator failed: {result.error}"
    assert "4" in result.output


def test_start_is_idempotent(running_instance):
    """A second start() on a running instance is a no-op that returns True."""
    assert running_instance.start() is True
    assert running_instance.state == "running"


def test_invoke_unknown_contract_returns_failure(running_instance):
    """Unknown names resolve to a failed ToolResult, not an exception."""
    result = running_instance.invoke("no_such_contract_xyz", {})
    assert result.success is False
    assert "Unknown tool" in (result.error or "")


def test_invoke_requires_running_instance():
    """invoke() outside the running state raises — fail loud, not None."""
    inst = create_instance(profile="quick")
    with pytest.raises(RuntimeError, match="not invocable"):
        inst.invoke("calculator", {"expression": "1 + 1"})
    # stopped instance too
    assert inst.stop() is True  # no-op from created
    with pytest.raises(RuntimeError, match="not invocable"):
        inst.invoke("calculator", {"expression": "1 + 1"})


def test_stop_before_start_is_noop():
    """stop() from created state is a safe no-op (embedder cleanup paths)."""
    inst = create_instance(profile="quick")
    assert inst.stop() is True
    assert inst.state == "stopped"


def test_invalid_profile_rejected_early():
    """Unknown profile fails at construction, not deep inside hooks."""
    with pytest.raises(ValueError, match="unknown profile"):
        SloughInstance.create(profile="nope")


def test_full_profile_without_loader_is_model_free():
    """full profile with no loader starts anyway — model-free instance."""
    inst = create_instance(profile="full")
    try:
        assert inst.start(), "full profile must start without a loader"
        assert inst.model is None
        assert inst.state == "running"
    finally:
        assert inst.stop() is True


def test_model_loader_seam_runs_in_full_profile():
    """A provided loader is invoked with config; its return becomes .model."""
    seen: dict = {}

    def loader(cfg):
        seen["cfg"] = cfg
        return "MODEL_HANDLE"

    inst = create_instance(profile="full", model_loader=loader)
    try:
        assert inst.start(), "full profile with loader must start"
        assert inst.model == "MODEL_HANDLE"
        assert "cfg" in seen, "loader must receive the resolved config"
    finally:
        assert inst.stop() is True


def test_lazy_barrel_export():
    """The public barrel exposes the facade lazily (cold-start safe)."""
    import domain.infrastructure as infra

    assert infra.SloughInstance is SloughInstance
    assert callable(infra.create_instance)
