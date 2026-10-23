"""Boundary tests: training router delegates to TrainingEngine, not domain._internal.

Card 50675952 (build order #4, part 1): the router keeps HTTP concerns only;
every `domain.training._internal.service|state|training_advisor|outcome_tracker|monitor`
import moves behind raw-passthrough TrainingEngine methods. Exceptions bubble
(HTTP error classification in the router is unchanged).

The executor/train_pipeline imports (get_training_executor, CheckpointManager,
SloughGPTTrainer) are part 2 (card c48aaaad) and are still allowed here.
"""

from pathlib import Path

from domain.training.engine import get_training_engine

ROUTER_SRC = (
    Path(__file__).resolve().parents[2] / "server" / "training" / "router.py"
).read_text(encoding="utf-8")

# _internal modules that part 1 must fully retire from the router source.
PART1_INTERNAL_MODULES = (
    "domain.training._internal.service",
    "domain.training._internal.state",
    "domain.training._internal.training_advisor",
    "domain.training._internal.outcome_tracker",
    "domain.training._internal.monitor",
)

# Raw-passthrough surface the router consumes (part 1).
DELEGATION_SURFACE = (
    "get_log",
    "list_stored_checkpoints",
    "delete_checkpoint",
    "load_stored_checkpoint",
    "download_checkpoint_path",
    "checkpoint_info",
    "get_all_checkpoint_data",
    "get_state",
    "is_valid_ckpt_name",
    "get_training_tips",
    "recommend_training_config",
    "load_outcomes",
    "get_training_monitor",
    "alert_severity",
)


def test_router_has_no_part1_internal_imports() -> None:
    """The router must not import part-1 _internal modules directly (delegated)."""
    offenders = [m for m in PART1_INTERNAL_MODULES if m in ROUTER_SRC]
    assert offenders == [], (
        f"router.py still imports {offenders} — route these through "
        "TrainingEngine raw-passthrough methods (card 50675952)"
    )


def test_engine_exposes_router_delegation_surface() -> None:
    """Every delegated capability is callable on the engine."""
    engine = get_training_engine()
    missing = [name for name in DELEGATION_SURFACE if not callable(getattr(engine, name, None))]
    assert missing == [], f"TrainingEngine missing raw-passthrough methods: {missing}"


def test_is_valid_ckpt_name_matches_state_regex() -> None:
    """is_valid_ckpt_name preserves VALID_CKPT_NAME semantics."""
    engine = get_training_engine()
    from domain.training._internal.state import VALID_CKPT_NAME

    for name in ("model-v1", "run_42.safetensors", "a" * 64):
        assert engine.is_valid_ckpt_name(name) is bool(VALID_CKPT_NAME.match(name))
    for name in ("../etc/passwd", "a/b", "bad name!", ""):
        assert engine.is_valid_ckpt_name(name) is bool(VALID_CKPT_NAME.match(name))


def test_get_training_monitor_returns_singleton() -> None:
    """get_training_monitor passthrough preserves the monitor singleton."""
    engine = get_training_engine()
    assert engine.get_training_monitor() is engine.get_training_monitor()


def test_alert_severity_passthrough_preserves_errors() -> None:
    """alert_severity returns the enum member and still raises on bad input."""
    engine = get_training_engine()
    from domain.training._internal.monitor import AlertSeverity

    assert engine.alert_severity("warning") is AlertSeverity.WARNING
    try:
        engine.alert_severity("bogus")
    except ValueError:
        pass
    else:  # pragma: no cover - failure path
        raise AssertionError("alert_severity('bogus') must raise ValueError")


def test_get_training_state_exposes_config() -> None:
    """get_state passthrough returns the state object (.config) the router reads."""
    engine = get_training_engine()
    assert hasattr(engine.get_state(), "config")
