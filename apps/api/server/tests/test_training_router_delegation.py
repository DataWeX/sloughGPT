"""Boundary tests: training router delegates to TrainingEngine, not domain._internal.

Cards 50675952 (part 1) + c48aaaad (part 2, build order #4): the router keeps
HTTP concerns only; every `domain.training._internal` import — service/state/
training_advisor/outcome_tracker/monitor (part 1) and executor/train_pipeline
recovery internals (part 2) — moves behind raw-passthrough TrainingEngine
methods. Exceptions bubble (HTTP error classification in the router is
unchanged); the single sanctioned mapping is resolve_resume_checkpoint's
ValueError -> 422.
"""

from pathlib import Path

from domain.training.engine import get_training_engine

ROUTER_SRC = (Path(__file__).resolve().parents[2] / "server" / "training" / "router.py").read_text(
    encoding="utf-8"
)

# Raw-passthrough surface the router consumes (part 1 + part 2).
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
    # part 2 (card c48aaaad): recovery internals
    "resolve_resume_checkpoint",
    "run_recovery_training",
    "executor_submit",
)


def test_router_has_no_internal_imports() -> None:
    """Router source must contain no domain.training._internal references at all.

    Part 1 retired service/state/advisor/outcome/monitor; part 2 (c48aaaad)
    retired executor/train_pipeline (CheckpointManager, SloughGPTTrainer).
    """
    assert "domain.training._internal" not in ROUTER_SRC, (
        "router.py still references domain.training._internal — route it "
        "through TrainingEngine raw-passthrough methods (cards 50675952 + c48aaaad)"
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


# ── part 2: recovery internals (card c48aaaad) ──────────────────────────────


def test_resolve_resume_checkpoint_not_resumable_raises(tmp_path, monkeypatch) -> None:
    """Recorded path that is not resumable raises ValueError (router maps to 422)."""
    from domain.training._internal.train_pipeline import CheckpointManager

    engine = get_training_engine()
    monkeypatch.setattr(CheckpointManager, "is_resumable", lambda p: False)
    try:
        engine.resolve_resume_checkpoint("x.soul", str(tmp_path))
    except ValueError as exc:
        assert "missing or unsupported" in str(exc)
    else:  # pragma: no cover - failure path
        raise AssertionError("must raise ValueError for non-resumable checkpoint")


def test_resolve_resume_checkpoint_unreadable_raises(tmp_path, monkeypatch) -> None:
    """Unreadable bundle raises ValueError carrying the loader error."""
    from domain.training._internal.train_pipeline import CheckpointManager

    engine = get_training_engine()
    monkeypatch.setattr(CheckpointManager, "is_resumable", lambda p: True)
    monkeypatch.setattr(
        CheckpointManager, "load_from_path", lambda p: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    try:
        engine.resolve_resume_checkpoint("x.soul", str(tmp_path))
    except ValueError as exc:
        assert "unreadable" in str(exc) and "boom" in str(exc)
    else:  # pragma: no cover - failure path
        raise AssertionError("must raise ValueError for unreadable checkpoint")


def test_resolve_resume_checkpoint_recorded_path_ok(tmp_path, monkeypatch) -> None:
    """A good recorded path returns (path, bundle) unchanged."""
    from domain.training._internal.train_pipeline import CheckpointManager

    engine = get_training_engine()
    bundle = {"weights": object()}
    monkeypatch.setattr(CheckpointManager, "is_resumable", lambda p: True)
    monkeypatch.setattr(CheckpointManager, "load_from_path", lambda p: bundle)
    assert engine.resolve_resume_checkpoint("x.soul", str(tmp_path)) == ("x.soul", bundle)


def test_run_recovery_training_builds_and_trains(monkeypatch) -> None:
    """Builds the trainer from config, trains once with resume+events, returns it."""
    from unittest.mock import MagicMock

    from domain.training._internal import train_pipeline

    trainer_cls = MagicMock()
    trainer_inst = trainer_cls.return_value
    monkeypatch.setattr(train_pipeline, "SloughGPTTrainer", trainer_cls)

    engine = get_training_engine()
    on_progress = lambda info: None  # noqa: E731
    cancel, pause = object(), object()
    got = engine.run_recovery_training(
        config={"data_path": "/tmp/corpus.txt"},
        on_progress=on_progress,
        resume={"w": 1},
        events={"cancel_event": cancel, "pause_event": pause},
    )
    assert got is trainer_inst
    trainer_cls.assert_called_once_with(data_path="/tmp/corpus.txt")
    trainer_inst.train.assert_called_once_with(
        on_progress=on_progress,
        resume=True,
        resume_checkpoint={"w": 1},
        cancel_event=cancel,
        pause_event=pause,
    )


def test_executor_submit_passthrough(monkeypatch) -> None:
    """executor_submit forwards fn+args to the training executor's submit."""
    from domain.training._internal import executor as executor_mod

    class _Exec:
        def __init__(self) -> None:
            self.submitted = []

        def submit(self, fn, *args):
            self.submitted.append((fn, args))

    exec_ = _Exec()
    monkeypatch.setattr(executor_mod, "get_training_executor", lambda: exec_)

    engine = get_training_engine()
    fn = lambda: None  # noqa: E731
    engine.executor_submit(fn, 42)
    assert exec_.submitted == [(fn, (42,))]
