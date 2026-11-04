"""Training-device parity gate: the NumPy engine vs each accelerator backend.

What this proves, and why it is separate from the per-op GPU tests:

* **Op parity is not trajectory parity.** ``test_slolib_gpu.py`` checks one op
  against a numpy reference. A training step does forward, backward, gradient
  accumulation and an optimizer update over ~20 dispatched matmuls; a backend
  that is right per-op can still be wrong per-step (dtype promotion, in-place
  aliasing, accumulation order). This module trains the same model, from the
  same weights, on the same batches, under two backends and compares the loss
  trajectory and the resulting parameters.
* **A silent CPU fallback is a false pass.** ``test_..._tripwire`` asserts the
  device arm actually dispatched matmuls (``dispatch_stats()["gpu_matmuls"]``),
  so "GPU parity" can never be green because the engine quietly stayed on NumPy
  — the degradation class the dispatch counters exist for.
* **The gate itself is tested.** ``test_drifted_backend_is_caught`` feeds the
  comparator a backend with a deliberate 1% matmul error and asserts it is
  reported as a mismatch. A parity gate that cannot fail is decoration.

Torch is optional: ``select_backend("torch")`` skips the arm when torch is not
installed, and CI/CPU-only hosts still run the NumPy, pin, and self-test arms.
"""

from __future__ import annotations

import importlib.util
import threading

import numpy as np
import pytest

from domain.slolib._internal import gpu as g
from domain.slolib._internal.gpu import (
    accelerator_for,
    dispatch_stats,
    override_accelerator,
    pop_accelerator,
    push_accelerator,
    select_backend,
)

_LOSS_TOL = 2e-3
_PARAM_TOL = 5e-3
_BACKEND_NAMES = ("cuda", "torch")


# =============================================================================
# Helpers
# =============================================================================


def _backend_or_skip(name: str):
    try:
        return select_backend(name)
    except (RuntimeError, ValueError):
        pytest.skip(f"{name} backend unavailable on this host")


@pytest.fixture(params=_BACKEND_NAMES)
def backend(request):
    return _backend_or_skip(request.param)


def _build_model(seed: int = 0):
    from domain.models import SloughGPTModel

    np.random.seed(seed)
    return SloughGPTModel(
        vocab_size=65,
        n_embed=128,
        n_layer=2,
        n_head=4,
        block_size=64,
        dropout=0.0,
    )


def _batches(count: int, seed: int = 7) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    return [(rng.integers(0, 65, (4, 64)), rng.integers(0, 65, (4, 64))) for _ in range(count)]


def _snapshot(model) -> list[np.ndarray]:
    return [np.array(p.data, copy=True) for p in model.parameters()]


def _restore(model, snapshot: list[np.ndarray]) -> None:
    from domain.training._internal.training_handler import zero_grads

    for param, saved in zip(model.parameters(), snapshot, strict=True):
        np.copyto(param.data, saved)
    zero_grads(list(model.parameters()))


def _run(model, batches, accel) -> tuple[list[float], list[np.ndarray]]:
    """Run identical steps under one backend; return losses + final params."""
    from domain.training._internal.slonet import SloAdamW
    from domain.training._internal.training_handler import zero_grads

    optimizer = SloAdamW(lr=1e-3, weight_decay=0.0)
    losses: list[float] = []
    previous = push_accelerator(accel)
    try:
        for x, y in batches:
            model.train()
            _logits, loss = model(x, y)
            loss.backward()
            optimizer.step([p for p in model.parameters() if p.grad is not None])
            zero_grads(list(model.parameters()))
            losses.append(float(loss.item()))
    finally:
        pop_accelerator(previous)
    return losses, _snapshot(model)


def _max_rel_drift(a: list[float], b: list[float]) -> float:
    """Largest relative difference between two loss trajectories."""
    assert len(a) == len(b)
    return max(abs(x - y) / max(abs(x), 1e-9) for x, y in zip(a, b, strict=True))


def _max_param_drift(a: list[np.ndarray], b: list[np.ndarray]) -> float:
    return max(
        float(np.max(np.abs(x - y)) / max(float(np.max(np.abs(x))), 1e-9))
        for x, y in zip(a, b, strict=True)
    )


class _DriftedBackend(g._CPUBackend):
    """CPU math with a deliberate 1% matmul error — the gate's negative control."""

    name = "cuda"  # non-cpu name so the engine dispatches to it

    def matmul(self, a, b):
        return super().matmul(a, b) * 1.01


# =============================================================================
# Selection: a declaration resolves or refuses — it never degrades silently
# =============================================================================


def test_cpu_resolves_to_the_numpy_backend():
    assert accelerator_for("cpu").name == "cpu"


def test_auto_is_never_torch_unless_asked():
    """Default detection keeps the historical priority order."""
    assert accelerator_for("auto").name != "torch"


def test_unknown_backend_refuses():
    with pytest.raises(ValueError):
        select_backend("banana")


def test_unavailable_backend_refuses(monkeypatch):
    class _Never(g._CPUBackend):
        def is_available(self) -> bool:
            return False

    monkeypatch.setattr(g, "_backend_factory", lambda name: _Never if name == "cpu" else None)
    with pytest.raises(RuntimeError):
        select_backend("cpu")


def test_env_declaration_selects_the_backend(monkeypatch):
    monkeypatch.setenv("SLO_ACCEL_BACKEND", "cpu")
    assert g._detect_best_backend().name == "cpu"


def test_env_declaration_that_cannot_be_served_still_boots(monkeypatch):
    """The boot probe logs and serves CPU; ``accelerator_for`` is the loud path."""
    monkeypatch.setenv("SLO_ACCEL_BACKEND", "banana")
    assert g._detect_best_backend().name == "cpu"


def test_trainer_config_device_is_no_longer_collapsed():
    from domain.training._internal.train_pipeline import TrainerConfig

    assert TrainerConfig(device="auto").device == "auto"
    assert TrainerConfig(device=None).device == "cpu"
    with pytest.raises(ValueError):
        TrainerConfig(device="tpu")


def test_setup_device_records_the_resolved_backend():
    from domain.training._internal.train_pipeline import SloughGPTTrainer, TrainerConfig

    trainer = object.__new__(SloughGPTTrainer)
    trainer.config = TrainerConfig(device="cpu")
    assert trainer._setup_device() == "cpu"
    assert trainer._accelerator.name == "cpu"


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is not None,
    reason="torch is installed here; the refusal path needs it absent",
)
def test_setup_device_refuses_an_unavailable_declaration():
    from domain.training._internal.train_pipeline import SloughGPTTrainer, TrainerConfig

    trainer = object.__new__(SloughGPTTrainer)
    trainer.config = TrainerConfig(device="torch")
    with pytest.raises(RuntimeError):
        trainer._setup_device()


# =============================================================================
# The pin: per-thread, and it actually changes where the math runs
# =============================================================================


def test_cpu_pin_forces_numpy_matmuls():
    from domain.training._internal.slonet import _ensure, _matmul

    a = _ensure(np.random.rand(128, 128).astype(np.float32))
    b = _ensure(np.random.rand(128, 128).astype(np.float32))
    before = dispatch_stats()
    previous = push_accelerator(g._CPUBackend())
    try:
        _matmul(a, b)
    finally:
        pop_accelerator(previous)
    delta = {key: dispatch_stats()[key] - before[key] for key in dispatch_stats()}
    assert delta["gpu_matmuls"] == 0
    assert delta["cpu_matmuls"] > 0


def test_pin_is_thread_scoped():
    previous = push_accelerator(g._CPUBackend())
    try:
        assert override_accelerator() is not None
        seen: list[object] = []

        def worker() -> None:
            seen.append(override_accelerator())

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        assert seen == [None], "a pin leaked into another thread"
    finally:
        pop_accelerator(previous)
    assert override_accelerator() is None


# =============================================================================
# Trajectory parity (the reason this module exists)
# =============================================================================


def test_trajectory_parity_with_backend(backend):
    model = _build_model()
    initial = _snapshot(model)
    batches = _batches(3)

    losses_cpu, params_cpu = _run(model, batches, g._CPUBackend())

    _restore(model, initial)
    before = dispatch_stats()
    losses_dev, params_dev = _run(model, batches, backend)
    delta = {key: dispatch_stats()[key] - before[key] for key in dispatch_stats()}

    assert delta["gpu_matmuls"] > 0, (
        f"{backend.name} arm never dispatched a matmul — it silently ran on the CPU, "
        "so the parity assertion below would prove nothing"
    )
    assert _max_rel_drift(losses_cpu, losses_dev) < _LOSS_TOL
    assert _max_param_drift(params_cpu, params_dev) < _PARAM_TOL


def test_drifted_backend_is_caught():
    """Negative control: the comparator must fail on a 1% matmul error.

    Checks the gate as written — either arm (loss trajectory or parameters)
    tripping is a catch. Parameters are the sensitive arm here: cross-entropy
    is ~4.6, so a 1% kernel error moves the loss by well under the loss
    tolerance at three steps while moving weights far past theirs.
    """
    model = _build_model()
    initial = _snapshot(model)
    batches = _batches(3)

    losses_cpu, params_cpu = _run(model, batches, g._CPUBackend())
    _restore(model, initial)
    losses_dev, params_dev = _run(model, batches, _DriftedBackend())

    loss_drift = _max_rel_drift(losses_cpu, losses_dev)
    param_drift = _max_param_drift(params_cpu, params_dev)
    assert loss_drift >= _LOSS_TOL or param_drift >= _PARAM_TOL, (
        "a backend with a 1% matmul error passed the parity gate "
        f"(loss drift {loss_drift:.2e} < {_LOSS_TOL}, "
        f"param drift {param_drift:.2e} < {_PARAM_TOL})"
    )
