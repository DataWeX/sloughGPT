"""Tests for domains/training/ewc.py — EWC continual learning."""

import numpy as np
import pytest

from domains.training.ewc import (
    _as_array,
    _scalar,
    _batch_size,
    _unpack_batch,
    EWCParameters,
    TaskSnapshot,
    DiagonalFisherEstimator,
    EwcContinualLearner,
)
from domains.training.slonet import (
    Tensor,
    tensor,
    SloLinear,
    cross_entropy,
)


class _LinearModel:
    """Real SloNet-backed model exposing the EWC-required API."""

    def __init__(self, in_f=4, out_f=3):
        self.lin = SloLinear(in_f, out_f)
        self._params = [self.lin.weight, self.lin.bias]

    def __call__(self, x):
        return self.lin.forward(x)

    def named_parameters(self):
        return [("weight", self.lin.weight), ("bias", self.lin.bias)]

    def parameters(self):
        return self._params

    def eval(self):
        return self

    def train(self, mode=True):
        return self

    def state_dict(self):
        return {
            "weight": self.lin.weight.data.copy(),
            "bias": self.lin.bias.data.copy(),
        }


def _make_batches(n=4, batch_size=4, in_f=4, out_f=3):
    rng = np.random.RandomState(0)
    batches = []
    for _ in range(n):
        x = rng.randn(batch_size, in_f).astype(np.float32)
        y = rng.randint(0, out_f, size=batch_size).astype(np.int64)
        batches.append((tensor(x), tensor(y)))
    return batches


def _ce_loss(outputs, targets):
    return cross_entropy(outputs, targets)


# ---- private helpers ---------------------------------------------------


def test_as_array_tensor_extracts_data():
    t = tensor([[1.0, 2.0]])
    arr = _as_array(t)
    assert isinstance(arr, np.ndarray)
    np.testing.assert_array_equal(arr, np.array([[1.0, 2.0]]))


def test_as_array_passes_ndarray_through():
    a = np.zeros((2, 2))
    assert _as_array(a) is a


def test_as_array_coerces_python_scalar():
    assert _as_array(3.0).ndim == 0
    assert float(_as_array(3.0)) == 3.0


def test_scalar_flat_float():
    assert _scalar(tensor([[2.5]])) == pytest.approx(2.5)
    assert _scalar(np.array([7.0])) == pytest.approx(7.0)
    assert _scalar(4.0) == pytest.approx(4.0)


def test_batch_size():
    assert _batch_size(np.zeros((8, 4))) == 8
    assert _batch_size(np.zeros((1, 4))) == 1
    assert _batch_size(np.zeros(4)) == 4
    assert _batch_size(np.float32(1.0)) == 1


def test_unpack_batch_list():
    x, y = _unpack_batch([np.zeros((2, 4)), np.ones(2)])
    assert x.shape == (2, 4)
    np.testing.assert_array_equal(y, np.ones(2))


def test_unpack_batch_tuple_no_targets():
    x, y = _unpack_batch((np.zeros((2, 4)),))
    assert y is None


def test_unpack_batch_raw_inputs():
    x, y = _unpack_batch(np.zeros((2, 4)))
    assert y is None
    assert x.shape == (2, 4)


def test_ewc_parameters_defaults():
    p = EWCParameters()
    assert p.lambda_ewc == 1000.0
    assert p.diagonal_approx is True
    assert p.batch_size == 32
    assert p.num_samples == 100
    assert p.clip_grad_norm == 10.0
    assert p.ema_decay == 0.9


def test_task_snapshot_fields():
    s = TaskSnapshot(
        task_id="t1", task_name="first",
        parameters={"w": np.ones(2)}, fisher_diagonal={"w": np.zeros(2)},
        optimal_loss=0.5, num_samples=10,
    )
    assert s.task_id == "t1"
    assert s.task_name == "first"
    assert s.optimal_loss == 0.5
    assert s.num_samples == 10


# ---- DiagonalFisherEstimator -------------------------------------------


def test_fisher_init_fills_accumulator():
    est = DiagonalFisherEstimator(_LinearModel())
    assert set(est.fisher_accum.keys()) == {"weight", "bias"}
    np.testing.assert_array_equal(est.fisher_accum["weight"], np.zeros((3, 4)))
    np.testing.assert_array_equal(est.fisher_accum["bias"], np.zeros(3))
    assert est.num_observations == 0


def test_fisher_init_skips_requires_grad_false():
    m = _LinearModel()
    for p in m.parameters():
        p.requires_grad = False
    est = DiagonalFisherEstimator(m)
    assert est.fisher_accum == {}


def test_fisher_estimate_accumulates_and_normalizes():
    est = DiagonalFisherEstimator(_LinearModel(), ema_decay=0.5)
    batches = _make_batches(n=2, batch_size=2)
    fisher = est.estimate(batches, _ce_loss, num_samples=4, accumulation_steps=10)
    assert set(fisher.keys()) == {"weight", "bias"}
    assert est.num_observations == 4
    assert (fisher["weight"] > 0).any()
    assert (fisher["weight"] >= 1e-8).all()


def test_fisher_estimate_stops_at_num_samples():
    est = DiagonalFisherEstimator(_LinearModel())
    batches = _make_batches(n=4, batch_size=2)
    est.estimate(batches, _ce_loss, num_samples=3, accumulation_steps=100)
    assert est.num_observations == 4


def test_fisher_estimate_stops_at_accumulation_steps():
    est = DiagonalFisherEstimator(_LinearModel())
    batches = _make_batches(n=8, batch_size=2)
    est.estimate(batches, _ce_loss, num_samples=1000, accumulation_steps=3)
    assert est.num_observations == 6


def test_fisher_estimate_without_targets_uses_mean():
    class _UnsupModel(_LinearModel):
        pass

    est = DiagonalFisherEstimator(_UnsupModel())
    batches = [tensor(np.random.RandomState(1).randn(2, 4).astype(np.float32))]
    fisher = est.estimate(batches, None, num_samples=10, accumulation_steps=1)
    assert (fisher["weight"] > 0).any()


def test_fisher_estimate_from_logits():
    est = DiagonalFisherEstimator(_LinearModel())
    rng = np.random.RandomState(2)
    x = tensor(rng.randn(4, 4).astype(np.float32))
    y = rng.randint(0, 3, size=4).astype(np.int64)
    fisher = est.estimate_from_logits(x, y, num_samples=3)
    assert (fisher["weight"] > 0).any()
    assert (fisher["weight"] >= 1e-8).all()


# ---- EwcContinualLearner ------------------------------------------------


def test_init_sets_model_and_estimator():
    learner = EwcContinualLearner(_LinearModel())
    assert learner.current_task is None
    assert learner.task_snapshots == {}
    assert isinstance(learner.fisher_estimator, DiagonalFisherEstimator)
    assert learner.params.lambda_ewc == 1000.0


def test_init_with_custom_params():
    p = EWCParameters(lambda_ewc=5.0, ema_decay=0.3)
    learner = EwcContinualLearner(_LinearModel(), params=p)
    assert learner.params.lambda_ewc == 5.0
    assert learner.params.ema_decay == 0.3
    assert learner.fisher_estimator.ema_decay == 0.3


def test_save_task_snapshot_stores_everything():
    learner = EwcContinualLearner(_LinearModel())
    snap = learner.save_task_snapshot("t1", "first task", _make_batches(n=2), _ce_loss)
    assert isinstance(snap, TaskSnapshot)
    assert snap.task_id == "t1"
    assert snap.task_name == "first task"
    assert set(snap.parameters.keys()) == {"weight", "bias"}
    assert set(snap.fisher_diagonal.keys()) == {"weight", "bias"}
    assert snap.optimal_loss > 0
    assert snap.num_samples == learner.params.num_samples
    assert learner.task_snapshots["t1"] is snap


def test_ewc_loss_no_snapshot():
    learner = EwcContinualLearner(_LinearModel())
    loss, stats = learner.ewc_loss("missing")
    assert _scalar(loss) == pytest.approx(0.0)
    assert stats == {"active_tasks": 0}


def test_ewc_loss_uses_current_task():
    learner = EwcContinualLearner(_LinearModel())
    learner.save_task_snapshot("t1", "t", _make_batches(n=1), _ce_loss)
    learner.current_task = "t1"
    loss, stats = learner.ewc_loss()
    assert stats["active_tasks"] == 1
    assert stats["param_count"] == 2
    assert stats["lambda"] == learner.params.lambda_ewc
    assert loss is not None


def test_ewc_loss_zero_when_params_match_snapshot():
    learner = EwcContinualLearner(_LinearModel())
    snap = learner.save_task_snapshot("t1", "t", _make_batches(n=1), _ce_loss)
    learner.model.lin.weight.data = snap.parameters["weight"].copy()
    learner.model.lin.bias.data = snap.parameters["bias"].copy()
    _, stats = learner.ewc_loss("t1")
    assert stats["raw_ewc_loss"] == pytest.approx(0.0, abs=1e-6)


def test_ewc_loss_penalizes_changed_params():
    learner = EwcContinualLearner(_LinearModel())
    learner.save_task_snapshot("t1", "t", _make_batches(n=1), _ce_loss)
    learner.model.lin.weight.data = learner.model.lin.weight.data + 5.0
    _, stats = learner.ewc_loss("t1")
    assert stats["raw_ewc_loss"] > 0


def test_multi_task_ewc_loss_empty():
    learner = EwcContinualLearner(_LinearModel())
    loss, stats = learner.multi_task_ewc_loss()
    assert _scalar(loss) == pytest.approx(0.0)
    assert stats == {"active_tasks": 0}


def test_multi_task_ewc_loss_two_tasks():
    learner = EwcContinualLearner(_LinearModel(), params=EWCParameters(num_samples=5))
    learner.save_task_snapshot("t1", "a", _make_batches(n=1), _ce_loss)
    learner.save_task_snapshot("t2", "b", _make_batches(n=1), _ce_loss)
    loss, stats = learner.multi_task_ewc_loss()
    assert stats["active_tasks"] == 2
    assert stats["param_count"] == 4
    assert _scalar(loss) >= 0


def test_forward_and_ewc_combines_losses():
    learner = EwcContinualLearner(_LinearModel(), params=EWCParameters(lambda_ewc=10.0))
    learner.save_task_snapshot("t1", "t", _make_batches(n=1), _ce_loss)
    learner.current_task = "t1"
    batch = _make_batches(n=1)[0]
    total, meta = learner.forward_and_ewc(batch, _ce_loss, task_id="t1")
    assert meta["task_loss"] > 0
    assert meta["active_tasks"] == 1
    assert meta["ewc_loss"] >= 0
    assert meta["total_loss"] == pytest.approx(meta["task_loss"] + meta["ewc_loss"])


def test_forward_and_ewc_without_task_just_task_loss():
    learner = EwcContinualLearner(_LinearModel())
    batch = _make_batches(n=1)[0]
    total, meta = learner.forward_and_ewc(batch, _ce_loss)
    assert meta["active_tasks"] == 0
    assert meta["ewc_loss"] == pytest.approx(0.0)
    assert meta["total_loss"] == pytest.approx(meta["task_loss"])


def test_prune_consolidation_empty():
    learner = EwcContinualLearner(_LinearModel())
    assert learner.prune_consolidation() == {}


def test_prune_consolidation_returns_important_counts():
    learner = EwcContinualLearner(_LinearModel())
    learner.save_task_snapshot("t1", "t", _make_batches(n=1), _ce_loss)
    important = learner.prune_consolidation(top_k_percent=100.0)
    assert set(important.keys()) <= {"weight", "bias"}
    assert important["weight"] > 0


def test_estimate_forgetting_empty():
    learner = EwcContinualLearner(_LinearModel())
    assert learner.estimate_forgetting() == {}


def test_estimate_forgetting_zero_when_unchanged():
    learner = EwcContinualLearner(_LinearModel())
    snap = learner.save_task_snapshot("t1", "t", _make_batches(n=1), _ce_loss)
    learner.model.lin.weight.data = snap.parameters["weight"].copy()
    learner.model.lin.bias.data = snap.parameters["bias"].copy()
    forgetting = learner.estimate_forgetting()
    assert forgetting["t1"] == pytest.approx(0.0, abs=1e-6)


def test_estimate_forgetting_increases_when_params_change():
    learner = EwcContinualLearner(_LinearModel())
    learner.save_task_snapshot("t1", "t", _make_batches(n=1), _ce_loss)
    learner.model.lin.weight.data = learner.model.lin.weight.data + 5.0
    forgetting = learner.estimate_forgetting()
    assert forgetting["t1"] > 0
