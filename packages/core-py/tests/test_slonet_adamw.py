"""Tests for SloAdamW: decoupled weight decay semantics, Adam moment math,
gradient clipping, state serialization, and checkpoint interchangeability
with SloAdam."""

import numpy as np
import pytest

from domains.training.slonet import SloAdam, SloAdamW, Tensor, mse_loss


class _NamedTensor(Tensor):
    def __init__(self, data, name=None, **kwargs):
        super().__init__(data, **kwargs)
        self.name = name


class TestSloAdamW:
    def test_subclasses_slo_adam(self):
        assert issubclass(SloAdamW, SloAdam)

    def test_inherits_shared_helpers(self):
        assert callable(SloAdamW._reduce_to_param_shape)
        assert callable(SloAdamW._adam_update)

    def test_default_weight_decay_is_0_01(self):
        assert SloAdamW().weight_decay == pytest.approx(0.01)

    def test_step_lowers_loss_direction(self):
        w = Tensor(np.array([1.0]), requires_grad=True)
        adamw = SloAdamW(lr=0.1, weight_decay=0.0)
        w.grad = Tensor(np.array([1.0]))
        adamw.step([w])
        assert w.data[0] == pytest.approx(0.9)

    def test_decoupled_decay_shrinks_weights_with_zero_grad(self):
        w = Tensor(np.array([1.0]), requires_grad=True)
        adamw = SloAdamW(lr=0.1, weight_decay=0.5)
        w.grad = Tensor(np.array([0.0]))
        adamw.step([w])
        assert w.data[0] == pytest.approx(1.0 - 0.1 * 0.5 * 1.0)

    def test_decay_annealed_by_learning_rate(self):
        w = Tensor(np.array([1.0]), requires_grad=True)
        adamw = SloAdamW(lr=0.0, weight_decay=0.5)
        w.grad = Tensor(np.array([1.0]))
        adamw.step([w])
        assert w.data[0] == pytest.approx(1.0)

    def test_decoupled_differs_from_l2_adam(self):
        params = {
            "adamw": Tensor(np.array([1.0]), requires_grad=True),
            "adam": Tensor(np.array([1.0]), requires_grad=True),
        }
        adamw = SloAdamW(lr=0.1, weight_decay=0.5)
        adam = SloAdam(lr=0.1, weight_decay=0.5)
        for p in params.values():
            p.grad = Tensor(np.array([0.0]))
        adamw.step([params["adamw"]])
        adam.step([params["adam"]])
        assert params["adamw"].data[0] == pytest.approx(0.95)
        assert params["adam"].data[0] != pytest.approx(0.95)

    def test_max_grad_norm_clips(self):
        w = Tensor(np.array([0.0]), requires_grad=True)
        adamw = SloAdamW(lr=0.1, weight_decay=0.0, max_grad_norm=1.0)
        w.grad = Tensor(np.array([100.0]))
        adamw.step([w])
        assert w.data[0] == pytest.approx(-0.1)

    def test_second_step_uses_bias_correction(self):
        w = Tensor(np.array([1.0]), requires_grad=True)
        adamw = SloAdamW(lr=0.1, weight_decay=0.0)
        for _ in range(2):
            w.grad = Tensor(np.array([1.0]))
            adamw.step([w])
        assert w.data[0] == pytest.approx(0.8, abs=1e-4)

    def test_skips_params_without_grad(self):
        w = Tensor(np.array([1.0]), requires_grad=True)
        w_no_grad = Tensor(np.array([1.0]))
        adamw = SloAdamW(lr=0.1, weight_decay=0.0)
        adamw.step([w, w_no_grad])
        assert w_no_grad.data[0] == pytest.approx(1.0)

    def test_state_dict_round_trip(self):
        w = _NamedTensor(np.array([1.0]), name="w", requires_grad=True)
        adamw = SloAdamW(lr=0.01, weight_decay=0.05)
        w.grad = Tensor(np.array([1.0]))
        adamw.step([w])
        state = adamw.state_dict([w])
        assert state["t"] == 1
        assert set(state["state"]["w"].keys()) == {"m", "v"}
        restored = SloAdamW()
        restored.load_state_dict(state, [w])
        assert restored._t == 1
        assert restored.weight_decay == pytest.approx(0.05)
        assert np.allclose(restored._m[id(w)], state["state"]["w"]["m"])

    def test_state_interchangeable_with_slo_adam(self):
        w = _NamedTensor(np.array([1.0]), name="w", requires_grad=True)
        adamw = SloAdamW(lr=0.01, weight_decay=0.05)
        w.grad = Tensor(np.array([1.0]))
        adamw.step([w])
        state = adamw.state_dict([w])
        loaded_adam = SloAdam()
        loaded_adam.load_state_dict(state, [w])
        assert loaded_adam.weight_decay == pytest.approx(0.05)
        assert np.allclose(loaded_adam._m[id(w)], state["state"]["w"]["m"])

    def test_matches_reference_amsgrad(self):
        rng = np.random.default_rng(7)
        p0 = rng.normal(size=(4,))
        gs = [rng.normal(size=(4,)) for _ in range(3)]
        lr, b1, b2, eps, wd = 0.01, 0.9, 0.999, 1e-8, 0.1

        opt = SloAdamW(lr=lr, b1=b1, b2=b2, eps=eps, weight_decay=wd, amsgrad=True)
        p = Tensor(p0.copy(), requires_grad=True)
        for g in gs:
            p.grad = Tensor(g.copy())
            opt.step([p])

        m = np.zeros(4)
        v = np.zeros(4)
        vmax = np.zeros(4)
        p_ref = p0.copy()
        for t, g in enumerate(gs, start=1):
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g ** 2
            mh = m / (1 - b1 ** t)
            vmax = np.maximum(vmax, v)
            vh = vmax / (1 - b2 ** t)
            p_ref -= lr * mh / (np.sqrt(vh) + eps)
            p_ref -= lr * wd * p_ref
        assert np.allclose(p.data, p_ref, atol=1e-12)

    def test_amsgrad_pins_denominator_on_shrinking_gradients(self):
        amsg = SloAdamW(lr=0.1, b2=0.9, weight_decay=0.0, amsgrad=True)
        plain = SloAdamW(lr=0.1, b2=0.9, weight_decay=0.0)
        pa = Tensor(np.array([0.0]), requires_grad=True)
        pb = Tensor(np.array([0.0]), requires_grad=True)
        for _ in range(20):
            pa.grad = Tensor(np.array([1.0]))
            pb.grad = Tensor(np.array([1.0]))
            amsg.step([pa]); plain.step([pb])
        for _ in range(20):
            pa.grad = Tensor(np.array([0.1]))
            pb.grad = Tensor(np.array([0.1]))
            amsg.step([pa]); plain.step([pb])
        # amsgrad holds the historical max of the second moment, so small
        # second-phase gradients cannot shrink the denominator; the parameter
        # therefore moves less and ends up closer to zero than plain AdamW.
        assert pa.data[0] > pb.data[0]

    def test_maximize_ascends_objective(self):
        w = Tensor(np.array([0.0]), requires_grad=True)
        opt = SloAdamW(lr=0.1, weight_decay=0.0, maximize=True)
        w.grad = Tensor(np.array([1.0]))
        opt.step([w])
        assert w.data[0] == pytest.approx(0.1)

    def test_maximize_with_decay(self):
        w = Tensor(np.array([1.0]), requires_grad=True)
        opt = SloAdamW(lr=0.1, weight_decay=0.5, maximize=True)
        w.grad = Tensor(np.array([1.0]))
        opt.step([w])
        # gradient inverted → ascent of +0.1, then decoupled decay by 0.1*0.5
        assert w.data[0] == pytest.approx(1.1 - 0.1 * 0.5 * 1.1)

    def test_amsgrad_state_round_trip(self):
        w = _NamedTensor(np.array([1.0]), name="w", requires_grad=True)
        opt = SloAdamW(lr=0.01, weight_decay=0.05, amsgrad=True)
        w.grad = Tensor(np.array([1.0]))
        opt.step([w])
        state = opt.state_dict([w])
        assert set(state["state"]["w"].keys()) == {"m", "v", "maxv"}
        assert state["hyperparameters"]["amsgrad"] is True
        restored = SloAdamW()
        restored.load_state_dict(state, [w])
        assert restored.amsgrad is True
        assert np.allclose(restored._vmax[id(w)], state["state"]["w"]["maxv"])

    def test_converges_on_linear_regression(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(64, 1))
        y = 2.0 * x + 1.0
        w = Tensor(rng.normal(size=(1, 1)), requires_grad=True)
        b = Tensor(np.zeros((1,)), requires_grad=True)
        adamw = SloAdamW(lr=0.05, weight_decay=0.0)
        y_t = Tensor(y)

        def loss_value():
            pred = w * Tensor(x) + b
            return float(np.mean((pred.data - y) ** 2))

        first = loss_value()
        for _ in range(200):
            mse_loss(w * Tensor(x) + b, y_t).backward()
            adamw.step([w, b])
        last = loss_value()
        assert last < first
        assert last < 1e-2

    def test_matches_reference_adamw(self):
        rng = np.random.default_rng(1)
        p0 = rng.normal(size=(4,))
        g0 = rng.normal(size=(4,))
        g1 = rng.normal(size=(4,))
        lr, b1, b2, eps, wd = 0.01, 0.9, 0.999, 1e-8, 0.1

        opt = SloAdamW(lr=lr, b1=b1, b2=b2, eps=eps, weight_decay=wd)
        p = Tensor(p0.copy(), requires_grad=True)
        for g in (g0, g1):
            p.grad = Tensor(g.copy())
            opt.step([p])

        m = np.zeros(4)
        v = np.zeros(4)
        p_ref = p0.copy()
        for t, g in enumerate((g0, g1), start=1):
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g ** 2
            mh = m / (1 - b1 ** t)
            vh = v / (1 - b2 ** t)
            p_ref -= lr * mh / (np.sqrt(vh) + eps)
            p_ref -= lr * wd * p_ref
        assert np.allclose(p.data, p_ref, atol=1e-12)

    def test_broadcast_grad_reduces_before_moments(self):
        rng = np.random.default_rng(2)
        p_shape = (3,)
        p0 = rng.normal(size=p_shape)
        batched = rng.normal(size=(2, 3))
        opt = SloAdamW(lr=0.01, weight_decay=0.0)
        p = Tensor(p0.copy(), requires_grad=True)
        p.grad = Tensor(batched.copy())
        opt.step([p])

        expected = SloAdamW(lr=0.01, weight_decay=0.0)
        q = Tensor(p0.copy(), requires_grad=True)
        q.grad = Tensor(batched.sum(axis=0))
        expected.step([q])
        assert p.data.shape == p_shape
        assert np.allclose(p.data, q.data, atol=1e-12)

    def test_broadcast_grad_with_multiple_leading_axes(self):
        rng = np.random.default_rng(3)
        p_shape = (2, 3)
        grad = rng.normal(size=(4, 2, 3))
        opt = SloAdamW(lr=0.01, weight_decay=0.05)
        p = Tensor(rng.normal(size=p_shape), requires_grad=True)
        p.grad = Tensor(grad.copy())
        opt.step([p])
        assert p.data.shape == p_shape
        assert np.all(np.isfinite(p.data))
