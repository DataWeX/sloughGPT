"""DpoStrategy — preference-pair DPO as a loss strategy on the ONE loop.

Card: 20261114_151505_unify-loss-modes-on-sloughgpttrainer-sft-dpo-distill-as-loss.

The strategy declares the DPO loss (Rafailov et al. 2023) at the existing
``StepStrategy`` seam — no second training loop, no engine changes. The
reference math here is written independently of the implementation so the
formula and the custom backward are both pinned from the outside.
"""

from __future__ import annotations

import numpy as np
import pytest

from domain.training._internal.meta_trainer import MetaTrainer
from domain.training._internal.slonet import Tensor
from domain.training._internal.step_strategy import DpoStrategy, StepStrategy
from domain.training._internal.train_pipeline import SloughGPTTrainer, TrainerConfig

DATA_TEXT = (
    "the quick brown fox jumps over the lazy dog and runs across the meadow "
    "again and again while the dog sleeps soundly in the warm sun all day "
    "long. " * 8
)


# ── Fixtures / helpers (pattern mirrors test_meta_trainer.py; no cross-imports) ──


@pytest.fixture
def data_path(tmp_path):
    p = tmp_path / "corpus.txt"
    p.write_text(DATA_TEXT, encoding="utf-8")
    return str(p)


def tiny_config(tmp_path, **overrides):
    cfg = TrainerConfig(
        vocab_size=0,
        n_embed=16,
        n_layer=1,
        n_head=2,
        block_size=8,
        dropout=0.0,
        batch_size=2,
        epochs=1,
        max_steps=None,
        gradient_accumulation_steps=1,
        checkpoint_dir=str(tmp_path / "ckpts"),
        log_interval=1,
        eval_interval=1000,
        checkpoint_interval=1000,
        warmup_steps=1,
        min_lr=1e-5,
        max_checkpoints=5,
        scheduler_type="cosine",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def make_trainer(data_path, cfg, **kwargs):
    return SloughGPTTrainer(data_path, config=cfg, **kwargs)


class _FixedLogitsModel:
    """Model stub: each call returns the next preset output (cycling).

    One ``compute`` calls the policy twice (chosen, rejected) and the
    reference twice — cycling keeps repeated evaluations reproducible.
    """

    def __init__(self, *outputs):
        self._outputs = list(outputs)
        self.calls = 0

    def __call__(self, x, y=None):
        out = self._outputs[self.calls % len(self._outputs)]
        self.calls += 1
        if isinstance(out, tuple):
            return out
        return out, None


_ZERO = Tensor(np.array(0.0))
_XY = np.zeros((2, 4), dtype=np.int64)


def _seq_logprobs(logits: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Reference: per-example sum of selected-token log-probs, [B,T,V],[B,T] → [B]."""
    mx = logits.max(axis=-1, keepdims=True)
    lp = logits - mx - np.log(np.exp(logits - mx).sum(axis=-1, keepdims=True))
    b, t = np.indices(ids.shape)
    return lp[b, t, ids].sum(axis=1)


def _ref_dpo(logits_ch, y_ch, logits_rj, y_rj, ref_ch, ref_rj, beta):
    """Reference: -E[log σ(β ((π_w − π_l) − (ref_w − ref_l)))] — mean over pairs."""
    pi_ch = _seq_logprobs(logits_ch, y_ch)
    pi_rj = _seq_logprobs(logits_rj, y_rj)
    margin = (pi_ch - pi_rj) - (ref_ch - ref_rj)
    return float(np.logaddexp(0.0, -beta * margin).mean()), float(margin.mean())


def _pair_setup(seed=7, b=2, seq_len=5, vocab=7, beta=0.37):
    """Fixed logits + ids for policy (Tensors) and reference (plain arrays).

    Logits are stored as float32 — the dtype ``Tensor`` casts to — so the
    strategy's forward and the reference math see bit-identical inputs.
    """
    rng = np.random.default_rng(seed)
    x_len = seq_len - 1
    chosen = rng.integers(0, vocab, size=(b, seq_len))
    rejected = rng.integers(0, vocab, size=(b, seq_len))
    setup = {
        "policy_ch": rng.normal(size=(b, x_len, vocab)).astype(np.float32),
        "policy_rj": rng.normal(size=(b, x_len, vocab)).astype(np.float32),
        "ref_ch": rng.normal(size=(b, x_len, vocab)).astype(np.float32),
        "ref_rj": rng.normal(size=(b, x_len, vocab)).astype(np.float32),
        "chosen": chosen,
        "rejected": rejected,
        "y_ch": chosen[:, 1:],
        "y_rj": rejected[:, 1:],
        "beta": beta,
    }
    return setup


def _policy_tensors(s):
    return (
        Tensor(s["policy_ch"].copy(), requires_grad=True),
        Tensor(s["policy_rj"].copy(), requires_grad=True),
    )


# ── Formula + metrics ────────────────────────────────────────────────────────


class TestDpoFormula:
    def test_loss_matches_reference_formula(self):
        s = _pair_setup()
        strat = DpoStrategy(
            _FixedLogitsModel(s["ref_ch"], s["ref_rj"]),
            lambda: (s["chosen"], s["rejected"]),
            beta=s["beta"],
        )
        policy = _FixedLogitsModel(*_policy_tensors(s))
        out, metrics = strat.compute(policy, _XY, _XY, _ZERO)

        want_loss, want_margin = _ref_dpo(
            s["policy_ch"],
            s["y_ch"],
            s["policy_rj"],
            s["y_rj"],
            _seq_logprobs(s["ref_ch"], s["y_ch"]),
            _seq_logprobs(s["ref_rj"], s["y_rj"]),
            s["beta"],
        )
        # loss metrics pass through Tensor's float32 storage → float32 tolerance
        assert float(out.item()) == pytest.approx(want_loss, rel=1e-6)
        assert set(metrics) == {"dpo_loss", "dpo_margin"}
        assert metrics["dpo_loss"] == pytest.approx(want_loss, rel=1e-6)
        # margin never enters a Tensor — op-identical to the reference
        assert metrics["dpo_margin"] == pytest.approx(want_margin, rel=1e-9)

    def test_zero_margin_with_reference_equal_to_policy(self):
        """π_θ = π_ref ⇒ margin 0 ⇒ loss = ln 2 — closed-form invariant."""
        s = _pair_setup()
        strat = DpoStrategy(
            _FixedLogitsModel(s["policy_ch"], s["policy_rj"]),
            lambda: (s["chosen"], s["rejected"]),
            beta=s["beta"],
        )
        policy = _FixedLogitsModel(*_policy_tensors(s))
        out, metrics = strat.compute(policy, _XY, _XY, _ZERO)
        assert metrics["dpo_margin"] == pytest.approx(0.0, abs=1e-9)
        assert metrics["dpo_loss"] == pytest.approx(np.log(2.0), rel=1e-6)


# ── Custom backward ─────────────────────────────────────────────────────────


class TestDpoGradients:
    def test_backward_matches_finite_differences(self):
        s = _pair_setup()
        ref_ch_lp = _seq_logprobs(s["ref_ch"], s["y_ch"])
        ref_rj_lp = _seq_logprobs(s["ref_rj"], s["y_rj"])

        def loss_value(policy_ch, policy_rj):
            # FD runs on the float64 reference: the strategy's loss Tensor is
            # stored as float32, whose rounding (~6e-8) would drown central
            # differences at eps=1e-6. The formula test pins forward parity.
            return _ref_dpo(
                policy_ch,
                s["y_ch"],
                policy_rj,
                s["y_rj"],
                ref_ch_lp,
                ref_rj_lp,
                s["beta"],
            )[0]

        # analytic gradients through the graph
        pc, pr = _policy_tensors(s)
        strat = DpoStrategy(
            _FixedLogitsModel(s["ref_ch"], s["ref_rj"]),
            lambda: (s["chosen"], s["rejected"]),
            beta=s["beta"],
        )
        out, _ = strat.compute(_FixedLogitsModel(pc, pr), _XY, _XY, _ZERO)
        out.backward()
        assert pc.grad is not None and pr.grad is not None

        # central differences on every logit of both sides (float64 inputs)
        eps = 1e-6
        for key, analytic in (("policy_ch", pc.grad.data), ("policy_rj", pr.grad.data)):
            base = s[key].astype(np.float64)
            numeric = np.zeros_like(base)
            for idx in np.ndindex(*base.shape):
                hi, lo = base.copy(), base.copy()
                hi[idx] += eps
                lo[idx] -= eps
                if key == "policy_ch":
                    d_hi = loss_value(hi, s["policy_rj"])
                    d_lo = loss_value(lo, s["policy_rj"])
                else:
                    d_hi = loss_value(s["policy_ch"], hi)
                    d_lo = loss_value(s["policy_ch"], lo)
                numeric[idx] = (d_hi - d_lo) / (2 * eps)
            np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-6)

    def test_reference_model_receives_no_gradients(self, tmp_path, data_path):
        """The ref policy is baked into the loss as constants — never in the graph."""
        trainer = make_trainer(data_path, tiny_config(tmp_path))
        vocab = int(np.max(trainer.train_data)) + 1
        x, y = trainer.get_batch("train")
        seqs = np.stack([np.concatenate([x[i], y[i][-1:]]) for i in range(x.shape[0])])
        rng = np.random.default_rng(11)
        logits = rng.normal(size=(seqs.shape[0], seqs.shape[1] - 1, vocab))
        policy = _FixedLogitsModel(Tensor(logits, requires_grad=True))

        strat = DpoStrategy(trainer.model, lambda: (seqs, seqs[::-1].copy()), beta=0.1)
        out, metrics = strat.compute(policy, _XY, _XY, _ZERO)
        out.backward()

        assert np.isfinite(metrics["dpo_loss"])
        assert all(p.grad is None for p in trainer.model.parameters())


# ── StepStrategy contract ───────────────────────────────────────────────────


class TestDpoStrategyContract:
    def test_is_structural_step_strategy(self):
        strat = DpoStrategy(_FixedLogitsModel(_XY), lambda: None)
        assert isinstance(strat, StepStrategy)
        assert strat.name == "dpo"

    def test_requires_reference_model(self):
        with pytest.raises(ValueError, match="reference"):
            DpoStrategy(None, lambda: None)

    def test_requires_callable_pairs(self):
        with pytest.raises(ValueError, match="callable"):
            DpoStrategy(_FixedLogitsModel(_XY), "not-callable")

    def test_beta_must_be_positive(self):
        with pytest.raises(ValueError, match="beta"):
            DpoStrategy(_FixedLogitsModel(_XY), lambda: None, beta=0.0)

    def test_exhausted_pairs_returns_loss_unchanged(self):
        s = _pair_setup()
        strat = DpoStrategy(_FixedLogitsModel(s["ref_ch"]), lambda: None, beta=s["beta"])
        loss_in = Tensor(np.array(3.0))
        out, metrics = strat.compute(_FixedLogitsModel(s["policy_ch"]), _XY, _XY, loss_in)
        assert out is loss_in
        assert metrics == {}

    def test_pair_sequences_must_be_integer_arrays(self):
        s = _pair_setup()
        strat = DpoStrategy(
            _FixedLogitsModel(s["ref_ch"], s["ref_rj"]),
            lambda: (s["chosen"].astype(float), s["rejected"].astype(float)),
            beta=s["beta"],
        )
        with pytest.raises(ValueError, match="integer"):
            strat.compute(_FixedLogitsModel(*_policy_tensors(s)), _XY, _XY, _ZERO)

    def test_pair_shapes_must_match(self):
        s = _pair_setup()
        strat = DpoStrategy(
            _FixedLogitsModel(s["ref_ch"], s["ref_rj"]),
            lambda: (s["chosen"], s["rejected"][:, :-1]),
            beta=s["beta"],
        )
        with pytest.raises(ValueError, match="shape"):
            strat.compute(_FixedLogitsModel(*_policy_tensors(s)), _XY, _XY, _ZERO)

    def test_pair_sequences_must_be_2d(self):
        s = _pair_setup()
        strat = DpoStrategy(
            _FixedLogitsModel(s["ref_ch"], s["ref_rj"]),
            lambda: (s["chosen"][0], s["rejected"][0]),
            beta=s["beta"],
        )
        with pytest.raises(ValueError, match="2-D"):
            strat.compute(_FixedLogitsModel(*_policy_tensors(s)), _XY, _XY, _ZERO)

    def test_pair_sequences_need_at_least_two_tokens(self):
        s = _pair_setup()
        strat = DpoStrategy(
            _FixedLogitsModel(s["ref_ch"], s["ref_rj"]),
            lambda: (s["chosen"][:, :1], s["rejected"][:, :1]),
            beta=s["beta"],
        )
        with pytest.raises(ValueError, match="two tokens"):
            strat.compute(_FixedLogitsModel(*_policy_tensors(s)), _XY, _XY, _ZERO)


# ── Loss mode: replace (default) vs compose (sft_weight > 0) ────────────────


class TestDpoLossMode:
    def _want(self, s):
        return _ref_dpo(
            s["policy_ch"],
            s["y_ch"],
            s["policy_rj"],
            s["y_rj"],
            _seq_logprobs(s["ref_ch"], s["y_ch"]),
            _seq_logprobs(s["ref_rj"], s["y_rj"]),
            s["beta"],
        )[0]

    def test_default_declares_dpo_as_the_step_loss(self):
        """Pure DPO mode: the strategy REPLACES the step's training signal."""
        s = _pair_setup()
        strat = DpoStrategy(
            _FixedLogitsModel(s["ref_ch"], s["ref_rj"]),
            lambda: (s["chosen"], s["rejected"]),
            beta=s["beta"],
        )
        sft_loss = Tensor(np.array(1.5))
        out, _ = strat.compute(_FixedLogitsModel(*_policy_tensors(s)), _XY, _XY, sft_loss)
        assert out is not sft_loss
        assert float(out.item()) == pytest.approx(self._want(s), rel=1e-6)

    def test_sft_weight_composes_with_the_loop_loss(self):
        s = _pair_setup()
        strat = DpoStrategy(
            _FixedLogitsModel(s["ref_ch"], s["ref_rj"]),
            lambda: (s["chosen"], s["rejected"]),
            beta=s["beta"],
            sft_weight=0.5,
        )
        sft_loss = Tensor(np.array(1.5))
        out, metrics = strat.compute(_FixedLogitsModel(*_policy_tensors(s)), _XY, _XY, sft_loss)
        assert float(out.item()) == pytest.approx(0.5 * 1.5 + self._want(s), rel=1e-6)
        # metrics stay raw (the weight is composition, not the term)
        assert metrics["dpo_loss"] == pytest.approx(self._want(s), rel=1e-6)


# ── On the one loop ──────────────────────────────────────────────────────────


class TestDpoOnTheTrainerLoop:
    def test_train_step_trains_on_preference_pairs(self, tmp_path, data_path):
        trainer = make_trainer(data_path, tiny_config(tmp_path))

        def feed():
            x, y = trainer.get_batch("train")
            seqs = [np.concatenate([x[i], y[i][-1:]]) for i in range(x.shape[0])]
            return np.stack(seqs), np.stack(seqs[::-1])

        # policy as its own reference ⇒ margin 0 ⇒ DPO loss = ln 2 (exact)
        trainer.strategies = [DpoStrategy(trainer.model, feed, beta=0.1)]
        m1 = trainer.train_step()

        assert {"dpo_loss", "dpo_margin"} <= set(m1)
        assert m1["raw_loss"] == pytest.approx(np.log(2.0), rel=1e-3)
        assert m1["dpo_loss"] == pytest.approx(np.log(2.0), rel=1e-3)
        assert m1["raw_loss"] == pytest.approx(m1["dpo_loss"], rel=1e-9)
        assert m1["dpo_margin"] == pytest.approx(0.0, abs=1e-4)
        # gradients reached the model through the strategy's graph
        assert trainer._last_grad_norm is not None
        assert np.isfinite(trainer._last_grad_norm)

        m2 = trainer.train_step()
        assert np.isfinite(m2["raw_loss"])
        assert m2["dpo_margin"] == pytest.approx(0.0, abs=1e-4)

    def test_with_dpo_factory_composes_the_strategy(self, tmp_path, data_path):
        ref = make_trainer(data_path, tiny_config(tmp_path)).model
        mt = MetaTrainer.with_dpo(
            data_path, ref, lambda: None, beta=0.25, config=tiny_config(tmp_path)
        )

        assert len(mt.strategies) == 1
        strat = mt.strategies[0]
        assert isinstance(strat, DpoStrategy)
        assert isinstance(strat, StepStrategy)
        assert strat.beta == 0.25
        trainer = mt.build_trainer()
        assert trainer.strategies[0] is strat
