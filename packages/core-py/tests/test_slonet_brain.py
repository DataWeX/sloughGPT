"""Native SloNet world brain + brain registry (card 7ff49d20).

Additive only: the locked ``Perceptron``/``ModelBrain`` proofs live in
``test_model_brain.py`` and are not touched by this card. These tests cover
what that file cannot — the ``slonet``/``random`` kinds, the registry itself,
the ``decide_every`` staleness window, the 0.5 tick fallback, whole-brain
inheritance, the fail-loud genome guards, the ``--brain`` CLI wiring, and the
trained-vs-untrained mean energy gate (the only gate that proves learning
HELPS rather than merely moves a tensor).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from domain.shell._internal.evolution import Genome
from domain.shell._internal.model_brain import (
    BRAIN_KINDS,
    ModelBrain,
    RandomBrain,
    SloNetBrain,
    brain_from_dict,
    make_brain,
)
from domain.shell._internal.simulation import (
    Perceptron,
    SimBaby,
    SimScene,
    Simulation,
    WorldParams,
)
from domain.shell._internal.world_driver import main as world_main

#: Every registered kind other than the two flat-tensor ones.
_STACK_KINDS = {"slonet": SloNetBrain, "random": RandomBrain}


def _params(**kw) -> WorldParams:
    return WorldParams(grid_size=(12, 6, 12), generate_world=True, world_seed=3, **kw)


# ── Registry ────────────────────────────────────────────────────────────────


class TestBrainRegistry:
    """``make_brain``/``brain_from_dict`` are the one seam for every kind."""

    def test_registry_lists_every_kind(self):
        assert set(BRAIN_KINDS) == {"perceptron", "model", "slonet", "random"}

    @pytest.mark.parametrize(
        ("kind", "expected"),
        [
            ("perceptron", Perceptron),
            ("model", ModelBrain),
            ("slonet", SloNetBrain),
            ("random", RandomBrain),
        ],
    )
    def test_make_brain_dispatch(self, kind, expected):
        assert isinstance(make_brain(kind, 4, 3), expected)

    def test_make_brain_unknown_kind_fails_loud(self):
        with pytest.raises(ValueError, match="unknown brain kind 'transformer'"):
            make_brain("transformer", 4, 3)

    def test_stack_knobs_are_ignored_by_the_flat_kinds(self):
        """One WorldParams configures every kind — flat brains drop the knobs."""
        p = make_brain("perceptron", 4, 2, 4, n_embed=99, n_layer=7, decide_every=5)
        assert isinstance(p, Perceptron) and p.hidden_units == 4
        m = make_brain("model", 4, 2, 4, n_embed=99, learn_every=3, optimizer="adam")
        assert isinstance(m, ModelBrain) and m.hidden_units == 4

    def test_slonet_width_falls_back_to_hidden_units_then_16(self):
        assert SloNetBrain(5, 3, hidden_units=4).n_embed == 4
        assert SloNetBrain(5, 3, hidden_units=4, n_embed=7).n_embed == 7
        assert SloNetBrain(5, 3).n_embed == 16

    def test_default_kind_still_constructs_the_identical_perceptron(self):
        """The default path goes through the registry unchanged: same class,
        same tensors, same RNG stream as the inline constructor it replaced."""
        np.random.seed(21)
        via_registry = make_brain("perceptron", 5, 3, 8)
        np.random.seed(21)
        inline = Perceptron(5, 3, hidden_units=8)
        np.testing.assert_array_equal(via_registry.W, inline.W)
        np.testing.assert_array_equal(via_registry.b, inline.b)
        np.testing.assert_array_equal(via_registry.H, inline.H)
        np.testing.assert_array_equal(via_registry.bh, inline.bh)


# ── Fail-loud deserialization ───────────────────────────────────────────────


class TestBrainFromDict:
    def test_roundtrip_slonet(self):
        b = SloNetBrain(5, 3, n_embed=6, n_layer=2, decide_every=3, learn_every=4)
        restored = brain_from_dict(b.to_dict())
        assert isinstance(restored, SloNetBrain)
        assert (restored.n_embed, restored.n_layer) == (6, 2)
        assert (restored.decide_every, restored.learn_every) == (3, 4)
        x = np.random.rand(5).astype(np.float32)
        np.testing.assert_allclose(restored.forward(x), b.forward(x))

    def test_roundtrip_random(self):
        r = RandomBrain(4, 2)
        assert isinstance(brain_from_dict(r.to_dict()), RandomBrain)

    def test_legacy_payload_without_kind_stays_perceptron(self):
        p = Perceptron(4, 2, hidden_units=4)
        restored = brain_from_dict(p.to_dict())
        assert isinstance(restored, Perceptron)
        np.testing.assert_array_equal(restored.W, p.W)

    def test_unknown_kind_fails_loud_instead_of_falling_through(self):
        """A typo in a saved payload must never silently resurrect a Perceptron."""
        payload = {"kind": "slonnet", "W": [[0.0]], "b": [0.0]}
        with pytest.raises(ValueError, match="unknown brain kind 'slonnet'"):
            brain_from_dict(payload)

    def test_corrupt_slonet_payload_fails_loud(self):
        payload = SloNetBrain(5, 3, n_embed=6, n_layer=1).to_dict()
        payload["layers"][0]["weight"] = [[0.0, 0.0]]  # wrong shape
        with pytest.raises(ValueError, match="layer weight"):
            brain_from_dict(payload)

    def test_roundtrip_consumes_no_rng(self):
        payload = SloNetBrain(5, 3, n_embed=6, n_layer=1).to_dict()
        np.random.seed(9)
        SloNetBrain.from_dict(payload)
        after_rebuild = np.random.random(3)
        np.random.seed(9)
        np.testing.assert_array_equal(after_rebuild, np.random.random(3))


class TestRandomBrain:
    def test_forward_is_uniform_gates(self):
        out = RandomBrain(4, 3).forward(np.zeros(4, dtype=np.float32))
        assert out.shape == (3,)
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_update_is_a_no_op_and_never_raises(self):
        RandomBrain(4, 3).update(np.ones(4, dtype=np.float32), np.array([1.0, -1.0, 0.0]))

    def test_identity_only_serialization_draws_no_rng(self):
        np.random.seed(13)
        RandomBrain(4, 3).to_dict()
        RandomBrain.from_dict({"kind": "random", "input_dim": 4, "output_dim": 3})
        with_constructors = np.random.random(3)
        np.random.seed(13)
        np.testing.assert_array_equal(with_constructors, np.random.random(3))


# ── The decide window: staleness by design ──────────────────────────────────


class TestDecideWindow:
    def test_decide_every_reuses_the_last_action(self):
        b = SloNetBrain(5, 3, decide_every=4)
        first = b.forward(np.random.rand(5).astype(np.float32))
        reused = [b.forward(np.random.rand(5).astype(np.float32)) for _ in range(3)]
        assert all(out is first for out in reused)

    def test_update_does_not_invalidate_the_cache(self):
        """learn() runs every tick; if update() invalidated the window the
        mitigation would never fire. The cache must survive a step untouched."""
        b = SloNetBrain(5, 3, decide_every=4)
        cached = b.forward(np.random.rand(5).astype(np.float32))
        calls, cache = b._calls, b._cached
        b.update(np.random.rand(5).astype(np.float32), np.array([1.0, 1.0, 1.0]))
        assert b._calls == calls and b._cached is cache
        assert b.forward(np.random.rand(5).astype(np.float32)) is cached

    def test_cache_refreshes_at_the_window_boundary(self):
        b = SloNetBrain(5, 3, decide_every=3)
        first = b.forward(np.random.rand(5).astype(np.float32))
        b.forward(np.random.rand(5).astype(np.float32))
        b.forward(np.random.rand(5).astype(np.float32))
        assert b.forward(np.random.rand(5).astype(np.float32)) is not first

    def test_window_of_one_always_recomputes(self):
        b = SloNetBrain(5, 3, decide_every=1)
        outs = [b.forward(np.random.rand(5).astype(np.float32)) for _ in range(3)]
        assert outs[1] is not outs[0]


# ── A dead brain must never kill the tick ───────────────────────────────────


class TestTickFallback:
    def test_model_failure_yields_neutral_gates(self):
        b = SloNetBrain(5, 3)
        out = b.forward(np.ones(9, dtype=np.float32))  # wrong width -> matmul raises
        np.testing.assert_array_equal(out, np.full(3, 0.5))

    def test_blown_up_stack_yields_neutral_gates(self):
        b = SloNetBrain(5, 3)
        b.proj.weight.data = np.full_like(b.proj.weight.data, np.nan)
        with np.errstate(invalid="ignore"):
            out = b.forward(np.random.rand(5).astype(np.float32))
        np.testing.assert_array_equal(out, np.full(3, 0.5))

    def test_fallback_is_not_cached(self):
        """The next call gets a fresh chance at real gates."""
        b = SloNetBrain(5, 3)
        b.head.weight.data = np.full_like(b.head.weight.data, np.inf)
        b.forward(np.random.rand(5).astype(np.float32))
        assert b._cached is None


# ── The brain itself ────────────────────────────────────────────────────────


class TestSloNetBrainCore:
    def test_forward_shape_and_range(self):
        out = SloNetBrain(7, 3).forward(np.random.rand(7).astype(np.float32))
        assert out.shape == (3,)
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_update_moves_the_whole_stack(self):
        b = SloNetBrain(5, 3, n_embed=6, n_layer=2)
        x = np.random.rand(5).astype(np.float32)
        before = json.dumps(b.to_dict(), sort_keys=True)
        for _ in range(5):
            b.update(x, np.array([1.0, 1.0, 1.0]), lr=0.1)
        assert json.dumps(b.to_dict(), sort_keys=True) != before

    def test_positive_error_reinforces_and_negative_weakens(self):
        x = np.ones(4, dtype=np.float32)
        up, down = SloNetBrain(4, 1, n_layer=1), SloNetBrain(4, 1, n_layer=1)
        up_before, down_before = float(up.forward(x)[0]), float(down.forward(x)[0])
        for _ in range(30):
            up.update(x, np.array([1.0]), lr=0.1)
            down.update(x, np.array([-1.0]), lr=0.1)
        assert float(up.forward(x)[0]) > up_before
        assert float(down.forward(x)[0]) < down_before

    def test_batched_update_defers_until_the_buffer_is_full(self):
        b = SloNetBrain(5, 3, learn_every=4)
        x = np.random.rand(5).astype(np.float32)
        before = json.dumps(b.to_dict(), sort_keys=True)
        for _ in range(3):
            b.update(x, np.array([1.0, 1.0, 1.0]), lr=0.1)
        assert json.dumps(b.to_dict(), sort_keys=True) == before
        b.update(x, np.array([1.0, 1.0, 1.0]), lr=0.1)
        assert json.dumps(b.to_dict(), sort_keys=True) != before

    @pytest.mark.parametrize("optimizer", ["sgd", "adam"])
    def test_serialize_roundtrip_is_exact(self, optimizer):
        b = SloNetBrain(5, 3, n_embed=6, n_layer=2, optimizer=optimizer)
        payload = b.to_dict()
        json.dumps(payload)  # must be JSON-safe
        restored = SloNetBrain.from_dict(payload)
        assert restored.optimizer == optimizer
        for a, c in zip(b._params, restored._params, strict=True):
            np.testing.assert_array_equal(a.data, c.data)

    def test_adam_optimizer_is_constructible(self):
        b = SloNetBrain(5, 3, optimizer="adam")
        b.update(np.random.rand(5).astype(np.float32), np.array([1.0, 1.0, 1.0]))
        assert b.optimizer == "adam"

    def test_unknown_optimizer_fails_loud(self):
        with pytest.raises(ValueError, match="unknown brain optimizer"):
            SloNetBrain(5, 3, optimizer="rmsprop")


# ── World wiring ────────────────────────────────────────────────────────────


class TestWorldWiring:
    @pytest.mark.parametrize("kind", sorted(_STACK_KINDS))
    def test_baby_births_registered_brains(self, kind):
        expected = _STACK_KINDS[kind]
        b = SimBaby(params=_params(brain_kind=kind))
        for slot in ("cells", "move", "body", "entity"):
            assert isinstance(getattr(b, f"perceptron_{slot}"), expected)

    def test_default_babies_are_still_perceptrons(self):
        b = SimBaby(params=WorldParams())
        assert isinstance(b.perceptron_cells, Perceptron)

    @pytest.mark.parametrize("kind", sorted(_STACK_KINDS))
    def test_message_brain_is_routed_through_the_registry(self, kind):
        """The last hard-coded ``Perceptron(...)`` construction in SimBaby."""
        expected = _STACK_KINDS[kind]
        b = SimBaby(params=_params(brain_kind=kind, message_enabled=True))
        assert isinstance(b.perceptron_message, expected)

    def test_message_brain_default_kind_is_unchanged(self):
        b = SimBaby(params=_params(message_enabled=True))
        assert isinstance(b.perceptron_message, Perceptron)

    @pytest.mark.parametrize("kind", sorted(_STACK_KINDS))
    def test_baby_serialization_preserves_kind(self, kind):
        expected = _STACK_KINDS[kind]
        params = _params(brain_kind=kind)
        b = SimBaby(params=params)
        b2 = SimBaby.from_dict(b.to_dict(), params=params)
        assert isinstance(b2.perceptron_cells, expected)
        for name in ("cells", "body", "entity", "move"):
            assert getattr(b2, f"perceptron_{name}").to_dict() == getattr(
                b, f"perceptron_{name}"
            ).to_dict()

    @pytest.mark.parametrize("kind", sorted(_STACK_KINDS))
    def test_message_brain_roundtrips(self, kind):
        expected = _STACK_KINDS[kind]
        params = _params(brain_kind=kind, message_enabled=True)
        b = SimBaby(params=params)
        b2 = SimBaby.from_dict(b.to_dict(), params=params)
        assert isinstance(b2.perceptron_message, expected)

    @pytest.mark.parametrize("kind", ["perceptron", "model", "slonet", "random"])
    def test_ticks_run(self, kind):
        params = _params(brain_kind=kind, start_agents=4)
        scene = SimScene(params=params)
        for _ in range(4):
            scene.add_baby(SimBaby(params=params))
        Simulation(scene, max_ticks=5).run()  # must not raise

    def test_spawn_child_inherits_the_stack_whole(self):
        """No flat tensors to copy: the child gets the parent's exact native
        stack through the no-RNG registry deserializer (its own freshly drawn
        weights are replaced, so a mismatch here means the branch never ran)."""
        parent = SimBaby(params=_params(brain_kind="slonet"))
        child = parent.spawn_child(np.array([1, 1, 1], dtype=np.float64))
        assert isinstance(child.perceptron_cells, SloNetBrain)
        assert child.perceptron_cells.to_dict() == parent.perceptron_cells.to_dict()

    def test_spawn_child_default_brains_still_copy_flat_tensors(self):
        parent = SimBaby(params=_params())
        child = parent.spawn_child(np.array([1, 1, 1], dtype=np.float64))
        np.testing.assert_array_equal(parent.perceptron_cells.W, child.perceptron_cells.W)

    def test_teach_survives_a_brain_it_cannot_blend(self):
        """Weight blending needs a flat readout; a native stack keeps its
        weights and still receives the episode transfer."""
        params = _params(brain_kind="slonet", teaching_enabled=True, teach_memotype_cap=1)
        teacher = SimBaby(params=params, initial_energy=params.start_energy)
        student = SimBaby(params=params, initial_energy=params.start_energy)
        teacher.memory.record(np.ones(3, dtype=np.float32), (0.5,), 1.0, 0)
        before = student.perceptron_cells.to_dict()
        assert teacher.teach(student, amplitude=1.0) >= 1  # memotype still moves
        assert student.perceptron_cells.to_dict() == before  # weights untouched


# ── The conservative genome guard ───────────────────────────────────────────


class TestGenomeFailLoud:
    def test_from_baby_refuses_a_native_stack(self):
        baby = SimBaby(params=_params(brain_kind="slonet"))
        with pytest.raises(NotImplementedError, match="slonet"):
            Genome.from_baby(baby)

    def test_apply_to_refuses_a_native_stack(self):
        params = _params(brain_kind="slonet")
        genome = Genome.random(params, np.random.default_rng(5))
        with pytest.raises(NotImplementedError, match="slonet"):
            genome.apply_to(SimBaby(params=params))

    @pytest.mark.parametrize("kind", ["perceptron", "model"])
    def test_flat_brains_are_still_breedable(self, kind):
        """Control: the guard fires on the kind, not on the code path."""
        assert isinstance(Genome.from_baby(SimBaby(params=_params(brain_kind=kind))), Genome)


# ── world_driver --brain wiring ─────────────────────────────────────────────


class TestWorldDriverCLI:
    @pytest.mark.parametrize("kind", ["slonet", "random"])
    def test_main_runs_with_a_registered_brain(self, capsys, kind):
        code = world_main(
            ["--grid", "6,4,6", "--seed", "5", "--babies", "2", "--ticks", "4", "--brain", kind]
        )
        capsys.readouterr()
        assert code == 0

    def test_main_rejects_an_unknown_brain(self, capsys):
        """``main`` turns argparse's SystemExit into a 2 (bad-args) exit code."""
        assert world_main(["--grid", "6,4,6", "--brain", "transformer"]) == 2
        assert "invalid choice" in capsys.readouterr().err

    def test_hidden_units_and_slonet_knobs_reach_world_params(self, monkeypatch, capsys):
        """Regression: ``--brain`` used to be parsed and then dropped on the
        floor, so ``brain_hidden_units`` stayed at its default and ``--brain
        model --hidden-units 8`` silently built a plain Perceptron world."""
        import domain.shell._internal.world_driver as wd

        captured: dict = {}
        real_init = wd.WorldDriver.__init__

        def spy(self, *args, **kwargs):
            captured["params"] = args[0] if args else kwargs.get("params")
            real_init(self, *args, **kwargs)

        monkeypatch.setattr(wd.WorldDriver, "__init__", spy)
        code = wd.main(
            [
                "--grid",
                "6,4,6",
                "--ticks",
                "2",
                "--babies",
                "2",
                "--brain",
                "slonet",
                "--hidden-units",
                "4",
                "--brain-n-embed",
                "7",
                "--brain-n-layer",
                "2",
                "--brain-decide-every",
                "3",
                "--brain-learn-every",
                "4",
                "--brain-optimizer",
                "adam",
            ]
        )
        capsys.readouterr()
        assert code == 0
        p = captured["params"]
        assert p.brain_kind == "slonet"
        assert p.brain_hidden_units == 4
        assert (p.brain_n_embed, p.brain_n_layer) == (7, 2)
        assert (p.brain_decide_every, p.brain_learn_every) == (3, 4)
        assert p.brain_optimizer == "adam"


# ── The gate that proves learning HELPS (card item 3) ───────────────────────


def _mean_energy(brain_kind: str, learning_enabled: bool, seed: int, ticks: int = 40) -> float:
    """Run one identical-seed world and return mean baby energy over the run.

    Seeding the global stream before construction gives the trained and
    untrained arms byte-identical worlds and byte-identical initial weights,
    so ``learning_enabled`` is the ONLY difference between them.
    """
    np.random.seed(seed)
    params = WorldParams(
        grid_size=(16, 8, 16),
        generate_world=True,
        world_seed=seed,
        start_agents=12,
        brain_kind=brain_kind,
        learning_enabled=learning_enabled,
    )
    scene = SimScene(params=params)
    for _ in range(12):
        scene.add_baby(SimBaby(params=params, initial_energy=params.start_energy))
    sim = Simulation(scene, max_ticks=ticks)
    sim.run()
    return float(np.mean([r["energy"] for r in sim.tick_log]))


class TestLearningPays:
    """Landed gates proved plumbing (a tensor moves). This one proves value."""

    def test_trained_slonet_beats_untrained_slonet_on_mean_energy(self):
        deltas = [
            _mean_energy("slonet", True, seed) - _mean_energy("slonet", False, seed)
            for seed in (1, 2, 3)
        ]
        assert float(np.mean(deltas)) > 0.0, (
            f"trained slonet brains did not out-earn untrained ones: "
            f"per-seed deltas {[f'{d:+.3f}' for d in deltas]}"
        )
