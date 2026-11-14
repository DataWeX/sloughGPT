"""Tests for the World–Model Bridge seam — ModelBrain (card 7cefbbcf)."""

from __future__ import annotations

import numpy as np

from domain.shell._internal.evolution import Genome, benchmark_emergence
from domain.shell._internal.model_brain import ModelBrain
from domain.shell._internal.simulation import (
    Perceptron,
    SimBaby,
    SimScene,
    Simulation,
    WorldParams,
    brain_from_dict,
    make_brain,
)

# ── Interface parity ────────────────────────────────────────────────────────


class TestInterfaceParity:
    """ModelBrain must be a drop-in for Perceptron at every call site."""

    def test_forward_shape_and_range(self):
        m = ModelBrain(5, 3, hidden_units=8)
        out = m.forward(np.random.rand(5).astype(np.float32))
        assert out.shape == (3,)
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_forward_no_hidden_matches_perceptron_math(self):
        """hidden=0 brains of both classes are the same delta-rule unit."""
        np.random.seed(3)
        m = ModelBrain(4, 2)
        np.random.seed(3)
        p = Perceptron(4, 2)
        x = np.random.rand(4).astype(np.float32)
        np.testing.assert_allclose(m.forward(x), p.forward(x), rtol=1e-6)

    def test_update_moves_weights(self):
        m = ModelBrain(5, 3, hidden_units=8)
        x = np.random.rand(5).astype(np.float32)
        w0, b0 = m.W.copy(), m.b.copy()
        m.update(x, np.array([1.0, 1.0, 1.0]), lr=0.1)
        assert not np.allclose(m.W, w0)
        assert not np.allclose(m.b, b0)

    def test_serialize_roundtrip(self):
        m = ModelBrain(5, 3, hidden_units=8)
        data = m.to_dict()
        assert data["kind"] == "model"
        m2 = ModelBrain.from_dict(data)
        np.testing.assert_array_equal(m.W, m2.W)
        np.testing.assert_array_equal(m.b, m2.b)
        np.testing.assert_array_equal(m.H, m2.H)
        np.testing.assert_array_equal(m.bh, m2.bh)

    def test_json_safe(self):
        import json

        m = ModelBrain(4, 2, hidden_units=4)
        json.dumps(m.to_dict())  # must not raise

    def test_brain_from_dict_dispatch(self):
        """kind marker routes to ModelBrain; legacy payloads stay Perceptron."""
        m = ModelBrain(4, 2, hidden_units=4)
        assert isinstance(brain_from_dict(m.to_dict()), ModelBrain)

        p = Perceptron(4, 2, hidden_units=4)
        restored = brain_from_dict(p.to_dict())
        assert isinstance(restored, Perceptron)
        np.testing.assert_array_equal(p.W, restored.W)


# ── The capability step ─────────────────────────────────────────────────────


class TestTrainableHidden:
    """The whole point: ModelBrain's H learns; Perceptron's H is frozen."""

    def test_hidden_projection_trains(self):
        m = ModelBrain(5, 3, hidden_units=8)
        x = np.random.rand(5).astype(np.float32)
        h0, bh0 = m.H.copy(), m.bh.copy()
        for _ in range(20):
            m.update(x, np.array([1.0, 1.0, 1.0]), lr=0.1)
        assert not np.allclose(m.H, h0)
        assert not np.allclose(m.bh, bh0)

    def test_perceptron_hidden_frozen_by_construction(self):
        """Control: the baseline brain must NOT move its H."""
        p = Perceptron(5, 3, hidden_units=8)
        x = np.random.rand(5).astype(np.float32)
        h0 = p.H.copy()
        for _ in range(20):
            p.update(x, np.array([1.0, 1.0, 1.0]), lr=0.1)
        np.testing.assert_array_equal(p.H, h0)

    def test_negative_error_weakens(self):
        m = ModelBrain(4, 1, hidden_units=4)
        x = np.ones(4, dtype=np.float32)
        out_before = float(m.forward(x)[0])
        for _ in range(30):
            m.update(x, np.array([-1.0]), lr=0.1)
        out_after = float(m.forward(x)[0])
        assert out_after < out_before


# ── Factory + RNG lock ──────────────────────────────────────────────────────


class TestFactory:
    def test_make_brain_dispatch(self):
        assert isinstance(make_brain("model", 4, 2, 4), ModelBrain)
        assert isinstance(make_brain("perceptron", 4, 2, 4), Perceptron)

    def test_default_kind_is_perceptron(self):
        """The world's default brain class must not change (locked proofs)."""
        assert WorldParams().brain_kind == "perceptron"

    def test_rng_draw_order_parity(self):
        """Same seed → ModelBrain and Perceptron draw identical initial
        tensors: the swap changes who learns, never luck."""
        np.random.seed(11)
        m = ModelBrain(5, 3, hidden_units=8)
        np.random.seed(11)
        p = Perceptron(5, 3, hidden_units=8)
        np.testing.assert_array_equal(m.W, p.W)
        np.testing.assert_array_equal(m.b, p.b)
        np.testing.assert_array_equal(m.H, p.H)
        np.testing.assert_array_equal(m.bh, p.bh)


# ── World wiring ────────────────────────────────────────────────────────────


class TestWorldWiring:
    def test_baby_births_model_brains(self):
        params = WorldParams(brain_kind="model", brain_hidden_units=4)
        b = SimBaby(params=params)
        assert isinstance(b.perceptron_cells, ModelBrain)
        assert isinstance(b.perceptron_move, ModelBrain)
        assert isinstance(b.perceptron_body, ModelBrain)
        assert isinstance(b.perceptron_entity, ModelBrain)

    def test_baby_default_brains_unchanged(self):
        b = SimBaby(params=WorldParams())
        assert isinstance(b.perceptron_cells, Perceptron)

    def test_decide_interface_works(self):
        """The decide_* methods must run unmodified against a ModelBrain."""
        from domain.shell._internal.simulation import Perception

        params = WorldParams(brain_kind="model", brain_hidden_units=4)
        scene = SimScene(params=params)
        b = SimBaby(params=params)
        scene.add_baby(b)
        perception = Perception(
            nearby_cells={
                "count": 2,
                "material": [1.0, 2.0],
                "energy": [5.0, 5.0],
                "temperature": [20.0, 20.0],
                "signal": [0.1, 0.1],
            },
            nearby_entities=[],
        )
        action = b.react(perception, energy_delta=0.0)
        assert action is not None
        move = b.decide_move(perception)
        assert move is None or move.shape == (3,)

    def test_baby_serialization_preserves_kind(self):
        params = WorldParams(brain_kind="model", brain_hidden_units=4)
        b = SimBaby(params=params)
        data = b.to_dict()
        b2 = SimBaby.from_dict(data, params=params)
        assert isinstance(b2.perceptron_cells, ModelBrain)
        np.testing.assert_array_equal(b.perceptron_cells.W, b2.perceptron_cells.W)
        np.testing.assert_array_equal(b.perceptron_cells.H, b2.perceptron_cells.H)

    def test_baby_serialization_legacy_payloads(self):
        """Pre-marker payloads rebuild as Perceptron brains."""
        b = SimBaby(params=WorldParams())
        data = b.to_dict()
        for key in ("perceptron_cells", "perceptron_body"):
            data[key].pop("kind", None)
        b2 = SimBaby.from_dict(data, params=WorldParams())
        assert isinstance(b2.perceptron_cells, Perceptron)

    def test_genome_roundtrip(self):
        """Genome.from_baby → apply_to must move ModelBrain weights intact."""
        params = WorldParams(brain_kind="model", brain_hidden_units=4)
        parent = SimBaby(params=params)
        g = Genome.from_baby(parent)
        child = SimBaby(params=params)
        g.apply_to(child)
        np.testing.assert_array_equal(parent.perceptron_cells.W, child.perceptron_cells.W)
        np.testing.assert_array_equal(parent.perceptron_cells.H, child.perceptron_cells.H)
        np.testing.assert_array_equal(parent.perceptron_move.W, child.perceptron_move.W)

    def test_evolved_genome_applies_to_model_baby(self):
        """Genome.random tensors (perceptron layout) fit ModelBrain shapes."""
        params = WorldParams(brain_kind="model", brain_hidden_units=4)
        rng = np.random.default_rng(5)
        b = SimBaby(params=params)
        g = Genome.random(params, rng)
        g.apply_to(b)  # must not raise on shape mismatch
        assert b.perceptron_cells.W.shape[1] == 3

    def test_ticks_run(self):
        """A model-brained scene must simulate end to end."""
        params = WorldParams(
            grid_size=(12, 6, 12),
            generate_world=True,
            world_seed=3,
            start_agents=4,
            brain_kind="model",
            brain_hidden_units=4,
        )
        scene = SimScene(params=params)
        for _ in range(4):
            scene.add_baby(SimBaby(params=params))
        Simulation(scene, max_ticks=5).run()  # must not raise


# ── Proofs ──────────────────────────────────────────────────────────────────


class TestProofs:
    def test_emergence_with_model_brains(self):
        """Evolved model brains must beat frozen-random ones."""
        params = WorldParams(grid_size=(16, 8, 16), brain_kind="model")
        result = benchmark_emergence(params, hidden_units=8, seed=1)
        assert "emerged" in result
        assert result["emerged"] is True, (
            f"model-brain emergence failed: evolved={result['evolved_last_avg']:.4f} "
            f"frozen={result['frozen_last_avg']:.4f}"
        )

    def test_learning_moves_hidden_in_world(self):
        """In-life learning must move the model brain's H inside a scene."""
        params = WorldParams(
            grid_size=(12, 6, 12),
            generate_world=True,
            world_seed=4,
            start_agents=4,
            brain_kind="model",
            brain_hidden_units=8,
            learning_enabled=True,
        )
        scene = SimScene(params=params)
        babies = []
        for _ in range(4):
            b = SimBaby(params=params)
            scene.add_baby(b)
            babies.append(b)
        h0 = [b.perceptron_cells.H.copy() for b in babies]
        Simulation(scene, max_ticks=10).run()
        drift = float(
            np.mean([np.abs(a - b.perceptron_cells.H).mean() for a, b in zip(h0, babies)])
        )
        assert drift > 0.0, "model brain H did not move under in-life learning"
